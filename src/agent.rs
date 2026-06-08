//! Per-turn agent Supervisor. Owns the OS-side process lifecycle (spawn,
//! signal escalation on cancel, stderr buffering, startup watchdog, prompt
//! file write/cleanup, power assertion). The agent-specific bits — building
//! the CLI command, normalizing stdout frames into claude-shape envelopes,
//! harvesting session ids/usage/compact boundaries — live behind the
//! `AgentAdapter` trait in `adapter.rs` and the concrete adapters in
//! `adapters/`. Supervisor stays agent-agnostic; future adapters plug in
//! through the registry without touching this file.
//!
//! Each `spawn_agent` call constructs a fresh adapter, hands it to the spawned
//! task, and discards it when the turn ends.
use nix::sys::signal::{self, Signal};
use nix::unistd::Pid;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex as StdMutex,
};
use std::time::{Duration, SystemTime};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn};

use crate::adapter::{AgentEvent, AgentKind, TurnContext};
use crate::envelope::{EnvelopeAttachment, MessageEnvelope};

/// Per-chat pending-attachments registry: the agent-side `attach-file` flow
/// writes `EnvelopeAttachment`s through these unbounded senders, and the
/// per-turn supervisor task (the receiver-owner) drains them just before
/// forwarding the next assistant frame. Cleared on agent exit so a future
/// `attach-file` against a stale chat-id returns a "no running agent" error
/// rather than queueing forever.
///
/// Std `Mutex` because (a) hold times are microseconds (one `HashMap` get +
/// `try_send`), (b) the control-socket handler is the only writer and the
/// supervisor side only touches it at spawn/cleanup. No async work happens
/// under the guard so a `tokio::sync::Mutex` would be wasted async overhead.
pub type PendingAttachments =
    Arc<StdMutex<HashMap<String, mpsc::UnboundedSender<EnvelopeAttachment>>>>;

/// Directory of live resident sessions' FSM state, shared with the control-
/// socket task so an inbound `prune-context` RPC can read a chat's `live_tasks`
/// without reaching the `Supervisor` (which the main loop owns exclusively).
/// Keyed by chat id; the value is the SAME `Arc<StdMutex<SessionState>>` the
/// reader mutates — one source of truth, no duplicated state to keep in sync.
/// Registered on resident spawn (overwriting any stale entry for the chat) and
/// dropped on teardown. The control task reads `live_tasks.len()` to refuse a
/// prune that would restart the agent out from under a running background task
/// / Monitor (whose runtime is in-process and not restored by `--resume`).
pub type LiveSessions = Arc<StdMutex<HashMap<String, Arc<StdMutex<SessionState>>>>>;

const AGENT_EXIT_TIMEOUT: Duration = Duration::from_secs(10);
const AGENT_STARTUP_TIMEOUT: Duration = Duration::from_secs(30);

/// Frame-derived obligations for ONE resident session, updated by the reader
/// task as `AgentEvent`s arrive and read by the Supervisor to answer
/// `is_running`. Only `running` crosses the wire (the boolean
/// `chats.agent_running`, true the whole time the agent is busy):
///   - `running = turn_in_flight || !live_tasks.is_empty()`
///   - `waiting = !turn_in_flight && !live_tasks.is_empty()`
///
/// i.e. Thinking = turn in flight; Waiting = no turn but a background task
/// (monitor / `run_in_background`) is still armed; Idle = neither. `running`
/// stays true across BOTH Thinking and Waiting, so the wire never distinguishes
/// them — `waiting()` documents the FSM contract for tests only; the
/// Thinking-vs-Waiting label is re-derived on the client.
#[derive(Debug, Default, Clone)]
pub struct SessionState {
    /// A user turn is in flight: set when we write a user frame, cleared on the
    /// user-turn's own `result` (origin absent — `Result{origin_is_task:false}`).
    pub turn_in_flight: bool,
    /// `task_id`s of armed background tasks (inserted on `TaskStarted`, removed
    /// on `TaskFinished`).
    pub live_tasks: std::collections::HashSet<String>,
}

impl SessionState {
    pub fn running(&self) -> bool {
        self.turn_in_flight || !self.live_tasks.is_empty()
    }
    /// The "Waiting" sub-state — busy with no turn in flight, only a background
    /// task / monitor still armed. NOT emitted on the wire (the client
    /// re-derives the Thinking-vs-Waiting label); kept solely to document and
    /// unit-test the FSM contract, so it's `#[cfg(test)]`.
    #[cfg(test)]
    pub fn waiting(&self) -> bool {
        !self.turn_in_flight && !self.live_tasks.is_empty()
    }
    /// Fully Idle — safe to tear down, and the only state in which a queued prune
    /// may be applied (a respawn would otherwise kill armed monitors). Exactly the
    /// negation of `running()`.
    pub fn is_idle(&self) -> bool {
        !self.running()
    }
}

/// PURE event→state reducer — the single source of truth for the resident
/// FSM, factored out so it's unit-testable without spawning a process. The
/// reader task calls this for each adapter event; everything else (`Frame`,
/// `ContextTokens`, …) leaves the run state untouched and is handled by the
/// reader directly.
///
/// `Result{origin_is_task:false}` ends the user turn; `Result{origin_is_task:
/// true}` is a background-task wake's result and must NOT clear the turn flag.
pub fn reduce(state: &mut SessionState, ev: &AgentEvent) {
    match ev {
        AgentEvent::Result {
            origin_is_task: false,
        } => {
            state.turn_in_flight = false;
        }
        AgentEvent::Result {
            origin_is_task: true,
        } => { /* background-task wake — leaves turn_in_flight unchanged */ }
        AgentEvent::TaskStarted(id) => {
            state.live_tasks.insert(id.clone());
        }
        AgentEvent::TaskFinished(id) => {
            state.live_tasks.remove(id);
        }
        // Non-state-bearing events.
        AgentEvent::Frame(_)
        | AgentEvent::ContextTokens(_)
        | AgentEvent::CompactBoundary(_)
        | AgentEvent::SessionIdHarvested(_)
        | AgentEvent::ToolResult => {}
    }
}

/// First-turn user text: on the first turn of a brand-new session, append the
/// adapter's `first_turn_prompt_suffix` (after a horizontal rule); later turns
/// (and adapters with no suffix, e.g. claude) send the prompt verbatim. Used by
/// `default_resident_spawn_fn` to build the first stdin frame.
fn first_turn_text(
    adapter: &dyn crate::adapter::AgentAdapter,
    prompt: String,
    first_turn: bool,
) -> String {
    match adapter.first_turn_prompt_suffix() {
        Some(suffix) if first_turn => format!("{prompt}\n\n---\n{suffix}"),
        _ => prompt,
    }
}

pub enum AgentResponse {
    Line {
        topic: String,
        content: String,
    },
    ContextTokens {
        topic: String,
        tokens: i64,
    },
    /// Manual `/compact` or auto-compact completed; carries `compactMetadata.postTokens`.
    CompactBoundary {
        topic: String,
        post_tokens: i64,
    },
    /// First `system/init` frame from the agent on a fresh chat — carries the
    /// session id the agent generated for itself. Caller writes it to
    /// `chats.agent_session_id` so subsequent turns can resume it.
    SessionIdHarvested {
        topic: String,
        session_id: String,
    },
    /// One-shot path: the per-turn process exited. `has_result=false` ⇒ it
    /// died without emitting a `result` (caller synthesizes `INTERRUPTED_RESULT`).
    /// RESIDENT path reuses this for process EXIT (EOF / crash / hard teardown):
    /// the reader sets `has_result=false` when the process exited while still
    /// busy (turn in flight or tasks armed) so the caller publishes
    /// `INTERRUPTED_RESULT`, `true` when it exited while Idle. Either way the
    /// caller clears `agent_running` (→ false / idle) and drops the session.
    Done {
        topic: String,
        has_result: bool,
    },
    /// RESIDENT path only — the session's busy/idle run state changed. The reader
    /// emits this whenever `running` transitions (deduped, so no spam of
    /// identical values). `running` stays true for the whole time the agent is
    /// busy — including while a background task / monitor is still armed (the
    /// internal "waiting" sub-state); only the busy↔idle edge crosses the wire.
    /// The idle (`running=false`) edge is emitted ONLY on a `Result` turn-boundary
    /// frame — never on a bare `TaskFinished` that empties `live_tasks`, because a
    /// claude-self-initiated continuation turn is about to run and will end with its
    /// own `Result`, so the false edge is deferred to that real boundary. Busy
    /// (`running=true`) edges always emit.
    /// `main.rs::handle_agent_response` maps it to `WriteEvent::chat_running(...)`
    /// via `send_run_state`. See `SessionState`.
    RunState {
        topic: String,
        running: bool,
    },
    /// The `prune-context` call's own result has persisted for `topic` — the main
    /// loop's cue to apply a queued prune (abort → rewrite → respawn). Call-keyed:
    /// the adapter only emits it for the prune call's own result, never a sibling's.
    /// Carries no body (the frame itself is skipped); a no-op unless a
    /// `PruneRequest` is pending for the chat. See `AgentEvent::ToolResult`.
    ToolResult {
        topic: String,
    },
}

impl AgentEvent {
    /// Maps one adapter-emitted event onto the supervisor's response channel.
    /// `AgentEvent::Result` is a supervisor-only signal — it flips `has_result`
    /// (so the eventual `AgentResponse::Done` carries the right flag) and
    /// returns `None`, suppressing wire emission. Every other variant maps
    /// 1:1 to an `AgentResponse` for the given chat topic.
    fn into_response(self, topic: &str, has_result: &mut bool) -> Option<AgentResponse> {
        Some(match self {
            AgentEvent::Frame(content) => AgentResponse::Line {
                topic: topic.to_string(),
                content,
            },
            AgentEvent::ContextTokens(tokens) => AgentResponse::ContextTokens {
                topic: topic.to_string(),
                tokens,
            },
            AgentEvent::CompactBoundary(post_tokens) => AgentResponse::CompactBoundary {
                topic: topic.to_string(),
                post_tokens,
            },
            AgentEvent::SessionIdHarvested(session_id) => AgentResponse::SessionIdHarvested {
                topic: topic.to_string(),
                session_id,
            },
            AgentEvent::Result { origin_is_task: _ } => {
                *has_result = true;
                return None;
            }
            // Inert on the one-shot spawn path: the `--print` process exits at
            // its first `result`, so no background task ever fires and these
            // never reach this mapper. The resident session reader consumes them
            // directly to drive the Thinking/Waiting state machine (it does not
            // route through `into_response`). Kept exhaustive so a stray frame
            // can't panic the match.
            AgentEvent::TaskStarted(_) | AgentEvent::TaskFinished(_) => {
                return None;
            }
            AgentEvent::ToolResult => AgentResponse::ToolResult {
                topic: topic.to_string(),
            },
        })
    }
}

/// One-shot input to `Supervisor::spawn_agent` — bundles every per-turn
/// parameter so `spawn_agent` is a single-argument call and the
/// `#[allow(clippy::too_many_arguments)]` waiver can stay off.
#[derive(Clone)]
pub struct SpawnRequest {
    pub chat_id: String,
    pub prompt: String,
    pub project_path: Option<String>,
    pub worktree: bool,
    /// `Some(_)` to resume a prior session, `None` for a brand-new chat.
    pub agent_session_id: Option<String>,
    pub agent_kind: AgentKind,
    pub is_sandboxed: bool,
    /// `chats.model` — verbatim `--model <X>` pass-through to the CLI
    /// (migration 0035). Empty / blank values are filtered to `None` at
    /// the `main.rs` construction site so adapters can read `Some(_)`
    /// as "user picked a non-default model".
    pub model: Option<String>,
    /// Sender's IANA timezone (`machine_users.timezone`, migration 0040), or
    /// `None`. Injects the user's current local time into the prompt each turn
    /// for `schedule-message --at`.
    pub user_timezone: Option<String>,
}

/// A command for a resident session's single writer task. The writer owns the
/// `ChildStdin` and writes the session's one (first) user turn directly before
/// draining this channel; with no warm reuse there are no follow-up user
/// frames, so the only message is `Shutdown` (flush + close on teardown).
#[derive(Debug)]
pub enum StdinMsg {
    /// Flush + close: the writer task exits (drops the `ChildStdin`).
    Shutdown,
}

/// Handles a resident spawn fn returns so the Supervisor can drive the session
/// after it's started. The reader+writer tasks are joined into the single
/// `reader` handle the prod impl returns (writer is spawned inside it / awaited
/// by it); `state` is the shared FSM the reader mutates and the Supervisor
/// reads for `is_running`.
pub struct ResidentHandles {
    pub stdin_tx: mpsc::UnboundedSender<StdinMsg>,
    pub reader: tokio::task::JoinHandle<()>,
    pub state: Arc<StdMutex<SessionState>>,
}

/// A live resident session for one chat. Held in `AgentEntry::Resident`.
pub struct ResidentSession {
    /// Writer-task input: user frames, shutdown.
    stdin_tx: mpsc::UnboundedSender<StdinMsg>,
    /// Owns the child stdout loop + child handle + writer task.
    reader: tokio::task::JoinHandle<()>,
    /// Hard kill (SIGTERM the process group) — teardown only, never /stop.
    cancel: CancellationToken,
    // NOTE: the harvested session id is NOT stored here — the reader emits
    // `SessionIdHarvested`, main.rs persists it to the mirror, and a respawn
    // (teardown / crash / next message) pulls it back from the mirror into
    // `SpawnRequest.agent_session_id` for `--resume`. One source of truth.
    /// FSM mutated by the reader, read by the Supervisor for `is_running`.
    state: Arc<StdMutex<SessionState>>,
}

/// Either a one-shot per-turn task (codex/gemini/cursor/hermes) or a resident
/// session (claude). One-shot keeps the historic `(handle, token)` pair; the
/// `agents` map holds one entry per chat regardless of kind.
enum AgentEntry {
    OneShot {
        handle: tokio::task::JoinHandle<()>,
        cancel: CancellationToken,
    },
    Resident(ResidentSession),
}

/// Per-spawn closure for ONE-SHOT adapters: owns the OS-side work (build
/// Command, spawn, drive stdout/stderr, emit `AgentResponse`s on `tx`, observe
/// `token` for cancel, return the driving `JoinHandle`). Default implementation
/// is `default_spawn_fn` which preserves the historic behavior; tests inject a
/// recorder via `Supervisor::with_spawn_fn` to capture `SpawnRequest`s without
/// ever touching `tokio::process::Command`.
///
/// `pending` is the per-chat `EnvelopeAttachment` mailbox the agent-side
/// `attach-file` flow pushes into; the per-turn spawn task registers its
/// receiver in it at spawn time and removes it on exit. Test spawn fns ignore
/// this — they never wrap frames.
pub type SpawnFn = Arc<
    dyn Fn(
            SpawnRequest,
            mpsc::Sender<AgentResponse>,
            CancellationToken,
            PendingAttachments,
        ) -> tokio::task::JoinHandle<()>
        + Send
        + Sync,
>;

/// Per-session spawn closure for RESIDENT adapters (claude): spawns the
/// resident process, starts the writer + reader tasks, and returns the handles
/// the Supervisor needs to drive it. Default is `default_resident_spawn_fn`
/// (real process); tests inject a recorder via `with_spawn_fns` that records the
/// session `SpawnRequest` and returns inert handles.
///
/// The `SpawnRequest.prompt` carries the turn's text — the spawn fn writes it as
/// the first stdin frame right after the process is up. Every new message
/// respawns a fresh `--resume` process (no warm reuse).
pub type ResidentSpawnFn = Arc<
    dyn Fn(
            SpawnRequest,
            mpsc::Sender<AgentResponse>,
            CancellationToken,
            PendingAttachments,
        ) -> ResidentHandles
        + Send
        + Sync,
>;

pub struct Supervisor {
    agents: HashMap<String, AgentEntry>,
    response_tx: mpsc::Sender<AgentResponse>,
    spawn_fn: SpawnFn,
    resident_spawn_fn: ResidentSpawnFn,
    /// Shared with the control-socket handler so an inbound `attach-file`
    /// RPC can look up the right per-chat mailbox by chat id.
    pending: PendingAttachments,
    /// Shared with the control-socket handler so an inbound `prune-context` RPC
    /// can read a chat's live-task count (resident sessions only). Points at the
    /// same per-session `SessionState` Arcs the reader mutates.
    live_sessions: LiveSessions,
}

impl Supervisor {
    pub fn new(response_tx: mpsc::Sender<AgentResponse>) -> Self {
        Self::with_spawn_fns(
            response_tx,
            Arc::new(default_spawn_fn),
            Arc::new(default_resident_spawn_fn),
        )
    }

    /// Inject a custom ONE-SHOT spawn implementation; resident kinds keep the
    /// real `default_resident_spawn_fn`. Test-only — the many existing recorder
    /// tests that only exercise one-shot capture use it; production goes through
    /// `new` / `with_spawn_fns`.
    #[cfg(test)]
    pub fn with_spawn_fn(response_tx: mpsc::Sender<AgentResponse>, spawn_fn: SpawnFn) -> Self {
        Self::with_spawn_fns(response_tx, spawn_fn, Arc::new(default_resident_spawn_fn))
    }

    /// Inject custom one-shot AND resident spawn implementations. The resident
    /// recorder tests use this to capture session `SpawnRequest`s (reuse vs
    /// respawn) without spawning a real `claude`.
    pub fn with_spawn_fns(
        response_tx: mpsc::Sender<AgentResponse>,
        spawn_fn: SpawnFn,
        resident_spawn_fn: ResidentSpawnFn,
    ) -> Self {
        Self {
            agents: HashMap::new(),
            response_tx,
            spawn_fn,
            resident_spawn_fn,
            pending: Arc::new(StdMutex::new(HashMap::new())),
            live_sessions: Arc::new(StdMutex::new(HashMap::new())),
        }
    }

    /// Hand the control-socket task a clone of the pending-attachments
    /// registry so it can route `attach-file` RPC results to the right
    /// per-chat mailbox. Cheap `Arc` clone — same shared state.
    pub fn pending_attachments(&self) -> PendingAttachments {
        self.pending.clone()
    }

    /// Hand the control-socket task a clone of the live-sessions directory so a
    /// `prune-context` RPC can read a chat's live-task count. Cheap `Arc` clone.
    pub fn live_sessions(&self) -> LiveSessions {
        self.live_sessions.clone()
    }

    /// Number of background tasks / monitors live in the chat's resident session
    /// right now (0 if no resident session). Read at prune apply-time to note in
    /// the respawn prompt how many tasks the restart terminated.
    pub fn live_task_count(&self, topic: &str) -> usize {
        self.live_sessions
            .lock()
            .expect("LiveSessions mutex")
            .get(topic)
            .map(|st| st.lock().expect("SessionState mutex").live_tasks.len())
            .unwrap_or(0)
    }

    /// Drop live-sessions entries whose chat no longer has a tracked agent. Cheap
    /// `retain` over a small map; call after any teardown that removes from
    /// `agents` so the directory doesn't leak stale `SessionState` Arcs.
    fn sync_live_sessions(&self) {
        self.live_sessions
            .lock()
            .expect("LiveSessions mutex")
            .retain(|topic, _| self.agents.contains_key(topic));
    }

    /// HARD teardown of a chat's agent (one-shot: cancel token → SIGTERM the
    /// group; resident: shutdown stdin then cancel → SIGTERM). Removes the entry
    /// and waits for the driving task to exit so the process is fully dead
    /// before any respawn on the same session. Used for interrupt-then-send /
    /// prune / idle-teardown / shutdown / /stop (all hard teardowns now — there is
    /// no in-place interrupt).
    pub async fn abort_agent(&mut self, topic: &str) -> bool {
        let removed = self.agents.remove(topic);
        self.sync_live_sessions();
        match removed {
            Some(AgentEntry::OneShot { handle, cancel }) => {
                info!(topic = %topic, "aborting running agent (one-shot)");
                cancel.cancel();
                await_agent_exit(handle, topic).await;
                true
            }
            Some(AgentEntry::Resident(session)) => {
                info!(topic = %topic, "tearing down resident session");
                // Ask the writer to flush+close first, then hard-cancel so the
                // reader SIGTERMs the group. Either alone suffices; doing both is
                // belt-and-suspenders for a wedged child.
                let _ = session.stdin_tx.send(StdinMsg::Shutdown);
                session.cancel.cancel();
                await_agent_exit(session.reader, topic).await;
                true
            }
            None => false,
        }
    }

    /// Check if there are no tracked agents.
    pub fn is_empty(&self) -> bool {
        self.agents.is_empty()
    }

    pub fn remove(&mut self, topic: &str) {
        self.agents.remove(topic);
        self.sync_live_sessions();
    }

    /// True when the chat has a tracked agent that hasn't finished. For a
    /// resident session this means the process is alive AND its FSM is non-Idle
    /// (a fully-Idle resident is torn down on its idle edge, so this is rarely
    /// observed Idle); for one-shot it's "task not finished".
    pub fn is_running(&self, topic: &str) -> bool {
        match self.agents.get(topic) {
            Some(AgentEntry::OneShot { handle, .. }) => !handle.is_finished(),
            Some(AgentEntry::Resident(s)) => !s.reader.is_finished() && session_state(s).running(),
            None => false,
        }
    }

    /// Dispatch a spawn request: resident kinds (claude) always (re)spawn a
    /// fresh `--resume` process; one-shot kinds keep the historic per-turn
    /// spawn. Branches on the adapter's `is_resident()`.
    pub fn spawn_agent(&mut self, req: SpawnRequest) {
        if req.agent_kind.make_adapter().is_resident() {
            self.spawn_resident(req);
        } else {
            self.spawn_oneshot(req);
        }
    }

    /// One-shot per-turn spawn (codex/gemini/cursor/hermes) — historic path.
    fn spawn_oneshot(&mut self, req: SpawnRequest) {
        let topic = req.chat_id.clone();
        let cancel = CancellationToken::new();
        let handle = (self.spawn_fn)(
            req,
            self.response_tx.clone(),
            cancel.clone(),
            self.pending.clone(),
        );
        self.agents
            .insert(topic, AgentEntry::OneShot { handle, cancel });
    }

    /// Resident path: always (re)spawn fresh with `--resume`; no warm reuse. A
    /// new user message never pushes a frame into an existing warm process —
    /// every turn gets a fresh `claude --resume <agent_session_id>` process.
    /// Drop any stale entry (finished reader, or an interrupt-then-send abort the
    /// caller already did) then spawn fresh and insert.
    fn spawn_resident(&mut self, req: SpawnRequest) {
        let topic = req.chat_id.clone();
        self.agents.remove(&topic);
        let cancel = CancellationToken::new();
        let handles = (self.resident_spawn_fn)(
            req.clone(),
            self.response_tx.clone(),
            cancel.clone(),
            self.pending.clone(),
        );
        let session = ResidentSession {
            stdin_tx: handles.stdin_tx,
            reader: handles.reader,
            cancel,
            state: handles.state,
        };
        // The spawn fn queues the first user turn (from `req.prompt`) itself but
        // returns a default (idle) FSM. Mark `turn_in_flight` here — the single
        // set covers both the real path (so an `is_running`/RunState read between
        // spawn and the first frame is correct) and the test-recorder path (which
        // runs no real writer). Done synchronously before the entry is stored.
        {
            let mut st = session.state.lock().expect("SessionState mutex");
            st.turn_in_flight = true;
        }
        // Publish this session's FSM to the shared directory (overwriting any
        // stale entry for the chat) so the control task can read its live-task
        // count when a `prune-context` RPC arrives.
        self.live_sessions
            .lock()
            .expect("LiveSessions mutex")
            .insert(topic.clone(), session.state.clone());
        self.agents.insert(topic, AgentEntry::Resident(session));
    }

    pub fn cleanup(&mut self) {
        self.agents.retain(|_, entry| match entry {
            AgentEntry::OneShot { handle, .. } => !handle.is_finished(),
            AgentEntry::Resident(s) => !s.reader.is_finished(),
        });
        self.sync_live_sessions();
    }

    pub async fn shutdown_all(&mut self) {
        let agents: Vec<_> = self.agents.drain().collect();
        self.live_sessions
            .lock()
            .expect("LiveSessions mutex")
            .clear();
        if agents.is_empty() {
            return;
        }
        info!("shutting down {} running agent(s)", agents.len());
        let mut joins: Vec<(String, tokio::task::JoinHandle<()>)> = Vec::new();
        for (topic, entry) in agents {
            info!(topic = %topic, "cancelling agent");
            match entry {
                AgentEntry::OneShot { handle, cancel } => {
                    cancel.cancel();
                    joins.push((topic, handle));
                }
                AgentEntry::Resident(s) => {
                    let _ = s.stdin_tx.send(StdinMsg::Shutdown);
                    s.cancel.cancel();
                    joins.push((topic, s.reader));
                }
            }
        }
        for (topic, handle) in joins {
            await_agent_exit(handle, &topic).await;
        }
    }
}

/// Snapshot a resident session's FSM (cloned out from under the std Mutex —
/// hold time is one `HashSet` clone, no async work).
fn session_state(s: &ResidentSession) -> SessionState {
    s.state.lock().expect("SessionState mutex").clone()
}

/// Default spawn implementation — verbatim lift of the historic body of
/// `Supervisor::spawn_agent`. Builds a `tokio::process::Command` via the
/// adapter, drives stdout/stderr, and emits `AgentResponse`s onto `tx`. The
/// per-turn `CancellationToken` is observed for /stop and abort-then-respawn.
///
/// `pending` is the shared per-chat-id mailbox registry. The spawned task
/// installs its receiver into the map at startup (and removes it on exit)
/// so the control-socket handler can push `EnvelopeAttachment`s in via
/// `attach-file`; after each assistant text frame is forwarded, the
/// supervisor drains the receiver and — if non-empty — emits a separate
/// follow-up `AgentResponse::Line` whose body is an attachments-only
/// `MessageEnvelope`. iOS sees that as its own row and renders an
/// attachment-only bubble below the assistant text.
fn default_spawn_fn(
    req: SpawnRequest,
    tx: mpsc::Sender<AgentResponse>,
    token: CancellationToken,
    pending: PendingAttachments,
) -> tokio::task::JoinHandle<()> {
    let SpawnRequest {
        chat_id: topic,
        prompt,
        project_path,
        worktree,
        agent_session_id,
        agent_kind,
        is_sandboxed,
        model,
        user_timezone,
    } = req;
    let mut adapter = agent_kind.make_adapter();

    let topic_clone = topic;
    let token_clone = token;

    // Register the per-chat pending-attachments mailbox before the spawn so
    // a fast `attach-file` RPC issued right after the agent starts isn't
    // dropped on the floor (the agent could already be reading the prompt
    // file). Removed in the `Drop` arm below on every exit path (clean
    // finish, cancel, error) so a future RPC against a stale chat id fails
    // loud with "no running agent" rather than queueing forever.
    let (attach_tx, mut attach_rx) = mpsc::unbounded_channel::<EnvelopeAttachment>();
    {
        let mut guard = pending.lock().expect("PendingAttachments mutex");
        // If a prior turn for this chat is somehow still in the map (no-op
        // under the abort-then-respawn path because the prior task's RAII
        // guard already removed it), prefer the fresh sender — drops the
        // old one and its mailbox.
        guard.insert(topic_clone.clone(), attach_tx);
    }

    tokio::spawn(async move {
        // RAII guard so cancellation paths (early `return` after the
        // outer-cancel arm, error returns from prompt-file write, etc.)
        // still clean the registry entry. Cheap: one `HashMap::remove`
        // under a short-held std Mutex.
        struct PendingGuard {
            map: PendingAttachments,
            chat_id: String,
        }
        impl Drop for PendingGuard {
            fn drop(&mut self) {
                if let Ok(mut g) = self.map.lock() {
                    g.remove(&self.chat_id);
                }
            }
        }
        let _pending_guard = PendingGuard {
            map: pending,
            chat_id: topic_clone.clone(),
        };
        let _power_assertion = crate::power::AgentPowerAssertion::acquire();
        let kind = adapter.kind();
        info!(
            topic = %topic_clone,
            kind = ?kind,
            resume = agent_session_id.is_some(),
            project_path = ?project_path,
            worktree,
            sandbox = is_sandboxed,
            "spawning agent"
        );

        // First turn only (`agent_session_id.is_none()`): append the adapter's
        // prompt suffix after the user's text. See `first_turn_prompt_suffix`.
        let prompt = if agent_session_id.is_none() {
            match adapter.first_turn_prompt_suffix() {
                Some(suffix) => format!("{prompt}\n\n---\n{suffix}"),
                None => prompt,
            }
        } else {
            prompt
        };

        // Write prompt to a temp file so it never touches the shell command string
        let unique = format!(
            "{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        );
        let prompt_file = PathBuf::from(format!("/tmp/zucchini-prompt-{}.txt", unique));

        let ctx = TurnContext {
            chat_id: &topic_clone,
            prompt_file: &prompt_file,
            project_path: project_path.as_deref(),
            worktree,
            agent_session_id: agent_session_id.as_deref(),
            is_sandboxed,
            model: model.as_deref(),
            user_timezone: user_timezone.as_deref(),
        };

        // Prepend the adapter's prompt-file prefixes (contracts on
        // `AgentAdapter::prompt_file_time_line` / `prompt_file_preamble`) ONLY into
        // the plaintext temp file, never the synced/persisted `messages` row.
        // Injection order: volatile time line, once-only preamble, then the message.
        let mut parts: Vec<String> = Vec::new();
        if let Some(time_line) = adapter.prompt_file_time_line(&ctx) {
            parts.push(time_line);
        }
        if let Some(preamble) = adapter.prompt_file_preamble(&ctx) {
            parts.push(preamble);
        }
        parts.push(prompt);
        let prompt_to_write = parts.join("\n\n---\n\n");
        if let Err(e) = tokio::fs::write(&prompt_file, &prompt_to_write).await {
            error!("failed to write prompt file: {}", e);
            fail_agent(
                &tx,
                &topic_clone,
                format!("failed to write prompt file: {}", e),
            )
            .await;
            return;
        }

        let cmd_string = match adapter.prepare_command(&ctx) {
            Ok(s) => s,
            Err(e) => {
                error!("adapter prepare_command failed: {}", e);
                let _ = tokio::fs::remove_file(&prompt_file).await;
                fail_agent(&tx, &topic_clone, format!("agent prepare failed: {}", e)).await;
                return;
            }
        };

        let user_shell = crate::shell::user_login_shell();
        info!(shell = %user_shell, kind = ?kind, "spawning agent via login shell");

        let mut cmd = Command::new(&user_shell);
        cmd.args(["-lic", &cmd_string])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .process_group(0); // new process group so we can kill shell + agent together
                               // Adapter prompts reference both vars to invoke
                               // `zucchini-spawner attach-file` / `schedule-message`. Exported on
                               // the spawn (not via shell rc) so a stale PATH can't pick the wrong
                               // binary — `current_exe()` is whatever launchd/systemd runs, which
                               // is also what the RPC handler listens on.
        cmd.env("ZUCCHINI_CHAT_ID", &topic_clone);
        if let Ok(exe) = std::env::current_exe() {
            cmd.env("ZUCCHINI_SPAWNER_BIN", exe);
        }
        let result = cmd.spawn();

        let mut child = match result {
            Ok(child) => child,
            Err(e) => {
                error!("failed to spawn agent: {}", e);
                let _ = tokio::fs::remove_file(&prompt_file).await;
                fail_agent(&tx, &topic_clone, format!("failed to spawn agent: {}", e)).await;
                return;
            }
        };

        // Shared flag: flipped to true once we receive the first stdout line from the agent.
        // The stderr task uses this to decide whether to buffer (startup noise) or warn (runtime).
        let agent_started = Arc::new(AtomicBool::new(false));
        let agent_started_stderr = agent_started.clone();

        // Read stderr in a separate task. Before the agent starts we buffer lines silently;
        // after startup any stderr is a genuine warning. We return the buffer so the main
        // task can report it to Sentry only if the agent never started.
        let stderr_handle = if let Some(stderr) = child.stderr.take() {
            let topic_for_stderr = topic_clone.clone();
            Some(tokio::spawn(async move {
                let reader = BufReader::new(stderr);
                let mut lines = reader.lines();
                let mut startup_buf: Vec<String> = Vec::new();
                while let Ok(Some(line)) = lines.next_line().await {
                    if agent_started_stderr.load(Ordering::Relaxed) {
                        warn!(topic = %topic_for_stderr, "agent stderr: {}", line);
                    } else if startup_buf.len() < 200 {
                        startup_buf.push(line);
                    }
                }
                startup_buf
            }))
        } else {
            None
        };

        let mut has_result = false;

        if let Some(stdout) = child.stdout.take() {
            let reader = BufReader::new(stdout);
            let mut lines = reader.lines();
            let startup_deadline = tokio::time::Instant::now() + AGENT_STARTUP_TIMEOUT;

            // `cancelled` is set by the inner per-send select when the outer
            // cancellation token fires mid-emit (slow writer back-pressures the mpsc
            // and a chatty turn can emit multiple events per line); we break out and
            // run the same SIGTERM/SIGKILL cleanup as the outer cancel arm.
            let mut cancelled_mid_send = false;
            loop {
                tokio::select! {
                    _ = token_clone.cancelled() => {
                        warn!(topic = %topic_clone, "agent cancelled, sending SIGTERM to process group");
                        terminate_agent_process_group(&mut child, Some(prompt_file.as_path()), &topic_clone).await;
                        // Don't send Done — caller publishes INTERRUPTED_RESULT and has
                        // already removed our entry from the map.
                        return;
                    }
                    _ = tokio::time::sleep_until(startup_deadline), if !agent_started.load(Ordering::Relaxed) => {
                        error!(topic = %topic_clone, "agent produced no output within {:?}, killing", AGENT_STARTUP_TIMEOUT);
                        let _ = tx.send(AgentResponse::Line {
                            topic: topic_clone.clone(),
                            content: format!("Error: agent failed to start — no output within {:?}. Check shell configuration (~/.zshrc / ~/.bashrc).", AGENT_STARTUP_TIMEOUT),
                        }).await;
                        kill_agent_process_group(&mut child).await;
                        break;
                    }
                    line_result = lines.next_line() => {
                        match line_result {
                            Ok(Some(line)) => {
                                // First stdout line means the agent is alive — silences any
                                // later stderr buffering and stops the startup watchdog.
                                agent_started.store(true, Ordering::Relaxed);

                                let events = adapter.handle_line(line);
                                let mut channel_closed = false;
                                for ev in events {
                                    // Result is a supervisor-only signal — set the latch and
                                    // emit nothing on the wire. Every other event maps 1:1 to
                                    // an AgentResponse via `into_response`.
                                    let Some(resp) = ev.into_response(&topic_clone, &mut has_result) else {
                                        continue;
                                    };
                                    // Compute a follow-up attachment row BEFORE forwarding the
                                    // original — `attach_followup_for` peeks at the response
                                    // shape and drains the mailbox only when the line is an
                                    // assistant text frame with queued attachments. Held
                                    // over the original send so cancellation between the two
                                    // emits drops both cleanly.
                                    let followup = if let AgentResponse::Line { topic, content } = &resp {
                                        attach_followup_for(topic, content, &mut attach_rx)
                                    } else {
                                        None
                                    };
                                    // Nested select so a mid-loop cancellation (Stop
                                    // tapped while the bounded mpsc is full) is observed
                                    // immediately instead of waiting for the writer to
                                    // drain. `biased` keeps cancel polled first.
                                    tokio::select! {
                                        biased;
                                        _ = token_clone.cancelled() => {
                                            cancelled_mid_send = true;
                                            break;
                                        }
                                        send_res = tx.send(resp) => {
                                            if send_res.is_err() {
                                                channel_closed = true;
                                                break;
                                            }
                                        }
                                    }
                                    // Emit the attachment row immediately after the text frame
                                    // so iOS renders the paperclip pill bubble right below the
                                    // assistant text. Same cancellation contract as above.
                                    if let Some(followup) = followup {
                                        tokio::select! {
                                            biased;
                                            _ = token_clone.cancelled() => {
                                                cancelled_mid_send = true;
                                                break;
                                            }
                                            send_res = tx.send(followup) => {
                                                if send_res.is_err() {
                                                    channel_closed = true;
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                }
                                if cancelled_mid_send {
                                    break;
                                }
                                if channel_closed {
                                    warn!(topic = %topic_clone, "response channel closed, killing agent");
                                    let _ = child.kill().await;
                                    break;
                                }
                            }
                            Ok(None) => break,
                            Err(e) => {
                                error!(topic = %topic_clone, "error reading stdout: {}", e);
                                break;
                            }
                        }
                    }
                }
            }

            // Mid-send cancellation: same cleanup as the outer cancel arm
            // (SIGTERM → 5s wait → SIGKILL → wait → remove prompt file → return; no Done).
            if cancelled_mid_send {
                warn!(topic = %topic_clone, "agent cancelled mid-emit, sending SIGTERM to process group");
                terminate_agent_process_group(
                    &mut child,
                    Some(prompt_file.as_path()),
                    &topic_clone,
                )
                .await;
                return;
            }
        }

        if let Some(h) = stderr_handle {
            if let Ok(startup_buf) = h.await {
                if !agent_started.load(Ordering::Relaxed) && !startup_buf.is_empty() {
                    let stderr = startup_buf.join("\n");
                    error!(topic = %topic_clone, "agent failed to start. startup stderr:\n{}", stderr);
                    let _ = tx
                        .send(AgentResponse::Line {
                            topic: topic_clone.clone(),
                            content: format!("Error: agent failed to start.\n{}", stderr),
                        })
                        .await;
                }
            }
        }

        match child.wait().await {
            Ok(status) => info!(topic = %topic_clone, %status, "agent exited"),
            Err(e) => error!(topic = %topic_clone, "error waiting for agent: {}", e),
        }

        let _ = tokio::fs::remove_file(&prompt_file).await;

        // Post-turn context-token correction for adapters (codex) that read
        // occupancy from the on-disk transcript, now flushed after exit.
        // `None` (stream-sourced adapters, or a read miss) ⇒ gauge keeps its
        // last value, not zeroed.
        if let Some(tokens) = adapter.post_turn_context_tokens(agent_session_id.as_deref()) {
            let _ = tx
                .send(AgentResponse::ContextTokens {
                    topic: topic_clone.clone(),
                    tokens,
                })
                .await;
        }

        let _ = tx
            .send(AgentResponse::Done {
                topic: topic_clone,
                has_result,
            })
            .await;
    })
}

/// Default RESIDENT spawn implementation (claude). Spawns one `claude` process
/// with stdin held open (`Stdio::piped()`), owns the `ChildStdin` in a single
/// writer task fed by an mpsc of `StdinMsg`, and drives stdout in a reader task
/// that never exits at the first `result` — it loops until stdout EOF / read
/// error / `cancel`. The reader runs each line through `adapter.handle_line`,
/// feeds run-state events through `reduce`, emits `RunState` on every busy↔idle
/// (`running`) transition, and forwards `Frame`/`ContextTokens`/`CompactBoundary`/
/// `SessionIdHarvested`/`ToolResult` exactly as the one-shot path does
/// (attachment-followup logic preserved; its lifetime is now the session). On
/// process exit it emits one `Done{has_result}` (`has_result=false` ⇒ exited
/// while busy ⇒ caller publishes `INTERRUPTED_RESULT`).
fn default_resident_spawn_fn(
    req: SpawnRequest,
    tx: mpsc::Sender<AgentResponse>,
    cancel: CancellationToken,
    pending: PendingAttachments,
) -> ResidentHandles {
    let (stdin_tx, mut stdin_rx) = mpsc::unbounded_channel::<StdinMsg>();
    // Default (idle) FSM. The first user turn is queued below, but `turn_in_flight`
    // is set by `spawn_resident` (the single setter) synchronously before the
    // session entry is stored, so a RunState/`is_running` read can't observe a
    // stale idle state.
    let state = Arc::new(StdMutex::new(SessionState::default()));

    let state_for_reader = state.clone();
    let stdin_tx_for_handles = stdin_tx.clone();

    let SpawnRequest {
        chat_id: topic,
        prompt,
        project_path,
        worktree,
        agent_session_id,
        agent_kind,
        is_sandboxed,
        model,
        user_timezone,
    } = req;

    // Register the per-chat attachment mailbox before the spawn (same rationale
    // as the one-shot path). Lifetime is now the SESSION — removed by the
    // reader's RAII guard on exit.
    let (attach_tx, mut attach_rx) = mpsc::unbounded_channel::<EnvelopeAttachment>();
    {
        let mut guard = pending.lock().expect("PendingAttachments mutex");
        guard.insert(topic.clone(), attach_tx);
    }

    let reader = tokio::spawn(async move {
        struct PendingGuard {
            map: PendingAttachments,
            chat_id: String,
        }
        impl Drop for PendingGuard {
            fn drop(&mut self) {
                if let Ok(mut g) = self.map.lock() {
                    g.remove(&self.chat_id);
                }
            }
        }
        let _pending_guard = PendingGuard {
            map: pending,
            chat_id: topic.clone(),
        };
        let _power_assertion = crate::power::AgentPowerAssertion::acquire();
        let mut adapter = agent_kind.make_adapter();
        info!(
            topic = %topic,
            resume = agent_session_id.is_some(),
            project_path = ?project_path,
            worktree,
            sandbox = is_sandboxed,
            "spawning resident agent"
        );

        let ctx = crate::adapter::SessionContext {
            chat_id: &topic,
            project_path: project_path.as_deref(),
            worktree,
            agent_session_id: agent_session_id.as_deref(),
            is_sandboxed,
            model: model.as_deref(),
            user_timezone: user_timezone.as_deref(),
        };
        let cmd_string = match adapter.prepare_session_command(&ctx) {
            Ok(s) => s,
            Err(e) => {
                error!("adapter prepare_session_command failed: {}", e);
                fail_agent(&tx, &topic, format!("agent prepare failed: {}", e)).await;
                return;
            }
        };

        let user_shell = crate::shell::user_login_shell();
        info!(shell = %user_shell, "spawning resident agent via login shell");
        let mut cmd = Command::new(&user_shell);
        cmd.args(["-lic", &cmd_string])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .process_group(0);
        cmd.env("ZUCCHINI_CHAT_ID", &topic);
        if let Ok(exe) = std::env::current_exe() {
            cmd.env("ZUCCHINI_SPAWNER_BIN", exe);
        }
        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                error!("failed to spawn resident agent: {}", e);
                fail_agent(&tx, &topic, format!("failed to spawn agent: {}", e)).await;
                return;
            }
        };

        // Writer task: owns ChildStdin, the single writer of stdin frames. Queue
        // the FIRST user turn (with the first-turn suffix when this is a brand-new
        // session — no resume id) before draining the mpsc so the turn is sent the
        // instant the process is up.
        let first_turn = agent_session_id.is_none();
        let first_text = first_turn_text(adapter.as_ref(), prompt, first_turn);
        let first_frame = adapter.encode_user_turn(&first_text);
        let writer_handle = child.stdin.take().map(|mut stdin| {
            tokio::spawn(async move {
                use tokio::io::AsyncWriteExt;
                if stdin.write_all(first_frame.as_bytes()).await.is_err()
                    || stdin.flush().await.is_err()
                {
                    return;
                }
                // With no warm reuse the only message is `Shutdown` — wait for it
                // (or the channel dropping). Either way fall through to drop stdin.
                let _ = stdin_rx.recv().await;
                // Dropping `stdin` here closes the pipe; claude ignores stdin EOF
                // (stays alive — see plan §8), so teardown is the SIGTERM below.
            })
        });

        let agent_started = Arc::new(AtomicBool::new(false));
        let agent_started_stderr = agent_started.clone();
        let stderr_handle = if let Some(stderr) = child.stderr.take() {
            let topic_for_stderr = topic.clone();
            Some(tokio::spawn(async move {
                let reader = BufReader::new(stderr);
                let mut lines = reader.lines();
                let mut startup_buf: Vec<String> = Vec::new();
                while let Ok(Some(line)) = lines.next_line().await {
                    if agent_started_stderr.load(Ordering::Relaxed) {
                        warn!(topic = %topic_for_stderr, "agent stderr: {}", line);
                    } else if startup_buf.len() < 200 {
                        startup_buf.push(line);
                    }
                }
                startup_buf
            }))
        } else {
            None
        };

        // Last `running` value emitted, so we dedup identical transitions. The
        // internal "waiting" sub-state (armed background task, no turn in flight)
        // is NOT distinguished on the wire — `running` stays true across it — so
        // a Thinking↔Waiting flip emits nothing; only the busy↔idle edge does.
        let mut last_emitted: Option<bool> = None;
        // True when the exit was a DELIBERATE `cancel` teardown (knob-change
        // respawn, prune-respawn, idle-teardown, shutdown). A deliberate teardown
        // emits NO `Done` — the caller (`abort_agent`) has already removed the
        // map entry and owns the post-state, exactly like the one-shot cancel
        // arm. Emitting a `Done` here would (a) post a spurious
        // INTERRUPTED_RESULT and (b) race a respawn: the stale `Done`, processed
        // after `spawn_agent` re-inserted a fresh session, would `remove` it.
        // Only a SPONTANEOUS exit (EOF / read error / startup timeout) emits a
        // `Done` so the crash-recovery path can clear columns + publish
        // INTERRUPTED_RESULT when it died mid-turn.
        let mut cancelled = false;

        if let Some(stdout) = child.stdout.take() {
            let bufreader = BufReader::new(stdout);
            let mut lines = bufreader.lines();
            let startup_deadline = tokio::time::Instant::now() + AGENT_STARTUP_TIMEOUT;
            'read: loop {
                tokio::select! {
                    _ = cancel.cancelled() => {
                        warn!(topic = %topic, "resident session cancelled, SIGTERM process group");
                        cancelled = true;
                        terminate_agent_process_group(&mut child, None, &topic).await;
                        break 'read;
                    }
                    _ = tokio::time::sleep_until(startup_deadline), if !agent_started.load(Ordering::Relaxed) => {
                        error!(topic = %topic, "resident agent produced no output within {:?}, killing", AGENT_STARTUP_TIMEOUT);
                        let _ = tx.send(AgentResponse::Line {
                            topic: topic.clone(),
                            content: format!("Error: agent failed to start — no output within {:?}. Check shell configuration (~/.zshrc / ~/.bashrc).", AGENT_STARTUP_TIMEOUT),
                        }).await;
                        kill_agent_process_group(&mut child).await;
                        break 'read;
                    }
                    line_result = lines.next_line() => {
                        match line_result {
                            Ok(Some(line)) => {
                                agent_started.store(true, Ordering::Relaxed);
                                let events = adapter.handle_line(line);
                                for ev in events {
                                    // Reduce the FSM and read back the resulting
                                    // `running` in one lock — the `match` below only
                                    // forwards frames and never mutates state, so
                                    // this value is final for the event. (The FSM
                                    // still tracks the waiting sub-state internally
                                    // via `SessionState`; it just doesn't cross the
                                    // wire.)
                                    let is_result = matches!(&ev, AgentEvent::Result { .. });
                                    let running = {
                                        let mut st = state_for_reader.lock().expect("SessionState mutex");
                                        reduce(&mut st, &ev);
                                        let r = st.running();
                                        // [DIAG tasktrace] TEMPORARY — log every
                                        // state-bearing event with the resulting FSM
                                        // so we can see why live_tasks never empties.
                                        match &ev {
                                            AgentEvent::TaskStarted(id) => tracing::info!(diag = "tasktrace", topic = %topic, %id, live = st.live_tasks.len(), turn = st.turn_in_flight, running = r, "reduce TaskStarted"),
                                            AgentEvent::TaskFinished(id) => tracing::info!(diag = "tasktrace", topic = %topic, %id, live = st.live_tasks.len(), turn = st.turn_in_flight, running = r, "reduce TaskFinished"),
                                            AgentEvent::Result { origin_is_task } => tracing::info!(diag = "tasktrace", topic = %topic, origin_is_task, live = st.live_tasks.len(), turn = st.turn_in_flight, running = r, "reduce Result"),
                                            _ => {}
                                        }
                                        r
                                    };
                                    match ev {
                                        AgentEvent::Frame(content) => {
                                            let resp = AgentResponse::Line { topic: topic.clone(), content };
                                            let followup = if let AgentResponse::Line { topic: t, content: c } = &resp {
                                                attach_followup_for(t, c, &mut attach_rx)
                                            } else { None };
                                            if tx.send(resp).await.is_err() { break 'read; }
                                            if let Some(f) = followup {
                                                if tx.send(f).await.is_err() { break 'read; }
                                            }
                                        }
                                        AgentEvent::ContextTokens(tokens) => {
                                            if tx.send(AgentResponse::ContextTokens { topic: topic.clone(), tokens }).await.is_err() { break 'read; }
                                        }
                                        AgentEvent::CompactBoundary(post_tokens) => {
                                            if tx.send(AgentResponse::CompactBoundary { topic: topic.clone(), post_tokens }).await.is_err() { break 'read; }
                                        }
                                        AgentEvent::SessionIdHarvested(session_id) => {
                                            if tx.send(AgentResponse::SessionIdHarvested { topic: topic.clone(), session_id }).await.is_err() { break 'read; }
                                        }
                                        AgentEvent::ToolResult => {
                                            if tx.send(AgentResponse::ToolResult { topic: topic.clone() }).await.is_err() { break 'read; }
                                        }
                                        // Run-state-bearing events (Result/TaskStarted/
                                        // TaskFinished) already updated the FSM via
                                        // `reduce`; the RunState recompute below emits a
                                        // transition if `running` changed.
                                        AgentEvent::Result { .. }
                                        | AgentEvent::TaskStarted(_)
                                        | AgentEvent::TaskFinished(_) => {}
                                    }
                                    // Emit the idle (`running=false`) edge ONLY on a `Result` frame — a real
                                    // turn boundary. A bare `TaskFinished` that empties `live_tasks` also drops
                                    // `running` to false, but a claude-self-initiated continuation turn is about
                                    // to run (and will end with its own `Result`), so suppressing the edge here
                                    // defers it to that turn's end. Busy (`running=true`) edges always emit.
                                    if last_emitted != Some(running) && (running || is_result) {
                                        last_emitted = Some(running);
                                        // [DIAG tasktrace] TEMPORARY — the busy↔idle edge that writes agent_running.
                                        tracing::info!(diag = "tasktrace", topic = %topic, running, "RunState transition -> chat_running");
                                        if tx.send(AgentResponse::RunState { topic: topic.clone(), running }).await.is_err() { break 'read; }
                                    }
                                }
                            }
                            Ok(None) => { break 'read; }
                            Err(e) => { error!(topic = %topic, "error reading resident stdout: {}", e); break 'read; }
                        }
                    }
                }
            }
        }

        if let Some(h) = stderr_handle {
            if let Ok(startup_buf) = h.await {
                if !agent_started.load(Ordering::Relaxed) && !startup_buf.is_empty() {
                    let stderr = startup_buf.join("\n");
                    error!(topic = %topic, "resident agent failed to start. startup stderr:\n{}", stderr);
                    let _ = tx
                        .send(AgentResponse::Line {
                            topic: topic.clone(),
                            content: format!("Error: agent failed to start.\n{}", stderr),
                        })
                        .await;
                }
            }
        }

        // Stop the writer (drops ChildStdin) and reap the child.
        if let Some(wh) = writer_handle {
            wh.abort();
        }
        match child.wait().await {
            Ok(status) => info!(topic = %topic, %status, "resident agent exited"),
            Err(e) => error!(topic = %topic, "error waiting for resident agent: {}", e),
        }

        // Deliberate teardown ⇒ no `Done` (caller owns the post-state; see the
        // `cancelled` declaration above). SPONTANEOUS exit ⇒ emit `Done`:
        // `has_result` repurposed as "exited cleanly" — true when the process
        // died while fully Idle (no in-flight turn, no armed tasks), false when
        // it died mid-turn / with tasks armed ⇒ caller publishes
        // INTERRUPTED_RESULT. Either way the caller clears the run-state columns
        // and drops the session; the next message respawns with `--resume`.
        if !cancelled {
            let was_idle = state_for_reader
                .lock()
                .expect("SessionState mutex")
                .is_idle();
            let _ = tx
                .send(AgentResponse::Done {
                    topic,
                    has_result: was_idle,
                })
                .await;
        }
    });

    ResidentHandles {
        stdin_tx: stdin_tx_for_handles,
        reader,
        state,
    }
}

async fn await_agent_exit(handle: tokio::task::JoinHandle<()>, topic: &str) {
    match tokio::time::timeout(AGENT_EXIT_TIMEOUT, handle).await {
        Ok(_) => info!(topic = %topic, "agent exited"),
        Err(_) => warn!(topic = %topic, "agent did not exit in {:?}", AGENT_EXIT_TIMEOUT),
    }
}

/// Wait this long after SIGTERM before escalating to SIGKILL. Same value used
/// by the outer-cancel and mid-send-cancel arms.
const AGENT_SIGTERM_GRACE: Duration = Duration::from_secs(5);

/// Graceful kill: SIGTERM the whole process group → wait up to
/// `AGENT_SIGTERM_GRACE` → SIGKILL if still alive → wait → remove the prompt
/// file (one-shot only — resident sessions pass `None`, having no prompt temp
/// file). Used by the one-shot outer-cancel + mid-send-cancel arms and by the
/// resident cancel teardown (knob-change / prune / idle-teardown / shutdown).
///
/// `process_group(0)` on spawn means `child.id()` IS the PGID, so a single
/// `killpg` reaches both the login shell and the agent process underneath it.
/// If `child.id()` is `None` (process already reaped) we fall back to
/// `child.kill()` which is a no-op in that case but keeps the API consistent.
async fn terminate_agent_process_group(
    child: &mut tokio::process::Child,
    prompt_file: Option<&std::path::Path>,
    topic: &str,
) {
    if let Some(pid) = child.id() {
        let pgid = Pid::from_raw(pid as i32);
        let _ = signal::killpg(pgid, Signal::SIGTERM);
        match tokio::time::timeout(AGENT_SIGTERM_GRACE, child.wait()).await {
            Ok(_) => info!(topic = %topic, "agent exited after SIGTERM"),
            Err(_) => {
                warn!(topic = %topic, "agent did not exit in {:?}, sending SIGKILL to process group", AGENT_SIGTERM_GRACE);
                let _ = signal::killpg(pgid, Signal::SIGKILL);
                let _ = child.wait().await;
            }
        }
    } else {
        let _ = child.kill().await;
    }
    if let Some(prompt_file) = prompt_file {
        let _ = tokio::fs::remove_file(prompt_file).await;
    }
}

/// Fast-path kill: SIGKILL the process group immediately, no grace period.
/// Used only by the startup-deadline arm — if the agent has produced no output
/// in `AGENT_STARTUP_TIMEOUT` it's almost certainly hung on shell-rc init
/// rather than actively doing work, so SIGTERM grace is wasted wait. Caller
/// is responsible for removing the prompt file afterwards (the post-loop
/// cleanup path already does).
async fn kill_agent_process_group(child: &mut tokio::process::Child) {
    if let Some(pid) = child.id() {
        let pgid = Pid::from_raw(pid as i32);
        let _ = signal::killpg(pgid, Signal::SIGKILL);
    } else {
        let _ = child.kill().await;
    }
}

async fn fail_agent(tx: &mpsc::Sender<AgentResponse>, topic: &str, msg: String) {
    let _ = tx
        .send(AgentResponse::Line {
            topic: topic.to_string(),
            content: format!("Error: {}", msg),
        })
        .await;
    let _ = tx
        .send(AgentResponse::Done {
            topic: topic.to_string(),
            has_result: false,
        })
        .await;
}

/// True when `line` is a text-bearing assistant frame — i.e. an outer
/// `{"type":"assistant", ...}` whose `message.content[]` contains at least
/// one `{"type":"text", ...}` block. A tool_use-only assistant frame is
/// rejected: iOS renders those as system/tool rows that strip attachments,
/// so pinning a file there silently drops it. A mixed text + tool_use frame
/// still counts (the bubble carries the text).
///
/// Substring fast-rejects non-assistant frames (tool_result wraps in
/// `"type":"user"`, etc.) before doing the real `serde_json` parse. On parse
/// failure or unexpected shape we return false so attachments stay queued
/// for the next frame (no data loss).
fn is_assistant_text_frame(line: &str) -> bool {
    if !line.contains("\"type\":\"assistant\"") {
        return false;
    }
    let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
        return false;
    };
    let Some(content) = v
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_array())
    else {
        return false;
    };
    content
        .iter()
        .any(|b| b.get("type").and_then(|t| t.as_str()) == Some("text"))
}

/// If `content` is an assistant text frame AND `attach_rx` has any queued
/// attachments, build a follow-up `AgentResponse::Line` whose body is a
/// `MessageEnvelope { text: "", attachments }` JSON string. The caller emits
/// it immediately after the original text frame, producing a dedicated
/// "attachment-only" `messages` row that iOS renders as a paperclip pill
/// bubble below the assistant text. Returns `None` in every other case
/// (non-assistant frames, empty mailbox).
///
/// Why a separate row instead of wrapping the text frame: keeps the spawner's
/// "one stream-json frame per row, body never grows" invariant intact — the
/// attachment row is just another frame the spawner generated. The text-frame
/// body stays a verbatim claude-SDK frame, which is what `SpawnerMessageDescriber`
/// has always parsed, so there's no envelope-vs-raw branching on the iOS hot
/// path.
fn attach_followup_for(
    topic: &str,
    content: &str,
    attach_rx: &mut mpsc::UnboundedReceiver<EnvelopeAttachment>,
) -> Option<AgentResponse> {
    if !is_assistant_text_frame(content) {
        return None;
    }
    let mut attachments: Vec<EnvelopeAttachment> = Vec::new();
    while let Ok(att) = attach_rx.try_recv() {
        attachments.push(att);
    }
    if attachments.is_empty() {
        return None;
    }
    let n = attachments.len();
    let envelope = MessageEnvelope {
        text: String::new(),
        attachments,
    };
    match serde_json::to_string(&envelope) {
        Ok(s) => {
            info!(
                topic = %topic,
                n,
                "emitting follow-up attachment row with {} attachment(s)",
                n
            );
            Some(AgentResponse::Line {
                topic: topic.to_string(),
                content: s,
            })
        }
        Err(e) => {
            // Effectively unreachable — `MessageEnvelope` is a trivial
            // struct of plain types. Drop the attachments rather than
            // crashing the turn; the assistant text already went out.
            warn!(error = %e, "failed to serialize MessageEnvelope; dropping attachment row");
            None
        }
    }
}

#[cfg(test)]
mod reduce_tests {
    use super::{reduce, AgentEvent, SessionState};

    /// Drive a frame sequence through the pure reducer and assert the derived
    /// (running, waiting) pair after each event — the resident FSM contract.
    /// Mirrors a real claude resident turn that arms a background task:
    ///   user frame → Thinking; user-turn result with a task live → Waiting;
    ///   task finished → Idle.
    #[test]
    fn user_turn_then_task_then_finish_walks_thinking_waiting_idle() {
        // A user turn is in flight (the Supervisor sets this when it writes the
        // user frame; here we model it explicitly).
        let mut s = SessionState {
            turn_in_flight: true,
            live_tasks: Default::default(),
        };
        assert_eq!((s.running(), s.waiting()), (true, false), "Thinking");
        assert!(!s.is_idle());

        // A background task starts while the turn is still running.
        reduce(&mut s, &AgentEvent::TaskStarted("t1".into()));
        assert_eq!(
            (s.running(), s.waiting()),
            (true, false),
            "still Thinking (turn in flight dominates)"
        );

        // The user turn's own result lands (origin absent) → turn ends, but the
        // task is still armed → Waiting.
        reduce(
            &mut s,
            &AgentEvent::Result {
                origin_is_task: false,
            },
        );
        assert_eq!((s.running(), s.waiting()), (true, true), "Waiting");
        assert!(!s.is_idle());

        // The task completes → fully Idle.
        reduce(&mut s, &AgentEvent::TaskFinished("t1".into()));
        assert_eq!((s.running(), s.waiting()), (false, false), "Idle");
        assert!(s.is_idle());
    }

    /// A task-driven wake's `result` (origin = task-notification) must NOT clear
    /// `turn_in_flight` — only the user-turn's own result does.
    #[test]
    fn task_origin_result_does_not_end_user_turn() {
        let mut s = SessionState {
            turn_in_flight: true,
            live_tasks: Default::default(),
        };
        s.live_tasks.insert("t1".into());
        reduce(
            &mut s,
            &AgentEvent::Result {
                origin_is_task: true,
            },
        );
        assert!(
            s.turn_in_flight,
            "a background-task wake result must not end the user turn"
        );
        assert_eq!((s.running(), s.waiting()), (true, false));
    }

    /// Two stacked tasks: both must finish before the session goes Idle.
    #[test]
    fn two_tasks_both_must_finish_for_idle() {
        let mut s = SessionState::default();
        reduce(&mut s, &AgentEvent::TaskStarted("a".into()));
        reduce(&mut s, &AgentEvent::TaskStarted("b".into()));
        // No turn in flight, tasks armed → Waiting.
        assert_eq!((s.running(), s.waiting()), (true, true));
        reduce(&mut s, &AgentEvent::TaskFinished("a".into()));
        assert_eq!((s.running(), s.waiting()), (true, true), "b still live");
        reduce(&mut s, &AgentEvent::TaskFinished("b".into()));
        assert!(s.is_idle());
    }

    /// Non-state-bearing events leave the FSM untouched.
    #[test]
    fn frame_and_tokens_are_inert_for_run_state() {
        let mut s = SessionState {
            turn_in_flight: true,
            live_tasks: Default::default(),
        };
        for ev in [
            AgentEvent::Frame("{}".into()),
            AgentEvent::ContextTokens(10),
            AgentEvent::CompactBoundary(5),
            AgentEvent::SessionIdHarvested("sid".into()),
            AgentEvent::ToolResult,
        ] {
            reduce(&mut s, &ev);
        }
        assert!(s.turn_in_flight);
        assert!(s.live_tasks.is_empty());
        assert_eq!((s.running(), s.waiting()), (true, false));
    }

    /// A finished `TaskFinished` for a never-started id is a harmless no-op
    /// (the adapter may drop a `task_started` parse and still see the finish).
    #[test]
    fn finish_of_unknown_task_is_noop() {
        let mut s = SessionState::default();
        reduce(&mut s, &AgentEvent::TaskFinished("ghost".into()));
        assert!(s.is_idle());
    }
}

#[cfg(test)]
mod attach_tests {
    use super::is_assistant_text_frame;

    #[test]
    fn text_only_assistant_frame_matches() {
        let f = r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"hi"}],"usage":{}}}"#;
        assert!(is_assistant_text_frame(f));
    }

    #[test]
    fn tool_use_only_assistant_frame_rejected() {
        let f = r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"Bash","input":{"command":"ls"}}],"usage":{}}}"#;
        assert!(!is_assistant_text_frame(f));
    }

    #[test]
    fn mixed_text_and_tool_use_matches() {
        let f = r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"let me check"},{"type":"tool_use","id":"t1","name":"Bash","input":{"command":"ls"}}],"usage":{}}}"#;
        assert!(is_assistant_text_frame(f));
    }

    #[test]
    fn user_tool_result_frame_rejected() {
        let f = r#"{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"ok"}]}}"#;
        assert!(!is_assistant_text_frame(f));
    }

    #[test]
    fn malformed_json_rejected() {
        assert!(!is_assistant_text_frame(r#"{"type":"assistant","#));
    }
}

/// Live-session-directory tests. These drive the REAL `live_sessions` directory
/// (read by the control task's `prune-context` guard via `live_task_count`)
/// against a recorder resident spawn fn that exposes its `state` Arc per chat id,
/// so a test can reshape the session FSM (insert/clear live tasks) without a real
/// `claude` process and assert the directory tracks + clears correctly.
#[cfg(test)]
mod live_session_tests {
    use super::*;
    use crate::adapter::AgentKind;
    use std::collections::HashMap;

    type Exposed = Arc<StdMutex<HashMap<String, Arc<StdMutex<SessionState>>>>>;

    /// A Supervisor whose resident spawn fn records the `state` Arc per chat id
    /// (so the test can reshape the session) and returns a reader that lives
    /// until cancelled (so `abort_agent` can join it).
    fn supervisor_with_exposed_sessions() -> (Supervisor, Exposed) {
        let exposed: Exposed = Arc::new(StdMutex::new(HashMap::new()));
        let exp = exposed.clone();
        let resident_fn: ResidentSpawnFn = Arc::new(move |req, _tx, cancel, _pending| {
            let (stdin_tx, _stdin_rx) = mpsc::unbounded_channel();
            let state = Arc::new(StdMutex::new(SessionState::default()));
            exp.lock()
                .unwrap()
                .insert(req.chat_id.clone(), state.clone());
            let reader = tokio::spawn(async move {
                let _keep = _stdin_rx;
                cancel.cancelled().await;
            });
            ResidentHandles {
                stdin_tx,
                reader,
                state,
            }
        });
        let one_shot: SpawnFn = Arc::new(|_req, _tx, _token, _pending| tokio::spawn(async {}));
        let (resp_tx, _resp_rx) = mpsc::channel::<AgentResponse>(64);
        (
            Supervisor::with_spawn_fns(resp_tx, one_shot, resident_fn),
            exposed,
        )
    }

    fn claude_req(chat_id: &str) -> SpawnRequest {
        SpawnRequest {
            chat_id: chat_id.to_string(),
            prompt: "hi".to_string(),
            project_path: None,
            worktree: false,
            agent_session_id: None,
            agent_kind: AgentKind::Claude,
            is_sandboxed: false,
            model: None,
            user_timezone: None,
        }
    }

    fn set_waiting(exposed: &Exposed, chat: &str) {
        let g = exposed.lock().unwrap();
        let mut st = g[chat].lock().unwrap();
        st.turn_in_flight = false;
        st.live_tasks.insert("t1".to_string());
    }

    /// The shared `live_sessions` directory (read by the control task's
    /// `prune-context` guard via `live_task_count`) reflects a resident session's
    /// live tasks while it's alive and is cleared on teardown. This is the data
    /// source for the prune-while-task-live block, so it must track exactly.
    #[tokio::test(start_paused = true)]
    async fn live_session_directory_tracks_tasks_and_clears_on_teardown() {
        let (mut sup, exposed) = supervisor_with_exposed_sessions();
        sup.spawn_agent(claude_req("deploying"));

        // Registered on spawn; no tasks yet.
        assert_eq!(sup.live_task_count("deploying"), 0);
        assert_eq!(sup.live_task_count("never-spawned"), 0);

        // A task armed in the session is visible through the shared directory.
        set_waiting(&exposed, "deploying"); // inserts live task "t1"
        assert_eq!(
            sup.live_task_count("deploying"),
            1,
            "an armed task is visible to the control-side guard"
        );

        // Teardown drops the directory entry — no stale count blocks a later prune.
        sup.abort_agent("deploying").await;
        assert_eq!(
            sup.live_task_count("deploying"),
            0,
            "teardown clears the live-sessions entry"
        );
    }

    /// Respawning the same chat overwrites the directory entry with the fresh
    /// session's FSM, so a stale count never lingers across a respawn.
    #[tokio::test(start_paused = true)]
    async fn respawn_overwrites_live_session_entry() {
        let (mut sup, exposed) = supervisor_with_exposed_sessions();
        sup.spawn_agent(claude_req("chat"));
        set_waiting(&exposed, "chat");
        assert_eq!(sup.live_task_count("chat"), 1);

        // A knob-changing respawn tears down + respawns; the recorder hands back a
        // fresh default state, so the directory must now read 0, not the stale 1.
        sup.abort_agent("chat").await;
        let mut req = claude_req("chat");
        req.model = Some("opus".to_string());
        sup.spawn_agent(req);
        assert_eq!(
            sup.live_task_count("chat"),
            0,
            "respawn publishes the fresh session's (empty) task set"
        );
    }
}
