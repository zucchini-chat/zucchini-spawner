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
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex as StdMutex,
};
use std::time::{Duration, SystemTime};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::{mpsc, Notify};
use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn};
use uuid::Uuid;

use crate::adapter::{AgentEvent, AgentKind, TurnContext};
use crate::envelope::{EnvelopeAttachment, MessageEnvelope};
use crate::sys;

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

/// True once ANY agent on this machine produced its first output line
/// (in-memory, since spawner start). Gates the silent-startup retry below:
/// with no success history, startup silence is almost certainly deterministic
/// (broken shell init, missing CLI) and a retry only doubles the user's wait.
static ANY_AGENT_STARTED: AtomicBool = AtomicBool::new(false);

/// One automatic respawn after a SILENT startup timeout — the transient-stall
/// signature (host overload / memory pressure; SPAWNER-13): the process hung
/// past the deadline (we had to kill it), emitted zero stderr, and agents are
/// known to start on this machine. Anything else fails immediately: captured
/// stderr means a real diagnostic is about to be surfaced, no success history
/// means environmental breakage, and the single-attempt cap keeps a
/// deterministic hang from looping. Only sound BEFORE the first output frame —
/// a started agent may have side effects, so mid-turn deaths are never retried.
fn startup_retry_allowed(attempt: u32, stderr_seen: &AtomicBool) -> bool {
    attempt == 0
        && !stderr_seen.load(Ordering::Relaxed)
        && ANY_AGENT_STARTED.load(Ordering::Relaxed)
}

fn startup_timeout_message(timeout: Duration, retried: bool) -> String {
    if retried {
        format!("Error: agent failed to start — no output within {timeout:?}, even after an automatic retry. The machine may be overloaded; try sending the message again.")
    } else if ANY_AGENT_STARTED.load(Ordering::Relaxed) {
        format!("Error: agent failed to start — no output within {timeout:?}.")
    } else {
        format!("Error: agent failed to start — no output within {timeout:?}. Check shell configuration (~/.zshrc / ~/.bashrc).")
    }
}

/// Frame-derived obligations for ONE resident session, updated by the reader
/// task as `AgentEvent`s arrive and read by the Supervisor to answer
/// `is_running`. Only `running` crosses the wire (the boolean
/// `chats.agent_running`, true the whole time the agent is busy):
///   - `running = turn_in_flight || !unconsumed.is_empty() || !live_tasks.is_empty()`
///   - `waiting = running && !turn_in_flight && unconsumed.is_empty()`
///
/// i.e. Thinking = a turn in flight or queued; Waiting = only a background task
/// (monitor / `run_in_background`) is still armed; Idle = none. `running`
/// stays true across BOTH Thinking and Waiting, so the wire never distinguishes
/// them — `waiting()` documents the FSM contract for tests only; the
/// Thinking-vs-Waiting label is re-derived on the client.
///
/// A queued turn stays `unconsumed` until claude echoes it: written mid-turn it
/// is answered by the running turn's single `result`; written after the last
/// tool call it runs as its own turn after that `result` — which therefore
/// can't end the busy state while the turn is still unconsumed.
#[derive(Debug, Default, Clone)]
pub struct SessionState {
    /// A consumed user turn is in flight: set on its `UserTurnConsumed` echo,
    /// cleared on a user-turn `result` (origin absent — `Result{origin_is_task:false}`).
    pub turn_in_flight: bool,
    /// `task_id`s of armed background tasks (inserted on `TaskStarted`, removed
    /// on `TaskFinished`).
    pub live_tasks: HashSet<String>,
    /// User turns queued for stdin and not yet echoed as consumed, in order.
    /// Registered before the write, so a racing `result` can't read Idle.
    pub unconsumed: Vec<QueuedTurn>,
    /// The reader left its stdout loop: the process is exiting and takes no
    /// more turns. Set before its `Done`.
    pub closed: bool,
}

#[derive(Debug, Clone)]
pub struct QueuedTurn {
    pub uuid: String,
    /// The user's prompt, re-queued by a hard restart that kills the process
    /// before it consumed the turn.
    pub prompt: String,
    frame: String,
    /// A slash command: claude expands it only as a turn of its own.
    own_turn: bool,
}

impl SessionState {
    pub fn running(&self) -> bool {
        self.turn_in_flight || !self.unconsumed.is_empty() || !self.live_tasks.is_empty()
    }
    /// Mid-turn or hosting background tasks — what a hard abort interrupts.
    fn working(&self) -> bool {
        self.turn_in_flight || !self.live_tasks.is_empty()
    }
    /// The "Waiting" sub-state — busy with no turn in flight, only a background
    /// task / monitor still armed. NOT emitted on the wire (the client
    /// re-derives the Thinking-vs-Waiting label); kept solely to document and
    /// unit-test the FSM contract, so it's `#[cfg(test)]`.
    #[cfg(test)]
    pub fn waiting(&self) -> bool {
        self.running() && !self.turn_in_flight && self.unconsumed.is_empty()
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
        AgentEvent::UserTurnConsumed(uuid) => {
            if let Some(i) = state.unconsumed.iter().position(|t| &t.uuid == uuid) {
                state.unconsumed.remove(i);
                state.turn_in_flight = true;
            }
        }
        // Non-state-bearing events.
        AgentEvent::Frame(_)
        | AgentEvent::ContextTokens(_)
        | AgentEvent::CompactBoundary(_)
        | AgentEvent::SessionIdHarvested(_)
        | AgentEvent::ToolResult => {}
    }
}

const TURN_PART_SEPARATOR: &str = "\n\n---\n\n";

/// An expanding CLI (`AgentAdapter::expands_slash_commands`) runs a slash
/// command only when the turn STARTS with it, so such a turn ships verbatim —
/// whole prompt, no time line, preamble or first-turn suffix (an expanding CLI
/// swallows a trailing suffix as the command's arguments). Only the first token
/// of the first line is inspected: `/compact`, `/plugin:cmd args` and `/tmp`
/// match; `/Users/...` does not, its `/` disqualifying the name. A turn carrying
/// attachments never matches — `blobs::build_prompt` puts the file list first —
/// which mirrors the `attachments.is_empty()` gate on `/stop` in `main.rs`.
fn is_slash_command(prompt: &str) -> bool {
    // Claude matches on `value.trim().startsWith("/")`, so leading blanks must
    // not defeat us either.
    let first_line = prompt.trim_start().lines().next().unwrap_or("");
    let Some(rest) = first_line.strip_prefix('/') else {
        return false;
    };
    let name = rest.split(char::is_whitespace).next().unwrap_or("");
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | ':'))
}

/// User-turn text shared by both lifecycles: the time line, then `preamble`,
/// then the prompt with the adapter's first-turn suffix appended after a
/// horizontal rule. Absent parts are skipped. `first_turn` == brand-new session
/// (no resume id). The time line rides the turn rather than the system prompt —
/// see `AgentAdapter::turn_time_line`.
fn turn_text(
    adapter: &dyn crate::adapter::AgentAdapter,
    prompt: String,
    first_turn: bool,
    user_timezone: Option<&str>,
    preamble: Option<String>,
) -> String {
    if adapter.expands_slash_commands() && is_slash_command(&prompt) {
        return prompt;
    }
    let suffix = first_turn
        .then(|| adapter.first_turn_prompt_suffix())
        .flatten();
    let body = match suffix {
        Some(suffix) => format!("{prompt}\n\n---\n{suffix}"),
        None => prompt,
    };
    let mut parts: Vec<String> = Vec::new();
    parts.extend(adapter.turn_time_line(user_timezone));
    parts.extend(preamble);
    parts.push(body);
    parts.join(TURN_PART_SEPARATOR)
}

#[cfg(test)]
mod turn_text_tests {
    use super::*;
    use crate::adapter::AgentKind;

    #[test]
    fn time_line_prefixes_the_turn() {
        let adapter = AgentKind::Claude.make_adapter();
        let time_line = adapter
            .turn_time_line(Some("Asia/Bangkok"))
            .expect("known tz yields a line");
        let text = turn_text(
            adapter.as_ref(),
            "hello".to_string(),
            false,
            Some("Asia/Bangkok"),
            None,
        );
        assert_eq!(text, format!("{time_line}{TURN_PART_SEPARATOR}hello"));
    }

    #[test]
    fn unknown_timezone_sends_the_prompt_verbatim() {
        let adapter = AgentKind::Claude.make_adapter();
        let text = turn_text(adapter.as_ref(), "hello".to_string(), false, None, None);
        assert_eq!(text, "hello");
    }

    #[test]
    fn slash_command_turns_are_verbatim_so_the_cli_expands_them() {
        let adapter = AgentKind::Claude.make_adapter();
        for prompt in ["/compact", "/compact keep the plan", "/plugin:cmd"] {
            let text = turn_text(
                adapter.as_ref(),
                prompt.to_string(),
                true,
                Some("Asia/Bangkok"),
                Some("PREAMBLE".to_string()),
            );
            assert_eq!(text, prompt);
        }
    }

    /// The verbatim path would cost a non-expanding adapter its one-per-chat
    /// capabilities preamble (and codex's prune suffix) for no slash expansion
    /// at all — this is the only call shape where those parts are ever `Some`.
    #[test]
    fn a_slash_command_keeps_every_part_for_a_non_expanding_adapter() {
        let adapter = AgentKind::Codex.make_adapter();
        let suffix = adapter
            .first_turn_prompt_suffix()
            .expect("codex sets a first-turn suffix");
        let text = turn_text(
            adapter.as_ref(),
            "/init".to_string(),
            true,
            Some("Asia/Bangkok"),
            Some("PREAMBLE".to_string()),
        );
        assert!(text.contains("PREAMBLE"), "{text}");
        assert!(text.contains(suffix), "{text}");
        assert!(text.contains("current local time"), "{text}");
    }

    #[test]
    fn slash_command_matching_edges() {
        assert!(is_slash_command(" /compact"));
        assert!(!is_slash_command("/Users/ivan/notes.txt what is this?"));
        assert!(!is_slash_command("/ compact"));
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
    /// Manual `/compact` or auto-compact completed; carries `compact_metadata.post_tokens`.
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
            AgentEvent::TaskStarted(_)
            | AgentEvent::TaskFinished(_)
            | AgentEvent::UserTurnConsumed(_) => {
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

/// Handles a resident spawn fn returns so the Supervisor can drive the session
/// after it's started. The reader+writer tasks are joined into the single
/// `reader` handle the prod impl returns (writer is spawned inside it / awaited
/// by it); `state` is the shared FSM the reader mutates and the Supervisor
/// reads for `is_running` and queues turns into.
pub struct ResidentHandles {
    /// Wakes the writer to write newly queued `SessionState::unconsumed` frames.
    pub wake: Arc<Notify>,
    pub reader: tokio::task::JoinHandle<()>,
    pub state: Arc<StdMutex<SessionState>>,
}

/// Spawn-time settings baked into a resident process. Not `agent_session_id`
/// (harvested mid-session) nor `user_timezone` (rides each turn).
#[derive(PartialEq)]
struct SessionKnobs {
    agent_kind: AgentKind,
    project_path: Option<String>,
    worktree: bool,
    is_sandboxed: bool,
    model: Option<String>,
}

impl SessionKnobs {
    fn of(req: &SpawnRequest) -> Self {
        Self {
            agent_kind: req.agent_kind,
            project_path: req.project_path.clone(),
            worktree: req.worktree,
            is_sandboxed: req.is_sandboxed,
            model: req.model.clone(),
        }
    }
}

/// A live resident session for one chat. Held in `AgentEntry::Resident`.
pub struct ResidentSession {
    wake: Arc<Notify>,
    knobs: SessionKnobs,
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

impl ResidentSession {
    fn queue_turn(&self, prompt: String, first_turn: bool, user_timezone: Option<&str>) {
        let adapter = self.knobs.agent_kind.make_adapter();
        let uuid = Uuid::now_v7().to_string();
        let text = turn_text(
            adapter.as_ref(),
            prompt.clone(),
            first_turn,
            user_timezone,
            None,
        );
        let frame = adapter.encode_user_turn(&uuid, &text);
        self.state
            .lock()
            .expect("SessionState mutex")
            .unconsumed
            .push(QueuedTurn {
                own_turn: adapter.expands_slash_commands() && is_slash_command(&prompt),
                uuid,
                prompt,
                frame,
            });
        self.wake.notify_one();
    }
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
/// The spawn fn only starts the process and its tasks; `spawn_resident` then
/// queues `SpawnRequest.prompt` as the first turn into the returned `state`.
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
    /// no in-place interrupt). Returns the resident's turns still unconsumed
    /// once its reader is joined (the final FSM, no racing echo).
    pub async fn abort_agent(&mut self, topic: &str) -> Vec<QueuedTurn> {
        let removed = self.agents.remove(topic);
        self.sync_live_sessions();
        match removed {
            Some(AgentEntry::OneShot { handle, cancel }) => {
                info!(topic = %topic, "aborting running agent (one-shot)");
                cancel.cancel();
                await_agent_exit(handle, topic).await;
                Vec::new()
            }
            Some(AgentEntry::Resident(session)) => {
                info!(topic = %topic, "tearing down resident session");
                // Closes stdin (writer) and SIGTERMs the group (reader).
                session.cancel.cancel();
                await_agent_exit(session.reader, topic).await;
                std::mem::take(&mut session.state.lock().expect("SessionState mutex").unconsumed)
            }
            None => Vec::new(),
        }
    }

    /// Check if there are no tracked agents.
    pub fn is_empty(&self) -> bool {
        self.agents.is_empty()
    }

    /// Drop the chat's entry on its `Done`. For a resident only the session
    /// that closed, so a late `Done` from a torn-down session can't drop (and
    /// orphan) its replacement.
    pub fn remove_finished(&mut self, topic: &str) {
        if let Some(AgentEntry::Resident(s)) = self.agents.get(topic) {
            if !s.state.lock().expect("SessionState mutex").closed {
                return;
            }
        }
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
            Some(AgentEntry::Resident(s)) => {
                let st = s.state.lock().expect("SessionState mutex");
                !s.reader.is_finished() && !st.closed && st.running()
            }
            None => false,
        }
    }

    /// Like `is_running`, minus turns that are only queued: whether a hard
    /// abort would interrupt work (queued turns are re-queued, not lost).
    pub fn is_working(&self, topic: &str) -> bool {
        match self.agents.get(topic) {
            Some(AgentEntry::Resident(s)) => {
                let st = s.state.lock().expect("SessionState mutex");
                !s.reader.is_finished() && !st.closed && st.working()
            }
            _ => self.is_running(topic),
        }
    }

    /// Deliver `req` as a new user turn to the chat's live resident process via
    /// stdin. `false` — the caller respawns instead — when there's no live
    /// session, its spawn knobs differ, or it's a slash command while a turn
    /// is in flight (that one interrupts, as before; a slash command queued
    /// behind other turns is held by the writer until it can run on its own).
    pub fn send_user_turn(&self, req: &SpawnRequest) -> bool {
        let Some(AgentEntry::Resident(s)) = self.agents.get(&req.chat_id) else {
            return false;
        };
        let (closed, in_flight) = {
            let st = s.state.lock().expect("SessionState mutex");
            (st.closed, st.turn_in_flight)
        };
        if closed || s.reader.is_finished() || s.knobs != SessionKnobs::of(req) {
            return false;
        }
        if in_flight
            && req.agent_kind.make_adapter().expands_slash_commands()
            && is_slash_command(&req.prompt)
        {
            return false;
        }
        s.queue_turn(req.prompt.clone(), false, req.user_timezone.as_deref());
        true
    }

    #[cfg(test)]
    pub fn unconsumed_prompts(&self, topic: &str) -> Vec<String> {
        match self.agents.get(topic) {
            Some(AgentEntry::Resident(s)) => s
                .state
                .lock()
                .expect("SessionState mutex")
                .unconsumed
                .iter()
                .map(|t| t.prompt.clone())
                .collect(),
            _ => Vec::new(),
        }
    }

    /// `spawn_agent(req)`, then queue each of `more` (resident only) behind its
    /// prompt as a separate user turn.
    pub fn spawn_agent_then(&mut self, req: SpawnRequest, more: Vec<String>) {
        let topic = req.chat_id.clone();
        let user_timezone = req.user_timezone.clone();
        self.spawn_agent(req);
        if let Some(AgentEntry::Resident(s)) = self.agents.get(&topic) {
            for prompt in more {
                s.queue_turn(prompt, false, user_timezone.as_deref());
            }
        }
    }

    /// Spawn a fresh agent process: resident kinds (claude) resume the chat's
    /// session with `req.prompt` as the first stdin turn; one-shot kinds keep
    /// the historic per-turn spawn. Branches on the adapter's `is_resident()`.
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

    /// Resident path: drop any stale entry (finished reader, or one the caller
    /// already aborted), spawn a fresh process (`--resume` when the chat has a
    /// session id) and queue `req.prompt` as its first turn.
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
            wake: handles.wake,
            knobs: SessionKnobs::of(&req),
            reader: handles.reader,
            cancel,
            state: handles.state,
        };
        session.queue_turn(
            req.prompt,
            req.agent_session_id.is_none(),
            req.user_timezone.as_deref(),
        );
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

        // Write prompt to a temp file so it never touches the shell command string
        let unique = format!(
            "{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        );
        let prompt_file = std::env::temp_dir().join(format!("zucchini-prompt-{}.txt", unique));

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

        // The adapter's prefixes (contracts on `AgentAdapter::turn_time_line` /
        // `prompt_file_preamble`) and the first-turn suffix land ONLY in the
        // plaintext temp file, never in the synced/persisted `messages` row.
        let prompt_to_write = turn_text(
            adapter.as_ref(),
            prompt,
            ctx.agent_session_id.is_none(),
            ctx.user_timezone,
            adapter.prompt_file_preamble(&ctx),
        );
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

        // Windows: launch the agent natively (no bash) when the adapter offers a
        // shell-free recipe. Unix keeps the login-shell path (byte-identical).
        let native = if cfg!(windows) {
            adapter.prepare_native(&ctx)
        } else {
            None
        };
        let mut cmd = if let Some(spawn) = native {
            info!(program = %spawn.program, kind = ?kind, "spawning agent natively (no shell)");
            match crate::shell::native_command(&spawn) {
                Ok(c) => c,
                Err(e) => {
                    error!("failed to build native agent command: {}", e);
                    let _ = tokio::fs::remove_file(&prompt_file).await;
                    fail_agent(&tx, &topic_clone, format!("failed to spawn agent: {}", e)).await;
                    return;
                }
            }
        } else {
            let cmd_string = match adapter.prepare_command(&ctx) {
                Ok(s) => s,
                Err(e) => {
                    error!("adapter prepare_command failed: {}", e);
                    let _ = tokio::fs::remove_file(&prompt_file).await;
                    fail_agent(&tx, &topic_clone, format!("agent prepare failed: {}", e)).await;
                    return;
                }
            };
            info!(shell = %crate::shell::user_login_shell(), kind = ?kind, "spawning agent via login shell");
            sys::login_shell_command(&cmd_string)
        };
        cmd.stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        // New process group so we can tear down shell + agent together.
        sys::configure_new_process_group(&mut cmd);
        // Adapter prompts reference both vars to invoke
        // `zucchini-spawner attach-file` / `schedule-message`. Exported on
        // the spawn (not via shell rc) so a stale PATH can't pick the wrong
        // binary — `current_exe()` is whatever launchd/systemd runs, which
        // is also what the RPC handler listens on.
        cmd.env("ZUCCHINI_CHAT_ID", &topic_clone);
        if let Ok(exe) = std::env::current_exe() {
            cmd.env("ZUCCHINI_SPAWNER_BIN", exe);
        }
        // One automatic respawn is allowed after a silent startup timeout (see
        // `startup_retry_allowed`); the retry attempt gets a doubled deadline.
        let mut attempt: u32 = 0;
        let (mut child, agent_started, stderr_handle, has_result) = 'spawn: loop {
            let mut child = match cmd.spawn() {
                Ok(child) => child,
                Err(e) => {
                    error!("failed to spawn agent: {}", e);
                    let _ = tokio::fs::remove_file(&prompt_file).await;
                    fail_agent(&tx, &topic_clone, format!("failed to spawn agent: {}", e)).await;
                    return;
                }
            };

            let group = match sys::ProcessGroup::adopt(&child) {
                Ok(g) => g,
                Err(e) => {
                    error!("failed to adopt agent process group: {}", e);
                    let _ = child.kill().await;
                    let _ = tokio::fs::remove_file(&prompt_file).await;
                    fail_agent(&tx, &topic_clone, format!("failed to spawn agent: {}", e)).await;
                    return;
                }
            };

            // Shared flag: flipped to true once we receive the first stdout line from the agent.
            // The stderr task uses this to decide whether to buffer (startup noise) or warn (runtime).
            let agent_started = Arc::new(AtomicBool::new(false));
            let stderr_seen = Arc::new(AtomicBool::new(false));

            // Read stderr in a separate task (see `spawn_stderr_reader`): buffer
            // startup lines silently, keep a rolling tail, warn-log the rest. The
            // capture is surfaced to the user only on a failed exit.
            let stderr_handle = child.stderr.take().map(|stderr| {
                spawn_stderr_reader(
                    stderr,
                    agent_started.clone(),
                    stderr_seen.clone(),
                    topic_clone.clone(),
                )
            });

            let mut has_result = false;

            if let Some(stdout) = child.stdout.take() {
                let reader = BufReader::new(stdout);
                let mut lines = reader.lines();
                let startup_timeout = AGENT_STARTUP_TIMEOUT * (attempt + 1);
                let startup_deadline = tokio::time::Instant::now() + startup_timeout;

                // `cancelled` is set by the inner per-send select when the outer
                // cancellation token fires mid-emit (slow writer back-pressures the mpsc
                // and a chatty turn can emit multiple events per line); we break out and
                // run the same SIGTERM/SIGKILL cleanup as the outer cancel arm.
                let mut cancelled_mid_send = false;
                loop {
                    tokio::select! {
                        _ = token_clone.cancelled() => {
                            warn!(topic = %topic_clone, "agent cancelled, tearing down process group");
                            group.terminate(&mut child, AGENT_SIGTERM_GRACE).await;
                            let _ = tokio::fs::remove_file(&prompt_file).await;
                            // Don't send Done — caller publishes INTERRUPTED_RESULT and has
                            // already removed our entry from the map.
                            return;
                        }
                        _ = tokio::time::sleep_until(startup_deadline), if !agent_started.load(Ordering::Relaxed) => {
                            group.kill(&mut child).await;
                            if startup_retry_allowed(attempt, &stderr_seen) {
                                warn!(topic = %topic_clone, "agent produced no output within {:?}, retrying once", startup_timeout);
                                let _ = child.wait().await;
                                attempt += 1;
                                continue 'spawn;
                            }
                            error!(topic = %topic_clone, "agent produced no output within {:?}, killing", startup_timeout);
                            let _ = tx.send(AgentResponse::Line {
                                topic: topic_clone.clone(),
                                content: startup_timeout_message(startup_timeout, attempt > 0),
                            }).await;
                            break;
                        }
                        line_result = lines.next_line() => {
                            match line_result {
                                Ok(Some(line)) => {
                                    // First stdout line means the agent is alive — silences any
                                    // later stderr buffering and stops the startup watchdog.
                                    agent_started.store(true, Ordering::Relaxed);
                                    ANY_AGENT_STARTED.store(true, Ordering::Relaxed);

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
                    warn!(topic = %topic_clone, "agent cancelled mid-emit, tearing down process group");
                    group.terminate(&mut child, AGENT_SIGTERM_GRACE).await;
                    let _ = tokio::fs::remove_file(&prompt_file).await;
                    return;
                }
            }

            break 'spawn (child, agent_started, stderr_handle, has_result);
        };

        // Surface stderr as the failure reason when the process never started,
        // or when it died mid-turn without a `result` (e.g. cursor printing
        // "You've hit your usage limit" to stderr after its init frame). The
        // cancel arms `return` earlier, so reaching here is always a spontaneous
        // exit — `has_result` is the clean-exit signal. Without this the user
        // only sees the bare INTERRUPTED_RESULT the `Done` below triggers.
        if let Some(h) = stderr_handle {
            if let Ok(cap) = h.await {
                let started = agent_started.load(Ordering::Relaxed);
                if let Some(msg) = stderr_failure_message(&cap, started, has_result) {
                    error!(topic = %topic_clone, "{}", msg);
                    let _ = tx
                        .send(AgentResponse::Line {
                            topic: topic_clone.clone(),
                            content: msg,
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
/// writer task (`spawn_stdin_writer`), and drives stdout in a reader task
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
    let wake = Arc::new(Notify::new());
    // `spawn_resident` queues the first turn right after this returns.
    let state = Arc::new(StdMutex::new(SessionState::default()));

    let state_for_reader = state.clone();
    let wake_for_writer = wake.clone();

    let SpawnRequest {
        chat_id: topic,
        prompt: _,
        project_path,
        worktree,
        agent_session_id,
        agent_kind,
        is_sandboxed,
        model,
        user_timezone: _,
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
        };
        let native = if cfg!(windows) {
            adapter.prepare_session_native(&ctx)
        } else {
            None
        };
        let mut cmd = if let Some(spawn) = native {
            info!(program = %spawn.program, "spawning resident agent natively (no shell)");
            match crate::shell::native_command(&spawn) {
                Ok(c) => c,
                Err(e) => {
                    error!("failed to build native resident command: {}", e);
                    close_session(&state_for_reader);
                    fail_agent(&tx, &topic, format!("failed to spawn agent: {}", e)).await;
                    return;
                }
            }
        } else {
            let cmd_string = match adapter.prepare_session_command(&ctx) {
                Ok(s) => s,
                Err(e) => {
                    error!("adapter prepare_session_command failed: {}", e);
                    close_session(&state_for_reader);
                    fail_agent(&tx, &topic, format!("agent prepare failed: {}", e)).await;
                    return;
                }
            };
            info!(shell = %crate::shell::user_login_shell(), "spawning resident agent via login shell");
            sys::login_shell_command(&cmd_string)
        };
        cmd.stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        sys::configure_new_process_group(&mut cmd);
        cmd.env("ZUCCHINI_CHAT_ID", &topic);
        if let Ok(exe) = std::env::current_exe() {
            cmd.env("ZUCCHINI_SPAWNER_BIN", exe);
        }
        // One automatic respawn is allowed after a silent startup timeout (see
        // `startup_retry_allowed`); the retry attempt gets a doubled deadline.
        let mut attempt: u32 = 0;
        let (mut child, agent_started, stderr_handle, writer_handle, cancelled) = 'spawn: loop {
            let mut child = match cmd.spawn() {
                Ok(c) => c,
                Err(e) => {
                    error!("failed to spawn resident agent: {}", e);
                    close_session(&state_for_reader);
                    fail_agent(&tx, &topic, format!("failed to spawn agent: {}", e)).await;
                    return;
                }
            };

            // Capture the group right after spawn and hold it for the resident's
            // whole lifetime — the resident `child` lives across turns in this same
            // task scope, so `group` (same scope) outlives it by construction.
            let group = match sys::ProcessGroup::adopt(&child) {
                Ok(g) => g,
                Err(e) => {
                    error!("failed to adopt resident agent process group: {}", e);
                    let _ = child.kill().await;
                    close_session(&state_for_reader);
                    fail_agent(&tx, &topic, format!("failed to spawn agent: {}", e)).await;
                    return;
                }
            };

            // Writer task: owns ChildStdin, the single writer of stdin frames (see
            // `spawn_stdin_writer`).
            let writer_handle = child.stdin.take().map(|stdin| {
                spawn_stdin_writer(
                    stdin,
                    state_for_reader.clone(),
                    wake_for_writer.clone(),
                    cancel.clone(),
                )
            });

            let agent_started = Arc::new(AtomicBool::new(false));
            let stderr_seen = Arc::new(AtomicBool::new(false));
            let stderr_handle = child.stderr.take().map(|stderr| {
                spawn_stderr_reader(
                    stderr,
                    agent_started.clone(),
                    stderr_seen.clone(),
                    topic.clone(),
                )
            });

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
                let startup_timeout = AGENT_STARTUP_TIMEOUT * (attempt + 1);
                let startup_deadline = tokio::time::Instant::now() + startup_timeout;
                'read: loop {
                    tokio::select! {
                        _ = cancel.cancelled() => {
                            warn!(topic = %topic, "resident session cancelled, tearing down process group");
                            cancelled = true;
                            group.terminate(&mut child, AGENT_SIGTERM_GRACE).await;
                            break 'read;
                        }
                        _ = tokio::time::sleep_until(startup_deadline), if !agent_started.load(Ordering::Relaxed) => {
                            group.kill(&mut child).await;
                            if startup_retry_allowed(attempt, &stderr_seen) {
                                warn!(topic = %topic, "resident agent produced no output within {:?}, retrying once", startup_timeout);
                                if let Some(wh) = &writer_handle {
                                    wh.abort();
                                }
                                let _ = child.wait().await;
                                attempt += 1;
                                continue 'spawn;
                            }
                            error!(topic = %topic, "resident agent produced no output within {:?}, killing", startup_timeout);
                            let _ = tx.send(AgentResponse::Line {
                                topic: topic.clone(),
                                content: startup_timeout_message(startup_timeout, attempt > 0),
                            }).await;
                            break 'read;
                        }
                        line_result = lines.next_line() => {
                            match line_result {
                                Ok(Some(line)) => {
                                    agent_started.store(true, Ordering::Relaxed);
                                    ANY_AGENT_STARTED.store(true, Ordering::Relaxed);
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
                                        // A slash turn held by the writer may be due now.
                                        if is_result {
                                            wake_for_writer.notify_one();
                                        }
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
                                            | AgentEvent::TaskFinished(_)
                                            | AgentEvent::UserTurnConsumed(_) => {}
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

            break 'spawn (
                child,
                agent_started,
                stderr_handle,
                writer_handle,
                cancelled,
            );
        };
        close_session(&state_for_reader);

        // Surface stderr on a failed exit (same contract as the one-shot path).
        // A deliberate teardown (`cancelled`) or an exit while fully idle is a
        // clean exit and shows nothing; a never-started or mid-turn-crash exit
        // surfaces its startup buffer / tail.
        if let Some(h) = stderr_handle {
            if let Ok(cap) = h.await {
                let started = agent_started.load(Ordering::Relaxed);
                let clean_exit = cancelled
                    || state_for_reader
                        .lock()
                        .expect("SessionState mutex")
                        .is_idle();
                if let Some(msg) = stderr_failure_message(&cap, started, clean_exit) {
                    error!(topic = %topic, "{}", msg);
                    let _ = tx
                        .send(AgentResponse::Line {
                            topic: topic.clone(),
                            content: msg,
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
        wake,
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
/// by the outer-cancel and mid-send-cancel arms. The `sys::ProcessGroup`
/// teardown applies it (unix: SIGTERM grace before SIGKILL; Windows: there is
/// no SIGTERM, so the grace is spent waiting for the child to exit on stdin-EOF
/// before `TerminateJobObject`).
const AGENT_SIGTERM_GRACE: Duration = Duration::from_secs(5);

/// How many trailing stderr lines we keep to surface as a failure diagnostic
/// when the process dies without a clean result.
const STDERR_TAIL_LINES: usize = 40;

/// Captured stderr from an agent process. `startup` holds the lines emitted
/// before the agent produced its first stdout line (used for "never started"
/// diagnostics); `tail` is a rolling window of the most recent lines overall.
/// The tail exists because some CLIs (e.g. cursor-agent) emit a `system/init`
/// frame to stdout — which flips `agent_started` true — and only *then* print
/// the real failure ("You've hit your usage limit") to stderr; the startup
/// buffer misses it, so without the tail the user would see a bare
/// "Agent interrupted" with no reason.
struct CapturedStderr {
    startup: Vec<String>,
    tail: VecDeque<String>,
}

/// Spawn the resident stdin writer task: write each queued
/// `SessionState::unconsumed` frame it hasn't written yet, then sleep until
/// `wake`. A startup-timeout retry's fresh writer re-writes them all to the new
/// process. Exits on `cancel`, closing stdin; claude ignores stdin EOF, so real
/// teardown is the SIGTERM.
fn spawn_stdin_writer(
    mut stdin: tokio::process::ChildStdin,
    state: Arc<StdMutex<SessionState>>,
    wake: Arc<Notify>,
    cancel: CancellationToken,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        use tokio::io::AsyncWriteExt;
        let mut written: HashSet<String> = HashSet::new();
        loop {
            let frames = frames_to_write(&state.lock().expect("SessionState mutex"), &mut written);
            for frame in frames {
                if stdin.write_all(frame.as_bytes()).await.is_err() || stdin.flush().await.is_err()
                {
                    return;
                }
            }
            tokio::select! {
                _ = wake.notified() => {}
                _ = cancel.cancelled() => return,
            }
        }
    })
}

/// Unwritten frames of `st.unconsumed`, in order. A slash-command turn is held
/// (with everything behind it) until it heads the queue with no turn in
/// flight: written earlier, claude would inject it into the running turn as
/// plain text at the next tool boundary.
fn frames_to_write(st: &SessionState, written: &mut HashSet<String>) -> Vec<String> {
    let mut frames = Vec::new();
    for (i, t) in st.unconsumed.iter().enumerate() {
        if t.own_turn && (i > 0 || st.turn_in_flight) {
            break;
        }
        if written.insert(t.uuid.clone()) {
            frames.push(t.frame.clone());
        }
    }
    frames
}

fn close_session(state: &Arc<StdMutex<SessionState>>) {
    state.lock().expect("SessionState mutex").closed = true;
}

/// Spawn the stderr reader task. Buffers startup lines silently (capped), keeps
/// a rolling tail of the last `STDERR_TAIL_LINES` lines, and `warn!`-logs every
/// post-startup line. Shared by the one-shot and resident spawn paths.
fn spawn_stderr_reader(
    stderr: tokio::process::ChildStderr,
    agent_started: Arc<AtomicBool>,
    stderr_seen: Arc<AtomicBool>,
    topic: String,
) -> tokio::task::JoinHandle<CapturedStderr> {
    tokio::spawn(async move {
        let reader = BufReader::new(stderr);
        let mut lines = reader.lines();
        let mut startup: Vec<String> = Vec::new();
        let mut tail: VecDeque<String> = VecDeque::with_capacity(STDERR_TAIL_LINES);
        while let Ok(Some(line)) = lines.next_line().await {
            stderr_seen.store(true, Ordering::Relaxed);
            if agent_started.load(Ordering::Relaxed) {
                warn!(topic = %topic, "agent stderr: {}", line);
            } else if startup.len() < 200 {
                startup.push(line.clone());
            }
            if tail.len() == STDERR_TAIL_LINES {
                tail.pop_front();
            }
            tail.push_back(line);
        }
        CapturedStderr { startup, tail }
    })
}

/// Build the diagnostic to surface when an agent process exits, or `None` when
/// there's nothing useful to show. `started` is the `agent_started` latch;
/// `clean_exit` is true when the turn completed normally (one-shot: emitted a
/// `result`; resident: died fully idle or via a deliberate teardown) — in that
/// case we never surface stderr. A process that never started reports its
/// startup buffer; one that started but died mid-turn reports its tail.
fn stderr_failure_message(cap: &CapturedStderr, started: bool, clean_exit: bool) -> Option<String> {
    if !started {
        if cap.startup.is_empty() {
            return None;
        }
        return Some(format!(
            "Error: agent failed to start.\n{}",
            cap.startup.join("\n")
        ));
    }
    if clean_exit {
        return None;
    }
    let tail: Vec<&str> = cap
        .tail
        .iter()
        .map(|l| l.trim_end())
        .filter(|l| !l.is_empty())
        .collect();
    if tail.is_empty() {
        return None;
    }
    Some(format!(
        "Error: agent exited unexpectedly.\n{}",
        tail.join("\n")
    ))
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
    use super::{frames_to_write, reduce, AgentEvent, QueuedTurn, SessionState};
    use std::collections::HashSet;

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
            ..Default::default()
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
            ..Default::default()
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
            ..Default::default()
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

    fn queued(uuid: &str) -> QueuedTurn {
        QueuedTurn {
            uuid: uuid.into(),
            prompt: String::new(),
            frame: uuid.into(),
            own_turn: false,
        }
    }

    #[test]
    fn a_slash_turn_is_written_only_at_the_head_with_no_turn_in_flight() {
        let mut s = SessionState::default();
        let slash = QueuedTurn {
            own_turn: true,
            ..queued("b")
        };
        s.unconsumed = vec![queued("a"), slash, queued("c")];
        let mut written = HashSet::new();
        assert_eq!(frames_to_write(&s, &mut written), vec!["a"]);
        s.unconsumed.remove(0);
        s.turn_in_flight = true;
        assert!(frames_to_write(&s, &mut written).is_empty());
        s.turn_in_flight = false;
        assert_eq!(frames_to_write(&s, &mut written), vec!["b", "c"]);
    }

    const RESULT: AgentEvent = AgentEvent::Result {
        origin_is_task: false,
    };

    /// Queued mid-turn: consumed at the next tool boundary and answered by the
    /// running turn's single `result`.
    #[test]
    fn injected_turn_echo_then_one_result_goes_idle() {
        let mut s = SessionState {
            turn_in_flight: true,
            ..Default::default()
        };
        s.unconsumed.push(queued("b"));
        reduce(&mut s, &AgentEvent::UserTurnConsumed("b".into()));
        assert!(s.unconsumed.is_empty() && s.running());
        reduce(&mut s, &RESULT);
        assert!(s.is_idle());
    }

    /// Queued after the last tool call: the running turn's `result` lands while
    /// the turn is still unconsumed, then it runs as a turn of its own.
    #[test]
    fn result_with_unconsumed_turn_stays_running_until_its_own_result() {
        let mut s = SessionState {
            turn_in_flight: true,
            ..Default::default()
        };
        s.unconsumed.push(queued("b"));
        reduce(&mut s, &RESULT);
        assert!(s.running(), "turn b is not consumed yet");
        reduce(&mut s, &AgentEvent::UserTurnConsumed("b".into()));
        assert!(s.turn_in_flight);
        reduce(&mut s, &RESULT);
        assert!(s.is_idle());
    }

    #[test]
    fn echo_of_an_unknown_uuid_is_ignored() {
        let mut s = SessionState::default();
        reduce(&mut s, &AgentEvent::UserTurnConsumed("ghost".into()));
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
            let state = Arc::new(StdMutex::new(SessionState::default()));
            exp.lock()
                .unwrap()
                .insert(req.chat_id.clone(), state.clone());
            let reader = tokio::spawn(async move { cancel.cancelled().await });
            ResidentHandles {
                wake: Arc::new(Notify::new()),
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
        st.unconsumed.clear();
        st.live_tasks.insert("t1".to_string());
    }

    #[tokio::test(start_paused = true)]
    async fn a_closed_session_takes_no_turns_and_only_it_is_removed_on_done() {
        let (mut sup, exposed) = supervisor_with_exposed_sessions();
        sup.spawn_agent(claude_req("chat"));
        sup.remove_finished("chat");
        assert!(
            sup.is_running("chat"),
            "a live session survives a stale Done"
        );

        exposed.lock().unwrap()["chat"].lock().unwrap().closed = true;
        assert!(!sup.is_running("chat"));
        assert!(!sup.send_user_turn(&claude_req("chat")));
        sup.remove_finished("chat");
        assert!(sup.is_empty());
    }

    #[tokio::test(start_paused = true)]
    async fn abort_returns_the_turns_left_unconsumed() {
        let (mut sup, _exposed) = supervisor_with_exposed_sessions();
        sup.spawn_agent(claude_req("chat"));
        let second = SpawnRequest {
            prompt: "second".to_string(),
            ..claude_req("chat")
        };
        assert!(sup.send_user_turn(&second));
        let prompts: Vec<String> = sup
            .abort_agent("chat")
            .await
            .into_iter()
            .map(|t| t.prompt)
            .collect();
        assert_eq!(prompts, vec!["hi", "second"]);
    }

    fn set_idle(exposed: &Exposed, chat: &str) {
        let g = exposed.lock().unwrap();
        let mut st = g[chat].lock().unwrap();
        st.turn_in_flight = false;
        st.unconsumed.clear();
    }

    /// A turn for a live session with the same knobs is queued on its stdin
    /// (uuid-stamped, time line prefixed like a respawn's) instead of respawning.
    #[tokio::test(start_paused = true)]
    async fn send_user_turn_queues_on_the_live_session() {
        let (mut sup, exposed) = supervisor_with_exposed_sessions();
        sup.spawn_agent(claude_req("chat"));
        let first_state = exposed.lock().unwrap()["chat"].clone();

        let next = SpawnRequest {
            prompt: "second".to_string(),
            agent_session_id: Some("sid-1".to_string()),
            user_timezone: Some("Asia/Bangkok".to_string()),
            ..claude_req("chat")
        };
        assert!(sup.send_user_turn(&next));
        assert!(
            Arc::ptr_eq(&first_state, &exposed.lock().unwrap()["chat"]),
            "no respawn"
        );
        assert_eq!(sup.unconsumed_prompts("chat"), vec!["hi", "second"]);
        let st = first_state.lock().unwrap();
        let frame: serde_json::Value = serde_json::from_str(&st.unconsumed[1].frame).unwrap();
        assert_eq!(frame["uuid"], st.unconsumed[1].uuid.as_str());
        let text = frame["message"]["content"][0]["text"].as_str().unwrap();
        assert!(
            text.contains("current local time") && text.ends_with("second"),
            "{text}"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn send_user_turn_refuses_other_knobs_and_a_mid_turn_slash_command() {
        let (mut sup, exposed) = supervisor_with_exposed_sessions();
        assert!(!sup.send_user_turn(&claude_req("chat")), "no session");
        sup.spawn_agent(claude_req("chat"));

        let other_model = SpawnRequest {
            model: Some("opus".to_string()),
            ..claude_req("chat")
        };
        assert!(!sup.send_user_turn(&other_model));

        let compact = SpawnRequest {
            prompt: "/compact".to_string(),
            ..claude_req("chat")
        };
        assert!(
            sup.send_user_turn(&compact),
            "queued behind the first turn; the writer holds it"
        );
        exposed.lock().unwrap()["chat"]
            .lock()
            .unwrap()
            .turn_in_flight = true;
        assert!(!sup.send_user_turn(&compact), "a turn is in flight");
        set_idle(&exposed, "chat");
        assert!(
            sup.send_user_turn(&compact),
            "a turn of its own gets expanded"
        );
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

#[cfg(test)]
mod stderr_diagnostic_tests {
    use super::*;

    fn cap(startup: &[&str], tail: &[&str]) -> CapturedStderr {
        CapturedStderr {
            startup: startup.iter().map(|s| s.to_string()).collect(),
            tail: tail.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn never_started_reports_startup_buffer() {
        let c = cap(
            &["zsh: command not found: cursor-agent"],
            &["zsh: command not found: cursor-agent"],
        );
        let msg = stderr_failure_message(&c, false, false).expect("startup failure surfaced");
        assert!(msg.starts_with("Error: agent failed to start."));
        assert!(msg.contains("command not found"));
    }

    #[test]
    fn never_started_but_empty_stderr_is_silent() {
        assert!(stderr_failure_message(&cap(&[], &[]), false, false).is_none());
    }

    #[test]
    fn started_then_died_midturn_surfaces_tail() {
        // The cursor no-subscription case: init frame flips `started` true, then
        // the usage-limit line lands on stderr and the process exits non-zero
        // without a `result` (clean_exit = false).
        let c = cap(
            &[],
            &["S: You've hit your usage limit Get Cursor Pro for more Agent usage."],
        );
        let msg = stderr_failure_message(&c, true, false).expect("mid-turn failure surfaced");
        assert!(msg.starts_with("Error: agent exited unexpectedly."));
        assert!(msg.contains("usage limit"));
    }

    #[test]
    fn clean_exit_never_surfaces_stderr() {
        // A normal turn that emitted a `result` (or a deliberate teardown) must
        // not surface benign stderr noise.
        let c = cap(&[], &["some benign warning"]);
        assert!(stderr_failure_message(&c, true, true).is_none());
    }

    #[test]
    fn started_died_with_only_blank_stderr_is_silent() {
        assert!(stderr_failure_message(&cap(&[], &["", "  ", "\t"]), true, false).is_none());
    }
}
