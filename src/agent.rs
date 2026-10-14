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
use crate::prune::PendingPrunes;
use crate::sys;
use crate::user_prompt::UserSystemPrompt;
use crate::watch::WatchDef;

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

/// How many background tasks / monitors the chat's resident has armed right now,
/// plus the codex shell commands still running (0 when it has no resident). The control task's `prune-context` guard: a
/// prune hard-restarts the agent and `--resume` does not re-arm in-process
/// tasks, so a non-zero count refuses the prune unless it was forced.
pub fn live_task_count(sessions: &LiveSessions, topic: &str) -> usize {
    sessions
        .lock()
        .expect("LiveSessions mutex")
        .get(topic)
        .map(|st| st.lock().expect("SessionState mutex").prune_victims().len())
        .unwrap_or(0)
}

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
///   - `running = in_turn || user_queued || !live_tasks.is_empty()`
///   - `waiting = running && !in_turn && !user_queued`
///   - `in_turn = turn_in_flight || self_turn`
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
    /// A consumed user turn is in flight: set on its `UserTurnConsumed` (or a
    /// peer session's `PeerTurnConsumed`) echo,
    /// cleared on a user-turn `result` (origin absent — `Result{origin_is_task:false}`).
    pub turn_in_flight: bool,
    /// The CLI started a turn that isn't one of our queued ones (a self-fired
    /// `CronCreate` prompt is never echoed); cleared by the next `result` of
    /// any origin, so it can't outlive a turn.
    pub self_turn: bool,
    /// Armed background tasks in arm order (appended on `TaskStarted`, retained
    /// out on `TaskFinished`). A `Vec`, not a map: it never exceeds a handful of
    /// entries and the client renders it in arm order.
    pub live_tasks: Vec<LiveTask>,
    /// `(id, command)` of codex shell commands still running. Only the prune
    /// guard reads them: they don't make the session busy.
    pub(crate) live_commands: Vec<(String, String)>,
    /// User turns queued for stdin and not yet echoed as consumed, in order.
    /// Registered before the write, so a racing `result` can't read Idle.
    pub unconsumed: Vec<QueuedTurn>,
    /// The consumed turn in flight is a command turn (`/compact`): nothing else
    /// is written until its `result` — a turn injected into a compaction has no
    /// defined outcome.
    pub(crate) command_in_flight: bool,
    /// The reader left its stdout loop: the process is exiting and takes no
    /// more turns. Set before its `Done`.
    pub closed: bool,
    /// When the reader last emitted the idle edge; cleared by a busy edge.
    /// Only a resident with one can be evicted (`warm.rs`).
    pub idle_since: Option<tokio::time::Instant>,
    /// The process group leader, for memory measurement.
    pub pid: Option<u32>,
    /// Uuids of internal `/rename` turns the spawner queued for itself and the
    /// CLI hasn't started yet. Several can coexist: a chat renamed twice in a
    /// row queues one per title.
    pub(crate) internal: Vec<String>,
    /// The internal turn the CLI started: its suppression window is open until
    /// the next `result`. Never busy.
    pub(crate) internal_running: Option<String>,
    /// What was last published as `chats.live_tasks`. Shared rather than local
    /// to the reader because the Supervisor publishes the queued count at the
    /// enqueue edge too (`live_tasks_change`) — a second dedup would let each
    /// side re-publish what the other already wrote, or swallow a change.
    pub(crate) published_live_tasks: LiveTasksDedup,
    /// What was last published as `chats.turn_running`; shared between the
    /// reader and the enqueue edge like `published_live_tasks`. A fresh session
    /// starts from false: the teardown before it cleared `agent_running`, and
    /// the backend resets `turn_running` with it.
    pub(crate) published_turn_running: bool,
    /// The context usage last reported, as the client's gauge shows it.
    pub(crate) context_tokens: Option<i64>,
    pub(crate) auto_compact: AutoCompact,
}

/// Where the session is in a spawner-issued compaction (`SessionState::after_turn`).
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub(crate) enum AutoCompact {
    #[default]
    Off,
    Compacting,
    /// The always-on notice that follows the compaction is queued or running.
    Noticing,
}

const AUTO_COMPACT_INSTRUCTIONS: &str = "Your system prompt and CLAUDE.md files stay in context: \
do not restate anything from them, in any section. Keep only what is nowhere else: the user's \
instructions from this conversation (verbatim), which background tasks are armed, and open \
commitments.";

const AUTO_COMPACT_NOTICE: &str = "[Zucchini] Your context was just auto-compacted. \
Re-read the files your instructions point to, check that your armed background tasks still \
match them, then continue.";

/// One armed background task, as the client lists it when the user taps the
/// Thinking/Waiting pill (`chats.live_tasks`). Claude's `task_started` frame
/// carries no start time, so `started_at` is stamped when the frame arrives.
#[derive(Debug, Clone)]
pub struct LiveTask {
    pub id: String,
    pub description: String,
    pub task_type: String,
    pub subagent_type: Option<String>,
    /// Epoch milliseconds.
    pub started_at: i64,
}

impl LiveTask {
    /// What to call the task in spawner-side prose (the forced-prune note).
    fn label(&self) -> &str {
        if self.description.is_empty() {
            &self.id
        } else {
            &self.description
        }
    }
}

fn now_epoch_ms() -> i64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or_default()
}

#[derive(Debug, Clone)]
pub struct QueuedTurn {
    pub uuid: String,
    /// The user's prompt, re-queued by a hard restart that kills the process
    /// before it consumed the turn.
    pub prompt: String,
    /// What goes to the CLI (`turn_text`), encoded by the writer at write time.
    text: String,
    /// A command turn (`AgentAdapter::is_command_turn`): runs only as a turn
    /// of its own.
    own_turn: bool,
    /// `AgentAdapter::command_runs_alone`: nothing behind it is written until
    /// its `result`.
    alone: bool,
    /// Queued by the spawner, not the user (`/rename`): invisible to
    /// `running()` and dropped rather than re-queued by a respawn.
    internal: bool,
}

impl SessionState {
    pub fn running(&self) -> bool {
        self.in_foreground() || !self.live_tasks.is_empty()
    }
    /// A turn runs or one of the user's waits: busy beyond background tasks.
    /// This is `chats.turn_running`.
    pub(crate) fn in_foreground(&self) -> bool {
        self.in_turn() || self.user_queued()
    }
    /// The `chats.turn_running` value to publish now, or `None` when it is
    /// unchanged since the last publication.
    pub(crate) fn observe_turn_running(&mut self) -> Option<bool> {
        let now = self.in_foreground();
        (self.published_turn_running != now).then(|| {
            self.published_turn_running = now;
            now
        })
    }
    /// A turn of the USER's is queued. An internal `/rename` doesn't count: it
    /// is the spawner's own, so it doesn't make the session busy.
    fn user_queued(&self) -> bool {
        self.user_queued_count() > 0
    }
    /// How many of the user's turns are queued — the `q` of the `live_tasks`
    /// blob. Same predicate as `user_queued`, so an internal `/rename` is never
    /// counted as a message the user is waiting on.
    fn user_queued_count(&self) -> usize {
        self.unconsumed.iter().filter(|t| !t.internal).count()
    }
    #[cfg(test)]
    pub(crate) fn has_live_task(&self, id: &str) -> bool {
        self.live_tasks.iter().any(|t| t.id == id)
    }
    /// A turn runs: a consumed one of ours or a peer's, or a self-fired one.
    fn in_turn(&self) -> bool {
        self.turn_in_flight || self.self_turn
    }
    /// Mid-turn or hosting background tasks — what a hard abort interrupts.
    fn working(&self) -> bool {
        self.in_turn() || !self.live_tasks.is_empty()
    }
    /// The "Waiting" sub-state — busy with no turn in flight, only a background
    /// task / monitor still armed. NOT emitted on the wire (the client
    /// re-derives the Thinking-vs-Waiting label); kept solely to document and
    /// unit-test the FSM contract, so it's `#[cfg(test)]`.
    #[cfg(test)]
    pub fn waiting(&self) -> bool {
        self.running() && !self.in_turn() && !self.user_queued()
    }
    /// Fully Idle — safe to tear down, and the only state in which a queued prune
    /// may be applied (a respawn would otherwise kill armed monitors). Exactly the
    /// negation of `running()`.
    pub fn is_idle(&self) -> bool {
        !self.running()
    }
    /// What a prune restart would kill, by label: armed tasks, then running
    /// codex commands.
    fn prune_victims(&self) -> Vec<String> {
        let tasks = self.live_tasks.iter().map(|t| t.label().to_string());
        let commands = self.live_commands.iter().map(|(_, c)| c.clone());
        tasks.chain(commands).collect()
    }
    fn subagent_armed(&self) -> bool {
        self.live_tasks.iter().any(|t| t.task_type == "local_agent")
    }
    /// A restart would cut no turn: none runs or is queued and no subagent is
    /// armed. Background shells and monitors may be.
    fn between_turns(&self) -> bool {
        !self.closed && !self.in_foreground() && !self.subagent_armed()
    }
    /// Kept warm past its idle edge with nothing pending: evictable.
    fn warm_idle_since(&self) -> Option<tokio::time::Instant> {
        self.idle_since.filter(|_| !self.closed && self.is_idle())
    }
    /// An internal turn is running: its frames are being swallowed right now.
    pub(crate) fn internal_in_flight(&self) -> bool {
        self.internal_running.is_some()
    }
    pub(crate) fn queue_internal(&mut self, uuid: String) {
        self.internal.push(uuid);
    }
    /// Opens the suppression window when `uuid` is an internal turn of ours
    /// (queued or already running); reports whether it is.
    fn start_internal(&mut self, uuid: &str) -> bool {
        if self.internal_running.as_deref() != Some(uuid)
            && !self.internal.iter().any(|u| u == uuid)
        {
            return false;
        }
        self.internal.retain(|u| u != uuid);
        self.internal_running = Some(uuid.to_string());
        true
    }
    /// Fail open: anything other than the internal turn's own `result` ends
    /// the suppression window, so real content can never be swallowed.
    fn close_internal_window(&mut self) {
        self.internal_running = None;
    }
    /// The `chats.live_tasks` blob to publish now, or `None` when it is
    /// unchanged since the last publication (see `LiveTasksDedup`).
    pub fn observe_live_tasks(&mut self) -> Option<Option<String>> {
        let payload = live_tasks_payload(self);
        self.published_live_tasks.observe(payload)
    }

    /// The spawner turn owed after a turn ended: `/compact` once the context
    /// reached `threshold` with no subagent armed and no user turn queued
    /// (background shells are fine), then on an always-on chat the notice.
    /// The end of either of those turns never compacts again.
    fn after_turn(&mut self, threshold: Option<i64>, always_on: bool) -> Option<String> {
        if self.closed || self.in_turn() {
            return None;
        }
        match self.auto_compact {
            AutoCompact::Off => {
                let due = threshold.is_some_and(|t| self.context_tokens.is_some_and(|n| n >= t))
                    && !self.user_queued()
                    && !self.subagent_armed();
                if !due {
                    return None;
                }
                self.auto_compact = AutoCompact::Compacting;
                Some(format!("/compact {AUTO_COMPACT_INSTRUCTIONS}"))
            }
            AutoCompact::Compacting if self.unconsumed.iter().any(|t| t.alone) => None,
            AutoCompact::Noticing if !self.unconsumed.is_empty() => None,
            AutoCompact::Compacting if always_on => {
                self.auto_compact = AutoCompact::Noticing;
                Some(AUTO_COMPACT_NOTICE.to_string())
            }
            AutoCompact::Compacting | AutoCompact::Noticing => {
                self.auto_compact = AutoCompact::Off;
                None
            }
        }
    }

    /// `after_turn` for a threshold the user just set: an idle session already
    /// past it compacts now, not after its next turn. A compaction under way
    /// is left to its own turn ends.
    fn after_threshold_set(&mut self, threshold: Option<i64>) -> Option<String> {
        if self.auto_compact != AutoCompact::Off {
            return None;
        }
        self.after_turn(threshold, false)
    }
}

/// PURE event→state reducer — the single source of truth for the resident
/// FSM, factored out so it's unit-testable without spawning a process. The
/// reader task calls this for each adapter event; everything else (`Frame`,
/// `SessionIdHarvested`, …) leaves the run state untouched and is handled by
/// the reader directly.
///
/// `Result{origin_is_task:false}` ends the user turn; `Result{origin_is_task:
/// true}` is a background-task wake's result and must NOT clear the turn flag.
pub fn reduce(state: &mut SessionState, ev: &AgentEvent) {
    match ev {
        AgentEvent::Result { origin_is_task } => {
            state.close_internal_window();
            if !origin_is_task {
                state.turn_in_flight = false;
                state.command_in_flight = false;
            }
            state.self_turn = false;
        }
        AgentEvent::TurnStarted(uuid) => {
            // The command's assistant text precedes its echo, so the window has
            // to open here rather than on `UserTurnConsumed`.
            if !state.start_internal(uuid) && !state.unconsumed.iter().any(|t| &t.uuid == uuid) {
                state.self_turn = true;
            }
        }
        AgentEvent::TaskStarted {
            id,
            description,
            task_type,
            subagent_type,
        } => {
            if !state.live_tasks.iter().any(|t| &t.id == id) {
                state.live_tasks.push(LiveTask {
                    id: id.clone(),
                    description: description.clone(),
                    task_type: task_type.clone(),
                    subagent_type: subagent_type.clone(),
                    started_at: now_epoch_ms(),
                });
            }
        }
        AgentEvent::TaskFinished(id) => {
            state.live_tasks.retain(|t| &t.id != id);
        }
        AgentEvent::CommandStarted { id, command } => {
            if !state.live_commands.iter().any(|(i, _)| i == id) {
                state.live_commands.push((id.clone(), command.clone()));
            }
        }
        AgentEvent::CommandFinished(id) => {
            state.live_commands.retain(|(i, _)| i != id);
        }
        AgentEvent::UserTurnConsumed(uuid) => {
            let ours = state.start_internal(uuid);
            if !ours {
                state.close_internal_window();
            }
            if let Some(i) = state.unconsumed.iter().position(|t| &t.uuid == uuid) {
                state.command_in_flight = state.unconsumed.remove(i).alone;
                if !ours {
                    state.turn_in_flight = true;
                }
            }
        }
        AgentEvent::PeerTurnConsumed { .. } => {
            state.close_internal_window();
            state.turn_in_flight = true;
        }
        AgentEvent::ContextTokens(n) | AgentEvent::CompactBoundary(n) => {
            state.context_tokens = Some(*n);
        }
        // Non-state-bearing events.
        AgentEvent::Frame(_) | AgentEvent::SessionIdHarvested(_) | AgentEvent::ToolResult => {}
    }
}

/// Ceiling on the tasks carried in the `live_tasks` blob — the OLDEST ones,
/// which are the ones the user has been waiting on. The chats row re-streams on
/// every writeback, so a fat blob multiplies sync traffic and `bucket_data`.
const MAX_LIVE_TASKS: usize = 20;
/// Claude's `description` is a one-liner; cap it anyway so a pathological one
/// can't bloat every re-stream of the row.
const MAX_LIVE_TASK_DESCRIPTION_CHARS: usize = 200;

/// The PLAINTEXT `chats.live_tasks` blob for a session — `{"t":[…],"q":N}` —
/// or `None` (⇒ column NULL) when nothing is armed and no user turn is queued.
/// `t` is omitted when empty, `q` when 0; per task `i` id, `d` description,
/// `k` kind, `a` subagent type (omitted when absent), `s0` start epoch ms.
/// The prompts themselves never ride along: they're already in the timeline as
/// user bubbles right above the pill.
pub fn live_tasks_payload(state: &SessionState) -> Option<String> {
    let queued = state.user_queued_count();
    if state.live_tasks.is_empty() && queued == 0 {
        return None;
    }
    let mut blob = serde_json::Map::new();
    if !state.live_tasks.is_empty() {
        let tasks: Vec<serde_json::Value> = state
            .live_tasks
            .iter()
            .take(MAX_LIVE_TASKS)
            .map(live_task_json)
            .collect();
        blob.insert("t".to_string(), serde_json::Value::Array(tasks));
    }
    if queued > 0 {
        blob.insert("q".to_string(), serde_json::json!(queued));
    }
    Some(serde_json::Value::Object(blob).to_string())
}

/// `session`'s blob (`None`: no agent process, or nothing armed or queued in
/// it) with the chat's watches listed ahead of its tasks, under the same cap.
/// The daemon runs them, so they stay listed through every teardown.
pub fn live_tasks_with_watches(session: Option<String>, watches: &[WatchDef]) -> Option<String> {
    if watches.is_empty() {
        return session;
    }
    let mut blob: serde_json::Map<String, serde_json::Value> = session
        .and_then(|blob| serde_json::from_str(&blob).ok())
        .unwrap_or_default();
    let mut tasks: Vec<serde_json::Value> = watches
        .iter()
        .map(|w| {
            live_task_json(&LiveTask {
                id: w.id.to_string(),
                description: w.description.clone(),
                task_type: "watch".to_string(),
                subagent_type: None,
                started_at: w.created_at.saturating_mul(1000),
            })
        })
        .collect();
    if let Some(serde_json::Value::Array(armed)) = blob.remove("t") {
        tasks.extend(armed);
    }
    tasks.truncate(MAX_LIVE_TASKS);
    blob.insert("t".to_string(), serde_json::Value::Array(tasks));
    Some(serde_json::Value::Object(blob).to_string())
}

fn live_task_json(task: &LiveTask) -> serde_json::Value {
    let kind = match task.task_type.as_str() {
        "local_bash" => "bash",
        "local_agent" => "agent",
        other => other,
    };
    let description: String = task
        .description
        .chars()
        .take(MAX_LIVE_TASK_DESCRIPTION_CHARS)
        .collect();
    let mut obj = serde_json::Map::new();
    obj.insert("i".to_string(), serde_json::json!(task.id));
    obj.insert("d".to_string(), serde_json::json!(description));
    obj.insert("k".to_string(), serde_json::json!(kind));
    if let Some(subagent) = &task.subagent_type {
        obj.insert("a".to_string(), serde_json::json!(subagent));
    }
    obj.insert("s0".to_string(), serde_json::json!(task.started_at));
    serde_json::Value::Object(obj)
}

/// Change detection for `chats.live_tasks`, the same shape as
/// `LastTokensDedup`: the blob is recomputed after every event and only a
/// CHANGED payload is published (`Some(None)` = clear the column). The reader
/// has to be one of the observers — a task edge publishes no frame, so a
/// main-loop diff à la `alive_changes` would not run until the next 60s
/// eviction pass. Lives in `SessionState` because queueing a turn changes the
/// blob outside the reader.
#[derive(Debug, Default, Clone)]
pub struct LiveTasksDedup {
    last: Option<String>,
}

impl LiveTasksDedup {
    /// `Some(payload)` when it differs from what was last published (`Some(None)`
    /// = clear the column), `None` when nothing changed. A fresh session starts
    /// from NULL — the teardown that preceded it cleared the column — so an
    /// empty payload publishes nothing until something is actually armed (an
    /// internal `/rename` turn must write nothing at all).
    pub fn observe(&mut self, payload: Option<String>) -> Option<Option<String>> {
        if self.last == payload {
            return None;
        }
        self.last = payload.clone();
        Some(payload)
    }
}

/// Spawner-synthesized message row for a turn the user didn't send: another
/// claude session's `SendMessage`, or a watch (`watch.rs`). Older clients hide
/// unknown system subtypes.
pub fn peer_message_frame_json(name: &str, body: &str) -> String {
    serde_json::json!({
        "type": "system",
        "subtype": "peer_message",
        "name": name,
        "body": body,
    })
    .to_string()
}

/// Dropped while the internal `/rename` turn's suppression window is open: its
/// own output writes no message row, and a local command's usage must not move
/// the context gauge. Everything else passes — and a foreign echo or a peer
/// turn closes the window in `reduce`, so real content is never swallowed.
fn internal_turn_swallows(ev: &AgentEvent) -> bool {
    matches!(ev, AgentEvent::Frame(_) | AgentEvent::ContextTokens(_))
}

/// The suppression window, latched for one stdout LINE. One line yields several
/// events (a `result` line yields `Result` and then the result frame), the
/// window is read before each is reduced, and the `Result` closes it — so a
/// line that starts inside the window stays inside it to its last event.
#[derive(Default)]
pub(crate) struct InternalLine(bool);

impl InternalLine {
    pub(crate) fn swallows(&mut self, window_open: bool, ev: &AgentEvent) -> bool {
        self.0 |= window_open;
        self.0 && internal_turn_swallows(ev)
    }
}

const TURN_PART_SEPARATOR: &str = "\n\n---\n\n";

/// User-turn text shared by both lifecycles: the time line, then `preamble`,
/// then the prompt with the adapter's first-turn suffix appended after a
/// horizontal rule. Absent parts are skipped. `first_turn` == brand-new session
/// (no resume id). The time line rides the turn rather than the system prompt —
/// see `AgentAdapter::turn_time_line`. A command turn ships verbatim: the CLI
/// recognizes it only when the turn STARTS with it (and claude swallows a
/// trailing suffix as the command's arguments) — unless it would drop a
/// first-turn preamble, which is never sent again (codex's capabilities): then
/// it goes as text.
fn turn_text(
    adapter: &dyn crate::adapter::AgentAdapter,
    prompt: String,
    first_turn: bool,
    user_timezone: Option<&str>,
    preamble: Option<String>,
) -> String {
    if adapter.is_command_turn(&prompt) && !(first_turn && preamble.is_some()) {
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
                adapter.session_first_turn_preamble(),
            );
            assert_eq!(text, prompt);
        }
    }

    /// The verbatim path would cost a non-expanding adapter its one-per-chat
    /// capabilities preamble (and its prune suffix) for no slash expansion
    /// at all — this is the only call shape where those parts are ever `Some`.
    /// Codex runs only `/compact` as a command; `/init` stays text.
    #[test]
    fn a_slash_command_keeps_every_part_for_a_non_expanding_adapter() {
        for kind in [AgentKind::Gemini, AgentKind::Codex] {
            let adapter = kind.make_adapter();
            let suffix = adapter
                .first_turn_prompt_suffix()
                .expect("sets a first-turn suffix");
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
    }

    #[test]
    fn codex_compact_ships_verbatim() {
        let adapter = AgentKind::Codex.make_adapter();
        let text = turn_text(
            adapter.as_ref(),
            "/compact keep the plan".to_string(),
            false,
            Some("Asia/Bangkok"),
            None,
        );
        assert_eq!(text, "/compact keep the plan");
    }

    /// The capabilities preamble goes out on the first turn only: a first
    /// `/compact` carries it as text rather than lose it for good.
    #[test]
    fn a_first_turn_codex_compact_keeps_the_preamble() {
        let adapter = AgentKind::Codex.make_adapter();
        let preamble = adapter
            .session_first_turn_preamble()
            .expect("codex has one");
        let text = turn_text(
            adapter.as_ref(),
            "/compact".to_string(),
            true,
            None,
            Some(preamble.clone()),
        );
        assert!(text.starts_with(&preamble), "{text}");
        assert!(!adapter.is_command_turn(&text), "goes as text: {text}");
    }

    #[test]
    fn slash_command_matching_edges() {
        use crate::adapter::slash_command_name;
        assert_eq!(slash_command_name(" /compact"), Some("compact"));
        assert_eq!(slash_command_name("/plugin:cmd args"), Some("plugin:cmd"));
        assert_eq!(
            slash_command_name("/Users/ivan/notes.txt what is this?"),
            None
        );
        assert_eq!(slash_command_name("/ compact"), None);
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
    /// RESIDENT path only — what the session has armed changed (a task started
    /// or finished, or a user turn was queued/consumed). `tasks` is the
    /// plaintext blob (`live_tasks_payload`), `None` to clear the column.
    /// Emitted by the READER, deduped on the serialized payload — see
    /// `LiveTasksDedup`. `session` identifies the sender so a late edge from a
    /// torn-down session can't resurrect phantom tasks on its replacement
    /// (`Supervisor::is_current_session`).
    LiveTasks {
        topic: String,
        session: Arc<StdMutex<SessionState>>,
        tasks: Option<String>,
    },
    /// RESIDENT path only — `SessionState::in_foreground` changed
    /// (`chats.turn_running`). Sent after the frames of the stdout line that
    /// changed it, so a turn's `result` row is queued for the writer before
    /// its false edge. `session` as in `LiveTasks`.
    TurnRunning {
        topic: String,
        session: Arc<StdMutex<SessionState>>,
        running: bool,
    },
    /// RESIDENT path only — a turn ended with none in flight (background tasks
    /// may still be armed): the main loop's cue for `Supervisor::after_turn`.
    /// Sent before that `result`'s idle edge, so a turn it queues keeps the
    /// chat busy.
    TurnEnded {
        topic: String,
    },
}

impl AgentResponse {
    pub fn topic(&self) -> &str {
        match self {
            AgentResponse::Line { topic, .. }
            | AgentResponse::ContextTokens { topic, .. }
            | AgentResponse::CompactBoundary { topic, .. }
            | AgentResponse::SessionIdHarvested { topic, .. }
            | AgentResponse::Done { topic, .. }
            | AgentResponse::RunState { topic, .. }
            | AgentResponse::ToolResult { topic }
            | AgentResponse::TurnEnded { topic }
            | AgentResponse::TurnRunning { topic, .. }
            | AgentResponse::LiveTasks { topic, .. } => topic,
        }
    }
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
            AgentEvent::TaskStarted { .. }
            | AgentEvent::TaskFinished(_)
            | AgentEvent::CommandStarted { .. }
            | AgentEvent::CommandFinished(_)
            | AgentEvent::UserTurnConsumed(_)
            | AgentEvent::PeerTurnConsumed { .. }
            | AgentEvent::TurnStarted(_) => {
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
    /// The chat's title sanitized by `session_name::from_title` — the name the
    /// resident registers for peer addressing (`claude --name`). Recomputed on
    /// every spawn, so a rename that missed the live process still lands on the
    /// next one. `None` for a blank/undecryptable title.
    pub session_name: Option<String>,
    /// `ChatState::user_system_prompt`. Rendered (its files read) at spawn; a
    /// render error fails the turn with the reason in the chat.
    pub user_system_prompt: UserSystemPrompt,
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
    user_system_prompt: UserSystemPrompt,
}

impl SessionKnobs {
    fn of(req: &SpawnRequest) -> Self {
        Self {
            agent_kind: req.agent_kind,
            project_path: req.project_path.clone(),
            worktree: req.worktree,
            is_sandboxed: req.is_sandboxed,
            model: req.model.clone(),
            user_system_prompt: req.user_system_prompt.clone(),
        }
    }
}

/// `Some(running)` for a session `Supervisor::retire_stale_prompt` may abort.
fn stale_prompt_retirable(
    knobs: &SessionKnobs,
    st: &SessionState,
    current: &UserSystemPrompt,
    kill_tasks: bool,
) -> Option<bool> {
    let retirable = knobs.user_system_prompt != *current
        && st.between_turns()
        && (kill_tasks || st.live_tasks.is_empty());
    retirable.then(|| st.running())
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
    /// The name this process was last ASKED for: `claude --name` at spawn, then
    /// whatever an internal `/rename` turn requested. A chats row re-streams on
    /// every writeback, so this is what keeps the rename from being re-queued
    /// on each one.
    asked_name: Option<String>,
}

impl ResidentSession {
    fn queue_turn(&self, prompt: String, first_turn: bool, user_timezone: Option<&str>) {
        self.push_turn(prompt, first_turn, user_timezone, false);
    }

    /// A turn of the spawner's own (`/rename <name>`). `is_command_turn` makes
    /// it an `own_turn`, so the writer holds it until it heads the queue with
    /// no turn in flight and `turn_text` ships it verbatim — nothing special.
    fn queue_internal_turn(&self, name: &str) {
        self.push_turn(format!("/rename {name}"), false, None, true);
    }

    fn push_turn(
        &self,
        prompt: String,
        first_turn: bool,
        user_timezone: Option<&str>,
        internal: bool,
    ) {
        let adapter = self.knobs.agent_kind.make_adapter();
        let uuid = Uuid::now_v7().to_string();
        let preamble = first_turn
            .then(|| adapter.session_first_turn_preamble())
            .flatten();
        let text = turn_text(
            adapter.as_ref(),
            prompt.clone(),
            first_turn,
            user_timezone,
            preamble,
        );
        let mut st = self.state.lock().expect("SessionState mutex");
        if internal {
            st.queue_internal(uuid.clone());
        }
        st.unconsumed.push(QueuedTurn {
            own_turn: adapter.is_command_turn(&text),
            alone: adapter.command_runs_alone(&text),
            internal,
            uuid,
            prompt,
            text,
        });
        drop(st);
        self.wake.notify_one();
    }
}

/// Either a one-shot per-turn task (gemini/cursor/hermes/pi) or a resident
/// session (claude, codex). One-shot keeps the historic `(handle, token)` pair; the
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

/// Per-session spawn closure for RESIDENT adapters (claude, codex): spawns the
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
    /// Shared with the control-socket handler, which parks `prune-context`
    /// requests here. An aborted session never delivers their cue or `Done`,
    /// so `abort_agent` drops its chat's entry: a stale one would respawn it.
    pending_prunes: PendingPrunes,
    /// Chats whose resident process we last published as alive
    /// (`chats.agent_alive`); see `alive_changes`.
    alive_published: HashSet<String>,
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
            pending_prunes: PendingPrunes::default(),
            alive_published: HashSet::new(),
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

    pub fn pending_prunes(&self) -> PendingPrunes {
        self.pending_prunes.clone()
    }

    /// The background tasks / monitors live in the chat's resident session right
    /// now, by label (empty if no resident session). Read at prune apply-time to
    /// NAME in the respawn prompt what the restart terminated.
    pub fn live_task_labels(&self, topic: &str) -> Vec<String> {
        self.live_sessions
            .lock()
            .expect("LiveSessions mutex")
            .get(topic)
            .map(|st| st.lock().expect("SessionState mutex").prune_victims())
            .unwrap_or_default()
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
    /// prune / eviction / shutdown / `/stop`. Returns the resident's turns still unconsumed
    /// once its reader is joined (the final FSM, no racing echo).
    pub async fn abort_agent(&mut self, topic: &str) -> Vec<QueuedTurn> {
        let removed = self.agents.remove(topic);
        self.sync_live_sessions();
        let unconsumed = match removed {
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
                // An internal `/rename` is not re-queued: the respawn passes
                // `--name` from the current title anyway.
                std::mem::take(&mut session.state.lock().expect("SessionState mutex").unconsumed)
                    .into_iter()
                    .filter(|t| !t.internal)
                    .collect()
            }
            None => Vec::new(),
        };
        self.pending_prunes.lock().await.remove(topic);
        unconsumed
    }

    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.agents.is_empty()
    }

    pub fn has_agent(&self, topic: &str) -> bool {
        self.agents.contains_key(topic)
    }

    /// The chat has a resident process that still takes turns.
    pub fn has_open_resident(&self, topic: &str) -> bool {
        match self.agents.get(topic) {
            Some(AgentEntry::Resident(s)) => {
                !s.reader.is_finished() && !s.state.lock().expect("SessionState mutex").closed
            }
            _ => false,
        }
    }

    /// Some chat's agent is busy. Idle residents don't count.
    pub fn any_running(&self) -> bool {
        self.agents.keys().any(|topic| self.is_running(topic))
    }

    /// The chat's resident has a turn running or queued. Armed background
    /// tasks don't count.
    pub fn in_foreground(&self, topic: &str) -> bool {
        match self.agents.get(topic) {
            Some(AgentEntry::Resident(s)) => {
                s.state.lock().expect("SessionState mutex").in_foreground()
            }
            _ => false,
        }
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
    /// (an Idle resident is kept warm and is not running); for one-shot it's
    /// "task not finished".
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
    /// session, its spawn knobs differ (another user system prompt only once
    /// no turn would be cut), or it's a claude slash command while a
    /// turn is in flight (that one interrupts, as before; a command queued
    /// behind other turns — or any codex `/compact` — is held by the writer
    /// until it can run on its own). A running `/compact` is never interrupted:
    /// the command waits for it.
    pub fn send_user_turn(&self, req: &SpawnRequest) -> bool {
        self.deliver_turn(req, true)
    }

    /// `send_user_turn` for a message the backend promoted from the schedule
    /// (`messages.auto_sent`): its slash command never interrupts a running
    /// turn, the writer holds it until that turn's `result`.
    pub fn send_scheduled_turn(&self, req: &SpawnRequest) -> bool {
        self.deliver_turn(req, false)
    }

    fn deliver_turn(&self, req: &SpawnRequest, command_may_interrupt: bool) -> bool {
        let Some(AgentEntry::Resident(s)) = self.agents.get(&req.chat_id) else {
            return false;
        };
        let (closed, in_flight, between_turns) = {
            let st = s.state.lock().expect("SessionState mutex");
            (
                st.closed,
                st.in_turn() && !st.command_in_flight,
                st.between_turns(),
            )
        };
        let mut knobs = SessionKnobs::of(req);
        if !between_turns {
            // A stale prompt alone never cuts a turn: `retire_stale_prompt`
            // applies it once this one is over.
            knobs.user_system_prompt = s.knobs.user_system_prompt.clone();
        }
        if closed || s.reader.is_finished() || s.knobs != knobs {
            return false;
        }
        let adapter = req.agent_kind.make_adapter();
        if command_may_interrupt
            && in_flight
            && adapter.is_command_turn(&req.prompt)
            && adapter.command_interrupts_turn()
        {
            return false;
        }
        s.queue_turn(req.prompt.clone(), false, req.user_timezone.as_deref());
        true
    }

    /// Queue the auto-compaction turn the chat's resident owes after a turn
    /// ended (`SessionState::after_turn`), whatever knobs it was spawned with.
    pub fn after_turn(
        &self,
        topic: &str,
        threshold: Option<i64>,
        always_on: bool,
        user_timezone: Option<&str>,
    ) -> bool {
        self.queue_auto_compaction(topic, user_timezone, |st| {
            st.after_turn(threshold, always_on)
        })
    }

    /// Queue the compaction a threshold just set on an idle live resident
    /// owes (`SessionState::after_threshold_set`).
    pub fn after_threshold_set(
        &self,
        topic: &str,
        threshold: Option<i64>,
        user_timezone: Option<&str>,
    ) -> bool {
        self.queue_auto_compaction(topic, user_timezone, |st| st.after_threshold_set(threshold))
    }

    fn queue_auto_compaction(
        &self,
        topic: &str,
        user_timezone: Option<&str>,
        decide: impl FnOnce(&mut SessionState) -> Option<String>,
    ) -> bool {
        let Some(AgentEntry::Resident(s)) = self.agents.get(topic) else {
            return false;
        };
        if s.reader.is_finished() {
            return false;
        }
        let prompt = decide(&mut s.state.lock().expect("SessionState mutex"));
        let Some(prompt) = prompt else {
            return false;
        };
        info!(topic = %topic, prompt = %prompt, "auto-compaction turn");
        s.queue_turn(prompt, false, user_timezone);
        true
    }

    /// The chat's `chats.live_tasks` blob when it changed since the last
    /// publication, `None` when it didn't (or the chat has no resident). Called
    /// right after `send_user_turn` so the queued count reaches the clients at
    /// the enqueue edge — the reader publishes only on adapter events, and a
    /// queued turn produces none until the CLI echoes it.
    pub fn live_tasks_change(&self, topic: &str) -> Option<Option<String>> {
        let Some(AgentEntry::Resident(s)) = self.agents.get(topic) else {
            return None;
        };
        s.state
            .lock()
            .expect("SessionState mutex")
            .observe_live_tasks()
    }

    /// The chat's resident's `live_tasks` blob as it is now, whatever was
    /// last published.
    pub fn live_tasks(&self, topic: &str) -> Option<String> {
        let Some(AgentEntry::Resident(s)) = self.agents.get(topic) else {
            return None;
        };
        live_tasks_payload(&s.state.lock().expect("SessionState mutex"))
    }

    /// The chat's `chats.turn_running` when it changed since the last
    /// publication: the enqueue-edge twin of the reader's `TurnRunning`.
    pub fn turn_running_change(&self, topic: &str) -> Option<bool> {
        let Some(AgentEntry::Resident(s)) = self.agents.get(topic) else {
            return None;
        };
        s.state
            .lock()
            .expect("SessionState mutex")
            .observe_turn_running()
    }

    /// `session` is the chat's live resident session.
    pub fn is_current_session(&self, topic: &str, session: &Arc<StdMutex<SessionState>>) -> bool {
        matches!(self.agents.get(topic), Some(AgentEntry::Resident(s)) if Arc::ptr_eq(&s.state, session))
    }

    /// Residents kept warm past their idle edge with nothing pending, by topic.
    /// `footprint_kb`, `pinned` and `always_on` are filled in by the caller (`main.rs`), which
    /// owns the process measurements and the chat mirror.
    pub fn idle_residents(&self) -> Vec<crate::warm::IdleResident> {
        let mut idle: Vec<_> = self
            .agents
            .iter()
            .filter_map(|(topic, entry)| {
                let AgentEntry::Resident(s) = entry else {
                    return None;
                };
                let st = s.state.lock().expect("SessionState mutex");
                let idle_since = st.warm_idle_since().filter(|_| !s.reader.is_finished())?;
                Some(crate::warm::IdleResident {
                    topic: topic.clone(),
                    idle_since,
                    pid: st.pid,
                    footprint_kb: None,
                    pinned: false,
                    always_on: false,
                })
            })
            .collect();
        idle.sort_by(|a, b| a.topic.cmp(&b.topic));
        idle
    }

    /// Hard-abort the chat's resident if `check` passes on its state, and hand
    /// back what `check` returned. The check and the kill share the lock the
    /// reader reduces under, so the resident can't take a turn in between.
    async fn abort_resident_if<T>(
        &mut self,
        topic: &str,
        check: impl FnOnce(&SessionKnobs, &SessionState) -> Option<T>,
    ) -> Option<T> {
        let Some(AgentEntry::Resident(s)) = self.agents.get(topic) else {
            return None;
        };
        let passed = {
            let st = s.state.lock().expect("SessionState mutex");
            let passed = check(&s.knobs, &st)?;
            s.cancel.cancel();
            passed
        };
        self.abort_agent(topic).await;
        Some(passed)
    }

    /// Hard-abort an idle resident. `false` if it took a turn meanwhile.
    pub async fn evict_idle(&mut self, topic: &str) -> bool {
        self.abort_resident_if(topic, |_, st| st.warm_idle_since().map(drop))
            .await
            .is_some()
    }

    /// `retire_stale_prompt` would tear the chat's resident down.
    pub fn has_retirable_stale_prompt(
        &self,
        topic: &str,
        current: &UserSystemPrompt,
        kill_tasks: bool,
    ) -> bool {
        let Some(AgentEntry::Resident(s)) = self.agents.get(topic) else {
            return false;
        };
        let st = s.state.lock().expect("SessionState mutex");
        stale_prompt_retirable(&s.knobs, &st, current, kill_tasks).is_some()
    }

    /// Hard-abort the chat's resident if it was spawned with another user
    /// system prompt than `current` and is between turns; its next spawn
    /// renders the current one. With armed background tasks it goes only when
    /// `kill_tasks`. `Some(was_running)` if torn down.
    pub async fn retire_stale_prompt(
        &mut self,
        topic: &str,
        current: &UserSystemPrompt,
        kill_tasks: bool,
    ) -> Option<bool> {
        self.abort_resident_if(topic, |knobs, st| {
            stale_prompt_retirable(knobs, st, current, kill_tasks)
        })
        .await
    }

    /// Rename the live resident to `name` (chat renamed in the app) with an
    /// internal `/rename` turn — claude handles it locally and updates both its
    /// session registry and the in-memory name it stamps on outbound peer
    /// messages. No-op when the chat has no live resident or that process was
    /// already asked for `name` (its `--name` at spawn included; a chats row
    /// re-streams on every writeback). `name` is `None` for a title that
    /// sanitizes to nothing: the resident keeps its current name.
    /// Returns whether a turn was queued.
    pub fn rename_resident(&mut self, topic: &str, name: Option<&str>) -> bool {
        let Some(name) = name else {
            return false;
        };
        let Some(AgentEntry::Resident(s)) = self.agents.get_mut(topic) else {
            return false;
        };
        if !s.knobs.agent_kind.make_adapter().names_session()
            || s.reader.is_finished()
            || s.state.lock().expect("SessionState mutex").closed
        {
            return false;
        }
        if s.asked_name.as_deref() == Some(name) {
            return false;
        }
        s.asked_name = Some(name.to_string());
        s.queue_internal_turn(name);
        info!(chat_id = %topic, name = %name, "renaming live claude session");
        true
    }

    /// `chats.agent_alive` edges since the last call: a chat gained or lost
    /// its resident process. The main loop diffs once per pass, so an
    /// abort + respawn within one pass (prune, knob change) publishes nothing.
    pub fn alive_changes(&mut self) -> Vec<(String, bool)> {
        let alive: HashSet<String> = self
            .agents
            .iter()
            .filter(|(_, e)| matches!(e, AgentEntry::Resident(_)))
            .map(|(topic, _)| topic.clone())
            .collect();
        let mut changes: Vec<(String, bool)> = alive
            .difference(&self.alive_published)
            .map(|topic| (topic.clone(), true))
            .collect();
        changes.extend(
            self.alive_published
                .difference(&alive)
                .map(|topic| (topic.clone(), false)),
        );
        self.alive_published = alive;
        changes
    }

    /// Drop the chat from `alive_changes` without an edge: its row is gone, or
    /// the caller writes `agent_alive` itself.
    pub fn forget_alive(&mut self, topic: &str) {
        self.alive_published.remove(topic);
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

    /// Spawn a fresh agent process: resident kinds (claude, codex) resume the chat's
    /// session with `req.prompt` as the first stdin turn; one-shot kinds keep
    /// the historic per-turn spawn. Branches on the adapter's `is_resident()`.
    pub fn spawn_agent(&mut self, req: SpawnRequest) {
        if req.agent_kind.make_adapter().is_resident() {
            self.spawn_resident(req);
        } else {
            self.spawn_oneshot(req);
        }
    }

    /// One-shot per-turn spawn (gemini/cursor/hermes/pi) — historic path.
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
            asked_name: req.session_name.clone(),
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

    /// Cancel every agent and hand back the tasks to join.
    pub fn cancel_all(&mut self) -> Vec<(String, tokio::task::JoinHandle<()>)> {
        let agents: Vec<_> = self.agents.drain().collect();
        self.live_sessions
            .lock()
            .expect("LiveSessions mutex")
            .clear();
        if !agents.is_empty() {
            info!("shutting down {} running agent(s)", agents.len());
        }
        agents
            .into_iter()
            .map(|(topic, entry)| {
                info!(topic = %topic, "cancelling agent");
                match entry {
                    AgentEntry::OneShot { handle, cancel } => {
                        cancel.cancel();
                        (topic, handle)
                    }
                    AgentEntry::Resident(s) => {
                        s.cancel.cancel();
                        (topic, s.reader)
                    }
                }
            })
            .collect()
    }

    pub async fn shutdown_all(&mut self) {
        for (topic, handle) in self.cancel_all() {
            await_agent_exit(handle, &topic).await;
        }
        // After the joins: a dying agent's `prune-context` RPC may park late.
        self.pending_prunes.lock().await.clear();
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
        session_name: _,
        // Only resident adapters take one (`takes_user_system_prompt`).
        user_system_prompt: _,
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
        let native_stdin_file = native.as_ref().and_then(|s| s.prompt_stdin_file.clone());
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
                                // The killed child shared the stdin handle's file
                                // position, so it may have left it at EOF.
                                if let Some(path) = &native_stdin_file {
                                    match std::fs::File::open(path) {
                                        Ok(f) => {
                                            cmd.stdin(f);
                                        }
                                        Err(e) => warn!(topic = %topic_clone, error = %e, "failed to reopen prompt file for the retry"),
                                    }
                                }
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

        let _ = tx
            .send(AgentResponse::Done {
                topic: topic_clone,
                has_result,
            })
            .await;
    })
}

/// Default RESIDENT spawn implementation (claude, codex). Spawns one agent
/// process with stdin held open (`Stdio::piped()`), owns the `ChildStdin` in a single
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
        session_name,
        user_system_prompt,
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
        // Held only while busy, so an idle resident doesn't keep the Mac awake.
        let mut _power_assertion: Option<crate::power::AgentPowerAssertion> = None;
        // Shared with the stdin writer, which encodes turns and protocol frames
        // from what this reader has seen (codex: the open thread, the running
        // turn).
        let adapter: SharedAdapter = Arc::new(StdMutex::new(agent_kind.make_adapter()));
        let (exits_on_stdin_eof, writer_follows_every_line) = {
            let a = lock_adapter(&adapter);
            (a.exits_on_stdin_eof(), a.writer_follows_every_line())
        };
        info!(
            topic = %topic,
            resume = agent_session_id.is_some(),
            project_path = ?project_path,
            worktree,
            sandbox = is_sandboxed,
            "spawning resident agent"
        );

        let user_system_prompt = match user_system_prompt.render(project_path.as_deref()) {
            Ok(rendered) => rendered,
            Err(e) => {
                error!(topic = %topic, "{}", e);
                close_session(&state_for_reader);
                fail_agent(&tx, &topic, e).await;
                return;
            }
        };

        let ctx = crate::adapter::SessionContext {
            chat_id: &topic,
            project_path: project_path.as_deref(),
            worktree,
            agent_session_id: agent_session_id.as_deref(),
            is_sandboxed,
            model: model.as_deref(),
            session_name: session_name.as_deref(),
            user_system_prompt: user_system_prompt.as_deref(),
        };
        let native = if cfg!(windows) {
            lock_adapter(&adapter).prepare_session_native(&ctx)
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
            let prepared = lock_adapter(&adapter).prepare_session_command(&ctx);
            let cmd_string = match prepared {
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
        let (mut child, pgid, agent_started, stderr_handle, writer_handle, cancelled) = 'spawn: loop {
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
            let pgid = child.id().unwrap_or_default();
            crate::orphans::record(pgid);
            state_for_reader.lock().expect("SessionState mutex").pid = child.id();

            // Writer task: owns ChildStdin, the single writer of stdin frames (see
            // `spawn_stdin_writer`).
            let writer_handle = child.stdin.take().map(|stdin| {
                spawn_stdin_writer(
                    stdin,
                    state_for_reader.clone(),
                    adapter.clone(),
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
            // respawn, prune-respawn, eviction, shutdown). A deliberate teardown
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
                    // A send blocked on a full response queue must still see a
                    // teardown: back to the loop, whose cancel arm wins. The
                    // send goes first, so a queue with room still takes the
                    // turn's last frames when shutdown cancels right after
                    // its `result` was reduced.
                    macro_rules! send_or_cancel {
                        ($resp:expr) => {
                            tokio::select! {
                                biased;
                                sent = tx.send($resp) => if sent.is_err() { break 'read; },
                                _ = cancel.cancelled() => continue 'read,
                            }
                        };
                    }
                    tokio::select! {
                        biased;
                        _ = cancel.cancelled() => {
                            warn!(topic = %topic, "resident session cancelled, tearing down process group");
                            cancelled = true;
                            if exits_on_stdin_eof {
                                // Dropping the writer closes stdin.
                                if let Some(wh) = &writer_handle {
                                    wh.abort();
                                }
                                // Keep draining stdout so the CLI can't block on
                                // a full pipe while it shuts down.
                                let drain = async { while let Ok(Some(_)) = lines.next_line().await {} };
                                let exit = async { tokio::join!(child.wait(), drain) };
                                if tokio::time::timeout(STDIN_EOF_GRACE, exit).await.is_err() {
                                    warn!(topic = %topic, "resident agent ignored stdin EOF for {:?}", STDIN_EOF_GRACE);
                                }
                            }
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
                                crate::orphans::forget(pgid);
                                // The new process needs its own handshake.
                                let mut fresh = agent_kind.make_adapter();
                                let _ = fresh.prepare_session_command(&ctx);
                                *lock_adapter(&adapter) = fresh;
                                attempt += 1;
                                continue 'spawn;
                            }
                            error!(topic = %topic, "resident agent produced no output within {:?}, killing", startup_timeout);
                            send_or_cancel!(AgentResponse::Line {
                                topic: topic.clone(),
                                content: startup_timeout_message(startup_timeout, attempt > 0),
                            });
                            break 'read;
                        }
                        line_result = lines.next_line() => {
                            match line_result {
                                Ok(Some(line)) => {
                                    let (events, started) = {
                                        let mut a = lock_adapter(&adapter);
                                        let events = a.handle_line(line);
                                        (events, a.startup_complete())
                                    };
                                    if started {
                                        agent_started.store(true, Ordering::Relaxed);
                                        ANY_AGENT_STARTED.store(true, Ordering::Relaxed);
                                    }
                                    let mut renaming = InternalLine::default();
                                    // Captured with the event that caused them (a
                                    // turn queued later can't hide a turn's end),
                                    // sent after the line's frames.
                                    let mut turn_edges = Vec::new();
                                    for ev in events {
                                        // Reduce the FSM and read back the resulting
                                        // `running` in one lock — the `match` below only
                                        // forwards frames and never mutates state, so
                                        // this value is final for the event. (The FSM
                                        // still tracks the waiting sub-state internally
                                        // via `SessionState`; it just doesn't cross the
                                        // wire.)
                                        let is_result = matches!(&ev, AgentEvent::Result { .. });
                                        // Our own `/rename` turn is running: it is silent,
                                        // so every frame it produces (and its usage) is
                                        // dropped.
                                        let (running, swallow, live_tasks, turn_ended) = {
                                            let mut st = state_for_reader.lock().expect("SessionState mutex");
                                            let internal = st.internal_in_flight();
                                            let swallow = renaming.swallows(internal, &ev);
                                            reduce(&mut st, &ev);
                                            let r = st.running();
                                            let turn_ended = is_result && !internal && !st.in_turn();
                                            let live_tasks = st.observe_live_tasks();
                                            turn_edges.extend(st.observe_turn_running());
                                            // [DIAG tasktrace] TEMPORARY — log every
                                            // state-bearing event with the resulting FSM
                                            // so we can see why live_tasks never empties.
                                            match &ev {
                                                AgentEvent::TaskStarted { id, .. } => tracing::info!(diag = "tasktrace", topic = %topic, %id, live = st.live_tasks.len(), turn = st.turn_in_flight, running = r, "reduce TaskStarted"),
                                                AgentEvent::TaskFinished(id) => tracing::info!(diag = "tasktrace", topic = %topic, %id, live = st.live_tasks.len(), turn = st.turn_in_flight, running = r, "reduce TaskFinished"),
                                                AgentEvent::Result { origin_is_task } => tracing::info!(diag = "tasktrace", topic = %topic, origin_is_task, live = st.live_tasks.len(), turn = st.turn_in_flight, running = r, "reduce Result"),
                                                _ => {}
                                            }
                                            (r, swallow, live_tasks, turn_ended)
                                        };
                                        // A slash turn held by the writer may be due now.
                                        if is_result {
                                            wake_for_writer.notify_one();
                                        }
                                        // Attachments no assistant text picked up
                                        // belong to this turn, not the next one.
                                        if is_result && !swallow {
                                            if let Some(row) = attachment_row(&topic, &mut attach_rx) {
                                                send_or_cancel!(row);
                                            }
                                        }
                                        match ev {
                                            _ if swallow => {}
                                            AgentEvent::Frame(content) => {
                                                let followup = attach_followup_for(&topic, &content, &mut attach_rx);
                                                send_or_cancel!(AgentResponse::Line { topic: topic.clone(), content });
                                                if let Some(f) = followup {
                                                    send_or_cancel!(f);
                                                }
                                            }
                                            AgentEvent::ContextTokens(tokens) => {
                                                send_or_cancel!(AgentResponse::ContextTokens { topic: topic.clone(), tokens });
                                            }
                                            AgentEvent::CompactBoundary(post_tokens) => {
                                                send_or_cancel!(AgentResponse::CompactBoundary { topic: topic.clone(), post_tokens });
                                            }
                                            AgentEvent::SessionIdHarvested(session_id) => {
                                                send_or_cancel!(AgentResponse::SessionIdHarvested { topic: topic.clone(), session_id });
                                            }
                                            AgentEvent::ToolResult => {
                                                send_or_cancel!(AgentResponse::ToolResult { topic: topic.clone() });
                                            }
                                            AgentEvent::PeerTurnConsumed { name, body } => {
                                                let content = peer_message_frame_json(&name, &body);
                                                send_or_cancel!(AgentResponse::Line { topic: topic.clone(), content });
                                            }
                                            // Run-state-bearing events (Result/TaskStarted/
                                            // TaskFinished) already updated the FSM via
                                            // `reduce`; the RunState recompute below emits a
                                            // transition if `running` changed.
                                            AgentEvent::Result { .. }
                                            | AgentEvent::TaskStarted { .. }
                                            | AgentEvent::TaskFinished(_)
                                            | AgentEvent::CommandStarted { .. }
                                            | AgentEvent::CommandFinished(_)
                                            | AgentEvent::UserTurnConsumed(_)
                                            | AgentEvent::TurnStarted(_) => {}
                                        }
                                        if turn_ended {
                                            send_or_cancel!(AgentResponse::TurnEnded { topic: topic.clone() });
                                        }
                                        // Emit the idle (`running=false`) edge ONLY on a `Result` frame — a real
                                        // turn boundary. A bare `TaskFinished` that empties `live_tasks` also drops
                                        // `running` to false, but a claude-self-initiated continuation turn is about
                                        // to run (and will end with its own `Result`), so suppressing the edge here
                                        // defers it to that turn's end. Busy (`running=true`) edges always emit.
                                        if last_emitted != Some(running) && (running || is_result) {
                                            last_emitted = Some(running);
                                            state_for_reader.lock().expect("SessionState mutex").idle_since =
                                                (!running).then(tokio::time::Instant::now);
                                            _power_assertion = running.then(crate::power::AgentPowerAssertion::acquire).flatten();
                                            // [DIAG tasktrace] TEMPORARY — the busy↔idle edge that writes agent_running.
                                            tracing::info!(diag = "tasktrace", topic = %topic, running, "RunState transition -> chat_running");
                                            send_or_cancel!(AgentResponse::RunState { topic: topic.clone(), running });
                                        }
                                        // What's armed changed: publish the blob (or a
                                        // clear). Unlike the idle edge this is NOT
                                        // deferred to a turn boundary — the point is to
                                        // show the task list while the turn is over and
                                        // the chat is Waiting.
                                        if let Some(tasks) = live_tasks {
                                            let session = state_for_reader.clone();
                                            send_or_cancel!(AgentResponse::LiveTasks { topic: topic.clone(), session, tasks });
                                        }
                                    }
                                    for running in turn_edges {
                                        let session = state_for_reader.clone();
                                        send_or_cancel!(AgentResponse::TurnRunning { topic: topic.clone(), session, running });
                                    }
                                    // A response or notification may have made a
                                    // held frame sendable.
                                    if writer_follows_every_line {
                                        wake_for_writer.notify_one();
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
                pgid,
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
        crate::orphans::forget(pgid);

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

pub async fn await_agent_exit(handle: tokio::task::JoinHandle<()>, topic: &str) {
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

/// A resident's adapter, shared by its stdout reader and stdin writer. Both
/// only hold the lock for synchronous work.
type SharedAdapter = Arc<StdMutex<Box<dyn crate::adapter::AgentAdapter>>>;

fn lock_adapter(
    adapter: &SharedAdapter,
) -> std::sync::MutexGuard<'_, Box<dyn crate::adapter::AgentAdapter>> {
    adapter.lock().expect("adapter mutex")
}

/// How long teardown waits for a CLI that exits on stdin EOF before
/// signalling its process group (codex exits in <100ms).
const STDIN_EOF_GRACE: Duration = Duration::from_secs(3);

/// Spawn the resident stdin writer task: write the adapter's protocol frames
/// and each queued `SessionState::unconsumed` turn it hasn't written yet, then
/// sleep until `wake`. A startup-timeout retry's fresh writer re-writes them all
/// to the new process. Exits on `cancel`, closing stdin; claude ignores stdin
/// EOF, so its real teardown is the SIGTERM.
fn spawn_stdin_writer(
    mut stdin: tokio::process::ChildStdin,
    state: Arc<StdMutex<SessionState>>,
    adapter: SharedAdapter,
    wake: Arc<Notify>,
    cancel: CancellationToken,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        use tokio::io::AsyncWriteExt;
        let mut written: HashSet<String> = HashSet::new();
        loop {
            // The state lock is released before the adapter's is taken: the
            // reader holds the adapter through a multi-MB line's parse, and the
            // Supervisor locks the state from the main loop.
            let due = due_writes(&state.lock().expect("SessionState mutex"), &written);
            let (frames, over) = {
                let mut adapter = lock_adapter(&adapter);
                let frames = encode_due(adapter.as_mut(), due, &mut written);
                (frames, adapter.session_over())
            };
            for frame in frames {
                if stdin.write_all(frame.as_bytes()).await.is_err() || stdin.flush().await.is_err()
                {
                    return;
                }
            }
            if over {
                return;
            }
            tokio::select! {
                _ = wake.notified() => {}
                _ = cancel.cancelled() => return,
            }
        }
    })
}

/// A turn the writer owes the CLI, read off the `SessionState`.
struct Due {
    uuid: String,
    text: String,
}

/// The unwritten turns of `st.unconsumed` in order. A command turn is held
/// (with everything behind it) until it heads the queue with no turn in
/// flight: written earlier, claude would inject it into the running turn as
/// plain text at the next tool boundary. Nothing behind an unconsumed turn
/// that runs alone (`QueuedTurn::alone`) is due either, nor anything while one
/// is in flight.
fn due_writes(st: &SessionState, written: &HashSet<String>) -> Vec<Due> {
    let mut due = Vec::new();
    if st.command_in_flight {
        return due;
    }
    for (i, t) in st.unconsumed.iter().enumerate() {
        if t.own_turn && (i > 0 || st.in_turn()) {
            break;
        }
        if !written.contains(&t.uuid) {
            due.push(Due {
                uuid: t.uuid.clone(),
                text: t.text.clone(),
            });
        }
        if t.alone {
            break;
        }
    }
    due
}

/// The adapter's protocol frames, then `due` encoded in order. A turn the
/// adapter can't encode yet holds the queue behind it.
fn encode_due(
    adapter: &mut dyn crate::adapter::AgentAdapter,
    due: Vec<Due>,
    written: &mut HashSet<String>,
) -> Vec<String> {
    let mut frames = adapter.outbound_frames();
    for Due { uuid, text } in due {
        let Some(frame) = adapter.encode_user_turn(&uuid, &text) else {
            break;
        };
        written.insert(uuid);
        frames.push(frame);
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
    attachment_row(topic, attach_rx)
}

/// Everything in `attach_rx` as one attachment-only row, or `None` when it is
/// empty.
fn attachment_row(
    topic: &str,
    attach_rx: &mut mpsc::UnboundedReceiver<EnvelopeAttachment>,
) -> Option<AgentResponse> {
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

/// A backgrounded Bash / Monitor arming, as the claude adapter parses it.
#[cfg(test)]
pub(crate) fn task_started(id: &str) -> AgentEvent {
    AgentEvent::TaskStarted {
        id: id.into(),
        description: format!("{id} description"),
        task_type: "local_bash".into(),
        subagent_type: None,
    }
}

/// The same, already reduced into a session's list.
#[cfg(test)]
pub(crate) fn live_task(id: &str) -> LiveTask {
    LiveTask {
        id: id.into(),
        description: format!("{id} description"),
        task_type: "local_bash".into(),
        subagent_type: None,
        started_at: 1_780_886_492_000,
    }
}

/// An adapter whose stdin frames are just the turn's uuid,
/// so writer tests can assert what goes out and in which order.
#[cfg(test)]
struct UuidFrames;

#[cfg(test)]
impl crate::adapter::AgentAdapter for UuidFrames {
    fn kind(&self) -> AgentKind {
        AgentKind::Claude
    }
    fn prepare_command(&mut self, _ctx: &TurnContext<'_>) -> anyhow::Result<String> {
        unreachable!()
    }
    fn handle_line(&mut self, _line: String) -> smallvec::SmallVec<[AgentEvent; 2]> {
        smallvec::SmallVec::new()
    }
    fn encode_user_turn(&mut self, uuid: &str, _text: &str) -> Option<String> {
        Some(uuid.to_string())
    }
}

#[cfg(test)]
fn test_frames_to_write(st: &SessionState, written: &mut HashSet<String>) -> Vec<String> {
    let due = due_writes(st, written);
    encode_due(&mut UuidFrames, due, written)
}

#[cfg(test)]
mod reduce_tests {
    use super::{
        internal_turn_swallows, live_task, live_tasks_payload, live_tasks_with_watches,
        peer_message_frame_json, reduce, task_started, test_frames_to_write as frames_to_write,
        AgentEvent, LiveTask, LiveTasksDedup, QueuedTurn, SessionState, WatchDef,
    };
    use std::collections::HashSet;

    /// A session with the internal `/rename` turn `uuid` queued, as
    /// `ResidentSession::queue_internal_turn` leaves it.
    fn renaming(uuid: &str) -> SessionState {
        let mut s = SessionState::default();
        queue_rename(&mut s, uuid);
        s
    }

    /// One more internal `/rename` turn queued, exactly as `push_turn` does.
    fn queue_rename(s: &mut SessionState, uuid: &str) {
        s.queue_internal(uuid.into());
        s.unconsumed.push(QueuedTurn {
            own_turn: true,
            internal: true,
            ..queued(uuid)
        });
    }

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
        reduce(&mut s, &task_started("t1"));
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
        s.live_tasks.push(live_task("t1"));
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

    /// Two stacked tasks: both must finish before the session goes Idle, and the
    /// list keeps them in ARM order (it is what the client renders).
    #[test]
    fn two_tasks_both_must_finish_for_idle() {
        let mut s = SessionState::default();
        reduce(&mut s, &task_started("a"));
        reduce(&mut s, &task_started("b"));
        // No turn in flight, tasks armed → Waiting.
        assert_eq!((s.running(), s.waiting()), (true, true));
        assert_eq!(ids(&s), vec!["a", "b"]);
        // A repeated `task_started` for the same id is not a second task.
        reduce(&mut s, &task_started("a"));
        assert_eq!(ids(&s), vec!["a", "b"]);
        reduce(&mut s, &AgentEvent::TaskFinished("a".into()));
        assert_eq!((s.running(), s.waiting()), (true, true), "b still live");
        assert_eq!(ids(&s), vec!["b"], "only the finished one is retained out");
        reduce(&mut s, &AgentEvent::TaskFinished("b".into()));
        assert!(s.is_idle());
    }

    fn ids(s: &SessionState) -> Vec<&str> {
        s.live_tasks.iter().map(|t| t.id.as_str()).collect()
    }

    /// The exact plaintext blob three clients parse. Kinds map `local_bash` →
    /// `bash` and `local_agent` → `agent`, an unknown kind rides verbatim, `a`
    /// is present only for a subagent, and `q` is omitted when no user turn is
    /// queued.
    #[test]
    fn live_tasks_blob_is_the_documented_wire_shape() {
        let mut s = SessionState::default();
        reduce(
            &mut s,
            &AgentEvent::TaskStarted {
                id: "buukg9452".into(),
                description: "npm test -- --watch".into(),
                task_type: "local_bash".into(),
                subagent_type: None,
            },
        );
        reduce(
            &mut s,
            &AgentEvent::TaskStarted {
                id: "a3bc46a79".into(),
                description: "count tmp files".into(),
                task_type: "local_agent".into(),
                subagent_type: Some("Explore".into()),
            },
        );
        reduce(
            &mut s,
            &AgentEvent::TaskStarted {
                id: "weird".into(),
                description: String::new(),
                task_type: "local_future_kind".into(),
                subagent_type: None,
            },
        );
        let blob: serde_json::Value =
            serde_json::from_str(&live_tasks_payload(&s).expect("armed")).expect("valid JSON");
        assert_eq!(blob["t"][0]["i"], "buukg9452");
        assert_eq!(blob["t"][0]["d"], "npm test -- --watch");
        assert_eq!(blob["t"][0]["k"], "bash");
        assert!(blob["t"][0].get("a").is_none(), "no subagent, no `a`");
        assert!(blob["t"][0]["s0"].as_i64().unwrap() > 1_700_000_000_000);
        assert_eq!(blob["t"][1]["k"], "agent");
        assert_eq!(blob["t"][1]["a"], "Explore");
        assert_eq!(
            blob["t"][2]["k"], "local_future_kind",
            "unknown kind rides verbatim"
        );
        assert_eq!(blob["t"][2]["d"], "");
        assert!(blob.get("q").is_none(), "no queued user turn, no `q`");
    }

    /// Nothing armed and nothing queued ⇒ the column is NULL, never an empty
    /// object: the clients read NULL as "no sheet content".
    #[test]
    fn an_empty_session_has_no_live_tasks_blob() {
        assert!(live_tasks_payload(&SessionState::default()).is_none());
    }

    /// `q` counts the USER's queued turns only — an internal `/rename` is the
    /// spawner's own and must never show up as a message the user is waiting on.
    #[test]
    fn q_counts_user_turns_not_internal_renames() {
        let mut s = SessionState::default();
        queue_rename(&mut s, "rename-1");
        assert!(
            live_tasks_payload(&s).is_none(),
            "an internal turn alone is not worth a blob"
        );
        s.unconsumed.push(queued("user-1"));
        s.unconsumed.push(queued("user-2"));
        let blob: serde_json::Value =
            serde_json::from_str(&live_tasks_payload(&s).expect("queued")).expect("valid JSON");
        assert_eq!(blob["q"], 2);
        assert!(blob.get("t").is_none(), "nothing armed, no `t`");
    }

    /// Caps: the OLDEST 20 tasks ride, and a pathological description is cut to
    /// 200 chars — the chats row re-streams on every writeback.
    #[test]
    fn the_blob_caps_tasks_at_20_and_descriptions_at_200_chars() {
        let mut s = SessionState::default();
        for i in 0..25 {
            s.live_tasks.push(LiveTask {
                description: "x".repeat(500),
                ..live_task(&format!("t{i}"))
            });
        }
        let blob: serde_json::Value =
            serde_json::from_str(&live_tasks_payload(&s).expect("armed")).expect("valid JSON");
        let tasks = blob["t"].as_array().expect("array");
        assert_eq!(tasks.len(), 20);
        assert_eq!(tasks[0]["i"], "t0", "the oldest survive the cap");
        assert_eq!(tasks[19]["i"], "t19");
        assert_eq!(tasks[0]["d"].as_str().expect("string").chars().count(), 200);
    }

    fn watch_def(description: &str) -> WatchDef {
        WatchDef::for_test("chat", uuid::Uuid::nil(), description)
    }

    #[test]
    fn watches_lead_the_blob_and_share_its_cap() {
        assert_eq!(live_tasks_with_watches(None, &[]), None);
        let session = Some(r#"{"q":2}"#.to_string());
        assert_eq!(live_tasks_with_watches(session.clone(), &[]), session);

        let watch = watch_def(&"x".repeat(500));
        let alone = live_tasks_with_watches(None, std::slice::from_ref(&watch)).expect("a blob");
        let blob: serde_json::Value = serde_json::from_str(&alone).expect("valid JSON");
        assert_eq!(
            blob,
            serde_json::json!({"t": [{
                "i": watch.id.to_string(),
                "d": "x".repeat(200),
                "k": "watch",
                "s0": 1_800_000_000_000_i64,
            }]})
        );

        let mut s = SessionState::default();
        for i in 0..25 {
            s.live_tasks.push(live_task(&format!("t{i}")));
        }
        s.unconsumed.push(queued("user-1"));
        let watches = [watch_def("mail"), watch_def("deploys")];
        let merged = live_tasks_with_watches(live_tasks_payload(&s), &watches).expect("a blob");
        let blob: serde_json::Value = serde_json::from_str(&merged).expect("valid JSON");
        let tasks = blob["t"].as_array().expect("array");
        assert_eq!(tasks.len(), 20);
        assert_eq!(tasks[0]["d"], "mail");
        assert_eq!(tasks[1]["d"], "deploys");
        assert_eq!(tasks[2]["i"], "t0");
        assert_eq!(tasks[19]["i"], "t17");
        assert_eq!(blob["q"], 1);
    }

    /// The reader publishes only CHANGES — an unchanged payload writes nothing,
    /// the way `LastTokensDedup` gates `ContextTokens`.
    #[test]
    fn live_tasks_are_published_only_when_the_payload_changes() {
        let mut dedup = LiveTasksDedup::default();
        let mut s = SessionState::default();
        assert_eq!(
            dedup.observe(live_tasks_payload(&s)),
            None,
            "a fresh session starts from a cleared column and publishes nothing"
        );

        reduce(&mut s, &task_started("a"));
        let armed = dedup.observe(live_tasks_payload(&s)).expect("a change");
        assert!(armed.expect("a blob").contains("\"a\""));
        assert_eq!(
            dedup.observe(live_tasks_payload(&s)),
            None,
            "an unrelated event does not re-write an unchanged set"
        );

        reduce(&mut s, &AgentEvent::TaskFinished("a".into()));
        assert_eq!(
            dedup.observe(live_tasks_payload(&s)),
            Some(None),
            "the last task finishing clears the column"
        );
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
            text: String::new(),
            own_turn: false,
            alone: false,
            internal: false,
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

    /// A message queued right behind `/compact` must not reach the CLI inside
    /// the compaction — not in the same pass, not before the compact is
    /// consumed, not while it runs.
    #[test]
    fn a_compact_turn_runs_alone() {
        let mut s = SessionState::default();
        let compact = QueuedTurn {
            own_turn: true,
            alone: true,
            ..queued("compact")
        };
        s.unconsumed = vec![compact, queued("next")];
        let mut written = HashSet::new();
        assert_eq!(frames_to_write(&s, &mut written), vec!["compact"]);
        assert!(frames_to_write(&s, &mut written).is_empty());

        reduce(&mut s, &AgentEvent::UserTurnConsumed("compact".into()));
        assert!(s.command_in_flight);
        assert!(frames_to_write(&s, &mut written).is_empty());

        reduce(
            &mut s,
            &AgentEvent::Result {
                origin_is_task: true,
            },
        );
        assert!(
            frames_to_write(&s, &mut written).is_empty(),
            "a background task's result doesn't end the compaction"
        );

        reduce(&mut s, &RESULT);
        assert_eq!(frames_to_write(&s, &mut written), vec!["next"]);
    }

    /// `chats.turn_running` across a turn that leaves a task armed and the
    /// continuation turn the task's completion wakes: false while the chat is
    /// only Waiting (still `running`), true again for the continuation.
    #[test]
    fn turn_running_follows_turns_not_background_tasks() {
        let mut s = SessionState::default();
        assert_eq!(s.observe_turn_running(), None, "a fresh session is false");

        s.unconsumed.push(queued("a"));
        assert_eq!(s.observe_turn_running(), Some(true), "queued");
        reduce(&mut s, &AgentEvent::UserTurnConsumed("a".into()));
        reduce(&mut s, &task_started("t1"));
        assert_eq!(s.observe_turn_running(), None, "still the same turn");

        reduce(&mut s, &RESULT);
        assert_eq!(s.observe_turn_running(), Some(false), "turn ended");
        assert!(s.running(), "the armed task keeps the agent running");

        reduce(&mut s, &AgentEvent::TaskFinished("t1".into()));
        assert_eq!(s.observe_turn_running(), None);
        reduce(&mut s, &AgentEvent::TurnStarted("continuation".into()));
        assert_eq!(s.observe_turn_running(), Some(true), "continuation turn");
        reduce(
            &mut s,
            &AgentEvent::Result {
                origin_is_task: true,
            },
        );
        assert_eq!(s.observe_turn_running(), Some(false));
        assert!(s.is_idle());
    }

    /// A user turn queued behind the running one keeps `turn_running` across
    /// that turn's `result`; the spawner's own `/rename` never raises it.
    #[test]
    fn turn_running_counts_queued_user_turns_but_no_internal_rename() {
        let mut s = renaming("r");
        assert_eq!(s.observe_turn_running(), None, "queued rename");
        reduce(&mut s, &AgentEvent::TurnStarted("r".into()));
        reduce(&mut s, &AgentEvent::UserTurnConsumed("r".into()));
        assert_eq!(s.observe_turn_running(), None, "running rename");
        reduce(&mut s, &RESULT);

        s.unconsumed.push(queued("a"));
        reduce(&mut s, &AgentEvent::UserTurnConsumed("a".into()));
        s.unconsumed.push(queued("b"));
        assert_eq!(s.observe_turn_running(), Some(true));
        reduce(&mut s, &RESULT);
        assert_eq!(s.observe_turn_running(), None, "b is still queued");
        reduce(&mut s, &AgentEvent::UserTurnConsumed("b".into()));

        // The turn's end has to be captured as the `result` is reduced: a turn
        // queued before a later look would hide it.
        reduce(&mut s, &RESULT);
        let ended = s.observe_turn_running();
        s.unconsumed.push(queued("b"));
        assert_eq!(ended, Some(false));
        assert_eq!(s.observe_turn_running(), Some(true), "the enqueue edge");
        reduce(&mut s, &AgentEvent::UserTurnConsumed("b".into()));
        reduce(&mut s, &RESULT);
        assert_eq!(s.observe_turn_running(), Some(false));
    }

    #[test]
    fn auto_compaction_compacts_then_notices_always_on_chats_once() {
        let mut s = SessionState::default();
        reduce(&mut s, &AgentEvent::ContextTokens(90_000));
        assert_eq!(s.after_turn(None, true), None);
        assert_eq!(s.after_turn(Some(100_000), true), None);

        reduce(&mut s, &AgentEvent::ContextTokens(100_000));
        s.live_tasks.push(LiveTask {
            task_type: "local_bash".into(),
            ..live_task("shell")
        });
        let compact = s.after_turn(Some(100_000), true).expect("compacts");
        assert!(compact.starts_with("/compact "));
        s.unconsumed.push(QueuedTurn {
            own_turn: true,
            alone: true,
            ..queued("compact")
        });
        assert_eq!(s.after_turn(Some(100_000), true), None, "still queued");

        s.unconsumed.clear();
        reduce(&mut s, &AgentEvent::CompactBoundary(8_000));
        assert_eq!(
            s.after_turn(Some(100_000), true).as_deref(),
            Some(super::AUTO_COMPACT_NOTICE)
        );
        reduce(&mut s, &AgentEvent::ContextTokens(150_000));
        assert_eq!(s.after_turn(Some(100_000), true), None, "notice's end");
        assert!(s.after_turn(Some(100_000), true).is_some());
    }

    #[test]
    fn task_result_before_the_notice_echo_keeps_noticing() {
        let mut s = SessionState {
            auto_compact: super::AutoCompact::Noticing,
            ..Default::default()
        };
        s.unconsumed.push(queued("notice"));
        reduce(&mut s, &AgentEvent::ContextTokens(200_000));
        reduce(
            &mut s,
            &AgentEvent::Result {
                origin_is_task: true,
            },
        );
        assert_eq!(s.after_turn(Some(100_000), true), None);
        assert_eq!(s.auto_compact, super::AutoCompact::Noticing);

        reduce(&mut s, &AgentEvent::UserTurnConsumed("notice".into()));
        reduce(&mut s, &RESULT);
        assert_eq!(s.after_turn(Some(100_000), true), None, "notice's end");
        assert!(s.after_turn(Some(100_000), true).is_some());
    }

    #[test]
    fn auto_compaction_waits_for_subagents_queued_turns_and_the_turn() {
        let mut s = SessionState::default();
        reduce(&mut s, &AgentEvent::ContextTokens(200_000));
        s.unconsumed.push(queued("user"));
        assert_eq!(s.after_turn(Some(100_000), false), None);
        s.unconsumed.clear();
        s.turn_in_flight = true;
        assert_eq!(s.after_turn(Some(100_000), false), None);
        s.turn_in_flight = false;
        s.live_tasks.push(LiveTask {
            task_type: "local_agent".into(),
            ..live_task("subagent")
        });
        assert_eq!(s.after_turn(Some(100_000), false), None);
        s.live_tasks.clear();

        assert!(s.after_turn(Some(100_000), false).is_some());
        assert_eq!(s.after_turn(Some(100_000), false), None, "no notice");
        assert!(s.after_turn(Some(100_000), false).is_some());
    }

    #[test]
    fn a_threshold_set_mid_compaction_leaves_it_alone() {
        let mut s = SessionState::default();
        reduce(&mut s, &AgentEvent::ContextTokens(200_000));
        assert!(s.after_threshold_set(Some(100_000)).is_some());
        assert_eq!(s.after_threshold_set(Some(100_000)), None);
        assert_eq!(s.auto_compact, super::AutoCompact::Compacting);
    }

    /// Turns behind any other claude slash command go at once.
    #[test]
    fn claude_writes_turns_behind_a_command_at_once() {
        let cmd = QueuedTurn {
            own_turn: true,
            ..queued("cmd")
        };
        let mut s = SessionState {
            unconsumed: vec![cmd, queued("next")],
            ..Default::default()
        };
        let mut written = HashSet::new();
        assert_eq!(frames_to_write(&s, &mut written), vec!["cmd", "next"]);

        reduce(&mut s, &AgentEvent::UserTurnConsumed("cmd".into()));
        s.unconsumed.push(queued("later"));
        assert_eq!(frames_to_write(&s, &mut written), vec!["later"]);
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

    /// A `CronCreate` prompt fires on an idle resident: no echo, only a
    /// started frame, and the result carries no origin (tmp/cron_probe.log).
    #[test]
    fn a_self_fired_turn_is_busy_until_its_result() {
        let mut s = SessionState::default();
        reduce(&mut s, &AgentEvent::TurnStarted("cron".into()));
        assert!(s.running() && s.working());
        reduce(&mut s, &RESULT);
        assert!(s.is_idle());

        // A monitor wake's turn ends with a task result.
        reduce(&mut s, &AgentEvent::TurnStarted("wake".into()));
        reduce(
            &mut s,
            &AgentEvent::Result {
                origin_is_task: true,
            },
        );
        assert!(s.is_idle());
    }

    /// Our own queued turn starts before its echo: it stays a queued turn.
    #[test]
    fn our_own_turn_starting_is_not_a_self_turn() {
        let mut s = SessionState::default();
        s.unconsumed.push(queued("a"));
        reduce(&mut s, &AgentEvent::TurnStarted("a".into()));
        assert!(!s.self_turn);
        reduce(&mut s, &AgentEvent::UserTurnConsumed("a".into()));
        reduce(&mut s, &RESULT);
        assert!(s.is_idle());
    }

    /// A peer `SendMessage` to an idle resident: busy from its echo, idle on
    /// the turn's `result`.
    #[test]
    fn peer_turn_is_busy_until_its_result() {
        let mut s = SessionState::default();
        let peer = AgentEvent::PeerTurnConsumed {
            name: "zucchini-9d".into(),
            body: "what is 6 x 7?".into(),
        };
        reduce(&mut s, &peer);
        assert!(s.turn_in_flight && s.running());
        reduce(&mut s, &RESULT);
        assert!(s.is_idle());
    }

    /// The spawner's own `/rename` turn: busy for nobody, from queue to
    /// `result`, so it writes no `agent_running` edge (hence no push, no
    /// scheduled-message promotion) and never touches `idle_since`. Everything
    /// it emits is dropped — the rename leaves no message row behind.
    #[test]
    fn an_internal_turn_is_never_busy_and_its_frames_are_swallowed() {
        let mut s = renaming("r1");
        assert!(s.is_idle(), "queued");

        reduce(&mut s, &AgentEvent::TurnStarted("r1".into()));
        assert!(s.is_idle() && !s.self_turn, "started");
        assert!(s.internal_in_flight());
        assert!(internal_turn_swallows(&AgentEvent::Frame("{}".into())));
        assert!(internal_turn_swallows(&AgentEvent::ContextTokens(9_000)));

        reduce(&mut s, &AgentEvent::UserTurnConsumed("r1".into()));
        assert!(s.is_idle() && !s.turn_in_flight, "consumed");
        assert!(s.unconsumed.is_empty());
        assert!(s.internal_in_flight());

        // The `result` LINE yields `Result` and then its own frame, so the
        // reader latches the window here and drops both (e2e:
        // `renaming_a_live_chat_writes_nothing_at_all`).
        let latched = s.internal_in_flight();
        reduce(&mut s, &RESULT);
        assert!(latched && internal_turn_swallows(&AgentEvent::Frame("{}".into())));
        assert!(
            !s.internal_in_flight(),
            "the window closes for the next line"
        );
        assert!(s.is_idle());
    }

    /// Fail open: anything that isn't the command's own `result` ends the
    /// window, so a turn racing it can never have its frames swallowed.
    #[test]
    fn a_foreign_turn_inside_the_window_closes_it() {
        for echo in [
            AgentEvent::UserTurnConsumed("other".into()),
            AgentEvent::PeerTurnConsumed {
                name: "zucchini-9d".into(),
                body: "hi".into(),
            },
        ] {
            let mut s = renaming("r1");
            s.unconsumed.push(queued("other"));
            reduce(&mut s, &AgentEvent::TurnStarted("r1".into()));
            reduce(&mut s, &echo);
            assert!(!s.internal_in_flight(), "{echo:?}");
            assert!(s.turn_in_flight, "{echo:?}");
        }
    }

    /// A user turn consumed while the rename is still QUEUED is an ordinary
    /// turn — the record it would need survives for its own echo.
    #[test]
    fn a_queued_internal_turn_outlives_another_turns_result() {
        let mut s = renaming("r1");
        s.unconsumed.insert(0, queued("a"));
        reduce(&mut s, &AgentEvent::UserTurnConsumed("a".into()));
        assert!(s.turn_in_flight && s.running());
        reduce(&mut s, &RESULT);
        assert!(s.is_idle());
        reduce(&mut s, &AgentEvent::TurnStarted("r1".into()));
        assert!(s.internal_in_flight());
    }

    /// It is a slash command, so the writer holds it — and the turns behind
    /// it — until it heads the queue with no turn in flight.
    #[test]
    fn an_internal_turn_is_held_like_any_slash_command() {
        let mut s = renaming("r1");
        s.unconsumed.insert(0, queued("a"));
        s.turn_in_flight = true;
        let mut written = HashSet::new();
        assert_eq!(frames_to_write(&s, &mut written), vec!["a"]);
        assert!(frames_to_write(&s, &mut written).is_empty());
        s.unconsumed.remove(0);
        s.turn_in_flight = false;
        assert_eq!(frames_to_write(&s, &mut written), vec!["r1"]);
    }

    /// Renamed twice before the first `/rename` even starts: both turns stay
    /// tracked, so the first is still invisible and still swallowed.
    #[test]
    fn a_rename_queued_behind_another_keeps_both_silent() {
        let mut s = renaming("r1");
        queue_rename(&mut s, "r2");

        reduce(&mut s, &AgentEvent::TurnStarted("r1".into()));
        reduce(&mut s, &AgentEvent::UserTurnConsumed("r1".into()));
        assert!(!s.turn_in_flight && !s.self_turn, "r1 is still invisible");
        assert!(s.is_idle());
        assert!(s.internal_in_flight(), "r1's frames are still swallowed");

        reduce(&mut s, &RESULT);
        assert!(!s.internal_in_flight());
        reduce(&mut s, &AgentEvent::TurnStarted("r2".into()));
        reduce(&mut s, &AgentEvent::UserTurnConsumed("r2".into()));
        assert!(!s.turn_in_flight && !s.self_turn, "r2 too");
        assert!(s.internal_in_flight(), "r2's own window");
        assert!(s.is_idle());
    }

    /// A rename queued while the previous one is mid-flight doesn't reopen or
    /// close its window.
    #[test]
    fn a_rename_queued_mid_flight_leaves_the_open_window_alone() {
        let mut s = renaming("r1");
        reduce(&mut s, &AgentEvent::TurnStarted("r1".into()));
        queue_rename(&mut s, "r2");
        assert!(s.internal_in_flight(), "r1's window survives r2 queueing");

        reduce(&mut s, &RESULT);
        assert!(!s.internal_in_flight());
        reduce(&mut s, &AgentEvent::TurnStarted("r2".into()));
        assert!(s.internal_in_flight() && s.is_idle(), "r2 is still tracked");
    }

    #[test]
    fn peer_message_frame_is_a_system_row() {
        let v: serde_json::Value =
            serde_json::from_str(&peer_message_frame_json("zucchini-9d", "hi \"there\"")).unwrap();
        assert_eq!(
            v,
            serde_json::json!({
                "type": "system",
                "subtype": "peer_message",
                "name": "zucchini-9d",
                "body": "hi \"there\"",
            })
        );
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
            session_name: None,
            user_system_prompt: Default::default(),
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
        st.live_tasks.push(live_task("t1"));
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
        let text = &st.unconsumed[1].text;
        assert!(
            text.contains("current local time") && text.ends_with("second"),
            "{text}"
        );
    }

    /// Only a resident past its idle edge with nothing pending is evictable,
    /// and a turn queued on it first saves it.
    #[tokio::test(start_paused = true)]
    async fn only_a_warm_idle_resident_is_evicted() {
        let (mut sup, exposed) = supervisor_with_exposed_sessions();
        sup.spawn_agent(claude_req("chat"));
        set_idle(&exposed, "chat");
        assert!(sup.idle_residents().is_empty(), "no idle edge yet");
        assert!(!sup.evict_idle("chat").await);

        exposed.lock().unwrap()["chat"].lock().unwrap().idle_since =
            Some(tokio::time::Instant::now());
        assert_eq!(sup.idle_residents()[0].topic, "chat");
        assert!(sup.send_user_turn(&claude_req("chat")), "warm stdin turn");
        assert!(sup.idle_residents().is_empty());
        assert!(!sup.evict_idle("chat").await, "busy is never evicted");

        set_idle(&exposed, "chat");
        assert!(sup.evict_idle("chat").await);
        assert!(sup.is_empty());
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
        let other_prompt = SpawnRequest {
            user_system_prompt: prompt_text("be terse"),
            ..claude_req("chat")
        };
        assert!(
            sup.send_user_turn(&other_prompt),
            "another prompt waits for the queued turn instead of cutting it"
        );
        let other_prompt_and_model = SpawnRequest {
            model: Some("opus".to_string()),
            ..other_prompt.clone()
        };
        assert!(!sup.send_user_turn(&other_prompt_and_model));

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
        assert!(
            sup.send_user_turn(&other_prompt),
            "another prompt doesn't cut the running turn either"
        );
        assert!(
            !sup.send_scheduled_turn(&other_model),
            "other knobs respawn a scheduled turn too"
        );
        let queued_before = sup.unconsumed_prompts("chat").len();
        assert!(
            sup.send_scheduled_turn(&compact),
            "a scheduled slash command waits for the turn instead of cutting it"
        );
        {
            let sessions = exposed.lock().unwrap();
            let mut st = sessions["chat"].lock().unwrap();
            st.unconsumed.drain(..queued_before);
            let mut written = HashSet::new();
            assert!(
                test_frames_to_write(&st, &mut written).is_empty(),
                "held while the turn runs"
            );
            st.turn_in_flight = false;
            assert_eq!(test_frames_to_write(&st, &mut written).len(), 1);
            st.turn_in_flight = true;
        }
        exposed.lock().unwrap()["chat"]
            .lock()
            .unwrap()
            .command_in_flight = true;
        assert!(
            sup.send_user_turn(&compact),
            "a running /compact is waited for, not interrupted"
        );
        set_idle(&exposed, "chat");
        assert!(
            sup.send_user_turn(&compact),
            "a turn of its own gets expanded"
        );
    }

    fn prompt_text(text: &str) -> UserSystemPrompt {
        UserSystemPrompt {
            text: Some(text.to_string()),
            ..Default::default()
        }
    }

    #[tokio::test(start_paused = true)]
    async fn a_resident_with_a_stale_system_prompt_is_retired_between_turns() {
        let (mut sup, exposed) = supervisor_with_exposed_sessions();
        let spawned_with = prompt_text("be terse");
        let current = prompt_text("be verbose");
        let spawn = |sup: &mut Supervisor| {
            sup.spawn_agent(SpawnRequest {
                user_system_prompt: spawned_with.clone(),
                ..claude_req("chat")
            })
        };
        let state = |exposed: &Exposed| exposed.lock().unwrap()["chat"].clone();

        spawn(&mut sup);
        assert_eq!(
            sup.retire_stale_prompt("chat", &current, true).await,
            None,
            "its first turn is queued"
        );
        state(&exposed).lock().unwrap().unconsumed.clear();
        state(&exposed).lock().unwrap().turn_in_flight = true;
        assert_eq!(sup.retire_stale_prompt("chat", &current, true).await, None);

        set_idle(&exposed, "chat");
        assert_eq!(
            sup.retire_stale_prompt("chat", &spawned_with, true).await,
            None,
            "the prompt it runs with"
        );
        assert_eq!(
            sup.retire_stale_prompt("chat", &current, false).await,
            Some(false)
        );
        assert!(!sup.has_agent("chat"));

        spawn(&mut sup);
        set_idle(&exposed, "chat");
        state(&exposed).lock().unwrap().live_tasks = vec![live_task("monitor")];
        assert_eq!(
            sup.retire_stale_prompt("chat", &current, false).await,
            None,
            "its monitor would die unnoticed"
        );
        let mut subagent = live_task("subagent");
        subagent.task_type = "local_agent".to_string();
        state(&exposed).lock().unwrap().live_tasks.push(subagent);
        assert_eq!(sup.retire_stale_prompt("chat", &current, true).await, None);
        state(&exposed).lock().unwrap().live_tasks.pop();
        assert_eq!(
            sup.retire_stale_prompt("chat", &current, true).await,
            Some(true)
        );
        assert!(!sup.has_agent("chat"));
    }

    /// A chats row re-streams on every writeback, so the rename is queued only
    /// when the title actually changed what this process was last asked for —
    /// its spawn-time `--name` included. The turn itself is invisible: it never
    /// makes the session busy, and a respawn drops it rather than re-queuing it
    /// (the fresh spawn passes `--name` from the current title).
    #[tokio::test(start_paused = true)]
    async fn renaming_queues_one_internal_turn_per_new_title() {
        let (mut sup, exposed) = supervisor_with_exposed_sessions();
        sup.spawn_agent(SpawnRequest {
            session_name: Some("cover letters writer".to_string()),
            ..claude_req("chat")
        });
        set_idle(&exposed, "chat");

        assert!(
            !sup.rename_resident("chat", Some("cover letters writer")),
            "came up under this --name"
        );
        assert!(!sup.rename_resident("chat", None), "unusable title");
        assert!(sup.rename_resident("chat", Some("my voice writer")));
        assert!(
            !sup.rename_resident("chat", Some("my voice writer")),
            "the same row re-streamed"
        );
        assert!(sup.rename_resident("chat", Some("third name")));
        assert!(!sup.rename_resident("other-chat", Some("x")), "no resident");

        assert_eq!(
            sup.unconsumed_prompts("chat"),
            vec!["/rename my voice writer", "/rename third name"]
        );
        assert!(!sup.is_running("chat"), "no busy edge for our own turns");

        // It is a slash command, so the writer holds it until it heads the
        // queue with no turn in flight.
        let session = exposed.lock().unwrap()["chat"].clone();
        assert_eq!(
            session.lock().unwrap().internal.len(),
            2,
            "both renames stay tracked, so neither turn is ever taken for the user's"
        );
        let mut written = HashSet::new();
        {
            let mut st = session.lock().expect("SessionState mutex");
            st.turn_in_flight = true;
            assert!(test_frames_to_write(&st, &mut written).is_empty());
            st.turn_in_flight = false;
            assert_eq!(
                test_frames_to_write(&st, &mut written).len(),
                1,
                "one at a time"
            );
        }

        let requeued: Vec<String> = sup
            .abort_agent("chat")
            .await
            .into_iter()
            .map(|t| t.prompt)
            .collect();
        assert!(requeued.is_empty(), "{requeued:?}");
    }

    /// The shared `live_sessions` directory (read by the control task's
    /// `prune-context` guard, and by `live_task_labels`) reflects a resident
    /// session's live tasks while it's alive and is cleared on teardown. This is
    /// the data source for the prune-while-task-live block, so it must track
    /// exactly — and the labels are what the forced-prune note names.
    #[tokio::test(start_paused = true)]
    async fn live_session_directory_tracks_tasks_and_clears_on_teardown() {
        let (mut sup, exposed) = supervisor_with_exposed_sessions();
        sup.spawn_agent(claude_req("deploying"));

        // Registered on spawn; no tasks yet.
        assert!(sup.live_task_labels("deploying").is_empty());
        assert!(sup.live_task_labels("never-spawned").is_empty());

        // A task armed in the session is visible through the shared directory.
        set_waiting(&exposed, "deploying"); // inserts live task "t1"
        assert_eq!(
            sup.live_task_labels("deploying"),
            vec!["t1 description"],
            "an armed task is visible (and named) to the control-side guard"
        );

        // Teardown drops the directory entry — no stale count blocks a later prune.
        sup.abort_agent("deploying").await;
        assert!(
            sup.live_task_labels("deploying").is_empty(),
            "teardown clears the live-sessions entry"
        );
    }

    /// The control task's `prune-context` guard reads the armed-task COUNT from
    /// the same directory: non-zero refuses the prune (a restart would kill the
    /// tasks), zero lets it through.
    #[tokio::test(start_paused = true)]
    async fn the_prune_guard_counts_a_residents_armed_tasks() {
        let (mut sup, exposed) = supervisor_with_exposed_sessions();
        let dir = sup.live_sessions();
        sup.spawn_agent(claude_req("chat"));
        assert_eq!(live_task_count(&dir, "chat"), 0, "a prune is allowed");
        assert_eq!(live_task_count(&dir, "never-spawned"), 0);

        set_waiting(&exposed, "chat"); // arms "t1"
        assert_eq!(live_task_count(&dir, "chat"), 1, "the prune is refused");

        let dev_server = AgentEvent::CommandStarted {
            id: "exec-1".into(),
            command: "npm run dev".into(),
        };
        reduce(
            &mut exposed.lock().unwrap()["chat"].lock().unwrap(),
            &dev_server,
        );
        assert_eq!(live_task_count(&dir, "chat"), 2, "a codex command counts");
        assert_eq!(sup.live_task_labels("chat")[1], "npm run dev");

        sup.abort_agent("chat").await;
        assert_eq!(live_task_count(&dir, "chat"), 0, "teardown unblocks it");
    }

    /// Respawning the same chat overwrites the directory entry with the fresh
    /// session's FSM, so a stale count never lingers across a respawn.
    #[tokio::test(start_paused = true)]
    async fn respawn_overwrites_live_session_entry() {
        let (mut sup, exposed) = supervisor_with_exposed_sessions();
        sup.spawn_agent(claude_req("chat"));
        set_waiting(&exposed, "chat");
        assert_eq!(sup.live_task_labels("chat").len(), 1);

        // A knob-changing respawn tears down + respawns; the recorder hands back a
        // fresh default state, so the directory must now read 0, not the stale 1.
        sup.abort_agent("chat").await;
        let mut req = claude_req("chat");
        req.model = Some("opus".to_string());
        sup.spawn_agent(req);
        assert!(
            sup.live_task_labels("chat").is_empty(),
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

/// End-to-end runs against the real agent CLIs in scratch dirs. They spend
/// real model calls, so they're opt-in:
/// `cargo test codex_resident_smoke -- --ignored --nocapture` (codex resident:
/// turn, message while busy, `/stop`, `/compact`, respawn with resume) and
/// `cargo test cursor_oneshot_smoke -- --ignored --nocapture` (cursor one-shot
/// through the spawn path the platform uses - the pwsh launcher on Windows).
#[cfg(test)]
mod cli_smoke {
    use super::*;

    const CHAT: &str = "5c0de000-0000-4000-8000-000000000000";
    const TURN_TIMEOUT: Duration = Duration::from_secs(240);

    fn req(prompt: &str, agent_session_id: Option<String>) -> SpawnRequest {
        req_for(
            AgentKind::Codex,
            "/tmp/codex-smoke",
            prompt,
            agent_session_id,
        )
    }

    fn req_for(
        agent_kind: AgentKind,
        project_path: &str,
        prompt: &str,
        agent_session_id: Option<String>,
    ) -> SpawnRequest {
        SpawnRequest {
            chat_id: CHAT.to_string(),
            prompt: prompt.to_string(),
            project_path: Some(project_path.to_string()),
            worktree: false,
            agent_session_id,
            agent_kind,
            is_sandboxed: false,
            model: None,
            user_timezone: Some("Asia/Bangkok".to_string()),
            session_name: None,
            user_system_prompt: Default::default(),
        }
    }

    fn describe(resp: &AgentResponse) -> String {
        match resp {
            AgentResponse::Line { content, .. } => format!("Line {content}"),
            AgentResponse::ContextTokens { tokens, .. } => format!("ContextTokens {tokens}"),
            AgentResponse::CompactBoundary { post_tokens, .. } => {
                format!("CompactBoundary {post_tokens}")
            }
            AgentResponse::SessionIdHarvested { session_id, .. } => {
                format!("SessionIdHarvested {session_id}")
            }
            AgentResponse::Done { has_result, .. } => format!("Done has_result={has_result}"),
            AgentResponse::RunState { running, .. } => format!("RunState running={running}"),
            AgentResponse::ToolResult { .. } => "ToolResult".to_string(),
            AgentResponse::LiveTasks { tasks, .. } => format!("LiveTasks {tasks:?}"),
            AgentResponse::TurnEnded { .. } => "TurnEnded".to_string(),
            AgentResponse::TurnRunning { running, .. } => format!("TurnRunning {running}"),
        }
    }

    /// Responses up to and including the first one `stop` matches.
    async fn until(
        rx: &mut mpsc::Receiver<AgentResponse>,
        stop: impl Fn(&str) -> bool,
    ) -> Vec<String> {
        let mut seen = Vec::new();
        let deadline = tokio::time::Instant::now() + TURN_TIMEOUT;
        loop {
            let resp = tokio::time::timeout_at(deadline, rx.recv())
                .await
                .unwrap_or_else(|_| panic!("timed out; seen: {seen:#?}"))
                .expect("response channel open");
            let d = describe(&resp);
            eprintln!("  <- {}", d.chars().take(240).collect::<String>());
            let done = stop(&d);
            seen.push(d);
            if done {
                return seen;
            }
        }
    }

    fn idle(d: &str) -> bool {
        d == "RunState running=false"
    }

    fn text_of(seen: &[String]) -> String {
        seen.iter()
            .filter(|d| d.contains("\"type\":\"assistant\""))
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn session_pid(sup: &Supervisor) -> Option<u32> {
        match sup.agents.get(CHAT) {
            Some(AgentEntry::Resident(s)) => s.state.lock().unwrap().pid,
            _ => None,
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "drives the real codex CLI"]
    async fn codex_resident_smoke() {
        std::fs::create_dir_all("/tmp/codex-smoke").unwrap();
        let (tx, mut rx) = mpsc::channel::<AgentResponse>(256);
        let mut sup = Supervisor::new(tx);

        eprintln!("== 1. first turn");
        sup.spawn_agent(req("Reply with the single word PONG.", None));
        let seen = until(&mut rx, idle).await;
        let sid = seen
            .iter()
            .find_map(|d| d.strip_prefix("SessionIdHarvested ").map(str::to_string))
            .expect("thread id harvested");
        assert!(text_of(&seen).contains("PONG"), "{seen:#?}");
        assert!(seen.iter().any(|d| d.starts_with("ContextTokens")));
        assert!(sup.has_agent(CHAT) && !sup.is_running(CHAT), "kept warm");
        let pid = session_pid(&sup).expect("pid");

        eprintln!("== 2. a message while busy joins the running turn");
        assert!(sup.send_user_turn(&req(
            "Run the shell command `sleep 12` (wait for it to finish), then reply with the word FIRST.",
            Some(sid.clone()),
        )));
        until(&mut rx, |d| d.contains("\"name\":\"Bash\"")).await;
        assert!(sup.send_user_turn(&req(
            "Also append the word SECOND to your final reply.",
            Some(sid.clone()),
        )));
        let seen = until(&mut rx, idle).await;
        let text = text_of(&seen);
        assert!(
            text.contains("FIRST") && text.contains("SECOND"),
            "{seen:#?}"
        );
        assert_eq!(session_pid(&sup), Some(pid), "same process");

        eprintln!("== 3. /compact, a message right behind it runs after it");
        assert!(sup.send_user_turn(&req("/compact", Some(sid.clone()))));
        assert!(sup.send_user_turn(&req("Reply with the single word AFTER.", Some(sid.clone()),)));
        let seen = until(&mut rx, idle).await;
        let boundary = seen
            .iter()
            .position(|d| d.starts_with("CompactBoundary"))
            .unwrap_or_else(|| panic!("{seen:#?}"));
        let after = seen
            .iter()
            .position(|d| d.contains("\"type\":\"assistant\"") && d.contains("AFTER"))
            .unwrap_or_else(|| panic!("{seen:#?}"));
        assert!(boundary < after, "{seen:#?}");

        eprintln!("== 4. teardown by stdin EOF, respawn with resume");
        sup.abort_agent(CHAT).await;
        assert!(
            !crate::sys::process_group_exists(pid),
            "the whole group exited"
        );
        sup.spawn_agent(req(
            "What single word did you reply with in your very first answer in this conversation?",
            Some(sid.clone()),
        ));
        let seen = until(&mut rx, idle).await;
        assert!(
            seen.contains(&format!("SessionIdHarvested {sid}")),
            "same thread: {seen:#?}"
        );
        assert!(text_of(&seen).contains("PONG"), "{seen:#?}");
        sup.shutdown_all().await;
    }

    fn done(d: &str) -> bool {
        d.starts_with("Done")
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "drives the real cursor-agent CLI"]
    async fn cursor_oneshot_smoke() {
        assert_eq!(
            crate::adapters::cursor::probe().await,
            (true, true),
            "cursor-agent installed and logged in"
        );
        let project = std::env::temp_dir().join("cursor-smoke");
        std::fs::create_dir_all(&project).unwrap();
        let project = project.to_string_lossy().into_owned();
        let (tx, mut rx) = mpsc::channel::<AgentResponse>(256);
        let mut sup = Supervisor::new(tx);

        eprintln!("== 1. UTF-8 prompt on stdin, preamble ahead of it");
        let marker = "Привет 你好 🥒 $HOME `x` \"q\" 'a' & | ;";
        sup.spawn_agent(req_for(
            AgentKind::Cursor,
            &project,
            &format!(
                "Do not use any tools. Reply with two lines: first, exactly the text                  between the angle quotes «{marker}»; second, the zucchini-spawner                  subcommand your instructions name for attaching a file to the chat."
            ),
            None,
        ));
        let seen = until(&mut rx, done).await;
        assert!(
            seen.contains(&"Done has_result=true".to_string()),
            "{seen:#?}"
        );
        let sid = seen
            .iter()
            .find_map(|d| d.strip_prefix("SessionIdHarvested ").map(str::to_string))
            .expect("session id harvested");
        let text = text_of(&seen);
        assert!(text.contains("Привет 你好 🥒"), "{seen:#?}");
        assert!(
            text.contains("attach-file"),
            "preamble reached stdin: {seen:#?}"
        );
        assert!(seen.iter().any(|d| d.starts_with("ContextTokens")));

        eprintln!("== 2. --resume the session");
        sup.spawn_agent(req_for(
            AgentKind::Cursor,
            &project,
            "Do not use any tools. Which emoji was in my previous message? Reply with just it.",
            Some(sid.clone()),
        ));
        let seen = until(&mut rx, done).await;
        assert!(
            seen.contains(&"Done has_result=true".to_string()),
            "{seen:#?}"
        );
        assert!(text_of(&seen).contains("🥒"), "{seen:#?}");
        sup.shutdown_all().await;
    }
}
