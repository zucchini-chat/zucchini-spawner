//! Per-agent adapter trait. Each adapter owns its CLI invocation and its
//! stdout-line normalization into claude-shape stream-json envelopes; the
//! Supervisor (in `agent.rs`) is generic infrastructure (process lifecycle,
//! prompt file, stderr buffering, signal escalation).
//!
//! Wire-format contract: every `AgentEvent::Frame(s)` lands verbatim in
//! `messages.body`. iOS's `SpawnerMessageDescriber` only renders claude's
//! shape, so adapters for other CLIs must normalize their frames to claude
//! shape before emitting.
//!
//! Adapter state lives for one turn (one user message → one agent run). The
//! Supervisor constructs a fresh adapter per `spawn_agent` call, holds it in
//! the spawned task, and drops it when the agent exits. Per-turn dedup state
//! like claude's `last_emitted_tokens` belongs on the adapter instance.

use anyhow::Result;
use futures::future::BoxFuture;
use smallvec::SmallVec;
use std::path::{Path, PathBuf};
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::writer::WriteEvent;

/// Progress callback handed to each adapter's `import()`. Adapter calls
/// `progress(pct).await` with monotonic 0..=100 values (throttled internally —
/// the per-percent gate lives inside each adapter, firing ~once per imported
/// chat, and the dispatcher's closure dedups identical rescaled values before
/// they hit the wire). The dispatcher in `main.rs` wraps this with a per-kind
/// rescaler (kind i of N takes the slice `i/N .. (i+1)/N`) and writes the
/// rescaled value to `machines.claude_history_import_status` via the writer
/// channel. Per-kind 100% is reserved for the dispatcher (it emits
/// `"finished"` exactly once at the very end, after every kind has run).
///
/// The callback is **async** so the dispatcher can deliver each status with a
/// blocking channel send rather than a droppable `try_send`: the status
/// channel is shared with the import's bulk row writes, and a dropped terminal
/// `finished` strands the client's progress modal forever. Awaiting also
/// applies natural backpressure — the import loop slows to the writer's drain
/// rate instead of overrunning the channel.
pub type ImportProgress = Box<dyn Fn(u8) -> BoxFuture<'static, ()> + Send + Sync>;

/// Picks the adapter at spawn time. Mirrors the `chats.agent_kind` enum
/// stored in Postgres.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentKind {
    Claude,
    Cursor,
    Codex,
    Hermes,
    Gemini,
}

/// Per-kind metadata + behavior table. Each `adapters/<kind>.rs` exports a
/// `pub const DESCRIPTOR: AdapterDescriptor` that wires its `AgentKind`
/// variant to the right constructor, columns, probe, and importer. The
/// dispatch methods on `AgentKind` (`make_adapter`, `install_columns`,
/// `probe`, `import`) all resolve through `AgentKind::descriptor(self)` →
/// the matching entry in `ADAPTERS`, so adding a new adapter is "new
/// `adapters/<kind>.rs` with a `DESCRIPTOR` + one slice entry in
/// `ADAPTERS`" — no per-variant match arms to touch.
pub struct AdapterDescriptor {
    pub kind: AgentKind,
    pub wire_name: &'static str,
    pub installed_col: &'static str,
    pub authenticated_col: &'static str,
    pub make: fn() -> Box<dyn AgentAdapter>,
    pub probe: fn() -> BoxFuture<'static, (bool, bool)>,
    pub import:
        fn(Uuid, Uuid, mpsc::Sender<WriteEvent>, ImportProgress) -> BoxFuture<'static, Result<()>>,
    /// Selective-forgetting ("prune-context") hooks, or `None` for hermes;
    /// see `crate::prune::PruneOps`. Set by each adapter's `PRUNE_OPS` const;
    /// dispatch goes through `AgentKind::prune_ops()`.
    pub prune: Option<crate::prune::PruneOps>,
}

/// The registry. Order is preserved for callers that iterate (probe fan-out,
/// import dispatcher); keep it stable so iOS sees a stable column order in
/// the startup-info PATCH and the per-kind progress slices line up across
/// builds. The `adapter_registry_consistent` test is the drift guard
/// coupling this slice to `AgentKind::ALL` and the per-kind `wire_name`s.
pub const ADAPTERS: &[&AdapterDescriptor] = &[
    &crate::adapters::claude::DESCRIPTOR,
    &crate::adapters::cursor::DESCRIPTOR,
    &crate::adapters::codex::DESCRIPTOR,
    &crate::adapters::hermes::DESCRIPTOR,
    &crate::adapters::gemini::DESCRIPTOR,
];

impl AgentKind {
    /// Strict parse: only the closed set of supported adapter kinds (those
    /// with a `DESCRIPTOR` in `ADAPTERS`) maps to `Some(_)`. Unknown values
    /// (e.g. `"hermes"`, `"gemini"` — permitted by the backend whitelist but
    /// not yet implemented as adapters) return `None`, and the caller refuses
    /// to mirror the chat rather than silently coercing to claude.
    /// Pre-migration rows where the column is absent are handled at the
    /// caller (defaulting to `AgentKind::Claude`) — strictness only kicks in
    /// when a value is present but unrecognized.
    pub fn parse(s: &str) -> Option<Self> {
        ADAPTERS.iter().find(|a| a.wire_name == s).map(|a| a.kind)
    }

    /// Every supported kind, for fan-outs (probe loops, startup-info report).
    /// Kept as a hand-listed `const` slice (and not derived from `ADAPTERS`
    /// at compile time — `const fn` can't iterate slices yet stably enough
    /// for this) so it stays `const`-evaluable; the
    /// `adapter_registry_consistent` test couples it to `ADAPTERS`.
    pub const ALL: &'static [AgentKind] = &[
        AgentKind::Claude,
        AgentKind::Cursor,
        AgentKind::Codex,
        AgentKind::Hermes,
        AgentKind::Gemini,
    ];

    /// Look up this variant's descriptor in `ADAPTERS`. Panics if the
    /// variant is missing from the registry — but the
    /// `adapter_registry_consistent` test fails before any panic could
    /// reach production: every variant in `AgentKind::ALL` must resolve to
    /// exactly one descriptor, and the test runs on every `cargo test`.
    pub fn descriptor(self) -> &'static AdapterDescriptor {
        ADAPTERS.iter().copied().find(|d| d.kind == self).expect(
            "AgentKind variant missing from ADAPTERS — coupling enforced by \
             adapter_registry_consistent test",
        )
    }

    /// Constructs the per-turn adapter for this kind. Single source of truth
    /// for the `AgentKind → Box<dyn AgentAdapter>` mapping — the supervisor
    /// in `agent.rs` calls this and anything else that needs a fresh adapter
    /// goes through here too.
    pub fn make_adapter(self) -> Box<dyn AgentAdapter> {
        (self.descriptor().make)()
    }

    /// Per-kind (`installed_col`, `authenticated_col`) on the `machines` row
    /// — the writer's startup PATCH builder fans out a single boolean pair per
    /// `AgentKind` into the nullable boolean columns. Keeping this on
    /// `AgentKind` (instead of inline in `writer.rs`) means a future variant
    /// only touches the descriptor in its `adapters/<kind>.rs`.
    pub fn install_columns(self) -> (&'static str, &'static str) {
        let d = self.descriptor();
        (d.installed_col, d.authenticated_col)
    }

    /// Probes install + auth state for this kind. Returns
    /// `(installed, authenticated)` — the writer flattens one pair per
    /// registered kind into a single PATCH on `machines`. `async` because
    /// probes may shell out or hit the filesystem.
    pub async fn probe(self) -> (bool, bool) {
        (self.descriptor().probe)().await
    }

    /// Selective-forgetting hooks for this kind, or `None` for hermes;
    /// see `crate::prune::PruneOps`. The `prune-context` flow (`control.rs`
    /// pre-scan, `main.rs` rewrite) dispatches through here rather than matching
    /// the variant, so prune support is a per-kind `AdapterDescriptor::prune` flip.
    pub fn prune_ops(self) -> Option<&'static crate::prune::PruneOps> {
        self.descriptor().prune.as_ref()
    }

    /// Base dir the kind's CLI stores per-session transcripts under — searched by
    /// the prune resolvers (`PruneOps::find_session`). Honors each CLI's
    /// relocation env var (semantics differ, hence the per-kind match), else
    /// `$HOME/.<cli>`. `None` for hermes (no `PruneOps`).
    ///
    /// gemini: `GEMINI_CLI_HOME` REPLACES `$HOME`, then gemini-cli appends
    /// `.gemini`, so we join `.gemini` onto it. Verified against the bundled
    /// gemini-cli `Storage.getGlobalGeminiDir` (`homedir()` = `GEMINI_CLI_HOME ||
    /// os.homedir()`, then `join(".gemini")`).
    ///
    /// An empty env value is treated as unset (the CLIs `||` past it) so a stray
    /// `CODEX_HOME=` doesn't resolve to `/`.
    pub fn cli_home(self) -> Option<PathBuf> {
        // `$HOME/.<subdir>` fallback shared by all three.
        fn home_subdir(subdir: &str) -> Option<PathBuf> {
            std::env::var_os("HOME")
                .filter(|h| !h.is_empty())
                .map(|h| PathBuf::from(h).join(subdir))
        }
        // The CLI's relocation env var when set non-empty.
        fn env_dir(var: &str) -> Option<PathBuf> {
            std::env::var_os(var)
                .filter(|s| !s.is_empty())
                .map(PathBuf::from)
        }
        match self {
            AgentKind::Claude => env_dir("CLAUDE_CONFIG_DIR").or_else(|| home_subdir(".claude")),
            AgentKind::Codex => env_dir("CODEX_HOME").or_else(|| home_subdir(".codex")),
            // GEMINI_CLI_HOME replaces $HOME; gemini-cli appends `.gemini` to it.
            AgentKind::Gemini => env_dir("GEMINI_CLI_HOME")
                .map(|h| h.join(".gemini"))
                .or_else(|| home_subdir(".gemini")),
            // Cursor's per-session content-addressed store lives under
            // `~/.cursor/chats/<projectHash>/<sessionUuid>/store.db` (the prune
            // path globs beneath this base). cursor-agent has no documented
            // home-relocation env var, so there's no `env_dir` override here.
            AgentKind::Cursor => home_subdir(".cursor"),
            AgentKind::Hermes => None,
        }
    }

    /// One-shot per-kind history importer. Each adapter walks its own
    /// transcript store (claude: `~/.claude/projects/*/*.jsonl`; cursor:
    /// `~/Library/Application Support/Cursor/User/{global,workspaceStorage}`
    /// sqlite blobs) and sends `WriteEvent::PutProject` / `PutChat` /
    /// `PutMessage` for everything it finds. Progress is reported via
    /// `progress(pct)` with pct in 0..=100 — the dispatcher in `main.rs`
    /// rescales per-kind progress into a single 0..99 bar (it owns 100 /
    /// `finished`). Adapters MUST NOT emit `WriteEvent::ImportStatus`
    /// themselves: status is owned by the dispatcher so per-kind progress
    /// composes correctly across the fan-out.
    ///
    /// Best-effort: an `Err` here is logged by the dispatcher and the next
    /// kind still runs — matches the existing per-session warn-and-continue
    /// posture inside the claude importer.
    pub async fn import(
        self,
        machine_id: Uuid,
        user_id: Uuid,
        write_tx: mpsc::Sender<WriteEvent>,
        progress: ImportProgress,
    ) -> Result<()> {
        (self.descriptor().import)(machine_id, user_id, write_tx, progress).await
    }
}

/// Inputs the Supervisor hands to `prepare_command` for each turn. Lifetime
/// borrows the prompt file path the Supervisor wrote (cleaned up after the
/// turn) and the chat metadata.
pub struct TurnContext<'a> {
    pub chat_id: &'a str,
    pub prompt_file: &'a Path,
    pub project_path: Option<&'a str>,
    pub worktree: bool,
    /// None on the first turn for the chat (or when the previous turn never
    /// reached the session-id harvest path). Adapter decides whether/how to
    /// pass it to the underlying CLI.
    pub agent_session_id: Option<&'a str>,
    /// Sender's `machine_users.is_sandboxed`. Claude adapter gates
    /// `--dangerously-skip-permissions` on `!is_sandboxed` so sandboxed
    /// invitees fall back to claude's default permission gating (which
    /// auto-denies tools in `--print`); cursor adapter gates `--force`
    /// (aka `--yolo`) the same way so cursor's default per-command deny
    /// path kicks in for sandboxed members. Workspace trust and MCP
    /// approval (`--trust`, `--approve-mcps`) stay on regardless — they're
    /// needed for headless mode to function — so the cursor sandbox is
    /// weaker than claude's; see `adapters/cursor.rs::prepare_command`.
    pub is_sandboxed: bool,
    /// `chats.model` — verbatim `--model <X>` pass-through to the underlying
    /// CLI (migration 0035). `None` means "no flag, let the CLI pick its
    /// default" — empty / blank values from the DB are filtered to `None`
    /// at the construction site in `main.rs`, so adapter logic stays
    /// `if let Some(m) = ctx.model { ... }`. Adapter is responsible only
    /// for shell-escaping the string; we don't validate the model name
    /// here because the closed set drifts per-CLI and per-release (claude
    /// uses `opus`/`sonnet`/`haiku`, cursor uses `Composer 2.5 Fast`-style
    /// labels) — an invalid value surfaces as a CLI error in the chat,
    /// which is the right place to learn about it.
    pub model: Option<&'a str>,
    /// Sender's IANA timezone (`machine_users.timezone`, migration 0040), or
    /// `None` (older client / NULL column). Feeds [`current_time_in_tz_line`]
    /// so the agent knows its current local time each turn for `schedule-message
    /// --at`. `None` ⇒ line omitted.
    pub user_timezone: Option<&'a str>,
}

/// Inputs the Supervisor hands to `prepare_session_command` for a RESIDENT
/// session (claude). Same fields as `TurnContext` MINUS `prompt_file` — a
/// resident session is spawned once with stdin held open, and per-turn prompts
/// arrive as `encode_user_turn` stdin frames rather than a piped prompt file.
/// The session-level knobs (`model`, `worktree`, `is_sandboxed`,
/// `project_path`) are fixed at spawn; a change in any of them forces a
/// teardown + respawn-with-`--resume` (the Supervisor compares them against the
/// live session).
pub struct SessionContext<'a> {
    pub chat_id: &'a str,
    pub project_path: Option<&'a str>,
    pub worktree: bool,
    /// `Some(_)` to resume a prior on-disk transcript when (re)starting a
    /// resident process for a chat that already harvested a session id
    /// (torn down / crashed / knob-changed). `None` for a brand-new chat — claude
    /// self-generates the id and we harvest it from `system/init`.
    pub agent_session_id: Option<&'a str>,
    /// See `TurnContext::is_sandboxed`.
    pub is_sandboxed: bool,
    /// See `TurnContext::model`.
    pub model: Option<&'a str>,
    /// See `TurnContext::user_timezone`. Folded into the resident session's
    /// per-(re)spawn `--append-system-prompt` time line; refreshed on every
    /// teardown + respawn (the resident process can't get a fresh time line
    /// mid-session, so it's only as current as the last spawn).
    pub user_timezone: Option<&'a str>,
}

/// Per-line normalized events the adapter forwards to Supervisor's response
/// channel. `Frame(s)` lands verbatim in `messages.body`; the other variants
/// are side-channel events written to `chats` columns.
#[derive(Debug)]
pub enum AgentEvent {
    /// Claude-shape stream-json line. Persisted verbatim into `messages.body`
    /// (with the per-user E2E key). For claude this is the line off stdout
    /// unchanged; for other adapters this is a normalized claude-shape JSON.
    Frame(String),
    /// Cumulative context tokens for this turn — caller overwrites
    /// `chats.context_tokens`.
    ContextTokens(i64),
    /// Claude only — manual `/compact` or auto-compact completed. Carries
    /// `compact_metadata.post_tokens`. Other adapters don't expose this signal.
    CompactBoundary(i64),
    /// Harvested from the agent's first stdout frame; persisted to
    /// `chats.agent_session_id` so subsequent turns can resume.
    SessionIdHarvested(String),
    /// Marks a `result` frame seen — Supervisor uses this to set
    /// `Done.has_result` (one-shot adapters) and, in the resident claude
    /// session model, to drive the turn/Waiting state machine.
    ///
    /// `origin_is_task` distinguishes the two kinds of `result` a resident
    /// claude process emits on the same stdout: a **user-turn** result (the
    /// `origin` field is absent — this ends the in-flight turn, `false`) vs a
    /// **background-task wake** result (`origin = {"kind":"task-notification"}`
    /// — a monitor/`run_in_background` task fired post-turn; the user turn is
    /// already done, so this must NOT clear `turn_in_flight`, `true`). One-shot
    /// adapters never produce task-driven results and always emit `false`.
    Result { origin_is_task: bool },
    /// Resident model only — a background task (a `run_in_background` Bash, a
    /// `Monitor`) was armed and started. Carries the claude `task_id`. The
    /// Supervisor inserts it into the session's `live_tasks` set; a non-empty
    /// set with no in-flight turn is the "Waiting…" state. One-shot adapters
    /// never emit this (their process exits at the first `result`, before any
    /// task can fire).
    TaskStarted(String),
    /// Resident model only — a previously-`TaskStarted` background task reached
    /// a terminal `task_notification` (completed/failed/cancelled). Carries the
    /// same `task_id`; the Supervisor removes it from `live_tasks`.
    TaskFinished(String),
    /// The agent's own `prune-context` call has PERSISTED its result. Content-free
    /// CUE the main loop uses to drive the prune restart: when the chat has a queued
    /// `PruneRequest`, the loop aborts → rewrites → respawns NOW (strictly after the
    /// result landed, so the resumed agent sees its own prune + summary).
    ///
    /// CALL-KEYED, not chat-keyed: each adapter emits this ONLY for the
    /// `prune-context` call's own result, never a sibling tool's. The adapter is
    /// the only layer that sees both the tool_use (command = prune-context) and the
    /// matching result id, so it does the correlation and the cue stays content-free.
    /// Without this, a sibling tool's result in the same parallel batch (common on
    /// gemini — its `update_topic` meta-tool's result usually lands first) would fire
    /// the restart before the prune's own result persists, losing the prune.
    /// Per-adapter "prune call persisted" boundary: claude/gemini match their
    /// `tool_result` frame's id against the recorded `prune-context` `tool_use` id;
    /// codex matches the completed `command_execution`'s `command` (it has no
    /// standalone tool_result frame). See `crate::prune` flow.
    ToolResult,
}

pub trait AgentAdapter: Send + Sync {
    fn kind(&self) -> AgentKind;

    /// Per-turn prep + final shell-command string. Adapter owns: cwd cd,
    /// prompt piping, session-id flag selection, agent-specific flags
    /// (e.g. `--worktree` for claude, `--force --trust` for cursor).
    /// Sync — both current adapters do pure string-building. If a future
    /// adapter needs to e.g. `git worktree add` first, switch back to async
    /// (or do the I/O upstream in the Supervisor before this hop).
    fn prepare_command(&mut self, ctx: &TurnContext<'_>) -> Result<String>;

    /// Text prepended to the PLAINTEXT prompt file (never the synced/persisted
    /// `messages` row) for adapters with NO system-prompt injection point, which
    /// can't convey [`agent_capabilities_instructions`] otherwise. Default `None`
    /// (claude/cursor inject in `prepare_command`); also `None` on non-first turns
    /// for prepend-path adapters, so capabilities land exactly once per chat.
    /// First turn == `ctx.agent_session_id.is_none()` (no harvested session id yet).
    fn prompt_file_preamble(&self, _ctx: &TurnContext<'_>) -> Option<String> {
        None
    }

    /// VOLATILE current-local-time line ([`current_time_in_tz_line`]) prepended
    /// to the prompt-file plaintext EVERY turn (never the persisted row). Split
    /// from [`Self::prompt_file_preamble`] (first-message-only) because the time
    /// must be fresh each turn. Default returns it (correct for prepend-path
    /// codex/gemini/hermes); claude/cursor fold it into their per-turn capability
    /// block instead and override to `None` to avoid double-injection. `None` when
    /// no tz is known.
    fn prompt_file_time_line(&self, ctx: &TurnContext<'_>) -> Option<String> {
        current_time_in_tz_line(ctx.user_timezone)
    }

    /// Resident adapters (claude) run ONE process across many turns with stdin
    /// held open; one-shot adapters (codex/gemini/cursor/hermes — default
    /// `false`) keep the per-turn `prepare_command` spawn path. The Supervisor
    /// branches on this to pick the session vs one-shot lifecycle.
    fn is_resident(&self) -> bool {
        false
    }

    /// Resident: build the per-SESSION shell command (spawned once; stdin
    /// piped). Like `prepare_command` but with no prompt piping — turns arrive
    /// as `encode_user_turn` stdin frames. Default `unreachable!` because the
    /// Supervisor only calls it when `is_resident()` returned `true`, and the
    /// one-shot adapters never set that.
    fn prepare_session_command(&mut self, _ctx: &SessionContext<'_>) -> Result<String> {
        unreachable!("prepare_session_command called on a one-shot adapter")
    }

    /// Resident: serialize one user turn as a newline-terminated stdin frame.
    /// Default `unreachable!` (one-shot adapters pipe the prompt instead).
    fn encode_user_turn(&self, _text: &str) -> String {
        unreachable!("encode_user_turn called on a one-shot adapter")
    }

    /// Per-stdout-line processing. Stateful for the lifetime of one turn so
    /// future delta-shaped adapters (hermes/gemini) can buffer internally.
    /// Returns 0..N events to forward to Supervisor's response channel.
    ///
    /// Takes the `String` by value so the adapter can move it into
    /// `AgentEvent::Frame(line)` without re-cloning. The Supervisor reads
    /// lines via `BufReader::lines()` which already hands out owned `String`s,
    /// so passing by value avoids a per-frame heap allocation on the hot path.
    fn handle_line(&mut self, line: String) -> SmallVec<[AgentEvent; 2]>;

    /// Static text appended (never prepended — see below) to the **first user
    /// message of the chat** when the adapter has no `--append-system-prompt`
    /// equivalent. The Supervisor calls this only on the FIRST turn
    /// (`agent_session_id.is_none()`); on resume the suffix is already baked into
    /// the reconstructed history, so re-appending would pollute every turn.
    ///
    /// Default `None`. Claude injects via `--append-system-prompt` instead;
    /// hermes doesn't support prune-context. Cursor supports it but doesn't use
    /// this hook — it has no resumable conversation history to bake a suffix into
    /// (`--resume` rebuilds from the local store), so it re-sends the prune nudge
    /// in its every-turn stdin preamble instead (`adapters/cursor.rs`). Gemini and
    /// codex override to their tool-name-corrected variants
    /// (`Some(PRUNE_CONTEXT_INSTRUCTION_GEMINI)` /
    /// `Some(PRUNE_CONTEXT_INSTRUCTION_CODEX)`) — their only hook for persisting
    /// the prune nudge into the conversation.
    ///
    /// MUST be appended after the user's text, not prepended: codex derives the
    /// chat title from the first user message (`first_fallback_user_text` →
    /// `collapse_title` in `adapters/codex.rs`), so prepending would corrupt
    /// imported titles. The text is invisible to the user — it reaches only the
    /// CLI's stdin (`prompt_file`), never the stored `messages` row / bubble, and
    /// both adapters drop their user-echo frames.
    fn first_turn_prompt_suffix(&self) -> Option<&'static str> {
        None
    }

    /// Called once after the agent process for a turn has fully exited (so any
    /// on-disk transcript is closed and flushed), just before the Supervisor
    /// emits `Done`. Lets an adapter publish a corrected context-token reading
    /// sourced from disk state the live stream can't expose; the returned value
    /// (if any) is emitted as this turn's `AgentEvent::ContextTokens`.
    ///
    /// Default `None`. Codex overrides it: the live `--json` stream only carries
    /// the cumulative `total_token_usage` (grows unbounded across turns), while
    /// the real context occupancy (`last_token_usage.input_tokens`) lives only
    /// in the rollout's `token_count` records — see `adapters/codex.rs`. The
    /// other adapters report occupancy inline from the stream and return `None`.
    ///
    /// `agent_session_id` is the resume id passed into the turn (`None` on a
    /// brand-new chat, where the adapter may instead use a session id it
    /// harvested from the stream this turn).
    fn post_turn_context_tokens(&self, _agent_session_id: Option<&str>) -> Option<i64> {
        None
    }
}

/// Per-line frame-size cap used by every adapter to skip a full
/// `serde_json::Value` parse (or even a full-line substring scan) on multi-MB
/// frames — tool_result/edit/read frames can legitimately blow past hundreds
/// of KB. Lives here so adapters can't drift on the threshold.
pub(crate) const MAX_STREAM_FRAME_BYTES: usize = 65_536;

/// Shared shell-escape helper. Every adapter uses single-quote escaping for
/// command strings handed to the user's login shell. Kept here so all
/// `adapters/*.rs` modules can share it without reaching back into
/// `agent.rs`.
pub fn shell_escape(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

/// Instruction telling the agent how to attach a local file to its next message
/// via the `zucchini-spawner attach-file` CLI. Chat id and binary path stay in
/// env vars (`ZUCCHINI_CHAT_ID`, `ZUCCHINI_SPAWNER_BIN`) exported on every spawn
/// — `--chat-id` defaults to `ZUCCHINI_CHAT_ID`, so it's dropped from the command
/// line and the text is chat-agnostic and cacheable. Tool name is left
/// unspecified ("run the command"): every coding agent has a shell-exec facility.
pub const ATTACH_FILE_INSTRUCTION: &str =
    "To send a file to the user, run the command `\"$ZUCCHINI_SPAWNER_BIN\" attach-file <absolute-path>` before writing the message that should accompany the attachment.";

/// Body of the prune-context nudge with a per-adapter `$example` clause spliced
/// in. The shared text lives here EXACTLY ONCE; only the worked example varies
/// (claude/gemini name tools the same way the matcher's claude-shape aliases do;
/// codex needs a different example — see `PRUNE_CONTEXT_INSTRUCTION_CODEX`).
macro_rules! prune_context_instruction {
    ($toolname_frag:literal, $example:literal) => {
        concat!(
            "Reclaim context as you go: once you've extracted what you need from a tool output, prune it — noisy grep, an already-answered file, a large file you've already edited and moved past, junk: prune with `\"$ZUCCHINI_SPAWNER_BIN\" prune-context ",
            $toolname_frag,
            "--args \"<value glob>\" --summary \"<1-line digest of what you're keeping — the key takeaway from this tool result>\"`. --args is a glob (supports *) over the call's argument VALUES, not key names, and selects which calls match — each prune blanks only the MOST RECENT matching call, so repeat the command to prune older ones. Pass --args \"\" for a call you made with no arguments. To forget several outputs at once, repeat the flags in ONE call (one --summary per output, which closes that target; --tool-name is per-target, so repeat it on each — an omitted one falls back to matching any tool) — they apply as a single batch. E.g. ",
            $example,
            ". Run prune-context alone, never alongside other tool calls."
        )
    };
}

/// Instruction telling the agent to proactively reclaim context by permanently
/// forgetting tool outputs it no longer needs. Delivered via
/// `--append-system-prompt` (claude) or appended to the first user message
/// (gemini, which lacks that hook — see `first_turn_prompt_suffix`). Codex uses
/// `PRUNE_CONTEXT_INSTRUCTION_CODEX` instead (its tool names differ). Same
/// env-var convention as `ATTACH_FILE_INSTRUCTION`, so it's chat-agnostic and
/// cacheable. The spawner aborts + respawns the agent with a "continue" prompt
/// after a prune, so the instruction doesn't tell the agent to stop.
pub const PRUNE_CONTEXT_INSTRUCTION: &str = prune_context_instruction!(
    "--tool-name <ToolName> ",
    "`--tool-name \"Read\" --args \"src/main.ts\" --summary \"main.ts: initApp() wires routes then listens on PORT — entry point confirmed\" --tool-name Read --args \"build.log\" --summary \"build: passed, nothing relevant\"`"
);

/// Codex variant. Codex has no `Read`/`Grep`/`Edit` tools — file reads, greps,
/// and edits ALL run through two generic shell tools (`shell`/`exec_command`), so
/// no single native tool name targets a given file op (and naming one shell tool
/// misses the other). Rather than force a claude-shape `Bash` alias the agent
/// never actually emitted, the codex example OMITS `--tool-name` entirely — the
/// empty selector prunes on the `--args` path alone, which already disambiguates.
/// (Gemini keeps native names because its reads/greps/edits land on distinct,
/// self-describing tools; codex has no such per-op name.)
pub const PRUNE_CONTEXT_INSTRUCTION_CODEX: &str = prune_context_instruction!(
    "",
    "`--args \"src/main.ts\" --summary \"main.ts: initApp() wires routes then listens on PORT — entry point confirmed\" --args \"build.log\" --summary \"build: passed, nothing relevant\"` (omit --tool-name on codex)"
);

/// Gemini variant. The matcher DOES alias claude-shape `Read`→`read_file`, so the
/// shared `Read` example technically works — but a gemini agent sees its own tool
/// calls named `read_file`/`list_directory`/`replace`/`grep_search` in its
/// transcript, and the `Read` example invites it to guess wrong variants
/// (`ReadFile`, `List`, …) that literal-compare to nothing and fail. The example
/// here uses gemini's native names so `--tool-name` matches what the agent sees.
pub const PRUNE_CONTEXT_INSTRUCTION_GEMINI: &str = prune_context_instruction!(
    "--tool-name <ToolName> ",
    "`--tool-name \"read_file\" --args \"src/main.ts\" --summary \"main.ts: initApp() wires routes then listens on PORT — entry point confirmed\" --tool-name read_file --args \"build.log\" --summary \"build: passed, nothing relevant\"` (use your own tool names — `read_file`, `list_directory`, `replace`, `grep_search`, … — not claude's `Read`/`Edit`/`Grep`)"
);

/// How the agent schedules a follow-up via `zucchini-spawner schedule-message`.
/// Same env-var contract as `ATTACH_FILE_INSTRUCTION` (`ZUCCHINI_CHAT_ID`,
/// `ZUCCHINI_SPAWNER_BIN`) so it stays chat-agnostic and cacheable.
///
/// The agent emits a NAIVE local wall-clock (no offset/`Z`); the spawner anchors
/// it in the user's timezone (`normalize_deliver_at`) so the agent never does
/// offset/DST math (which it gets wrong). It just echoes the digits off the
/// per-turn current-local-time line ([`current_time_in_tz_line`]).
pub const SCHEDULE_MESSAGE_INSTRUCTION: &str =
    "To schedule a follow-up message to yourself in this chat for a specific future time, run the command `\"$ZUCCHINI_SPAWNER_BIN\" schedule-message --chat-id \"$ZUCCHINI_CHAT_ID\" --body \"<text>\" --at <local-datetime>`. Pass `--at` as a naive local datetime — the user's wall-clock time (e.g. \"2026-06-07T09:00:00\" for 9am), read off the current-local-time line. ALWAYS use this command for any scheduling, reminder, or self-follow-up request — it is the only mechanism that actually delivers a message back into this chat. Do NOT use your own built-in scheduling tools (the `/schedule` skill, ScheduleWakeup, cron routines, etc.); those fire in a context the user never sees here and will silently do nothing for them.";

/// Per-turn current-local-time line for the agent prompt, from the sender's IANA
/// timezone (`machine_users.timezone`, migration 0040). `None` when `iana_tz` is
/// absent or unknown — the caller omits the line (`SCHEDULE_MESSAGE_INSTRUCTION`
/// still works, just no tz hint). MUST be recomputed every turn (embeds `Utc::now()`).
pub fn current_time_in_tz_line(iana_tz: Option<&str>) -> Option<String> {
    let tz = iana_tz?.parse::<chrono_tz::Tz>().ok()?;
    let now = chrono::Utc::now().with_timezone(&tz);
    // Naive wall-clock (no `±HH:MM`/`Z`) — exactly what `schedule-message --at`
    // wants echoed back; an offset here only invites wrong offset math. The
    // `(timezone: …)` suffix carries zone identity; the spawner does the
    // offset/DST in `normalize_deliver_at`.
    Some(format!(
        "The user's current local time is {} (timezone: {}).",
        now.format("%Y-%m-%dT%H:%M:%S"),
        tz.name(),
    ))
}

/// Worktree-containment guidance (absolute worktree path + parent repo). Single
/// source of truth shared by the system-prompt path (claude, cursor) and the
/// first-message prepend path (codex, gemini, hermes). Why it's needed: the
/// harness chdirs into the worktree but doesn't tell the agent (or subagents) to
/// stay there, so absolute paths into the parent repo "just work" and edits leak
/// out — this rule plugs that hole.
pub fn worktree_instructions(worktree_abs: &str, parent_repo: &str) -> String {
    format!(
        "Worktree: {worktree_abs}\nParent repo: {parent_repo} (do not touch unless the user explicitly asks).\nKeep all edits and shell commands inside the worktree. If a path under the parent repo appears in context, rewrite it to the worktree before editing or running commands against it. When delegating to a subagent, repeat this rule and the worktree path — subagents don't inherit it."
    )
}

/// The capability instructions every adapter conveys — attach-file,
/// schedule-message, and (when `worktree` is `Some`) the worktree rule. One
/// helper so the texts can't drift across conveyance mechanisms: claude's
/// `--append-system-prompt`, cursor's stdin preamble, and the first-message
/// prepend (codex/gemini/hermes, no system-prompt injection point). `worktree`
/// is `Some` only when the turn runs in a worktree the agent must stay inside;
/// `None` omits the rule. `time_line` rides along for claude/cursor (rebuilt
/// fresh every turn). codex/gemini/hermes pass `None` here and inject the time
/// line separately every turn (`agent.rs` `prompt_file_time_line`), since this
/// static block reaches them only on turn 1.
pub fn agent_capabilities_instructions(
    worktree: Option<&WorktreeInstructions>,
    time_line: Option<&str>,
) -> String {
    let mut out = format!("{ATTACH_FILE_INSTRUCTION}\n\n{SCHEDULE_MESSAGE_INSTRUCTION}");
    if let Some(wt) = worktree {
        out.push_str("\n\n");
        out.push_str(&worktree_instructions(&wt.worktree_abs, &wt.parent_repo));
    }
    if let Some(line) = time_line {
        out.push_str("\n\n");
        out.push_str(line);
    }
    out
}

/// Shared `prompt_file_preamble` body for the prepend-path adapters
/// (codex/gemini/hermes): no system-prompt injection point, worktree ignored,
/// time line carried separately via `prompt_file_time_line`. Emits the static
/// capability block on the FIRST turn only (`agent_session_id.is_none()`), `None`
/// on resume. Centralized so the first-turn gate can't drift between the three.
/// `worktree`/`time_line` are both `None` here on purpose.
pub fn first_message_capabilities_preamble(ctx: &TurnContext<'_>) -> Option<String> {
    if ctx.agent_session_id.is_some() {
        return None;
    }
    Some(agent_capabilities_instructions(None, None))
}

/// Already-resolved absolute worktree path + parent repo for
/// [`agent_capabilities_instructions`] / [`worktree_instructions`]. Each adapter
/// computes these from its own `--worktree` naming scheme (claude:
/// `<project>/.claude/worktrees/`; cursor: `~/.cursor/worktrees/<repo>/`).
pub struct WorktreeInstructions {
    pub worktree_abs: String,
    pub parent_repo: String,
}

/// Trim + `starts_with('{')` gate + `serde_json::from_str::<Value>` for
/// adapters that need to dispatch on a frame's `type` field. Returns `None`
/// when the line isn't an object — caller decides whether to forward as a
/// raw `Frame` (codex's permissive path) or drop with a debug log (cursor's
/// stricter path). Logs parse failures at debug so a wire-format drift leaves
/// a breadcrumb in the spawner log without spamming on every healthy line.
pub(crate) fn parse_json_obj(line: &str) -> Option<serde_json::Value> {
    let s = line.trim();
    if !s.starts_with('{') {
        return None;
    }
    match serde_json::from_str::<serde_json::Value>(s) {
        Ok(v) => v.is_object().then_some(v),
        Err(e) => {
            tracing::debug!("JSON parse failed: {}", e);
            None
        }
    }
}

/// Four-zero `usage` block in claude's snake_case shape — the same constant
/// every non-claude adapter stamps into mid-turn assistant envelopes when
/// per-frame usage isn't available. iOS reads `chats.context_tokens` for the
/// live counter (driven by `AgentEvent::ContextTokens`) so the zeros are
/// cosmetic — the persisted body still has to carry *something* here because
/// `SpawnerMessageDescriber` and other downstream consumers may parse it.
pub(crate) fn claude_zero_usage() -> serde_json::Value {
    serde_json::json!({
        "input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "output_tokens": 0,
    })
}

/// Wraps an already-built array of claude content blocks (`[{"type":"text",...}]`,
/// mixed text + tool_use, etc.) in the claude-shape assistant envelope with
/// zero usage. Output is the serialized stream-json line ready to ship as
/// `AgentEvent::Frame(_)`. Used by cursor's live `normalize_assistant_frame`
/// (where the blocks come straight off the cursor wire) — codex and the
/// cursor importer use the more specialized text / tool_use builders below
/// because they build single-block envelopes from primitives.
pub(crate) fn claude_assistant_envelope(content: serde_json::Value) -> String {
    serde_json::json!({
        "type": "assistant",
        "message": {
            "content": content,
            "usage": claude_zero_usage(),
        },
    })
    .to_string()
}

/// Convenience: assistant envelope carrying a single `text` block. Codex's
/// `agent_message` item and the cursor importer's text-only bubble both emit
/// this shape; calling this from both sites keeps the wire identical (same
/// key order, same zero usage) so iOS sees one envelope shape per text-only
/// frame across all adapters.
pub(crate) fn claude_assistant_text_envelope(text: &str) -> String {
    claude_assistant_envelope(serde_json::json!([
        { "type": "text", "text": text },
    ]))
}

/// Convenience: assistant envelope carrying a single `tool_use` block.
/// `input` is forwarded verbatim — caller picks the shape (the full cursor
/// args object, the full codex item, claude-renamed keys for cursor's tool
/// re-mapping, ...). Same call sites as `claude_assistant_text_envelope`:
/// codex's file_change / command_execution / web_search items, cursor's live
/// tool_call.completed normalizer + oversize-frame synthesizer, and the
/// cursor importer's persisted tool_use builder.
pub(crate) fn claude_tool_use_envelope(id: &str, name: &str, input: serde_json::Value) -> String {
    claude_assistant_envelope(serde_json::json!([
        {
            "type": "tool_use",
            "id": id,
            "name": name,
            "input": input,
        },
    ]))
}

/// Per-turn dedup helper for adapters whose underlying CLI reports a
/// cumulative context-token count and may legitimately emit the same value
/// twice in a turn (claude: usage repeats across thinking-then-text frames
/// in the same turn; codex: hypothetical streamed-usage variant or a
/// re-emitted final frame). Returns `Some(tokens)` on a fresh value (and
/// stores it as the new high-water mark), `None` when the value matches the
/// last emission. Cursor doesn't use this — its tokens come from a single
/// `result.usage` so dedup is structurally impossible.
#[derive(Default)]
pub(crate) struct LastTokensDedup {
    last: Option<i64>,
}

impl LastTokensDedup {
    pub(crate) fn observe(&mut self, tokens: i64) -> Option<i64> {
        if self.last == Some(tokens) {
            return None;
        }
        self.last = Some(tokens);
        Some(tokens)
    }
}

/// Shared `probe()` shape for adapters whose auth check is a sync
/// filesystem-only function — runs the binary-on-PATH check first, then
/// `spawn_blocking`s the sync auth fn so a slow disk doesn't stall the
/// runtime thread. Returns `(installed, authenticated)`; an `auth_fn` panic
/// surfaces as `(true, false)` via the `.unwrap_or(false)` on join.
///
/// Cursor's probe is NOT layered on top of this because its auth check is
/// async (it shells out to `cursor-agent status`) — see `cursor::probe()`
/// for the bespoke path.
pub(crate) async fn probe_with_blocking_auth(bin: &str, auth_fn: fn() -> bool) -> (bool, bool) {
    if !crate::shell::binary_on_path(bin).await {
        return (false, false);
    }
    let authed = tokio::task::spawn_blocking(auth_fn).await.unwrap_or(false);
    (true, authed)
}

/// File-presence test shared by adapters' filesystem-only auth probes: `true`
/// only if `path` exists, is `stat`-able, and is non-empty. A missing file,
/// any metadata error, or a zero-length file all map to `false`. Centralizing
/// this keeps the "presence + non-empty" auth contract identical across
/// adapters so a future hardening (symlink handling, zero-byte sentinels, …)
/// can't be applied to one probe and forgotten in another.
pub(crate) fn file_nonempty(path: &std::path::Path) -> bool {
    std::fs::metadata(path)
        .map(|m| m.len() > 0)
        .unwrap_or(false)
}

#[cfg(test)]
impl<'a> TurnContext<'a> {
    /// Minimal test `TurnContext` (fixed chat id, `/tmp/proj`, non-worktree, no
    /// tz). The four args are what adapter tests vary; other fields go via
    /// struct-update: `TurnContext { user_timezone: Some(..), ..for_test(..) }`.
    /// One builder so a new field isn't an N-site edit across the test modules.
    pub(crate) fn for_test(
        prompt_file: &'a Path,
        agent_session_id: Option<&'a str>,
        is_sandboxed: bool,
        model: Option<&'a str>,
    ) -> Self {
        TurnContext {
            chat_id: "00000000-0000-0000-0000-000000000000",
            prompt_file,
            project_path: Some("/tmp/proj"),
            worktree: false,
            agent_session_id,
            is_sandboxed,
            model,
            user_timezone: None,
        }
    }
}

/// Test-only stringifier shared by every adapter's `run()` helper. Maps a
/// turn's emitted events to human-readable tags for assertion equality. Lives
/// here (next to the enum) so a new `AgentEvent` variant forces exactly ONE
/// update instead of three drifting per-adapter copies.
#[cfg(test)]
pub(crate) fn stringify_events(events: SmallVec<[AgentEvent; 2]>) -> Vec<String> {
    events
        .into_iter()
        .map(|e| match e {
            AgentEvent::Frame(s) => format!("Frame({})", s),
            AgentEvent::ContextTokens(n) => format!("ContextTokens({})", n),
            AgentEvent::CompactBoundary(n) => format!("CompactBoundary({})", n),
            AgentEvent::SessionIdHarvested(s) => format!("SessionIdHarvested({})", s),
            AgentEvent::Result {
                origin_is_task: false,
            } => "Result".to_string(),
            AgentEvent::Result {
                origin_is_task: true,
            } => "Result(task)".to_string(),
            AgentEvent::TaskStarted(s) => format!("TaskStarted({})", s),
            AgentEvent::TaskFinished(s) => format!("TaskFinished({})", s),
            AgentEvent::ToolResult => "ToolResult".to_string(),
        })
        .collect()
}

/// Test-only: unwrap a `Frame(<json>)` tag produced by [`stringify_events`]
/// back into parsed JSON. Panics if the tag isn't a Frame or the payload isn't
/// valid JSON — adapters assert on normalized claude-shape frames a lot.
#[cfg(test)]
pub(crate) fn frame_value(event: &str) -> serde_json::Value {
    let inner = event
        .strip_prefix("Frame(")
        .and_then(|s| s.strip_suffix(')'))
        .expect("event was not a Frame");
    serde_json::from_str(inner).expect("frame payload was not valid JSON")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Single drift guard for the descriptor registry. Replaces the old
    /// per-method exhaustive-match tests now that dispatch is data-driven
    /// (one entry in `ADAPTERS` per kind, not seven match arms).
    ///
    /// Asserts:
    ///   1. `ADAPTERS.len() == AgentKind::ALL.len()` — no variant in `ALL`
    ///      without a descriptor and no descriptor without a variant in
    ///      `ALL`.
    ///   2. `wire_name`s are unique across `ADAPTERS` — copy-paste in a new
    ///      `adapters/<kind>.rs` that forgets to change the wire string
    ///      would otherwise silently mis-route `parse`.
    ///   3. Every variant in `AgentKind::ALL` resolves to exactly one
    ///      descriptor — catches an `AgentKind` variant added to `ALL`
    ///      without a corresponding `ADAPTERS` entry.
    ///   4. Round-trip: `AgentKind::parse(d.wire_name) == Some(d.kind)` for
    ///      every descriptor — catches a wire-string change in one place but
    ///      not the other.
    #[test]
    fn adapter_registry_consistent() {
        // (1) Length parity.
        assert_eq!(
            ADAPTERS.len(),
            AgentKind::ALL.len(),
            "ADAPTERS.len() ({}) != AgentKind::ALL.len() ({}) — add the missing entry to \
             whichever slice is short",
            ADAPTERS.len(),
            AgentKind::ALL.len(),
        );

        // (2) Wire names unique. Sort+dedup over a Vec of &'static str.
        let mut wires: Vec<&'static str> = ADAPTERS.iter().map(|d| d.wire_name).collect();
        wires.sort_unstable();
        let before = wires.len();
        wires.dedup();
        assert_eq!(
            wires.len(),
            before,
            "ADAPTERS has duplicate wire_name entries — every adapter must use a distinct wire \
             string (it's the key for AgentKind::parse)",
        );

        // (3) Every AgentKind::ALL variant resolves to exactly one descriptor.
        for v in AgentKind::ALL {
            let matches: Vec<&&AdapterDescriptor> =
                ADAPTERS.iter().filter(|d| d.kind == *v).collect();
            assert_eq!(
                matches.len(),
                1,
                "AgentKind::{:?} resolves to {} descriptors in ADAPTERS, expected exactly 1",
                v,
                matches.len(),
            );
        }

        // (4) Round-trip parse for every descriptor.
        for d in ADAPTERS {
            assert_eq!(
                AgentKind::parse(d.wire_name),
                Some(d.kind),
                "AgentKind::parse({:?}) did not round-trip to {:?} — likely a wire_name mismatch \
                 in the descriptor",
                d.wire_name,
                d.kind,
            );
        }
    }

    /// Only gemini/codex (no `--append-system-prompt` hook) carry the prune nudge
    /// on the first user message. The rest must return `None`: a non-`None` would
    /// double-inject (claude via `--append-system-prompt`, cursor via its every-turn
    /// stdin preamble) or inject a no-op nudge (hermes has no prune support). Gemini
    /// and codex each use a tool-name-corrected variant (native names / Bash, not Read).
    #[test]
    fn first_turn_prompt_suffix_only_for_gemini_and_codex() {
        for v in AgentKind::ALL {
            let suffix = v.make_adapter().first_turn_prompt_suffix();
            match v {
                AgentKind::Gemini => assert_eq!(
                    suffix,
                    Some(PRUNE_CONTEXT_INSTRUCTION_GEMINI),
                    "Gemini must append the gemini-specific prune nudge (read_file example) \
                     on the first turn (it has no --append-system-prompt hook)",
                ),
                AgentKind::Codex => assert_eq!(
                    suffix,
                    Some(PRUNE_CONTEXT_INSTRUCTION_CODEX),
                    "Codex must append the codex-specific prune nudge (Bash example)",
                ),
                _ => assert_eq!(
                    suffix, None,
                    "AgentKind::{v:?} must NOT set a first-turn prompt suffix",
                ),
            }
        }
    }

    #[test]
    fn current_time_line_known_tz_is_naive_wall_clock() {
        // Known IANA id → line names the zone + carries a NAIVE wall-clock (no
        // offset), the shape the agent echoes to `schedule-message --at`.
        let line = current_time_in_tz_line(Some("America/Los_Angeles"))
            .expect("known IANA id yields a line");
        assert!(
            line.contains("America/Los_Angeles"),
            "line names the timezone: {line}"
        );
        // No signed `±HH:MM` offset and no `Z` — the timestamp is bare local time.
        assert!(
            !line.contains("+00:00")
                && !line.contains("-08:00")
                && !line.contains("-07:00")
                && !line.contains('Z'),
            "line must carry no UTC offset: {line}"
        );
    }

    #[test]
    fn current_time_line_absent_or_garbage_is_none() {
        // No tz (older client / NULL column) → no line.
        assert!(current_time_in_tz_line(None).is_none());
        // Unparseable IANA id → no line (no tz hint, not a bogus offset).
        assert!(current_time_in_tz_line(Some("Not/AZone")).is_none());
        assert!(current_time_in_tz_line(Some("")).is_none());
    }

    #[test]
    fn capabilities_block_appends_time_line_when_present() {
        // Time line lands when passed, absent otherwise.
        let with = agent_capabilities_instructions(None, Some("CLOCK-MARKER"));
        assert!(with.contains("CLOCK-MARKER"), "time line appended: {with}");
        assert!(with.contains("attach-file"), "static block still present");
        let without = agent_capabilities_instructions(None, None);
        assert!(
            !without.contains("CLOCK-MARKER"),
            "no time line when None: {without}"
        );
    }

    #[test]
    fn first_message_preamble_first_turn_then_resume() {
        // Prepend-path body: capability block on first turn (no session id),
        // omitted on resume; worktree never conveyed this way.
        use std::path::PathBuf;
        let prompt_file = PathBuf::from("/tmp/p.txt");
        let first = TurnContext::for_test(&prompt_file, None, false, None);
        let preamble = first_message_capabilities_preamble(&first).expect("first turn → Some");
        assert!(preamble.contains("attach-file"), "got: {preamble}");
        assert!(preamble.contains("schedule-message"), "got: {preamble}");
        assert!(!preamble.contains("Worktree:"), "got: {preamble}");

        let resume = TurnContext::for_test(&prompt_file, Some("sid"), false, None);
        assert!(
            first_message_capabilities_preamble(&resume).is_none(),
            "resume → None"
        );
    }
}
