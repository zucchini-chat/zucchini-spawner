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
use std::path::Path;
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::writer::WriteEvent;

/// Progress callback handed to each adapter's `import()`. Adapter calls
/// `progress(pct)` with monotonic 0..=100 values (throttled internally — the
/// 5%-step gate lives inside each adapter, so the dispatcher's closure can
/// fire on every call without flooding). The dispatcher in `main.rs` wraps
/// this with a per-kind rescaler (kind i of N takes the slice
/// `i/N .. (i+1)/N`) and writes the rescaled value to `machines.
/// claude_history_import_status` via the writer channel. Per-kind 100% is
/// reserved for the dispatcher (it emits `"finished"` exactly once at the
/// very end, after every kind has run).
pub type ImportProgress = Box<dyn Fn(u8) + Send + Sync>;

/// Picks the adapter at spawn time. Mirrors the `chats.agent_kind` enum
/// stored in Postgres.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentKind {
    Claude,
    Cursor,
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
}

/// The registry. Order is preserved for callers that iterate (probe fan-out,
/// import dispatcher); keep it stable so iOS sees a stable column order in
/// the startup-info PATCH and the per-kind progress slices line up across
/// builds. The `adapter_registry_consistent` test is the drift guard
/// coupling this slice to `AgentKind::ALL` and the per-kind `wire_name`s.
pub const ADAPTERS: &[&AdapterDescriptor] = &[
    &crate::adapters::claude::DESCRIPTOR,
    &crate::adapters::cursor::DESCRIPTOR,
];

impl AgentKind {
    /// Strict parse: only the closed set of supported adapter kinds (those
    /// with a `DESCRIPTOR` in `ADAPTERS`) maps to `Some(_)`. Unknown values
    /// (e.g. `"codex"`, `"hermes"`, `"gemini"` — permitted by the backend
    /// whitelist but not yet implemented as adapters) return `None`, and the
    /// caller refuses to mirror the chat rather than silently coercing to
    /// claude. Pre-migration rows where the column is absent are handled at
    /// the caller (defaulting to `AgentKind::Claude`) — strictness only
    /// kicks in when a value is present but unrecognized.
    pub fn parse(s: &str) -> Option<Self> {
        ADAPTERS.iter().find(|a| a.wire_name == s).map(|a| a.kind)
    }

    /// Every supported kind, for fan-outs (probe loops, startup-info report).
    /// Kept as a hand-listed `const` slice (and not derived from `ADAPTERS`
    /// at compile time — `const fn` can't iterate slices yet stably enough
    /// for this) so it stays `const`-evaluable; the
    /// `adapter_registry_consistent` test couples it to `ADAPTERS`.
    pub const ALL: &'static [AgentKind] = &[AgentKind::Claude, AgentKind::Cursor];

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
    /// `AgentKind` into the four nullable boolean columns. Keeping this on
    /// `AgentKind` (instead of inline in `writer.rs`) means a future variant
    /// only touches the descriptor in its `adapters/<kind>.rs`.
    pub fn install_columns(self) -> (&'static str, &'static str) {
        let d = self.descriptor();
        (d.installed_col, d.authenticated_col)
    }

    /// Probes install + auth state for this kind. Returns
    /// `(installed, authenticated)` — the writer flattens both pairs into a
    /// single PATCH on `machines`'s four boolean columns. `async` because
    /// probes shell out for both kinds.
    pub async fn probe(self) -> (bool, bool) {
        (self.descriptor().probe)().await
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
    /// `compactMetadata.postTokens`. Other adapters don't expose this signal.
    CompactBoundary(i64),
    /// Harvested from the agent's first stdout frame; persisted to
    /// `chats.agent_session_id` so subsequent turns can resume.
    SessionIdHarvested(String),
    /// Marks a final/result frame seen — Supervisor uses this to set
    /// `Done.has_result`.
    Result,
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

    /// Per-stdout-line processing. Stateful for the lifetime of one turn so
    /// future delta-shaped adapters (hermes/gemini) can buffer internally.
    /// Returns 0..N events to forward to Supervisor's response channel.
    ///
    /// Takes the `String` by value so the adapter can move it into
    /// `AgentEvent::Frame(line)` without re-cloning. The Supervisor reads
    /// lines via `BufReader::lines()` which already hands out owned `String`s,
    /// so passing by value avoids a per-frame heap allocation on the hot path.
    fn handle_line(&mut self, line: String) -> SmallVec<[AgentEvent; 2]>;
}

/// Per-line frame-size cap used by both adapters to skip a full
/// `serde_json::Value` parse (or even a full-line substring scan) on multi-MB
/// frames — tool_result/edit/read frames can legitimately blow past hundreds
/// of KB. Lives here so the two adapters can't drift on the threshold.
pub(crate) const MAX_STREAM_FRAME_BYTES: usize = 65_536;

/// Shared shell-escape helper. Both adapters use single-quote escaping for
/// command strings handed to the user's login shell. Kept here so both
/// `adapters/claude.rs` and `adapters/cursor.rs` can share it without
/// reaching back into `agent.rs`.
pub fn shell_escape(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
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
}
