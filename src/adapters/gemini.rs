//! gemini (Google `gemini-cli`) adapter. Normalizes gemini's `-o stream-json`
//! JSONL frames into claude-shape envelopes on the wire so iOS's
//! `SpawnerMessageDescriber` (which only knows claude's wire format) can render
//! them WITHOUT any gemini-specific branches in the iOS code: every gemini tool
//! call is remapped to the equivalent claude tool name on the wire so iOS's
//! `toolSummary` table picks it up as if it were a claude `tool_use`. Single
//! seam, single source of truth — same posture as `codex.rs`.
//!
//! Frame mapping (gemini → claude-shape), observed empirically on gemini-cli
//! 0.44.1 (`-o stream-json`, JSONL, one object per line):
//!
//! - `init` → `SessionIdHarvested` only (no Frame; matches claude's init-skip).
//!   `session_id` echoes the exact UUID we minted and passed via `--session-id`;
//!   harvest it regardless so the resume path on the next turn is exact.
//! - `message` role=user → drop (echo of our own prompt).
//! - `message` role=assistant → buffered, NOT emitted immediately. Assistant
//!   text arrives as MULTIPLE incremental `delta:true` chunks per turn (verified
//!   against the captures in `tmp/gemini-probe/*.jsonl`: each chunk carries only
//!   the NEW text, never a cumulative re-send). We APPEND every chunk's content
//!   into a per-turn `pending_assistant_text` buffer and flush it as ONE
//!   claude-shape assistant text envelope at the next boundary (tool_use,
//!   result). The spawner writes one `messages` row per emitted Frame and those
//!   rows are immutable / never grow (see crate `CLAUDE.md` message-frame
//!   invariant), so emitting one Frame PER CHUNK would fragment a single reply
//!   into N separate chat bubbles. Coalescing is mandatory, not cosmetic — it's
//!   the only place the fragmentation can be fixed (the rows don't exist yet).
//!   Mirrors the hermes adapter's buffer-and-flush-at-boundary fix.
//! - `tool_use` → flush any pending assistant text FIRST (so a "text → tool →
//!   text" turn renders as alternating coherent text/tool bubbles), THEN Frame:
//!   claude tool_use envelope under the mapped claude tool name (see
//!   `normalize_tool_use`). Gemini meta-tools `update_topic` / `exit_plan_mode`
//!   are filtered out entirely AND do NOT flush — they produce no Frame, so
//!   flushing on them would split the surrounding text needlessly.
//! - `tool_result` → drop (claude UI shows tool_use only; claude itself infers
//!   the result. Matches codex, which emits no explicit tool_result frame).
//! - `result` status=success → flush pending assistant text first, then
//!   ContextTokens (see `parse_result_context_tokens` — the gemini `stats` are
//!   SUMMED across every model round-trip in the turn, so we divide
//!   `input_tokens` by an estimated call count instead of reporting the raw
//!   `total_tokens`, which over-reports context by ~the number of round-trips)
//!   + Frame (claude-shape success result envelope) + Result.
//! - `result` status=error → flush pending assistant text first, then surface
//!   the error text as its OWN claude-shape assistant message bubble, THEN the
//!   Frame (claude-shape error result envelope) + Result. The extra text bubble
//!   is the whole point: iOS's `SpawnerMessageDescriber` renders a result frame
//!   as only `[result: error]` and never reads `error.message`, so without it a
//!   failed turn (e.g. a bad `-m <model>`) shows a bare terminator with no
//!   reason. Emitting the message as a normal assistant frame matches how
//!   claude's own plain-text error lines reach the chat. (Gemini's structured
//!   `error.message` is often generic — `[API Error: ...]` — because the rich
//!   diagnostic, e.g. `ModelNotFoundError`, lands on STDERR, which the
//!   Supervisor drains separately and `handle_line` never sees; we surface the
//!   best text the JSON frame carries.)
//! - anything else → forwarded as-is (defensive against gemini format drift;
//!   iOS will likely drop, but we avoid silently losing the line).
//!
//! Retries / throttling / error stacks arrive on STDERR (not stdout JSON) and
//! are drained by the Supervisor — `handle_line` only ever sees stdout JSON
//! lines; we never try to parse stderr here.
//!
//! Also hosts the install/auth `probe()` for gemini (free function, not on the
//! `AgentAdapter` trait — `dyn AgentAdapter` can't dispatch statics). For
//! "authenticated" we stat `~/.gemini/oauth_creds.json` (Google OAuth login
//! writes its token blob there) OR check `GEMINI_API_KEY` — same pragmatic
//! presence + non-empty check codex uses for `~/.codex/auth.json`.
//!
//! `import()` is a stub for v1, like codex/hermes.

use std::path::PathBuf;

use anyhow::Result;
use serde::Deserialize;
use serde_json::{json, Value};
use smallvec::SmallVec;
use tokio::sync::mpsc;
use tracing::{debug, info};
use uuid::Uuid;

use crate::adapter::{
    claude_assistant_text_envelope, claude_tool_use_envelope, file_nonempty, parse_json_obj,
    probe_with_blocking_auth, shell_escape, AdapterDescriptor, AgentAdapter, AgentEvent, AgentKind,
    ImportProgress, LastTokensDedup, TurnContext, MAX_STREAM_FRAME_BYTES,
};
use crate::writer::WriteEvent;

/// Wired into `adapter::ADAPTERS`. See `adapter::AdapterDescriptor` for the
/// shape; the `probe` / `import` slots are filled by `_boxed` wrappers below
/// the `probe()` / `import()` definitions in this file.
///
/// `installed_col` / `authenticated_col` follow the same per-kind boolean pair
/// as `codex_*` (migration 0037) and `hermes_*` (0038); the matching
/// `gemini_*` columns land in migration 0039 (implemented in parallel and
/// deployed first, so `backend_has_install_columns` returns `true` for Gemini).
pub const DESCRIPTOR: AdapterDescriptor = AdapterDescriptor {
    kind: AgentKind::Gemini,
    wire_name: "gemini",
    installed_col: "gemini_installed",
    authenticated_col: "gemini_authenticated",
    make: make_boxed,
    probe: probe_boxed,
    import: import_boxed,
};

fn make_boxed() -> Box<dyn AgentAdapter> {
    Box::new(GeminiAdapter::new())
}

/// asdf-managed `gemini` shim no-ops with "No version is set for command
/// gemini" unless this env var is set right before the binary. Pinned to the
/// node version the spawner host has installed for gemini-cli 0.44.1.
const ASDF_NODE_ENV: &str = "ASDF_NODEJS_VERSION=24.14.0 ";

/// Per-turn state for the gemini adapter. Carries:
///   - `session_id`: a fresh UUID minted per session, passed via
///     `--session-id` on the first turn (gemini echoes it back in the `init`
///     frame; we still harvest it so the persisted `chats.agent_session_id` is
///     exactly what gemini knows about for the resume path).
///   - `last_emitted_tokens`: dedup so a re-emitted `result` frame doesn't
///     double-fire ContextTokens on an identical value. Mirrors codex's
///     per-turn dedup field (`adapter::LastTokensDedup`).
///   - `pending_assistant_text`: accumulates the incremental assistant text
///     deltas for the current text run; flushed as ONE Frame at the next
///     tool_use / result boundary to honor the one-frame-per-row invariant
///     (otherwise the reply fragments into one bubble per delta chunk). Per-turn
///     state is correct here — the adapter is constructed fresh per turn (same
///     as `last_emitted_tokens`).
pub struct GeminiAdapter {
    session_id: Uuid,
    last_emitted_tokens: LastTokensDedup,
    pending_assistant_text: String,
}

impl Default for GeminiAdapter {
    fn default() -> Self {
        Self {
            session_id: Uuid::now_v7(),
            last_emitted_tokens: LastTokensDedup::default(),
            pending_assistant_text: String::new(),
        }
    }
}

impl GeminiAdapter {
    pub fn new() -> Self {
        Self::default()
    }

    /// If assistant text deltas have accumulated in `pending_assistant_text`,
    /// build ONE claude-shape assistant text envelope from the whole buffer,
    /// clear the buffer, and return it. Returns `None` when there's nothing
    /// buffered. Called at every boundary (tool_use, result, init) so a turn's
    /// text run is emitted as a single immutable `messages` row — coalescing the
    /// per-chunk deltas instead of fragmenting them into one bubble each.
    fn flush_pending_text(&mut self) -> Option<AgentEvent> {
        if self.pending_assistant_text.is_empty() {
            return None;
        }
        let frame = claude_assistant_text_envelope(&self.pending_assistant_text);
        self.pending_assistant_text.clear();
        Some(AgentEvent::Frame(frame))
    }
}

impl AgentAdapter for GeminiAdapter {
    fn kind(&self) -> AgentKind {
        AgentKind::Gemini
    }

    fn prepare_command(&mut self, ctx: &TurnContext<'_>) -> Result<String> {
        let mut cmd = String::new();
        if let Some(pp) = ctx.project_path {
            cmd.push_str(&format!("cd {} && ", shell_escape(pp)));
        }

        // Prompt is piped via stdin (prompts can be multi-MB with attachments;
        // never inline via `-p` argv). Piped non-TTY stdin triggers gemini's
        // non-interactive/headless mode — verified empirically; no `-p` needed.
        // The `ASDF_NODEJS_VERSION` env prefix must sit right before the binary
        // (after the pipe) so the asdf shim resolves the right node.
        cmd.push_str(&format!(
            "cat {} | {}gemini",
            shell_escape(&ctx.prompt_file.to_string_lossy()),
            ASDF_NODE_ENV,
        ));

        // Session id. FIRST turn: pass our freshly-minted UUID via
        // `--session-id` (gemini adopts it and echoes it in the `init` frame).
        // RESUME: `--resume <sid>` with the harvested id genuinely continues
        // context — verified. The two flags are mutually exclusive.
        if let Some(sid) = ctx.agent_session_id {
            cmd.push_str(&format!(" --resume {}", shell_escape(sid)));
        } else {
            cmd.push_str(&format!(
                " --session-id {}",
                shell_escape(&self.session_id.to_string())
            ));
        }

        cmd.push_str(" -o stream-json");

        // Sender's `machine_users.is_sandboxed`. Non-sandboxed = bypass all
        // approval + trust prompts (`--approval-mode yolo --skip-trust`),
        // mirroring claude's `--dangerously-skip-permissions`. Sandboxed =
        // OMIT both: gemini then relies on the user's own ~/.gemini
        // settings/policy — that is intentional and the user's responsibility
        // (thin-spawn-layer scope). NOTE: `--approval-mode plan` is NOT an
        // enforced read-only boundary (the model self-writes a plan then calls
        // exit_plan_mode and actually writes files — verified), so we never use
        // it as a sandbox.
        if !ctx.is_sandboxed {
            cmd.push_str(" --approval-mode yolo --skip-trust");
        }

        // Verbatim pass-through of `chats.model` (migration 0035). Gemini uses
        // `-m, --model`; the model label drifts per-release so we don't
        // validate it locally — an invalid value surfaces as a gemini error
        // frame in the chat.
        if let Some(model) = ctx.model {
            cmd.push_str(&format!(" -m {}", shell_escape(model)));
        }

        // TODO(gemini): worktree=true is ignored for v1, same as codex.
        let _ = ctx.worktree;

        Ok(cmd)
    }

    fn handle_line(&mut self, line: String) -> SmallVec<[AgentEvent; 2]> {
        let mut out: SmallVec<[AgentEvent; 2]> = SmallVec::new();

        // Oversize-frame guard. Forward verbatim above the cap so a single big
        // tool_result/message line doesn't churn the heap on a full
        // `serde_json::Value` parse. Mirrors codex/claude `handle_line`.
        if line.len() > MAX_STREAM_FRAME_BYTES {
            out.push(AgentEvent::Frame(line));
            return out;
        }

        let Some(obj) = parse_json_obj(&line) else {
            // Non-JSON line: forward as-is (matches codex's permissive path).
            out.push(AgentEvent::Frame(line));
            return out;
        };
        let Some(ty) = obj.get("type").and_then(|v| v.as_str()) else {
            out.push(AgentEvent::Frame(line));
            return out;
        };

        match ty {
            "init" => {
                // Defensive flush — init shouldn't carry pending text mid-turn,
                // but flushing here is harmless and keeps ordering safe.
                if let Some(ev) = self.flush_pending_text() {
                    out.push(ev);
                }
                // Harvest session_id → `chats.agent_session_id`; drop the frame
                // (matches claude's init-skip). It echoes our minted uuid;
                // harvest regardless so resume is exact.
                if let Some(sid) = obj.get("session_id").and_then(|v| v.as_str()) {
                    out.push(AgentEvent::SessionIdHarvested(sid.to_string()));
                } else {
                    debug!("gemini init frame without session_id");
                }
            }
            "message" => {
                let role = obj.get("role").and_then(|v| v.as_str()).unwrap_or("");
                match role {
                    // Echo of our own prompt — drop.
                    "user" => debug!("gemini user message dropped (prompt echo)"),
                    "assistant" => {
                        // Assistant text arrives as multiple INCREMENTAL delta
                        // chunks per turn (verified non-cumulative against
                        // tmp/gemini-probe captures). APPEND into the per-turn
                        // buffer; do NOT emit yet. Flushed as ONE Frame at the
                        // next boundary so the reply lands in a single immutable
                        // `messages` row instead of one bubble per chunk.
                        let content = obj.get("content").and_then(|v| v.as_str()).unwrap_or("");
                        self.pending_assistant_text.push_str(content);
                    }
                    other => debug!("gemini message with unknown role {:?}, dropping", other),
                }
            }
            "tool_use" => {
                if let Some(frame) = normalize_tool_use(&obj) {
                    // Real tool → flush the text run that preceded it FIRST so
                    // text and tool become separate, correctly-ordered bubbles.
                    if let Some(ev) = self.flush_pending_text() {
                        out.push(ev);
                    }
                    out.push(AgentEvent::Frame(frame));
                }
                // None = filtered meta-tool (update_topic / exit_plan_mode) —
                // dropped silently inside normalize_tool_use. We deliberately do
                // NOT flush in that case: a meta-tool arriving mid-text must not
                // split the surrounding text into two bubbles.
            }
            "tool_result" => {
                // Claude UI shows tool_use only and infers the result — match
                // codex and drop.
                debug!("gemini tool_result dropped (claude infers results)");
            }
            "result" => {
                // Flush the trailing text run before the turn terminator so the
                // final assistant text lands as its own row, ahead of Result.
                if let Some(ev) = self.flush_pending_text() {
                    out.push(ev);
                }
                let status = obj.get("status").and_then(|v| v.as_str()).unwrap_or("");
                if status == "error" {
                    // Surface the error text as its OWN assistant bubble first
                    // so the user sees WHY the turn failed — iOS renders the
                    // result frame below as only `[result: error]` and drops
                    // `error.message`. Same posture as claude's plain-text
                    // errors arriving as a message.
                    let message = result_error_message(&obj).to_string();
                    out.push(AgentEvent::Frame(claude_assistant_text_envelope(&message)));
                    out.push(AgentEvent::Frame(normalize_result_error_frame(&obj)));
                    out.push(AgentEvent::Result);
                } else {
                    // success (or anything non-error): harvest tokens, emit a
                    // claude-shape success result envelope, then Result.
                    if let Some(tokens) = parse_result_context_tokens(&obj) {
                        if let Some(t) = self.last_emitted_tokens.observe(tokens) {
                            out.push(AgentEvent::ContextTokens(t));
                        }
                    }
                    out.push(AgentEvent::Frame(normalize_result_success_frame()));
                    out.push(AgentEvent::Result);
                }
            }
            other => {
                // Defensive forward — gemini format drift shouldn't silently
                // drop content. Matches codex's unknown-passthrough.
                debug!("gemini unknown frame type, forwarding: {}", other);
                out.push(AgentEvent::Frame(line));
            }
        }

        out
    }
}

/// Maps a gemini `tool_use` frame to a claude-shape tool_use envelope. Returns
/// `None` for the gemini meta-tools `update_topic` / `exit_plan_mode` (dropped
/// entirely — they have no claude analog and would render as noise). `tool_id`
/// is used as the claude `tool_use.id` so iOS can key its row diffing off it.
///
/// iOS's `SpawnerMessage.toolSummary` only renders a one-line detail for a
/// fixed set of claude tool names keyed on specific input fields:
/// `Bash{command}`, `Read`/`Write`/`Edit`{file_path}, `Grep`/`Glob`{pattern},
/// `WebSearch{query}`, `Agent{description}`. Gemini emits its OWN snake_case
/// tool names, so every gemini tool we want detail for MUST be remapped to one
/// of those claude names AND have its summary argument copied under the claude
/// input key iOS reads — otherwise the chat shows a bare tool name with no
/// command/path/pattern. (Same posture as hermes's `_TOOL_NAME_MAP` and codex's
/// `normalize_item_completed`.) iOS has no `WebFetch` case, so `web_fetch` is
/// folded into `WebSearch` using its `prompt` as the query.
///
/// Mapping (gemini `tool_name` → claude tool name + iOS-rendered input key;
/// gemini param field names confirmed empirically on gemini-cli 0.44.1
/// `-o stream-json`):
///   read_file         → `Read`   `{file_path}`
///   write_file        → `Write`  `{file_path, content}`
///   replace           → `Edit`   `{file_path}`           (in-place edit)
///   run_shell_command → `Bash`   `{command}`             (THE shell tool)
///   list_directory    → `Bash`   `{command: "ls <dir_path>"}` (no claude
///                        dir-list tool; `ls` is what claude itself emits)
///   grep_search       → `Grep`   `{pattern}`
///   glob              → `Glob`   `{pattern}`
///   read_many_files   → `Read`   `{file_path: <include[0]>}` (carries an
///                        `include` array; show the first glob/path)
///   google_web_search → `WebSearch` `{query}`
///   web_fetch         → `WebSearch` `{query: <prompt>}`   (iOS has no WebFetch)
///   update_topic / exit_plan_mode → filtered (None)
///   <unknown>         → forwarded under its native name (defensive)
fn normalize_tool_use(obj: &Value) -> Option<String> {
    let tool_name = obj.get("tool_name").and_then(|v| v.as_str()).unwrap_or("");
    // Gemini-injected meta-tools — filter out entirely.
    if matches!(tool_name, "update_topic" | "exit_plan_mode") {
        debug!("gemini meta-tool {:?} filtered out", tool_name);
        return None;
    }
    let id = obj.get("tool_id").and_then(|v| v.as_str()).unwrap_or("");
    let params = obj.get("parameters");
    let param_str = |key: &str| -> String {
        params
            .and_then(|p| p.get(key))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string()
    };
    // First element of an array-valued param (e.g. read_many_files.include),
    // as a string. Empty string if absent / not an array of strings.
    let param_first = |key: &str| -> String {
        params
            .and_then(|p| p.get(key))
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.first())
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string()
    };

    Some(match tool_name {
        "read_file" => {
            claude_tool_use_envelope(id, "Read", json!({ "file_path": param_str("file_path") }))
        }
        "write_file" => claude_tool_use_envelope(
            id,
            "Write",
            json!({ "file_path": param_str("file_path"), "content": param_str("content") }),
        ),
        // In-place edit. Gemini carries file_path/old_string/new_string; iOS's
        // Edit summary only reads file_path.
        "replace" => {
            claude_tool_use_envelope(id, "Edit", json!({ "file_path": param_str("file_path") }))
        }
        // Shell command — THE reported bug. Gemini's `command` param maps onto
        // claude Bash's `command` so iOS renders the actual command line.
        "run_shell_command" => {
            claude_tool_use_envelope(id, "Bash", json!({ "command": param_str("command") }))
        }
        "list_directory" => claude_tool_use_envelope(
            id,
            "Bash",
            json!({ "command": format!("ls {}", param_str("dir_path")) }),
        ),
        // Codebase text search. Gemini's tool_name is `grep_search` (not the
        // older `search_file_content`); `pattern` matches claude Grep's key.
        "grep_search" => {
            claude_tool_use_envelope(id, "Grep", json!({ "pattern": param_str("pattern") }))
        }
        "glob" => claude_tool_use_envelope(id, "Glob", json!({ "pattern": param_str("pattern") })),
        // Bulk read by glob/path list. Show the first include entry as the
        // file_path so iOS's Read summary renders something meaningful.
        "read_many_files" => {
            claude_tool_use_envelope(id, "Read", json!({ "file_path": param_first("include") }))
        }
        "google_web_search" => {
            claude_tool_use_envelope(id, "WebSearch", json!({ "query": param_str("query") }))
        }
        // No WebFetch case in iOS — fold into WebSearch using the prompt text.
        "web_fetch" => {
            claude_tool_use_envelope(id, "WebSearch", json!({ "query": param_str("prompt") }))
        }
        // Unknown gemini tool — forward under its native name with whatever
        // parameters it carried (defensive against gemini adding tools).
        _ => {
            let input = params.cloned().unwrap_or_else(|| json!({}));
            claude_tool_use_envelope(id, tool_name, input)
        }
    })
}

/// Estimates the context-window occupancy from a `result` frame's `stats`.
///
/// THE FIX for the context over-count: gemini's `stats` are SUMMED across every
/// model round-trip in the turn (each tool call re-sends the growing context),
/// so `total_tokens` / `input_tokens` report ~`#round-trips × context_size`,
/// not how full the context window actually is. Empirically (gemini-cli 0.44.1,
/// captures in `tmp/gemini-probe/*.jsonl`): two trivial questions in one resumed
/// session reached `total_tokens` 82108 / `input_tokens` 80528 with
/// `tool_calls` 5, while the live context was only ~tens of thousands — an easy
/// many-tool turn can read 480k. Reporting the raw value made the context gauge
/// meaningless.
///
/// We approximate the per-call prompt size (≈ the context resident in the
/// window) by dividing the summed `input_tokens` by an estimated model-call
/// count, `tool_calls + 1` (each tool result triggers a fresh round-trip, plus
/// the final answer call). This is the same posture as `cursor.rs`, which
/// divides its summed `input + cacheRead + cacheWrite` by `n_calls`. We use
/// `input_tokens` (the prompt = non-cached `input` + `cached`), NOT
/// `total_tokens`, because context occupancy is the prompt size and excludes
/// generated output — matching codex's `ContextTokens(input_tokens)`.
///
/// The estimate slightly under-counts (divides by a bit too much) when one model
/// response emits PARALLEL tool calls, since those add `tool_calls` without an
/// extra round-trip; that's the same imprecision cursor accepts and is far
/// better than the linear blow-up.
///
/// Returns `None` (skip the ContextTokens emission entirely) when the frame
/// carries no usable `input_tokens`, so a malformed / usage-less result never
/// zeroes the live counter. Narrow Deserialize struct so serde skips the rest.
fn parse_result_context_tokens(obj: &Value) -> Option<i64> {
    #[derive(Deserialize)]
    struct Frame {
        #[serde(default)]
        stats: Option<Stats>,
    }
    #[derive(Deserialize, Default)]
    struct Stats {
        #[serde(default)]
        input_tokens: i64,
        #[serde(default)]
        tool_calls: i64,
    }
    let stats = match serde_json::from_value::<Frame>(obj.clone()) {
        Ok(f) => f.stats?,
        Err(e) => {
            debug!("failed to parse gemini result stats: {}", e);
            return None;
        }
    };
    if stats.input_tokens <= 0 {
        return None;
    }
    // `tool_calls` is only negative on malformed input; clamp so the divisor is
    // always >= 1 (a no-tool turn is a single model call → divide by 1).
    let n_calls = stats.tool_calls.max(0) + 1;
    Some(stats.input_tokens / n_calls)
}

/// Builds the claude-shape result envelope emitted on a successful `result`.
/// iOS's describer renders this as `[result: success]`. Matches codex's
/// `normalize_turn_completed_frame`.
fn normalize_result_success_frame() -> String {
    json!({
        "type": "result",
        "subtype": "success",
        "is_error": false,
    })
    .to_string()
}

/// Pulls the human-readable error string out of a gemini `result` error frame
/// (`error.message`), falling back to a generic label when the frame omits it.
/// Shared by `normalize_result_error_frame` (stored in the result envelope) and
/// the `handle_line` error branch (emitted as a visible assistant bubble) so
/// both render the exact same text.
fn result_error_message(obj: &Value) -> &str {
    obj.get("error")
        .and_then(|e| e.get("message"))
        .and_then(|v| v.as_str())
        .unwrap_or("turn failed")
}

/// Builds the claude-shape error result envelope emitted on a failed `result`.
/// Gemini's error shape is `{"type":"result","status":"error",
/// "error":{"type":"...","message":"..."}}`. iOS uses `subtype` for the visible
/// terminator (`[result: error]`); preserving `error.message` keeps the stored
/// frame useful for logs. Matches codex's `normalize_turn_failed_frame`.
fn normalize_result_error_frame(obj: &Value) -> String {
    let message = result_error_message(obj);
    json!({
        "type": "result",
        "subtype": "error",
        "is_error": true,
        "error": {
            "message": message,
        },
    })
    .to_string()
}

/// Probe install + auth state in one go. Returns `(installed, authenticated)`.
/// The shared `adapter::probe_with_blocking_auth` helper handles the
/// `binary_on_path` + `spawn_blocking` boilerplate.
pub async fn probe() -> (bool, bool) {
    probe_with_blocking_auth("gemini", is_authenticated).await
}

/// `fn`-pointer-shaped wrapper around `probe()` for `AdapterDescriptor.probe`.
fn probe_boxed() -> futures::future::BoxFuture<'static, (bool, bool)> {
    Box::pin(probe())
}

/// Gemini stores Google-OAuth state in `~/.gemini/oauth_creds.json`. We check
/// for presence + non-empty (same pragmatic check codex uses for
/// `~/.codex/auth.json`), OR a non-empty `GEMINI_API_KEY` env var (API-key
/// auth path). Either satisfies "authenticated".
fn is_authenticated() -> bool {
    if std::env::var("GEMINI_API_KEY")
        .map(|v| !v.is_empty())
        .unwrap_or(false)
    {
        return true;
    }
    let Some(home) = std::env::var_os("HOME").map(PathBuf::from) else {
        return false;
    };
    let creds = home.join(".gemini").join("oauth_creds.json");
    file_nonempty(&creds)
}

/// One-shot per-kind history importer. Gemini stores session transcripts under
/// `~/.gemini/` but the on-disk format isn't yet wired up to PutChat/PutMessage
/// — stub for v1, like codex/hermes: log + report 100% so the dispatcher's
/// per-kind progress slice closes cleanly.
pub(crate) async fn import(
    _machine_id: Uuid,
    _user_id: Uuid,
    _write_tx: mpsc::Sender<WriteEvent>,
    progress: ImportProgress,
) -> Result<()> {
    info!("gemini history import not yet implemented, skipping");
    progress(100);
    Ok(())
}

/// `fn`-pointer-shaped wrapper around `import()` for `AdapterDescriptor.import`.
fn import_boxed(
    machine_id: Uuid,
    user_id: Uuid,
    write_tx: mpsc::Sender<WriteEvent>,
    progress: ImportProgress,
) -> futures::future::BoxFuture<'static, Result<()>> {
    Box::pin(import(machine_id, user_id, write_tx, progress))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Maps a turn's emitted events to human-readable tags + payload for easy
    /// assertion equality. Delegates to the shared stringifier in `adapter.rs`.
    fn run(adapter: &mut GeminiAdapter, line: &str) -> Vec<String> {
        crate::adapter::stringify_events(adapter.handle_line(line.to_string()))
    }

    /// Strips the `Frame(...)` test wrapper and parses the inner JSON.
    fn frame_value(event: &str) -> Value {
        crate::adapter::frame_value(event)
    }

    fn ctx<'a>(
        prompt_file: &'a std::path::Path,
        agent_session_id: Option<&'a str>,
        is_sandboxed: bool,
        model: Option<&'a str>,
    ) -> TurnContext<'a> {
        TurnContext {
            chat_id: "00000000-0000-0000-0000-000000000000",
            prompt_file,
            project_path: Some("/tmp/proj"),
            worktree: false,
            agent_session_id,
            is_sandboxed,
            model,
        }
    }

    #[test]
    fn init_frame_harvests_session_id_and_drops_frame() {
        let mut a = GeminiAdapter::new();
        let line = r#"{"type":"init","session_id":"11111111-2222-3333-4444-555555555555"}"#;
        let events = run(&mut a, line);
        assert_eq!(
            events,
            vec!["SessionIdHarvested(11111111-2222-3333-4444-555555555555)"]
        );
    }

    #[test]
    fn user_message_dropped() {
        let mut a = GeminiAdapter::new();
        let line = r#"{"type":"message","role":"user","content":"hello"}"#;
        assert!(run(&mut a, line).is_empty());
    }

    #[test]
    fn single_assistant_delta_is_buffered_not_emitted() {
        // A lone assistant delta emits NOTHING — it's buffered until a boundary.
        let mut a = GeminiAdapter::new();
        let line = r#"{"type":"message","role":"assistant","content":"Hello ","delta":true}"#;
        assert!(run(&mut a, line).is_empty());
        // It's still pending in the buffer.
        assert_eq!(a.pending_assistant_text, "Hello ");
    }

    #[test]
    fn multi_delta_then_result_emits_one_coalesced_text_frame() {
        // Two consecutive deltas buffer (zero frames each), then the result
        // boundary flushes ONE assistant text frame with the concatenation, in
        // order, ahead of the result/Result markers. This is the fragmentation
        // fix: N delta chunks → exactly ONE assistant `messages` row.
        let mut a = GeminiAdapter::new();
        let c1 = r#"{"type":"message","role":"assistant","content":"Hello ","delta":true}"#;
        let c2 = r#"{"type":"message","role":"assistant","content":"world","delta":true}"#;
        assert!(run(&mut a, c1).is_empty());
        assert!(run(&mut a, c2).is_empty());

        // input_tokens 30 over (tool_calls 2 + 1) model calls → ContextTokens(10).
        let res =
            r#"{"type":"result","status":"success","stats":{"input_tokens":30,"tool_calls":2}}"#;
        let events = run(&mut a, res);
        // flush(text) + ContextTokens + result Frame + Result = 4 events.
        assert_eq!(events.len(), 4);
        let text = frame_value(&events[0]);
        assert_eq!(text["type"], "assistant");
        assert_eq!(text["message"]["content"][0]["type"], "text");
        assert_eq!(text["message"]["content"][0]["text"], "Hello world");
        assert_eq!(events[1], "ContextTokens(10)");
        assert_eq!(frame_value(&events[2])["subtype"], "success");
        assert_eq!(events[3], "Result");
    }

    #[test]
    fn text_tool_text_result_splits_at_tool_boundary_only() {
        // text chunk → tool_use → text chunk → result must yield two SEPARATE
        // coherent text bubbles split exactly at the tool boundary, with the
        // tool bubble between them and the result terminator last.
        let mut a = GeminiAdapter::new();

        let t1 = r#"{"type":"message","role":"assistant","content":"Let me check.","delta":true}"#;
        assert!(run(&mut a, t1).is_empty());

        let tool = r#"{"type":"tool_use","tool_id":"sh1","tool_name":"run_shell_command","parameters":{"command":"ls"}}"#;
        let tool_events = run(&mut a, tool);
        // flush(text "Let me check.") + tool_use Frame = 2 events, in order.
        assert_eq!(tool_events.len(), 2);
        assert_eq!(
            frame_value(&tool_events[0])["message"]["content"][0]["text"],
            "Let me check."
        );
        assert_eq!(
            frame_value(&tool_events[1])["message"]["content"][0]["name"],
            "Bash"
        );

        let t2 = r#"{"type":"message","role":"assistant","content":"Found it.","delta":true}"#;
        assert!(run(&mut a, t2).is_empty());

        // input_tokens 15 over (tool_calls 2 + 1) model calls → ContextTokens(5).
        let res =
            r#"{"type":"result","status":"success","stats":{"input_tokens":15,"tool_calls":2}}"#;
        let res_events = run(&mut a, res);
        // flush(text "Found it.") + ContextTokens + result Frame + Result.
        assert_eq!(res_events.len(), 4);
        assert_eq!(
            frame_value(&res_events[0])["message"]["content"][0]["text"],
            "Found it."
        );
        assert_eq!(res_events[1], "ContextTokens(5)");
        assert_eq!(res_events[3], "Result");
    }

    #[test]
    fn meta_tool_mid_text_does_not_split_the_bubble() {
        // A meta-tool (update_topic) between two text deltas produces no Frame
        // and must NOT flush, so the surrounding text coalesces into ONE bubble.
        let mut a = GeminiAdapter::new();

        let t1 = r#"{"type":"message","role":"assistant","content":"Part one ","delta":true}"#;
        assert!(run(&mut a, t1).is_empty());

        let meta = r#"{"type":"tool_use","tool_id":"u1","tool_name":"update_topic","parameters":{"summary":"x"}}"#;
        assert!(run(&mut a, meta).is_empty());
        // Buffer untouched by the meta-tool.
        assert_eq!(a.pending_assistant_text, "Part one ");

        let t2 = r#"{"type":"message","role":"assistant","content":"part two.","delta":true}"#;
        assert!(run(&mut a, t2).is_empty());

        let res =
            r#"{"type":"result","status":"success","stats":{"input_tokens":9,"tool_calls":2}}"#;
        let events = run(&mut a, res);
        assert_eq!(events.len(), 4);
        assert_eq!(
            frame_value(&events[0])["message"]["content"][0]["text"],
            "Part one part two."
        );
    }

    #[test]
    fn tool_use_read_file_maps_to_claude_read() {
        let mut a = GeminiAdapter::new();
        let line = r#"{"type":"tool_use","tool_id":"t1","tool_name":"read_file","parameters":{"file_path":"src/foo.rs","end_line":40}}"#;
        let events = run(&mut a, line);
        assert_eq!(events.len(), 1);
        let v = frame_value(&events[0]);
        assert_eq!(v["message"]["content"][0]["name"], "Read");
        assert_eq!(v["message"]["content"][0]["id"], "t1");
        assert_eq!(
            v["message"]["content"][0]["input"]["file_path"],
            "src/foo.rs"
        );
    }

    #[test]
    fn tool_use_write_file_maps_to_claude_write() {
        let mut a = GeminiAdapter::new();
        let line = r#"{"type":"tool_use","tool_id":"t2","tool_name":"write_file","parameters":{"file_path":"out.txt","content":"hi"}}"#;
        let events = run(&mut a, line);
        assert_eq!(events.len(), 1);
        let v = frame_value(&events[0]);
        assert_eq!(v["message"]["content"][0]["name"], "Write");
        assert_eq!(v["message"]["content"][0]["input"]["file_path"], "out.txt");
        assert_eq!(v["message"]["content"][0]["input"]["content"], "hi");
    }

    #[test]
    fn tool_use_list_directory_maps_to_claude_bash_ls() {
        let mut a = GeminiAdapter::new();
        let line = r#"{"type":"tool_use","tool_id":"t3","tool_name":"list_directory","parameters":{"dir_path":"/tmp/proj"}}"#;
        let events = run(&mut a, line);
        assert_eq!(events.len(), 1);
        let v = frame_value(&events[0]);
        assert_eq!(v["message"]["content"][0]["name"], "Bash");
        assert_eq!(
            v["message"]["content"][0]["input"]["command"],
            "ls /tmp/proj"
        );
    }

    #[test]
    fn tool_use_run_shell_command_maps_to_claude_bash() {
        // THE reported bug: shell command must carry its command line under
        // claude Bash's `command` key so iOS's toolSummary renders it.
        let mut a = GeminiAdapter::new();
        let line = r#"{"type":"tool_use","tool_id":"sh1","tool_name":"run_shell_command","parameters":{"command":"echo hello","description":"print hello"}}"#;
        let events = run(&mut a, line);
        assert_eq!(events.len(), 1);
        let v = frame_value(&events[0]);
        assert_eq!(v["message"]["content"][0]["name"], "Bash");
        assert_eq!(v["message"]["content"][0]["id"], "sh1");
        assert_eq!(v["message"]["content"][0]["input"]["command"], "echo hello");
    }

    #[test]
    fn tool_use_replace_maps_to_claude_edit() {
        let mut a = GeminiAdapter::new();
        let line = r#"{"type":"tool_use","tool_id":"r1","tool_name":"replace","parameters":{"file_path":"editme.txt","old_string":"banana","new_string":"apple","instruction":"swap"}}"#;
        let events = run(&mut a, line);
        assert_eq!(events.len(), 1);
        let v = frame_value(&events[0]);
        assert_eq!(v["message"]["content"][0]["name"], "Edit");
        assert_eq!(v["message"]["content"][0]["id"], "r1");
        assert_eq!(
            v["message"]["content"][0]["input"]["file_path"],
            "editme.txt"
        );
    }

    #[test]
    fn tool_use_grep_search_maps_to_claude_grep() {
        let mut a = GeminiAdapter::new();
        let line = r#"{"type":"tool_use","tool_id":"g1","tool_name":"grep_search","parameters":{"pattern":"banana"}}"#;
        let events = run(&mut a, line);
        assert_eq!(events.len(), 1);
        let v = frame_value(&events[0]);
        assert_eq!(v["message"]["content"][0]["name"], "Grep");
        assert_eq!(v["message"]["content"][0]["input"]["pattern"], "banana");
    }

    #[test]
    fn tool_use_glob_maps_to_claude_glob() {
        let mut a = GeminiAdapter::new();
        let line = r#"{"type":"tool_use","tool_id":"gl1","tool_name":"glob","parameters":{"pattern":"*.txt"}}"#;
        let events = run(&mut a, line);
        assert_eq!(events.len(), 1);
        let v = frame_value(&events[0]);
        assert_eq!(v["message"]["content"][0]["name"], "Glob");
        assert_eq!(v["message"]["content"][0]["input"]["pattern"], "*.txt");
    }

    #[test]
    fn tool_use_read_many_files_maps_to_claude_read_first_include() {
        let mut a = GeminiAdapter::new();
        let line = r#"{"type":"tool_use","tool_id":"rm1","tool_name":"read_many_files","parameters":{"include":["src/**/*.rs","README.md"]}}"#;
        let events = run(&mut a, line);
        assert_eq!(events.len(), 1);
        let v = frame_value(&events[0]);
        assert_eq!(v["message"]["content"][0]["name"], "Read");
        assert_eq!(
            v["message"]["content"][0]["input"]["file_path"],
            "src/**/*.rs"
        );
    }

    #[test]
    fn tool_use_google_web_search_maps_to_claude_websearch() {
        let mut a = GeminiAdapter::new();
        let line = r#"{"type":"tool_use","tool_id":"w1","tool_name":"google_web_search","parameters":{"query":"rust async"}}"#;
        let events = run(&mut a, line);
        assert_eq!(events.len(), 1);
        let v = frame_value(&events[0]);
        assert_eq!(v["message"]["content"][0]["name"], "WebSearch");
        assert_eq!(v["message"]["content"][0]["input"]["query"], "rust async");
    }

    #[test]
    fn tool_use_web_fetch_folds_into_websearch_with_prompt() {
        let mut a = GeminiAdapter::new();
        let line = r#"{"type":"tool_use","tool_id":"wf1","tool_name":"web_fetch","parameters":{"prompt":"summarize https://example.com"}}"#;
        let events = run(&mut a, line);
        assert_eq!(events.len(), 1);
        let v = frame_value(&events[0]);
        assert_eq!(v["message"]["content"][0]["name"], "WebSearch");
        assert_eq!(
            v["message"]["content"][0]["input"]["query"],
            "summarize https://example.com"
        );
    }

    #[test]
    fn tool_use_meta_tools_filtered_out() {
        let mut a = GeminiAdapter::new();
        let ut = r#"{"type":"tool_use","tool_id":"t4","tool_name":"update_topic","parameters":{"topic":"x"}}"#;
        let ep =
            r#"{"type":"tool_use","tool_id":"t5","tool_name":"exit_plan_mode","parameters":{}}"#;
        assert!(run(&mut a, ut).is_empty());
        assert!(run(&mut a, ep).is_empty());
    }

    #[test]
    fn tool_use_unknown_forwarded_under_native_name() {
        let mut a = GeminiAdapter::new();
        let line = r#"{"type":"tool_use","tool_id":"t6","tool_name":"some_future_tool","parameters":{"k":"v"}}"#;
        let events = run(&mut a, line);
        assert_eq!(events.len(), 1);
        let v = frame_value(&events[0]);
        assert_eq!(v["message"]["content"][0]["name"], "some_future_tool");
        assert_eq!(v["message"]["content"][0]["input"]["k"], "v");
    }

    #[test]
    fn tool_result_dropped() {
        let mut a = GeminiAdapter::new();
        let line = r#"{"type":"tool_result","tool_id":"t1","output":"some output"}"#;
        assert!(run(&mut a, line).is_empty());
    }

    #[test]
    fn result_success_emits_tokens_frame_and_result_marker() {
        let mut a = GeminiAdapter::new();
        // ContextTokens uses input_tokens / (tool_calls + 1), NOT the summed
        // total_tokens: 1200 / (2 + 1) = 400. output_tokens is ignored (context
        // occupancy is the prompt size, not generated tokens).
        let line = r#"{"type":"result","status":"success","stats":{"total_tokens":1434,"input_tokens":1200,"output_tokens":234,"tool_calls":2}}"#;
        let events = run(&mut a, line);
        assert_eq!(events.len(), 3);
        assert_eq!(events[0], "ContextTokens(400)");
        assert!(events[1].starts_with("Frame("));
        assert_eq!(events[2], "Result");
        let v = frame_value(&events[1]);
        assert_eq!(v["type"], "result");
        assert_eq!(v["subtype"], "success");
        assert_eq!(v["is_error"], false);
    }

    #[test]
    fn result_no_tool_turn_reports_input_tokens_unchanged() {
        // A turn with no tool calls is a single model round-trip, so the divisor
        // is 1 and input_tokens passes through as-is.
        let mut a = GeminiAdapter::new();
        let line = r#"{"type":"result","status":"success","stats":{"total_tokens":520,"input_tokens":500,"output_tokens":20,"tool_calls":0}}"#;
        let events = run(&mut a, line);
        assert_eq!(events[0], "ContextTokens(500)");
    }

    #[test]
    fn result_many_round_trips_divides_out_the_summed_overcount() {
        // Regression for the real-world over-count: the captured turn-2 stats
        // (two trivial questions, 5 tool calls) reported total_tokens 82108 /
        // input_tokens 80528. Reporting the raw value made the context gauge
        // read ~80k for a tiny session; dividing by (5 + 1) calls yields a
        // realistic ~13.4k that's far below the summed total.
        let mut a = GeminiAdapter::new();
        let line = r#"{"type":"result","status":"success","stats":{"total_tokens":82108,"input_tokens":80528,"output_tokens":319,"cached":55143,"input":25385,"tool_calls":5}}"#;
        let events = run(&mut a, line);
        assert_eq!(events[0], "ContextTokens(13421)"); // 80528 / 6
    }

    #[test]
    fn result_without_input_tokens_skips_context_tokens() {
        // A result frame missing usable input_tokens must NOT emit ContextTokens
        // (and must not zero the live counter) — just the result Frame + Result.
        let mut a = GeminiAdapter::new();
        let line = r#"{"type":"result","status":"success","stats":{"tool_calls":1}}"#;
        let events = run(&mut a, line);
        assert_eq!(events.len(), 2);
        assert!(events[0].starts_with("Frame("));
        assert_eq!(events[1], "Result");
    }

    #[test]
    fn result_error_emits_assistant_bubble_then_error_frame_and_result_marker() {
        // A failed turn surfaces the error text as a VISIBLE assistant bubble
        // first (iOS only shows `[result: error]` for the result frame and
        // drops error.message), then the error result envelope, then Result.
        let mut a = GeminiAdapter::new();
        let line = r#"{"type":"result","status":"error","error":{"type":"quota","message":"rate limited"},"stats":{"total_tokens":0}}"#;
        let events = run(&mut a, line);
        assert_eq!(events.len(), 3);

        // 1) assistant text bubble carrying the human-readable error.
        let bubble = frame_value(&events[0]);
        assert_eq!(bubble["type"], "assistant");
        assert_eq!(bubble["message"]["content"][0]["type"], "text");
        assert_eq!(bubble["message"]["content"][0]["text"], "rate limited");

        // 2) error result envelope (the `[result: error]` terminator on iOS).
        let v = frame_value(&events[1]);
        assert_eq!(v["type"], "result");
        assert_eq!(v["subtype"], "error");
        assert_eq!(v["is_error"], true);
        assert_eq!(v["error"]["message"], "rate limited");

        // 3) Result marker.
        assert_eq!(events[2], "Result");
    }

    #[test]
    fn result_error_without_message_falls_back_to_generic_bubble() {
        // Frame missing error.message still produces a non-empty bubble so the
        // user never sees a bare `[result: error]` with no context.
        let mut a = GeminiAdapter::new();
        let line = r#"{"type":"result","status":"error","error":{"type":"unknown"},"stats":{"total_tokens":0}}"#;
        let events = run(&mut a, line);
        assert_eq!(events.len(), 3);
        assert_eq!(
            frame_value(&events[0])["message"]["content"][0]["text"],
            "turn failed"
        );
        assert_eq!(frame_value(&events[1])["subtype"], "error");
        assert_eq!(events[2], "Result");
    }

    #[test]
    fn pending_text_flushes_before_error_bubble() {
        // Trailing assistant text before a failed turn must flush as its own
        // bubble AHEAD of the error bubble — the error must not swallow or
        // reorder real reply text.
        let mut a = GeminiAdapter::new();
        let t = r#"{"type":"message","role":"assistant","content":"Working on it.","delta":true}"#;
        assert!(run(&mut a, t).is_empty());

        let line = r#"{"type":"result","status":"error","error":{"message":"boom"},"stats":{"total_tokens":0}}"#;
        let events = run(&mut a, line);
        // flush(text) + error bubble + error result frame + Result = 4 events.
        assert_eq!(events.len(), 4);
        assert_eq!(
            frame_value(&events[0])["message"]["content"][0]["text"],
            "Working on it."
        );
        assert_eq!(
            frame_value(&events[1])["message"]["content"][0]["text"],
            "boom"
        );
        assert_eq!(frame_value(&events[2])["subtype"], "error");
        assert_eq!(events[3], "Result");
    }

    #[test]
    fn repeated_result_success_dedups_context_tokens() {
        let mut a = GeminiAdapter::new();
        // input_tokens 1234, no tool calls → ContextTokens(1234); re-emitting the
        // identical frame must not double-fire it.
        let line =
            r#"{"type":"result","status":"success","stats":{"input_tokens":1234,"tool_calls":0}}"#;
        let first = run(&mut a, line);
        assert_eq!(first.len(), 3);
        assert_eq!(first[0], "ContextTokens(1234)");
        let second = run(&mut a, line);
        assert_eq!(second.len(), 2);
        assert!(second[0].starts_with("Frame("));
        assert_eq!(second[1], "Result");
    }

    #[test]
    fn unknown_frame_type_passed_through() {
        let mut a = GeminiAdapter::new();
        let line = r#"{"type":"some.future.event","payload":{"k":"v"}}"#;
        let events = run(&mut a, line);
        assert_eq!(events.len(), 1);
        assert!(events[0].starts_with("Frame("));
    }

    #[test]
    fn non_json_line_kept_as_frame() {
        let mut a = GeminiAdapter::new();
        let events = run(&mut a, "non-json-line");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0], "Frame(non-json-line)");
    }

    #[test]
    fn first_turn_uses_session_id_pipe_stdin_and_node_env() {
        let mut a = GeminiAdapter::new();
        let prompt_file = PathBuf::from("/tmp/p.txt");
        let c = ctx(&prompt_file, None, false, None);
        let cmd = a.prepare_command(&c).unwrap();
        assert!(cmd.contains("cat '/tmp/p.txt' |"), "got: {}", cmd);
        assert!(
            cmd.contains("ASDF_NODEJS_VERSION=24.14.0 gemini"),
            "got: {}",
            cmd
        );
        assert!(cmd.contains("--session-id "), "got: {}", cmd);
        assert!(!cmd.contains("--resume"), "got: {}", cmd);
        assert!(cmd.contains("-o stream-json"), "got: {}", cmd);
        // Session id passed must be the adapter's minted uuid.
        assert!(
            cmd.contains(&format!("--session-id '{}'", a.session_id)),
            "got: {}",
            cmd
        );
    }

    #[test]
    fn resume_turn_uses_resume_flag_not_session_id() {
        let mut a = GeminiAdapter::new();
        let prompt_file = PathBuf::from("/tmp/p.txt");
        let c = ctx(&prompt_file, Some("existing-sid"), false, None);
        let cmd = a.prepare_command(&c).unwrap();
        assert!(cmd.contains("--resume 'existing-sid'"), "got: {}", cmd);
        assert!(!cmd.contains("--session-id"), "got: {}", cmd);
    }

    #[test]
    fn non_sandboxed_appends_yolo_and_skip_trust() {
        let mut a = GeminiAdapter::new();
        let prompt_file = PathBuf::from("/tmp/p.txt");
        let c = ctx(&prompt_file, None, false, None);
        let cmd = a.prepare_command(&c).unwrap();
        assert!(
            cmd.contains("--approval-mode yolo --skip-trust"),
            "got: {}",
            cmd
        );
    }

    #[test]
    fn sandboxed_omits_yolo_and_skip_trust() {
        let mut a = GeminiAdapter::new();
        let prompt_file = PathBuf::from("/tmp/p.txt");
        let c = ctx(&prompt_file, None, true, None);
        let cmd = a.prepare_command(&c).unwrap();
        assert!(!cmd.contains("--approval-mode"), "got: {}", cmd);
        assert!(!cmd.contains("--skip-trust"), "got: {}", cmd);
    }

    #[test]
    fn model_some_appends_model_flag() {
        let mut a = GeminiAdapter::new();
        let prompt_file = PathBuf::from("/tmp/p.txt");
        let c = ctx(&prompt_file, None, false, Some("gemini-2.5-pro"));
        let cmd = a.prepare_command(&c).unwrap();
        assert!(cmd.contains("-m 'gemini-2.5-pro'"), "got: {}", cmd);
    }

    #[test]
    fn model_none_omits_model_flag() {
        let mut a = GeminiAdapter::new();
        let prompt_file = PathBuf::from("/tmp/p.txt");
        let c = ctx(&prompt_file, None, false, None);
        let cmd = a.prepare_command(&c).unwrap();
        assert!(!cmd.contains(" -m "), "got: {}", cmd);
    }
}
