//! cursor-agent adapter. Normalizes cursor-agent's stream-json frames into
//! claude-shape envelopes on the wire so iOS's `SpawnerMessageDescriber`
//! (which only knows claude's wire format) can render them.
//!
//! Frame mapping (cursor → claude-shape):
//!  - `system.init`              → `SessionIdHarvested` only (no Frame; matches claude's init-skip)
//!  - `user`                     → drop (iOS doesn't render user frames)
//!  - `assistant` (text blocks)  → Frame: `{"type":"assistant","message":{"content":[...],"usage":{...zeros}}}`
//!  - `tool_call.started`        → drop (mutable; the immutable version arrives on `completed`)
//!  - `tool_call.completed`      → Frame: one `assistant` envelope with `tool_use` block
//!  - `thinking.delta`           → drop (claude doesn't stream thinking mid-turn)
//!  - `thinking.completed`       → drop (claude's skip filter drops assistant-thinking-only frames too)
//!  - `result.success`           → ContextTokens + Frame (claude-shape result envelope) + Result
//!  - `result.*` (error variants) → Frame + Result
//!
//! Cursor's actual wire format observed via `cursor-agent --print
//! --output-format stream-json --force --trust` on 2026.05.20: tool calls use
//! `{"type":"tool_call","subtype":"started|completed","call_id":"...","tool_call":{"<verb>ToolCall":{"args":...,"result":...}}}`
//! (NOT the `tool_call.started` literal-type the research doc suggested).
//! Tool input/output schemas are per-verb (readToolCall, shellToolCall, etc.) —
//! we forward them as-is inside the claude-shape `tool_use.input` field; iOS
//! renders them via `SpawnerMessageDescriber.toolSummary` which only knows
//! claude tool names anyway, so cursor tool_use bubbles fall through to the
//! "tool name only" branch. That's acceptable for now.
//!
//! Also hosts the install/auth `probe()` for cursor (free function, not on
//! the `AgentAdapter` trait — `dyn AgentAdapter` can't dispatch statics).
//! `cursor-agent status --format json` is the canonical auth signal: a stable,
//! documented JSON output mode that reports both install presence (the binary
//! must be on PATH to run at all) and `isAuthenticated` in one call —
//! preferred over poking at `~/.cursor/cli-config.json`, which only stores
//! permissions, not tokens. The probe runs under a 15s timeout (above
//! realistic shell-rc cold-starts but well below the supervisor's 30s startup
//! ceiling) so a hung cursor-agent or slow `-lic` rc load can never starve
//! the install-status report.

use std::process::Stdio;
use std::time::Duration;

use anyhow::Result;
use serde_json::{json, Value};
use smallvec::SmallVec;
use tokio::process::Command;
use tracing::{debug, warn};

use crate::adapter::{
    shell_escape, AdapterDescriptor, AgentAdapter, AgentEvent, AgentKind, TurnContext,
    ATTACH_FILE_INSTRUCTION, MAX_STREAM_FRAME_BYTES,
};

/// Wired into `adapter::ADAPTERS`. See `adapter::AdapterDescriptor` for the
/// shape; the `probe` / `import` slots are filled by `_boxed` wrappers below
/// the `probe()` / `import()` definitions in this file.
pub const DESCRIPTOR: AdapterDescriptor = AdapterDescriptor {
    kind: AgentKind::Cursor,
    wire_name: "cursor",
    installed_col: "cursor_installed",
    authenticated_col: "cursor_authenticated",
    make: make_boxed,
    probe: probe_boxed,
    import: import_boxed,
};

fn make_boxed() -> Box<dyn AgentAdapter> {
    Box::new(CursorAdapter::new())
}

// Per-line frame-size cap lives in `adapter.rs::MAX_STREAM_FRAME_BYTES`, shared
// with the claude adapter. For cursor specifically, the frame type that can
// legitimately blow past the cap is `tool_call.completed`, whose
// `tool_call.<verb>ToolCall.result` may embed full file contents
// (`streamContent`, read output, etc.). For oversize lines we still try to
// recover `call_id` + the verb via a cheap substring sniff — if that
// succeeds we emit a minimal tool_use envelope with `input = {}` so the chat
// still shows the tool ran; otherwise we drop the line. Either way we avoid
// allocating a megabytes-large `Value` tree just to dispatch.

/// Per-turn state for the cursor adapter. Today this exists only to count
/// the distinct LLM calls that ran during a turn — cursor's `result.usage`
/// is *summed* across every internal call, so dividing by the call count is
/// the only way to get a per-call (≈ context-window) number out of the wire.
/// One adapter instance per spawn (`agent.rs:224`), so the state is
/// naturally scoped to a single turn; we also defensively reset on every
/// `Result` emission so a hypothetical multi-result run still behaves.
pub struct CursorAdapter {
    /// Distinct `model_call_id` values observed during the current turn,
    /// each representing one LLM call. Collected from assistant + tool_call
    /// frames. See the `result` branch in `handle_line` for how the count
    /// is used to rescale `result.usage` (sums across all calls) into a
    /// per-call context size.
    model_call_ids: std::collections::HashSet<String>,
    /// True when at least one assistant frame in the current turn arrived
    /// without a `model_call_id`. Observed empirically on the final
    /// summarizing assistant text frame in multi-call cursor turns —
    /// cursor doesn't surface its id, but it's still a real LLM call.
    /// Counted as +1 alongside `model_call_ids.len()` when computing
    /// `N_calls` at result time.
    saw_unidentified_assistant: bool,
}

impl CursorAdapter {
    pub fn new() -> Self {
        Self {
            model_call_ids: std::collections::HashSet::new(),
            saw_unidentified_assistant: false,
        }
    }

    /// Records the `model_call_id` from an already-parsed frame, if present.
    /// Called from the assistant + tool_call.completed branches in
    /// `handle_line`. `tool_call.started` frames are dropped by the prefilter
    /// before reaching here, but every `model_call_id` we'd see on a started
    /// frame is also present on its matching completed frame (verified
    /// against live cursor-agent stdout), so the count stays correct.
    fn record_model_call_id(&mut self, obj: &Value) {
        if let Some(id) = obj.get("model_call_id").and_then(|v| v.as_str()) {
            self.model_call_ids.insert(id.to_string());
        }
    }
}

impl Default for CursorAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentAdapter for CursorAdapter {
    fn kind(&self) -> AgentKind {
        AgentKind::Cursor
    }

    fn prepare_command(&mut self, ctx: &TurnContext<'_>) -> Result<String> {
        let mut cmd = String::new();
        if let Some(pp) = ctx.project_path {
            cmd.push_str(&format!("cd {} && ", shell_escape(pp)));
        }
        // cursor-agent has no `--append-system-prompt` (or any other system /
        // prompt-prefix flag — confirmed against `cursor-agent --help` 2026.05).
        // The only injection point is the stdin prompt itself, so we prepend
        // a short preamble before the user's prompt body.
        let preamble = format!("{}\n\n---\n\n", ATTACH_FILE_INSTRUCTION);
        cmd.push_str(&format!(
            "{{ printf %s {}; cat {}; }} | cursor-agent",
            shell_escape(&preamble),
            shell_escape(&ctx.prompt_file.to_string_lossy())
        ));
        // `--approve-mcps` is the headless equivalent of "user already clicked
        // approve in interactive mode" — without it, any MCP server the user
        // configured in `~/.cursor/` hangs waiting for a prompt that can't
        // appear under `--print`. Not policy curation: we only unblock what
        // the user already opted into.
        cmd.push_str(" --print --output-format stream-json --trust --approve-mcps");
        // Sender's `machine_users.is_sandboxed`. Non-sandboxed = `--force`
        // (aka `--yolo`) bypasses cursor's per-command permission gating;
        // sandboxed = omit it so cursor falls back to its default deny path.
        // Mirrors claude's `--dangerously-skip-permissions` gate. Note: this
        // only covers per-command permissions — `--trust` (workspace trust)
        // and `--approve-mcps` (MCP server approval) stay on for sandboxed
        // users too, because without them headless cursor either hangs on a
        // trust prompt or skips user-configured MCP servers.
        if !ctx.is_sandboxed {
            cmd.push_str(" --force");
        }
        // No `--session-id` flag on cursor-agent — use the harvest path:
        // omit `--resume` on first turn (cursor mints a session id and emits
        // it in `system.init`); pass it on subsequent turns.
        if let Some(sid) = ctx.agent_session_id {
            cmd.push_str(&format!(" --resume {}", shell_escape(sid)));
        }
        // Verbatim pass-through of `chats.model` (migration 0035). Empty /
        // blank values are already filtered to `None` at the construction
        // site in `main.rs`, so any `Some` here is a non-empty model name
        // the user picked in the composer's agent roster. Cursor's model
        // labels look like "Composer 2.5 Fast" — keep the shell-escape so
        // spaces survive into argv[].
        if let Some(model) = ctx.model {
            cmd.push_str(&format!(" --model {}", shell_escape(model)));
        }
        if ctx.worktree {
            // Deterministic name = idempotent dir, persists across --resume
            // because cursor's --resume doesn't restore cwd.
            let short: String = ctx.chat_id.chars().take(12).collect();
            let wt_name = format!("zcm-{}", short);
            cmd.push_str(&format!(" --worktree {}", shell_escape(&wt_name)));
            // NOTE: cursor-agent's `--worktree <name>` puts the directory at
            // `~/.cursor/worktrees/<reponame>/<name>` (per `cursor-agent --help`)
            // — NOT under the user's project tree like claude's --worktree
            // does. The path is hardcoded inside cursor-agent and cannot be
            // overridden from the CLI, so the same iOS Worktree toggle has
            // two very different mental models depending on the agent kind.
            // Users running `git status` in the project will see no changes;
            // they need to look under `~/.cursor/worktrees/<reponame>/zcm-*`
            // to find cursor's edits. iOS must surface this divergence in the
            // Worktree pill / chat header, or refuse worktree=true for cursor
            // entirely — anything user-facing is cross-file from here.
        }
        Ok(cmd)
    }

    fn handle_line(&mut self, line: String) -> SmallVec<[AgentEvent; 2]> {
        let mut out: SmallVec<[AgentEvent; 2]> = SmallVec::new();

        // Cheap substring prefilter for the high-frequency drops, so we
        // avoid a full `serde_json::Value` parse (which allocates a tree
        // of `Map<String, Value>` nodes) on frames we'd discard anyway.
        // Mirrors the pattern in `claude.rs::handle_line`. Uses `&line`
        // (borrow) so `line` is still owned for the normalize path below
        // — cursor synthesizes its own envelope via `json!`, so the
        // original line isn't reused as a `Frame` payload anyway.
        if line.starts_with('{') {
            if line.contains("\"type\":\"thinking\"") {
                // Drop both `delta` and `completed`: claude doesn't emit
                // mid-turn thinking either, and the iOS skip filter drops
                // assistant-thinking-only frames anyway.
                debug!("cursor-agent dropping thinking frame (prefilter)");
                return out;
            }
            if line.contains("\"type\":\"user\"") {
                // iOS doesn't render user frames; claude skip-filters them too.
                debug!("cursor-agent dropping user frame (prefilter)");
                return out;
            }
            if line.contains("\"type\":\"tool_call\"") && line.contains("\"subtype\":\"started\"") {
                // `started` is mutable; wait for `completed`.
                debug!("cursor-agent dropping tool_call.started frame (prefilter)");
                return out;
            }
        }

        // Oversize-frame guard. `serde_json::from_str` on a multi-MB line
        // allocates a tree of `Map<String, Value>` nodes; the `tool_call`
        // dispatch below needs the parsed tree, so without this guard a
        // single big edit can churn the heap by megabytes. For lines past
        // the cap we sniff the type substring and synthesize a minimal
        // envelope without a full parse. Mirrors the pattern in
        // `claude.rs::handle_line` (see `MAX_USAGE_FRAME_BYTES`).
        if line.len() > MAX_STREAM_FRAME_BYTES {
            if line.starts_with('{')
                && line.contains("\"type\":\"tool_call\"")
                && line.contains("\"subtype\":\"completed\"")
            {
                if let Some(frame) = synthesize_oversize_tool_call_completed(&line) {
                    out.push(AgentEvent::Frame(frame));
                } else {
                    warn!(
                        "cursor-agent dropping oversize tool_call.completed frame ({} bytes): \
                         could not recover call_id/verb via substring sniff",
                        line.len()
                    );
                }
                return out;
            }
            // Any other oversize frame type is dropped — `result` and
            // `assistant` payloads are normally tiny, so this branch
            // signals something unexpected on the wire.
            warn!(
                "cursor-agent dropping oversize frame ({} bytes) without full parse",
                line.len()
            );
            return out;
        }

        let Some(obj) = parse_obj(&line) else {
            // Non-JSON line (shouldn't happen on a healthy cursor-agent
            // stdout, but the line loop doesn't filter). Log and drop.
            debug!("cursor-agent non-JSON stdout line, dropping: {}", line);
            return out;
        };
        let Some(ty) = obj.get("type").and_then(|v| v.as_str()) else {
            debug!("cursor-agent frame missing type, dropping");
            return out;
        };
        let subtype = obj.get("subtype").and_then(|v| v.as_str());

        match ty {
            "system" => {
                // `system.init` → harvest session id; everything else dropped.
                if subtype == Some("init") {
                    if let Some(sid) = obj.get("session_id").and_then(|v| v.as_str()) {
                        out.push(AgentEvent::SessionIdHarvested(sid.to_string()));
                    } else {
                        debug!("cursor-agent dropping system.init frame without session_id");
                    }
                } else {
                    debug!(
                        "cursor-agent dropping system frame with subtype={:?}",
                        subtype
                    );
                }
            }
            "user" => {
                // Defensive — the prefilter above already drops these,
                // but keep the arm so the match stays exhaustive over
                // the cursor wire and any future code-path that bypasses
                // the prefilter still drops correctly.
                debug!("cursor-agent dropping user frame");
            }
            "assistant" => {
                // Track LLM-call count for the `result` arm's per-call
                // context-token rescale. Assistant frames usually carry
                // `model_call_id`, but the final summarizing assistant
                // frame in multi-call turns omits it — flag that as
                // `saw_unidentified_assistant` so it's counted too.
                if obj.get("model_call_id").is_some() {
                    self.record_model_call_id(&obj);
                } else {
                    self.saw_unidentified_assistant = true;
                }
                if let Some(frame) = normalize_assistant_frame(&obj) {
                    out.push(AgentEvent::Frame(frame));
                }
            }
            "tool_call" => {
                if subtype == Some("completed") {
                    // Count toward `model_call_ids` even if the
                    // tool_call.completed parse below fails — the id is
                    // the source of truth for "how many LLM calls ran."
                    self.record_model_call_id(&obj);
                    if let Some(frame) = normalize_tool_call_completed(&obj) {
                        out.push(AgentEvent::Frame(frame));
                    }
                } else {
                    // `started` is mutable; wait for `completed`. The
                    // prefilter handles started; any other subtype lands
                    // here.
                    debug!(
                        "cursor-agent dropping tool_call frame with subtype={:?}",
                        subtype
                    );
                }
            }
            "thinking" => {
                // Defensive — see `user` arm above.
                debug!("cursor-agent dropping thinking frame");
            }
            "result" => {
                let usage = obj.get("usage");
                // Cursor's `result.usage` is summed across every internal
                // LLM call in the turn (planner + each tool/subagent step),
                // NOT a per-call snapshot. Empirically (`tmp/cursor_frames*.ndjson`):
                //
                //   "windows app?" → 2 calls, sum(in+cR+cW) = 105k → /2 ≈ 52k
                //                    (claude answers the same in ~45k)
                //   "spawner tests?" → 6 calls, sum(in+cR+cW) = 301k → /6 ≈ 50k
                //
                // The constant ~50k across very different turn shapes is mostly
                // cursor's system prompt that gets carried into every call.
                // Dividing the cumulative billing by N_calls reconstructs an
                // approximate per-call context-window size — which is what
                // claude's adapter emits via per-frame `usage`, and what iOS
                // expects in `chats.context_tokens`.
                //
                // N_calls = distinct model_call_ids + 1 if any assistant frame
                // arrived without an id (the final summary frame, observed on
                // every multi-call turn). Clamped to ≥1 so a malformed turn
                // with no observed calls falls back to the raw sum.
                //
                // `inputTokens` is itself pre-adjusted by cursor
                // (max(raw − cacheRead − cacheWrite, 0)) so `in + cR + cW`
                // really does sum to the total per-call context across calls.
                // Omitted on interrupt/crash — leave context_tokens untouched.
                if let Some(u) = usage {
                    let input = u.get("inputTokens").and_then(|x| x.as_i64()).unwrap_or(0);
                    let cache_read = u
                        .get("cacheReadTokens")
                        .and_then(|x| x.as_i64())
                        .unwrap_or(0);
                    let cache_write = u
                        .get("cacheWriteTokens")
                        .and_then(|x| x.as_i64())
                        .unwrap_or(0);
                    let n_calls = (self.model_call_ids.len()
                        + usize::from(self.saw_unidentified_assistant))
                    .max(1) as i64;
                    out.push(AgentEvent::ContextTokens(
                        (input + cache_read + cache_write) / n_calls,
                    ));
                }
                // Emit claude-shape result envelope (subtype passes through,
                // duration_ms preserved, is_error preserved if present).
                let frame = normalize_result_frame(&obj);
                out.push(AgentEvent::Frame(frame));
                // Emit Result on every result frame; the supervisor latches it
                // (so AgentResponse::Done.has_result is set once and only once).
                out.push(AgentEvent::Result);
                // Reset per-turn state. One adapter per spawn today, but a
                // future supervisor change that reuses an adapter across
                // turns would otherwise carry stale call ids forward.
                self.model_call_ids.clear();
                self.saw_unidentified_assistant = false;
            }
            other => {
                debug!("cursor-agent unknown frame type, dropping: {}", other);
            }
        }
        out
    }
}

fn parse_obj(line: &str) -> Option<Value> {
    let s = line.trim();
    if !s.starts_with('{') {
        return None;
    }
    match serde_json::from_str::<Value>(s) {
        Ok(v) => v.is_object().then_some(v),
        Err(e) => {
            debug!("cursor-agent JSON parse failed: {}", e);
            None
        }
    }
}

/// Converts a cursor `assistant` frame to claude-shape. cursor's content
/// blocks already use `{"type":"text","text":"..."}` (identical to claude),
/// so we forward them verbatim and wrap the envelope to claude's shape.
fn normalize_assistant_frame(obj: &Value) -> Option<String> {
    let blocks = obj
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_array())
        .cloned()
        .unwrap_or_default();
    // Skip if no renderable blocks (claude does the same — assistant frames
    // with only thinking blocks are filtered out).
    let renderable = blocks.iter().any(|b| {
        b.get("type")
            .and_then(|t| t.as_str())
            .map(|s| s == "text" || s == "tool_use")
            .unwrap_or(false)
    });
    if !renderable {
        return None;
    }
    let envelope = json!({
        "type": "assistant",
        "message": {
            "content": blocks,
            // Usage is only known on `result.success` for cursor — zero here.
            // iOS treats per-frame usage as cumulative on claude too, so
            // emitting zeros mid-turn matches "no progress yet" (the real
            // ContextTokens event lands at end-of-turn via the result frame).
            "usage": {
                "input_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "output_tokens": 0,
            },
        },
    });
    Some(envelope.to_string())
}

/// Converts a cursor `tool_call.completed` frame to a claude-shape assistant
/// frame with a `tool_use` block. Cursor's per-verb nesting
/// (`tool_call.<verb>ToolCall.args`) is collapsed to claude's flat
/// `{name, input}`; output goes into `input` is NOT possible (claude only has
/// input on tool_use), so output is discarded — iOS's
/// `SpawnerMessageDescriber.toolSummary` reads `input` only anyway.
fn normalize_tool_call_completed(obj: &Value) -> Option<String> {
    let call_id = obj.get("call_id").and_then(|v| v.as_str())?;
    let tool_call = obj.get("tool_call")?.as_object()?;
    // Pick the verb entry by suffix, not by iteration order. `serde_json::Map`
    // is a `BTreeMap` without the `preserve_order` feature in our Cargo.toml,
    // so keys iterate in alphabetical order — a sibling like `_meta` or
    // `callId` added by a future cursor release would sort before the verb
    // and steal the `iter().next()` slot, hiding the real tool. Falling back
    // to first-entry + warn keeps unknown shapes visible in logs without
    // dropping the bubble.
    let (verb_key, verb_payload) = tool_call
        .iter()
        .find(|(k, _)| k.ends_with("ToolCall"))
        .or_else(|| {
            let first = tool_call.iter().next()?;
            warn!(
                "cursor-agent tool_call has no *ToolCall key; falling back to first entry '{}'",
                first.0
            );
            Some(first)
        })?;
    let raw_args = verb_payload.get("args").cloned().unwrap_or(Value::Null);
    let (name, input) = map_cursor_tool(verb_key, &raw_args);
    let envelope = json!({
        "type": "assistant",
        "message": {
            "content": [{
                "type": "tool_use",
                "id": call_id,
                "name": name,
                "input": input,
            }],
            "usage": {
                "input_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "output_tokens": 0,
            },
        },
    });
    Some(envelope.to_string())
}

/// Substring-based synthesis of a minimal claude-shape tool_use envelope for
/// oversize `tool_call.completed` frames. We can't parse the full JSON tree
/// without allocating megabytes (the `result` field embeds file contents), so
/// we sniff `call_id` and the verb key directly from the line and emit an
/// empty `input`. iOS's `SpawnerMessageDescriber.toolSummary` returns the
/// tool-name-only branch for unknown inputs, which is the right thing to
/// render here — the user still sees "ran Edit" instead of a missing bubble.
/// Returns `None` if either sniff fails; caller logs + drops the line.
fn synthesize_oversize_tool_call_completed(line: &str) -> Option<String> {
    let call_id = sniff_string_field(line, "\"call_id\":\"")?;
    let verb_key = sniff_tool_call_verb(line)?;
    let (name, _input) = map_cursor_tool(&verb_key, &Value::Null);
    let envelope = json!({
        "type": "assistant",
        "message": {
            "content": [{
                "type": "tool_use",
                "id": call_id,
                "name": name,
                "input": {},
            }],
            "usage": {
                "input_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "output_tokens": 0,
            },
        },
    });
    Some(envelope.to_string())
}

/// Extracts a JSON string field's value by literal substring after `needle`,
/// up to the next unescaped `"`. Best-effort and tolerant of escapes only at
/// the boundary — fine for `call_id` (uuid-shaped, no quotes/backslashes) and
/// the verb key (alphanumeric). Returns `None` if the needle is absent or the
/// closing quote isn't found within a small window.
fn sniff_string_field(line: &str, needle: &str) -> Option<String> {
    let start = line.find(needle)? + needle.len();
    let rest = &line[start..];
    // Cap the search window so a malformed/huge value can't pin us scanning
    // multi-MB strings for a closing quote that isn't there.
    let window = &rest[..rest.len().min(512)];
    let end = window.find('"')?;
    Some(window[..end].to_string())
}

/// Locates the verb key inside `"tool_call":{ "<verb>ToolCall": ... }` via
/// substring search. We look for the `"tool_call":{` opener, then the first
/// `"…ToolCall":` key after it. Cap the scan to a fixed window so an oversize
/// line doesn't drag this O(n) scan through megabytes of payload.
fn sniff_tool_call_verb(line: &str) -> Option<String> {
    let anchor = line.find("\"tool_call\":{")? + "\"tool_call\":{".len();
    let rest = &line[anchor..];
    let window = &rest[..rest.len().min(512)];
    // Expect: `"<verbKey>":`. Find the first `"`, then the next `"`.
    let open = window.find('"')? + 1;
    let after_open = &window[open..];
    let close = after_open.find('"')?;
    let key = &after_open[..close];
    if key.ends_with("ToolCall") {
        Some(key.to_string())
    } else {
        None
    }
}

/// Maps a cursor `<verb>ToolCall` key + its raw `args` to a claude tool name
/// and a claude-shape input. iOS's `SpawnerMessageDescriber.toolSummary`
/// dispatches on the claude name and reads claude-named arg keys
/// (`command`, `file_path`, `pattern`, ...), so we rename keys here on the
/// spawner side. iOS is e2e-encrypted and in users' hands — it can't be
/// force-upgraded to learn cursor's verbs, so this map is the only place
/// it'll work.
///
/// Known verbs (cursor → claude):
///   shellToolCall  → Bash   (args.command unchanged)
///   readToolCall   → Read   (args.path → input.file_path)
///   editToolCall   → Edit   (args.path → input.file_path)
///   writeToolCall  → Write  (args.path → input.file_path)
///   grepToolCall   → Grep   (args.pattern unchanged — assumed)
///   globToolCall   → Glob   (args.pattern unchanged — assumed)
///   anything else  → strip `"ToolCall"` suffix, args passthrough.
///
/// For Read/Edit/Write the `path → file_path` rename is in-place on a clone
/// of the args object; sibling keys (offset, streamContent, etc.) are
/// preserved untouched — iOS ignores them but keeping the wire faithful
/// makes the persisted body useful for future tooling.
fn map_cursor_tool(verb_key: &str, args: &Value) -> (String, Value) {
    match verb_key {
        "shellToolCall" => ("Bash".to_string(), args.clone()),
        "readToolCall" => ("Read".to_string(), rename_path_to_file_path(args)),
        "editToolCall" => ("Edit".to_string(), rename_path_to_file_path(args)),
        "writeToolCall" => ("Write".to_string(), rename_path_to_file_path(args)),
        "grepToolCall" => ("Grep".to_string(), args.clone()),
        "globToolCall" => ("Glob".to_string(), args.clone()),
        other => {
            let name = other.strip_suffix("ToolCall").unwrap_or(other).to_string();
            (name, args.clone())
        }
    }
}

/// Clones `args` and, if it's an object containing `path`, moves the value
/// under `file_path` (claude's expected key) and removes `path`. Other keys
/// are preserved. Non-object args pass through unchanged (defensive — this
/// shouldn't happen on the cursor wire).
fn rename_path_to_file_path(args: &Value) -> Value {
    let Some(map) = args.as_object() else {
        return args.clone();
    };
    let mut out = map.clone();
    if let Some(path_val) = out.remove("path") {
        out.insert("file_path".to_string(), path_val);
    }
    Value::Object(out)
}

/// Converts a cursor `result` frame to claude-shape. Preserves `subtype`,
/// `duration_ms`, `is_error` so iOS's describer renders `[result: success
/// (Ns)]` correctly. Usage is rewritten to claude's snake_case shape if
/// present (purely cosmetic for the persisted body — the ContextTokens event
/// is what updates the chat-row column).
fn normalize_result_frame(obj: &Value) -> String {
    let subtype = obj
        .get("subtype")
        .and_then(|v| v.as_str())
        .unwrap_or("success")
        .to_string();
    let duration_ms = obj.get("duration_ms").cloned().unwrap_or(Value::Null);
    let is_error = obj.get("is_error").cloned().unwrap_or(Value::Bool(false));
    let usage = obj.get("usage").map(|u| {
        json!({
            "input_tokens": u.get("inputTokens").cloned().unwrap_or(json!(0)),
            "cache_read_input_tokens": u.get("cacheReadTokens").cloned().unwrap_or(json!(0)),
            "cache_creation_input_tokens": u.get("cacheWriteTokens").cloned().unwrap_or(json!(0)),
            "output_tokens": u.get("outputTokens").cloned().unwrap_or(json!(0)),
        })
    });
    let mut envelope = json!({
        "type": "result",
        "subtype": subtype,
        "duration_ms": duration_ms,
        "is_error": is_error,
    });
    if let Some(u) = usage {
        envelope["usage"] = u;
    }
    envelope.to_string()
}

/// Hard ceiling on the `cursor-agent status` shell-out. Includes the user's
/// `-lic` rc load (asdf, nvm, slow prompt init) plus the cursor-agent process
/// itself. 15s is comfortably above realistic shell-rc cold-starts (~1-3s) but
/// well below the supervisor's 30s `AGENT_STARTUP_TIMEOUT` so a hang here can
/// never starve the install-status report indefinitely.
const STATUS_PROBE_TIMEOUT: Duration = Duration::from_secs(15);

/// Probe install + auth in one shot. Returns `(installed, authenticated)`
/// — the writer flattens both pairs (claude + cursor) into a single PATCH on
/// `machines`'s four boolean columns.
///
/// Auth detection: `cursor-agent status --format json` returns a JSON object
/// with `isAuthenticated: true|false`. Stable contract (the JSON output mode
/// is documented in the public CLI help) so this is preferred over poking at
/// `~/.cursor/cli-config.json` (which only stores permissions, not tokens —
/// the actual token lives elsewhere and isn't documented). When the binary
/// is absent we report `(false, false)`; when it's on PATH but the auth
/// check fails (timeout, parse error, non-zero exit, isAuthenticated=false)
/// we report `(true, false)` — same wire shape as "logged out", and each
/// failure mode emits a distinctive `warn!` so a future cursor schema
/// rename leaves a breadcrumb in the spawner log.
pub async fn probe() -> (bool, bool) {
    if !crate::shell::binary_on_path("cursor-agent").await {
        return (false, false);
    }
    (true, is_authenticated().await)
}

/// `fn`-pointer-shaped wrapper around `probe()` for `AdapterDescriptor.probe`.
/// `BoxFuture` erases the concrete async-fn type so the descriptor can hold
/// all adapters' probes in a single slice.
fn probe_boxed() -> futures::future::BoxFuture<'static, (bool, bool)> {
    Box::pin(probe())
}

/// Runs `cursor-agent status --format json` through the user's login shell so
/// PATH/asdf/etc resolve the same way as the agent spawn. Returns `false` on
/// any failure (binary missing under shell PATH, non-zero exit, unparseable
/// JSON, timeout, or `isAuthenticated: false`).
///
/// Every failure mode collapses to `false`, which `probe()` then reports as
/// `(true, false)` — same wire shape as "user not logged in" on iOS. We
/// can't widen the wire locally (it's just two booleans), so each failure
/// mode emits a distinctive `warn!` to leave a breadcrumb in the spawner
/// log when cursor's schema drifts.
async fn is_authenticated() -> bool {
    let run = Command::new(crate::shell::user_login_shell())
        .args(["-lic", "cursor-agent status --format json"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output();

    let output = match tokio::time::timeout(STATUS_PROBE_TIMEOUT, run).await {
        Ok(Ok(o)) => o,
        Ok(Err(e)) => {
            warn!(error = %e, "cursor-agent status: spawn failed");
            return false;
        }
        Err(_) => {
            warn!(
                timeout_secs = STATUS_PROBE_TIMEOUT.as_secs(),
                "cursor-agent status: timed out (slow shell rc or hung cursor-agent); reporting not_authenticated"
            );
            return false;
        }
    };

    if !output.status.success() {
        warn!(
            exit_code = ?output.status.code(),
            "cursor-agent status: non-zero exit; reporting not_authenticated (may actually be a broken install)"
        );
        return false;
    }

    // The JSON object may be preceded by shell-rc noise (e.g. asdf banner
    // lines printed on `-lic`). Find the first '{' and parse from there.
    let stdout = String::from_utf8_lossy(&output.stdout);
    let Some(start) = stdout.find('{') else {
        warn!(
            stdout_len = stdout.len(),
            "cursor-agent status: no '{{' in stdout (binary printed non-JSON?); reporting not_authenticated"
        );
        return false;
    };
    match serde_json::from_str::<StatusJson>(&stdout[start..]) {
        Ok(s) if s.is_authenticated => true,
        Ok(_) => {
            // Parsed cleanly but `isAuthenticated` was false or absent. The
            // `#[serde(default)]` on the field means a schema rename (e.g.
            // cursor switching to `authenticated`) silently lands here — log
            // the raw payload tail so an operator can spot drift.
            let tail = &stdout[start..];
            let snippet: String = tail.chars().take(200).collect();
            warn!(payload_snippet = %snippet, "cursor-agent status: isAuthenticated=false or field missing (schema drift?)");
            false
        }
        Err(e) => {
            warn!(error = %e, "cursor-agent status: JSON parse failed; reporting not_authenticated");
            false
        }
    }
}

#[derive(serde::Deserialize)]
struct StatusJson {
    #[serde(rename = "isAuthenticated", default)]
    is_authenticated: bool,
}

// ===========================================================================
// Cursor-history importer. Walks the global
// `~/Library/Application Support/Cursor/User/globalStorage/state.vscdb` SQLite
// db read-only and emits `PutProject` / `PutChat` / `PutMessage` events shaped
// identically to claude's importer output. Per-workspace `state.vscdb` files
// are NOT read — `composer.composerHeaders` in the global ItemTable already
// carries `workspaceIdentifier.uri.path` for chats that lived in a workspace,
// and headerless composers (subagent / best-of-N spawns) have no UI
// representation in Cursor either, so we skip them.
//
// Wire-shape mapping (cursor persisted bubble → claude-shape stream-json):
//   type=1 (user)       → MessageEnvelope { text } JSON (sender="user")
//   type=2 with text    → `{"type":"assistant","message":{"content":[{"type":"text",...}], "usage":zeros}}`
//   type=2 + toolFormer → one extra assistant frame with a `tool_use` block
//                         (toolFormerData.rawArgs is a JSON STRING — parse
//                         with `serde_json::from_str` before re-keying;
//                         claude name + key renames come from
//                         `map_cursor_persisted_tool`).
//   text-then-tool      → text frame first, tool_use frame second (each is its
//                         own messages row; iOS orders by writer-assigned seq).
//   thinking-only / empty + no tool → dropped (matches the live adapter's
//                         skip filter for assistant-thinking-only frames).
//
// Filtering rules (`accept_header`): drop drafts (never sent),
// best-of-N subcomposers (defensive — `allComposers` filter usually catches
// them), and composers whose `composerData:` row is missing.
//
// Idempotency: chat ids and message ids are the original cursor UUIDs
// (composerId / bubbleId), so re-imports converge through the backend's
// `INSERT ... ON CONFLICT (id) DO NOTHING` clause (writer.rs:46-49). Project
// ids share `mint_project_id` with claude so a project that has transcripts
// from both CLIs collapses to a single `projects` row.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Duration as ChronoDuration, TimeZone, Utc};
use rusqlite::OpenFlags;
use tokio::sync::mpsc;
use tracing::info;
use uuid::Uuid;

use crate::adapter::ImportProgress;
use crate::adapters::import_shared::{basename_or, collapse_title, mint_project_id};
use crate::envelope::MessageEnvelope;
use crate::writer::WriteEvent;

/// Sentinel `projects.path` for cursor composers whose `workspaceIdentifier`
/// carries only an opaque id (no `uri.path`). Real project paths are
/// absolute (always start with `/`), so this string can't collide. Kept
/// short + human-readable because it surfaces in the iOS project list and
/// (via `basename_or`) in the project name fallback.
const CURSOR_NO_PROJECT_PATH: &str = "<no project>";
const CURSOR_NO_PROJECT_NAME: &str = "Cursor (no project)";

/// Parsed `composer.composerHeaders.allComposers[i]` entry. Only the fields
/// we actually use — serde drops the rest via the catch-all (no #[serde(deny_unknown_fields)]).
struct ComposerHeader {
    id: String,
    /// Resolved workspace path, or `None` when only `workspaceIdentifier.id`
    /// is present (no `uri.path`). Caller buckets `None` under
    /// `CURSOR_NO_PROJECT_PATH`.
    project_path: Option<String>,
    created_at_ms: i64,
    /// `composerData.name` is preferred for the title; the header name is
    /// the fallback if the per-chat row is missing it.
    header_name: Option<String>,
}

pub(crate) async fn import(
    machine_id: Uuid,
    user_id: Uuid,
    write_tx: mpsc::Sender<WriteEvent>,
    progress: ImportProgress,
) -> anyhow::Result<()> {
    let Some(db_path) = cursor_state_db_path() else {
        info!("HOME not set, skipping cursor history import");
        return Ok(());
    };
    if !db_path.exists() {
        // Same posture as claude when `~/.claude/projects` is absent: no
        // Cursor installed (or never opened) → nothing to import. Don't
        // touch `progress` — the dispatcher's per-kind rescaler treats
        // no-progress as zero-weight, which is what we want here.
        info!(path = %db_path.display(), "no cursor state.vscdb, nothing to import");
        return Ok(());
    }
    info!(path = %db_path.display(), "scanning cursor-agent transcripts");

    // All rusqlite work happens on a blocking thread — rusqlite is sync and
    // even with WAL the SQLite file lock is held for the duration of each
    // query. Doing this on a tokio runtime worker would block the executor.
    // We pull every row off the disk on the blocking thread, drop the
    // connection, then iterate + emit on the async side so progress
    // throttling and channel back-pressure work normally.
    let raw =
        tokio::task::spawn_blocking(move || -> anyhow::Result<RawScan> { read_state_db(&db_path) })
            .await
            .map_err(|e| anyhow::anyhow!("cursor sqlite blocking task panicked: {e}"))??;

    let RawScan {
        headers,
        composer_rows,
        bubble_rows,
    } = raw;

    if headers.is_empty() {
        info!("cursor: composer.composerHeaders empty, nothing to import");
        return Ok(());
    }

    // Group composers by project path (sentinel for unresolved). Sort each
    // group's composers by created_at so progress is deterministic across
    // re-runs and the iOS UI walks oldest→newest.
    let mut by_project: BTreeMap<String, Vec<ComposerHeader>> = BTreeMap::new();
    for h in headers {
        let key = h
            .project_path
            .clone()
            .unwrap_or_else(|| CURSOR_NO_PROJECT_PATH.to_string());
        by_project.entry(key).or_default().push(h);
    }
    let total_composers: usize = by_project.values().map(|v| v.len()).sum();
    if total_composers == 0 {
        return Ok(());
    }
    info!(
        projects = by_project.len(),
        composers = total_composers,
        "starting cursor import"
    );

    let mut done_composers: usize = 0;
    let mut last_pct: i32 = 0;
    for (project_path, mut composers) in by_project {
        composers.sort_by_key(|h| h.created_at_ms);
        let project_id = mint_project_id(machine_id, &project_path);
        let project_name = if project_path == CURSOR_NO_PROJECT_PATH {
            CURSOR_NO_PROJECT_NAME.to_string()
        } else {
            basename_or(&project_path, "project")
        };
        let _ = write_tx
            .send(WriteEvent::PutProject {
                id: project_id,
                machine_id,
                name: project_name,
                path: project_path.clone(),
            })
            .await;

        for header in composers {
            if let Err(e) = import_composer(
                &header,
                project_id,
                user_id,
                &composer_rows,
                &bubble_rows,
                &write_tx,
            )
            .await
            {
                tracing::warn!(composer_id = %header.id, error = %e, "cursor composer import failed, skipping");
            }
            done_composers += 1;
            // Same 5%-step throttle as the claude importer — each progress
            // call fans out via PowerSync to every connected client, so
            // throttling avoids burning watch wakeups for negligible UX gain.
            let pct = ((done_composers as f64 / total_composers as f64) * 100.0) as i32;
            if pct >= last_pct + 5 {
                last_pct = pct;
                progress(pct.clamp(0, 100) as u8);
            }
        }
    }

    info!(composers = done_composers, "cursor history import complete");
    Ok(())
}

fn cursor_state_db_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME").map(PathBuf::from);
    // Empty XDG_CONFIG_HOME → treat as unset (VS Code / Cursor follow XDG spec
    // which says empty values fall back to defaults).
    let xdg = std::env::var_os("XDG_CONFIG_HOME")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from);
    cursor_state_db_path_for(std::env::consts::OS, home.as_deref(), xdg.as_deref())
}

/// Pure-function half of `cursor_state_db_path` so unit tests can exercise
/// each platform without env mutation. Cursor inherits VS Code's user-data
/// layout — `<dataDir>/User/globalStorage/state.vscdb` — and only the `dataDir`
/// differs per OS. `install.sh` ships macOS + Linux only, so other targets
/// return `None`.
fn cursor_state_db_path_for(
    os: &str,
    home: Option<&Path>,
    xdg_config_home: Option<&Path>,
) -> Option<PathBuf> {
    let data_dir = match os {
        "macos" => home?.join("Library/Application Support/Cursor"),
        "linux" => xdg_config_home
            .map(PathBuf::from)
            .or_else(|| home.map(|h| h.join(".config")))?
            .join("Cursor"),
        _ => return None,
    };
    Some(data_dir.join("User/globalStorage/state.vscdb"))
}

/// Raw blob payloads from the SQLite walk. We materialize the per-composer
/// and per-bubble JSON values up-front (after applying the
/// `composer.composerHeaders` allow-list) so the async side can iterate
/// without holding a connection — rusqlite is sync, and Tokio's
/// `spawn_blocking` worker shouldn't outlive the bounded blob loads.
struct RawScan {
    headers: Vec<ComposerHeader>,
    /// `composerId` → parsed `composerData:<composerId>` row. Missing entries
    /// mean the header pointed at a row that doesn't exist (rare; usually a
    /// stale header from before cleanup); caller warns + skips.
    composer_rows: std::collections::HashMap<String, serde_json::Value>,
    /// `(composerId, bubbleId)` → parsed bubble row.
    bubble_rows: std::collections::HashMap<(String, String), serde_json::Value>,
}

fn read_state_db(db_path: &std::path::Path) -> anyhow::Result<RawScan> {
    // Read-only + no-mutex: read-only protects against accidental corruption
    // if Cursor is running, and no-mutex is fine for a single-connection load.
    let conn = rusqlite::Connection::open_with_flags(
        db_path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    let headers = read_composer_headers(&conn)?;
    if headers.is_empty() {
        return Ok(RawScan {
            headers,
            composer_rows: Default::default(),
            bubble_rows: Default::default(),
        });
    }
    let ids: Vec<String> = headers.iter().map(|h| h.id.clone()).collect();
    let composer_rows = load_composer_rows(&conn, &ids)?;
    let bubble_rows = load_bubble_rows(&conn, &composer_rows)?;
    Ok(RawScan {
        headers,
        composer_rows,
        bubble_rows,
    })
}

fn read_composer_headers(conn: &rusqlite::Connection) -> anyhow::Result<Vec<ComposerHeader>> {
    let blob: Option<Vec<u8>> = conn
        .query_row(
            "SELECT value FROM ItemTable WHERE key = 'composer.composerHeaders'",
            [],
            |r| r.get::<_, Vec<u8>>(0),
        )
        .optional_or_none()?;
    let Some(blob) = blob else {
        return Ok(Vec::new());
    };
    let parsed: serde_json::Value = serde_json::from_slice(&blob)
        .map_err(|e| anyhow::anyhow!("composer.composerHeaders not JSON: {e}"))?;
    let Some(all) = parsed.get("allComposers").and_then(|v| v.as_array()) else {
        return Ok(Vec::new());
    };
    let mut out = Vec::with_capacity(all.len());
    for entry in all {
        if !accept_header(entry) {
            continue;
        }
        let Some(id) = entry.get("composerId").and_then(|v| v.as_str()) else {
            continue;
        };
        let project_path = entry
            .get("workspaceIdentifier")
            .and_then(|w| w.get("uri"))
            .and_then(|u| u.get("path"))
            .and_then(|p| p.as_str())
            .map(|s| s.to_string());
        let created_at_ms = entry.get("createdAt").and_then(|v| v.as_i64()).unwrap_or(0);
        let header_name = entry
            .get("name")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        out.push(ComposerHeader {
            id: id.to_string(),
            project_path,
            created_at_ms,
            header_name,
        });
    }
    Ok(out)
}

/// Header-level gate: drop composers that shouldn't import.
///
/// - `isDraft`: composer started but never sent; bubbles empty.
/// - `isBestOfNSubcomposer`: A/B branches the UI never shows.
/// - `subagentInfo` present: spawned by Cursor's subagent feature (e.g.
///   `subagentTypeName:"explore"`). These DO appear in `allComposers` with
///   their task as `name` — so `allComposers`-membership is NOT enough to
///   filter them; observed ~5% of headers on a real Cursor install. The
///   parent composer's tool_use bubble already references the subagent's
///   work; importing the subagent as a standalone chat would surface junk
///   like "Time picker and platform detection search" in the user's chat
///   list.
fn accept_header(entry: &serde_json::Value) -> bool {
    if entry.get("isDraft").and_then(|v| v.as_bool()) == Some(true) {
        return false;
    }
    if entry.get("isBestOfNSubcomposer").and_then(|v| v.as_bool()) == Some(true) {
        return false;
    }
    if entry.get("subagentInfo").is_some() {
        return false;
    }
    true
}

fn load_composer_rows(
    conn: &rusqlite::Connection,
    ids: &[String],
) -> anyhow::Result<std::collections::HashMap<String, serde_json::Value>> {
    let mut stmt = conn.prepare("SELECT value FROM cursorDiskKV WHERE key = ?")?;
    let mut out = std::collections::HashMap::with_capacity(ids.len());
    for id in ids {
        let key = format!("composerData:{id}");
        let blob: Option<Vec<u8>> = stmt
            .query_row([&key], |r| r.get::<_, Vec<u8>>(0))
            .optional_or_none()?;
        let Some(blob) = blob else {
            tracing::warn!(composer_id = %id, "cursor: composerData row missing for header, skipping");
            continue;
        };
        match serde_json::from_slice::<serde_json::Value>(&blob) {
            Ok(v) => {
                out.insert(id.clone(), v);
            }
            Err(e) => {
                tracing::warn!(composer_id = %id, error = %e, "cursor: composerData JSON parse failed");
            }
        }
    }
    Ok(out)
}

fn load_bubble_rows(
    conn: &rusqlite::Connection,
    composer_rows: &std::collections::HashMap<String, serde_json::Value>,
) -> anyhow::Result<std::collections::HashMap<(String, String), serde_json::Value>> {
    let mut stmt = conn.prepare("SELECT value FROM cursorDiskKV WHERE key = ?")?;
    let mut out = std::collections::HashMap::new();
    for (composer_id, row) in composer_rows {
        let Some(headers) = row
            .get("fullConversationHeadersOnly")
            .and_then(|v| v.as_array())
        else {
            continue;
        };
        for h in headers {
            let Some(bubble_id) = h.get("bubbleId").and_then(|v| v.as_str()) else {
                continue;
            };
            let key = format!("bubbleId:{composer_id}:{bubble_id}");
            let blob: Option<Vec<u8>> = stmt
                .query_row([&key], |r| r.get::<_, Vec<u8>>(0))
                .optional_or_none()?;
            let Some(blob) = blob else {
                tracing::warn!(
                    composer_id = %composer_id,
                    bubble_id = %bubble_id,
                    "cursor: bubble row missing, skipping"
                );
                continue;
            };
            match serde_json::from_slice::<serde_json::Value>(&blob) {
                Ok(v) => {
                    out.insert((composer_id.clone(), bubble_id.to_string()), v);
                }
                Err(e) => {
                    tracing::warn!(
                        composer_id = %composer_id,
                        bubble_id = %bubble_id,
                        error = %e,
                        "cursor: bubble JSON parse failed"
                    );
                }
            }
        }
    }
    Ok(out)
}

/// Small rusqlite helper: convert `QueryReturnedNoRows` into `Ok(None)`,
/// keep other errors as `Err`. Mirrors `rusqlite::OptionalExtension::optional`
/// but spelled out so we don't pull an extension trait into the module.
trait OptionalOrNone<T> {
    fn optional_or_none(self) -> rusqlite::Result<Option<T>>;
}

impl<T> OptionalOrNone<T> for rusqlite::Result<T> {
    fn optional_or_none(self) -> rusqlite::Result<Option<T>> {
        match self {
            Ok(v) => Ok(Some(v)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e),
        }
    }
}

async fn import_composer(
    header: &ComposerHeader,
    project_id: Uuid,
    user_id: Uuid,
    composer_rows: &std::collections::HashMap<String, serde_json::Value>,
    bubble_rows: &std::collections::HashMap<(String, String), serde_json::Value>,
    write_tx: &mpsc::Sender<WriteEvent>,
) -> anyhow::Result<()> {
    let Some(row) = composer_rows.get(&header.id) else {
        // Already warned at load time; nothing to do.
        return Ok(());
    };
    let chat_id = Uuid::parse_str(&header.id)
        .map_err(|e| anyhow::anyhow!("composerId {} is not a UUID: {e}", header.id))?;
    let chat_created_at = epoch_ms_to_utc(header.created_at_ms);

    // Walk bubbles in fullConversationHeadersOnly order — that's the
    // canonical message order (bubbles have no per-message timestamp).
    let Some(headers) = row
        .get("fullConversationHeadersOnly")
        .and_then(|v| v.as_array())
    else {
        return Ok(());
    };

    let mut emitted: Vec<PendingMsg> = Vec::new();
    let mut first_user_text: Option<String> = None;
    for h in headers {
        let Some(bubble_id) = h.get("bubbleId").and_then(|v| v.as_str()) else {
            continue;
        };
        let Some(bubble) = bubble_rows.get(&(header.id.clone(), bubble_id.to_string())) else {
            continue;
        };
        match classify_bubble(bubble) {
            BubbleOut::User { text } => {
                if first_user_text.is_none() {
                    first_user_text = Some(text.clone());
                }
                let env = MessageEnvelope {
                    text,
                    attachments: Vec::new(),
                };
                let body =
                    serde_json::to_string(&env).expect("MessageEnvelope is always serializable");
                emitted.push(PendingMsg {
                    bubble_id: bubble_id.to_string(),
                    sender: "user",
                    body,
                });
            }
            BubbleOut::Assistant {
                text_frame,
                tool_frame,
            } => {
                if let Some(frame) = text_frame {
                    emitted.push(PendingMsg {
                        bubble_id: bubble_id.to_string(),
                        sender: "agent",
                        body: frame,
                    });
                }
                if let Some(frame) = tool_frame {
                    emitted.push(PendingMsg {
                        bubble_id: bubble_id.to_string(),
                        sender: "agent",
                        body: frame,
                    });
                }
            }
            BubbleOut::Skip => {}
        }
    }

    if emitted.is_empty() {
        return Ok(());
    }

    // Title preference: composerData.name → header.name → first user
    // prompt → generic fallback. Matches the live UX (Cursor itself shows
    // the auto-titled `name` in its chat list).
    let title = row
        .get("name")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .or_else(|| header.header_name.clone())
        .or_else(|| first_user_text.as_deref().map(collapse_title))
        .unwrap_or_else(|| "Imported chat".to_string());

    // PutChat MUST land before its messages — matches the claude importer
    // ordering, and the backend's FK on `messages.chat_id` rejects orphans.
    let _ = write_tx
        .send(WriteEvent::PutChat {
            id: chat_id,
            project_id,
            user_id,
            title,
            created_at: chat_created_at,
        })
        .await;

    // Per-message `created_at`: bubbles have no clock. Use
    // `chat_created_at + idx ms` so the writer's `created_at` is monotonic
    // per row (text+tool_use from the same bubble get distinct timestamps
    // because they occupy consecutive emit slots).
    for (seq, msg) in emitted.into_iter().enumerate() {
        let bubble_uuid = Uuid::parse_str(&msg.bubble_id).ok();
        let ts = chat_created_at + ChronoDuration::milliseconds(seq as i64);
        let _ = write_tx
            .send(WriteEvent::PutMessage {
                id: bubble_uuid,
                chat_id: chat_id.to_string(),
                user_id,
                sender: msg.sender,
                content: msg.body,
                created_at: Some(ts),
                imported: true,
            })
            .await;
    }

    Ok(())
}

struct PendingMsg {
    bubble_id: String,
    sender: &'static str,
    body: String,
}

enum BubbleOut {
    User {
        text: String,
    },
    Assistant {
        text_frame: Option<String>,
        tool_frame: Option<String>,
    },
    Skip,
}

fn classify_bubble(bubble: &serde_json::Value) -> BubbleOut {
    let bubble_type = bubble.get("type").and_then(|v| v.as_i64()).unwrap_or(-1);
    match bubble_type {
        1 => classify_user_bubble(bubble),
        2 => classify_assistant_bubble(bubble),
        _ => BubbleOut::Skip,
    }
}

fn classify_user_bubble(bubble: &serde_json::Value) -> BubbleOut {
    let text = bubble
        .get("text")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim();
    if text.is_empty() {
        return BubbleOut::Skip;
    }
    // Defensive: no observed cases of <system-reminder>-style wrappers in
    // cursor bubbles (Cursor isn't claude code, so the TUI's synthetic
    // wrapper rows don't appear), but the check is cheap and prevents a
    // future drift from sneaking harness messages into the chat.
    if super_is_synthetic_wrapper(text) {
        return BubbleOut::Skip;
    }
    BubbleOut::User {
        text: text.to_string(),
    }
}

fn classify_assistant_bubble(bubble: &serde_json::Value) -> BubbleOut {
    let text = bubble.get("text").and_then(|v| v.as_str()).unwrap_or("");
    let tool_former = bubble.get("toolFormerData");

    let text_frame = if text.is_empty() {
        None
    } else {
        Some(assistant_text_frame(text))
    };
    let tool_frame = tool_former.and_then(|t| assistant_tool_use_frame_from_persisted(t, bubble));

    if text_frame.is_none() && tool_frame.is_none() {
        // Empty text + no tool = either a placeholder row (pre-Cursor-3 turns
        // that ran no tools) or a thinking-only bubble. Both match the live
        // adapter's skip filter for assistant-thinking-only frames.
        return BubbleOut::Skip;
    }
    BubbleOut::Assistant {
        text_frame,
        tool_frame,
    }
}

fn assistant_text_frame(text: &str) -> String {
    let envelope = serde_json::json!({
        "type": "assistant",
        "message": {
            "content": [{"type": "text", "text": text}],
            // Zeros: the persisted bubble has no per-message usage, and iOS
            // reads `chats.context_tokens` for the live counter anyway.
            // Matches the live cursor adapter's `normalize_assistant_frame`.
            "usage": {
                "input_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "output_tokens": 0,
            },
        },
    });
    envelope.to_string()
}

/// Build a claude-shape `tool_use` frame from a persisted `toolFormerData`
/// payload. Dispatch is on `toolFormerData.name` (NOT the `tool` int — the
/// int drifts between Cursor releases). `rawArgs` is a JSON **string** in
/// the persisted shape, so we `from_str` before re-keying. Falls back to
/// the raw payload + bubbleId-derived call id if `rawArgs` is missing or
/// unparseable.
fn assistant_tool_use_frame_from_persisted(
    tool_former: &serde_json::Value,
    bubble: &serde_json::Value,
) -> Option<String> {
    let name = tool_former
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if name.is_empty() {
        return None;
    }
    let call_id = tool_former
        .get("toolCallId")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .or_else(|| {
            bubble
                .get("bubbleId")
                .and_then(|v| v.as_str())
                .map(|s| format!("cursor_persisted_{s}"))
        })
        .unwrap_or_else(|| "cursor_persisted_unknown".to_string());

    let raw_args: serde_json::Value = tool_former
        .get("rawArgs")
        .and_then(|v| v.as_str())
        .and_then(|s| serde_json::from_str(s).ok())
        .unwrap_or(serde_json::Value::Object(Default::default()));

    let (claude_name, input) = map_cursor_persisted_tool(name, &raw_args);
    let envelope = serde_json::json!({
        "type": "assistant",
        "message": {
            "content": [{
                "type": "tool_use",
                "id": call_id,
                "name": claude_name,
                "input": input,
            }],
            "usage": {
                "input_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "output_tokens": 0,
            },
        },
    });
    Some(envelope.to_string())
}

/// Map a persisted `toolFormerData.name` string to a claude tool name + a
/// claude-shape input object. Separate from `map_cursor_tool` because the
/// persisted shape uses tool *names* (`run_terminal_cmd`, `read_file_v2`,
/// ...), while the live wire uses *verb keys* (`shellToolCall`, ...). iOS's
/// `SpawnerMessageDescriber` dispatches on claude names and reads claude-
/// named arg keys, so we rename here on the spawner side. `_v2` suffix is
/// stripped for dispatch but the original key names in `rawArgs` (where
/// present) are still passed through unchanged for sibling fields — iOS
/// doesn't read them but keeping the wire faithful helps future tooling.
fn map_cursor_persisted_tool(
    name: &str,
    raw_args: &serde_json::Value,
) -> (String, serde_json::Value) {
    // Strip `_v2` for dispatch; the inner args layout doesn't change between
    // v1 and v2 for our purposes.
    let base = name.strip_suffix("_v2").unwrap_or(name);
    match base {
        "run_terminal_cmd" | "run_terminal_command" => ("Bash".to_string(), raw_args.clone()),
        "read_file" => (
            "Read".to_string(),
            rename_key(raw_args, "target_file", "file_path"),
        ),
        "edit_file" | "search_replace" | "apply_patch" => (
            "Edit".to_string(),
            rename_either_key(
                raw_args,
                &["target_file", "relativeWorkspacePath"],
                "file_path",
            ),
        ),
        "write" => (
            "Write".to_string(),
            rename_either_key(
                raw_args,
                &["target_file", "relativeWorkspacePath"],
                "file_path",
            ),
        ),
        "grep" | "grep_search" | "ripgrep_raw_search" => (
            "Grep".to_string(),
            rename_either_key(raw_args, &["query", "pattern"], "pattern"),
        ),
        "glob_file_search" => (
            "Glob".to_string(),
            rename_either_key(raw_args, &["glob_pattern", "pattern"], "pattern"),
        ),
        "todo_write" => ("TodoWrite".to_string(), raw_args.clone()),
        // Pass-throughs: iOS falls back to the tool-name-only branch in
        // `SpawnerMessageDescriber.toolSummary`, which renders fine.
        // Includes `delete_file`, `list_dir`, `mcp_*`, `web_search`,
        // `codebase_search`, and anything we haven't seen yet.
        _ => (base.to_string(), raw_args.clone()),
    }
}

/// Clone `args` (if it's an object) and rename `from` → `to` in place.
/// Non-object args pass through unchanged.
fn rename_key(args: &serde_json::Value, from: &str, to: &str) -> serde_json::Value {
    let Some(map) = args.as_object() else {
        return args.clone();
    };
    let mut out = map.clone();
    if let Some(v) = out.remove(from) {
        out.insert(to.to_string(), v);
    }
    serde_json::Value::Object(out)
}

/// Like `rename_key` but tries each `from_candidates` in order — the first
/// one that's present wins. Useful when Cursor uses different arg key names
/// across versions of the same tool (e.g. `target_file` vs
/// `relativeWorkspacePath`).
fn rename_either_key(
    args: &serde_json::Value,
    from_candidates: &[&str],
    to: &str,
) -> serde_json::Value {
    let Some(map) = args.as_object() else {
        return args.clone();
    };
    let mut out = map.clone();
    for from in from_candidates {
        if let Some(v) = out.remove(*from) {
            out.insert(to.to_string(), v);
            break;
        }
    }
    serde_json::Value::Object(out)
}

/// User-content strings that the harness would treat as synthetic. No
/// observed cases in cursor bubbles (Cursor isn't claude code), but the
/// check is cheap and matches the claude importer's posture; rename-only
/// wrapper around the claude prefix list so a future centralization is one
/// edit. The `super_` prefix is just because `is_synthetic_wrapper` is a
/// private fn in `adapters/claude.rs` and this module doesn't see it.
fn super_is_synthetic_wrapper(s: &str) -> bool {
    const PREFIXES: &[&str] = &[
        "<local-command-caveat>",
        "<local-command-stdout>",
        "<local-command-stderr>",
        "<command-name>",
        "<system-reminder>",
        "<task-notification>",
    ];
    PREFIXES.iter().any(|p| s.starts_with(p))
}

/// Cursor stores `createdAt` as ms since unix epoch. Saturating cast: the
/// real range of values is 2023..2030-ish, well within i64. The fallback
/// is `epoch` itself so a malformed/zero header still produces a valid
/// timestamp (the chat just sorts to the top).
fn epoch_ms_to_utc(ms: i64) -> DateTime<Utc> {
    Utc.timestamp_millis_opt(ms)
        .single()
        .unwrap_or_else(|| Utc.timestamp_opt(0, 0).unwrap())
}

/// `fn`-pointer-shaped wrapper around `import()` for `AdapterDescriptor.import`.
/// Same boxing trick as `probe_boxed` above — erases the async-fn return type
/// so all adapters' imports share one `fn` signature in the descriptor slice.
fn import_boxed(
    machine_id: Uuid,
    user_id: Uuid,
    write_tx: mpsc::Sender<WriteEvent>,
    progress: ImportProgress,
) -> futures::future::BoxFuture<'static, anyhow::Result<()>> {
    Box::pin(import(machine_id, user_id, write_tx, progress))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(a: &mut CursorAdapter, line: &str) -> Vec<String> {
        a.handle_line(line.to_string())
            .into_iter()
            .map(|e| match e {
                AgentEvent::Frame(s) => format!("Frame({})", s),
                AgentEvent::ContextTokens(n) => format!("ContextTokens({})", n),
                AgentEvent::CompactBoundary(n) => format!("CompactBoundary({})", n),
                AgentEvent::SessionIdHarvested(s) => format!("SessionIdHarvested({})", s),
                AgentEvent::Result => "Result".to_string(),
            })
            .collect()
    }

    #[test]
    fn init_frame_harvests_session_id() {
        let mut a = CursorAdapter::new();
        let line = r#"{"type":"system","subtype":"init","apiKeySource":"login","cwd":"/tmp","session_id":"abc-1","model":"Composer 2.5 Fast","permissionMode":"default"}"#;
        assert_eq!(run(&mut a, line), vec!["SessionIdHarvested(abc-1)"]);
    }

    #[test]
    fn user_frame_dropped() {
        let mut a = CursorAdapter::new();
        let line = r#"{"type":"user","message":{"role":"user","content":[{"type":"text","text":"hi"}]},"session_id":"abc-1"}"#;
        assert!(run(&mut a, line).is_empty());
    }

    #[test]
    fn assistant_text_normalized_to_claude_shape() {
        let mut a = CursorAdapter::new();
        let line = r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"Hi"}]},"session_id":"abc-1"}"#;
        let events = run(&mut a, line);
        assert_eq!(events.len(), 1);
        let Some(rest) = events[0].strip_prefix("Frame(") else {
            panic!("not Frame")
        };
        let body = rest.strip_suffix(')').unwrap();
        let v: Value = serde_json::from_str(body).unwrap();
        assert_eq!(v["type"], "assistant");
        assert_eq!(v["message"]["content"][0]["type"], "text");
        assert_eq!(v["message"]["content"][0]["text"], "Hi");
        assert_eq!(v["message"]["usage"]["input_tokens"], 0);
    }

    #[test]
    fn tool_call_completed_normalized_to_tool_use_block() {
        let mut a = CursorAdapter::new();
        let line = r#"{"type":"tool_call","subtype":"completed","call_id":"tool_xx","tool_call":{"shellToolCall":{"args":{"command":"ls"},"result":{"success":{"stdout":"a"}}}},"session_id":"abc-1"}"#;
        let events = run(&mut a, line);
        assert_eq!(events.len(), 1);
        let body = events[0]
            .strip_prefix("Frame(")
            .unwrap()
            .strip_suffix(')')
            .unwrap();
        let v: Value = serde_json::from_str(body).unwrap();
        assert_eq!(v["type"], "assistant");
        let block = &v["message"]["content"][0];
        assert_eq!(block["type"], "tool_use");
        assert_eq!(block["id"], "tool_xx");
        assert_eq!(block["name"], "Bash");
        assert_eq!(block["input"]["command"], "ls");
    }

    #[test]
    fn cursor_read_tool_renamed_to_claude_shape() {
        let mut a = CursorAdapter::new();
        let line = r#"{"type":"tool_call","subtype":"completed","call_id":"r1","tool_call":{"readToolCall":{"args":{"path":"/etc/hosts","offset":0},"result":{}}}}"#;
        let events = run(&mut a, line);
        assert_eq!(events.len(), 1);
        let body = events[0]
            .strip_prefix("Frame(")
            .unwrap()
            .strip_suffix(')')
            .unwrap();
        let v: Value = serde_json::from_str(body).unwrap();
        let block = &v["message"]["content"][0];
        assert_eq!(block["name"], "Read");
        assert_eq!(block["input"]["file_path"], "/etc/hosts");
        assert!(block["input"].get("path").is_none());
        assert_eq!(block["input"]["offset"], 0);
    }

    #[test]
    fn cursor_edit_tool_renamed_to_claude_shape() {
        let mut a = CursorAdapter::new();
        let line = r#"{"type":"tool_call","subtype":"completed","call_id":"e1","tool_call":{"editToolCall":{"args":{"path":"/tmp/x","streamContent":"hello"},"result":{}}}}"#;
        let events = run(&mut a, line);
        assert_eq!(events.len(), 1);
        let body = events[0]
            .strip_prefix("Frame(")
            .unwrap()
            .strip_suffix(')')
            .unwrap();
        let v: Value = serde_json::from_str(body).unwrap();
        let block = &v["message"]["content"][0];
        assert_eq!(block["name"], "Edit");
        assert_eq!(block["input"]["file_path"], "/tmp/x");
        assert!(block["input"].get("path").is_none());
        assert_eq!(block["input"]["streamContent"], "hello");
    }

    #[test]
    fn cursor_write_tool_renamed_to_claude_shape() {
        let mut a = CursorAdapter::new();
        let line = r#"{"type":"tool_call","subtype":"completed","call_id":"w1","tool_call":{"writeToolCall":{"args":{"path":"/tmp/y"},"result":{}}}}"#;
        let events = run(&mut a, line);
        assert_eq!(events.len(), 1);
        let body = events[0]
            .strip_prefix("Frame(")
            .unwrap()
            .strip_suffix(')')
            .unwrap();
        let v: Value = serde_json::from_str(body).unwrap();
        let block = &v["message"]["content"][0];
        assert_eq!(block["name"], "Write");
        assert_eq!(block["input"]["file_path"], "/tmp/y");
        assert!(block["input"].get("path").is_none());
    }

    #[test]
    fn cursor_unknown_tool_passes_through_bare() {
        let mut a = CursorAdapter::new();
        let line = r#"{"type":"tool_call","subtype":"completed","call_id":"u1","tool_call":{"fooBarToolCall":{"args":{"x":1},"result":{}}}}"#;
        let events = run(&mut a, line);
        assert_eq!(events.len(), 1);
        let body = events[0]
            .strip_prefix("Frame(")
            .unwrap()
            .strip_suffix(')')
            .unwrap();
        let v: Value = serde_json::from_str(body).unwrap();
        let block = &v["message"]["content"][0];
        assert_eq!(block["name"], "fooBar");
        assert_eq!(block["input"]["x"], 1);
    }

    #[test]
    fn tool_call_completed_picks_verb_by_suffix_not_alpha_order() {
        // Regression guard: `serde_json::Map` is a `BTreeMap` (no
        // `preserve_order` feature), so keys iterate alphabetically. A
        // sibling key that sorts before `shellToolCall` (e.g. `_meta`)
        // must NOT win — we pick the entry whose key ends with `ToolCall`.
        let mut a = CursorAdapter::new();
        let line = r#"{"type":"tool_call","subtype":"completed","call_id":"tool_yy","tool_call":{"_meta":{"t":0},"shellToolCall":{"args":{"command":"ls"},"result":{}}}}"#;
        let events = run(&mut a, line);
        assert_eq!(events.len(), 1);
        let body = events[0]
            .strip_prefix("Frame(")
            .unwrap()
            .strip_suffix(')')
            .unwrap();
        let v: Value = serde_json::from_str(body).unwrap();
        let block = &v["message"]["content"][0];
        assert_eq!(block["name"], "Bash");
        assert_eq!(block["input"]["command"], "ls");
    }

    #[test]
    fn oversize_tool_call_completed_synthesizes_minimal_envelope() {
        // Frames over MAX_STREAM_FRAME_BYTES must not trigger a full serde_json
        // parse. Build a line whose `result` payload pushes well past the
        // cap; the substring sniffer should still recover call_id + verb
        // and emit a minimal tool_use bubble.
        let mut a = CursorAdapter::new();
        let big = "x".repeat(MAX_STREAM_FRAME_BYTES + 1024);
        let line = format!(
            r#"{{"type":"tool_call","subtype":"completed","call_id":"big_1","tool_call":{{"editToolCall":{{"args":{{"path":"/tmp/x"}},"result":{{"streamContent":"{}"}}}}}}}}"#,
            big
        );
        assert!(line.len() > MAX_STREAM_FRAME_BYTES);
        let events = run(&mut a, &line);
        assert_eq!(events.len(), 1);
        let body = events[0]
            .strip_prefix("Frame(")
            .unwrap()
            .strip_suffix(')')
            .unwrap();
        let v: Value = serde_json::from_str(body).unwrap();
        let block = &v["message"]["content"][0];
        assert_eq!(block["type"], "tool_use");
        assert_eq!(block["id"], "big_1");
        assert_eq!(block["name"], "Edit");
        // Oversize path emits empty input — iOS falls back to tool-name-only.
        assert!(block["input"].as_object().unwrap().is_empty());
    }

    #[test]
    fn oversize_non_tool_call_frame_dropped() {
        let mut a = CursorAdapter::new();
        // A pathological huge `assistant` frame — drop without parse.
        let big = "x".repeat(MAX_STREAM_FRAME_BYTES + 1024);
        let line = format!(
            r#"{{"type":"assistant","message":{{"content":[{{"type":"text","text":"{}"}}]}}}}"#,
            big
        );
        assert!(run(&mut a, &line).is_empty());
    }

    #[test]
    fn tool_call_started_dropped() {
        let mut a = CursorAdapter::new();
        let line = r#"{"type":"tool_call","subtype":"started","call_id":"tool_xx","tool_call":{"shellToolCall":{"args":{"command":"ls"}}},"session_id":"abc-1"}"#;
        assert!(run(&mut a, line).is_empty());
    }

    #[test]
    fn thinking_dropped() {
        let mut a = CursorAdapter::new();
        let delta =
            r#"{"type":"thinking","subtype":"delta","text":"plan...","session_id":"abc-1"}"#;
        let done =
            r#"{"type":"thinking","subtype":"completed","text":"plan done","session_id":"abc-1"}"#;
        assert!(run(&mut a, delta).is_empty());
        assert!(run(&mut a, done).is_empty());
    }

    #[test]
    fn result_success_emits_tokens_frame_and_marker() {
        let mut a = CursorAdapter::new();
        let line = r#"{"type":"result","subtype":"success","duration_ms":7787,"is_error":false,"usage":{"inputTokens":22446,"outputTokens":44,"cacheReadTokens":2432,"cacheWriteTokens":0}}"#;
        let events = run(&mut a, line);
        // ContextTokens, Frame, Result. No prior assistant/tool_call frames
        // were observed, so N_calls falls back to 1 and the emission equals
        // the raw sum (22446 + 2432 + 0 = 24878).
        assert_eq!(events.len(), 3);
        assert_eq!(events[0], "ContextTokens(24878)");
        assert!(events[1].starts_with("Frame("));
        assert_eq!(events[2], "Result");

        // Adapter emits Result on every result line; the supervisor (not the
        // adapter) latches it so AgentResponse::Done.has_result is set once.
        // Per-turn state is reset on the first Result, so the second line
        // re-runs through the same N_calls=1 fallback path.
        let again = run(&mut a, line);
        assert_eq!(again.len(), 3);
        assert_eq!(again[0], "ContextTokens(24878)");
        assert!(again[1].starts_with("Frame("));
        assert_eq!(again[2], "Result");
    }

    #[test]
    fn result_context_tokens_divides_by_observed_llm_call_count() {
        // Reproduces the spawn shape captured live in
        // `tmp/cursor_frames2.ndjson`: 5 distinct model_call_ids on
        // assistant + tool_call.completed frames, plus a final assistant
        // frame with no model_call_id (the summary). N_calls = 5 + 1 = 6.
        // The result.usage from that turn was
        //   inputTokens=101650, cacheReadTokens=200160, cacheWriteTokens=0
        // and the rescaled per-call context = (101650+200160+0)/6 = 50301.
        let mut a = CursorAdapter::new();

        // Five LLM calls. Each call gets one of: assistant text, or a
        // tool_call.completed — both branches must populate the id set.
        let assistant_with_id = |id: &str| {
            format!(
                r#"{{"type":"assistant","message":{{"role":"assistant","content":[{{"type":"text","text":"step"}}]}},"model_call_id":"{}"}}"#,
                id
            )
        };
        let tool_completed_with_id = |id: &str, call_id: &str| {
            format!(
                r#"{{"type":"tool_call","subtype":"completed","call_id":"{}","tool_call":{{"shellToolCall":{{"args":{{"command":"ls"}},"result":{{}}}}}},"model_call_id":"{}"}}"#,
                call_id, id
            )
        };

        // Call 1: assistant text.
        let _ = run(&mut a, &assistant_with_id("req-0-aaa"));
        // Call 2: tool_call (no assistant text frame this call).
        let _ = run(&mut a, &tool_completed_with_id("req-1-bbb", "t1"));
        // Call 3: assistant text + tool_call share an id — must dedupe.
        let _ = run(&mut a, &assistant_with_id("req-2-ccc"));
        let _ = run(&mut a, &tool_completed_with_id("req-2-ccc", "t2"));
        // Call 4 + 5: tool_call only.
        let _ = run(&mut a, &tool_completed_with_id("req-3-ddd", "t3"));
        let _ = run(&mut a, &tool_completed_with_id("req-4-eee", "t4"));
        // Call 6: final summary assistant frame WITHOUT model_call_id.
        let line_no_id = r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"summary"}]}}"#;
        let _ = run(&mut a, line_no_id);

        // Result frame with the live-captured cumulative usage.
        let result_line = r#"{"type":"result","subtype":"success","duration_ms":31069,"is_error":false,"usage":{"inputTokens":101650,"outputTokens":1859,"cacheReadTokens":200160,"cacheWriteTokens":0}}"#;
        let events = run(&mut a, result_line);
        assert_eq!(events.len(), 3);
        // (101650 + 200160 + 0) / 6 = 50301
        assert_eq!(events[0], "ContextTokens(50301)");
        assert!(events[1].starts_with("Frame("));
        assert_eq!(events[2], "Result");

        // State must reset after Result: a follow-up result with no
        // intervening frames should fall back to N_calls=1.
        let again = run(&mut a, result_line);
        assert_eq!(again.len(), 3);
        assert_eq!(again[0], "ContextTokens(301810)"); // raw sum
    }

    #[test]
    fn result_without_usage_skips_context_tokens() {
        let mut a = CursorAdapter::new();
        let line = r#"{"type":"result","subtype":"error","duration_ms":12,"is_error":true}"#;
        let events = run(&mut a, line);
        assert_eq!(events.len(), 2);
        assert!(events[0].starts_with("Frame("));
        assert_eq!(events[1], "Result");
    }

    #[test]
    fn assistant_without_text_or_tool_use_dropped() {
        let mut a = CursorAdapter::new();
        // e.g. a hypothetical thinking-only assistant frame from cursor.
        let line = r#"{"type":"assistant","message":{"content":[{"type":"thinking","thinking":"..."}]},"session_id":"abc-1"}"#;
        assert!(run(&mut a, line).is_empty());
    }

    #[test]
    fn worktree_true_passes_deterministic_name() {
        use crate::adapter::TurnContext;
        use std::path::PathBuf;
        let mut a = CursorAdapter::new();
        let prompt_file = PathBuf::from("/tmp/p.txt");
        let ctx = TurnContext {
            chat_id: "abcdef012345-6789-...",
            prompt_file: &prompt_file,
            project_path: Some("/tmp/proj"),
            worktree: true,
            agent_session_id: None,
            is_sandboxed: false,
            model: None,
        };
        let cmd = a.prepare_command(&ctx).unwrap();
        assert!(
            cmd.contains("--worktree 'zcm-abcdef012345'"),
            "got: {}",
            cmd
        );
    }

    #[test]
    fn worktree_true_also_works_with_resume() {
        use crate::adapter::TurnContext;
        use std::path::PathBuf;
        let mut a = CursorAdapter::new();
        let prompt_file = PathBuf::from("/tmp/p.txt");
        let ctx = TurnContext {
            chat_id: "abcdef012345-6789-...",
            prompt_file: &prompt_file,
            project_path: Some("/tmp/proj"),
            worktree: true,
            agent_session_id: Some("sess-1"),
            is_sandboxed: false,
            model: None,
        };
        let cmd = a.prepare_command(&ctx).unwrap();
        assert!(cmd.contains("--resume 'sess-1'"), "got: {}", cmd);
        assert!(
            cmd.contains("--worktree 'zcm-abcdef012345'"),
            "got: {}",
            cmd
        );
    }

    #[test]
    fn sandboxed_omits_force_flag() {
        use crate::adapter::TurnContext;
        use std::path::PathBuf;
        let mut a = CursorAdapter::new();
        let prompt_file = PathBuf::from("/tmp/p.txt");
        let ctx = TurnContext {
            chat_id: "abcdef012345-6789-...",
            prompt_file: &prompt_file,
            project_path: Some("/tmp/proj"),
            worktree: false,
            agent_session_id: None,
            is_sandboxed: true,
            model: None,
        };
        let cmd = a.prepare_command(&ctx).unwrap();
        assert!(!cmd.contains("--force"), "got: {}", cmd);
        // --trust and --approve-mcps still required for headless to function.
        assert!(cmd.contains("--trust"), "got: {}", cmd);
        assert!(cmd.contains("--approve-mcps"), "got: {}", cmd);
    }

    #[test]
    fn non_sandboxed_includes_force_flag() {
        use crate::adapter::TurnContext;
        use std::path::PathBuf;
        let mut a = CursorAdapter::new();
        let prompt_file = PathBuf::from("/tmp/p.txt");
        let ctx = TurnContext {
            chat_id: "abcdef012345-6789-...",
            prompt_file: &prompt_file,
            project_path: Some("/tmp/proj"),
            worktree: false,
            agent_session_id: None,
            is_sandboxed: false,
            model: None,
        };
        let cmd = a.prepare_command(&ctx).unwrap();
        assert!(cmd.contains("--force"), "got: {}", cmd);
    }

    // ===== Importer tests =====

    use serde_json::json;

    #[test]
    fn map_persisted_run_terminal_cmd_to_bash() {
        let args = json!({"command": "ls", "is_background": false});
        let (name, input) = map_cursor_persisted_tool("run_terminal_cmd", &args);
        assert_eq!(name, "Bash");
        assert_eq!(input["command"], "ls");
        assert_eq!(input["is_background"], false);
    }

    #[test]
    fn map_persisted_v2_suffix_stripped() {
        let args = json!({"target_file": "/tmp/x"});
        let (name, input) = map_cursor_persisted_tool("read_file_v2", &args);
        assert_eq!(name, "Read");
        assert_eq!(input["file_path"], "/tmp/x");
        assert!(input.get("target_file").is_none());
    }

    #[test]
    fn map_persisted_read_file_renames_target_file() {
        let args = json!({"target_file": "/etc/hosts", "limit": 200});
        let (name, input) = map_cursor_persisted_tool("read_file", &args);
        assert_eq!(name, "Read");
        assert_eq!(input["file_path"], "/etc/hosts");
        assert!(input.get("target_file").is_none());
        assert_eq!(input["limit"], 200);
    }

    #[test]
    fn map_persisted_edit_file_renames_target_file() {
        let args = json!({"target_file": "/tmp/y"});
        let (name, input) = map_cursor_persisted_tool("edit_file", &args);
        assert_eq!(name, "Edit");
        assert_eq!(input["file_path"], "/tmp/y");
        assert!(input.get("target_file").is_none());
    }

    #[test]
    fn map_persisted_search_replace_is_edit() {
        let args = json!({"target_file": "/tmp/z"});
        let (name, input) = map_cursor_persisted_tool("search_replace", &args);
        assert_eq!(name, "Edit");
        assert_eq!(input["file_path"], "/tmp/z");
    }

    #[test]
    fn map_persisted_apply_patch_is_edit() {
        let args = json!({"target_file": "/tmp/w"});
        let (name, input) = map_cursor_persisted_tool("apply_patch", &args);
        assert_eq!(name, "Edit");
        assert_eq!(input["file_path"], "/tmp/w");
    }

    #[test]
    fn map_persisted_write_renames_target_file() {
        let args = json!({"target_file": "/tmp/n"});
        let (name, input) = map_cursor_persisted_tool("write", &args);
        assert_eq!(name, "Write");
        assert_eq!(input["file_path"], "/tmp/n");
    }

    #[test]
    fn map_persisted_grep_search_to_grep() {
        let args = json!({"query": "foo"});
        let (name, input) = map_cursor_persisted_tool("grep_search", &args);
        assert_eq!(name, "Grep");
        assert_eq!(input["pattern"], "foo");
        assert!(input.get("query").is_none());
    }

    #[test]
    fn map_persisted_glob_file_search_to_glob() {
        let args = json!({"glob_pattern": "**/*.rs"});
        let (name, input) = map_cursor_persisted_tool("glob_file_search", &args);
        assert_eq!(name, "Glob");
        assert_eq!(input["pattern"], "**/*.rs");
        assert!(input.get("glob_pattern").is_none());
    }

    #[test]
    fn map_persisted_mcp_tool_passes_through() {
        let args = json!({"foo": 1});
        let (name, input) = map_cursor_persisted_tool("mcp_anything", &args);
        assert_eq!(name, "mcp_anything");
        assert_eq!(input["foo"], 1);
    }

    #[test]
    fn raw_args_json_string_parsed_before_rekeying() {
        // Build a synthetic toolFormerData with rawArgs as a JSON string
        // (the persisted shape). After parsing, the renamer must see
        // `target_file` and rewrite it to `file_path`.
        let tool_former = json!({
            "name": "read_file",
            "toolCallId": "tc1",
            "rawArgs": "{\"target_file\":\"x\"}",
            "status": "completed",
        });
        let bubble = json!({"bubbleId": "b1"});
        let frame = assistant_tool_use_frame_from_persisted(&tool_former, &bubble).unwrap();
        let v: serde_json::Value = serde_json::from_str(&frame).unwrap();
        let block = &v["message"]["content"][0];
        assert_eq!(block["type"], "tool_use");
        assert_eq!(block["id"], "tc1");
        assert_eq!(block["name"], "Read");
        assert_eq!(block["input"]["file_path"], "x");
        assert!(block["input"].get("target_file").is_none());
    }

    #[test]
    fn classify_user_bubble_returns_text() {
        let bubble = json!({"type": 1, "text": "Hello there"});
        match classify_bubble(&bubble) {
            BubbleOut::User { text } => assert_eq!(text, "Hello there"),
            _ => panic!("expected User"),
        }
    }

    #[test]
    fn classify_user_bubble_empty_text_skipped() {
        let bubble = json!({"type": 1, "text": ""});
        assert!(matches!(classify_bubble(&bubble), BubbleOut::Skip));
    }

    #[test]
    fn classify_user_bubble_system_reminder_skipped() {
        let bubble = json!({"type": 1, "text": "<system-reminder>foo</system-reminder>"});
        assert!(matches!(classify_bubble(&bubble), BubbleOut::Skip));
    }

    #[test]
    fn classify_assistant_text_and_tool_emits_two_frames() {
        let bubble = json!({
            "type": 2,
            "text": "Sure, running it.",
            "bubbleId": "b1",
            "toolFormerData": {
                "name": "run_terminal_cmd",
                "toolCallId": "tc1",
                "rawArgs": "{\"command\":\"ls\"}",
                "status": "completed",
            },
        });
        let out = classify_bubble(&bubble);
        let BubbleOut::Assistant {
            text_frame,
            tool_frame,
        } = out
        else {
            panic!("expected Assistant");
        };
        let text = text_frame.expect("text frame present");
        let tool = tool_frame.expect("tool frame present");
        let t: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(t["message"]["content"][0]["type"], "text");
        assert_eq!(t["message"]["content"][0]["text"], "Sure, running it.");
        let u: serde_json::Value = serde_json::from_str(&tool).unwrap();
        assert_eq!(u["message"]["content"][0]["type"], "tool_use");
        assert_eq!(u["message"]["content"][0]["name"], "Bash");
        assert_eq!(u["message"]["content"][0]["input"]["command"], "ls");
    }

    #[test]
    fn classify_assistant_empty_text_no_tool_skipped() {
        let bubble = json!({"type": 2, "text": ""});
        assert!(matches!(classify_bubble(&bubble), BubbleOut::Skip));
    }

    #[test]
    fn classify_assistant_thinking_only_skipped() {
        // Cursor doesn't persist `thinking` separately from `text` the way
        // claude does — empty `text` + no tool is the closest analog of a
        // thinking-only frame, and we already skip those.
        let bubble = json!({"type": 2, "text": "", "thinking": {"text": "planning..."}});
        assert!(matches!(classify_bubble(&bubble), BubbleOut::Skip));
    }

    #[test]
    fn classify_unknown_type_skipped() {
        let bubble = json!({"type": 99, "text": "weird"});
        assert!(matches!(classify_bubble(&bubble), BubbleOut::Skip));
    }

    #[test]
    fn accept_header_drops_draft_best_of_n_and_subagent() {
        let draft = json!({"composerId": "x", "isDraft": true});
        let bestof = json!({"composerId": "y", "isBestOfNSubcomposer": true});
        // Real-data shape (subagentTypeName:"explore" with isBestOfNSubcomposer:false)
        // — these DO appear in allComposers so the membership filter doesn't catch
        // them; gate must trigger on `subagentInfo` presence directly.
        let subagent = json!({
            "composerId": "s",
            "isBestOfNSubcomposer": false,
            "subagentInfo": {"subagentTypeName": "explore", "parentComposerId": "p"},
        });
        let normal = json!({"composerId": "z"});
        assert!(!accept_header(&draft));
        assert!(!accept_header(&bestof));
        assert!(!accept_header(&subagent));
        assert!(accept_header(&normal));
    }

    /// End-to-end with an in-memory SQLite db: insert a composerHeaders
    /// row + a couple of composerData / bubbleId entries, run import, and
    /// drain the writer channel. Verifies project/chat/message ordering
    /// and that the `allComposers` allow-list correctly filters orphan
    /// composers (rows present in cursorDiskKV but absent from headers).
    #[tokio::test]
    async fn end_to_end_in_memory_db() {
        let db_path = {
            let mut p = std::env::temp_dir();
            p.push(format!(
                "zucchini_cursor_import_test_{}.sqlite",
                std::process::id()
            ));
            // Ensure clean.
            let _ = std::fs::remove_file(&p);
            p
        };
        // Build a fixture db on disk (we test the on-disk path).
        {
            let conn = rusqlite::Connection::open(&db_path).unwrap();
            conn.execute_batch(
                "CREATE TABLE ItemTable (key TEXT UNIQUE, value BLOB);
                 CREATE TABLE cursorDiskKV (key TEXT UNIQUE, value BLOB);",
            )
            .unwrap();
            // Two composers — one with a path, one without. A third row
            // exists in cursorDiskKV but NOT in the headers allow-list and
            // must be dropped.
            let headers = json!({
                "allComposers": [
                    {
                        "composerId": "11111111-1111-4111-8111-111111111111",
                        "name": "First chat",
                        "createdAt": 1_700_000_000_000_i64,
                        "workspaceIdentifier": {"uri": {"path": "/tmp/proj-a"}},
                    },
                    {
                        "composerId": "22222222-2222-4222-8222-222222222222",
                        "name": "Second chat (no path)",
                        "createdAt": 1_700_000_001_000_i64,
                        "workspaceIdentifier": {"id": "opaque"}
                    },
                    {
                        "composerId": "33333333-3333-4333-8333-333333333333",
                        "isDraft": true,
                        "createdAt": 1_700_000_002_000_i64,
                    }
                ]
            });
            conn.execute(
                "INSERT INTO ItemTable(key, value) VALUES('composer.composerHeaders', ?)",
                rusqlite::params![serde_json::to_vec(&headers).unwrap()],
            )
            .unwrap();
            // composer 1: 1 user + 1 assistant text + 1 assistant tool
            let composer1 = json!({
                "composerId": "11111111-1111-4111-8111-111111111111",
                "name": "First chat",
                "createdAt": 1_700_000_000_000_i64,
                "fullConversationHeadersOnly": [
                    {"bubbleId": "aaaaaaaa-1111-4111-8111-111111111111", "type": 1},
                    {"bubbleId": "bbbbbbbb-1111-4111-8111-111111111111", "type": 2},
                    {"bubbleId": "cccccccc-1111-4111-8111-111111111111", "type": 2},
                ]
            });
            conn.execute(
                "INSERT INTO cursorDiskKV(key, value) VALUES(?, ?)",
                rusqlite::params![
                    "composerData:11111111-1111-4111-8111-111111111111",
                    serde_json::to_vec(&composer1).unwrap()
                ],
            )
            .unwrap();
            let bubble_user = json!({"type": 1, "text": "hello"});
            conn.execute(
                "INSERT INTO cursorDiskKV(key, value) VALUES(?, ?)",
                rusqlite::params![
                    "bubbleId:11111111-1111-4111-8111-111111111111:aaaaaaaa-1111-4111-8111-111111111111",
                    serde_json::to_vec(&bubble_user).unwrap()
                ],
            )
            .unwrap();
            let bubble_text = json!({"type": 2, "text": "world"});
            conn.execute(
                "INSERT INTO cursorDiskKV(key, value) VALUES(?, ?)",
                rusqlite::params![
                    "bubbleId:11111111-1111-4111-8111-111111111111:bbbbbbbb-1111-4111-8111-111111111111",
                    serde_json::to_vec(&bubble_text).unwrap()
                ],
            )
            .unwrap();
            let bubble_tool = json!({
                "type": 2,
                "text": "",
                "bubbleId": "cccccccc-1111-4111-8111-111111111111",
                "toolFormerData": {
                    "name": "read_file",
                    "toolCallId": "tc1",
                    "rawArgs": "{\"target_file\":\"/tmp/x\"}",
                    "status": "completed",
                }
            });
            conn.execute(
                "INSERT INTO cursorDiskKV(key, value) VALUES(?, ?)",
                rusqlite::params![
                    "bubbleId:11111111-1111-4111-8111-111111111111:cccccccc-1111-4111-8111-111111111111",
                    serde_json::to_vec(&bubble_tool).unwrap()
                ],
            )
            .unwrap();
            // composer 2: 1 user
            let composer2 = json!({
                "composerId": "22222222-2222-4222-8222-222222222222",
                "name": "Second chat (no path)",
                "createdAt": 1_700_000_001_000_i64,
                "fullConversationHeadersOnly": [
                    {"bubbleId": "dddddddd-2222-4222-8222-222222222222", "type": 1}
                ]
            });
            conn.execute(
                "INSERT INTO cursorDiskKV(key, value) VALUES(?, ?)",
                rusqlite::params![
                    "composerData:22222222-2222-4222-8222-222222222222",
                    serde_json::to_vec(&composer2).unwrap()
                ],
            )
            .unwrap();
            let bubble_user2 = json!({"type": 1, "text": "no project here"});
            conn.execute(
                "INSERT INTO cursorDiskKV(key, value) VALUES(?, ?)",
                rusqlite::params![
                    "bubbleId:22222222-2222-4222-8222-222222222222:dddddddd-2222-4222-8222-222222222222",
                    serde_json::to_vec(&bubble_user2).unwrap()
                ],
            )
            .unwrap();
            // Orphan composer present in cursorDiskKV but absent from
            // composer.composerHeaders.allComposers — must be skipped.
            let orphan = json!({
                "composerId": "99999999-9999-4999-8999-999999999999",
                "name": "orphan subagent",
                "createdAt": 1_700_000_999_000_i64,
                "fullConversationHeadersOnly": [
                    {"bubbleId": "eeeeeeee-9999-4999-8999-999999999999", "type": 1}
                ]
            });
            conn.execute(
                "INSERT INTO cursorDiskKV(key, value) VALUES(?, ?)",
                rusqlite::params![
                    "composerData:99999999-9999-4999-8999-999999999999",
                    serde_json::to_vec(&orphan).unwrap()
                ],
            )
            .unwrap();
        }

        // Run via the read path the importer uses (same `read_state_db`).
        let scan = read_state_db(&db_path).unwrap();
        assert_eq!(scan.headers.len(), 2, "draft + orphan dropped");
        assert!(scan
            .composer_rows
            .contains_key("11111111-1111-4111-8111-111111111111"));
        assert!(scan
            .composer_rows
            .contains_key("22222222-2222-4222-8222-222222222222"));
        assert!(
            !scan
                .composer_rows
                .contains_key("99999999-9999-4999-8999-999999999999"),
            "orphan composer must not be loaded"
        );

        // Drain the writer channel via a real `import` call. Use a generous
        // channel so a slow drain doesn't backpressure the sender.
        let (tx, mut rx) = mpsc::channel::<WriteEvent>(64);
        let machine_id = Uuid::nil();
        let user_id = Uuid::nil();

        // Reroute the importer to read from our fixture db by overriding
        // HOME just for this test process. The exact path is OS-dependent
        // (macOS uses `~/Library/Application Support/Cursor/...`, Linux uses
        // `~/.config/Cursor/...`), so ask the path resolver itself where to
        // materialize the tree. Passing `xdg_config_home = None` forces the
        // HOME-based fallback on Linux so we only need to override HOME.
        let tempdir = tempfile_tempdir();
        let target_db = cursor_state_db_path_for(std::env::consts::OS, Some(&tempdir), None)
            .expect("cursor importer supports this test OS");
        std::fs::create_dir_all(target_db.parent().unwrap()).unwrap();
        std::fs::copy(&db_path, &target_db).unwrap();
        // SAFETY: tests run single-threaded per-test by default, but this is
        // a process-wide mutation. The only other test that reads HOME in
        // this crate is the claude importer, which is `ignore`d by default;
        // we set + restore here to be safe. Also clear XDG_CONFIG_HOME so the
        // Linux branch falls back to `$HOME/.config` (matches what we wrote).
        let prev_home = std::env::var_os("HOME");
        let prev_xdg = std::env::var_os("XDG_CONFIG_HOME");
        // SAFETY: env mutation is single-threaded per test process; no
        // concurrent threads read HOME / XDG_CONFIG_HOME during this test.
        unsafe {
            std::env::set_var("HOME", &tempdir);
            std::env::remove_var("XDG_CONFIG_HOME");
        }

        let result = import(machine_id, user_id, tx, Box::new(|_| {}) as ImportProgress).await;

        // Restore HOME / XDG_CONFIG_HOME before any assertions panic.
        // SAFETY: same as above — single-threaded test process.
        unsafe {
            match prev_home {
                Some(v) => std::env::set_var("HOME", v),
                None => std::env::remove_var("HOME"),
            }
            if let Some(v) = prev_xdg {
                std::env::set_var("XDG_CONFIG_HOME", v);
            }
        }
        let _ = std::fs::remove_file(&db_path);
        let _ = std::fs::remove_dir_all(&tempdir);

        result.expect("import ok");

        // Collect events.
        let mut events: Vec<WriteEvent> = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }

        // Sanity: count by variant.
        let projects: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, WriteEvent::PutProject { .. }))
            .collect();
        let chats: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, WriteEvent::PutChat { .. }))
            .collect();
        let messages: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, WriteEvent::PutMessage { .. }))
            .collect();
        assert_eq!(projects.len(), 2, "/tmp/proj-a + <no project>");
        assert_eq!(chats.len(), 2, "one per accepted composer");
        // composer1: user + assistant_text + assistant_tool = 3; composer2: 1
        assert_eq!(messages.len(), 4);

        // Project ordering: BTreeMap iterates by path, so `/tmp/proj-a`
        // (starts with '/') comes before `<no project>` (starts with '<')
        // — '/' (0x2F) < '<' (0x3C).
        let WriteEvent::PutProject {
            path: first_path, ..
        } = &events[0]
        else {
            panic!("first event should be PutProject");
        };
        assert_eq!(first_path, "/tmp/proj-a");

        // Verify PutChat precedes its PutMessage rows (FK ordering).
        let mut seen_chat: std::collections::HashSet<Uuid> = std::collections::HashSet::new();
        for ev in &events {
            match ev {
                WriteEvent::PutChat { id, .. } => {
                    seen_chat.insert(*id);
                }
                WriteEvent::PutMessage { chat_id, .. } => {
                    let cid = Uuid::parse_str(chat_id).unwrap();
                    assert!(
                        seen_chat.contains(&cid),
                        "PutMessage for chat {} arrived before PutChat",
                        chat_id
                    );
                }
                _ => {}
            }
        }
    }

    fn tempfile_tempdir() -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "zucchini_cursor_import_home_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn cursor_state_db_path_macos_uses_app_support() {
        let home = Path::new("/Users/jane");
        let p = cursor_state_db_path_for("macos", Some(home), None).unwrap();
        assert_eq!(
            p,
            Path::new(
                "/Users/jane/Library/Application Support/Cursor/User/globalStorage/state.vscdb"
            )
        );
    }

    #[test]
    fn cursor_state_db_path_macos_without_home_is_none() {
        assert!(cursor_state_db_path_for("macos", None, None).is_none());
    }

    #[test]
    fn cursor_state_db_path_linux_uses_xdg_config_home_when_set() {
        let home = Path::new("/home/jane");
        let xdg = Path::new("/custom/xdg");
        let p = cursor_state_db_path_for("linux", Some(home), Some(xdg)).unwrap();
        assert_eq!(
            p,
            Path::new("/custom/xdg/Cursor/User/globalStorage/state.vscdb")
        );
    }

    #[test]
    fn cursor_state_db_path_linux_falls_back_to_home_dot_config() {
        let home = Path::new("/home/jane");
        let p = cursor_state_db_path_for("linux", Some(home), None).unwrap();
        assert_eq!(
            p,
            Path::new("/home/jane/.config/Cursor/User/globalStorage/state.vscdb")
        );
    }

    #[test]
    fn cursor_state_db_path_linux_without_home_or_xdg_is_none() {
        assert!(cursor_state_db_path_for("linux", None, None).is_none());
    }

    #[test]
    fn cursor_state_db_path_unsupported_os_is_none() {
        // install.sh only ships Darwin + Linux, so anything else (e.g.
        // windows / freebsd) should refuse rather than guess a path.
        let home = Path::new("/home/jane");
        assert!(cursor_state_db_path_for("windows", Some(home), None).is_none());
        assert!(cursor_state_db_path_for("freebsd", Some(home), None).is_none());
    }

    #[test]
    fn worktree_false_omits_flag() {
        use crate::adapter::TurnContext;
        use std::path::PathBuf;
        let mut a = CursorAdapter::new();
        let prompt_file = PathBuf::from("/tmp/p.txt");
        let ctx = TurnContext {
            chat_id: "abcdef012345-6789-...",
            prompt_file: &prompt_file,
            project_path: Some("/tmp/proj"),
            worktree: false,
            agent_session_id: None,
            is_sandboxed: false,
            model: None,
        };
        let cmd = a.prepare_command(&ctx).unwrap();
        assert!(!cmd.contains("--worktree"), "got: {}", cmd);
    }

    #[test]
    fn model_some_appends_model_flag() {
        // `chats.model = Some("Composer 2.5 Fast")` (migration 0035) →
        // cursor command carries `--model 'Composer 2.5 Fast'` verbatim.
        // Shell escaping preserves the spaces so cursor sees one argv[].
        use crate::adapter::TurnContext;
        use std::path::PathBuf;
        let mut a = CursorAdapter::new();
        let prompt_file = PathBuf::from("/tmp/p.txt");
        let ctx = TurnContext {
            chat_id: "abcdef012345-6789-...",
            prompt_file: &prompt_file,
            project_path: Some("/tmp/proj"),
            worktree: false,
            agent_session_id: None,
            is_sandboxed: false,
            model: Some("Composer 2.5 Fast"),
        };
        let cmd = a.prepare_command(&ctx).unwrap();
        assert!(
            cmd.contains("--model 'Composer 2.5 Fast'"),
            "got: {}",
            cmd
        );
    }

    #[test]
    fn model_none_omits_model_flag() {
        // `chats.model = None` → no `--model` flag at all; cursor picks
        // the user's default from its own config.
        use crate::adapter::TurnContext;
        use std::path::PathBuf;
        let mut a = CursorAdapter::new();
        let prompt_file = PathBuf::from("/tmp/p.txt");
        let ctx = TurnContext {
            chat_id: "abcdef012345-6789-...",
            prompt_file: &prompt_file,
            project_path: Some("/tmp/proj"),
            worktree: false,
            agent_session_id: None,
            is_sandboxed: false,
            model: None,
        };
        let cmd = a.prepare_command(&ctx).unwrap();
        assert!(!cmd.contains("--model"), "got: {}", cmd);
    }
}
