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
    agent_capabilities_instructions, claude_assistant_envelope, claude_tool_use_envelope,
    current_time_in_tz_line, extract_json_type, parse_json_obj, shell_escape, AdapterDescriptor,
    AgentAdapter, AgentEvent, AgentKind, TurnContext, WorktreeInstructions, MAX_STREAM_FRAME_BYTES,
    PRUNE_CONTEXT_INSTRUCTION,
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
    // cursor-agent's LIVE CLI session is a content-addressed (git-like) Merkle
    // blob store in SQLite at `~/.cursor/chats/<projectHash>/<sessionUuid>/
    // store.db` — NOT the opaque VS Code `state.vscdb` the importer reads (that
    // earlier "infeasible" note conflated the two). Each blob is keyed by
    // `sha256(data)`; message blobs are JSON, the conversation root is a
    // protobuf whose repeated field-1 entries are the message-blob hashes in
    // order. We prune a tool result by blanking its message blob, re-minting the
    // root with that hash swapped, and re-pointing `meta.latestRootBlobId`.
    // VERIFIED ("local replay"): after this + `cursor-agent --resume`, the model
    // reports the pruned output as `[pruned]` in its own context — so cursor
    // replays from the LOCAL store, not server-canonical history. Full spec +
    // PoC: `tmp/cursor-agent-prune-plan.md`.
    prune: Some(PRUNE_OPS),
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
        // Worktree containment, computed up front so the preamble below carries
        // the "stay inside" rule. cursor-agent hardcodes the dir at
        // `~/.cursor/worktrees/<basename(project)>/<wt_name>` (see NOTE at the flag
        // site below). Wording via the shared `adapter::worktree_instructions`.
        let worktree_info = ctx.worktree.then(|| {
            let short: String = ctx.chat_id.chars().take(12).collect();
            let wt_name = format!("zcm-{}", short);
            let repo = ctx
                .project_path
                .map(|pp| pp.trim_end_matches('/').rsplit('/').next().unwrap_or(pp))
                .unwrap_or("project");
            let home = std::env::var("HOME").unwrap_or_else(|_| "~".to_string());
            let worktree_abs = format!("{home}/.cursor/worktrees/{repo}/{wt_name}");
            WorktreeInstructions {
                worktree_abs,
                parent_repo: ctx.project_path.unwrap_or(repo).to_string(),
            }
        });
        // cursor-agent has no `--append-system-prompt` (or any other system /
        // prompt-prefix flag — confirmed against `cursor-agent --help` 2026.05).
        // a short preamble before the user's prompt body. `agent_capabilities_instructions`
        // bundles the attach-file how-to, schedule-message, worktree containment,
        // and the per-turn time line; we then append the prune-context standing
        // order (cursor's selective-forgetting nudge — its reads run through
        // `Read`, so the claude-shape `PRUNE_CONTEXT_INSTRUCTION` is the right
        // variant, NOT the codex/gemini tool-name variants). The whole preamble
        // rides every turn (not just the first, unlike codex/gemini's
        // `first_turn_prompt_suffix`); cursor's `--resume` reconstructs history
        // from the local store and drops our user-echo frame, so the preamble
        // never persists and must be re-sent — hence `first_turn_prompt_suffix`
        // stays `None` and the per-turn time line lives here too.
        let preamble = format!(
            "{}\n\n{}\n\n---\n\n",
            agent_capabilities_instructions(
                worktree_info.as_ref(),
                current_time_in_tz_line(ctx.user_timezone).as_deref(),
            ),
            PRUNE_CONTEXT_INSTRUCTION
        );
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

    /// The current-local-time line is folded into the per-turn stdin preamble
    /// (`agent_capabilities_instructions` in `prepare_command`), so suppress the
    /// prompt-file prepend to avoid injecting it twice.
    fn prompt_file_time_line(&self, _ctx: &TurnContext<'_>) -> Option<String> {
        None
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

        // Oversize-frame fast-path. `serde_json::from_str` on a multi-MB line
        // allocates a tree of `Map<String, Value>` nodes; without this guard a
        // single big edit can churn the heap by megabytes. cursor frames are
        // NEVER iOS-wire-compatible (we always transform), so forwarding a raw
        // cursor line verbatim is wrong — iOS renders an unknown
        // `{"type":"X"}` as literal `[X]`. So classify CHEAPLY via
        // `extract_json_type` (no full parse) and branch per type. Mirrors the
        // pattern in `pi.rs::handle_line` / `claude.rs::handle_line`.
        if line.len() > MAX_STREAM_FRAME_BYTES {
            match extract_json_type(&line) {
                // Content-bearing, handled without a full parse: a big edit's
                // `tool_call.completed` is recovered via substring sniff (we
                // still need the `subtype` substring to confirm it's the
                // immutable variant). The synth path is the whole reason this
                // fast-path exists.
                Some("tool_call") if line.contains("\"subtype\":\"completed\"") => {
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
                // The body (the reply text) is what we render, so a >64KB
                // assistant frame needs the full parse. Rare, and correctness
                // beats the one-time heap blip — DON'T early-return; fall
                // through to the normal `"assistant"` arm below (mirrors pi's
                // `message_end => {}` fall-through).
                Some("assistant") => {}
                // CRITICAL: the `result` arm is the TERMINAL — it emits
                // ContextTokens (from usage), the result Frame, AND
                // `AgentEvent::Result` (which latches `has_result`). All three
                // are body-derived, so we must full-parse. Dropping an oversize
                // `result` means the turn never latches Result → the user sees
                // "Agent interrupted" instead of `[result: success]`. cursor
                // result frames are realistically tiny (so the heap blip
                // basically never happens), but falling through GUARANTEES
                // correctness if one is ever oversize. DON'T early-return.
                Some("result") => {}
                // Lifecycle (system, user, thinking, tool_call.started) + any
                // unknown type → DROP. Forwarding raw cursor JSON is strictly
                // worse than dropping (iOS can't render it). Matches the
                // prefilter drops above.
                Some(other) => {
                    debug!(ty = %other, "cursor-agent oversize lifecycle frame dropped");
                    return out;
                }
                // Couldn't classify (no `"type"` near the start, escaped value,
                // …). Drop: raw cursor JSON is never renderable.
                None => {
                    debug!("cursor-agent oversize frame without classifiable type dropped");
                    return out;
                }
            }
        }

        let Some(obj) = parse_json_obj(&line) else {
            // Non-JSON line (shouldn't happen on a healthy cursor-agent
            // stdout, but the line loop doesn't filter). The shared parser
            // logs the parse failure at debug; we additionally log + drop
            // here because cursor's wire protocol is strict-JSON-only.
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
                    // Call-keyed prune cue: fire ONLY when THIS completed tool is
                    // the `prune-context` shell call itself, matched on its own
                    // `tool_call.<verb>ToolCall.args` (the shellToolCall args
                    // object carrying `{"command":"…prune-context…"}`). A sibling
                    // tool's completion in the same batch must NOT drive the apply
                    // — it would abort→respawn before this call's result is
                    // persisted to the local store, so the resumed agent re-runs
                    // the prune. The command is right here in the frame, so no
                    // cross-frame state is needed. Mirrors codex's
                    // `is_prune_command` posture.
                    let is_prune_command = tool_call_args_value(&obj)
                        .is_some_and(|args| crate::prune::value_is_prune_context_call(&args));
                    if let Some(frame) = normalize_tool_call_completed(&obj) {
                        out.push(AgentEvent::Frame(frame));
                    }
                    // Emit the content-free `ToolResult` cue AFTER any visible
                    // frame (so a restart never preempts the tool's own bubble),
                    // so the main loop applies the queued prune strictly after the
                    // result landed (the mechanism claude/gemini/codex use too).
                    // No-op unless a `PruneRequest` is pending. Without this the
                    // queued prune is silently dropped at turn end — cursor has no
                    // timeout fallback in the main loop.
                    if is_prune_command {
                        out.push(AgentEvent::ToolResult);
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
                out.push(AgentEvent::Result {
                    origin_is_task: false,
                });
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
    // Usage is only known on `result.success` for cursor — the shared
    // envelope helper stamps zeros here. iOS treats per-frame usage as
    // cumulative on claude too, so emitting zeros mid-turn matches "no
    // progress yet" (the real ContextTokens event lands at end-of-turn
    // via the result frame).
    Some(claude_assistant_envelope(Value::Array(blocks)))
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
    Some(claude_tool_use_envelope(call_id, &name, input))
}

/// Extract a `tool_call.completed` frame's verb args object
/// (`tool_call.<verb>ToolCall.args`) for the call-keyed prune cue. Picks the
/// verb entry by the `…ToolCall` suffix (NOT iteration order — same hazard
/// `normalize_tool_call_completed` guards against: an alphabetically-earlier
/// sibling like `_meta` would steal `iter().next()`), then returns its `args`
/// value. `None` when the frame has no `tool_call` object, no `…ToolCall` entry,
/// or that entry carries no `args`. Used only to detect the agent's OWN
/// `prune-context` shell call via `crate::prune::value_is_prune_context_call`.
fn tool_call_args_value(obj: &Value) -> Option<Value> {
    let tool_call = obj.get("tool_call")?.as_object()?;
    let (_verb_key, verb_payload) = tool_call.iter().find(|(k, _)| k.ends_with("ToolCall"))?;
    verb_payload.get("args").cloned()
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
    Some(claude_tool_use_envelope(&call_id, &name, json!({})))
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
///   updateTodosToolCall → TodoWrite (args.todos unchanged)
///   anything else  → strip `"ToolCall"` suffix, args passthrough.
///
/// For Read/Edit/Write the `path → file_path` rename is in-place on a clone
/// of the args object; sibling keys (offset, streamContent, etc.) are
/// preserved untouched — iOS ignores them but keeping the wire faithful
/// makes the persisted body useful for future tooling.
///
/// SIBLING MAP — keep in sync with [`map_cursor_persisted_tool`]. That fn
/// maps the SAME cursor→claude tool set from the importer's vocabulary
/// (persisted tool *names* like `run_terminal_cmd`, `read_file`) instead of
/// these live verb *keys*. They are deliberately NOT merged: the input
/// vocabularies differ AND the source arg keys differ (live `path` vs
/// persisted `target_file`/`relativeWorkspacePath`), so a unified dispatch
/// would be a leaky abstraction. When you add or change a tool here, check
/// whether the persisted side needs the matching change too.
///
/// TODO PARITY: the persisted side maps `todo_write` → `TodoWrite`; the live
/// equivalent is `updateTodosToolCall` (verb key confirmed against live
/// `cursor-agent --print --output-format stream-json` output — args carry a
/// `todos` array plus a `merge` flag). Both forward args verbatim so iOS's
/// `SpawnerMessageDescriber` sees a claude-shape `TodoWrite` with `input.todos`.
fn map_cursor_tool(verb_key: &str, args: &Value) -> (String, Value) {
    match verb_key {
        "shellToolCall" => ("Bash".to_string(), args.clone()),
        "readToolCall" => ("Read".to_string(), rename_path_to_file_path(args)),
        "editToolCall" => ("Edit".to_string(), rename_path_to_file_path(args)),
        "writeToolCall" => ("Write".to_string(), rename_path_to_file_path(args)),
        "grepToolCall" => ("Grep".to_string(), args.clone()),
        "globToolCall" => ("Glob".to_string(), args.clone()),
        "updateTodosToolCall" => ("TodoWrite".to_string(), args.clone()),
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
    rename_either_key(args, &["path"], "file_path")
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
/// — the writer flattens one pair per registered kind into a single PATCH on
/// `machines`.
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
// Selective forgetting ("prune-context") for the cursor adapter.
//
// Unlike claude/gemini/codex (line-oriented JSONL files that share `prune.rs`'s
// `rewrite_jsonl_*` helpers), cursor's LIVE CLI session is a content-addressed
// (git-like) Merkle blob store in SQLite at
// `~/.cursor/chats/<projectHash>/<sessionUuid>/store.db`:
//   - `blobs(id TEXT PRIMARY KEY = sha256(data), data BLOB)` — message blobs are
//     JSON `{role, content, …}`; tree nodes (the conversation root + per-turn
//     checkpoints) are protobuf.
//   - `meta(key, value)` — key `'0'`'s value is HEX-encoded JSON text; decode →
//     `{latestRootBlobId: "<hex sha256>", …}`. `latestRootBlobId` is HEAD (the
//     only root `--resume` loads).
//   - The root protobuf's repeated field-1 entries (`0x0A 0x20` + raw 32-byte
//     hash) are the conversation's message-blob hashes in document order, and
//     the root references the message blobs DIRECTLY (flat — no intermediate
//     tree), so re-pointing = byte-replacing one (or more) 32-byte hash(es).
//
// To prune a tool result we blank its `tool` blob's `result` +
// `experimental_content`, re-mint that blob (new sha = new id), byte-replace its
// old hash in the root, re-sha the root, and move `meta.latestRootBlobId`. We
// NEVER delete blobs or rewrite older checkpoint roots — old roots stay valid so
// cursor's rewind feature keeps working. All the dialect-agnostic matchers
// (`tool_name_matches`, `value_glob_match`, `value_is_prune_context_call`,
// `args_value_is_empty`, `blank_string_field`, `PRUNED_PLACEHOLDER`) are reused
// from `prune.rs`; only the storage surgery is cursor-specific.
//
// Safety (mirrors claude/codex TOCTOU posture): no-op on ANY mismatch, never a
// partial write. Every guard failure logs a `warn!` breadcrumb (format-drift
// signal) and returns `PruneStats::default()` (0 blanked ⇒ frame skipped). All
// writes happen in ONE transaction so a mid-way failure rolls back. Spec + PoC:
// `tmp/cursor-agent-prune-plan.md`.

use sha2::{Digest, Sha256};

/// `crate::prune::PruneOps` for cursor (content-addressed store surgery). Wired
/// into the descriptor via `prune: Some(PRUNE_OPS)`.
pub(crate) const PRUNE_OPS: crate::prune::PruneOps = crate::prune::PruneOps {
    find_session: find_cursor_store,
    count_matches: count_cursor_matches,
    prune_batch: prune_batch_cursor,
};

/// Hex-encode bytes (lowercase) — sha256 ids + raw-hash hex are compared as
/// lowercase hex strings throughout.
fn hex_encode(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}

/// Decode a lowercase-hex string to bytes; `None` on any non-hex char or odd
/// length. Used to turn a blob id (hex) into the raw 32-byte hash the root
/// protobuf embeds.
fn hex_decode(s: &str) -> Option<Vec<u8>> {
    if !s.len().is_multiple_of(2) {
        return None;
    }
    let mut out = Vec::with_capacity(s.len() / 2);
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let hi = (bytes[i] as char).to_digit(16)?;
        let lo = (bytes[i + 1] as char).to_digit(16)?;
        out.push((hi * 16 + lo) as u8);
        i += 2;
    }
    Some(out)
}

/// `sha256(data)` as lowercase hex — the cursor blob id of `data`.
fn blob_id(data: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(data);
    hex_encode(&h.finalize())
}

/// `find_session(base, session_id)`: `base` = `~/.cursor`. cursor nests each
/// session's store under an opaque `<projectHash>` dir, so we glob
/// `base/chats/*/<session_id>/store.db` via `std::fs::read_dir` (no glob crate)
/// and return the first one that exists. `None` when the base/chats dir is
/// absent or no matching store.db exists yet.
fn find_cursor_store(base: &Path, session_id: &str) -> Option<PathBuf> {
    let chats = base.join("chats");
    let entries = std::fs::read_dir(&chats).ok()?;
    for entry in entries.flatten() {
        if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            continue;
        }
        let candidate = entry.path().join(session_id).join("store.db");
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// Everything read out of a cursor store needed by both the count pre-scan and
/// the apply: every blob (`id → data`), the decoded meta JSON text + the
/// `latestRootBlobId` (HEAD), and the conversation's message-blob ids in
/// DOCUMENT ORDER (decoded from the root protobuf's field-1 entries).
struct CursorStore {
    /// `blob id (hex) → raw blob bytes`.
    blobs: std::collections::HashMap<String, Vec<u8>>,
    /// `blob id (hex) → parsed JSON`, for the message blobs that ARE JSON
    /// (protobuf tree nodes — root + checkpoints — have no entry). Parsed ONCE
    /// here so the eligibility scan, the blob-target grouping, and the blank
    /// pass all read the same `Value` instead of `from_slice`-ing each blob 2-3×.
    parsed: std::collections::HashMap<String, serde_json::Value>,
    /// The decoded (un-hexlified) `meta['0']` JSON text — string-replaced on
    /// write to preserve byte-for-byte formatting, exactly like the PoC.
    meta_text: String,
    /// `meta_json["latestRootBlobId"]` — HEAD, the root `--resume` loads.
    root_id: String,
    /// Message-blob ids (hex) in document order, decoded from the root's field-1
    /// entries (each `0x0A 0x20` + a 32-byte hash that IS a known blob id).
    order: Vec<String>,
}

impl CursorStore {
    /// The parsed JSON `role` of blob `id`, or `None` for a non-JSON (protobuf)
    /// blob or one with no string `role`.
    fn role(&self, id: &str) -> Option<&str> {
        self.parsed.get(id)?.get("role").and_then(|r| r.as_str())
    }
}

/// Read `meta['0']` robustly (stored as TEXT or BLOB, per the PoC's
/// `if isinstance(v, bytes)`) and return its decoded JSON text (un-hexlified).
fn read_meta_text(conn: &rusqlite::Connection) -> Option<String> {
    // SQLite may hand the value back as TEXT or BLOB depending on how it was
    // written (the PoC's `if isinstance(v, bytes)` handles both). rusqlite's
    // `Vec<u8>`/`String` FromSql are storage-class-strict, so pull the
    // `ValueRef` and accept either Text or Blob, normalizing to a hex string.
    let hex: String = conn
        .query_row("SELECT value FROM meta WHERE key='0'", [], |r| {
            use rusqlite::types::ValueRef;
            match r.get_ref(0)? {
                ValueRef::Text(b) | ValueRef::Blob(b) => {
                    Ok(String::from_utf8_lossy(b).into_owned())
                }
                _ => Err(rusqlite::Error::InvalidColumnType(
                    0,
                    "value".into(),
                    rusqlite::types::Type::Text,
                )),
            }
        })
        .ok()?;
    let json_bytes = hex_decode(hex.trim())?;
    String::from_utf8(json_bytes).ok()
}

/// Scan the root protobuf for the conversation's message-blob hashes in document
/// order: each is a `0x0A 0x20` (field 1, wire 2, length 32) prefix followed by
/// 32 raw bytes whose hex IS a known blob id (exactly as `cursor_prune_decode.py`
/// does — only treat it as a message hash if it's in `blobs`, which skips field-8
/// aux roots `0x42 0x20 …` and any incidental `0A 20` byte pair). Operates on
/// raw bytes; no protobuf library.
fn decode_root_order(
    root_data: &[u8],
    blobs: &std::collections::HashMap<String, Vec<u8>>,
) -> Vec<String> {
    // Decode the blob-id hexes to raw 32-byte hashes ONCE, so each `0A 20`
    // candidate compares raw bytes (no per-candidate `hex_encode` alloc); we
    // only hex-encode the confirmed match for the returned ordered ids.
    let raw_ids: std::collections::HashSet<[u8; 32]> = blobs
        .keys()
        .filter_map(|id| hex_decode(id))
        .filter_map(|v| <[u8; 32]>::try_from(v).ok())
        .collect();
    let mut order = Vec::new();
    let mut i = 0usize;
    while i + 34 <= root_data.len() {
        if root_data[i] == 0x0A && root_data[i + 1] == 0x20 {
            let hash: [u8; 32] = root_data[i + 2..i + 34]
                .try_into()
                .expect("34-i slice is 32");
            if raw_ids.contains(&hash) {
                order.push(hex_encode(&hash));
                i += 34;
                continue;
            }
        }
        i += 1;
    }
    order
}

/// Load a [`CursorStore`] from `conn`. `None` (with a `warn!`) when the meta head
/// is unreadable, the root blob is missing, or any blob fails the
/// `sha256(data) == id` content-address invariant — fail-closed so a corrupt /
/// drifted store is never edited.
fn load_cursor_store(conn: &rusqlite::Connection) -> Option<CursorStore> {
    let mut stmt = conn.prepare("SELECT id, data FROM blobs").ok()?;
    let rows = stmt
        .query_map([], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, Vec<u8>>(1)?))
        })
        .ok()?;
    let mut blobs: std::collections::HashMap<String, Vec<u8>> = std::collections::HashMap::new();
    for row in rows {
        let (id, data) = row.ok()?;
        // Content-address invariant: a blob whose id isn't sha256(data) means the
        // store is corrupt or a future cursor changed the hashing — bail rather
        // than re-mint against a broken baseline.
        if blob_id(&data) != id {
            warn!("cursor prune: blob id != sha256(data) — store drift, skipping");
            return None;
        }
        blobs.insert(id, data);
    }

    let meta_text = read_meta_text(conn)?;
    let meta_json: serde_json::Value = serde_json::from_str(&meta_text).ok()?;
    let root_id = meta_json
        .get("latestRootBlobId")
        .and_then(|v| v.as_str())?
        .to_string();
    let Some(root_data) = blobs.get(&root_id) else {
        warn!("cursor prune: meta latestRootBlobId not in blobs — store drift, skipping");
        return None;
    };
    let order = decode_root_order(root_data, &blobs);
    // Parse every JSON blob ONCE. Protobuf tree nodes (root + checkpoints) fail
    // `from_slice` and are simply absent from the map — downstream only ever
    // looks up message blobs by id.
    let parsed: std::collections::HashMap<String, serde_json::Value> = blobs
        .iter()
        .filter_map(|(id, data)| {
            serde_json::from_slice::<serde_json::Value>(data)
                .ok()
                .map(|v| (id.clone(), v))
        })
        .collect();
    Some(CursorStore {
        blobs,
        parsed,
        meta_text,
        root_id,
        order,
    })
}

/// The toolCallIds (in document order) ELIGIBLE to prune for `(tool_name,
/// needle)`: matched assistant `tool-call` parts MINUS toolCallIds whose paired
/// `tool-result` is ALREADY `[pruned]` (idempotent re-prune ⇒ clean no-op).
/// Mirrors the JSONL adapters' "ordered matched minus already-pruned", but over
/// the blob store's `order`. Reuses the `prune.rs` matchers: `toolName` is
/// already claude-shape, so `tool_name_matches` gets `prune::no_tool_map`
/// (literal compare / any-tool selector); `needle == ""` is the no-args selector.
fn cursor_eligible_ids(store: &CursorStore, tool_name: &str, needle: &str) -> Vec<String> {
    let mut matched: Vec<String> = Vec::new();
    let mut pruned: std::collections::HashSet<String> = std::collections::HashSet::new();
    for id in &store.order {
        let Some(blob) = store.parsed.get(id) else {
            continue;
        };
        let role = blob.get("role").and_then(|r| r.as_str()).unwrap_or("");
        let Some(parts) = blob.get("content").and_then(|c| c.as_array()) else {
            continue;
        };
        if role == "assistant" {
            for part in parts {
                if part.get("type").and_then(|t| t.as_str()) != Some("tool-call") {
                    continue;
                }
                let Some(call_id) = part.get("toolCallId").and_then(|v| v.as_str()) else {
                    continue;
                };
                let part_tool = part.get("toolName").and_then(|v| v.as_str()).unwrap_or("");
                if !crate::prune::tool_name_matches(part_tool, tool_name, crate::prune::no_tool_map)
                {
                    continue;
                }
                let args = part.get("args").cloned().unwrap_or(serde_json::Value::Null);
                // Never target the agent's OWN prune-context shell call.
                if crate::prune::value_is_prune_context_call(&args) {
                    continue;
                }
                let arg_ok = if needle.is_empty() {
                    crate::prune::args_value_is_empty(Some(&args))
                } else {
                    crate::prune::value_glob_match(&args, needle)
                };
                if arg_ok {
                    matched.push(call_id.to_string());
                }
            }
        } else if role == "tool" {
            for part in parts {
                if part.get("type").and_then(|t| t.as_str()) != Some("tool-result") {
                    continue;
                }
                let Some(call_id) = part.get("toolCallId").and_then(|v| v.as_str()) else {
                    continue;
                };
                if part.get("result").and_then(|r| r.as_str())
                    == Some(crate::prune::PRUNED_PLACEHOLDER)
                {
                    pruned.insert(call_id.to_string());
                }
            }
        }
    }
    matched.retain(|id| !pruned.contains(id));
    matched
}

/// `count_matches`: read-only pre-scan (opens `?immutable=1`). How many ELIGIBLE
/// matches across the conversation. `0` → control errors back to the live agent
/// (no abort). Returns `0` (not an error) on any store-read failure so a drifted
/// store reads as "nothing to prune" rather than aborting the turn.
fn count_cursor_matches(path: &Path, tool_name: &str, needle: &str) -> std::io::Result<usize> {
    let Some(conn) = open_store_readonly(path) else {
        return Ok(0);
    };
    let Some(store) = load_cursor_store(&conn) else {
        return Ok(0);
    };
    Ok(cursor_eligible_ids(&store, tool_name, needle).len())
}

/// Open the store read-only + immutable (`?immutable=1`) for the count/scan
/// paths — never locks a (dead) cursor-agent's db file. `None` on open failure.
fn open_store_readonly(path: &Path) -> Option<rusqlite::Connection> {
    let uri = format!("file:{}?immutable=1", path.display());
    rusqlite::Connection::open_with_flags(
        uri,
        OpenFlags::SQLITE_OPEN_READ_ONLY
            | OpenFlags::SQLITE_OPEN_NO_MUTEX
            | OpenFlags::SQLITE_OPEN_URI,
    )
    .ok()
}

/// `prune_batch`: blank the last-only target of EACH `(tool_name, needle)` in
/// `targets`, reproducing K sequential last-only passes (the contract codex's
/// `prune_batch` follows). For each target in order: compute its eligible ids
/// (document order, minus already-`[pruned]` on disk AND ids already chosen in
/// THIS batch), take the LAST, add to `chosen`. Then blank every chosen
/// toolCallId's tool-result (`result` + `experimental_content`), re-mint the
/// affected `tool` blobs, byte-replace ALL changed hashes in the root in one
/// pass, re-sha the root, and move `meta.latestRootBlobId` — all in ONE
/// transaction. No-op (`PruneStats::default()`) on any mismatch.
fn prune_batch_cursor(
    path: &Path,
    targets: &[crate::prune::PruneTarget],
) -> std::io::Result<crate::prune::PruneStats> {
    let default = crate::prune::PruneStats::default();
    // RW connection (the apply edits the db). A dead cursor-agent means no lock
    // fight; the main loop only prunes after aborting the agent.
    let Ok(conn) = rusqlite::Connection::open(path) else {
        warn!("cursor prune: failed to open store RW, skipping");
        return Ok(default);
    };
    let Some(store) = load_cursor_store(&conn) else {
        // load_cursor_store already logged the specific guard failure.
        return Ok(default);
    };

    // Select the chosen toolCallIds across all targets (last-only per target,
    // subtracting earlier rounds' picks — reproduces K on-disk passes).
    let mut chosen: Vec<String> = Vec::new();
    let mut chosen_set: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (tool_name, needle) in targets {
        let eligible = cursor_eligible_ids(&store, tool_name, needle);
        if let Some(pick) = eligible
            .into_iter()
            .rev()
            .find(|id| !chosen_set.contains(id))
        {
            chosen_set.insert(pick.clone());
            chosen.push(pick);
        }
    }
    if chosen.is_empty() {
        // Nothing eligible (TOCTOU after the control pre-check, or all already
        // pruned) — safe no-op, frame skipped.
        return Ok(default);
    }

    // Group chosen ids by the `tool` blob that holds their result, so a blob with
    // multiple tool-result parts is re-minted once. `tool_blob_id → set(callId)`.
    let mut blob_targets: std::collections::HashMap<String, std::collections::HashSet<String>> =
        std::collections::HashMap::new();
    for id in &store.order {
        if store.role(id) != Some("tool") {
            continue;
        }
        let Some(blob) = store.parsed.get(id) else {
            continue;
        };
        let Some(parts) = blob.get("content").and_then(|c| c.as_array()) else {
            continue;
        };
        for part in parts {
            if part.get("type").and_then(|t| t.as_str()) != Some("tool-result") {
                continue;
            }
            if let Some(call_id) = part.get("toolCallId").and_then(|v| v.as_str()) {
                if chosen_set.contains(call_id) {
                    blob_targets
                        .entry(id.clone())
                        .or_default()
                        .insert(call_id.to_string());
                }
            }
        }
    }
    if blob_targets.is_empty() {
        // A chosen call had no tool-result blob to blank (in-flight, or store
        // drift) — fail closed, no partial write.
        warn!("cursor prune: chosen toolCallId(s) have no tool-result blob, skipping");
        return Ok(default);
    }

    // Build the new blobs + accumulate (old_hash → new_hash) re-point pairs and
    // the freed-byte / blanked-result tally.
    let mut repoints: Vec<(String, String)> = Vec::new();
    let mut new_blobs: Vec<(String, Vec<u8>)> = Vec::new();
    let mut results_blanked = 0usize;
    let mut freed_bytes = 0usize;
    for (old_tool_id, call_ids) in &blob_targets {
        // Clone the parsed-once value (these are JSON message blobs, present in
        // `parsed`) so we can blank in place without re-`from_slice`-ing bytes.
        let Some(mut blob) = store.parsed.get(old_tool_id).cloned() else {
            warn!("cursor prune: tool blob vanished mid-build, skipping");
            return Ok(default);
        };
        let Some(parts) = blob.get_mut("content").and_then(|c| c.as_array_mut()) else {
            warn!("cursor prune: tool blob has no content array, skipping");
            return Ok(default);
        };
        for part in parts.iter_mut() {
            let Some(obj) = part.as_object_mut() else {
                continue;
            };
            if obj.get("type").and_then(|t| t.as_str()) != Some("tool-result") {
                continue;
            }
            let is_target = obj
                .get("toolCallId")
                .and_then(|v| v.as_str())
                .map(|c| call_ids.contains(c))
                .unwrap_or(false);
            if !is_target {
                continue;
            }
            freed_bytes += blank_tool_result_twins(obj);
            results_blanked += 1;
        }
        // serde_json::to_vec is compact (no spaces) — matches the PoC's
        // separators=(",",":") so the re-minted blob round-trips byte-stably.
        let Ok(new_data) = serde_json::to_vec(&blob) else {
            warn!("cursor prune: re-serialize of blanked tool blob failed, skipping");
            return Ok(default);
        };
        let new_id = blob_id(&new_data);
        new_blobs.push((new_id.clone(), new_data));
        repoints.push((old_tool_id.clone(), new_id));
    }
    if results_blanked == 0 {
        // Every chosen result was already `[pruned]` (idempotent) — no-op.
        return Ok(default);
    }

    // Re-mint the root: byte-replace each changed 32-byte hash. The root
    // references message blobs DIRECTLY (flat), so each old hash must appear
    // EXACTLY ONCE — guard it (a drifted/nested layout where the count isn't 1
    // bails the whole prune rather than corrupt the tree).
    let Some(old_root_data) = store.blobs.get(&store.root_id) else {
        warn!("cursor prune: root blob vanished, skipping");
        return Ok(default);
    };
    let mut new_root_data = old_root_data.clone();
    for (old_id, new_id) in &repoints {
        let (Some(old_raw), Some(new_raw)) = (hex_decode(old_id), hex_decode(new_id)) else {
            warn!("cursor prune: bad hash hex during re-point, skipping");
            return Ok(default);
        };
        // old/new are both 32-byte hashes (equal length), so re-point is an
        // in-place `copy_from_slice` at the unique offset — no realloc.
        let (occ, first) = scan_subslice(&new_root_data, &old_raw);
        if occ != 1 {
            warn!(
                "cursor prune: old hash appears {occ} times in root (expected 1) — drift, skipping"
            );
            return Ok(default);
        }
        let at = first.expect("occ == 1 implies an offset");
        new_root_data[at..at + new_raw.len()].copy_from_slice(&new_raw);
    }
    let new_root_id = blob_id(&new_root_data);

    // Re-point meta: string-replace the old root id with the new (preserves the
    // JSON's original formatting, exactly like the PoC). Must be uniquely
    // replaceable or we bail.
    if store.meta_text.matches(&store.root_id).count() != 1 {
        warn!("cursor prune: root id not uniquely replaceable in meta json, skipping");
        return Ok(default);
    }
    let new_meta_text = store.meta_text.replace(&store.root_id, &new_root_id);
    let new_meta_hex = hex_encode(new_meta_text.as_bytes());

    // Single transaction: INSERT OR IGNORE the new blobs + root (never delete —
    // old roots stay valid checkpoints), then move the meta head. A mid-way
    // failure rolls back, so the store is never left half-edited.
    let apply = || -> rusqlite::Result<()> {
        let tx = conn.unchecked_transaction()?;
        for (id, data) in &new_blobs {
            tx.execute(
                "INSERT OR IGNORE INTO blobs(id, data) VALUES(?, ?)",
                rusqlite::params![id, data],
            )?;
        }
        tx.execute(
            "INSERT OR IGNORE INTO blobs(id, data) VALUES(?, ?)",
            rusqlite::params![new_root_id, new_root_data],
        )?;
        tx.execute(
            "UPDATE meta SET value = ? WHERE key='0'",
            rusqlite::params![new_meta_hex],
        )?;
        tx.commit()
    };
    if let Err(e) = apply() {
        warn!(error = %e, "cursor prune: transaction failed, rolled back");
        return Ok(default);
    }
    // Best-effort WAL checkpoint so the edit is visible to the next `--resume`
    // even if it opens the main db file directly (matches the PoC).
    let _ = conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);");

    Ok(crate::prune::PruneStats {
        results_blanked,
        freed_bytes,
    })
}

/// Blank BOTH twins of one `tool-result` part in place: the text twin `result`
/// (via the shared `prune::blank_string_field`) and the multimodal twin
/// `experimental_content` (the model could otherwise still read the output from
/// it). Returns the combined freed-byte estimate — both twins' original
/// serialized length counts, so the `~Nk freed` tally doesn't undercount when
/// the output lived in `experimental_content`.
fn blank_tool_result_twins(obj: &mut serde_json::Map<String, serde_json::Value>) -> usize {
    let mut freed = 0usize;
    if let Some(f) =
        crate::prune::blank_string_field(obj, "result", crate::prune::PRUNED_PLACEHOLDER)
    {
        freed += f;
    }
    // `experimental_content` is an array (not a plain string), so it can't go
    // through `blank_string_field`; count its original serialized length minus
    // the replacement's footprint, then overwrite with the placeholder block.
    let replacement =
        serde_json::json!([{"type": "text", "text": crate::prune::PRUNED_PLACEHOLDER}]);
    if let Some(existing) = obj.get("experimental_content") {
        freed += existing
            .to_string()
            .len()
            .saturating_sub(replacement.to_string().len());
    }
    obj.insert("experimental_content".to_string(), replacement);
    freed
}

/// One-pass scan: count non-overlapping occurrences of `needle` in `haystack`
/// (raw bytes) and report the FIRST offset. The re-point caller asserts the
/// count is exactly 1, then `copy_from_slice`s the equal-length replacement at
/// `first` — so one scan serves both the uniqueness guard and the locate.
fn scan_subslice(haystack: &[u8], needle: &[u8]) -> (usize, Option<usize>) {
    if needle.is_empty() || haystack.len() < needle.len() {
        return (0, None);
    }
    let mut count = 0;
    let mut first = None;
    let mut i = 0;
    while i + needle.len() <= haystack.len() {
        if &haystack[i..i + needle.len()] == needle {
            if first.is_none() {
                first = Some(i);
            }
            count += 1;
            i += needle.len();
        } else {
            i += 1;
        }
    }
    (count, first)
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
use crate::adapters::import_shared::{
    basename_or, collapse_title, emit_chat, is_synthetic_wrapper, mint_project_id,
    user_message_body, ImportedChat, ImportedMessage, ProgressThrottle,
};
use crate::writer::WriteEvent;

/// Parsed `composer.composerHeaders.allComposers[i]` entry. Only the fields
/// we actually use — serde drops the rest via the catch-all (no #[serde(deny_unknown_fields)]).
struct ComposerHeader {
    id: String,
    /// Resolved workspace path, or `None` when only `workspaceIdentifier.id`
    /// is present (no `uri.path`). Caller DROPS `None` composers — a chat with
    /// no project folder can't be opened or resumed in Zucchini.
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

    // Group composers by project path. Composers whose `workspaceIdentifier`
    // carries only an opaque id (no resolvable `uri.path`) are DROPPED: a chat
    // with no project folder can't be opened or resumed in Zucchini (the agent
    // is spawned in the project's cwd), so a "<no project>" bucket would only
    // create dead chats. Sort each group's composers by created_at so progress
    // is deterministic across re-runs and the iOS UI walks oldest→newest.
    let mut by_project: BTreeMap<String, Vec<ComposerHeader>> = BTreeMap::new();
    let mut skipped_no_project = 0usize;
    for h in headers {
        let Some(key) = h.project_path.clone() else {
            skipped_no_project += 1;
            continue;
        };
        by_project.entry(key).or_default().push(h);
    }
    if skipped_no_project > 0 {
        info!(
            count = skipped_no_project,
            "cursor: skipped composers with no resolvable project path"
        );
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
    let mut throttle = ProgressThrottle::new();
    for (project_path, mut composers) in by_project {
        composers.sort_by_key(|h| h.created_at_ms);
        let project_id = mint_project_id(machine_id, &project_path);
        let project_name = basename_or(&project_path, "project");
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
            // Per-percent throttle shared with every importer; see `ProgressThrottle`.
            throttle
                .step(done_composers, total_composers, &progress)
                .await;
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

/// Read the `value` column (index 0) as raw bytes regardless of whether SQLite
/// stored it as BLOB or TEXT. Cursor's `ItemTable`/`cursorDiskKV` declare
/// `value BLOB`, but some Cursor versions persist the JSON as TEXT; rusqlite's
/// `Vec<u8>` `FromSql` rejects the TEXT storage class with "Invalid column type
/// Text at index: 0" (SPAWNER-Y), so coerce both here. Downstream callers feed
/// the bytes straight into `serde_json::from_slice`, which is byte-agnostic.
fn value_bytes(r: &rusqlite::Row) -> rusqlite::Result<Vec<u8>> {
    use rusqlite::types::ValueRef;
    match r.get_ref(0)? {
        ValueRef::Text(t) => Ok(t.to_vec()),
        ValueRef::Blob(b) => Ok(b.to_vec()),
        // Null / numeric: defer to the typed getter so the error stays accurate
        // (these never occur for Cursor's JSON value columns).
        _ => r.get::<_, Vec<u8>>(0),
    }
}

fn read_composer_headers(conn: &rusqlite::Connection) -> anyhow::Result<Vec<ComposerHeader>> {
    let blob: Option<Vec<u8>> = conn
        .query_row(
            "SELECT value FROM ItemTable WHERE key = 'composer.composerHeaders'",
            [],
            value_bytes,
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
        let blob: Option<Vec<u8>> = stmt.query_row([&key], value_bytes).optional_or_none()?;
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
            let blob: Option<Vec<u8>> = stmt.query_row([&key], value_bytes).optional_or_none()?;
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

    // Per-message `created_at`: bubbles have no clock. Use
    // `chat_created_at + idx ms` (idx = emit slot, == `messages.len()` at push
    // time) so the writer's `created_at` is monotonic per row — text+tool_use
    // from the same bubble get distinct timestamps because they occupy
    // consecutive slots. The bubbleId is a UUID, threaded as the message id so
    // re-imports converge in place. The PutChat-then-PutMessage emit is shared
    // via `emit_chat`.
    let mut messages: Vec<ImportedMessage> = Vec::new();
    let mut first_user_text: Option<String> = None;
    for h in headers {
        let Some(bubble_id) = h.get("bubbleId").and_then(|v| v.as_str()) else {
            continue;
        };
        let Some(bubble) = bubble_rows.get(&(header.id.clone(), bubble_id.to_string())) else {
            continue;
        };
        let id = Uuid::parse_str(bubble_id).ok();
        let (sender, frames) = match classify_bubble(bubble) {
            BubbleOut::User { text } => {
                if first_user_text.is_none() {
                    first_user_text = Some(text.clone());
                }
                ("user", vec![user_message_body(&text)])
            }
            BubbleOut::Assistant {
                text_frame,
                tool_frame,
            } => (
                "agent",
                [text_frame, tool_frame].into_iter().flatten().collect(),
            ),
            BubbleOut::Skip => continue,
        };
        for body in frames {
            let ts = chat_created_at + ChronoDuration::milliseconds(messages.len() as i64);
            messages.push(ImportedMessage {
                id,
                sender,
                body,
                created_at: ts,
            });
        }
    }

    if messages.is_empty() {
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

    emit_chat(
        write_tx,
        user_id,
        ImportedChat {
            id: chat_id,
            project_id,
            title,
            created_at: chat_created_at,
            messages,
        },
    )
    .await;

    Ok(())
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
    if is_synthetic_wrapper(text) {
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
    // The persisted bubble has no per-message usage, and iOS reads
    // `chats.context_tokens` for the live counter anyway — the shared
    // envelope helper stamps the claude-shape zero usage. Matches the
    // live cursor adapter's `normalize_assistant_frame`.
    crate::adapter::claude_assistant_text_envelope(text)
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
    Some(claude_tool_use_envelope(&call_id, &claude_name, input))
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
///
/// SIBLING MAP — keep in sync with [`map_cursor_tool`] (the live-wire path).
/// Both encode the same cursor→claude tool set; they are deliberately NOT
/// merged because the input vocabularies and the source arg keys differ
/// (persisted `target_file`/`relativeWorkspacePath` vs live `path`). When you
/// add or change a tool here, check whether the live side needs it too.
///
/// KNOWN ASYMMETRY: `todo_write` → `TodoWrite` exists here but has no
/// counterpart in [`map_cursor_tool`]. That is intentional, not an oversight:
/// `todo_write` is an observed persisted tool name, but the live wire's todo
/// verb key (if any) is unconfirmed, so we don't guess a spelling there.
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
    rename_either_key(args, &[from], to)
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

    /// Delegates to the shared stringifier in `adapter.rs` so the event→tag
    /// mapping has a single source of truth.
    fn run(a: &mut CursorAdapter, line: &str) -> Vec<String> {
        crate::adapter::stringify_events(a.handle_line(line.to_string()))
    }

    /// Local alias for the shared Frame-payload parser, so the many cursor
    /// normalization tests below can `frame_value(&events[0])` instead of
    /// open-coding `strip_prefix("Frame(").unwrap().strip_suffix(')')…`.
    fn frame_value(event: &str) -> Value {
        crate::adapter::frame_value(event)
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
        let v = frame_value(&events[0]);
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
        let v = frame_value(&events[0]);
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
        let v = frame_value(&events[0]);
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
        let v = frame_value(&events[0]);
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
        let v = frame_value(&events[0]);
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
        let v = frame_value(&events[0]);
        let block = &v["message"]["content"][0];
        assert_eq!(block["name"], "fooBar");
        assert_eq!(block["input"]["x"], 1);
    }

    #[test]
    fn cursor_update_todos_maps_to_claude_todowrite() {
        // Live verb key `updateTodosToolCall` (confirmed against live
        // cursor-agent stream-json) → claude `TodoWrite` with the `todos`
        // array forwarded verbatim, mirroring the importer's `todo_write`
        // → TodoWrite mapping so both surfaces render the same iOS summary.
        let mut a = CursorAdapter::new();
        let line = r#"{"type":"tool_call","subtype":"completed","call_id":"td1","tool_call":{"updateTodosToolCall":{"args":{"todos":[{"id":"1","content":"step one","status":"TODO_STATUS_PENDING"}],"merge":false},"result":{}}}}"#;
        let events = run(&mut a, line);
        assert_eq!(events.len(), 1);
        let v = frame_value(&events[0]);
        let block = &v["message"]["content"][0];
        assert_eq!(block["type"], "tool_use");
        assert_eq!(block["name"], "TodoWrite");
        assert_eq!(block["id"], "td1");
        assert_eq!(block["input"]["todos"][0]["content"], "step one");
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
        let v = frame_value(&events[0]);
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
        let v = frame_value(&events[0]);
        let block = &v["message"]["content"][0];
        assert_eq!(block["type"], "tool_use");
        assert_eq!(block["id"], "big_1");
        assert_eq!(block["name"], "Edit");
        // Oversize path emits empty input — iOS falls back to tool-name-only.
        assert!(block["input"].as_object().unwrap().is_empty());
    }

    #[test]
    fn oversize_assistant_frame_falls_through_and_renders() {
        // A >64KB assistant reply is body-derived (the text is what we render),
        // so the oversize fast-path must fall through to the normal `assistant`
        // arm rather than drop. Rare, but correctness beats the heap blip.
        let mut a = CursorAdapter::new();
        let big = "x".repeat(MAX_STREAM_FRAME_BYTES + 1024);
        let line = format!(
            r#"{{"type":"assistant","message":{{"role":"assistant","content":[{{"type":"text","text":"{}"}}]}},"session_id":"abc-1"}}"#,
            big
        );
        assert!(line.len() > MAX_STREAM_FRAME_BYTES);
        let events = run(&mut a, &line);
        assert_eq!(events.len(), 1);
        let v = frame_value(&events[0]);
        assert_eq!(v["type"], "assistant");
        assert_eq!(v["message"]["content"][0]["text"], big);
    }

    #[test]
    fn oversize_result_frame_falls_through_emits_terminal() {
        // CRITICAL regression guard: the `result` arm is the TERMINAL. An
        // oversize `result` must NOT be dropped — dropping it means the turn
        // never latches Result and the user sees "Agent interrupted" instead of
        // `[result: success]`. cursor result frames are realistically tiny, but
        // we pad past the cap to force the oversize path, with `"type":"result"`
        // first so `extract_json_type` classifies it before the padding.
        let mut a = CursorAdapter::new();
        let pad = "x".repeat(MAX_STREAM_FRAME_BYTES + 1024);
        let line = format!(
            r#"{{"type":"result","subtype":"success","duration_ms":42,"is_error":false,"usage":{{"inputTokens":100,"outputTokens":5,"cacheReadTokens":20,"cacheWriteTokens":0}},"pad":"{}"}}"#,
            pad
        );
        assert!(line.len() > MAX_STREAM_FRAME_BYTES);
        let events = run(&mut a, &line);
        // ContextTokens (100+20+0 over N_calls=1), Frame, Result.
        assert_eq!(events.len(), 3);
        assert_eq!(events[0], "ContextTokens(120)");
        let v = frame_value(&events[1]);
        assert_eq!(v["type"], "result");
        assert_eq!(v["subtype"], "success");
        assert_eq!(events[2], "Result");
    }

    #[test]
    fn oversize_lifecycle_frame_dropped_not_forwarded() {
        // An oversize lifecycle/unknown frame emits NOTHING (dropped, never
        // forwarded raw — raw cursor JSON renders as literal `[type]` on iOS).
        let mut a = CursorAdapter::new();
        let big = "x".repeat(MAX_STREAM_FRAME_BYTES + 1024);
        let thinking = format!(
            r#"{{"type":"thinking","subtype":"delta","text":"{}"}}"#,
            big
        );
        assert!(thinking.len() > MAX_STREAM_FRAME_BYTES);
        assert!(run(&mut a, &thinking).is_empty());

        let system = format!(r#"{{"type":"system","subtype":"init","pad":"{}"}}"#, big);
        assert!(system.len() > MAX_STREAM_FRAME_BYTES);
        assert!(run(&mut a, &system).is_empty());
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
            // chat_id prefix names the worktree dir, asserted below.
            chat_id: "abcdef012345-6789-...",
            worktree: true,
            ..TurnContext::for_test(&prompt_file, None, false, None)
        };
        let cmd = a.prepare_command(&ctx).unwrap();
        assert!(
            cmd.contains("--worktree 'zcm-abcdef012345'"),
            "got: {}",
            cmd
        );
        // Stdin preamble carries capabilities + (worktree on) the worktree rule
        // naming cursor's `~/.cursor/worktrees/<repo>/<name>` dir.
        assert!(cmd.contains("attach-file"), "got: {}", cmd);
        assert!(cmd.contains("schedule-message"), "got: {}", cmd);
        assert!(cmd.contains("Worktree:"), "worktree rule present: {}", cmd);
        assert!(
            cmd.contains(".cursor/worktrees/proj/zcm-abcdef012345"),
            "cursor worktree abs path in rule: {}",
            cmd
        );
    }

    #[test]
    fn worktree_off_omits_worktree_rule_from_preamble() {
        use crate::adapter::TurnContext;
        use std::path::PathBuf;
        let mut a = CursorAdapter::new();
        let prompt_file = PathBuf::from("/tmp/p.txt");
        let ctx = TurnContext::for_test(&prompt_file, None, false, None);
        let cmd = a.prepare_command(&ctx).unwrap();
        // Capabilities always present; worktree rule only when worktree is on.
        assert!(cmd.contains("attach-file"), "got: {}", cmd);
        assert!(cmd.contains("schedule-message"), "got: {}", cmd);
        assert!(
            !cmd.contains("Worktree:"),
            "worktree off → no rule: {}",
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
            // chat_id prefix names the worktree dir, asserted below.
            chat_id: "abcdef012345-6789-...",
            worktree: true,
            ..TurnContext::for_test(&prompt_file, Some("sess-1"), false, None)
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
        let ctx = TurnContext::for_test(&prompt_file, None, true, None);
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
        let ctx = TurnContext::for_test(&prompt_file, None, false, None);
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

    /// SPAWNER-Y: Cursor declares `value BLOB` but some versions persist the
    /// JSON as TEXT. `value_bytes` must read both storage classes, where the
    /// plain `get::<Vec<u8>>` getter rejects TEXT with "Invalid column type".
    #[test]
    fn value_bytes_reads_text_and_blob() {
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        conn.execute_batch("CREATE TABLE ItemTable (key TEXT UNIQUE, value BLOB);")
            .unwrap();
        conn.execute(
            "INSERT INTO ItemTable(key, value) VALUES('as_text', ?)",
            rusqlite::params!["{\"a\":1}"], // bound as TEXT
        )
        .unwrap();
        conn.execute(
            "INSERT INTO ItemTable(key, value) VALUES('as_blob', ?)",
            rusqlite::params![b"{\"a\":1}".to_vec()], // bound as BLOB
        )
        .unwrap();

        for key in ["as_text", "as_blob"] {
            let bytes: Vec<u8> = conn
                .query_row(
                    "SELECT value FROM ItemTable WHERE key = ?",
                    [key],
                    value_bytes,
                )
                .unwrap();
            assert_eq!(&bytes, b"{\"a\":1}", "value_bytes failed for {key}");
        }

        // Document the regression: the typed getter still fails on the TEXT row.
        let direct: rusqlite::Result<Vec<u8>> = conn.query_row(
            "SELECT value FROM ItemTable WHERE key = 'as_text'",
            [],
            |r| r.get::<_, Vec<u8>>(0),
        );
        assert!(direct.is_err());
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

        let result = import(
            machine_id,
            user_id,
            tx,
            Box::new(|_| Box::pin(async {}) as futures::future::BoxFuture<'static, ()>)
                as ImportProgress,
        )
        .await;

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
        // composer2 has no resolvable project path → dropped entirely (no
        // "<no project>" bucket), so only `/tmp/proj-a` survives.
        assert_eq!(projects.len(), 1, "only /tmp/proj-a (no-project dropped)");
        assert_eq!(chats.len(), 1, "one per accepted composer");
        // composer1: user + assistant_text + assistant_tool = 3.
        assert_eq!(messages.len(), 3);

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
        let ctx = TurnContext::for_test(&prompt_file, None, false, None);
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
        let ctx = TurnContext::for_test(&prompt_file, None, false, Some("Composer 2.5 Fast"));
        let cmd = a.prepare_command(&ctx).unwrap();
        assert!(cmd.contains("--model 'Composer 2.5 Fast'"), "got: {}", cmd);
    }

    #[test]
    fn model_none_omits_model_flag() {
        // `chats.model = None` → no `--model` flag at all; cursor picks
        // the user's default from its own config.
        use crate::adapter::TurnContext;
        use std::path::PathBuf;
        let mut a = CursorAdapter::new();
        let prompt_file = PathBuf::from("/tmp/p.txt");
        let ctx = TurnContext::for_test(&prompt_file, None, false, None);
        let cmd = a.prepare_command(&ctx).unwrap();
        assert!(!cmd.contains("--model"), "got: {}", cmd);
    }

    #[test]
    fn preamble_carries_attach_and_prune_instructions() {
        // The stdin preamble must inject BOTH the attach-file how-to and the
        // prune-context standing order (cursor has no system-prompt flag).
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
            user_timezone: None,
        };
        let cmd = a.prepare_command(&ctx).unwrap();
        // The instructions are shell-escaped into the `printf %s '<preamble>'`
        // arg; assert a distinctive token from each is present.
        assert!(cmd.contains("prune-context"), "got: {}", cmd);
        assert!(
            cmd.contains("attach"),
            "missing attach instruction: {}",
            cmd
        );
    }

    // ===== prune-context cue tests (correction B) =====

    #[test]
    fn prune_context_shell_call_completion_fires_tool_result_cue() {
        // A `tool_call.completed` whose shellToolCall command is a prune-context
        // invocation must emit the visible bubble THEN the ToolResult cue.
        let mut a = CursorAdapter::new();
        let cmd = r#"\"$ZUCCHINI_SPAWNER_BIN\" prune-context --tool-name Read --args \"*BACKLOG.md*\" --reason x"#;
        let line = format!(
            r#"{{"type":"tool_call","subtype":"completed","call_id":"tool_p","tool_call":{{"shellToolCall":{{"args":{{"command":"{cmd}"}},"result":{{"success":{{"stdout":"ok"}}}}}}}}}}"#
        );
        let events = run(&mut a, &line);
        assert_eq!(events.len(), 2, "bubble + cue, got {events:?}");
        assert_eq!(events[1], "ToolResult");
    }

    #[test]
    fn ordinary_shell_call_completion_does_not_fire_cue() {
        // A sibling/ordinary shell call's completion renders its bubble only —
        // no cue (call-keyed). Firing here would drop the queued prune.
        let mut a = CursorAdapter::new();
        let line = r#"{"type":"tool_call","subtype":"completed","call_id":"tool_s","tool_call":{"shellToolCall":{"args":{"command":"grep -rn foo src/"},"result":{"success":{"stdout":"hits"}}}}}"#;
        let events = run(&mut a, line);
        assert_eq!(events.len(), 1, "bubble only, got {events:?}");
        assert_ne!(events[0], "ToolResult");
    }

    // ===== prune-context store surgery tests (correction A + plan §7) =====

    /// Owns a unique temp dir holding a `store.db`, deleted on drop. Hand-rolled
    /// (no `tempfile` dep), mirroring the importer tests' temp-file pattern.
    struct TempStore {
        dir: std::path::PathBuf,
    }

    impl TempStore {
        fn db_path(&self) -> std::path::PathBuf {
            self.dir.join("store.db")
        }
    }

    impl Drop for TempStore {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    /// sha256 → hex, exactly the production `blob_id`.
    fn sha_hex(data: &[u8]) -> String {
        blob_id(data)
    }

    /// Compact JSON bytes (matches the production `serde_json::to_vec`).
    fn json_bytes(v: &serde_json::Value) -> Vec<u8> {
        serde_json::to_vec(v).unwrap()
    }

    /// Build a minimal cursor store.db from a list of (already-serialized)
    /// message blobs IN ORDER, minting a flat root protobuf whose field-1 entries
    /// (`0A 20` + raw hash) reference each blob, and a hex-encoded meta head. The
    /// blobs are content-addressed (id = sha256(data)), so the store passes the
    /// production load guard. Returns the temp store + the ordered blob ids.
    fn build_store(message_blobs: &[Vec<u8>]) -> (TempStore, Vec<String>) {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let mut dir = std::env::temp_dir();
        dir.push(format!(
            "zucchini_cursor_prune_{}_{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let store = TempStore { dir };

        let mut ids: Vec<String> = Vec::new();
        let mut root = Vec::new();
        for data in message_blobs {
            let id = sha_hex(data);
            // field 1, wire 2, len 32 = 0x0A 0x20, then the raw hash.
            root.push(0x0Au8);
            root.push(0x20u8);
            root.extend_from_slice(&hex_decode(&id).unwrap());
            ids.push(id);
        }
        let root_id = sha_hex(&root);
        let meta_json =
            serde_json::json!({"agentId": "agent-1", "latestRootBlobId": root_id, "name": "t"});
        let meta_text = serde_json::to_string(&meta_json).unwrap();
        let meta_hex = hex_encode(meta_text.as_bytes());

        let conn = rusqlite::Connection::open(store.db_path()).unwrap();
        conn.execute_batch(
            "CREATE TABLE blobs (id TEXT PRIMARY KEY, data BLOB);
             CREATE TABLE meta  (key TEXT PRIMARY KEY, value TEXT);",
        )
        .unwrap();
        for (data, id) in message_blobs.iter().zip(ids.iter()) {
            conn.execute(
                "INSERT INTO blobs(id, data) VALUES(?, ?)",
                rusqlite::params![id, data],
            )
            .unwrap();
        }
        conn.execute(
            "INSERT INTO blobs(id, data) VALUES(?, ?)",
            rusqlite::params![root_id, root],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO meta(key, value) VALUES('0', ?)",
            rusqlite::params![meta_hex],
        )
        .unwrap();
        (store, ids)
    }

    /// Read the current meta head + that root's referenced (blanked-or-not) tool
    /// blob, from a freshly opened connection.
    fn read_head(path: &std::path::Path) -> (String, Vec<String>) {
        let conn = rusqlite::Connection::open(path).unwrap();
        let store = load_cursor_store(&conn).unwrap();
        (store.root_id, store.order)
    }

    fn blob_data(path: &std::path::Path, id: &str) -> Option<Vec<u8>> {
        let conn = rusqlite::Connection::open(path).unwrap();
        conn.query_row(
            "SELECT data FROM blobs WHERE id = ?",
            rusqlite::params![id],
            |r| r.get::<_, Vec<u8>>(0),
        )
        .ok()
    }

    fn standard_convo() -> Vec<Vec<u8>> {
        let system = json_bytes(&serde_json::json!({"role":"system","content":"you are helpful"}));
        let user = json_bytes(&serde_json::json!({"role":"user","content":"read BACKLOG.md"}));
        let assistant = json_bytes(&serde_json::json!({
            "role":"assistant",
            "content":[{"type":"tool-call","toolCallId":"tool_1","toolName":"Read","args":{"path":"/repo/BACKLOG.md"}}]
        }));
        let tool = json_bytes(&serde_json::json!({
            "role":"tool",
            "content":[{"type":"tool-result","toolCallId":"tool_1","result":"line1\nline2\nlots of text here","experimental_content":[{"type":"text","text":"line1\nline2\nlots of text here"}]}]
        }));
        vec![system, user, assistant, tool]
    }

    #[test]
    fn count_and_prune_blanks_last_only_and_repoints_head() {
        let (store, ids) = build_store(&standard_convo());
        let path = store.db_path();
        let (old_root, _old_order) = read_head(&path);
        let old_tool_id = ids[3].clone();

        // count: exactly one eligible Read match (any-needle path).
        assert_eq!(
            count_cursor_matches(&path, "Read", "*BACKLOG.md*").unwrap(),
            1
        );

        // prune one target.
        let stats =
            prune_batch_cursor(&path, &[("Read".to_string(), "*BACKLOG.md*".to_string())]).unwrap();
        assert_eq!(stats.results_blanked, 1);
        assert!(stats.freed_bytes > 0, "freed_bytes should be positive");

        // meta head moved to a NEW root.
        let (new_root, new_order) = read_head(&path);
        assert_ne!(new_root, old_root, "head must move to a new root");
        // new root references the blanked (new) tool blob, not the old one.
        assert!(!new_order.contains(&old_tool_id), "old tool hash repointed");
        let new_tool_id = new_order[3].clone();
        let blanked: serde_json::Value =
            serde_json::from_slice(&blob_data(&path, &new_tool_id).unwrap()).unwrap();
        assert_eq!(
            blanked["content"][0]["result"],
            crate::prune::PRUNED_PLACEHOLDER
        );
        assert_eq!(
            blanked["content"][0]["experimental_content"][0]["text"],
            crate::prune::PRUNED_PLACEHOLDER
        );

        // checkpoints intact: old root + old tool blob still present.
        assert!(blob_data(&path, &old_root).is_some(), "old root preserved");
        assert!(
            blob_data(&path, &old_tool_id).is_some(),
            "old tool blob preserved"
        );

        // idempotent: a second identical prune is a clean no-op.
        let stats2 =
            prune_batch_cursor(&path, &[("Read".to_string(), "*BACKLOG.md*".to_string())]).unwrap();
        assert_eq!(stats2.results_blanked, 0, "re-prune must be a no-op");
        let (root_after, _) = read_head(&path);
        assert_eq!(root_after, new_root, "no-op must not move the head");
    }

    #[test]
    fn multi_tool_call_blob_prunes_one_sibling_untouched() {
        // One assistant blob with two tool-calls; their results live in one tool
        // blob with two tool-result parts. Prune Read of A only → its result
        // blanked, the sibling (Read of B) left intact.
        let system = json_bytes(&serde_json::json!({"role":"system","content":"sys"}));
        let assistant = json_bytes(&serde_json::json!({
            "role":"assistant",
            "content":[
                {"type":"tool-call","toolCallId":"tc_a","toolName":"Read","args":{"path":"/repo/A.md"}},
                {"type":"tool-call","toolCallId":"tc_b","toolName":"Read","args":{"path":"/repo/B.md"}}
            ]
        }));
        let tool = json_bytes(&serde_json::json!({
            "role":"tool",
            "content":[
                {"type":"tool-result","toolCallId":"tc_a","result":"AAA content","experimental_content":[{"type":"text","text":"AAA content"}]},
                {"type":"tool-result","toolCallId":"tc_b","result":"BBB content","experimental_content":[{"type":"text","text":"BBB content"}]}
            ]
        }));
        let (store, _ids) = build_store(&[system, assistant, tool]);
        let path = store.db_path();

        let stats =
            prune_batch_cursor(&path, &[("Read".to_string(), "*A.md*".to_string())]).unwrap();
        assert_eq!(stats.results_blanked, 1);

        let (_new_root, new_order) = read_head(&path);
        let tool_blob: serde_json::Value =
            serde_json::from_slice(&blob_data(&path, &new_order[2]).unwrap()).unwrap();
        let parts = tool_blob["content"].as_array().unwrap();
        let a = parts.iter().find(|p| p["toolCallId"] == "tc_a").unwrap();
        let b = parts.iter().find(|p| p["toolCallId"] == "tc_b").unwrap();
        assert_eq!(a["result"], crate::prune::PRUNED_PLACEHOLDER, "A pruned");
        assert_eq!(b["result"], "BBB content", "sibling B untouched");
    }

    #[test]
    fn prune_context_shell_call_is_never_selected() {
        // An assistant whose tool-call is the agent's own prune-context shell
        // invocation must never be eligible (self-exclusion).
        let system = json_bytes(&serde_json::json!({"role":"system","content":"sys"}));
        let assistant = json_bytes(&serde_json::json!({
            "role":"assistant",
            "content":[{"type":"tool-call","toolCallId":"tc_p","toolName":"Bash","args":{"command":"\"$ZUCCHINI_SPAWNER_BIN\" prune-context --tool-name Read --args \"*A.md*\" --reason x"}}]
        }));
        let tool = json_bytes(&serde_json::json!({
            "role":"tool",
            "content":[{"type":"tool-result","toolCallId":"tc_p","result":"pruned 1 tool output","experimental_content":[{"type":"text","text":"pruned 1 tool output"}]}]
        }));
        let (store, _ids) = build_store(&[system, assistant, tool]);
        let path = store.db_path();

        // any-tool, prune-context's own needle → zero eligible.
        assert_eq!(count_cursor_matches(&path, "", "*A.md*").unwrap(), 0);
        let stats = prune_batch_cursor(&path, &[("".to_string(), "*A.md*".to_string())]).unwrap();
        assert_eq!(stats.results_blanked, 0, "self-call must not be pruned");
    }
}
