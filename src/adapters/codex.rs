//! codex (openai codex-cli) adapter. Normalizes codex's `exec --json` stream
//! frames into claude-shape envelopes on the wire so iOS's
//! `SpawnerMessageDescriber` (which only knows claude's wire format) can
//! render them WITHOUT any codex-specific branches in the iOS code: every
//! codex tool item is remapped to the equivalent claude tool name on the
//! wire so iOS's toolSummary table picks it up as if it were a claude
//! tool_use. Single seam, single source of truth.
//!
//! Frame mapping (codex → claude-shape). Tool items surface on `item.started`
//! (so they show IN-FLIGHT — codex populates input fields at
//! `status:in_progress`); the matching `item.completed` is deduped away
//! (`dedup_item`), acting only as a fallback when no usable started arrived:
//!  - `thread.started`     → `SessionIdHarvested` only (no Frame; matches claude's init-skip)
//!  - `turn.started`       → drop (no claude analog)
//!  - command_execution    → claude `Bash` `{command}`
//!  - file_change          → one Frame per change: add→`Write`, update→`Edit`,
//!    delete→`Bash {command: "rm <path>"}` (no claude delete tool; `rm` is what
//!    claude itself emits)
//!  - web_search           → claude `WebSearch` `{query}`
//!  - mcp_tool_call        → claude `mcp__<server>__<tool>`, codex's raw
//!    `arguments` object as input (`mcp__server__tool` IS claude's own MCP naming)
//!  - collab_tool_call     → `spawn_agent` w/ prompt → claude `Agent` `{description}`;
//!    other collab subtools (or `spawn_agent` w/o prompt) → collab tool name verbatim
//!  - agent_message        → Frame on `item.completed` only (text empty at start):
//!    claude-shape assistant text envelope
//!  - `turn.completed`     → Frame (result envelope) + Result (context tokens are
//!    sourced post-turn from the rollout, not this frame)
//!  - `turn.failed`        → Frame (error result envelope) + Result
//!  - anything else        → forwarded as-is (defensive against codex format
//!    drift; iOS will likely drop, but we avoid silently losing the line)
//!
//! Codex's actual wire format observed via `codex exec --json` on
//! codex-cli 0.133.0: each turn starts with `thread.started` carrying a
//! UUIDv7 thread_id, ends with `turn.completed` carrying a `usage` block
//! ({input_tokens, cached_input_tokens, output_tokens, reasoning_output_tokens}).
//! That `usage.input_tokens` is cumulative, not live occupancy, so context
//! tokens come post-turn from the rollout instead — see
//! `read_rollout_last_context_tokens`. Item payload shapes are pinned by
//! `codex-rs/exec/src/exec_events.rs` (CommandExecutionItem, FileChangeItem,
//! WebSearchItem).
//!
//! Also hosts the install/auth `probe()` for codex (free function, not on
//! the `AgentAdapter` trait — `dyn AgentAdapter` can't dispatch statics).
//! For "authenticated" we stat `~/.codex/auth.json` (chatgpt SSO writes its
//! token blob there); missing file → not authenticated. Don't shell out —
//! `codex` has no `status`-style sub-command stable enough to lean on
//! across versions (TODO: revisit once codex ships one).
//!
//! `import()` walks Codex's on-disk rollouts at
//! `~/.codex/sessions/YYYY/MM/DD/rollout-<ISO8601>-<uuid>.jsonl` (one file per
//! session) and emits PutProject/PutChat/PutMessage like the claude / cursor
//! importers. See the importer section at the bottom of this file for the
//! line→frame mapping and the persisted-tool-name map.

use std::path::PathBuf;

use anyhow::Result;
use serde_json::{json, Value};
use smallvec::SmallVec;
use tokio::sync::mpsc;
use tracing::{debug, info};
use uuid::Uuid;

use crate::adapter::{
    claude_assistant_text_envelope, claude_tool_use_envelope, file_nonempty, parse_json_obj,
    probe_with_blocking_auth, shell_escape, AdapterDescriptor, AgentAdapter, AgentEvent, AgentKind,
    ImportProgress, TurnContext, MAX_STREAM_FRAME_BYTES, PRUNE_CONTEXT_INSTRUCTION_CODEX,
};
use crate::writer::WriteEvent;

use crate::adapters::import_shared::{
    basename_or, collapse_title, emit_chat, mint_project_id, parse_rfc3339_utc, user_message_body,
    ImportedChat, ImportedMessage, ProgressThrottle,
};
#[cfg(test)]
use crate::envelope::MessageEnvelope;
use anyhow::Context;
use chrono::{DateTime, Utc};
use std::collections::{BTreeMap, HashSet};
use std::io::ErrorKind;
use std::path::Path;
use tokio::io::{AsyncBufReadExt, BufReader};
use tracing::warn;

/// Wired into `adapter::ADAPTERS`. See `adapter::AdapterDescriptor` for the
/// shape; the `probe` / `import` slots are filled by `_boxed` wrappers below
/// the `probe()` / `import()` definitions in this file.
///
/// `installed_col` / `authenticated_col` follow the same per-kind boolean
/// pair as `claude_code_*` (migration 0022) and `cursor_*` (0033); the
/// matching `codex_*` columns landed in migration 0037. Cross-version
/// compatibility: pre-codex spawners never PATCH these columns, and
/// pre-0037 backends will reject a PATCH that does — keep both directions
/// in mind when bumping the wire.
pub const DESCRIPTOR: AdapterDescriptor = AdapterDescriptor {
    kind: AgentKind::Codex,
    wire_name: "codex",
    installed_col: "codex_installed",
    authenticated_col: "codex_authenticated",
    make: make_boxed,
    probe: probe_boxed,
    import: import_boxed,
    prune: Some(PRUNE_OPS),
};

fn make_boxed() -> Box<dyn AgentAdapter> {
    Box::new(CodexAdapter::new())
}

/// Per-turn state for the codex adapter. Carries the thread id harvested from
/// the `thread.started` frame so `post_turn_context_tokens` can locate the
/// session's rollout even on the FIRST turn (where the spawner hasn't persisted
/// `chats.agent_session_id` yet, so the turn context's `agent_session_id` is
/// `None`). On resume turns the harvested value and the turn context's id agree.
///
/// `emitted_item_ids` dedupes tool bubbles across the `item.started` /
/// `item.completed` pair: we surface tool items (command/file_change/web_search)
/// on `item.started` so they show in-flight, then suppress the matching
/// `item.completed` so the same id isn't rendered twice. A fresh adapter is
/// built per turn (`agent.rs`), so item ids never collide across turns.
#[derive(Default)]
pub struct CodexAdapter {
    session_id: Option<String>,
    emitted_item_ids: HashSet<String>,
}

impl CodexAdapter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Dedupe a rendered item by its `item.id` across the started/completed
    /// pair: returns `frames` only the FIRST non-empty render for an id (then
    /// records it); later sightings return empty. An EMPTY render is never
    /// recorded, so an item whose `item.started` carried no usable payload still
    /// renders from its `item.completed`. Items without an id bypass dedup.
    fn dedup_item(&mut self, obj: &Value, frames: Vec<String>) -> Vec<String> {
        if frames.is_empty() {
            return frames;
        }
        let id = obj
            .get("item")
            .and_then(|i| i.get("id"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if id.is_empty() {
            return frames;
        }
        if self.emitted_item_ids.contains(id) {
            return Vec::new();
        }
        self.emitted_item_ids.insert(id.to_string());
        frames
    }
}

impl AgentAdapter for CodexAdapter {
    fn kind(&self) -> AgentKind {
        AgentKind::Codex
    }

    fn prepare_command(&mut self, ctx: &TurnContext<'_>) -> Result<String> {
        let mut cmd = String::new();
        if let Some(pp) = ctx.project_path {
            cmd.push_str(&format!("cd {} && ", shell_escape(pp)));
        }
        // Subcommand selection. First turn uses `codex exec`; follow-up turns
        // use `codex exec resume <thread_id>`. `--json --skip-git-repo-check`
        // work on both. Flag availability differs across the two:
        //   - `-s` (sandbox) and `-C` (cwd) ONLY exist on the first-turn
        //     `codex exec` — they are locked at session creation, so
        //     `codex exec resume` rejects them with `unexpected argument`.
        //   - `-a` / `--ask-for-approval` does NOT exist on `codex exec` at
        //     all (only on the top-level `codex` interactive command). Approval
        //     policy on `exec` is set via `-c approval_policy=never` (config
        //     override, allowed on BOTH first turn and resume) or folded into
        //     `--dangerously-bypass-approvals-and-sandbox` (also allowed on
        //     both).
        let is_resume = ctx.agent_session_id.is_some();
        cmd.push_str("codex exec");
        if let Some(sid) = ctx.agent_session_id {
            cmd.push_str(&format!(" resume {}", shell_escape(sid)));
        }
        cmd.push_str(" --json --skip-git-repo-check");

        // Sender's `machine_users.is_sandboxed`. Non-sandboxed =
        // `--dangerously-bypass-approvals-and-sandbox` (no sandbox at all +
        // skip approval prompts in one flag), mirroring claude's
        // `--dangerously-skip-permissions` and cursor's `--force` — the
        // BYO-spawner trust model is "the owner accepts what their agent
        // does on their machine". Sandboxed = `read-only` sandbox (first
        // turn only — locked at session creation) plus `-c approval_policy=
        // never` every turn so a refusal (or any prompt) surfaces as an
        // error frame instead of hanging on a TTY.
        if ctx.is_sandboxed {
            cmd.push_str(" -c approval_policy=never");
            if !is_resume {
                cmd.push_str(" -s read-only");
                if let Some(pp) = ctx.project_path {
                    cmd.push_str(&format!(" -C {}", shell_escape(pp)));
                }
            }
        } else {
            cmd.push_str(" --dangerously-bypass-approvals-and-sandbox");
            if !is_resume {
                if let Some(pp) = ctx.project_path {
                    cmd.push_str(&format!(" -C {}", shell_escape(pp)));
                }
            }
        }

        // TODO(codex): worktree=true is ignored for v1. Codex has no
        // first-class worktree flag; a follow-up pass will either `git
        // worktree add` upstream of this and pass `-C <worktree>` in
        // place of `project_path`, or refuse the toggle with a clearer
        // signal back to the UI. Today we just spawn in the project root.
        let _ = ctx.worktree;

        // TODO(codex): no system-prompt injection in v1. Codex has no
        // `--append-system-prompt` equivalent — the options are (a) write
        // an `AGENTS.md` in cwd, which clobbers the user's existing one
        // and persists across turns even after this spawner exits, or
        // (b) pass `-c instructions="..."` to override the global codex
        // instructions, which loses the user's own instructions for the
        // turn. Neither is acceptable as a default. ATTACH_FILE_INSTRUCTION
        // is therefore NOT injected — the agent-side attach-file flow is
        // unavailable on codex until this is resolved.
        //
        // prune-context: no `--append-system-prompt`, so the nudge rides in on the
        // first user message via `first_turn_prompt_suffix` (see
        // `AgentAdapter::first_turn_prompt_suffix`). ATTACH_FILE_INSTRUCTION can't
        // reuse it — attach-file needs the nudge every turn, and per-turn appends
        // would pollute the rollout — so it stays deferred above.

        // Verbatim pass-through of `chats.model` (migration 0035). Codex
        // uses `-m, --model`; the model label drifts per-release so we
        // don't validate it locally — an invalid value surfaces as a
        // codex error frame in the chat.
        if let Some(model) = ctx.model {
            cmd.push_str(&format!(" --model {}", shell_escape(model)));
        }

        // Prompt is read from stdin. `-` as the positional argument tells
        // codex to consume stdin; the shell `<` redirect feeds it the
        // prompt file the supervisor wrote.
        cmd.push_str(&format!(
            " - < {}",
            shell_escape(&ctx.prompt_file.to_string_lossy())
        ));

        Ok(cmd)
    }

    /// Codex has no `--append-system-prompt`, so the prune nudge rides in on the
    /// first user message instead (the Supervisor appends it on the first turn
    /// only). Uses the codex-specific variant: codex reads/greps/edits run through
    /// the shell (mapped to `Bash`), so its example targets `Bash`, not `Read`.
    /// See `AgentAdapter::first_turn_prompt_suffix`.
    fn first_turn_prompt_suffix(&self) -> Option<&'static str> {
        Some(PRUNE_CONTEXT_INSTRUCTION_CODEX)
    }

    /// Read live context-window occupancy from the rollout once the `codex exec`
    /// process has exited (each turn is a fresh, short-lived process, so the
    /// rollout is closed and flushed by then). The occupancy isn't in the `--json`
    /// stream — see `read_rollout_last_context_tokens`. Prefer the id harvested
    /// this turn (`thread.started`), falling back to the resume id from the turn
    /// context. Returns `None` (suppressing the PATCH) when the rollout can't be
    /// located or read, leaving the prior gauge value in place rather than zeroing.
    fn post_turn_context_tokens(&self, agent_session_id: Option<&str>) -> Option<i64> {
        let sid = self.session_id.as_deref().or(agent_session_id)?;
        // Same base-dir resolution as the prune path (honors CODEX_HOME, else
        // $HOME/.codex) — `find_codex_rollout` searches `<base>/sessions/**`.
        let base = AgentKind::Codex.cli_home()?;
        let path = find_codex_rollout(&base, sid)?;
        read_rollout_last_context_tokens(&path)
    }

    fn handle_line(&mut self, line: String) -> SmallVec<[AgentEvent; 2]> {
        let mut out: SmallVec<[AgentEvent; 2]> = SmallVec::new();

        // Oversize-frame guard. `serde_json::from_str` on a multi-MB line
        // allocates a tree of `Map<String, Value>` nodes; the per-item
        // dispatch below needs the parsed tree, so without this guard a
        // single big shell output (command_execution often embeds full
        // stdout) can churn the heap by megabytes. For lines past the cap
        // we forward verbatim as a Frame — iOS will likely fail to render
        // it cleanly, but at least the content reaches the chat instead
        // of disappearing. Mirrors the pattern in `claude.rs::handle_line`.
        if line.len() > MAX_STREAM_FRAME_BYTES {
            out.push(AgentEvent::Frame(line));
            return out;
        }

        let Some(obj) = parse_json_obj(&line) else {
            // Non-JSON line: forward as-is so spawner-side noise (or a
            // codex pre-banner) still surfaces. Mirrors claude's
            // permissive non-JSON branch.
            out.push(AgentEvent::Frame(line));
            return out;
        };
        let Some(ty) = obj.get("type").and_then(|v| v.as_str()) else {
            // Object without a `type` — defensive forward.
            out.push(AgentEvent::Frame(line));
            return out;
        };

        match ty {
            "thread.started" => {
                // Harvest thread id → `chats.agent_session_id`; drop the
                // frame so it never reaches the chat (matches claude's
                // init-skip).
                if let Some(sid) = obj.get("thread_id").and_then(|v| v.as_str()) {
                    // Stash it so `post_turn_context_tokens` can find this
                    // session's rollout after the process exits, even on the
                    // first turn (before the id is persisted to chats).
                    self.session_id = Some(sid.to_string());
                    out.push(AgentEvent::SessionIdHarvested(sid.to_string()));
                } else {
                    debug!("codex thread.started without thread_id");
                }
            }
            "turn.started" => {
                // No claude analog — drop.
                debug!("codex turn.started dropped");
            }
            "item.started" => {
                // Surface tool items in-flight (codex populates input fields at
                // `status:in_progress`); the matching `item.completed` is deduped
                // below. `agent_message` is skipped here (text empty at start).
                let frames = self.dedup_item(&obj, normalize_item_started(&obj));
                for frame in frames {
                    out.push(AgentEvent::Frame(frame));
                }
            }
            "item.completed" => {
                // `agent_message` renders here (text final now); tool items already
                // surfaced on `item.started` are deduped, but one with no usable
                // started still renders here as a fallback.
                // Call-keyed prune cue: fire ONLY when the completed command is
                // the `prune-context` call itself, matched on its own `command`
                // (present on the completed item). A sibling shell call's
                // completion in the same batch must NOT drive the apply — it
                // would abort→respawn before this call's `function_call_output`
                // is flushed to the rollout, so the resumed agent re-runs the
                // prune. No cross-frame state needed: the command is right here.
                let is_prune_command = item_parts(&obj)
                    .filter(|(_, t, _)| *t == "command_execution")
                    .and_then(|(item, _, _)| item.get("command"))
                    .is_some_and(crate::prune::value_is_prune_context_call);
                let frames = self.dedup_item(&obj, normalize_item_completed(&obj));
                if frames.is_empty() {
                    debug!("codex item.completed: no new renderable item (dropped or already surfaced on item.started)");
                }
                for frame in frames {
                    out.push(AgentEvent::Frame(frame));
                }
                // The `prune-context` command's completion is codex's "tool
                // result persisted" signal — its `function_call_output` is now in
                // the rollout. Emit the content-free `ToolResult` cue (AFTER any
                // visible frame, so a restart never preempts the item's own
                // bubble) so the main loop applies the queued prune strictly after
                // the result landed (the mechanism claude/gemini use too). No-op
                // unless a `PruneRequest` is pending. Codex emits no standalone
                // tool_result frame, so this completion is the only per-tool
                // boundary.
                if is_prune_command {
                    out.push(AgentEvent::ToolResult);
                }
            }
            "turn.completed" => {
                // Deliberately NO ContextTokens here: the frame's
                // `usage.input_tokens` is cumulative, not live occupancy. The
                // gauge is sourced post-turn from the rollout instead — see
                // `read_rollout_last_context_tokens` / `post_turn_context_tokens`.
                //
                // Emit claude-shape result envelope so iOS's describer
                // renders the `[result: success]` line.
                out.push(AgentEvent::Frame(normalize_turn_completed_frame()));
                // Emit Result on every turn.completed; the supervisor
                // latches it (so AgentResponse::Done.has_result is set
                // once and only once).
                out.push(AgentEvent::Result);
            }
            "turn.failed" => {
                // Codex emits a terminal failure frame instead of a claude
                // `result` frame. Normalize it to the same result shape iOS
                // already renders and emit the Result marker so the
                // supervisor does not append the generic interrupted line.
                out.push(AgentEvent::Frame(normalize_turn_failed_frame(&obj)));
                out.push(AgentEvent::Result);
            }
            other => {
                // Defensive forward — codex format drift shouldn't silently
                // drop content. iOS will fall through to its "unknown
                // frame" branch (rendered as raw text), which is the right
                // failure mode while we observe the drift.
                debug!("codex unknown frame type, forwarding: {}", other);
                out.push(AgentEvent::Frame(line));
            }
        }

        out
    }
}

/// Read the live context-window occupancy from a codex rollout: the
/// `last_token_usage.input_tokens` of the LAST `token_count` record in the file.
///
/// Codex writes a `token_count` `event_msg` after each model round-trip:
/// `{"type":"event_msg","payload":{"type":"token_count","info":{
///    "total_token_usage":{...}, "last_token_usage":{"input_tokens":N,...},
///    "model_context_window":W}}}`. `total_token_usage` is the cumulative sum
/// across the whole thread (what the `--json` stream's `turn.completed.usage`
/// echoes); `last_token_usage` is just the final round-trip's prompt — i.e. the
/// current context occupancy, bounded by `model_context_window`. We want the
/// latter, taken from the LAST such record (the most recent round-trip).
///
/// Scans the whole file because the final `token_count` is near (but not
/// guaranteed at) EOF — codex may append a few non-token records after it. A
/// rollout is at most a few MB and this runs once per turn, off the hot path.
/// Returns `None` if the file is unreadable or carries no `token_count` record,
/// so the caller leaves the prior gauge value untouched.
fn read_rollout_last_context_tokens(path: &Path) -> Option<i64> {
    let text = std::fs::read_to_string(path).ok()?;
    let mut last: Option<i64> = None;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(entry) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        // `event_msg` wrapper → `payload.type == "token_count"` →
        // `payload.info.last_token_usage.input_tokens`.
        let info = entry
            .get("payload")
            .filter(|p| p.get("type").and_then(|t| t.as_str()) == Some("token_count"))
            .and_then(|p| p.get("info"));
        let Some(info) = info else {
            continue;
        };
        if let Some(n) = info
            .get("last_token_usage")
            .and_then(|u| u.get("input_tokens"))
            .and_then(|v| v.as_i64())
        {
            last = Some(n);
        }
    }
    last
}

/// Extract `(item, item_type, item_id)` from an `item.*` frame, or `None` when
/// the frame has no typed `item`. `item_id` defaults to `""` when absent (the
/// dedup + per-change-id logic tolerate an empty id).
fn item_parts(obj: &Value) -> Option<(&Value, &str, String)> {
    let item = obj.get("item")?;
    let item_type = item.get("type").and_then(|v| v.as_str())?;
    let item_id = item
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    Some((item, item_type, item_id))
}

/// Render the codex TOOL items (input field populated at `status:in_progress`) →
/// claude-shape `tool_use` envelopes, so they surface in-flight on `item.started`.
/// `agent_message` is absent (text empty at start; renders on `item.completed`);
/// unknown types → empty. A render with an empty primary field returns empty so
/// the started/completed dedup falls through to the completed frame (no empty
/// bubble that then suppresses the real one). `file_change` is the only multi-
/// frame case — codex bundles N edits in one item, fanned out one bubble each.
///
/// Mapping (see file-level doc for the full table). Field projection only — we
/// drop codex-side metadata (aggregated_output, exit_code, action variants, …)
/// since iOS only summarizes the primary input field; matching claude's wire
/// shape means no codex-specific branches in iOS.
fn normalize_item_started(obj: &Value) -> Vec<String> {
    let Some((item, item_type, item_id)) = item_parts(obj) else {
        return Vec::new();
    };
    match item_type {
        "command_execution" => {
            let command = item.get("command").and_then(|v| v.as_str()).unwrap_or("");
            // Empty → fall through to the completed frame (see fn doc).
            if command.is_empty() {
                return Vec::new();
            }
            vec![claude_tool_use_envelope(
                &item_id,
                "Bash",
                json!({ "command": command }),
            )]
        }
        "file_change" => {
            let Some(changes) = item.get("changes").and_then(|v| v.as_array()) else {
                return Vec::new();
            };
            changes
                .iter()
                .enumerate()
                .filter_map(|(idx, change)| {
                    let path = change.get("path").and_then(|v| v.as_str())?.to_string();
                    let kind = change
                        .get("kind")
                        .and_then(|v| v.as_str())
                        .unwrap_or("update");
                    // Per-change unique id so iOS can key its row diffing
                    // off `tool_use.id` without collisions when N>1 (claude
                    // itself mints unique ids per tool_use).
                    let id = if changes.len() == 1 {
                        item_id.clone()
                    } else {
                        format!("{}.{}", item_id, idx)
                    };
                    Some(match kind {
                        "add" => {
                            claude_tool_use_envelope(&id, "Write", json!({ "file_path": path }))
                        }
                        "delete" => claude_tool_use_envelope(
                            &id,
                            "Bash",
                            json!({ "command": format!("rm {}", path) }),
                        ),
                        // "update" + defensive fallback for any future kind
                        // we haven't seen — Edit conveys "an existing file
                        // was changed", which is the best generic match.
                        _ => claude_tool_use_envelope(&id, "Edit", json!({ "file_path": path })),
                    })
                })
                .collect()
        }
        "web_search" => {
            let query = item.get("query").and_then(|v| v.as_str()).unwrap_or("");
            // Empty → fall through to the completed frame (see fn doc).
            if query.is_empty() {
                return Vec::new();
            }
            vec![claude_tool_use_envelope(
                &item_id,
                "WebSearch",
                json!({ "query": query }),
            )]
        }
        "mcp_tool_call" => render_mcp_tool_call(item, &item_id),
        "collab_tool_call" => render_collab_tool_call(item, &item_id),
        // agent_message: text not final at start. Unknown: drop.
        _ => Vec::new(),
    }
}

/// codex `mcp_tool_call` item → claude-shape `tool_use` named
/// `mcp__<server>__<tool>` — claude's own MCP naming, so iOS's default summary
/// branch renders the qualified name with no codex-specific branching.
///
/// The exec stream's `arguments` is a RAW JSON object, NOT the JSON-encoded STRING
/// the PERSISTED `function_call.arguments` carries, so we pass it through under
/// `input` (no decode). Degrades when a name part is missing: with one present we
/// emit the best qualified name (`mcp__<server>` / `mcp__<tool>`); with neither we
/// return empty so the completed fallback fires instead of a garbage `mcp__`.
fn render_mcp_tool_call(item: &Value, item_id: &str) -> Vec<String> {
    let server = item.get("server").and_then(|v| v.as_str()).unwrap_or("");
    let tool = item.get("tool").and_then(|v| v.as_str()).unwrap_or("");
    let name = match (server.is_empty(), tool.is_empty()) {
        (false, false) => format!("mcp__{server}__{tool}"),
        (false, true) => format!("mcp__{server}"),
        (true, false) => format!("mcp__{tool}"),
        (true, true) => return Vec::new(),
    };
    // Raw JSON object, not an encoded string — forward verbatim. Absent → `{}`.
    let input = item.get("arguments").cloned().unwrap_or_else(|| json!({}));
    vec![claude_tool_use_envelope(item_id, &name, input)]
}

/// codex `collab_tool_call` item (sub-agent orchestration: `spawn_agent`,
/// `send_input`, `wait`, `close_agent`) → claude-shape `tool_use`.
///
/// A `spawn_agent` with a `prompt` → claude's `Agent` `{description}` (iOS's
/// `toolSummary` reads `Agent`→`description`, showing the sub-agent's task). Any
/// other subtool (or `spawn_agent` w/o prompt) passes the `tool` name through
/// verbatim with its routing fields under `input` (iOS renders tool-name-only).
/// Empty/absent `tool` → empty so the completed fallback fires.
fn render_collab_tool_call(item: &Value, item_id: &str) -> Vec<String> {
    let tool = item.get("tool").and_then(|v| v.as_str()).unwrap_or("");
    if tool.is_empty() {
        return Vec::new();
    }
    let prompt = item.get("prompt").and_then(|v| v.as_str());
    if tool == "spawn_agent" {
        if let Some(desc) = prompt.filter(|p| !p.is_empty()) {
            return vec![claude_tool_use_envelope(
                item_id,
                "Agent",
                json!({ "description": desc }),
            )];
        }
    }
    // Pass-through: keep the collab subtool name; surface the thread routing +
    // prompt under input so the stored frame stays useful (iOS renders the bare
    // tool name). Drop `agents_states` (verbose, not summarized by iOS).
    let mut input = serde_json::Map::new();
    if let Some(p) = prompt {
        input.insert("prompt".to_string(), json!(p));
    }
    if let Some(s) = item.get("sender_thread_id") {
        input.insert("sender_thread_id".to_string(), s.clone());
    }
    if let Some(r) = item.get("receiver_thread_ids") {
        input.insert("receiver_thread_ids".to_string(), r.clone());
    }
    vec![claude_tool_use_envelope(
        item_id,
        tool,
        Value::Object(input),
    )]
}

/// Render an `item.completed` frame. `agent_message`'s `text` is final only here,
/// so it renders now. Tool items reuse `normalize_item_started`'s renderers as a
/// FALLBACK (one whose `item.started` was absent/empty still gets a bubble);
/// `dedup_item` at the call site suppresses the common already-surfaced case.
fn normalize_item_completed(obj: &Value) -> Vec<String> {
    let Some((item, item_type, _item_id)) = item_parts(obj) else {
        return Vec::new();
    };
    if item_type == "agent_message" {
        // Context tokens are sourced post-turn from the rollout, not here. The
        // shared envelope helper stamps the claude-shape zero `usage` so iOS sees
        // the same key set across adapters mid-turn.
        let text = item.get("text").and_then(|v| v.as_str()).unwrap_or("");
        return vec![claude_assistant_text_envelope(text)];
    }
    // Tool items: same renderers as the started path; deduped by id at the call site.
    normalize_item_started(obj)
}

/// Builds the claude-shape result envelope emitted on `turn.completed`. We
/// don't carry codex's per-turn duration through. iOS's describer renders
/// this as `[result: success]`.
fn normalize_turn_completed_frame() -> String {
    json!({
        "type": "result",
        "subtype": "success",
        "is_error": false,
    })
    .to_string()
}

/// Builds the claude-shape result envelope emitted on `turn.failed`.
/// Upstream Codex shape is:
/// `{"type":"turn.failed","error":{"message":"..."}}`.
/// iOS only uses `subtype` for the visible terminator (`[result: error]`),
/// but preserving `error.message` keeps the stored frame useful for logs and
/// future clients without leaking the raw Codex event shape into chat rows.
fn normalize_turn_failed_frame(obj: &Value) -> String {
    let message = obj
        .get("error")
        .and_then(|e| e.get("message"))
        .and_then(|v| v.as_str())
        .unwrap_or("turn failed");
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

/// Probe install + auth state in one go. Returns `(installed, authenticated)`
/// — the writer flattens both pairs into a single PATCH on `machines`'s
/// per-kind boolean columns. Pure filesystem check, no shell-out — codex
/// has no stable `status`-style sub-command we can lean on across versions
/// (TODO: revisit once it ships one). The shared
/// `adapter::probe_with_blocking_auth` helper takes care of the
/// `binary_on_path` + `spawn_blocking` boilerplate.
pub async fn probe() -> (bool, bool) {
    probe_with_blocking_auth("codex", is_authenticated).await
}

/// `fn`-pointer-shaped wrapper around `probe()` for `AdapterDescriptor.probe`.
/// `BoxFuture` erases the concrete async-fn type so the descriptor can hold
/// all adapters' probes in a single slice.
fn probe_boxed() -> futures::future::BoxFuture<'static, (bool, bool)> {
    Box::pin(probe())
}

/// Codex stores ChatGPT-SSO auth state in `~/.codex/auth.json`. The exact
/// schema isn't documented as a stable contract, so we only check for
/// presence + non-empty — anything else risks a false negative on a schema
/// rename. False positives here just over-report "codex ready" in the UI,
/// which is acceptable for v1 (the actual spawn will surface an auth error
/// frame if the file is stale).
fn is_authenticated() -> bool {
    let Some(home) = std::env::var_os("HOME").map(PathBuf::from) else {
        return false;
    };
    let auth = home.join(".codex").join("auth.json");
    file_nonempty(&auth)
}

// ===========================================================================
// One-shot codex-history importer. Walks `~/.codex/sessions/**/rollout-*.jsonl`
// recursively (one file per session), groups by the session's `cwd` so each
// project emits one PutProject before its chats, and emits
// PutProject/PutChat/PutMessage events shaped identically to the claude /
// cursor importer output. Status emission lives in the dispatcher in
// `main.rs`; this fn only reports raw 0..=100 progress via `progress`.
//
// Idempotent: project ids are UUIDv5(machine_id || cwd) via the SAME
// `mint_project_id` namespace as claude/cursor (so a project with transcripts
// from multiple CLIs collapses to one row); chat ids are the session UUID
// (`session_meta.payload.id`, also in the filename), so re-runs reconverge.
//
// Line→frame mapping (current format; see file-level doc for the live wire):
//   session_meta            → harvest `payload.id` (chat id) + `payload.cwd`
//                             (project) + `payload.timestamp` (chat created_at)
//   event_msg/user_message  → real typed prompt → MessageEnvelope (sender "user")
//   event_msg/agent_message → final assistant text → claude_assistant_text_envelope
//   response_item/function_call → claude_tool_use_envelope (persisted-tool map)
//   everything else (response_item message/reasoning, function_call_output,
//   token_count, task_started/complete, turn_context, developer msgs) → drop.
//
// Double-emit avoidance: codex carries the same assistant text BOTH in
// `event_msg/agent_message` AND `response_item`/role:assistant output_text,
// and the same user prompt in `event_msg/user_message` AND
// `response_item`/role:user input_text (alongside the injected
// `<user_instructions>`/`<environment_context>` developer dumps). We source
// BOTH user and assistant TEXT exclusively from `event_msg`, which also
// naturally filters the injected response_item user messages — only tool calls
// come from `response_item`. If a future codex build stops emitting event_msg
// text (a non-tool-using turn that only has response_item rows), we fall back
// to response_item message text (still filtering the injected dumps).
//
// SKIP-NO-CWD: legacy 2025 bare-head files (line 1 = `{id,timestamp,instructions}`
// with no `type`/`cwd`) and any session_meta lacking `cwd` are skipped entirely
// — a chat with no project folder can't be opened or resumed in Zucchini.

pub(crate) async fn import(
    machine_id: Uuid,
    user_id: Uuid,
    write_tx: mpsc::Sender<WriteEvent>,
    progress: ImportProgress,
) -> Result<()> {
    let Some(sessions_dir) = codex_sessions_dir() else {
        info!("HOME not set, skipping codex history import");
        progress(100).await;
        return Ok(());
    };
    info!(path = %sessions_dir.display(), "scanning codex rollouts");

    // Recursively collect every `rollout-*.jsonl` under the sessions dir. The
    // year/month/day nesting is just for the operator's benefit; we flatten it
    // and read each file's `session_meta.cwd` to group by project.
    let mut files: Vec<PathBuf> = Vec::new();
    match collect_rollouts(&sessions_dir, &mut files) {
        Ok(()) => {}
        Err(e) if e.kind() == ErrorKind::NotFound => {
            info!(path = %sessions_dir.display(), "no ~/.codex/sessions, nothing to import");
            progress(100).await;
            return Ok(());
        }
        Err(e) => {
            return Err(anyhow::Error::new(e))
                .with_context(|| format!("walk {}", sessions_dir.display()));
        }
    }

    if files.is_empty() {
        info!("no codex rollout files found");
        progress(100).await;
        return Ok(());
    }
    files.sort();
    let total_files = files.len();
    info!(files = total_files, "starting codex import");

    // Group the parsed sessions by cwd so each project emits one PutProject
    // before its chats. We parse every file once up-front (cheap: each rollout
    // is a few hundred small JSON lines), bucket by cwd, then emit per-project.
    // A session with no cwd is skipped (see SKIP-NO-CWD above).
    let mut by_project: BTreeMap<String, Vec<ParsedSession>> = BTreeMap::new();
    let mut skipped_no_cwd = 0usize;
    let mut done_files = 0usize;
    let mut throttle = ProgressThrottle::new();
    for path in &files {
        match parse_session(path).await {
            Ok(Some(session)) => {
                by_project
                    .entry(session.cwd.clone())
                    .or_default()
                    .push(session);
            }
            Ok(None) => {
                skipped_no_cwd += 1;
            }
            Err(e) => {
                warn!(file = %path.display(), error = %e, "codex session parse failed, skipping");
            }
        }
        done_files += 1;
        // Per-percent throttle shared with every importer; see `ProgressThrottle`.
        throttle.step(done_files, total_files, &progress).await;
    }

    if skipped_no_cwd > 0 {
        info!(
            count = skipped_no_cwd,
            "codex: skipped sessions with no cwd (legacy bare-head files or session_meta without cwd)"
        );
    }

    info!(
        projects = by_project.len(),
        "codex: emitting parsed sessions"
    );

    for (cwd, mut sessions) in by_project {
        // Deterministic order across re-runs and oldest→newest in the UI.
        sessions.sort_by_key(|s| s.created_at);
        let project_id = mint_project_id(machine_id, &cwd);
        let project_name = basename_or(&cwd, "project");
        let _ = write_tx
            .send(WriteEvent::PutProject {
                id: project_id,
                machine_id,
                name: project_name,
                path: cwd.clone(),
            })
            .await;

        for session in sessions {
            emit_chat(
                &write_tx,
                user_id,
                ImportedChat {
                    id: session.chat_id,
                    project_id,
                    title: session.title,
                    created_at: session.created_at,
                    messages: session.messages,
                },
            )
            .await;
        }
    }

    info!("codex history import complete");
    Ok(())
}

/// `~/.codex/sessions`, or `None` when HOME is unset.
fn codex_sessions_dir() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    Some(PathBuf::from(home).join(".codex").join("sessions"))
}

/// Recursively collect `rollout-*.jsonl` files under `dir` into `out`. Codex
/// nests sessions under `YYYY/MM/DD/`, but an older flat layout
/// (`~/.codex/sessions/rollout-*.json[l]`) may coexist, so we walk the whole
/// subtree rather than assuming the date nesting. Returns the first
/// `NotFound` so the caller can early-out exactly like claude's missing-dir
/// branch; other IO errors on nested dirs are logged and skipped.
pub(crate) fn collect_rollouts(dir: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    let entries = std::fs::read_dir(dir)?;
    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                warn!(error = %e, dir = %dir.display(), "skipping unreadable codex sessions entry");
                continue;
            }
        };
        let path = entry.path();
        let file_type = match entry.file_type() {
            Ok(t) => t,
            Err(_) => continue,
        };
        if file_type.is_dir() {
            // Nested IO errors (permissions on a stray dir) shouldn't abort the
            // whole walk — log + continue.
            if let Err(e) = collect_rollouts(&path, out) {
                warn!(error = %e, dir = %path.display(), "skipping unreadable codex sessions subdir");
            }
        } else if file_type.is_file() {
            let is_rollout = path
                .file_name()
                .and_then(|s| s.to_str())
                .map(|n| n.starts_with("rollout-") && n.ends_with(".jsonl"))
                .unwrap_or(false);
            if is_rollout {
                out.push(path);
            }
        }
    }
    Ok(())
}

/// A fully-parsed codex session ready to emit. `cwd` is guaranteed non-empty
/// (sessions without one are dropped at parse time). `messages` is in file
/// order; the writer assigns `seq` on insert.
struct ParsedSession {
    chat_id: Uuid,
    cwd: String,
    title: String,
    created_at: DateTime<Utc>,
    messages: Vec<ImportedMessage>,
}

/// Codex `call_id`s are `call_<base62>`, not UUIDs, so every imported message
/// leaves `id` as `None` and the writer mints `Uuid::now_v7()`. These two
/// constructors keep the per-sender shaping (user → `MessageEnvelope` body,
/// agent → already-shaped claude frame) at the call sites that build the
/// session message list.
fn imported_user(text: String, created_at: DateTime<Utc>) -> ImportedMessage {
    ImportedMessage {
        id: None,
        sender: "user",
        body: user_message_body(&text),
        created_at,
    }
}

fn imported_agent(body: String, created_at: DateTime<Utc>) -> ImportedMessage {
    ImportedMessage {
        id: None,
        sender: "agent",
        body,
        created_at,
    }
}

/// Parse a single rollout file. Returns `Ok(None)` when the session has no
/// `cwd` (legacy bare-head file, or a `session_meta` lacking `cwd`) so the
/// caller skips it. Errors only on unreadable files.
async fn parse_session(path: &Path) -> anyhow::Result<Option<ParsedSession>> {
    let file = tokio::fs::File::open(path)
        .await
        .with_context(|| format!("open {}", path.display()))?;
    let mut lines = BufReader::new(file).lines();

    let mut chat_id: Option<Uuid> = None;
    let mut cwd: Option<String> = None;
    let mut created_at: Option<DateTime<Utc>> = None;
    let mut title: Option<String> = None;
    let mut messages: Vec<ImportedMessage> = Vec::new();
    // Fallback assistant/user text harvested from response_item rows, used only
    // when a session emitted NO event_msg text rows at all (defensive against a
    // future codex build that stops mirroring text into event_msg).
    let mut had_event_text = false;
    let mut response_text_fallback: Vec<ImportedMessage> = Vec::new();
    // First user prompt harvested from the fallback path, kept as plain text so
    // the title can use it without re-deserializing a `MessageEnvelope` body.
    let mut first_fallback_user_text: Option<String> = None;

    while let Some(line) = lines.next_line().await? {
        if line.is_empty() {
            continue;
        }
        let entry: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                warn!(error = %e, "skipping malformed codex jsonl line");
                continue;
            }
        };

        let line_ts = entry
            .get("timestamp")
            .and_then(|v| v.as_str())
            .and_then(parse_rfc3339_utc);

        let Some(ty) = entry.get("type").and_then(|v| v.as_str()) else {
            // LEGACY 2025 bare-head line (`{id,timestamp,instructions}`) has no
            // `type`. It carries no cwd, so the whole session is unimportable —
            // bail out as no-cwd.
            return Ok(None);
        };

        match ty {
            "session_meta" => {
                let payload = entry.get("payload");
                chat_id = payload
                    .and_then(|p| p.get("id"))
                    .and_then(|v| v.as_str())
                    .and_then(|s| Uuid::parse_str(s).ok());
                cwd = payload
                    .and_then(|p| p.get("cwd"))
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string());
                created_at = payload
                    .and_then(|p| p.get("timestamp"))
                    .and_then(|v| v.as_str())
                    .and_then(parse_rfc3339_utc)
                    .or(line_ts);
                // No cwd → unimportable session, skip the whole file.
                if cwd.is_none() {
                    return Ok(None);
                }
            }
            "event_msg" => {
                let payload = entry.get("payload");
                let pty = payload.and_then(|p| p.get("type")).and_then(|v| v.as_str());
                let ts = line_ts.unwrap_or_else(Utc::now);
                match pty {
                    Some("user_message") => {
                        if let Some(msg) = payload
                            .and_then(|p| p.get("message"))
                            .and_then(|v| v.as_str())
                            .filter(|s| !s.is_empty())
                        {
                            had_event_text = true;
                            if title.is_none() {
                                title = Some(collapse_title(msg));
                            }
                            messages.push(imported_user(msg.to_string(), ts));
                        }
                    }
                    Some("agent_message") => {
                        if let Some(msg) = payload
                            .and_then(|p| p.get("message"))
                            .and_then(|v| v.as_str())
                            .filter(|s| !s.is_empty())
                        {
                            had_event_text = true;
                            messages.push(imported_agent(claude_assistant_text_envelope(msg), ts));
                        }
                    }
                    // token_count / task_started / task_complete /
                    // agent_reasoning → drop.
                    _ => {}
                }
            }
            "response_item" => {
                let payload = entry.get("payload");
                let pty = payload.and_then(|p| p.get("type")).and_then(|v| v.as_str());
                let ts = line_ts.unwrap_or_else(Utc::now);
                match pty {
                    Some("function_call") => {
                        if let Some(frame) = normalize_persisted_function_call(payload.unwrap()) {
                            messages.push(imported_agent(frame, ts));
                        }
                    }
                    Some("message") => {
                        // Text is normally sourced from event_msg; collect a
                        // fallback here in case event_msg text is absent. The
                        // injected `<user_instructions>` / `<environment_context>`
                        // / developer dumps are filtered by
                        // `response_item_message_text`.
                        let role = payload
                            .and_then(|p| p.get("role"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        if let Some(text) = response_item_message_text(payload.unwrap()) {
                            match role {
                                "user" => {
                                    if first_fallback_user_text.is_none() {
                                        first_fallback_user_text = Some(text.clone());
                                    }
                                    response_text_fallback.push(imported_user(text, ts));
                                }
                                "assistant" => {
                                    response_text_fallback.push(imported_agent(
                                        claude_assistant_text_envelope(&text),
                                        ts,
                                    ));
                                }
                                // developer / system → drop.
                                _ => {}
                            }
                        }
                    }
                    // reasoning / function_call_output → drop.
                    _ => {}
                }
            }
            // turn_context → ignore. Unknown future types → drop.
            _ => {}
        }
    }

    let Some(cwd) = cwd else {
        // No session_meta with cwd was ever seen.
        return Ok(None);
    };
    let Some(chat_id) = chat_id else {
        // session_meta had a cwd but no parseable id — we can't mint a stable
        // chat id, so skip (defensive; not observed in practice).
        warn!(file = %path.display(), "codex session_meta has cwd but no UUID id, skipping");
        return Ok(None);
    };

    // Merge the response_item fallback only if the session emitted no event_msg
    // text at all (tool-only / function_call rows still landed in `messages`).
    if !had_event_text && !response_text_fallback.is_empty() {
        if title.is_none() {
            title = first_fallback_user_text.as_deref().map(collapse_title);
        }
        messages.extend(response_text_fallback);
        // Keep file order stable after the merge.
        messages.sort_by_key(|m| m.created_at);
    }

    if messages.is_empty() {
        return Ok(None);
    }

    Ok(Some(ParsedSession {
        chat_id,
        cwd,
        title: title.unwrap_or_else(|| "Imported chat".to_string()),
        created_at: created_at.unwrap_or_else(Utc::now),
        messages,
    }))
}

/// Extract the joined text of a `response_item`/`message` payload, filtering
/// codex's injected developer dumps (`<user_instructions>` /
/// `<environment_context>`). Concatenates `content[].text` for
/// `input_text`/`output_text` blocks. Returns `None` when nothing renderable
/// remains.
fn response_item_message_text(payload: &Value) -> Option<String> {
    let blocks = payload.get("content").and_then(|c| c.as_array())?;
    let mut parts: Vec<&str> = Vec::new();
    for b in blocks {
        let bt = b.get("type").and_then(|t| t.as_str());
        if matches!(bt, Some("input_text") | Some("output_text")) {
            if let Some(t) = b.get("text").and_then(|t| t.as_str()) {
                parts.push(t);
            }
        }
    }
    if parts.is_empty() {
        return None;
    }
    let joined = parts.join("\n");
    // Filter the injected AGENTS.md / permissions / environment dumps that
    // codex prepends as the first response_item user messages — they are not
    // user-typed prompts. The real prompt has no such wrapper.
    if joined.starts_with("<user_instructions>") || joined.starts_with("<environment_context>") {
        return None;
    }
    Some(joined)
}

/// Convert a codex `response_item`/`function_call` payload to a claude-shape
/// `tool_use` frame. `arguments` is a JSON-encoded STRING (e.g.
/// `"{\"cmd\":\"ls\"}"`) — we `from_str` it before re-keying, exactly like the
/// cursor importer's `rawArgs` handling. Returns `None` when there's no name.
fn normalize_persisted_function_call(payload: &Value) -> Option<String> {
    let name = payload.get("name").and_then(|v| v.as_str())?;
    if name.is_empty() {
        return None;
    }
    // `call_id` is `call_<base62>`, not a UUID — used verbatim as the
    // claude-shape `tool_use.id` (iOS keys row diffing off it; uniqueness
    // within a chat is enough).
    let call_id = payload
        .get("call_id")
        .and_then(|v| v.as_str())
        .unwrap_or("codex_persisted_unknown");
    // `arguments` is a JSON-encoded string; parse it. Missing/unparseable →
    // empty object so the bubble still renders (iOS falls back to tool-name-only).
    let args: Value = payload
        .get("arguments")
        .and_then(|v| v.as_str())
        .and_then(|s| serde_json::from_str(s).ok())
        .unwrap_or_else(|| json!({}));
    let (claude_name, input) = map_codex_persisted_tool(name, &args);
    Some(claude_tool_use_envelope(call_id, &claude_name, input))
}

/// Map a codex persisted `function_call.name` to a claude tool name + a
/// claude-shape input object. iOS's `SpawnerMessageDescriber` dispatches on
/// claude names and reads claude-named arg keys (`command`, `file_path`,
/// `pattern`, `query`, `todos`), so we rename here on the spawner side.
///
/// SEPARATE from the live adapter's `normalize_item_completed` map: the
/// persisted names are the raw OpenAI tool names (`shell`, `exec_command`,
/// `apply_patch`, `update_plan`, ...), NOT the live `item.completed` item
/// types (`command_execution`, `file_change`, `web_search`). Confirmed against
/// real rollouts on disk: `shell`, `exec_command`, `update_plan`,
/// `write_stdin`, `spawn_agent`, `wait_agent`, `close_agent`.
///
/// Known codex → claude:
///   shell           → Bash   (`command` is an ARRAY ["bash","-lc","..."] →
///                              join to a single `{command}` string)
///   exec_command    → Bash   (`cmd` string → `{command}`)
///   apply_patch     → Edit   (`{file_path}` from `path`/`file_path` if present,
///                              else the raw patch under `{command}` so the
///                              bubble still shows something)
///   update_plan     → TodoWrite (`{todos}` from the `plan` array)
///   web_search      → WebSearch (`{query}`)
///   anything else (write_stdin, spawn_agent, wait_agent, close_agent, mcp_*,
///                   future tools) → pass-through name + args verbatim; iOS
///                   falls back to the tool-name-only branch, which renders fine.
fn map_codex_persisted_tool(name: &str, args: &Value) -> (String, Value) {
    match name {
        "shell" => {
            // `args.command` is typically an array of argv tokens; join to a
            // single shell-ish string for the claude `Bash` summary. Fall back
            // to a string command if codex ever sends one.
            let command = match args.get("command") {
                Some(Value::Array(parts)) => parts
                    .iter()
                    .filter_map(|p| p.as_str())
                    .collect::<Vec<_>>()
                    .join(" "),
                Some(Value::String(s)) => s.clone(),
                _ => String::new(),
            };
            ("Bash".to_string(), json!({ "command": command }))
        }
        "exec_command" => {
            let command = args
                .get("cmd")
                .and_then(|v| v.as_str())
                .or_else(|| args.get("command").and_then(|v| v.as_str()))
                .unwrap_or("")
                .to_string();
            ("Bash".to_string(), json!({ "command": command }))
        }
        "apply_patch" => {
            // codex's apply_patch carries the edit as a unified-patch blob; the
            // target path isn't always a discrete field. Prefer an explicit
            // path key; otherwise surface the raw patch body as a Bash-ish
            // command so the bubble isn't empty.
            if let Some(path) = args
                .get("path")
                .and_then(|v| v.as_str())
                .or_else(|| args.get("file_path").and_then(|v| v.as_str()))
            {
                ("Edit".to_string(), json!({ "file_path": path }))
            } else if let Some(patch) = args
                .get("input")
                .and_then(|v| v.as_str())
                .or_else(|| args.get("patch").and_then(|v| v.as_str()))
            {
                ("Bash".to_string(), json!({ "command": patch }))
            } else {
                ("Edit".to_string(), args.clone())
            }
        }
        "update_plan" => {
            // `plan` array → claude `TodoWrite`'s `todos`. iOS reads
            // `input.todos`; we forward the array verbatim under that key.
            let todos = args.get("plan").cloned().unwrap_or_else(|| json!([]));
            ("TodoWrite".to_string(), json!({ "todos": todos }))
        }
        "web_search" | "web_search_call" => {
            let query = args
                .get("query")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            ("WebSearch".to_string(), json!({ "query": query }))
        }
        // Pass-through: write_stdin, spawn_agent, wait_agent, close_agent,
        // mcp_*, and anything we haven't seen. iOS renders the tool-name-only
        // branch.
        other => (other.to_string(), args.clone()),
    }
}

/// `fn`-pointer-shaped wrapper around `import()` for `AdapterDescriptor.import`.
/// Same boxing trick as `probe_boxed` above — erases the async-fn return
/// type so all adapters' imports share one `fn` signature in the descriptor
/// slice.
fn import_boxed(
    machine_id: Uuid,
    user_id: Uuid,
    write_tx: mpsc::Sender<WriteEvent>,
    progress: ImportProgress,
) -> futures::future::BoxFuture<'static, Result<()>> {
    Box::pin(import(machine_id, user_id, write_tx, progress))
}

// ===========================================================================
// Selective forgetting ("prune-context") — codex dialect. Shared contract:
// `crate::prune`. Codex-specific delta below.
//
// codex persists each session as a plaintext newline-delimited JSON "rollout"
// at `~/.codex/sessions/YYYY/MM/DD/rollout-<ISO8601>-<thread_id>.jsonl`, each line
// `{timestamp, type, payload}`. The trailing filename UUID equals
// `session_meta.payload.id` on line 0 (= the harvested `chats.agent_session_id`):
// verified against 767/767 real local rollouts on codex-cli 0.x, all bare `.jsonl`
// (NOT `.jsonl.zst`), so blank-in-place has a plaintext target. `find_codex_rollout`
// resolves a thread id by that filename suffix.
//
// Pruned fields (paired by `call_id`, like claude's `tool_use_id`):
//   - `function_call.arguments` → `"{}"` — a JSON-ENCODED STRING (e.g.
//     `"{\"command\":[...]}"`).
//   - `function_call_output.output` → `PRUNED_PLACEHOLDER` — the bulky tool result
//     (also a JSON-encoded string). Field name verified on a real rollout:
//     `output` (NOT `content`).
//
// Matching mirrors claude/gemini (`--tool-name` is a CLAUDE-shape name inverted to
// raw codex `function_call.name`s). Two codex-specific wrinkles:
//   - MCP calls persist as plain `function_call`s under the BARE tool name (server
//     dropped from the on-disk record; verified 0.135.0), so an `mcp__<server>__
//     <tool>` request is inverted by `codex_mcp_ondisk_name_candidates` instead of
//     the static map.
//   - `arguments` is a JSON-ENCODED STRING, so `codex_arguments_match` DECODES it
//     and globs the decoded value leaves (never keys/escapes) — a needle copied
//     from the chat (rendered decoded) matches as the agent saw it.
//
// Last-only / blank-in-place / resume-cache rationale: see `crate::prune`. Blanking
// preserves every line + `call_id` so `codex exec resume` keeps its prefix.
//
// Wired into the adapter via `PRUNE_OPS` (→ `AdapterDescriptor::prune`).

/// `crate::prune::PruneOps` for codex. Points the descriptor at this module's
/// dialect pruners.
pub(crate) const PRUNE_OPS: crate::prune::PruneOps = crate::prune::PruneOps {
    find_session: find_codex_rollout,
    count_matches: count_codex_matches,
    prune_batch: prune_batch_codex_jsonl,
};

/// Locate the codex rollout for `thread_id` under `<base>/sessions/**` (`base` is
/// codex's home dir from `AgentKind::cli_home`; enumerated via `collect_rollouts`).
/// Two-tier resolution (provenance in the section header above):
///   1. FAST PATH (no file reads): prefer the file whose name ends with
///      `-<thread_id>.jsonl`. `thread_id ↔ file` is 1:1 (resume APPENDS to the same
///      rollout), so first-match is correct.
///   2. FALLBACK (content scan): if no filename matches (defensive against a future
///      naming change), match `session_meta.payload.id` (`codex_rollout_thread_id`).
fn find_codex_rollout(base: &Path, thread_id: &str) -> Option<PathBuf> {
    let sessions = base.join("sessions");
    let mut files: Vec<PathBuf> = Vec::new();
    collect_rollouts(&sessions, &mut files).ok()?;

    // Fast path: match the filename suffix `-<thread_id>.jsonl` without opening
    // any file. collect_rollouts already guarantees `rollout-*.jsonl`, so a suffix
    // match is a real rollout.
    let suffix = format!("-{thread_id}.jsonl");
    if let Some(hit) = files.iter().find(|path| {
        path.file_name()
            .and_then(|s| s.to_str())
            .is_some_and(|n| n.ends_with(&suffix))
    }) {
        return Some(hit.clone());
    }

    // Fallback: content scan of line 0 (defensive — only reached if no filename
    // carries the thread id).
    files
        .into_iter()
        .find(|path| codex_rollout_thread_id(path).as_deref() == Some(thread_id))
}

/// Read the `session_meta.payload.id` (thread id) from a codex rollout. It's on
/// the first non-empty line (`{"type":"session_meta","payload":{"id":...}}`), so
/// we read only enough to parse that line rather than slurping the whole file.
/// Returns `None` for an unreadable file or a first line that isn't a
/// `session_meta` with an `id`.
fn codex_rollout_thread_id(path: &Path) -> Option<String> {
    use std::io::BufRead;
    let file = std::fs::File::open(path).ok()?;
    let mut reader = std::io::BufReader::new(file);
    let mut first = String::new();
    loop {
        first.clear();
        let n = reader.read_line(&mut first).ok()?;
        if n == 0 {
            return None; // EOF before any non-blank line.
        }
        if first.trim().is_empty() {
            continue;
        }
        let header: serde_json::Value = serde_json::from_str(first.trim()).ok()?;
        if header.get("type").and_then(|t| t.as_str()) != Some("session_meta") {
            return None;
        }
        return header
            .get("payload")
            .and_then(|p| p.get("id"))
            .and_then(|v| v.as_str())
            .map(str::to_string);
    }
}

/// Invert the codex persisted-tool-name → claude-name map
/// (`map_codex_persisted_tool` in this module): given a CLAUDE-shape tool name
/// (what `--tool-name` carries), return every raw codex
/// `function_call.name` that maps to it. An unknown name is matched literally
/// (codex forwards unknown tools under their native name — pass-through in
/// `map_codex_persisted_tool` — so a user could prune by the raw codex name too).
///
/// Derives from `CODEX_TOOL_NAME_MAP`, the single source of truth shared with the
/// forward map `map_codex_persisted_tool` — so the inverse can't drift (the old
/// failure mode: a renamed codex tool that silently stopped matching, leaking that
/// tool's output past a prune). The map records each codex tool's PRIMARY claude
/// name; `apply_patch`'s args-conditional `Bash` fallback (no discrete path) is a
/// rendering detail and intentionally not a prune target — same as before.
fn claude_to_codex_tool_names(claude_name: &str) -> Vec<&'static str> {
    // Unmapped / already-codex name yields empty, so the caller matches it
    // literally (a prune by the raw codex tool name, or any forwarded/future
    // tool, still works).
    CODEX_TOOL_NAME_MAP
        .iter()
        .filter(|(_, c)| *c == claude_name)
        .map(|(t, _)| *t)
        .collect()
}

/// Single source of truth for the codex-persisted-tool ↔ claude-tool-name mapping,
/// as `(codex function_call.name, claude name)` pairs. Each codex tool's PRIMARY
/// claude name (see `map_codex_persisted_tool` for the args reshaping and
/// `apply_patch`'s conditional `Bash` fallback). The prune inverse
/// `claude_to_codex_tool_names` filters this; the `codex_tool_name_map_round_trips`
/// test asserts every entry round-trips through the forward map.
const CODEX_TOOL_NAME_MAP: &[(&str, &str)] = &[
    ("shell", "Bash"),
    ("exec_command", "Bash"),
    ("apply_patch", "Edit"),
    ("update_plan", "TodoWrite"),
    ("web_search", "WebSearch"),
    ("web_search_call", "WebSearch"),
];

/// Invert an `mcp__<server>__<tool>` wire name (what `--tool-name` carries) to the
/// on-disk `function_call.name` candidates. codex persists MCP calls under the BARE
/// tool name (server dropped — see the section header). The tool is everything
/// after the FIRST `__` in the post-`mcp__` remainder (also yields the bare tool for
/// the degraded `mcp__<tool>` form). Returns `None` for a non-MCP name, which then
/// goes through `claude_to_codex_tool_names`. Candidate set:
///   - the BARE tool (`get_last_id`) — the confirmed 0.135.0 shape,
///   - the `<server>__<tool>` remainder (defensive: a future codex could prefix
///     the server to disambiguate same-named tools),
///   - the full `mcp__<server>__<tool>` string (defensive).
fn codex_mcp_ondisk_name_candidates(want: &str) -> Option<Vec<String>> {
    let rest = want.strip_prefix("mcp__")?;
    if rest.is_empty() {
        return None;
    }
    let tool = rest.split_once("__").map_or(rest, |(_server, tool)| tool);
    let mut cands = vec![tool.to_string()];
    if rest != tool {
        cands.push(rest.to_string());
    }
    cands.push(want.to_string());
    Some(cands)
}

/// Does a codex `function_call` (`name`, `arguments`) match (`tool_name`,
/// `needle`)? The call's `name` must be in the codex set the claude `tool_name`
/// maps to (or equal it literally, for forwarded unknown tools) — and for an
/// `mcp__<server>__<tool>` wire name, in the on-disk candidate set from
/// `codex_mcp_ondisk_name_candidates` (codex stores the bare tool name, not our
/// server-qualified wire name). The `needle` must glob the `arguments`' decoded
/// value leaves per `codex_arguments_match`, or, when `needle` is empty, the call
/// must have been made with no arguments.
fn codex_function_call_matches(
    name: &str,
    arguments: Option<&str>,
    tool_name: &str,
    needle: &str,
) -> bool {
    let name_ok = match codex_mcp_ondisk_name_candidates(tool_name) {
        Some(cands) => cands.iter().any(|c| c == name),
        None => crate::prune::tool_name_matches(name, tool_name, claude_to_codex_tool_names),
    };
    if !name_ok {
        return false;
    }
    // Never target the agent's own in-flight prune-context CLI call. codex's
    // `arguments` is a JSON-ENCODED STRING, so decode then scan its value leaves
    // (an unparseable blob is treated as one leaf). See
    // crate::prune::value_is_prune_context_call.
    if let Some(a) = arguments {
        let decoded = serde_json::from_str::<serde_json::Value>(a)
            .unwrap_or_else(|_| serde_json::Value::String(a.to_string()));
        if crate::prune::value_is_prune_context_call(&decoded) {
            return false;
        }
    }
    // Empty needle = the empty-args selector (`--args ""`): match no-argument
    // calls. codex's `arguments` is a JSON-ENCODED STRING (a no-arg MCP call
    // persists `"{}"`), so decode then apply the shared emptiness predicate;
    // empty/absent counts too. An UNPARSEABLE non-empty string is NOT empty (fail
    // closed — don't blank a call we can't read).
    if needle.is_empty() {
        return match arguments {
            None => true,
            Some(a) => {
                let t = a.trim();
                t.is_empty()
                    || serde_json::from_str::<serde_json::Value>(t)
                        .ok()
                        .is_some_and(|v| crate::prune::args_value_is_empty(Some(&v)))
            }
        };
    }
    match arguments {
        Some(a) => codex_arguments_match(a, needle),
        None => false,
    }
}

/// Glob `needle` against a codex `function_call`'s `arguments` VALUE leaves.
/// `arguments` is a JSON-ENCODED STRING, so we DECODE first and glob the decoded
/// string VALUE leaves via shared [`crate::prune::value_glob_match`] (never keys,
/// never the `\"`-escaped wrapper — that was the over-match bug), so a chat-copied
/// needle matches as the agent saw it. Non-JSON `arguments` (defensive) → glob the
/// raw string leaf so the call stays reachable.
fn codex_arguments_match(arguments: &str, needle: &str) -> bool {
    match serde_json::from_str::<serde_json::Value>(arguments) {
        Ok(v) => crate::prune::value_glob_match(&v, needle),
        Err(_) => crate::prune::glob_leaf_match(arguments, needle),
    }
}

/// Matching `function_call` `call_id`s in DOCUMENT ORDER, EXCLUDING any whose
/// paired `function_call_output.output` is already the `[pruned]` placeholder (so
/// repeated prunes walk newest→oldest instead of re-hitting an already-blanked
/// call). A `function_call` with no output yet (in-flight) stays eligible. The
/// "ordered matched minus already-pruned" combinator (shared with claude/gemini)
/// lives in [`crate::prune::select_eligible_ids`]; here we just supply the
/// codex-shape per-entry collectors ([`codex_collect_matched`] /
/// [`codex_collect_pruned`]) — the SAME two `prune_codex_jsonl` feeds the
/// single-read apply driver, so the count and apply paths can't drift. The
/// collectors are order-independent — a rollout could record an output before its
/// call (see `codex_prune_blanks_output_appearing_before_its_call`).
fn eligible_matches(path: &Path, tool_name: &str, needle: &str) -> std::io::Result<Vec<String>> {
    crate::prune::select_eligible_ids(
        path,
        |entry, matched| codex_collect_matched(entry, tool_name, needle, matched),
        codex_collect_pruned,
    )
}

/// Pass-1 matched collector shared by `eligible_matches` (the control-side count
/// path, via [`crate::prune::select_eligible_ids`]) and `prune_codex_jsonl` (the
/// apply path, via [`crate::prune::rewrite_jsonl_last_only`]). Pushes the matching
/// `function_call`'s `call_id` in DOCUMENT ORDER onto `matched`. A function_call
/// yields at most its own single call_id.
fn codex_collect_matched(
    entry: &serde_json::Value,
    tool_name: &str,
    needle: &str,
    matched: &mut Vec<String>,
) {
    // Same match predicate as the blank pass (via the shared collect helper) so
    // the count can't drift from what gets blanked.
    let mut ids = std::collections::HashSet::new();
    collect_codex_call_id_if_matches(entry, tool_name, needle, &mut ids);
    matched.extend(ids);
}

/// Pass-1 already-pruned collector shared by `eligible_matches` and
/// `prune_codex_jsonl` (same call sites as [`codex_collect_matched`]). Marks a
/// call ineligible from a `function_call_output` whose `output` is already the
/// `[pruned]` placeholder.
fn codex_collect_pruned(
    entry: &serde_json::Value,
    already_pruned: &mut std::collections::HashSet<String>,
) {
    if entry.get("type").and_then(|t| t.as_str()) != Some("response_item") {
        return;
    }
    let Some(payload) = entry.get("payload") else {
        return;
    };
    if payload.get("type").and_then(|t| t.as_str()) != Some("function_call_output") {
        return;
    }
    // A `[pruned]` output marks its call ineligible.
    if payload.get("output").and_then(|o| o.as_str()) == Some(crate::prune::PRUNED_PLACEHOLDER) {
        if let Some(id) = payload.get("call_id").and_then(|v| v.as_str()) {
            already_pruned.insert(id.to_string());
        }
    }
}

/// Read-only pre-scan twin of the claude/gemini count for codex: how many ELIGIBLE
/// matches (drives the zero-check and the "N remain" CLI message). Zero → the
/// control task returns an error to the still-alive agent instead of killing +
/// respawning it.
fn count_codex_matches(path: &Path, tool_name: &str, needle: &str) -> std::io::Result<usize> {
    Ok(eligible_matches(path, tool_name, needle)?.len())
}

/// If `entry` is a `response_item`/`function_call` matching (`tool_name`,
/// `needle`), record its `call_id` into `ids`. The single source of the codex
/// function-call match predicate, called per-entry by `eligible_matches` so the
/// count + the last-only selection can't drift on what "matches" means.
fn collect_codex_call_id_if_matches(
    entry: &serde_json::Value,
    tool_name: &str,
    needle: &str,
    ids: &mut std::collections::HashSet<String>,
) {
    if entry.get("type").and_then(|t| t.as_str()) != Some("response_item") {
        return;
    }
    let Some(payload) = entry.get("payload") else {
        return;
    };
    if payload.get("type").and_then(|t| t.as_str()) != Some("function_call") {
        return;
    }
    let name = payload.get("name").and_then(|v| v.as_str()).unwrap_or("");
    let arguments = payload.get("arguments").and_then(|v| v.as_str());
    if codex_function_call_matches(name, arguments, tool_name, needle) {
        if let Some(id) = payload.get("call_id").and_then(|v| v.as_str()) {
            ids.insert(id.to_string());
        }
    }
}

/// The real codex rewrite, LAST-ONLY: blank just the MOST RECENT eligible match
/// (its `function_call.arguments` → `"{}"` + paired `function_call_output.output`
/// → `[pruned]`) — never deleting a line — then atomic write back. The agent
/// reclaims more by calling again (eligibility walks newest→oldest). Empty set
/// (TOCTOU after the control pre-check, or nothing eligible) is a safe no-op
/// (`results_blanked` 0 → timeline frame skipped).
///
/// The last-only target is chosen INSIDE the single-read driver
/// ([`crate::prune::rewrite_jsonl_last_only`]) from the SAME codex collectors
/// `eligible_matches` feeds ([`codex_collect_matched`] / [`codex_collect_pruned`])
/// — one transcript read does pass-1 selection + the blank pass, dropping the
/// redundant `eligible_matches` read the old code did before the driver. The blank
/// pass (`blank_codex_entry`) blanks exactly that call's arguments + its paired
/// output (running per-entry, it catches an output recorded BEFORE its call — see
/// `codex_prune_blanks_output_appearing_before_its_call`). An empty target (TOCTOU
/// after the control pre-check, or nothing eligible) is a safe no-op: the blank
/// pass touches nothing → `results_blanked` 0 → timeline frame skipped. Codex has
/// no fail-closed post-scan, so the returned final entries are ignored.
///
/// TEST-ONLY now: the one-round case of [`prune_batch_codex_jsonl`], kept as the
/// equivalence oracle the tests pin the batch path against. Production goes through
/// the batch fn.
#[cfg(test)]
fn prune_codex_jsonl(
    path: &Path,
    tool_name: &str,
    needle: &str,
) -> std::io::Result<crate::prune::PruneStats> {
    let (stats, _final_entries) = crate::prune::rewrite_jsonl_last_only(
        path,
        |entry, matched| codex_collect_matched(entry, tool_name, needle, matched),
        codex_collect_pruned,
        blank_codex_entry,
    )?;
    Ok(stats)
}

/// Batch entry point behind `PRUNE_OPS::prune_batch`: blank the last-only target of
/// every `(tool_name, needle)` in `targets` in ONE read/write via the shared batch
/// driver ([`crate::prune::rewrite_jsonl_batch_last_only`]), reproducing exactly
/// what running [`prune_codex_jsonl`] once per target produced. Same codex
/// collectors `prune_codex_jsonl` feeds; codex has no fail-closed post-scan, so the
/// final entries are ignored (mirrors claude).
fn prune_batch_codex_jsonl(
    path: &Path,
    targets: &[crate::prune::PruneTarget],
) -> std::io::Result<crate::prune::PruneStats> {
    let (stats, _final_entries) = crate::prune::rewrite_jsonl_batch_last_only(
        path,
        targets.len(),
        |idx, entry, matched| {
            let (tool_name, needle) = &targets[idx];
            codex_collect_matched(entry, tool_name, needle, matched)
        },
        codex_collect_pruned,
        blank_codex_entry,
    )?;
    Ok(stats)
}

/// Pass-2 helper: blank one parsed codex rollout line in place:
///   - `function_call` whose `call_id` ∈ `pruned_ids`: blank `arguments` to
///     `"{}"`.
///   - `function_call_output` whose `call_id` ∈ `pruned_ids`: blank `output` to
///     `[pruned]` and record the `call_id` in `outputs_blanked` (the user-facing
///     count counts outputs actually dropped, not calls merely matched).
///
/// Returns `None` when the entry was left untouched, or `Some(freed_bytes)` when
/// a field was blanked (`freed` may legitimately be `0` for a tiny output that
/// was shorter than the placeholder — the caller still re-serializes).
fn blank_codex_entry(
    entry: &mut serde_json::Value,
    pruned_ids: &std::collections::HashSet<String>,
    outputs_blanked: &mut std::collections::HashSet<String>,
) -> Option<usize> {
    if entry.get("type").and_then(|t| t.as_str()) != Some("response_item") {
        return None;
    }
    let payload = entry.get_mut("payload").and_then(|p| p.as_object_mut())?;
    match payload.get("type").and_then(|t| t.as_str()) {
        Some("function_call") => {
            let matched = payload
                .get("call_id")
                .and_then(|v| v.as_str())
                .map(|id| pruned_ids.contains(id))
                .unwrap_or(false);
            if !matched {
                return None;
            }
            // Blank `arguments` (a JSON-encoded string) to the encoded empty
            // object `"{}"`, keeping the wire type (string) intact.
            crate::prune::blank_string_field(payload, "arguments", "{}")
        }
        Some("function_call_output") => {
            let call_id = payload.get("call_id").and_then(|v| v.as_str());
            let pruned = call_id.map(|id| pruned_ids.contains(id)).unwrap_or(false);
            if !pruned {
                return None;
            }
            let call_id = call_id.map(str::to_string);
            // Only count (and re-serialize) when the output was ACTUALLY blanked
            // this run. `blank_string_field` returns `None` when it was
            // already `[pruned]` (idempotent re-prune), so a re-run over a
            // fully-pruned rollout reports `results_blanked = 0` and skips the
            // timeline frame — matching the claude pruner's `already` guard.
            let freed = crate::prune::blank_string_field(
                payload,
                "output",
                crate::prune::PRUNED_PLACEHOLDER,
            )?;
            if let Some(id) = call_id {
                outputs_blanked.insert(id);
            }
            Some(freed)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Returns the SmallVec as a Vec of human-readable tags + payload for
    /// easy assertion equality. Delegates to the shared stringifier in
    /// `adapter.rs` so the event→tag mapping has a single source of truth.
    fn run(adapter: &mut CodexAdapter, line: &str) -> Vec<String> {
        crate::adapter::stringify_events(adapter.handle_line(line.to_string()))
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
    fn thread_started_frame_harvests_session_id_and_drops_frame() {
        let mut a = CodexAdapter::new();
        let line =
            r#"{"type":"thread.started","thread_id":"0192f00d-7ce0-7e9a-8d6f-abcdef012345"}"#;
        let events = run(&mut a, line);
        assert_eq!(
            events,
            vec!["SessionIdHarvested(0192f00d-7ce0-7e9a-8d6f-abcdef012345)"]
        );
    }

    #[test]
    fn item_completed_agent_message_normalizes_to_assistant_text_frame() {
        let mut a = CodexAdapter::new();
        let line = r#"{"type":"item.completed","item":{"id":"item_1","type":"agent_message","text":"Hello there"}}"#;
        let events = run(&mut a, line);
        assert_eq!(events.len(), 1);
        assert!(events[0].starts_with("Frame("));
        // Strip the wrapper and re-parse to assert structure.
        let v = frame_value(&events[0]);
        assert_eq!(v["type"], "assistant");
        assert_eq!(v["message"]["content"][0]["type"], "text");
        assert_eq!(v["message"]["content"][0]["text"], "Hello there");
    }

    /// Strips the `Frame(...)` test wrapper and parses the inner JSON. Used
    /// by the per-tool tests below to keep them readable.
    fn frame_value(event: &str) -> Value {
        crate::adapter::frame_value(event)
    }

    #[test]
    fn item_completed_file_change_update_normalizes_to_claude_edit() {
        // Codex's `file_change` with kind=update should land as a claude
        // `Edit` tool_use with `input.file_path` — iOS's existing toolSummary
        // switch picks the file path off `Edit`, so no codex-specific iOS
        // branch is needed.
        let mut a = CodexAdapter::new();
        let line = r#"{"type":"item.completed","item":{"id":"item_2","type":"file_change","changes":[{"path":"src/foo.rs","kind":"update"}],"status":"completed"}}"#;
        let events = run(&mut a, line);
        assert_eq!(events.len(), 1);
        let v = frame_value(&events[0]);
        assert_eq!(v["type"], "assistant");
        assert_eq!(v["message"]["content"][0]["type"], "tool_use");
        assert_eq!(v["message"]["content"][0]["name"], "Edit");
        assert_eq!(v["message"]["content"][0]["id"], "item_2");
        assert_eq!(
            v["message"]["content"][0]["input"]["file_path"],
            "src/foo.rs"
        );
    }

    #[test]
    fn item_completed_file_change_add_normalizes_to_claude_write() {
        let mut a = CodexAdapter::new();
        let line = r#"{"type":"item.completed","item":{"id":"item_2","type":"file_change","changes":[{"path":"src/new.rs","kind":"add"}],"status":"completed"}}"#;
        let events = run(&mut a, line);
        assert_eq!(events.len(), 1);
        let v = frame_value(&events[0]);
        assert_eq!(v["message"]["content"][0]["name"], "Write");
        assert_eq!(
            v["message"]["content"][0]["input"]["file_path"],
            "src/new.rs"
        );
    }

    #[test]
    fn item_completed_file_change_delete_normalizes_to_claude_bash_rm() {
        // Claude has no dedicated delete tool, and claude code itself uses
        // `Bash` with `rm <path>` for deletes — mirror that so the iOS row
        // reads naturally ("Bash: rm src/old.rs").
        let mut a = CodexAdapter::new();
        let line = r#"{"type":"item.completed","item":{"id":"item_2","type":"file_change","changes":[{"path":"src/old.rs","kind":"delete"}],"status":"completed"}}"#;
        let events = run(&mut a, line);
        assert_eq!(events.len(), 1);
        let v = frame_value(&events[0]);
        assert_eq!(v["message"]["content"][0]["name"], "Bash");
        assert_eq!(
            v["message"]["content"][0]["input"]["command"],
            "rm src/old.rs"
        );
    }

    #[test]
    fn item_completed_file_change_multi_emits_one_frame_per_change() {
        // Codex bundles N edits in a single `file_change` item; fan them out
        // so each shows up as its own bubble line (claude itself emits one
        // tool_use per edit). Ids are suffixed `.<idx>` to stay unique.
        let mut a = CodexAdapter::new();
        let line = r#"{"type":"item.completed","item":{"id":"item_2","type":"file_change","changes":[{"path":"a.rs","kind":"update"},{"path":"b.rs","kind":"add"}],"status":"completed"}}"#;
        let events = run(&mut a, line);
        assert_eq!(events.len(), 2);
        let v0 = frame_value(&events[0]);
        assert_eq!(v0["message"]["content"][0]["name"], "Edit");
        assert_eq!(v0["message"]["content"][0]["id"], "item_2.0");
        assert_eq!(v0["message"]["content"][0]["input"]["file_path"], "a.rs");
        let v1 = frame_value(&events[1]);
        assert_eq!(v1["message"]["content"][0]["name"], "Write");
        assert_eq!(v1["message"]["content"][0]["id"], "item_2.1");
        assert_eq!(v1["message"]["content"][0]["input"]["file_path"], "b.rs");
    }

    #[test]
    fn item_completed_command_execution_normalizes_to_claude_bash() {
        let mut a = CodexAdapter::new();
        let line = r#"{"type":"item.completed","item":{"id":"item_3","type":"command_execution","command":"ls -la","aggregated_output":"out","exit_code":0,"status":"completed"}}"#;
        let events = run(&mut a, line);
        // Just the visible bubble — an ordinary command is NOT the prune-context
        // call, so the call-keyed cue does not fire.
        assert_eq!(events.len(), 1);
        let v = frame_value(&events[0]);
        assert_eq!(v["message"]["content"][0]["name"], "Bash");
        assert_eq!(v["message"]["content"][0]["input"]["command"], "ls -la");
        // We drop codex-side metadata — iOS doesn't render it and keeping it
        // would just bloat the persisted body.
        assert!(v["message"]["content"][0]["input"]["aggregated_output"].is_null());
        assert!(v["message"]["content"][0]["input"]["exit_code"].is_null());
    }

    #[test]
    fn item_completed_web_search_normalizes_to_claude_websearch() {
        let mut a = CodexAdapter::new();
        let line = r#"{"type":"item.completed","item":{"id":"item_4","type":"web_search","query":"rust async traits","action":{"type":"search","query":"rust async traits"}}}"#;
        let events = run(&mut a, line);
        assert_eq!(events.len(), 1);
        let v = frame_value(&events[0]);
        assert_eq!(v["message"]["content"][0]["name"], "WebSearch");
        assert_eq!(
            v["message"]["content"][0]["input"]["query"],
            "rust async traits"
        );
    }

    #[test]
    fn turn_completed_emits_result_frame_and_marker_but_no_context_tokens() {
        let mut a = CodexAdapter::new();
        // `usage` here is the cumulative `total_token_usage` — deliberately NOT
        // surfaced as ContextTokens (occupancy comes post-turn from the
        // rollout, see `read_rollout_last_context_tokens`).
        let line = r#"{"type":"turn.completed","usage":{"input_tokens":1000,"cached_input_tokens":500,"output_tokens":200,"reasoning_output_tokens":50}}"#;
        let events = run(&mut a, line);
        assert_eq!(events.len(), 2);
        assert!(events.iter().all(|e| !e.starts_with("ContextTokens")));
        assert!(events[0].starts_with("Frame("));
        assert_eq!(events[1], "Result");
        // Verify the result envelope shape.
        let v = frame_value(&events[0]);
        assert_eq!(v["type"], "result");
        assert_eq!(v["subtype"], "success");
        assert_eq!(v["is_error"], false);
    }

    #[test]
    fn turn_failed_emits_error_result_frame_and_result_marker() {
        let mut a = CodexAdapter::new();
        let line =
            r#"{"type":"turn.failed","error":{"message":"sandbox denied: read-only filesystem"}}"#;
        let events = run(&mut a, line);
        assert_eq!(events.len(), 2);
        assert!(events[0].starts_with("Frame("));
        assert_eq!(events[1], "Result");

        let v = frame_value(&events[0]);
        assert_eq!(v["type"], "result");
        assert_eq!(v["subtype"], "error");
        assert_eq!(v["is_error"], true);
        assert_eq!(
            v["error"]["message"],
            "sandbox denied: read-only filesystem"
        );
    }

    #[test]
    fn turn_failed_without_message_uses_generic_error_result() {
        let mut a = CodexAdapter::new();
        let line = r#"{"type":"turn.failed","error":{}}"#;
        let events = run(&mut a, line);
        assert_eq!(events.len(), 2);
        let v = frame_value(&events[0]);
        assert_eq!(v["subtype"], "error");
        assert_eq!(v["is_error"], true);
        assert_eq!(v["error"]["message"], "turn failed");
        assert_eq!(events[1], "Result");
    }

    #[test]
    fn read_rollout_last_context_tokens_returns_last_records_last_token_usage() {
        // A rollout with two `token_count` records: the gauge must read the
        // LAST record's `last_token_usage.input_tokens` (occupancy), NOT
        // `total_token_usage` (cumulative) and NOT the earlier record.
        let dir = std::env::temp_dir().join(format!(
            "zucchini-codex-rollout-{}-{}",
            std::process::id(),
            line!()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("rollout-test.jsonl");
        let body = concat!(
            r#"{"type":"session_meta","payload":{"id":"abc"}}"#,
            "\n",
            r#"{"type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":53584},"last_token_usage":{"input_tokens":19590}}}}"#,
            "\n",
            r#"{"type":"response_item","payload":{"type":"message"}}"#,
            "\n",
            r#"{"type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":122951},"last_token_usage":{"input_tokens":24930},"model_context_window":258400}}}"#,
            "\n",
        );
        std::fs::write(&path, body).unwrap();

        assert_eq!(read_rollout_last_context_tokens(&path), Some(24930));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn read_rollout_last_context_tokens_none_without_token_count() {
        let dir = std::env::temp_dir().join(format!(
            "zucchini-codex-rollout-{}-{}",
            std::process::id(),
            line!()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("rollout-empty.jsonl");
        std::fs::write(
            &path,
            "{\"type\":\"session_meta\",\"payload\":{\"id\":\"abc\"}}\n",
        )
        .unwrap();

        assert_eq!(read_rollout_last_context_tokens(&path), None);
        // Unreadable path → None (caller leaves the gauge untouched).
        assert_eq!(
            read_rollout_last_context_tokens(&dir.join("does-not-exist.jsonl")),
            None
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn post_turn_context_tokens_none_when_no_session_id() {
        // No harvested id and no resume id → nothing to locate, returns None.
        let a = CodexAdapter::new();
        assert_eq!(a.post_turn_context_tokens(None), None);
    }

    #[test]
    fn turn_started_and_agent_message_item_started_dropped() {
        // `turn.started` has no claude analog; an `agent_message` `item.started`
        // carries no final text yet (it renders on `item.completed`). Tool
        // `item.started`s, by contrast, DO render — see the tests below.
        let mut a = CodexAdapter::new();
        let ts = r#"{"type":"turn.started"}"#;
        let is_ = r#"{"type":"item.started","item":{"id":"item_4","type":"agent_message"}}"#;
        assert!(run(&mut a, ts).is_empty());
        assert!(run(&mut a, is_).is_empty());
    }

    #[test]
    fn item_started_command_execution_surfaces_in_flight_bash() {
        // The whole point of the started-not-completed switch: a command shows
        // the instant codex starts it (status:in_progress), not after it ends.
        let mut a = CodexAdapter::new();
        let line = r#"{"type":"item.started","item":{"id":"item_3","type":"command_execution","command":"npm ci","status":"in_progress"}}"#;
        let events = run(&mut a, line);
        assert_eq!(events.len(), 1);
        let v = frame_value(&events[0]);
        assert_eq!(v["message"]["content"][0]["name"], "Bash");
        assert_eq!(v["message"]["content"][0]["id"], "item_3");
        assert_eq!(v["message"]["content"][0]["input"]["command"], "npm ci");
    }

    #[test]
    fn item_started_then_completed_command_dedupes_to_single_bubble() {
        // started renders the bubble; the matching completed (same id) is
        // suppressed so the command isn't shown twice.
        let mut a = CodexAdapter::new();
        let started = r#"{"type":"item.started","item":{"id":"item_3","type":"command_execution","command":"ls -la","status":"in_progress"}}"#;
        let completed = r#"{"type":"item.completed","item":{"id":"item_3","type":"command_execution","command":"ls -la","aggregated_output":"out","exit_code":0,"status":"completed"}}"#;
        let first = run(&mut a, started);
        assert_eq!(first.len(), 1, "started should surface the command");
        let second = run(&mut a, completed);
        // The bubble is deduped (already surfaced on started); an ordinary
        // command fires no prune cue (call-keyed), so nothing is left.
        assert!(
            second.is_empty(),
            "deduped non-prune completion emits nothing, got {second:?}"
        );
    }

    #[test]
    fn item_completed_command_execution_without_started_still_renders() {
        // Fallback path: if a tool item never produced a usable item.started,
        // its item.completed must still render (nothing vanishes). The id was
        // never recorded, so dedup doesn't suppress it.
        let mut a = CodexAdapter::new();
        let line = r#"{"type":"item.completed","item":{"id":"item_9","type":"command_execution","command":"echo hi","status":"completed"}}"#;
        let events = run(&mut a, line);
        // Fallback render bubble only — an ordinary command fires no prune cue.
        assert_eq!(events.len(), 1);
        let v = frame_value(&events[0]);
        assert_eq!(v["message"]["content"][0]["name"], "Bash");
        assert_eq!(v["message"]["content"][0]["input"]["command"], "echo hi");
    }

    #[test]
    fn item_completed_prune_context_command_fires_cue_siblings_do_not() {
        // Call-keyed: only the `prune-context` command's own completion drives
        // the queued prune's apply. A sibling shell command completing first in
        // the same batch must not preempt it.
        let mut a = CodexAdapter::new();
        let sibling = r#"{"type":"item.completed","item":{"id":"item_1","type":"command_execution","command":"grep -rn foo src/","aggregated_output":"hits","exit_code":0,"status":"completed"}}"#;
        let sib_events = run(&mut a, sibling);
        assert_eq!(sib_events.len(), 1, "sibling renders its bubble, no cue");
        assert_ne!(sib_events[0], "ToolResult");
        let prune = r#"{"type":"item.completed","item":{"id":"item_2","type":"command_execution","command":"\"$ZUCCHINI_SPAWNER_BIN\" prune-context --tool-name Bash --args \"*foo*\" --reason y","aggregated_output":"pruned","exit_code":0,"status":"completed"}}"#;
        let prune_events = run(&mut a, prune);
        // Bubble, then the prune cue (the call's own completion).
        assert_eq!(prune_events.len(), 2);
        assert_eq!(prune_events[1], "ToolResult");
    }

    #[test]
    fn item_started_then_completed_file_change_dedupes() {
        // Multi-change file_change surfaces on started (one bubble per change),
        // and the completed for the same item id is deduped to nothing.
        let mut a = CodexAdapter::new();
        let started = r#"{"type":"item.started","item":{"id":"item_2","type":"file_change","changes":[{"path":"a.rs","kind":"update"},{"path":"b.rs","kind":"add"}],"status":"in_progress"}}"#;
        let completed = r#"{"type":"item.completed","item":{"id":"item_2","type":"file_change","changes":[{"path":"a.rs","kind":"update"},{"path":"b.rs","kind":"add"}],"status":"completed"}}"#;
        assert_eq!(run(&mut a, started).len(), 2);
        assert!(run(&mut a, completed).is_empty());
    }

    #[test]
    fn item_started_mcp_tool_call_surfaces_qualified_name_and_raw_args() {
        // codex's exec-stream `arguments` is a RAW JSON object (not the encoded
        // string the persisted function_call carries), so it passes straight
        // through under `input`. The tool_use name is claude's own
        // `mcp__<server>__<tool>` convention → iOS's default summary branch
        // renders the qualified name with no codex-specific iOS code.
        let mut a = CodexAdapter::new();
        let line = r#"{"type":"item.started","item":{"id":"item_5","type":"mcp_tool_call","server":"github","tool":"create_issue","arguments":{"title":"bug","repo":"acme/app"},"status":"in_progress"}}"#;
        let events = run(&mut a, line);
        assert_eq!(events.len(), 1);
        let v = frame_value(&events[0]);
        assert_eq!(v["message"]["content"][0]["type"], "tool_use");
        assert_eq!(
            v["message"]["content"][0]["name"],
            "mcp__github__create_issue"
        );
        assert_eq!(v["message"]["content"][0]["id"], "item_5");
        assert_eq!(v["message"]["content"][0]["input"]["title"], "bug");
        assert_eq!(v["message"]["content"][0]["input"]["repo"], "acme/app");
    }

    #[test]
    fn item_started_then_completed_mcp_tool_call_dedupes_to_single_bubble() {
        // started surfaces the MCP bubble; the matching completed (same id,
        // now carrying a result) is suppressed so it isn't rendered twice.
        let mut a = CodexAdapter::new();
        let started = r#"{"type":"item.started","item":{"id":"item_5","type":"mcp_tool_call","server":"github","tool":"create_issue","arguments":{"title":"bug"},"status":"in_progress"}}"#;
        let completed = r#"{"type":"item.completed","item":{"id":"item_5","type":"mcp_tool_call","server":"github","tool":"create_issue","arguments":{"title":"bug"},"result":{"ok":true},"status":"completed"}}"#;
        assert_eq!(run(&mut a, started).len(), 1);
        assert!(
            run(&mut a, completed).is_empty(),
            "completed for an already-surfaced MCP id must be deduped"
        );
    }

    #[test]
    fn item_started_mcp_tool_call_missing_both_names_falls_through_to_completed() {
        // No server AND no tool → started renders nothing (no garbage `mcp__`
        // bubble), so the completed fallback fires instead. Here completed
        // carries a tool name, so the fallback renders a usable bubble.
        let mut a = CodexAdapter::new();
        let started = r#"{"type":"item.started","item":{"id":"item_6","type":"mcp_tool_call","arguments":{},"status":"in_progress"}}"#;
        assert!(
            run(&mut a, started).is_empty(),
            "started with no server/tool must render nothing"
        );
        let completed = r#"{"type":"item.completed","item":{"id":"item_6","type":"mcp_tool_call","server":"fs","tool":"read","arguments":{"path":"/x"},"status":"completed"}}"#;
        let events = run(&mut a, completed);
        assert_eq!(events.len(), 1, "completed fallback must render the bubble");
        let v = frame_value(&events[0]);
        assert_eq!(v["message"]["content"][0]["name"], "mcp__fs__read");
    }

    #[test]
    fn item_started_collab_spawn_agent_with_prompt_maps_to_claude_agent() {
        // A `spawn_agent` carrying a prompt → claude `Agent` `{description}`, so
        // iOS's toolSummary (`Agent`→`description`) shows the sub-agent's task.
        let mut a = CodexAdapter::new();
        let line = r#"{"type":"item.started","item":{"id":"item_7","type":"collab_tool_call","tool":"spawn_agent","sender_thread_id":"t0","receiver_thread_ids":["t1"],"prompt":"investigate the flaky test","agents_states":{},"status":"in_progress"}}"#;
        let events = run(&mut a, line);
        assert_eq!(events.len(), 1);
        let v = frame_value(&events[0]);
        assert_eq!(v["message"]["content"][0]["type"], "tool_use");
        assert_eq!(v["message"]["content"][0]["name"], "Agent");
        assert_eq!(v["message"]["content"][0]["id"], "item_7");
        assert_eq!(
            v["message"]["content"][0]["input"]["description"],
            "investigate the flaky test"
        );
    }

    #[test]
    fn item_started_collab_send_input_passes_tool_name_through() {
        // Non-spawn collab subtools (or spawn with no prompt) pass the collab
        // tool name through verbatim → iOS renders the tool-name-only branch.
        let mut a = CodexAdapter::new();
        let line = r#"{"type":"item.started","item":{"id":"item_8","type":"collab_tool_call","tool":"send_input","sender_thread_id":"t0","receiver_thread_ids":["t1"],"prompt":"keep going","agents_states":{},"status":"in_progress"}}"#;
        let events = run(&mut a, line);
        assert_eq!(events.len(), 1);
        let v = frame_value(&events[0]);
        assert_eq!(v["message"]["content"][0]["name"], "send_input");
        assert_eq!(v["message"]["content"][0]["id"], "item_8");
        // routing + prompt preserved under input (iOS ignores them, but the
        // stored frame stays useful).
        assert_eq!(v["message"]["content"][0]["input"]["prompt"], "keep going");
        assert_eq!(
            v["message"]["content"][0]["input"]["receiver_thread_ids"][0],
            "t1"
        );
    }

    #[test]
    fn item_started_then_completed_collab_dedupes_to_single_bubble() {
        let mut a = CodexAdapter::new();
        let started = r#"{"type":"item.started","item":{"id":"item_7","type":"collab_tool_call","tool":"spawn_agent","prompt":"do X","status":"in_progress"}}"#;
        let completed = r#"{"type":"item.completed","item":{"id":"item_7","type":"collab_tool_call","tool":"spawn_agent","prompt":"do X","status":"completed"}}"#;
        assert_eq!(run(&mut a, started).len(), 1);
        assert!(
            run(&mut a, completed).is_empty(),
            "completed for an already-surfaced collab id must be deduped"
        );
    }

    #[test]
    fn unknown_frame_type_passed_through_as_frame() {
        // Defensive against codex format drift — a future frame type we
        // don't know about still reaches the chat (iOS will likely render
        // it as raw text via the unknown-frame branch). Matches claude's
        // permissive behavior for non-stream-event frames.
        let mut a = CodexAdapter::new();
        let line = r#"{"type":"some.future.event","payload":{"k":"v"}}"#;
        let events = run(&mut a, line);
        assert_eq!(events.len(), 1);
        assert!(events[0].starts_with("Frame("));
    }

    #[test]
    fn non_json_line_kept_as_frame() {
        // A spawner stderr-merged line (shouldn't happen — stderr is buffered
        // separately — but the line loop still gets bytes from stdout). Lines
        // that don't start with `{` skip every JSON branch and are kept.
        let mut a = CodexAdapter::new();
        let events = run(&mut a, "non-json-line");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0], "Frame(non-json-line)");
    }

    #[test]
    fn model_some_appends_model_flag_on_first_turn() {
        let mut a = CodexAdapter::new();
        let prompt_file = PathBuf::from("/tmp/p.txt");
        let c = ctx(&prompt_file, None, false, Some("gpt-5.5"));
        let cmd = a.prepare_command(&c).unwrap();
        assert!(cmd.contains("--model 'gpt-5.5'"), "got: {}", cmd);
    }

    #[test]
    fn model_none_omits_model_flag() {
        let mut a = CodexAdapter::new();
        let prompt_file = PathBuf::from("/tmp/p.txt");
        let c = ctx(&prompt_file, None, false, None);
        let cmd = a.prepare_command(&c).unwrap();
        assert!(!cmd.contains("--model"), "got: {}", cmd);
    }

    #[test]
    fn resume_command_omits_sandbox_and_cwd_flags() {
        // `codex exec resume` does NOT accept -s, -C, --add-dir (locked at
        // session creation). prepare_command must omit them on the resume
        // path or the spawn fails on argv parse. The bypass flag IS allowed
        // on resume and survives across turns so the agent stays unsandboxed.
        let mut a = CodexAdapter::new();
        let prompt_file = PathBuf::from("/tmp/p.txt");
        let c = ctx(&prompt_file, Some("uuid-here"), false, None);
        let cmd = a.prepare_command(&c).unwrap();
        assert!(
            cmd.contains("exec resume 'uuid-here'"),
            "expected resume subcommand, got: {}",
            cmd
        );
        assert!(
            !cmd.contains(" -s "),
            "resume must not pass sandbox flag, got: {}",
            cmd
        );
        assert!(
            !cmd.contains(" -C "),
            "resume must not pass cwd flag, got: {}",
            cmd
        );
        assert!(
            cmd.contains("--dangerously-bypass-approvals-and-sandbox"),
            "non-sandboxed resume should still emit the bypass flag, got: {}",
            cmd
        );
    }

    #[test]
    fn first_turn_command_uses_bypass_flag_and_cwd() {
        // Non-sandboxed first turn collapses sandbox + approval skip into a
        // single `--dangerously-bypass-approvals-and-sandbox` flag (codex
        // exec has no `-a` / `--ask-for-approval`; the previous code path
        // was invalid and failed at argv parse).
        let mut a = CodexAdapter::new();
        let prompt_file = PathBuf::from("/tmp/p.txt");
        let c = ctx(&prompt_file, None, false, None);
        let cmd = a.prepare_command(&c).unwrap();
        assert!(
            cmd.contains("--dangerously-bypass-approvals-and-sandbox"),
            "expected bypass flag for non-sandboxed sender, got: {}",
            cmd
        );
        assert!(
            !cmd.contains(" -a "),
            "codex exec rejects -a / --ask-for-approval, must not emit it, got: {}",
            cmd
        );
        assert!(
            cmd.contains("-C '/tmp/proj'"),
            "expected -C cwd flag, got: {}",
            cmd
        );
    }

    #[test]
    fn sandboxed_first_turn_uses_read_only_and_config_approval() {
        // Sandboxed invitees get the most restrictive codex sandbox
        // (read-only) plus `-c approval_policy=never` (the only way to
        // skip prompts on `codex exec` — the top-level `-a` short flag
        // doesn't exist on the subcommand).
        let mut a = CodexAdapter::new();
        let prompt_file = PathBuf::from("/tmp/p.txt");
        let c = ctx(&prompt_file, None, true, None);
        let cmd = a.prepare_command(&c).unwrap();
        assert!(
            cmd.contains("-s read-only"),
            "expected read-only sandbox for sandboxed sender, got: {}",
            cmd
        );
        assert!(
            cmd.contains("-c approval_policy=never"),
            "expected -c approval_policy=never to suppress TTY prompts, got: {}",
            cmd
        );
        assert!(
            !cmd.contains("--dangerously-bypass-approvals-and-sandbox"),
            "sandboxed sender must not get the bypass flag, got: {}",
            cmd
        );
    }

    #[test]
    fn sandboxed_resume_drops_sandbox_keeps_approval_override() {
        // Sandbox is locked at session creation, so resume drops `-s`. The
        // approval-policy config override is allowed on resume and we want
        // it on every turn so a stale config.toml can't reintroduce prompts.
        let mut a = CodexAdapter::new();
        let prompt_file = PathBuf::from("/tmp/p.txt");
        let c = ctx(&prompt_file, Some("uuid-here"), true, None);
        let cmd = a.prepare_command(&c).unwrap();
        assert!(
            !cmd.contains(" -s "),
            "sandboxed resume must not pass -s, got: {}",
            cmd
        );
        assert!(
            !cmd.contains(" -C "),
            "sandboxed resume must not pass -C, got: {}",
            cmd
        );
        assert!(
            cmd.contains("-c approval_policy=never"),
            "expected -c approval_policy=never on sandboxed resume, got: {}",
            cmd
        );
    }

    // ===================== importer tests =====================

    #[test]
    fn persisted_shell_joins_argv_to_bash_command() {
        // `shell.command` is an argv ARRAY; the persisted-tool map joins it to
        // a single claude `Bash` `{command}` string.
        let args = json!({"command":["bash","-lc","ls .."],"workdir":"/tmp"});
        let (name, input) = map_codex_persisted_tool("shell", &args);
        assert_eq!(name, "Bash");
        assert_eq!(input["command"], "bash -lc ls ..");
    }

    #[test]
    fn persisted_exec_command_maps_cmd_to_bash() {
        let args = json!({"cmd":"tail -20 log.txt","workdir":"/tmp","yield_time_ms":1000});
        let (name, input) = map_codex_persisted_tool("exec_command", &args);
        assert_eq!(name, "Bash");
        assert_eq!(input["command"], "tail -20 log.txt");
        // codex metadata dropped — only `command` survives.
        assert!(input.get("workdir").is_none());
    }

    #[test]
    fn persisted_update_plan_maps_to_todowrite() {
        let args = json!({"plan":[{"step":"do X","status":"completed"}]});
        let (name, input) = map_codex_persisted_tool("update_plan", &args);
        assert_eq!(name, "TodoWrite");
        assert_eq!(input["todos"][0]["step"], "do X");
    }

    #[test]
    fn persisted_apply_patch_with_path_maps_to_edit() {
        let args = json!({"path":"src/foo.rs"});
        let (name, input) = map_codex_persisted_tool("apply_patch", &args);
        assert_eq!(name, "Edit");
        assert_eq!(input["file_path"], "src/foo.rs");
    }

    #[test]
    fn persisted_unknown_tool_passes_through() {
        let args = json!({"agent_type":"worker"});
        let (name, input) = map_codex_persisted_tool("spawn_agent", &args);
        assert_eq!(name, "spawn_agent");
        assert_eq!(input["agent_type"], "worker");
    }

    #[test]
    fn function_call_arguments_string_is_parsed() {
        // `arguments` arrives as a JSON-encoded STRING (like cursor's rawArgs).
        let payload = json!({
            "type":"function_call",
            "name":"exec_command",
            "call_id":"call_abc",
            "arguments":"{\"cmd\":\"echo hi\"}"
        });
        let frame = normalize_persisted_function_call(&payload).expect("frame");
        let v: Value = serde_json::from_str(&frame).unwrap();
        let block = &v["message"]["content"][0];
        assert_eq!(block["type"], "tool_use");
        assert_eq!(block["name"], "Bash");
        assert_eq!(block["id"], "call_abc");
        assert_eq!(block["input"]["command"], "echo hi");
    }

    #[test]
    fn response_item_message_filters_injected_dumps() {
        let inj = json!({"role":"user","content":[{"type":"input_text","text":"<user_instructions>\nfoo"}]});
        assert!(response_item_message_text(&inj).is_none());
        let env = json!({"role":"user","content":[{"type":"input_text","text":"<environment_context>\n<cwd>/x</cwd>"}]});
        assert!(response_item_message_text(&env).is_none());
        let real = json!({"role":"user","content":[{"type":"input_text","text":"real prompt"}]});
        assert_eq!(
            response_item_message_text(&real).as_deref(),
            Some("real prompt")
        );
    }

    #[tokio::test]
    async fn parse_session_skips_legacy_bare_head() {
        // Legacy 2025 file: line 1 has `id` but NO `type` and NO `cwd`.
        let dir = std::env::temp_dir().join(format!("codex_test_{}", Uuid::now_v7()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("rollout-2025-07-02T14-36-01-19855ba8.jsonl");
        std::fs::write(
            &path,
            "{\"id\":\"19855ba8-f15f-4d81-a2fe-502e28d2e08a\",\"timestamp\":\"2025-07-02T14:36:01.490Z\",\"instructions\":\"x\"}\n{\"type\":\"message\",\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"hi\"}]}\n",
        )
        .unwrap();
        let parsed = parse_session(&path).await.unwrap();
        assert!(
            parsed.is_none(),
            "legacy bare-head (no cwd) must be skipped"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn parse_session_current_format_end_to_end() {
        let dir = std::env::temp_dir().join(format!("codex_test_{}", Uuid::now_v7()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("rollout-2025-10-03T07-16-55-0199a76d.jsonl");
        // session_meta + injected response_item user dumps + real event_msg
        // prompt + a function_call + an agent_message + the mirrored
        // response_item assistant text (which must NOT double-emit).
        let content = concat!(
            r#"{"timestamp":"2025-10-03T00:16:55.629Z","type":"session_meta","payload":{"id":"0199a76d-cf26-7761-a338-3d456edb725f","timestamp":"2025-10-03T00:16:55.590Z","cwd":"/Users/me/projects/demo"}}"#,
            "\n",
            r#"{"timestamp":"2025-10-03T00:16:56.000Z","type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"<user_instructions>\nblah"}]}}"#,
            "\n",
            r#"{"timestamp":"2025-10-03T00:16:56.100Z","type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"<environment_context>\n<cwd>/x</cwd>"}]}}"#,
            "\n",
            r#"{"timestamp":"2025-10-03T00:16:57.000Z","type":"event_msg","payload":{"type":"user_message","message":"list files please"}}"#,
            "\n",
            r#"{"timestamp":"2025-10-03T00:16:58.000Z","type":"response_item","payload":{"type":"function_call","name":"shell","call_id":"call_1","arguments":"{\"command\":[\"ls\",\"-la\"]}"}}"#,
            "\n",
            r#"{"timestamp":"2025-10-03T00:16:59.000Z","type":"event_msg","payload":{"type":"agent_message","message":"Here are the files."}}"#,
            "\n",
            r#"{"timestamp":"2025-10-03T00:16:59.500Z","type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Here are the files."}]}}"#,
            "\n"
        );
        std::fs::write(&path, content).unwrap();

        let session = parse_session(&path).await.unwrap().expect("session");
        assert_eq!(session.cwd, "/Users/me/projects/demo");
        assert_eq!(
            session.chat_id.to_string(),
            "0199a76d-cf26-7761-a338-3d456edb725f"
        );
        assert_eq!(session.title, "list files please");
        // user prompt + shell tool_use + agent text = 3. The injected dumps
        // and the mirrored response_item assistant text are NOT emitted.
        assert_eq!(session.messages.len(), 3, "expected 3 messages");
        assert_eq!(session.messages[0].sender, "user");
        let user_env: MessageEnvelope = serde_json::from_str(&session.messages[0].body).unwrap();
        assert_eq!(user_env.text, "list files please");
        // shell tool_use
        assert_eq!(session.messages[1].sender, "agent");
        let tool: Value = serde_json::from_str(&session.messages[1].body).unwrap();
        assert_eq!(tool["message"]["content"][0]["name"], "Bash");
        assert_eq!(tool["message"]["content"][0]["input"]["command"], "ls -la");
        // agent text (from event_msg, not the mirrored response_item)
        assert_eq!(session.messages[2].sender, "agent");
        let txt: Value = serde_json::from_str(&session.messages[2].body).unwrap();
        assert_eq!(txt["message"]["content"][0]["text"], "Here are the files.");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn parse_session_falls_back_to_response_item_text_when_no_event_msg() {
        // A session with NO event_msg text rows still imports its real prompt
        // and assistant text from response_item (injected dumps filtered).
        let dir = std::env::temp_dir().join(format!("codex_test_{}", Uuid::now_v7()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("rollout-2025-10-03T07-16-55-0199a770.jsonl");
        let content = concat!(
            r#"{"timestamp":"2025-10-03T00:16:55.629Z","type":"session_meta","payload":{"id":"0199a770-b04e-76c3-b7b9-e556f59ddeab","timestamp":"2025-10-03T00:16:55.590Z","cwd":"/Users/me/projects/demo2"}}"#,
            "\n",
            r#"{"timestamp":"2025-10-03T00:16:56.000Z","type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"<user_instructions>\nblah"}]}}"#,
            "\n",
            r#"{"timestamp":"2025-10-03T00:16:57.000Z","type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"what is 1+1"}]}}"#,
            "\n",
            r#"{"timestamp":"2025-10-03T00:16:58.000Z","type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"2"}]}}"#,
            "\n"
        );
        std::fs::write(&path, content).unwrap();

        let session = parse_session(&path).await.unwrap().expect("session");
        assert_eq!(session.title, "what is 1+1");
        assert_eq!(session.messages.len(), 2);
        assert_eq!(session.messages[0].sender, "user");
        assert_eq!(session.messages[1].sender, "agent");

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ===== prune-context (codex dialect) =================================
    mod prune {
        use super::super::{
            claude_to_codex_tool_names, count_codex_matches, eligible_matches, find_codex_rollout,
            map_codex_persisted_tool, prune_codex_jsonl, CODEX_TOOL_NAME_MAP,
        };
        use crate::prune::test_util::{read_lines, write_jsonl};

        /// codex `response_item`/`function_call` line. `arguments` is a
        /// JSON-ENCODED STRING on the wire (e.g. `"{\"command\":[...]}"`); we store
        /// `args_json` as that string verbatim — the pruner substring-probes the
        /// string and blanks it to `"{}"`, never re-parses it.
        fn codex_call(call_id: &str, name: &str, args_json: &str) -> String {
            serde_json::json!({
                "timestamp": "2025-10-04T12:00:00.000Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": name,
                    "arguments": args_json,
                    "call_id": call_id,
                }
            })
            .to_string()
        }

        /// codex `response_item`/`function_call_output` line keyed by `call_id`. The
        /// bulky `output` is a JSON-encoded string on the real wire; a plain string
        /// is sufficient for the field-blanking test.
        fn codex_output(call_id: &str, output: &str) -> String {
            serde_json::json!({
                "timestamp": "2025-10-04T12:00:01.000Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output,
                }
            })
            .to_string()
        }

        fn session_meta(id: &str, cwd: &str) -> String {
            serde_json::json!({
                "timestamp": "2025-10-04T12:00:00.000Z",
                "type": "session_meta",
                "payload": { "id": id, "cwd": cwd }
            })
            .to_string()
        }

        #[test]
        fn codex_tool_name_map_round_trips() {
            // Lockstep guard: every mapped codex tool must round-trip — the forward
            // map `map_codex_persisted_tool` emits it under the table's claude name,
            // and `claude_to_codex_tool_names` maps that claude name back to the
            // codex tool. Driving CODEX_TOOL_NAME_MAP as the single source means a
            // rename can't silently break prune matching for that tool. Empty args
            // exercise each tool's PRIMARY path (e.g. `apply_patch` → `Edit`, not
            // its no-path `Bash` rendering fallback).
            for &(codex, claude) in CODEX_TOOL_NAME_MAP {
                let (emitted, _) = map_codex_persisted_tool(codex, &serde_json::json!({}));
                assert_eq!(
                    emitted, claude,
                    "map_codex_persisted_tool({codex}) must emit claude name {claude}",
                );
                assert!(
                    claude_to_codex_tool_names(claude).contains(&codex),
                    "claude_to_codex_tool_names({claude}) must contain {codex}",
                );
            }
            // Fan-out order is preserved; an unmapped / already-codex name yields
            // empty (the caller then matches it literally).
            assert_eq!(
                claude_to_codex_tool_names("Bash"),
                vec!["shell", "exec_command"]
            );
            assert!(claude_to_codex_tool_names("shell").is_empty());
        }

        #[test]
        fn codex_count_matches_maps_claude_name_to_codex_tools() {
            let f = write_jsonl(&[
                &session_meta("0199af34-e7f8-7f32-b1da-a5a4053adb84", "/p"),
                &codex_call("c1", "shell", r#"{"command":["bash","-lc","cat junk.rs"]}"#),
                &codex_output("c1", "BIG"),
                &codex_call("c2", "shell", r#"{"command":["bash","-lc","cat keep.rs"]}"#),
                &codex_call("c3", "apply_patch", r#"{"path":"junk.rs"}"#),
            ]);
            // claude "Bash" + needle "junk.rs" → only c1 (shell with junk.rs).
            assert_eq!(count_codex_matches(f.path(), "Bash", "junk.rs").unwrap(), 1);
            // claude "Edit" maps to apply_patch → c3 only.
            assert_eq!(count_codex_matches(f.path(), "Edit", "junk.rs").unwrap(), 1);
            assert_eq!(count_codex_matches(f.path(), "Bash", "nope").unwrap(), 0);
        }

        #[test]
        fn codex_empty_tool_name_matches_any_tool_by_args() {
            // Omitting `--tool-name` (the empty "any tool" selector) prunes on the
            // args needle alone — across BOTH codex shell tools (shell AND
            // exec_command) and any other tool. This is why the codex instruction
            // drops --tool-name: no single native name covers its file ops.
            let f = write_jsonl(&[
                &session_meta("0199af34-e7f8-7f32-b1da-a5a4053adb84", "/p"),
                &codex_call("c1", "shell", r#"{"command":["bash","-lc","cat junk.rs"]}"#),
                &codex_output("c1", "BIG"),
                &codex_call("c2", "exec_command", r#"{"cmd":"rg junk.rs ."}"#),
                &codex_output("c2", "BIG2"),
                &codex_call("c3", "shell", r#"{"command":["bash","-lc","cat keep.rs"]}"#),
                &codex_output("c3", "KEEP"),
            ]);
            // Empty tool name + "junk.rs" → c1 (shell) AND c2 (exec_command); not c3.
            assert_eq!(count_codex_matches(f.path(), "", "junk.rs").unwrap(), 2);
            // Last-only prune blanks the most recent match (c2); c1 and c3 survive.
            let stats = prune_codex_jsonl(f.path(), "", "junk.rs").unwrap();
            assert_eq!(stats.results_blanked, 1);
            let lines = read_lines(f.path());
            assert_eq!(lines[2]["payload"]["output"], "BIG");
            assert_eq!(lines[4]["payload"]["output"], "[pruned]");
            assert_eq!(lines[6]["payload"]["output"], "KEEP");
        }

        #[test]
        fn codex_empty_args_selector_matches_only_no_arg_calls() {
            // A no-arg MCP call persists `arguments` as the encoded string "{}"
            // (verified on disk against codex 0.135.0). `--args ""` must select it
            // and spare the same tool's with-args call. The MCP name inverts to the
            // bare on-disk tool name via codex_mcp_ondisk_name_candidates.
            let f = write_jsonl(&[
                &session_meta("0199af34-e7f8-7f32-b1da-a5a4053adb84", "/p"),
                &codex_call("m1", "get_last_id", "{}"),
                &codex_output("m1", "BULKY RESULT"),
                &codex_call("m2", "get_last_id", r#"{"limit":10}"#),
                &codex_output("m2", "OTHER"),
            ]);
            // Empty args + MCP wire name → only the no-arg m1.
            assert_eq!(
                count_codex_matches(f.path(), "mcp__zdbg__get_last_id", "").unwrap(),
                1
            );
            // Prune blanks m1's output, leaves m2 intact.
            let stats = prune_codex_jsonl(f.path(), "mcp__zdbg__get_last_id", "").unwrap();
            assert_eq!(stats.results_blanked, 1);
            let lines = read_lines(f.path());
            assert_eq!(lines[2]["payload"]["output"], "[pruned]");
            assert_eq!(lines[4]["payload"]["output"], "OTHER");
        }

        #[test]
        fn codex_matches_quoted_needle_against_decoded_arguments() {
            // REGRESSION: a bare-quote needle copied from the chat must match the
            // DECODED `cmd` leaf, not the `\"`-escaped raw form (see
            // `codex_arguments_match`).
            let f = write_jsonl(&[
                &session_meta("0199af34-e7f8-7f32-b1da-a5a4053adb84", "/p"),
                &codex_call("c1", "exec_command", r#"{"cmd":"rg -n \"ABN\" ."}"#),
                &codex_output("c1", "BULKY rg OUTPUT WITH ABN HITS"),
            ]);
            let needle = r#"rg -n "ABN""#; // bare quotes, as rendered in the chat
                                           // The raw escaped form does NOT contain the bare-quote needle…
            assert!(!r#"{"cmd":"rg -n \"ABN\" ."}"#.contains(needle));
            // …but the matcher finds it via the decoded `cmd` leaf.
            assert_eq!(count_codex_matches(f.path(), "Bash", needle).unwrap(), 1);
            let stats = prune_codex_jsonl(f.path(), "Bash", needle).unwrap();
            assert_eq!(stats.results_blanked, 1);
            let lines = read_lines(f.path());
            assert_eq!(lines[1]["payload"]["arguments"], "{}");
            assert_eq!(lines[2]["payload"]["output"], "[pruned]");
            let raw = std::fs::read_to_string(f.path()).unwrap();
            assert!(
                !raw.contains("BULKY rg OUTPUT"),
                "stale output survived: {raw}"
            );
        }

        #[test]
        fn codex_prune_blanks_function_call_and_paired_output() {
            // THE pairing test: a function_call matched by (mapped name, needle) has
            // its arguments blanked to "{}" and its paired function_call_output
            // (same call_id) has its output blanked to [pruned]. An unmatched
            // call/output pair is untouched. No line deleted.
            let f = write_jsonl(&[
                &session_meta("0199af34-e7f8-7f32-b1da-a5a4053adb84", "/p"),
                &codex_call("c1", "shell", r#"{"command":["bash","-lc","cat junk.rs"]}"#),
                &codex_output("c1", "BULKY FILE BODY OF junk.rs"),
                &codex_call("c2", "shell", r#"{"command":["bash","-lc","cat keep.rs"]}"#),
                &codex_output("c2", "KEEP BODY"),
            ]);
            let stats = prune_codex_jsonl(f.path(), "Bash", "junk.rs").unwrap();
            // One distinct matched call_id (c1).
            assert_eq!(stats.results_blanked, 1);
            // No surviving copy of the bulky body.
            let raw = std::fs::read_to_string(f.path()).unwrap();
            assert!(
                !raw.contains("BULKY FILE BODY"),
                "stale output survived: {raw}"
            );
            let lines = read_lines(f.path());
            // Nothing deleted: session_meta + 4 body lines.
            assert_eq!(lines.len(), 5);
            // c1 arguments blanked to "{}" (still a JSON-encoded string).
            assert_eq!(lines[1]["payload"]["arguments"], "{}");
            // c1 output blanked.
            assert_eq!(lines[2]["payload"]["output"], "[pruned]");
            // c2 untouched.
            assert_eq!(
                lines[3]["payload"]["arguments"],
                r#"{"command":["bash","-lc","cat keep.rs"]}"#
            );
            assert_eq!(lines[4]["payload"]["output"], "KEEP BODY");
        }

        #[test]
        fn codex_needle_equal_to_key_name_does_not_match() {
            // `--args` globs argument VALUES, never KEY names: "command" / "path" are
            // keys → no match (the old raw-string probe over-matched them); values match.
            let f = write_jsonl(&[
                &session_meta("0199af34-e7f8-7f32-b1da-a5a4053adb84", "/p"),
                &codex_call(
                    "c1",
                    "shell",
                    r#"{"command":["bash","-lc","cat src/main.rs"]}"#,
                ),
                &codex_output("c1", "x"),
                &codex_call("c2", "apply_patch", r#"{"path":"src/main.rs"}"#),
                &codex_output("c2", "y"),
            ]);
            // Key names never match.
            assert_eq!(count_codex_matches(f.path(), "Bash", "command").unwrap(), 0);
            assert_eq!(count_codex_matches(f.path(), "Edit", "path").unwrap(), 0);
            // The value still matches.
            assert_eq!(count_codex_matches(f.path(), "Bash", "main.rs").unwrap(), 1);
            assert_eq!(count_codex_matches(f.path(), "Edit", "main.rs").unwrap(), 1);
        }

        #[test]
        fn codex_glob_wildcard_matches_value_leaf() {
            // `*`-separated segments must appear in order within one decoded VALUE
            // leaf (mirrors claude `glob_wildcard_matches_value_leaf`).
            let f = write_jsonl(&[
                &session_meta("0199af34-e7f8-7f32-b1da-a5a4053adb84", "/p"),
                &codex_call(
                    "c1",
                    "shell",
                    r#"{"command":["bash","-lc","psql -c 'SELECT x FROM analytics_186081460 WHERE day = 2026-05-25 AND channel = organic'"]}"#,
                ),
                &codex_output("c1", "ROWS"),
            ]);
            assert_eq!(
                count_codex_matches(
                    f.path(),
                    "Bash",
                    "analytics_186081460*WHERE*2026-05-25*organic"
                )
                .unwrap(),
                1
            );
            // Out-of-order segments don't match.
            assert_eq!(
                count_codex_matches(f.path(), "Bash", "organic*analytics_186081460").unwrap(),
                0
            );
        }

        #[test]
        fn codex_last_only_walks_newest_to_oldest_across_calls() {
            // Two same-tool calls both matching the needle: each prune blanks only the
            // MOST RECENT eligible one, the next prune blanks the older, a third reports
            // zero (mirrors claude `last_only_walks_newest_to_oldest_across_calls`).
            let f = write_jsonl(&[
                &session_meta("0199af34-e7f8-7f32-b1da-a5a4053adb84", "/p"),
                &codex_call("c1", "shell", r#"{"command":["bash","-lc","cat junk.rs"]}"#),
                &codex_output("c1", "FIRST BODY"),
                &codex_call("c2", "shell", r#"{"command":["bash","-lc","cat junk.rs"]}"#),
                &codex_output("c2", "SECOND BODY"),
            ]);
            // Both eligible; document order is [c1, c2].
            assert_eq!(
                eligible_matches(f.path(), "Bash", "junk.rs").unwrap(),
                vec!["c1", "c2"]
            );

            // First prune: only the most recent (c2) is blanked; c1 untouched.
            assert_eq!(
                prune_codex_jsonl(f.path(), "Bash", "junk.rs")
                    .unwrap()
                    .results_blanked,
                1
            );
            let lines = read_lines(f.path());
            assert_eq!(lines.len(), 5);
            assert_eq!(lines[2]["payload"]["output"], "FIRST BODY");
            assert_eq!(lines[3]["payload"]["arguments"], "{}");
            assert_eq!(lines[4]["payload"]["output"], "[pruned]");
            // c2 now ineligible (output is `[pruned]`); only c1 remains.
            assert_eq!(
                eligible_matches(f.path(), "Bash", "junk.rs").unwrap(),
                vec!["c1"]
            );

            // Second prune: walks back to c1.
            assert_eq!(
                prune_codex_jsonl(f.path(), "Bash", "junk.rs")
                    .unwrap()
                    .results_blanked,
                1
            );
            let lines = read_lines(f.path());
            assert_eq!(lines[1]["payload"]["arguments"], "{}");
            assert_eq!(lines[2]["payload"]["output"], "[pruned]");

            // Third prune: nothing eligible left → reports 0, safe no-op.
            assert_eq!(count_codex_matches(f.path(), "Bash", "junk.rs").unwrap(), 0);
            assert_eq!(
                prune_codex_jsonl(f.path(), "Bash", "junk.rs")
                    .unwrap()
                    .results_blanked,
                0
            );
        }

        #[test]
        fn codex_empty_args_selector_walks_newest_to_oldest() {
            // `--args ""` over multiple no-arg calls of one tool: the blanked args stay
            // `"{}"` (still "no-args"), so without the output-pruned eligibility guard a
            // second prune would re-hit the same call. The guard makes it walk back
            // (mirrors claude `empty_args_selector_walks_newest_to_oldest`).
            let f = write_jsonl(&[
                &session_meta("0199af34-e7f8-7f32-b1da-a5a4053adb84", "/p"),
                &codex_call("m1", "get_last_id", "{}"),
                &codex_output("m1", "PING1"),
                &codex_call("m2", "get_last_id", "{}"),
                &codex_output("m2", "PING2"),
            ]);
            assert_eq!(
                count_codex_matches(f.path(), "mcp__zdbg__get_last_id", "").unwrap(),
                2
            );
            // First prune → most recent (m2).
            prune_codex_jsonl(f.path(), "mcp__zdbg__get_last_id", "").unwrap();
            let lines = read_lines(f.path());
            assert_eq!(lines[2]["payload"]["output"], "PING1");
            assert_eq!(lines[4]["payload"]["output"], "[pruned]");
            assert_eq!(
                count_codex_matches(f.path(), "mcp__zdbg__get_last_id", "").unwrap(),
                1
            );
            // Second prune → walks back to m1 (guard excludes the pruned m2).
            prune_codex_jsonl(f.path(), "mcp__zdbg__get_last_id", "").unwrap();
            let lines = read_lines(f.path());
            assert_eq!(lines[2]["payload"]["output"], "[pruned]");
            assert_eq!(lines[4]["payload"]["output"], "[pruned]");
            assert_eq!(
                count_codex_matches(f.path(), "mcp__zdbg__get_last_id", "").unwrap(),
                0
            );
        }

        #[test]
        fn codex_prune_output_without_matched_call_is_untouched() {
            // A function_call_output whose call_id was never matched stays intact
            // (the forward pass only blanks outputs for collected ids).
            let f = write_jsonl(&[
                &session_meta("0199af34-e7f8-7f32-b1da-a5a4053adb84", "/p"),
                &codex_output("orphan", "SOME OUTPUT"),
            ]);
            let stats = prune_codex_jsonl(f.path(), "Bash", "junk.rs").unwrap();
            assert_eq!(stats.results_blanked, 0);
            assert_eq!(stats.freed_bytes, 0);
            let lines = read_lines(f.path());
            assert_eq!(lines[1]["payload"]["output"], "SOME OUTPUT");
        }

        #[test]
        fn codex_prune_reports_freed_bytes() {
            // A bulky output → freed_bytes reflects (roughly) the blanked payload.
            let big = "x".repeat(4000);
            let f = write_jsonl(&[
                &session_meta("0199af34-e7f8-7f32-b1da-a5a4053adb84", "/p"),
                &codex_call("c1", "shell", r#"{"command":["bash","-lc","cat junk.rs"]}"#),
                &codex_output("c1", &big),
            ]);
            let stats = prune_codex_jsonl(f.path(), "Bash", "junk.rs").unwrap();
            assert_eq!(stats.results_blanked, 1);
            // The 4000-char body dominates; freed should be well above it.
            assert!(
                stats.freed_bytes > 3900,
                "freed_bytes = {}",
                stats.freed_bytes
            );
        }

        #[test]
        fn codex_prune_unknown_tool_name_matches_literally() {
            let f = write_jsonl(&[
                &session_meta("0199af34-e7f8-7f32-b1da-a5a4053adb84", "/p"),
                &codex_call("c1", "some_mcp_tool", r#"{"path":"junk.rs"}"#),
                &codex_output("c1", "BODY"),
            ]);
            // An unknown/forwarded tool name (not in the claude→codex map) is
            // matched literally against the on-disk function_call name.
            let stats = prune_codex_jsonl(f.path(), "some_mcp_tool", "junk.rs").unwrap();
            assert_eq!(stats.results_blanked, 1);
            let lines = read_lines(f.path());
            assert_eq!(lines[1]["payload"]["arguments"], "{}");
            assert_eq!(lines[2]["payload"]["output"], "[pruned]");
        }

        #[test]
        fn codex_prune_matches_mcp_wire_name_against_bare_ondisk_tool() {
            // Wire name `mcp__zdbg__get_last_id` inverts to the BARE on-disk name
            // `get_last_id` (server dropped; see `codex_mcp_ondisk_name_candidates`).
            // The bare name collides across servers; last-only keeps it safe (blanks
            // only the newest of two same-named calls, agent walks back).
            let f = write_jsonl(&[
                &session_meta("0199af34-e7f8-7f32-b1da-a5a4053adb84", "/p"),
                // First bare `get_last_id` (e.g. server `zdbg`).
                &codex_call("m1", "get_last_id", r#"{"query":"SELECT 1"}"#),
                &codex_output("m1", "BULKY MCP RESULT A"),
                // Second bare `get_last_id` from ANOTHER server — same on-disk name.
                &codex_call("m2", "get_last_id", r#"{"query":"SELECT 2"}"#),
                &codex_output("m2", "BULKY MCP RESULT B"),
            ]);
            // The wire name maps to the bare tool, so BOTH bare calls are eligible
            // (empty needle → no-args selector would miss them; use a value the SQL
            // shares). Document order is [m1, m2].
            assert_eq!(
                eligible_matches(f.path(), "mcp__zdbg__get_last_id", "SELECT").unwrap(),
                vec!["m1", "m2"]
            );
            assert_eq!(
                count_codex_matches(f.path(), "mcp__zdbg__get_last_id", "SELECT").unwrap(),
                2
            );

            // First prune: only the most recent (m2) is blanked; m1 untouched.
            let stats = prune_codex_jsonl(f.path(), "mcp__zdbg__get_last_id", "SELECT").unwrap();
            assert_eq!(stats.results_blanked, 1);
            let lines = read_lines(f.path());
            assert_eq!(lines[2]["payload"]["output"], "BULKY MCP RESULT A");
            assert_eq!(lines[3]["payload"]["arguments"], "{}");
            assert_eq!(lines[4]["payload"]["output"], "[pruned]");
            // m2 now ineligible; only m1 remains.
            assert_eq!(
                eligible_matches(f.path(), "mcp__zdbg__get_last_id", "SELECT").unwrap(),
                vec!["m1"]
            );

            // Second prune walks back to m1.
            let stats = prune_codex_jsonl(f.path(), "mcp__zdbg__get_last_id", "SELECT").unwrap();
            assert_eq!(stats.results_blanked, 1);
            let lines = read_lines(f.path());
            assert_eq!(lines[1]["payload"]["arguments"], "{}");
            assert_eq!(lines[2]["payload"]["output"], "[pruned]");

            // Third prune: nothing eligible left → reports 0, safe no-op.
            assert_eq!(
                count_codex_matches(f.path(), "mcp__zdbg__get_last_id", "SELECT").unwrap(),
                0
            );
            assert_eq!(
                prune_codex_jsonl(f.path(), "mcp__zdbg__get_last_id", "SELECT")
                    .unwrap()
                    .results_blanked,
                0
            );
            let raw = std::fs::read_to_string(f.path()).unwrap();
            assert!(!raw.contains("BULKY MCP RESULT"), "stale MCP output: {raw}");
        }

        #[test]
        fn codex_mcp_ondisk_name_candidates_inverts_wire_name() {
            use super::super::codex_mcp_ondisk_name_candidates;
            // Full server-qualified wire name → [bare tool, server__tool, full].
            assert_eq!(
                codex_mcp_ondisk_name_candidates("mcp__zdbg__get_last_id"),
                Some(vec![
                    "get_last_id".to_string(),
                    "zdbg__get_last_id".to_string(),
                    "mcp__zdbg__get_last_id".to_string(),
                ])
            );
            // Degraded `mcp__<tool>` form (only the tool was known at render time):
            // bare tool == remainder, so no duplicate remainder entry.
            assert_eq!(
                codex_mcp_ondisk_name_candidates("mcp__solo_tool"),
                Some(vec!["solo_tool".to_string(), "mcp__solo_tool".to_string()])
            );
            // Non-MCP names fall through to the static map (None here).
            assert_eq!(codex_mcp_ondisk_name_candidates("Bash"), None);
            assert_eq!(codex_mcp_ondisk_name_candidates("exec_command"), None);
            // Empty remainder is not a valid MCP name.
            assert_eq!(codex_mcp_ondisk_name_candidates("mcp__"), None);
        }

        #[test]
        fn codex_prune_context_call_excluded_so_real_read_is_pruned() {
            // The agent's own in-flight prune-context call carries the needle in its
            // argv but is excluded from eligibility (see
            // `crate::prune::value_is_prune_context_call`), so the real read (c1) is
            // pruned instead of it.
            let f = write_jsonl(&[
                &session_meta("0199af34-e7f8-7f32-b1da-a5a4053adb84", "/p"),
                // A real prior call WITH an output.
                &codex_call("c1", "shell", r#"{"command":["bash","-lc","cat junk.rs"]}"#),
                &codex_output("c1", "BULKY junk.rs BODY"),
                // The in-flight prune-context call: matches the needle via its args
                // but is the agent's own pruning command → must be excluded.
                &codex_call(
                    "inflight",
                    "shell",
                    r#"{"command":["bash","-lc","zucchini-spawner prune-context --tool-name Bash --args junk.rs"]}"#,
                ),
            ]);
            // Only the real read is eligible; the prune-context call is excluded.
            assert_eq!(
                eligible_matches(f.path(), "Bash", "junk.rs").unwrap(),
                vec!["c1"]
            );
            let stats = prune_codex_jsonl(f.path(), "Bash", "junk.rs").unwrap();
            // The real read's output IS blanked → count 1, frame shown.
            assert_eq!(stats.results_blanked, 1);
            let lines = read_lines(f.path());
            assert_eq!(lines.len(), 4);
            // c1's arguments + output are pruned.
            assert_eq!(lines[1]["payload"]["arguments"], "{}");
            assert_eq!(lines[2]["payload"]["output"], "[pruned]");
            // The in-flight prune-context call is left FULLY intact.
            assert_eq!(
                lines[3]["payload"]["arguments"],
                r#"{"command":["bash","-lc","zucchini-spawner prune-context --tool-name Bash --args junk.rs"]}"#
            );
        }

        #[test]
        fn codex_prune_blanks_output_appearing_before_its_call() {
            // Defensive two-pass: even if a function_call_output somehow precedes its
            // own function_call in the file (a single forward pass would MISS it and
            // silently fail open — resume reloads the unpruned output), the two-pass
            // collect-then-blank still catches it.
            let bulky = "OUT-OF-ORDER BULKY BODY THAT MUST STILL BE PRUNED";
            let f = write_jsonl(&[
                &session_meta("0199af34-e7f8-7f32-b1da-a5a4053adb84", "/p"),
                // Output line FIRST.
                &codex_output("c1", bulky),
                // Its matching call SECOND (reversed order).
                &codex_call("c1", "shell", r#"{"command":["bash","-lc","cat junk.rs"]}"#),
            ]);
            let stats = prune_codex_jsonl(f.path(), "Bash", "junk.rs").unwrap();
            assert_eq!(stats.results_blanked, 1);
            let raw = std::fs::read_to_string(f.path()).unwrap();
            assert!(!raw.contains(bulky), "out-of-order output survived: {raw}");
            let lines = read_lines(f.path());
            assert_eq!(lines[1]["payload"]["output"], "[pruned]");
            assert_eq!(lines[2]["payload"]["arguments"], "{}");
        }

        #[test]
        fn codex_re_prune_of_already_pruned_rollout_reports_zero() {
            // Idempotency: running prune twice over the same rollout must not
            // double-count or corrupt. The second run sees `[pruned]`/`"{}"` already
            // in place and reports `results_blanked == 0` (matching claude's `already`
            // guard), leaving the file byte-stable.
            let f = write_jsonl(&[
                &session_meta("0199af34-e7f8-7f32-b1da-a5a4053adb84", "/p"),
                &codex_call("c1", "shell", r#"{"command":["bash","-lc","cat junk.rs"]}"#),
                &codex_output("c1", "BULKY BODY"),
            ]);
            let first = prune_codex_jsonl(f.path(), "Bash", "junk.rs").unwrap();
            assert_eq!(first.results_blanked, 1);
            let after_first = std::fs::read_to_string(f.path()).unwrap();
            let second = prune_codex_jsonl(f.path(), "Bash", "junk.rs").unwrap();
            assert_eq!(second.results_blanked, 0, "re-prune should report nothing");
            assert_eq!(second.freed_bytes, 0);
            let after_second = std::fs::read_to_string(f.path()).unwrap();
            assert_eq!(after_first, after_second, "re-prune must be byte-stable");
        }

        #[test]
        fn find_codex_rollout_matches_session_meta_payload_id() {
            // Fallback content-scan path: the filename suffix here (`-DEADBEEF.jsonl`)
            // does NOT embed the thread id, so the fast path misses and resolution
            // falls back to matching `session_meta.payload.id` on line 0.
            //
            // `base` is the codex home dir (what `AgentKind::cli_home` resolves);
            // `find_codex_rollout` searches `<base>/sessions/**`. We pass a temp
            // dir directly — no `HOME` mutation, no env races.
            let base = std::env::temp_dir().join(format!(
                "zucchini_codex_find_{}_{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            let day = base.join("sessions").join("2025").join("10").join("04");
            std::fs::create_dir_all(&day).unwrap();
            let tid = "0199af34-e7f8-7f32-b1da-a5a4053adb84";
            // Filename uuid deliberately differs from the header payload.id.
            let file = day.join("rollout-2025-10-04T19-31-44-DEADBEEF.jsonl");
            std::fs::write(
                &file,
                format!("{}\n{}\n", session_meta(tid, "/p"), codex_output("c1", "x")),
            )
            .unwrap();

            let found = find_codex_rollout(&base, tid);
            let miss = find_codex_rollout(&base, "00000000-0000-0000-0000-000000000000");
            assert_eq!(found.as_deref(), Some(file.as_path()));
            assert!(miss.is_none());
            let _ = std::fs::remove_dir_all(&base);
        }

        #[test]
        fn find_codex_rollout_uses_filename_suffix_fast_path() {
            // The thread id is the trailing UUID in the filename
            // (`rollout-<ISO8601>-<thread_id>.jsonl`). The fast path must resolve it
            // WITHOUT depending on line-0 content: here the header `session_meta`
            // carries a DIFFERENT id, yet the filename suffix still wins.
            //
            // `base` is the codex home dir passed straight in (no `HOME` swap).
            let base = std::env::temp_dir().join(format!(
                "zucchini_codex_fastpath_{}_{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            let day = base.join("sessions").join("2025").join("10").join("04");
            std::fs::create_dir_all(&day).unwrap();
            let tid = "0199af34-e7f8-7f32-b1da-a5a4053adb84";
            let file = day.join(format!("rollout-2025-10-04T19-31-44-{tid}.jsonl"));
            // Header id deliberately DIFFERS from the filename suffix — a content
            // scan would NOT find `tid` here, so a pass proves the filename path.
            std::fs::write(
                &file,
                format!(
                    "{}\n{}\n",
                    session_meta("00000000-0000-0000-0000-000000000000", "/p"),
                    codex_output("c1", "x")
                ),
            )
            .unwrap();

            let found = find_codex_rollout(&base, tid);
            assert_eq!(
                found.as_deref(),
                Some(file.as_path()),
                "filename suffix fast-path must resolve the rollout"
            );
            let _ = std::fs::remove_dir_all(&base);
        }
    }
}
