//! codex (openai codex-cli) adapter. Normalizes codex's `exec --json` stream
//! frames into claude-shape envelopes on the wire so iOS's
//! `SpawnerMessageDescriber` (which only knows claude's wire format) can
//! render them WITHOUT any codex-specific branches in the iOS code: every
//! codex tool item is remapped to the equivalent claude tool name on the
//! wire so iOS's toolSummary table picks it up as if it were a claude
//! tool_use. Single seam, single source of truth.
//!
//! Frame mapping (codex → claude-shape):
//!  - `thread.started`           → `SessionIdHarvested` only (no Frame; matches claude's init-skip)
//!  - `turn.started`             → drop (no claude analog)
//!  - `item.started`             → drop for v1 (we forward completions only;
//!                                 a future pass may surface in-flight tool
//!                                 status the way cursor's tool_call.started
//!                                 is currently dropped)
//!  - `item.completed` agent_message       → Frame: claude-shape assistant text envelope
//!  - `item.completed` command_execution   → Frame: claude `Bash` tool_use `{command}`
//!  - `item.completed` file_change         → one Frame per change:
//!                                            kind=add    → claude `Write` tool_use `{file_path}`
//!                                            kind=update → claude `Edit`  tool_use `{file_path}`
//!                                            kind=delete → claude `Bash`  tool_use `{command: "rm <path>"}`
//!                                            (no dedicated claude delete tool; `rm` is what claude
//!                                            code itself emits for the same intent)
//!  - `item.completed` web_search          → Frame: claude `WebSearch` tool_use `{query}`
//!  - `turn.completed`           → ContextTokens(input_tokens)
//!                                 + Frame (claude-shape result envelope)
//!                                 + Result
//!  - `turn.failed`              → Frame (claude-shape error result envelope)
//!                                 + Result
//!  - anything else              → forwarded as-is (defensive against codex
//!                                 format drift; iOS will likely drop, but
//!                                 we avoid silently losing the line)
//!
//! Codex's actual wire format observed via `codex exec --json` on
//! codex-cli 0.133.0: each turn starts with `thread.started` carrying a
//! UUIDv7 thread_id, ends with `turn.completed` carrying a `usage` block
//! ({input_tokens, cached_input_tokens, output_tokens, reasoning_output_tokens}).
//! Between them, every artifact (agent text, file edit, shell command, web
//! search) lands as an `item.completed` carrying a payload whose shape is
//! pinned by `codex-rs/exec/src/exec_events.rs` (CommandExecutionItem,
//! FileChangeItem, WebSearchItem). Codex tool names never reach iOS — the
//! field projection below picks the canonical claude tool's primary input
//! field (`command` / `file_path` / `query`) so the existing
//! `SpawnerMessageDescriber.toolSummary` switch renders them.
//!
//! Also hosts the install/auth `probe()` for codex (free function, not on
//! the `AgentAdapter` trait — `dyn AgentAdapter` can't dispatch statics).
//! For "authenticated" we stat `~/.codex/auth.json` (chatgpt SSO writes its
//! token blob there); missing file → not authenticated. Don't shell out —
//! `codex` has no `status`-style sub-command stable enough to lean on
//! across versions (TODO: revisit once codex ships one).
//!
//! `import()` is a stub for v1. Codex's on-disk rollouts live at
//! `~/.codex/sessions/YYYY/MM/DD/rollout-<ts>-<uuid>.jsonl`; a follow-up
//! pass will walk them and emit PutProject/PutChat/PutMessage like the
//! claude / cursor importers do.

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
};

fn make_boxed() -> Box<dyn AgentAdapter> {
    Box::new(CodexAdapter::new())
}

/// Per-turn state for the codex adapter. Only carries the
/// `last_emitted_tokens` dedup so a hypothetical multi-`turn.completed` run
/// (today: one per turn; tomorrow: maybe streaming usage updates) doesn't
/// double-fire ContextTokens on identical values. Mirrors claude's per-turn
/// dedup field (`adapter::LastTokensDedup`).
#[derive(Default)]
pub struct CodexAdapter {
    last_emitted_tokens: LastTokensDedup,
}

impl CodexAdapter {
    pub fn new() -> Self {
        Self::default()
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
                // v1: forward only completions. A future pass may surface
                // in-flight tool starts the way cursor's tool_call.started
                // is currently dropped.
                debug!("codex item.started dropped (v1)");
            }
            "item.completed" => {
                let frames = normalize_item_completed(&obj);
                if frames.is_empty() {
                    debug!("codex item.completed without renderable item, dropping");
                }
                for frame in frames {
                    out.push(AgentEvent::Frame(frame));
                }
            }
            "turn.completed" => {
                // Cumulative for the thread — closest analog to claude's
                // per-frame `input + cache_creation + cache_read`. Codex's
                // upstream `input_tokens` already includes cache hits
                // (`non_cached_input = input_tokens - cached_input_tokens`),
                // so adding `cached_input_tokens` here double-counts cached
                // context. We dedup against `last_emitted_tokens` so a
                // hypothetical streamed-usage variant (or a re-emitted final
                // frame) doesn't double-fire PATCH on identical values.
                if let Some(tokens) = parse_turn_completed_tokens(&obj) {
                    if let Some(t) = self.last_emitted_tokens.observe(tokens) {
                        out.push(AgentEvent::ContextTokens(t));
                    }
                }
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

/// Reads cumulative input tokens for the turn from a `turn.completed` frame.
/// `input_tokens` is the context size: Codex reports cached input separately
/// as a subset of total input, not an additional count. Narrow Deserialize
/// struct so serde skips the rest of the frame without allocating it.
fn parse_turn_completed_tokens(obj: &Value) -> Option<i64> {
    #[derive(Deserialize)]
    struct Frame {
        #[serde(default)]
        usage: Option<Usage>,
    }
    #[derive(Deserialize, Default)]
    struct Usage {
        #[serde(default)]
        input_tokens: i64,
    }
    // `serde_json::from_value` clones the subtree, but `usage` is a tiny
    // four-number object so this is cheap. Going via `from_value` keeps the
    // dispatch code path (`Value`-shaped) consistent with the rest of the
    // handler.
    match serde_json::from_value::<Frame>(obj.clone()) {
        Ok(f) => {
            let u = f.usage?;
            Some(u.input_tokens)
        }
        Err(e) => {
            debug!("failed to parse codex turn.completed usage: {}", e);
            None
        }
    }
}

/// Converts a codex `item.completed` frame to claude-shape envelopes. Returns
/// an empty vec when the item type isn't one we know how to render (caller
/// drops with a debug log). `file_change` is the only multi-frame case —
/// codex bundles N file edits in one item, we fan them out so each shows up
/// as its own claude-tool bubble line.
///
/// Mapping (see file-level doc for the full table):
///   agent_message     → assistant text envelope
///   command_execution → claude `Bash`     tool_use `{command}`
///   file_change       → per-change: claude `Write` / `Edit` / `Bash(rm)`
///                       tool_use `{file_path}` (or `{command}` for delete)
///   web_search        → claude `WebSearch` tool_use `{query}`
///
/// Field projection only — we drop codex-side metadata (aggregated_output,
/// exit_code, durationMs, action variants, patch status, …) because iOS
/// only summarizes the primary input field anyway. Keeping the wire close
/// to claude's own shape means no codex-specific branches in iOS.
fn normalize_item_completed(obj: &Value) -> Vec<String> {
    let Some(item) = obj.get("item") else {
        return Vec::new();
    };
    let Some(item_type) = item.get("type").and_then(|v| v.as_str()) else {
        return Vec::new();
    };
    let item_id = item
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    match item_type {
        "agent_message" => {
            // Per-frame usage isn't available on item.completed; the real
            // ContextTokens lands at turn.completed. The shared envelope
            // helper stamps the claude-shape zero `usage` so iOS sees the
            // same key set across adapters mid-turn.
            let text = item.get("text").and_then(|v| v.as_str()).unwrap_or("");
            vec![claude_assistant_text_envelope(text)]
        }
        "command_execution" => {
            let command = item
                .get("command")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
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
            let query = item
                .get("query")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            vec![claude_tool_use_envelope(
                &item_id,
                "WebSearch",
                json!({ "query": query }),
            )]
        }
        _ => Vec::new(),
    }
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

/// One-shot per-kind history importer. Codex stores rollouts at
/// `~/.codex/sessions/YYYY/MM/DD/rollout-<ts>-<uuid>.jsonl` (one file per
/// turn-or-session, format roughly analogous to claude's `~/.claude/projects`
/// jsonl transcripts). A follow-up pass will walk them and emit
/// PutProject/PutChat/PutMessage events shaped identically to the claude /
/// cursor importer output. For v1 we stub: log + report 100% so the
/// dispatcher's per-kind progress slice closes cleanly.
pub(crate) async fn import(
    _machine_id: Uuid,
    _user_id: Uuid,
    _write_tx: mpsc::Sender<WriteEvent>,
    progress: ImportProgress,
) -> Result<()> {
    info!("codex history import not yet implemented, skipping");
    progress(100);
    Ok(())
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
    fn turn_completed_emits_context_tokens_result_frame_and_result_marker() {
        let mut a = CodexAdapter::new();
        let line = r#"{"type":"turn.completed","usage":{"input_tokens":1000,"cached_input_tokens":500,"output_tokens":200,"reasoning_output_tokens":50}}"#;
        let events = run(&mut a, line);
        assert_eq!(events.len(), 3);
        assert_eq!(events[0], "ContextTokens(1000)");
        assert!(events[1].starts_with("Frame("));
        assert_eq!(events[2], "Result");
        // Verify the result envelope shape.
        let v = frame_value(&events[1]);
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
    fn repeated_turn_completed_usage_dedups_context_tokens() {
        let mut a = CodexAdapter::new();
        let line = r#"{"type":"turn.completed","usage":{"input_tokens":1000,"cached_input_tokens":500,"output_tokens":200,"reasoning_output_tokens":50}}"#;
        let first = run(&mut a, line);
        assert_eq!(first.len(), 3);
        assert_eq!(first[0], "ContextTokens(1000)");
        // Same usage → no ContextTokens on the second emission, just Frame + Result.
        let second = run(&mut a, line);
        assert_eq!(second.len(), 2);
        assert!(second[0].starts_with("Frame("));
        assert_eq!(second[1], "Result");
    }

    #[test]
    fn turn_started_and_item_started_dropped() {
        let mut a = CodexAdapter::new();
        let ts = r#"{"type":"turn.started"}"#;
        let is_ = r#"{"type":"item.started","item":{"id":"item_4","type":"agent_message"}}"#;
        assert!(run(&mut a, ts).is_empty());
        assert!(run(&mut a, is_).is_empty());
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
}
