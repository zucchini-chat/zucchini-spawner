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
//!    a future pass may surface in-flight tool
//!    status the way cursor's tool_call.started
//!    is currently dropped)
//!  - `item.completed` agent_message       → Frame: claude-shape assistant text envelope
//!  - `item.completed` command_execution   → Frame: claude `Bash` tool_use `{command}`
//!  - `item.completed` file_change         → one Frame per change:
//!    kind=add    → claude `Write` tool_use `{file_path}`
//!    kind=update → claude `Edit`  tool_use `{file_path}`
//!    kind=delete → claude `Bash`  tool_use `{command: "rm <path>"}`
//!    (no dedicated claude delete tool; `rm` is what claude
//!    code itself emits for the same intent)
//!  - `item.completed` web_search          → Frame: claude `WebSearch` tool_use `{query}`
//!  - `turn.completed`           → ContextTokens(input_tokens)
//!    + Frame (claude-shape result envelope)
//!    + Result
//!  - `turn.failed`              → Frame (claude-shape error result envelope)
//!    + Result
//!  - anything else              → forwarded as-is (defensive against codex
//!    format drift; iOS will likely drop, but
//!    we avoid silently losing the line)
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
//! `import()` walks Codex's on-disk rollouts at
//! `~/.codex/sessions/YYYY/MM/DD/rollout-<ISO8601>-<uuid>.jsonl` (one file per
//! session) and emits PutProject/PutChat/PutMessage like the claude / cursor
//! importers. See the importer section at the bottom of this file for the
//! line→frame mapping and the persisted-tool-name map.

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

use crate::adapters::import_shared::{
    basename_or, collapse_title, emit_chat, mint_project_id, parse_rfc3339_utc, user_message_body,
    ImportedChat, ImportedMessage, ProgressThrottle,
};
#[cfg(test)]
use crate::envelope::MessageEnvelope;
use anyhow::Context;
use chrono::{DateTime, Utc};
use std::collections::BTreeMap;
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
fn collect_rollouts(dir: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
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
}
