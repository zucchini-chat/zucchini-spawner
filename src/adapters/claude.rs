//! Claude-code adapter. **Iso-claude guarantee**: the bytes written to
//! `messages.body` here must be byte-identical to the pre-refactor spawner
//! output. The command builder, skip filter, session-id harvest, and
//! per-frame usage parsing are direct lifts from the pre-refactor `agent.rs`.
//! When in doubt: do not edit; move only.
//!
//! Also hosts the install/auth `probe()` for claude (free function, not on
//! the `AgentAdapter` trait — `dyn AgentAdapter` can't dispatch statics).
//! `main.rs::probe_install` calls into it from the startup-info report.

use std::collections::HashSet;
use std::path::PathBuf;

use anyhow::Result;
use serde::Deserialize;
use smallvec::SmallVec;
use tracing::debug;

use crate::adapter::{
    agent_capabilities_instructions, current_time_in_tz_line, file_nonempty,
    probe_with_blocking_auth, shell_escape, AdapterDescriptor, AgentAdapter, AgentEvent, AgentKind,
    LastTokensDedup, TurnContext, WorktreeInstructions, MAX_STREAM_FRAME_BYTES,
    PRUNE_CONTEXT_INSTRUCTION,
};

/// Wired into `adapter::ADAPTERS`. See `adapter::AdapterDescriptor` for the
/// shape; the `probe` / `import` slots are filled by `_boxed` wrappers below
/// the `probe()` / `import()` definitions in this file.
pub const DESCRIPTOR: AdapterDescriptor = AdapterDescriptor {
    kind: AgentKind::Claude,
    wire_name: "claude",
    // Pre-multi-agent schema keeps the long `claude_code_*` column names;
    // cursor used the short form from migration 0033.
    installed_col: "claude_code_installed",
    authenticated_col: "claude_code_authenticated",
    make: make_boxed,
    probe: probe_boxed,
    import: import_boxed,
    prune: Some(PRUNE_OPS),
};

fn make_boxed() -> Box<dyn AgentAdapter> {
    Box::new(ClaudeAdapter::new())
}

#[derive(Default)]
pub struct ClaudeAdapter {
    /// Per-turn dedup so repeated usage frames (e.g. thinking frame followed
    /// by the text frame after it carries the same usage) don't fire a
    /// redundant PATCH on each one. State lives for one turn. See
    /// `adapter::LastTokensDedup`.
    last_emitted_tokens: LastTokensDedup,
    /// `tool_use` ids of in-flight `prune-context` calls seen on assistant
    /// frames this turn. The `tool_result` cue that drives a queued prune's
    /// apply fires ONLY when the result's own `tool_use_id` is in this set —
    /// call-keyed, not chat-keyed. Without this, a sibling tool's result in
    /// the same parallel batch would fire abort→respawn before the
    /// `prune-context` call's own result persists, so the resumed agent never
    /// sees its prune + summary and re-runs it. See `AgentEvent::ToolResult`.
    pending_prune_tool_use_ids: HashSet<String>,
}

impl ClaudeAdapter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Scan an assistant frame for `prune-context` `tool_use` blocks and record
    /// their ids so the matching `tool_result` (and only it) can later drive the
    /// queued prune's apply. Caller gates on `line.contains("prune-context")`,
    /// so the parse only runs on frames that actually mention the subcommand.
    fn record_prune_tool_use_ids(&mut self, line: &str) {
        let Ok(entry) = serde_json::from_str::<serde_json::Value>(line) else {
            return;
        };
        let Some(blocks) = entry
            .get("message")
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_array())
        else {
            return;
        };
        for block in blocks {
            if block.get("type").and_then(|t| t.as_str()) != Some("tool_use") {
                continue;
            }
            let Some(input) = block.get("input") else {
                continue;
            };
            if crate::prune::value_is_prune_context_call(input) {
                if let Some(id) = block.get("id").and_then(|i| i.as_str()) {
                    self.pending_prune_tool_use_ids.insert(id.to_string());
                }
            }
        }
    }
}

/// Extract a `tool_result` frame's `tool_use_id` WITHOUT a full parse of the
/// (possibly multi-MB) result body — the id is a short token near the head of
/// the content block. The structural field is unescaped (`"tool_use_id":"`);
/// any occurrence inside the result's own text is JSON-escaped (`\"…\"`), so
/// this substring only matches the real field — same reasoning the frame-skip
/// filters above rely on.
fn extract_tool_result_id(line: &str) -> Option<&str> {
    const KEY: &str = "\"tool_use_id\":\"";
    let start = line.find(KEY)? + KEY.len();
    let rest = &line[start..];
    let end = rest.find('"')?;
    Some(&rest[..end])
}

/// Append every claude flag AFTER the base invocation (`cat … | claude` for
/// one-shot, bare `claude` for resident) to `cmd`. Shared by `prepare_command`
/// and `prepare_session_command` so the flag set (session-id resume, worktree
/// containment, `--append-system-prompt`, `--print --verbose
/// --output-format stream-json`, sandbox, model, the prune `--settings` hook)
/// can't drift between the two paths. `sys_leadin` is the first sentence of the
/// `--append-system-prompt` block (the only intentional difference); `resident`
/// adds `--input-format stream-json` before `--output-format` so claude reads
/// turns from the held-open stdin.
#[allow(clippy::too_many_arguments)]
fn append_claude_flags(
    cmd: &mut String,
    sys_leadin: &str,
    chat_id: &str,
    project_path: Option<&str>,
    worktree: bool,
    agent_session_id: Option<&str>,
    is_sandboxed: bool,
    model: Option<&str>,
    user_timezone: Option<&str>,
    resident: bool,
) {
    // First turn / fresh session: pass no session flag — claude generates a
    // session id and emits it in the `system/init` stdout frame, which we
    // harvest and persist to `chats.agent_session_id`. Subsequent (re)starts
    // resume that id. Pre-migration rows were backfilled with
    // `agent_session_id = id::text`, so existing chats keep using the same id
    // claude already knows about.
    if let Some(sid) = agent_session_id {
        cmd.push_str(&format!(" --resume {}", shell_escape(sid)));
    }

    let mut sys = String::from(sys_leadin);
    // Worktree containment: when `--worktree` is on and the project path is
    // known, resolve the absolute worktree path so the capability block can
    // carry the "stay inside" rule (wording in `adapter::worktree_instructions`).
    let mut worktree_info: Option<WorktreeInstructions> = None;
    if worktree {
        // Use the chat_id prefix so the worktree directory name stays short.
        let worktree_name: String = chat_id.chars().take(8).collect();
        cmd.push_str(&format!(" --worktree {}", shell_escape(&worktree_name)));
        if let Some(pp) = project_path {
            let worktree_abs = format!(
                "{}/.claude/worktrees/{}",
                pp.trim_end_matches('/'),
                worktree_name
            );
            worktree_info = Some(WorktreeInstructions {
                worktree_abs,
                parent_repo: pp.to_string(),
            });
        }
    }
    sys.push_str("\n\n");
    // Fold the per-turn time line into this fresh-every-turn block (hence the
    // `prompt_file_time_line` override returns `None`); prune nudge appended after.
    sys.push_str(&agent_capabilities_instructions(
        worktree_info.as_ref(),
        current_time_in_tz_line(user_timezone).as_deref(),
    ));
    sys.push_str("\n\n");
    sys.push_str(PRUNE_CONTEXT_INSTRUCTION);
    cmd.push_str(&format!(" --append-system-prompt {}", shell_escape(&sys)));
    // Resident: read user/control turns from the held-open stdin pipe.
    if resident {
        cmd.push_str(" --input-format stream-json");
    }
    cmd.push_str(
        " --print --verbose --output-format stream-json --disallowedTools AskUserQuestion",
    );
    // Sender's `machine_users.is_sandboxed`. Non-sandboxed = bypass permission
    // gating; sandboxed = claude's default permission mode auto-denies tools
    // in `--print`, which is the actual sandboxing mechanism.
    if !is_sandboxed {
        cmd.push_str(" --dangerously-skip-permissions");
    }
    // Verbatim pass-through of `chats.model` (migration 0035). Empty / blank
    // values are already filtered to `None` at the construction site in
    // `main.rs`, so any `Some` here is a non-empty model name the user picked in
    // the composer's agent roster. We don't validate the model name — claude
    // prints a clean error if it doesn't recognize it, and the closed set drifts
    // per-release.
    if let Some(model) = model {
        cmd.push_str(&format!(" --model {}", shell_escape(model)));
    }
    // Register a `PostToolUse` hook that nudges the agent to prune large tool
    // outputs once they're no longer needed (selective forgetting). The hook
    // command re-invokes THIS binary (`prune-reminder-hook`), so there's no
    // bash+jq script and no external dep — the spawner parses the hook JSON and
    // applies the size gate in Rust. `$ZUCCHINI_SPAWNER_BIN` is exported on the
    // spawn (agent.rs); shell_escape wraps the settings JSON in single quotes, so
    // the var is passed through LITERALLY and claude's hook runner expands it at
    // hook time against the inherited env (NOT the outer login shell). Added
    // unconditionally (harmless when nothing's large), same as the prune-context
    // system-prompt nudge. `--settings` MERGES with (never replaces) the user's
    // settings.json / project hooks, so user-configured hooks survive the spawn
    // (verified empirically). See `zucchini-spawner/CLAUDE.md`.
    let settings = serde_json::json!({
        "hooks": {
            "PostToolUse": [{
                // Match-all. claude treats the matcher as a REGEX, so the
                // documented match-all is the empty string (`""`), NOT `"*"`
                // (a bare `*` is not a valid standalone quantifier).
                "matcher": "",
                "hooks": [{
                    "type": "command",
                    "command": "\"$ZUCCHINI_SPAWNER_BIN\" prune-reminder-hook"
                }]
            }]
        }
    });
    cmd.push_str(&format!(
        " --settings {}",
        shell_escape(&settings.to_string())
    ));
}

impl AgentAdapter for ClaudeAdapter {
    fn kind(&self) -> AgentKind {
        AgentKind::Claude
    }

    fn prepare_command(&mut self, ctx: &TurnContext<'_>) -> Result<String> {
        let mut claude_cmd = String::new();
        if let Some(pp) = ctx.project_path {
            claude_cmd.push_str(&format!("cd {} && ", shell_escape(pp)));
        }
        // One-shot: pipe the prompt file into `claude`.
        claude_cmd.push_str(&format!(
            "cat {} | claude",
            shell_escape(&ctx.prompt_file.to_string_lossy())
        ));
        // One-shot `--print` is print-and-exit: the process dies at its first
        // `result`, so any background subagent it armed is abandoned. Steer the
        // agent away from `run_in_background:true`. (The resident session uses a
        // DIFFERENT lead-in — see `RESIDENT_SYS_LEADIN` — because background
        // tasks work there.)
        const ONESHOT_SYS_LEADIN: &str =
            "You are spawned via a harness, no background subagents will wake you when finished, use subagents with `run_in_background: false` only.";
        append_claude_flags(
            &mut claude_cmd,
            ONESHOT_SYS_LEADIN,
            ctx.chat_id,
            ctx.project_path,
            ctx.worktree,
            ctx.agent_session_id,
            ctx.is_sandboxed,
            ctx.model,
            ctx.user_timezone,
            /* resident = */ false,
        );
        Ok(claude_cmd)
    }

    /// The current-local-time line is folded into the per-turn system prompt
    /// (`agent_capabilities_instructions` in `append_claude_flags`), so suppress
    /// the prompt-file prepend to avoid injecting it twice.
    fn prompt_file_time_line(&self, _ctx: &TurnContext<'_>) -> Option<String> {
        None
    }

    fn is_resident(&self) -> bool {
        true
    }

    fn prepare_session_command(
        &mut self,
        ctx: &crate::adapter::SessionContext<'_>,
    ) -> Result<String> {
        let mut claude_cmd = String::new();
        if let Some(pp) = ctx.project_path {
            claude_cmd.push_str(&format!("cd {} && ", shell_escape(pp)));
        }
        // Resident: NO prompt piping — turns arrive as `encode_user_turn` stdin
        // frames over the held-open pipe. Just spawn bare `claude`; the
        // `--input-format stream-json` flag (added by `append_claude_flags` with
        // `resident=true`) tells it to read user/control frames from stdin.
        claude_cmd.push_str("claude");
        // The whole point of the resident model is that background tasks/monitors
        // run to completion and stream their results back, so DROP the one-shot
        // "no background subagents" lead-in.
        const RESIDENT_SYS_LEADIN: &str =
            "You are running in a resident session; background tasks and monitors you arm will run to completion and stream their results back.";
        append_claude_flags(
            &mut claude_cmd,
            RESIDENT_SYS_LEADIN,
            ctx.chat_id,
            ctx.project_path,
            ctx.worktree,
            ctx.agent_session_id,
            ctx.is_sandboxed,
            ctx.model,
            ctx.user_timezone,
            /* resident = */ true,
        );
        Ok(claude_cmd)
    }

    fn encode_user_turn(&self, text: &str) -> String {
        // Build via serde so the user's text (quotes / newlines / control chars)
        // is encoded safely — never string concat. Newline-terminated: claude's
        // stream-json input is newline-delimited frames.
        let frame = serde_json::json!({
            "type": "user",
            "message": {
                "role": "user",
                "content": [{ "type": "text", "text": text }],
            },
        });
        format!("{}\n", frame)
    }

    fn handle_line(&mut self, line: String) -> SmallVec<[AgentEvent; 2]> {
        let mut out: SmallVec<[AgentEvent; 2]> = SmallVec::new();

        // [DIAG tasktrace] TEMPORARY — capture every frame that touches a
        // background task/monitor so we can see the exact terminal frame the
        // LIVE resident claude emits (Monitor tool vs Bash run_in_background,
        // task_started/updated/notification subtype + the id field actually
        // present). Remove once the stuck-Waiting bug is pinned.
        if line.starts_with('{')
            && (line.contains("task_id")
                || line.contains("task-notification")
                || line.contains("\"subtype\":\"task_"))
        {
            let snip: String = line.chars().take(900).collect();
            tracing::info!(diag = "tasktrace", "raw task frame: {snip}");
        }

        // Pre-skip-filter: thinking-only frames also carry usage. The
        // token-usage parse stays under MAX_STREAM_FRAME_BYTES — it runs on
        // EVERY assistant frame, so it must skip pathologically large bodies.
        if line.len() < MAX_STREAM_FRAME_BYTES
            && line.starts_with('{')
            && line.contains("\"type\":\"assistant\"")
        {
            if let Some(tokens) = parse_assistant_usage(&line) {
                if let Some(t) = self.last_emitted_tokens.observe(tokens) {
                    out.push(AgentEvent::ContextTokens(t));
                }
            }
        }

        // Record the `tool_use` id of any `prune-context` call so its own
        // `tool_result` (and only its own) can later drive the queued prune's
        // apply. The `prune-context` substring is the cheap pre-filter that
        // keeps ordinary assistant frames from paying the parse.
        //
        // Deliberately NOT under the MAX_STREAM_FRAME_BYTES gate (unlike the
        // usage parse above): record and the matching `tool_result` EMIT below
        // share one policy. A batched multi-target prune call (many
        // --tool-name/--args/--summary triples with large needles/summaries)
        // can push the assistant frame past 64KiB; if we skipped the record
        // there, the cue would silently never fire and the queued prune would
        // be a no-op (leaking in `pending_prunes` until the turn's Done clears
        // it). The substring + starts_with('{') + assistant-type check keep
        // this scoped to assistant-shaped prune-bearing frames (a user/
        // tool_result frame can also echo "prune-context" in its result text,
        // but record_prune_tool_use_ids only inspects `tool_use` blocks, so
        // restricting to assistant frames just avoids a needless parse).
        if line.starts_with('{')
            && line.contains("\"type\":\"assistant\"")
            && line.contains("prune-context")
        {
            self.record_prune_tool_use_ids(&line);
        }

        // Pre-skip harvest: `system/init` frames carry claude's
        // self-generated session id (we no longer pass --session-id
        // on the first turn). Emit it BEFORE the skip decision so
        // the init frame itself still gets filtered out below and
        // never reaches the chat. Init frames are tiny — no need
        // for the MAX_STREAM_FRAME_BYTES guard.
        if line.starts_with('{')
            && line.contains("\"type\":\"system\"")
            && line.contains("\"subtype\":\"init\"")
        {
            if let Some(session_id) = parse_init_session_id(&line) {
                out.push(AgentEvent::SessionIdHarvested(session_id));
            }
        }

        // Skip frames the UI never renders so they don't flicker
        // the chat-list preview to empty between visible messages.
        // Substring match: tool_result bodies can be multi-MB, so
        // we avoid a full JSON parse on every line. Filtered:
        //   stream_event                — per-token deltas, high-freq
        //   system (non-status)         — init/shutdown frames
        //   user, rate_limit_event      — never shown
        //   assistant with only thinking — SpawnerMessageDescriber
        //     returns nil for these, so they contribute nothing to
        //     ChatView and would otherwise clobber `last_agent_body`
        //     with unreadable content (chat list falls back to
        //     "Thinking…" even though prior text/tool_use exists).
        //     JSON-string-escaping renders \"type\":\"text\" as
        //     `\"type\":\"text\"` in the source, so the substring
        //     below only matches structural block types.
        let mut skip = false;
        if line.starts_with('{') {
            if line.contains("\"type\":\"system\"") && !line.contains("\"subtype\":\"status\"") {
                // System frames are skipped, but compact_boundary
                // carries postTokens we need — harvest it here
                // instead of dropping the line silently.
                if line.contains("\"subtype\":\"compact_boundary\"") {
                    if let Some(post_tokens) = parse_compact_post_tokens(&line) {
                        out.push(AgentEvent::CompactBoundary(post_tokens));
                    }
                } else if line.contains("\"subtype\":\"task_started\"") {
                    // Resident model: a background task / Monitor was armed. The
                    // `task_id` lets the Supervisor track it in `live_tasks`
                    // (Waiting state) until its terminal `task_notification`.
                    if let Some(id) = parse_task_id(&line) {
                        out.push(AgentEvent::TaskStarted(id));
                    }
                } else if line.contains("\"subtype\":\"task_notification\"")
                    || (line.contains("\"subtype\":\"task_updated\"")
                        && task_updated_is_terminal(&line))
                {
                    // Terminal task signal (completed/failed/cancelled) — clears
                    // the task from `live_tasks`. Two frame shapes carry it:
                    // claude's one-shot `--print` mode emits a dedicated
                    // `task_notification`, but the resident `stream-json` session
                    // signals completion via `task_updated` with `patch.status` =
                    // completed/failed/cancelled and never sends a
                    // `task_notification` (verified live: chat c06c30b2, task
                    // bo87ftywa — only a task_started then a task_updated{completed}
                    // ever crossed the wire). Match BOTH, or the resident session's
                    // `live_tasks` never empties → `running` sticks true → no
                    // `chat_running(false)` is ever written and the chat is pinned
                    // to agent_running / "Waiting…" forever after a Monitor ends. A
                    // `task_updated` PROGRESS frame (status running/pending) is not
                    // terminal and still falls through to skip.
                    if let Some(id) = parse_task_id(&line) {
                        out.push(AgentEvent::TaskFinished(id));
                    }
                }
                skip = true;
            } else if line.contains("\"type\":\"user\"") {
                // User frames are never rendered — but a `tool_result` user frame
                // is claude's signal that it has PERSISTED a finished tool call's
                // result. Emit a content-free `ToolResult` cue (still skipping the
                // frame) so the main loop can apply a queued prune the instant the
                // `prune-context` call's own result lands on disk — strictly after
                // it persists, so the resumed agent sees its prune + summary.
                // Substring match (not a parse): a real `tool_result` block type
                // has unescaped quotes; any occurrence inside the result's own
                // text is escaped (`\"`), so this only matches the structural block.
                //
                // Call-keyed: fire ONLY when this result's own `tool_use_id`
                // matches a recorded `prune-context` call. A sibling tool's
                // result in the same parallel batch must NOT preempt the apply
                // (it would abort→respawn before the prune's own result
                // persists). The set is empty for every ordinary turn, so the
                // id extraction is skipped entirely unless a prune is in flight.
                if line.contains("\"type\":\"tool_result\"")
                    && !self.pending_prune_tool_use_ids.is_empty()
                {
                    if let Some(id) = extract_tool_result_id(&line) {
                        if self.pending_prune_tool_use_ids.remove(id) {
                            out.push(AgentEvent::ToolResult);
                        }
                    }
                }
                skip = true;
            } else if line.contains("\"type\":\"stream_event\"")
                || line.contains("\"type\":\"rate_limit_event\"")
                || (line.contains("\"type\":\"assistant\"")
                    && !line.contains("\"type\":\"text\"")
                    && !line.contains("\"type\":\"tool_use\""))
            {
                skip = true;
            } else if line.contains("\"type\":\"result\"") {
                // Emit Result on every result frame; the supervisor latches it
                // (so AgentResponse::Done.has_result is set once and only once).
                // `origin = {"kind":"task-notification"}` marks a background-task
                // wake's result (vs an absent `origin` on a user-turn result) so
                // the resident session FSM doesn't clear `turn_in_flight` for it.
                // Parse-free: both needles are structural (any occurrence inside
                // the JSON-escaped result text would be `\"`-escaped).
                let origin_is_task =
                    line.contains("\"origin\"") && line.contains("\"kind\":\"task-notification\"");
                out.push(AgentEvent::Result { origin_is_task });
                // Resident interrupt suppression. Now mainly covers the /stop
                // SIGTERM race (and interrupt-then-send, also a SIGTERM): we
                // publish our own canonical `{"type":"result","subtype":
                // "interrupted"}` row, and claude may flush a real abort result
                // for the killed turn before the process dies.
                // claude ALSO emits a real abort result for the killed turn —
                // empirically (real claude 2.1.165, tmp/claude_proto_probe.py):
                //   {"subtype":"error_during_execution",...,
                //    "terminal_reason":"aborted_streaming"}
                // Forwarding that as a Frame too would render a SECOND terminal
                // row ("[result: error_during_execution]" right after "Agent
                // interrupted"). Suppress the row (keep the Result above so the
                // FSM still clears `turn_in_flight`). Gated on BOTH needles so we
                // fail OPEN — a genuine execution error (different subtype or
                // terminal_reason) still renders.
                if line.contains("\"subtype\":\"error_during_execution\"")
                    && line.contains("\"terminal_reason\":\"aborted_streaming\"")
                {
                    skip = true;
                }
            }
        }

        if !skip {
            // Move the owned `String` into the Frame; no per-frame heap
            // allocation on the hot path. The Supervisor's line loop in
            // `agent.rs` reads owned `String`s from `BufReader::lines()`,
            // so this is just a move.
            out.push(AgentEvent::Frame(line));
        }

        out
    }
}

/// Cumulative for the current turn — caller overwrites `chats.context_tokens`
/// with each emission. Uses a narrow Deserialize struct so serde skips the rest
/// of the frame (text blocks, tool calls) without allocating it.
///
/// `message.usage` is `Option<_>` so a claude format drift that ships an
/// interim assistant frame without `usage` (or renames the field) doesn't
/// poison the whole-frame parse — we just return `None` and skip the
/// ContextTokens emission for that line. The next usage-carrying frame
/// resumes the cumulative counter. Without this, a single missing-usage
/// frame would silently freeze `chats.context_tokens` mid-turn.
///
/// Subagent (Task) frames carry the parent's tool_use id in top-level
/// `parent_tool_use_id`; their `usage` describes the subagent's own context,
/// not the main chat's, so we skip them — otherwise `chats.context_tokens`
/// would jitter to the subagent's count and back on every interleaved frame.
fn parse_assistant_usage(line: &str) -> Option<i64> {
    #[derive(serde::Deserialize)]
    struct Frame {
        message: Message,
        #[serde(default)]
        parent_tool_use_id: Option<String>,
    }
    #[derive(serde::Deserialize)]
    struct Message {
        #[serde(default)]
        usage: Option<Usage>,
    }
    #[derive(serde::Deserialize, Default)]
    struct Usage {
        #[serde(default)]
        input_tokens: i64,
        #[serde(default)]
        cache_creation_input_tokens: i64,
        #[serde(default)]
        cache_read_input_tokens: i64,
    }
    match serde_json::from_str::<Frame>(line) {
        Ok(f) => {
            if f.parent_tool_use_id.is_some() {
                return None;
            }
            let u = f.message.usage?;
            Some(u.input_tokens + u.cache_creation_input_tokens + u.cache_read_input_tokens)
        }
        Err(e) => {
            debug!("failed to parse assistant frame for usage: {}", e);
            None
        }
    }
}

/// Reads `session_id` from a `system/init` stream-json frame. Narrow Deserialize
/// struct so serde skips the rest of the (init payload is small but heterogeneous —
/// `cwd`, `model`, `tools`, etc.) without allocating it.
fn parse_init_session_id(line: &str) -> Option<String> {
    #[derive(serde::Deserialize)]
    struct Frame {
        session_id: String,
    }
    match serde_json::from_str::<Frame>(line) {
        Ok(f) => Some(f.session_id),
        Err(e) => {
            debug!("failed to parse init frame for session_id: {}", e);
            None
        }
    }
}

/// Reads `task_id` from a `task_started` / `task_notification` system frame.
/// Both carry it at the top level. Narrow Deserialize struct so serde skips the
/// rest of the (small) frame. `None` on parse failure or a missing id — the
/// Supervisor treats a dropped start/finish as "task never tracked", which is
/// safe (a never-inserted task can't wedge the Waiting state).
fn parse_task_id(line: &str) -> Option<String> {
    #[derive(serde::Deserialize)]
    struct Frame {
        task_id: String,
    }
    match serde_json::from_str::<Frame>(line) {
        Ok(f) => Some(f.task_id),
        Err(e) => {
            debug!("failed to parse task frame for task_id: {}", e);
            None
        }
    }
}

/// True when a `task_updated` system frame reports a TERMINAL `patch.status`
/// (completed / failed / cancelled) rather than progress (running / pending).
/// The resident `stream-json` session uses this frame — not `task_notification`
/// — to signal a background task / Monitor has finished, so it's the only edge
/// that empties `live_tasks` for a live session (see the caller). Parses the
/// nested `patch.status`; an unparsable frame or a non-terminal/absent status is
/// treated as non-terminal, leaving the frame to skip (a missed completion would
/// wedge Waiting, but a never-tracked task can't, so failing non-terminal here is
/// only safe because `task_started` is parsed by the same strict path).
fn task_updated_is_terminal(line: &str) -> bool {
    #[derive(serde::Deserialize)]
    struct Frame {
        patch: Patch,
    }
    #[derive(serde::Deserialize)]
    struct Patch {
        #[serde(default)]
        status: Option<String>,
    }
    match serde_json::from_str::<Frame>(line) {
        Ok(f) => matches!(
            f.patch.status.as_deref(),
            Some("completed") | Some("failed") | Some("cancelled")
        ),
        Err(_) => false,
    }
}

/// Reads `compact_metadata.post_tokens` from a `compact_boundary` system frame.
/// Narrow Deserialize struct so serde skips the rest of the frame without allocating it.
///
/// The public `--output-format stream-json` wire frame is snake_case
/// (`compact_metadata` / `post_tokens`) — verified live against claude 2.1.170.
/// The on-disk transcript jsonl and older builds use camelCase
/// (`compactMetadata` / `postTokens`); the `alias` accepts both so a future
/// format flip doesn't silently break the harvest again. (The earlier struct
/// only knew the camelCase transcript shape, so every real stdout frame failed
/// to parse with "missing field 'compactMetadata'".)
fn parse_compact_post_tokens(line: &str) -> Option<i64> {
    #[derive(serde::Deserialize)]
    struct Frame {
        #[serde(rename = "compact_metadata", alias = "compactMetadata")]
        metadata: Metadata,
    }
    #[derive(serde::Deserialize)]
    struct Metadata {
        #[serde(rename = "post_tokens", alias = "postTokens")]
        post_tokens: i64,
    }
    match serde_json::from_str::<Frame>(line) {
        Ok(f) => Some(f.metadata.post_tokens),
        Err(e) => {
            debug!("failed to parse compact_boundary frame: {}", e);
            None
        }
    }
}

/// Probe install + auth state in one go. Returns `(installed, authenticated)`
/// — the writer flattens one pair per registered kind into a single PATCH on
/// `machines`. `is_authenticated` is sync because it only touches the
/// filesystem; the shared
/// `adapter::probe_with_blocking_auth` helper wraps it in `spawn_blocking`
/// so a slow filesystem read doesn't block the runtime thread.
pub async fn probe() -> (bool, bool) {
    probe_with_blocking_auth("claude", is_authenticated).await
}

/// `fn`-pointer-shaped wrapper around `probe()` for `AdapterDescriptor.probe`.
/// `BoxFuture` erases the concrete async-fn type so the descriptor can hold
/// all adapters' probes in a single slice.
fn probe_boxed() -> futures::future::BoxFuture<'static, (bool, bool)> {
    Box::pin(probe())
}

/// claude code stores OAuth state in `~/.claude.json` under `oauthAccount` —
/// cross-platform, regardless of where the actual token lives (macOS Keychain,
/// `~/.claude/.credentials.json` on Linux). Fallback covers older / non-Keychain
/// installs.
fn is_authenticated() -> bool {
    let Some(home) = std::env::var_os("HOME").map(PathBuf::from) else {
        return false;
    };

    #[derive(Deserialize)]
    struct ClaudeCfg {
        #[serde(default, rename = "oauthAccount")]
        oauth_account: Option<serde_json::Value>,
    }

    let cfg = home.join(".claude.json");
    if let Ok(bytes) = std::fs::read(&cfg) {
        match serde_json::from_slice::<ClaudeCfg>(&bytes) {
            Ok(c) if c.oauth_account.is_some() => return true,
            Ok(_) => {}
            Err(e) => debug!(error = %e, path = %cfg.display(), "claude config not parseable"),
        }
    }

    let creds = home.join(".claude").join(".credentials.json");
    file_nonempty(&creds)
}

// ===========================================================================
// One-shot claude-history importer. Walks `~/.claude/projects/*/*.jsonl`,
// dedups via `(machine_id, project_path)`-derived project ids + filename-
// derived chat ids, and emits PutProject/PutChat/PutMessage events. Status
// emission lives in the dispatcher in `main.rs` (the dispatcher rescales
// per-kind progress into a single 0..99 bar and emits `finished` once at the
// very end), so this function only reports raw 0..=100 progress via the
// `progress` callback and never sends `WriteEvent::ImportStatus`.
//
// Idempotent: project ids are UUIDv5(machine_id || path), chat ids are the
// sessionId from the filename, so re-runs reconverge on the same rows.
//
// Frame filter mirrors the substring filter in `agent.rs` (the stdout reader
// in `Supervisor::spawn`) — keep `user` with string content (wrap in
// MessageEnvelope) and `assistant` with text/tool_use blocks (re-emit as
// stream-json so SpawnerMessageDescriber reads it). Skip tool_result echoes,
// thinking-only frames, sidechain (subagent) transcripts, and
// `queue-operation`/`last-prompt`/`attachment`. `ai-title` is harvested into
// chats.title.
//
// User strings also get a synthetic-wrapper screen (see `is_synthetic_wrapper`):
// the TUI logs `/exit`, `/clear`, `/compact`, `<system-reminder>`, etc. as
// pseudo-user rows that the user never typed and the model never sees.
// Custom slash commands (`<command-message>`-prefixed) are kept — those go
// to the model and produce real assistant replies.

use std::collections::BTreeMap;
use std::io::ErrorKind;
use std::path::Path;

use anyhow::Context;
use chrono::{DateTime, Utc};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::mpsc;
use tracing::{info, warn};
use uuid::Uuid;

use crate::adapter::ImportProgress;
use crate::adapters::import_shared::{
    basename_or, collapse_title, emit_chat, is_synthetic_wrapper, mint_project_id,
    parse_rfc3339_utc, user_message_body, ImportedChat, ImportedMessage, ProgressThrottle,
};
use crate::writer::WriteEvent;

/// One-shot. Triggered exactly once, immediately after a machine is added —
/// the iOS app blocks the UI on the import-progress sheet, so no live agent
/// can be spawned while this runs and the writer's batch channel is ours
/// alone. That's why we don't bump `MAX_OPS_PER_BATCH` for the importer
/// path — contention isn't possible by construction.
pub(crate) async fn import(
    machine_id: Uuid,
    user_id: Uuid,
    write_tx: mpsc::Sender<WriteEvent>,
    progress: ImportProgress,
) -> anyhow::Result<()> {
    let projects_dir = claude_projects_dir().context("locate ~/.claude/projects")?;
    info!(path = %projects_dir.display(), "scanning claude-code transcripts");

    // `BTreeMap` keeps the order stable for logs.
    let mut sessions_by_path: BTreeMap<String, Vec<std::path::PathBuf>> = BTreeMap::new();
    let mut total_sessions: usize = 0;
    let dir = match std::fs::read_dir(&projects_dir) {
        Ok(d) => d,
        Err(e) if e.kind() == ErrorKind::NotFound => {
            info!(path = %projects_dir.display(), "no ~/.claude/projects, nothing to import");
            progress(100).await;
            return Ok(());
        }
        Err(e) => return Err(e).with_context(|| format!("read_dir {}", projects_dir.display())),
    };
    for entry in dir {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                warn!(error = %e, "skipping unreadable entry under ~/.claude/projects");
                continue;
            }
        };
        if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            continue;
        }
        let dir_name = match entry.file_name().into_string() {
            Ok(s) => s,
            Err(_) => continue,
        };
        let Some(decoded) = decode_project_dir(&dir_name) else {
            warn!(dir = %dir_name, "could not resolve project dir to a real path, skipping");
            continue;
        };
        // Worktree sessions are tied to a transient checkout the user usually
        // cleans up; they'd land under a project the user never created.
        // Cleaner to skip them than to roll them up under the parent repo
        // and re-import their messages on every future import.
        if decoded.contains("/.claude/worktrees/") {
            info!(dir = %dir_name, "skipping worktree session transcripts");
            continue;
        }
        let bucket = sessions_by_path.entry(decoded).or_default();
        let inner = match std::fs::read_dir(entry.path()) {
            Ok(it) => it,
            Err(e) => {
                warn!(error = %e, dir = %entry.path().display(), "read_dir failed");
                continue;
            }
        };
        for f in inner {
            let f = match f {
                Ok(x) => x,
                Err(_) => continue,
            };
            let p = f.path();
            if p.extension().and_then(|s| s.to_str()) == Some("jsonl") {
                bucket.push(p);
                total_sessions += 1;
            }
        }
    }

    if total_sessions == 0 {
        info!("no .jsonl transcripts found");
        progress(100).await;
        return Ok(());
    }
    info!(
        projects = sessions_by_path.len(),
        sessions = total_sessions,
        "starting import"
    );

    let mut done_sessions: usize = 0;
    let mut throttle = ProgressThrottle::new();
    for (path, mut sessions) in sessions_by_path {
        sessions.sort();
        let project_id = mint_project_id(machine_id, &path);
        let project_name = basename_or(&path, "project");
        let _ = write_tx
            .send(WriteEvent::PutProject {
                id: project_id,
                machine_id,
                name: project_name,
                path: path.clone(),
            })
            .await;

        for jsonl in sessions {
            if let Err(e) = import_session(&jsonl, project_id, user_id, &write_tx).await {
                warn!(file = %jsonl.display(), error = %e, "session import failed, skipping");
            }
            done_sessions += 1;
            // Per-percent throttle shared with every importer; see `ProgressThrottle`.
            throttle
                .step(done_sessions, total_sessions, &progress)
                .await;
        }
    }

    info!(sessions = done_sessions, "claude history import complete");
    Ok(())
}

fn claude_projects_dir() -> Option<std::path::PathBuf> {
    let home = std::env::var_os("HOME")?;
    Some(
        std::path::PathBuf::from(home)
            .join(".claude")
            .join("projects"),
    )
}

/// Reverse the dirname encoding claude-code uses for cwd (every `/` replaced
/// with `-`). Naive `replace('-', "/")` is lossy on segments that contain a
/// literal `-` (e.g. `/Users/me/projects/berkshire-hathaway` round-trips to
/// `/Users/me/projects/berkshire/hathaway`). We disambiguate by walking the
/// real filesystem: at each step pick the longest prefix of remaining
/// segments that resolves to an existing directory.
///
/// Returns `None` when no walk reaches the leaf — usually because the project
/// was deleted or moved after the transcript was written. Caller skips those.
fn decode_project_dir(name: &str) -> Option<String> {
    let parts: Vec<&str> = name.split('-').collect();
    // Encoded form starts with the leading '/', so the first split yields "".
    if parts.first() != Some(&"") {
        return None;
    }
    let segments = &parts[1..];
    if segments.is_empty() {
        return None;
    }
    let mut current = std::path::PathBuf::from("/");
    let mut i = 0;
    while i < segments.len() {
        let mut matched: Option<usize> = None;
        // Greedy longest-prefix match so `berkshire-hathaway` wins over `berkshire`.
        for j in (i + 1..=segments.len()).rev() {
            let candidate = segments[i..j].join("-");
            let probe = current.join(&candidate);
            if probe.is_dir() {
                current = probe;
                matched = Some(j);
                break;
            }
        }
        match matched {
            Some(j) => i = j,
            None => return None,
        }
    }
    current.into_os_string().into_string().ok()
}

async fn import_session(
    jsonl: &Path,
    project_id: Uuid,
    user_id: Uuid,
    write_tx: &mpsc::Sender<WriteEvent>,
) -> anyhow::Result<()> {
    let session_id_str = jsonl
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| anyhow::anyhow!("no file stem"))?;
    let chat_id = Uuid::parse_str(session_id_str)
        .with_context(|| format!("filename is not a UUID: {session_id_str}"))?;

    let file = tokio::fs::File::open(jsonl)
        .await
        .with_context(|| format!("open {}", jsonl.display()))?;
    let mut lines = BufReader::new(file).lines();

    let mut title: Option<String> = None;
    let mut first_ts: Option<DateTime<Utc>> = None;
    let mut keepers: Vec<(DateTime<Utc>, ImportedMsg)> = Vec::new();

    while let Some(line) = lines.next_line().await? {
        if line.is_empty() {
            continue;
        }
        let entry: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                warn!(error = %e, "skipping malformed jsonl line");
                continue;
            }
        };
        let entry_type = entry.get("type").and_then(|v| v.as_str()).unwrap_or("");

        if entry_type == "ai-title" {
            if let Some(s) = entry.get("aiTitle").and_then(|v| v.as_str()) {
                title = Some(s.to_string());
            }
            continue;
        }
        if matches!(entry_type, "queue-operation" | "last-prompt" | "attachment") {
            continue;
        }
        if entry.get("isSidechain").and_then(|v| v.as_bool()) == Some(true) {
            continue;
        }

        let ts = entry
            .get("timestamp")
            .and_then(|v| v.as_str())
            .and_then(parse_rfc3339_utc);

        // Claude code preserves the entry uuid across `--continue`/`--resume`
        // replays of the same conversation entry — we thread it into
        // `WriteEvent::PutMessage::id` so replays dedup.
        let entry_uuid = crate::parse_uuid_field(&entry, "uuid");

        let imported = match entry_type {
            "user" => match classify_user(&entry) {
                UserContent::Prompt(text) => {
                    if title.is_none() {
                        title = Some(collapse_title(&text));
                    }
                    Some(ImportedMsg::user(text, entry_uuid))
                }
                UserContent::ToolResult | UserContent::Empty => None,
            },
            "assistant" => match classify_assistant(&entry) {
                AssistantContent::Renderable(body) => Some(ImportedMsg::agent(body, entry_uuid)),
                AssistantContent::Skip => None,
            },
            _ => None,
        };

        if let (Some(ts), Some(msg)) = (ts, imported) {
            if first_ts.is_none() {
                first_ts = Some(ts);
            }
            keepers.push((ts, msg));
        }
    }

    if keepers.is_empty() {
        return Ok(());
    }

    let chat_created_at = first_ts.unwrap_or_else(Utc::now);
    let chat_title = title.unwrap_or_else(|| "Imported chat".to_string());

    // Each keeper carries its sort timestamp + entry uuid (threaded into the
    // message id so `--continue`/`--resume` replays dedup in place). The PutChat
    // + per-row PutMessage emit is shared via `emit_chat`.
    let messages: Vec<ImportedMessage> = keepers
        .into_iter()
        .map(|(ts, msg)| ImportedMessage {
            id: msg.uuid,
            sender: msg.sender,
            body: msg.body,
            created_at: ts,
        })
        .collect();

    emit_chat(
        write_tx,
        user_id,
        ImportedChat {
            id: chat_id,
            project_id,
            title: chat_title,
            created_at: chat_created_at,
            messages,
        },
    )
    .await;

    Ok(())
}

struct ImportedMsg {
    sender: &'static str,
    body: String,
    uuid: Option<Uuid>,
}

impl ImportedMsg {
    fn user(text: String, uuid: Option<Uuid>) -> Self {
        Self {
            sender: "user",
            body: user_message_body(&text),
            uuid,
        }
    }

    fn agent(stream_json_frame: String, uuid: Option<Uuid>) -> Self {
        Self {
            sender: "agent",
            body: stream_json_frame,
            uuid,
        }
    }
}

enum UserContent {
    Prompt(String),
    ToolResult,
    Empty,
}

fn classify_user(entry: &serde_json::Value) -> UserContent {
    // isMeta=true is claude-code's marker for system-injected user messages
    // (slash-command caveats, "Continue from where you left off." pads, etc.).
    if entry.get("isMeta").and_then(|v| v.as_bool()) == Some(true) {
        return UserContent::Empty;
    }
    let Some(content) = entry.get("message").and_then(|m| m.get("content")) else {
        return UserContent::Empty;
    };
    // The CLI entrypoint writes typed prompts as a bare string, but the
    // VS Code extension (`entrypoint:"claude-vscode"`) always wraps them as
    // `[{type:"text", text:"..."}]`. Normalize both shapes to a single string;
    // arrays without any text blocks are tool_result echoes and skipped.
    let text = if let Some(s) = content.as_str() {
        s.to_string()
    } else if let Some(blocks) = content.as_array() {
        let mut parts: Vec<&str> = Vec::new();
        for b in blocks {
            if b.get("type").and_then(|t| t.as_str()) == Some("text") {
                if let Some(t) = b.get("text").and_then(|t| t.as_str()) {
                    parts.push(t);
                }
            }
        }
        if parts.is_empty() {
            return UserContent::ToolResult;
        }
        parts.join("\n")
    } else {
        return UserContent::Empty;
    };
    if is_synthetic_wrapper(&text) {
        return UserContent::Empty;
    }
    UserContent::Prompt(text)
}

enum AssistantContent {
    /// Reshaped as the live stream-json frame so SpawnerMessageDescriber reads
    /// it the same way as live agent output.
    Renderable(String),
    Skip,
}

fn classify_assistant(entry: &serde_json::Value) -> AssistantContent {
    let Some(message) = entry.get("message") else {
        return AssistantContent::Skip;
    };
    let has_text_or_tool = message
        .get("content")
        .and_then(|c| c.as_array())
        .map(|blocks| {
            blocks.iter().any(|b| {
                matches!(
                    b.get("type").and_then(|t| t.as_str()),
                    Some("text") | Some("tool_use")
                )
            })
        })
        .unwrap_or(false);
    if !has_text_or_tool {
        return AssistantContent::Skip;
    }
    let frame = serde_json::json!({ "type": "assistant", "message": message });
    AssistantContent::Renderable(frame.to_string())
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

// ===========================================================================
// Selective forgetting ("prune-context") — claude dialect. Shared contract:
// `crate::prune`. Claude shape: a tool result lives in a `user` entry paired to
// the assistant's `tool_use` by `tool_use_id`; a prune blanks the `tool_use`
// `input` to `{}` and the paired `tool_result` `content` to `PRUNED_PLACEHOLDER`
// — preserving every line + id/threading field byte-for-byte so `--resume` keeps
// its prompt-cache prefix up to the edit. Wired in via `PRUNE_OPS` (→
// `AdapterDescriptor::prune`). Single-pass (assistant input + paired user
// content blank in one read pass), distinct from the shared two-pass driver
// gemini/codex use.

/// `crate::prune::PruneOps` for claude. Points the descriptor at this module's
/// dialect pruners.
pub(crate) const PRUNE_OPS: crate::prune::PruneOps = crate::prune::PruneOps {
    find_session: find_session_jsonl,
    count_matches,
    prune_batch: prune_batch_jsonl,
};

/// Locate `~/.claude/projects/*/<session_id>.jsonl` under `base` (the CLI home
/// resolved by `AgentKind::cli_home`). claude shards transcripts into one
/// per-cwd subdir, so we scan the immediate children of `projects/` for the
/// `<session_id>.jsonl` leaf. `None` when no transcript for that id exists yet.
fn find_session_jsonl(base: &Path, session_id: &str) -> Option<PathBuf> {
    let projects = base.join("projects");
    let file_name = format!("{session_id}.jsonl");
    for entry in std::fs::read_dir(&projects).ok()?.flatten() {
        if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            continue;
        }
        let candidate = entry.path().join(&file_name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// Does this `tool_use` block match the (`tool_name`, `needle`) selector? Block
/// must be a `tool_use`; an empty `--tool-name` matches any tool (prune on the
/// args needle alone), else the claude-shape `name` must match exactly. The
/// agent's own in-flight `prune-context` call is never a target (see
/// [`crate::prune::value_is_prune_context_call`]). Empty `needle` is the no-args
/// selector ([`crate::prune::args_value_is_empty`]); otherwise glob the input
/// VALUES ([`crate::prune::value_glob_match`]).
fn tool_use_matches(block: &serde_json::Value, tool_name: &str, needle: &str) -> bool {
    if block.get("type").and_then(|t| t.as_str()) != Some("tool_use") {
        return false;
    }
    if !tool_name.is_empty() && block.get("name").and_then(|n| n.as_str()) != Some(tool_name) {
        return false;
    }
    // Skip the agent's own in-flight prune-context call (its command line carries
    // the same needle, and last-only would otherwise pick it over the real target).
    if block
        .get("input")
        .is_some_and(crate::prune::value_is_prune_context_call)
    {
        return false;
    }
    if needle.is_empty() {
        return crate::prune::args_value_is_empty(block.get("input"));
    }
    block
        .get("input")
        .is_some_and(|i| crate::prune::value_glob_match(i, needle))
}

/// Eligible (matched, not already `[pruned]`) `tool_use` ids in document order.
/// The ordered-minus-pruned combinator is the shared
/// [`crate::prune::select_eligible_ids`]; here we supply the claude-shape
/// collectors ([`claude_collect_matched`] / [`claude_collect_pruned`]) — the SAME
/// two `prune_jsonl` feeds the single-read apply driver, so the count and apply
/// paths can't drift.
fn eligible_matches(path: &Path, tool_name: &str, needle: &str) -> std::io::Result<Vec<String>> {
    crate::prune::select_eligible_ids(
        path,
        |entry, matched| claude_collect_matched(entry, tool_name, needle, matched),
        claude_collect_pruned,
    )
}

/// Pass-1 matched collector shared by `eligible_matches` (the control-side count
/// path, via [`crate::prune::select_eligible_ids`]) and `prune_jsonl` (the apply
/// path, via [`crate::prune::rewrite_jsonl_last_only`]). Walks assistant `tool_use`
/// blocks via [`tool_use_matches`], pushing matching ids in DOCUMENT ORDER.
fn claude_collect_matched(
    entry: &serde_json::Value,
    tool_name: &str,
    needle: &str,
    matched: &mut Vec<String>,
) {
    if entry.get("type").and_then(|t| t.as_str()) != Some("assistant") {
        return;
    }
    let Some(blocks) = entry
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_array())
    else {
        return;
    };
    for block in blocks {
        if tool_use_matches(block, tool_name, needle) {
            if let Some(id) = block.get("id").and_then(|i| i.as_str()) {
                matched.push(id.to_string());
            }
        }
    }
}

/// Pass-1 already-pruned collector shared by `eligible_matches` and `prune_jsonl`
/// (same call sites as [`claude_collect_matched`]). Walks user `tool_result`
/// blocks whose `content` is already the `[pruned]` placeholder, marking their
/// `tool_use_id` ineligible.
fn claude_collect_pruned(
    entry: &serde_json::Value,
    already_pruned: &mut std::collections::HashSet<String>,
) {
    use crate::prune::PRUNED_PLACEHOLDER;
    if entry.get("type").and_then(|t| t.as_str()) != Some("user") {
        return;
    }
    let Some(blocks) = entry
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_array())
    else {
        return;
    };
    for block in blocks {
        if block.get("type").and_then(|t| t.as_str()) != Some("tool_result") {
            continue;
        }
        // A `[pruned]` tool_result marks its tool_use ineligible.
        if block.get("content").and_then(|c| c.as_str()) == Some(PRUNED_PLACEHOLDER) {
            if let Some(id) = block.get("tool_use_id").and_then(|i| i.as_str()) {
                already_pruned.insert(id.to_string());
            }
        }
    }
}

/// Read-only pre-scan: count eligible matches (control-side, before aborting).
fn count_matches(path: &Path, tool_name: &str, needle: &str) -> std::io::Result<usize> {
    Ok(eligible_matches(path, tool_name, needle)?.len())
}

/// Pass-2 helper: blank one parsed claude transcript entry in place for any
/// `tool_use`/`tool_result` whose id ∈ `pruned_ids` (the singleton target seeded
/// by [`prune_jsonl`]):
///   - assistant: each `tool_use` whose `id` ∈ set → `input` blanked to `{}` (the
///     path/command lives in the agent's summary; the args carry nothing it still
///     needs). Does NOT count toward `results_blanked`/`freed_bytes` — only the
///     paired output does.
///   - user: each `tool_result` whose `tool_use_id` ∈ set → `content` blanked to
///     the placeholder and the id recorded in `outputs_blanked` (drives the
///     user-facing `results_blanked` count + `freed_bytes`).
///
/// Cross-line by design: the tool_use lives on the assistant line and its
/// tool_result on a later user line; the driver runs this per-entry, so each is
/// blanked when its own line comes through. Returns `None` when the entry was
/// left untouched, or `Some(freed_bytes)` when a field was blanked (`freed` may be
/// `0` for an input-only blank or a tiny output — the driver still re-serializes).
/// Idempotent: already-blank fields no-op so a re-prune stays byte-stable.
fn blank_claude_entry(
    entry: &mut serde_json::Value,
    pruned_ids: &std::collections::HashSet<String>,
    outputs_blanked: &mut std::collections::HashSet<String>,
) -> Option<usize> {
    use crate::prune::PRUNED_PLACEHOLDER;

    let entry_type = entry.get("type").and_then(|t| t.as_str());
    let is_assistant = entry_type == Some("assistant");
    let is_user = entry_type == Some("user");
    if !is_assistant && !is_user {
        return None;
    }

    let blocks = entry
        .get_mut("message")
        .and_then(|m| m.get_mut("content"))
        .and_then(|c| c.as_array_mut())?;

    let mut freed_total = 0usize;
    let mut changed = false;
    for block in blocks.iter_mut() {
        let id = block
            .get(if is_assistant { "id" } else { "tool_use_id" })
            .and_then(|i| i.as_str())
            .map(str::to_string);
        let Some(id) = id.filter(|id| pruned_ids.contains(id)) else {
            continue;
        };
        let Some(obj) = block.as_object_mut() else {
            continue;
        };
        if is_assistant {
            if obj.get("type").and_then(|t| t.as_str()) != Some("tool_use") {
                continue;
            }
            let already_empty = matches!(
                obj.get("input"),
                Some(serde_json::Value::Object(m)) if m.is_empty()
            );
            if !already_empty {
                obj.insert("input".into(), serde_json::json!({}));
                changed = true;
            }
        } else {
            if obj.get("type").and_then(|t| t.as_str()) != Some("tool_result") {
                continue;
            }
            // `None` when already blank — keeps re-prune byte-stable. Idempotency +
            // freed-byte arithmetic live in the shared helper.
            if let Some(freed) =
                crate::prune::blank_string_field(obj, "content", PRUNED_PLACEHOLDER)
            {
                freed_total += freed;
                changed = true;
                outputs_blanked.insert(id);
            }
        }
    }

    changed.then_some(freed_total)
}

/// Single-target last-only rewrite — the one-round case of [`prune_batch_jsonl`].
/// Kept as a TEST-ONLY oracle: the batch path must be byte-identical to running
/// this once per target, so the tests pin that equivalence. Production prunes go
/// through `prune_batch_jsonl` (one read for the whole burst). Blanks the most
/// recent eligible match's tool_use input + paired tool_result content; an empty
/// target (TOCTOU / nothing eligible) is a safe no-op. Claude has no fail-closed
/// post-scan, so the driver's returned final entries are ignored.
#[cfg(test)]
fn prune_jsonl(
    path: &Path,
    tool_name: &str,
    needle: &str,
) -> std::io::Result<crate::prune::PruneStats> {
    let (stats, _final_entries) = crate::prune::rewrite_jsonl_last_only(
        path,
        |entry, matched| claude_collect_matched(entry, tool_name, needle, matched),
        claude_collect_pruned,
        blank_claude_entry,
    )?;
    Ok(stats)
}

/// Batch entry point behind `PRUNE_OPS::prune_batch`: blank the last-only target of
/// EVERY `(tool_name, needle)` in `targets` in ONE read/write, reproducing exactly
/// what running [`prune_jsonl`] once per target produced (same blanked set + freed
/// bytes). The single-read batch driver
/// ([`crate::prune::rewrite_jsonl_batch_last_only`]) re-derives the per-target
/// matches from the SAME claude collectors `prune_jsonl` uses, subtracting both the
/// on-disk `[pruned]` set and the ids already chosen earlier in the batch — so two
/// same-needle targets pick two DISTINCT successive matches, just as two separate
/// `prune_jsonl` calls did. Claude has no fail-closed post-scan, so the driver's
/// final entries are ignored.
fn prune_batch_jsonl(
    path: &Path,
    targets: &[crate::prune::PruneTarget],
) -> std::io::Result<crate::prune::PruneStats> {
    let (stats, _final_entries) = crate::prune::rewrite_jsonl_batch_last_only(
        path,
        targets.len(),
        |idx, entry, matched| {
            let (tool_name, needle) = &targets[idx];
            claude_collect_matched(entry, tool_name, needle, matched)
        },
        claude_collect_pruned,
        blank_claude_entry,
    )?;
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Returns the SmallVec as a Vec of human-readable tags + payload for
    /// easy assertion equality. Delegates to the shared stringifier in
    /// `adapter.rs` so the event→tag mapping has a single source of truth.
    fn run(adapter: &mut ClaudeAdapter, line: &str) -> Vec<String> {
        crate::adapter::stringify_events(adapter.handle_line(line.to_string()))
    }

    #[test]
    fn init_frame_harvests_session_id_and_drops_frame() {
        let mut a = ClaudeAdapter::new();
        let line = r#"{"type":"system","subtype":"init","session_id":"abc-123","model":"sonnet"}"#;
        let events = run(&mut a, line);
        assert_eq!(events, vec!["SessionIdHarvested(abc-123)"]);
    }

    #[test]
    fn task_started_frame_emits_taskstarted_and_drops_frame() {
        let mut a = ClaudeAdapter::new();
        let line = r#"{"type":"system","subtype":"task_started","task_id":"t-7","tool_use_id":"tu9","task_type":"local_bash","description":"monitor"}"#;
        assert_eq!(run(&mut a, line), vec!["TaskStarted(t-7)"]);
    }

    #[test]
    fn task_notification_frame_emits_taskfinished_and_drops_frame() {
        let mut a = ClaudeAdapter::new();
        let line = r#"{"type":"system","subtype":"task_notification","task_id":"t-7","status":"completed","output_file":"/tmp/x","summary":"done"}"#;
        assert_eq!(run(&mut a, line), vec!["TaskFinished(t-7)"]);
    }

    #[test]
    fn task_updated_progress_frame_is_skipped_silently() {
        let mut a = ClaudeAdapter::new();
        let line = r#"{"type":"system","subtype":"task_updated","task_id":"t-7","patch":{"status":"running"}}"#;
        assert!(run(&mut a, line).is_empty());
    }

    /// The resident `stream-json` session signals a finished Monitor with a
    /// `task_updated{patch.status:"completed"}` and NO `task_notification` — verbatim
    /// frame captured live (chat c06c30b2, task bo87ftywa). It must emit TaskFinished
    /// so `live_tasks` empties; otherwise `running` sticks true forever (stuck Waiting).
    #[test]
    fn task_updated_completed_frame_emits_taskfinished() {
        let mut a = ClaudeAdapter::new();
        let line = r#"{"type":"system","subtype":"task_updated","task_id":"bo87ftywa","patch":{"status":"completed","end_time":1780889095302},"uuid":"c2a6a1a4-a71d-4b06-b7bd-187eb6147796","session_id":"d1883d0c-8e31-4537-9963-47d282a6ab31"}"#;
        assert_eq!(run(&mut a, line), vec!["TaskFinished(bo87ftywa)"]);
    }

    #[test]
    fn task_updated_failed_and_cancelled_frames_emit_taskfinished() {
        let mut a = ClaudeAdapter::new();
        let failed = r#"{"type":"system","subtype":"task_updated","task_id":"t-7","patch":{"status":"failed"}}"#;
        assert_eq!(run(&mut a, failed), vec!["TaskFinished(t-7)"]);
        let cancelled = r#"{"type":"system","subtype":"task_updated","task_id":"t-8","patch":{"status":"cancelled"}}"#;
        assert_eq!(run(&mut a, cancelled), vec!["TaskFinished(t-8)"]);
    }

    #[test]
    fn encode_user_turn_is_valid_json_with_trailing_newline() {
        let a = ClaudeAdapter::new();
        let frame = a.encode_user_turn("hello \"world\"\nwith newline");
        assert!(frame.ends_with('\n'), "frame must be newline-terminated");
        let v: serde_json::Value = serde_json::from_str(frame.trim_end()).unwrap();
        assert_eq!(v["type"], "user");
        assert_eq!(v["message"]["role"], "user");
        assert_eq!(v["message"]["content"][0]["type"], "text");
        assert_eq!(
            v["message"]["content"][0]["text"],
            "hello \"world\"\nwith newline"
        );
    }

    #[test]
    fn prepare_session_command_drops_prompt_pipe_and_adds_input_format() {
        use crate::adapter::SessionContext;
        let mut a = ClaudeAdapter::new();
        let ctx = SessionContext {
            chat_id: "abcdef012345-6789",
            project_path: Some("/tmp/proj"),
            worktree: false,
            agent_session_id: Some("sid-9"),
            is_sandboxed: false,
            model: Some("opus"),
            user_timezone: Some("Asia/Bangkok"),
        };
        let cmd = a.prepare_session_command(&ctx).unwrap();
        // No prompt-file piping in the resident path.
        assert!(
            !cmd.contains("cat "),
            "resident cmd must not pipe a prompt file: {cmd}"
        );
        // Reads turns from stdin.
        assert!(cmd.contains("--input-format stream-json"), "got: {cmd}");
        assert!(cmd.contains("--output-format stream-json"), "got: {cmd}");
        // Resume + model + sandbox bypass + prune hook all preserved.
        assert!(cmd.contains("--resume 'sid-9'"), "got: {cmd}");
        assert!(cmd.contains("--model 'opus'"), "got: {cmd}");
        assert!(cmd.contains("--dangerously-skip-permissions"), "got: {cmd}");
        assert!(cmd.contains("prune-reminder-hook"), "got: {cmd}");
        // cd prefix retained.
        assert!(cmd.starts_with("cd '/tmp/proj' && claude"), "got: {cmd}");
    }

    #[test]
    fn claude_adapter_is_resident() {
        assert!(ClaudeAdapter::new().is_resident());
    }

    #[test]
    fn user_turn_result_is_not_task_origin() {
        let mut a = ClaudeAdapter::new();
        // `result` frames are latched (Result) AND forwarded as the result row
        // (Frame) — they are not skipped, unlike init/task frames.
        let line = r#"{"type":"result","subtype":"success","is_error":false,"result":"ok"}"#;
        let events = run(&mut a, line);
        assert_eq!(events[0], "Result");
        assert!(events[1].starts_with("Frame("));
    }

    #[test]
    fn task_wake_result_is_task_origin() {
        let mut a = ClaudeAdapter::new();
        let line = r#"{"type":"result","subtype":"success","is_error":false,"origin":{"kind":"task-notification"},"result":"task done"}"#;
        let events = run(&mut a, line);
        assert_eq!(events[0], "Result(task)");
        assert!(events[1].starts_with("Frame("));
    }

    // --- Regression fixtures captured from REAL claude 2.1.165 ---------------
    // (tmp/claude_proto_probe.py, 2026-06-07). These lock the parser + FSM to
    // the genuine wire shapes so a silent format drift breaks a test, not a
    // user's Thinking/Waiting indicator or /stop.

    /// Drives the verbatim background-task frames through `handle_line` →
    /// `reduce` and asserts the headline transition: arming-turn result with a
    /// live task ⇒ WAITING; the task's `task_notification` ⇒ back to Idle; and
    /// the task-wake result (origin = task-notification) must NOT clear an
    /// unrelated in-flight turn.
    #[test]
    fn real_background_task_lifecycle_drives_fsm_idle_waiting_idle() {
        use crate::agent::{reduce, SessionState};
        let mut a = ClaudeAdapter::new();
        let mut st = SessionState {
            turn_in_flight: true, // the arming user turn is in flight
            ..Default::default()
        };
        let feed = |a: &mut ClaudeAdapter, st: &mut SessionState, line: &str| {
            for ev in a.handle_line(line.to_string()) {
                reduce(st, &ev);
            }
        };

        // 1. Background task armed (run_in_background Bash). Real shape.
        feed(
            &mut a,
            &mut st,
            r#"{"type":"system","subtype":"task_started","task_id":"bzdt1ec2n","tool_use_id":"toolu_01","description":"sleep 8 && echo BG_DONE","task_type":"local_bash","session_id":"s1"}"#,
        );
        assert!(
            st.live_tasks.contains("bzdt1ec2n"),
            "task_started must register the task"
        );

        // 2. The arming turn finishes (no origin). turn clears; task still live.
        feed(
            &mut a,
            &mut st,
            r#"{"type":"result","subtype":"success","is_error":false,"terminal_reason":"completed","result":"started"}"#,
        );
        assert!(st.waiting(), "turn done + live task ⇒ WAITING");
        assert!(st.running(), "WAITING is a running sub-state");
        assert!(!st.is_idle());

        // 3. Progress ping (non-terminal status) — must be a no-op (task stays live).
        feed(
            &mut a,
            &mut st,
            r#"{"type":"system","subtype":"task_updated","task_id":"bzdt1ec2n","patch":{"status":"running"}}"#,
        );
        assert!(
            st.waiting(),
            "task_updated progress must not change run state"
        );

        // 4. Terminal task signal ⇒ task removed ⇒ Idle. The resident stream-json
        //    session emits this as a `task_updated{patch.status:"completed"}` (NOT a
        //    `task_notification`); failing to treat it as terminal here is exactly the
        //    bug that pinned chats to agent_running/"Waiting…" after a Monitor ended
        //    (chat c06c30b2). The `task_notification` shape is covered by
        //    `task_notification_frame_emits_taskfinished_and_drops_frame`.
        feed(
            &mut a,
            &mut st,
            r#"{"type":"system","subtype":"task_updated","task_id":"bzdt1ec2n","patch":{"status":"completed","end_time":1780889095302}}"#,
        );
        assert!(
            st.is_idle(),
            "task_updated{{completed}} clears the last live task"
        );

        // 5. The task-wake turn's result carries origin=task-notification — it
        //    must NOT touch turn state. Pretend a real user turn is in flight to
        //    prove the wake result doesn't clear it.
        st.turn_in_flight = true;
        feed(
            &mut a,
            &mut st,
            r#"{"type":"result","subtype":"success","is_error":false,"origin":{"kind":"task-notification"},"terminal_reason":"completed","result":"bg output: BG_DONE"}"#,
        );
        assert!(
            st.turn_in_flight,
            "a task-wake result must NOT clear an unrelated in-flight user turn"
        );
    }

    /// Real claude emits an abort result for the killed turn on /stop. We
    /// publish our own canonical `interrupted` row, so this one must be
    /// suppressed (no Frame) to avoid a double terminal indicator — while still
    /// emitting `Result` so the FSM clears `turn_in_flight`.
    #[test]
    fn interrupt_abort_result_suppresses_row_but_emits_result() {
        let mut a = ClaudeAdapter::new();
        let line = r#"{"type":"result","subtype":"error_during_execution","is_error":true,"terminal_reason":"aborted_streaming","result":"[Request interrupted by user]"}"#;
        let events = run(&mut a, line);
        assert_eq!(
            events,
            vec!["Result"],
            "abort result must emit Result for the FSM but NO Frame (row suppressed)"
        );
    }

    /// Fail-open guard: a genuine execution error (NOT the aborted-streaming
    /// interrupt signature) still renders as a row so real failures stay visible.
    #[test]
    fn genuine_execution_error_still_renders_a_row() {
        let mut a = ClaudeAdapter::new();
        let line = r#"{"type":"result","subtype":"error_during_execution","is_error":true,"terminal_reason":"error","result":"boom"}"#;
        let events = run(&mut a, line);
        assert_eq!(events[0], "Result");
        assert!(
            events[1].starts_with("Frame("),
            "non-abort execution errors must still render: {events:?}"
        );
    }

    #[test]
    fn assistant_text_frame_emits_frame_and_context_tokens() {
        let mut a = ClaudeAdapter::new();
        let line = r#"{"type":"assistant","message":{"content":[{"type":"text","text":"hi"}],"usage":{"input_tokens":10,"cache_creation_input_tokens":2,"cache_read_input_tokens":3,"output_tokens":1}}}"#;
        let events = run(&mut a, line);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0], "ContextTokens(15)");
        assert!(events[1].starts_with("Frame("));
    }

    #[test]
    fn assistant_thinking_only_frame_skipped_but_tokens_emitted() {
        let mut a = ClaudeAdapter::new();
        let line = r#"{"type":"assistant","message":{"content":[{"type":"thinking","thinking":"..."}],"usage":{"input_tokens":10,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":1}}}"#;
        let events = run(&mut a, line);
        // Tokens still harvested even though the frame itself is skipped (UI doesn't render thinking-only).
        assert_eq!(events, vec!["ContextTokens(10)"]);
    }

    #[test]
    fn assistant_tool_use_frame_kept() {
        let mut a = ClaudeAdapter::new();
        let line = r#"{"type":"assistant","message":{"content":[{"type":"tool_use","id":"tu1","name":"Bash","input":{"command":"ls"}}],"usage":{"input_tokens":1,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":1}}}"#;
        let events = run(&mut a, line);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0], "ContextTokens(1)");
        assert!(events[1].starts_with("Frame("));
    }

    #[test]
    fn tool_result_without_pending_prune_emits_no_cue() {
        let mut a = ClaudeAdapter::new();
        // No `prune-context` call recorded this turn ⇒ an ordinary tool_result
        // is skipped silently (the cue is call-keyed, not chat-keyed).
        let line = r#"{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"tu1","content":"out"}]}}"#;
        assert!(run(&mut a, line).is_empty());
    }

    #[test]
    fn prune_context_result_emits_cue_only_for_its_own_id() {
        let mut a = ClaudeAdapter::new();
        // Assistant frame: a `prune-context` call (tu_prune) batched in parallel
        // with a sibling Read (tu_read).
        let assistant = r#"{"type":"assistant","message":{"content":[{"type":"tool_use","id":"tu_read","name":"Read","input":{"file_path":"/x"}},{"type":"tool_use","id":"tu_prune","name":"Bash","input":{"command":"\"$ZUCCHINI_SPAWNER_BIN\" prune-context --tool-name Read --args \"*x*\" --reason y"}}],"usage":{"input_tokens":1,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":1}}}"#;
        let _ = run(&mut a, assistant);
        // The sibling's result lands FIRST — must NOT fire the cue.
        let read_result = r#"{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"tu_read","content":"big file body"}]}}"#;
        assert!(run(&mut a, read_result).is_empty());
        // The prune-context call's OWN result lands second — fires the cue once.
        let prune_result = r#"{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"tu_prune","content":"pruned 2 outputs"}]}}"#;
        assert_eq!(run(&mut a, prune_result), vec!["ToolResult"]);
        // Id is consumed: a duplicate/late result for the same id does not re-fire.
        assert!(run(&mut a, prune_result).is_empty());
    }

    #[test]
    fn oversized_prune_assistant_frame_still_records_id_and_emits_cue() {
        // Regression: the prune-context id RECORD must NOT be size-gated. A
        // batched multi-target prune call (many --tool-name/--args/--summary
        // triples) can push the assistant frame past MAX_STREAM_FRAME_BYTES.
        // Previously RECORD lived under that gate while the tool_result EMIT
        // did not, so an oversized prune frame never recorded its id ⇒ the cue
        // silently never fired ⇒ the queued prune was a no-op. Record and emit
        // now share one policy; prove an >64KiB prune frame still records and
        // its result still emits ToolResult.
        let mut a = ClaudeAdapter::new();
        // Pad the prune-context command's --summary with a large needle/summary
        // so the whole assistant frame exceeds MAX_STREAM_FRAME_BYTES.
        let huge = "x".repeat(MAX_STREAM_FRAME_BYTES + 4096);
        let assistant = format!(
            r#"{{"type":"assistant","message":{{"content":[{{"type":"tool_use","id":"tu_prune","name":"Bash","input":{{"command":"\"$ZUCCHINI_SPAWNER_BIN\" prune-context --tool-name Read --args needle --summary {huge}"}}}}],"usage":{{"input_tokens":1,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":1}}}}}}"#
        );
        assert!(
            assistant.len() > MAX_STREAM_FRAME_BYTES,
            "test frame must exceed the size gate (got {} bytes)",
            assistant.len()
        );
        // Oversized assistant frame: usage parse is skipped (size-gated), but
        // the prune id must still be recorded.
        let _ = run(&mut a, &assistant);
        // The prune-context call's own result fires the cue exactly once,
        // proving the id was recorded despite the oversized frame.
        let prune_result = r#"{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"tu_prune","content":"pruned 1 output"}]}}"#;
        assert_eq!(run(&mut a, prune_result), vec!["ToolResult"]);
        // Consumed: a duplicate result does not re-fire.
        assert!(run(&mut a, prune_result).is_empty());
    }

    #[test]
    fn user_frame_without_tool_result_dropped_silently() {
        let mut a = ClaudeAdapter::new();
        // A plain user frame (no tool_result block) is skipped with no cue.
        let line = r#"{"type":"user","message":{"content":[{"type":"text","text":"hi"}]}}"#;
        assert!(run(&mut a, line).is_empty());
    }

    #[test]
    fn stream_event_dropped() {
        let mut a = ClaudeAdapter::new();
        let line = r#"{"type":"stream_event","event":{}}"#;
        assert!(run(&mut a, line).is_empty());
    }

    #[test]
    fn compact_boundary_frame_harvests_post_tokens_and_skips() {
        let mut a = ClaudeAdapter::new();
        // Real `--output-format stream-json` wire shape (snake_case), captured
        // live from claude 2.1.170. The frame the spawner actually sees.
        let line = r#"{"type":"system","subtype":"compact_boundary","session_id":"s","uuid":"u","compact_metadata":{"trigger":"manual","pre_tokens":46139,"post_tokens":2388,"duration_ms":37287}}"#;
        let events = run(&mut a, line);
        assert_eq!(events, vec!["CompactBoundary(2388)"]);
    }

    #[test]
    fn compact_boundary_frame_accepts_legacy_camelcase_shape() {
        let mut a = ClaudeAdapter::new();
        // Transcript-jsonl / older-build camelCase shape — accepted via serde alias.
        let line = r#"{"type":"system","subtype":"compact_boundary","compactMetadata":{"postTokens":1234,"trigger":"manual"}}"#;
        let events = run(&mut a, line);
        assert_eq!(events, vec!["CompactBoundary(1234)"]);
    }

    #[test]
    fn result_frame_emits_result_marker_and_frame() {
        let mut a = ClaudeAdapter::new();
        let line = r#"{"type":"result","subtype":"success","duration_ms":1234,"is_error":false}"#;
        let events = run(&mut a, line);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0], "Result");
        assert!(events[1].starts_with("Frame("));
        // Adapter emits Result on every result line; the supervisor (not the
        // adapter) latches it so AgentResponse::Done.has_result is set once.
        let again = run(&mut a, line);
        assert_eq!(again.len(), 2);
        assert_eq!(again[0], "Result");
        assert!(again[1].starts_with("Frame("));
    }

    #[test]
    fn repeated_usage_dedups() {
        let mut a = ClaudeAdapter::new();
        let line = r#"{"type":"assistant","message":{"content":[{"type":"text","text":"hi"}],"usage":{"input_tokens":10,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":1}}}"#;
        let first = run(&mut a, line);
        assert_eq!(first[0], "ContextTokens(10)");
        let second = run(&mut a, line);
        // Same tokens → no ContextTokens event the second time, just the Frame.
        assert_eq!(second.len(), 1);
        assert!(second[0].starts_with("Frame("));
    }

    #[test]
    fn assistant_frame_without_usage_skips_context_tokens_but_keeps_frame() {
        // Defensive against claude format drift: an interim assistant frame
        // that omits `usage` (renamed field, partial-flush variant, etc.)
        // must not poison the parse — the frame still reaches the chat
        // (preserving the text/tool_use body), and ContextTokens is
        // simply not emitted for that line. The next usage-carrying
        // frame resumes the counter from the dedup state.
        let mut a = ClaudeAdapter::new();
        let line = r#"{"type":"assistant","message":{"content":[{"type":"text","text":"hi"}]}}"#;
        let events = run(&mut a, line);
        // No ContextTokens — usage absent, parse returned None.
        // Frame still emitted — text content path is unaffected.
        assert_eq!(events.len(), 1);
        assert!(events[0].starts_with("Frame("));
        // Dedup state stays clean so the next usage-bearing frame fires.
        let next = r#"{"type":"assistant","message":{"content":[{"type":"text","text":"hi"}],"usage":{"input_tokens":7,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":1}}}"#;
        let events = run(&mut a, next);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0], "ContextTokens(7)");
        assert!(events[1].starts_with("Frame("));
    }

    #[test]
    fn subagent_frame_skips_context_tokens_but_keeps_frame() {
        // Subagent (Task) assistant frames carry `parent_tool_use_id` at the
        // top level. Their `usage` is the subagent's own context, not the
        // main chat's, so emitting ContextTokens here would clobber
        // `chats.context_tokens` with the subagent count and jitter back
        // on the next interleaved main-agent frame.
        let mut a = ClaudeAdapter::new();
        // First, a main-agent frame establishes a baseline.
        let main = r#"{"type":"assistant","parent_tool_use_id":null,"message":{"content":[{"type":"text","text":"hi"}],"usage":{"input_tokens":10,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":1}}}"#;
        let events = run(&mut a, main);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0], "ContextTokens(10)");
        // Now a subagent frame with a very different usage count — must NOT emit.
        let sub = r#"{"type":"assistant","parent_tool_use_id":"toolu_abc","message":{"content":[{"type":"text","text":"sub"}],"usage":{"input_tokens":9999,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":1}}}"#;
        let events = run(&mut a, sub);
        // Frame still passes through (rendered as a regular assistant message),
        // but no ContextTokens was emitted for the subagent's usage.
        assert_eq!(events.len(), 1);
        assert!(events[0].starts_with("Frame("));
        // Next main-agent frame with a NEW value still fires (dedup state untouched).
        let main2 = r#"{"type":"assistant","message":{"content":[{"type":"text","text":"hi2"}],"usage":{"input_tokens":11,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":1}}}"#;
        let events = run(&mut a, main2);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0], "ContextTokens(11)");
        assert!(events[1].starts_with("Frame("));
    }

    #[test]
    fn non_json_line_kept_as_frame() {
        let mut a = ClaudeAdapter::new();
        // A spawner stderr-merged line (shouldn't happen — stderr is buffered
        // separately — but the line loop still gets bytes from stdout). Lines
        // that don't start with `{` skip every JSON branch and are kept.
        let events = run(&mut a, "non-json-line");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0], "Frame(non-json-line)");
    }

    #[test]
    fn model_some_appends_model_flag() {
        // `chats.model = Some("opus")` (migration 0035, picked from the
        // composer's agent roster) → claude command carries `--model 'opus'`
        // verbatim. Shell-escaping is via `adapter::shell_escape` so the
        // single quotes are part of the contract — the same form the agent
        // session id and prompt path use.
        use crate::adapter::TurnContext;
        use std::path::PathBuf;
        let mut a = ClaudeAdapter::new();
        let prompt_file = PathBuf::from("/tmp/p.txt");
        let ctx = TurnContext::for_test(&prompt_file, None, false, Some("opus"));
        let cmd = a.prepare_command(&ctx).unwrap();
        assert!(cmd.contains("--model 'opus'"), "got: {}", cmd);
    }

    #[test]
    fn model_none_omits_model_flag() {
        // `chats.model = None` (no override; NULL or empty filtered upstream)
        // → no `--model` flag at all; claude picks the user's default from
        // its own config.
        use crate::adapter::TurnContext;
        use std::path::PathBuf;
        let mut a = ClaudeAdapter::new();
        let prompt_file = PathBuf::from("/tmp/p.txt");
        let ctx = TurnContext::for_test(&prompt_file, None, false, None);
        let cmd = a.prepare_command(&ctx).unwrap();
        assert!(!cmd.contains("--model"), "got: {}", cmd);
    }

    #[test]
    fn injected_system_prompt_carries_capabilities_and_no_worktree_when_off() {
        // claude injects via --append-system-prompt: block carries capabilities,
        // no worktree rule (off), and the prepend path is unused.
        use crate::adapter::TurnContext;
        use std::path::PathBuf;
        let mut a = ClaudeAdapter::new();
        let prompt_file = PathBuf::from("/tmp/p.txt");
        let ctx = TurnContext::for_test(&prompt_file, None, false, None);
        let cmd = a.prepare_command(&ctx).unwrap();
        assert!(cmd.contains("--append-system-prompt"), "got: {}", cmd);
        assert!(cmd.contains("attach-file"), "got: {}", cmd);
        assert!(cmd.contains("schedule-message"), "got: {}", cmd);
        assert!(
            !cmd.contains("Worktree:"),
            "worktree off → no rule: {}",
            cmd
        );
        assert!(
            a.prompt_file_preamble(&ctx).is_none(),
            "claude conveys via system prompt, not the prepend path"
        );
    }

    #[test]
    fn worktree_on_folds_worktree_rule_into_injected_block() {
        // worktree=true + known project path → worktree rule (abs path + parent
        // repo) lands in the same --append-system-prompt block.
        use crate::adapter::TurnContext;
        use std::path::PathBuf;
        let mut a = ClaudeAdapter::new();
        let prompt_file = PathBuf::from("/tmp/p.txt");
        let ctx = TurnContext {
            // chat_id prefix names the worktree dir, asserted below.
            chat_id: "abcdef012345-6789-...",
            worktree: true,
            ..TurnContext::for_test(&prompt_file, None, false, None)
        };
        let cmd = a.prepare_command(&ctx).unwrap();
        assert!(cmd.contains("--worktree"), "got: {}", cmd);
        assert!(cmd.contains("Worktree:"), "worktree rule present: {}", cmd);
        // chat_id prefix (first 8 chars) names the worktree dir under the project.
        assert!(
            cmd.contains("/tmp/proj/.claude/worktrees/abcdef01"),
            "absolute worktree path in rule: {}",
            cmd
        );
        assert!(cmd.contains("Parent repo: /tmp/proj"), "got: {}", cmd);
        // Capability instructions still present in the same block.
        assert!(cmd.contains("attach-file"), "got: {}", cmd);
        assert!(cmd.contains("schedule-message"), "got: {}", cmd);
    }

    #[test]
    fn prepare_command_wires_prune_feature() {
        use crate::adapter::TurnContext;
        use std::path::PathBuf;
        let mut a = ClaudeAdapter::new();
        let prompt_file = PathBuf::from("/tmp/p.txt");
        let ctx = TurnContext::for_test(&prompt_file, None, false, None);
        let cmd = a.prepare_command(&ctx).unwrap();
        // The standing prune-context instruction rides --append-system-prompt …
        assert!(cmd.contains("prune-context"), "got: {}", cmd);
        // … and the PostToolUse reminder hook is injected via --settings.
        assert!(cmd.contains("--settings"), "got: {}", cmd);
        assert!(cmd.contains("prune-reminder-hook"), "got: {}", cmd);
        assert!(cmd.contains("PostToolUse"), "got: {}", cmd);
    }

    mod prune {
        use super::super::*;
        use crate::prune::test_util::{read_lines, write_jsonl};

        /// One assistant `tool_use` (Read junk.rs) paired to a user `tool_result`.
        /// Pruning by (Read, junk) blanks the result content to `[pruned]` and the
        /// tool_use input to `{}`, preserving every other field.
        #[test]
        fn prunes_paired_tool_use_and_result() {
            let f = write_jsonl(&[
                r#"{"type":"assistant","message":{"content":[{"type":"tool_use","id":"tu1","name":"Read","input":{"file_path":"src/junk.rs"}}]},"uuid":"a1","parentUuid":null}"#,
                r#"{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"tu1","content":"a big file body"}]},"uuid":"u1","parentUuid":"a1"}"#,
            ]);
            assert_eq!(count_matches(f.path(), "Read", "junk").unwrap(), 1);
            let stats = prune_jsonl(f.path(), "Read", "junk").unwrap();
            assert_eq!(stats.results_blanked, 1);
            assert!(stats.freed_bytes > 0);
            let lines = read_lines(f.path());
            // tool_use input blanked to {}, threading fields intact.
            assert_eq!(
                lines[0]["message"]["content"][0]["input"],
                serde_json::json!({})
            );
            assert_eq!(lines[0]["uuid"], "a1");
            // paired tool_result content blanked to the placeholder.
            assert_eq!(lines[1]["message"]["content"][0]["content"], "[pruned]");
            assert_eq!(lines[1]["uuid"], "u1");
        }

        /// Last-only: with two eligible matches, only the most recent is blanked;
        /// a repeat blanks the older one; a third call finds nothing.
        #[test]
        fn last_only_then_older_then_zero() {
            let f = write_jsonl(&[
                r#"{"type":"assistant","message":{"content":[{"type":"tool_use","id":"t1","name":"Read","input":{"file_path":"junk1.rs"}}]}}"#,
                r#"{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"t1","content":"body one"}]}}"#,
                r#"{"type":"assistant","message":{"content":[{"type":"tool_use","id":"t2","name":"Read","input":{"file_path":"junk2.rs"}}]}}"#,
                r#"{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"t2","content":"body two"}]}}"#,
            ]);
            assert_eq!(count_matches(f.path(), "Read", "junk").unwrap(), 2);

            assert_eq!(
                prune_jsonl(f.path(), "Read", "junk")
                    .unwrap()
                    .results_blanked,
                1
            );
            let lines = read_lines(f.path());
            // Newest (t2) blanked, oldest (t1) still live.
            assert_eq!(lines[1]["message"]["content"][0]["content"], "body one");
            assert_eq!(lines[3]["message"]["content"][0]["content"], "[pruned]");
            assert_eq!(count_matches(f.path(), "Read", "junk").unwrap(), 1);

            assert_eq!(
                prune_jsonl(f.path(), "Read", "junk")
                    .unwrap()
                    .results_blanked,
                1
            );
            assert_eq!(
                read_lines(f.path())[1]["message"]["content"][0]["content"],
                "[pruned]"
            );

            // Nothing eligible left; a further prune is a byte-stable no-op.
            assert_eq!(count_matches(f.path(), "Read", "junk").unwrap(), 0);
            assert_eq!(
                prune_jsonl(f.path(), "Read", "junk")
                    .unwrap()
                    .results_blanked,
                0
            );
        }

        /// A prune NEVER targets the agent's own in-flight `prune-context` call,
        /// even though its command line carries the same needle (here both the Read
        /// and the prune call mention `junk.rs`). Only the real Read is blanked.
        #[test]
        fn never_targets_own_prune_context_call() {
            let f = write_jsonl(&[
                r#"{"type":"assistant","message":{"content":[{"type":"tool_use","id":"r1","name":"Read","input":{"file_path":"junk.rs"}}]}}"#,
                r#"{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"r1","content":"body"}]}}"#,
                r#"{"type":"assistant","message":{"content":[{"type":"tool_use","id":"p1","name":"Bash","input":{"command":"\"$ZUCCHINI_SPAWNER_BIN\" prune-context --tool-name Read --args junk.rs --summary x"}}]}}"#,
                r#"{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"p1","content":"pruned 1 tool output"}]}}"#,
            ]);
            // Only the real Read is eligible — the prune call is excluded.
            assert_eq!(count_matches(f.path(), "", "junk").unwrap(), 1);
            assert_eq!(
                prune_jsonl(f.path(), "", "junk").unwrap().results_blanked,
                1
            );
            let lines = read_lines(f.path());
            assert_eq!(lines[1]["message"]["content"][0]["content"], "[pruned]");
            // The prune call's own result is untouched.
            assert_eq!(
                lines[3]["message"]["content"][0]["content"],
                "pruned 1 tool output"
            );
        }

        /// `find_session_jsonl` resolves `<base>/projects/<dir>/<sid>.jsonl`.
        #[test]
        fn find_session_locates_sharded_transcript() {
            let base = std::env::temp_dir().join(format!(
                "zucchini_claude_find_{}_{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            let proj = base.join("projects").join("-tmp-proj");
            std::fs::create_dir_all(&proj).unwrap();
            let sid = "11111111-2222-3333-4444-555555555555";
            let target = proj.join(format!("{sid}.jsonl"));
            std::fs::write(&target, "{}\n").unwrap();
            assert_eq!(
                find_session_jsonl(&base, sid).as_deref(),
                Some(target.as_path())
            );
            assert!(find_session_jsonl(&base, "no-such-session").is_none());
            let _ = std::fs::remove_dir_all(&base);
        }

        /// A batch of K DISTINCT-needle targets blanks all K outputs in ONE
        /// `prune_batch_jsonl` read/write — the coalesced-burst case
        /// `apply_prune_group` drives. Stats sum to K.
        #[test]
        fn batch_distinct_needles_blanks_all_in_one_pass() {
            let f = write_jsonl(&[
                r#"{"type":"assistant","message":{"content":[{"type":"tool_use","id":"t1","name":"Read","input":{"file_path":"junk1.rs"}}]}}"#,
                r#"{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"t1","content":"body one"}]}}"#,
                r#"{"type":"assistant","message":{"content":[{"type":"tool_use","id":"t2","name":"Read","input":{"file_path":"junk2.rs"}}]}}"#,
                r#"{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"t2","content":"body two"}]}}"#,
                r#"{"type":"assistant","message":{"content":[{"type":"tool_use","id":"t3","name":"Read","input":{"file_path":"junk3.rs"}}]}}"#,
                r#"{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"t3","content":"body three"}]}}"#,
            ]);
            let targets = vec![
                ("Read".to_string(), "junk1.rs".to_string()),
                ("Read".to_string(), "junk2.rs".to_string()),
                ("Read".to_string(), "junk3.rs".to_string()),
            ];
            let stats = prune_batch_jsonl(f.path(), &targets).unwrap();
            assert_eq!(stats.results_blanked, 3, "all 3 distinct targets blanked");
            assert!(stats.freed_bytes > 0);
            let lines = read_lines(f.path());
            for i in [1usize, 3, 5] {
                assert_eq!(
                    lines[i]["message"]["content"][0]["content"], "[pruned]",
                    "line {i} should be pruned"
                );
            }
        }

        /// Two SAME-needle targets in one batch blank two DISTINCT successive
        /// matches (newest two of three), exactly as two separate `prune_jsonl`
        /// calls would — the in-memory K-round selection re-derives last-only with a
        /// running "chosen this batch" set. The oldest match stays live.
        #[test]
        fn batch_same_needle_blanks_two_distinct_matches() {
            let f = write_jsonl(&[
                r#"{"type":"assistant","message":{"content":[{"type":"tool_use","id":"t1","name":"Read","input":{"file_path":"junk.rs"}}]}}"#,
                r#"{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"t1","content":"body one"}]}}"#,
                r#"{"type":"assistant","message":{"content":[{"type":"tool_use","id":"t2","name":"Read","input":{"file_path":"junk.rs"}}]}}"#,
                r#"{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"t2","content":"body two"}]}}"#,
                r#"{"type":"assistant","message":{"content":[{"type":"tool_use","id":"t3","name":"Read","input":{"file_path":"junk.rs"}}]}}"#,
                r#"{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"t3","content":"body three"}]}}"#,
            ]);
            let targets = vec![
                ("Read".to_string(), "junk".to_string()),
                ("Read".to_string(), "junk".to_string()),
            ];
            let stats = prune_batch_jsonl(f.path(), &targets).unwrap();
            assert_eq!(
                stats.results_blanked, 2,
                "two same-needle targets blank two distinct matches"
            );
            let lines = read_lines(f.path());
            // Oldest (t1) untouched; newest two (t2, t3) pruned.
            assert_eq!(lines[1]["message"]["content"][0]["content"], "body one");
            assert_eq!(lines[3]["message"]["content"][0]["content"], "[pruned]");
            assert_eq!(lines[5]["message"]["content"][0]["content"], "[pruned]");
            // One eligible match remains for a follow-up prune.
            assert_eq!(count_matches(f.path(), "Read", "junk").unwrap(), 1);
        }

        /// The batch path is BYTE-IDENTICAL to running `prune_jsonl` once per target
        /// in sequence (the equivalence the single-read fold rests on). Same
        /// transcript, same targets, two independent runs → identical file bytes +
        /// identical summed stats.
        #[test]
        fn batch_is_byte_identical_to_sequential_single_calls() {
            let lines_src: &[&str] = &[
                r#"{"type":"assistant","message":{"content":[{"type":"tool_use","id":"t1","name":"Read","input":{"file_path":"junk.rs"}}]}}"#,
                r#"{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"t1","content":"body one"}]}}"#,
                r#"{"type":"assistant","message":{"content":[{"type":"tool_use","id":"t2","name":"Read","input":{"file_path":"junk.rs"}}]}}"#,
                r#"{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"t2","content":"body two longer"}]}}"#,
                r#"{"type":"assistant","message":{"content":[{"type":"tool_use","id":"t3","name":"Read","input":{"file_path":"other.rs"}}]}}"#,
                r#"{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"t3","content":"unrelated body"}]}}"#,
            ];
            // Sequential: two same-needle junk prunes, then one other.rs prune.
            let seq = write_jsonl(lines_src);
            let mut seq_stats = crate::prune::PruneStats::default();
            for (tn, nd) in [("Read", "junk"), ("Read", "junk"), ("Read", "other")] {
                let s = prune_jsonl(seq.path(), tn, nd).unwrap();
                seq_stats.results_blanked += s.results_blanked;
                seq_stats.freed_bytes += s.freed_bytes;
            }
            let seq_bytes = std::fs::read_to_string(seq.path()).unwrap();

            // Batch: the same three targets in one call.
            let batch = write_jsonl(lines_src);
            let targets = vec![
                ("Read".to_string(), "junk".to_string()),
                ("Read".to_string(), "junk".to_string()),
                ("Read".to_string(), "other".to_string()),
            ];
            let batch_stats = prune_batch_jsonl(batch.path(), &targets).unwrap();
            let batch_bytes = std::fs::read_to_string(batch.path()).unwrap();

            assert_eq!(
                batch_bytes, seq_bytes,
                "batch rewrite must be byte-identical"
            );
            assert_eq!(batch_stats.results_blanked, seq_stats.results_blanked);
            assert_eq!(batch_stats.freed_bytes, seq_stats.freed_bytes);
            assert_eq!(batch_stats.results_blanked, 3);
        }
    }
}
