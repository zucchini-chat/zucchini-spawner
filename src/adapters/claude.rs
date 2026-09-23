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
    agent_capabilities_instructions, file_nonempty, probe_with_blocking_auth, shell_escape,
    AdapterDescriptor, AgentAdapter, AgentEvent, AgentKind, LastTokensDedup, NativeSpawn,
    TurnContext, WorktreeInstructions, MAX_STREAM_FRAME_BYTES, PRUNE_CONTEXT_INSTRUCTION,
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

/// First sentence of the one-shot `--append-system-prompt` block. One-shot
/// `--print` is print-and-exit: the process dies at its first `result`, so any
/// background subagent it armed is abandoned. Steer the agent away from
/// `run_in_background:true`. The resident session has no lead-in: claude's own
/// tool descriptions already say background work wakes the agent.
const ONESHOT_SYS_LEADIN: &str =
    "You are spawned via a harness, no background subagents will wake you when finished, use subagents with `run_in_background: false` only.";

/// Claude-only addendum to `PRUNE_CONTEXT_INSTRUCTION` (shared with cursor/pi,
/// whose pruners don't move the summary).
const PRUNE_SUMMARY_MOVE_INSTRUCTION: &str =
    "Each --summary is moved into the pruned tool result (the prune command keeps a marker in its place), so write it to stand on its own next to that call.";

/// One token of the claude invocation. `Lit` is emitted verbatim (unquoted in
/// the shell string) — the constant flags/values that need no escaping. `Val`
/// is a dynamic value: shell-escaped in the POSIX string, one argv element
/// natively. Splitting them lets the POSIX renderer reproduce the exact
/// historical string (constant flags stay unquoted) while the native renderer
/// yields clean argv — one source of truth for the flag list.
enum ClaudeArg {
    Lit(&'static str),
    Val(String),
}

/// Build the ordered claude flag tokens AFTER the base invocation (`cat … |
/// claude` for one-shot, bare `claude` for resident). Shared by the POSIX
/// string path (`prepare_command`/`prepare_session_command` via
/// `push_posix_flags`) and the Windows-native argv path (`prepare_native`/
/// `prepare_session_native` via `flags_to_argv`) so the flag set (session-id
/// resume, worktree containment, `--append-system-prompt`, `--print --verbose
/// --output-format stream-json`, sandbox, model, the prune `--settings` hook)
/// can't drift between the two paths. `sys_leadin` is the optional first sentence of the
/// `--append-system-prompt` block (the only intentional difference); `resident`
/// adds `--input-format stream-json` before `--output-format` so claude reads
/// turns from the held-open stdin. `chrome` is the user's Claude in Chrome
/// "enabled by default" setting (`claude_in_chrome_enabled`).
#[allow(clippy::too_many_arguments)]
fn claude_flag_args(
    sys_leadin: Option<&str>,
    chat_id: &str,
    project_path: Option<&str>,
    worktree: bool,
    agent_session_id: Option<&str>,
    is_sandboxed: bool,
    model: Option<&str>,
    session_name: Option<&str>,
    resident: bool,
    chrome: bool,
) -> Vec<ClaudeArg> {
    let mut args: Vec<ClaudeArg> = Vec::new();

    // Register the chat's title as this process's peer-addressable name
    // (`~/.claude/sessions/<pid>.json`), so another agent on the machine can
    // `SendMessage` it by chat title instead of re-deriving its context.
    // Claude resolves a collision with another live session itself
    // (`session_name_collision`) by taking a suffixed variant, so we pass the
    // title as-is. Sanitizing is `session_name::from_title`'s job; the value is
    // one argv element (POSIX path single-quotes it), so a title with spaces or
    // quotes can't split into flags.
    if let Some(name) = session_name {
        args.push(ClaudeArg::Lit("--name"));
        args.push(ClaudeArg::Val(name.to_string()));
    }

    // First turn / fresh session: pass no session flag — claude generates a
    // session id and emits it in the `system/init` stdout frame, which we
    // harvest and persist to `chats.agent_session_id`. Subsequent (re)starts
    // resume that id. Pre-migration rows were backfilled with
    // `agent_session_id = id::text`, so existing chats keep using the same id
    // claude already knows about.
    if let Some(sid) = agent_session_id {
        args.push(ClaudeArg::Lit("--resume"));
        args.push(ClaudeArg::Val(sid.to_string()));
    }

    let mut sys = sys_leadin.map(|l| format!("{l}\n\n")).unwrap_or_default();
    // Worktree containment: when `--worktree` is on and the project path is
    // known, resolve the absolute worktree path so the capability block can
    // carry the "stay inside" rule (wording in `adapter::worktree_instructions`).
    let mut worktree_info: Option<WorktreeInstructions> = None;
    if worktree {
        // Use the chat_id prefix so the worktree directory name stays short.
        let worktree_name: String = chat_id.chars().take(8).collect();
        args.push(ClaudeArg::Lit("--worktree"));
        args.push(ClaudeArg::Val(worktree_name.clone()));
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
    // Must stay byte-stable across respawns — see `AgentAdapter::turn_time_line`.
    sys.push_str(&agent_capabilities_instructions(worktree_info.as_ref()));
    sys.push_str("\n\n");
    sys.push_str(PRUNE_CONTEXT_INSTRUCTION);
    sys.push(' ');
    sys.push_str(PRUNE_SUMMARY_MOVE_INSTRUCTION);
    args.push(ClaudeArg::Lit("--append-system-prompt"));
    args.push(ClaudeArg::Val(sys));
    // Resident: read user/control turns from the held-open stdin pipe, and echo
    // each one back when consumed (`UserTurnConsumed` drives the session FSM).
    if resident {
        args.push(ClaudeArg::Lit("--input-format"));
        args.push(ClaudeArg::Lit("stream-json"));
        args.push(ClaudeArg::Lit("--replay-user-messages"));
    }
    args.push(ClaudeArg::Lit("--print"));
    args.push(ClaudeArg::Lit("--verbose"));
    args.push(ClaudeArg::Lit("--output-format"));
    args.push(ClaudeArg::Lit("stream-json"));
    // Sender's `machine_users.is_sandboxed`. Non-sandboxed = bypass permission
    // gating; sandboxed = claude's default permission mode auto-denies tools
    // in `--print`, which is the actual sandboxing mechanism. The default mode
    // still allows `SendMessage`, which would let an invitee's session drive
    // the owner's bypass-mode sessions on this machine.
    let disallowed_tools = if is_sandboxed {
        "AskUserQuestion SendMessage ListAgents"
    } else {
        "AskUserQuestion"
    };
    args.push(ClaudeArg::Lit("--disallowedTools"));
    args.push(ClaudeArg::Val(disallowed_tools.to_string()));
    if !is_sandboxed {
        args.push(ClaudeArg::Lit("--dangerously-skip-permissions"));
    }
    // The browser is signed into the owner's accounts, so an invitee never gets it.
    if chrome && !is_sandboxed {
        args.push(ClaudeArg::Lit("--chrome"));
    }
    // Verbatim pass-through of `chats.model` (migration 0035). Empty / blank
    // values are already filtered to `None` at the construction site in
    // `main.rs`, so any `Some` here is a non-empty model name the user picked in
    // the composer's agent roster. We don't validate the model name — claude
    // prints a clean error if it doesn't recognize it, and the closed set drifts
    // per-release.
    if let Some(model) = model {
        args.push(ClaudeArg::Lit("--model"));
        args.push(ClaudeArg::Val(model.to_string()));
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
    args.push(ClaudeArg::Lit("--settings"));
    args.push(ClaudeArg::Val(settings.to_string()));

    args
}

/// Render the flag tokens into the historical POSIX shell string: a leading
/// space before each token, `Lit` verbatim (unquoted), `Val` single-quoted via
/// `shell_escape`. Byte-identical to the pre-refactor `append_claude_flags`.
fn push_posix_flags(cmd: &mut String, args: &[ClaudeArg]) {
    for a in args {
        cmd.push(' ');
        match a {
            ClaudeArg::Lit(s) => cmd.push_str(s),
            ClaudeArg::Val(s) => cmd.push_str(&shell_escape(s)),
        }
    }
}

/// Render the flag tokens into clean argv (one OS arg per token, no escaping)
/// for the Windows-native shell-free spawn.
fn flags_to_argv(args: Vec<ClaudeArg>) -> Vec<String> {
    args.into_iter()
        .map(|a| match a {
            ClaudeArg::Lit(s) => s.to_string(),
            ClaudeArg::Val(s) => s,
        })
        .collect()
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
        let args = claude_flag_args(
            Some(ONESHOT_SYS_LEADIN),
            ctx.chat_id,
            ctx.project_path,
            ctx.worktree,
            ctx.agent_session_id,
            ctx.is_sandboxed,
            ctx.model,
            /* session_name = */ None,
            /* resident = */ false,
            claude_in_chrome_enabled(),
        );
        push_posix_flags(&mut claude_cmd, &args);
        Ok(claude_cmd)
    }

    fn prepare_native(&mut self, ctx: &TurnContext<'_>) -> Option<NativeSpawn> {
        let args = claude_flag_args(
            Some(ONESHOT_SYS_LEADIN),
            ctx.chat_id,
            ctx.project_path,
            ctx.worktree,
            ctx.agent_session_id,
            ctx.is_sandboxed,
            ctx.model,
            /* session_name = */ None,
            false,
            claude_in_chrome_enabled(),
        );
        Some(NativeSpawn {
            program: "claude".to_string(),
            args: flags_to_argv(args),
            cwd: ctx.project_path.map(std::path::PathBuf::from),
            prompt_stdin_file: Some(ctx.prompt_file.to_path_buf()),
        })
    }

    fn is_resident(&self) -> bool {
        true
    }

    fn is_command_turn(&self, prompt: &str) -> bool {
        crate::adapter::slash_command_name(prompt).is_some()
    }

    fn names_session(&self) -> bool {
        true
    }

    fn can_interrupt(&self) -> bool {
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
        // `--input-format stream-json` flag (added by `claude_flag_args` with
        // `resident=true`) tells it to read user/control frames from stdin.
        claude_cmd.push_str("claude");
        let args = claude_flag_args(
            None,
            ctx.chat_id,
            ctx.project_path,
            ctx.worktree,
            ctx.agent_session_id,
            ctx.is_sandboxed,
            ctx.model,
            ctx.session_name,
            /* resident = */ true,
            claude_in_chrome_enabled(),
        );
        push_posix_flags(&mut claude_cmd, &args);
        Ok(claude_cmd)
    }

    fn prepare_session_native(
        &mut self,
        ctx: &crate::adapter::SessionContext<'_>,
    ) -> Option<NativeSpawn> {
        let args = claude_flag_args(
            None,
            ctx.chat_id,
            ctx.project_path,
            ctx.worktree,
            ctx.agent_session_id,
            ctx.is_sandboxed,
            ctx.model,
            ctx.session_name,
            true,
            claude_in_chrome_enabled(),
        );
        Some(NativeSpawn {
            program: "claude".to_string(),
            args: flags_to_argv(args),
            cwd: ctx.project_path.map(std::path::PathBuf::from),
            prompt_stdin_file: None,
        })
    }

    fn encode_user_turn(&mut self, uuid: &str, text: &str) -> Option<String> {
        // Build via serde so the user's text (quotes / newlines / control chars)
        // is encoded safely — never string concat. Newline-terminated: claude's
        // stream-json input is newline-delimited frames.
        let frame = serde_json::json!({
            "type": "user",
            "uuid": uuid,
            "message": {
                "role": "user",
                "content": [{ "type": "text", "text": text }],
            },
        });
        Some(format!("{}\n", frame))
    }

    fn encode_interrupt(&mut self, request_id: &str) -> Option<String> {
        let frame = serde_json::json!({
            "type": "control_request",
            "request_id": request_id,
            "request": { "subtype": "interrupt" },
        });
        Some(format!("{}\n", frame))
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
                    // Resident model: a background task / Monitor / backgrounded
                    // subagent was armed. The `task_id` lets the Supervisor track
                    // it in `live_tasks` (Waiting state) until its terminal
                    // `task_notification`; the rest of the frame names it for the
                    // client (`chats.live_tasks`).
                    out.extend(parse_task_started(&line));
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
                // `--replay-user-messages` echo of a stdin frame we wrote (claude
                // has consumed that turn; the user's own message row already
                // exists) or of a peer session's `SendMessage`.
                if line.contains("\"isReplay\":true") {
                    out.extend(parse_replay_echo(&line));
                }
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
            } else if line.contains("\"type\":\"control_response\"") {
                // Ack of our interrupt `control_request`. An error ack is no
                // ack: `/stop`'s timeout then hard-aborts a still-running turn.
                if line.contains("\"subtype\":\"success\"") {
                    out.extend(parse_control_response_id(&line).map(AgentEvent::InterruptAcked));
                }
                skip = true;
            } else if line.contains("\"type\":\"command_lifecycle\"") {
                // Queue bookkeeping emitted only under `--replay-user-messages`.
                if line.contains("\"state\":\"started\"") {
                    out.extend(parse_started_command_uuid(&line).map(AgentEvent::TurnStarted));
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

fn parse_started_command_uuid(line: &str) -> Option<String> {
    #[derive(serde::Deserialize)]
    struct Lifecycle {
        command_uuid: String,
    }
    serde_json::from_str::<Lifecycle>(line)
        .ok()
        .map(|l| l.command_uuid)
}

fn parse_control_response_id(line: &str) -> Option<String> {
    #[derive(serde::Deserialize)]
    struct Response {
        request_id: String,
    }
    #[derive(serde::Deserialize)]
    struct ControlResponse {
        response: Response,
    }
    serde_json::from_str::<ControlResponse>(line)
        .ok()
        .map(|r| r.response.request_id)
}

fn parse_replay_echo(line: &str) -> Option<AgentEvent> {
    // `origin` stays untyped so an unexpected shape can't cost us our own echo.
    #[derive(serde::Deserialize)]
    struct Replay {
        uuid: String,
        #[serde(default)]
        origin: serde_json::Value,
    }
    let r = serde_json::from_str::<Replay>(line).ok()?;
    if r.origin["kind"] != "peer" {
        return Some(AgentEvent::UserTurnConsumed(r.uuid));
    }
    let field = |k: &str| r.origin[k].as_str().unwrap_or_default().to_string();
    Some(AgentEvent::PeerTurnConsumed {
        name: field("name"),
        body: field("body"),
    })
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

/// Reads a `task_started` system frame into the event the Supervisor tracks.
/// Everything but `task_id` is optional: a frame shape drift must still arm the
/// task (a never-inserted task wedges the Waiting state), it may just cost the
/// client its label. Verbatim shapes in `tmp/monitor_raw.jsonl` (backgrounded
/// Bash / Monitor: `task_type: "local_bash"`, no `subagent_type`) and
/// `tmp/agent_task_raw.jsonl` (backgrounded subagent: `task_type:
/// "local_agent"` + `subagent_type`).
fn parse_task_started(line: &str) -> Option<AgentEvent> {
    #[derive(serde::Deserialize)]
    struct Frame {
        task_id: String,
        #[serde(default)]
        description: Option<String>,
        #[serde(default)]
        task_type: Option<String>,
        #[serde(default)]
        subagent_type: Option<String>,
    }
    match serde_json::from_str::<Frame>(line) {
        Ok(f) => Some(AgentEvent::TaskStarted {
            id: f.task_id,
            description: f.description.unwrap_or_default(),
            task_type: f.task_type.unwrap_or_default(),
            subagent_type: f.subagent_type.filter(|s| !s.is_empty()),
        }),
        Err(e) => {
            debug!("failed to parse task_started frame: {}", e);
            None
        }
    }
}

/// Reads `task_id` from a terminal `task_notification` / `task_updated` system
/// frame. Narrow Deserialize struct so serde skips the rest of the (small)
/// frame. `None` on parse failure or a missing id — the Supervisor treats a
/// dropped finish as "task never tracked", which is safe (a never-inserted task
/// can't wedge the Waiting state).
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

#[derive(Deserialize)]
struct ClaudeGlobalConfig {
    #[serde(default, rename = "oauthAccount")]
    oauth_account: Option<serde_json::Value>,
    #[serde(default, rename = "claudeInChromeDefaultEnabled")]
    chrome_enabled_by_default: bool,
}

fn read_claude_global_config() -> Option<ClaudeGlobalConfig> {
    let path = crate::home_dir_opt()?.join(".claude.json");
    let bytes = std::fs::read(&path).ok()?;
    serde_json::from_slice(&bytes)
        .map_err(|e| debug!(error = %e, path = %path.display(), "claude config not parseable"))
        .ok()
}

/// claude code stores OAuth state in `~/.claude.json` under `oauthAccount` —
/// cross-platform, regardless of where the actual token lives (macOS Keychain,
/// `~/.claude/.credentials.json` on Linux). Fallback covers older / non-Keychain
/// installs.
fn is_authenticated() -> bool {
    if read_claude_global_config().is_some_and(|c| c.oauth_account.is_some()) {
        return true;
    }
    crate::home_dir_opt()
        .is_some_and(|home| file_nonempty(&home.join(".claude").join(".credentials.json")))
}

/// Interactive `claude` loads the Claude in Chrome tools whenever the user set
/// them "enabled by default" (`/chrome`), but `--print` ignores that setting and
/// loads them only with an explicit `--chrome`. Read per spawn, so toggling it
/// takes effect on the chat's next fresh process.
fn claude_in_chrome_enabled() -> bool {
    read_claude_global_config().is_some_and(|c| c.chrome_enabled_by_default)
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
        // and re-import their messages on every future import. The decoded
        // path is host-native, so match both separators (Windows uses `\`).
        if decoded.contains("/.claude/worktrees/") || decoded.contains("\\.claude\\worktrees\\") {
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
    Some(crate::home_dir_opt()?.join(".claude").join("projects"))
}

/// Reverse the dirname encoding claude-code uses for cwd (every path separator
/// replaced with `-`). Naive `replace('-', "/")` is lossy on segments that
/// contain a literal `-` (e.g. `/Users/me/projects/berkshire-hathaway`
/// round-trips to `/Users/me/projects/berkshire/hathaway`). We disambiguate by
/// walking the real filesystem: at each step pick the longest prefix of
/// remaining segments that resolves to an existing directory.
///
/// Two host encodings are handled, chosen by the shape of the name:
///   - Unix:    `/Users/me/proj`   → `-Users-me-proj`   (leading `/` → leading `-`)
///   - Windows: `C:\Users\me\proj` → `C--Users-me-proj` (the drive `:` and every
///     `\` both map to `-`, so the `C:\` prefix becomes `C--`)
///
/// The greedy longest-prefix walk is identical for both; only the root and the
/// list of segments differ.
///
/// Returns `None` when no walk reaches the leaf — usually because the project
/// was deleted or moved after the transcript was written. Caller skips those.
fn decode_project_dir(name: &str) -> Option<String> {
    let parts: Vec<&str> = name.split('-').collect();

    let (mut current, segments): (std::path::PathBuf, &[&str]) = if parts.first() == Some(&"") {
        // Unix: the split before the leading '/' yields an empty first element.
        (std::path::PathBuf::from("/"), &parts[1..])
    } else if parts.len() >= 2
        && parts[1].is_empty()
        && parts[0].len() == 1
        && parts[0].as_bytes()[0].is_ascii_alphabetic()
    {
        // Windows: `C`, `` (from the doubled dash of `:\`), then the segments.
        (
            std::path::PathBuf::from(format!("{}:\\", parts[0])),
            &parts[2..],
        )
    } else {
        return None;
    };

    if segments.is_empty() {
        return None;
    }
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
            agent_kind: DESCRIPTOR.kind,
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
// the assistant's `tool_use` by `tool_use_id`. A prune sets the `tool_use` `input`
// to `pruned_input` and the paired `tool_result` `content` to `[pruned] <summary>`,
// and swaps that summary in the requesting `prune-context` command for
// `SUMMARY_MOVED_MARKER`, so the summary sits in context once, next to the call it
// describes. Every line + id/threading field is preserved. Wired in via
// `PRUNE_OPS` (→ `AdapterDescriptor::prune`).

use std::collections::HashMap;
use std::ops::Range;

use crate::prune::{
    BatchCollision, ChosenTargets, ParsedLine, PruneStats, PruneTarget, PRUNED_PLACEHOLDER,
};

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

/// The `message.content` blocks of a transcript entry of `entry_type`
/// (`assistant`/`user`); empty for any other entry.
fn claude_content_blocks<'a>(
    entry: &'a serde_json::Value,
    entry_type: &str,
) -> &'a [serde_json::Value] {
    if entry.get("type").and_then(|t| t.as_str()) != Some(entry_type) {
        return &[];
    }
    entry
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_array())
        .map_or(&[], Vec::as_slice)
}

/// Pass-1 matched collector shared by [`count_matches`] and
/// [`prune_batch_jsonl`], so the count and apply paths can't drift. Walks
/// assistant `tool_use` blocks via [`tool_use_matches`], pushing matching ids in
/// DOCUMENT ORDER.
fn claude_collect_matched(
    entry: &serde_json::Value,
    tool_name: &str,
    needle: &str,
    matched: &mut Vec<String>,
) {
    for block in claude_content_blocks(entry, "assistant") {
        if tool_use_matches(block, tool_name, needle) {
            if let Some(id) = block.get("id").and_then(|i| i.as_str()) {
                matched.push(id.to_string());
            }
        }
    }
}

/// Top-level `sourceToolUseID` linking a Skill / slash-command expansion BODY frame
/// (a `user`/`isMeta` entry, see [`blank_claude_entry`]) back to its `tool_use`. It's
/// at the entry top level, NOT in a content block. One accessor so the wire key isn't
/// retyped in [`claude_collect_pruned`] + [`blank_claude_entry`].
fn claude_entry_source_id(entry: &serde_json::Value) -> Option<&str> {
    entry.get("sourceToolUseID").and_then(|i| i.as_str())
}

/// Is this claude output already pruned? The bare placeholder (older transcripts)
/// or the placeholder followed by a moved summary.
fn is_pruned_text(text: &str) -> bool {
    text.strip_prefix(PRUNED_PLACEHOLDER)
        .is_some_and(|rest| rest.is_empty() || rest.starts_with(' '))
}

/// Does an output field still hold content to prune?
fn holds_unpruned_output(field: Option<&serde_json::Value>) -> bool {
    field.is_some_and(|v| !v.as_str().is_some_and(is_pruned_text))
}

/// Already-pruned collector. Marks a `tool_use` pruned when either its
/// `tool_result` content OR its expansion-body text is — the latter keeps re-prune
/// a no-op even if the tiny ack is absent.
fn claude_collect_pruned(entry: &serde_json::Value, already_pruned: &mut HashSet<String>) {
    let source_id = claude_entry_source_id(entry);
    let is_pruned = |block: &serde_json::Value, key| {
        block
            .get(key)
            .and_then(|v| v.as_str())
            .is_some_and(is_pruned_text)
    };
    for block in claude_content_blocks(entry, "user") {
        let pruned_id = match block.get("type").and_then(|t| t.as_str()) {
            Some("tool_result") if is_pruned(block, "content") => {
                block.get("tool_use_id").and_then(|i| i.as_str())
            }
            Some("text") if is_pruned(block, "text") => source_id,
            _ => None,
        };
        if let Some(id) = pruned_id {
            already_pruned.insert(id.to_string());
        }
    }
}

/// Ids a prune may still pick: the call has a `tool_result` and it isn't pruned
/// yet. A call killed mid-tool has no result: picking it would blank nothing, and
/// once its input is `pruned_args` that glob would match it on every later prune.
fn claude_prunable_ids(parsed: &[ParsedLine]) -> HashSet<String> {
    let mut answered = HashSet::new();
    let mut pruned = HashSet::new();
    for entry in crate::prune::parsed_entries(parsed) {
        for block in claude_content_blocks(entry, "user") {
            if block.get("type").and_then(|t| t.as_str()) == Some("tool_result") {
                if let Some(id) = block.get("tool_use_id").and_then(|i| i.as_str()) {
                    answered.insert(id.to_string());
                }
            }
        }
        claude_collect_pruned(entry, &mut pruned);
    }
    answered.retain(|id| !pruned.contains(id));
    answered
}

/// Read-only pre-scan: count eligible matches (control-side, before aborting).
fn count_matches(path: &Path, tool_name: &str, needle: &str) -> std::io::Result<usize> {
    let parsed = crate::prune::read_parsed_lines(path)?;
    let prunable = claude_prunable_ids(&parsed);
    let mut matched = Vec::new();
    for entry in crate::prune::parsed_entries(&parsed) {
        claude_collect_matched(entry, tool_name, needle, &mut matched);
    }
    Ok(matched.iter().filter(|id| prunable.contains(*id)).count())
}

/// The `input` a pruned call keeps: its `--args` glob, so the agent can tell the
/// pruned calls apart. The no-args selector (empty needle) keeps `{}`.
fn pruned_input(needle: &str) -> serde_json::Value {
    if needle.is_empty() {
        serde_json::json!({})
    } else {
        serde_json::json!({ "pruned_args": needle })
    }
}

/// The `content` a pruned `tool_result` keeps: the placeholder, plus the target's
/// summary when it moved out of the prune command.
fn pruned_result(moved_summary: Option<&str>) -> String {
    match moved_summary {
        Some(s) => format!("{PRUNED_PLACEHOLDER} {s}"),
        None => PRUNED_PLACEHOLDER.to_string(),
    }
}

/// Left in the `prune-context` command where a moved `--summary` value was.
const SUMMARY_MOVED_MARKER: &str = "[was here, now written into the pruned tool result]";

fn is_shell_word_break(c: char) -> bool {
    c.is_whitespace() || matches!(c, ';' | '|' | '&')
}

/// The words of a shell `command` with their raw byte spans, quotes and backslash
/// escapes resolved. Expansions (`$VAR`, `$(...)`) stay literal.
fn shell_words(command: &str) -> Vec<(String, Range<usize>)> {
    let mut words = Vec::new();
    let mut chars = command.char_indices().peekable();
    loop {
        while chars.next_if(|&(_, c)| is_shell_word_break(c)).is_some() {}
        let Some(&(start, _)) = chars.peek() else {
            break;
        };
        let mut word = String::new();
        while let Some((_, c)) = chars.next_if(|&(_, c)| !is_shell_word_break(c)) {
            match c {
                '\'' => word.extend(chars.by_ref().map(|(_, c)| c).take_while(|&c| c != '\'')),
                '"' => {
                    while let Some((_, c)) = chars.next() {
                        match c {
                            '"' => break,
                            '\\' => {
                                match chars.next_if(|&(_, c)| matches!(c, '"' | '\\' | '$' | '`')) {
                                    Some((_, escaped)) => word.push(escaped),
                                    None => word.push('\\'),
                                }
                            }
                            _ => word.push(c),
                        }
                    }
                }
                '\\' => word.extend(chars.next().map(|(_, c)| c).filter(|&c| c != '\n')),
                _ => word.push(c),
            }
        }
        let end = chars.peek().map_or(command.len(), |&(i, _)| i);
        words.push((word, start..end));
    }
    words
}

/// The targets of a `prune-context` command, read from its shell words the way
/// `parse_prune_args` reads argv, each with the raw span of its summary value.
fn prune_command_items(command: &str) -> Vec<(PruneTarget, Range<usize>)> {
    const VALUE_FLAGS: [&str; 5] = [
        "--tool-name",
        "--args",
        "--summary",
        "--reason",
        "--chat-id",
    ];
    let words = shell_words(command);
    let Some(subcommand) = words.iter().position(|(word, _)| word == "prune-context") else {
        return Vec::new();
    };
    let mut items = Vec::new();
    let mut tool_name = String::new();
    let mut needle = None;
    let mut it = words[subcommand + 1..].iter();
    while let Some((word, span)) = it.next() {
        let (flag, (value, value_span)) = match word.split_once('=') {
            Some((flag, value)) if command[span.clone()].starts_with(&format!("{flag}=")) => (
                flag,
                (value.to_string(), span.start + flag.len() + 1..span.end),
            ),
            _ if VALUE_FLAGS.contains(&word.as_str()) => match it.next() {
                Some(value) => (word.as_str(), value.clone()),
                None => break,
            },
            _ => continue,
        };
        match flag {
            "--tool-name" => tool_name = value,
            "--args" => needle = Some(value),
            "--summary" | "--reason" => {
                if let Some(needle) = needle.take() {
                    let tool_name = std::mem::take(&mut tool_name);
                    items.push((PruneTarget::new(&tool_name, &needle, &value), value_span));
                }
            }
            _ => {}
        }
    }
    items
}

/// `command` with the summary value of its first item equal to `target` replaced
/// by [`SUMMARY_MOVED_MARKER`]. `None` when no item matches, e.g. the summary came
/// from a shell variable.
fn replace_summary_flag_value(command: &str, target: &PruneTarget) -> Option<String> {
    let (_, span) = prune_command_items(command)
        .into_iter()
        .find(|(item, _)| item == target)?;
    Some(format!(
        "{}{SUMMARY_MOVED_MARKER}{}",
        &command[..span.start],
        &command[span.end..]
    ))
}

/// Moves `target`'s summary out of the first `prune-context` command leaf of
/// `input` that carries `target`.
fn move_summary_out_of_prune_call(input: &mut serde_json::Value, target: &PruneTarget) -> bool {
    match input {
        serde_json::Value::String(leaf) if crate::prune::leaf_is_prune_context_command(leaf) => {
            match replace_summary_flag_value(leaf, target) {
                Some(rewritten) => {
                    *leaf = rewritten;
                    true
                }
                None => false,
            }
        }
        serde_json::Value::Array(items) => items
            .iter_mut()
            .any(|v| move_summary_out_of_prune_call(v, target)),
        serde_json::Value::Object(map) => map
            .values_mut()
            .any(|v| move_summary_out_of_prune_call(v, target)),
        _ => false,
    }
}

/// Where a batch's summaries go.
#[derive(Default)]
struct SummaryMoves {
    /// Rewritten `input` of each `prune-context` call a summary moved out of, by
    /// tool_use id.
    prune_call_inputs: HashMap<String, serde_json::Value>,
    /// Targets (indexes into the batch) whose summary moved. Any other picked
    /// target's result gets the bare placeholder, so a summary never sits in
    /// context twice.
    moved: HashSet<usize>,
}

/// For each picked target, in batch order, move its summary out of the most
/// recent `prune-context` call that carries the same target (tool name, args and
/// summary). The apply cue carries no tool_use id, so the call is found by content.
fn plan_summary_moves(
    parsed: &[ParsedLine],
    targets: &[PruneTarget],
    chosen: &ChosenTargets,
) -> SummaryMoves {
    let prune_calls: Vec<(&str, &serde_json::Value)> = crate::prune::parsed_entries(parsed)
        .flat_map(|entry| claude_content_blocks(entry, "assistant"))
        .filter_map(|block| {
            let input = block
                .get("input")
                .filter(|input| crate::prune::value_is_prune_context_call(input))?;
            Some((block.get("id")?.as_str()?, input))
        })
        .collect();
    let mut picked: Vec<usize> = chosen.values().copied().collect();
    picked.sort_unstable();

    let mut moves = SummaryMoves::default();
    for idx in picked {
        let target = &targets[idx];
        if target.summary.is_empty() {
            continue;
        }
        for &(id, input) in prune_calls.iter().rev() {
            let mut rewritten = moves.prune_call_inputs.get(id).unwrap_or(input).clone();
            if move_summary_out_of_prune_call(&mut rewritten, target) {
                moves.prune_call_inputs.insert(id.to_string(), rewritten);
                moves.moved.insert(idx);
                break;
            }
        }
    }
    moves
}

/// Serialized bytes freed by setting `obj[key]` to `value` (negative when it grows).
fn replace_field(
    obj: &mut serde_json::Map<String, serde_json::Value>,
    key: &str,
    value: serde_json::Value,
) -> isize {
    let new_len = value.to_string().len() as isize;
    let old_len = obj
        .insert(key.to_string(), value)
        .map_or(0, |old| old.to_string().len() as isize);
    old_len - new_len
}

/// Pass-2 helper: rewrite one parsed claude transcript entry in place:
///   - assistant: each picked `tool_use` → `input` = [`pruned_input`]; each
///     `prune-context` call in `moves` → its rewritten `input`.
///   - user: each picked `tool_result` → `content` = [`pruned_result`], the id
///     recorded in `outputs_blanked` (drives the user-facing `results_blanked`).
///   - user EXPANSION BODY: a Skill/command expansion's real payload rides in a
///     SEPARATE `user`/`isMeta` entry (a `text` block, NOT a `tool_result`) threaded
///     by [`claude_entry_source_id`]; its `tool_result` is just a tiny ack, which
///     gets the summary. When that source id is picked we blank the text to the bare
///     placeholder and record the SAME id in `outputs_blanked` — the set dedups it
///     against the ack ("1 output") while the freed bytes pick up the body. Without
///     this a `--tool-name Skill` prune freed ~nothing (a 570KB claude-api body once
///     survived a prune that freed 19 bytes).
///
/// Cross-line by design: the tool_use lives on the assistant line, its tool_result
/// and any expansion body on later user lines; the driver runs this per-entry.
/// Returns `None` when the entry was left untouched, else the serialized bytes
/// freed (negative when it grew). Already-pruned outputs are left as they are, so a
/// re-prune stays byte-stable.
fn blank_claude_entry(
    entry: &mut serde_json::Value,
    chosen: &ChosenTargets,
    targets: &[PruneTarget],
    moves: &SummaryMoves,
    outputs_blanked: &mut HashSet<String>,
) -> Option<isize> {
    let entry_type = entry.get("type").and_then(|t| t.as_str());
    let is_assistant = entry_type == Some("assistant");
    let is_user = entry_type == Some("user");
    if !is_assistant && !is_user {
        return None;
    }

    // Resolved before the mutable content borrow; `Some` only for a user body whose
    // source tool_use is a prune target.
    let body_source_id = is_user
        .then(|| claude_entry_source_id(entry))
        .flatten()
        .filter(|id| chosen.contains_key(*id))
        .map(str::to_string);

    let blocks = entry
        .get_mut("message")
        .and_then(|m| m.get_mut("content"))
        .and_then(|c| c.as_array_mut())?;

    let mut freed = 0isize;
    let mut changed = false;
    for block in blocks.iter_mut() {
        let Some(obj) = block.as_object_mut() else {
            continue;
        };
        let block_type = obj.get("type").and_then(|t| t.as_str());
        if let Some(source_id) = &body_source_id {
            if block_type == Some("text") {
                if holds_unpruned_output(obj.get("text")) {
                    freed += replace_field(obj, "text", PRUNED_PLACEHOLDER.into());
                    changed = true;
                    outputs_blanked.insert(source_id.clone());
                }
                continue;
            }
        }
        if is_assistant && block_type == Some("tool_use") {
            let Some(id) = obj.get("id").and_then(|i| i.as_str()) else {
                continue;
            };
            let input = match (chosen.get(id), moves.prune_call_inputs.get(id)) {
                (Some(&idx), _) => pruned_input(&targets[idx].needle),
                (None, Some(rewritten)) => rewritten.clone(),
                (None, None) => continue,
            };
            if obj.get("input") != Some(&input) {
                freed += replace_field(obj, "input", input);
                changed = true;
            }
        } else if is_user && block_type == Some("tool_result") {
            let Some((id, &idx)) = obj
                .get("tool_use_id")
                .and_then(|i| i.as_str())
                .and_then(|id| chosen.get_key_value(id))
            else {
                continue;
            };
            if holds_unpruned_output(obj.get("content")) {
                let summary = moves
                    .moved
                    .contains(&idx)
                    .then(|| targets[idx].summary.as_str());
                freed += replace_field(obj, "content", pruned_result(summary).into());
                changed = true;
                outputs_blanked.insert(id.clone());
            }
        }
    }

    changed.then_some(freed)
}

/// Single-target, summary-less [`prune_batch_jsonl`], a test shorthand.
#[cfg(test)]
fn prune_jsonl(path: &Path, tool_name: &str, needle: &str) -> std::io::Result<PruneStats> {
    prune_batch_jsonl(path, &[PruneTarget::new(tool_name, needle, "")])
}

/// Batch entry point behind `PRUNE_OPS::prune_batch`, in ONE read/write: pick the
/// most recent match of each target among [`claude_prunable_ids`] (a target whose
/// match an earlier target with a different selector picked is a miss — its
/// summary would land on an output it wasn't written about), move the summaries
/// out of the prune command ([`plan_summary_moves`]), and rewrite.
/// `freed_bytes` is the net serialized shrink of all edits, floored at 0.
fn prune_batch_jsonl(path: &Path, targets: &[PruneTarget]) -> std::io::Result<PruneStats> {
    let parsed = crate::prune::read_parsed_lines(path)?;
    let prunable = claude_prunable_ids(&parsed);
    let chosen = crate::prune::select_batch_last_only(
        &parsed,
        targets,
        BatchCollision::SameSelectorOnly,
        |target, entry, matched| {
            claude_collect_matched(entry, &target.tool_name, &target.needle, matched)
        },
        |id| prunable.contains(id),
    );
    let moves = plan_summary_moves(&parsed, targets, &chosen);
    let (stats, _final_entries) =
        crate::prune::blank_pass(path, parsed, &chosen, |entry, chosen, outputs_blanked| {
            blank_claude_entry(entry, chosen, targets, &moves, outputs_blanked)
        })?;
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

    /// Minimal resident `SessionContext` (mirrors `TurnContext::for_test`). Vary
    /// a field via struct-update: `SessionContext { worktree: true, ..session_ctx() }`.
    fn session_ctx() -> crate::adapter::SessionContext<'static> {
        crate::adapter::SessionContext {
            session_name: None,
            chat_id: "abcdef012345-6789",
            project_path: Some("/tmp/proj"),
            worktree: false,
            agent_session_id: Some("sid-9"),
            is_sandboxed: false,
            model: Some("opus"),
        }
    }

    #[test]
    fn init_frame_harvests_session_id_and_drops_frame() {
        let mut a = ClaudeAdapter::new();
        let line = r#"{"type":"system","subtype":"init","session_id":"abc-123","model":"sonnet"}"#;
        let events = run(&mut a, line);
        assert_eq!(events, vec!["SessionIdHarvested(abc-123)"]);
    }

    /// Real claude 2.1.265 shape (tmp/claude_stdin_probe.py): the echo carries
    /// the uuid we stamped and must not become a message row.
    #[test]
    fn replay_echo_emits_user_turn_consumed_and_drops_frame() {
        let mut a = ClaudeAdapter::new();
        let line = r#"{"type":"user","message":{"role":"user","content":[{"type":"text","text":"Also, reply with BANANA."}]},"session_id":"s1","parent_tool_use_id":null,"uuid":"11111111-2222-3333-4444-555555555555","timestamp":"2026-09-10T10:00:00.000Z","isReplay":true}"#;
        assert_eq!(
            run(&mut a, line),
            vec!["UserTurnConsumed(11111111-2222-3333-4444-555555555555)"]
        );
    }

    #[test]
    fn command_lifecycle_frame_is_dropped() {
        let mut a = ClaudeAdapter::new();
        let line = r#"{"type":"command_lifecycle","command_uuid":"01a08b19-5a10-7bf2-a142-2dd72f212045","state":"queued","uuid":"20832165-4c87-43cf-b694-bb2c0b58bce5","session_id":"s1"}"#;
        assert!(run(&mut a, line).is_empty());
    }

    /// Real claude 2.1.265 (tmp/cron_probe.log): a `CronCreate` prompt firing
    /// on an idle resident is announced only by this frame; no replay echo.
    #[test]
    fn command_lifecycle_started_emits_turn_started() {
        let mut a = ClaudeAdapter::new();
        let line = r#"{"type":"command_lifecycle","command_uuid":"b3bed1f2-4e92-48a7-8f16-f6481fb94f1a","state":"started","uuid":"420478f0-552a-444b-9cda-d65c04cbd262","session_id":"47264b88-bc6c-42b8-aef8-e69544efdf36"}"#;
        assert_eq!(
            run(&mut a, line),
            vec!["TurnStarted(b3bed1f2-4e92-48a7-8f16-f6481fb94f1a)"]
        );
    }

    #[test]
    fn task_started_frame_emits_taskstarted_and_drops_frame() {
        let mut a = ClaudeAdapter::new();
        let line = r#"{"type":"system","subtype":"task_started","task_id":"t-7","tool_use_id":"tu9","task_type":"local_bash","description":"monitor"}"#;
        assert_eq!(run(&mut a, line), vec!["TaskStarted(t-7)"]);
    }

    /// The `task_started` frame carries everything the client needs to NAME the
    /// task; the adapter must keep all of it. Verbatim shapes from
    /// `tmp/monitor_raw.jsonl` (backgrounded Bash) and `tmp/agent_task_raw.jsonl`
    /// (backgrounded subagent).
    #[test]
    fn task_started_carries_description_kind_and_subagent_type() {
        let mut a = ClaudeAdapter::new();
        let bash = r#"{"type":"system","subtype":"task_started","task_id":"buukg9452","tool_use_id":"tu9","description":"npm test -- --watch","task_type":"local_bash"}"#;
        match a.handle_line(bash.to_string()).into_iter().next() {
            Some(AgentEvent::TaskStarted {
                id,
                description,
                task_type,
                subagent_type,
            }) => {
                assert_eq!(id, "buukg9452");
                assert_eq!(description, "npm test -- --watch");
                assert_eq!(task_type, "local_bash");
                assert_eq!(subagent_type, None);
            }
            other => panic!("expected TaskStarted, got {other:?}"),
        }

        let subagent = r#"{"type":"system","subtype":"task_started","task_id":"a3bc46a79","description":"count tmp files","subagent_type":"Explore","is_backgrounded":true,"spawn_depth":1,"task_type":"local_agent","prompt":"Count the files"}"#;
        match a.handle_line(subagent.to_string()).into_iter().next() {
            Some(AgentEvent::TaskStarted {
                task_type,
                subagent_type,
                ..
            }) => {
                assert_eq!(task_type, "local_agent");
                assert_eq!(subagent_type.as_deref(), Some("Explore"));
            }
            other => panic!("expected TaskStarted, got {other:?}"),
        }
    }

    /// A frame missing everything but the id must still ARM the task — dropping
    /// it would wedge the Waiting state forever.
    #[test]
    fn task_started_without_a_description_still_arms_the_task() {
        let mut a = ClaudeAdapter::new();
        let line = r#"{"type":"system","subtype":"task_started","task_id":"t-7"}"#;
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
        let mut a = ClaudeAdapter::new();
        let frame = a
            .encode_user_turn("u-1", "hello \"world\"\nwith newline")
            .unwrap();
        assert!(frame.ends_with('\n'), "frame must be newline-terminated");
        let v: serde_json::Value = serde_json::from_str(frame.trim_end()).unwrap();
        assert_eq!(v["type"], "user");
        assert_eq!(v["uuid"], "u-1");
        assert_eq!(v["message"]["role"], "user");
        assert_eq!(v["message"]["content"][0]["type"], "text");
        assert_eq!(
            v["message"]["content"][0]["text"],
            "hello \"world\"\nwith newline"
        );
    }

    /// Real claude 2.1.265 ack (tmp/queued_interrupt_probe.log) is not a row;
    /// an error ack is no ack.
    #[test]
    fn interrupt_request_and_its_ack() {
        let mut a = ClaudeAdapter::new();
        let frame = a.encode_interrupt("int-1").unwrap();
        let v: serde_json::Value = serde_json::from_str(frame.trim_end()).unwrap();
        assert_eq!(
            v,
            serde_json::json!({
                "type": "control_request",
                "request_id": "int-1",
                "request": { "subtype": "interrupt" },
            })
        );
        let ack = r#"{"type":"control_response","response":{"subtype":"success","request_id":"int-1","response":{"still_queued":[]}}}"#;
        assert_eq!(run(&mut a, ack), vec!["InterruptAcked(int-1)"]);
        let error = r#"{"type":"control_response","response":{"subtype":"error","request_id":"int-1","error":"x"}}"#;
        assert!(run(&mut a, error).is_empty());
    }

    #[test]
    fn prepare_session_command_drops_prompt_pipe_and_adds_input_format() {
        let mut a = ClaudeAdapter::new();
        let ctx = session_ctx();
        let cmd = a.prepare_session_command(&ctx).unwrap();
        // No prompt-file piping in the resident path.
        assert!(
            !cmd.contains("cat "),
            "resident cmd must not pipe a prompt file: {cmd}"
        );
        // Reads turns from stdin.
        assert!(cmd.contains("--input-format stream-json"), "got: {cmd}");
        assert!(cmd.contains("--replay-user-messages"), "got: {cmd}");
        assert!(cmd.contains("--output-format stream-json"), "got: {cmd}");
        // Resume + model + sandbox bypass + prune hook all preserved.
        assert!(cmd.contains("--resume 'sid-9'"), "got: {cmd}");
        assert!(cmd.contains("--model 'opus'"), "got: {cmd}");
        assert!(cmd.contains("--dangerously-skip-permissions"), "got: {cmd}");
        assert!(cmd.contains("prune-reminder-hook"), "got: {cmd}");
        // cd prefix retained.
        assert!(cmd.starts_with("cd '/tmp/proj' && claude"), "got: {cmd}");
    }

    /// The chat title becomes the process's peer-addressable name, on both
    /// spawn paths — a respawn (resume, prune, knob change) renders through the
    /// same builder, so the name can't silently revert to claude's derived one.
    #[test]
    fn session_name_becomes_the_claude_name_flag() {
        let mut a = ClaudeAdapter::new();
        let named = crate::adapter::SessionContext {
            session_name: Some("cover letters writer"),
            ..session_ctx()
        };
        let cmd = a.prepare_session_command(&named).unwrap();
        // One argv element: a title with spaces can't split into flags.
        assert!(cmd.contains("--name 'cover letters writer'"), "got: {cmd}");

        let argv = a.prepare_session_native(&named).unwrap().args;
        let i = argv.iter().position(|s| s == "--name").unwrap();
        assert_eq!(argv[i + 1], "cover letters writer");

        // No title (blank, or key not unsealed yet): no flag, claude derives one.
        let unnamed = a.prepare_session_command(&session_ctx()).unwrap();
        assert!(!unnamed.contains("--name"), "got: {unnamed}");
    }

    #[test]
    fn sandboxed_spawns_disallow_peer_messaging() {
        let mut a = ClaudeAdapter::new();
        let owner = a.prepare_session_command(&session_ctx()).unwrap();
        assert!(
            owner.contains("--disallowedTools 'AskUserQuestion' --dangerously-skip-permissions"),
            "got: {owner}"
        );
        let sandboxed = crate::adapter::SessionContext {
            is_sandboxed: true,
            ..session_ctx()
        };
        let cmd = a.prepare_session_command(&sandboxed).unwrap();
        assert!(
            cmd.contains("--disallowedTools 'AskUserQuestion SendMessage ListAgents'"),
            "got: {cmd}"
        );
        assert!(
            !cmd.contains("--dangerously-skip-permissions"),
            "got: {cmd}"
        );
        let argv = a.prepare_session_native(&sandboxed).unwrap().args;
        let i = argv.iter().position(|s| s == "--disallowedTools").unwrap();
        assert_eq!(argv[i + 1], "AskUserQuestion SendMessage ListAgents");
    }

    #[test]
    fn chrome_flag_follows_the_setting_for_owners_only() {
        let has_chrome = |is_sandboxed: bool, chrome: bool| {
            let args = claude_flag_args(
                None,
                "chat-1",
                None,
                false,
                None,
                is_sandboxed,
                None,
                None,
                true,
                chrome,
            );
            flags_to_argv(args).iter().any(|a| a == "--chrome")
        };
        assert!(has_chrome(false, true));
        assert!(!has_chrome(false, false));
        assert!(!has_chrome(true, true));
    }

    /// Real claude 2.1.265 shape (tmp/peer_probe2.log): a cross-session
    /// `SendMessage` to this session is echoed with `origin.kind == "peer"`
    /// and a uuid we never wrote.
    #[test]
    fn peer_replay_echo_emits_peer_turn_consumed() {
        let mut a = ClaudeAdapter::new();
        let line = r#"{"type":"user","message":{"role":"user","content":"Another Claude session sent a message:\n<cross-session-message from=\"uds:/tmp/cc-socks/95696.sock\" from-name=\"zucchini-9d\" from-mode=\"bypass\">\nProbe question 2: what is 6 x 7? Reply to me via SendMessage.\n</cross-session-message>"},"session_id":"011e1e77-dbc3-44e8-b423-06b9379903f5","parent_tool_use_id":null,"uuid":"153cc7ac-438d-4e49-a41e-2ab0afbaf756","timestamp":"2026-09-10T15:28:41.059Z","isReplay":true,"isSynthetic":true,"origin":{"kind":"peer","from":"uds:/tmp/cc-socks/95696.sock","verifiedPeerPid":95696,"msg_id":"a2abd1f2-a0c3-4ff6-a3f6-8940043e28e4","name":"zucchini-9d","fromMode":"bypass","body":"Probe question 2: what is 6 x 7? Reply to me via SendMessage."}}"#;
        assert_eq!(
            run(&mut a, line),
            vec!["PeerTurnConsumed(zucchini-9d: Probe question 2: what is 6 x 7? Reply to me via SendMessage.)"]
        );
    }

    /// Must stay byte-stable across respawns — see `AgentAdapter::turn_time_line`.
    #[test]
    fn session_system_prompt_is_stable_across_respawns() {
        // The two-spawn equality only catches a same-second rebuild; the
        // time-line check catches the clock itself.
        let ctx = session_ctx();
        let first = ClaudeAdapter::new().prepare_session_command(&ctx).unwrap();
        let second = ClaudeAdapter::new().prepare_session_command(&ctx).unwrap();
        assert!(
            !first.contains("current local time"),
            "no time line in the system prompt: {first}"
        );
        assert_eq!(first, second, "respawn must reproduce the same command");

        // A rename changes the `--name` flag and NOTHING in the appended system
        // prompt — that block heads the cacheable prefix.
        let renamed = ClaudeAdapter::new()
            .prepare_session_command(&crate::adapter::SessionContext {
                session_name: Some("cover letters writer"),
                ..session_ctx()
            })
            .unwrap();
        let sys_of = |cmd: &str| {
            let start = cmd.find("--append-system-prompt").expect("system prompt");
            cmd[start..cmd.find("--print").expect("--print")].to_string()
        };
        assert_eq!(sys_of(&first), sys_of(&renamed));
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
            st.has_live_task("bzdt1ec2n"),
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
    fn decode_project_dir_roundtrips_real_dir_with_dashes() {
        // Build a real directory tree, derive the exact name claude-code would
        // encode for its cwd (every path separator + a Windows drive colon →
        // '-'), and assert decode walks back to the original absolute path.
        // The `berkshire-hathaway` component pins the greedy longest-prefix
        // handling of a literal '-' inside a segment. Cross-platform by
        // construction: the encoding is derived from the real host path, so
        // this drives the Unix `/` root on Unix and the `C:\` drive root on
        // Windows — the exact case that made every Windows import a silent
        // no-op before the drive-aware branch.
        let base = std::env::temp_dir().join(format!("zucc-decode-{}", std::process::id()));
        let leaf = base.join("berkshire-hathaway").join("sub");
        std::fs::create_dir_all(&leaf).unwrap();

        let leaf_str = leaf.to_str().unwrap();
        let encoded: String = leaf_str
            .chars()
            .map(|c| {
                if c == '/' || c == '\\' || c == ':' {
                    '-'
                } else {
                    c
                }
            })
            .collect();

        let decoded = decode_project_dir(&encoded).expect("decode should resolve a real dir");
        assert_eq!(std::path::PathBuf::from(&decoded), leaf);

        std::fs::remove_dir_all(&base).ok();
    }

    #[test]
    fn decode_project_dir_rejects_unresolvable() {
        // A drive-shaped name whose segments don't resolve on disk returns
        // None (caller skips it), and a bare non-encoded name is rejected too.
        assert_eq!(
            decode_project_dir("Z--nope-does-not-exist-anywhere-42"),
            None
        );
        assert_eq!(decode_project_dir("plainname"), None);
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
        assert!(cmd.contains("scheduled-messages create"), "got: {}", cmd);
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
        assert!(cmd.contains("scheduled-messages create"), "got: {}", cmd);
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
        assert!(cmd.contains(PRUNE_SUMMARY_MOVE_INSTRUCTION), "got: {}", cmd);
        // … and the PostToolUse reminder hook is injected via --settings.
        assert!(cmd.contains("--settings"), "got: {}", cmd);
        assert!(cmd.contains("prune-reminder-hook"), "got: {}", cmd);
        assert!(cmd.contains("PostToolUse"), "got: {}", cmd);
    }

    #[test]
    fn prepare_native_yields_shell_free_recipe() {
        use crate::adapter::TurnContext;
        use std::path::PathBuf;
        let mut a = ClaudeAdapter::new();
        let prompt_file = PathBuf::from("/tmp/p.txt");
        // for_test sets project_path = Some("/tmp/proj").
        let ctx = TurnContext::for_test(&prompt_file, Some("sid-9"), false, None);
        let spawn = a
            .prepare_native(&ctx)
            .expect("claude offers a native recipe");
        assert_eq!(spawn.program, "claude");
        assert_eq!(spawn.cwd, Some(PathBuf::from("/tmp/proj")));
        assert_eq!(spawn.prompt_stdin_file, Some(prompt_file.clone()));
        // Flags land as clean argv (one OS arg per token, no quoting).
        assert!(
            spawn.args.iter().any(|a| a == "--print"),
            "{:?}",
            spawn.args
        );
        assert!(
            spawn.args.iter().any(|a| a == "--append-system-prompt"),
            "{:?}",
            spawn.args
        );
        // --resume is immediately followed by the bare session id.
        let ri = spawn
            .args
            .iter()
            .position(|a| a == "--resume")
            .expect("resume flag present");
        assert_eq!(spawn.args[ri + 1], "sid-9");
        // The native argv must carry the SAME flag set the POSIX string does:
        // every native arg appears (verbatim or single-quoted) in the shell cmd.
        let cmd = a.prepare_command(&ctx).unwrap();
        for arg in &spawn.args {
            assert!(
                cmd.contains(arg.as_str()) || cmd.contains(&shell_escape(arg)),
                "posix cmd missing native arg {arg:?}: {cmd}"
            );
        }
    }

    #[test]
    fn prepare_session_native_holds_stdin_open() {
        let mut a = ClaudeAdapter::new();
        let ctx = session_ctx();
        let spawn = a
            .prepare_session_native(&ctx)
            .expect("claude offers a resident native recipe");
        assert_eq!(spawn.program, "claude");
        // Resident holds stdin open for frames — no prompt file.
        assert!(spawn.prompt_stdin_file.is_none());
        assert!(
            spawn.args.iter().any(|a| a == "--input-format"),
            "{:?}",
            spawn.args
        );
    }

    mod prune {
        use super::super::*;
        use crate::prune::test_util::{read_lines, write_jsonl};

        /// One assistant `tool_use` (Read junk.rs) paired to a user `tool_result`.
        /// Pruning by (Read, junk) with no summary sets the result content to
        /// `[pruned]` and the tool_use input to the glob, preserving every other field.
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
            // tool_use input keeps only the glob, threading fields intact.
            assert_eq!(
                lines[0]["message"]["content"][0]["input"],
                serde_json::json!({"pruned_args": "junk"})
            );
            assert_eq!(lines[0]["uuid"], "a1");
            // paired tool_result content blanked to the placeholder.
            assert_eq!(lines[1]["message"]["content"][0]["content"], "[pruned]");
            assert_eq!(lines[1]["uuid"], "u1");
        }

        /// A Skill prune blanks BOTH the tiny ack AND the big body in the separate
        /// `sourceToolUseID`-threaded frame. Regression for chat bf70c589, where the
        /// body survived a prune that freed only 19 bytes.
        #[test]
        fn prunes_skill_ack_and_expansion_body() {
            let big = "MODEL claude-opus-4-8 ".repeat(50);
            let f = write_jsonl(&[
                r#"{"type":"assistant","message":{"content":[{"type":"tool_use","id":"sk1","name":"Skill","input":{"skill":"claude-api","args":""}}]},"uuid":"a1"}"#,
                r#"{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"sk1","content":"Running claude-api skill"}]},"uuid":"u1"}"#,
                &format!(
                    r#"{{"type":"user","isMeta":true,"sourceToolUseID":"sk1","message":{{"content":[{{"type":"text","text":"{big}"}}]}},"uuid":"u2"}}"#
                ),
            ]);
            assert_eq!(count_matches(f.path(), "Skill", "claude-api").unwrap(), 1);
            let stats = prune_jsonl(f.path(), "Skill", "claude-api").unwrap();
            // One user-facing output (the source id dedups ack + body), but the
            // freed bytes reflect the BODY, not just the 24-char ack.
            assert_eq!(stats.results_blanked, 1);
            assert!(
                stats.freed_bytes > big.len() - 16,
                "freed {} should reflect the {}-byte body",
                stats.freed_bytes,
                big.len()
            );
            let lines = read_lines(f.path());
            assert_eq!(
                lines[0]["message"]["content"][0]["input"],
                serde_json::json!({"pruned_args": "claude-api"})
            );
            assert_eq!(lines[1]["message"]["content"][0]["content"], "[pruned]");
            // The expansion body text is blanked; threading fields intact.
            assert_eq!(lines[2]["message"]["content"][0]["text"], "[pruned]");
            assert_eq!(lines[2]["sourceToolUseID"], "sk1");
            assert_eq!(lines[2]["isMeta"], true);
            // Idempotent + already-pruned aware: re-prune finds nothing.
            assert_eq!(count_matches(f.path(), "Skill", "claude-api").unwrap(), 0);
            assert_eq!(
                prune_jsonl(f.path(), "Skill", "claude-api")
                    .unwrap()
                    .results_blanked,
                0
            );
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
                r#"{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"t1","content":"body one, long enough to free bytes"}]}}"#,
                r#"{"type":"assistant","message":{"content":[{"type":"tool_use","id":"t2","name":"Read","input":{"file_path":"junk2.rs"}}]}}"#,
                r#"{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"t2","content":"body two, long enough to free bytes"}]}}"#,
                r#"{"type":"assistant","message":{"content":[{"type":"tool_use","id":"t3","name":"Read","input":{"file_path":"junk3.rs"}}]}}"#,
                r#"{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"t3","content":"body three, long enough to free bytes"}]}}"#,
            ]);
            let targets = [
                PruneTarget::new("Read", "junk1.rs", ""),
                PruneTarget::new("Read", "junk2.rs", ""),
                PruneTarget::new("Read", "junk3.rs", ""),
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
            let targets = [
                PruneTarget::new("Read", "junk", ""),
                PruneTarget::new("Read", "junk", ""),
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
            let targets = [
                PruneTarget::new("Read", "junk", ""),
                PruneTarget::new("Read", "junk", ""),
                PruneTarget::new("Read", "other", ""),
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

        fn tool_use(id: &str, name: &str, input: serde_json::Value) -> String {
            serde_json::json!({"type":"assistant","message":{"content":[
                {"type":"tool_use","id":id,"name":name,"input":input}
            ]}})
            .to_string()
        }

        fn tool_result(id: &str, content: &str) -> String {
            serde_json::json!({"type":"user","message":{"content":[
                {"type":"tool_result","tool_use_id":id,"content":content}
            ]}})
            .to_string()
        }

        fn prune_call(id: &str, command: &str) -> [String; 2] {
            [
                tool_use(id, "Bash", serde_json::json!({ "command": command })),
                tool_result(id, "queued 1 prune(s)"),
            ]
        }

        fn write(lines: &[String]) -> crate::prune::test_util::TempJsonl {
            write_jsonl(&lines.iter().map(String::as_str).collect::<Vec<_>>())
        }

        fn input_of(line: &serde_json::Value) -> &serde_json::Value {
            &line["message"]["content"][0]["input"]
        }

        fn content_of(line: &serde_json::Value) -> &serde_json::Value {
            &line["message"]["content"][0]["content"]
        }

        #[test]
        fn summary_flag_value_is_read_as_shell_words() {
            let m = SUMMARY_MOVED_MARKER;
            let cases = [
                (
                    r#"p prune-context --tool-name Read --args "build.log" --summary "build""#,
                    ("Read", "build.log", "build"),
                    Some(format!(
                        r#"p prune-context --tool-name Read --args "build.log" --summary {m}"#
                    )),
                ),
                (
                    "p prune-context --args x --summary 'kept: a b' --chat-id c",
                    ("", "x", "kept: a b"),
                    Some(format!(
                        "p prune-context --args x --summary {m} --chat-id c"
                    )),
                ),
                (
                    "p prune-context --args x --summary kept && echo ok",
                    ("", "x", "kept"),
                    Some(format!("p prune-context --args x --summary {m} && echo ok")),
                ),
                (
                    r#"p prune-context --args x --summary="kept it""#,
                    ("", "x", "kept it"),
                    Some(format!("p prune-context --args x --summary={m}")),
                ),
                (
                    r#"p prune-context --args x --reason "kept""#,
                    ("", "x", "kept"),
                    Some(format!("p prune-context --args x --reason {m}")),
                ),
                (
                    r#"p prune-context --args x --summary "say \"hi\"""#,
                    ("", "x", r#"say "hi""#),
                    Some(format!("p prune-context --args x --summary {m}")),
                ),
                // Only a whole flag value of the same target counts.
                (
                    r#"p prune-context --args "build" --summary "other""#,
                    ("", "build", "build"),
                    None,
                ),
                (
                    r#"p prune-context --args x --summary "build passed""#,
                    ("", "x", "build"),
                    None,
                ),
                (
                    r#"p prune-context --args y --summary "kept""#,
                    ("", "x", "kept"),
                    None,
                ),
                (
                    r#"p prune-context --args "a --summary kept" --summary kept"#,
                    ("", "a --summary kept", "kept"),
                    Some(format!(
                        r#"p prune-context --args "a --summary kept" --summary {m}"#
                    )),
                ),
                (
                    r#"p prune-context --args x --summary "$S""#,
                    ("", "x", "kept"),
                    None,
                ),
            ];
            for (command, (tool, needle, summary), expected) in cases {
                assert_eq!(
                    replace_summary_flag_value(command, &PruneTarget::new(tool, needle, summary)),
                    expected,
                    "{command}"
                );
            }
        }

        /// The summary lands in the pruned result, the pruned call keeps its glob,
        /// and the prune command keeps a marker instead of the summary.
        #[test]
        fn summary_moves_from_prune_command_into_result() {
            let command = r#""$ZUCCHINI_SPAWNER_BIN" prune-context --tool-name Read --args "build.log" --summary "build""#;
            let mut lines = vec![
                tool_use("r1", "Read", serde_json::json!({"file_path": "build.log"})),
                tool_result("r1", "log body"),
            ];
            lines.extend(prune_call("p1", command));
            let f = write(&lines);

            let stats =
                prune_batch_jsonl(f.path(), &[PruneTarget::new("Read", "build.log", "build")])
                    .unwrap();
            assert_eq!(stats.results_blanked, 1);
            let out = read_lines(f.path());
            assert_eq!(
                input_of(&out[0]),
                &serde_json::json!({"pruned_args": "build.log"})
            );
            assert_eq!(content_of(&out[1]), "[pruned] build");
            assert_eq!(
                input_of(&out[2])["command"],
                command.replace(r#""build""#, SUMMARY_MOVED_MARKER)
            );
            assert_eq!(content_of(&out[3]), "queued 1 prune(s)");
        }

        #[test]
        fn empty_needle_keeps_empty_input() {
            let mut lines = vec![
                tool_use("l1", "LS", serde_json::json!({})),
                tool_result("l1", "a b c"),
            ];
            lines.extend(prune_call(
                "p1",
                r#"p prune-context --tool-name LS --args "" --summary "three files""#,
            ));
            let f = write(&lines);
            prune_batch_jsonl(f.path(), &[PruneTarget::new("LS", "", "three files")]).unwrap();
            let out = read_lines(f.path());
            assert_eq!(input_of(&out[0]), &serde_json::json!({}));
            assert_eq!(content_of(&out[1]), "[pruned] three files");
        }

        /// Each target of a multi-target command gets its own summary, and each
        /// flag value becomes a marker.
        #[test]
        fn multi_target_command_moves_each_summary() {
            let command = r#"p prune-context --tool-name Read --args a.rs --summary "same" --tool-name Read --args b.rs --summary "same""#;
            let mut lines = vec![
                tool_use("a", "Read", serde_json::json!({"file_path": "a.rs"})),
                tool_result("a", "A body"),
                tool_use("b", "Read", serde_json::json!({"file_path": "b.rs"})),
                tool_result("b", "B body"),
            ];
            lines.extend(prune_call("p1", command));
            let f = write(&lines);
            let targets = [
                PruneTarget::new("Read", "a.rs", "same"),
                PruneTarget::new("Read", "b.rs", "same"),
            ];
            assert_eq!(
                prune_batch_jsonl(f.path(), &targets)
                    .unwrap()
                    .results_blanked,
                2
            );
            let out = read_lines(f.path());
            assert_eq!(content_of(&out[1]), "[pruned] same");
            assert_eq!(content_of(&out[3]), "[pruned] same");
            assert_eq!(
                input_of(&out[4])["command"],
                command.replace(r#""same""#, SUMMARY_MOVED_MARKER)
            );
        }

        /// A summary the requesting command doesn't carry literally stays in it, an
        /// older prune call with the same summary for another target is left alone,
        /// and the result gets the bare placeholder.
        #[test]
        fn unfindable_summary_leaves_commands_and_bare_placeholder() {
            let older = r#"p prune-context --tool-name Read --args b.rs --summary "kept""#;
            let command = r#"p prune-context --tool-name Read --args a.rs --summary "$S""#;
            let mut lines = vec![
                tool_use("a", "Read", serde_json::json!({"file_path": "a.rs"})),
                tool_result("a", "A body"),
            ];
            lines.extend(prune_call("p0", older));
            lines.extend(prune_call("p1", command));
            let f = write(&lines);
            prune_batch_jsonl(f.path(), &[PruneTarget::new("Read", "a.rs", "kept")]).unwrap();
            let out = read_lines(f.path());
            assert_eq!(content_of(&out[1]), "[pruned]");
            assert_eq!(input_of(&out[2])["command"], older);
            assert_eq!(input_of(&out[4])["command"], command);
        }

        #[test]
        fn long_summary_moves_whole() {
            let summary = "a".repeat(4096);
            let command =
                format!("p prune-context --tool-name Read --args a.rs --summary {summary}");
            let mut lines = vec![
                tool_use("a", "Read", serde_json::json!({"file_path": "a.rs"})),
                tool_result("a", "A body"),
            ];
            lines.extend(prune_call("p1", &command));
            let f = write(&lines);
            prune_batch_jsonl(f.path(), &[PruneTarget::new("Read", "a.rs", &summary)]).unwrap();
            let out = read_lines(f.path());
            assert_eq!(content_of(&out[1]), format!("[pruned] {summary}").as_str());
        }

        /// A Skill's ack gets the summary; the expansion body stays bare.
        #[test]
        fn skill_summary_goes_to_ack_not_body() {
            let mut lines = vec![
                tool_use("sk1", "Skill", serde_json::json!({"skill": "claude-api"})),
                tool_result("sk1", "Running claude-api skill"),
                serde_json::json!({"type":"user","isMeta":true,"sourceToolUseID":"sk1",
                    "message":{"content":[{"type":"text","text":"big body"}]}})
                .to_string(),
            ];
            lines.extend(prune_call(
                "p1",
                r#"p prune-context --tool-name Skill --args claude-api --summary "models listed""#,
            ));
            let f = write(&lines);
            prune_batch_jsonl(
                f.path(),
                &[PruneTarget::new("Skill", "claude-api", "models listed")],
            )
            .unwrap();
            let out = read_lines(f.path());
            assert_eq!(content_of(&out[1]), "[pruned] models listed");
            assert_eq!(out[2]["message"]["content"][0]["text"], "[pruned]");
        }

        /// A result already pruned with a summary (or a genuine output starting
        /// with `[pruned] `) is not re-selected, and a repeat leaves the bytes alone.
        #[test]
        fn summarised_result_is_already_pruned() {
            let mut lines = vec![
                tool_use("a", "Read", serde_json::json!({"file_path": "a.rs"})),
                tool_result("a", "A body"),
            ];
            lines.extend(prune_call(
                "p1",
                r#"p prune-context --tool-name Read --args a.rs --summary "kept""#,
            ));
            let f = write(&lines);
            let target = [PruneTarget::new("Read", "a.rs", "kept")];
            prune_batch_jsonl(f.path(), &target).unwrap();
            let after_first = std::fs::read_to_string(f.path()).unwrap();

            assert_eq!(count_matches(f.path(), "Read", "a.rs").unwrap(), 0);
            assert_eq!(
                prune_batch_jsonl(f.path(), &target).unwrap(),
                PruneStats::default()
            );
            assert_eq!(std::fs::read_to_string(f.path()).unwrap(), after_first);

            let genuine = write(&[
                tool_use("g", "Read", serde_json::json!({"file_path": "g.rs"})),
                tool_result("g", "[pruned] looks pruned"),
            ]);
            assert_eq!(count_matches(genuine.path(), "Read", "g.rs").unwrap(), 0);
        }

        /// A call killed mid-tool (no `tool_result`) is never picked and doesn't
        /// hide an older answered match.
        #[test]
        fn unanswered_tool_use_is_never_selected() {
            let f = write(&[
                tool_use("old", "Read", serde_json::json!({"file_path": "junk.rs"})),
                tool_result("old", "old body"),
                tool_use(
                    "killed",
                    "Read",
                    serde_json::json!({"file_path": "junk.rs"}),
                ),
            ]);
            assert_eq!(count_matches(f.path(), "Read", "junk").unwrap(), 1);
            assert_eq!(
                prune_jsonl(f.path(), "Read", "junk")
                    .unwrap()
                    .results_blanked,
                1
            );
            let out = read_lines(f.path());
            assert_eq!(content_of(&out[1]), "[pruned]");
            assert_eq!(
                input_of(&out[2]),
                &serde_json::json!({"file_path": "junk.rs"})
            );
        }

        /// In a batch, a target whose newest match a DIFFERENT selector already
        /// picked is a miss rather than falling through to an older output.
        #[test]
        fn batch_collision_with_different_selector_is_a_miss() {
            let f = write(&[
                tool_use("t1", "Read", serde_json::json!({"file_path": "junk1.rs"})),
                tool_result("t1", "body one"),
                tool_use("t2", "Read", serde_json::json!({"file_path": "junk2.rs"})),
                tool_result("t2", "body two"),
            ]);
            let targets = [
                PruneTarget::new("Read", "junk2", "two"),
                PruneTarget::new("Read", "junk", "any junk"),
            ];
            assert_eq!(
                prune_batch_jsonl(f.path(), &targets)
                    .unwrap()
                    .results_blanked,
                1
            );
            let out = read_lines(f.path());
            assert_eq!(content_of(&out[1]), "body one");
            assert_eq!(content_of(&out[3]), "[pruned]");
        }

        /// `freed_bytes` is the net serialized shrink — input, result and the
        /// command's summary→marker swap — and floors at 0 when the edits grow the
        /// transcript.
        #[test]
        fn freed_bytes_is_net_serialized_shrink() {
            let transcript = |body: &str| {
                let mut lines = vec![
                    tool_use("a", "Read", serde_json::json!({"file_path": "src/a.rs"})),
                    tool_result("a", body),
                ];
                lines.extend(prune_call(
                    "p1",
                    r#"p prune-context --tool-name Read --args a.rs --summary "a.rs: \"quoted\" takeaway""#,
                ));
                lines
            };
            let target = [PruneTarget::new(
                "Read",
                "a.rs",
                r#"a.rs: "quoted" takeaway"#,
            )];

            let big = write(&transcript(&"line\n".repeat(200)));
            let before = std::fs::metadata(big.path()).unwrap().len() as usize;
            let stats = prune_batch_jsonl(big.path(), &target).unwrap();
            let after = std::fs::metadata(big.path()).unwrap().len() as usize;
            assert_eq!(stats.freed_bytes, before - after);
            assert_eq!(
                input_of(&read_lines(big.path())[2])["command"],
                format!(
                    "p prune-context --tool-name Read --args a.rs --summary {SUMMARY_MOVED_MARKER}"
                )
            );

            let tiny = write(&transcript("x"));
            let before = std::fs::metadata(tiny.path()).unwrap().len();
            assert_eq!(
                prune_batch_jsonl(tiny.path(), &target).unwrap().freed_bytes,
                0
            );
            assert!(std::fs::metadata(tiny.path()).unwrap().len() > before);
        }
    }
}
