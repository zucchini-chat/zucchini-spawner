//! hermes adapter — Telegram-shape. One long-lived `hermes gateway run`
//! process per machine, ONE Unix socket between the spawner and the plugin,
//! all chats multiplexed by `chat_id`. Per-turn isolation lives inside the
//! plugin (asyncio task per turn, hermes `task_id == chat_id`); from the
//! `AgentAdapter` trait's perspective hermes is fork-per-turn, just like
//! claude/codex/cursor — `prepare_command` re-execs the spawner binary as a
//! `hermes-turn` subcommand and the trampoline shuttles claude-shape NDJSON
//! between the plugin socket and its own stdout.
//!
//! The single-socket architecture means cross-chat fan-in/fan-out happens
//! in `crate::hermes_support::socket_server` (spawner is server; plugin and every
//! trampoline are clients). The socket server demuxes inbound envelopes
//! by `chat_id` and routes proactive (cron-fired) envelopes through the
//! existing `send_agent_line` writer path. See that module for the wiring.
//!
//! Wire-format contract (verbatim from
//! `~/.hermes/plugins/zucchini/adapter.py`):
//!
//!   Inbound to plugin: `{"type":"turn","chat_id":...,"user_prompt":...,
//!                        "project_path":...,"yolo":bool,"model":<str|null>,
//!                        "channel_prompt":<str|null>,"resume":<str|null>,
//!                        "attachments":[abs,...]}`
//!   Outbound from plugin: `{"chat_id":...,"proactive":bool,"event":{...claude-shape...}}`
//!
//! The trampoline strips the outer wrapper and emits the inner `event` line
//! by line to stdout; `handle_line` here dispatches on `event.type` exactly
//! like codex. Proactive envelopes should never reach this adapter (they're
//! handled by the socket server); we guard anyway.
//!
//! Sandbox mapping mirrors codex's bypass flag: `!is_sandboxed` → `--yolo`
//! (full bypass on the plugin side); sandboxed invitees get no `--yolo` and
//! the plugin's default approval policy auto-denies tools in headless mode.

use std::path::PathBuf;

use anyhow::Result;
use smallvec::SmallVec;
use tokio::sync::mpsc;
use tracing::{debug, info};
use uuid::Uuid;

use crate::adapter::{
    claude_assistant_text_envelope, file_nonempty, first_message_capabilities_preamble,
    parse_json_obj, probe_with_blocking_auth, shell_escape, AdapterDescriptor, AgentAdapter,
    AgentEvent, AgentKind, ImportProgress, LastTokensDedup, TurnContext, MAX_STREAM_FRAME_BYTES,
};
use crate::writer::WriteEvent;

/// Wired into `adapter::ADAPTERS`. `installed_col` / `authenticated_col`
/// follow the `claude_code_*` (0022) / `cursor_*` (0033) / `codex_*` (0037)
/// per-kind nullable-BOOLEAN pair convention. Backend column allowlist must
/// learn about `hermes_installed` / `hermes_authenticated` in a follow-up
/// migration (see report); until then the writer-side PATCH 4xx's on these
/// columns and the queue head-of-line blocks.
pub const DESCRIPTOR: AdapterDescriptor = AdapterDescriptor {
    kind: AgentKind::Hermes,
    wire_name: "hermes",
    installed_col: "hermes_installed",
    authenticated_col: "hermes_authenticated",
    make: make_boxed,
    probe: probe_boxed,
    import: import_boxed,
    // Long-lived in-process gateway with a lossy resume and no per-turn local
    // transcript the spawner can edit — prune-context is infeasible.
    prune: None,
};

fn make_boxed() -> Box<dyn AgentAdapter> {
    Box::new(HermesAdapter::new())
}

/// Per-turn state. Only `last_emitted_tokens` (mirrors codex / claude) for
/// dedup of re-emitted `system.context_tokens` envelopes. Everything else
/// is stateless.
#[derive(Default)]
pub struct HermesAdapter {
    last_emitted_tokens: LastTokensDedup,
}

impl HermesAdapter {
    pub fn new() -> Self {
        Self::default()
    }
}

impl AgentAdapter for HermesAdapter {
    fn kind(&self) -> AgentKind {
        AgentKind::Hermes
    }

    fn prepare_command(&mut self, ctx: &TurnContext<'_>) -> Result<String> {
        // Re-exec the spawner binary as a `hermes-turn` subcommand. The
        // supervisor exports `ZUCCHINI_SPAWNER_BIN=<current_exe>` on every
        // spawn (agent.rs default_spawn_fn), and the socket server task
        // exports `ZUCCHINI_SPAWNER_SOCK=<server-socket-path>` BEFORE we
        // get here (see `main.rs`); the trampoline reads both from env.
        //
        // Long strings (prompt, channel prompt) go through temp files
        // mirroring codex's `- < <prompt-file>` pattern — keeps argv bounded
        // and side-steps shell-escape pitfalls for multi-MB attachment-laden
        // bodies. cwd intentionally not changed here: hermes runs in the
        // gateway process, the plugin sets per-task cwd via
        // `register_task_env_overrides(chat_id, {"cwd": project_path})`
        // before each turn.
        let mut cmd = String::from("\"$ZUCCHINI_SPAWNER_BIN\" hermes-turn");
        cmd.push_str(&format!(" --chat-id={}", shell_escape(ctx.chat_id)));
        cmd.push_str(&format!(
            " --user-prompt-file={}",
            shell_escape(&ctx.prompt_file.to_string_lossy())
        ));

        // Sandbox → yolo mapping. `!is_sandboxed` → `--yolo` (full bypass on
        // the plugin side, mirroring codex's
        // `--dangerously-bypass-approvals-and-sandbox`). Sandboxed invitees
        // get NO `--yolo` and the plugin's default approval policy
        // auto-denies tools in headless mode — same trust posture as the
        // other multi-agent flows on a shared machine.
        if !ctx.is_sandboxed {
            cmd.push_str(" --yolo");
        }

        if let Some(pp) = ctx.project_path {
            cmd.push_str(&format!(" --project-path={}", shell_escape(pp)));
        }

        // Verbatim pass-through of `chats.model` (migration 0035). Plugin
        // routes it as `model_override` to `_create_agent`. Empty / blank
        // is filtered to `None` upstream (main.rs::handle_message_put), so
        // `Some(_)` here means "user explicitly picked a model".
        if let Some(model) = ctx.model {
            cmd.push_str(&format!(" --model={}", shell_escape(model)));
        }

        // Hermes session resume — IDs look like `YYYYMMDD_HHMMSS_hex6`, NOT
        // UUIDs (per agent_log 2026-05-28). We reuse the existing multi-agent
        // `chats.agent_session_id` column for now; a dedicated column
        // (`chats.hermes_session_id`) is a follow-up that doesn't gate v1.
        if let Some(sid) = ctx.agent_session_id {
            cmd.push_str(&format!(" --resume={}", shell_escape(sid)));
        }

        // Worktree handling: hermes has no first-class worktree flag (same
        // as codex). For v1 we ignore it; a follow-up pass can `git worktree
        // add` upstream and pass the worktree dir as `--project-path`.
        let _ = ctx.worktree;

        // No stdin redirect — the trampoline reads the prompt from the file
        // path passed via `--user-prompt-file`. This is cleaner than codex's
        // `- < <prompt-file>` stdin pattern because the supervisor only
        // cleans up the file after the spawn returns, so the trampoline
        // can re-read it on connect failure / retry without racing the
        // cleanup. Same trick the supervisor itself uses (`tokio::fs::write`
        // → spawn → `tokio::fs::remove_file` after `child.wait`).

        Ok(cmd)
    }

    /// No system-prompt injection point, so capabilities ride the first user
    /// message (prepended to the `--user-prompt-file` plaintext); worktree
    /// ignored. See [`first_message_capabilities_preamble`].
    fn prompt_file_preamble(&self, ctx: &TurnContext<'_>) -> Option<String> {
        first_message_capabilities_preamble(ctx)
    }

    fn handle_line(&mut self, line: String) -> SmallVec<[AgentEvent; 2]> {
        // Same dispatch shape as codex.rs: parse the inner claude-shape
        // envelope (the trampoline has already stripped the outer
        // `{chat_id, proactive, event}` wrapper) and route based on
        // `event.type`. Oversize-frame guard prevents a tool_result frame
        // embedding a multi-MB stdout from churning the heap on the
        // `serde_json::Value` parse — forward verbatim above the cap.
        let mut out: SmallVec<[AgentEvent; 2]> = SmallVec::new();

        if line.len() > MAX_STREAM_FRAME_BYTES {
            out.push(AgentEvent::Frame(line));
            return out;
        }

        let Some(obj) = parse_json_obj(&line) else {
            // Non-JSON / non-object: forward as-is. The trampoline shouldn't
            // emit these, but a stderr-merged spawner-side line still
            // surfaces here; codex / claude take the same permissive path.
            out.push(AgentEvent::Frame(line));
            return out;
        };

        let Some(ty) = obj.get("type").and_then(|v| v.as_str()) else {
            out.push(AgentEvent::Frame(line));
            return out;
        };

        match ty {
            "system" => {
                // Two `system` subtypes from the plugin:
                //   - subtype="init"           → harvest session_id, drop
                //                                the frame (matches claude's
                //                                init-skip).
                //   - subtype="context_tokens" → ContextTokens via dedup.
                let subtype = obj.get("subtype").and_then(|v| v.as_str()).unwrap_or("");
                match subtype {
                    "init" => {
                        if let Some(sid) = obj.get("session_id").and_then(|v| v.as_str()) {
                            out.push(AgentEvent::SessionIdHarvested(sid.to_string()));
                        } else {
                            debug!("hermes system/init without session_id");
                        }
                    }
                    "context_tokens" => {
                        if let Some(tokens) = obj.get("context_tokens").and_then(|v| v.as_i64()) {
                            if let Some(t) = self.last_emitted_tokens.observe(tokens) {
                                out.push(AgentEvent::ContextTokens(t));
                            }
                        } else {
                            debug!("hermes system/context_tokens without context_tokens int");
                        }
                    }
                    other => {
                        debug!(subtype = %other, "hermes system frame with unknown subtype, forwarding");
                        out.push(AgentEvent::Frame(line));
                    }
                }
            }
            "assistant" | "user" => {
                // assistant: streaming text delta OR tool_use envelope.
                // user: tool_result envelope (plugin emits per
                // `tool_complete_callback`). Both are forwarded verbatim —
                // iOS's `SpawnerMessageDescriber` already renders both via
                // the existing claude-shape branches.
                out.push(AgentEvent::Frame(line));
            }
            "result" => {
                // Terminal envelope. Plugin guarantees exactly one per turn
                // (subtype=success or subtype=error). Forward so iOS renders
                // `[result: ...]`, then signal Result so the supervisor
                // latches `Done.has_result = true`. Matches codex's
                // turn.completed branch.
                out.push(AgentEvent::Frame(line));
                out.push(AgentEvent::Result {
                    origin_is_task: false,
                });
            }
            "error" => {
                // Plugin emits this when the turn never produces a usable
                // answer (auth crash, AIAgent init failure, run exception).
                // Normalize to a claude-shape result envelope so iOS's
                // terminator pill renders consistently with codex's
                // `turn.failed` path.
                let message = obj
                    .get("message")
                    .and_then(|v| v.as_str())
                    .unwrap_or("hermes error")
                    .to_string();
                let result_frame = serde_json::json!({
                    "type": "result",
                    "subtype": "error",
                    "is_error": true,
                    "error": { "message": message },
                })
                .to_string();
                out.push(AgentEvent::Frame(result_frame));
                out.push(AgentEvent::Result {
                    origin_is_task: false,
                });
            }
            other => {
                // Defensive forward — wire-format drift shouldn't silently
                // swallow content. iOS will fall through to its
                // "unknown frame" branch.
                debug!(ty = %other, "hermes unknown envelope type, forwarding");
                out.push(AgentEvent::Frame(line));
            }
        }

        out
    }
}

/// Filesystem-only probe — same shape as codex.rs. Returns
/// `(installed, authenticated)`. `installed = (hermes on PATH)`,
/// `authenticated = (~/.hermes/auth.json exists and non-empty)`.
///
/// Caveats accepted for v1: a broken install where `hermes --version` exits
/// non-zero still reports installed=true; a stale `auth.json` (token revoked
/// upstream) reports authenticated=true. Both surfaces show up as runtime
/// error envelopes from the plugin, which is the right place for them.
pub async fn probe() -> (bool, bool) {
    probe_with_blocking_auth("hermes", is_authenticated).await
}

fn probe_boxed() -> futures::future::BoxFuture<'static, (bool, bool)> {
    Box::pin(probe())
}

fn is_authenticated() -> bool {
    let Some(home) = std::env::var_os("HOME").map(PathBuf::from) else {
        return false;
    };
    file_nonempty(&home.join(".hermes").join("auth.json"))
}

/// Stub importer — same pattern as codex.rs. Hermes session state lives in
/// `~/.hermes/state.db` (sqlite) + `~/.hermes/sessions/`; a follow-up pass
/// will walk them. For v1 we report 100% so the per-kind progress slice
/// closes cleanly.
pub(crate) async fn import(
    _machine_id: Uuid,
    _user_id: Uuid,
    _write_tx: mpsc::Sender<WriteEvent>,
    progress: ImportProgress,
) -> Result<()> {
    info!("hermes history import not yet implemented, skipping");
    progress(100).await;
    Ok(())
}

fn import_boxed(
    machine_id: Uuid,
    user_id: Uuid,
    write_tx: mpsc::Sender<WriteEvent>,
    progress: ImportProgress,
) -> futures::future::BoxFuture<'static, Result<()>> {
    Box::pin(import(machine_id, user_id, write_tx, progress))
}

/// Build a synthetic claude-shape assistant text envelope from a plain
/// string. Used by the socket server's proactive lane when the inner event
/// the plugin sent isn't already a full assistant envelope (e.g. cron-fired
/// `send()` calls that hand us raw strings via `BasePlatformAdapter.send()`
/// pass-through). Exposed here (next to the wire-format constants) so the
/// socket server doesn't have to learn claude's envelope shape directly.
pub(crate) fn synthesize_assistant_text(text: &str) -> String {
    claude_assistant_text_envelope(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use std::path::PathBuf;

    fn run(adapter: &mut HermesAdapter, line: &str) -> Vec<String> {
        crate::adapter::stringify_events(adapter.handle_line(line.to_string()))
    }

    fn frame_value(event: &str) -> Value {
        crate::adapter::frame_value(event)
    }

    fn ctx<'a>(
        prompt_file: &'a std::path::Path,
        agent_session_id: Option<&'a str>,
        is_sandboxed: bool,
        model: Option<&'a str>,
    ) -> TurnContext<'a> {
        TurnContext::for_test(prompt_file, agent_session_id, is_sandboxed, model)
    }

    #[test]
    fn prepare_command_owner_first_turn_passes_yolo_and_project_path() {
        let mut a = HermesAdapter::new();
        let prompt_file = PathBuf::from("/tmp/p.txt");
        let c = ctx(&prompt_file, None, false, None);
        let cmd = a.prepare_command(&c).unwrap();
        assert!(cmd.contains("hermes-turn"), "got: {}", cmd);
        assert!(
            cmd.contains("--yolo"),
            "owner non-sandboxed must get --yolo: {}",
            cmd
        );
        assert!(cmd.contains("--project-path='/tmp/proj'"), "got: {}", cmd);
        assert!(
            cmd.contains("--user-prompt-file='/tmp/p.txt'"),
            "prompt via file path: {}",
            cmd
        );
        assert!(
            !cmd.contains("--resume"),
            "first turn must not pass --resume: {}",
            cmd
        );
    }

    #[test]
    fn prepare_command_sandboxed_invitee_omits_yolo() {
        let mut a = HermesAdapter::new();
        let prompt_file = PathBuf::from("/tmp/p.txt");
        let c = ctx(&prompt_file, None, true, None);
        let cmd = a.prepare_command(&c).unwrap();
        assert!(
            !cmd.contains("--yolo"),
            "sandboxed invitee must not get --yolo: {}",
            cmd
        );
    }

    #[test]
    fn prepare_command_resume_passes_session_id() {
        let mut a = HermesAdapter::new();
        let prompt_file = PathBuf::from("/tmp/p.txt");
        let c = ctx(&prompt_file, Some("20260528_214220_e6100c"), false, None);
        let cmd = a.prepare_command(&c).unwrap();
        assert!(
            cmd.contains("--resume='20260528_214220_e6100c'"),
            "got: {}",
            cmd
        );
    }

    // First-turn-prepend / resume-omit covered once in adapter.rs
    // (`first_message_preamble_first_turn_then_resume`); hermes delegates to
    // `first_message_capabilities_preamble`, not re-asserted here.

    #[test]
    fn prepare_command_model_pass_through() {
        let mut a = HermesAdapter::new();
        let prompt_file = PathBuf::from("/tmp/p.txt");
        let c = ctx(&prompt_file, None, false, Some("claude-3.5-sonnet"));
        let cmd = a.prepare_command(&c).unwrap();
        assert!(cmd.contains("--model='claude-3.5-sonnet'"), "got: {}", cmd);
    }

    #[test]
    fn system_init_harvests_session_id_and_drops_frame() {
        let mut a = HermesAdapter::new();
        let line = r#"{"type":"system","subtype":"init","session_id":"20260528_214220_e6100c","tools":[]}"#;
        let events = run(&mut a, line);
        assert_eq!(events, vec!["SessionIdHarvested(20260528_214220_e6100c)"]);
    }

    #[test]
    fn assistant_text_envelope_forwarded_verbatim() {
        let mut a = HermesAdapter::new();
        let line = r#"{"type":"assistant","message":{"content":[{"type":"text","text":"hi"}],"usage":{"input_tokens":0,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":0}}}"#;
        let events = run(&mut a, line);
        assert_eq!(events.len(), 1);
        let v = frame_value(&events[0]);
        assert_eq!(v["type"], "assistant");
        assert_eq!(v["message"]["content"][0]["text"], "hi");
    }

    #[test]
    fn user_tool_result_envelope_forwarded_verbatim() {
        let mut a = HermesAdapter::new();
        let line = r#"{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"t1","content":"ok","is_error":false}]}}"#;
        let events = run(&mut a, line);
        assert_eq!(events.len(), 1);
        let v = frame_value(&events[0]);
        assert_eq!(v["type"], "user");
        assert_eq!(v["message"]["content"][0]["type"], "tool_result");
    }

    #[test]
    fn system_context_tokens_envelope_emits_context_tokens() {
        let mut a = HermesAdapter::new();
        let line = r#"{"type":"system","subtype":"context_tokens","context_tokens":12345}"#;
        let events = run(&mut a, line);
        assert_eq!(events, vec!["ContextTokens(12345)"]);
    }

    #[test]
    fn repeated_context_tokens_dedup() {
        let mut a = HermesAdapter::new();
        let line = r#"{"type":"system","subtype":"context_tokens","context_tokens":12345}"#;
        let first = run(&mut a, line);
        assert_eq!(first, vec!["ContextTokens(12345)"]);
        // Same value → no second emission.
        let second = run(&mut a, line);
        assert_eq!(second.len(), 0);
    }

    #[test]
    fn result_envelope_emits_frame_and_result_marker() {
        let mut a = HermesAdapter::new();
        let line = r#"{"type":"result","subtype":"success","is_error":false,"result":"done","duration_ms":42,"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}"#;
        let events = run(&mut a, line);
        assert_eq!(events.len(), 2);
        assert!(events[0].starts_with("Frame("));
        assert_eq!(events[1], "Result");
        let v = frame_value(&events[0]);
        assert_eq!(v["type"], "result");
        assert_eq!(v["subtype"], "success");
    }

    #[test]
    fn error_envelope_normalized_to_result_error_frame() {
        let mut a = HermesAdapter::new();
        let line = r#"{"type":"error","message":"AIAgent init failed: provider 500"}"#;
        let events = run(&mut a, line);
        assert_eq!(events.len(), 2);
        let v = frame_value(&events[0]);
        assert_eq!(v["type"], "result");
        assert_eq!(v["subtype"], "error");
        assert_eq!(v["is_error"], true);
        assert_eq!(v["error"]["message"], "AIAgent init failed: provider 500");
        assert_eq!(events[1], "Result");
    }

    #[test]
    fn non_json_line_kept_as_frame() {
        let mut a = HermesAdapter::new();
        let events = run(&mut a, "non-json-noise");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0], "Frame(non-json-noise)");
    }

    #[test]
    fn synthesize_assistant_text_wraps_into_claude_shape() {
        let line = synthesize_assistant_text("hello world");
        let v: Value = serde_json::from_str(&line).unwrap();
        assert_eq!(v["type"], "assistant");
        assert_eq!(v["message"]["content"][0]["type"], "text");
        assert_eq!(v["message"]["content"][0]["text"], "hello world");
    }
}
