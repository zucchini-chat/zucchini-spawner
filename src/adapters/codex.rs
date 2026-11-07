//! codex (openai codex-cli) adapter. A RESIDENT: one long-lived `codex
//! app-server` per chat (JSON-RPC over stdio), kept warm between turns exactly
//! like the claude resident. Protocol facts (codex-cli 0.154) and the raw
//! captures behind the tests: `tmp/codex-appserver-spike.md`.
//!
//! Session: `initialize` → `initialized` + `thread/resume <agent_session_id>`
//! (threads `codex exec` created resume too) or `thread/start`; the thread id in
//! its response is the session id. Every user turn is a `turn/start` stamped
//! with `clientUserMessageId`: on a busy thread it joins the running turn, and
//! codex echoes the id on the `userMessage` item once the input is recorded
//! (`UserTurnConsumed`). `/compact` is
//! `thread/compact/start`, held until the chat is idle — on a busy thread it
//! would interrupt the running turn. Teardown closes stdin: SIGTERM leaves the
//! shell commands codex runs (in process groups of their own) orphaned.
//!
//! JSON-RPC ids are strings naming what they answer: `init`, `thread`,
//! `turn:<uuid>`, `compact:<uuid>`.
//!
//! Notification mapping (codex → claude-shape; codex frames are never
//! iOS-renderable as-is). Tool items surface on `item/started` (in flight); the
//! matching `item/completed` is deduped away (`render_item_once`), acting only as a
//! fallback when no usable started arrived:
//!  - commandExecution    → claude `Bash` `{command}`, plus `CommandStarted` /
//!    `CommandFinished` for the prune guard
//!  - fileChange         → one Frame per change by `kind.type`: add→`Write`,
//!    update→`Edit`, delete→`Bash {command: "rm <path>"}` (no claude delete tool)
//!  - webSearch           → claude `WebSearch` `{query}` (query only on completed)
//!  - mcpToolCall         → claude `mcp__<server>__<tool>`, raw `arguments` as input
//!  - collabAgentToolCall → `spawnAgent` with a prompt → claude `Agent`
//!    `{description}`; any other tool passes through snake_cased (`send_input`)
//!  - agentMessage        → assistant text envelope on `item/completed`
//!  - userMessage         → `UserTurnConsumed(clientId)`, no frame
//!  - contextCompaction   → `CompactBoundary` (the last `tokenUsage.last.totalTokens`)
//!  - thread/tokenUsage/updated → `ContextTokens(last.totalTokens)`
//!  - turn/started        → `TurnStarted(turn id)`
//!  - turn/completed      → `Result` + result envelope by status (completed →
//!    success, failed → error, interrupted → interrupted)
//!  - everything else     → dropped with a debug breadcrumb
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
    agent_capabilities_instructions, claude_assistant_text_envelope, claude_tool_use_envelope,
    effective_cwd, extract_json_string_field, file_nonempty, parse_json_obj,
    probe_with_blocking_auth, slash_command_name, worktree_cwd_prefix, AdapterDescriptor,
    AgentAdapter, AgentEvent, AgentKind, ImportProgress, LastTokensDedup, SessionContext,
    TurnContext, INTERRUPTED_RESULT, MAX_STREAM_FRAME_BYTES, PRUNE_CONTEXT_INSTRUCTION_CODEX,
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

/// Chatter the spawner never reads, opted out of at `initialize` so it isn't
/// even parsed (`turn/diff/updated` carries the whole cumulative diff).
const QUIET_NOTIFICATIONS: &[&str] = &[
    "item/agentMessage/delta",
    "item/reasoning/textDelta",
    "item/reasoning/summaryTextDelta",
    "item/reasoning/summaryPartAdded",
    "item/commandExecution/outputDelta",
    "item/fileChange/outputDelta",
    "item/plan/delta",
    "turn/diff/updated",
    "account/rateLimits/updated",
    "mcpServer/startupStatus/updated",
    "remoteControl/status/changed",
];

/// State of one `codex app-server` session, shared by the stdout reader
/// (`handle_line`) and the stdin writer (`outbound_frames`, `encode_*`).
#[derive(Default)]
pub struct CodexAdapter {
    /// `thread/start` or `thread/resume`, sent once `initialize` is answered.
    thread_request: Option<String>,
    /// The `thread/start` a resume falls back to when its thread has no
    /// rollout: codex writes none until the first turn, so a thread torn down
    /// before that can't be resumed and would fail the chat for good.
    fresh_thread_request: Option<String>,
    handshake: Handshake,
    /// Frames for the writer: the handshake and refusals of server requests.
    outbound: Vec<String>,
    /// The running turn.
    active_turn: Option<String>,
    /// `(uuid, turn id)` of accepted `turn/start`s whose `clientId` echo hasn't
    /// arrived. A turn that ends without recording the input (it failed first,
    /// or a CLI ignores `clientUserMessageId`) still consumes them — otherwise
    /// the chat would stay busy forever: never evicted, and blocking the
    /// spawner's self-update.
    awaiting_echo: Vec<(String, String)>,
    /// The thread failed to open and a turn has reported it: close the session.
    over: bool,
    /// `tokenUsage.last.totalTokens` of the latest model round-trip — after a
    /// compaction, the compacted context (`last.inputTokens` reads 0 there).
    context_tokens: Option<i64>,
    tokens: LastTokensDedup,
    /// Tool items rendered from `item/started` and not completed yet.
    emitted_item_ids: HashSet<String>,
}

#[derive(Default)]
enum Handshake {
    #[default]
    NotStarted,
    Initializing,
    OpeningThread,
    Open(String),
    /// The thread couldn't be opened; every turn fails with this message.
    Failed(String),
}

type Events = SmallVec<[AgentEvent; 2]>;

impl CodexAdapter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Render a tool item once across its started/completed pair: from
    /// `item/started` when that carries a usable payload, else from
    /// `item/completed`. A rendered id stays recorded until its completion,
    /// which can arrive after the turn ended (a command `/stop` left running).
    /// Items without an id bypass dedup.
    fn render_item_once(&mut self, item: &Value, completed: bool) -> Vec<String> {
        let id = item["id"].as_str().unwrap_or("");
        if completed && self.emitted_item_ids.remove(id) {
            return Vec::new();
        }
        let frames = render_tool_item(item);
        if !completed
            && !frames.is_empty()
            && !id.is_empty()
            && !self.emitted_item_ids.insert(id.to_string())
        {
            return Vec::new();
        }
        frames
    }

    fn handle_response(&mut self, id: &Value, msg: &Value, out: &mut Events) {
        let Some(id) = id.as_str() else {
            debug!(%id, "codex response to an id we never sent");
            return;
        };
        let error = msg.get("error").map(|e| {
            e["message"]
                .as_str()
                .unwrap_or("codex request failed")
                .to_string()
        });
        let (kind, key) = id.split_once(':').unwrap_or((id, ""));
        match (kind, error) {
            ("init", None) => {
                self.handshake = Handshake::OpeningThread;
                self.outbound.push(rpc_notification("initialized"));
                self.outbound.extend(self.thread_request.take());
            }
            ("thread", None) => match msg["result"]["thread"]["id"].as_str() {
                Some(thread_id) => {
                    self.handshake = Handshake::Open(thread_id.to_string());
                    out.push(AgentEvent::SessionIdHarvested(thread_id.to_string()));
                }
                None => self.handshake = Handshake::Failed("codex opened no thread".to_string()),
            },
            ("thread", Some(e))
                if e.contains("no rollout found") && self.fresh_thread_request.is_some() =>
            {
                warn!(error = %e, "codex thread has no rollout; starting a new one");
                self.outbound.extend(self.fresh_thread_request.take());
            }
            ("init" | "thread", Some(e)) => {
                warn!(error = %e, "codex could not open the thread");
                self.handshake = Handshake::Failed(format!("codex could not open the thread: {e}"));
            }
            ("turn" | "compact", Some(e)) => self.fail_turn(key, &e, out),
            ("turn", None) => {
                if let Some(turn_id) = msg["result"]["turn"]["id"].as_str() {
                    self.awaiting_echo
                        .push((key.to_string(), turn_id.to_string()));
                }
            }
            ("compact", None) => out.push(AgentEvent::UserTurnConsumed(key.to_string())),
            _ => {}
        }
    }

    /// A turn request codex refused: consumed, so it no longer holds the chat
    /// busy, and answered with an error result.
    fn fail_turn(&mut self, uuid: &str, error: &str, out: &mut Events) {
        let message = match &self.handshake {
            Handshake::Failed(why) => {
                self.over = true;
                why.clone()
            }
            _ => error.to_string(),
        };
        out.push(AgentEvent::UserTurnConsumed(uuid.to_string()));
        if self.active_turn.is_none() {
            out.push(AgentEvent::Result);
        }
        out.push(AgentEvent::Frame(error_result_frame(&message)));
    }

    /// Approvals and elicitations can't reach the user here (turns run with
    /// `approvalPolicy: never`), and an unanswered request hangs the turn.
    fn refuse_server_request(&mut self, id: &Value, method: &str) {
        warn!(%method, "codex asked the spawner for something it can't answer; refusing");
        let reply = json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": { "code": -32601, "message": format!("zucchini does not handle {method}") },
        });
        self.outbound.push(format!("{reply}\n"));
    }

    fn handle_notification(&mut self, method: &str, params: &Value, out: &mut Events) {
        match method {
            // Busy even when no turn of ours started it (a goal continuation),
            // like claude's self-fired turns; its `turn/completed` ends it.
            "turn/started" => {
                self.active_turn = params["turn"]["id"].as_str().map(str::to_string);
                out.extend(self.active_turn.clone().map(AgentEvent::TurnStarted));
            }
            "turn/completed" => {
                self.active_turn = None;
                let turn_id = params["turn"]["id"].as_str().unwrap_or("");
                let (unechoed, rest) = std::mem::take(&mut self.awaiting_echo)
                    .into_iter()
                    .partition(|(_, t)| t == turn_id);
                self.awaiting_echo = rest;
                for (uuid, _) in unechoed {
                    warn!(%uuid, "codex turn ended without echoing this input");
                    out.push(AgentEvent::UserTurnConsumed(uuid));
                }
                out.push(AgentEvent::Result);
                out.push(AgentEvent::Frame(turn_result_frame(&params["turn"])));
            }
            "item/started" => {
                let item = &params["item"];
                let frames = self.render_item_once(item, false);
                out.extend(frames.into_iter().map(AgentEvent::Frame));
                out.extend(live_command(item));
            }
            "item/completed" => self.item_completed(&params["item"], out),
            "thread/tokenUsage/updated" => {
                if let Some(n) = params["tokenUsage"]["last"]["totalTokens"].as_i64() {
                    self.context_tokens = Some(n);
                    out.extend(self.tokens.observe(n).map(AgentEvent::ContextTokens));
                }
            }
            other => debug!(method = %other, "codex notification dropped"),
        }
    }

    fn item_completed(&mut self, item: &Value, out: &mut Events) {
        match item["type"].as_str() {
            Some("userMessage") => {
                if let Some(uuid) = item["clientId"].as_str() {
                    self.awaiting_echo.retain(|(u, _)| u != uuid);
                    out.push(AgentEvent::UserTurnConsumed(uuid.to_string()));
                }
            }
            Some("agentMessage") => {
                let text = item["text"].as_str().unwrap_or("");
                if !text.is_empty() {
                    out.push(AgentEvent::Frame(claude_assistant_text_envelope(text)));
                }
            }
            Some("contextCompaction") => {
                out.extend(self.context_tokens.map(AgentEvent::CompactBoundary));
            }
            _ => {
                // Call-keyed prune cue: only the `prune-context` command's own
                // completion — its output is in the rollout now — may drive
                // the restart, never a sibling's. Emitted after the item's own
                // bubble so the restart can't preempt it.
                let is_command = item["type"] == "commandExecution";
                let is_prune =
                    is_command && crate::prune::value_is_prune_context_call(&item["command"]);
                let frames = self.render_item_once(item, true);
                out.extend(frames.into_iter().map(AgentEvent::Frame));
                if let Some(id) = item["id"].as_str().filter(|_| is_command && !is_prune) {
                    out.push(AgentEvent::CommandFinished(id.to_string()));
                }
                if is_prune {
                    out.push(AgentEvent::ToolResult);
                }
            }
        }
    }
}

impl AgentAdapter for CodexAdapter {
    fn kind(&self) -> AgentKind {
        AgentKind::Codex
    }

    fn prepare_command(&mut self, _ctx: &TurnContext<'_>) -> Result<String> {
        unreachable!("codex runs as a resident")
    }

    fn is_resident(&self) -> bool {
        true
    }

    /// `codex app-server` in the chat's cwd (the shared worktree prefix creates
    /// the worktree on a brand-new chat). Sandbox, model and cwd ride the
    /// thread request instead of flags. Sandboxed = `read-only` with approvals
    /// off, so a refused write fails instead of waiting on a prompt nobody can
    /// answer; otherwise the owner's agent runs unsandboxed, like claude's
    /// `--dangerously-skip-permissions`.
    fn prepare_session_command(&mut self, ctx: &SessionContext<'_>) -> Result<String> {
        let cwd = ctx.cwd();
        let (start, resume) = thread_requests(ctx, effective_cwd(&cwd).as_deref());
        match resume {
            Some(resume) => {
                self.thread_request = Some(resume);
                self.fresh_thread_request = Some(start);
            }
            None => self.thread_request = Some(start),
        }
        Ok(format!("{}codex app-server", worktree_cwd_prefix(&cwd)))
    }

    /// Codex has no system-prompt injection point (`AGENTS.md` would clobber
    /// the user's, `developerInstructions` would have to be byte-stable across
    /// every resume of every older thread), so capabilities ride the first
    /// user message; worktree ignored.
    fn session_first_turn_preamble(&self) -> Option<String> {
        Some(agent_capabilities_instructions(&[], None))
    }

    /// No `--append-system-prompt` either, so the prune nudge rides in on the
    /// first user message. The codex variant omits `--tool-name`: its reads,
    /// greps and edits all run through the shell.
    fn first_turn_prompt_suffix(&self) -> Option<&'static str> {
        Some(PRUNE_CONTEXT_INSTRUCTION_CODEX)
    }

    fn outbound_frames(&mut self) -> Vec<String> {
        if matches!(self.handshake, Handshake::NotStarted) {
            self.handshake = Handshake::Initializing;
            self.outbound.insert(0, initialize_request());
        }
        std::mem::take(&mut self.outbound)
    }

    fn encode_user_turn(&mut self, uuid: &str, text: &str) -> Option<String> {
        let thread_id = match &self.handshake {
            Handshake::Open(id) => id.as_str(),
            // Sent anyway: codex refuses it, and `fail_turn` reports why the
            // thread isn't open.
            Handshake::Failed(_) => "",
            _ => return None,
        };
        Some(if is_compact(text) {
            rpc_request(
                &format!("compact:{uuid}"),
                "thread/compact/start",
                json!({ "threadId": thread_id }),
            )
        } else {
            rpc_request(
                &format!("turn:{uuid}"),
                "turn/start",
                json!({
                    "threadId": thread_id,
                    "clientUserMessageId": uuid,
                    "input": [{ "type": "text", "text": text, "text_elements": [] }],
                }),
            )
        })
    }

    fn exits_on_stdin_eof(&self) -> bool {
        true
    }

    /// `/compact [instructions]` → `thread/compact/start`, which takes no
    /// instructions. Other slash commands are the codex TUI's and reach the
    /// model as text.
    fn is_command_turn(&self, prompt: &str) -> bool {
        is_compact(prompt)
    }

    fn command_interrupts_turn(&self) -> bool {
        false
    }

    fn command_runs_alone(&self, prompt: &str) -> bool {
        is_compact(prompt)
    }

    fn writer_follows_every_line(&self) -> bool {
        true
    }

    fn session_over(&self) -> bool {
        self.over
    }

    fn startup_complete(&self) -> bool {
        matches!(self.handshake, Handshake::Open(_) | Handshake::Failed(_))
    }

    fn handle_line(&mut self, line: String) -> SmallVec<[AgentEvent; 2]> {
        let mut out = Events::new();
        // Skip the parse of a multi-MB line nothing is rendered from.
        if line.len() > MAX_STREAM_FRAME_BYTES && !oversize_line_renders(&line) {
            debug!("codex oversize notification dropped");
            return out;
        }
        let Some(msg) = parse_json_obj(&line) else {
            debug!("codex non-object stdout line dropped");
            return out;
        };
        match (msg["method"].as_str(), msg.get("id")) {
            (Some(method), Some(id)) => self.refuse_server_request(id, method),
            (Some(method), None) => self.handle_notification(method, &msg["params"], &mut out),
            (None, Some(id)) => self.handle_response(id, &msg, &mut out),
            (None, None) => debug!("codex message without method or id dropped"),
        }
        out
    }
}

/// A started shell command a prune restart would kill. The `prune-context`
/// call itself is exempt: it can't block its own prune.
fn live_command(item: &Value) -> Option<AgentEvent> {
    let command = item["command"].as_str()?;
    let id = item["id"].as_str()?;
    let counts = item["type"] == "commandExecution"
        && !crate::prune::value_is_prune_context_call(&item["command"]);
    counts.then(|| AgentEvent::CommandStarted {
        id: id.to_string(),
        command: command.to_string(),
    })
}

fn is_compact(prompt: &str) -> bool {
    slash_command_name(prompt) == Some("compact")
}

/// Whether an oversize line is worth a full parse: tool items (a command's
/// whole output rides on its `item/completed`), the turn's end, responses, and
/// anything with an `id` key — it may be a server request, which must be
/// refused or its turn hangs.
fn oversize_line_renders(line: &str) -> bool {
    line.contains(r#""id""#)
        || matches!(
            extract_json_string_field(line, "method"),
            None | Some("item/started" | "item/completed" | "turn/completed")
        )
}

fn rpc_request(id: &str, method: &str, params: Value) -> String {
    let request = json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params });
    format!("{request}\n")
}

fn rpc_notification(method: &str) -> String {
    format!("{}\n", json!({ "jsonrpc": "2.0", "method": method }))
}

fn initialize_request() -> String {
    rpc_request(
        "init",
        "initialize",
        json!({
            "clientInfo": {
                "name": "zucchini",
                "title": "Zucchini",
                "version": env!("CARGO_PKG_VERSION"),
            },
            "capabilities": {
                "experimentalApi": true,
                "requestAttestation": false,
                "optOutNotificationMethods": QUIET_NOTIFICATIONS,
            },
        }),
    )
}

/// `thread/start`, and the `thread/resume` to send instead when the chat has a
/// session. A resume re-passes every override: codex falls back to the user's
/// config for any it isn't given. `excludeTurns` keeps a long chat's history
/// out of the response.
fn thread_requests(ctx: &SessionContext<'_>, cwd: Option<&str>) -> (String, Option<String>) {
    let mut params = json!({
        "cwd": cwd,
        "model": ctx.model,
        "approvalPolicy": "never",
        "sandbox": if ctx.is_sandboxed { "read-only" } else { "danger-full-access" },
    });
    let start = rpc_request("thread", "thread/start", params.clone());
    let resume = ctx.agent_session_id.map(|thread_id| {
        params["threadId"] = json!(thread_id);
        params["excludeTurns"] = json!(true);
        rpc_request("thread", "thread/resume", params)
    });
    (start, resume)
}

/// Render a codex TOOL item → claude-shape `tool_use` envelopes. Unknown types
/// → empty. A render with an empty primary field returns empty so the
/// started/completed dedup falls through to the completed item (no empty bubble
/// that then suppresses the real one). Field projection only — iOS summarizes
/// just the primary input field, so matching claude's wire shape means no
/// codex-specific branches in iOS.
fn render_tool_item(item: &Value) -> Vec<String> {
    let id = item["id"].as_str().unwrap_or("");
    let non_empty = |field: &str| item[field].as_str().filter(|s| !s.is_empty());
    match item["type"].as_str().unwrap_or("") {
        "commandExecution" => non_empty("command")
            .map(|command| claude_tool_use_envelope(id, "Bash", json!({ "command": command })))
            .into_iter()
            .collect(),
        "fileChange" => render_file_changes(item, id),
        "webSearch" => non_empty("query")
            .map(|query| claude_tool_use_envelope(id, "WebSearch", json!({ "query": query })))
            .into_iter()
            .collect(),
        "mcpToolCall" => render_mcp_tool_call(item, id),
        "collabAgentToolCall" => render_collab_tool_call(item, id),
        _ => Vec::new(),
    }
}

/// One bubble per change — codex bundles N edits in one item.
fn render_file_changes(item: &Value, item_id: &str) -> Vec<String> {
    let Some(changes) = item["changes"].as_array() else {
        return Vec::new();
    };
    changes
        .iter()
        .enumerate()
        .filter_map(|(idx, change)| {
            let path = change["path"].as_str()?;
            // Per-change unique id so iOS can key its row diffing off
            // `tool_use.id` without collisions when N>1.
            let id = if changes.len() == 1 {
                item_id.to_string()
            } else {
                format!("{item_id}.{idx}")
            };
            Some(match change["kind"]["type"].as_str() {
                Some("add") => claude_tool_use_envelope(&id, "Write", json!({ "file_path": path })),
                Some("delete") => claude_tool_use_envelope(
                    &id,
                    "Bash",
                    json!({ "command": format!("rm {path}") }),
                ),
                // "update" + any future kind: an existing file was changed.
                _ => claude_tool_use_envelope(&id, "Edit", json!({ "file_path": path })),
            })
        })
        .collect()
}

/// codex `mcpToolCall` → claude-shape `tool_use` named `mcp__<server>__<tool>`
/// — claude's own MCP naming, so iOS renders the qualified name with no codex
/// branch. `arguments` is a raw JSON object, passed through as `input`.
/// Degrades when a name part is missing; with neither, returns empty so the
/// completed fallback fires instead of a garbage `mcp__`.
fn render_mcp_tool_call(item: &Value, item_id: &str) -> Vec<String> {
    let server = item["server"].as_str().unwrap_or("");
    let tool = item["tool"].as_str().unwrap_or("");
    let name = match (server.is_empty(), tool.is_empty()) {
        (false, false) => format!("mcp__{server}__{tool}"),
        (false, true) => format!("mcp__{server}"),
        (true, false) => format!("mcp__{tool}"),
        (true, true) => return Vec::new(),
    };
    let input = item.get("arguments").cloned().unwrap_or_else(|| json!({}));
    vec![claude_tool_use_envelope(item_id, &name, input)]
}

/// codex `collabAgentToolCall` (sub-agent orchestration) → claude-shape
/// `tool_use`. `spawnAgent` with a prompt → claude's `Agent` `{description}`
/// (iOS shows the sub-agent's task). Any other tool passes through under its
/// snake_case name (what `codex exec` called it) with the routing fields as
/// input (iOS renders the bare name). Empty `tool` → empty (completed fallback).
fn render_collab_tool_call(item: &Value, item_id: &str) -> Vec<String> {
    let tool = item["tool"].as_str().unwrap_or("");
    if tool.is_empty() {
        return Vec::new();
    }
    let prompt = item["prompt"].as_str();
    if tool == "spawnAgent" {
        if let Some(description) = prompt.filter(|p| !p.is_empty()) {
            return vec![claude_tool_use_envelope(
                item_id,
                "Agent",
                json!({ "description": description }),
            )];
        }
    }
    let mut input = serde_json::Map::new();
    if let Some(p) = prompt {
        input.insert("prompt".to_string(), json!(p));
    }
    if let Some(s) = item.get("senderThreadId") {
        input.insert("sender_thread_id".to_string(), s.clone());
    }
    if let Some(r) = item.get("receiverThreadIds") {
        input.insert("receiver_thread_ids".to_string(), r.clone());
    }
    vec![claude_tool_use_envelope(
        item_id,
        &snake_case(tool),
        Value::Object(input),
    )]
}

fn snake_case(name: &str) -> String {
    let mut out = String::with_capacity(name.len() + 4);
    for c in name.chars() {
        if c.is_ascii_uppercase() {
            out.push('_');
            out.push(c.to_ascii_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}

/// The claude-shape `result` envelope for a finished turn — what iOS renders
/// as `[result: success]` / `[result: error]` / "Agent interrupted".
fn turn_result_frame(turn: &Value) -> String {
    match turn["status"].as_str() {
        Some("failed") => {
            error_result_frame(turn["error"]["message"].as_str().unwrap_or("turn failed"))
        }
        Some("interrupted") => INTERRUPTED_RESULT.to_string(),
        _ => json!({ "type": "result", "subtype": "success", "is_error": false }).to_string(),
    }
}

/// Keeps the error message in the stored frame (useful for logs and future
/// clients) without leaking codex's own error shape into chat rows.
fn error_result_frame(message: &str) -> String {
    json!({
        "type": "result",
        "subtype": "error",
        "is_error": true,
        "error": { "message": message },
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
    let Some(home) = crate::home_dir_opt() else {
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
                    agent_kind: DESCRIPTOR.kind,
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
    Some(crate::home_dir_opt()?.join(".codex").join("sessions"))
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
async fn parse_session(path: &Path) -> Result<Option<ParsedSession>> {
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
                        if let Some(frame) = payload.and_then(normalize_persisted_function_call) {
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
                        if let Some(text) = payload.and_then(response_item_message_text) {
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
        let header: Value = serde_json::from_str(first.trim()).ok()?;
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
        let decoded =
            serde_json::from_str::<Value>(a).unwrap_or_else(|_| Value::String(a.to_string()));
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
                    || serde_json::from_str::<Value>(t)
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
    match serde_json::from_str::<Value>(arguments) {
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
/// [`codex_collect_pruned`]) — the SAME two `prune_batch_codex_jsonl` feeds the
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
/// path, via [`crate::prune::select_eligible_ids`]) and `prune_batch_codex_jsonl` (the
/// apply path, via [`crate::prune::rewrite_jsonl_batch_last_only`]). Pushes the matching
/// `function_call`'s `call_id` in DOCUMENT ORDER onto `matched`. A function_call
/// yields at most its own single call_id.
fn codex_collect_matched(entry: &Value, tool_name: &str, needle: &str, matched: &mut Vec<String>) {
    // Same match predicate as the blank pass (via the shared collect helper) so
    // the count can't drift from what gets blanked.
    let mut ids = HashSet::new();
    collect_codex_call_id_if_matches(entry, tool_name, needle, &mut ids);
    matched.extend(ids);
}

/// Pass-1 already-pruned collector shared by `eligible_matches` and
/// `prune_batch_codex_jsonl` (same call sites as [`codex_collect_matched`]). Marks a
/// call ineligible from a `function_call_output` whose `output` is already the
/// `[pruned]` placeholder.
fn codex_collect_pruned(entry: &Value, already_pruned: &mut HashSet<String>) {
    if entry.get("type").and_then(|t| t.as_str()) != Some("response_item") {
        return;
    }
    let Some(payload) = entry.get("payload") else {
        return;
    };
    let pruned = match payload.get("type").and_then(|t| t.as_str()) {
        Some("function_call_output") => {
            payload.get("output").and_then(|o| o.as_str()) == Some(crate::prune::PRUNED_PLACEHOLDER)
        }
        Some("custom_tool_call_output") => payload
            .get("output")
            .is_some_and(|o| *o == pruned_custom_output(o)),
        _ => false,
    };
    if pruned {
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
    entry: &Value,
    tool_name: &str,
    needle: &str,
    ids: &mut HashSet<String>,
) {
    if entry.get("type").and_then(|t| t.as_str()) != Some("response_item") {
        return;
    }
    let Some(payload) = entry.get("payload") else {
        return;
    };
    // codex ≥0.15x "code mode" persists shell calls as `custom_tool_call`
    // `exec` whose freeform `input` is a JS script; it globs as one raw leaf.
    let args_key = match payload.get("type").and_then(|t| t.as_str()) {
        Some("function_call") => "arguments",
        Some("custom_tool_call") => "input",
        _ => return,
    };
    let name = payload.get("name").and_then(|v| v.as_str()).unwrap_or("");
    let arguments = payload.get(args_key).and_then(|v| v.as_str());
    if codex_function_call_matches(name, arguments, tool_name, needle) {
        if let Some(id) = payload.get("call_id").and_then(|v| v.as_str()) {
            ids.insert(id.to_string());
        }
    }
}

/// Single-target, summary-less [`prune_batch_codex_jsonl`], a test shorthand.
#[cfg(test)]
fn prune_codex_jsonl(
    path: &Path,
    tool_name: &str,
    needle: &str,
) -> std::io::Result<crate::prune::PruneStats> {
    prune_batch_codex_jsonl(
        path,
        &[crate::prune::PruneTarget::new(tool_name, needle, "")],
    )
}

/// Batch entry point behind `PRUNE_OPS::prune_batch`: blank the last-only target of
/// every `(tool_name, needle)` in `targets` in ONE read/write via the shared batch
/// driver ([`crate::prune::rewrite_jsonl_batch_last_only`]): `function_call.arguments`
/// → `"{}"` and the paired `function_call_output.output` → `[pruned]`, never deleting
/// a line. The blank runs per entry, so it also catches an output recorded BEFORE its
/// call (`codex_prune_blanks_output_appearing_before_its_call`). Codex has no
/// fail-closed post-scan, so the final entries are ignored.
fn prune_batch_codex_jsonl(
    path: &Path,
    targets: &[crate::prune::PruneTarget],
) -> std::io::Result<crate::prune::PruneStats> {
    let (stats, _final_entries) = crate::prune::rewrite_jsonl_batch_last_only(
        path,
        targets,
        |target, entry, matched| {
            codex_collect_matched(entry, &target.tool_name, &target.needle, matched)
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
    entry: &mut Value,
    pruned_ids: &crate::prune::ChosenTargets,
    outputs_blanked: &mut HashSet<String>,
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
                .map(|id| pruned_ids.contains_key(id))
                .unwrap_or(false);
            if !matched {
                return None;
            }
            // Blank `arguments` (a JSON-encoded string) to the encoded empty
            // object `"{}"`, keeping the wire type (string) intact.
            crate::prune::blank_string_field(payload, "arguments", "{}")
        }
        Some("custom_tool_call") => {
            let id = payload.get("call_id").and_then(|v| v.as_str())?;
            if !pruned_ids.contains_key(id) {
                return None;
            }
            crate::prune::blank_string_field(payload, "input", "")
        }
        Some(kind @ ("function_call_output" | "custom_tool_call_output")) => {
            let custom = kind == "custom_tool_call_output";
            let call_id = payload.get("call_id").and_then(|v| v.as_str());
            let pruned = call_id
                .map(|id| pruned_ids.contains_key(id))
                .unwrap_or(false);
            if !pruned {
                return None;
            }
            let call_id = call_id.map(str::to_string);
            // Only count (and re-serialize) when the output was ACTUALLY blanked
            // this run. `blank_string_field` returns `None` when it was
            // already `[pruned]` (idempotent re-prune), so a re-run over a
            // fully-pruned rollout reports `results_blanked = 0` and skips the
            // timeline frame — matching the claude pruner's `already` guard.
            let freed = if custom {
                blank_custom_output(payload)
            } else {
                crate::prune::blank_string_field(
                    payload,
                    "output",
                    crate::prune::PRUNED_PLACEHOLDER,
                )
            }?;
            if let Some(id) = call_id {
                outputs_blanked.insert(id);
            }
            Some(freed)
        }
        _ => None,
    }
}

/// The `[pruned]` placeholder in the shape of `output`: a `custom_tool_call_output`
/// carries either a plain string or (codex ≥0.15x) an `input_text` item array, and
/// the placeholder keeps that shape so resume deserializes it as before.
fn pruned_custom_output(output: &Value) -> Value {
    let placeholder = crate::prune::PRUNED_PLACEHOLDER;
    if output.is_array() {
        serde_json::json!([{ "type": "input_text", "text": placeholder }])
    } else {
        Value::String(placeholder.to_string())
    }
}

/// Blank a `custom_tool_call_output.output` to [`pruned_custom_output`]. `None` when
/// absent or already blanked (idempotent); else the freed serialized bytes.
fn blank_custom_output(payload: &mut serde_json::Map<String, Value>) -> Option<usize> {
    let existing = payload.get("output")?;
    let blank = pruned_custom_output(existing);
    if *existing == blank {
        return None;
    }
    let freed = existing
        .to_string()
        .len()
        .saturating_sub(blank.to_string().len());
    payload.insert("output".into(), blank);
    Some(freed)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Returns the SmallVec as a Vec of human-readable tags + payload for
    /// easy assertion equality. Delegates to the shared stringifier in
    /// `adapter.rs` so the event→tag mapping has a single source of truth.
    fn run(adapter: &mut CodexAdapter, line: &str) -> Vec<String> {
        crate::adapter::stringify_events(adapter.handle_line(line.to_string()))
    }

    fn frame_value(event: &str) -> Value {
        crate::adapter::frame_value(event)
    }

    fn session_ctx(agent_session_id: Option<&str>) -> SessionContext<'_> {
        SessionContext {
            chat_id: "00000000-0000-0000-0000-000000000000",
            project_path: Some("/tmp/proj"),
            worktree: false,
            agent_session_id,
            is_sandboxed: false,
            model: None,
            session_name: None,
            user_system_prompt: None,
        }
    }

    fn rpc(frame: &str) -> Value {
        assert!(
            frame.ends_with('\n'),
            "frames are newline-terminated: {frame}"
        );
        serde_json::from_str(frame.trim_end()).unwrap()
    }

    // Real codex-cli 0.154 lines (tmp/spike-logs/), ids swapped for ours where
    // the spawner picks them.
    const THREAD: &str = "01a0ccc6-1c43-75b3-b5c7-637798bff872";
    const INIT_OK: &str = r#"{"id":"init","result":{"userAgent":"zucchini/0.154.0 (Mac OS 26.6.2; arm64) unknown (zucchini; 0.0.1)","codexHome":"/Users/ivanpchelintsev/.codex","platformFamily":"unix","platformOs":"macos"}}"#;
    const THREAD_OK: &str = r#"{"id":"thread","result":{"thread":{"id":"01a0ccc6-1c43-75b3-b5c7-637798bff872","sessionId":"01a0ccc6-1c43-75b3-b5c7-637798bff872","preview":"","ephemeral":false,"historyMode":"paginated","modelProvider":"openai","model":"gpt-6-astra","reasoningEffort":"low","status":{"type":"idle"},"path":"/Users/ivanpchelintsev/.codex/sessions/2026/09/23/rollout-2026-09-23T06-38-50-01a0ccc6-1c43-75b3-b5c7-637798bff872.jsonl","cwd":"/tmp/codex-spike","cliVersion":"0.154.0","originator":"zucchini","source":"vscode","turns":[]},"model":"gpt-6-astra","cwd":"/tmp/codex-spike","approvalPolicy":"never","approvalsReviewer":"user","sandbox":{"type":"dangerFullAccess"},"reasoningEffort":"low"}}"#;
    const TURN_STARTED: &str = r#"{"method":"turn/started","params":{"threadId":"01a0ccc6-1c43-75b3-b5c7-637798bff872","turn":{"id":"01a0ccc6-1d09-7902-a2b6-7d39a30c6290","items":[],"itemsView":"notLoaded","status":"inProgress","error":null,"startedAt":1790141930,"completedAt":null,"durationMs":null}},"emittedAtMs":1790141930784}"#;
    const TURN_COMPLETED: &str = r#"{"method":"turn/completed","params":{"threadId":"01a0ccc6-1c43-75b3-b5c7-637798bff872","turn":{"id":"01a0ccc6-1d09-7902-a2b6-7d39a30c6290","items":[{"type":"agentMessage","id":"msg_0b690fad2e21428f016ab365f3163c87d289aea24e41508b56","text":"done","phase":"final_answer","memoryCitation":null,"delivery":null,"questions":null}],"itemsView":"summary","status":"completed","error":null,"startedAt":1790141930,"completedAt":1790141939,"durationMs":8405}},"emittedAtMs":1790141939174}"#;
    const TURN_FAILED: &str = r#"{"method":"turn/completed","params":{"threadId":"01a0cccf-0fbb-7340-af02-1b26c6ee2f69","turn":{"id":"01a0cccf-3516-7842-bb38-1c23c4249439","items":[],"itemsView":"notLoaded","status":"failed","error":{"message":"{\"type\":\"error\",\"status\":400,\"error\":{\"type\":\"invalid_request_error\",\"message\":\"The 'no-such-model-xyz' model is not supported when using Codex with a ChatGPT account.\"}}","codexErrorInfo":"other","additionalDetails":null,"misalignment":null},"startedAt":1790142526,"completedAt":1790142528,"durationMs":2008}},"emittedAtMs":1790142528763}"#;
    const TURN_INTERRUPTED: &str = r#"{"method":"turn/completed","params":{"threadId":"01a0cccf-6a10-75a2-bda0-84a0c78626b3","turn":{"id":"01a0cccf-6a66-7112-a065-afd89626f6da","items":[],"itemsView":"notLoaded","status":"interrupted","error":null,"startedAt":1790142540,"completedAt":1790142547,"durationMs":6738}},"emittedAtMs":1790142547132}"#;
    const USER_ECHO: &str = r#"{"method":"item/completed","params":{"item":{"type":"userMessage","id":"01a0cccf-18c6-7831-93df-a6cc121a59b5","clientId":"cli-1","content":[{"type":"text","text":"Run the shell command `sleep 20` (wait for it to finish) and then reply FIRST.","text_elements":[]}]},"threadId":"01a0cccf-0fbb-7340-af02-1b26c6ee2f69","turnId":"01a0cccf-1040-7e23-b565-ff9c831e0521","completedAtMs":1790142519494},"emittedAtMs":1790142519498}"#;
    const USER_NO_CLIENT_ID: &str = r#"{"method":"item/completed","params":{"item":{"type":"userMessage","id":"01a0ccc6-2627-7530-83bf-35f0d4c4dc87","clientId":null,"content":[{"type":"text","text":"hi","text_elements":[]}]},"threadId":"01a0ccc6-1c43-75b3-b5c7-637798bff872","turnId":"01a0ccc6-1d09-7902-a2b6-7d39a30c6290","completedAtMs":1790141933095},"emittedAtMs":1790141933098}"#;
    const AGENT_MESSAGE_STARTED: &str = r#"{"method":"item/started","params":{"item":{"type":"agentMessage","id":"msg_0b690fad2e21428f016ab365ef131087d288e8aad9a75f024b","text":"","phase":"commentary","memoryCitation":null,"delivery":null,"questions":null},"threadId":"01a0ccc6-1c43-75b3-b5c7-637798bff872","turnId":"01a0ccc6-1d09-7902-a2b6-7d39a30c6290","startedAtMs":1790141935005},"emittedAtMs":1790141935005}"#;
    const AGENT_MESSAGE: &str = r#"{"method":"item/completed","params":{"item":{"type":"agentMessage","id":"msg_0b690fad2e21428f016ab365ef131087d288e8aad9a75f024b","text":"I’ll run the command and create `note.txt`.\n","phase":"commentary","memoryCitation":null,"delivery":null,"questions":null},"threadId":"01a0ccc6-1c43-75b3-b5c7-637798bff872","turnId":"01a0ccc6-1d09-7902-a2b6-7d39a30c6290","completedAtMs":1790141935989},"emittedAtMs":1790141935990}"#;
    const COMMAND_STARTED: &str = r#"{"method":"item/started","params":{"item":{"type":"commandExecution","id":"exec-565056be-aca3-4c8e-85fc-1f837d1825d0","pluginId":null,"scriptPath":null,"command":"/bin/zsh -lc 'echo hello && ls'","cwd":"/tmp/codex-spike","processId":"42287","source":"unifiedExecStartup","status":"inProgress","commandActions":[{"type":"listFiles","command":"ls","path":null}],"aggregatedOutput":null,"exitCode":null,"durationMs":null},"threadId":"01a0ccc6-1c43-75b3-b5c7-637798bff872","turnId":"01a0ccc6-1d09-7902-a2b6-7d39a30c6290","startedAtMs":1790141937567},"emittedAtMs":1790141937569}"#;
    const COMMAND_COMPLETED: &str = r#"{"method":"item/completed","params":{"item":{"type":"commandExecution","id":"exec-565056be-aca3-4c8e-85fc-1f837d1825d0","pluginId":null,"scriptPath":null,"command":"/bin/zsh -lc 'echo hello && ls'","cwd":"/tmp/codex-spike","processId":"42287","source":"unifiedExecStartup","status":"completed","commandActions":[{"type":"listFiles","command":"ls","path":null}],"aggregatedOutput":"hello\n","exitCode":0,"durationMs":0},"threadId":"01a0ccc6-1c43-75b3-b5c7-637798bff872","turnId":"01a0ccc6-1d09-7902-a2b6-7d39a30c6290","completedAtMs":1790141937567},"emittedAtMs":1790141937570}"#;
    const FILE_CHANGE_STARTED: &str = r#"{"method":"item/started","params":{"item":{"type":"fileChange","id":"exec-05d666b1-db56-4043-96bb-37b7f8274ea7","changes":[{"path":"/tmp/codex-spike/note.txt","kind":{"type":"add"},"diff":"hi\n"}],"status":"inProgress"},"threadId":"01a0ccc6-1c43-75b3-b5c7-637798bff872","turnId":"01a0ccc6-1d09-7902-a2b6-7d39a30c6290","startedAtMs":1790141937574},"emittedAtMs":1790141937575}"#;
    const WEB_SEARCH_STARTED: &str = r#"{"method":"item/started","params":{"item":{"type":"webSearch","id":"exec-dee48c2b-3563-4ebb-81fe-f288de91aa60","query":"","action":null,"results":null},"threadId":"01a0ccca-7626-7971-818c-a8f9ff5cc90f","turnId":"01a0ccca-7695-72a1-ae08-2ac1524dd00e","startedAtMs":1790142224261},"emittedAtMs":1790142224262}"#;
    const WEB_SEARCH_COMPLETED: &str = r#"{"method":"item/completed","params":{"item":{"type":"webSearch","id":"exec-dee48c2b-3563-4ebb-81fe-f288de91aa60","query":"site:github.com/openai/codex/releases latest codex cli release","action":{"type":"search","query":"site:github.com/openai/codex/releases latest codex cli release","queries":null},"results":[]},"threadId":"01a0ccca-7626-7971-818c-a8f9ff5cc90f","turnId":"01a0ccca-7695-72a1-ae08-2ac1524dd00e","completedAtMs":1790142224900},"emittedAtMs":1790142224901}"#;
    const MCP_STARTED: &str = r#"{"method":"item/started","params":{"item":{"type":"mcpToolCall","id":"exec-73f8c546-00bd-4d39-b1ca-d7edc3e73c8c","server":"node_repl","tool":"js","status":"inProgress","arguments":{"code":"nodeRepl.write(2+2)","title":"Evaluate 2+2"},"appContext":null,"pluginId":null,"readOnlyHint":true,"result":null,"error":null,"durationMs":null},"threadId":"01a0ccd0-0188-7071-9eff-0a3429828ea5","turnId":"01a0ccd0-01ed-7761-8dc4-9c39a1758dc1","startedAtMs":1790142589227},"emittedAtMs":1790142589229}"#;
    const COMPACT_TOKENS: &str = r#"{"method":"thread/tokenUsage/updated","params":{"threadId":"01a0ccc8-f265-7603-9514-213024507e6c","turnId":"01a0ccc9-1c1b-7b72-88b9-5be8efb5dcdf","tokenUsage":{"total":{"totalTokens":32404,"inputTokens":32350,"cachedInputTokens":28416,"cacheWriteInputTokens":0,"outputTokens":54,"reasoningOutputTokens":0},"last":{"totalTokens":5575,"inputTokens":0,"cachedInputTokens":0,"cacheWriteInputTokens":0,"outputTokens":0,"reasoningOutputTokens":0},"modelContextWindow":258400}},"emittedAtMs":1790142133096}"#;
    const TURN_TOKENS: &str = r#"{"method":"thread/tokenUsage/updated","params":{"threadId":"01a0ccc6-1c43-75b3-b5c7-637798bff872","turnId":"01a0ccc6-1d09-7902-a2b6-7d39a30c6290","tokenUsage":{"total":{"totalTokens":15917,"inputTokens":15844,"cachedInputTokens":12928,"cacheWriteInputTokens":0,"outputTokens":73,"reasoningOutputTokens":0},"last":{"totalTokens":15917,"inputTokens":15844,"cachedInputTokens":12928,"cacheWriteInputTokens":0,"outputTokens":73,"reasoningOutputTokens":0},"modelContextWindow":258400}},"emittedAtMs":1790141937581}"#;
    const COMPACTION_DONE: &str = r#"{"method":"item/completed","params":{"item":{"type":"contextCompaction","id":"01a0ccc9-1c2a-7b30-8baf-5e2f3bc113d2"},"threadId":"01a0ccc8-f265-7603-9514-213024507e6c","turnId":"01a0ccc9-1c1b-7b72-88b9-5be8efb5dcdf","completedAtMs":1790142133096},"emittedAtMs":1790142133098}"#;

    /// A session through its handshake with the thread open.
    fn open_session() -> CodexAdapter {
        let mut a = CodexAdapter::new();
        a.prepare_session_command(&session_ctx(None)).unwrap();
        a.outbound_frames();
        run(&mut a, INIT_OK);
        a.outbound_frames();
        run(&mut a, THREAD_OK);
        a
    }

    #[test]
    fn handshake_opens_the_thread_before_any_turn() {
        let mut a = CodexAdapter::new();
        let cmd = a.prepare_session_command(&session_ctx(None)).unwrap();
        assert_eq!(cmd, "cd '/tmp/proj' && codex app-server");
        assert!(a.encode_user_turn("u1", "hi").is_none(), "no thread yet");

        let init = a.outbound_frames();
        assert_eq!(init.len(), 1);
        let init = rpc(&init[0]);
        assert_eq!(
            (init["id"].as_str(), init["method"].as_str()),
            (Some("init"), Some("initialize"))
        );
        assert_eq!(init["params"]["clientInfo"]["name"], "zucchini");
        assert!(a.outbound_frames().is_empty(), "initialize goes out once");

        assert!(run(&mut a, INIT_OK).is_empty());
        let frames = a.outbound_frames();
        assert_eq!(
            rpc(&frames[0]),
            json!({ "jsonrpc": "2.0", "method": "initialized" })
        );
        let start = rpc(&frames[1]);
        assert_eq!(start["method"], "thread/start");
        assert_eq!(
            start["params"],
            json!({
                "cwd": "/tmp/proj",
                "model": null,
                "approvalPolicy": "never",
                "sandbox": "danger-full-access",
            })
        );
        assert!(
            a.encode_user_turn("u1", "hi").is_none(),
            "thread not open yet"
        );
        assert!(
            !a.startup_complete(),
            "handshake chatter doesn't stop the startup watchdog"
        );

        assert_eq!(
            run(&mut a, THREAD_OK),
            vec![format!("SessionIdHarvested({THREAD})")]
        );
        assert!(a.startup_complete());
        let turn = rpc(&a.encode_user_turn("u1", "hello \"there\"").unwrap());
        assert_eq!(turn["id"], "turn:u1");
        assert_eq!(turn["method"], "turn/start");
        assert_eq!(
            turn["params"],
            json!({
                "threadId": THREAD,
                "clientUserMessageId": "u1",
                "input": [{ "type": "text", "text": "hello \"there\"", "text_elements": [] }],
            })
        );
    }

    #[test]
    fn a_chat_with_a_session_resumes_its_thread_with_every_override() {
        let mut a = CodexAdapter::new();
        let ctx = SessionContext {
            is_sandboxed: true,
            model: Some("gpt-6-astra"),
            ..session_ctx(Some(THREAD))
        };
        a.prepare_session_command(&ctx).unwrap();
        a.outbound_frames();
        run(&mut a, INIT_OK);
        let resume = rpc(&a.outbound_frames()[1]);
        assert_eq!(resume["method"], "thread/resume");
        assert_eq!(
            resume["params"],
            json!({
                "threadId": THREAD,
                "excludeTurns": true,
                "cwd": "/tmp/proj",
                "model": "gpt-6-astra",
                "approvalPolicy": "never",
                "sandbox": "read-only",
            })
        );
    }

    #[test]
    fn a_new_worktree_chat_creates_the_worktree_and_runs_the_thread_in_it() {
        let wt = crate::adapter::worktree_for("00000000-0000-0000-0000-000000000000");
        let wt_dir = wt.dir.to_string_lossy().into_owned();
        let mut a = CodexAdapter::new();
        let ctx = SessionContext {
            worktree: true,
            ..session_ctx(None)
        };
        let cmd = a.prepare_session_command(&ctx).unwrap();
        assert_eq!(
            cmd,
            format!(
                "git -C '/tmp/proj' worktree add '{wt_dir}' -b {} && cd '{wt_dir}' && codex app-server",
                wt.branch
            )
        );
        a.outbound_frames();
        run(&mut a, INIT_OK);
        assert_eq!(
            rpc(&a.outbound_frames()[1])["params"]["cwd"],
            wt_dir.as_str()
        );
    }

    #[test]
    fn a_thread_that_fails_to_open_fails_every_turn_with_the_reason() {
        let mut a = CodexAdapter::new();
        a.prepare_session_command(&session_ctx(Some(THREAD)))
            .unwrap();
        a.outbound_frames();
        run(&mut a, INIT_OK);
        a.outbound_frames();
        run(
            &mut a,
            r#"{"id":"thread","error":{"code":-32600,"message":"permission denied"}}"#,
        );
        assert!(a.startup_complete(), "a failed thread ends the startup");
        let turn = rpc(&a.encode_user_turn("u1", "hi").expect("sent to be refused"));
        assert_eq!(turn["params"]["threadId"], "");
        let events = run(
            &mut a,
            r#"{"id":"turn:u1","error":{"code":-32600,"message":"thread not found"}}"#,
        );
        assert_eq!(events[..2], ["UserTurnConsumed(u1)", "Result"]);
        assert_eq!(
            frame_value(&events[2])["error"]["message"],
            "codex could not open the thread: permission denied"
        );
    }

    /// A thread torn down before its first turn has no rollout to resume
    /// (real codex 0.154 error): a new thread replaces it.
    #[test]
    fn a_thread_without_a_rollout_is_replaced_by_a_new_one() {
        let mut a = CodexAdapter::new();
        a.prepare_session_command(&session_ctx(Some(THREAD)))
            .unwrap();
        a.outbound_frames();
        run(&mut a, INIT_OK);
        a.outbound_frames();
        assert!(run(
            &mut a,
            r#"{"id":"thread","error":{"code":-32600,"message":"no rollout found for thread id 01a0ccc6-1c43-75b3-b5c7-637798bff872"}}"#,
        )
        .is_empty());
        let start = rpc(&a.outbound_frames()[0]);
        assert_eq!(start["method"], "thread/start");
        assert!(start["params"].get("threadId").is_none());
        assert!(!a.startup_complete(), "still opening");
        assert_eq!(
            run(&mut a, THREAD_OK),
            vec![format!("SessionIdHarvested({THREAD})")]
        );
        assert!(a.startup_complete());
    }

    /// A refused turn must not hold the chat busy; inside a running turn it
    /// doesn't end that turn.
    #[test]
    fn a_refused_turn_is_consumed_and_answered_with_an_error() {
        let refused = r#"{"error":{"code":-32600,"message":"bad input"},"id":"turn:u2"}"#;
        let mut a = open_session();
        let events = run(&mut a, refused);
        assert_eq!(events[..2], ["UserTurnConsumed(u2)", "Result"]);
        assert_eq!(frame_value(&events[2])["subtype"], "error");

        run(&mut a, TURN_STARTED);
        let in_turn = run(&mut a, refused);
        assert_eq!(in_turn[0], "UserTurnConsumed(u2)");
        assert!(!in_turn.contains(&"Result".to_string()), "{in_turn:?}");
    }

    #[test]
    fn the_client_id_echo_marks_the_turn_consumed() {
        let mut a = open_session();
        assert_eq!(run(&mut a, USER_ECHO), vec!["UserTurnConsumed(cli-1)"]);
        assert!(run(&mut a, USER_NO_CLIENT_ID).is_empty());
    }

    #[test]
    fn an_interrupted_turn_renders_as_interrupted() {
        let mut a = open_session();
        run(&mut a, TURN_STARTED);
        let events = run(&mut a, TURN_INTERRUPTED);
        assert_eq!(events[0], "Result");
        assert_eq!(events[1], format!("Frame({INTERRUPTED_RESULT})"));
    }

    /// A turn that ends without echoing an accepted input still consumes it,
    /// or the chat would stay busy forever.
    #[test]
    fn an_unechoed_input_is_consumed_when_its_turn_ends() {
        let mut a = open_session();
        let accepted = r#"{"id":"turn:u1","result":{"turn":{"id":"01a0ccc6-1d09-7902-a2b6-7d39a30c6290","items":[],"itemsView":"notLoaded","status":"inProgress","error":null}}}"#;
        let echoed = accepted.replace("turn:u1", "turn:cli-1");
        assert!(run(&mut a, accepted).is_empty());
        assert!(run(&mut a, &echoed).is_empty());
        run(&mut a, TURN_STARTED);
        assert_eq!(run(&mut a, USER_ECHO), vec!["UserTurnConsumed(cli-1)"]);
        let events = run(&mut a, TURN_COMPLETED);
        assert_eq!(events[..2], ["UserTurnConsumed(u1)", "Result"]);
        assert_eq!(run(&mut a, TURN_COMPLETED)[0], "Result", "only once");
    }

    /// A thread that won't open reports it on the next turn, then ends the
    /// session so the next message respawns.
    #[test]
    fn a_failed_thread_ends_the_session_after_reporting() {
        let mut a = CodexAdapter::new();
        a.prepare_session_command(&session_ctx(Some(THREAD)))
            .unwrap();
        a.outbound_frames();
        run(
            &mut a,
            r#"{"id":"init","error":{"code":-32600,"message":"boom"}}"#,
        );
        assert!(!a.session_over(), "not before a turn has reported it");
        run(
            &mut a,
            r#"{"id":"turn:u1","error":{"code":-32600,"message":"Not initialized"}}"#,
        );
        assert!(a.session_over());
    }

    #[test]
    fn compact_is_a_command_turn_that_waits_for_idle() {
        let mut a = open_session();
        assert!(a.is_command_turn("/compact"));
        assert!(a.is_command_turn(" /compact keep the plan"));
        assert!(!a.is_command_turn("/init"));
        assert!(!a.is_command_turn("please /compact"));
        assert!(!a.command_interrupts_turn());
        assert!(a.command_runs_alone("/compact keep the plan"));

        let compact = rpc(&a.encode_user_turn("u3", "/compact keep the plan").unwrap());
        assert_eq!(compact["id"], "compact:u3");
        assert_eq!(compact["method"], "thread/compact/start");
        assert_eq!(compact["params"], json!({ "threadId": THREAD }));
        assert_eq!(
            run(&mut a, r#"{"id":"compact:u3","result":{}}"#),
            vec!["UserTurnConsumed(u3)"]
        );
        // After compaction `last.inputTokens` is 0; `last.totalTokens` is the
        // compacted context.
        assert_eq!(run(&mut a, COMPACT_TOKENS), vec!["ContextTokens(5575)"]);
        assert_eq!(run(&mut a, COMPACTION_DONE), vec!["CompactBoundary(5575)"]);
    }

    #[test]
    fn token_usage_drives_the_context_gauge_deduped() {
        let mut a = open_session();
        assert_eq!(run(&mut a, TURN_TOKENS), vec!["ContextTokens(15917)"]);
        assert!(run(&mut a, TURN_TOKENS).is_empty());
    }

    #[test]
    fn turn_completed_emits_result_then_its_frame() {
        let mut a = open_session();
        let events = run(&mut a, TURN_COMPLETED);
        assert_eq!(events[0], "Result");
        assert_eq!(
            frame_value(&events[1]),
            json!({ "type": "result", "subtype": "success", "is_error": false })
        );
        let failed_events = run(&mut a, TURN_FAILED);
        assert_eq!(failed_events[0], "Result");
        let failed = frame_value(&failed_events[1]);
        assert_eq!(
            (failed["subtype"].as_str(), failed["is_error"].as_bool()),
            (Some("error"), Some(true))
        );
        assert!(failed["error"]["message"]
            .as_str()
            .unwrap()
            .contains("no-such-model-xyz"));
    }

    #[test]
    fn agent_message_renders_on_completed_only() {
        let mut a = open_session();
        assert!(run(&mut a, AGENT_MESSAGE_STARTED).is_empty());
        let events = run(&mut a, AGENT_MESSAGE);
        assert_eq!(events.len(), 1);
        let v = frame_value(&events[0]);
        assert_eq!(v["type"], "assistant");
        assert_eq!(
            v["message"]["content"][0],
            json!({ "type": "text", "text": "I\u{2019}ll run the command and create `note.txt`.\n" })
        );
    }

    #[test]
    fn a_command_surfaces_in_flight_and_its_completion_is_deduped() {
        let mut a = open_session();
        let id = "exec-565056be-aca3-4c8e-85fc-1f837d1825d0";
        let events = run(&mut a, COMMAND_STARTED);
        assert_eq!(events[1], format!("CommandStarted({id})"));
        let v = frame_value(&events[0]);
        assert_eq!(
            v["message"]["content"][0],
            json!({
                "type": "tool_use",
                "id": id,
                "name": "Bash",
                "input": { "command": "/bin/zsh -lc 'echo hello && ls'" },
            })
        );
        assert_eq!(
            run(&mut a, COMMAND_COMPLETED),
            vec![format!("CommandFinished({id})")]
        );
        // A completion with no started renders.
        let unstarted = run(&mut a, COMMAND_COMPLETED);
        assert_eq!(unstarted.len(), 2);
        assert!(unstarted[0].starts_with("Frame("), "{unstarted:?}");
    }

    /// A command `/stop` left running completes after its turn ended: still
    /// one bubble (real sequence, `tmp/spike-logs/int3.jsonl`).
    #[test]
    fn a_command_completing_after_its_turn_renders_once() {
        let mut a = open_session();
        run(&mut a, COMMAND_STARTED);
        run(&mut a, TURN_INTERRUPTED);
        let events = run(&mut a, COMMAND_COMPLETED);
        assert!(
            !events.iter().any(|e| e.starts_with("Frame(")),
            "{events:?}"
        );
    }

    /// A turn nobody sent (a goal continuation) is busy like our own.
    #[test]
    fn every_started_turn_is_announced() {
        let mut a = open_session();
        assert_eq!(
            run(&mut a, TURN_STARTED),
            vec!["TurnStarted(01a0ccc6-1d09-7902-a2b6-7d39a30c6290)"]
        );
        let mut st = crate::agent::SessionState::default();
        crate::agent::reduce(
            &mut st,
            &AgentEvent::TurnStarted("01a0ccc6-1d09-7902-a2b6-7d39a30c6290".into()),
        );
        assert!(st.running());
        crate::agent::reduce(&mut st, &AgentEvent::Result);
        assert!(!st.running());
    }

    /// Running commands block a prune but never keep the chat busy; the prune
    /// call itself doesn't count.
    #[test]
    fn running_commands_block_a_prune_without_making_the_chat_busy() {
        let mut a = open_session();
        let mut st = crate::agent::SessionState::default();
        let prune = COMMAND_STARTED.replace(
            "/bin/zsh -lc 'echo hello && ls'",
            r#"/bin/zsh -lc '\"$ZUCCHINI_SPAWNER_BIN\" prune-context --args x --summary y'"#,
        );
        for ev in a.handle_line(prune) {
            crate::agent::reduce(&mut st, &ev);
        }
        assert!(st.live_commands.is_empty(), "the prune call is exempt");
        for ev in a.handle_line(COMMAND_STARTED.to_string()) {
            crate::agent::reduce(&mut st, &ev);
        }
        assert_eq!(st.live_commands.len(), 1);
        assert!(!st.running());
        for ev in a.handle_line(COMMAND_COMPLETED.to_string()) {
            crate::agent::reduce(&mut st, &ev);
        }
        assert!(st.live_commands.is_empty());
    }

    #[test]
    fn only_the_prune_context_command_completion_cues_the_prune() {
        let mut a = open_session();
        let prune = COMMAND_COMPLETED.replace(
            "/bin/zsh -lc 'echo hello && ls'",
            r#"/bin/zsh -lc '\"$ZUCCHINI_SPAWNER_BIN\" prune-context --args \"*.log\" --summary x'"#,
        );
        let events = run(&mut a, &prune);
        assert_eq!(events.len(), 2, "{events:?}");
        assert!(events[0].starts_with("Frame("));
        assert_eq!(events[1], "ToolResult");
        assert!(!run(&mut a, TURN_COMPLETED).contains(&"ToolResult".to_string()));
        assert!(!run(&mut a, COMMAND_COMPLETED).contains(&"ToolResult".to_string()));
    }

    #[test]
    fn file_changes_map_by_kind_type() {
        let mut a = open_session();
        let events = run(&mut a, FILE_CHANGE_STARTED);
        let v = frame_value(&events[0]);
        assert_eq!(v["message"]["content"][0]["name"], "Write");
        assert_eq!(
            v["message"]["content"][0]["input"],
            json!({ "file_path": "/tmp/codex-spike/note.txt" })
        );
        assert!(run(
            &mut a,
            &FILE_CHANGE_STARTED.replace("item/started", "item/completed")
        )
        .is_empty());

        let multi = r#"{"method":"item/started","params":{"item":{"type":"fileChange","id":"fc","changes":[{"path":"/a","kind":{"type":"update","move_path":null},"diff":""},{"path":"/b","kind":{"type":"delete"},"diff":""}],"status":"inProgress"}}}"#;
        let multi_events = run(&mut a, multi);
        let edit = frame_value(&multi_events[0]);
        assert_eq!(edit["message"]["content"][0]["id"], "fc.0");
        assert_eq!(edit["message"]["content"][0]["name"], "Edit");
        let rm = frame_value(&multi_events[1]);
        assert_eq!(rm["message"]["content"][0]["id"], "fc.1");
        assert_eq!(
            rm["message"]["content"][0]["input"],
            json!({ "command": "rm /b" })
        );
    }

    #[test]
    fn web_search_renders_once_its_query_is_known() {
        let mut a = open_session();
        assert!(run(&mut a, WEB_SEARCH_STARTED).is_empty());
        let events = run(&mut a, WEB_SEARCH_COMPLETED);
        let v = frame_value(&events[0]);
        assert_eq!(v["message"]["content"][0]["name"], "WebSearch");
        assert_eq!(
            v["message"]["content"][0]["input"],
            json!({ "query": "site:github.com/openai/codex/releases latest codex cli release" })
        );
    }

    #[test]
    fn mcp_tool_call_uses_claude_mcp_naming() {
        let mut a = open_session();
        let v = frame_value(&run(&mut a, MCP_STARTED)[0]);
        assert_eq!(v["message"]["content"][0]["name"], "mcp__node_repl__js");
        assert_eq!(
            v["message"]["content"][0]["input"],
            json!({ "code": "nodeRepl.write(2+2)", "title": "Evaluate 2+2" })
        );
    }

    /// Schema-only shapes (collabAgentToolCall never fired in the spike).
    #[test]
    fn collab_tools_map_to_agent_or_their_exec_names() {
        let mut a = open_session();
        let spawn = r#"{"method":"item/started","params":{"item":{"type":"collabAgentToolCall","id":"c1","tool":"spawnAgent","status":"inProgress","senderThreadId":"t0","receiverThreadIds":[],"prompt":"audit the parser","model":null}}}"#;
        let spawned = frame_value(&run(&mut a, spawn)[0]);
        assert_eq!(spawned["message"]["content"][0]["name"], "Agent");
        assert_eq!(
            spawned["message"]["content"][0]["input"],
            json!({ "description": "audit the parser" })
        );
        let send = r#"{"method":"item/started","params":{"item":{"type":"collabAgentToolCall","id":"c2","tool":"sendInput","status":"inProgress","senderThreadId":"t0","receiverThreadIds":["t1"],"prompt":"go on","model":null}}}"#;
        let sent = frame_value(&run(&mut a, send)[0]);
        assert_eq!(sent["message"]["content"][0]["name"], "send_input");
        assert_eq!(
            sent["message"]["content"][0]["input"],
            json!({ "prompt": "go on", "sender_thread_id": "t0", "receiver_thread_ids": ["t1"] })
        );
    }

    #[test]
    fn a_server_request_is_refused_so_the_turn_cannot_hang() {
        let mut a = open_session();
        let request = r#"{"method":"item/commandExecution/requestApproval","id":7,"params":{"threadId":"t","turnId":"u","itemId":"i"}}"#;
        assert!(run(&mut a, request).is_empty());
        let reply = rpc(&a.outbound_frames()[0]);
        assert_eq!(reply["id"], 7);
        assert_eq!(reply["error"]["code"], -32601);
    }

    #[test]
    fn noise_is_dropped() {
        let mut a = open_session();
        for line in [
            r#"{"method":"thread/status/changed","params":{"threadId":"01a0ccc6-1c43-75b3-b5c7-637798bff872","status":{"type":"idle"}},"emittedAtMs":1790141939174}"#,
            r#"{"method":"item/completed","params":{"item":{"type":"reasoning","id":"rs_0f03","summary":[],"content":[]},"threadId":"t","turnId":"u","completedAtMs":1}}"#,
            r#"{"method":"warning","params":{"threadId":"t","message":"Model metadata for `x` not found."}}"#,
            r#"{"id":"turn:u9","result":{"turn":{"id":"u","items":[],"itemsView":"notLoaded","status":"inProgress","error":null}}}"#,
            "not json",
        ] {
            assert!(run(&mut a, line).is_empty(), "{line}");
        }
    }

    #[test]
    fn an_oversize_line_is_parsed_only_when_something_renders_from_it() {
        let mut a = open_session();
        let big = "x".repeat(MAX_STREAM_FRAME_BYTES + 1);
        let diff = format!(
            r#"{{"method":"turn/diff/updated","params":{{"threadId":"t","turnId":"u","diff":"{big}"}}}}"#
        );
        assert!(run(&mut a, &diff).is_empty());
        let output = COMMAND_COMPLETED.replace(
            r#""aggregatedOutput":"hello\n""#,
            &format!(r#""aggregatedOutput":"{big}""#),
        );
        assert!(output.len() > MAX_STREAM_FRAME_BYTES);
        assert!(
            run(&mut a, &output)[0].starts_with("Frame("),
            "a completed command still renders"
        );
        let elicitation = format!(
            r#"{{"method":"mcpServer/elicitation/request","id":7,"params":{{"message":"{big}"}}}}"#
        );
        run(&mut a, &elicitation);
        assert!(
            a.outbound_frames()
                .iter()
                .any(|f| f.contains(r#""id":7"#) && f.contains("error")),
            "an oversize server request is still refused"
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
        fn codex_prune_matches_code_mode_exec_custom_tool_call() {
            // codex-cli 0.154.0 "code mode" rollout shapes (redacted from a real
            // rollout): shell calls are `custom_tool_call` `exec` with a JS `input`,
            // outputs are `custom_tool_call_output` with an `input_text` array.
            let f = write_jsonl(&[
                &session_meta("01a0cc82-6d96-7c62-b3fd-952ee1d7c12b", "/p"),
                r#"{"timestamp":"2026-09-23T04:25:04.642Z","ordinal":12,"type":"response_item","payload":{"type":"custom_tool_call","id":"ctc_1","status":"completed","call_id":"call_A","name":"exec","input":"const results = await Promise.allSettled([\n  tools.exec_command({cmd:\"git pull\", max_output_tokens:1500}),\n  tools.exec_command({cmd:\"cat tools.md\", max_output_tokens:4000}),\n]);","internal_chat_message_metadata_passthrough":{"turn_id":"t1","create_time":1790137498.4}}}"#,
                r#"{"timestamp":"2026-09-23T04:25:06.099Z","ordinal":18,"type":"response_item","payload":{"type":"custom_tool_call_output","id":"ctco_1","call_id":"call_A","output":[{"type":"input_text","text":"Script completed\nWall time 1.4 seconds\nOutput:\n"},{"type":"input_text","text":"BULKY TOOLS BODY"}],"internal_chat_message_metadata_passthrough":{"turn_id":"t1","create_time":1790137506.1}}}"#,
                r#"{"timestamp":"2026-09-23T04:26:46.887Z","ordinal":66,"type":"response_item","payload":{"type":"custom_tool_call","id":"ctc_2","status":"completed","call_id":"call_B","name":"exec","input":"text(await tools.exec_command({cmd:'\"$ZUCCHINI_SPAWNER_BIN\" prune-context --args \"*cat tools.md*\" --summary \"s\"'}))","internal_chat_message_metadata_passthrough":{"turn_id":"t1","create_time":1790137606.8}}}"#,
            ]);
            assert_eq!(
                count_codex_matches(f.path(), "", "*cat tools.md*").unwrap(),
                1
            );
            let stats = prune_codex_jsonl(f.path(), "", "*cat tools.md*").unwrap();
            assert_eq!(stats.results_blanked, 1);
            let lines = read_lines(f.path());
            assert_eq!(lines.len(), 4);
            assert_eq!(lines[1]["payload"]["input"], "");
            assert_eq!(
                lines[2]["payload"]["output"],
                serde_json::json!([{ "type": "input_text", "text": "[pruned]" }])
            );
            assert!(lines[3]["payload"]["input"]
                .as_str()
                .unwrap()
                .contains("prune-context"));
            assert_eq!(
                count_codex_matches(f.path(), "", "*cat tools.md*").unwrap(),
                0
            );
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
            let after_second = read_lines(f.path());
            assert_eq!(after_second[1]["payload"]["arguments"], "{}");
            assert_eq!(after_second[2]["payload"]["output"], "[pruned]");

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
            let after_second = read_lines(f.path());
            assert_eq!(after_second[2]["payload"]["output"], "[pruned]");
            assert_eq!(after_second[4]["payload"]["output"], "[pruned]");
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
            let second = prune_codex_jsonl(f.path(), "mcp__zdbg__get_last_id", "SELECT").unwrap();
            assert_eq!(second.results_blanked, 1);
            let after_second = read_lines(f.path());
            assert_eq!(after_second[1]["payload"]["arguments"], "{}");
            assert_eq!(after_second[2]["payload"]["output"], "[pruned]");

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
