//! Claude-code adapter. **Iso-claude guarantee**: the bytes written to
//! `messages.body` here must be byte-identical to the pre-refactor spawner
//! output. The command builder, skip filter, session-id harvest, and
//! per-frame usage parsing are direct lifts from the pre-refactor `agent.rs`.
//! When in doubt: do not edit; move only.
//!
//! Also hosts the install/auth `probe()` for claude (free function, not on
//! the `AgentAdapter` trait — `dyn AgentAdapter` can't dispatch statics).
//! `main.rs::probe_install` calls into it from the startup-info report.

use std::path::PathBuf;

use anyhow::Result;
use serde::Deserialize;
use smallvec::SmallVec;
use tracing::debug;

use crate::adapter::{
    file_nonempty, probe_with_blocking_auth, shell_escape, AdapterDescriptor, AgentAdapter,
    AgentEvent, AgentKind, LastTokensDedup, TurnContext, ATTACH_FILE_INSTRUCTION,
    MAX_STREAM_FRAME_BYTES,
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
}

impl ClaudeAdapter {
    pub fn new() -> Self {
        Self::default()
    }
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
        claude_cmd.push_str(&format!(
            "cat {} | claude",
            shell_escape(&ctx.prompt_file.to_string_lossy())
        ));
        // First turn: pass no session flag — claude generates a session id
        // and emits it in the `system/init` stdout frame, which we harvest
        // below and persist to `chats.agent_session_id`. Subsequent turns
        // resume that id. Pre-migration rows were backfilled with
        // `agent_session_id = id::text`, so existing chats keep using the
        // same id claude already knows about.
        if let Some(sid) = ctx.agent_session_id {
            claude_cmd.push_str(&format!(" --resume {}", shell_escape(sid)));
        }
        // claude --print is one-shot; once the result frame emits, the process
        // exits and the spawner stops reading stdout. Background work has no
        // way to reach the chat afterwards, so steer agents away from it.
        let mut sys = String::from(
            "You are spawned via a harness, no background subagents will wake you when finished, use subagents with `run_in_background: false` only."
        );

        if ctx.worktree {
            // Use the chat_id prefix so the worktree directory name stays short.
            let worktree_name: String = ctx.chat_id.chars().take(8).collect();
            claude_cmd.push_str(&format!(" --worktree {}", shell_escape(&worktree_name)));
            // Plug the path-containment hole in --worktree: the harness chdirs into
            // the worktree but doesn't tell the agent (or its subagents) to stay
            // there, so absolute paths into the parent repo "just work" and edits
            // leak out. Inject the absolute worktree path with an explicit rule.
            if let Some(pp) = ctx.project_path {
                let worktree_abs = format!(
                    "{}/.claude/worktrees/{}",
                    pp.trim_end_matches('/'),
                    worktree_name
                );
                sys.push_str(&format!(
                    "\n\nWorktree: {}\nParent repo: {} (do not touch unless the user explicitly asks).\nKeep all edits and Bash commands inside the worktree. If a path under the parent repo appears in context, rewrite it to the worktree before calling Edit/Write/Bash. When delegating via Task, repeat this rule and the worktree path — subagents don't inherit it.",
                    worktree_abs, pp
                ));
            }
        }
        sys.push_str("\n\n");
        sys.push_str(ATTACH_FILE_INSTRUCTION);
        claude_cmd.push_str(&format!(" --append-system-prompt {}", shell_escape(&sys)));
        claude_cmd.push_str(
            " --print --verbose --output-format stream-json --disallowedTools AskUserQuestion",
        );
        // Sender's `machine_users.is_sandboxed`. Non-sandboxed = bypass permission
        // gating; sandboxed = claude's default permission mode auto-denies tools
        // in `--print`, which is the actual sandboxing mechanism.
        if !ctx.is_sandboxed {
            claude_cmd.push_str(" --dangerously-skip-permissions");
        }
        // Verbatim pass-through of `chats.model` (migration 0035). Empty /
        // blank values are already filtered to `None` at the construction
        // site in `main.rs`, so any `Some` here is a non-empty model name
        // the user picked in the composer's agent roster. We don't validate
        // the model name — claude prints a clean error if it doesn't
        // recognize it, and the closed set drifts per-release.
        if let Some(model) = ctx.model {
            claude_cmd.push_str(&format!(" --model {}", shell_escape(model)));
        }
        Ok(claude_cmd)
    }

    fn handle_line(&mut self, line: String) -> SmallVec<[AgentEvent; 2]> {
        let mut out: SmallVec<[AgentEvent; 2]> = SmallVec::new();

        // Pre-skip-filter: thinking-only frames also carry usage.
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
                }
                skip = true;
            } else if line.contains("\"type\":\"stream_event\"")
                || line.contains("\"type\":\"user\"")
                || line.contains("\"type\":\"rate_limit_event\"")
                || (line.contains("\"type\":\"assistant\"")
                    && !line.contains("\"type\":\"text\"")
                    && !line.contains("\"type\":\"tool_use\""))
            {
                skip = true;
            } else if line.contains("\"type\":\"result\"") {
                // Emit Result on every result frame; the supervisor latches it
                // (so AgentResponse::Done.has_result is set once and only once).
                out.push(AgentEvent::Result);
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

/// Reads `compactMetadata.postTokens` from a `compact_boundary` system frame.
/// Narrow Deserialize struct so serde skips the rest of the frame without allocating it.
fn parse_compact_post_tokens(line: &str) -> Option<i64> {
    #[derive(serde::Deserialize)]
    struct Frame {
        #[serde(rename = "compactMetadata")]
        metadata: Metadata,
    }
    #[derive(serde::Deserialize)]
    struct Metadata {
        #[serde(rename = "postTokens")]
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
    basename_or, collapse_title, is_synthetic_wrapper, mint_project_id, ProgressThrottle,
};
use crate::envelope::MessageEnvelope;
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
            progress(100);
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
        progress(100);
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
            // 5%-step throttle shared with every importer; see `ProgressThrottle`.
            throttle.step(done_sessions, total_sessions, &progress);
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
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&Utc));

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
    let _ = write_tx
        .send(WriteEvent::PutChat {
            id: chat_id,
            project_id,
            user_id,
            title: chat_title,
            created_at: chat_created_at,
        })
        .await;

    for (ts, msg) in keepers {
        let _ = write_tx
            .send(WriteEvent::PutMessage {
                id: msg.uuid,
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

struct ImportedMsg {
    sender: &'static str,
    body: String,
    uuid: Option<Uuid>,
}

impl ImportedMsg {
    fn user(text: String, uuid: Option<Uuid>) -> Self {
        let env = MessageEnvelope {
            text,
            attachments: Vec::new(),
        };
        Self {
            sender: "user",
            body: serde_json::to_string(&env).expect("envelope serializable"),
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
    fn user_frame_dropped() {
        let mut a = ClaudeAdapter::new();
        let line = r#"{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"tu1","content":"out"}]}}"#;
        let events = run(&mut a, line);
        assert!(events.is_empty());
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
        let ctx = TurnContext {
            chat_id: "abcdef012345-6789-...",
            prompt_file: &prompt_file,
            project_path: Some("/tmp/proj"),
            worktree: false,
            agent_session_id: None,
            is_sandboxed: false,
            model: Some("opus"),
        };
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
