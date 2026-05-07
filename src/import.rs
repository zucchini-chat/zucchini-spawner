//! One-shot importer for `~/.claude/projects/*/*.jsonl`. Idempotent: project
//! ids are UUIDv5(machine_id || path), chat ids are the sessionId from the
//! filename, so re-runs reconverge.
//!
//! Frame filter mirrors the substring filter in `agent.rs` (the stdout reader
//! in `Supervisor::spawn`) — keep `user` with string content (wrap in
//! MessageEnvelope) and `assistant` with text/tool_use blocks (re-emit as
//! stream-json so SpawnerMessageDescriber reads it). Skip tool_result echoes,
//! thinking-only frames, sidechain (subagent) transcripts, and
//! `queue-operation`/`last-prompt`/`attachment`. `ai-title` is harvested into
//! chats.title.
//!
//! User strings also get a synthetic-wrapper screen (see `is_synthetic_wrapper`):
//! the TUI logs `/exit`, `/clear`, `/compact`, `<system-reminder>`, etc. as
//! pseudo-user rows that the user never typed and the model never sees.
//! Custom slash commands (`<command-message>`-prefixed) are kept — those go
//! to the model and produce real assistant replies.

use std::collections::BTreeMap;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::mpsc;
use tracing::{info, warn};
use uuid::Uuid;

use crate::envelope::MessageEnvelope;
use crate::writer::WriteEvent;

// "zucchiniprojects" — fixed so re-imports converge on the same project ids.
const PROJECT_NS: Uuid = Uuid::from_bytes([
    0x7a, 0x75, 0x63, 0x63, 0x68, 0x69, 0x6e, 0x69,
    0x70, 0x72, 0x6f, 0x6a, 0x65, 0x63, 0x74, 0x73,
]);

/// One-shot. Triggered exactly once, immediately after a machine is added —
/// the iOS app blocks the UI on the import-progress sheet, so no live agent
/// can be spawned while this runs and the writer's batch channel is ours
/// alone. That's why we don't bump `MAX_OPS_PER_BATCH` for the importer
/// path — contention isn't possible by construction.
pub async fn run(
    machine_id: Uuid,
    write_tx: mpsc::Sender<WriteEvent>,
) -> Result<()> {
    let projects_dir = claude_projects_dir().context("locate ~/.claude/projects")?;
    info!(path = %projects_dir.display(), "scanning claude-code transcripts");
    send_status(&write_tx, machine_id, "running-0").await;

    // `BTreeMap` keeps the order stable for logs.
    let mut sessions_by_path: BTreeMap<String, Vec<PathBuf>> = BTreeMap::new();
    let mut total_sessions: usize = 0;
    let dir = match std::fs::read_dir(&projects_dir) {
        Ok(d) => d,
        Err(e) if e.kind() == ErrorKind::NotFound => {
            info!(path = %projects_dir.display(), "no ~/.claude/projects, nothing to import");
            send_status(&write_tx, machine_id, "finished").await;
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
            let f = match f { Ok(x) => x, Err(_) => continue };
            let p = f.path();
            if p.extension().and_then(|s| s.to_str()) == Some("jsonl") {
                bucket.push(p);
                total_sessions += 1;
            }
        }
    }

    if total_sessions == 0 {
        info!("no .jsonl transcripts found");
        send_status(&write_tx, machine_id, "finished").await;
        return Ok(());
    }
    info!(
        projects = sessions_by_path.len(),
        sessions = total_sessions,
        "starting import"
    );

    let mut done_sessions: usize = 0;
    let mut last_pct: i32 = 0;
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
            if let Err(e) = import_session(&jsonl, project_id, &write_tx).await {
                warn!(file = %jsonl.display(), error = %e, "session import failed, skipping");
            }
            done_sessions += 1;
            // Step in 5% increments. Each PATCH fans out via PowerSync to
            // every connected client; flooding 100 of them inside one minute
            // burns watch wakeups for negligible UX gain.
            let pct = ((done_sessions as f64 / total_sessions as f64) * 100.0) as i32;
            if pct >= last_pct + 5 {
                last_pct = pct;
                send_status(&write_tx, machine_id, &format!("running-{pct}")).await;
            }
        }
    }

    send_status(&write_tx, machine_id, "finished").await;
    info!(sessions = done_sessions, "claude history import complete");
    Ok(())
}

async fn send_status(tx: &mpsc::Sender<WriteEvent>, machine_id: Uuid, status: &str) {
    let _ = tx
        .send(WriteEvent::ImportStatus { machine_id, status: status.to_string() })
        .await;
}

fn claude_projects_dir() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    Some(PathBuf::from(home).join(".claude").join("projects"))
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
    let mut current = PathBuf::from("/");
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

fn basename_or(path: &str, fallback: &str) -> String {
    Path::new(path)
        .file_name()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or(fallback)
        .to_string()
}

fn mint_project_id(machine_id: Uuid, path: &str) -> Uuid {
    // \0 separator so `(a, b\0c)` and `(a\0b, c)` can't collide.
    Uuid::new_v5(
        &PROJECT_NS,
        &[machine_id.as_bytes().as_slice(), b"\0", path.as_bytes()].concat(),
    )
}

async fn import_session(
    jsonl: &Path,
    project_id: Uuid,
    write_tx: &mpsc::Sender<WriteEvent>,
) -> Result<()> {
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
        if matches!(
            entry_type,
            "queue-operation" | "last-prompt" | "attachment"
        ) {
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

        let imported = match entry_type {
            "user" => match classify_user(&entry) {
                UserContent::Prompt(text) => {
                    if title.is_none() {
                        title = Some(collapse_title(&text));
                    }
                    Some(ImportedMsg::user(text))
                }
                UserContent::ToolResult | UserContent::Empty => None,
            },
            "assistant" => match classify_assistant(&entry) {
                AssistantContent::Renderable(body) => Some(ImportedMsg::agent(body)),
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
            title: chat_title,
            created_at: chat_created_at,
        })
        .await;

    for (ts, msg) in keepers {
        let _ = write_tx
            .send(WriteEvent::PutMessage {
                chat_id: chat_id.to_string(),
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
}

impl ImportedMsg {
    fn user(text: String) -> Self {
        let env = MessageEnvelope { text, attachments: Vec::new() };
        Self { sender: "user", body: serde_json::to_string(&env).expect("envelope serializable") }
    }

    fn agent(stream_json_frame: String) -> Self {
        Self { sender: "agent", body: stream_json_frame }
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
    if let Some(s) = content.as_str() {
        if is_synthetic_wrapper(s) {
            return UserContent::Empty;
        }
        return UserContent::Prompt(s.to_string());
    }
    if content.as_array().is_some() {
        // tool_result or other block-shaped content the live spawner skips too.
        return UserContent::ToolResult;
    }
    UserContent::Empty
}

/// User-content strings that claude-code wraps in these tags are synthetic —
/// either local CLI commands handled by the TUI (`<command-name>`,
/// `<local-command-stdout>`, `<local-command-stderr>`, `<local-command-caveat>`)
/// or harness-injected reminders/notifications (`<system-reminder>`,
/// `<task-notification>`). None of them are user-typed prompts, and the TUI
/// itself strips them before rendering. `<command-message>` is the contrast
/// case: it's how custom slash commands like `/simplify` introduce a real
/// prompt that gets sent to the model — keep those.
fn is_synthetic_wrapper(s: &str) -> bool {
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

fn collapse_title(text: &str) -> String {
    let collapsed: String = text
        .replace('\r', " ")
        .replace('\n', " ");
    collapsed.chars().take(100).collect()
}

