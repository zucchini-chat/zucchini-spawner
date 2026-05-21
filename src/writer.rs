//! Drains agent output into the backend's `/api/writes` endpoint.
//!
//! Scope: one-way HTTP writer. Takes high-level events (agent line, session
//! started, heartbeat), encrypts message bodies (only field that's E2E),
//! batches ops into a single POST, and retries with exponential backoff until
//! the request succeeds. Items stay in the channel until the POST that carries
//! them returns 2xx, so a network flap doesn't lose messages.

use std::collections::VecDeque;
use std::fmt::Display;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use chrono::{DateTime, Utc};
use futures_util::future::BoxFuture;
use serde::Serialize;
use tokio::sync::mpsc;
use tokio::time::sleep;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::crypto::{encrypt_field_b64, KUser, KeyStore};

const MAX_OPS_PER_BATCH: usize = 32;
const INITIAL_BACKOFF: Duration = Duration::from_millis(500);
const MAX_BACKOFF: Duration = Duration::from_secs(30);

pub type TokenFetcher =
    Box<dyn Fn() -> BoxFuture<'static, Result<String, anyhow::Error>> + Send + Sync>;

/// High-level event the main loop hands to the writer.
#[derive(Debug, Clone)]
pub enum WriteEvent {
    /// `created_at` is `Some` only on the importer path — backdated to the
    /// transcript timestamp; live writes leave it `None` so the server stamps `now()`.
    /// `imported` is true only on the importer path; the spawner's sync consumer
    /// skips imported rows so re-streaming our own bucket doesn't re-spawn an
    /// agent for every backfilled user prompt.
    PutMessage {
        /// When `Some`, used verbatim as `messages.id`. The importer passes the
        /// claude-code `entry.uuid` here so replays of the same conversation
        /// entry (claude rewrites the .jsonl on every `--continue`/`--resume`,
        /// preserving entry uuids) collapse via the backend's
        /// `INSERT ... ON CONFLICT (id) DO NOTHING` clause. Live writes leave
        /// it `None` and the writer mints a fresh `Uuid::now_v7()`.
        id: Option<Uuid>,
        chat_id: String,
        /// Owner of the chat — picks which per-user key encrypts `body`.
        user_id: Uuid,
        sender: &'static str,
        content: String,
        created_at: Option<DateTime<Utc>>,
        imported: bool,
    },
    ChatRunning { chat_id: String, agent_running: bool },
    ContextTokens { chat_id: String, tokens: i64 },
    /// `compactMetadata.postTokens` from a `compact_boundary` system frame.
    /// Backend resolves `context_tokens = baseline_tokens + post_tokens`.
    CompactBoundary { chat_id: String, post_tokens: i64 },
    Heartbeat { machine_id: Uuid },
    /// Sent once per process startup. Re-evaluating on each restart picks up
    /// `claude /login` after a service kick — without that, the iOS app would
    /// never see auth flips.
    ReportStartupInfo {
        machine_id: Uuid,
        claude_code_installed: bool,
        claude_code_authenticated: bool,
    },
    /// Importer-only.
    PutChat {
        id: Uuid,
        project_id: Uuid,
        user_id: Uuid,
        title: String,
        created_at: DateTime<Utc>,
    },
    /// Importer-only.
    PutProject {
        id: Uuid,
        machine_id: Uuid,
        name: String,
        path: String,
    },
    /// Importer-only: machines.PATCH carrying the import progress string
    /// (`requested` | `running-N` | `finished`).
    ImportStatus { machine_id: Uuid, status: String },
}

impl WriteEvent {
    pub fn agent_line(chat_id: String, user_id: Uuid, content: String) -> Self {
        WriteEvent::PutMessage {
            id: None,
            chat_id,
            user_id,
            sender: "agent",
            content,
            created_at: None,
            imported: false,
        }
    }

    pub fn chat_running(chat_id: String, agent_running: bool) -> Self {
        WriteEvent::ChatRunning { chat_id, agent_running }
    }
}

pub struct WriterConfig {
    pub base_url: String,
    pub fetch_token: TokenFetcher,
}

/// Handle returned from `start`. `tx` is the usual mpsc sender; `pending`
/// counts events the writer task has pulled off the channel and not yet
/// successfully flushed (used by the updater to wait for a clean drain
/// before swapping the binary). Events still sitting in the channel are
/// observable via `tx.max_capacity() - tx.capacity()`.
pub struct Writer {
    pub tx: mpsc::Sender<WriteEvent>,
    pub pending: Arc<AtomicUsize>,
}

impl Writer {
    /// True when the channel is empty AND the writer task has flushed every
    /// event it pulled. Callers must also ensure no new events are being
    /// produced (supervisor idle + heartbeat cancelled) — otherwise this
    /// races and can flip back to false a moment later.
    pub fn is_idle(&self) -> bool {
        let queued = self.tx.max_capacity() - self.tx.capacity();
        queued == 0 && self.pending.load(Ordering::Relaxed) == 0
    }
}

pub fn start(config: WriterConfig, keys: Arc<KeyStore>) -> Writer {
    let (tx, rx) = mpsc::channel::<WriteEvent>(1024);
    let pending = Arc::new(AtomicUsize::new(0));
    let pending_for_task = pending.clone();
    tokio::spawn(async move {
        run(config, keys, rx, pending_for_task).await;
    });
    Writer { tx, pending }
}

// ---------- internals ----------

#[derive(Debug, Clone, Serialize)]
struct BatchOp {
    op: &'static str,
    table: &'static str,
    id: Uuid,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
struct BatchReq<'a> {
    ops: &'a [BatchOp],
}

fn resolve_key_or_warn(
    keys: &KeyStore,
    user_id: &Uuid,
    chat_id: impl Display,
    label: &str,
) -> Option<Arc<KUser>> {
    match keys.get(user_id) {
        Ok(k) => Some(k),
        Err(e) => {
            warn!(%user_id, chat_id = %chat_id, error = %e, "no key for user, dropping {}", label);
            None
        }
    }
}

fn encode_event(event: &WriteEvent, keys: &KeyStore) -> Option<BatchOp> {
    Some(match event {
        WriteEvent::PutMessage { id, chat_id, user_id, sender, content, created_at, imported } => {
            let k = resolve_key_or_warn(keys, user_id, chat_id, "message")?;
            let mut data = serde_json::json!({
                "chat_id": chat_id,
                "sender": sender,
                "body": encrypt_field_b64(&k, content),
                "imported": imported,
            });
            if let Some(ts) = created_at {
                data["created_at"] = serde_json::Value::String(ts.to_rfc3339());
            }
            BatchOp {
                op: "PUT",
                table: "messages",
                id: id.unwrap_or_else(Uuid::now_v7),
                data: Some(data),
            }
        }
        WriteEvent::ChatRunning { chat_id, agent_running } => {
            chats_patch(chat_id, "ChatRunning", serde_json::json!({ "agent_running": agent_running }))?
        }
        WriteEvent::ContextTokens { chat_id, tokens } => {
            chats_patch(chat_id, "ContextTokens", serde_json::json!({ "context_tokens": tokens }))?
        }
        WriteEvent::CompactBoundary { chat_id, post_tokens } => {
            chats_patch(
                chat_id,
                "CompactBoundary",
                serde_json::json!({ "compact_boundary_post_tokens": post_tokens }),
            )?
        }
        WriteEvent::Heartbeat { machine_id } => BatchOp {
            op: "PATCH",
            table: "machines",
            id: *machine_id,
            // Server stamps now() for last_heartbeat_at; the null is just a presence marker.
            data: Some(serde_json::json!({ "last_heartbeat_at": null })),
        },
        WriteEvent::ReportStartupInfo { machine_id, claude_code_installed, claude_code_authenticated } => BatchOp {
            op: "PATCH",
            table: "machines",
            id: *machine_id,
            data: Some(serde_json::json!({
                "spawner_version": env!("CARGO_PKG_VERSION"),
                "claude_code_installed": claude_code_installed,
                "claude_code_authenticated": claude_code_authenticated,
            })),
        },
        WriteEvent::PutChat { id, project_id, user_id, title, created_at } => {
            let k = resolve_key_or_warn(keys, user_id, id, "chat")?;
            BatchOp {
                op: "PUT",
                table: "chats",
                id: *id,
                data: Some(serde_json::json!({
                    "project_id": project_id.to_string(),
                    "title": encrypt_field_b64(&k, title),
                    "worktree": false,
                    "created_at": created_at.to_rfc3339(),
                })),
            }
        }
        WriteEvent::PutProject { id, machine_id, name, path } => BatchOp {
            op: "PUT",
            table: "projects",
            id: *id,
            data: Some(serde_json::json!({
                "machine_id": machine_id.to_string(),
                "name": name,
                "path": path,
            })),
        },
        WriteEvent::ImportStatus { machine_id, status } => BatchOp {
            op: "PATCH",
            table: "machines",
            id: *machine_id,
            data: Some(serde_json::json!({ "claude_history_import_status": status })),
        },
    })
}

fn chats_patch(chat_id: &str, label: &str, data: serde_json::Value) -> Option<BatchOp> {
    match Uuid::parse_str(chat_id) {
        Ok(id) => Some(BatchOp { op: "PATCH", table: "chats", id, data: Some(data) }),
        Err(e) => {
            warn!(chat_id = %chat_id, error = %e, "{} chat_id is not a UUID, dropping", label);
            None
        }
    }
}

async fn run(
    config: WriterConfig,
    keys: Arc<KeyStore>,
    mut rx: mpsc::Receiver<WriteEvent>,
    pending_counter: Arc<AtomicUsize>,
) {
    let http = reqwest::Client::builder()
        .pool_idle_timeout(Duration::from_secs(60))
        .timeout(Duration::from_secs(20))
        .build()
        .expect("reqwest client");
    let url = format!("{}/api/writes", config.base_url.trim_end_matches('/'));
    let mut pending: VecDeque<BatchOp> = VecDeque::new();
    let mut backoff = INITIAL_BACKOFF;
    let mut channel_closed = false;

    let enqueue = |ev: WriteEvent, pending: &mut VecDeque<BatchOp>| {
        if let Some(op) = encode_event(&ev, &keys) {
            pending.push_back(op);
            pending_counter.fetch_add(1, Ordering::Relaxed);
        }
    };

    loop {
        if pending.is_empty() {
            if channel_closed {
                return;
            }
            match rx.recv().await {
                Some(ev) => enqueue(ev, &mut pending),
                None => {
                    info!("writer channel closed, exiting");
                    return;
                }
            }
        }

        while pending.len() < MAX_OPS_PER_BATCH {
            match rx.try_recv() {
                Ok(ev) => enqueue(ev, &mut pending),
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    channel_closed = true;
                    info!(remaining = pending.len(), "writer channel closed, draining");
                    break;
                }
            }
        }

        let batch: Vec<BatchOp> = pending.iter().take(MAX_OPS_PER_BATCH).cloned().collect();
        match send_batch(&http, &url, &config.fetch_token, &batch).await {
            Ok(()) => {
                debug!(count = batch.len(), "flushed batch to /api/writes");
                for _ in 0..batch.len() {
                    pending.pop_front();
                }
                pending_counter.fetch_sub(batch.len(), Ordering::Relaxed);
                backoff = INITIAL_BACKOFF;
            }
            Err(e) => {
                warn!(error = %e, queued = pending.len(), ?backoff, "write failed, retrying");
                sleep(backoff).await;
                backoff = (backoff * 2).min(MAX_BACKOFF);
            }
        }
    }
}

async fn send_batch(
    http: &reqwest::Client,
    url: &str,
    fetch_token: &TokenFetcher,
    ops: &[BatchOp],
) -> Result<(), anyhow::Error> {
    let token = fetch_token().await?;
    let resp = http
        .post(url)
        .bearer_auth(token)
        .json(&BatchReq { ops })
        .send()
        .await?;
    let status = resp.status();
    if status.is_success() {
        return Ok(());
    }
    let body = resp.text().await.unwrap_or_default();
    if status.is_client_error() {
        // 4xx means the server rejected the batch for a reason that won't fix itself
        // (bad data, missing chat, forbidden sender). Dropping is worse than retrying —
        // we escalate so the operator sees it; keep the items queued so manual recovery
        // (fix schema, add chat) can still flush them.
        error!(%status, %body, "POST /api/writes rejected (client error)");
    }
    anyhow::bail!("POST /api/writes {}: {}", status, body);
}
