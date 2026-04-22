//! Drains agent output into the backend's `/api/writes` endpoint.
//!
//! Scope: one-way HTTP writer. Takes high-level events (agent line, session
//! started, heartbeat), encrypts message bodies (only field that's E2E),
//! batches ops into a single POST, and retries with exponential backoff until
//! the request succeeds. Items stay in the channel until the POST that carries
//! them returns 2xx, so a network flap doesn't lose messages.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use futures_util::future::BoxFuture;
use serde::Serialize;
use tokio::sync::mpsc;
use tokio::time::sleep;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::crypto::{encrypt_field_b64, DevKey};

const MAX_OPS_PER_BATCH: usize = 32;
const INITIAL_BACKOFF: Duration = Duration::from_millis(500);
const MAX_BACKOFF: Duration = Duration::from_secs(30);

pub type TokenFetcher =
    Box<dyn Fn() -> BoxFuture<'static, Result<String, anyhow::Error>> + Send + Sync>;

/// High-level event the main loop hands to the writer.
#[derive(Debug, Clone)]
pub enum WriteEvent {
    AgentLine { chat_id: String, content: String },
    ChatStatus { chat_id: String, status: &'static str },
    Heartbeat { machine_id: Uuid },
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

pub fn start(config: WriterConfig, dev_key: Option<DevKey>) -> Writer {
    let (tx, rx) = mpsc::channel::<WriteEvent>(1024);
    let pending = Arc::new(AtomicUsize::new(0));
    let pending_for_task = pending.clone();
    tokio::spawn(async move {
        run(config, dev_key, rx, pending_for_task).await;
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

fn encode_event(event: &WriteEvent, dev_key: Option<&DevKey>) -> Option<BatchOp> {
    Some(match event {
        WriteEvent::AgentLine { chat_id, content } => BatchOp {
            op: "PUT",
            table: "messages",
            id: Uuid::now_v7(),
            data: Some(serde_json::json!({
                "chat_id": chat_id,
                "sender": "agent",
                "body": encrypt_field_b64(dev_key, content),
            })),
        },
        WriteEvent::ChatStatus { chat_id, status } => {
            let id = match Uuid::parse_str(chat_id) {
                Ok(u) => u,
                Err(e) => {
                    warn!(chat_id = %chat_id, error = %e, "ChatStatus chat_id is not a UUID, dropping");
                    return None;
                }
            };
            BatchOp {
                op: "PATCH",
                table: "chats",
                id,
                data: Some(serde_json::json!({ "agent_status": status })),
            }
        }
        WriteEvent::Heartbeat { machine_id } => BatchOp {
            op: "PATCH",
            table: "machines",
            id: *machine_id,
            // Server stamps now() for last_heartbeat_at; the null is just a presence marker.
            data: Some(serde_json::json!({ "last_heartbeat_at": null })),
        },
    })
}

async fn run(
    config: WriterConfig,
    dev_key: Option<DevKey>,
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
        if let Some(op) = encode_event(&ev, dev_key.as_ref()) {
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
