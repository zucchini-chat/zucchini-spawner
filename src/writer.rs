//! Drains agent output into the backend's `/api/writes` endpoint.
//!
//! Scope: one-way HTTP writer. Takes high-level events (agent line, session
//! started, heartbeat), encrypts message bodies (only field that's E2E),
//! batches ops into a single POST, and retries with exponential backoff until
//! the request succeeds. Items stay in the channel until the POST that carries
//! them returns 2xx, so a network flap doesn't lose messages.

use std::collections::VecDeque;
use std::fmt::Display;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use futures_util::future::BoxFuture;
use serde::Serialize;
use tokio::sync::mpsc;
use tokio::time::sleep;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::adapter::AgentKind;
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
    ChatRunning {
        chat_id: String,
        agent_running: bool,
    },
    ContextTokens {
        chat_id: String,
        tokens: i64,
    },
    /// `compactMetadata.postTokens` from a `compact_boundary` system frame.
    /// Backend resolves `context_tokens = baseline_tokens + post_tokens`.
    CompactBoundary {
        chat_id: String,
        post_tokens: i64,
    },
    /// Persists the session id claude generated on its own (we no longer pass
    /// `--session-id`). Written once per chat — the harvest path in `main.rs`
    /// stashes it locally first to avoid racing the sync round-trip.
    AgentSessionId {
        chat_id: String,
        session_id: String,
    },
    Heartbeat {
        machine_id: Uuid,
    },
    /// Sent once per process startup. Re-evaluating on each restart picks up
    /// `claude /login`, `cursor-agent login`, or `codex login` after a service
    /// kick — without that, the iOS app would never see auth flips. The wire
    /// shape on `machines` is a flat pair of nullable BOOLEANs per agent kind
    /// (`claude_code_installed` / `claude_code_authenticated`,
    /// `cursor_installed` / `cursor_authenticated`, `codex_installed` /
    /// `codex_authenticated`); we carry one `(installed, authenticated)` tuple
    /// per `AgentKind` so the encode site can fan them out generically.
    ReportStartupInfo {
        machine_id: Uuid,
        statuses: Vec<(AgentKind, (bool, bool))>,
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
    ImportStatus {
        machine_id: Uuid,
        status: String,
    },
    /// Machine-sharing handshake: publish our X25519 sealedbox public key.
    SetSpawnerPubkey {
        machine_id: Uuid,
        pubkey_b64: String,
    },
    /// Seed the owner's `machine_users.agents` JSON column when it lands
    /// NULL (migration 0035 — see `seed_default_agents_if_needed` in
    /// main.rs). The backend routes this through `put_machine_user_envelope`
    /// (same path iOS uses for `wrapped_key` / `sealed_blob`). For a
    /// machine principal that handler gates writes to the `agents` field
    /// only AND requires `expected_machine_id == jwt.machine_id`, so a
    /// spawner can never touch its peer-machines' rosters. The owner's row
    /// is hit naturally because a machine-token's JWT carries the owner's
    /// user_id in `sub` (backend main.rs:691) and the UPDATE WHERE is
    /// `user_id = $p.user_id`. `agents_json` is the literal JSON string
    /// the backend's `validate_agents_json` will round-trip (we serialize
    /// it once here and never re-parse). `row_id` is `machine_users.id`,
    /// not `machine_users.user_id` — the backend looks up the row by id.
    SetMachineUserAgents {
        row_id: Uuid,
        machine_id: Uuid,
        agents_json: String,
    },
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
        WriteEvent::ChatRunning {
            chat_id,
            agent_running,
        }
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
pub(crate) struct BatchOp {
    pub(crate) op: &'static str,
    pub(crate) table: &'static str,
    pub(crate) id: Uuid,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) data: Option<serde_json::Value>,
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

pub(crate) fn encode_event(event: &WriteEvent, keys: &KeyStore) -> Option<BatchOp> {
    Some(match event {
        WriteEvent::PutMessage {
            id,
            chat_id,
            user_id,
            sender,
            content,
            created_at,
            imported,
        } => {
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
        WriteEvent::ChatRunning {
            chat_id,
            agent_running,
        } => chats_patch(
            chat_id,
            "ChatRunning",
            serde_json::json!({ "agent_running": agent_running }),
        )?,
        WriteEvent::ContextTokens { chat_id, tokens } => chats_patch(
            chat_id,
            "ContextTokens",
            serde_json::json!({ "context_tokens": tokens }),
        )?,
        WriteEvent::CompactBoundary {
            chat_id,
            post_tokens,
        } => chats_patch(
            chat_id,
            "CompactBoundary",
            serde_json::json!({ "compact_boundary_post_tokens": post_tokens }),
        )?,
        WriteEvent::AgentSessionId {
            chat_id,
            session_id,
        } => chats_patch(
            chat_id,
            "AgentSessionId",
            serde_json::json!({ "agent_session_id": session_id }),
        )?,
        WriteEvent::Heartbeat { machine_id } => {
            // Server stamps now() for last_heartbeat_at; the null is just a presence marker.
            machines_patch(
                *machine_id,
                serde_json::json!({ "last_heartbeat_at": null }),
            )
        }
        WriteEvent::ReportStartupInfo {
            machine_id,
            statuses,
        } => {
            // Per-agent install/auth is a pair of nullable BOOLEAN columns per kind on
            // `machines` (`claude_code_installed` / `claude_code_authenticated`
            // for the legacy claude columns inherited from pre-multi-agent
            // schema, `cursor_installed` / `cursor_authenticated` added in
            // migration 0033_multi_agent_support, `codex_installed` /
            // `codex_authenticated` added in migration 0037, and
            // `hermes_installed` / `hermes_authenticated` added in migration
            // 0038). iOS derives its `AgentInstallStatus` UI helper from these
            // booleans locally.
            //
            // `backend_has_install_columns` stays as a per-kind guard so a
            // future kind whose backend migration is still in flight can be
            // filtered out here: shipping a PATCH with a column the backend's
            // `reject_unknown_fields(obj, allowed)` doesn't know would 4xx the
            // whole batch and head-of-line block the writer (items stay queued
            // on 4xx, retry forever with backoff).
            let mut data = serde_json::Map::new();
            data.insert(
                "spawner_version".to_string(),
                serde_json::Value::String(env!("CARGO_PKG_VERSION").to_string()),
            );
            for (kind, (installed, authenticated)) in statuses {
                if !backend_has_install_columns(*kind) {
                    continue;
                }
                let (installed_col, authenticated_col) = kind.install_columns();
                data.insert(
                    installed_col.to_string(),
                    serde_json::Value::Bool(*installed),
                );
                data.insert(
                    authenticated_col.to_string(),
                    serde_json::Value::Bool(*authenticated),
                );
            }
            machines_patch(*machine_id, serde_json::Value::Object(data))
        }
        WriteEvent::PutChat {
            id,
            project_id,
            user_id,
            title,
            created_at,
        } => {
            let k = resolve_key_or_warn(keys, user_id, id, "chat")?;
            // Importer contract: chat id IS the claude session id (migration
            // 0019). The pre-0032 backfill `UPDATE chats SET agent_session_id =
            // id::text WHERE agent_session_id IS NULL` only ran once at deploy,
            // so chats imported AFTER deploy would otherwise leave the column
            // NULL and the next user message would spawn claude without
            // `--resume`, losing the imported transcript context.
            BatchOp {
                op: "PUT",
                table: "chats",
                id: *id,
                data: Some(serde_json::json!({
                    "project_id": project_id.to_string(),
                    "title": encrypt_field_b64(&k, title),
                    "worktree": false,
                    "created_at": created_at.to_rfc3339(),
                    "agent_session_id": id.to_string(),
                })),
            }
        }
        WriteEvent::PutProject {
            id,
            machine_id,
            name,
            path,
        } => BatchOp {
            op: "PUT",
            table: "projects",
            id: *id,
            data: Some(serde_json::json!({
                "machine_id": machine_id.to_string(),
                "name": name,
                "path": path,
            })),
        },
        WriteEvent::ImportStatus { machine_id, status } => machines_patch(
            *machine_id,
            serde_json::json!({ "claude_history_import_status": status }),
        ),
        WriteEvent::SetSpawnerPubkey {
            machine_id,
            pubkey_b64,
        } => machines_patch(
            *machine_id,
            serde_json::json!({ "spawner_pubkey": pubkey_b64 }),
        ),
        WriteEvent::SetMachineUserAgents {
            row_id,
            machine_id,
            agents_json,
        } => {
            // Routes through the backend's `put_machine_user_envelope`
            // handler (table=machine_users, op=PATCH — see writes.rs
            // dispatch table). The handler demands `machine_id` in the
            // body for a defense-in-depth ownership check on top of the
            // `user_id = $principal` gate. `agents` is the raw JSON
            // string the backend's `validate_agents_json` will round-trip
            // before commit; we never re-parse it on the spawner side.
            BatchOp {
                op: "PATCH",
                table: "machine_users",
                id: *row_id,
                data: Some(serde_json::json!({
                    "machine_id": machine_id.to_string(),
                    "agents": agents_json,
                })),
            }
        }
    })
}

fn machines_patch(machine_id: Uuid, data: serde_json::Value) -> BatchOp {
    BatchOp {
        op: "PATCH",
        table: "machines",
        id: machine_id,
        data: Some(data),
    }
}

fn chats_patch(chat_id: &str, label: &str, data: serde_json::Value) -> Option<BatchOp> {
    match Uuid::parse_str(chat_id) {
        Ok(id) => Some(BatchOp {
            op: "PATCH",
            table: "chats",
            id,
            data: Some(data),
        }),
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

/// Whether the deployed backend's `machines` allowlist already carries the
/// install/authenticated columns for `kind`. Per-kind override that exists
/// only while a kind's backend migration is in flight — once the columns are
/// live (and `process_machine_patch`'s `machine_fields` allowlist + COALESCE
/// block know about them), drop the kind from this filter.
///
/// Why this exists: `ReportStartupInfo` ships a PATCH carrying one
/// `<kind>_installed` / `<kind>_authenticated` pair per `AgentKind::ALL`. The
/// backend's `reject_unknown_fields` 4xx's the whole batch on any unknown
/// field, items stay queued on 4xx (writer retries forever with backoff),
/// and every subsequent flush is blocked behind the same poison message.
/// Filter here = no poison message gets queued in the first place.
fn backend_has_install_columns(kind: AgentKind) -> bool {
    match kind {
        AgentKind::Claude
        | AgentKind::Cursor
        | AgentKind::Codex
        | AgentKind::Hermes
        | AgentKind::Gemini => true,
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
