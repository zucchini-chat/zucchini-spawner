mod agent;
mod auth;
mod crypto;
mod powersync;
mod state;
mod updater;
mod writer;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use agent::{AgentResponse, Supervisor};
use auth::AuthClient;
use crypto::DevKey;
use powersync::{SyncConfig, SyncEvent};
use state::Mirror;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use uuid::Uuid;
use writer::{WriteEvent, Writer, WriterConfig};

const INTERRUPTED_RESULT: &str = r#"{"type":"result","subtype":"interrupted"}"#;
const PROD_BASE_URL: &str = "https://api.zucchini.chat";
const DEV_SYNC_BASE_URL: &str = "http://localhost:8080";
const DEV_API_BASE_URL: &str = "http://localhost:3100";
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(10);
// Cap the wait so a server that won't accept writes can't block the update forever.
const UPDATE_DRAIN_TIMEOUT: Duration = Duration::from_secs(30);
const UPDATE_DRAIN_POLL: Duration = Duration::from_millis(100);

fn init_logging() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info,zucchini_spawner=debug"));
    tracing_subscriber::registry()
        .with(filter)
        .with(tracing_subscriber::fmt::layer().with_writer(std::io::stdout))
        .init();
}

pub(crate) fn zucchini_spawner_dir() -> PathBuf {
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
    home.join(".zucchini-spawner")
}

/// Prod vs dev auth mode. Prod needs both MACHINE_ID and SPAWNER_TOKEN — one without
/// the other is an install bug, so we fall through to dev rather than half-configuring.
struct ProdConfig {
    machine_id: Uuid,
    auth: Arc<AuthClient>,
}

fn load_prod_config() -> Option<ProdConfig> {
    let machine_id = std::env::var("ZUCCHINI_MACHINE_ID").ok()?;
    let spawner_token = std::env::var("ZUCCHINI_SPAWNER_TOKEN").ok()?;
    let machine_id = match Uuid::parse_str(&machine_id) {
        Ok(u) => u,
        Err(e) => {
            warn!(error = %e, "ZUCCHINI_MACHINE_ID not a valid UUID, falling back to dev");
            return None;
        }
    };
    Some(ProdConfig {
        machine_id,
        auth: Arc::new(AuthClient::new(PROD_BASE_URL, spawner_token)),
    })
}

fn dev_token_fetcher() -> Box<
    dyn Fn() -> futures_util::future::BoxFuture<'static, Result<String, anyhow::Error>>
        + Send
        + Sync,
> {
    Box::new(|| {
        Box::pin(async {
            std::env::var("ZUCCHINI_DEV_JWT")
                .map_err(|_| anyhow::anyhow!("ZUCCHINI_DEV_JWT env var not set"))
        })
    })
}

fn build_sync_config(
    prod: Option<&ProdConfig>,
    initial_buckets: std::collections::HashMap<String, String>,
) -> SyncConfig {
    let hostname = gethostname::gethostname().to_string_lossy().to_string();
    let (base_url, fetch_token) = match prod {
        Some(p) => (PROD_BASE_URL.to_string(), auth::token_fetcher(p.auth.clone())),
        None => (DEV_SYNC_BASE_URL.to_string(), dev_token_fetcher()),
    };
    SyncConfig {
        base_url,
        client_id: format!("zucchini-spawner-{}", hostname),
        initial_buckets,
        fetch_token,
    }
}

fn state_path() -> PathBuf {
    zucchini_spawner_dir().join("state.json")
}

fn save_mirror(path: &Path, mirror: &Mirror) {
    if let Err(e) = mirror.save(path) {
        warn!(error = %e, "failed to persist state.json");
    }
}

fn build_writer_config(prod: Option<&ProdConfig>) -> WriterConfig {
    let (base_url, fetch_token) = match prod {
        Some(p) => (PROD_BASE_URL.to_string(), auth::token_fetcher(p.auth.clone())),
        None => (DEV_API_BASE_URL.to_string(), dev_token_fetcher()),
    };
    WriterConfig { base_url, fetch_token }
}

/// Returns true when the op mutated persistent state (projects) and the caller
/// should save `mirror` to disk. Cursor/buckets are persisted on CheckpointComplete
/// instead — per-row saves would multiply disk writes for no durability gain.
async fn handle_sync_event(
    event: SyncEvent,
    mirror: &mut Mirror,
    supervisor: &mut Supervisor,
    dev_key: Option<&DevKey>,
) -> bool {
    match event {
        SyncEvent::Put { table, id, data } => match table.as_str() {
            "projects" => {
                mirror.upsert_project(id, &data);
                true
            }
            "chats" => {
                mirror.upsert_chat(id, &data);
                false
            }
            "messages" => {
                handle_message_put(&data, mirror, supervisor, dev_key).await;
                false
            }
            _ => false,
        },
        SyncEvent::Remove { table, id } => match table.as_str() {
            "chats" => {
                // PowerSync emits PUT → REMOVE → PUT on every row update (moves the
                // storage slot, not a real delete), so a lone REMOVE here does NOT
                // mean the user deleted the chat — abort-on-delete needs a separate
                // signal (TODO: debounce until CheckpointComplete shows the row is
                // really gone, or add a chats.deleted_at column).
                mirror.remove_chat(&id);
                false
            }
            "projects" => {
                mirror.remove_project(&id);
                true
            }
            _ => false,
        },
        SyncEvent::CheckpointComplete { buckets } => {
            mirror.buckets = buckets;
            true
        }
    }
}

async fn handle_message_put(
    data: &str,
    mirror: &mut Mirror,
    supervisor: &mut Supervisor,
    dev_key: Option<&DevKey>,
) {
    let row: serde_json::Value = match serde_json::from_str(data) {
        Ok(v) => v,
        Err(e) => {
            warn!(error = %e, "failed to parse message row");
            return;
        }
    };

    let Some(chat_id) = row.get("chat_id").and_then(|v| v.as_str()).map(str::to_string) else {
        warn!("message row missing chat_id");
        return;
    };
    let seq = match row.get("seq").and_then(json_to_i64) {
        Some(s) => s,
        None => {
            warn!(chat_id = %chat_id, "message row missing seq");
            return;
        }
    };
    let sender = row.get("sender").and_then(|v| v.as_str()).unwrap_or("");
    if sender != "user" {
        return;
    }

    let (project_id, last_processed) = match mirror.chats.get(&chat_id) {
        Some(c) => (c.project_id.clone(), c.last_processed_seq),
        None => {
            warn!(chat_id = %chat_id, "message arrived before chat row, skipping");
            return;
        }
    };
    if seq <= last_processed {
        return;
    }

    let Some(body_field) = row.get("body") else {
        let keys: Vec<&String> = row.as_object().map(|o| o.keys().collect()).unwrap_or_default();
        warn!(
            chat_id = %chat_id,
            seq,
            keys = ?keys,
            raw = %data,
            "message row missing body (dump of raw PowerSync row)"
        );
        return;
    };
    let Some(prompt) = crypto::decrypt_field(dev_key, body_field) else {
        warn!(
            chat_id = %chat_id,
            seq,
            body_field = %body_field,
            "failed to decrypt message body (dump of body field)"
        );
        return;
    };

    let Some(project_path) = mirror.projects.get(&project_id).cloned() else {
        warn!(chat_id = %chat_id, project_id = %project_id, "project not yet synced, skipping message");
        return;
    };

    supervisor.spawn_agent(chat_id.clone(), prompt, Some(project_path));

    if let Some(chat) = mirror.chats.get_mut(&chat_id) {
        chat.last_processed_seq = seq;
    }
}

fn json_to_i64(v: &serde_json::Value) -> Option<i64> {
    v.as_i64().or_else(|| v.as_str().and_then(|s| s.parse().ok()))
}

fn spawn_heartbeat(
    machine_id: Uuid,
    write_tx: mpsc::Sender<WriteEvent>,
    cancel: CancellationToken,
) {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(HEARTBEAT_INTERVAL);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tokio::select! {
                _ = cancel.cancelled() => return,
                _ = ticker.tick() => {
                    if write_tx
                        .send(WriteEvent::Heartbeat { machine_id })
                        .await
                        .is_err()
                    {
                        return;
                    }
                }
            }
        }
    });
}

/// Wait for in-flight writes to drain (supervisor is already empty at this
/// point). Bounded by `UPDATE_DRAIN_TIMEOUT` so a stalled backend can't block
/// the update forever — `writer`'s retry-with-backoff already keeps anything
/// un-flushed in its queue, so a post-timeout restart picks it up again.
async fn wait_for_writer_idle(writer: &Writer) -> bool {
    let deadline = Instant::now() + UPDATE_DRAIN_TIMEOUT;
    while Instant::now() < deadline {
        if writer.is_idle() {
            return true;
        }
        tokio::time::sleep(UPDATE_DRAIN_POLL).await;
    }
    false
}

#[tokio::main]
async fn main() {
    init_logging();
    info!("zucchini-spawner starting");

    let prod = load_prod_config();
    if let Some(p) = &prod {
        info!(machine_id = %p.machine_id, base_url = PROD_BASE_URL, "prod auth configured");
    } else {
        info!("no ZUCCHINI_MACHINE_ID/SPAWNER_TOKEN set — running in dev mode");
    }

    let state_path = state_path();
    let mut mirror = Mirror::load(&state_path);
    info!(
        projects = mirror.projects.len(),
        buckets = mirror.buckets.len(),
        path = %state_path.display(),
        "loaded persisted state"
    );

    let sync_config = build_sync_config(prod.as_ref(), mirror.buckets.clone());
    info!(
        base_url = %sync_config.base_url,
        client_id = %sync_config.client_id,
        "starting PowerSync sync stream"
    );
    let mut sync_rx = powersync::start(sync_config);

    let writer_config = build_writer_config(prod.as_ref());
    info!(base_url = %writer_config.base_url, "starting write API sender");
    let writer = writer::start(writer_config, DevKey::load_or_warn());
    let write_tx = writer.tx.clone();

    let heartbeat_cancel = CancellationToken::new();
    if let Some(p) = &prod {
        info!(machine_id = %p.machine_id, "starting heartbeat task");
        spawn_heartbeat(p.machine_id, write_tx.clone(), heartbeat_cancel.clone());
    }

    let (update_tx, mut update_rx) = mpsc::channel::<String>(1);
    tokio::spawn(updater::run_update_loop(update_tx));
    let mut update_pending: Option<String> = None;

    let read_key = DevKey::load_or_warn();

    let (response_tx, mut response_rx) = mpsc::channel::<AgentResponse>(256);
    let mut supervisor = Supervisor::new(response_tx);

    let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .expect("failed to register SIGTERM handler");
    let mut sigint = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())
        .expect("failed to register SIGINT handler");

    info!("zucchini-spawner ready, waiting for sync + agent responses");

    'main_loop: loop {
        tokio::select! {
            _ = sigterm.recv() => {
                info!("received SIGTERM, shutting down");
                supervisor.shutdown_all().await;
                break;
            }
            _ = sigint.recv() => {
                info!("received SIGINT, shutting down");
                supervisor.shutdown_all().await;
                break;
            }
            Some(new_version) = update_rx.recv(), if update_pending.is_none() => {
                info!(new_version = %new_version, "update available, will apply when idle");
                update_pending = Some(new_version);
                supervisor.cleanup();
            }
            Some(event) = sync_rx.recv() => {
                let changed = handle_sync_event(event, &mut mirror, &mut supervisor, read_key.as_ref()).await;
                if changed {
                    save_mirror(&state_path, &mirror);
                }
            }
            Some(resp) = response_rx.recv() => {
                match resp {
                    AgentResponse::Line { topic, content } => {
                        let _ = write_tx
                            .send(WriteEvent::AgentLine { chat_id: topic, content })
                            .await;
                    }
                    AgentResponse::Done { topic, has_result } => {
                        info!(topic = %topic, has_result, "agent done");
                        if !has_result {
                            let _ = write_tx
                                .send(WriteEvent::AgentLine {
                                    chat_id: topic.clone(),
                                    content: INTERRUPTED_RESULT.to_string(),
                                })
                                .await;
                        }
                        supervisor.remove(&topic);
                    }
                }
            }
        }

        // If an update is pending and no agents are running, stop producing
        // heartbeats, wait for the writer to drain, then swap the binary and
        // exit — launchd/systemd will respawn the new version.
        if let Some(new_version) = update_pending.as_ref() {
            if supervisor.is_empty() {
                info!(
                    current = env!("CARGO_PKG_VERSION"),
                    new = %new_version,
                    "all agents finished, draining writer before update"
                );
                heartbeat_cancel.cancel();
                let drained = wait_for_writer_idle(&writer).await;
                if !drained {
                    warn!("writer drain timed out, proceeding with update anyway");
                }
                match updater::download_and_replace(new_version).await {
                    Ok(()) => {
                        info!("binary replaced, exiting for restart");
                        break 'main_loop;
                    }
                    Err(e) => {
                        error!("update failed: {}", e);
                        update_pending = None;
                    }
                }
            }
        }
    }

    info!("zucchini-spawner exiting");
}
