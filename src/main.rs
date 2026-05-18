mod agent;
mod auth;
mod blobs;
mod claude_code;
mod crypto;
mod envelope;
mod import;
mod power;
mod powersync;
mod shell;
mod state;
mod uninstall;
mod updater;
mod writer;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use agent::{AgentResponse, Supervisor};
use auth::AuthClient;
use blobs::BlobDownloader;
use crypto::KeyStore;
use power::WakeSignal;
use powersync::{SyncConfig, SyncEvent};
use state::Mirror;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use sentry_tracing::EventFilter;
use tracing::{error, info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use uuid::Uuid;
use writer::{WriteEvent, Writer, WriterConfig};

const SENTRY_DSN: &str = "https://05d5deab2efce04e4f801af41ea39def@o4511216603234304.ingest.de.sentry.io/4511216616669264";
const INTERRUPTED_RESULT: &str = r#"{"type":"result","subtype":"interrupted"}"#;
const PROD_BASE_URL: &str = "https://api.zucchini.chat";
const DEV_SYNC_BASE_URL: &str = "http://localhost:8080";
const DEV_API_BASE_URL: &str = "http://localhost:3100";
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(60);
// Cap the wait so a server that won't accept writes can't block the update forever.
const UPDATE_DRAIN_TIMEOUT: Duration = Duration::from_secs(30);
const UPDATE_DRAIN_POLL: Duration = Duration::from_millis(100);

fn init_logging() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info,zucchini_spawner=debug"));
    let sentry_layer = sentry_tracing::layer().event_filter(|md| match *md.level() {
        tracing::Level::ERROR => EventFilter::Event,
        _ => EventFilter::Ignore,
    });
    tracing_subscriber::registry()
        .with(filter)
        .with(tracing_subscriber::fmt::layer().with_writer(std::io::stdout))
        .with(sentry_layer)
        .init();
}

pub(crate) fn zucchini_spawner_dir() -> PathBuf {
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
    home.join(".zucchini-spawner")
}

/// Prod vs dev auth mode. Prod needs both MACHINE_ID and SPAWNER_TOKEN — one without
/// the other is an install bug, so we fall through to dev rather than half-configuring.
///
/// Note: `user_id` deliberately does NOT live here. Older `install.sh` runs (pre
/// per-user keys) never wrote `ZUCCHINI_USER_ID` into `config.env`, and binary-only
/// upgrades don't re-run the installer, so any boot-time env-seed would be `None`
/// for a non-trivial slice of prod hosts. We harvest user_id lazily from the first
/// `machines` PUT in the by_machine bucket (`Mirror::user_id`) instead — single
/// source of truth, no two-paths split. `install.sh` keeps writing the env var; the
/// spawner just ignores it.
struct ProdConfig {
    machine_id: Uuid,
    auth: Arc<AuthClient>,
}

/// Parse the named env var as a UUID. Returns `None` if unset or unparseable
/// (with a warn! on parse failure).
fn env_uuid(name: &str) -> Option<Uuid> {
    std::env::var(name).ok().and_then(|s| match Uuid::parse_str(&s) {
        Ok(u) => Some(u),
        Err(e) => {
            warn!(env = name, error = %e, "env var is not a valid UUID, ignoring");
            None
        }
    })
}

fn load_prod_config() -> Option<ProdConfig> {
    let spawner_token = std::env::var("ZUCCHINI_SPAWNER_TOKEN").ok()?;
    let machine_id = env_uuid("ZUCCHINI_MACHINE_ID")?;
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

/// Pairs the right base URL with the matching token fetcher. Prod arm is
/// identical for sync and API surfaces (same auth client, same PROD_BASE_URL);
/// only the dev URL differs.
fn base_and_token(prod: Option<&ProdConfig>, dev_url: &str) -> (String, writer::TokenFetcher) {
    match prod {
        Some(p) => (PROD_BASE_URL.to_string(), auth::token_fetcher(p.auth.clone())),
        None => (dev_url.to_string(), dev_token_fetcher()),
    }
}

fn build_sync_config(
    prod: Option<&ProdConfig>,
    initial_buckets: std::collections::HashMap<String, String>,
    wake_signal: WakeSignal,
    revoked: CancellationToken,
) -> SyncConfig {
    let hostname = gethostname::gethostname().to_string_lossy().to_string();
    let (base_url, fetch_token) = base_and_token(prod, DEV_SYNC_BASE_URL);
    SyncConfig {
        base_url,
        client_id: format!("zucchini-spawner-{}", hostname),
        initial_buckets,
        fetch_token,
        wake_signal,
        revoked,
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

pub(crate) enum SyncEventOutcome {
    Nothing,
    StateChanged,
    ImportRequested,
    ImportAborted,
    UninstallRequested,
}

async fn handle_sync_event(
    event: SyncEvent,
    machine_id: Option<Uuid>,
    mirror: &mut Mirror,
    supervisor: &mut Supervisor,
    blobs: &BlobDownloader,
    keys: &KeyStore,
    write_tx: &mpsc::Sender<WriteEvent>,
) -> SyncEventOutcome {
    match event {
        SyncEvent::Put { table, id, data } => match table.as_str() {
            "projects" => {
                mirror.upsert_project(id, &data);
                SyncEventOutcome::StateChanged
            }
            "chats" => {
                mirror.upsert_chat(id, &data);
                SyncEventOutcome::Nothing
            }
            "messages" => {
                handle_message_put(&data, mirror, supervisor, blobs, keys, write_tx).await;
                SyncEventOutcome::Nothing
            }
            "machines" => {
                // Defense-in-depth: the bucket already scopes to this spawner.
                let is_self = machine_id
                    .map(|m| Uuid::parse_str(&id).map(|x| x == m).unwrap_or(false))
                    .unwrap_or(false);
                if !is_self {
                    return SyncEventOutcome::Nothing;
                }
                let Some(row) = state::parse_row_json(&data, "machine", &id) else {
                    return SyncEventOutcome::Nothing;
                };
                // Harvest user_id from the row (sync-rules.yaml `by_machine`
                // bucket includes machines.user_id via SELECT *). First-time
                // transition persists to state.json so subsequent boots have
                // it ready synchronously, before the by_machine round-trip.
                let user_id_changed = row
                    .get("user_id")
                    .and_then(|p| p.as_str())
                    .and_then(|s| Uuid::parse_str(s).ok())
                    .map(|uid| mirror.set_user_id(uid))
                    .unwrap_or(false);
                if json_pg_bool(row.get("to_uninstall")) {
                    return SyncEventOutcome::UninstallRequested;
                }
                let status = row
                    .get("claude_history_import_status")
                    .and_then(|f| f.as_str());
                match (mirror.set_import_status(status), status) {
                    (true, Some("requested")) => SyncEventOutcome::ImportRequested,
                    (true, Some("aborted")) => SyncEventOutcome::ImportAborted,
                    _ if user_id_changed => SyncEventOutcome::StateChanged,
                    _ => SyncEventOutcome::Nothing,
                }
            }
            _ => SyncEventOutcome::Nothing,
        },
        SyncEvent::Remove { table, id } => match table.as_str() {
            "chats" => {
                // PowerSync emits PUT → REMOVE → PUT on every row update (moves the
                // storage slot, not a real delete), so a lone REMOVE here does NOT
                // mean the user deleted the chat — abort-on-delete needs a separate
                // signal (TODO: debounce until CheckpointComplete shows the row is
                // really gone, or add a chats.deleted_at column).
                mirror.remove_chat(&id);
                SyncEventOutcome::Nothing
            }
            "projects" => {
                mirror.remove_project(&id);
                SyncEventOutcome::StateChanged
            }
            _ => SyncEventOutcome::Nothing,
        },
        SyncEvent::CheckpointComplete { buckets } => {
            mirror.buckets = buckets;
            SyncEventOutcome::StateChanged
        }
    }
}

async fn handle_message_put(
    data: &str,
    mirror: &mut Mirror,
    supervisor: &mut Supervisor,
    blobs: &BlobDownloader,
    keys: &KeyStore,
    write_tx: &mpsc::Sender<WriteEvent>,
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
    // Server-stamped marker for rows backfilled by the claude-history importer.
    // We subscribe to our own by_machine bucket, so every imported message
    // round-trips back as a sync PUT — without this skip the importer would
    // re-spawn an agent for every imported user prompt (potentially thousands
    // concurrent on first import).
    if json_pg_bool(row.get("imported")) {
        return;
    }

    let (project_id, worktree, chat_last_seq, user_id) = match mirror.chats.get(&chat_id) {
        Some(c) => (c.project_id.clone(), c.worktree, c.last_seq, c.user_id),
        None => {
            warn!(chat_id = %chat_id, "message arrived before chat row, skipping");
            return;
        }
    };

    // chats.last_seq is the bucket-authoritative "latest message in chat".
    // A user message with seq < last_seq is a replayed copy of an
    // already-answered prompt — happens when PowerSync re-streams the bucket
    // from op_id 0 (e.g. sync-rules rename, fresh state.json). Without this
    // guard every replayed user message respawns its long-finished agent.
    if seq < chat_last_seq {
        info!(chat_id = %chat_id, seq, chat_last_seq, "skipping replayed user message");
        return;
    }

    let Some(body_str) = row.get("body").and_then(|v| v.as_str()) else {
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
    let key = match keys.get(&user_id) {
        Ok(k) => k,
        Err(e) => {
            warn!(chat_id = %chat_id, seq, error = %e, "no key for user, skipping message");
            return;
        }
    };
    let envelope = match envelope::decode(body_str, &key) {
        Ok(env) => env,
        Err(e) => {
            warn!(
                chat_id = %chat_id,
                seq,
                error = %e,
                "failed to decode message envelope"
            );
            return;
        }
    };

    // A new user message interrupts any agent still running on this chat:
    // kill it, publish an interrupted result so the UI flips to "done", then
    // fall through to either handle /stop or start the new agent.
    let was_running = supervisor.is_running(&chat_id);
    if was_running {
        info!(chat_id = %chat_id, "aborting running agent before handling new message");
        supervisor.abort_agent(&chat_id).await;
        let _ = write_tx
            .send(WriteEvent::agent_line(chat_id.clone(), user_id, INTERRUPTED_RESULT.to_string()))
            .await;
    }

    if envelope.attachments.is_empty() && envelope.text.trim() == "/stop" {
        info!(chat_id = %chat_id, "stop command received");
        let _ = write_tx
            .send(WriteEvent::chat_running(chat_id.clone(), false))
            .await;
        return;
    }

    let Some(project_path) = mirror.projects.get(&project_id).cloned() else {
        warn!(chat_id = %chat_id, project_id = %project_id, "project not yet synced, skipping message");
        return;
    };

    let downloaded = match blobs.fetch_all(&envelope.attachments, &key).await {
        Ok(d) => d,
        Err(e) => {
            warn!(
                chat_id = %chat_id,
                seq,
                error = %e,
                "failed to download attachments, skipping message"
            );
            return;
        }
    };
    let prompt = blobs::build_prompt(&envelope.text, &downloaded);

    // Only PATCH when transitioning idle→running. The abort-then-respawn path
    // (was_running==true) leaves agent_running already true from the prior
    // spawn — re-sending it would fan out a no-op write to every listening
    // client and re-trigger their chat-list re-decrypt.
    if !was_running {
        let _ = write_tx
            .send(WriteEvent::chat_running(chat_id.clone(), true))
            .await;
    }
    let is_resume = seq > 1;
    supervisor.spawn_agent(chat_id.clone(), prompt, Some(project_path), worktree, is_resume);
}

pub(crate) fn json_to_i64(v: &serde_json::Value) -> Option<i64> {
    v.as_i64().or_else(|| v.as_str().and_then(|s| s.parse().ok()))
}

/// PowerSync serializes Postgres BOOLEAN as JSON Number 0/1, not bool — `as_bool()`
/// returns None and silently falls through to `false`. Treat absent fields as false.
pub(crate) fn json_pg_bool(v: Option<&serde_json::Value>) -> bool {
    v.and_then(|x| x.as_i64()) == Some(1)
}

/// Off the main task — the login-shell probe in `is_installed` is hundreds of
/// ms and we don't want to delay select-loop entry.
fn spawn_startup_info_report(machine_id: Uuid, write_tx: mpsc::Sender<WriteEvent>) {
    tokio::spawn(async move {
        let (installed, authenticated) = tokio::join!(
            claude_code::is_installed(),
            tokio::task::spawn_blocking(claude_code::is_authenticated),
        );
        let authenticated = authenticated.unwrap_or(false);
        info!(installed, authenticated, "reporting startup info");
        let _ = write_tx
            .send(WriteEvent::ReportStartupInfo {
                machine_id,
                claude_code_installed: installed,
                claude_code_authenticated: authenticated,
            })
            .await;
    });
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

/// Drops the op with a warn when the chat's user_id isn't mirrored yet.
async fn send_agent_line(
    write_tx: &mpsc::Sender<WriteEvent>,
    mirror: &Mirror,
    chat_id: &str,
    content: String,
) {
    match mirror.chats.get(chat_id).map(|c| c.user_id) {
        Some(user_id) => {
            let _ = write_tx
                .send(WriteEvent::agent_line(chat_id.to_string(), user_id, content))
                .await;
        }
        None => {
            warn!(chat_id = %chat_id, "agent op for chat without mirrored user_id, dropping");
        }
    }
}

/// Shared shutdown path used by both the `to_uninstall` PowerSync signal
/// and the 410-from-/auth/token revoked signal. Caller breaks the main
/// loop after this returns.
async fn run_uninstall(
    supervisor: &mut Supervisor,
    heartbeat_cancel: &CancellationToken,
    writer: &Writer,
) {
    supervisor.shutdown_all().await;
    heartbeat_cancel.cancel();
    let _ = wait_for_writer_idle(writer).await;
    uninstall::spawn_detached_cleanup();
}

#[tokio::main]
async fn main() {
    if std::env::args().any(|a| a == "--version" || a == "-V") {
        println!("{}", env!("CARGO_PKG_VERSION"));
        return;
    }

    let _sentry_guard = sentry::init((
        SENTRY_DSN,
        sentry::ClientOptions {
            release: sentry::release_name!(),
            send_default_pii: true,
            ..Default::default()
        },
    ));

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

    // In dev mode there's no AuthClient; an unsignalled token never cancels.
    let revoked_token = prod
        .as_ref()
        .map(|p| p.auth.revoked_signal())
        .unwrap_or_default();

    let wake_signal = power::start_wake_watcher();
    let sync_config = build_sync_config(
        prod.as_ref(),
        mirror.buckets.clone(),
        wake_signal,
        revoked_token.clone(),
    );
    info!(
        base_url = %sync_config.base_url,
        client_id = %sync_config.client_id,
        "starting PowerSync sync stream"
    );
    let mut sync_rx = powersync::start(sync_config);

    let keys = Arc::new(KeyStore::new());

    let (writer_base_url, writer_token) = base_and_token(prod.as_ref(), DEV_API_BASE_URL);
    info!(base_url = %writer_base_url, "starting write API sender");
    let writer = writer::start(
        WriterConfig { base_url: writer_base_url, fetch_token: writer_token },
        keys.clone(),
    );
    let write_tx = writer.tx.clone();

    let heartbeat_cancel = CancellationToken::new();
    if let Some(p) = &prod {
        info!(machine_id = %p.machine_id, "starting heartbeat task");
        spawn_heartbeat(p.machine_id, write_tx.clone(), heartbeat_cancel.clone());
        spawn_startup_info_report(p.machine_id, write_tx.clone());
    }

    let (update_tx, mut update_rx) = mpsc::channel::<String>(1);
    tokio::spawn(updater::run_update_loop(update_tx));
    let mut update_pending: Option<String> = None;

    let blob_downloader = {
        let (base_url, fetch_token) = base_and_token(prod.as_ref(), DEV_API_BASE_URL);
        BlobDownloader::new(&base_url, fetch_token)
    };

    let (response_tx, mut response_rx) = mpsc::channel::<AgentResponse>(256);
    let mut supervisor = Supervisor::new(response_tx);
    // One-shot: spawned on ImportRequested, aborted on ImportAborted. After the
    // task finishes naturally the slot stays Some(handle), but the FSM's
    // terminal `finished` blocks any further ImportRequested, so it never gets
    // reused. Aborting an already-finished JoinHandle is a no-op.
    let mut import_task: Option<tokio::task::JoinHandle<()>> = None;

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
            _ = revoked_token.cancelled() => {
                info!("auth revoked (410 from /auth/token) — running self-uninstall and exiting");
                run_uninstall(&mut supervisor, &heartbeat_cancel, &writer).await;
                break 'main_loop;
            }
            Some(new_version) = update_rx.recv(), if update_pending.is_none() => {
                info!(new_version = %new_version, "update available, will apply when idle");
                update_pending = Some(new_version);
                supervisor.cleanup();
            }
            Some(event) = sync_rx.recv() => {
                let machine_id = prod.as_ref().map(|p| p.machine_id);
                let outcome = handle_sync_event(
                    event,
                    machine_id,
                    &mut mirror,
                    &mut supervisor,
                    &blob_downloader,
                    &keys,
                    &write_tx,
                ).await;
                match outcome {
                    SyncEventOutcome::StateChanged => save_mirror(&state_path, &mirror),
                    // Claude-history import is a one-shot triggered ONLY from
                    // iOS AddMachineView, immediately after the machine row is
                    // created (see SyncStore::requestClaudeHistoryImport, sole
                    // call site AddMachineView::startImport). There is no
                    // "Import History" button anywhere else. user_id is
                    // sourced from `mirror.user_id`, harvested from the
                    // by_machine bucket's machines row — the same PUT that
                    // flipped claude_history_import_status to `requested`
                    // already populated it, so the guard below is a safety
                    // net for the impossible case where they ever diverge.
                    SyncEventOutcome::ImportRequested => 'arm: {
                        let Some(mid) = machine_id else {
                            warn!("import requested but spawner is in dev mode (no machine id) — ignoring");
                            break 'arm;
                        };
                        let Some(uid) = mirror.user_id else {
                            warn!("import requested but mirror.user_id not populated yet — ignoring");
                            break 'arm;
                        };
                        info!(machine_id = %mid, user_id = %uid, "claude history import requested by user");
                        let tx_clone = write_tx.clone();
                        let handle = tokio::spawn(async move {
                            if let Err(e) = import::run(mid, uid, tx_clone).await {
                                error!(error = %e, "claude history import failed");
                            }
                        });
                        // FSM only allows NotStarted→Requested once, so any
                        // pre-existing handle here is a leftover from a
                        // finished run — abort is a no-op on it.
                        if let Some(prev) = import_task.replace(handle) {
                            prev.abort();
                        }
                    }
                    SyncEventOutcome::ImportAborted => {
                        if let Some(handle) = import_task.take() {
                            info!("user aborted import — cancelling import task");
                            handle.abort();
                        }
                    }
                    SyncEventOutcome::UninstallRequested => {
                        info!("to_uninstall=true — running self-uninstall and exiting");
                        run_uninstall(&mut supervisor, &heartbeat_cancel, &writer).await;
                        break 'main_loop;
                    }
                    SyncEventOutcome::Nothing => {}
                }
            }
            Some(resp) = response_rx.recv() => {
                match resp {
                    AgentResponse::Line { topic, content } => {
                        send_agent_line(&write_tx, &mirror, &topic, content).await;
                    }
                    AgentResponse::ContextTokens { topic, tokens } => {
                        let _ = write_tx
                            .send(WriteEvent::ContextTokens { chat_id: topic, tokens })
                            .await;
                    }
                    AgentResponse::Done { topic, has_result } => {
                        info!(topic = %topic, has_result, "agent done");
                        if !has_result {
                            send_agent_line(&write_tx, &mirror, &topic, INTERRUPTED_RESULT.to_string()).await;
                        }
                        let _ = write_tx
                            .send(WriteEvent::chat_running(topic.clone(), false))
                            .await;
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
