mod adapter;
mod adapters;
mod agent;
mod atomic;
mod auth;
mod blobs;
mod control;
mod crypto;
mod envelope;
mod power;
mod powersync;
mod shell;
mod state;
mod uninstall;
mod updater;
mod writer;
mod x25519;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use adapter::AgentKind;
use agent::{AgentResponse, SpawnRequest, Supervisor};
use auth::AuthClient;
use blobs::BlobDownloader;
use crypto::KeyStore;
use crypto_box::SecretKey;
use power::WakeSignal;
use powersync::{SyncConfig, SyncEvent};
use sentry_tracing::EventFilter;
use state::{Mirror, SharedMirror};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use uuid::Uuid;
use writer::{WriteEvent, Writer, WriterConfig};

/// Shared probe-results cache shape — fan-out from `spawn_startup_info_report`
/// into the sync-event handler so `seed_default_agents_if_needed` can decide
/// whether claude/cursor are installed without re-shelling. `OnceLock` lets
/// the producer set it exactly once and lets readers snapshot without locking.
type ProbeStatusesCache = Arc<std::sync::OnceLock<Vec<(AgentKind, (bool, bool))>>>;

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
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    home.join(".zucchini-spawner")
}

/// Force 0o700 on the spawner dir so `key_<uuid>` filenames don't leak to
/// local cohorts on hosts where umask is the default 022. We can't pass the
/// mode to `create_dir_all` portably, so chmod after the fact — best-effort,
/// failure is logged but not fatal (matches the existing
/// "failed to ensure spawner dir exists" pattern).
pub(crate) fn ensure_spawner_dir_private(dir: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Err(e) = std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700)) {
            warn!(error = %e, path = %dir.display(), "failed to chmod 0700 on spawner dir");
        }
    }
    #[cfg(not(unix))]
    let _ = dir;
}

/// Prod vs dev auth mode. Prod needs both MACHINE_ID and SPAWNER_TOKEN — one without
/// the other is an install bug, so we fall through to dev rather than half-configuring.
///
/// `user_id` is harvested from the first `machines` PUT in the by_machine bucket
/// (`Mirror::user_id`) and persisted to state.json. As a safety net for the first
/// boot — before the machines PUT lands — we also seed `mirror.user_id` from the
/// `ZUCCHINI_USER_ID` env var written by `install.sh` (since v1+; older hosts
/// without the env var simply continue using lazy harvest). Without the seed, by_machine
/// can deliver a `machine_users` row OR a `messages` row BEFORE the `machines` PUT in
/// the same checkpoint, which would otherwise (a) misclassify the owner as a non-owner
/// and delete `key_<owner>`, (b) drop the owner's user message because the membership
/// gate has no entry yet.
struct ProdConfig {
    machine_id: Uuid,
    auth: Arc<AuthClient>,
}

/// Parse the named env var as a UUID. Returns `None` if unset or unparseable
/// (with a warn! on parse failure).
fn env_uuid(name: &str) -> Option<Uuid> {
    std::env::var(name)
        .ok()
        .and_then(|s| match Uuid::parse_str(&s) {
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
        Some(p) => (
            PROD_BASE_URL.to_string(),
            auth::token_fetcher(p.auth.clone()),
        ),
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

/// Persist `state.json`. Locks `mirror` for read internally so callers don't
/// have to juggle a guard alongside the IO; `Mirror::save` is `&self` so a
/// read guard suffices. We pay the small cost of an async lock acquire to
/// keep the call-sites in `main()` readable (compare: `let g =
/// mirror.read().await; save_mirror(&p, &*g);` at every site).
async fn save_mirror(path: &Path, mirror: &SharedMirror) {
    let g = mirror.read().await;
    if let Err(e) = g.save(path) {
        warn!(error = %e, "failed to persist state.json");
    }
}

pub(crate) enum SyncEventOutcome {
    Nothing,
    StateChanged,
    /// PowerSync delivered a CheckpointComplete — every bucket op the server
    /// has for us up to this op_id is now in `mirror`. Main uses this as the
    /// "by_machine fully streamed" trigger for one-shot boot tasks (e.g. the
    /// orphan-key reconciliation pass).
    CheckpointReached,
    ImportRequested,
    ImportAborted,
    UninstallRequested,
}

#[allow(clippy::too_many_arguments)]
async fn handle_sync_event(
    event: SyncEvent,
    machine_id: Option<Uuid>,
    mirror: &mut Mirror,
    supervisor: &mut Supervisor,
    blobs: &BlobDownloader,
    keys: &KeyStore,
    x25519_secret: Option<&SecretKey>,
    our_pubkey_b64: Option<&str>,
    write_tx: &mpsc::Sender<WriteEvent>,
    // Probe statuses fan-out cache (`spawn_startup_info_report` fills it
    // once at boot). `None` here means probes haven't completed yet — the
    // seeding pass is a no-op in that case and re-tries on the next
    // re-emission of the row (heartbeat fan-out OR after the probe lands).
    probe_statuses: Option<&[(AgentKind, (bool, bool))]>,
    // Process-lifetime "already attempted seeding once" flag. Flipped to
    // `true` after we emit the seeding PATCH; subsequent re-emissions of
    // the same `machine_users` row (heartbeat fan-out, ~every 60s) skip
    // re-seeding even if the DB hasn't transitioned NULL → `[]` yet. The
    // DB transition is the durable guard; this flag prevents a transient
    // re-seed before the round-trip lands. Restart re-checks (state.json
    // doesn't persist it).
    agents_seed_attempted: &mut bool,
) -> SyncEventOutcome {
    // `&mut Mirror` here is borrowed from the same `Arc<tokio::sync::RwLock<Mirror>>`
    // the control task reads (see `control::ControlState::mirror`); callers in
    // `main()` acquire the write guard once per sync event before invoking this
    // helper. Tests construct `Mirror::default()` directly and never involve
    // the lock.
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
                let is_self = machine_id.is_some() && parse_uuid_str(&id) == machine_id;
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
                let user_id_changed = parse_uuid_field(&row, "user_id")
                    .map(|uid| mirror.set_user_id(uid))
                    .unwrap_or(false);
                // Track the server-side spawner_pubkey value so the boot path
                // can skip the upload when it matches our on-disk secret.
                // Older backends without the column simply omit the field —
                // that's harmless, the mirror keeps `None` and the boot path
                // will upload to a server that ignores unknown PATCH keys.
                let pubkey_changed =
                    mirror.set_spawner_pubkey(row.get("spawner_pubkey").and_then(|p| p.as_str()));
                // If a server-side clear/rotation flips the column at runtime
                // (e.g. ops nukes spawner_pubkey to force a re-publish), upload
                // immediately rather than waiting for the next boot.
                if pubkey_changed {
                    if let Some(mid) = machine_id {
                        publish_spawner_pubkey_if_needed(mid, our_pubkey_b64, mirror, write_tx)
                            .await;
                    }
                }
                if json_pg_bool(row.get("to_uninstall")) {
                    return SyncEventOutcome::UninstallRequested;
                }
                let status = row
                    .get("claude_history_import_status")
                    .and_then(|f| f.as_str());
                // Snapshot the user's checkbox selection alongside the status
                // flip so the dispatcher reads the kinds the user picked at
                // the same moment they tapped "Import" — not a later
                // heartbeat-driven re-stream. Mid-import changes to the
                // column don't retro-affect the in-flight run.
                mirror.set_import_kinds(
                    row.get("claude_history_import_kinds")
                        .and_then(|f| f.as_str()),
                );
                match (mirror.set_import_status(status), status) {
                    (true, Some("requested")) => SyncEventOutcome::ImportRequested,
                    (true, Some("aborted")) => SyncEventOutcome::ImportAborted,
                    _ if user_id_changed || pubkey_changed => SyncEventOutcome::StateChanged,
                    _ => SyncEventOutcome::Nothing,
                }
            }
            "machine_users" => {
                let Some(row) = state::parse_row_json(&data, "machine_users", &id) else {
                    return SyncEventOutcome::Nothing;
                };
                let Some(user_id) = parse_uuid_field(&row, "user_id") else {
                    warn!(row_id = %id, "machine_users row missing user_id");
                    return SyncEventOutcome::Nothing;
                };
                // Members are persisted across restarts so the sandbox-spawn
                // gate doesn't fail open after the bucket cursor advances
                // past historical machine_users rows.
                let state_changed =
                    apply_machine_users_put(&id, user_id, &row, x25519_secret, keys, mirror);

                // `agents` (migration 0035) is a TEXT column carrying a JSON
                // array. Mirror the raw value (NULL → None; "[]" → Some) so
                // the seeding decision below has the freshest value, AND so
                // future per-member features (cross-user agent broadcasts,
                // etc.) have a single source of truth. The `by_machine`
                // bucket only ships the spawner's own row + other actives'
                // rows; only the OWN row matters for seeding.
                let agents_raw = row
                    .get("agents")
                    .and_then(|v| v.as_str())
                    .map(str::to_string);
                mirror.set_member_agents(&user_id, agents_raw);

                // Seeding fires when ALL of:
                //   - row is the spawner-owner's own row
                //   - column landed NULL (not `[]` — distinct values; `[]`
                //     means user emptied the list and we leave it alone)
                //   - probe results are in AND at least one CLI is installed
                //   - we haven't already attempted this process lifetime
                // The DB NULL → `[]`-or-non-empty transition is the durable
                // guard that prevents re-seeding on subsequent boots; the
                // in-memory flag covers the round-trip window before that
                // transition lands.
                if let Some(mid) = machine_id {
                    seed_default_agents_if_needed(
                        &id,
                        user_id,
                        mid,
                        mirror,
                        probe_statuses,
                        agents_seed_attempted,
                        write_tx,
                    )
                    .await;
                }

                if state_changed {
                    SyncEventOutcome::StateChanged
                } else {
                    SyncEventOutcome::Nothing
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
            "machine_users" => {
                // Owner K_user lifecycle is install.sh / uninstall.sh territory
                // — a stray REMOVE for the owner's machine_users row must not
                // delete `key_<owner>` out from under a still-running install.
                let Some(user_id) = mirror.user_for_row_id(&id) else {
                    return SyncEventOutcome::Nothing;
                };
                if mirror.is_owner(user_id) {
                    warn!(%user_id, "ignoring machine_users REMOVE for owner's own row");
                    return SyncEventOutcome::Nothing;
                }
                mirror.remove_member(&id);
                revoke_local_access(&user_id, keys);
                SyncEventOutcome::StateChanged
            }
            _ => SyncEventOutcome::Nothing,
        },
        SyncEvent::CheckpointComplete { buckets } => {
            mirror.buckets = buckets;
            SyncEventOutcome::CheckpointReached
        }
    }
}

/// Consume one `AgentResponse` (the other half of the main loop's `select!`,
/// matching `handle_sync_event`'s role on the sync side). Maps each variant
/// onto its `WriteEvent` and — for `Done` — synthesizes the canned
/// `INTERRUPTED_RESULT` line when the agent died without emitting `result`,
/// then drops the per-chat supervisor slot.
pub(crate) async fn handle_agent_response(
    resp: AgentResponse,
    mirror: &mut Mirror,
    write_tx: &mpsc::Sender<WriteEvent>,
    supervisor: &mut Supervisor,
) {
    match resp {
        AgentResponse::Line { topic, content } => {
            send_agent_line(write_tx, mirror, &topic, content).await;
        }
        AgentResponse::ContextTokens { topic, tokens } => {
            let _ = write_tx
                .send(WriteEvent::ContextTokens { chat_id: topic, tokens })
                .await;
        }
        AgentResponse::CompactBoundary { topic, post_tokens } => {
            let _ = write_tx
                .send(WriteEvent::CompactBoundary { chat_id: topic, post_tokens })
                .await;
        }
        AgentResponse::SessionIdHarvested { topic, session_id } => {
            // Stash locally first so a fast-followup user message
            // in the same chat doesn't race the writer / PowerSync
            // round-trip and spawn a second claude without --resume.
            mirror.set_agent_session_id(&topic, session_id.clone());
            let _ = write_tx
                .send(WriteEvent::AgentSessionId { chat_id: topic, session_id })
                .await;
        }
        AgentResponse::Done { topic, has_result } => {
            info!(topic = %topic, has_result, "agent done");
            if !has_result {
                send_agent_line(write_tx, mirror, &topic, INTERRUPTED_RESULT.to_string()).await;
            }
            let _ = write_tx
                .send(WriteEvent::chat_running(topic.clone(), false))
                .await;
            supervisor.remove(&topic);
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

    let Some(chat_id) = row
        .get("chat_id")
        .and_then(|v| v.as_str())
        .map(str::to_string)
    else {
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

    let (project_id, worktree, chat_last_seq, user_id, agent_session_id, agent_kind, model) =
        match mirror.chats.get(&chat_id) {
            Some(c) => (
                c.project_id.clone(),
                c.worktree,
                c.last_seq,
                c.user_id,
                c.agent_session_id.clone(),
                c.agent_kind,
                // `chats.model` (migration 0035) — filter empty strings to
                // `None` HERE (not in the adapter) so adapter logic stays
                // `if let Some(m) = ctx.model { ... }`. `state.rs::upsert_chat`
                // already does the same filter on insert, but we re-apply
                // defensively in case a future code path bypasses it.
                c.model.as_deref().filter(|s| !s.is_empty()).map(str::to_string),
            ),
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
        let keys: Vec<&String> = row
            .as_object()
            .map(|o| o.keys().collect())
            .unwrap_or_default();
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

    // Detect /stop early but defer the return: the abort+INTERRUPTED publish
    // below must still run for an explicit user stop (so the chat gets the
    // "Agent interrupted" system frame), while the membership gate that
    // follows must run BEFORE any abort so an invitee with an unsynced
    // machine_users row can never trigger a stray INTERRUPTED.
    let is_stop = envelope.attachments.is_empty() && envelope.text.trim() == "/stop";
    if is_stop {
        info!(chat_id = %chat_id, "stop command received");
    }

    // Sandbox gate: owners (`user_id == mirror.user_id`) are never sandboxed
    // per the schema invariant — `machine_users.is_sandboxed` only applies to
    // invitees. For invitees, refuse to spawn if the `machine_users` row hasn't
    // been mirrored yet; an unknown member would otherwise default to NOT
    // sandboxed, and PowerSync resumes from the saved bucket cursor so a row
    // that streamed in a previous boot won't be re-emitted after restart.
    //
    // Evaluated BEFORE the abort+INTERRUPTED publish below so an invitee whose
    // membership row hasn't synced can't trigger a stray agent abort on an
    // already-running chat. Also BEFORE `chat_running=true`: a `return` here
    // after sending true would strand the UI on a perpetual spinner.
    let is_owner = mirror.is_owner(user_id);
    let is_sandboxed = if is_owner {
        false
    } else {
        let Some(s) = mirror.member_is_sandboxed(&user_id) else {
            warn!(
                chat_id = %chat_id,
                %user_id,
                "machine_users row not synced for user, skipping message"
            );
            return;
        };
        s
    };

    // Membership gate passed — safe to abort any running agent on this chat
    // and publish INTERRUPTED_RESULT. Runs for both the explicit /stop button
    // and the "interrupt then send new message" UX (non-/stop body arriving
    // while an agent is still running). For /stop with no running agent this
    // is a no-op (no INTERRUPTED frame emitted out of nowhere).
    let was_running = supervisor.is_running(&chat_id);
    if was_running {
        info!(chat_id = %chat_id, "aborting running agent before handling new message");
        supervisor.abort_agent(&chat_id).await;
        let _ = write_tx
            .send(WriteEvent::agent_line(
                chat_id.clone(),
                user_id,
                INTERRUPTED_RESULT.to_string(),
            ))
            .await;
    }

    if is_stop {
        // Explicit stop: signal chat_running=false (idempotent if already
        // false) and return. The abort+INTERRUPTED above (if was_running)
        // is the full story — don't spawn a replacement agent.
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
    // Resume keys off `agent_session_id`, not `seq>1`: a freshly created chat
    // may already have a backfilled `agent_session_id` (pre-migration rows) and
    // a brand-new chat where the first turn aborted before harvest still has
    // `agent_session_id = None`, so we want a fresh session there too.
    // `agent_kind` picks the adapter (claude/cursor) at spawn time.
    supervisor.spawn_agent(SpawnRequest {
        chat_id: chat_id.clone(),
        prompt,
        project_path: Some(project_path),
        worktree,
        agent_session_id,
        agent_kind,
        is_sandboxed,
        model,
    });
}

/// Seed `machine_users.agents` defaults (migration 0035) when the owner's
/// row lands with `agents IS NULL`. Emits exactly one
/// `WriteEvent::SetMachineUserAgents` carrying a default list — one entry
/// per installed CLI per `probe_statuses` — and flips the
/// `agents_seed_attempted` in-memory guard to skip re-seeding on the next
/// heartbeat-driven re-emission.
///
/// The contract with the iOS app: NULL means "spawner hasn't seeded yet"
/// and `[]` means "user explicitly removed everything". So we only seed
/// on NULL, never on `[]`. The DB write (NULL → non-NULL) is the durable
/// guard against repeat-seeding across boots; the in-memory flag handles
/// the round-trip window before the write lands and is re-streamed.
///
/// Defaults shape, deterministic order:
///   1. `{id: <uuid-v7>, agent_kind: "claude", model: "", name: ""}` (if claude installed)
///   2. `{id: <uuid-v7>, agent_kind: "cursor", model: "", name: ""}` (if cursor installed)
///
/// If neither is installed, no PATCH is emitted (would be `[]`, which the
/// contract treats as user-emptied — never the spawner's call to make).
async fn seed_default_agents_if_needed(
    row_id: &str,
    row_user_id: Uuid,
    machine_id: Uuid,
    mirror: &Mirror,
    probe_statuses: Option<&[(AgentKind, (bool, bool))]>,
    agents_seed_attempted: &mut bool,
    write_tx: &mpsc::Sender<WriteEvent>,
) {
    if *agents_seed_attempted {
        return;
    }
    if !mirror.is_owner(row_user_id) {
        return;
    }
    // NULL = "not seeded yet" → seed; Some(_) (including `"[]"`) = leave alone.
    if mirror.member_agents(&row_user_id).is_some() {
        return;
    }
    let Some(statuses) = probe_statuses else {
        // Probes haven't completed yet — the next re-emission of this row
        // will retry. (Heartbeat re-fans `machine_users` rows every ~60s,
        // probe latency is typically < 1s, so this race window is short.)
        return;
    };
    // Closed deterministic order: claude first, then cursor — matches the
    // task spec so iOS sees a stable agent-picker ordering across builds.
    let mut entries: Vec<serde_json::Value> = Vec::new();
    for kind in AgentKind::ALL {
        let installed = statuses
            .iter()
            .find_map(|(k, (i, _))| (k == kind).then_some(*i))
            .unwrap_or(false);
        if !installed {
            continue;
        }
        let id = Uuid::now_v7();
        entries.push(serde_json::json!({
            "id": id.to_string(),
            "agent_kind": kind.descriptor().wire_name,
            "model": "",
            "name": "",
        }));
    }
    if entries.is_empty() {
        // No CLI installed → nothing to seed. Skip without setting the
        // attempted-flag so a future re-emission (after the user runs
        // `claude /login` / `cursor-agent login` and restarts the spawner)
        // can still seed. NOTE: probe results don't re-fire in-process
        // today, so this branch effectively waits for a process restart.
        return;
    }
    let Some(row_uuid) = parse_uuid_str(row_id) else {
        warn!(row_id, "machine_users.id is not a UUID; cannot seed agents");
        return;
    };
    let agents_json =
        serde_json::to_string(&entries).expect("array of small objects is always serializable");
    info!(
        %row_uuid,
        n = entries.len(),
        agents_json = %agents_json,
        "seeding default machine_users.agents for owner row"
    );
    *agents_seed_attempted = true;
    let _ = write_tx
        .send(WriteEvent::SetMachineUserAgents {
            row_id: row_uuid,
            machine_id,
            agents_json,
        })
        .await;
}

fn apply_machine_users_put(
    row_id: &str,
    user_id: Uuid,
    row: &serde_json::Value,
    x25519_secret: Option<&SecretKey>,
    keys: &KeyStore,
    mirror: &mut Mirror,
) -> bool {
    let is_sandboxed_new = json_pg_bool(row.get("is_sandboxed"));
    let sealed_b64 = row.get("sealed_blob").and_then(|v| v.as_str());

    // Owner short-circuit, symmetric with the REMOVE arm. Backend doesn't
    // populate `sealed_blob` on the owner's row today, but any future bug /
    // migration that does so would otherwise archive `key_<owner>` and
    // overwrite it with whatever the blob decrypts to, bricking historical
    // ciphertext. Refuse to touch the owner's key file from this path —
    // applies to both empty-blob (soft-revoke) and non-empty-blob arms.
    if mirror.is_owner(user_id) {
        return mirror.upsert_member(row_id.to_string(), user_id, is_sandboxed_new);
    }

    // Server-side soft-revoke = patch `sealed_blob` to NULL/empty while keeping
    // the row. Treat that as "tear down local access for this user".
    let Some(sealed_b64) = sealed_b64.filter(|s| !s.is_empty()) else {
        revoke_local_access(&user_id, keys);
        // Keep the membership row so `member_is_sandboxed` still gates spawns;
        // clear the cached blob so a future non-empty PUT is treated as fresh.
        let upserted = mirror.upsert_member(row_id.to_string(), user_id, is_sandboxed_new);
        let cleared = mirror.clear_sealed_blob(&user_id);
        return upserted || cleared;
    };

    // Cache-hit short-circuit must run before any `is_sandboxed` mutation:
    // letting a server-only flip of `is_sandboxed` (without rotating
    // `sealed_blob`) take effect would mean an attacker who only controls the
    // is_sandboxed column can flip the sandbox bit while keeping a prior
    // K_machine. Rebind is_sandboxed only after the blob has been re-validated.
    if mirror.member_sealed_blob_matches(&user_id, sealed_b64) {
        return mirror.upsert_member(row_id.to_string(), user_id, is_sandboxed_new);
    }
    let Some(secret) = x25519_secret else {
        // Still record the member entry so `member_is_sandboxed` returns
        // Some(_), otherwise every inbound message from this invitee is silently
        // dropped by the spawn gate and the PowerSync cursor advances with no
        // retry. The decrypt path will then fail loudly with "no key for user"
        // — surfaced via warn! — instead of failing invisibly. Log at error so
        // operators can see the missing-secret root cause in Sentry.
        //
        // Fail-closed on the sandbox bit: we never validated this row's
        // sealed_blob against our key, so an attacker who only controls the
        // is_sandboxed column could otherwise flip the bit. Force-sandbox
        // until a future boot with a real x25519 secret re-validates.
        error!(row_id = %row_id, %user_id, "sealed_blob present but spawner has no x25519 secret, recording member without key");
        return mirror.upsert_member(row_id.to_string(), user_id, true);
    };
    let plaintext = match x25519::open_sealed(secret, sealed_b64) {
        Ok(p) => p,
        Err(e) => {
            error!(row_id = %row_id, %user_id, error = %e, "sealed_blob open failed");
            return false;
        }
    };
    if plaintext.len() != 32 {
        error!(row_id = %row_id, %user_id, len = plaintext.len(), "sealed_blob plaintext not 32 bytes");
        return false;
    }
    if let Err(e) = persist_user_key(&user_id, &plaintext) {
        // AlreadyExists = K_machine mismatch (see `persist_user_key` doc);
        // skip upsert_member/record_sealed_blob so the mirror doesn't pretend
        // the new sealed_blob landed.
        error!(row_id = %row_id, %user_id, error = %e, "failed to persist sealed K_machine; skipping mirror upsert");
        return false;
    }
    // Sealed_blob validated and persisted — only NOW is it safe to bind the
    // new is_sandboxed value into the in-memory mirror. No
    // `keys.forget` here: K_machine is not rotated within an active
    // membership (see persist_user_key doc), so any cached Arc<KUser> in the
    // KeyStore is either absent (first PUT for this user) or already the
    // matching one (idempotent re-emit). The REMOVE branch is the sole
    // forget point.
    mirror.upsert_member(row_id.to_string(), user_id, is_sandboxed_new);
    mirror.record_sealed_blob(&user_id, sealed_b64);
    info!(%user_id, "stored K_machine from machine_users.sealed_blob");
    // record_sealed_blob persists the new blob → state changed even if the
    // upsert was a no-op on row_id/is_sandboxed.
    true
}

/// Persist `bytes` as the base64-encoded K_machine for `user_id`.
///
/// K_machine is minted once per membership (iOS `KMachine.generate()` at
/// `put_machine` for the owner and `acceptInvitation` for the invitee) and
/// is NEVER rotated within an active membership — see the lifecycle in
/// sync-rules.yaml `by_machine` (only `status='active'` rows are streamed)
/// plus `remove_membership` (hard DELETE → REMOVE arrives before any
/// re-invite's PUT, lower op_id per bucket). So the only call patterns are:
///
///   - File missing → write fresh.
///   - File present, byte-identical → no-op (heartbeat re-emit of the same
///     `sealed_blob`, or a recovered cache after restart).
///   - File present, DIFFERENT bytes → refuse loudly. This can only happen
///     via (1) a lost REMOVE due to offline + bucket compaction, (2) a
///     direct SQL UPDATE bypassing the invitation flow, or (3) a future bug.
///     All three are operator-visible incidents; silent overwrite would
///     orphan every existing ciphertext written under the prior key. The
///     caller catches `AlreadyExists` and skips the `upsert_member` /
///     `record_sealed_blob` so a loud Sentry event is produced and the
///     incident is surfaced.
fn persist_user_key(user_id: &Uuid, bytes: &[u8]) -> std::io::Result<()> {
    let path = crypto::user_key_path(user_id);
    match crypto::read_b64_32(&path) {
        Ok(existing) => {
            if existing.as_slice() == bytes {
                return Ok(());
            }
            error!(
                %user_id,
                path = %path.display(),
                "K_machine mismatch with existing key file; refusing to overwrite (manual operator intervention required)"
            );
            Err(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                "K_machine mismatch with existing key file",
            ))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            let encoded = crypto::encode_b64_zeroized(bytes);
            atomic::atomic_write_private(&path, encoded.as_bytes())
        }
        Err(e) => Err(e),
    }
}

/// One-shot boot pass to delete `key_<uuid>` files left behind by offline
/// revocations + bucket compaction. Called after the first CheckpointComplete
/// on by_machine, when `mirror.members` reflects the server-authoritative
/// active membership set. Owner's key file is always preserved (install.sh /
/// uninstall.sh territory).
fn reconcile_key_files(mirror: &Mirror, keys: &KeyStore) {
    let dir = zucchini_spawner_dir();
    let entries = match std::fs::read_dir(&dir) {
        Ok(e) => e,
        Err(e) => {
            warn!(error = %e, "orphan-key reconciliation: failed to read spawner dir");
            return;
        }
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(s) = name.to_str() else { continue };
        // Strict `key_<uuid>` match — ignore the legacy `key` file,
        // `state.json`, `x25519_secret`, etc. (No rotated `.prev.<ts>`
        // archives exist anymore — persist_user_key refuses mismatches
        // rather than archiving.)
        let Some(uid_str) = s.strip_prefix("key_") else {
            continue;
        };
        if uid_str.contains('.') {
            continue;
        }
        let Some(uid) = parse_uuid_str(uid_str) else {
            continue;
        };
        if mirror.user_id == Some(uid) {
            continue;
        }
        if mirror.has_member(&uid) {
            continue;
        }
        info!(user_id = %uid, "orphan-key reconciliation: removing key file for non-member");
        // Only forget when removal actually succeeded — same rationale as
        // the SyncEvent::Remove branch.
        match remove_user_key_file(&uid) {
            Ok(()) => {
                keys.forget(&uid);
            }
            Err(e) => {
                error!(user_id = %uid, error = %e, "orphan-key reconciliation: remove failed; keeping in-memory cache to mirror on-disk state");
            }
        }
    }
}

/// Tear down local access for `user_id`: remove the key file from disk, and
/// only on success drop the in-memory cache too. If removal failed (EROFS, MAC
/// denial, immutable bit), keep the cache so it mirrors on-disk state — the
/// next `keys.get(uid)` would otherwise re-load from disk and silently restore
/// decryption for the revoked member. Shared by the `machine_users` REMOVE arm
/// and the soft-revoke (empty `sealed_blob`) arm.
fn revoke_local_access(user_id: &Uuid, keys: &KeyStore) {
    match remove_user_key_file(user_id) {
        Ok(()) => {
            keys.forget(user_id);
        }
        Err(e) => {
            warn!(%user_id, error = %e, "failed to remove key file on revoke; keeping in-memory cache to mirror on-disk state");
        }
    }
}

/// `NotFound` is fine: invited member who never sent a message has no key file.
fn remove_user_key_file(user_id: &Uuid) -> std::io::Result<()> {
    let path = crypto::user_key_path(user_id);
    match std::fs::remove_file(&path) {
        Ok(()) => {
            info!(%user_id, path = %path.display(), "removed user key file on member removal");
            Ok(())
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e),
    }
}

pub(crate) fn json_to_i64(v: &serde_json::Value) -> Option<i64> {
    v.as_i64()
        .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
}

pub(crate) fn parse_uuid_str(s: &str) -> Option<Uuid> {
    Uuid::parse_str(s).ok()
}

pub(crate) fn parse_uuid_field(row: &serde_json::Value, field: &str) -> Option<Uuid> {
    row.get(field)
        .and_then(|v| v.as_str())
        .and_then(parse_uuid_str)
}

/// PowerSync serializes Postgres BOOLEAN as JSON Number 0/1, not bool — `as_bool()`
/// returns None and silently falls through to `false`. Treat absent fields as false.
pub(crate) fn json_pg_bool(v: Option<&serde_json::Value>) -> bool {
    v.and_then(|x| x.as_i64()) == Some(1)
}

/// Off the main task — the login-shell probes (one per agent) each shell out
/// for hundreds of ms, so we run them concurrently via `join_all` and don't
/// block select-loop entry. Iterates `AgentKind::ALL` so adding a new variant
/// requires no edits here.
///
/// `cache` is the shared snapshot the sync-event handler consults on each
/// `machine_users` PUT to decide whether to seed `agents` defaults
/// (migration 0035). Filled once at startup; subsequent `claude /login` /
/// `cursor-agent login` flips show up on the next spawner restart, same as
/// the wire-side install-status report — there's no live re-probe.
fn spawn_startup_info_report(
    machine_id: Uuid,
    write_tx: mpsc::Sender<WriteEvent>,
    cache: ProbeStatusesCache,
) {
    tokio::spawn(async move {
        let statuses: Vec<(AgentKind, (bool, bool))> = futures_util::future::join_all(
            AgentKind::ALL
                .iter()
                .map(|kind| async move { (*kind, kind.probe().await) }),
        )
        .await;
        info!(?statuses, "reporting startup info");
        // Best-effort: if the cache is already populated (impossible under
        // the current "spawn once at boot" call pattern, but cheap defense)
        // we keep the first value and log nothing — the wire-side report
        // below still runs.
        let _ = cache.set(statuses.clone());
        let _ = write_tx
            .send(WriteEvent::ReportStartupInfo {
                machine_id,
                statuses,
            })
            .await;
    });
}

/// Dispatcher for the history import. Fans out across the user-selected
/// `kinds` sequentially (kinds run one after the other so the writer's
/// batch channel — capacity 1024 — never sees both importers piling on at
/// once; iOS's blocking import sheet is the lock that lets us assume
/// single-tenant access to the channel for the duration). Each kind's
/// 0..=100 progress is rescaled into its slice of the shared 0..99 bar so
/// iOS sees one continuous progress bar across all selected kinds.
///
/// `kinds` comes from `mirror.parsed_import_kinds()` which reads
/// `claude_history_import_kinds` (CSV the iOS modal writes alongside the
/// `requested` status flip) and falls back to `AgentKind::ALL` when the
/// column is absent / NULL — older iOS without the checkbox UI keeps the
/// historic both-kinds behavior.
///
/// Status-emission contract (owned here, not in per-kind `import` fns):
///  - Emit `running-0` once at the very start.
///  - Per-kind `progress(pct)` callback emits `running-{scaled}` via
///    `write_tx.try_send` — `ImportStatus` is a tiny machines PATCH, channel
///    backpressure that drops one of these is acceptable (the next 5%-step
///    will reapply the correct value), so we log on a failed `try_send` but
///    don't await.
///  - Emit `finished` exactly once at the very end, after every kind has
///    been attempted. EXCEPTION: if every kind errored, leave the status at
///    its last `running-{scaled}` value — the backend FSM permits
///    `Running(n) → Running(m)`, so the next "Import History" request from
///    iOS would have to come via the FSM's `Running → Aborted → NotStarted →
///    Requested` path anyway. Logging `error!` here surfaces in Sentry.
///
/// Per-kind errors are logged-and-continued (matches the existing
/// per-session warn-and-skip posture inside the claude importer): one kind
/// failing doesn't strand the user with no chats from the other kind.
async fn run_history_import(
    machine_id: Uuid,
    user_id: Uuid,
    write_tx: mpsc::Sender<WriteEvent>,
    kinds: Vec<AgentKind>,
) {
    // Helper: try to push a status update; channel cap is 1024 and import
    // status writes are sparse (≤20 per kind via the 5%-step throttle), so
    // a full channel here means the writer is wedged on a network failure
    // — log and move on rather than blocking the dispatcher.
    fn emit_status(tx: &mpsc::Sender<WriteEvent>, machine_id: Uuid, status: String) {
        if let Err(e) = tx.try_send(WriteEvent::ImportStatus {
            machine_id,
            status: status.clone(),
        }) {
            warn!(error = %e, %status, "dropped import status update (writer channel full?)");
        }
    }

    // Caller (parsed_import_kinds) already falls back to ALL on
    // None/empty/all-unknown — so an empty Vec here would be a programmer
    // bug. Don't divide by zero in the rescaler; emit finished and bail.
    if kinds.is_empty() {
        warn!("run_history_import called with empty kinds list — nothing to do");
        emit_status(&write_tx, machine_id, "finished".to_string());
        return;
    }

    emit_status(&write_tx, machine_id, "running-0".to_string());

    let n = kinds.len() as u32;
    let mut ok_count = 0usize;
    for (idx, kind) in kinds.iter().enumerate() {
        let idx_u32 = idx as u32;
        let tx_for_progress = write_tx.clone();
        // Per-kind rescaler. `scaled = (idx*100 + pct) / N`, capped at 99
        // (the dispatcher owns 100/finished). Adapter callbacks are
        // throttled internally (5%-step), so we don't bother dedup-ing
        // duplicate scaled values here — at most ~20 fires per kind × N
        // kinds = ~40 PATCHes per import.
        let progress: crate::adapter::ImportProgress = Box::new(move |pct: u8| {
            let pct = pct.min(100) as u32;
            let scaled = ((idx_u32 * 100 + pct) / n).min(99) as u8;
            emit_status(&tx_for_progress, machine_id, format!("running-{scaled}"));
        });

        match kind
            .import(machine_id, user_id, write_tx.clone(), progress)
            .await
        {
            Ok(()) => {
                info!(?kind, "history import kind completed");
                ok_count += 1;
            }
            Err(e) => {
                error!(?kind, error = %e, "history import kind failed, continuing with next kind");
            }
        }
    }

    if ok_count == 0 {
        // Every kind errored. Leave the row at the last `running-X` — the
        // backend FSM allows `Running → Running`, and there's no path back
        // to `Requested` from `Finished`, so emitting `finished` here would
        // permanently strand the machine with no transcripts. Surfaces in
        // Sentry via the `error!` level.
        error!(
            n_kinds = kinds.len(),
            "every kind failed during history import; leaving status at last running-X (no `finished` emitted)"
        );
        return;
    }

    emit_status(&write_tx, machine_id, "finished".to_string());
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
                .send(WriteEvent::agent_line(
                    chat_id.to_string(),
                    user_id,
                    content,
                ))
                .await;
        }
        None => {
            warn!(chat_id = %chat_id, "agent op for chat without mirrored user_id, dropping");
        }
    }
}

/// Server-side `spawner_pubkey` is the source of truth — only the by_machine
/// round-trip flips `mirror.spawner_pubkey`. Marking it locally on a partial
/// server-side write (older backend, missing column) would skip the upload
/// forever.
async fn publish_spawner_pubkey_if_needed(
    machine_id: Uuid,
    our_pubkey: Option<&str>,
    mirror: &Mirror,
    write_tx: &mpsc::Sender<WriteEvent>,
) {
    let Some(our_pubkey) = our_pubkey else { return };
    if mirror.spawner_pubkey.as_deref() == Some(our_pubkey) {
        info!(%machine_id, "spawner_pubkey already published, skipping upload");
        return;
    }
    info!(%machine_id, "publishing spawner_pubkey for machine sharing");
    let _ = write_tx
        .send(WriteEvent::SetSpawnerPubkey {
            machine_id,
            pubkey_b64: our_pubkey.to_string(),
        })
        .await;
}

/// CLI entry point for `zucchini-spawner attach-file --chat-id <UUID> <abs-path>`.
///
/// Parses argv (hand-rolled — clap is overkill for one flag and a positional
/// arg), connects to the daemon's `~/.zucchini-spawner/control.sock`, runs
/// one `attach_file` RPC, and prints a human-readable result. Exits 0 on
/// success, 1 on any failure. The CLI itself does no crypto or HTTP — the
/// daemon owns the JWT and K_user.
async fn run_attach_file_cli(args: &[String]) {
    fn usage_and_exit() -> ! {
        eprintln!("usage: zucchini-spawner attach-file --chat-id <UUID> <absolute-path>");
        std::process::exit(2);
    }

    let mut chat_id: Option<String> = None;
    let mut path: Option<String> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--chat-id" => {
                chat_id = it.next().cloned();
            }
            "-h" | "--help" => usage_and_exit(),
            s if s.starts_with("--chat-id=") => {
                chat_id = Some(s["--chat-id=".len()..].to_string());
            }
            s if path.is_none() => path = Some(s.to_string()),
            _ => usage_and_exit(),
        }
    }
    let (Some(chat_id), Some(path)) = (chat_id, path) else {
        usage_and_exit();
    };

    match control::attach_file_via_socket(&chat_id, &path).await {
        Ok((blob_key, name)) => {
            // Stable, machine-parseable first line so the agent can grep for
            // success; details follow on stderr. Keep `blob_key` in here so a
            // future debug dump in a transcript can still cross-reference.
            println!("attached {name} ({blob_key}) to chat {chat_id}");
            std::process::exit(0);
        }
        Err(e) => {
            eprintln!("attach-file failed: {e:#}");
            std::process::exit(1);
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

    // CLI subcommand dispatch — keeps the binary single-entry so the daemon
    // and the agent-side `attach-file` client are the same on-disk
    // executable. Hand-rolled (no clap) because we have a single subcommand
    // and the daemon path needs no parsing. The subcommand is a thin RPC
    // client over `~/.zucchini-spawner/control.sock`; secrets/JWT/K_user
    // never leave the long-running daemon process.
    let raw_args: Vec<String> = std::env::args().skip(1).collect();
    if let Some(first) = raw_args.first() {
        if first == "attach-file" {
            run_attach_file_cli(&raw_args[1..]).await;
            return;
        }
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
    // `Mirror` is shared with the control socket task (see
    // `control::ControlState::mirror`). Wrapped in `Arc<tokio::sync::RwLock<…>>`
    // so the control task can take a read guard for `chat_id → user_id`
    // lookups while the main loop holds the write guard across `.await`s in
    // `handle_sync_event` / `handle_agent_response`. Must be `tokio::sync` —
    // both call sites yield under the guard.
    let mirror: SharedMirror = Arc::new(tokio::sync::RwLock::new(Mirror::load(&state_path)));
    {
        let g = mirror.read().await;
        info!(
            projects = g.projects.len(),
            buckets = g.buckets.len(),
            path = %state_path.display(),
            "loaded persisted state"
        );
    }

    // Seed `mirror.user_id` from the env var written by install.sh
    // BEFORE the sync loop starts, so the owner-check (`is_owner = ... == mirror.user_id`)
    // works regardless of bucket op ordering on first boot. Without this, a by_machine
    // checkpoint that delivers our own `machine_users` row before the `machines` PUT
    // would misclassify the owner as a non-owner and delete `key_<owner>`; a `messages`
    // PUT in that same window would be dropped because the membership gate sees no entry.
    // `set_user_id` is no-op when `user_id` is already populated (e.g. from state.json),
    // so older hosts without the env var fall back to lazy harvest unchanged.
    {
        let mut g = mirror.write().await;
        if g.user_id.is_none() {
            if let Some(env_uid) = env_uuid("ZUCCHINI_USER_ID") {
                if g.set_user_id(env_uid) {
                    info!(user_id = %env_uid, "seeded mirror.user_id from ZUCCHINI_USER_ID");
                    drop(g);
                    save_mirror(&state_path, &mirror).await;
                }
            }
        }
    }

    // In dev mode there's no AuthClient; an unsignalled token never cancels.
    let revoked_token = prod
        .as_ref()
        .map(|p| p.auth.revoked_signal())
        .unwrap_or_default();

    let wake_signal = power::start_wake_watcher();
    let initial_buckets = mirror.read().await.buckets.clone();
    let sync_config = build_sync_config(
        prod.as_ref(),
        initial_buckets,
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

    // Hoist dir creation here so per-message persist_user_key on the hot path
    // doesn't recheck it (heartbeat fan-out re-runs every machine_users PUT).
    let spawner_dir = zucchini_spawner_dir();
    if let Err(e) = std::fs::create_dir_all(&spawner_dir) {
        warn!(error = %e, "failed to ensure spawner dir exists");
    }
    ensure_spawner_dir_private(&spawner_dir);

    // Machine-sharing handshake (best-effort): load or generate the X25519
    // sealedbox secret. Generation/persistence failure is logged but
    // non-fatal — older spawners that never get a secret simply never
    // participate in sharing; the existing single-user flow is unaffected
    // because nothing else depends on this secret.
    let x25519_secret: Option<SecretKey> = match x25519::load_or_generate_secret() {
        Ok(s) => Some(s),
        Err(e) => {
            warn!(error = %e, "failed to load/generate x25519 secret — machine sharing disabled this boot");
            None
        }
    };
    let our_pubkey_b64: Option<String> = x25519_secret.as_ref().map(x25519::public_key_b64);

    let (writer_base_url, writer_token) = base_and_token(prod.as_ref(), DEV_API_BASE_URL);
    info!(base_url = %writer_base_url, "starting write API sender");
    let writer = writer::start(
        WriterConfig {
            base_url: writer_base_url,
            fetch_token: writer_token,
        },
        keys.clone(),
    );
    let write_tx = writer.tx.clone();

    // Probe results cache: populated once by `spawn_startup_info_report`,
    // read by `seed_default_agents_if_needed` on every `machine_users` PUT.
    // `OnceLock` (not `RwLock`) because the write happens exactly once and
    // the read side never blocks. `None` means "probes not in yet"; the
    // seeding pass treats that as "skip this PUT, retry on the next
    // re-emission".
    let probe_statuses_cache: ProbeStatusesCache = Arc::new(std::sync::OnceLock::new());

    let heartbeat_cancel = CancellationToken::new();
    if let Some(p) = &prod {
        info!(machine_id = %p.machine_id, "starting heartbeat task");
        spawn_heartbeat(p.machine_id, write_tx.clone(), heartbeat_cancel.clone());
        spawn_startup_info_report(p.machine_id, write_tx.clone(), probe_statuses_cache.clone());

        // Startup pubkey publish: read guard is fine — we only inspect
        // `mirror.spawner_pubkey`. The call inside `handle_sync_event` runs
        // under the write guard already; both paths use the same helper.
        let g = mirror.read().await;
        publish_spawner_pubkey_if_needed(
            p.machine_id,
            our_pubkey_b64.as_deref(),
            &g,
            &write_tx,
        )
        .await;
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

    // Control socket for agent-side CLI subcommands (`attach-file`). Bound
    // before we start consuming sync events so a fast `attach-file` issued
    // right after a chat-created PUT can connect. Failure to bind is logged
    // but non-fatal — sending files back from the agent is a feature, not a
    // hard requirement; the rest of the spawner still works. The control
    // task shares `mirror` (the same `Arc<tokio::sync::RwLock<Mirror>>` the
    // main loop holds) for chat → user_id lookups — no parallel projection.
    {
        let (api_base_url, api_token) = base_and_token(prod.as_ref(), DEV_API_BASE_URL);
        let control_state = control::ControlState {
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(60))
                .build()
                .expect("control http client"),
            api_base_url,
            fetch_token: Arc::new(api_token),
            keys: keys.clone(),
            pending: supervisor.pending_attachments(),
            mirror: mirror.clone(),
        };
        if let Err(e) = control::start(control_state).await {
            warn!(error = %e, "failed to start control socket — `zucchini-spawner attach-file` will be unavailable");
        }
    }
    // One-shot: spawned on ImportRequested, aborted on ImportAborted. After the
    // task finishes naturally the slot stays Some(handle), but the FSM's
    // terminal `finished` blocks any further ImportRequested, so it never gets
    // reused. Aborting an already-finished JoinHandle is a no-op.
    let mut import_task: Option<tokio::task::JoinHandle<()>> = None;

    let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .expect("failed to register SIGTERM handler");
    let mut sigint = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())
        .expect("failed to register SIGINT handler");

    // Latched on the FIRST CheckpointComplete: by then `mirror.members`
    // reflects everything by_machine had to send, so we can safely sweep
    // `key_<uuid>` files for non-members. Process-lifetime flag.
    let mut key_files_reconciled = false;

    // Process-lifetime guard for the `machine_users.agents` seeding pass
    // (migration 0035 — see `seed_default_agents_if_needed`). Once we
    // emit the seeding PATCH this flag flips to `true` so heartbeat-driven
    // re-emissions of the owner's row don't re-seed before the DB
    // transition lands. Restart re-checks via the DB NULL → non-NULL
    // durable guard.
    let mut agents_seed_attempted = false;

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
                let probe_snap = probe_statuses_cache.get().map(|v| v.as_slice());
                // Take the write guard once per sync event. `handle_sync_event`
                // is `async fn` and yields under the guard (decrypt, R2
                // download, writer-channel sends) — that's exactly what
                // `tokio::sync::RwLock` is for. The control task's read guards
                // can interleave only between events, never mid-event.
                let outcome = {
                    let mut g = mirror.write().await;
                    handle_sync_event(
                        event,
                        machine_id,
                        &mut g,
                        &mut supervisor,
                        &blob_downloader,
                        &keys,
                        x25519_secret.as_ref(),
                        our_pubkey_b64.as_deref(),
                        &write_tx,
                        probe_snap,
                        &mut agents_seed_attempted,
                    ).await
                };
                match outcome {
                    SyncEventOutcome::StateChanged => save_mirror(&state_path, &mirror).await,
                    SyncEventOutcome::CheckpointReached => {
                        save_mirror(&state_path, &mirror).await;
                        // Gate reconcile on `mirror.user_id.is_some()` so
                        // the FIRST CheckpointComplete from a by_user-only
                        // checkpoint (dev mode without by_machine, or first
                        // boot before the by_machine bucket has streamed)
                        // doesn't sweep every dev-placed `key_<uuid>` file.
                        // The latch flips only once the by_machine round-trip
                        // has populated mirror.user_id, after which subsequent
                        // CheckpointComplete events reflect full membership.
                        let g = mirror.read().await;
                        if !key_files_reconciled && g.user_id.is_some() {
                            key_files_reconciled = true;
                            reconcile_key_files(&g, &keys);
                        }
                    }
                    // History import is a one-shot triggered ONLY from iOS
                    // AddMachineView, immediately after the machine row is
                    // created (see SyncStore::requestClaudeHistoryImport,
                    // sole call site AddMachineView::startImport). There is
                    // no "Import History" button anywhere else. user_id is
                    // sourced from `mirror.user_id`, harvested from the
                    // by_machine bucket's machines row — the same PUT that
                    // flipped claude_history_import_status to `requested`
                    // already populated it, so the guard below is a safety
                    // net for the impossible case where they ever diverge.
                    //
                    // Multi-kind fan-out: we iterate `AgentKind::ALL`
                    // sequentially, rescaling each kind's 0..=100 progress
                    // into its slice of the shared 0..99 bar (kind i of N
                    // takes `i/N .. (i+1)/N`). The dispatcher owns
                    // `running-0` and the final `finished` so iOS sees one
                    // continuous progress bar across both kinds. iOS still
                    // labels this "Importing claude history" today — that's
                    // a future cleanup (rename the column or relabel the
                    // bar; out of scope for step 1).
                    SyncEventOutcome::ImportRequested => 'arm: {
                        let Some(mid) = machine_id else {
                            warn!("import requested but spawner is in dev mode (no machine id) — ignoring");
                            break 'arm;
                        };
                        // Snapshot user_id + parsed kinds under one read
                        // guard, then drop it — the spawned task below
                        // doesn't need the mirror.
                        let (uid, selected_kinds) = {
                            let g = mirror.read().await;
                            let Some(uid) = g.user_id else {
                                warn!("import requested but mirror.user_id not populated yet — ignoring");
                                break 'arm;
                            };
                            // Snapshot the user's checkbox selection at request
                            // time (not in the spawned task) — column changes
                            // mid-import don't retro-affect the in-flight run.
                            // Falls back to AgentKind::ALL when the column is
                            // absent / NULL (older iOS / older backend) so the
                            // historic both-kinds behavior is preserved.
                            (uid, g.parsed_import_kinds())
                        };
                        info!(
                            machine_id = %mid,
                            user_id = %uid,
                            ?selected_kinds,
                            "history import requested by user"
                        );
                        let tx_clone = write_tx.clone();
                        let handle = tokio::spawn(async move {
                            run_history_import(mid, uid, tx_clone, selected_kinds).await;
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
                // Same shape as the sync arm: write guard scoped tightly so
                // the control task can interleave a read between agent
                // responses but not within one. `handle_agent_response`
                // `.await`s on writer-channel sends and the supervisor
                // remove path, so this MUST be `tokio::sync::RwLock`.
                let mut g = mirror.write().await;
                handle_agent_response(resp, &mut g, &write_tx, &mut supervisor).await;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::SpawnFn;
    use crate::powersync::SyncEvent;
    use crate::writer::WriteEvent;
    use base64::{engine::general_purpose::STANDARD as B64, Engine as _};
    use serde_json::json;
    use std::sync::Mutex as StdMutex;

    /// Sync → spawn pipeline: a `projects` PUT + `chats` PUT + user `messages`
    /// PUT must converge into one `Supervisor::spawn_agent` call carrying the
    /// expected `SpawnRequest` (chat id, decrypted prompt, project path,
    /// claude adapter kind, owner = not sandboxed).
    #[tokio::test]
    async fn user_message_through_handle_sync_event_spawns_one_agent() {
        let user_id = Uuid::now_v7();
        let machine_id = Uuid::now_v7();
        let chat_id = Uuid::now_v7().to_string();
        let project_id = Uuid::now_v7().to_string();
        let project_path = "/tmp/zucchini-test-project".to_string();
        let msg_id = Uuid::now_v7().to_string();

        let mut mirror = Mirror::default();
        // Owner classification → is_sandboxed=false branch (skips the
        // machine_users membership gate).
        mirror.set_user_id(user_id);

        // KeyStore seeded with a deterministic key; envelope::encode signs
        // against this so the message decodes cleanly inside handle_message_put.
        let key_bytes = [0u8; 32];
        let keys = KeyStore::with_keys([(user_id, key_bytes)]);
        let key = keys.get(&user_id).expect("seeded key");

        // BlobDownloader is constructed but never reached — the test message
        // has zero attachments so `fetch_all` returns Ok(vec![]) without any
        // network IO. Token fetcher is a panicking stub for the same reason.
        let blobs = BlobDownloader::new(
            "http://test.invalid",
            Box::new(|| Box::pin(async { Err(anyhow::anyhow!("unused in test")) })),
        );

        let (write_tx, mut write_rx) = mpsc::channel::<WriteEvent>(64);
        let (resp_tx, _resp_rx) = mpsc::channel::<AgentResponse>(64);

        // Recorder spawn fn: captures the SpawnRequest and returns a dummy
        // JoinHandle (no shell, no Command::spawn).
        let recorded: Arc<StdMutex<Vec<SpawnRequest>>> = Arc::new(StdMutex::new(Vec::new()));
        let recorder = recorded.clone();
        let spawn_fn: SpawnFn = Arc::new(move |req, _tx, _token, _pending| {
            recorder.lock().unwrap().push(req);
            tokio::spawn(async {})
        });
        let mut supervisor = Supervisor::with_spawn_fn(resp_tx, spawn_fn);

        // Seeding-guard flag the test reuses across calls (the original
        // single-spawn flow doesn't trigger seeding because mirror.user_id
        // is set but no `machine_users` PUT is fed; flag stays `false`).
        let mut seed_attempted = false;

        // 1. Project PUT — populates mirror.projects so handle_message_put
        // can resolve project_path from project_id.
        let project_row =
            json!({ "id": project_id, "path": project_path, "name": "test-proj" }).to_string();
        handle_sync_event(
            SyncEvent::Put {
                table: "projects".into(),
                id: project_id.clone(),
                data: project_row,
            },
            Some(machine_id),
            &mut mirror,
            &mut supervisor,
            &blobs,
            &keys,
            None,
            None,
            &write_tx,
            None,
            &mut seed_attempted,
        )
        .await;

        // 2. Chat PUT — populates mirror.chats. last_seq=0 so seq=1 passes
        // the replay guard. agent_session_id=null → SpawnRequest carries None.
        let chat_row = json!({
            "id": chat_id,
            "project_id": project_id,
            "user_id": user_id.to_string(),
            "last_seq": 0,
            "agent_session_id": serde_json::Value::Null,
            "agent_kind": "claude",
            "worktree": false,
        })
        .to_string();
        handle_sync_event(
            SyncEvent::Put {
                table: "chats".into(),
                id: chat_id.clone(),
                data: chat_row,
            },
            Some(machine_id),
            &mut mirror,
            &mut supervisor,
            &blobs,
            &keys,
            None,
            None,
            &write_tx,
            None,
            &mut seed_attempted,
        )
        .await;

        // 3. Encrypt the prompt the way the iOS client would: envelope JSON
        // (text + empty attachments) → XChaCha20 AEAD with K_user → base64.
        let plaintext = "hello world from test";
        let envelope_json = serde_json::json!({ "text": plaintext, "attachments": [] }).to_string();
        let body_b64 = B64.encode(crypto::encrypt(&key, envelope_json.as_bytes()));

        let msg_row = json!({
            "id": msg_id,
            "chat_id": chat_id,
            "sender": "user",
            "seq": 1,
            "body": body_b64,
            "imported": false,
        })
        .to_string();
        handle_sync_event(
            SyncEvent::Put {
                table: "messages".into(),
                id: msg_id,
                data: msg_row,
            },
            Some(machine_id),
            &mut mirror,
            &mut supervisor,
            &blobs,
            &keys,
            None,
            None,
            &write_tx,
            None,
            &mut seed_attempted,
        )
        .await;

        // Assert: exactly one spawn captured with the expected SpawnRequest.
        let captured = recorded.lock().unwrap();
        assert_eq!(
            captured.len(),
            1,
            "expected one spawn, got {}",
            captured.len()
        );
        let req = &captured[0];
        assert_eq!(req.chat_id, chat_id);
        assert_eq!(req.agent_kind, AgentKind::Claude);
        assert!(!req.is_sandboxed, "owner spawn must not be sandboxed");
        assert_eq!(req.project_path.as_deref(), Some(project_path.as_str()));
        assert!(req.agent_session_id.is_none(), "fresh chat → no resume");
        assert!(!req.worktree);
        // No attachments → prompt is the raw envelope text (see blobs::build_prompt).
        assert_eq!(req.prompt, plaintext);

        // The idle→running transition must have emitted exactly one
        // ChatRunning(true) PATCH on the writer channel.
        let mut saw_running_true = false;
        while let Ok(ev) = write_rx.try_recv() {
            if let WriteEvent::ChatRunning {
                chat_id: cid,
                agent_running: true,
            } = ev
            {
                if cid == chat_id {
                    saw_running_true = true;
                }
            }
        }
        assert!(
            saw_running_true,
            "expected one ChatRunning(true) on the writer channel"
        );
    }

    /// `AgentResponse → WriteEvent` half of the agent pipeline: every variant
    /// of `handle_agent_response` must produce the right `WriteEvent`s on
    /// `write_tx`, and the `PutMessage` body must round-trip through
    /// `writer::encode_event` → `envelope::decode` back to its plaintext under
    /// the seeded `K_user`.
    #[tokio::test]
    async fn agent_responses_produce_correct_write_events_and_encrypt_body() {
        let user_id = Uuid::now_v7();
        let chat_id = Uuid::now_v7().to_string();
        let project_id = Uuid::now_v7().to_string();

        // Deterministic K_user — same value used to encrypt and decrypt below.
        let key_bytes = [7u8; 32];
        let keys = KeyStore::with_keys([(user_id, key_bytes)]);

        let mut mirror = Mirror::default();
        mirror.set_user_id(user_id);
        // Seed mirror.{projects,chats} so send_agent_line can resolve user_id.
        mirror.upsert_project(
            project_id.clone(),
            &json!({ "id": project_id, "path": "/tmp/zucchini-test", "name": "t" }).to_string(),
        );
        mirror.upsert_chat(
            chat_id.clone(),
            &json!({
                "id": chat_id,
                "project_id": project_id,
                "user_id": user_id.to_string(),
                "last_seq": 0,
                "agent_session_id": serde_json::Value::Null,
                "agent_kind": "claude",
                "worktree": false,
            })
            .to_string(),
        );

        // Supervisor is only needed because the `Done` arm calls
        // `supervisor.remove(&topic)`. The spawn closure is never invoked.
        let (resp_tx, _resp_rx) = mpsc::channel::<AgentResponse>(64);
        let spawn_fn: SpawnFn = Arc::new(|_req, _tx, _token, _pending| tokio::spawn(async {}));
        let mut supervisor = Supervisor::with_spawn_fn(resp_tx, spawn_fn);

        let (write_tx, mut write_rx) = mpsc::channel::<WriteEvent>(256);

        // Helper: drain everything currently queued on the writer channel.
        fn drain(rx: &mut mpsc::Receiver<WriteEvent>) -> Vec<WriteEvent> {
            let mut out = Vec::new();
            while let Ok(ev) = rx.try_recv() {
                out.push(ev);
            }
            out
        }

        // --- 1. Line → one PutMessage with sender=agent + plaintext intact.
        let line_plaintext = "hello from agent";
        handle_agent_response(
            AgentResponse::Line {
                topic: chat_id.clone(),
                content: line_plaintext.to_string(),
            },
            &mut mirror,
            &write_tx,
            &mut supervisor,
        )
        .await;
        let events = drain(&mut write_rx);
        assert_eq!(events.len(), 1, "Line → one WriteEvent");
        let put = match &events[0] {
            WriteEvent::PutMessage {
                id,
                chat_id: cid,
                user_id: uid,
                sender,
                content,
                created_at,
                imported,
            } => {
                assert!(id.is_none(), "live agent line carries no pre-minted id");
                assert_eq!(cid, &chat_id);
                assert_eq!(uid, &user_id);
                assert_eq!(*sender, "agent");
                assert_eq!(content, line_plaintext);
                assert!(created_at.is_none(), "live agent line: server stamps now()");
                assert!(!*imported, "live writes are never imported=true");
                events[0].clone()
            }
            other => panic!("expected PutMessage, got {:?}", other),
        };

        // --- 2. Round-trip the encryption through writer::encode_event.
        let op = writer::encode_event(&put, &keys).expect("encode_event returns BatchOp");
        assert_eq!(op.op, "PUT");
        assert_eq!(op.table, "messages");
        let data = op.data.as_ref().expect("PutMessage carries data");
        let body_b64 = data
            .get("body")
            .and_then(|v| v.as_str())
            .expect("data.body is a base64 string");
        let key = keys.get(&user_id).expect("seeded key resolves");
        // writer.rs::encode_event encrypts the raw `content` string (no
        // envelope wrap on the writer side — the iOS client wraps on incoming
        // user messages; agent-side outgoing messages are written raw).
        let cipher_bytes = B64.decode(body_b64).expect("body is base64");
        let plaintext_bytes =
            crypto::decrypt_bytes(&key, &cipher_bytes).expect("body decrypts under K_user");
        let plaintext =
            String::from_utf8(plaintext_bytes).expect("agent line is UTF-8 plaintext");
        assert_eq!(plaintext, line_plaintext);

        // --- 3. ContextTokens → ContextTokens passthrough.
        handle_agent_response(
            AgentResponse::ContextTokens {
                topic: chat_id.clone(),
                tokens: 12345,
            },
            &mut mirror,
            &write_tx,
            &mut supervisor,
        )
        .await;
        let events = drain(&mut write_rx);
        assert_eq!(events.len(), 1);
        match &events[0] {
            WriteEvent::ContextTokens { chat_id: cid, tokens } => {
                assert_eq!(cid, &chat_id);
                assert_eq!(*tokens, 12345);
            }
            other => panic!("expected ContextTokens, got {:?}", other),
        }

        // --- 4. SessionIdHarvested: local stash THEN the write event.
        handle_agent_response(
            AgentResponse::SessionIdHarvested {
                topic: chat_id.clone(),
                session_id: "sess-abc".into(),
            },
            &mut mirror,
            &write_tx,
            &mut supervisor,
        )
        .await;
        assert_eq!(
            mirror
                .chats
                .get(&chat_id)
                .and_then(|c| c.agent_session_id.clone())
                .as_deref(),
            Some("sess-abc"),
            "session id must be stashed locally before the writer round-trip"
        );
        let events = drain(&mut write_rx);
        assert_eq!(events.len(), 1);
        match &events[0] {
            WriteEvent::AgentSessionId { chat_id: cid, session_id } => {
                assert_eq!(cid, &chat_id);
                assert_eq!(session_id, "sess-abc");
            }
            other => panic!("expected AgentSessionId, got {:?}", other),
        }

        // --- 5. Done{has_result=false}: synthesize INTERRUPTED_RESULT line,
        // then flip agent_running=false. Supervisor slot is gone.
        // Pre-register a fake handle so we can observe `supervisor.remove`.
        supervisor.spawn_agent(SpawnRequest {
            chat_id: chat_id.clone(),
            prompt: String::new(),
            project_path: None,
            worktree: false,
            agent_session_id: None,
            agent_kind: AgentKind::Claude,
            is_sandboxed: false,
            model: None,
        });
        assert!(
            supervisor.is_running(&chat_id),
            "sanity: spawn_agent inserted the topic"
        );
        handle_agent_response(
            AgentResponse::Done {
                topic: chat_id.clone(),
                has_result: false,
            },
            &mut mirror,
            &write_tx,
            &mut supervisor,
        )
        .await;
        let events = drain(&mut write_rx);
        assert_eq!(events.len(), 2, "INTERRUPTED line + ChatRunning(false)");
        match &events[0] {
            WriteEvent::PutMessage { content, sender, chat_id: cid, .. } => {
                assert_eq!(*sender, "agent");
                assert_eq!(cid, &chat_id);
                assert_eq!(content, INTERRUPTED_RESULT);
            }
            other => panic!("expected synthesized INTERRUPTED PutMessage, got {:?}", other),
        }
        match &events[1] {
            WriteEvent::ChatRunning { chat_id: cid, agent_running: false } => {
                assert_eq!(cid, &chat_id);
            }
            other => panic!("expected ChatRunning(false), got {:?}", other),
        }
        assert!(
            !supervisor.is_running(&chat_id),
            "Done arm must remove the topic from supervisor"
        );

        // --- 6. Done{has_result=true}: just the ChatRunning(false), no synthesized line.
        handle_agent_response(
            AgentResponse::Done {
                topic: chat_id.clone(),
                has_result: true,
            },
            &mut mirror,
            &write_tx,
            &mut supervisor,
        )
        .await;
        let events = drain(&mut write_rx);
        assert_eq!(events.len(), 1);
        match &events[0] {
            WriteEvent::ChatRunning { chat_id: cid, agent_running: false } => {
                assert_eq!(cid, &chat_id);
            }
            other => panic!("expected ChatRunning(false), got {:?}", other),
        }
    }

    // ===== machine_users.agents seeding (migration 0035) =====
    //
    // These tests exercise the seeding decision in `handle_sync_event`'s
    // `machine_users` arm via `seed_default_agents_if_needed`. The contract:
    //   - NULL `agents` + ≥1 CLI installed + owner row → emit one PATCH
    //     carrying a valid JSON array with one entry per installed kind
    //     (claude first, then cursor; ids are uuid-v7 strings).
    //   - Non-NULL `agents` (including `"[]"`) → no PATCH (user-emptied is
    //     distinct from spawner-not-seeded-yet).
    //   - Re-emission of the same row within process lifetime → no PATCH
    //     (in-memory `agents_seed_attempted` flag).

    /// Helper: build the bare-bones args set `handle_sync_event` needs that
    /// the agents-seeding tests reuse — keeps each test focused on the
    /// scenario, not the boilerplate.
    fn empty_blob_downloader() -> BlobDownloader {
        BlobDownloader::new(
            "http://test.invalid",
            Box::new(|| Box::pin(async { Err(anyhow::anyhow!("unused in test")) })),
        )
    }

    #[tokio::test]
    async fn machine_users_null_agents_with_claude_installed_seeds_defaults() {
        let user_id = Uuid::now_v7();
        let machine_id = Uuid::now_v7();
        let row_id = Uuid::now_v7();

        let mut mirror = Mirror::default();
        mirror.set_user_id(user_id); // owner classification
        let keys = KeyStore::with_keys([(user_id, [0u8; 32])]);
        let blobs = empty_blob_downloader();
        let (write_tx, mut write_rx) = mpsc::channel::<WriteEvent>(64);
        let (resp_tx, _resp_rx) = mpsc::channel::<AgentResponse>(64);
        let spawn_fn: crate::agent::SpawnFn =
            Arc::new(|_req, _tx, _token, _pending| tokio::spawn(async {}));
        let mut supervisor = Supervisor::with_spawn_fn(resp_tx, spawn_fn);

        // Claude installed + authenticated; cursor not installed.
        let probe_statuses: Vec<(AgentKind, (bool, bool))> = vec![
            (AgentKind::Claude, (true, true)),
            (AgentKind::Cursor, (false, false)),
        ];
        let mut seed_attempted = false;

        // `machine_users` PUT with `agents = null` (the missing-key shape
        // is treated identically; we use explicit null here to mirror the
        // wire shape when iOS deliberately clears the column).
        let mu_row = json!({
            "id": row_id.to_string(),
            "user_id": user_id.to_string(),
            "machine_id": machine_id.to_string(),
            "is_sandboxed": 0,
            "sealed_blob": serde_json::Value::Null,
            "agents": serde_json::Value::Null,
        })
        .to_string();
        handle_sync_event(
            SyncEvent::Put {
                table: "machine_users".into(),
                id: row_id.to_string(),
                data: mu_row,
            },
            Some(machine_id),
            &mut mirror,
            &mut supervisor,
            &blobs,
            &keys,
            None,
            None,
            &write_tx,
            Some(&probe_statuses),
            &mut seed_attempted,
        )
        .await;

        // Exactly one SetMachineUserAgents PATCH on the writer channel.
        let mut patches: Vec<WriteEvent> = Vec::new();
        while let Ok(ev) = write_rx.try_recv() {
            if matches!(ev, WriteEvent::SetMachineUserAgents { .. }) {
                patches.push(ev);
            }
        }
        assert_eq!(patches.len(), 1, "expected one seed PATCH, got {}", patches.len());
        let WriteEvent::SetMachineUserAgents {
            row_id: emitted_row,
            machine_id: emitted_machine,
            agents_json,
        } = &patches[0]
        else {
            panic!("variant mismatch")
        };
        assert_eq!(*emitted_row, row_id);
        assert_eq!(*emitted_machine, machine_id);
        // Decode + shape-check: one claude entry, model+name empty, id parses as UUID.
        let parsed: serde_json::Value =
            serde_json::from_str(agents_json).expect("agents_json is JSON");
        let arr = parsed.as_array().expect("agents is array");
        assert_eq!(arr.len(), 1, "claude installed only → one entry");
        let entry = &arr[0];
        assert_eq!(entry["agent_kind"], "claude");
        assert_eq!(entry["model"], "");
        assert_eq!(entry["name"], "");
        let id_str = entry["id"].as_str().expect("id is string");
        Uuid::parse_str(id_str).expect("id is uuid");

        // Seeding flag flipped → second feed of the same row is a no-op.
        assert!(seed_attempted, "seed flag must flip after one PATCH");
    }

    #[tokio::test]
    async fn machine_users_null_agents_with_both_installed_seeds_two_entries_claude_first() {
        let user_id = Uuid::now_v7();
        let machine_id = Uuid::now_v7();
        let row_id = Uuid::now_v7();

        let mut mirror = Mirror::default();
        mirror.set_user_id(user_id);
        let keys = KeyStore::with_keys([(user_id, [0u8; 32])]);
        let blobs = empty_blob_downloader();
        let (write_tx, mut write_rx) = mpsc::channel::<WriteEvent>(64);
        let (resp_tx, _resp_rx) = mpsc::channel::<AgentResponse>(64);
        let spawn_fn: crate::agent::SpawnFn =
            Arc::new(|_req, _tx, _token, _pending| tokio::spawn(async {}));
        let mut supervisor = Supervisor::with_spawn_fn(resp_tx, spawn_fn);

        let probe_statuses: Vec<(AgentKind, (bool, bool))> = vec![
            (AgentKind::Claude, (true, true)),
            (AgentKind::Cursor, (true, true)),
        ];
        let mut seed_attempted = false;

        let mu_row = json!({
            "id": row_id.to_string(),
            "user_id": user_id.to_string(),
            "machine_id": machine_id.to_string(),
            "is_sandboxed": 0,
            "agents": serde_json::Value::Null,
        })
        .to_string();
        handle_sync_event(
            SyncEvent::Put {
                table: "machine_users".into(),
                id: row_id.to_string(),
                data: mu_row,
            },
            Some(machine_id),
            &mut mirror,
            &mut supervisor,
            &blobs,
            &keys,
            None,
            None,
            &write_tx,
            Some(&probe_statuses),
            &mut seed_attempted,
        )
        .await;

        let ev = write_rx
            .try_recv()
            .expect("expected one writer event for the seed PATCH");
        let WriteEvent::SetMachineUserAgents { agents_json, .. } = ev else {
            panic!("expected SetMachineUserAgents")
        };
        let arr: Vec<serde_json::Value> = serde_json::from_str(&agents_json).unwrap();
        assert_eq!(arr.len(), 2, "claude + cursor → two entries");
        // Deterministic order: claude first, then cursor. Matches the task
        // spec so iOS sees a stable agent-picker ordering across builds.
        assert_eq!(arr[0]["agent_kind"], "claude");
        assert_eq!(arr[1]["agent_kind"], "cursor");
    }

    #[tokio::test]
    async fn machine_users_non_null_agents_does_not_seed() {
        // `agents = "[]"` is the explicit "user emptied the list" state —
        // the spawner MUST NOT re-seed over it. Same for any non-empty array.
        let user_id = Uuid::now_v7();
        let machine_id = Uuid::now_v7();
        let row_id = Uuid::now_v7();

        let mut mirror = Mirror::default();
        mirror.set_user_id(user_id);
        let keys = KeyStore::with_keys([(user_id, [0u8; 32])]);
        let blobs = empty_blob_downloader();
        let (write_tx, mut write_rx) = mpsc::channel::<WriteEvent>(64);
        let (resp_tx, _resp_rx) = mpsc::channel::<AgentResponse>(64);
        let spawn_fn: crate::agent::SpawnFn =
            Arc::new(|_req, _tx, _token, _pending| tokio::spawn(async {}));
        let mut supervisor = Supervisor::with_spawn_fn(resp_tx, spawn_fn);

        let probe_statuses: Vec<(AgentKind, (bool, bool))> = vec![
            (AgentKind::Claude, (true, true)),
            (AgentKind::Cursor, (true, true)),
        ];
        let mut seed_attempted = false;

        let mu_row = json!({
            "id": row_id.to_string(),
            "user_id": user_id.to_string(),
            "machine_id": machine_id.to_string(),
            "is_sandboxed": 0,
            "agents": "[]",
        })
        .to_string();
        handle_sync_event(
            SyncEvent::Put {
                table: "machine_users".into(),
                id: row_id.to_string(),
                data: mu_row,
            },
            Some(machine_id),
            &mut mirror,
            &mut supervisor,
            &blobs,
            &keys,
            None,
            None,
            &write_tx,
            Some(&probe_statuses),
            &mut seed_attempted,
        )
        .await;

        // No SetMachineUserAgents PATCH on the channel.
        let mut saw_seed_patch = false;
        while let Ok(ev) = write_rx.try_recv() {
            if matches!(ev, WriteEvent::SetMachineUserAgents { .. }) {
                saw_seed_patch = true;
            }
        }
        assert!(
            !saw_seed_patch,
            "non-NULL agents (including empty array) must not trigger seeding"
        );
        assert!(!seed_attempted, "non-NULL path must not flip the seed flag");
    }

    /// `chats.model` flows from the row → ChatState → SpawnRequest.model.
    /// Empty / NULL both collapse to `None`; non-empty is preserved verbatim
    /// (the adapter is responsible for `--model <X>` shell-escaping).
    #[tokio::test]
    async fn chat_model_threads_into_spawn_request() {
        let user_id = Uuid::now_v7();
        let machine_id = Uuid::now_v7();
        let project_id = Uuid::now_v7().to_string();
        let project_path = "/tmp/zucchini-test-project".to_string();

        // Two chats: one with `model="opus"`, one with `model=""` (empty
        // sentinel → None). Each gets its own user message + spawn capture
        // so we can compare the resulting SpawnRequest.model fields side by side.
        let chat_with = Uuid::now_v7().to_string();
        let chat_empty = Uuid::now_v7().to_string();

        let mut mirror = Mirror::default();
        mirror.set_user_id(user_id);
        let keys = KeyStore::with_keys([(user_id, [0u8; 32])]);
        let key = keys.get(&user_id).expect("seeded key");
        let blobs = empty_blob_downloader();
        let (write_tx, _write_rx) = mpsc::channel::<WriteEvent>(64);
        let (resp_tx, _resp_rx) = mpsc::channel::<AgentResponse>(64);

        let recorded: Arc<StdMutex<Vec<SpawnRequest>>> = Arc::new(StdMutex::new(Vec::new()));
        let recorder = recorded.clone();
        let spawn_fn: crate::agent::SpawnFn = Arc::new(move |req, _tx, _token, _pending| {
            recorder.lock().unwrap().push(req);
            tokio::spawn(async {})
        });
        let mut supervisor = Supervisor::with_spawn_fn(resp_tx, spawn_fn);

        let mut seed_attempted = true; // skip seeding path in this test

        // Project PUT.
        handle_sync_event(
            SyncEvent::Put {
                table: "projects".into(),
                id: project_id.clone(),
                data: json!({ "id": project_id, "path": project_path, "name": "t" }).to_string(),
            },
            Some(machine_id),
            &mut mirror,
            &mut supervisor,
            &blobs,
            &keys,
            None,
            None,
            &write_tx,
            None,
            &mut seed_attempted,
        )
        .await;

        // Chat A: model="opus".
        handle_sync_event(
            SyncEvent::Put {
                table: "chats".into(),
                id: chat_with.clone(),
                data: json!({
                    "id": chat_with,
                    "project_id": project_id,
                    "user_id": user_id.to_string(),
                    "last_seq": 0,
                    "agent_session_id": serde_json::Value::Null,
                    "agent_kind": "claude",
                    "worktree": false,
                    "model": "opus",
                })
                .to_string(),
            },
            Some(machine_id),
            &mut mirror,
            &mut supervisor,
            &blobs,
            &keys,
            None,
            None,
            &write_tx,
            None,
            &mut seed_attempted,
        )
        .await;

        // Chat B: model="" (empty sentinel → None).
        handle_sync_event(
            SyncEvent::Put {
                table: "chats".into(),
                id: chat_empty.clone(),
                data: json!({
                    "id": chat_empty,
                    "project_id": project_id,
                    "user_id": user_id.to_string(),
                    "last_seq": 0,
                    "agent_session_id": serde_json::Value::Null,
                    "agent_kind": "claude",
                    "worktree": false,
                    "model": "",
                })
                .to_string(),
            },
            Some(machine_id),
            &mut mirror,
            &mut supervisor,
            &blobs,
            &keys,
            None,
            None,
            &write_tx,
            None,
            &mut seed_attempted,
        )
        .await;

        // Feed a user message to each chat.
        for (cid, label) in [(chat_with.clone(), "with"), (chat_empty.clone(), "empty")] {
            let envelope_json =
                serde_json::json!({ "text": format!("hi {label}"), "attachments": [] })
                    .to_string();
            let body_b64 = B64.encode(crypto::encrypt(&key, envelope_json.as_bytes()));
            let msg_id = Uuid::now_v7().to_string();
            handle_sync_event(
                SyncEvent::Put {
                    table: "messages".into(),
                    id: msg_id.clone(),
                    data: json!({
                        "id": msg_id,
                        "chat_id": cid,
                        "sender": "user",
                        "seq": 1,
                        "body": body_b64,
                        "imported": false,
                    })
                    .to_string(),
                },
                Some(machine_id),
                &mut mirror,
                &mut supervisor,
                &blobs,
                &keys,
                None,
                None,
                &write_tx,
                None,
                &mut seed_attempted,
            )
            .await;
        }

        let captured = recorded.lock().unwrap();
        assert_eq!(captured.len(), 2, "two chats → two spawns");
        let with_spawn = captured
            .iter()
            .find(|r| r.chat_id == chat_with)
            .expect("chat A spawn captured");
        let empty_spawn = captured
            .iter()
            .find(|r| r.chat_id == chat_empty)
            .expect("chat B spawn captured");
        assert_eq!(
            with_spawn.model.as_deref(),
            Some("opus"),
            "non-empty model passes through verbatim"
        );
        assert!(
            empty_spawn.model.is_none(),
            "empty model collapses to None at the SpawnRequest construction site"
        );
    }
}
