mod agent;
mod atomic;
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
mod x25519;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use agent::{AgentResponse, Supervisor};
use auth::AuthClient;
use blobs::BlobDownloader;
use crypto::KeyStore;
use crypto_box::SecretKey;
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
                let pubkey_changed = mirror.set_spawner_pubkey(
                    row.get("spawner_pubkey").and_then(|p| p.as_str()),
                );
                // If a server-side clear/rotation flips the column at runtime
                // (e.g. ops nukes spawner_pubkey to force a re-publish), upload
                // immediately rather than waiting for the next boot.
                if pubkey_changed {
                    if let Some(mid) = machine_id {
                        publish_spawner_pubkey_if_needed(mid, our_pubkey_b64, mirror, write_tx).await;
                    }
                }
                if json_pg_bool(row.get("to_uninstall")) {
                    return SyncEventOutcome::UninstallRequested;
                }
                let status = row
                    .get("claude_history_import_status")
                    .and_then(|f| f.as_str());
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
                if apply_machine_users_put(&id, user_id, &row, x25519_secret, keys, mirror) {
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
            .send(WriteEvent::agent_line(chat_id.clone(), user_id, INTERRUPTED_RESULT.to_string()))
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

    let is_resume = seq > 1;
    // Only PATCH when transitioning idle→running. The abort-then-respawn path
    // (was_running==true) leaves agent_running already true from the prior
    // spawn — re-sending it would fan out a no-op write to every listening
    // client and re-trigger their chat-list re-decrypt.
    if !was_running {
        let _ = write_tx
            .send(WriteEvent::chat_running(chat_id.clone(), true))
            .await;
    }
    supervisor.spawn_agent(
        chat_id.clone(),
        prompt,
        Some(project_path),
        worktree,
        is_resume,
        is_sandboxed,
    );
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
        let Some(uid_str) = s.strip_prefix("key_") else { continue };
        if uid_str.contains('.') {
            continue;
        }
        let Some(uid) = parse_uuid_str(uid_str) else { continue };
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
    v.as_i64().or_else(|| v.as_str().and_then(|s| s.parse().ok()))
}

pub(crate) fn parse_uuid_str(s: &str) -> Option<Uuid> {
    Uuid::parse_str(s).ok()
}

pub(crate) fn parse_uuid_field(row: &serde_json::Value, field: &str) -> Option<Uuid> {
    row.get(field).and_then(|v| v.as_str()).and_then(parse_uuid_str)
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
        .send(WriteEvent::SetSpawnerPubkey { machine_id, pubkey_b64: our_pubkey.to_string() })
        .await;
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

    // Seed `mirror.user_id` from the env var written by install.sh
    // BEFORE the sync loop starts, so the owner-check (`is_owner = ... == mirror.user_id`)
    // works regardless of bucket op ordering on first boot. Without this, a by_machine
    // checkpoint that delivers our own `machine_users` row before the `machines` PUT
    // would misclassify the owner as a non-owner and delete `key_<owner>`; a `messages`
    // PUT in that same window would be dropped because the membership gate sees no entry.
    // `set_user_id` is no-op when `user_id` is already populated (e.g. from state.json),
    // so older hosts without the env var fall back to lazy harvest unchanged.
    if mirror.user_id.is_none() {
        if let Some(env_uid) = env_uuid("ZUCCHINI_USER_ID") {
            if mirror.set_user_id(env_uid) {
                info!(user_id = %env_uid, "seeded mirror.user_id from ZUCCHINI_USER_ID");
                save_mirror(&state_path, &mirror);
            }
        }
    }

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
        WriterConfig { base_url: writer_base_url, fetch_token: writer_token },
        keys.clone(),
    );
    let write_tx = writer.tx.clone();

    let heartbeat_cancel = CancellationToken::new();
    if let Some(p) = &prod {
        info!(machine_id = %p.machine_id, "starting heartbeat task");
        spawn_heartbeat(p.machine_id, write_tx.clone(), heartbeat_cancel.clone());
        spawn_startup_info_report(p.machine_id, write_tx.clone());

        publish_spawner_pubkey_if_needed(
            p.machine_id,
            our_pubkey_b64.as_deref(),
            &mirror,
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
                    x25519_secret.as_ref(),
                    our_pubkey_b64.as_deref(),
                    &write_tx,
                ).await;
                match outcome {
                    SyncEventOutcome::StateChanged => save_mirror(&state_path, &mirror),
                    SyncEventOutcome::CheckpointReached => {
                        save_mirror(&state_path, &mirror);
                        // Gate reconcile on `mirror.user_id.is_some()` so
                        // the FIRST CheckpointComplete from a by_user-only
                        // checkpoint (dev mode without by_machine, or first
                        // boot before the by_machine bucket has streamed)
                        // doesn't sweep every dev-placed `key_<uuid>` file.
                        // The latch flips only once the by_machine round-trip
                        // has populated mirror.user_id, after which subsequent
                        // CheckpointComplete events reflect full membership.
                        if !key_files_reconciled && mirror.user_id.is_some() {
                            key_files_reconciled = true;
                            reconcile_key_files(&mirror, &keys);
                        }
                    }
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
                    AgentResponse::CompactBoundary { topic, post_tokens } => {
                        let _ = write_tx
                            .send(WriteEvent::CompactBoundary { chat_id: topic, post_tokens })
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
