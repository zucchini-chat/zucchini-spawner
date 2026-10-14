mod adapter;
mod adapters;
mod agent;
mod atomic;
mod auth;
mod blobs;
mod control;
mod crypto;
mod envelope;
mod hermes_support;
mod net;
mod orphans;
mod power;
mod powersync;
mod prune;
mod session_name;
mod shell;
mod state;
mod sys;
mod uninstall;
mod updater;
mod user_prompt;
mod useragent;
mod warm;
mod watch;
mod writer;
mod x25519;

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use adapter::AgentKind;
use agent::{AgentResponse, QueuedTurn, SpawnRequest, Supervisor};
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
/// into the sync-event handler so `seed_initial_agents_if_pending` can decide
/// which compatibility-seeded agents are installed without re-shelling. `OnceLock` lets
/// the producer set it exactly once and lets readers snapshot without locking.
type ProbeStatusesCache = Arc<std::sync::OnceLock<Vec<(AgentKind, (bool, bool))>>>;

const SENTRY_DSN: &str = "https://05d5deab2efce04e4f801af41ea39def@o4511216603234304.ingest.de.sentry.io/4511216616669264";
use adapter::INTERRUPTED_RESULT;
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

/// The user's home directory, or `None` when it can't be determined. Wraps
/// `dirs::home_dir()` (reads `$HOME` when set non-empty, else the OS default) so
/// every spawner-side `$HOME` read is cross-platform and goes through one place.
/// Use this where an absent home is a meaningful skip signal.
pub(crate) fn home_dir_opt() -> Option<PathBuf> {
    dirs::home_dir()
}

/// The user's home directory, falling back to `"."` when it can't be determined.
/// Use this where a path must always be produced (the historical fallback).
pub(crate) fn home_dir() -> PathBuf {
    home_dir_opt().unwrap_or_else(|| PathBuf::from("."))
}

pub(crate) fn zucchini_spawner_dir() -> PathBuf {
    home_dir().join(".zucchini-spawner")
}

/// Force 0o700 on the spawner dir so `key_<uuid>` filenames don't leak to
/// local cohorts on hosts where umask is the default 022. We can't pass the
/// mode to `create_dir_all` portably, so chmod after the fact — best-effort,
/// failure is logged but not fatal (matches the existing
/// "failed to ensure spawner dir exists" pattern).
pub(crate) fn ensure_spawner_dir_private(dir: &Path) {
    if let Err(e) = sys::harden_dir(dir) {
        warn!(error = %e, path = %dir.display(), "failed to harden spawner dir permissions");
    }
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

/// Per-checkpoint-window staging for hazard tables. PowerSync emits
/// PUT → REMOVE → PUT on every row update (slot move, not a delete); staging
/// the last op per id and applying at `CheckpointComplete` collapses that to
/// the net result, so a transient REMOVE never evicts a chat mid-turn (which
/// would drop the turn's `result` frame via `send_agent_line`). Owned by the
/// main loop, NEVER part of `Mirror` (not serialized, not visible to the
/// control task). Apply order at checkpoint: projects → chats → messages.
#[derive(Default)]
struct SyncStaging {
    chats: std::collections::HashMap<String, StagedRow>, // id -> last op this window
    projects: std::collections::HashMap<String, StagedRow>, // id -> last op this window
    messages: Vec<String>, // deferred user-message raw row JSON, arrival order
    /// The window CLEARed the bucket: chats it doesn't re-PUT were deleted.
    cleared: bool,
}

/// A staged chats/projects op for one id within a window (last-write-wins).
enum StagedRow {
    Put(String), // raw row JSON
    Remove,
}

#[allow(clippy::too_many_arguments)]
async fn handle_sync_event(
    event: SyncEvent,
    machine_id: Option<Uuid>,
    mirror: &mut Mirror,
    // Per-checkpoint-window staging for hazard tables (chats/projects/messages).
    // Stage arms record the last op per id (chats/projects) or push deferred
    // user-message rows; the `CheckpointComplete` arm applies them in order
    // (projects → chats → messages) against the now-consistent mirror. Owned by
    // the main loop, never part of `Mirror` (not serialized, control task never
    // sees it).
    staging: &mut SyncStaging,
    supervisor: &mut Supervisor,
    blobs: &BlobDownloader,
    keys: &KeyStore,
    x25519_secret: Option<&SecretKey>,
    our_pubkey_b64: Option<&str>,
    write_tx: &mpsc::Sender<WriteEvent>,
    // Probe statuses fan-out cache (`spawn_startup_info_report` fills it
    // once at boot). `None` here means probes haven't completed yet — the
    // initial agents seed defers (see `seed_initial_agents_if_pending`).
    probe_statuses: Option<&[(AgentKind, (bool, bool))]>,
) -> SyncEventOutcome {
    // `&mut Mirror` here is borrowed from the same `Arc<tokio::sync::RwLock<Mirror>>`
    // the control task reads (see `control::ControlState::mirror`); callers in
    // `main()` acquire the write guard once per sync event before invoking this
    // helper. Tests construct `Mirror::default()` directly and never involve
    // the lock.
    match event {
        SyncEvent::Put { table, id, data } => match table.as_str() {
            "projects" => {
                // Stage; applied (and persisted via CheckpointReached) at the
                // checkpoint. No per-op StateChanged anymore.
                staging.projects.insert(id, StagedRow::Put(data));
                SyncEventOutcome::Nothing
            }
            "chats" => {
                // Stage last-write-wins; a transient PUT→REMOVE→PUT slot move
                // collapses to the net result at the checkpoint, so a stale
                // REMOVE never evicts a chat mid-turn.
                staging.chats.insert(id, StagedRow::Put(data));
                SyncEventOutcome::Nothing
            }
            "messages" => {
                // Defer the spawn to the checkpoint so a new chat + its first
                // message arriving in the same window apply in dependency order
                // (chat present before the message spawns). Cheap pre-filter
                // (mirrors the early-outs in `handle_message_put`) keeps staging
                // free of agent/imported noise — `handle_message_put` still does
                // the authoritative filtering/decode/replay-guard at apply time.
                let stage = match serde_json::from_str::<serde_json::Value>(&data) {
                    Ok(row) => {
                        let is_user = row.get("sender").and_then(|v| v.as_str()) == Some("user");
                        let imported = json_pg_bool(row.get("imported"));
                        is_user && !imported
                    }
                    // Unparseable here → stage it; `handle_message_put` logs +
                    // bails on the same parse failure at apply.
                    Err(_) => true,
                };
                if stage {
                    staging.messages.push(data);
                }
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
                let sandboxed_before = mirror.member_is_sandboxed(&user_id);
                let state_changed =
                    apply_machine_users_put(&id, user_id, &row, x25519_secret, keys, mirror);
                if !mirror.is_owner(user_id) {
                    let revoked = row
                        .get("sealed_blob")
                        .and_then(|v| v.as_str())
                        .is_none_or(str::is_empty);
                    let sandbox_flipped = sandboxed_before
                        .is_some_and(|b| mirror.member_is_sandboxed(&user_id) != Some(b));
                    if revoked || sandbox_flipped {
                        abort_member_agents(user_id, revoked, mirror, supervisor, write_tx).await;
                    }
                }

                // `agents` (migration 0035) is deliberately NOT mirrored: the
                // spawner never needs the roster's server state (see
                // `seed_initial_agents_if_pending`).

                // `timezone` (migration 0040) — IANA id of the member's most-
                // recently-active device (validated server-side in `/api/devices`).
                // Mirror raw (NULL → None) for the spawn-site prompt clock; see
                // `MemberInfo.timezone`.
                let timezone_raw = row
                    .get("timezone")
                    .and_then(|v| v.as_str())
                    .map(str::to_string);
                mirror.set_member_timezone(&user_id, timezone_raw);

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
                // PowerSync emits PUT → REMOVE → PUT on every row update (moves
                // the storage slot, not a real delete). Stage last-write-wins:
                // a window whose final op is REMOVE applies remove_chat at the
                // checkpoint (real delete); a transient REMOVE followed by a
                // re-PUT collapses to the PUT, so a chat is never evicted
                // mid-turn (which would drop the turn's `result` frame).
                staging.chats.insert(id, StagedRow::Remove);
                SyncEventOutcome::Nothing
            }
            "projects" => {
                staging.projects.insert(id, StagedRow::Remove);
                SyncEventOutcome::Nothing
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
                abort_member_agents(user_id, true, mirror, supervisor, write_tx).await;
                SyncEventOutcome::StateChanged
            }
            _ => SyncEventOutcome::Nothing,
        },
        SyncEvent::Clear => {
            // Everything staged before it is superseded: the bucket re-sends
            // every live row after the CLEAR.
            *staging = SyncStaging {
                cleared: true,
                ..Default::default()
            };
            SyncEventOutcome::Nothing
        }
        SyncEvent::CheckpointComplete { buckets } => {
            // Apply the staged window in dependency order: projects → chats →
            // messages, so chat.project_id and message.chat resolve against a
            // now-consistent mirror. Then advance the persisted cursor.

            // 1. projects
            for (id, row) in staging.projects.drain() {
                match row {
                    StagedRow::Put(json) => mirror.upsert_project(id, &json),
                    StagedRow::Remove => {
                        // Its chats go with it (ON DELETE CASCADE).
                        for chat_id in mirror.chat_ids(|c| c.project_id == id) {
                            delete_chat(&chat_id, mirror, supervisor).await;
                        }
                        mirror.remove_project(&id);
                    }
                }
            }

            // 2. chats (with agent_session_id restore — replaces the old inline
            //    merge-on-upsert hack in state.rs::upsert_chat)
            if std::mem::take(&mut staging.cleared) {
                for id in mirror.chat_ids(|_| true) {
                    staging.chats.entry(id).or_insert(StagedRow::Remove);
                }
            }
            let mut thresholds_set = Vec::new();
            let mut chats_put = Vec::new();
            for (id, row) in staging.chats.drain() {
                match row {
                    StagedRow::Put(json) => {
                        let old_threshold =
                            mirror.chats.get(&id).and_then(|c| c.auto_compact_tokens);
                        let old_prompt =
                            mirror.chats.get(&id).map(|c| c.user_system_prompt.clone());
                        // Preserve a locally-harvested session id across a
                        // checkpoint that re-delivers the row with
                        // agent_session_id=NULL (the harvest is set locally
                        // between checkpoints via set_agent_session_id; PowerSync
                        // can re-stream a stale NULL). A window containing a real
                        // REMOVE collapses to remove_chat first (last-write-wins),
                        // so nothing is restored for a genuine delete.
                        let local_session = mirror
                            .chats
                            .get(&id)
                            .and_then(|c| c.agent_session_id.clone());
                        mirror.upsert_chat(id.clone(), &json, keys);
                        if let Some(local) = local_session {
                            if mirror
                                .chats
                                .get(&id)
                                .map(|c| c.agent_session_id.is_none())
                                .unwrap_or(false)
                            {
                                mirror.set_agent_session_id(&id, local);
                            }
                        }
                        // Chat renamed in the app: carry the new name to the
                        // live resident so peers keep addressing it by title.
                        // A cold chat needs nothing - its next spawn passes
                        // `--name` from this same title.
                        if let Some(chat) = mirror.chats.get(&id) {
                            let name = chat.title.as_deref().and_then(session_name::from_title);
                            supervisor.rename_resident(&id, name.as_deref());
                            if chat.auto_compact_tokens.is_some()
                                && chat.auto_compact_tokens != old_threshold
                            {
                                thresholds_set.push(id.clone());
                            }
                            // A new prompt gets the full retry budget: the
                            // old one may be what kept the resident from
                            // starting.
                            if old_prompt.is_some_and(|old| old != chat.user_system_prompt) {
                                mirror.always_on_starts.remove(&id);
                            }
                            chats_put.push(id.clone());
                        }
                    }
                    StagedRow::Remove => delete_chat(&id, mirror, supervisor).await,
                }
            }

            // Boot: clear the run flags no agent backs anymore, before any
            // message below spawns one. Stopped-by-user: no "finished" push.
            if mirror.first_checkpoint() {
                // A watch's command runs unsandboxed: one whose chat is gone or
                // lost that access while the spawner was down must not start.
                // Before the reset, which keeps a chat with an admitted watch
                // busy and lists it in `live_tasks`.
                watch::admit_restored(|w| mirror.can_watch(&w.chat_id, keys));
                for chat_id in mirror.flagged_chat_ids() {
                    info!(chat_id = %chat_id, "resetting stale agent_running/agent_alive/live_tasks");
                    send_agent_idle(write_tx, chat_id.clone(), true).await;
                    let _ = write_tx
                        .send(WriteEvent::chat_alive(chat_id.clone(), false))
                        .await;
                    clear_live_tasks(write_tx, &chat_id).await;
                }
            }

            // 3. messages — spawn against the now-consistent mirror, arrival order
            let pending_messages = std::mem::take(&mut staging.messages);
            let window = window_user_seqs(&pending_messages);
            for raw in pending_messages {
                handle_message_put(&raw, &window, mirror, supervisor, blobs, keys, write_tx).await;
            }

            // After the messages: a user turn they queued comes before any compaction.
            for chat_id in thresholds_set {
                let Some(chat) = mirror.chats.get(&chat_id) else {
                    continue;
                };
                let timezone = mirror.member_timezone(&chat.user_id);
                if supervisor.after_threshold_set(&chat_id, chat.auto_compact_tokens, timezone) {
                    publish_queued_turn(supervisor, write_tx, mirror, chat_id).await;
                }
            }

            // After the messages: one that came with a new system prompt
            // already respawned its chat.
            for chat_id in chats_put {
                retire_stale_prompt(&chat_id, mirror, supervisor, write_tx).await;
            }

            // After the messages: a client switches always-on off no later than
            // its `/stop`, and a chat a message spawned already has its agent.
            reconcile_always_on(mirror, supervisor, write_tx).await;

            // One-shot agents seed — driven here because the owner row only
            // arrives via sync; the set flag persists with the cursor at the
            // CheckpointReached save.
            seed_initial_agents_if_pending(machine_id, mirror, probe_statuses, write_tx).await;

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
    if is_deleted_chat(mirror, resp.topic()) {
        if let AgentResponse::Done { topic, .. } = &resp {
            supervisor.remove_finished(topic);
        }
        return;
    }
    match resp {
        AgentResponse::Line { topic, content } => {
            send_agent_line(write_tx, mirror, &topic, content).await;
        }
        AgentResponse::ContextTokens { topic, tokens } => {
            let _ = write_tx
                .send(WriteEvent::ContextTokens {
                    chat_id: topic,
                    tokens,
                })
                .await;
        }
        AgentResponse::CompactBoundary { topic, post_tokens } => {
            let _ = write_tx
                .send(WriteEvent::CompactBoundary {
                    chat_id: topic,
                    post_tokens,
                })
                .await;
        }
        AgentResponse::SessionIdHarvested { topic, session_id } => {
            // Stash locally first so a fast-followup user message
            // in the same chat doesn't race the writer / PowerSync
            // round-trip and spawn a second claude without --resume.
            mirror.set_agent_session_id(&topic, session_id.clone());
            let _ = write_tx
                .send(WriteEvent::AgentSessionId {
                    chat_id: topic,
                    session_id,
                })
                .await;
        }
        AgentResponse::Done { topic, has_result } => {
            info!(topic = %topic, has_result, "agent done");
            if !has_result {
                send_agent_line(write_tx, mirror, &topic, INTERRUPTED_RESULT.to_string()).await;
            }
            // Idle clears agent_running (→ false). Required for the resident path
            // (a session that exits while still busy — turn in flight or a task
            // armed — must clear the indicator); for one-shot it's the normal
            // busy→idle flip. A late `Done` from a torn-down resident leaves its
            // replacement and its indicator alone.
            let had_agent = supervisor.has_agent(&topic);
            supervisor.remove_finished(&topic);
            if had_agent && !supervisor.has_agent(&topic) {
                // This `Done` is the one that ended the tracked process, and
                // anything it had armed died with it. A late `Done` from a
                // session a respawn already replaced leaves its successor's task
                // list alone.
                clear_live_tasks(write_tx, &topic).await;
            }
            if !supervisor.is_running(&topic) {
                send_run_state(write_tx, topic.clone(), false).await;
            }
        }
        // Resident path: the reader emits this on every busy↔idle transition.
        // Drives the boolean `agent_running` (true the whole time the agent is
        // busy, including while a background task / monitor is still armed —
        // the Thinking-vs-Waiting sub-state is re-derived on the client, not on
        // the wire). No optimistic flips — these come straight from the reader's
        // frame-derived FSM (CLAUDE.md "No optimistic patches").
        AgentResponse::RunState { topic, running } => {
            send_run_state(write_tx, topic, running).await;
        }
        // The main loop's `response_rx` arm intercepts `ToolResult` (it drives the
        // prune restart against `pending_prunes`, which this fn doesn't own), so it
        // never reaches here at runtime. The arm exists only to keep the match
        // exhaustive (and stays a no-op if a future caller routes one through).
        AgentResponse::ToolResult { .. } => {}
        // What the resident has armed changed. From a session a teardown
        // already replaced it is stale: that teardown cleared the column, and
        // re-writing its task list would show phantom tasks on the live one.
        AgentResponse::LiveTasks {
            topic,
            session,
            tasks,
        } => {
            if !supervisor.is_current_session(&topic, &session) {
                return;
            }
            // Enqueue writes directly to the writer while reader snapshots
            // wait here; an older one must not overwrite the newer count.
            if agent::live_tasks_payload(&session.lock().expect("SessionState mutex")) != tasks {
                return;
            }
            send_live_tasks(write_tx, mirror, topic, tasks).await;
        }
        // From a session a teardown replaced it is stale, like `LiveTasks`.
        AgentResponse::TurnRunning {
            topic,
            session,
            running,
        } => {
            if !supervisor.is_current_session(&topic, &session) {
                return;
            }
            let now = {
                let mut st = session.lock().expect("SessionState mutex");
                let now = st.in_foreground();
                if !running {
                    st.published_turn_running = now;
                }
                now
            };
            // The backend releases queued messages on the false edge, so every
            // turn end reaches it, even one the next turn has overtaken: false,
            // then true again. A true edge whose turn is over writes nothing.
            if !running {
                send_turn_running(write_tx, topic.clone(), false).await;
            }
            if now {
                send_turn_running(write_tx, topic, true).await;
            }
        }
        AgentResponse::TurnEnded { topic } => {
            if let Some(chat) = mirror.chats.get(&topic) {
                supervisor.after_turn(
                    &topic,
                    chat.auto_compact_tokens,
                    mirror.is_always_on(&topic),
                    mirror.member_timezone(&chat.user_id),
                );
            }
            // Left with background tasks armed, the chat sends no idle edge.
            if mirror.is_always_on(&topic) && supervisor.is_running(&topic) {
                retire_stale_prompt(&topic, mirror, supervisor, write_tx).await;
            }
        }
    }
}

/// A chat whose agent outlived its row (its responses were queued before the
/// delete aborted it). Any write against it 4xx's the whole batch server-side,
/// and the writer retries a 4xx forever.
fn is_deleted_chat(mirror: &Mirror, chat_id: &str) -> bool {
    !mirror.chats.contains_key(chat_id)
}

/// Delete `chat_id` from the mirror and kill its agent, with no writes: the
/// row is gone server-side (see `is_deleted_chat`).
async fn delete_chat(chat_id: &str, mirror: &mut Mirror, supervisor: &mut Supervisor) {
    mirror.remove_chat(chat_id);
    watch::chat_deleted(chat_id);
    supervisor.abort_agent(chat_id).await;
    supervisor.forget_alive(chat_id);
}

/// Kill a member's agents after their access changed, so a running turn can't
/// finish under the old key or sandbox mode. A revoked member gets no writes
/// (their key is gone and their chats may be cascaded away); after a sandbox
/// flip, a running or queued turn is closed out as interrupted.
async fn abort_member_agents(
    user_id: Uuid,
    revoked: bool,
    mirror: &Mirror,
    supervisor: &mut Supervisor,
    write_tx: &mpsc::Sender<WriteEvent>,
) {
    watch::member_restricted(user_id);
    for chat_id in mirror.chat_ids(|c| c.user_id == user_id) {
        let running = supervisor.is_running(&chat_id);
        let torn_down = supervisor.has_agent(&chat_id);
        supervisor.abort_agent(&chat_id).await;
        if revoked {
            supervisor.forget_alive(&chat_id);
            continue;
        }
        if running {
            info!(chat_id = %chat_id, %user_id, "sandbox flipped: aborted running agent");
            publish_interrupted(write_tx, &chat_id, user_id).await;
        }
        // Also for a WARM IDLE resident: its reader's own clear is dropped as a
        // replaced session's, so the teardown has to write the NULL itself.
        if torn_down {
            clear_live_tasks(write_tx, &chat_id).await;
        }
        if running {
            send_run_state(write_tx, chat_id, false).await;
        }
    }
}

/// Publish the canned `INTERRUPTED_RESULT` system frame for an aborted /
/// interrupted turn. Shared by the resident interrupt arms (/stop +
/// interrupt-then-send) and the one-shot abort arm so the frame is emitted
/// identically from all three.
async fn publish_interrupted(write_tx: &mpsc::Sender<WriteEvent>, chat_id: &str, user_id: Uuid) {
    let _ = write_tx
        .send(WriteEvent::agent_line(
            chat_id.to_string(),
            user_id,
            INTERRUPTED_RESULT.to_string(),
        ))
        .await;
}

/// Emit the resident run state as a single boolean `agent_running` patch.
/// `running` stays true the whole time the agent is busy — including while a
/// background task / monitor is still armed (the old "Waiting…" window). The
/// Thinking-vs-Waiting sub-state is re-derived on the client, so the spawner
/// does NOT distinguish them on the wire. One write per busy↔idle transition.
/// A turn was queued on a live resident: the queued count is part of the
/// pill's task list, and nothing else publishes it until the CLI echoes the
/// turn.
async fn publish_queued_turn(
    supervisor: &Supervisor,
    write_tx: &mpsc::Sender<WriteEvent>,
    mirror: &Mirror,
    chat_id: String,
) {
    if let Some(tasks) = supervisor.live_tasks_change(&chat_id) {
        send_live_tasks(write_tx, mirror, chat_id.clone(), tasks).await;
    }
    send_run_state(write_tx, chat_id.clone(), true).await;
    if let Some(running) = supervisor.turn_running_change(&chat_id) {
        send_turn_running(write_tx, chat_id, running).await;
    }
}

/// Write `chats.turn_running`. Only ever sent after the `agent_running` write
/// of the same edge: the backend resets it whenever `agent_running` goes false.
async fn send_turn_running(write_tx: &mpsc::Sender<WriteEvent>, chat_id: String, running: bool) {
    let _ = write_tx
        .send(WriteEvent::ChatTurnRunning {
            chat_id,
            turn_running: running,
        })
        .await;
}

async fn send_run_state(write_tx: &mpsc::Sender<WriteEvent>, chat_id: String, running: bool) {
    if running {
        let _ = write_tx.send(WriteEvent::chat_running(chat_id, true)).await;
    } else {
        send_agent_idle(write_tx, chat_id, false).await;
    }
}

/// The chat's agent is idle or gone: `agent_running=false`. Every such write
/// comes through here, because a running watch keeps the chat busy like an
/// armed background task: then only the turn's end is written, which the
/// backend otherwise takes from `agent_running` going false.
async fn send_agent_idle(
    write_tx: &mpsc::Sender<WriteEvent>,
    chat_id: String,
    stopped_by_user: bool,
) {
    if !watch::running_in(&chat_id).is_empty() {
        send_turn_running(write_tx, chat_id, false).await;
        return;
    }
    let idle = if stopped_by_user {
        WriteEvent::chat_stopped_by_user(chat_id)
    } else {
        WriteEvent::chat_running(chat_id, false)
    };
    let _ = write_tx.send(idle).await;
}

/// Clear the agent's part of `chats.live_tasks`. Armed tasks live inside the
/// agent process, so every teardown has to drop them — a `--resume` does not
/// re-arm them and PowerSync won't re-stream the row, so a stale blob would
/// show phantom tasks forever. Called from each teardown that keeps the row (a
/// deleted chat and a revoked member get no writes at all).
async fn clear_live_tasks(write_tx: &mpsc::Sender<WriteEvent>, chat_id: &str) {
    write_live_tasks(write_tx, chat_id.to_string(), None, None).await;
}

/// Write `chats.live_tasks` for the session blob `tasks` (`None`: nothing
/// armed or queued).
async fn send_live_tasks(
    write_tx: &mpsc::Sender<WriteEvent>,
    mirror: &Mirror,
    chat_id: String,
    tasks: Option<String>,
) {
    let user_id = mirror.chats.get(&chat_id).map(|c| c.user_id);
    write_live_tasks(write_tx, chat_id, user_id, tasks).await;
}

/// The one writer of `chats.live_tasks`: the session's blob with the chat's
/// watches ahead of it (plaintext; the writer encrypts it under the chat
/// owner's key), NULL when both are empty.
async fn write_live_tasks(
    write_tx: &mpsc::Sender<WriteEvent>,
    chat_id: String,
    user_id: Option<Uuid>,
    session_tasks: Option<String>,
) {
    let watches = watch::running_in(&chat_id);
    // A clear names no user; a watch carries its chat's.
    let user_id = user_id.or(watches.first().map(|w| w.user_id));
    let tasks = agent::live_tasks_with_watches(session_tasks, &watches);
    let _ = write_tx
        .send(WriteEvent::ChatLiveTasks {
            chat_id,
            user_id,
            tasks,
        })
        .await;
}

/// A chat's watches changed: rewrite its `chats.live_tasks` from its resident's
/// session, if it has one, and the watches it has now, and `agent_running`
/// unless its agent holds it true anyway.
async fn publish_watch_change(
    chat_id: &str,
    mirror: &Mirror,
    keys: &KeyStore,
    supervisor: &Supervisor,
    write_tx: &mpsc::Sender<WriteEvent>,
) {
    let Some(chat) = mirror.chats.get(chat_id) else {
        return;
    };
    // Before the first checkpoint the row may be gone server-side; a revoked
    // member gets no writes.
    if !mirror.booted() || keys.get_existing(&chat.user_id).is_none() {
        return;
    }
    let tasks = supervisor.live_tasks(chat_id);
    send_live_tasks(write_tx, mirror, chat_id.to_string(), tasks).await;
    if !supervisor.is_running(chat_id) {
        let watched = !watch::running_in(chat_id).is_empty();
        send_run_state(write_tx, chat_id.to_string(), watched).await;
    }
}

/// Write `agent_alive` for every chat whose resident process appeared or went
/// away since the last call. Once per main-loop pass (see `alive_changes`).
async fn publish_alive_changes(supervisor: &mut Supervisor, write_tx: &mpsc::Sender<WriteEvent>) {
    for (chat_id, alive) in supervisor.alive_changes() {
        let _ = write_tx.send(WriteEvent::chat_alive(chat_id, alive)).await;
    }
}

/// How often idle residents' memory is re-measured for eviction.
const WARM_TICK: Duration = Duration::from_secs(60);

/// Evict idle claude residents per `warm::choose` until none is due (one per
/// call under memory pressure); returns when the next one would be due. Their
/// turns already ended, so no `interrupted`
/// frame; `agent_alive` goes false via `publish_alive_changes`.
async fn evict_idle_residents(
    supervisor: &mut Supervisor,
    memory: fn() -> Option<warm::Memory>,
    write_tx: &mpsc::Sender<WriteEvent>,
    pinned: &HashSet<String>,
    always_on: &HashSet<String>,
) -> Option<tokio::time::Instant> {
    loop {
        let mut idle = supervisor.idle_residents();
        if idle.len() < 2 {
            return None;
        }
        measure_idle_footprint(&mut idle).await;
        for r in idle.iter_mut() {
            r.pinned = pinned.contains(&r.topic);
            r.always_on = always_on.contains(&r.topic);
        }
        let eviction = match warm::choose(&idle, memory().as_ref(), tokio::time::Instant::now()) {
            warm::Decision::Evict(e) => e,
            warm::Decision::WaitUntil(deadline) => return Some(deadline),
            warm::Decision::KeepAll => return None,
        };
        let topic = &idle[eviction.victim].topic;
        info!(
            topic = %topic,
            n = idle.len(),
            n_eff = eviction.n_eff,
            ttl = ?eviction.ttl,
            age = ?eviction.age,
            pressure = eviction.pressure,
            pinned = idle[eviction.victim].pinned,
            "evicting idle resident"
        );
        let topic = topic.clone();
        if supervisor.evict_idle(&topic).await {
            // An evictable resident is idle, so its list is already empty and
            // this is normally a no-op write — but it is the only thing standing
            // between a missed edge and a chat pinned to phantom tasks.
            clear_live_tasks(write_tx, &topic).await;
        }
        if eviction.pressure {
            // Memory stats lag the kill; the next tick re-checks.
            return None;
        }
    }
}

async fn measure_idle_footprint(idle: &mut [warm::IdleResident]) {
    let Some(pids) = idle.iter().map(|r| r.pid).collect::<Option<Vec<u32>>>() else {
        return;
    };
    let Some(footprints) = sys::process_tree_footprint_kb(&pids).await else {
        return;
    };
    for (r, kb) in idle.iter_mut().zip(footprints) {
        r.footprint_kb = Some(kb);
    }
}

/// Which residents are idle since when; a change re-runs eviction.
fn idle_set(supervisor: &Supervisor) -> Vec<(String, tokio::time::Instant)> {
    supervisor
        .idle_residents()
        .into_iter()
        .map(|r| (r.topic, r.idle_since))
        .collect()
}

/// The main loop's `response_rx` arm.
async fn route_agent_response(
    resp: AgentResponse,
    pending_prunes: &prune::PendingPrunes,
    mirror: &SharedMirror,
    supervisor: &mut Supervisor,
    write_tx: &mpsc::Sender<WriteEvent>,
) {
    match resp {
        // This cue is the call-keyed signal that the `prune-context`
        // call's OWN result has persisted to the transcript (the adapter
        // emits it only for that call, never a sibling tool in the same
        // parallel batch). Aborting → rewriting → respawning here lands
        // the prune strictly AFTER that result (and its summary) reached
        // disk — the resumed agent sees its own prune and won't re-run it.
        // A no-op for any chat with nothing pending. The control task
        // parked the `PruneRequest` in `pending_prunes` during the
        // `prune-context` RPC, which returned before this cue could fire,
        // so the entry is guaranteed visible here. Take the lock only long
        // enough to lift the chat's batch out, then apply unlocked.
        AgentResponse::ToolResult { topic } => {
            // The `prune-context` call's own result has persisted to the
            // transcript. Apply NOW regardless of resident-busy state:
            // abort (SIGTERM) the agent → rewrite the jsonl → respawn-with-
            // `--resume`, so the freed context takes effect immediately,
            // within the SAME task — not deferred to the turn boundary
            // (which made the prune useless to the turn that asked for it).
            // This is a HARD RESTART of the resident session, and it has to
            // be: a transcript rewrite only lands via respawn (a live in-
            // memory session ignores the on-disk edit), so pruning a
            // resident agent is necessarily kill+resume. The cue fires only
            // AFTER the prune call's result reached disk, so the resumed
            // transcript carries the agent's prune call + summary and it
            // won't re-issue the now-satisfied command. A no-op for any chat
            // with nothing pending or no agent (`take_prunes`). The control task parked the
            // `PruneRequest` in `pending_prunes` during the `prune-context`
            // RPC (which returned before this cue could fire), so the entry
            // is guaranteed visible here.
            //
            // This restart kills any background task / Monitor armed in
            // the resident session — their runtime is in-process, not in
            // the transcript, so `--resume` does not re-arm them. The
            // control task GUARDS against this: `prune_context` refuses a
            // prune while `live_tasks` is non-empty unless the agent
            // passed `--force`, so reaching here with live tasks means a
            // forced prune (or a task armed in the same instant the RPC's
            // check passed). Either way `apply_prune_group` reads the
            // live-task count first and tells the resumed agent in the
            // respawn prompt how many it terminated, so the kill is never
            // silent. (Earlier this DEFERRED until Idle to spare the
            // monitor; the immediate apply — so the freed context helps the
            // turn that asked for it — is the behavior we kept.)
            //
            // An idle chat's cue is stale: applying it would restart a chat no
            // turn is running in, so it is dropped.
            if !supervisor.is_running(&topic) {
                pending_prunes.lock().await.remove(&topic);
            } else if let Some(reqs) = take_prunes(&topic, pending_prunes, supervisor).await {
                apply_prune_group(&topic, reqs, mirror, supervisor, write_tx).await;
            }
        }
        AgentResponse::RunState { topic, running } => {
            handle_run_state(
                &topic,
                running,
                pending_prunes,
                mirror,
                supervisor,
                write_tx,
            )
            .await;
        }
        other => {
            // A turn ending with a prune still parked means its triggering
            // cue never arrived (the agent exited/errored mid-`prune-context`,
            // so the call's own result was never persisted). Drop the parked
            // request: leaving it in `pending_prunes` leaks the entry for the
            // process lifetime and would mis-fire on a LATER turn's
            // `prune-context` cue for this chat (applying a stale rewrite).
            if let AgentResponse::Done { topic, .. } = &other {
                pending_prunes.lock().await.remove(topic);
            }
            // Write guard scoped tightly so the control task can interleave
            // a read between agent responses but not within one.
            // `handle_agent_response` `.await`s on writer-channel sends and
            // the supervisor remove path, so this MUST be `tokio::sync::RwLock`.
            let mut g = mirror.write().await;
            handle_agent_response(other, &mut g, write_tx, supervisor).await;
        }
    }
}

/// The chat's parked prunes, if its agent is still tracked. A `prune-context`
/// RPC can park after `/stop` or shutdown cleared the entry; applying it would
/// respawn the torn-down chat, so it is dropped instead.
async fn take_prunes(
    topic: &str,
    pending_prunes: &prune::PendingPrunes,
    supervisor: &Supervisor,
) -> Option<Vec<prune::PruneRequest>> {
    let reqs = pending_prunes.lock().await.remove(topic);
    reqs.filter(|_| supervisor.has_agent(topic))
}

/// Tear every agent down while processing what their readers send meanwhile:
/// a resident reduces a turn's `result` (so `any_running` goes false) before it
/// forwards that line's frames and idle edge, and a reader blocked on a full
/// `response_rx` only exits once it is consumed. With no agent tracked,
/// `take_prunes` keeps anything routed here from respawning one.
async fn shutdown_agents(
    supervisor: &mut Supervisor,
    response_rx: &mut mpsc::Receiver<AgentResponse>,
    mirror: &SharedMirror,
    write_tx: &mpsc::Sender<WriteEvent>,
) {
    let pending_prunes = supervisor.pending_prunes();
    let mut readers = tokio::task::JoinSet::new();
    let mut torn_down = Vec::new();
    for (topic, handle) in supervisor.cancel_all() {
        torn_down.push(topic.clone());
        readers.spawn(async move { agent::await_agent_exit(handle, &topic).await });
    }
    loop {
        tokio::select! {
            Some(resp) = response_rx.recv() => {
                route_agent_response(resp, &pending_prunes, mirror, supervisor, write_tx).await;
            }
            joined = readers.join_next() => {
                if joined.is_none() {
                    break;
                }
            }
        }
    }
    pending_prunes.lock().await.clear();
    while let Ok(resp) = response_rx.try_recv() {
        route_agent_response(resp, &pending_prunes, mirror, supervisor, write_tx).await;
    }
    publish_alive_changes(supervisor, write_tx).await;
    // These sessions were CANCELLED, not finished: `cancel_all` drained the
    // supervisor, so a `Done` routed above no longer recognizes the chat's agent
    // and its clear never fires (and a reader that misses `AGENT_EXIT_TIMEOUT`
    // sends no `Done` at all). Last write of the drain, so a late edge can't
    // re-arm the column behind it.
    let mirror = mirror.read().await;
    for topic in torn_down {
        if !is_deleted_chat(&mirror, &topic) {
            clear_live_tasks(write_tx, &topic).await;
        }
    }
}

/// Stop heartbeats, tear the agents down and flush the writer, whose queue
/// lives only in memory.
async fn shutdown_and_flush(
    supervisor: &mut Supervisor,
    response_rx: &mut mpsc::Receiver<AgentResponse>,
    mirror: &SharedMirror,
    write_tx: &mpsc::Sender<WriteEvent>,
    heartbeat_cancel: &CancellationToken,
    writer: &Writer,
) {
    heartbeat_cancel.cancel();
    shutdown_agents(supervisor, response_rx, mirror, write_tx).await;
    if !wait_for_writer_idle(writer).await {
        warn!("writer drain timed out, unsent writes are lost");
    }
}

/// `/stop` (and `/kill`, its alias for older clients): kill the chat's agent
/// process group, background tasks included. The transcript stays, so the
/// next message resumes it. The chat's watches go too; the agent learns of
/// that from a note on its next turn (`start_turn`).
async fn stop_agent(
    chat_id: &str,
    user_id: Uuid,
    supervisor: &mut Supervisor,
    write_tx: &mpsc::Sender<WriteEvent>,
) {
    let running = supervisor.is_running(chat_id);
    if supervisor.is_working(chat_id) {
        publish_interrupted(write_tx, chat_id, user_id).await;
    }
    let watched = watch::chat_stopped(chat_id);
    supervisor.abort_agent(chat_id).await;
    if running || watched {
        send_agent_idle(write_tx, chat_id.to_string(), true).await;
    }
    clear_live_tasks(write_tx, chat_id).await;
    // Written even with no process: the row may be stale.
    supervisor.forget_alive(chat_id);
    let _ = write_tx
        .send(WriteEvent::chat_alive(chat_id.to_string(), false))
        .await;
}

/// Per chat, the `seq`s of the user messages staged in one checkpoint window.
type WindowUserSeqs = std::collections::HashMap<String, std::collections::HashSet<i64>>;

fn window_user_seqs(raw_rows: &[String]) -> WindowUserSeqs {
    let mut seqs = WindowUserSeqs::new();
    for raw in raw_rows {
        let Ok(row) = serde_json::from_str::<serde_json::Value>(raw) else {
            continue;
        };
        let chat_id = row.get("chat_id").and_then(|v| v.as_str());
        let seq = row.get("seq").and_then(json_to_i64);
        if let (Some(chat_id), Some(seq)) = (chat_id, seq) {
            seqs.entry(chat_id.to_string()).or_default().insert(seq);
        }
    }
    seqs
}

async fn handle_message_put(
    data: &str,
    window: &WindowUserSeqs,
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

    let (
        project_id,
        worktree,
        chat_last_seq,
        user_id,
        agent_session_id,
        agent_kind,
        model,
        session_name,
    ) = match mirror.chats.get(&chat_id) {
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
            c.model
                .as_deref()
                .filter(|s| !s.is_empty())
                .map(str::to_string),
            c.title.as_deref().and_then(session_name::from_title),
        ),
        None => {
            warn!(chat_id = %chat_id, "message arrived before chat row, skipping");
            return;
        }
    };

    // Replay guard: PowerSync re-streams the bucket from op_id 0 (sync-rules
    // rename, fresh state.json), and without it every replayed user message
    // respawns its long-finished agent. A message at or below the chat's
    // handled high-water mark is a replay. Without a mark, fall back to
    // chats.last_seq, the bucket-authoritative latest message: a message below
    // it is a replay — unless (resident chat) only user messages of this same
    // window follow it (seqs are contiguous per chat). The mark exists because
    // last_seq also counts agent rows, which can share a window with a
    // mid-turn user message.
    let is_resident = agent_kind.make_adapter().is_resident();
    let only_window_messages_follow = || {
        window
            .get(&chat_id)
            .is_some_and(|seqs| (seq + 1..=chat_last_seq).all(|s| seqs.contains(&s)))
    };
    let replayed = match mirror.handled_user_seq(&chat_id) {
        Some(handled) => seq <= handled,
        None => seq < chat_last_seq && !(is_resident && only_window_messages_follow()),
    };
    // One-shot adapters answer only the latest of a window's messages.
    let superseded = !is_resident
        && window
            .get(&chat_id)
            .is_some_and(|seqs| seqs.iter().any(|&s| s > seq));
    if replayed || superseded {
        info!(chat_id = %chat_id, seq, chat_last_seq, "skipping replayed user message");
        return;
    }
    mirror.set_handled_user_seq(&chat_id, seq);

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

    // Detect /stop and /kill early but act after the membership gate, so an
    // invitee with an unsynced machine_users row can never trigger a stray
    // abort.
    let command = |name: &str| envelope.attachments.is_empty() && envelope.text.trim() == name;
    let is_stop = command("/stop");
    let is_kill = command("/kill");
    if is_stop || is_kill {
        info!(chat_id = %chat_id, text = %envelope.text.trim(), "stop command received");
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
    // already-running chat. Also BEFORE the optimistic `agent_running=true`
    // flip: a `return` here after sending it would strand the UI on a
    // perpetual spinner.
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

    if is_stop || is_kill {
        stop_agent(&chat_id, user_id, supervisor, write_tx).await;
        return;
    }
    // One-shot adapters (gemini/cursor/hermes/pi) spawn per turn: a new message
    // aborts the running one. Resident adapters (claude, codex) take it on
    // stdin (below).
    let was_running = supervisor.is_running(&chat_id);
    if !is_resident && was_running {
        info!(chat_id = %chat_id, "aborting running one-shot agent before handling new message");
        supervisor.abort_agent(&chat_id).await;
        publish_interrupted(write_tx, &chat_id, user_id).await;
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

    // Resume keys off `agent_session_id`, not `seq>1`: a freshly created chat
    // may already have a backfilled `agent_session_id` (pre-migration rows) and
    // a brand-new chat where the first turn aborted before harvest still has
    // `agent_session_id = None`, so we want a fresh session there too.
    // `agent_kind` picks the adapter at spawn time.
    // Sender's IANA timezone (migration 0040). `None` → adapter omits the time
    // line. Looked up here so the volatile clock is fresh per turn.
    let user_timezone = mirror.member_timezone(&user_id).map(str::to_string);

    let req = SpawnRequest {
        chat_id: chat_id.clone(),
        prompt,
        project_path: Some(project_path),
        worktree,
        agent_session_id,
        agent_kind,
        is_sandboxed,
        model,
        user_timezone,
        session_name,
        user_system_prompt: mirror.user_system_prompt(&chat_id),
    };

    // A message the backend promoted from the schedule can arrive mid-turn at
    // any time: its slash command waits for the turn instead of cutting it.
    let command_may_interrupt = !json_pg_bool(row.get("auto_sent"));
    start_turn(
        req,
        command_may_interrupt,
        was_running,
        user_id,
        mirror,
        supervisor,
        write_tx,
    )
    .await;
}

/// Give `req`'s prompt to its chat's agent as a turn: on a live resident's
/// stdin, else in a fresh process. `was_running` is the chat's run state
/// before a one-shot agent was aborted for this turn.
async fn start_turn(
    mut req: SpawnRequest,
    command_may_interrupt: bool,
    was_running: bool,
    user_id: Uuid,
    mirror: &Mirror,
    supervisor: &mut Supervisor,
    write_tx: &mpsc::Sender<WriteEvent>,
) {
    let chat_id = req.chat_id.clone();
    let agent_kind = req.agent_kind;
    let adapter = agent_kind.make_adapter();
    // A command turn ships verbatim, so the note waits for the next one.
    if !adapter.is_command_turn(&req.prompt) {
        if let Some(note) = watch::take_stop_note(&chat_id) {
            req.prompt = format!("{note}\n\n{}", req.prompt);
        }
    }
    if adapter.is_resident() {
        // A live process spawned with the same knobs takes the turn on stdin;
        // claude answers it within the running turn or right after it.
        let queued = if command_may_interrupt {
            supervisor.send_user_turn(&req)
        } else {
            supervisor.send_scheduled_turn(&req)
        };
        if queued {
            info!(chat_id = %chat_id, "queued user turn on the live resident session");
            publish_queued_turn(supervisor, write_tx, mirror, chat_id).await;
            return;
        }
        // Otherwise (no live process, other knobs, or a slash command mid-turn)
        // spawn a fresh `--resume` process, re-queuing turns the old one never
        // consumed ahead of this one. Hard-aborting a running turn also kills
        // its armed monitors. NO optimistic agent_running flip for the spawn —
        // the reader's RunState drives it.
        let interrupting = supervisor.is_working(&chat_id);
        if interrupting {
            info!(chat_id = %chat_id, "interrupt-then-send: aborting in-flight resident turn, will respawn fresh");
            publish_interrupted(write_tx, &chat_id, user_id).await;
        }
        let respawning = supervisor.has_agent(&chat_id);
        let aborted = supervisor.abort_agent(&chat_id).await;
        if interrupting {
            send_run_state(write_tx, chat_id.clone(), false).await;
        }
        if respawning {
            // The respawn re-arms nothing (`--resume` doesn't restore in-process
            // tasks); the fresh session republishes from empty.
            clear_live_tasks(write_tx, &chat_id).await;
        }
        let mut turns = turns_to_requeue(aborted, agent_kind, req.agent_session_id.as_deref());
        turns.push(req.prompt.clone());
        let first = turns.remove(0);
        supervisor.spawn_agent_then(
            SpawnRequest {
                prompt: first,
                ..req
            },
            turns,
        );
    } else {
        // One-shot: optimistic-free idle→running flip preserved. The
        // abort-then-respawn path already had agent_running=true, so only PATCH
        // when transitioning idle→running. One-shot adapters never arm
        // background tasks, so they're simply busy until the turn ends.
        if !was_running {
            let _ = write_tx
                .send(WriteEvent::chat_running(chat_id.clone(), true))
                .await;
        }
        // A one-shot's turn is its whole run. Written on every spawn: the
        // chat may have been a resident's, left with tasks armed and no turn.
        send_turn_running(write_tx, chat_id.clone(), true).await;
        // A fresh turn: anything parked is a torn-down session's (the resident
        // path's `abort_agent` drops it too).
        supervisor.pending_prunes().lock().await.remove(&chat_id);
        supervisor.spawn_agent(req);
    }
}

/// A watch's message (`watch.rs`): a system row in its chat, then a turn with
/// the same text. An error leaves it in the watch's outbox to be offered again.
async fn deliver_watch_message(
    out: watch::Outgoing,
    mirror: &Mirror,
    keys: &KeyStore,
    supervisor: &mut Supervisor,
    write_tx: &mpsc::Sender<WriteEvent>,
) -> anyhow::Result<()> {
    let chat_id = &out.watch.chat_id;
    if out.cancelled.is_cancelled() {
        anyhow::bail!("the watch was stopped");
    }
    // Before the first checkpoint the chat's run flags are still to be reset.
    if !mirror.booted() {
        anyhow::bail!("not synced yet");
    }
    // Also a message queued before its watch was cancelled by a revoke.
    if !mirror.can_watch(chat_id, keys) {
        anyhow::bail!("chat {chat_id} is not synced, is sandboxed or has no key");
    }
    let prompt = format!("{}\n{}", out.watch.label(), out.text);
    let req = chat_spawn_request(mirror, chat_id, prompt)
        .ok_or_else(|| anyhow::anyhow!("chat {chat_id} can't spawn yet"))?;
    // A one-shot agent would be cut by a new turn; the message waits for it.
    let was_running = supervisor.is_running(chat_id);
    if was_running && !req.agent_kind.make_adapter().is_resident() {
        anyhow::bail!("the chat's agent is busy");
    }
    let user_id = out.watch.user_id;
    let row = agent::peer_message_frame_json(&out.watch.sender_name(), &out.text);
    let _ = write_tx
        .send(WriteEvent::agent_line(chat_id.clone(), user_id, row))
        .await;
    start_turn(
        req,
        false,
        was_running,
        user_id,
        mirror,
        supervisor,
        write_tx,
    )
    .await;
    Ok(())
}

/// `chat_id`'s spawn with `prompt`, resolved from the `Mirror` with
/// `handle_message_put`'s gates (project-path resolution, owner-vs-member
/// sandbox). `None` (with a logged reason) if the chat/project isn't synced or
/// a non-owner's `machine_users` row hasn't landed.
fn chat_spawn_request(mirror: &Mirror, chat_id: &str, prompt: String) -> Option<SpawnRequest> {
    let chat = mirror.chats.get(chat_id)?;
    let project_path = match mirror.projects.get(&chat.project_id) {
        Some(p) => p.clone(),
        None => {
            warn!(chat_id = %chat_id, project_id = %chat.project_id, "project not synced; cannot resolve spawn params");
            return None;
        }
    };
    // Same owner-vs-member sandbox gate as handle_message_put.
    let is_sandboxed = if mirror.is_owner(chat.user_id) {
        false
    } else {
        match mirror.member_is_sandboxed(&chat.user_id) {
            Some(s) => s,
            None => {
                warn!(chat_id = %chat_id, user_id = %chat.user_id, "machine_users row not synced; cannot resolve spawn params");
                return None;
            }
        }
    };
    Some(SpawnRequest {
        chat_id: chat_id.to_string(),
        prompt,
        project_path: Some(project_path),
        worktree: chat.worktree,
        agent_session_id: chat.agent_session_id.clone(),
        agent_kind: chat.agent_kind,
        is_sandboxed,
        model: chat
            .model
            .as_deref()
            .filter(|s| !s.is_empty())
            .map(str::to_string),
        user_timezone: mirror.member_timezone(&chat.user_id).map(str::to_string),
        session_name: chat.title.as_deref().and_then(session_name::from_title),
        user_system_prompt: mirror.user_system_prompt(chat_id),
    })
}

/// Tear down `chat_id`'s resident if it runs with another user system prompt
/// than the chat has now and no turn would be cut. Its next spawn - the next
/// message, or right here for an always-on chat - renders the current one. An
/// always-on resident goes even with background tasks armed: its restart
/// notice has the agent re-arm them. Elsewhere they would die unnoticed, so
/// that resident stays until it is idle or a message respawns it. A prompt that doesn't
/// render retires nothing: the resident keeps working until a message's
/// respawn reports the error.
async fn retire_stale_prompt(
    chat_id: &str,
    mirror: &mut Mirror,
    supervisor: &mut Supervisor,
    write_tx: &mpsc::Sender<WriteEvent>,
) {
    let prompt = mirror.user_system_prompt(chat_id);
    let always_on = mirror.is_always_on(chat_id);
    if !supervisor.has_retirable_stale_prompt(chat_id, &prompt, always_on) {
        return;
    }
    let Some(project_path) = mirror
        .chats
        .get(chat_id)
        .and_then(|c| mirror.projects.get(&c.project_id))
    else {
        return;
    };
    if let Err(e) = prompt.render(Some(project_path)) {
        warn!(chat_id = %chat_id, error = %e, "the new system prompt doesn't render, keeping the resident");
        return;
    }
    let Some(was_running) = supervisor
        .retire_stale_prompt(chat_id, &prompt, always_on)
        .await
    else {
        return;
    };
    info!(chat_id = %chat_id, "system prompt changed, resident torn down");
    if was_running {
        send_agent_idle(write_tx, chat_id.to_string(), true).await;
    }
    clear_live_tasks(write_tx, chat_id).await;
    if always_on {
        reconcile_always_on(mirror, supervisor, write_tx).await;
    }
}

/// The warm tick's always-on pass: an always-on resident whose stale prompt
/// had to wait for its turn to end is retired now, then whatever is
/// missing starts.
async fn tend_always_on(
    mirror: &mut Mirror,
    supervisor: &mut Supervisor,
    write_tx: &mpsc::Sender<WriteEvent>,
) {
    for chat_id in mirror.always_on_chats() {
        retire_stale_prompt(&chat_id, mirror, supervisor, write_tx).await;
    }
    reconcile_always_on(mirror, supervisor, write_tx).await;
}

/// The first turn of an always-on resident the spawner (re)starts.
const RESTART_NOTICE: &str = "Your process was (re)started with this session resumed \
(spawner restart, machine reboot, always-on switched on, or a crash). Background tasks, monitors, \
dev servers, watchers and scheduled loops you started may have stopped. Check what is still \
running before restarting anything that should be; if nothing should be, reply in one line.";

/// The notice's system row: clients show "Message from Zucchini: …".
const RESTART_NOTICE_SENDER: &str = "Zucchini";

fn restart_notice_turn() -> String {
    format!("[{RESTART_NOTICE_SENDER}] {RESTART_NOTICE}")
}

/// How many starts of an always-on resident `reconcile_always_on` makes
/// before a pass sees it through its notice turn.
const ALWAYS_ON_RETRIES: u32 = 3;

/// Start the resident of every always-on chat that has a session to resume,
/// a resident adapter and no agent, with `RESTART_NOTICE` as its turn and as
/// a system row in the chat. A chat whose starts no pass saw through the
/// notice turn `ALWAYS_ON_RETRIES` times is left alone.
async fn reconcile_always_on(
    mirror: &mut Mirror,
    supervisor: &mut Supervisor,
    write_tx: &mpsc::Sender<WriteEvent>,
) {
    if !mirror.booted() {
        return;
    }
    let always_on = mirror.always_on_chats();
    // A resident that is over its notice turn has started; one still in it,
    // or closed and dying, keeps its count.
    mirror.always_on_starts.retain(|id, _| {
        always_on.contains(id)
            && (!supervisor.has_open_resident(id) || supervisor.in_foreground(id))
    });
    for chat_id in always_on {
        let chat = &mirror.chats[&chat_id];
        let startable = chat.agent_session_id.is_some()
            && chat.agent_kind.make_adapter().is_resident()
            && !supervisor.has_agent(&chat_id)
            && mirror.always_on_starts.get(&chat_id).copied().unwrap_or(0) < ALWAYS_ON_RETRIES;
        if !startable {
            continue;
        }
        let user_id = chat.user_id;
        *mirror.always_on_starts.entry(chat_id.clone()).or_default() += 1;
        let Some(req) = chat_spawn_request(mirror, &chat_id, restart_notice_turn()) else {
            continue;
        };
        info!(chat_id = %chat_id, "starting always-on resident");
        let row = agent::peer_message_frame_json(RESTART_NOTICE_SENDER, RESTART_NOTICE);
        let _ = write_tx
            .send(WriteEvent::agent_line(chat_id.clone(), user_id, row))
            .await;
        supervisor.spawn_agent(req);
    }
}

/// Abort `chat_id`'s agent (if any) and apply its coalesced prune requests, then
/// respawn ONCE. Called from the `response_rx` arm on the `ToolResult` cue — i.e.
/// once claude has emitted (and thus persisted) the `prune-context` call's own
/// result frame. The abort runs FIRST; because it fires only after that result
/// landed, the resumed transcript carries the agent's prune call + summary, and
/// the respawn re-reads the now-rewritten jsonl so the freed context takes effect.
/// Rewrites run in arrival order, each re-reading the transcript (the last-only /
/// already-pruned-folds-to-0 contract is `crate::prune`'s). Stats are summed into a
/// single timeline frame + respawn prompt. No INTERRUPTED frame (transparent
/// restart, not a user stop); `agent_running` stays true throughout (no
/// ChatRunning write) so the UI never flickers idle.
async fn apply_prune_group(
    chat_id: &str,
    reqs: Vec<prune::PruneRequest>,
    mirror: &SharedMirror,
    supervisor: &mut Supervisor,
    write_tx: &mpsc::Sender<WriteEvent>,
) {
    // Snapshot how many background tasks / monitors this restart will terminate
    // BEFORE the abort drops the session (and clears its live-task set). Normally
    // 0 — the control task blocks a prune while tasks are live unless `--force`,
    // so a non-zero count here means the agent forced it (or a task armed in the
    // same instant the RPC's check passed). Surfaced in the respawn prompt so the
    // resumed agent knows the tasks are gone and can re-arm/re-run consciously.
    let killed_tasks = supervisor.live_task_labels(chat_id);

    // Kill the (now-idle-at-a-tool-boundary) agent before touching its transcript:
    // the rewrite needs exclusive access and the resume must re-read from disk. A
    // no-op if the turn already exited on its own.
    let aborted = supervisor.abort_agent(chat_id).await;
    // The respawn re-arms none of them; the fresh session republishes from empty.
    clear_live_tasks(write_tx, chat_id).await;

    // Rewrite the transcript via the per-adapter `prune_batch` behind `PruneOps`.
    // The whole burst for one transcript collapses into ONE read/serialize/fsync
    // (vs the old K separate `prune` calls) while reproducing the identical
    // blanked set + freed bytes. Group defensively by `jsonl_path` (in practice a
    // group is one chat = one transcript, so this is a single bucket) and batch
    // per path. Even on error we continue/respawn so the chat isn't left with no
    // running agent. `prune_ops()` is `Some` for any kind that produced a
    // `PruneRequest` (control.rs only enqueues those); guard anyway rather than
    // panic if somehow `None`.
    let mut total = prune::PruneStats::default();
    let mut groups: std::collections::HashMap<std::path::PathBuf, Vec<&prune::PruneRequest>> =
        std::collections::HashMap::new();
    for req in &reqs {
        groups.entry(req.jsonl_path.clone()).or_default().push(req);
    }
    for (jsonl_path, path_reqs) in &groups {
        // All reqs for a path share an `agent_kind` (control.rs derives both from
        // the same chat), so resolve ops from the first.
        let Some(first) = path_reqs.first() else {
            continue;
        };
        match first.agent_kind.prune_ops() {
            Some(ops) => {
                let targets: Vec<prune::PruneTarget> = path_reqs
                    .iter()
                    .map(|r| prune::PruneTarget::new(&r.tool_name, &r.needle, &r.reason))
                    .collect();
                match (ops.prune_batch)(jsonl_path, &targets) {
                    Ok(stats) => {
                        total.results_blanked += stats.results_blanked;
                        total.freed_bytes += stats.freed_bytes;
                    }
                    Err(e) => {
                        let reasons = path_reqs
                            .iter()
                            .map(|r| r.reason.as_str())
                            .collect::<Vec<_>>()
                            .join("; ");
                        error!(chat_id = %chat_id, error = %e, reasons = %reasons, "prune batch rewrite failed; skipping this transcript")
                    }
                }
            }
            None => {
                error!(chat_id = %chat_id, kind = ?first.agent_kind, "prune request for kind without PruneOps; skipping")
            }
        }
    }
    info!(
        chat_id = %chat_id,
        requests = reqs.len(),
        results_blanked = total.results_blanked,
        freed_bytes = total.freed_bytes,
        "pruned context (coalesced), respawning"
    );

    // Emit a synthetic `agent` frame (the user-visible "context pruned" line)
    // before the respawn so it orders ahead of the resumed agent's frames. One
    // combined frame for the whole burst; `results_blanked == 0` skips it (the
    // frame-skip contract is `crate::prune`'s).
    if total.results_blanked > 0 {
        let content = prune::pruned_frame_json(total);
        let g = mirror.read().await;
        send_agent_line(write_tx, &g, chat_id, content).await;
    }

    // Build the respawn prompt. On a real prune (≥1 output blanked) tell the agent
    // explicitly that it succeeded and not to re-issue the command — it was killed
    // mid-`prune-context`, so otherwise it can't distinguish success from a miss
    // and loops re-running the now-satisfied prune (which returns "no … call
    // found"). Fall back to a generic nudge when every rewrite errored or blanked
    // nothing.
    let mut prompt = if total.results_blanked > 0 {
        prune::pruned_respawn_prompt(total)
    } else {
        "context pruning complete, continue".to_string()
    };
    // If the restart killed live background tasks / monitors (forced prune, or a
    // same-instant race), tell the resumed agent: their in-process runtime is
    // gone and `--resume` does not bring it back, so it must re-arm/re-run any it
    // still needs (and re-check the status of anything side-effecting, e.g. a
    // deploy that was mid-flight).
    if !killed_tasks.is_empty() {
        let count = killed_tasks.len();
        let names = killed_tasks.join("; ");
        prompt.push_str(&format!(
            " NOTE: this prune restarted the session and terminated {count} \
             background task(s)/monitor(s) that were running ({names}) — they are NOT \
             automatically resumed. Re-arm or re-run any you still need, and check \
             the status of anything that was making changes (e.g. a deploy/build)."
        ));
    }

    // Resolve spawn params from the mirror (read guard) and respawn.
    let req = {
        let g = mirror.read().await;
        chat_spawn_request(&g, chat_id, prompt)
    };
    let Some(req) = req else {
        error!(chat_id = %chat_id, "cannot resolve spawn params after prune; chat left idle");
        return;
    };
    if req.agent_session_id.is_none() {
        // A prune request only fires mid-turn, so the session id is already
        // harvested — but guard anyway: resuming requires it, and a fresh session
        // would re-run from scratch with the wrong prompt.
        error!(chat_id = %chat_id, "no agent_session_id after prune; cannot resume, chat left idle");
        return;
    }
    let unconsumed = turns_to_requeue(aborted, req.agent_kind, req.agent_session_id.as_deref());
    supervisor.spawn_agent_then(req, unconsumed);
}

/// Prompts an aborted resident never consumed, minus those its transcript
/// already holds: claude persists a consumed turn (the user entry's `uuid`, or
/// a mid-turn `queued_command` attachment's `source_uuid`) and can die before
/// our reader sees the echo; `--resume` carries those itself.
fn turns_to_requeue(
    turns: Vec<QueuedTurn>,
    kind: AgentKind,
    session_id: Option<&str>,
) -> Vec<String> {
    if turns.is_empty() {
        return Vec::new();
    }
    let transcript = session_id
        .and_then(|sid| (kind.prune_ops()?.find_session)(&kind.cli_home()?, sid))
        .and_then(|path| std::fs::read_to_string(path).ok())
        .unwrap_or_default();
    unpersisted_prompts(turns, &transcript)
}

fn unpersisted_prompts(turns: Vec<QueuedTurn>, transcript: &str) -> Vec<String> {
    turns
        .into_iter()
        .filter(|t| !transcript.contains(&t.uuid))
        .map(|t| t.prompt)
        .collect()
}

/// Main-loop handling of a resident busy↔idle edge: write `agent_running`. The
/// idle process is kept warm for the next message's stdin until evicted
/// (`evict_idle_residents`). The idle edge is emitted only on a `Result`
/// turn-boundary frame with no live background task (see the reader's
/// emission gate), so it means the turn ended and nothing is in flight.
async fn handle_run_state(
    topic: &str,
    running: bool,
    pending_prunes: &prune::PendingPrunes,
    mirror: &SharedMirror,
    supervisor: &mut Supervisor,
    write_tx: &mpsc::Sender<WriteEvent>,
) {
    if is_deleted_chat(&*mirror.read().await, topic) {
        return;
    }
    // Stale busy edge, e.g. queued before a /stop tore the session down.
    if running && !supervisor.is_running(topic) {
        info!(topic = %topic, "ignoring stale busy edge: no live session");
        return;
    }
    if !running {
        // Stale edge: a turn was queued on stdin after the reader emitted it.
        if supervisor.is_running(topic) {
            info!(topic = %topic, "ignoring stale idle edge: a user turn is queued");
            return;
        }
        // Safety net only: a prune is normally applied the instant its
        // `ToolResult` cue fires. If that cue never arrived for a still-parked
        // request, apply it here rather than leak the entry; the respawn emits
        // its own RunState.
        if let Some(reqs) = take_prunes(topic, pending_prunes, supervisor).await {
            apply_prune_group(topic, reqs, mirror, supervisor, write_tx).await;
            return;
        }
    }
    send_run_state(write_tx, topic.to_string(), running).await;
    if !running {
        retire_stale_prompt(topic, &mut *mirror.write().await, supervisor, write_tx).await;
    }
}

/// Keep this list narrower than `AgentKind::ALL`: new adapter kinds are not
/// safe to auto-insert until all older shipped clients can decode their
/// `agent_kind` values in `machine_users.agents`. Users can still add newer
/// kinds explicitly from clients that know how to write them.
const DEFAULT_SEED_AGENT_KINDS: &[AgentKind] = &[AgentKind::Claude, AgentKind::Cursor];

/// One seeded roster entry. Serialized field order = declaration order and
/// MUST stay alphabetical — the payload bytes are a frozen contract with the
/// client-side seeders (iOS `.sortedKeys`, Android hand-sorted keys).
#[derive(serde::Serialize)]
struct SeedAgentEntry {
    agent_kind: &'static str,
    /// Deterministic `"seed-<wire_name>"` so concurrent seeds from different
    /// writers converge on identical rows. Opaque — never parsed as a UUID.
    id: String,
    model: &'static str,
    name: &'static str,
}

/// One-shot install-time seed of `machine_users.agents` defaults (migration
/// 0035; design + back-compat rationale: CLAUDE.md "Agent-roster seeding").
/// Gates: flag unset, machine id known (dev never seeds), owner row streamed,
/// probes cached. Drivers: every `CheckpointComplete` plus the
/// `probe_ready_rx` arm in `main`. Correctness doesn't depend on the flag —
/// the backend write is seed-only (`WHERE agents IS NULL`), so a redundant
/// attempt drains as a 0-rows no-op.
///
/// Returns true iff the flag was set — the caller must then persist
/// state.json.
async fn seed_initial_agents_if_pending(
    machine_id: Option<Uuid>,
    mirror: &mut Mirror,
    probe_statuses: Option<&[(AgentKind, (bool, bool))]>,
    write_tx: &mpsc::Sender<WriteEvent>,
) -> bool {
    if mirror.initial_agents_seed_done {
        return false;
    }
    let Some(machine_id) = machine_id else {
        return false; // dev mode (no machine id) never seeds
    };
    let Some(row_id) = mirror.owner_row() else {
        return false; // owner's machine_users row hasn't streamed yet
    };
    let Some(statuses) = probe_statuses else {
        // Boot checkpoint applies ~5s before the login-shell probes finish,
        // and an idle machine may never see another checkpoint — the
        // `probe_ready_rx` arm in `main` is the retry.
        return false;
    };
    let Some(row_uuid) = parse_uuid_str(&row_id) else {
        warn!(row_id, "machine_users.id is not a UUID; cannot seed agents");
        return false;
    };
    // Closed deterministic order — do not iterate `AgentKind::ALL` (see
    // `DEFAULT_SEED_AGENT_KINDS`).
    let entries: Vec<SeedAgentEntry> = DEFAULT_SEED_AGENT_KINDS
        .iter()
        .filter(|kind| {
            statuses
                .iter()
                .find_map(|(k, (installed, _))| (k == *kind).then_some(*installed))
                .unwrap_or(false)
        })
        .map(|kind| SeedAgentEntry {
            agent_kind: kind.descriptor().wire_name,
            id: format!("seed-{}", kind.descriptor().wire_name),
            model: "",
            name: "",
        })
        .collect();
    // Flag set BEFORE the write round-trips: a crash before the caller's
    // persist just re-attempts next boot and drains on the seed-only guard.
    mirror.initial_agents_seed_done = true;
    if entries.is_empty() {
        // Nothing installed → no write (an explicit `[]` would read as
        // user-emptied), but still drain the one-shot — clients seed residual
        // NULL rows themselves.
        info!(
            "no default-seed CLI installed; initial agents seed skipped (clients own the roster)"
        );
        return true;
    }
    let agents_json =
        serde_json::to_string(&entries).expect("array of small structs is always serializable");
    info!(
        %row_uuid,
        n = entries.len(),
        agents_json = %agents_json,
        "seeding initial machine_users.agents defaults for owner row"
    );
    let _ = write_tx
        .send(WriteEvent::SetMachineUserAgents {
            row_id: row_uuid,
            machine_id,
            agents_json,
        })
        .await;
    true
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
    // new is_sandboxed value into the in-memory mirror. Drop any cached entry:
    // K_machine is not rotated within an active membership (see
    // persist_user_key doc), so a cached Arc<KUser> is already the matching
    // one — but a NEGATIVE entry ("no key file") can have been cached in
    // between by anything that reads keys for this user (`upsert_chat`
    // decrypting a title, a message body), and it would outlive the key we
    // just wrote: every later message from that user would be dropped until
    // the spawner restarts.
    keys.forget(&user_id);
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
    probe_ready_tx: mpsc::Sender<()>,
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
        // Cache is now populated — nudge the main loop to retry the owner-row
        // agents seed that the boot-time PUT deferred (the PUT lands ~5s before
        // this point, with no probe data, and the row is never re-PUT). Send
        // AFTER `cache.set` so the retry observes the populated snapshot. A full
        // channel (capacity 1) means a signal is already pending — fine, drop.
        let _ = probe_ready_tx.try_send(());
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
/// column is absent / NULL — older iOS without the checkbox UI imports every
/// registered kind.
///
/// Status-emission contract (owned here, not in per-kind `import` fns):
///  - Emit `running-0` once at the very start.
///  - Per-kind `progress(pct)` callback emits `running-{scaled}` via
///    `write_tx.try_send` — `ImportStatus` is a tiny machines PATCH, channel
///    backpressure that drops one of these is acceptable (the next percent
///    step will reapply the correct value), so we log on a failed `try_send`
///    but don't await. The callback fires roughly once per imported chat
///    (per-percent throttle inside the adapter); a shared `last_scaled` gate
///    here suppresses the duplicate `running-{scaled}` values that several
///    per-kind percents collapse to once they're rescaled into this kind's
///    slice of the shared 0..99 bar, so the wire still only moves on a real
///    bar advance.
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
    // Helper: deliver a status update with a blocking send so it can't be
    // dropped. The 1024-cap writer channel is SHARED with this import's bulk
    // PutChat/PutMessage row writes, so on a large import (hundreds of
    // sessions) it legitimately fills with row writes faster than the writer
    // drains; a non-blocking `try_send` here would silently drop updates —
    // including the terminal `finished`, which is the only signal that
    // dismisses the client's progress modal (observed: a 331-session import
    // dropped `finished` → modal hung forever at the last delivered percent).
    // Blocking instead applies backpressure (the import loop waits for the
    // writer to drain a slot) and `finished` is guaranteed to land. Safe to
    // block here: this runs in its own spawned task (see the `tokio::spawn`
    // caller), so it never stalls the dispatcher, and the writer's
    // retry-with-backoff keeps draining the channel.
    async fn emit_status(tx: &mpsc::Sender<WriteEvent>, machine_id: Uuid, status: String) {
        if let Err(e) = tx
            .send(WriteEvent::ImportStatus {
                machine_id,
                status: status.clone(),
            })
            .await
        {
            warn!(error = %e, %status, "failed to enqueue import status (writer channel closed)");
        }
    }

    // Caller (parsed_import_kinds) already falls back to ALL on
    // None/empty/all-unknown — so an empty Vec here would be a programmer
    // bug. Don't divide by zero in the rescaler; emit finished and bail.
    if kinds.is_empty() {
        warn!("run_history_import called with empty kinds list — nothing to do");
        emit_status(&write_tx, machine_id, "finished".to_string()).await;
        return;
    }

    emit_status(&write_tx, machine_id, "running-0".to_string()).await;

    let n = kinds.len() as u32;
    // Last `running-{scaled}` value we put on the wire, shared across every
    // kind's callback. The bar is monotonic non-decreasing (idx grows, and
    // each kind's percent grows), so deduping consecutive identical values
    // collapses the per-chat callbacks down to one PATCH per integer-percent
    // advance — `running-0` is already on the wire, so seed it to 0.
    let last_scaled = Arc::new(AtomicU8::new(0));
    let mut ok_count = 0usize;
    for (idx, kind) in kinds.iter().enumerate() {
        let idx_u32 = idx as u32;
        let tx_for_progress = write_tx.clone();
        let last_scaled = Arc::clone(&last_scaled);
        // Per-kind rescaler. `scaled = (idx*100 + pct) / N`, capped at 99
        // (the dispatcher owns 100/finished). The adapter calls this ~once per
        // imported chat; we emit only when the rescaled bar value actually
        // moves, so multi-kind imports don't re-send the same percent N times.
        let progress: crate::adapter::ImportProgress = Box::new(move |pct: u8| {
            let tx = tx_for_progress.clone();
            let last_scaled = Arc::clone(&last_scaled);
            Box::pin(async move {
                let pct = pct.min(100) as u32;
                let scaled = ((idx_u32 * 100 + pct) / n).min(99) as u8;
                if last_scaled.swap(scaled, Ordering::Relaxed) != scaled {
                    emit_status(&tx, machine_id, format!("running-{scaled}")).await;
                }
            })
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

    emit_status(&write_tx, machine_id, "finished".to_string()).await;
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
/// the exit forever. The queue isn't persisted: writes still unsent at the
/// timeout are lost.
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

/// Match `arg` against `--name value` (consuming the next argv item) or
/// `--name=value`, returning the value when it matches. Lets each flag in a
/// hand-rolled argv parse be a one-liner instead of two match arms apiece.
/// Shared by every `*_cli` subcommand's parser.
fn take_flag<'a>(
    arg: &str,
    name: &str,
    it: &mut impl Iterator<Item = &'a String>,
) -> Option<String> {
    if arg == name {
        it.next().cloned()
    } else {
        arg.strip_prefix(name)
            .and_then(|rest| rest.strip_prefix('='))
            .map(str::to_string)
    }
}

/// The chat an agent-run subcommand acts on: `ZUCCHINI_CHAT_ID`, which agent.rs
/// exports on every spawn.
fn env_chat_id() -> Option<String> {
    std::env::var("ZUCCHINI_CHAT_ID").ok()
}

/// Consumes a `--chat-id` flag and its value. The flag is gone, but resumed
/// conversations still replay calls that pass it, so it's ignored, not rejected.
fn skip_legacy_chat_id<'a>(arg: &str, it: &mut impl Iterator<Item = &'a String>) -> bool {
    take_flag(arg, "--chat-id", it).is_some()
}

/// CLI entry point for `zucchini-spawner attach-file <abs-path>` (chat from
/// `ZUCCHINI_CHAT_ID`).
///
/// Parses argv (hand-rolled — clap is overkill for one flag and a positional
/// arg), connects to the daemon's `~/.zucchini-spawner/control.sock`, runs
/// one `attach_file` RPC, and prints a human-readable result. Exits 0 on
/// success, 1 on any failure. The CLI itself does no crypto or HTTP — the
/// daemon owns the JWT and K_user.
async fn run_attach_file_cli(args: &[String]) {
    const USAGE: &str = "usage: zucchini-spawner attach-file <absolute-path>\n  Attaches the file to your next message in this chat; run it before writing that message.";

    let mut path: Option<String> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let s = a.as_str();
        if s == "-h" || s == "--help" {
            usage_and_exit(USAGE, 0);
        } else if skip_legacy_chat_id(s, &mut it) {
        } else if path.is_none() {
            path = Some(s.to_string());
        } else {
            usage_and_exit(USAGE, 2);
        }
    }
    let (Some(chat_id), Some(path)) = (env_chat_id(), path) else {
        usage_and_exit(USAGE, 2);
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

/// Append a one-line CLI-invocation record to `spawner.log` (the daemon's
/// `StandardOutPath`). Used by `run_prune_context_cli` to log that the
/// subprocess started, BEFORE its RPC — see the call site for why the
/// daemon-side log can't answer "was the CLI executed?". Best-effort:
/// open-append-write with O_APPEND so concurrent single-line writes from
/// parallel prune CLIs (and the daemon) don't interleave; any error is
/// swallowed (a diagnostic log must never disrupt the prune).
fn log_prune_cli_invoked(chat_id: &str, tool_name: &str, needle: &str) {
    use std::io::Write;
    let path = zucchini_spawner_dir().join("spawner.log");
    let line = format!(
        "{} PRUNE-CLI invoked pid={} chat_id={} tool_name={} needle={}\n",
        chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Micros, true),
        std::process::id(),
        chat_id,
        tool_name,
        needle,
    );
    if let Ok(mut f) = std::fs::OpenOptions::new().append(true).open(&path) {
        let _ = f.write_all(line.as_bytes());
    }
}

/// Parse `prune-context` CLI args into the call-level `--force` plus the batch
/// of prune targets. `Err(())` signals any structural problem (so the
/// caller prints usage + exits 2). Pure (no env, no I/O, no process exit) so the
/// batching/terminator semantics are unit-testable.
///
/// Each `--summary` (or `--reason` alias) CLOSES the current target: a target is
/// the run of `[--tool-name] --args` flags before it. Per target, `--args` is
/// required by PRESENCE (`--args ""` is the no-args selector); `--tool-name` is
/// optional (`""` = any-tool selector). A `--summary` with no preceding `--args`,
/// a dangling `--tool-name`/`--args` not closed by a `--summary`, an unknown flag,
/// `-h`/`--help`, or an empty batch are all errors. `--flag=value` accepted.
fn parse_prune_args(args: &[String]) -> Result<(bool, Vec<control::PruneItem>), ()> {
    fn close_item(
        items: &mut Vec<control::PruneItem>,
        tool: &mut Option<String>,
        needle: &mut Option<String>,
        summary: String,
    ) -> Result<(), ()> {
        let needle = needle.take().ok_or(())?;
        items.push(control::PruneItem {
            tool_name: tool.take().unwrap_or_default(),
            needle,
            reason: summary,
        });
        Ok(())
    }

    let mut force = false;
    let mut items: Vec<control::PruneItem> = Vec::new();
    let mut cur_tool: Option<String> = None;
    let mut cur_needle: Option<String> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let s = a.as_str();
        if s == "-h" || s == "--help" {
            return Err(());
        } else if s == "--force" {
            // Call-level boolean flag (no value): prune even while background
            // tasks / monitors are live, terminating them on the restart.
            force = true;
        } else if let Some(v) = take_flag(s, "--tool-name", &mut it) {
            cur_tool = Some(v);
        } else if let Some(v) = take_flag(s, "--args", &mut it) {
            cur_needle = Some(v);
        } else if skip_legacy_chat_id(s, &mut it) {
        } else if let Some(summary) = take_flag(s, "--summary", &mut it) {
            close_item(&mut items, &mut cur_tool, &mut cur_needle, summary)?;
        } else if let Some(summary) = take_flag(s, "--reason", &mut it) {
            close_item(&mut items, &mut cur_tool, &mut cur_needle, summary)?;
        } else {
            return Err(());
        }
    }
    // Dangling `--tool-name`/`--args` with no closing `--summary` is a malformed
    // trailing target, not a silent drop. An empty batch is also an error.
    if cur_tool.is_some() || cur_needle.is_some() || items.is_empty() {
        return Err(());
    }
    Ok((force, items))
}

/// CLI entry point for `zucchini-spawner prune-context [--tool-name <Tool>]
/// --args "<glob>" --summary "<digest>" [... more triples ...]`.
///
/// Thin RPC client over the control socket (mirrors `run_attach_file_cli`).
/// Accepts ONE OR MORE prune targets per call: each `--summary` (alias
/// `--reason`) CLOSES the current target (the `[--tool-name] --args` run before
/// it). Batching is the point — several outputs forgotten in one process → one
/// RPC → one restart, vs parallel processes that reap each other (`control.rs`
/// `PruneContext`). A lone triple is a 1-item batch, byte-identical to the old
/// single-call form (older-binary mid-turn stays hot-reload compatible). Per
/// target: `--args` required (`""` = no-args selector); `--tool-name` optional
/// (omit/`""` to match on `--args` alone, as codex does); `--summary` required
/// (terminates the target). Chat from `ZUCCHINI_CHAT_ID`. `--flag=value` accepted. `--args` globs argument values
/// and blanks only the most recent match (output reports how many remain). The
/// daemon aborts the agent right after replying, so a post-send connection-reset
/// is expected — the prune proceeds regardless.
async fn run_prune_context_cli(args: &[String]) {
    const USAGE: &str = "usage: zucchini-spawner prune-context [--tool-name <ToolName>] --args \"<glob>\" --summary \"<digest>\" [... repeat per output ...] [--force]\n  Prune one or more tool outputs in a single call: each --summary CLOSES the target made of the --tool-name/--args before it, so repeat the triple to forget several outputs at once (one restart for the whole batch). --tool-name is optional per target — omit it to match on --args alone. --summary is the takeaway from that output you still need going forward — the slice that matters for the task at hand, NOT a recap of the whole output (--reason accepted as a legacy alias). --args is a glob (supports *) over the call's argument VALUES, not key names; blanks only the most recent matching call (repeat to prune older ones). Use --args \"\" to prune a call you made with no arguments. --force prunes even while background tasks/monitors are running (the restart terminates them); without it, prune-context errors so you can wait for them to finish.\n  E.g. --tool-name Read --args \"src/main.ts\" --summary \"main.ts: initApp() wires routes then listens on PORT\" --args \"*build.log*\" --summary \"build: passed, nothing relevant\"";

    if args.first().is_some_and(|a| a == "-h" || a == "--help") {
        usage_and_exit(USAGE, 0);
    }
    // `--reason` is the legacy alias for `--summary` (older binary mid-turn keeps
    // parsing across a hot-reload); both feed the same wire field `reason`.
    let Ok((force, items)) = parse_prune_args(args) else {
        usage_and_exit(USAGE, 2);
    };
    let Some(chat_id) = env_chat_id() else {
        usage_and_exit(USAGE, 2);
    };

    // Record the CLI INVOCATION itself — written here, at the top of the
    // subprocess and BEFORE any RPC, so it answers "was the prune-context CLI
    // actually executed?" independently of whether the call ever reaches the
    // daemon. A CLI subprocess's stdout goes to claude (its parent), not to
    // `spawner.log`, and the daemon-side `prune-context called` log only fires
    // after the socket connect. One line per batch item (the pid + high-res
    // timestamp ties them together). Best-effort — never blocks or fails the
    // prune.
    for item in &items {
        log_prune_cli_invoked(&chat_id, &item.tool_name, &item.needle);
    }

    match control::prune_context_via_socket(&chat_id, items.clone(), force).await {
        Ok(counts) => {
            // Per-target feedback (parallel to `items`): a `0` count is a miss the
            // batch tolerated (≥1 other item matched), reported so the agent sees
            // which needle to fix; a non-zero `n` blanked the most recent of `n`
            // eligible matches, leaving `n-1`.
            let mut queued = 0usize;
            for (item, n) in items.iter().zip(counts.iter()) {
                let what = control::describe_prune_target(&item.tool_name, &item.needle);
                match *n {
                    0 => println!("· no {what} found — skipped"),
                    1 => {
                        queued += 1;
                        println!("· pruned the most recent {what} (the only match)");
                    }
                    n => {
                        queued += 1;
                        println!(
                            "· pruned the most recent {what}; {} remain — repeat to prune the next",
                            n - 1
                        );
                    }
                }
            }
            if queued > 0 {
                // Exit 0 CLEANLY — do NOT trigger the restart from here. The
                // daemon already has the prune queued; it applies it (abort →
                // rewrite → respawn) when claude emits THIS call's `tool_result`
                // frame, i.e. strictly AFTER claude has persisted this call's
                // result to the transcript. That ordering is the whole point: the
                // resumed agent then sees its own completed prune call in
                // context, so it won't re-run the now-satisfied prune. Triggering
                // the restart here instead would SIGTERM this CLI mid-call, before
                // the result persisted — the lost-tool_result bug this avoids.
                println!("queued {queued} prune(s); the agent will restart to apply them");
            } else {
                println!("no eligible matches — nothing pruned");
            }
            std::process::exit(0);
        }
        Err(e) => {
            eprintln!("prune-context failed: {e:#}");
            std::process::exit(1);
        }
    }
}

/// Prints `usage` and exits with `code`: to stdout for an asked-for `--help`
/// (0), to stderr for a misuse.
fn usage_and_exit(usage: &str, code: i32) -> ! {
    if code == 0 {
        println!("{usage}");
    } else {
        eprintln!("{usage}");
    }
    std::process::exit(code);
}

const SCHEDULED_MESSAGES_USAGE: &str = "usage: zucchini-spawner scheduled-messages <command>
commands:
  create --body <TEXT> --at <local datetime, e.g. 2026-06-07T09:00:00>
                      schedule a message into this chat. --at is the user's
                      wall-clock time, without an offset or Z
  list                list this chat's pending messages (delivered ones are gone)
  delete --id <UUID>  cancel a pending message";

fn scheduled_messages_usage_and_exit(code: i32) -> ! {
    usage_and_exit(SCHEDULED_MESSAGES_USAGE, code)
}

/// `scheduled-messages` subcommand: the agent's create/list/delete of this
/// chat's pending scheduled messages. Each is one RPC over the control socket;
/// the daemon owns K_user, the JWT and the user's timezone.
async fn run_scheduled_messages_cli(args: &[String]) {
    match args.first().map(String::as_str) {
        Some("create") => run_create_scheduled_message_cli(&args[1..]).await,
        Some("list") => run_list_scheduled_messages_cli(&args[1..]).await,
        Some("delete") => run_delete_scheduled_message_cli(&args[1..]).await,
        Some("-h" | "--help") => scheduled_messages_usage_and_exit(0),
        _ => scheduled_messages_usage_and_exit(2),
    }
}

/// `scheduled-messages create`. `--at <local-datetime>` is required. Forwarded
/// raw, zoned + validated daemon-side by `control::normalize_deliver_at` (naive
/// local anchored in the user's tz, which only the daemon holds; unparseable
/// values rejected so a garbage timestamp can't misfire). The queue-when-free
/// path (`deliver_at` null) is composer-only — no use for the agent, which
/// already runs at turn end.
async fn run_create_scheduled_message_cli(args: &[String]) {
    let mut body: Option<String> = None;
    let mut at: Option<String> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let s = a.as_str();
        if s == "-h" || s == "--help" {
            scheduled_messages_usage_and_exit(0);
        } else if skip_legacy_chat_id(s, &mut it) {
        } else if let Some(v) = take_flag(s, "--body", &mut it) {
            body = Some(v);
        } else if let Some(v) = take_flag(s, "--at", &mut it) {
            at = Some(v);
        } else {
            scheduled_messages_usage_and_exit(2);
        }
    }
    let (Some(chat_id), Some(body), Some(at)) = (env_chat_id(), body, at) else {
        scheduled_messages_usage_and_exit(2);
    };

    match control::schedule_message_via_socket(&chat_id, &body, Some(at)).await {
        Ok(id) => {
            // Stable, machine-parseable first line so the agent can grep for
            // success; the row id follows for cross-referencing.
            println!("scheduled message {id} on chat {chat_id}");
            std::process::exit(0);
        }
        Err(e) => {
            eprintln!("scheduled-messages create failed: {e:#}");
            std::process::exit(1);
        }
    }
}

/// `scheduled-messages list`: what this chat still has pending. Rows come back
/// decrypted and already in the user's local wall-clock, so this only formats
/// and prints. Exit 0 whether or not anything is pending; 1 only on an RPC
/// failure.
async fn run_list_scheduled_messages_cli(args: &[String]) {
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let s = a.as_str();
        if s == "-h" || s == "--help" {
            scheduled_messages_usage_and_exit(0);
        } else if !skip_legacy_chat_id(s, &mut it) {
            scheduled_messages_usage_and_exit(2);
        }
    }
    let Some(chat_id) = env_chat_id() else {
        scheduled_messages_usage_and_exit(2);
    };

    match control::list_scheduled_messages_via_socket(&chat_id).await {
        Ok(messages) if messages.is_empty() => {
            println!("no pending scheduled messages in this chat");
            std::process::exit(0);
        }
        Ok(messages) => {
            for m in messages {
                let when = m
                    .deliver_at
                    .unwrap_or_else(|| "when the agent is next free".to_string());
                println!("== {}  {}", m.id, when);
                if !m.attachments.is_empty() {
                    println!("attachments: {}", m.attachments.join(", "));
                }
                println!("{}\n", m.text);
            }
            std::process::exit(0);
        }
        Err(e) => {
            eprintln!("scheduled-messages list failed: {e:#}");
            std::process::exit(1);
        }
    }
}

/// `scheduled-messages delete`. The daemon refuses an id that isn't pending in
/// this chat, so an already-delivered row reports as such here.
async fn run_delete_scheduled_message_cli(args: &[String]) {
    let mut id: Option<String> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let s = a.as_str();
        if s == "-h" || s == "--help" {
            scheduled_messages_usage_and_exit(0);
        } else if skip_legacy_chat_id(s, &mut it) {
        } else if let Some(v) = take_flag(s, "--id", &mut it) {
            id = Some(v);
        } else {
            scheduled_messages_usage_and_exit(2);
        }
    }
    let (Some(chat_id), Some(id)) = (env_chat_id(), id) else {
        scheduled_messages_usage_and_exit(2);
    };

    match control::cancel_scheduled_message_via_socket(&chat_id, &id).await {
        Ok(id) => {
            println!("deleted scheduled message {id}");
            std::process::exit(0);
        }
        Err(e) => {
            eprintln!("scheduled-messages delete failed: {e:#}");
            std::process::exit(1);
        }
    }
}

const WATCH_USAGE: &str = "usage: zucchini-spawner watch <command>
commands:
  create --cmd <SHELL COMMAND> --ttl <30m|12h|7d|2w|forever> --description <TEXT>
                      run the command in the current directory until it exits or
                      the TTL passes; each stdout line becomes a message into this
                      chat. Any exit (whatever the code) ends and removes the watch,
                      and the chat is told how it ended. All three flags are required.
                      The command is started again after a spawner restart, so it
                      must be safe to run twice. It gets $ZUCCHINI_WATCH_STATE_DIR,
                      a directory that survives that, to keep its cursor in. The
                      description names the watch, so it is unique within the chat.
                      Every line is a full turn for the agent and is read as
                      trusted input: print one short line per real event with ids
                      and event types only, never raw subjects, comment bodies or
                      other third-party text (fetch those once woken). A watch
                      that floods is stopped.
  list                this chat's watches
  delete --description <TEXT>
                      stop and remove this chat's watch of that description";

fn watch_usage_and_exit(code: i32) -> ! {
    usage_and_exit(WATCH_USAGE, code)
}

/// `watch` subcommand: the agent's create/list/delete of daemon-run watches
/// (`watch.rs`). Each is one RPC over the control socket.
async fn run_watch_cli(args: &[String]) {
    let Some(command) = args.first().map(String::as_str) else {
        watch_usage_and_exit(2);
    };
    let (mut cmd, mut ttl, mut description) = (None, None, None);
    let mut it = args[1..].iter();
    while let Some(a) = it.next() {
        let s = a.as_str();
        if s == "-h" || s == "--help" {
            watch_usage_and_exit(0);
        } else if let Some(v) = take_flag(s, "--cmd", &mut it) {
            cmd = Some(v);
        } else if let Some(v) = take_flag(s, "--ttl", &mut it) {
            ttl = Some(v);
        } else if let Some(v) = take_flag(s, "--description", &mut it) {
            description = Some(v);
        } else {
            watch_usage_and_exit(2);
        }
    }
    let Some(chat_id) = env_chat_id() else {
        watch_usage_and_exit(2);
    };
    let result = match (command, cmd, description) {
        ("delete", None, Some(description)) => {
            control::delete_watch_via_socket(&chat_id, description)
                .await
                .map(|w| println!("deleted watch \"{}\"", w.description))
        }
        ("create", cmd, description) => {
            let required = |value: Option<String>, flag: &str| {
                value.unwrap_or_else(|| {
                    eprintln!("watch create: {flag} is required");
                    watch_usage_and_exit(2)
                })
            };
            let cmd = required(cmd, "--cmd");
            let ttl = required(ttl, "--ttl");
            let description = required(description, "--description");
            let cwd = std::env::current_dir()
                .map(|d| d.to_string_lossy().into_owned())
                .unwrap_or_default();
            control::create_watch_via_socket(&chat_id, cmd, cwd, ttl, description)
                .await
                .map(|w| println!("created watch\n{}", w.summary()))
        }
        ("list", None, None) => control::list_watches_via_socket(&chat_id)
            .await
            .map(|watches| {
                if watches.is_empty() {
                    println!("no watches");
                }
                for w in watches {
                    println!("{}", w.summary());
                }
            }),
        ("-h" | "--help", ..) => watch_usage_and_exit(0),
        _ => watch_usage_and_exit(2),
    };
    if let Err(e) = result {
        eprintln!("watch {command} failed: {e:#}");
        std::process::exit(1);
    }
}

const SYSTEM_PROMPT_USAGE: &str = "usage: zucchini-spawner system-prompt <command>
commands:
  show                print this chat's system prompt: the text, and the files the
                      user listed (their contents are part of the prompt; edit a
                      file itself to change what it says)
  set --text <TEXT>   replace the text with TEXT (at most 8000 bytes; empty clears it).
                      The whole text is replaced, so include what `show` printed
                      and should stay. Keep it short. The user sees and edits
                      the same text in the app. It applies from the next start
                      of the agent process; on an always-on chat that start
                      comes right after this turn and ends background tasks.";

fn system_prompt_usage_and_exit(code: i32) -> ! {
    usage_and_exit(SYSTEM_PROMPT_USAGE, code)
}

/// `system-prompt` subcommand: the agent's read and rewrite of its chat's user
/// system prompt text (`user_prompt.rs`). Each is one RPC over the control socket.
async fn run_system_prompt_cli(args: &[String]) {
    let Some(command) = args.first().map(String::as_str) else {
        system_prompt_usage_and_exit(2);
    };
    let mut text = None;
    let mut it = args[1..].iter();
    while let Some(a) = it.next() {
        let s = a.as_str();
        if s == "-h" || s == "--help" {
            system_prompt_usage_and_exit(0);
        } else if let Some(v) = take_flag(s, "--text", &mut it) {
            text = Some(v);
        } else {
            system_prompt_usage_and_exit(2);
        }
    }
    if matches!(command, "-h" | "--help") {
        system_prompt_usage_and_exit(0);
    }
    let Some(chat_id) = env_chat_id() else {
        system_prompt_usage_and_exit(2);
    };
    let result = match (command, text) {
        ("show", None) => control::get_system_prompt_via_socket(&chat_id)
            .await
            .map(|prompt| {
                if let Some(files) = prompt.files {
                    println!("files: {files}");
                }
                match prompt.text {
                    Some(text) => println!("text:\n{text}"),
                    None => println!("no text"),
                }
            }),
        ("set", Some(text)) => control::set_system_prompt_via_socket(&chat_id, text)
            .await
            .map(|prompt| {
                match prompt.text {
                    Some(_) => println!("system prompt text replaced"),
                    None => println!("system prompt text cleared"),
                }
                if prompt.sync_pending {
                    println!("saved, but not synced back yet: `show` may still print the old text for a while, so don't build the next `set` on it");
                }
            }),
        _ => system_prompt_usage_and_exit(2),
    };
    if let Err(e) = result {
        eprintln!("system-prompt {command} failed: {e:#}");
        std::process::exit(1);
    }
}

/// `prune-reminder-hook` subcommand: claude's match-all `PostToolUse` hook
/// trampoline. Reads the hook JSON payload from stdin, delegates the size gate
/// to the pure [`crate::prune::prune_reminder_output`], and prints the
/// `additionalContext` line on a large `tool_response` (nudging the agent to
/// prune). ALWAYS exits 0 — a failing hook could disrupt claude, so any
/// read/parse failure is swallowed (silent, exit 0). This hook is reminder-only:
/// the prune RESTART is driven by the main loop when claude emits the
/// `prune-context` call's `tool_result` frame, not from here.
fn run_prune_reminder_hook_cli() {
    use std::io::Read;
    let mut payload = String::new();
    // Best-effort read; on any stdin error, treat as empty → silent.
    let _ = std::io::stdin().read_to_string(&mut payload);

    if let Some(line) = prune::prune_reminder_output(&payload) {
        println!("{line}");
    }
    std::process::exit(0);
}

/// Shared shutdown path used by both the `to_uninstall` PowerSync signal
/// and the 410-from-/auth/token revoked signal. Caller breaks the main
/// loop after this returns.
async fn run_uninstall(
    supervisor: &mut Supervisor,
    heartbeat_cancel: &CancellationToken,
    writer: &Writer,
) {
    watch::shutdown().await;
    supervisor.shutdown_all().await;
    heartbeat_cancel.cancel();
    let _ = wait_for_writer_idle(writer).await;
    uninstall::spawn_detached_cleanup();
}

/// Hold an exclusive single-instance lock on the pidfile
/// `~/.zucchini-spawner/spawner.pid` for the daemon's whole lifetime so only
/// ONE spawner can run against this runtime dir. A second daemon — a stray
/// `zucchini-spawner … | head` probe that fell through to boot, a leftover dev
/// `cargo run`, a double start — would join the same PowerSync stream and spawn
/// a *duplicate* agent for every user message, so the chat shows two responses
/// and two `[result]`s (the bug this guards).
///
/// The platform mechanics live in `sys::acquire_single_instance_lock` (unix:
/// crash-safe `flock`, released by the kernel on exit; lock fd is `O_CLOEXEC`
/// so spawned agents don't inherit it). On contention this prints the running
/// pid and exits. On any I/O failure taking the lock it logs and continues
/// unlocked (best-effort — never wedge the service over a lock we couldn't
/// acquire). The returned handle MUST be kept alive by the caller; dropping it
/// unlocks.
fn acquire_single_instance_lock() -> Option<sys::InstanceLock> {
    let path = zucchini_spawner_dir().join("spawner.pid");
    match sys::acquire_single_instance_lock(&path) {
        Ok(Some(lock)) => Some(lock),
        Ok(None) => {
            let existing = std::fs::read_to_string(&path).unwrap_or_default();
            let pid = existing.trim();
            let pid = if pid.is_empty() { "unknown" } else { pid };
            eprintln!(
                "zucchini-spawner: another instance is already running (pid {pid}); exiting."
            );
            std::process::exit(0);
        }
        Err(e) => {
            warn!(error = %e, path = %path.display(), "could not take single-instance lock; continuing unlocked");
            None
        }
    }
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
        if first == "prune-context" {
            run_prune_context_cli(&raw_args[1..]).await;
            return;
        }
        if first == "prune-reminder-hook" {
            // claude `PostToolUse` hook (wired via `--settings`; see
            // `adapters/claude.rs`). Reads the hook JSON from stdin, gates on
            // `tool_response` size, and on a large result prints the
            // additionalContext line that claude surfaces as a
            // `<system-reminder>` nudging the agent to prune. Inert/exit-0 on any
            // failure — a failing hook could disrupt claude.
            run_prune_reminder_hook_cli();
            return;
        }
        if first == "scheduled-messages" {
            run_scheduled_messages_cli(&raw_args[1..]).await;
            return;
        }
        if first == "watch" {
            run_watch_cli(&raw_args[1..]).await;
            return;
        }
        if first == "system-prompt" {
            run_system_prompt_cli(&raw_args[1..]).await;
            return;
        }
        // Legacy spelling: conversations started before `scheduled-messages`
        // still carry the old instruction and resume with it.
        if first == "schedule-message" {
            run_create_scheduled_message_cli(&raw_args[1..]).await;
            return;
        }
        if first == "hermes-turn" {
            // Per-turn trampoline child for the hermes adapter. Connects to
            // the spawner's hermes socket, sends one `turn` frame, shuttles
            // envelopes back to stdout as claude-shape NDJSON. See
            // `adapters/hermes/trampoline.rs` for the wire-format contract.
            let parsed = match hermes_support::trampoline::HermesTurnArgs::parse(&raw_args[1..]) {
                Ok(a) => a,
                Err(e) => {
                    eprintln!("hermes-turn: {e:#}");
                    std::process::exit(2);
                }
            };
            // `run_hermes_turn` returns the trampoline's process exit code
            // (0 on clean result, 1 on any error). The supervisor
            // synthesises INTERRUPTED_RESULT on non-zero exits so the
            // chat lands a terminator either way.
            let code = hermes_support::trampoline::run_hermes_turn(parsed).await;
            std::process::exit(code);
        }

        // Any other first argument is NOT a recognized subcommand. Print usage
        // and exit WITHOUT falling through to the daemon boot below. The daemon
        // is only ever started with zero args (launchd/systemd `ProgramArguments`
        // is just the binary path), so a stray flag, a typo, or a probe like
        // `zucchini-spawner --help | head` must never silently launch a SECOND
        // spawner — a second instance joins the same sync stream and double-spawns
        // an agent for every user message (real incident: a `--help` invocation
        // ran as a 52-minute zombie spawner alongside the service).
        let is_help = first == "--help" || first == "-h";
        eprintln!(
            "zucchini-spawner — run with NO arguments to start the daemon.\n\
             subcommands: attach-file, prune-context, prune-reminder-hook, scheduled-messages, watch, system-prompt, hermes-turn\n\
             flags: --version|-V, --help|-h"
        );
        std::process::exit(if is_help { 0 } else { 2 });
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

    // Clean up the `<exe>.old` a prior self-update parked aside (Windows can't
    // overwrite a live .exe, so the updater renames it out of the way and we
    // unlink it here once it's no longer mapped). No-op on unix.
    sys::sweep_stale_exe();

    // Single-instance guard. MUST stay bound for the whole daemon lifetime —
    // dropping the handle releases the lock. A second daemon would otherwise
    // join the same sync stream and spawn a duplicate agent for every user
    // message (two responses + two `[result]`s per chat). See
    // `acquire_single_instance_lock`.
    let _instance_lock = acquire_single_instance_lock();
    if _instance_lock.is_some() {
        orphans::sweep();
        sys::ensure_supervisor_keepalive();
    }

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
    mirror.write().await.unseal_loaded_chats(&keys);

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
    // read by `seed_initial_agents_if_pending` at every CheckpointComplete.
    // `OnceLock` (not `RwLock`) because the write happens exactly once and
    // the read side never blocks. `None` means "probes not in yet"; the
    // seeding pass defers until they're cached.
    let probe_statuses_cache: ProbeStatusesCache = Arc::new(std::sync::OnceLock::new());

    // Hermes plugin self-heal: write the embedded plugin payload to
    // `~/.hermes/plugins/zucchini/` if missing or byte-different from the
    // embedded copy. Runs unconditionally — cheap (3 file-stats + 0-3
    // writes per boot) and removes a class of "is the plugin installed?"
    // failure modes. Logged-and-skipped on filesystem error so a
    // permission glitch doesn't gate the rest of the spawner.
    if let Err(e) = hermes_support::plugin_install::ensure_hermes_plugin_installed() {
        warn!(error = %e, "hermes plugin install/self-heal failed; hermes chats may fail until resolved");
    }

    // Hermes socket server: binds the single-socket multiplexer at
    // `~/.zucchini-spawner/hermes.sock` (configurable via
    // ZUCCHINI_SPAWNER_SOCK env var for dev/tests). Spawns the
    // `hermes gateway run` child process under the user's login shell so
    // the plugin can dial back in. The trampoline children read the same
    // env var to find the socket. Export the path so the env var is
    // inherited by every subprocess (`agent.rs::default_spawn_fn` doesn't
    // override it).
    let hermes_socket = match hermes_support::socket_server::start(write_tx.clone(), mirror.clone())
    {
        Ok(handle) => {
            std::env::set_var("ZUCCHINI_SPAWNER_SOCK", &handle.socket_path);
            Some(handle)
        }
        Err(e) => {
            warn!(error = %e, "hermes socket server failed to start; hermes chats will fail");
            None
        }
    };

    // Probe-completion signal: `spawn_startup_info_report` sends once after
    // it fills `probe_statuses_cache` — the retry for the seed the boot
    // checkpoint deferred. Capacity 1 + `try_send` = at-most-one pending
    // nudge. Never fires in dev (probe task not spawned; dev never seeds).
    let (probe_ready_tx, mut probe_ready_rx) = mpsc::channel::<()>(1);

    let heartbeat_cancel = CancellationToken::new();
    if let Some(p) = &prod {
        info!(machine_id = %p.machine_id, "starting heartbeat task");
        spawn_heartbeat(p.machine_id, write_tx.clone(), heartbeat_cancel.clone());
        spawn_startup_info_report(
            p.machine_id,
            write_tx.clone(),
            probe_statuses_cache.clone(),
            probe_ready_tx,
        );

        // Startup pubkey publish: read guard is fine — we only inspect
        // `mirror.spawner_pubkey`. The call inside `handle_sync_event` runs
        // under the write guard already; both paths use the same helper.
        let g = mirror.read().await;
        publish_spawner_pubkey_if_needed(p.machine_id, our_pubkey_b64.as_deref(), &g, &write_tx)
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

    // Watches hand their messages to the main loop, the only owner of the
    // supervisor, and learn from the reply whether the chat took them.
    type WatchDelivery = (
        watch::Outgoing,
        tokio::sync::oneshot::Sender<anyhow::Result<()>>,
    );
    let (watch_tx, mut watch_rx) = mpsc::channel::<WatchDelivery>(16);
    let (watched_chat_tx, mut watched_chat_rx) = mpsc::unbounded_channel::<String>();
    if _instance_lock.is_some() {
        let deliver: watch::Deliver = Box::new(move |out| {
            let watch_tx = watch_tx.clone();
            Box::pin(async move {
                let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
                watch_tx.send((out, reply_tx)).await?;
                reply_rx.await?
            })
        });
        watch::start(deliver, watched_chat_tx);
    }

    // Prune-context: the control task does the read-only lookup + transcript
    // pre-scan, then parks the request in this shared table so the main loop
    // (sole owner of `supervisor`) can abort the agent, rewrite the jsonl, and
    // respawn on the same session with a "continue" prompt. A shared lock (not a
    // channel) so the park is visible the instant the RPC returns — the apply cue
    // on `response_rx` can't race ahead of it. Coalescing several targets from one
    // batched call onto a single `Vec` entry folds the burst into ONE respawn.
    let pending_prunes = supervisor.pending_prunes();

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
                .user_agent(crate::useragent::user_agent())
                .timeout(Duration::from_secs(60))
                .build()
                .expect("control http client"),
            api_base_url,
            fetch_token: Arc::new(api_token),
            keys: keys.clone(),
            pending: supervisor.pending_attachments(),
            mirror: mirror.clone(),
            live_sessions: supervisor.live_sessions(),
            pending_prunes: pending_prunes.clone(),
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

    let mut shutdown = sys::ShutdownSignal::new().expect("failed to register OS shutdown handler");

    // Latched on the FIRST CheckpointComplete: by then `mirror.members`
    // reflects everything by_machine had to send, so we can safely sweep
    // `key_<uuid>` files for non-members. Process-lifetime flag.
    let mut key_files_reconciled = false;

    // Per-checkpoint-window staging for hazard tables (chats/projects/messages).
    // Lives across loop iterations — accumulates ops within a checkpoint window
    // and is drained/applied at `CheckpointComplete`. Main()-local, passed by
    // `&mut`; the control task never sees it. Unused this step (threaded only).
    let mut staging = SyncStaging::default();

    // `pending_prunes` (the shared table above) accumulates requests the control
    // task parks while claude runs, each waiting for that `prune-context` call's
    // own `tool_result` frame. The `response_rx` `ToolResult`-cue arm drains a
    // chat's entry once the result has persisted — coalescing a batch into ONE
    // abort→rewrite→respawn.
    // Warm-keep eviction runs when the idle set changes, at the next TTL
    // deadline, and on a periodic tick (idle memory drifts).
    let mut warm_tick = tokio::time::interval(WARM_TICK);
    warm_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut warm_deadline: Option<tokio::time::Instant> = None;
    let mut warm_idle_seen = Vec::new();

    info!("zucchini-spawner ready, waiting for sync + agent responses");

    'main_loop: loop {
        let mut warm_due = false;
        tokio::select! {
            _ = warm_tick.tick() => {
                warm_due = true;
            }
            _ = tokio::time::sleep_until(warm_deadline.unwrap_or_else(tokio::time::Instant::now)), if warm_deadline.is_some() => {
                warm_due = true;
            }
            _ = shutdown.recv() => {
                info!("received OS shutdown signal, shutting down");
                watch::shutdown().await;
                shutdown_and_flush(&mut supervisor, &mut response_rx, &mirror, &write_tx, &heartbeat_cancel, &writer).await;
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
            Some(()) = probe_ready_rx.recv() => {
                // Probes just cached — retry the seed the boot checkpoint
                // deferred (all gates live in `seed_initial_agents_if_pending`).
                // Fires at most once: the probe task drops its sender, so this
                // arm self-disables after the `None`.
                let probe_snap = probe_statuses_cache.get().map(|v| v.as_slice());
                let seed_done = {
                    let mut g = mirror.write().await;
                    seed_initial_agents_if_pending(
                        prod.as_ref().map(|p| p.machine_id),
                        &mut g,
                        probe_snap,
                        &write_tx,
                    )
                    .await
                };
                if seed_done {
                    save_mirror(&state_path, &mirror).await;
                }
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
                        &mut staging,
                        &mut supervisor,
                        &blob_downloader,
                        &keys,
                        x25519_secret.as_ref(),
                        our_pubkey_b64.as_deref(),
                        &write_tx,
                        probe_snap,
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
                    // continuous progress bar across all selected kinds. iOS still
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
                            // historic "all supported kinds" behavior is preserved.
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
                route_agent_response(resp, &pending_prunes, &mirror, &mut supervisor, &write_tx).await;
            }
            Some((out, reply)) = watch_rx.recv() => {
                let delivered =
                    deliver_watch_message(out, &*mirror.read().await, &keys, &mut supervisor, &write_tx).await;
                let _ = reply.send(delivered);
            }
            Some(chat_id) = watched_chat_rx.recv() => {
                publish_watch_change(&chat_id, &*mirror.read().await, &keys, &supervisor, &write_tx).await;
            }
        }

        if warm_due || idle_set(&supervisor) != warm_idle_seen {
            let (pinned, always_on) = {
                let g = mirror.read().await;
                (g.pinned_chats(), g.always_on_chats())
            };
            warm_deadline =
                evict_idle_residents(&mut supervisor, sys::memory, &write_tx, &pinned, &always_on)
                    .await;
            warm_idle_seen = idle_set(&supervisor);
        }
        if warm_due {
            tend_always_on(&mut *mirror.write().await, &mut supervisor, &write_tx).await;
        }
        publish_alive_changes(&mut supervisor, &write_tx).await;

        // If an update is pending and no agent is busy, stop producing
        // heartbeats, shut down idle agents, wait for the writer to drain
        // (including their final writes), then swap the binary and exit —
        // launchd/systemd will respawn the new version.
        if let Some(new_version) = update_pending.as_ref() {
            if !supervisor.any_running() {
                info!(
                    current = env!("CARGO_PKG_VERSION"),
                    new = %new_version,
                    "all agents finished, draining writer before update"
                );
                shutdown_and_flush(
                    &mut supervisor,
                    &mut response_rx,
                    &mirror,
                    &write_tx,
                    &heartbeat_cancel,
                    &writer,
                )
                .await;
                match updater::download_and_replace(new_version).await {
                    Ok(()) => {
                        // Only now: after a failed download the daemon runs
                        // on, and so do its watches.
                        watch::shutdown().await;
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

    if let Some(h) = hermes_socket.as_ref() {
        h.cancel.cancel();
    }
    info!("zucchini-spawner exiting");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{ResidentHandles, ResidentSpawnFn, SessionState, SpawnFn};
    use crate::powersync::SyncEvent;
    use crate::writer::WriteEvent;
    use base64::{engine::general_purpose::STANDARD as B64, Engine as _};
    use serde_json::json;
    use std::sync::{Arc as StdArc, Mutex as StdMutex};

    /// Build a Supervisor whose RESIDENT spawn fn records every session
    /// `SpawnRequest` into `recorded` and returns inert handles: an empty state
    /// (`spawn_resident` queues the first turn into it) and a reader task that
    /// stays alive until its `cancel` token fires (so `is_running` sees a live
    /// session, and `abort_agent` can join it). No writer runs.
    /// This is the resident analogue of the one-shot recorder seam — claude is
    /// resident, so claude-flow tests route here, not through the one-shot fn.
    fn supervisor_with_resident_recorder(
        resp_tx: mpsc::Sender<AgentResponse>,
    ) -> (Supervisor, StdArc<StdMutex<Vec<SpawnRequest>>>) {
        let recorded: StdArc<StdMutex<Vec<SpawnRequest>>> = StdArc::new(StdMutex::new(Vec::new()));
        let rec = recorded.clone();
        let resident_fn: ResidentSpawnFn = StdArc::new(move |req, _tx, cancel, _pending| {
            rec.lock().unwrap().push(req);
            let reader = tokio::spawn(async move { cancel.cancelled().await });
            ResidentHandles {
                wake: StdArc::new(tokio::sync::Notify::new()),
                reader,
                state: StdArc::new(StdMutex::new(SessionState::default())),
            }
        });
        // One-shot fn is unused for claude tests but required by the ctor.
        let one_shot: SpawnFn = StdArc::new(|_req, _tx, _token, _pending| tokio::spawn(async {}));
        let supervisor = Supervisor::with_spawn_fns(resp_tx, one_shot, resident_fn);
        (supervisor, recorded)
    }

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

        // Claude is RESIDENT, so the spawn routes through the resident recorder
        // (captures the session SpawnRequest; returns a live inert session).
        let (mut supervisor, recorded) = supervisor_with_resident_recorder(resp_tx);

        // Shared staging across the window — chats/projects/messages now stage
        // and only apply (spawn) at the trailing CheckpointComplete.
        let mut staging = SyncStaging::default();

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
            &mut staging,
            &mut supervisor,
            &blobs,
            &keys,
            None,
            None,
            &write_tx,
            None,
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
            // Sealed like the clients write it; becomes `claude --name`.
            "title": crypto::encrypt_field_b64(&key, "  cover letters\nwriter  "),
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
            &mut staging,
            &mut supervisor,
            &blobs,
            &keys,
            None,
            None,
            &write_tx,
            None,
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
            &mut staging,
            &mut supervisor,
            &blobs,
            &keys,
            None,
            None,
            &write_tx,
            None,
        )
        .await;

        // 4. CheckpointComplete — applies the staged window (projects → chats →
        // messages) and triggers the deferred spawn.
        handle_sync_event(
            SyncEvent::CheckpointComplete {
                buckets: Default::default(),
            },
            Some(machine_id),
            &mut mirror,
            &mut staging,
            &mut supervisor,
            &blobs,
            &keys,
            None,
            None,
            &write_tx,
            None,
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
        // Decrypted + sanitized chat title: the handle peers address it by.
        assert_eq!(req.session_name.as_deref(), Some("cover letters writer"));

        // RESIDENT path: NO optimistic `ChatRunning(true)` PATCH at message-put
        // time (CLAUDE.md "No optimistic patches of local sync state") — the
        // reader's `RunState` drives `agent_running` instead. The session IS live
        // + busy, though (turn marked in flight).
        let mut saw_running_true = false;
        while let Ok(ev) = write_rx.try_recv() {
            if let WriteEvent::ChatRunning {
                agent_running: true,
                ..
            } = ev
            {
                saw_running_true = true;
            }
        }
        assert!(
            !saw_running_true,
            "resident path must NOT optimistically flip agent_running at put time"
        );
        drop(captured);
        assert!(
            supervisor.is_running(&chat_id),
            "resident session is live and busy after the first user turn"
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
            &keys,
        );

        // Supervisor is only needed because the `Done` arm calls
        // `supervisor.remove(&topic)` and step 5 pre-registers a claude
        // (resident) session via `spawn_agent`. The resident recorder returns a
        // live inert session so `is_running` is observable.
        let (resp_tx, _resp_rx) = mpsc::channel::<AgentResponse>(64);
        let (mut supervisor, _recorded) = supervisor_with_resident_recorder(resp_tx);

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
        let plaintext = String::from_utf8(plaintext_bytes).expect("agent line is UTF-8 plaintext");
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
            WriteEvent::ContextTokens {
                chat_id: cid,
                tokens,
            } => {
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
            WriteEvent::AgentSessionId {
                chat_id: cid,
                session_id,
            } => {
                assert_eq!(cid, &chat_id);
                assert_eq!(session_id, "sess-abc");
            }
            other => panic!("expected AgentSessionId, got {:?}", other),
        }

        // --- 5. Done{has_result=false}: synthesize INTERRUPTED_RESULT line,
        // then clear agent_running (→ false / idle). Supervisor slot is gone.
        // Pre-register a fake handle so we can observe `supervisor.remove`.
        supervisor.spawn_agent(SpawnRequest {
            session_name: None,
            user_system_prompt: Default::default(),
            chat_id: chat_id.clone(),
            prompt: String::new(),
            project_path: None,
            worktree: false,
            agent_session_id: None,
            agent_kind: AgentKind::Claude,
            is_sandboxed: false,
            model: None,
            user_timezone: None,
        });
        assert!(
            supervisor.is_running(&chat_id),
            "sanity: spawn_agent inserted the topic"
        );
        // A session's own `Done` follows its reader closing.
        supervisor.live_sessions().lock().unwrap()[&chat_id]
            .lock()
            .unwrap()
            .closed = true;
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
        assert_eq!(
            events.len(),
            3,
            "INTERRUPTED line + cleared live_tasks + ChatRunning(false) (idle clears the run state)"
        );
        match &events[0] {
            WriteEvent::PutMessage {
                content,
                sender,
                chat_id: cid,
                ..
            } => {
                assert_eq!(*sender, "agent");
                assert_eq!(cid, &chat_id);
                assert_eq!(content, INTERRUPTED_RESULT);
            }
            other => panic!(
                "expected synthesized INTERRUPTED PutMessage, got {:?}",
                other
            ),
        }
        assert!(
            matches!(&events[1], WriteEvent::ChatLiveTasks { tasks: None, .. }),
            "the dead process's armed tasks are cleared: {:?}",
            events[1]
        );
        match &events[2] {
            WriteEvent::ChatRunning {
                chat_id: cid,
                agent_running: false,
                // Process died without a result — not a /stop, push still fires.
                stopped_by_user: false,
            } => {
                assert_eq!(cid, &chat_id);
            }
            other => panic!("expected ChatRunning(false), got {:?}", other),
        }
        assert!(
            !supervisor.is_running(&chat_id),
            "Done arm must remove the topic from supervisor"
        );

        // --- 6. Done{has_result=true}: ChatRunning(false), no synthesized line.
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
            WriteEvent::ChatRunning {
                chat_id: cid,
                agent_running: false,
                stopped_by_user: false,
            } => {
                assert_eq!(cid, &chat_id);
            }
            other => panic!("expected ChatRunning(false), got {:?}", other),
        }
    }

    // ===== machine_users.agents one-shot install-time seeding (migration 0035) =====

    /// Helper: build the bare-bones args set `handle_sync_event` needs that
    /// the agents-seeding tests reuse — keeps each test focused on the
    /// scenario, not the boilerplate.
    fn empty_blob_downloader() -> BlobDownloader {
        BlobDownloader::new(
            "http://test.invalid",
            Box::new(|| Box::pin(async { Err(anyhow::anyhow!("unused in test")) })),
        )
    }

    /// Supervisor with an inert spawn fn. The receiver is returned so a stray
    /// send doesn't error on a dropped channel.
    fn inert_supervisor() -> (Supervisor, mpsc::Receiver<AgentResponse>) {
        let (resp_tx, resp_rx) = mpsc::channel::<AgentResponse>(64);
        let spawn_fn: crate::agent::SpawnFn =
            Arc::new(|_req, _tx, _token, _pending| tokio::spawn(async {}));
        (Supervisor::with_spawn_fn(resp_tx, spawn_fn), resp_rx)
    }

    /// Owner mirror plus the inert collaborators `handle_sync_event` needs.
    struct SeedFixture {
        user_id: Uuid,
        machine_id: Uuid,
        row_id: Uuid,
        mirror: Mirror,
        keys: KeyStore,
        blobs: BlobDownloader,
        write_tx: mpsc::Sender<WriteEvent>,
        write_rx: mpsc::Receiver<WriteEvent>,
        supervisor: Supervisor,
        /// Kept alive so a stray Supervisor send doesn't error.
        _resp_rx: mpsc::Receiver<AgentResponse>,
    }

    impl SeedFixture {
        /// Owner classified and the owner's `machine_users` row latched.
        fn new() -> Self {
            let mut f = Self::without_member();
            assert!(f
                .mirror
                .upsert_member(f.row_id.to_string(), f.user_id, false));
            f
        }

        /// Owner classified but no `machine_users` row yet.
        fn without_member() -> Self {
            let user_id = Uuid::now_v7();
            let mut mirror = Mirror::default();
            mirror.set_user_id(user_id); // owner classification
            let (write_tx, write_rx) = mpsc::channel::<WriteEvent>(64);
            let (supervisor, _resp_rx) = inert_supervisor();
            Self {
                user_id,
                machine_id: Uuid::now_v7(),
                row_id: Uuid::now_v7(),
                mirror,
                keys: KeyStore::with_keys([(user_id, [0u8; 32])]),
                blobs: empty_blob_downloader(),
                write_tx,
                write_rx,
                supervisor,
                _resp_rx,
            }
        }

        /// Drive one `CheckpointComplete` with an empty staging window.
        /// `probe_statuses` stays explicit — some tests pass `None` to pin
        /// which driver seeds.
        async fn checkpoint(&mut self, probe_statuses: Option<&[(AgentKind, (bool, bool))]>) {
            handle_sync_event(
                SyncEvent::CheckpointComplete {
                    buckets: Default::default(),
                },
                Some(self.machine_id),
                &mut self.mirror,
                &mut SyncStaging::default(),
                &mut self.supervisor,
                &self.blobs,
                &self.keys,
                None,
                None,
                &self.write_tx,
                probe_statuses,
            )
            .await;
        }

        /// Drain the writer channel, returning only seed PATCHes.
        fn seed_patches(&mut self) -> Vec<WriteEvent> {
            let mut patches = Vec::new();
            while let Ok(ev) = self.write_rx.try_recv() {
                if matches!(ev, WriteEvent::SetMachineUserAgents { .. }) {
                    patches.push(ev);
                }
            }
            patches
        }
    }

    #[tokio::test]
    async fn fresh_install_with_claude_installed_seeds_frozen_payload_once() {
        let mut f = SeedFixture::without_member();
        assert!(
            !f.mirror.initial_agents_seed_done,
            "a brand-new state must default to seed-not-done"
        );

        // Codex installed but must NOT be seeded (see `DEFAULT_SEED_AGENT_KINDS`).
        let probe_statuses: Vec<(AgentKind, (bool, bool))> = vec![
            (AgentKind::Claude, (true, true)),
            (AgentKind::Cursor, (false, false)),
            (AgentKind::Codex, (true, true)),
        ];

        let mu_row = json!({
            "id": f.row_id.to_string(),
            "user_id": f.user_id.to_string(),
            "machine_id": f.machine_id.to_string(),
            "is_sandboxed": 0,
            "sealed_blob": serde_json::Value::Null,
            "agents": serde_json::Value::Null,
        })
        .to_string();
        handle_sync_event(
            SyncEvent::Put {
                table: "machine_users".into(),
                id: f.row_id.to_string(),
                data: mu_row,
            },
            Some(f.machine_id),
            &mut f.mirror,
            &mut SyncStaging::default(),
            &mut f.supervisor,
            &f.blobs,
            &f.keys,
            None,
            None,
            &f.write_tx,
            Some(&probe_statuses),
        )
        .await;
        f.checkpoint(Some(&probe_statuses)).await;

        let patches = f.seed_patches();
        assert_eq!(
            patches.len(),
            1,
            "expected one seed PATCH, got {}",
            patches.len()
        );
        let WriteEvent::SetMachineUserAgents {
            row_id: emitted_row,
            machine_id: emitted_machine,
            agents_json,
        } = &patches[0]
        else {
            panic!("variant mismatch")
        };
        assert_eq!(*emitted_row, f.row_id);
        assert_eq!(*emitted_machine, f.machine_id);
        // Frozen payload bytes shared with the client-side seeders (see
        // `SeedAgentEntry`).
        assert_eq!(
            agents_json,
            r#"[{"agent_kind":"claude","id":"seed-claude","model":"","name":""}]"#
        );
        assert!(
            f.mirror.initial_agents_seed_done,
            "one-shot drained after the send"
        );

        f.checkpoint(Some(&probe_statuses)).await;
        assert!(
            f.seed_patches().is_empty(),
            "done flag → no second write on the next checkpoint"
        );
    }

    #[tokio::test]
    async fn claude_and_cursor_installed_seed_in_frozen_order() {
        let mut f = SeedFixture::new();
        let probe_statuses: Vec<(AgentKind, (bool, bool))> = vec![
            (AgentKind::Claude, (true, true)),
            (AgentKind::Cursor, (true, true)),
            (AgentKind::Codex, (true, true)),
        ];

        f.checkpoint(Some(&probe_statuses)).await;

        let patches = f.seed_patches();
        assert_eq!(patches.len(), 1, "exactly one seed PATCH");
        let WriteEvent::SetMachineUserAgents { agents_json, .. } = &patches[0] else {
            panic!("variant mismatch")
        };
        assert_eq!(
            agents_json,
            r#"[{"agent_kind":"claude","id":"seed-claude","model":"","name":""},{"agent_kind":"cursor","id":"seed-cursor","model":"","name":""}]"#
        );
    }

    #[tokio::test]
    async fn done_seed_flag_never_writes() {
        // Post-seed restart: every other gate open — the flag alone must block.
        let mut f = SeedFixture::new();
        f.mirror.initial_agents_seed_done = true;
        let probe_statuses: Vec<(AgentKind, (bool, bool))> =
            vec![(AgentKind::Claude, (true, true))];

        f.checkpoint(Some(&probe_statuses)).await;

        assert!(
            f.seed_patches().is_empty(),
            "drained one-shot must never write again"
        );
    }

    #[tokio::test]
    async fn codex_only_install_does_not_seed_but_drains_flag() {
        // Codex never auto-seeds (see `DEFAULT_SEED_AGENT_KINDS`); with claude
        // and cursor absent this also covers the zero-installs shape.
        let mut f = SeedFixture::new();
        let probe_statuses: Vec<(AgentKind, (bool, bool))> = vec![
            (AgentKind::Claude, (false, false)),
            (AgentKind::Cursor, (false, false)),
            (AgentKind::Codex, (true, true)),
        ];

        f.checkpoint(Some(&probe_statuses)).await;

        assert!(
            f.seed_patches().is_empty(),
            "codex-only install must not seed a roster older clients cannot decode"
        );
        assert!(
            f.mirror.initial_agents_seed_done,
            "the one-shot still drains — clients cover the install-CLI-later case"
        );
    }

    /// Checkpoint-before-probes and missing-owner-row both defer WITHOUT
    /// draining the one-shot; the `probe_ready_rx` retry then completes the
    /// seed.
    #[tokio::test]
    async fn deferred_seed_retries_once_probes_complete() {
        let mut f = SeedFixture::without_member();
        let probe_statuses: Vec<(AgentKind, (bool, bool))> = vec![
            (AgentKind::Claude, (true, true)),
            (AgentKind::Cursor, (false, false)),
        ];

        // (1) Probes cached but no owner row → defer.
        let done = seed_initial_agents_if_pending(
            Some(f.machine_id),
            &mut f.mirror,
            Some(&probe_statuses),
            &f.write_tx,
        )
        .await;
        assert!(!done, "no owner row → defer");
        assert!(
            !f.mirror.initial_agents_seed_done,
            "deferral must not drain the one-shot"
        );

        // (2) Owner row lands, probes still in flight → defer.
        assert!(f
            .mirror
            .upsert_member(f.row_id.to_string(), f.user_id, false));
        f.checkpoint(None).await; // probes not in yet
        assert!(
            f.seed_patches().is_empty(),
            "checkpoint before probes complete must defer (no PATCH)"
        );
        assert!(
            !f.mirror.initial_agents_seed_done,
            "probe deferral must not drain the one-shot"
        );

        // (3) Probes complete → the `probe_ready_rx` retry seeds.
        let done = seed_initial_agents_if_pending(
            Some(f.machine_id),
            &mut f.mirror,
            Some(&probe_statuses),
            &f.write_tx,
        )
        .await;
        assert!(done, "the probe driver persists state on `true`");
        let patches = f.seed_patches();
        assert_eq!(
            patches.len(),
            1,
            "retry after probes complete must emit exactly one seed PATCH"
        );
        assert!(
            f.mirror.initial_agents_seed_done,
            "one-shot drained after the send"
        );
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

        // Claude is resident → capture via the resident recorder.
        let (mut supervisor, recorded) = supervisor_with_resident_recorder(resp_tx);

        // Shared staging — apply (spawn) happens at the trailing CheckpointComplete.
        let mut staging = SyncStaging::default();

        // Project PUT.
        handle_sync_event(
            SyncEvent::Put {
                table: "projects".into(),
                id: project_id.clone(),
                data: json!({ "id": project_id, "path": project_path, "name": "t" }).to_string(),
            },
            Some(machine_id),
            &mut mirror,
            &mut staging,
            &mut supervisor,
            &blobs,
            &keys,
            None,
            None,
            &write_tx,
            None,
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
            &mut staging,
            &mut supervisor,
            &blobs,
            &keys,
            None,
            None,
            &write_tx,
            None,
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
            &mut staging,
            &mut supervisor,
            &blobs,
            &keys,
            None,
            None,
            &write_tx,
            None,
        )
        .await;

        // Feed a user message to each chat.
        for (cid, label) in [(chat_with.clone(), "with"), (chat_empty.clone(), "empty")] {
            let envelope_json =
                serde_json::json!({ "text": format!("hi {label}"), "attachments": [] }).to_string();
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
                &mut staging,
                &mut supervisor,
                &blobs,
                &keys,
                None,
                None,
                &write_tx,
                None,
            )
            .await;
        }

        // CheckpointComplete — applies the staged window and fires both spawns.
        handle_sync_event(
            SyncEvent::CheckpointComplete {
                buckets: Default::default(),
            },
            Some(machine_id),
            &mut mirror,
            &mut staging,
            &mut supervisor,
            &blobs,
            &keys,
            None,
            None,
            &write_tx,
            None,
        )
        .await;

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

    /// Within ONE checkpoint window: project PUT, chat PUT, chat REMOVE, chat
    /// PUT (same id). Last-write-wins per id collapses the transient REMOVE, so
    /// at CheckpointComplete the chat is PRESENT (the dropped-`[result]`-frame
    /// PUT→REMOVE→PUT slot-move race, reproduced at the mirror level). A
    /// follow-up user message in a fresh window must then spawn exactly once.
    #[tokio::test]
    async fn chats_put_remove_put_nets_present() {
        let user_id = Uuid::now_v7();
        let machine_id = Uuid::now_v7();
        let chat_id = Uuid::now_v7().to_string();
        let project_id = Uuid::now_v7().to_string();
        let project_path = "/tmp/zucchini-test-project".to_string();

        let mut mirror = Mirror::default();
        mirror.set_user_id(user_id);
        let key_bytes = [0u8; 32];
        let keys = KeyStore::with_keys([(user_id, key_bytes)]);
        let key = keys.get(&user_id).expect("seeded key");
        let blobs = empty_blob_downloader();
        let (write_tx, _write_rx) = mpsc::channel::<WriteEvent>(64);
        let (resp_tx, _resp_rx) = mpsc::channel::<AgentResponse>(64);
        let (mut supervisor, recorded) = supervisor_with_resident_recorder(resp_tx);
        let mut staging = SyncStaging::default();

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

        // Window 1: project PUT, chat PUT, chat REMOVE, chat PUT (same id).
        for ev in [
            SyncEvent::Put {
                table: "projects".into(),
                id: project_id.clone(),
                data: json!({ "id": project_id, "path": project_path, "name": "t" }).to_string(),
            },
            SyncEvent::Put {
                table: "chats".into(),
                id: chat_id.clone(),
                data: chat_row.clone(),
            },
            SyncEvent::Remove {
                table: "chats".into(),
                id: chat_id.clone(),
            },
            SyncEvent::Put {
                table: "chats".into(),
                id: chat_id.clone(),
                data: chat_row.clone(),
            },
            SyncEvent::CheckpointComplete {
                buckets: Default::default(),
            },
        ] {
            handle_sync_event(
                ev,
                Some(machine_id),
                &mut mirror,
                &mut staging,
                &mut supervisor,
                &blobs,
                &keys,
                None,
                None,
                &write_tx,
                None,
            )
            .await;
        }

        assert!(
            mirror.chats.contains_key(&chat_id),
            "transient REMOVE inside a window must not evict the re-PUT chat"
        );

        // Window 2: a user message must spawn exactly one agent.
        let envelope_json =
            serde_json::json!({ "text": "hi after slot-move", "attachments": [] }).to_string();
        let body_b64 = B64.encode(crypto::encrypt(&key, envelope_json.as_bytes()));
        let msg_id = Uuid::now_v7().to_string();
        for ev in [
            SyncEvent::Put {
                table: "messages".into(),
                id: msg_id.clone(),
                data: json!({
                    "id": msg_id,
                    "chat_id": chat_id,
                    "sender": "user",
                    "seq": 1,
                    "body": body_b64,
                    "imported": false,
                })
                .to_string(),
            },
            SyncEvent::CheckpointComplete {
                buckets: Default::default(),
            },
        ] {
            handle_sync_event(
                ev,
                Some(machine_id),
                &mut mirror,
                &mut staging,
                &mut supervisor,
                &blobs,
                &keys,
                None,
                None,
                &write_tx,
                None,
            )
            .await;
        }

        let captured = recorded.lock().unwrap();
        assert_eq!(
            captured.len(),
            1,
            "exactly one spawn after the survived chat takes a message"
        );
        assert_eq!(captured[0].chat_id, chat_id);
    }

    /// A genuine delete still applies: chat PUT + CheckpointComplete (present),
    /// then chat REMOVE with NO re-PUT + CheckpointComplete → the chat is gone.
    #[tokio::test]
    async fn chats_put_remove_nets_absent() {
        let user_id = Uuid::now_v7();
        let machine_id = Uuid::now_v7();
        let chat_id = Uuid::now_v7().to_string();
        let project_id = Uuid::now_v7().to_string();
        let project_path = "/tmp/zucchini-test-project".to_string();

        let mut mirror = Mirror::default();
        mirror.set_user_id(user_id);
        let keys = KeyStore::with_keys([(user_id, [0u8; 32])]);
        let blobs = empty_blob_downloader();
        let (write_tx, _write_rx) = mpsc::channel::<WriteEvent>(64);
        let (resp_tx, _resp_rx) = mpsc::channel::<AgentResponse>(64);
        let (mut supervisor, _recorded) = supervisor_with_resident_recorder(resp_tx);
        let mut staging = SyncStaging::default();

        // Window 1: project + chat PUT → chat present.
        for ev in [
            SyncEvent::Put {
                table: "projects".into(),
                id: project_id.clone(),
                data: json!({ "id": project_id, "path": project_path, "name": "t" }).to_string(),
            },
            SyncEvent::Put {
                table: "chats".into(),
                id: chat_id.clone(),
                data: json!({
                    "id": chat_id,
                    "project_id": project_id,
                    "user_id": user_id.to_string(),
                    "last_seq": 0,
                    "agent_session_id": serde_json::Value::Null,
                    "agent_kind": "claude",
                    "worktree": false,
                })
                .to_string(),
            },
            SyncEvent::CheckpointComplete {
                buckets: Default::default(),
            },
        ] {
            handle_sync_event(
                ev,
                Some(machine_id),
                &mut mirror,
                &mut staging,
                &mut supervisor,
                &blobs,
                &keys,
                None,
                None,
                &write_tx,
                None,
            )
            .await;
        }
        assert!(
            mirror.chats.contains_key(&chat_id),
            "chat present after its first checkpoint"
        );

        // Window 2: REMOVE with no re-PUT → genuine delete applies at checkpoint.
        for ev in [
            SyncEvent::Remove {
                table: "chats".into(),
                id: chat_id.clone(),
            },
            SyncEvent::CheckpointComplete {
                buckets: Default::default(),
            },
        ] {
            handle_sync_event(
                ev,
                Some(machine_id),
                &mut mirror,
                &mut staging,
                &mut supervisor,
                &blobs,
                &keys,
                None,
                None,
                &write_tx,
                None,
            )
            .await;
        }
        assert!(
            !mirror.chats.contains_key(&chat_id),
            "a real REMOVE with no re-PUT evicts the chat at checkpoint"
        );
    }

    /// Apply order is projects → chats → messages regardless of delivery order
    /// within the window: feed the user message PUT BEFORE the chat PUT, then
    /// the project PUT, then CheckpointComplete → exactly one spawn.
    #[tokio::test]
    async fn new_chat_plus_first_message_same_window_spawns() {
        let user_id = Uuid::now_v7();
        let machine_id = Uuid::now_v7();
        let chat_id = Uuid::now_v7().to_string();
        let project_id = Uuid::now_v7().to_string();
        let project_path = "/tmp/zucchini-test-project".to_string();

        let mut mirror = Mirror::default();
        mirror.set_user_id(user_id);
        let key_bytes = [0u8; 32];
        let keys = KeyStore::with_keys([(user_id, key_bytes)]);
        let key = keys.get(&user_id).expect("seeded key");
        let blobs = empty_blob_downloader();
        let (write_tx, _write_rx) = mpsc::channel::<WriteEvent>(64);
        let (resp_tx, _resp_rx) = mpsc::channel::<AgentResponse>(64);
        let (mut supervisor, recorded) = supervisor_with_resident_recorder(resp_tx);
        let mut staging = SyncStaging::default();

        let envelope_json =
            serde_json::json!({ "text": "first turn", "attachments": [] }).to_string();
        let body_b64 = B64.encode(crypto::encrypt(&key, envelope_json.as_bytes()));
        let msg_id = Uuid::now_v7().to_string();

        // Deliberately out of dependency order: message, then chat, then project.
        for ev in [
            SyncEvent::Put {
                table: "messages".into(),
                id: msg_id.clone(),
                data: json!({
                    "id": msg_id,
                    "chat_id": chat_id,
                    "sender": "user",
                    "seq": 1,
                    "body": body_b64,
                    "imported": false,
                })
                .to_string(),
            },
            SyncEvent::Put {
                table: "chats".into(),
                id: chat_id.clone(),
                data: json!({
                    "id": chat_id,
                    "project_id": project_id,
                    "user_id": user_id.to_string(),
                    "last_seq": 0,
                    "agent_session_id": serde_json::Value::Null,
                    "agent_kind": "claude",
                    "worktree": false,
                })
                .to_string(),
            },
            SyncEvent::Put {
                table: "projects".into(),
                id: project_id.clone(),
                data: json!({ "id": project_id, "path": project_path, "name": "t" }).to_string(),
            },
            SyncEvent::CheckpointComplete {
                buckets: Default::default(),
            },
        ] {
            handle_sync_event(
                ev,
                Some(machine_id),
                &mut mirror,
                &mut staging,
                &mut supervisor,
                &blobs,
                &keys,
                None,
                None,
                &write_tx,
                None,
            )
            .await;
        }

        let captured = recorded.lock().unwrap();
        assert_eq!(
            captured.len(),
            1,
            "apply orders projects→chats→messages regardless of in-window delivery order"
        );
        assert_eq!(captured[0].chat_id, chat_id);
        assert_eq!(
            captured[0].project_path.as_deref(),
            Some(project_path.as_str())
        );
    }

    /// Guards the merge-hack retirement: a locally-harvested `agent_session_id`
    /// must survive a later checkpoint that re-streams the row with a stale NULL
    /// session id, and a fast-followup user message must `--resume` it.
    #[tokio::test]
    async fn fast_followup_resume_survives_checkpoint_window() {
        let user_id = Uuid::now_v7();
        let machine_id = Uuid::now_v7();
        let chat_id = Uuid::now_v7().to_string();
        let project_id = Uuid::now_v7().to_string();
        let project_path = "/tmp/zucchini-test-project".to_string();

        let mut mirror = Mirror::default();
        mirror.set_user_id(user_id);
        let key_bytes = [0u8; 32];
        let keys = KeyStore::with_keys([(user_id, key_bytes)]);
        let key = keys.get(&user_id).expect("seeded key");
        let blobs = empty_blob_downloader();
        let (write_tx, _write_rx) = mpsc::channel::<WriteEvent>(64);
        let (resp_tx, _resp_rx) = mpsc::channel::<AgentResponse>(64);
        let (mut supervisor, recorded) = supervisor_with_resident_recorder(resp_tx);
        let mut staging = SyncStaging::default();

        // The chat row always carries agent_session_id=NULL (PowerSync never
        // round-trips the locally-harvested id back into the streamed row here).
        let null_chat_row = json!({
            "id": chat_id,
            "project_id": project_id,
            "user_id": user_id.to_string(),
            "last_seq": 0,
            "agent_session_id": serde_json::Value::Null,
            "agent_kind": "claude",
            "worktree": false,
        })
        .to_string();

        // Window 1: project PUT + chat PUT (NULL session id).
        for ev in [
            SyncEvent::Put {
                table: "projects".into(),
                id: project_id.clone(),
                data: json!({ "id": project_id, "path": project_path, "name": "t" }).to_string(),
            },
            SyncEvent::Put {
                table: "chats".into(),
                id: chat_id.clone(),
                data: null_chat_row.clone(),
            },
            SyncEvent::CheckpointComplete {
                buckets: Default::default(),
            },
        ] {
            handle_sync_event(
                ev,
                Some(machine_id),
                &mut mirror,
                &mut staging,
                &mut supervisor,
                &blobs,
                &keys,
                None,
                None,
                &write_tx,
                None,
            )
            .await;
        }

        // Mimic the harvest (main.rs SessionIdHarvested → set_agent_session_id).
        mirror.set_agent_session_id(&chat_id, "sess-1".into());

        // Window 2: a stale re-stream of the same row, STILL NULL.
        for ev in [
            SyncEvent::Put {
                table: "chats".into(),
                id: chat_id.clone(),
                data: null_chat_row.clone(),
            },
            SyncEvent::CheckpointComplete {
                buckets: Default::default(),
            },
        ] {
            handle_sync_event(
                ev,
                Some(machine_id),
                &mut mirror,
                &mut staging,
                &mut supervisor,
                &blobs,
                &keys,
                None,
                None,
                &write_tx,
                None,
            )
            .await;
        }

        assert_eq!(
            mirror
                .chats
                .get(&chat_id)
                .and_then(|c| c.agent_session_id.clone()),
            Some("sess-1".to_string()),
            "apply-phase restore must preserve the harvested id across a NULL re-stream"
        );

        // Window 3: a fast-followup user message must resume sess-1.
        let envelope_json =
            serde_json::json!({ "text": "follow up", "attachments": [] }).to_string();
        let body_b64 = B64.encode(crypto::encrypt(&key, envelope_json.as_bytes()));
        let msg_id = Uuid::now_v7().to_string();
        for ev in [
            SyncEvent::Put {
                table: "messages".into(),
                id: msg_id.clone(),
                data: json!({
                    "id": msg_id,
                    "chat_id": chat_id,
                    "sender": "user",
                    "seq": 1,
                    "body": body_b64,
                    "imported": false,
                })
                .to_string(),
            },
            SyncEvent::CheckpointComplete {
                buckets: Default::default(),
            },
        ] {
            handle_sync_event(
                ev,
                Some(machine_id),
                &mut mirror,
                &mut staging,
                &mut supervisor,
                &blobs,
                &keys,
                None,
                None,
                &write_tx,
                None,
            )
            .await;
        }

        let captured = recorded.lock().unwrap();
        assert_eq!(captured.len(), 1, "fast-followup spawns exactly once");
        assert_eq!(
            captured[0].agent_session_id.as_deref(),
            Some("sess-1"),
            "fast-followup --resume preserved the harvested session id"
        );
    }

    fn sv(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn parse_prune_args_single_triple_is_one_item() {
        let (force, items) = super::parse_prune_args(&sv(&[
            "--tool-name",
            "Read",
            "--args",
            "a.ts",
            "--summary",
            "kept x",
        ]))
        .expect("valid");
        assert!(!force, "force defaults off");
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].tool_name, "Read");
        assert_eq!(items[0].needle, "a.ts");
        assert_eq!(items[0].reason, "kept x");
    }

    #[test]
    fn parse_prune_args_summary_terminates_each_target() {
        // Three targets in one call; a legacy `--chat-id` is ignored wherever it sits.
        let (force, items) = super::parse_prune_args(&sv(&[
            "--tool-name",
            "Read",
            "--args",
            "a.ts",
            "--summary",
            "s1", //
            "--args",
            "TODO",
            "--summary",
            "s2", // no --tool-name → any-tool selector
            "--chat-id",
            "c1",      //
            "--force", // call-level, can sit anywhere
            "--tool-name",
            "Grep",
            "--args",
            "",
            "--summary",
            "s3", // no-args selector
        ]))
        .expect("valid");
        assert!(force, "--force parsed as call-level boolean");
        assert_eq!(items.len(), 3);
        assert_eq!(
            (items[0].tool_name.as_str(), items[0].needle.as_str()),
            ("Read", "a.ts")
        );
        assert_eq!(
            (items[1].tool_name.as_str(), items[1].needle.as_str()),
            ("", "TODO")
        );
        assert_eq!(
            (items[2].tool_name.as_str(), items[2].needle.as_str()),
            ("Grep", "")
        );
        assert_eq!(items[2].reason, "s3");
    }

    #[test]
    fn parse_prune_args_accepts_eq_form_and_reason_alias() {
        let (_, items) = super::parse_prune_args(&sv(&[
            "--tool-name=Read",
            "--args=a.ts",
            "--reason=legacy", //
            "--args=b.ts",
            "--summary=new",
        ]))
        .expect("valid");
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].reason, "legacy");
        assert_eq!(items[1].needle, "b.ts");
    }

    #[test]
    fn parse_prune_args_rejects_malformed() {
        // --summary with no preceding --args for this target.
        assert!(super::parse_prune_args(&sv(&["--tool-name", "Read", "--summary", "s"])).is_err());
        // Dangling target not closed by a --summary.
        assert!(super::parse_prune_args(&sv(&[
            "--args",
            "a.ts",
            "--summary",
            "s",
            "--args",
            "b.ts"
        ]))
        .is_err());
        // Empty batch (only call-level flags).
        assert!(super::parse_prune_args(&sv(&["--chat-id", "c1"])).is_err());
        // Unknown flag / help.
        assert!(super::parse_prune_args(&sv(&["--bogus"])).is_err());
        assert!(super::parse_prune_args(&sv(&["--help"])).is_err());
        // Flag expecting a value at end of args.
        assert!(super::parse_prune_args(&sv(&["--args", "a.ts", "--summary"])).is_err());
    }

    /// Regression for chat 1398d148: a single turn fires several `prune-context`
    /// calls; they must coalesce into ONE abort→rewrite→respawn. Before the fix
    /// each request respawned independently, the next request's `abort` SIGTERM'd
    /// the fresh respawn (thrash), and the storm's tail raced the mirror into
    /// `resolve → None`, leaving the chat idle ("agent failed to continue").
    /// Here three distinct prunes are applied as one coalesced batch: assert all
    /// three transcript outputs get blanked but the agent respawns exactly once.
    #[tokio::test]
    async fn prune_burst_coalesces_into_single_respawn() {
        use crate::prune::test_util::{read_lines, write_jsonl};

        let user_id = Uuid::now_v7();
        let machine_id = Uuid::now_v7();
        let project_id = Uuid::now_v7().to_string();
        let chat_id = Uuid::now_v7().to_string();
        let session_id = Uuid::now_v7().to_string();

        // Three distinct Read tool_use/tool_result pairs in one transcript.
        let transcript = write_jsonl(&[
            r#"{"type":"assistant","message":{"content":[{"type":"tool_use","id":"t1","name":"Read","input":{"file_path":"junk1.rs"}}]},"uuid":"a1","parentUuid":null}"#,
            r#"{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"t1","content":"body one"}]},"uuid":"u1","parentUuid":"a1"}"#,
            r#"{"type":"assistant","message":{"content":[{"type":"tool_use","id":"t2","name":"Read","input":{"file_path":"junk2.rs"}}]},"uuid":"a2","parentUuid":"u1"}"#,
            r#"{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"t2","content":"body two"}]},"uuid":"u2","parentUuid":"a2"}"#,
            r#"{"type":"assistant","message":{"content":[{"type":"tool_use","id":"t3","name":"Read","input":{"file_path":"junk3.rs"}}]},"uuid":"a3","parentUuid":"u2"}"#,
            r#"{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"t3","content":"body three"}]},"uuid":"u3","parentUuid":"a3"}"#,
        ]);
        let jsonl_path = transcript.path().to_path_buf();

        let mut mirror = Mirror::default();
        mirror.set_user_id(user_id);
        let keys = KeyStore::with_keys([(user_id, [0u8; 32])]);
        let blobs = empty_blob_downloader();
        let (write_tx, _write_rx) = mpsc::channel::<WriteEvent>(64);
        let (resp_tx, _resp_rx) = mpsc::channel::<AgentResponse>(64);

        // Claude is resident; `apply_prune_group`'s respawn routes through the
        // resident recorder. The pre-prune abort_agent is a no-op (no live
        // session in this unit test), then the respawn records exactly one.
        let (mut supervisor, recorded) = supervisor_with_resident_recorder(resp_tx);

        // Shared staging; the trailing CheckpointComplete (before the SharedMirror
        // wrap) applies the project + chat into the mirror.
        let mut staging = SyncStaging::default();

        handle_sync_event(
            SyncEvent::Put {
                table: "projects".into(),
                id: project_id.clone(),
                data:
                    json!({ "id": project_id, "path": "/tmp/zucchini-test-project", "name": "t" })
                        .to_string(),
            },
            Some(machine_id),
            &mut mirror,
            &mut staging,
            &mut supervisor,
            &blobs,
            &keys,
            None,
            None,
            &write_tx,
            None,
        )
        .await;
        handle_sync_event(
            SyncEvent::Put {
                table: "chats".into(),
                id: chat_id.clone(),
                data: json!({
                    "id": chat_id,
                    "project_id": project_id,
                    "user_id": user_id.to_string(),
                    "last_seq": 0,
                    "agent_session_id": session_id,
                    "agent_kind": "claude",
                    "worktree": false,
                    "model": "",
                })
                .to_string(),
            },
            Some(machine_id),
            &mut mirror,
            &mut staging,
            &mut supervisor,
            &blobs,
            &keys,
            None,
            None,
            &write_tx,
            None,
        )
        .await;

        // Apply the staged project + chat before wrapping in SharedMirror.
        handle_sync_event(
            SyncEvent::CheckpointComplete {
                buckets: Default::default(),
            },
            Some(machine_id),
            &mut mirror,
            &mut staging,
            &mut supervisor,
            &blobs,
            &keys,
            None,
            None,
            &write_tx,
            None,
        )
        .await;

        let shared: SharedMirror = Arc::new(tokio::sync::RwLock::new(mirror));

        // The three prunes from one turn coalesce onto a single `pending_prunes`
        // `Vec` entry in the main loop; `apply_prune_group` is what the loop calls
        // when claude emits the `prune-context` call's `tool_result` frame, so
        // exercise it with the coalesced batch directly.
        let mk = |needle: &str, reason: &str| prune::PruneRequest {
            jsonl_path: jsonl_path.clone(),
            agent_kind: crate::adapter::AgentKind::Claude,
            tool_name: "Read".into(),
            needle: needle.into(),
            reason: reason.into(),
        };
        let reqs = vec![
            mk("junk1.rs", "got first"),
            mk("junk2.rs", "got second"),
            mk("junk3.rs", "got third"),
        ];

        apply_prune_group(&chat_id, reqs, &shared, &mut supervisor, &write_tx).await;

        // Exactly ONE respawn for the whole burst — not three.
        let captured = recorded.lock().unwrap();
        assert_eq!(
            captured.len(),
            1,
            "burst of 3 prunes must coalesce into ONE respawn"
        );
        assert_eq!(captured[0].chat_id, chat_id);
        assert_eq!(
            captured[0].agent_session_id.as_deref(),
            Some(session_id.as_str()),
            "respawn resumes the harvested session"
        );
        assert!(
            captured[0].prompt.contains("Context pruning succeeded"),
            "respawn prompt signals success, got: {}",
            captured[0].prompt
        );

        // All three tool_result outputs blanked — every queued prune applied, not
        // just the last (stats summed across the coalesced batch).
        let lines = read_lines(&jsonl_path);
        let pruned = lines
            .iter()
            .filter(|l| l["message"]["content"][0]["content"] == "[pruned]")
            .count();
        assert_eq!(
            pruned, 3,
            "every queued prune was applied, not just the last"
        );
    }

    /// A respawn after the previous process is gone resumes the harvested
    /// session id. Drives `Supervisor::spawn_agent` directly so the assertion
    /// is on the recorded session `SpawnRequest`s.
    #[tokio::test]
    async fn resident_respawn_resumes_the_harvested_session() {
        let chat_id = Uuid::now_v7().to_string();
        let (resp_tx, _resp_rx) = mpsc::channel::<AgentResponse>(64);
        let (mut supervisor, recorded) = supervisor_with_resident_recorder(resp_tx);

        let mk = |prompt: &str, model: Option<&str>, sid: Option<&str>| SpawnRequest {
            session_name: None,
            user_system_prompt: Default::default(),
            chat_id: chat_id.clone(),
            prompt: prompt.to_string(),
            project_path: Some("/tmp/p".to_string()),
            worktree: false,
            agent_session_id: sid.map(str::to_string),
            agent_kind: AgentKind::Claude,
            is_sandboxed: false,
            model: model.map(str::to_string),
            user_timezone: None,
        };

        supervisor.spawn_agent(mk("first", Some("opus"), None));
        assert_eq!(recorded.lock().unwrap().len(), 1, "first turn spawns once");
        assert!(supervisor.is_running(&chat_id), "session live + busy");

        supervisor.abort_agent(&chat_id).await;
        supervisor.spawn_agent(mk("second", Some("opus"), Some("sid-1")));
        let captured = recorded.lock().unwrap();
        assert_eq!(captured.len(), 2);
        assert_eq!(
            captured[1].agent_session_id.as_deref(),
            Some("sid-1"),
            "respawn resumes the harvested session id"
        );
    }

    /// One claude chat on an owner machine, driven window by window through
    /// `handle_sync_event` against the resident recorder.
    struct ResidentChat {
        user_id: Uuid,
        chat_id: String,
        project_id: String,
        mirror: Mirror,
        keys: KeyStore,
        blobs: BlobDownloader,
        write_tx: mpsc::Sender<WriteEvent>,
        write_rx: mpsc::Receiver<WriteEvent>,
        supervisor: Supervisor,
        recorded: StdArc<StdMutex<Vec<SpawnRequest>>>,
        /// What the chat's reader sends on.
        resp_tx: mpsc::Sender<AgentResponse>,
        resp_rx: mpsc::Receiver<AgentResponse>,
        /// The real claude line parser, for `reader_raw`.
        adapter: Box<dyn adapter::AgentAdapter>,
    }

    impl ResidentChat {
        fn new() -> Self {
            let user_id = Uuid::now_v7();
            let mut mirror = Mirror::default();
            mirror.set_user_id(user_id);
            let (write_tx, write_rx) = mpsc::channel::<WriteEvent>(64);
            let (resp_tx, resp_rx) = mpsc::channel::<AgentResponse>(64);
            let (supervisor, recorded) = supervisor_with_resident_recorder(resp_tx.clone());
            Self {
                user_id,
                chat_id: Uuid::now_v7().to_string(),
                project_id: Uuid::now_v7().to_string(),
                mirror,
                keys: KeyStore::with_keys([(user_id, [0u8; 32])]),
                blobs: empty_blob_downloader(),
                write_tx,
                write_rx,
                supervisor,
                recorded,
                resp_tx,
                resp_rx,
                adapter: AgentKind::Claude.make_adapter(),
            }
        }

        fn project(&self) -> SyncEvent {
            SyncEvent::Put {
                table: "projects".into(),
                id: self.project_id.clone(),
                data: json!({ "id": self.project_id, "path": "/tmp/p", "name": "p" }).to_string(),
            }
        }

        fn chat(&self, last_seq: i64, model: &str) -> SyncEvent {
            SyncEvent::Put {
                table: "chats".into(),
                id: self.chat_id.clone(),
                data: json!({
                    "id": self.chat_id,
                    "project_id": self.project_id,
                    "user_id": self.user_id.to_string(),
                    "last_seq": last_seq,
                    "agent_session_id": serde_json::Value::Null,
                    "agent_kind": "claude",
                    "worktree": false,
                    "model": model,
                })
                .to_string(),
            }
        }

        /// `chat`, plus the sealed `title` the clients write — which becomes
        /// `claude --name` and, on a change, the internal `/rename` turn.
        fn titled_chat(&self, last_seq: i64, title: &str) -> SyncEvent {
            let key = self.keys.get(&self.user_id).expect("seeded key");
            let title = crypto::encrypt_field_b64(&key, title);
            put_with(self.chat(last_seq, ""), json!({ "title": title }))
        }

        fn message(&self, seq: i64, text: &str) -> SyncEvent {
            let key = self.keys.get(&self.user_id).expect("seeded key");
            let envelope = json!({ "text": text, "attachments": [] }).to_string();
            let id = Uuid::now_v7().to_string();
            SyncEvent::Put {
                table: "messages".into(),
                id: id.clone(),
                data: json!({
                    "id": id,
                    "chat_id": self.chat_id,
                    "sender": "user",
                    "seq": seq,
                    "body": B64.encode(crypto::encrypt(&key, envelope.as_bytes())),
                    "imported": false,
                })
                .to_string(),
            }
        }

        /// Apply `events` as one checkpoint window.
        async fn window(&mut self, events: Vec<SyncEvent>) {
            let mut staging = SyncStaging::default();
            let checkpoint = SyncEvent::CheckpointComplete {
                buckets: Default::default(),
            };
            for event in events.into_iter().chain([checkpoint]) {
                handle_sync_event(
                    event,
                    Some(Uuid::now_v7()),
                    &mut self.mirror,
                    &mut staging,
                    &mut self.supervisor,
                    &self.blobs,
                    &self.keys,
                    None,
                    None,
                    &self.write_tx,
                    None,
                )
                .await;
            }
            publish_alive_changes(&mut self.supervisor, &self.write_tx).await;
        }

        async fn respond(&mut self, resp: AgentResponse) {
            handle_agent_response(resp, &mut self.mirror, &self.write_tx, &mut self.supervisor)
                .await;
            publish_alive_changes(&mut self.supervisor, &self.write_tx).await;
        }

        /// The live session's FSM, as its reader would mutate it.
        fn session(&self) -> StdArc<StdMutex<SessionState>> {
            self.supervisor.live_sessions().lock().unwrap()[&self.chat_id].clone()
        }

        /// Drive the events of ONE stdout line exactly as the resident reader
        /// would: reduce each in order and publish what it leaves standing,
        /// with the internal `/rename` suppression window latched for the whole
        /// line (its `result` closes the window, but that same line's frame and
        /// usage are still the rename's).
        async fn reader_line(&mut self, events: Vec<adapter::AgentEvent>) {
            let mut renaming = crate::agent::InternalLine::default();
            let mut turn_edges = Vec::new();
            for ev in events {
                let session = self.session();
                let (swallow, live_tasks) = {
                    let mut st = session.lock().unwrap();
                    let swallow = renaming.swallows(st.internal_in_flight(), &ev);
                    crate::agent::reduce(&mut st, &ev);
                    turn_edges.extend(st.observe_turn_running());
                    (swallow, st.observe_live_tasks())
                };
                let topic = self.chat_id.clone();
                match ev {
                    _ if swallow => {}
                    adapter::AgentEvent::Frame(content) => {
                        self.respond(AgentResponse::Line { topic, content }).await
                    }
                    adapter::AgentEvent::ContextTokens(tokens) => {
                        self.respond(AgentResponse::ContextTokens { topic, tokens })
                            .await
                    }
                    _ => {}
                }
                if let Some(tasks) = live_tasks {
                    self.respond(AgentResponse::LiveTasks {
                        topic: self.chat_id.clone(),
                        session,
                        tasks,
                    })
                    .await;
                }
            }
            for running in turn_edges {
                self.respond(AgentResponse::TurnRunning {
                    topic: self.chat_id.clone(),
                    session: self.session(),
                    running,
                })
                .await;
            }
        }

        /// The same, for a raw claude stdout line parsed by the real adapter —
        /// so the events and their ORDER are the production ones.
        async fn reader_raw(&mut self, line: &str) {
            let events: Vec<_> = self
                .adapter
                .handle_line(line.to_string())
                .into_iter()
                .collect();
            self.reader_line(events).await;
        }

        /// The first turn was consumed and is in flight.
        fn consume_first_turn(&self) {
            let s = self.session();
            let mut st = s.lock().unwrap();
            st.unconsumed.clear();
            st.turn_in_flight = true;
        }

        /// Moves the mirror out, for the main-loop helpers that take it shared.
        fn shared_mirror(&mut self) -> SharedMirror {
            Arc::new(tokio::sync::RwLock::new(std::mem::take(&mut self.mirror)))
        }

        fn spawned_prompts(&self) -> Vec<String> {
            self.recorded
                .lock()
                .unwrap()
                .iter()
                .map(|r| r.prompt.clone())
                .collect()
        }

        fn unconsumed(&self) -> Vec<String> {
            self.supervisor.unconsumed_prompts(&self.chat_id)
        }

        fn writes(&mut self) -> Vec<WriteEvent> {
            let mut writes = Vec::new();
            while let Ok(w) = self.write_rx.try_recv() {
                writes.push(w);
            }
            writes
        }

        fn message_rows(&mut self) -> Vec<serde_json::Value> {
            self.writes()
                .into_iter()
                .filter_map(|w| match w {
                    WriteEvent::PutMessage { content, .. } => {
                        Some(serde_json::from_str(&content).expect("message row is JSON"))
                    }
                    _ => None,
                })
                .collect()
        }
    }

    fn is_interrupted(w: &WriteEvent) -> bool {
        matches!(w, WriteEvent::PutMessage { content, .. } if content == INTERRUPTED_RESULT)
    }

    fn remove(table: &str, id: &str) -> SyncEvent {
        SyncEvent::Remove {
            table: table.into(),
            id: id.into(),
        }
    }

    /// Makes `c.user_id` an unsandboxed invitee on someone else's machine,
    /// with its `machine_users` row `row_id` already unsealed.
    fn invite(c: &mut ResidentChat, row_id: &str) {
        c.mirror.set_user_id(Uuid::now_v7());
        c.mirror.upsert_member(row_id.into(), c.user_id, false);
        c.mirror.record_sealed_blob(&c.user_id, "blob");
    }

    /// `c.chat(last_seq, "")` with `fields` overriding its columns.
    fn chat_with(c: &ResidentChat, last_seq: i64, fields: serde_json::Value) -> SyncEvent {
        put_with(c.chat(last_seq, ""), fields)
    }

    /// The row PUT `put`, with `fields` set on it.
    fn put_with(put: SyncEvent, fields: serde_json::Value) -> SyncEvent {
        let SyncEvent::Put { table, id, data } = put else {
            unreachable!("a row PUT")
        };
        let mut row: serde_json::Value = serde_json::from_str(&data).expect("row");
        for (k, v) in fields.as_object().expect("fields object") {
            row[k] = v.clone();
        }
        SyncEvent::Put {
            table,
            id,
            data: row.to_string(),
        }
    }

    fn always_on_chat(c: &ResidentChat, last_seq: i64, always_on: bool) -> SyncEvent {
        chat_with(
            c,
            last_seq,
            json!({ "always_on": always_on as i64, "agent_session_id": "sid" }),
        )
    }

    #[tokio::test]
    async fn a_turn_ending_past_the_threshold_queues_a_compaction() {
        let mut c = ResidentChat::new();
        let chat = chat_with(&c, 1, json!({ "auto_compact_tokens": 100_000 }));
        c.window(vec![c.project(), chat, c.message(1, "first")])
            .await;
        c.consume_first_turn();
        c.reader_line(vec![
            adapter::AgentEvent::ContextTokens(120_000),
            adapter::AgentEvent::Result {
                origin_is_task: false,
            },
        ])
        .await;
        c.respond(AgentResponse::TurnEnded {
            topic: c.chat_id.clone(),
        })
        .await;
        let queued = c.unconsumed();
        assert_eq!(queued.len(), 1);
        assert!(queued[0].starts_with("/compact "));
        assert!(c.supervisor.is_running(&c.chat_id), "no idle edge");
    }

    #[tokio::test]
    async fn a_threshold_set_below_an_idle_residents_context_compacts_at_once() {
        let mut c = ResidentChat::new();
        c.window(vec![c.project(), c.chat(1, ""), c.message(1, "first")])
            .await;
        c.consume_first_turn();
        c.reader_line(vec![
            adapter::AgentEvent::ContextTokens(120_000),
            adapter::AgentEvent::Result {
                origin_is_task: false,
            },
        ])
        .await;
        c.window(vec![chat_with(
            &c,
            2,
            json!({ "auto_compact_tokens": 200_000 }),
        )])
        .await;
        assert!(c.unconsumed().is_empty(), "below the threshold");

        c.window(vec![
            chat_with(&c, 2, json!({ "auto_compact_tokens": 100_000 })),
            c.message(2, "second"),
        ])
        .await;
        assert_eq!(c.unconsumed().len(), 1, "the user's turn goes first");
        c.consume_first_turn();
        c.reader_line(vec![adapter::AgentEvent::Result {
            origin_is_task: false,
        }])
        .await;

        c.window(vec![chat_with(
            &c,
            3,
            json!({ "auto_compact_tokens": 110_000 }),
        )])
        .await;
        let queued = c.unconsumed();
        assert_eq!(queued.len(), 1);
        assert!(queued[0].starts_with("/compact "));
        c.window(vec![chat_with(
            &c,
            3,
            json!({ "auto_compact_tokens": 90_000 }),
        )])
        .await;
        assert_eq!(c.unconsumed().len(), 1, "already compacting");
    }

    #[tokio::test]
    async fn a_watch_message_is_a_system_row_and_a_turn() {
        let mut c = ResidentChat::new();
        c.window(vec![c.project(), c.chat(0, "")]).await;
        c.writes();
        let message = |chat_id: &str| watch::Outgoing {
            watch: watch::WatchDef {
                id: Uuid::nil(),
                ..watch::WatchDef::for_test(chat_id, c.user_id, "mail")
            },
            text: "new_mail id=1".into(),
            cancelled: Default::default(),
        };

        let stopped = message(&c.chat_id);
        stopped.cancelled.cancel();
        let refused =
            deliver_watch_message(stopped, &c.mirror, &c.keys, &mut c.supervisor, &c.write_tx);
        assert!(
            refused.await.is_err(),
            "queued before its watch was stopped"
        );

        // An unknown chat, and one whose user's key is gone (a revoked member).
        let revoked = KeyStore::new();
        for (chat_id, keys) in [("no-such-chat", &c.keys), (c.chat_id.as_str(), &revoked)] {
            let out = message(chat_id);
            let refused =
                deliver_watch_message(out, &c.mirror, keys, &mut c.supervisor, &c.write_tx).await;
            assert!(refused.is_err(), "stays in the outbox");
        }
        assert!(c.spawned_prompts().is_empty());

        let out = message(&c.chat_id);
        deliver_watch_message(out, &c.mirror, &c.keys, &mut c.supervisor, &c.write_tx)
            .await
            .unwrap();
        assert_eq!(
            c.spawned_prompts(),
            vec!["[zucchini watch \"mail\"]\nnew_mail id=1".to_string()]
        );
        assert_eq!(
            c.message_rows(),
            vec![json!({
                "type": "system",
                "subtype": "peer_message",
                "name": "watch \"mail\"",
                "body": "new_mail id=1",
            })]
        );
    }

    fn notices(c: &ResidentChat) -> usize {
        c.spawned_prompts()
            .iter()
            .filter(|p| **p == restart_notice_turn())
            .count()
    }

    #[tokio::test]
    async fn always_on_chat_starts_with_a_notice_and_stops_once_its_flag_is_off() {
        let mut c = ResidentChat::new();
        c.window(vec![c.project()]).await;
        c.writes();
        c.window(vec![always_on_chat(&c, 1, true)]).await;
        assert_eq!(c.spawned_prompts(), vec![restart_notice_turn()]);
        assert_eq!(
            c.recorded.lock().unwrap()[0].agent_session_id.as_deref(),
            Some("sid")
        );
        assert_eq!(
            c.message_rows(),
            vec![json!({
                "type": "system",
                "subtype": "peer_message",
                "name": "Zucchini",
                "body": RESTART_NOTICE,
            })]
        );

        c.window(vec![always_on_chat(&c, 2, false), c.message(2, "/stop")])
            .await;
        assert!(!c.supervisor.has_agent(&c.chat_id));
        assert_eq!(notices(&c), 1, "flag off in the /stop's window: no restart");

        c.window(vec![always_on_chat(&c, 2, true)]).await;
        assert_eq!(notices(&c), 2, "turning it on starts it again");

        c.window(vec![always_on_chat(&c, 2, false)]).await;
        c.window(vec![always_on_chat(&c, 3, false), c.message(3, "/stop")])
            .await;
        assert!(!c.supervisor.has_agent(&c.chat_id));
        assert_eq!(notices(&c), 2, "flag off a window earlier: no restart");
    }

    #[tokio::test]
    async fn a_clear_drops_the_chats_it_does_not_re_stream() {
        let mut c = ResidentChat::new();
        c.window(vec![c.project(), always_on_chat(&c, 1, true)])
            .await;
        restart(&mut c);
        c.window(vec![
            SyncEvent::Clear,
            c.project(),
            always_on_chat(&c, 1, true),
        ])
        .await;
        assert_eq!(notices(&c), 1, "a re-streamed chat survives the clear");

        restart(&mut c);
        c.window(vec![SyncEvent::Clear, c.project()]).await;
        assert_eq!(notices(&c), 0, "a chat deleted meanwhile doesn't start");
        assert!(!c.mirror.chats.contains_key(&c.chat_id));
    }

    #[tokio::test]
    async fn a_clear_discards_a_chat_staged_before_it() {
        let mut c = ResidentChat::new();
        c.window(vec![c.project(), always_on_chat(&c, 1, true)])
            .await;
        restart(&mut c);
        c.window(vec![
            always_on_chat(&c, 1, true),
            SyncEvent::Clear,
            c.project(),
        ])
        .await;
        assert_eq!(notices(&c), 0);
        assert!(!c.mirror.chats.contains_key(&c.chat_id));
    }

    /// The spawner restarts: state.json reloads, the supervisor is empty.
    fn restart(c: &mut ResidentChat) {
        let saved = serde_json::to_string(&c.mirror).unwrap();
        c.mirror = serde_json::from_str(&saved).unwrap();
        (c.supervisor, c.recorded) = supervisor_with_resident_recorder(c.resp_tx.clone());
    }

    #[tokio::test]
    async fn boot_restarts_always_on_chats_unless_stopped() {
        let mut c = ResidentChat::new();
        c.window(vec![c.project(), always_on_chat(&c, 1, true)])
            .await;
        restart(&mut c);
        c.window(vec![]).await;
        assert_eq!(notices(&c), 1);

        restart(&mut c);
        c.window(vec![always_on_chat(&c, 2, false), c.message(2, "/stop")])
            .await;
        assert_eq!(notices(&c), 0, "a /stop synced at boot wins");

        restart(&mut c);
        c.window(vec![]).await;
        assert_eq!(notices(&c), 0);
    }

    #[tokio::test]
    async fn always_on_chats_start_together_and_while_another_chat_is_busy() {
        let mut c = ResidentChat::new();
        c.window(vec![c.project(), always_on_chat(&c, 1, true)])
            .await;
        c.chat_id = Uuid::now_v7().to_string();
        c.window(vec![always_on_chat(&c, 1, true)]).await;
        restart(&mut c);
        c.window(vec![]).await;
        assert_eq!(notices(&c), 2, "both start at boot");

        c.chat_id = Uuid::now_v7().to_string();
        c.window(vec![always_on_chat(&c, 1, true)]).await;
        assert_eq!(notices(&c), 3, "a third starts while the two are in a turn");
    }

    #[tokio::test]
    async fn a_dying_always_on_resident_is_restarted_a_bounded_number_of_times() {
        let mut c = ResidentChat::new();
        c.window(vec![c.project(), always_on_chat(&c, 1, true)])
            .await;
        for _ in 0..5 {
            c.supervisor.abort_agent(&c.chat_id).await;
            reconcile_always_on(&mut c.mirror, &mut c.supervisor, &c.write_tx).await;
        }
        assert_eq!(notices(&c), ALWAYS_ON_RETRIES as usize);
    }

    #[tokio::test]
    async fn toggling_always_on_resets_the_start_count() {
        let mut c = ResidentChat::new();
        c.window(vec![c.project(), always_on_chat(&c, 1, true)])
            .await;
        for _ in 0..5 {
            c.supervisor.abort_agent(&c.chat_id).await;
            reconcile_always_on(&mut c.mirror, &mut c.supervisor, &c.write_tx).await;
        }
        c.window(vec![always_on_chat(&c, 2, false)]).await;
        c.window(vec![always_on_chat(&c, 3, true)]).await;
        assert_eq!(notices(&c), ALWAYS_ON_RETRIES as usize + 1);
    }

    #[tokio::test]
    async fn a_pass_during_a_dying_residents_cleanup_keeps_its_start_count() {
        let mut c = ResidentChat::new();
        c.window(vec![c.project(), always_on_chat(&c, 1, true)])
            .await;
        for _ in 0..5 {
            for session in c.supervisor.live_sessions().lock().unwrap().values() {
                let mut st = session.lock().unwrap();
                st.unconsumed.clear();
                st.closed = true;
            }
            reconcile_always_on(&mut c.mirror, &mut c.supervisor, &c.write_tx).await;
            c.supervisor.abort_agent(&c.chat_id).await;
            reconcile_always_on(&mut c.mirror, &mut c.supervisor, &c.write_tx).await;
        }
        assert_eq!(notices(&c), ALWAYS_ON_RETRIES as usize);
    }

    #[tokio::test]
    async fn a_members_always_on_chat_is_neither_started_nor_exempt() {
        let mut c = ResidentChat::new();
        invite(&mut c, "row");
        c.window(vec![c.project(), always_on_chat(&c, 1, true)])
            .await;
        assert!(c.spawned_prompts().is_empty());
        assert!(c.mirror.always_on_chats().is_empty());
    }

    /// `plain` as the clients seal a `chats` column.
    fn sealed(c: &ResidentChat, plain: &str) -> serde_json::Value {
        let key = c.keys.get(&c.user_id).expect("seeded key");
        json!(crypto::encrypt_field_b64(&key, plain))
    }

    fn prompted_chat(c: &ResidentChat, last_seq: i64, text: &str) -> SyncEvent {
        chat_with(c, last_seq, json!({ "system_prompt": sealed(c, text) }))
    }

    fn always_on_prompted_chat(c: &ResidentChat, last_seq: i64, text: &str) -> SyncEvent {
        chat_with(
            c,
            last_seq,
            json!({ "always_on": 1, "agent_session_id": "sid", "system_prompt": sealed(c, text) }),
        )
    }

    fn spawned_prompt_texts(c: &ResidentChat) -> Vec<Option<String>> {
        c.recorded
            .lock()
            .unwrap()
            .iter()
            .map(|r| r.user_system_prompt.text.clone())
            .collect()
    }

    /// The live session finished its turn: idle.
    fn finish_turn(c: &ResidentChat) {
        let s = c.session();
        let mut st = s.lock().unwrap();
        st.unconsumed.clear();
        st.turn_in_flight = false;
    }

    fn arm_monitor(c: &ResidentChat) {
        c.session()
            .lock()
            .unwrap()
            .live_tasks
            .push(agent::LiveTask {
                id: "monitor".into(),
                description: String::new(),
                task_type: "bash".into(),
                subagent_type: None,
                started_at: 0,
            });
    }

    #[tokio::test]
    async fn a_chats_system_prompt_reaches_the_spawn_decrypted() {
        let mut c = ResidentChat::new();
        let chat = chat_with(
            &c,
            1,
            json!({
                "system_prompt": sealed(&c, "be terse"),
                "system_prompt_files": sealed(&c, "docs/a.md, ~/b.md"),
            }),
        );
        c.window(vec![c.project(), chat, c.message(1, "first")])
            .await;
        assert_eq!(
            c.recorded.lock().unwrap()[0].user_system_prompt,
            user_prompt::UserSystemPrompt {
                files: Some("docs/a.md, ~/b.md".to_string()),
                text: Some("be terse".to_string()),
                undecryptable: false,
            }
        );

        c.chat_id = Uuid::now_v7().to_string();
        let undecryptable = chat_with(&c, 1, json!({ "system_prompt": "bm90IHNlYWxlZA==" }));
        c.window(vec![undecryptable, c.message(1, "first")]).await;
        assert_eq!(
            c.recorded.lock().unwrap()[1].user_system_prompt,
            user_prompt::UserSystemPrompt {
                undecryptable: true,
                ..Default::default()
            },
            "its spawn fails rather than run without the prompt"
        );
        c.chat_id = Uuid::now_v7().to_string();
        let blank = chat_with(&c, 1, json!({ "system_prompt": sealed(&c, " \n") }));
        c.window(vec![blank, c.message(1, "first")]).await;
        c.chat_id = Uuid::now_v7().to_string();
        let codex = chat_with(
            &c,
            1,
            json!({ "system_prompt": sealed(&c, "be terse"), "agent_kind": "codex" }),
        );
        c.window(vec![codex, c.message(1, "first")]).await;
        for unset in &c.recorded.lock().unwrap()[2..] {
            assert_eq!(unset.user_system_prompt, Default::default());
        }
    }

    #[tokio::test]
    async fn a_sandboxed_chat_gets_no_system_prompt() {
        let mut c = ResidentChat::new();
        invite(&mut c, "row");
        c.window(vec![
            c.project(),
            prompted_chat(&c, 1, "be terse"),
            c.message(1, "first"),
        ])
        .await;
        assert_eq!(spawned_prompt_texts(&c), vec![Some("be terse".to_string())]);

        finish_turn(&c);
        c.mirror.upsert_member("row".into(), c.user_id, true);
        c.window(vec![
            prompted_chat(&c, 2, "be terse"),
            c.message(2, "second"),
        ])
        .await;
        let sandboxed = c.recorded.lock().unwrap()[1].clone();
        assert!(sandboxed.is_sandboxed);
        assert_eq!(sandboxed.user_system_prompt, Default::default());

        finish_turn(&c);
        let undecryptable = chat_with(&c, 2, json!({ "system_prompt": "bm90IHNlYWxlZA==" }));
        c.window(vec![prompted_chat(&c, 2, "be verbose")]).await;
        c.window(vec![undecryptable, c.message(3, "third")]).await;
        assert_eq!(
            c.spawned_prompts().len(),
            2,
            "its prompt columns change nothing: no retire, no respawn"
        );
        assert!(c.supervisor.has_agent(&c.chat_id));
    }

    #[tokio::test]
    async fn a_message_after_a_system_prompt_change_respawns_with_the_new_one() {
        let mut c = ResidentChat::new();
        c.window(vec![
            c.project(),
            prompted_chat(&c, 1, "be terse"),
            c.message(1, "first"),
        ])
        .await;
        finish_turn(&c);
        c.window(vec![
            prompted_chat(&c, 2, "be terse"),
            c.message(2, "second"),
        ])
        .await;
        assert_eq!(c.spawned_prompts(), vec!["first"], "same prompt: stdin");

        c.window(vec![
            prompted_chat(&c, 3, "be verbose"),
            c.message(3, "third"),
        ])
        .await;
        assert_eq!(
            c.spawned_prompts(),
            vec!["first"],
            "the queued turn is not cut: stdin"
        );
        assert_eq!(c.unconsumed(), vec!["second", "third"]);

        finish_turn(&c);
        c.window(vec![
            prompted_chat(&c, 4, "be verbose"),
            c.message(4, "fourth"),
        ])
        .await;
        assert_eq!(c.spawned_prompts(), vec!["first", "fourth"]);
        assert_eq!(
            spawned_prompt_texts(&c),
            vec![Some("be terse".to_string()), Some("be verbose".to_string())]
        );
        assert!(c.supervisor.has_agent(&c.chat_id), "not retired again");
    }

    #[tokio::test]
    async fn a_system_prompt_change_retires_the_resident_once_it_is_idle() {
        let mut c = ResidentChat::new();
        c.window(vec![
            c.project(),
            prompted_chat(&c, 1, "be terse"),
            c.message(1, "first"),
        ])
        .await;
        c.consume_first_turn();
        c.window(vec![prompted_chat(&c, 1, "be verbose")]).await;
        assert!(c.supervisor.is_running(&c.chat_id), "the turn is not cut");

        finish_turn(&c);
        c.writes();
        let prunes = prune::PendingPrunes::default();
        let chat_id = c.chat_id.clone();
        let mirror = c.shared_mirror();
        handle_run_state(
            &chat_id,
            false,
            &prunes,
            &mirror,
            &mut c.supervisor,
            &c.write_tx,
        )
        .await;
        c.mirror = std::mem::take(&mut *mirror.write().await);
        assert!(!c.supervisor.has_agent(&c.chat_id));
        let writes = c.writes();
        assert!(matches!(
            writes[0],
            WriteEvent::ChatRunning {
                agent_running: false,
                stopped_by_user: false,
                ..
            }
        ));
        assert!(matches!(
            writes[1],
            WriteEvent::ChatLiveTasks { tasks: None, .. }
        ));
        assert_eq!(writes.len(), 2);

        c.window(vec![
            prompted_chat(&c, 2, "be verbose"),
            c.message(2, "second"),
        ])
        .await;
        assert_eq!(
            spawned_prompt_texts(&c),
            vec![Some("be terse".to_string()), Some("be verbose".to_string())]
        );
        finish_turn(&c);
        c.window(vec![prompted_chat(&c, 2, "be verbose")]).await;
        assert!(
            c.supervisor.has_agent(&c.chat_id),
            "a re-stream changes nothing"
        );

        arm_monitor(&c);
        c.window(vec![prompted_chat(&c, 2, "be brief")]).await;
        assert!(c.supervisor.has_agent(&c.chat_id), "its monitor is spared");
        c.session().lock().unwrap().live_tasks.clear();
        c.window(vec![chat_with(&c, 2, json!({}))]).await;
        assert!(!c.supervisor.has_agent(&c.chat_id), "idle: retired at once");
        assert_eq!(
            c.spawned_prompts().len(),
            2,
            "and left for the next message"
        );
    }

    #[tokio::test]
    async fn an_always_on_chat_restarts_with_each_new_system_prompt() {
        let mut c = ResidentChat::new();
        c.window(vec![c.project(), always_on_prompted_chat(&c, 1, "v0")])
            .await;
        let edits = ALWAYS_ON_RETRIES as usize + 2;
        for edit in 1..=edits {
            c.window(vec![always_on_prompted_chat(&c, 1, &format!("v{edit}"))])
                .await;
            assert_eq!(notices(&c), edit, "the notice turn is not cut");
            finish_turn(&c);
            c.respond(AgentResponse::TurnEnded {
                topic: c.chat_id.clone(),
            })
            .await;
            assert_eq!(
                notices(&c),
                edit,
                "idle: left to the idle edge or the next row"
            );
            c.window(vec![always_on_prompted_chat(&c, 1, &format!("v{edit}"))])
                .await;
            assert_eq!(notices(&c), edit + 1);
        }
        let expected: Vec<_> = (0..=edits).map(|v| Some(format!("v{v}"))).collect();
        assert_eq!(spawned_prompt_texts(&c), expected);

        finish_turn(&c);
        arm_monitor(&c);
        c.writes();
        c.window(vec![always_on_prompted_chat(&c, 1, "armed")])
            .await;
        assert_eq!(notices(&c), edits + 2, "restarted with its monitor armed");
        assert!(c.writes().iter().any(|w| matches!(
            w,
            WriteEvent::ChatRunning {
                agent_running: false,
                stopped_by_user: true,
                ..
            }
        )));

        c.window(vec![always_on_prompted_chat(&c, 1, "mid-turn")])
            .await;
        finish_turn(&c);
        arm_monitor(&c);
        c.respond(AgentResponse::TurnEnded {
            topic: c.chat_id.clone(),
        })
        .await;
        assert_eq!(
            notices(&c),
            edits + 3,
            "left waiting: restarted at the turn end"
        );
        assert_eq!(
            spawned_prompt_texts(&c).last().unwrap().as_deref(),
            Some("mid-turn")
        );
    }

    #[tokio::test]
    async fn a_system_prompt_that_does_not_render_keeps_the_resident() {
        let mut c = ResidentChat::new();
        c.window(vec![
            c.project(),
            prompted_chat(&c, 1, "be terse"),
            c.message(1, "first"),
        ])
        .await;
        finish_turn(&c);
        let typo = chat_with(
            &c,
            1,
            json!({
                "system_prompt": sealed(&c, "be terse"),
                "system_prompt_files": sealed(&c, "zucchini-no-such-file.md"),
            }),
        );
        c.window(vec![typo]).await;
        assert!(c.supervisor.has_agent(&c.chat_id), "a typo'd path");

        let undecryptable = chat_with(&c, 1, json!({ "system_prompt": "bm90IHNlYWxlZA==" }));
        c.window(vec![undecryptable]).await;
        assert!(c.mirror.chats[&c.chat_id].user_system_prompt.undecryptable);
        assert!(c.supervisor.has_agent(&c.chat_id));

        c.window(vec![prompted_chat(&c, 1, "be terse")]).await;
        assert!(
            c.supervisor.has_agent(&c.chat_id),
            "decrypted again and unchanged: nothing to retire"
        );
        c.window(vec![prompted_chat(&c, 1, "be verbose")]).await;
        assert!(!c.supervisor.has_agent(&c.chat_id), "the one real change");
        assert_eq!(c.spawned_prompts().len(), 1);
    }

    #[tokio::test]
    async fn an_always_on_chat_with_tasks_armed_restarts_while_another_chat_is_busy() {
        let mut c = ResidentChat::new();
        c.window(vec![c.project(), always_on_prompted_chat(&c, 1, "v0")])
            .await;
        finish_turn(&c);
        arm_monitor(&c);
        let always_on = c.chat_id.clone();
        c.chat_id = Uuid::now_v7().to_string();
        c.window(vec![c.chat(0, ""), c.message(1, "busy")]).await;
        c.chat_id = always_on;

        c.window(vec![always_on_prompted_chat(&c, 1, "v1")]).await;
        assert_eq!(notices(&c), 2);
        assert_eq!(
            spawned_prompt_texts(&c).last().unwrap().as_deref(),
            Some("v1")
        );
    }

    #[tokio::test]
    async fn a_fixed_system_prompt_gets_an_always_on_chat_its_retries_back() {
        let mut c = ResidentChat::new();
        c.window(vec![c.project(), always_on_prompted_chat(&c, 1, "broken")])
            .await;
        for _ in 0..5 {
            c.supervisor.abort_agent(&c.chat_id).await;
            reconcile_always_on(&mut c.mirror, &mut c.supervisor, &c.write_tx).await;
        }
        assert_eq!(notices(&c), ALWAYS_ON_RETRIES as usize);
        c.window(vec![always_on_prompted_chat(&c, 1, "broken")])
            .await;
        assert_eq!(notices(&c), ALWAYS_ON_RETRIES as usize);
        c.window(vec![always_on_prompted_chat(&c, 1, "fixed")])
            .await;
        assert_eq!(notices(&c), ALWAYS_ON_RETRIES as usize + 1);
    }

    #[tokio::test]
    async fn boot_restarts_an_always_on_chat_with_its_system_prompt() {
        let mut c = ResidentChat::new();
        c.window(vec![
            c.project(),
            always_on_prompted_chat(&c, 1, "be terse"),
        ])
        .await;
        assert!(!serde_json::to_string(&c.mirror)
            .unwrap()
            .contains("be terse"));
        restart(&mut c);
        c.mirror.unseal_loaded_chats(&c.keys);
        c.window(vec![]).await;
        assert_eq!(spawned_prompt_texts(&c), vec![Some("be terse".to_string())]);
    }

    #[tokio::test]
    async fn boot_restarts_an_always_on_chat_under_its_title() {
        let mut c = ResidentChat::new();
        let key = c.keys.get(&c.user_id).expect("seeded key");
        let title = crypto::encrypt_field_b64(&key, "cover letters writer");
        let chat = put_with(always_on_chat(&c, 1, true), json!({ "title": title }));
        c.window(vec![c.project(), chat]).await;
        restart(&mut c);
        c.mirror.unseal_loaded_chats(&c.keys);
        c.window(vec![]).await;
        let recorded = c.recorded.lock().unwrap();
        assert_eq!(recorded.len(), 1);
        assert_eq!(recorded[0].prompt, restart_notice_turn());
        assert_eq!(
            recorded[0].session_name.as_deref(),
            Some("cover letters writer")
        );
    }

    #[test]
    fn a_saved_chat_missing_fields_still_loads() {
        let mirror: Mirror =
            serde_json::from_str(r#"{"chats":{"c":{"always_on":true}},"projects":{"p":"/p"}}"#)
                .unwrap();
        assert!(mirror.chats["c"].always_on);
        assert_eq!(mirror.projects["p"], "/p");
    }

    #[tokio::test]
    async fn only_always_on_chats_persist_and_with_a_sealed_title() {
        let mut c = ResidentChat::new();
        let key = c.keys.get(&c.user_id).expect("seeded key");
        for (id, on) in [("on", true), ("off", false)] {
            c.chat_id = id.into();
            let SyncEvent::Put { data, .. } = always_on_chat(&c, 1, on) else {
                unreachable!("always_on_chat() is a Put")
            };
            let mut row: serde_json::Value = serde_json::from_str(&data).unwrap();
            row["title"] = json!(crypto::encrypt_field_b64(&key, "secret"));
            c.mirror.upsert_chat(id.into(), &row.to_string(), &c.keys);
        }
        assert_eq!(c.mirror.chats["on"].title.as_deref(), Some("secret"));
        let json = serde_json::to_string(&c.mirror).unwrap();
        assert!(!json.contains("secret"));
        let mut loaded: Mirror = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.always_on_chats(), HashSet::from(["on".to_string()]));
        assert_eq!(loaded.chats["on"].agent_session_id.as_deref(), Some("sid"));
        assert!(loaded.chats["on"].title.is_none());
        loaded.unseal_loaded_chats(&c.keys);
        assert_eq!(loaded.chats["on"].title.as_deref(), Some("secret"));
    }

    #[tokio::test]
    async fn chat_delete_aborts_its_agent_and_drops_its_late_writes() {
        let mut c = ResidentChat::new();
        c.window(vec![c.project(), c.chat(1, ""), c.message(1, "first")])
            .await;
        c.writes();
        c.window(vec![remove("chats", &c.chat_id)]).await;
        assert!(!c.supervisor.is_running(&c.chat_id));

        let mirror = c.shared_mirror();
        let prunes = prune::PendingPrunes::default();
        handle_run_state(
            &c.chat_id,
            false,
            &prunes,
            &mirror,
            &mut c.supervisor,
            &c.write_tx,
        )
        .await;
        handle_agent_response(
            AgentResponse::ContextTokens {
                topic: c.chat_id.clone(),
                tokens: 1,
            },
            &mut *mirror.write().await,
            &c.write_tx,
            &mut c.supervisor,
        )
        .await;
        assert!(c.writes().is_empty());
    }

    #[tokio::test]
    async fn project_delete_aborts_its_chats_agents() {
        let mut c = ResidentChat::new();
        c.window(vec![c.project(), c.chat(1, ""), c.message(1, "first")])
            .await;
        c.writes();
        c.window(vec![remove("projects", &c.project_id)]).await;
        assert!(!c.supervisor.is_running(&c.chat_id));
        assert!(!c.mirror.chats.contains_key(&c.chat_id));
        assert!(c.writes().is_empty());
    }

    #[tokio::test]
    async fn invitee_revoke_aborts_their_agents_without_writes() {
        let mut c = ResidentChat::new();
        invite(&mut c, "row-1");
        c.window(vec![c.project(), c.chat(1, ""), c.message(1, "first")])
            .await;
        assert!(c.supervisor.is_running(&c.chat_id));
        c.writes();
        c.window(vec![remove("machine_users", "row-1")]).await;
        assert!(!c.supervisor.is_running(&c.chat_id));
        assert!(c.writes().is_empty());
    }

    #[tokio::test]
    async fn sandbox_flip_aborts_the_running_turn() {
        let mut c = ResidentChat::new();
        invite(&mut c, "row-1");
        c.window(vec![c.project(), c.chat(1, ""), c.message(1, "first")])
            .await;
        c.writes();
        let flip = SyncEvent::Put {
            table: "machine_users".into(),
            id: "row-1".into(),
            data: json!({
                "user_id": c.user_id.to_string(),
                "is_sandboxed": 1,
                "sealed_blob": "blob",
            })
            .to_string(),
        };
        c.window(vec![flip]).await;
        assert!(!c.supervisor.is_running(&c.chat_id));
        let writes = c.writes();
        assert!(writes.iter().any(is_interrupted), "{writes:?}");
        assert!(writes.iter().any(|w| matches!(
            w,
            WriteEvent::ChatRunning {
                agent_running: false,
                ..
            }
        )));
    }

    /// The flip tears down a WARM IDLE resident too, and the clear its reader
    /// already published is dropped as a replaced session's — so the teardown
    /// has to write the NULL itself or the row keeps showing finished tasks.
    #[tokio::test]
    async fn sandbox_flip_clears_the_live_tasks_of_an_idle_resident() {
        let mut c = ResidentChat::new();
        invite(&mut c, "row-1");
        c.window(vec![c.project(), c.chat(1, ""), c.message(1, "first")])
            .await;
        c.consume_first_turn();
        c.reader_raw(r#"{"type":"system","subtype":"task_started","task_id":"t-7","description":"deploy","task_type":"local_bash"}"#).await;

        // The task finished and the turn ended: the reader's clear is still in
        // the response queue when the flip lands.
        let session = c.session();
        {
            let mut st = session.lock().unwrap();
            st.live_tasks.clear();
            st.turn_in_flight = false;
        }
        let queued_clear = AgentResponse::LiveTasks {
            topic: c.chat_id.clone(),
            session,
            tasks: None,
        };
        assert!(!c.supervisor.is_running(&c.chat_id), "warm idle");
        c.writes();

        c.window(vec![SyncEvent::Put {
            table: "machine_users".into(),
            id: "row-1".into(),
            data: json!({
                "user_id": c.user_id.to_string(),
                "is_sandboxed": 1,
                "sealed_blob": "blob",
            })
            .to_string(),
        }])
        .await;
        c.respond(queued_clear).await;
        assert_eq!(
            live_task_blobs(&c.writes()),
            vec![None],
            "the idle teardown clears the column"
        );
    }

    #[tokio::test]
    async fn an_idle_resident_does_not_hold_back_an_update() {
        let mut c = ResidentChat::new();
        c.window(vec![c.project(), c.chat(1, ""), c.message(1, "first")])
            .await;
        assert!(c.supervisor.any_running());
        c.supervisor.live_sessions().lock().unwrap()[&c.chat_id]
            .lock()
            .unwrap()
            .unconsumed
            .clear();
        assert!(!c.supervisor.any_running());
    }

    #[tokio::test]
    async fn message_to_a_live_resident_is_queued_on_stdin_not_respawned() {
        let mut c = ResidentChat::new();
        c.window(vec![c.project(), c.chat(1, ""), c.message(1, "first")])
            .await;
        c.writes();
        c.window(vec![c.chat(2, ""), c.message(2, "second")]).await;

        assert_eq!(c.spawned_prompts(), vec!["first"], "no abort + respawn");
        assert_eq!(c.unconsumed(), vec!["first", "second"]);
        let writes = c.writes();
        assert!(!writes.iter().any(is_interrupted), "{writes:?}");
        assert!(writes.iter().any(|w| matches!(
            w,
            WriteEvent::ChatRunning {
                agent_running: true,
                ..
            }
        )));
    }

    /// The queued count must reach the clients AT the enqueue edge: the reader
    /// publishes only on adapter events and a queued turn produces none until
    /// the CLI echoes it, so the pill would list a stale count until unrelated
    /// output happened to arrive.
    #[tokio::test]
    async fn a_turn_queued_on_a_live_resident_publishes_its_count_at_once() {
        let mut c = ResidentChat::new();
        c.window(vec![c.project(), c.chat(1, ""), c.message(1, "first")])
            .await;
        c.consume_first_turn();
        c.writes();

        c.window(vec![c.chat(2, ""), c.message(2, "second")]).await;
        assert_eq!(
            live_task_blobs(&c.writes()),
            vec![Some(r#"{"q":1}"#.to_string())],
            "the queued count rides the row with no adapter output"
        );

        // The change detection is shared with the reader, so the echo that
        // consumes the turn clears the column — and nothing re-writes it.
        let uuid = c.session().lock().unwrap().unconsumed[0].uuid.clone();
        c.reader_raw(&format!(
            r#"{{"type":"user","message":{{"role":"user","content":[{{"type":"text","text":"second"}}]}},"session_id":"s1","uuid":"{uuid}","isReplay":true}}"#
        ))
        .await;
        assert_eq!(
            live_task_blobs(&c.writes()),
            vec![None],
            "the echo clears the count exactly once"
        );
    }

    #[tokio::test]
    async fn a_buffered_task_clear_cannot_overwrite_a_newer_queued_count() {
        let mut c = ResidentChat::new();
        c.window(vec![c.project(), c.chat(1, ""), c.message(1, "first")])
            .await;
        c.consume_first_turn();
        c.reader_raw(r#"{"type":"system","subtype":"task_started","task_id":"t1","description":"build","task_type":"local_bash"}"#).await;
        assert_eq!(live_task_blobs(&c.writes()).len(), 1);

        let session = c.session();
        let tasks = {
            let mut st = session.lock().unwrap();
            crate::agent::reduce(&mut st, &adapter::AgentEvent::TaskFinished("t1".into()));
            st.observe_live_tasks().expect("the finished task clears")
        };
        assert_eq!(tasks, None);
        c.resp_tx
            .send(AgentResponse::LiveTasks {
                topic: c.chat_id.clone(),
                session,
                tasks,
            })
            .await
            .unwrap();

        // The sync checkpoint wins the select while the reader's older clear
        // is buffered, so the enqueue publishes q=1 before main sees the clear.
        c.window(vec![c.chat(2, ""), c.message(2, "second")]).await;
        assert_eq!(
            live_task_blobs(&c.writes()),
            vec![Some(r#"{"q":1}"#.to_string())]
        );
        let buffered = c.resp_rx.try_recv().expect("buffered clear");
        c.respond(buffered).await;
        c.reader_raw(r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"still working"}]}}"#).await;
        assert!(
            live_task_blobs(&c.writes()).is_empty(),
            "the stale clear must be rejected; shared dedup cannot repair it"
        );

        let uuid = c.session().lock().unwrap().unconsumed[0].uuid.clone();
        c.reader_raw(
            &json!({
                "type": "user", "uuid": uuid, "isReplay": true,
                "message": { "role": "user", "content": "second" }
            })
            .to_string(),
        )
        .await;
        assert_eq!(live_task_blobs(&c.writes()), vec![None]);
    }

    /// A chat renamed in the app while its resident is warm: the `/rename`
    /// turn is the spawner's own and completely silent — no message row (which
    /// would bump the chat: unread, preview, push), no `agent_running` edge (no
    /// "finished" push, no queued-message promotion), no context-gauge move.
    #[tokio::test]
    async fn renaming_a_live_chat_writes_nothing_at_all() {
        let mut c = ResidentChat::new();
        c.window(vec![
            c.titled_chat(1, "cover letters"),
            c.project(),
            c.message(1, "first"),
        ])
        .await;
        c.consume_first_turn();
        c.reader_line(vec![adapter::AgentEvent::Result {
            origin_is_task: false,
        }])
        .await;
        c.writes();

        c.window(vec![c.titled_chat(1, "my voice writer")]).await;
        assert_eq!(c.unconsumed(), vec!["/rename my voice writer"]);
        let uuid = c.session().lock().unwrap().unconsumed[0].uuid.clone();

        for line in [
            format!(r#"{{"type":"command_lifecycle","command_uuid":"{uuid}","state":"started","session_id":"s1"}}"#),
            r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"Session renamed to: my voice writer"}],"usage":{"input_tokens":9000,"output_tokens":1}}}"#.to_string(),
            format!(r#"{{"type":"user","message":{{"role":"user","content":[{{"type":"text","text":"/rename my voice writer"}}]}},"session_id":"s1","uuid":"{uuid}","isReplay":true}}"#),
            r#"{"type":"result","subtype":"success","is_error":false,"result":"ok"}"#.to_string(),
        ] {
            c.reader_raw(&line).await;
            assert!(!c.supervisor.is_running(&c.chat_id), "never busy");
        }

        let writes = c.writes();
        assert!(writes.is_empty(), "the rename is silent: {writes:?}");
    }

    #[tokio::test]
    async fn user_messages_sharing_a_window_all_reach_the_resident_in_order() {
        let mut c = ResidentChat::new();
        c.window(vec![
            c.project(),
            c.chat(3, ""),
            c.message(1, "a"),
            c.message(2, "b"),
            c.message(3, "c"),
        ])
        .await;
        assert_eq!(c.spawned_prompts(), vec!["a"]);
        assert_eq!(c.unconsumed(), vec!["a", "b", "c"]);
    }

    /// Replay guard still holds: a user message followed by a non-window
    /// message (its answer) is a re-streamed, already-answered prompt.
    #[tokio::test]
    async fn a_replayed_answered_message_is_still_skipped() {
        let mut c = ResidentChat::new();
        c.window(vec![c.project(), c.chat(2, ""), c.message(1, "old")])
            .await;
        assert!(c.spawned_prompts().is_empty());
    }

    #[tokio::test]
    async fn knob_change_respawns_requeuing_unconsumed_turns_first() {
        let mut c = ResidentChat::new();
        c.window(vec![c.project(), c.chat(1, ""), c.message(1, "first")])
            .await;
        c.writes();
        c.window(vec![c.chat(2, "opus"), c.message(2, "second")])
            .await;

        assert_eq!(c.spawned_prompts(), vec!["first", "first"]);
        assert_eq!(c.recorded.lock().unwrap()[1].model.as_deref(), Some("opus"));
        assert_eq!(c.unconsumed(), vec!["first", "second"]);
        assert!(
            !c.writes().iter().any(is_interrupted),
            "only a queued turn, nothing in flight"
        );
    }

    #[tokio::test]
    async fn requeue_skips_turns_the_transcript_already_holds() {
        let mut c = ResidentChat::new();
        c.window(vec![c.project(), c.chat(1, ""), c.message(1, "first")])
            .await;
        let turns = c.supervisor.abort_agent(&c.chat_id).await;
        let uuid = turns[0].uuid.clone();
        assert_eq!(unpersisted_prompts(turns.clone(), ""), vec!["first"]);
        let queued_command = format!(
            r#"{{"type":"attachment","attachment":{{"type":"queued_command","source_uuid":"{uuid}"}}}}"#
        );
        assert!(unpersisted_prompts(turns, &queued_command).is_empty());
    }

    #[tokio::test]
    async fn late_done_leaves_the_live_session_and_its_indicator() {
        let mut c = ResidentChat::new();
        c.window(vec![c.project(), c.chat(1, ""), c.message(1, "first")])
            .await;
        c.writes();
        handle_agent_response(
            AgentResponse::Done {
                topic: c.chat_id.clone(),
                has_result: false,
            },
            &mut c.mirror,
            &c.write_tx,
            &mut c.supervisor,
        )
        .await;
        assert!(c.supervisor.is_running(&c.chat_id));
        assert!(!c.writes().iter().any(|w| matches!(
            w,
            WriteEvent::ChatRunning {
                agent_running: false,
                ..
            }
        )));
    }

    #[tokio::test]
    async fn stale_busy_edge_without_a_session_is_dropped() {
        let mut c = ResidentChat::new();
        c.window(vec![c.project(), c.chat(1, "")]).await;
        let mirror = c.shared_mirror();
        let prunes = prune::PendingPrunes::default();
        handle_run_state(
            &c.chat_id,
            true,
            &prunes,
            &mirror,
            &mut c.supervisor,
            &c.write_tx,
        )
        .await;
        assert!(c.writes().is_empty());
    }

    /// `last_seq` counts the running turn's agent rows, so a mid-turn user
    /// message can sit below it with a non-user row after it in its window.
    #[tokio::test]
    async fn mid_turn_message_behind_agent_rows_is_delivered_and_replays_are_not() {
        let mut c = ResidentChat::new();
        c.window(vec![c.project(), c.chat(1, ""), c.message(1, "first")])
            .await;
        c.window(vec![c.chat(3, ""), c.message(2, "second")]).await;
        assert_eq!(c.unconsumed(), vec!["first", "second"]);

        c.window(vec![
            c.chat(3, ""),
            c.message(1, "first"),
            c.message(2, "second"),
        ])
        .await;
        assert_eq!(c.unconsumed(), vec!["first", "second"], "re-stream skipped");
        assert_eq!(c.spawned_prompts(), vec!["first"]);
    }

    #[tokio::test]
    async fn stale_idle_edge_keeps_a_session_with_a_queued_turn() {
        let mut c = ResidentChat::new();
        c.window(vec![c.project(), c.chat(1, ""), c.message(1, "first")])
            .await;
        c.writes();
        let mirror = c.shared_mirror();
        let prunes = prune::PendingPrunes::default();

        handle_run_state(
            &c.chat_id,
            false,
            &prunes,
            &mirror,
            &mut c.supervisor,
            &c.write_tx,
        )
        .await;
        assert!(c.supervisor.is_running(&c.chat_id), "not torn down");
        assert!(c.writes().is_empty(), "idle column not written");

        c.supervisor.abort_agent(&c.chat_id).await;
        handle_run_state(
            &c.chat_id,
            false,
            &prunes,
            &mirror,
            &mut c.supervisor,
            &c.write_tx,
        )
        .await;
        assert!(matches!(
            c.writes()[..],
            [WriteEvent::ChatRunning {
                agent_running: false,
                ..
            }]
        ));
    }

    #[tokio::test]
    async fn idle_edge_keeps_the_resident_warm_for_the_next_turn() {
        use crate::agent::reduce;
        use adapter::AgentEvent;
        let mut c = ResidentChat::new();
        c.window(vec![c.project(), c.chat(1, ""), c.message(1, "first")])
            .await;
        let session = c.session();
        let first = session.lock().unwrap().unconsumed[0].uuid.clone();
        reduce(
            &mut session.lock().unwrap(),
            &AgentEvent::UserTurnConsumed(first),
        );
        reduce(
            &mut session.lock().unwrap(),
            &AgentEvent::Result {
                origin_is_task: false,
            },
        );
        c.writes();

        let mirror = c.shared_mirror();
        handle_run_state(
            &c.chat_id,
            false,
            &prune::PendingPrunes::default(),
            &mirror,
            &mut c.supervisor,
            &c.write_tx,
        )
        .await;
        c.mirror = std::mem::take(&mut *mirror.write().await);
        publish_alive_changes(&mut c.supervisor, &c.write_tx).await;
        assert!(
            matches!(
                c.writes()[..],
                [WriteEvent::ChatRunning {
                    agent_running: false,
                    ..
                }]
            ),
            "idle, and still alive"
        );

        c.window(vec![c.chat(3, ""), c.message(2, "second")]).await;
        assert_eq!(c.spawned_prompts(), vec!["first"], "same process");
        assert_eq!(c.unconsumed(), vec!["second"]);
        let second = session.lock().unwrap().unconsumed[0].uuid.clone();
        reduce(
            &mut session.lock().unwrap(),
            &AgentEvent::UserTurnConsumed(second),
        );
        assert!(session.lock().unwrap().running());
        reduce(
            &mut session.lock().unwrap(),
            &AgentEvent::Result {
                origin_is_task: false,
            },
        );
        assert!(session.lock().unwrap().is_idle());
    }

    /// Five residents: one busy past its idle edge, one with no idle edge,
    /// three idle. Three idle get 24h, so the one idle 25h goes; then two
    /// idle get 48h.
    #[tokio::test(start_paused = true)]
    async fn eviction_kills_the_oldest_idle_past_its_ttl_and_clears_its_alive_flag() {
        let (write_tx, mut write_rx) = mpsc::channel::<WriteEvent>(64);
        let (resp_tx, _resp_rx) = mpsc::channel::<AgentResponse>(64);
        let (mut sup, _) = supervisor_with_resident_recorder(resp_tx);
        let hour = Duration::from_secs(3600);
        let mut since = std::collections::HashMap::new();
        for chat in ["busy", "idle-25h", "idle-24h", "no-edge", "idle-22h"] {
            sup.spawn_agent(SpawnRequest {
                session_name: None,
                user_system_prompt: Default::default(),
                chat_id: chat.to_string(),
                prompt: "hi".to_string(),
                project_path: None,
                worktree: false,
                agent_session_id: None,
                agent_kind: AgentKind::Claude,
                is_sandboxed: false,
                model: None,
                user_timezone: None,
            });
            {
                let state = sup.live_sessions().lock().unwrap()[chat].clone();
                let mut st = state.lock().unwrap();
                st.unconsumed.clear();
                st.turn_in_flight = chat == "busy";
                if chat != "no-edge" {
                    st.idle_since = Some(tokio::time::Instant::now());
                    since.insert(chat, tokio::time::Instant::now());
                }
            }
            tokio::time::advance(hour).await;
        }
        publish_alive_changes(&mut sup, &write_tx).await;
        tokio::time::advance(hour * 21).await;

        let pinned = HashSet::from(["idle-25h".to_string(), "idle-24h".to_string()]);
        evict_idle_residents(&mut sup, || None, &write_tx, &pinned, &HashSet::new()).await;
        assert_eq!(
            sup.idle_residents().len(),
            3,
            "the two past 24h are pinned, the third is inside its TTL"
        );

        let deadline = evict_idle_residents(
            &mut sup,
            || None,
            &write_tx,
            &HashSet::new(),
            &HashSet::new(),
        )
        .await;
        assert_eq!(deadline, Some(since["idle-24h"] + hour * 48));
        publish_alive_changes(&mut sup, &write_tx).await;
        let mut alive = Vec::new();
        let mut cleared = Vec::new();
        while let Ok(w) = write_rx.try_recv() {
            match w {
                WriteEvent::ChatAlive {
                    chat_id,
                    agent_alive,
                } => alive.push((chat_id, agent_alive)),
                WriteEvent::ChatLiveTasks {
                    chat_id,
                    tasks: None,
                    ..
                } => cleared.push(chat_id),
                _ => {}
            }
        }
        assert_eq!(
            cleared,
            vec!["idle-25h".to_string()],
            "the evicted process's task list is cleared with it"
        );
        assert_eq!(alive.len(), 6, "{alive:?}");
        assert_eq!(alive[5], ("idle-25h".to_string(), false));
        assert!(sup.is_running("busy"));
        assert_eq!(sup.idle_residents().len(), 2);
    }

    fn alive_writes(writes: &[WriteEvent]) -> Vec<bool> {
        writes
            .iter()
            .filter_map(|w| match w {
                WriteEvent::ChatAlive { agent_alive, .. } => Some(*agent_alive),
                _ => None,
            })
            .collect()
    }

    fn is_stopped_by_user(w: &WriteEvent) -> bool {
        matches!(
            w,
            WriteEvent::ChatRunning {
                agent_running: false,
                stopped_by_user: true,
                ..
            }
        )
    }

    /// `/stop` and its old alias `/kill` end the turn and the process; the next
    /// message resumes the session in a new one.
    #[tokio::test]
    async fn stop_ends_the_turn_and_the_process() {
        for stop in ["/stop", "/kill"] {
            let mut c = ResidentChat::new();
            c.window(vec![c.project(), c.chat(1, ""), c.message(1, "first")])
                .await;
            c.consume_first_turn();
            assert_eq!(alive_writes(&c.writes()), vec![true]);
            c.window(vec![c.chat(2, ""), c.message(2, stop)]).await;

            assert!(!c.supervisor.has_agent(&c.chat_id), "{stop}");
            let writes = c.writes();
            assert!(writes.iter().any(is_interrupted), "{writes:?}");
            assert!(writes.iter().any(is_stopped_by_user), "{writes:?}");
            assert_eq!(alive_writes(&writes), vec![false]);

            c.window(vec![c.chat(3, ""), c.message(3, "second")]).await;
            assert_eq!(c.spawned_prompts(), vec!["first", "second"]);
        }
    }

    #[tokio::test]
    async fn stop_on_an_idle_resident_kills_it_quietly() {
        let mut c = ResidentChat::new();
        c.window(vec![c.project(), c.chat(1, ""), c.message(1, "first")])
            .await;
        c.session().lock().unwrap().unconsumed.clear();
        c.writes();
        c.window(vec![c.chat(2, ""), c.message(2, "/stop")]).await;

        assert!(!c.supervisor.has_agent(&c.chat_id));
        let writes = c.writes();
        assert!(!writes.iter().any(is_interrupted), "{writes:?}");
        assert_eq!(alive_writes(&writes), vec![false]);
    }

    fn watch_in(c: &ResidentChat) -> String {
        let def = watch::WatchDef::for_test(&c.chat_id, c.user_id, "mail");
        watch::register_idle(def.clone());
        def.id.to_string()
    }

    /// The task ids of each `chats.live_tasks` write, watches included.
    fn live_task_ids(writes: &[WriteEvent]) -> Vec<Vec<String>> {
        let ids = |blob: Option<String>| {
            let blob: serde_json::Value =
                serde_json::from_str(&blob.unwrap_or("{}".into())).expect("valid JSON");
            let tasks = blob["t"].as_array().cloned().unwrap_or_default();
            tasks
                .iter()
                .map(|t| t["i"].as_str().expect("id").to_string())
                .collect()
        };
        live_task_blobs(writes).into_iter().map(ids).collect()
    }

    #[tokio::test]
    async fn stop_removes_the_chats_watches_and_the_agents_next_turn_says_so() {
        let mut c = ResidentChat::new();
        c.window(vec![c.project(), c.chat(1, ""), c.message(1, "first")])
            .await;
        c.consume_first_turn();
        let watch = watch_in(&c);
        c.writes();
        c.reader_raw(r#"{"type":"system","subtype":"task_started","task_id":"t-7","description":"deploy","task_type":"local_bash"}"#).await;
        assert_eq!(
            live_task_ids(&c.writes()),
            vec![vec![watch, "t-7".to_string()]],
            "the watch leads the agent's tasks"
        );

        c.window(vec![c.chat(2, ""), c.message(2, "/stop")]).await;
        assert!(watch::running_in(&c.chat_id).is_empty());
        let writes = c.writes();
        assert!(writes.iter().any(is_interrupted), "{writes:?}");
        assert!(writes.iter().any(is_stopped_by_user), "{writes:?}");
        assert_eq!(live_task_blobs(&writes), vec![None]);
        assert_eq!(
            c.spawned_prompts(),
            vec!["first"],
            "the stop starts no turn"
        );

        // A command turn ships verbatim; the note rides the first turn after.
        let note = "[zucchini] The user pressed Stop, which also removed this chat's watches: \
                    \"mail\" (true). They are not running any more.";
        for (seq, text) in [(3, "/compact"), (4, "second"), (5, "third")] {
            c.window(vec![c.chat(seq, ""), c.message(seq, text)]).await;
        }
        assert_eq!(c.spawned_prompts(), vec!["first", "/compact"]);
        assert_eq!(
            c.unconsumed(),
            vec![
                "/compact".to_string(),
                format!("{note}\n\nsecond"),
                "third".to_string()
            ]
        );
    }

    #[tokio::test]
    async fn stop_on_a_chat_with_only_watches_clears_its_run_flag() {
        let mut c = ResidentChat::new();
        c.window(vec![c.project(), c.chat(0, "")]).await;
        watch_in(&c);
        c.writes();
        c.window(vec![c.chat(1, ""), c.message(1, "/stop")]).await;
        let writes = c.writes();
        assert!(writes.iter().any(is_stopped_by_user), "{writes:?}");
        assert_eq!(live_task_blobs(&writes), vec![None]);
    }

    fn run_flags(writes: &[WriteEvent]) -> Vec<(&'static str, bool)> {
        writes
            .iter()
            .filter_map(|w| match w {
                WriteEvent::ChatRunning { agent_running, .. } => Some(("agent", *agent_running)),
                WriteEvent::ChatTurnRunning { turn_running, .. } => Some(("turn", *turn_running)),
                _ => None,
            })
            .collect()
    }

    #[tokio::test]
    async fn a_watch_keeps_its_chat_running_until_the_last_one_is_gone() {
        let mut c = ResidentChat::new();
        c.window(vec![c.project(), c.chat(0, "")]).await;
        let watch = watch_in(&c);
        c.writes();

        let revoked = KeyStore::new();
        publish_watch_change(&c.chat_id, &c.mirror, &revoked, &c.supervisor, &c.write_tx).await;
        publish_watch_change("deleted", &c.mirror, &c.keys, &c.supervisor, &c.write_tx).await;
        assert!(c.writes().is_empty());

        publish_watch_change(&c.chat_id, &c.mirror, &c.keys, &c.supervisor, &c.write_tx).await;
        let writes = c.writes();
        assert_eq!(live_task_ids(&writes), vec![vec![watch.clone()]]);
        assert_eq!(run_flags(&writes), vec![("agent", true)]);

        // A turn ends: the watch holds `agent_running`, the turn's end is told.
        c.window(vec![c.chat(1, ""), c.message(1, "first")]).await;
        c.consume_first_turn();
        finish_turn(&c);
        c.writes();
        c.respond(AgentResponse::RunState {
            topic: c.chat_id.clone(),
            running: false,
        })
        .await;
        assert_eq!(run_flags(&c.writes()), vec![("turn", false)]);

        let removed = watch::global()
            .unwrap()
            .remove(watch.parse().unwrap(), None);
        removed.await.expect("registered");
        publish_watch_change(&c.chat_id, &c.mirror, &c.keys, &c.supervisor, &c.write_tx).await;
        let writes = c.writes();
        assert_eq!(live_task_blobs(&writes), vec![None]);
        assert_eq!(run_flags(&writes), vec![("agent", false)]);
        assert!(!writes.iter().any(is_stopped_by_user), "an ordinary finish");
    }

    #[tokio::test]
    async fn stop_with_a_queued_turn_drops_it() {
        let mut c = ResidentChat::new();
        c.window(vec![c.project(), c.chat(1, ""), c.message(1, "first")])
            .await;
        c.consume_first_turn();
        c.window(vec![c.chat(2, ""), c.message(2, "second")]).await;
        c.writes();
        c.window(vec![c.chat(3, ""), c.message(3, "/stop")]).await;

        assert!(!c.supervisor.has_agent(&c.chat_id));
        let writes = c.writes();
        assert!(writes.iter().any(is_interrupted), "{writes:?}");
        assert_eq!(alive_writes(&writes), vec![false]);
    }

    #[tokio::test]
    async fn stop_without_a_process_only_clears_the_alive_flag() {
        let mut c = ResidentChat::new();
        c.window(vec![c.project(), c.chat(1, ""), c.message(1, "/kill")])
            .await;
        let writes = c.writes();
        assert!(
            matches!(
                writes[..],
                [
                    WriteEvent::ChatLiveTasks { tasks: None, .. },
                    WriteEvent::ChatAlive {
                        agent_alive: false,
                        ..
                    }
                ]
            ),
            "{writes:?}"
        );
    }

    /// The blob the clients read: a real `task_started` line puts the armed task
    /// on the chats row, an unchanged set writes nothing more, and the task
    /// finishing clears the column.
    #[tokio::test]
    async fn armed_tasks_ride_the_chats_row_and_clear_when_they_finish() {
        let mut c = ResidentChat::new();
        c.window(vec![c.project(), c.chat(1, ""), c.message(1, "first")])
            .await;
        c.consume_first_turn();
        c.writes();

        c.reader_raw(r#"{"type":"system","subtype":"task_started","task_id":"bzdt1ec2n","tool_use_id":"toolu_01","description":"sleep 8 && echo BG_DONE","task_type":"local_bash","session_id":"s1"}"#).await;
        let blob = live_task_blobs(&c.writes());
        assert_eq!(blob.len(), 1, "one write per change");
        let armed = blob[0].clone().expect("the armed task rides the row");
        assert!(
            armed.contains("bzdt1ec2n") && armed.contains("sleep 8"),
            "{armed}"
        );
        assert!(armed.contains("\"k\":\"bash\""), "{armed}");

        // An ordinary assistant frame doesn't touch the set: no second write.
        c.reader_raw(r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"working"}],"usage":{"input_tokens":10}}}"#).await;
        assert!(
            live_task_blobs(&c.writes()).is_empty(),
            "an unchanged payload is not re-written"
        );

        c.reader_raw(r#"{"type":"system","subtype":"task_updated","task_id":"bzdt1ec2n","patch":{"status":"completed"}}"#).await;
        assert_eq!(
            live_task_blobs(&c.writes()),
            vec![None],
            "the last task finishing clears the column"
        );
    }

    /// The `chats.turn_running` writes in arrival order.
    fn turn_running_writes(writes: &[WriteEvent]) -> Vec<bool> {
        writes
            .iter()
            .filter_map(|w| match w {
                WriteEvent::ChatTurnRunning { turn_running, .. } => Some(*turn_running),
                _ => None,
            })
            .collect()
    }

    const RESULT_LINE: &str =
        r#"{"type":"result","subtype":"success","is_error":false,"result":"ok"}"#;

    /// A turn that leaves a task armed ends `turn_running` (after its result
    /// row) while the chat stays `agent_running`; the continuation turn the
    /// task's completion wakes raises it again.
    #[tokio::test]
    async fn turn_running_ends_with_the_turn_while_tasks_keep_the_agent_running() {
        let mut c = ResidentChat::new();
        c.window(vec![c.project(), c.chat(1, ""), c.message(1, "first")])
            .await;
        c.writes();
        let uuid = c.session().lock().unwrap().unconsumed[0].uuid.clone();
        c.reader_raw(
            &json!({
                "type": "user", "uuid": uuid, "isReplay": true,
                "message": { "role": "user", "content": "first" }
            })
            .to_string(),
        )
        .await;
        assert_eq!(turn_running_writes(&c.writes()), vec![true]);

        c.reader_raw(r#"{"type":"system","subtype":"task_started","task_id":"t1","description":"build","task_type":"local_bash"}"#).await;
        c.writes();
        c.reader_raw(RESULT_LINE).await;
        let writes = c.writes();
        assert_eq!(turn_running_writes(&writes), vec![false]);
        let result_row = writes
            .iter()
            .position(|w| matches!(w, WriteEvent::PutMessage { .. }))
            .expect("the result row");
        let edge = writes
            .iter()
            .position(|w| matches!(w, WriteEvent::ChatTurnRunning { .. }))
            .expect("the edge");
        assert!(result_row < edge, "{writes:?}");
        assert!(c.supervisor.is_running(&c.chat_id), "the task is armed");

        c.reader_raw(r#"{"type":"system","subtype":"task_updated","task_id":"t1","patch":{"status":"completed"}}"#).await;
        c.reader_line(vec![adapter::AgentEvent::TurnStarted("woken".into())])
            .await;
        assert_eq!(turn_running_writes(&c.writes()), vec![true]);
        c.reader_raw(RESULT_LINE).await;
        assert_eq!(turn_running_writes(&c.writes()), vec![false]);
    }

    /// The reader captures a turn's end as it reduces the `result` and sends it
    /// after that line's frames. A message the main loop queues in between
    /// raises `turn_running` at the enqueue edge, and the captured false edge
    /// still reaches the backend, which releases queued messages on it: false,
    /// then true.
    #[tokio::test]
    async fn a_turn_end_overtaken_by_the_next_turn_still_reaches_the_backend() {
        let mut c = ResidentChat::new();
        c.window(vec![c.project(), c.chat(1, ""), c.message(1, "first")])
            .await;
        c.consume_first_turn();
        c.reader_line(vec![adapter::AgentEvent::ContextTokens(1)])
            .await;
        assert_eq!(turn_running_writes(&c.writes()), vec![true]);

        let session = c.session();
        let ended = {
            let mut st = session.lock().unwrap();
            let result = adapter::AgentEvent::Result {
                origin_is_task: false,
            };
            crate::agent::reduce(&mut st, &result);
            st.observe_turn_running().expect("the turn ended")
        };
        c.window(vec![c.chat(2, ""), c.message(2, "second")]).await;
        assert_eq!(turn_running_writes(&c.writes()), vec![true]);
        c.respond(AgentResponse::TurnRunning {
            topic: c.chat_id.clone(),
            session: session.clone(),
            running: ended,
        })
        .await;
        assert_eq!(turn_running_writes(&c.writes()), vec![false, true]);
        assert_eq!(
            c.session().lock().unwrap().observe_turn_running(),
            None,
            "the dedup holds what was written last"
        );

        // A busy edge whose turn is over by now is dropped: its false edge follows.
        let started = {
            let mut st = session.lock().unwrap();
            st.unconsumed.clear();
            st.observe_turn_running().expect("the queued turn is gone");
            st.turn_in_flight = true;
            let started = st.observe_turn_running().expect("a turn started");
            st.turn_in_flight = false;
            started
        };
        c.respond(AgentResponse::TurnRunning {
            topic: c.chat_id.clone(),
            session: session.clone(),
            running: started,
        })
        .await;
        assert!(turn_running_writes(&c.writes()).is_empty());

        // Edges from a session a teardown replaced are dropped, a pulse included.
        c.window(vec![c.chat(3, ""), c.message(3, "/stop")]).await;
        c.window(vec![c.chat(4, ""), c.message(4, "third")]).await;
        c.writes();
        for running in [true, false] {
            c.respond(AgentResponse::TurnRunning {
                topic: c.chat_id.clone(),
                session: session.clone(),
                running,
            })
            .await;
        }
        assert!(turn_running_writes(&c.writes()).is_empty());
    }

    /// A per-turn adapter's turn is its whole run, and every spawn says so.
    #[tokio::test]
    async fn a_one_shot_spawn_writes_turn_running() {
        let mut c = ResidentChat::new();
        let cursor = |c: &ResidentChat, seq| chat_with(c, seq, json!({ "agent_kind": "cursor" }));
        c.window(vec![c.project(), cursor(&c, 1), c.message(1, "first")])
            .await;
        assert_eq!(turn_running_writes(&c.writes()), vec![true]);
    }

    /// A scheduled slash command landing mid-turn is queued behind the turn; the
    /// same command typed live cuts it.
    #[tokio::test]
    async fn a_scheduled_slash_command_never_cuts_a_running_turn() {
        let mut c = ResidentChat::new();
        c.window(vec![c.project(), c.chat(1, ""), c.message(1, "first")])
            .await;
        c.consume_first_turn();
        c.writes();

        let scheduled = put_with(c.message(2, "/compact"), json!({ "auto_sent": 1 }));
        c.window(vec![c.chat(2, ""), scheduled]).await;
        assert_eq!(c.spawned_prompts(), vec!["first"], "no respawn");
        assert_eq!(c.unconsumed(), vec!["/compact"]);
        let writes = c.writes();
        assert!(!writes.iter().any(is_interrupted), "{writes:?}");

        c.window(vec![c.chat(3, ""), c.message(3, "/compact")])
            .await;
        assert_eq!(c.spawned_prompts(), vec!["first", "/compact"]);
        assert!(c.writes().iter().any(is_interrupted));
    }

    /// A prune hard-restarts the resident, and `--resume` re-arms nothing: the
    /// respawn prompt NAMES what it terminated and the column is cleared.
    #[tokio::test]
    async fn a_prune_restart_names_the_terminated_tasks_and_clears_them() {
        let mut c = ResidentChat::new();
        c.window(vec![c.project(), c.chat(1, ""), c.message(1, "first")])
            .await;
        c.consume_first_turn();
        c.respond(AgentResponse::SessionIdHarvested {
            topic: c.chat_id.clone(),
            session_id: Uuid::now_v7().to_string(),
        })
        .await;
        c.reader_raw(r#"{"type":"system","subtype":"task_started","task_id":"t-7","description":"deploy to prod","task_type":"local_bash"}"#).await;
        c.writes();

        let transcript = crate::prune::test_util::write_jsonl(&[
            r#"{"type":"assistant","message":{"content":[{"type":"tool_use","id":"t1","name":"Read","input":{"file_path":"junk.rs"}}]},"uuid":"a1","parentUuid":null}"#,
            r#"{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"t1","content":"body"}]},"uuid":"u1","parentUuid":"a1"}"#,
        ]);
        let reqs = vec![prune::PruneRequest {
            jsonl_path: transcript.path().to_path_buf(),
            agent_kind: AgentKind::Claude,
            tool_name: "Read".into(),
            needle: "junk.rs".into(),
            reason: "done with it".into(),
        }];
        let (chat_id, write_tx) = (c.chat_id.clone(), c.write_tx.clone());
        let shared = c.shared_mirror();
        apply_prune_group(&chat_id, reqs, &shared, &mut c.supervisor, &write_tx).await;

        let prompt = c.recorded.lock().unwrap().last().unwrap().prompt.clone();
        assert!(
            prompt.contains("terminated 1 background task(s)/monitor(s)")
                && prompt.contains("deploy to prod"),
            "the respawn prompt must name the killed task, got: {prompt}"
        );
        assert_eq!(
            live_task_blobs(&c.writes()),
            vec![None],
            "the restart clears the armed-task list"
        );
    }

    /// Armed tasks die with the process, so every teardown clears the column —
    /// here the spontaneous exit (`Done`) and a `/kill`.
    #[tokio::test]
    async fn teardown_clears_the_live_task_list() {
        async fn armed() -> ResidentChat {
            let mut c = ResidentChat::new();
            c.window(vec![c.project(), c.chat(1, ""), c.message(1, "first")])
                .await;
            c.consume_first_turn();
            c.reader_raw(r#"{"type":"system","subtype":"task_started","task_id":"t-7","description":"deploy","task_type":"local_bash"}"#).await;
            assert_eq!(live_task_blobs(&c.writes()).len(), 1, "armed");
            c
        }

        // The process exited on its own with the task still armed: its reader
        // closes the session, then sends `Done`.
        let mut c = armed().await;
        c.session().lock().unwrap().closed = true;
        c.respond(AgentResponse::Done {
            topic: c.chat_id.clone(),
            has_result: false,
        })
        .await;
        assert_eq!(
            live_task_blobs(&c.writes()),
            vec![None],
            "a spontaneous exit clears the column"
        );

        // A hard abort (`/kill`) does the same.
        let mut c = armed().await;
        c.window(vec![c.chat(2, ""), c.message(2, "/kill")]).await;
        assert_eq!(
            live_task_blobs(&c.writes()),
            vec![None],
            "a hard abort clears the column"
        );

        // The spawner is going down (SIGTERM / update): the agents are
        // cancelled, not finished, so no `Done` arm can clear the column.
        let mut c = armed().await;
        let mirror = c.shared_mirror();
        shutdown_agents(&mut c.supervisor, &mut c.resp_rx, &mirror, &c.write_tx).await;
        assert_eq!(
            live_task_blobs(&c.writes()),
            vec![None],
            "shutdown clears the column"
        );
    }

    fn park_prune(prunes: &prune::PendingPrunes, chat_id: &str) {
        prunes.try_lock().unwrap().insert(
            chat_id.to_string(),
            vec![prune::PruneRequest {
                jsonl_path: "/nonexistent.jsonl".into(),
                agent_kind: AgentKind::Claude,
                tool_name: "Read".into(),
                needle: "x".into(),
                reason: "r".into(),
            }],
        );
    }

    #[tokio::test]
    async fn a_prune_cue_after_kill_does_not_resurrect_the_agent() {
        let mut c = ResidentChat::new();
        c.window(vec![c.project(), c.chat(1, ""), c.message(1, "first")])
            .await;
        c.consume_first_turn();
        let prunes = c.supervisor.pending_prunes();
        park_prune(&prunes, &c.chat_id);
        c.window(vec![c.chat(2, ""), c.message(2, "/kill")]).await;
        assert!(prunes.lock().await.is_empty());

        c.mirror.set_agent_session_id(&c.chat_id, "sid".into());
        let mirror = c.shared_mirror();
        let topic = c.chat_id.clone();
        for resp in [
            AgentResponse::ToolResult {
                topic: topic.clone(),
            },
            AgentResponse::RunState {
                topic,
                running: false,
            },
        ] {
            route_agent_response(resp, &prunes, &mirror, &mut c.supervisor, &c.write_tx).await;
        }
        assert_eq!(c.spawned_prompts(), vec!["first"]);
        assert!(!c.supervisor.is_running(&c.chat_id));
    }

    /// The control task parks a `prune-context` RPC that raced `/kill` after
    /// `abort_agent` cleared the table.
    #[tokio::test]
    async fn a_prune_parked_after_kill_does_not_resurrect_the_agent() {
        let mut c = ResidentChat::new();
        c.window(vec![c.project(), c.chat(1, ""), c.message(1, "first")])
            .await;
        c.consume_first_turn();
        c.window(vec![c.chat(2, ""), c.message(2, "/kill")]).await;

        c.mirror.set_agent_session_id(&c.chat_id, "sid".into());
        let prunes = c.supervisor.pending_prunes();
        let mirror = c.shared_mirror();
        let topic = c.chat_id.clone();
        for resp in [
            AgentResponse::ToolResult {
                topic: topic.clone(),
            },
            AgentResponse::RunState {
                topic: topic.clone(),
                running: false,
            },
        ] {
            park_prune(&prunes, &topic);
            route_agent_response(resp, &prunes, &mirror, &mut c.supervisor, &c.write_tx).await;
            assert!(prunes.lock().await.is_empty());
        }
        assert_eq!(c.spawned_prompts(), vec!["first"]);
    }

    /// A prune cue arriving after `/stop` (a late RPC) must not restart the
    /// stopped chat.
    #[tokio::test]
    async fn a_prune_of_a_stopped_turn_does_not_restart_the_chat() {
        let mut c = ResidentChat::new();
        c.window(vec![c.project(), c.chat(1, ""), c.message(1, "first")])
            .await;
        c.consume_first_turn();
        c.mirror.set_agent_session_id(&c.chat_id, "sid".into());
        let prunes = c.supervisor.pending_prunes();
        park_prune(&prunes, &c.chat_id);
        c.window(vec![c.chat(2, ""), c.message(2, "/stop")]).await;
        assert!(prunes.lock().await.is_empty());

        let mirror = c.shared_mirror();
        let topic = c.chat_id.clone();
        park_prune(&prunes, &topic);
        let cue = AgentResponse::ToolResult {
            topic: topic.clone(),
        };
        route_agent_response(cue, &prunes, &mirror, &mut c.supervisor, &c.write_tx).await;
        assert!(prunes.lock().await.is_empty());
        assert_eq!(c.spawned_prompts(), vec!["first"]);
        assert!(!c.supervisor.has_agent(&topic));
    }

    #[tokio::test]
    async fn a_fresh_spawn_drops_a_prune_parked_after_kill() {
        let mut c = ResidentChat::new();
        c.window(vec![c.project(), c.chat(1, ""), c.message(1, "first")])
            .await;
        c.consume_first_turn();
        c.window(vec![c.chat(2, ""), c.message(2, "/kill")]).await;
        let prunes = c.supervisor.pending_prunes();
        park_prune(&prunes, &c.chat_id);
        c.window(vec![c.chat(3, ""), c.message(3, "second")]).await;

        assert_eq!(c.spawned_prompts(), vec!["first", "second"]);
        assert!(prunes.lock().await.is_empty());
    }

    #[tokio::test(start_paused = true)]
    async fn shutdown_routes_the_frames_of_a_reader_blocked_on_a_full_queue() {
        let mut c = ResidentChat::new();
        let (resp_tx, resp_rx) = mpsc::channel::<AgentResponse>(1);
        let resident_fn: ResidentSpawnFn = StdArc::new(|req, tx, cancel, _pending| {
            let reader = tokio::spawn(async move {
                cancel.cancelled().await;
                for n in 0..3 {
                    let content = format!(r#"{{"type":"assistant","n":{n}}}"#);
                    let topic = req.chat_id.clone();
                    let _ = tx.send(AgentResponse::Line { topic, content }).await;
                }
            });
            ResidentHandles {
                wake: StdArc::new(tokio::sync::Notify::new()),
                reader,
                state: StdArc::new(StdMutex::new(SessionState::default())),
            }
        });
        let one_shot: SpawnFn = StdArc::new(|_req, _tx, _token, _pending| tokio::spawn(async {}));
        c.supervisor = Supervisor::with_spawn_fns(resp_tx, one_shot, resident_fn);
        c.resp_rx = resp_rx;
        c.window(vec![c.project(), c.chat(1, ""), c.message(1, "first")])
            .await;
        c.writes();

        let mirror = c.shared_mirror();
        shutdown_agents(&mut c.supervisor, &mut c.resp_rx, &mirror, &c.write_tx).await;

        let writes = c.writes();
        let lines = writes
            .iter()
            .filter(|w| matches!(w, WriteEvent::PutMessage { .. }))
            .count();
        assert_eq!(lines, 3, "{writes:?}");
    }

    /// The reader reduces a turn's `result` (so `any_running` goes false and
    /// the update proceeds) before it forwards the line's frames and idle edge.
    #[tokio::test]
    async fn shutdown_publishes_the_final_frames_still_queued() {
        let mut c = ResidentChat::new();
        c.window(vec![c.project(), c.chat(1, ""), c.message(1, "first")])
            .await;
        c.writes();
        park_prune(&c.supervisor.pending_prunes(), &c.chat_id);
        let topic = c.chat_id.clone();
        for resp in [
            AgentResponse::ToolResult {
                topic: topic.clone(),
            },
            AgentResponse::Line {
                topic: topic.clone(),
                content: r#"{"type":"result"}"#.into(),
            },
            AgentResponse::RunState {
                topic,
                running: false,
            },
        ] {
            c.resp_tx.send(resp).await.unwrap();
        }

        c.mirror.set_agent_session_id(&c.chat_id, "sid".into());
        let mirror = c.shared_mirror();
        shutdown_agents(&mut c.supervisor, &mut c.resp_rx, &mirror, &c.write_tx).await;

        assert_eq!(c.spawned_prompts(), vec!["first"], "no prune respawn");
        let writes = c.writes();
        assert!(
            matches!(
                writes[..],
                [
                    WriteEvent::PutMessage { .. },
                    WriteEvent::ChatRunning {
                        agent_running: false,
                        ..
                    },
                    WriteEvent::ChatAlive {
                        agent_alive: false,
                        ..
                    },
                    WriteEvent::ChatLiveTasks { tasks: None, .. },
                ]
            ),
            "{writes:?}"
        );
    }

    #[tokio::test]
    async fn respawns_do_not_flicker_the_alive_flag() {
        let mut c = ResidentChat::new();
        c.window(vec![c.project(), c.chat(1, ""), c.message(1, "first")])
            .await;
        assert_eq!(alive_writes(&c.writes()), vec![true]);
        c.window(vec![c.chat(2, "opus"), c.message(2, "second")])
            .await;
        assert_eq!(c.recorded.lock().unwrap().len(), 2, "respawned");
        let writes = c.writes();
        assert!(alive_writes(&writes).is_empty());
        assert_eq!(
            live_task_blobs(&writes),
            vec![None],
            "the respawn re-arms nothing, so the task list is cleared"
        );

        c.respond(AgentResponse::Done {
            topic: c.chat_id.clone(),
            has_result: true,
        })
        .await;
        assert!(
            alive_writes(&c.writes()).is_empty(),
            "late Done, live session"
        );
    }

    /// The mirror as the next spawner process loads it.
    fn restarted(mirror: &Mirror) -> Mirror {
        let path = std::env::temp_dir().join(format!("zucchini-state-{}.json", Uuid::now_v7()));
        mirror.save(&path).unwrap();
        let loaded = Mirror::load(&path);
        let _ = std::fs::remove_file(&path);
        loaded
    }

    /// The `chats.live_tasks` payloads in arrival order (`None` = a clear).
    fn live_task_blobs(writes: &[WriteEvent]) -> Vec<Option<String>> {
        writes
            .iter()
            .filter_map(|w| match w {
                WriteEvent::ChatLiveTasks { tasks, .. } => Some(tasks.clone()),
                _ => None,
            })
            .collect()
    }

    fn live_tasks_reset_chats(writes: &[WriteEvent]) -> std::collections::HashSet<&str> {
        writes
            .iter()
            .filter_map(|w| match w {
                WriteEvent::ChatLiveTasks {
                    chat_id,
                    tasks: None,
                    ..
                } => Some(chat_id.as_str()),
                _ => None,
            })
            .collect()
    }

    fn alive_reset_chats(writes: &[WriteEvent]) -> std::collections::HashSet<&str> {
        writes
            .iter()
            .filter_map(|w| match w {
                WriteEvent::ChatAlive {
                    chat_id,
                    agent_alive: false,
                } => Some(chat_id.as_str()),
                _ => None,
            })
            .collect()
    }

    #[tokio::test]
    async fn boot_resets_stale_run_flags_before_spawning() {
        let mut c = ResidentChat::new();
        let (project_id, user_id) = (c.project_id.clone(), c.user_id);
        let row = |flag: i64| {
            json!({
                "project_id": project_id,
                "user_id": user_id.to_string(),
                "agent_running": flag,
                "agent_alive": flag,
            })
            .to_string()
        };
        // Flagged before the restart: persisted, not re-streamed.
        let persisted = Uuid::now_v7().to_string();
        c.mirror
            .upsert_chat(persisted.clone(), &row(1), &KeyStore::new());
        c.mirror = restarted(&c.mirror);

        let restreamed = SyncEvent::Put {
            table: "chats".into(),
            id: c.chat_id.clone(),
            data: row(1),
        };
        c.window(vec![c.project(), restreamed, c.message(1, "first")])
            .await;
        let writes = c.writes();
        assert_eq!(
            alive_reset_chats(&writes),
            [persisted.as_str(), c.chat_id.as_str()].into()
        );
        assert_eq!(
            live_tasks_reset_chats(&writes),
            [persisted.as_str(), c.chat_id.as_str()].into(),
            "boot clears phantom tasks too — PowerSync won't re-stream these rows"
        );
        assert_eq!(writes.iter().filter(|w| is_stopped_by_user(w)).count(), 2);
        assert_eq!(
            alive_writes(&writes).last(),
            Some(&true),
            "the spawn publishes after the reset"
        );

        c.window(vec![c.chat(1, "")]).await;
        assert!(c.writes().is_empty(), "once per boot");

        // Restarted before the reset synced back: reset again.
        c.mirror = restarted(&c.mirror);
        c.window(vec![]).await;
        assert_eq!(alive_reset_chats(&c.writes()), [persisted.as_str()].into());

        let synced_back = SyncEvent::Put {
            table: "chats".into(),
            id: persisted.clone(),
            data: row(0),
        };
        c.window(vec![synced_back]).await;
        c.mirror = restarted(&c.mirror);
        c.window(vec![]).await;
        assert!(alive_reset_chats(&c.writes()).is_empty());
    }

    /// A teardown that missed its clear leaves a task blob on a row whose run
    /// flags are already false — boot recovery has to pick that up too, or the
    /// chat shows phantom tasks forever (PowerSync never re-streams the row).
    #[tokio::test]
    async fn boot_resets_a_chat_left_with_tasks_but_no_run_flags() {
        let mut c = ResidentChat::new();
        let idle_with_tasks = json!({
            "project_id": c.project_id,
            "user_id": c.user_id.to_string(),
            "agent_running": 0,
            "agent_alive": 0,
            "live_tasks": "cipher",
        })
        .to_string();
        let chat_id = Uuid::now_v7().to_string();
        c.mirror
            .upsert_chat(chat_id.clone(), &idle_with_tasks, &KeyStore::new());
        c.mirror = restarted(&c.mirror);

        c.window(vec![]).await;
        let writes = c.writes();
        assert_eq!(live_tasks_reset_chats(&writes), [chat_id.as_str()].into());
    }

    #[tokio::test]
    async fn boot_keeps_a_watched_chat_running_with_its_watches_listed() {
        let mut c = ResidentChat::new();
        let was_in_a_turn = json!({
            "project_id": c.project_id,
            "user_id": c.user_id.to_string(),
            "agent_running": 1,
            "live_tasks": "cipher",
        })
        .to_string();
        c.mirror
            .upsert_chat(c.chat_id.clone(), &was_in_a_turn, &c.keys);
        let watch = watch_in(&c);
        c.mirror = restarted(&c.mirror);

        c.window(vec![]).await;
        let writes = c.writes();
        assert_eq!(run_flags(&writes), vec![("turn", false)]);
        assert_eq!(live_task_ids(&writes), vec![vec![watch]]);
    }
}
