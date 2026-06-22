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
mod power;
mod powersync;
mod prune;
mod shell;
mod state;
mod uninstall;
mod updater;
mod writer;
mod x25519;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU8, Ordering};
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
/// into the sync-event handler so `seed_initial_agents_if_pending` can decide
/// which compatibility-seeded agents are installed without re-shelling. `OnceLock` lets
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
                let state_changed =
                    apply_machine_users_put(&id, user_id, &row, x25519_secret, keys, mirror);

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
                SyncEventOutcome::StateChanged
            }
            _ => SyncEventOutcome::Nothing,
        },
        SyncEvent::CheckpointComplete { buckets } => {
            // Apply the staged window in dependency order: projects → chats →
            // messages, so chat.project_id and message.chat resolve against a
            // now-consistent mirror. Then advance the persisted cursor.

            // 1. projects
            for (id, row) in staging.projects.drain() {
                match row {
                    StagedRow::Put(json) => mirror.upsert_project(id, &json),
                    StagedRow::Remove => mirror.remove_project(&id),
                }
            }

            // 2. chats (with agent_session_id restore — replaces the old inline
            //    merge-on-upsert hack in state.rs::upsert_chat)
            for (id, row) in staging.chats.drain() {
                match row {
                    StagedRow::Put(json) => {
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
                        mirror.upsert_chat(id.clone(), &json);
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
                    }
                    StagedRow::Remove => mirror.remove_chat(&id),
                }
            }

            // 3. messages — spawn against the now-consistent mirror, arrival order
            let pending_messages = std::mem::take(&mut staging.messages);
            for raw in pending_messages {
                handle_message_put(&raw, mirror, supervisor, blobs, keys, write_tx).await;
            }

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
            // busy→idle flip.
            send_run_state(write_tx, topic.clone(), false).await;
            supervisor.remove(&topic);
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
async fn send_run_state(write_tx: &mpsc::Sender<WriteEvent>, chat_id: String, running: bool) {
    let _ = write_tx
        .send(WriteEvent::chat_running(chat_id, running))
        .await;
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
                c.model
                    .as_deref()
                    .filter(|s| !s.is_empty())
                    .map(str::to_string),
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

    // Resident adapters (claude) host one process across a turn PLUS its
    // in-process background tasks/Monitors; one-shot adapters (codex/gemini/
    // cursor/hermes) spawn per turn. There is NO warm reuse: a new message
    // hard-aborts any in-flight turn (SIGTERM the group — also kills armed
    // monitors) then respawns a FRESH `--resume` process below.
    let is_resident = agent_kind.make_adapter().is_resident();
    let was_running = supervisor.is_running(&chat_id);

    if is_resident {
        if is_stop {
            // RESIDENT /stop is an explicit HARD stop: SIGTERM the process
            // group, killing claude AND any armed background tasks/monitors, so
            // the chat returns to Idle. (A new message below also hard-aborts —
            // /stop differs only in that it returns without respawning.)
            // `abort_agent` on a resident session emits NO `Done`, so we clear
            // the run state explicitly. The next message respawns with `--resume`.
            if was_running {
                info!(chat_id = %chat_id, "stop: tearing down resident session (kills armed monitors)");
                publish_interrupted(write_tx, &chat_id, user_id).await;
                supervisor.abort_agent(&chat_id).await;
                let _ = write_tx
                    .send(WriteEvent::chat_stopped_by_user(chat_id.clone()))
                    .await;
            }
            return;
        }
        // RESIDENT new message ("Send now (Interrupt)"). Hard-abort the running
        // turn (SIGTERM the process group — also kills armed monitors), publish
        // INTERRUPTED_RESULT, then fall through to spawn a FRESH `--resume`
        // process below. `abort_agent` emits NO `Done`, so we clear the run state
        // explicitly (mirrors the /stop arm above). Only abort when actually
        // running — an Idle/torn-down session has nothing to interrupt and we
        // must NOT abort a Waiting session's armed monitors prematurely.
        if was_running {
            info!(chat_id = %chat_id, "interrupt-then-send: aborting in-flight resident turn, will respawn fresh");
            publish_interrupted(write_tx, &chat_id, user_id).await;
            supervisor.abort_agent(&chat_id).await;
            send_run_state(write_tx, chat_id.clone(), false).await;
        }
    } else {
        // ONE-SHOT path (historic): abort any running agent + publish
        // INTERRUPTED_RESULT, then either return (/stop) or spawn a fresh turn.
        if was_running {
            info!(chat_id = %chat_id, "aborting running one-shot agent before handling new message");
            supervisor.abort_agent(&chat_id).await;
            publish_interrupted(write_tx, &chat_id, user_id).await;
        }
        if is_stop {
            let _ = write_tx
                .send(WriteEvent::chat_stopped_by_user(chat_id.clone()))
                .await;
            return;
        }
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
    };

    if is_resident {
        // Always (re)spawn a FRESH `claude --resume <agent_session_id>` process —
        // no warm reuse. Any in-flight turn was already hard-aborted above (the
        // `was_running` interrupt-then-send path); `spawn_resident` drops any
        // stale entry and spawns fresh, threading the harvested session id (from
        // the mirror, in `SpawnRequest.agent_session_id`) into `--resume`. NO
        // optimistic agent_running flip — the reader's RunState drives it.
        supervisor.spawn_agent(req);
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
        supervisor.spawn_agent(req);
    }
}

/// Per-chat spawn parameters resolved from the `Mirror`, shared between the
/// message-put path and the prune-context respawn. `agent_session_id` is `Some`
/// for any chat that's had at least one turn (always true for a prune request —
/// it's issued from inside a turn).
struct ChatSpawnParams {
    project_path: String,
    worktree: bool,
    agent_session_id: Option<String>,
    agent_kind: AgentKind,
    is_sandboxed: bool,
    model: Option<String>,
    /// Sender's IANA timezone (migration 0040), mirrored per `handle_message_put`
    /// so the prune respawn keeps injecting the current-local-time line.
    user_timezone: Option<String>,
}

/// Resolve spawn params for `chat_id`, mirroring `handle_message_put`'s gates
/// (project-path resolution, owner-vs-member sandbox). `None` (with a logged
/// reason) if the chat/project isn't synced or a non-owner's `machine_users`
/// row hasn't landed.
fn resolve_chat_spawn_params(mirror: &Mirror, chat_id: &str) -> Option<ChatSpawnParams> {
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
    Some(ChatSpawnParams {
        project_path,
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
    })
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
    let killed_tasks = supervisor.live_task_count(chat_id);

    // Kill the (now-idle-at-a-tool-boundary) agent before touching its transcript:
    // the rewrite needs exclusive access and the resume must re-read from disk. A
    // no-op if the turn already exited on its own.
    supervisor.abort_agent(chat_id).await;

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
                    .map(|r| (r.tool_name.clone(), r.needle.clone()))
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
    if killed_tasks > 0 {
        prompt.push_str(&format!(
            " NOTE: this prune restarted the session and terminated {killed_tasks} \
             background task(s)/monitor(s) that were running — they are NOT \
             automatically resumed. Re-arm or re-run any you still need, and check \
             the status of anything that was making changes (e.g. a deploy/build)."
        ));
    }

    // Resolve spawn params from the mirror (read guard) and respawn.
    let params = {
        let g = mirror.read().await;
        resolve_chat_spawn_params(&g, chat_id)
    };
    let Some(params) = params else {
        error!(chat_id = %chat_id, "cannot resolve spawn params after prune; chat left idle");
        return;
    };
    if params.agent_session_id.is_none() {
        // A prune request only fires mid-turn, so the session id is already
        // harvested — but guard anyway: resuming requires it, and a fresh session
        // would re-run from scratch with the wrong prompt.
        error!(chat_id = %chat_id, "no agent_session_id after prune; cannot resume, chat left idle");
        return;
    }

    supervisor.spawn_agent(SpawnRequest {
        chat_id: chat_id.to_string(),
        prompt,
        project_path: Some(params.project_path),
        worktree: params.worktree,
        agent_session_id: params.agent_session_id,
        agent_kind: params.agent_kind,
        is_sandboxed: params.is_sandboxed,
        model: params.model,
        user_timezone: params.user_timezone,
    });
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

/// Default a parsed `--chat-id` from the `ZUCCHINI_CHAT_ID` env var (agent.rs
/// exports it on every spawn); an explicit flag still wins. Shared so all
/// subcommands resolve `--chat-id` identically.
fn chat_id_or_env(chat_id: Option<String>) -> Option<String> {
    chat_id.or_else(|| std::env::var("ZUCCHINI_CHAT_ID").ok())
}

/// CLI entry point for `zucchini-spawner attach-file <abs-path> [--chat-id <UUID>]`
/// (`--chat-id` defaults to the `ZUCCHINI_CHAT_ID` env var exported on the spawn).
///
/// Parses argv (hand-rolled — clap is overkill for one flag and a positional
/// arg), connects to the daemon's `~/.zucchini-spawner/control.sock`, runs
/// one `attach_file` RPC, and prints a human-readable result. Exits 0 on
/// success, 1 on any failure. The CLI itself does no crypto or HTTP — the
/// daemon owns the JWT and K_user.
async fn run_attach_file_cli(args: &[String]) {
    fn usage_and_exit() -> ! {
        eprintln!("usage: zucchini-spawner attach-file <absolute-path> [--chat-id <UUID>]\n  --chat-id defaults to $ZUCCHINI_CHAT_ID (exported on every spawn).");
        std::process::exit(2);
    }

    let mut chat_id: Option<String> = None;
    let mut path: Option<String> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let s = a.as_str();
        if s == "-h" || s == "--help" {
            usage_and_exit();
        } else if let Some(v) = take_flag(s, "--chat-id", &mut it) {
            chat_id = Some(v);
        } else if path.is_none() {
            path = Some(s.to_string());
        } else {
            usage_and_exit();
        }
    }
    let chat_id = chat_id_or_env(chat_id);
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

/// Parse `prune-context` CLI args into the call-level `--chat-id` (if given) plus
/// the batch of prune targets. `Err(())` signals any structural problem (so the
/// caller prints usage + exits 2). Pure (no env, no I/O, no process exit) so the
/// batching/terminator semantics are unit-testable.
///
/// Each `--summary` (or `--reason` alias) CLOSES the current target: a target is
/// the run of `[--tool-name] --args` flags before it. Per target, `--args` is
/// required by PRESENCE (`--args ""` is the no-args selector); `--tool-name` is
/// optional (`""` = any-tool selector). A `--summary` with no preceding `--args`,
/// a dangling `--tool-name`/`--args` not closed by a `--summary`, an unknown flag,
/// `-h`/`--help`, or an empty batch are all errors. `--flag=value` accepted.
fn parse_prune_args(
    args: &[String],
) -> Result<(Option<String>, bool, Vec<control::PruneItem>), ()> {
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

    let mut chat_id: Option<String> = None;
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
        } else if let Some(v) = take_flag(s, "--chat-id", &mut it) {
            chat_id = Some(v);
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
    Ok((chat_id, force, items))
}

/// CLI entry point for `zucchini-spawner prune-context [--tool-name <Tool>]
/// --args "<glob>" --summary "<digest>" [... more triples ...] [--chat-id <UUID>]`.
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
/// (terminates the target). `--chat-id` is call-level, defaults to
/// `ZUCCHINI_CHAT_ID`. `--flag=value` accepted. `--args` globs argument values
/// and blanks only the most recent match (output reports how many remain). The
/// daemon aborts the agent right after replying, so a post-send connection-reset
/// is expected — the prune proceeds regardless.
async fn run_prune_context_cli(args: &[String]) {
    fn usage_and_exit() -> ! {
        eprintln!(
            "usage: zucchini-spawner prune-context [--tool-name <ToolName>] --args \"<glob>\" --summary \"<digest>\" [... repeat per output ...] [--chat-id <UUID>] [--force]\n  Prune one or more tool outputs in a single call: each --summary CLOSES the target made of the --tool-name/--args before it, so repeat the triple to forget several outputs at once (one restart for the whole batch). --tool-name is optional per target — omit it to match on --args alone. --summary is the takeaway from that output you still need going forward — the slice that matters for the task at hand, NOT a recap of the whole output (--reason accepted as a legacy alias). --args is a glob (supports *) over the call's argument VALUES, not key names; blanks only the most recent matching call (repeat to prune older ones). --chat-id is call-level and defaults to $ZUCCHINI_CHAT_ID (exported on every spawn); use --args \"\" to prune a call you made with no arguments. --force prunes even while background tasks/monitors are running (the restart terminates them); without it, prune-context errors so you can wait for them to finish."
        );
        std::process::exit(2);
    }

    // `--reason` is the legacy alias for `--summary` (older binary mid-turn keeps
    // parsing across a hot-reload); both feed the same wire field `reason`.
    let Ok((chat_id, force, items)) = parse_prune_args(args) else {
        usage_and_exit();
    };
    // Default `--chat-id` from `ZUCCHINI_CHAT_ID` (inherited from the agent
    // subprocess); explicit flag still wins.
    let Some(chat_id) = chat_id.or_else(|| std::env::var("ZUCCHINI_CHAT_ID").ok()) else {
        usage_and_exit();
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
                // frame, i.e. strictly AFTER claude has persisted our stdout
                // summary to the transcript. That ordering is the whole point: the
                // resumed agent then sees its own prune call + this summary in
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

/// `schedule-message` subcommand. Twin of `run_attach_file_cli`: hand-rolled
/// argv parse, one RPC over the control socket, human-readable result, exit
/// 0/1. The daemon owns K_user + the JWT; this CLI only forwards the request.
///
/// `--at <local-datetime>` is required. Forwarded raw, zoned + validated
/// daemon-side by `control::normalize_deliver_at` (naive local anchored in the
/// user's tz, which only the daemon holds; unparseable values rejected so a
/// garbage timestamp can't misfire). The queue-when-free path (`deliver_at` null)
/// is composer-only — no use for the agent, which already runs at turn end.
async fn run_schedule_message_cli(args: &[String]) {
    fn usage_and_exit() -> ! {
        eprintln!(
            "usage: zucchini-spawner schedule-message --chat-id <UUID> --body <TEXT> --at <local-datetime, e.g. 2026-06-07T09:00:00>"
        );
        std::process::exit(2);
    }

    let mut chat_id: Option<String> = None;
    let mut body: Option<String> = None;
    let mut at: Option<String> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let s = a.as_str();
        if s == "-h" || s == "--help" {
            usage_and_exit();
        } else if let Some(v) = take_flag(s, "--chat-id", &mut it) {
            chat_id = Some(v);
        } else if let Some(v) = take_flag(s, "--body", &mut it) {
            body = Some(v);
        } else if let Some(v) = take_flag(s, "--at", &mut it) {
            at = Some(v);
        } else {
            usage_and_exit();
        }
    }
    // `--chat-id` defaults to $ZUCCHINI_CHAT_ID like the other subcommands (the
    // agent always passes it explicitly, so this is just consistency).
    let chat_id = chat_id_or_env(chat_id);
    let (Some(chat_id), Some(body), Some(at)) = (chat_id, body, at) else {
        usage_and_exit();
    };

    match control::schedule_message_via_socket(&chat_id, &body, Some(at)).await {
        Ok(id) => {
            // Stable, machine-parseable first line so the agent can grep for
            // success; the row id follows for cross-referencing.
            println!("scheduled message {id} on chat {chat_id}");
            std::process::exit(0);
        }
        Err(e) => {
            eprintln!("schedule-message failed: {e:#}");
            std::process::exit(1);
        }
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
    supervisor.shutdown_all().await;
    heartbeat_cancel.cancel();
    let _ = wait_for_writer_idle(writer).await;
    uninstall::spawn_detached_cleanup();
}

/// Hold an exclusive `flock` on the pidfile `~/.zucchini-spawner/spawner.pid`
/// for the
/// daemon's whole lifetime so only ONE spawner can run against this runtime
/// dir. A second daemon — a stray `zucchini-spawner … | head` probe that fell
/// through to boot, a leftover dev `cargo run`, a double start — would join the
/// same PowerSync stream and spawn a *duplicate* agent for every user message,
/// so the chat shows two responses and two `[result]`s (the bug this guards).
///
/// `flock` is crash-safe: the kernel releases it when the holder exits, so
/// there's no stale-lock recovery. The lock fd is `O_CLOEXEC` (Rust std default)
/// so spawned agents don't inherit it and pin the lock past the daemon's death.
///
/// On contention this prints the running pid and exits. On any I/O failure
/// taking the lock it logs and continues unlocked (best-effort — never wedge
/// the service over a lock we couldn't acquire). The returned handle MUST be
/// kept alive by the caller; dropping it unlocks.
fn acquire_single_instance_lock() -> Option<std::fs::File> {
    use std::io::Write;
    use std::os::unix::io::AsRawFd;

    let path = zucchini_spawner_dir().join("spawner.pid");
    let file = match std::fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        // Never truncate on open: a contender that opens the file but fails the
        // flock below must not erase the holder's recorded pid. We clear+rewrite
        // it via `set_len` only after we actually hold the lock.
        .truncate(false)
        .open(&path)
    {
        Ok(f) => f,
        Err(e) => {
            warn!(error = %e, path = %path.display(), "could not open single-instance lock; continuing unlocked");
            return None;
        }
    };

    if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
        let err = std::io::Error::last_os_error();
        if err.raw_os_error() == Some(libc::EWOULDBLOCK) {
            let existing = std::fs::read_to_string(&path).unwrap_or_default();
            let pid = existing.trim();
            let pid = if pid.is_empty() { "unknown" } else { pid };
            eprintln!(
                "zucchini-spawner: another instance is already running (pid {pid}); exiting."
            );
            error!(pid = %pid, "another spawner instance already holds the lock; exiting");
            std::process::exit(0);
        }
        warn!(error = %err, "flock on single-instance lock failed; continuing unlocked");
        return None;
    }

    // Lock held — record our pid so a future contender can name us.
    let _ = file.set_len(0);
    let _ = (&file).write_all(format!("{}\n", std::process::id()).as_bytes());
    Some(file)
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
        if first == "schedule-message" {
            run_schedule_message_cli(&raw_args[1..]).await;
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
             subcommands: attach-file, prune-context, prune-reminder-hook, schedule-message, hermes-turn\n\
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

    // Single-instance guard. MUST stay bound for the whole daemon lifetime —
    // dropping the handle releases the lock. A second daemon would otherwise
    // join the same sync stream and spawn a duplicate agent for every user
    // message (two responses + two `[result]`s per chat). See
    // `acquire_single_instance_lock`.
    let _instance_lock = acquire_single_instance_lock();

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

    // Prune-context: the control task does the read-only lookup + transcript
    // pre-scan, then parks the request in this shared table so the main loop
    // (sole owner of `supervisor`) can abort the agent, rewrite the jsonl, and
    // respawn on the same session with a "continue" prompt. A shared lock (not a
    // channel) so the park is visible the instant the RPC returns — the apply cue
    // on `response_rx` can't race ahead of it. Coalescing several targets from one
    // batched call onto a single `Vec` entry folds the burst into ONE respawn.
    let pending_prunes: prune::PendingPrunes =
        Arc::new(tokio::sync::Mutex::new(std::collections::HashMap::new()));

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

    let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .expect("failed to register SIGTERM handler");
    let mut sigint = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())
        .expect("failed to register SIGINT handler");

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
                        // with nothing pending. The control task parked the
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
                        let reqs = pending_prunes.lock().await.remove(&topic);
                        if let Some(reqs) = reqs {
                            apply_prune_group(&topic, reqs, &mirror, &mut supervisor, &write_tx).await;
                        }
                    }
                    AgentResponse::RunState { topic, running } => {
                        // Safety net only: a prune is normally applied the instant its
                        // `ToolResult` cue fires (arm above). If that cue never arrived
                        // for a still-parked request and the session reaches Idle
                        // (running=false), drain + apply it here rather than leak the
                        // entry. Normal prunes don't reach this path. Then write the
                        // run-state column.
                        if !running {
                            let reqs = pending_prunes.lock().await.remove(&topic);
                            if let Some(reqs) = reqs {
                                apply_prune_group(&topic, reqs, &mirror, &mut supervisor, &write_tx).await;
                                // `apply_prune_group` respawned the session, which
                                // will emit its own RunState; don't also write the
                                // idle column from this (now-stale) transition.
                                continue 'main_loop;
                            }
                        }
                        {
                            let mut g = mirror.write().await;
                            handle_agent_response(
                                AgentResponse::RunState { topic: topic.clone(), running },
                                &mut g,
                                &write_tx,
                                &mut supervisor,
                            ).await;
                        }
                        // Idle sessions are not kept warm. `RunState{running:false}`
                        // is emitted only on a `Result` turn-boundary frame with no
                        // live background task (see the reader's emission gate), so
                        // this edge means the turn ended and nothing is in flight:
                        // tear the resident down to free its child process and let
                        // `is_empty()` go true (e.g. for a pending self-update). The
                        // next message respawns with `--resume`.
                        if !running {
                            supervisor.abort_agent(&topic).await;
                        }
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
                        handle_agent_response(other, &mut g, &write_tx, &mut supervisor).await;
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
    /// `SpawnRequest` into `recorded` and returns inert handles: a real (but
    /// never-drained) stdin channel, a `turn_in_flight=true` state, and a reader
    /// task that stays alive until its `cancel` token fires (so `is_running`
    /// sees a live session, and `abort_agent` can join it).
    /// This is the resident analogue of the one-shot recorder seam — claude is
    /// resident, so claude-flow tests route here, not through the one-shot fn.
    fn supervisor_with_resident_recorder(
        resp_tx: mpsc::Sender<AgentResponse>,
    ) -> (Supervisor, StdArc<StdMutex<Vec<SpawnRequest>>>) {
        let recorded: StdArc<StdMutex<Vec<SpawnRequest>>> = StdArc::new(StdMutex::new(Vec::new()));
        let rec = recorded.clone();
        let resident_fn: ResidentSpawnFn = StdArc::new(move |req, _tx, cancel, _pending| {
            rec.lock().unwrap().push(req);
            let (stdin_tx, _stdin_rx) = mpsc::unbounded_channel();
            let state = StdArc::new(StdMutex::new(SessionState {
                turn_in_flight: true,
                live_tasks: Default::default(),
            }));
            // Keep the stdin receiver alive for the reader's lifetime so stdin
            // sends don't error; the reader exits on cancel.
            let reader = tokio::spawn(async move {
                let _keep = _stdin_rx;
                cancel.cancelled().await;
            });
            ResidentHandles {
                stdin_tx,
                reader,
                state,
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
            2,
            "INTERRUPTED line + ChatRunning(false) (idle clears the run state)"
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
        match &events[1] {
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
        let (chat, force, items) = super::parse_prune_args(&sv(&[
            "--tool-name",
            "Read",
            "--args",
            "a.ts",
            "--summary",
            "kept x",
        ]))
        .expect("valid");
        assert!(chat.is_none());
        assert!(!force, "force defaults off");
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].tool_name, "Read");
        assert_eq!(items[0].needle, "a.ts");
        assert_eq!(items[0].reason, "kept x");
    }

    #[test]
    fn parse_prune_args_summary_terminates_each_target() {
        // Three targets in one call; `--chat-id` is call-level and can sit anywhere.
        let (chat, force, items) = super::parse_prune_args(&sv(&[
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
        assert_eq!(chat.as_deref(), Some("c1"));
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
        let (_, _, items) = super::parse_prune_args(&sv(&[
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

    /// Resident lifecycle: NO warm reuse. Every user turn spawns a FRESH
    /// `--resume` process, so two turns for the same chat record TWO spawns —
    /// the second carrying the harvested session id for `--resume`. Drives
    /// `Supervisor::spawn_agent` directly (it's the dispatch the message-put path
    /// calls) so the assertion is on the recorded session `SpawnRequest`s.
    #[tokio::test]
    async fn resident_each_turn_respawns_fresh_with_resume() {
        let chat_id = Uuid::now_v7().to_string();
        let (resp_tx, _resp_rx) = mpsc::channel::<AgentResponse>(64);
        let (mut supervisor, recorded) = supervisor_with_resident_recorder(resp_tx);

        let mk = |prompt: &str, model: Option<&str>, sid: Option<&str>| SpawnRequest {
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

        // First turn: fresh session → one recorded spawn.
        supervisor.spawn_agent(mk("first", Some("opus"), None));
        assert_eq!(recorded.lock().unwrap().len(), 1, "first turn spawns once");
        assert!(supervisor.is_running(&chat_id), "session live + busy");

        // Second turn: the message-put path hard-aborts the running turn first
        // (interrupt-then-send), then spawns FRESH with `--resume`. Mirror that
        // here — abort then respawn — and assert a SECOND spawn was recorded
        // (no warm reuse) carrying the harvested session id.
        supervisor.abort_agent(&chat_id).await;
        supervisor.spawn_agent(mk("second", Some("opus"), Some("sid-1")));
        let captured = recorded.lock().unwrap();
        assert_eq!(
            captured.len(),
            2,
            "every turn respawns fresh — no warm reuse"
        );
        assert_eq!(
            captured[1].agent_session_id.as_deref(),
            Some("sid-1"),
            "respawn resumes the harvested session id"
        );
    }
}
