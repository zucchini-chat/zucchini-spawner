//! Control socket for agent-side CLI subcommands.
//!
//! `~/.zucchini-spawner/control.sock` — newline-delimited JSON. The daemon
//! listens; CLI sub-invocations (e.g. `zucchini-spawner attach-file …`) are
//! thin RPC clients. Putting the heavy lifting (JWT, K_user, R2 PUT) on the
//! daemon side means the CLI never touches secrets and a stale binary on
//! PATH can't independently authenticate against the backend.
//!
//! Wire protocol:
//!   - request: `{ "action": "attach_file", "chat_id": "<uuid>", "path": "<abs>" }\n`
//!     or `{ "action": "schedule_message", "chat_id": "<uuid>", "body": "<text>",
//!     "deliver_at": "<naive-local-datetime|rfc3339>"|null }\n` (null = queue for
//!     the next agent-free window; non-null = that wall-clock time, zoned daemon-side)
//!     or `{ "action": "list_scheduled_messages", "chat_id": "<uuid>" }\n`
//!     or `{ "action": "cancel_scheduled_message", "chat_id": "<uuid>", "id": "<uuid>" }\n`
//!   - response: `{ "ok": true, "blob_key": "...", "name": "..." }\n`
//!     or `{ "ok": true, "scheduled_message_id": "<uuid>" }\n` (schedule + cancel)
//!     or `{ "ok": true, "scheduled_messages": [ { "id": "<uuid>",
//!     "deliver_at": "<local-wall-clock>"|null, "text": "...", "attachments": ["<name>"] } ] }\n`
//!     or `{ "ok": false, "error": "..." }\n`
//!
//! Connection is one request/one response then close — the CLI exits and we
//! release the per-conn task. Concurrent agents in different chats are
//! independent connections.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tracing::{debug, error, info, warn};

use crate::agent::PendingAttachments;
use crate::blobs;
use crate::crypto::KeyStore;
use crate::envelope::EnvelopeAttachment;
use crate::prune::{PendingPrunes, PruneRequest};
use crate::state::{Mirror, SharedMirror};
use crate::sys;
use crate::watch::WatchDef;
use crate::writer::TokenFetcher;

/// Where the daemon binds and the CLI connects. Lives under the same
/// `~/.zucchini-spawner` dir whose perms we already lock to 0o700 in
/// `main.rs`, so the socket inherits that confinement.
pub fn control_socket_path() -> PathBuf {
    crate::zucchini_spawner_dir().join("control.sock")
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum ControlRequest {
    /// Upload `path` and queue an `EnvelopeAttachment` for the next assistant
    /// frame on `chat_id`. Caller (the CLI process started by the agent) is
    /// trusted: the daemon already runs as the user, the socket is 0700, and
    /// per-path policy (no `/etc/passwd` confinement, no size cap) is
    /// explicit user-product decision — see CLAUDE.md scope-of-responsibility
    /// rule. Validation is restricted to "is there a running agent for this
    /// chat".
    AttachFile { chat_id: String, path: String },
    /// Selective context forgetting (claude/gemini/codex). Carries one or more
    /// [`PruneItem`]s — the agent batches every output it wants to forget into a
    /// SINGLE CLI call so the burst is one process → one RPC → one restart.
    /// (Separate parallel `prune-context` processes used to reap each other: the
    /// first one's restart SIGTERMs the agent's whole process group, killing
    /// still-running sibling prune CLIs before their RPC lands — see
    /// `tmp/agent_log.txt`.) Each item with ≥1 eligible match is queued as its
    /// own `PruneRequest`; the actual abort/rewrite/respawn is NOT triggered by
    /// this call. The `prune-context` CLI just prints its summary and exits 0; the
    /// main loop applies the queued prune when claude emits that call's
    /// `tool_result` frame (so the rewrite lands strictly after claude persists the
    /// result). If NO item matches anything, nothing is queued and the call errors
    /// so the live agent can retry.
    PruneContext {
        chat_id: String,
        items: Vec<PruneItem>,
        /// When false (default) the daemon REFUSES to prune while the resident
        /// session has live background tasks / monitors — a prune restarts the
        /// agent (abort → rewrite → respawn) and their in-process runtime is not
        /// restored by `--resume`, so they'd be silently killed. `--force`
        /// overrides (prune anyway, terminating them). `#[serde(default)]` keeps
        /// an older CLI binary that omits the field decoding across a hot-reload.
        #[serde(default)]
        force: bool,
    },
    /// Insert a `scheduled_messages` row for `chat_id` — the running agent
    /// enqueues a message at a wall-clock time (`deliver_at` = naive local datetime
    /// or offset-bearing RFC3339) or for the next agent-free window (`null`). The
    /// nullness IS the fire condition; the daemon zones a naive value to UTC
    /// (`normalize_deliver_at`) before the write. `body` is plaintext here; the
    /// daemon encrypts it with the owner's key into the `messages.body` wire format
    /// before the `/api/writes` PUT. Same trust model as `AttachFile`.
    ScheduleMessage {
        chat_id: String,
        body: String,
        deliver_at: Option<String>,
    },
    /// Still-pending `scheduled_messages` rows for `chat_id`, soonest first. The
    /// daemon decrypts each body with the owner's key and renders `deliver_at`
    /// into the user's local wall-clock — the CLI holds neither the key nor the tz.
    ListScheduledMessages { chat_id: String },
    /// Cancel one pending row. Scoped to `chat_id`: the daemon only deletes an id
    /// that `ListScheduledMessages` returns for that chat, so an agent can't reach
    /// into another chat's schedule and an already-fired row reports as such
    /// instead of succeeding silently.
    CancelScheduledMessage { chat_id: String, id: String },
    /// Start a watch (`watch.rs`) owned by `chat_id`: the daemon runs `cmd` in
    /// `cwd` and turns its stdout lines into messages. Refused for a sandboxed
    /// member's chat, since the command runs outside any sandbox.
    CreateWatch {
        chat_id: String,
        cmd: String,
        cwd: String,
        ttl: String,
        description: String,
    },
    /// The watches of `chat_id`; a chat never sees another chat's.
    ListWatches { chat_id: String },
    /// Stop and remove `chat_id`'s watch of that `description`.
    DeleteWatch {
        chat_id: String,
        description: String,
    },
}

/// One pending scheduled message as the CLI sees it. `deliver_at` is already in
/// the user's LOCAL wall-clock (see `local_wall_clock`; what `schedule-message --at`
/// accepts) — rendered daemon-side because only the daemon's mirror holds the
/// member timezone, same reason `normalize_deliver_at` lives here. `None` means
/// the row is queued for the next agent-free window rather than a fixed time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduledMessageInfo {
    pub id: String,
    pub deliver_at: Option<String>,
    pub text: String,
    #[serde(default)]
    pub attachments: Vec<String>,
}

/// One prune target inside a (possibly batched) `prune-context` call. `tool_name`
/// is the CLAUDE-shape tool to prune, or `""` for the "any tool" selector (match
/// on `needle` alone — codex omits `--tool-name`); `needle` is the `--args` glob,
/// `""` selecting no-args calls; `reason` is the agent's `--summary` (the
/// task-relevant takeaway it keeps after the raw output is blanked). A single CLI
/// invocation carries one or more of these (one per `--summary`-terminated triple
/// on the command line); the daemon queues a `PruneRequest` per matching item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruneItem {
    pub tool_name: String,
    pub needle: String,
    pub reason: String,
}

/// Wire response. `ok=true` always carries `blob_key`+`name`; `ok=false`
/// always carries `error`. The unused fields are `None` and elided from the
/// JSON via `skip_serializing_if`. One struct (not an untagged enum) keeps
/// the CLI-side decode a single `serde_json::from_str` instead of poking
/// at `serde_json::Value`.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ControlResponse {
    pub ok: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blob_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// The `scheduled_messages.id` the call acted on — minted by
    /// `schedule_message`, deleted by `cancel_scheduled_message`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scheduled_message_id: Option<String>,
    /// Set only by the `list_scheduled_messages` handler. An empty vec is a
    /// successful "nothing pending", distinct from `None` (not that handler).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scheduled_messages: Option<Vec<ScheduledMessageInfo>>,
    /// Set by the watch handlers: the one created or deleted, or the list.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub watches: Option<Vec<WatchDef>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Per-item eligible-match counts, parallel to the request's `items` — set
    /// only on the `prune_context` success path. A `0` entry means that item
    /// matched nothing (queued nothing); the CLI reports it as a per-item miss
    /// without failing the whole batch (the batch succeeds as long as ≥1 item
    /// matched). Each non-zero `n` is the count BEFORE this prune, so `n-1`
    /// matches remain for a repeat.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pruned_counts: Option<Vec<usize>>,
}

impl ControlResponse {
    fn ok(blob_key: String, name: String) -> Self {
        Self {
            ok: true,
            blob_key: Some(blob_key),
            name: Some(name),
            ..Default::default()
        }
    }
    fn scheduled(id: String) -> Self {
        Self {
            ok: true,
            scheduled_message_id: Some(id),
            ..Default::default()
        }
    }
    fn scheduled_list(messages: Vec<ScheduledMessageInfo>) -> Self {
        Self {
            ok: true,
            scheduled_messages: Some(messages),
            ..Default::default()
        }
    }
    fn watches(watches: Vec<WatchDef>) -> Self {
        Self {
            ok: true,
            watches: Some(watches),
            ..Default::default()
        }
    }
    fn pruned_ok(counts: Vec<usize>) -> Self {
        Self {
            ok: true,
            pruned_counts: Some(counts),
            ..Default::default()
        }
    }
    fn err(msg: impl Into<String>) -> Self {
        // `ok` defaults to false here — that's the only failure constructor.
        Self {
            error: Some(msg.into()),
            ..Default::default()
        }
    }
}

/// State the per-conn handlers need to actually do the work. Cheap to clone —
/// everything inside is `Arc`-ish or `Clone`-able shallow handles. `mirror`
/// is the same `Arc<tokio::sync::RwLock<Mirror>>` the main loop holds; the
/// control task only ever takes the READ guard (resolve `chat_id → user_id`
/// from `mirror.chats`) and drops it before the slow R2-PUT work begins.
#[derive(Clone)]
pub struct ControlState {
    pub http: reqwest::Client,
    pub api_base_url: String,
    pub fetch_token: Arc<TokenFetcher>,
    pub keys: Arc<KeyStore>,
    pub pending: PendingAttachments,
    pub mirror: SharedMirror,
    /// Shared directory of live resident sessions (same `SessionState` Arcs the
    /// reader mutates), so `prune_context` can read a chat's live-task count and
    /// refuse a prune that would restart the agent out from under a running task.
    pub live_sessions: crate::agent::LiveSessions,
    /// Shared park-table of `PruneRequest`s, keyed by `chat_id`. We push under the
    /// lock here, inside the `prune-context` RPC; the main `select!` loop (sole
    /// owner of the `Supervisor`) drains the chat's entry and applies the
    /// abort + jsonl rewrite + respawn when claude emits the `prune-context`
    /// call's `tool_result` frame. The push completing before this RPC returns is
    /// what guarantees the request is visible before that cue can fire.
    pub pending_prunes: PendingPrunes,
}

/// Bind the listener and spawn the accept loop. Stale-socket cleanup + the
/// 0600 inode hardening live inside `sys::IpcListener::bind` (shared with the
/// hermes socket server), so a crash that left the inode behind doesn't
/// permanently break the RPC path (the bind would otherwise fail with
/// EADDRINUSE on every restart).
pub async fn start(state: ControlState) -> Result<()> {
    let path = control_socket_path();
    let listener = sys::IpcListener::bind(&path)
        .with_context(|| format!("bind control socket {}", path.display()))?;
    info!(path = %path.display(), "control socket listening");

    tokio::spawn(async move {
        loop {
            match listener.accept().await {
                Ok(stream) => {
                    let st = state.clone();
                    tokio::spawn(async move {
                        if let Err(e) = handle_conn(stream, st).await {
                            warn!(error = %e, "control-socket handler returned error");
                        }
                    });
                }
                Err(e) => {
                    error!(error = %e, "control listener accept failed");
                    // Don't bail the whole loop on a transient accept error;
                    // a real bind-level failure would have errored out of
                    // `start` above.
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                }
            }
        }
    });
    Ok(())
}

async fn handle_conn(stream: sys::IpcStream, state: ControlState) -> Result<()> {
    let (read_half, mut write_half) = tokio::io::split(stream);
    let mut reader = BufReader::new(read_half);
    let mut line = String::new();
    let n = reader
        .read_line(&mut line)
        .await
        .context("read control request")?;
    if n == 0 {
        return Err(anyhow!("empty control request"));
    }
    let trimmed = line.trim_end_matches('\n');
    let resp = match serde_json::from_str::<ControlRequest>(trimmed) {
        Ok(req) => dispatch(req, &state).await,
        Err(e) => ControlResponse::err(format!("invalid request JSON: {e}")),
    };
    let mut out = serde_json::to_string(&resp).unwrap_or_else(|_| {
        // Last-ditch hand-crafted error so the CLI sees something parseable.
        r#"{"ok":false,"error":"failed to serialize response"}"#.to_string()
    });
    out.push('\n');
    write_half
        .write_all(out.as_bytes())
        .await
        .context("write control response")?;
    write_half.shutdown().await.ok();
    Ok(())
}

async fn dispatch(req: ControlRequest, state: &ControlState) -> ControlResponse {
    match req {
        ControlRequest::AttachFile { chat_id, path } => {
            match attach_file(&chat_id, &path, state).await {
                Ok((blob_key, name)) => ControlResponse::ok(blob_key.to_string(), name),
                Err(e) => {
                    warn!(chat_id = %chat_id, path = %path, error = %e, "attach_file failed");
                    ControlResponse::err(format!("{e:#}"))
                }
            }
        }
        ControlRequest::PruneContext {
            chat_id,
            items,
            force,
        } => {
            info!(chat_id = %chat_id, n_items = items.len(), force, "prune-context called");
            match prune_context(&chat_id, &items, force, state).await {
                Ok(counts) => ControlResponse::pruned_ok(counts),
                Err(e) => {
                    warn!(chat_id = %chat_id, n_items = items.len(), error = %e, "prune_context failed");
                    ControlResponse::err(format!("{e:#}"))
                }
            }
        }
        ControlRequest::ScheduleMessage {
            chat_id,
            body,
            deliver_at,
        } => match schedule_message(&chat_id, &body, deliver_at.as_deref(), state).await {
            Ok(id) => ControlResponse::scheduled(id.to_string()),
            Err(e) => {
                warn!(chat_id = %chat_id, deliver_at = ?deliver_at, error = %e, "schedule_message failed");
                ControlResponse::err(format!("{e:#}"))
            }
        },
        ControlRequest::ListScheduledMessages { chat_id } => {
            match list_scheduled_messages(&chat_id, state).await {
                Ok(messages) => ControlResponse::scheduled_list(messages),
                Err(e) => {
                    warn!(chat_id = %chat_id, error = %e, "list_scheduled_messages failed");
                    ControlResponse::err(format!("{e:#}"))
                }
            }
        }
        ControlRequest::CancelScheduledMessage { chat_id, id } => {
            match cancel_scheduled_message(&chat_id, &id, state).await {
                Ok(()) => ControlResponse::scheduled(id),
                Err(e) => {
                    warn!(chat_id = %chat_id, %id, error = %e, "cancel_scheduled_message failed");
                    ControlResponse::err(format!("{e:#}"))
                }
            }
        }
        ControlRequest::CreateWatch {
            chat_id,
            cmd,
            cwd,
            ttl,
            description,
        } => watch_response(
            create_watch(&chat_id, &cmd, &cwd, &ttl, &description, state)
                .await
                .map(|def| vec![def]),
        ),
        ControlRequest::ListWatches { chat_id } => {
            watch_response(list_watches(&chat_id, state).await)
        }
        ControlRequest::DeleteWatch {
            chat_id,
            description,
        } => watch_response(
            delete_watch(&chat_id, &description, state)
                .await
                .map(|def| vec![def]),
        ),
    }
}

fn watch_response(result: Result<Vec<WatchDef>>) -> ControlResponse {
    match result {
        Ok(watches) => ControlResponse::watches(watches),
        Err(e) => {
            warn!(error = %e, "watch request failed");
            ControlResponse::err(format!("{e:#}"))
        }
    }
}

/// Human label for a prune target, naming the empty-args/any-tool selectors
/// distinctly so both the daemon's retry/miss error and the CLI's per-target
/// output read unambiguously. Shared (`pub`) so `main.rs`'s `prune-context` CLI
/// uses the SAME phrasing — no second copy to drift.
pub fn describe_prune_target(tool_name: &str, needle: &str) -> String {
    match (tool_name.is_empty(), needle.is_empty()) {
        (true, true) => "no-argument call".to_string(),
        (true, false) => format!("call with args matching \"{needle}\""),
        (false, true) => format!("no-argument \"{tool_name}\" call"),
        (false, false) => format!("\"{tool_name}\" call with args matching \"{needle}\""),
    }
}

/// Control-side `prune-context`: validate + locate the session transcript ONCE
/// (shared across the batch), then pre-scan each item for ELIGIBLE matches. Every
/// item with ≥1 match is queued as its own `PruneRequest` for the main loop; the
/// returned `Vec<usize>` is the per-item eligible count (parallel to `items`, `0`
/// = miss) the CLI turns into per-item "N remain" / "no … found" messages. If NO
/// item matched anything, nothing is queued and we error (agent still alive, can
/// retry) — this keeps the single-item behavior intact. The prunes are applied
/// when claude emits the `prune-context` call's `tool_result` frame, folded into
/// ONE abort→respawn.
async fn prune_context(
    chat_id: &str,
    items: &[PruneItem],
    force: bool,
    state: &ControlState,
) -> Result<Vec<usize>> {
    if items.is_empty() {
        return Err(anyhow!("prune-context called with no items"));
    }

    // Refuse to prune while the resident session has live background tasks /
    // monitors, unless `--force`. Applying a prune hard-restarts the agent
    // (abort → rewrite jsonl → respawn-with-`--resume`), and a task/monitor's
    // runtime lives in the agent process — `--resume` does NOT re-arm it — so the
    // restart would silently kill it (e.g. a running deploy whose status the
    // agent then can't recover). Block by default; the agent should wait for the
    // task to finish, or pass `--force` to prune and terminate it.
    //
    // The count is read from the shared `live_sessions` directory (the same
    // `SessionState` the stdout reader mutates). It's a hair behind the stdout
    // stream — a task armed in the SAME instant as this call can slip through —
    // but for the real flow (arm a task, decide to prune seconds later) the
    // `task_started` frame is long since reduced by the time this RPC lands. A
    // synchronous error here is the only way to tell the agent at the tool
    // boundary (exit 1); the perfectly-ordered signal arrives only post-exit.
    if !force {
        let live = crate::agent::live_task_count(&state.live_sessions, chat_id);
        if live > 0 {
            return Err(anyhow!(
                "{live} background task(s)/monitor(s) still running in this session — \
                 pruning now restarts the agent and terminates them (a `--resume` does \
                 not re-arm them). Wait until they finish and prune then, or re-run with \
                 --force to prune and kill them."
            ));
        }
    }
    // Tight read-guard scope: no `.await` held under it.
    let (agent_kind, agent_session_id) = {
        let g = state.mirror.read().await;
        g.chats
            .get(chat_id)
            .map(|c| (c.agent_kind, c.agent_session_id.clone()))
            .ok_or_else(|| anyhow!("no chat with id {chat_id} (unknown or not yet synced)"))?
    };
    // Selective-forgetting hooks (claude/gemini/codex/pi only).
    let ops = agent_kind
        .prune_ops()
        .ok_or_else(|| anyhow!("prune-context supports claude, gemini, codex and pi only"))?;

    // Locate by the agent SESSION id, which for spawner-created chats is NOT the
    // chat id. Fall back to chat_id for backfilled/imported rows (or before one's
    // been harvested), where the two coincide.
    let session_id = agent_session_id.as_deref().unwrap_or(chat_id);

    // CLI transcript base dir (honors CLAUDE_CONFIG_DIR / CODEX_HOME /
    // GEMINI_CLI_HOME, else $HOME/.<cli>).
    let base = agent_kind
        .cli_home()
        .ok_or_else(|| anyhow!("cannot resolve home dir for {agent_kind:?}"))?;

    // Locate the transcript ONCE; every item in the batch prunes the same file.
    let jsonl_path = (ops.find_session)(&base, session_id)
        .ok_or_else(|| anyhow!("no transcript for chat {chat_id} (session {session_id})"))?;

    // Pre-scan each item against the (as-yet-unmodified) transcript and queue a
    // `PruneRequest` for every item with ≥1 eligible match. `count_matches`
    // reads the on-disk jsonl, which isn't rewritten until `apply_prune_group`
    // runs at restart — so two items naming the same needle each see the full
    // count here and both queue (the group then blanks the most-recent-remaining
    // per request, exactly as repeated separate calls did before batching).
    let mut counts = Vec::with_capacity(items.len());
    let mut misses = Vec::new();
    let mut queued = 0usize;
    for item in items {
        let n = (ops.count_matches)(&jsonl_path, &item.tool_name, &item.needle)
            .with_context(|| format!("scan transcript {}", jsonl_path.display()))?;
        counts.push(n);
        if n == 0 {
            // Don't queue a no-op; record the miss for the batch error message
            // (only surfaced if EVERY item missed).
            misses.push(describe_prune_target(&item.tool_name, &item.needle));
            continue;
        }
        // Push under the lock — a tight, non-`await` critical section. This
        // completes before the RPC returns to the CLI, so the request is parked
        // before claude's `prune-context` `tool_result` (and thus the apply cue)
        // can land. No `Sender` to fail, so no shutdown-race branch here; an
        // entry parked moments before SIGTERM is simply dropped with the table
        // (the agent is being torn down anyway).
        state
            .pending_prunes
            .lock()
            .await
            .entry(chat_id.to_string())
            .or_default()
            .push(PruneRequest {
                jsonl_path: jsonl_path.clone(),
                agent_kind,
                tool_name: item.tool_name.clone(),
                needle: item.needle.clone(),
                reason: item.reason.clone(),
            });
        queued += 1;
    }

    if queued == 0 {
        // Nothing matched anywhere — no PruneRequest queued, so no restart fires
        // and the agent stays alive to retry. Mirror the single-item error so the
        // existing "no … found" handling (and tests) hold for a 1-item batch.
        return Err(anyhow!("no {} found in transcript", misses.join("; no ")));
    }

    Ok(counts)
}

async fn attach_file(
    chat_id: &str,
    path: &str,
    state: &ControlState,
) -> Result<(uuid::Uuid, String)> {
    // Read guard scope is tight: a single map lookup, no `.await`s held
    // under it. The slow work (presign-mint HTTP, file read, encrypt, R2
    // PUT) all happens after the guard is dropped at the end of this block.
    let user_id = {
        let g = state.mirror.read().await;
        g.chats
            .get(chat_id)
            .map(|c| c.user_id)
            .ok_or_else(|| anyhow!("no chat with id {chat_id} (unknown or not yet synced)"))?
    };

    // Resolve K_user so we can encrypt without leaning on file-system
    // discovery inside the hot path. `KeyStore::get` is a single-lock cache
    // lookup; first call per user does the disk read.
    let key = state
        .keys
        .get(&user_id)
        .with_context(|| format!("no key for user {user_id}"))?;

    // Clone the per-chat mailbox sender up front. Doing it once at the top
    // (a) confirms the agent is running before we mint an upload URL, and
    // (b) gives us a Sender we can push on without re-locking after the
    // mint. The std-Mutex section is microseconds — no async work under
    // the guard.
    let attach_tx = {
        let g = state.pending.lock().expect("PendingAttachments mutex");
        g.get(chat_id)
            .cloned()
            .ok_or_else(|| anyhow!("no running agent for chat {chat_id} (start a turn first)"))?
    };

    let path_buf = Path::new(path).to_path_buf();
    // Per task spec: NO path confinement. The user (or their agent) asked
    // for this file; the daemon already runs as that user. UI / agent prompt
    // is responsible for not asking for sensitive paths.
    //
    // Mint the presigned PUT URL synchronously — fast (one stat + one
    // backend RTT) and the only step that produces `blob_key` + `name`
    // (what the CLI prints + iOS renders on the pill). The slow R2 PUT
    // (encrypt + body bytes over the network) is deferred to a detached
    // task below so the CLI / shell tool / LLM all unblock ~immediately.
    let minted = blobs::mint_url(
        &state.http,
        &state.api_base_url,
        state.fetch_token.as_ref(),
        &path_buf,
    )
    .await
    .with_context(|| format!("mint upload-url for {}", path_buf.display()))?;

    let blobs::MintedUpload {
        blob_key,
        presigned_url,
        name,
        plaintext_size,
    } = minted;

    // Push the envelope into the per-chat mailbox before the upload starts.
    // `agent.rs::attach_followup_for` drains this on the next assistant text
    // frame and emits a follow-up attachment row; we want the envelope queued
    // by then, not stuck behind a 200 MB R2 PUT.
    let att = EnvelopeAttachment {
        blob_key,
        size: plaintext_size,
        name: name.clone(),
    };
    if attach_tx.send(att).is_err() {
        // Agent exited between the lock-section above and now. Vanishingly
        // unlikely (microseconds) but possible. The presigned URL is
        // already minted; nothing on R2 will reference it. Surface a
        // CLI-side error so the agent knows the attach failed.
        return Err(anyhow!(
            "agent on chat {chat_id} exited before envelope could be queued"
        ));
    }
    debug!(chat_id, %blob_key, name = %name, "queued attachment for next assistant frame");

    // Detached upload. `crypto::encrypt` is sync CPU-bound — wrap it in
    // `spawn_blocking` so it doesn't stall a worker thread on multi-hundred-MB
    // files. The HTTP client clones cheaply (internal Arc); `key` is already
    // `Arc<KUser>` so its `.clone()` is a refcount bump, not a key copy.
    let http = state.http.clone();
    let key_for_task: Arc<crate::crypto::KUser> = Arc::clone(&key);
    let chat_id_for_task = chat_id.to_string();
    let path_for_task = path_buf.clone();
    let name_for_task = name.clone();
    tokio::spawn(async move {
        // Read the file from inside the detached task: even tokio::fs::read
        // for a 200 MB file takes real wall-clock, and we want the control
        // reply out ASAP. The pre-mint stat already validated readability;
        // a races-with-deletion error here is logged but otherwise dropped
        // (iOS's retry-on-404 surfaces as permanent unavailable to the user).
        let read_res = tokio::fs::read(&path_for_task).await;
        let plaintext = match read_res {
            Ok(b) => b,
            Err(e) => {
                error!(
                    chat_id = %chat_id_for_task,
                    %blob_key,
                    path = %path_for_task.display(),
                    error = %e,
                    "background attach-file read failed"
                );
                sentry::with_scope(
                    |scope| {
                        scope.set_tag("chat_id", &chat_id_for_task);
                        scope.set_tag("blob_key", blob_key.to_string());
                        scope.set_tag("path", path_for_task.display().to_string());
                    },
                    || {
                        sentry::capture_message(
                            &format!("attach-file read failed: {e}"),
                            sentry::Level::Error,
                        );
                    },
                );
                return;
            }
        };

        // CPU-bound; offload so we don't stall a tokio worker.
        let cipher_res =
            tokio::task::spawn_blocking(move || crate::crypto::encrypt(&key_for_task, &plaintext))
                .await;
        let ciphertext = match cipher_res {
            Ok(c) => c,
            Err(e) => {
                error!(
                    chat_id = %chat_id_for_task,
                    %blob_key,
                    error = %e,
                    "background attach-file encrypt task panicked"
                );
                sentry::with_scope(
                    |scope| {
                        scope.set_tag("chat_id", &chat_id_for_task);
                        scope.set_tag("blob_key", blob_key.to_string());
                    },
                    || {
                        sentry::capture_message(
                            &format!("attach-file encrypt panicked: {e}"),
                            sentry::Level::Error,
                        );
                    },
                );
                return;
            }
        };

        match blobs::put_ciphertext(&http, &presigned_url, ciphertext).await {
            Ok(()) => {
                info!(
                    chat_id = %chat_id_for_task,
                    %blob_key,
                    name = %name_for_task,
                    size = plaintext_size,
                    "uploaded agent attachment (background)"
                );
            }
            Err(e) => {
                error!(
                    chat_id = %chat_id_for_task,
                    %blob_key,
                    path = %path_for_task.display(),
                    error = %format!("{e:#}"),
                    "background attach-file upload failed"
                );
                sentry::with_scope(
                    |scope| {
                        scope.set_tag("chat_id", &chat_id_for_task);
                        scope.set_tag("blob_key", blob_key.to_string());
                        scope.set_tag("path", path_for_task.display().to_string());
                    },
                    || {
                        sentry::capture_message(
                            &format!("attach-file R2 PUT failed: {e:#}"),
                            sentry::Level::Error,
                        );
                    },
                );
                // Nothing to retract — the envelope was already pushed to
                // iOS. The iOS retry-on-404 loop (BlobClient.swift) will
                // eventually surface the failure as a permanent 404 to the
                // user, which is the agreed UX per the task spec.
            }
        }
    });

    Ok((blob_key, name))
}

/// `chat_id → (owner, owner's IANA zone)`. The read guard spans a couple of map
/// lookups with no `.await` under it; the slow crypto/HTTP work happens in the
/// callers after it drops.
async fn chat_owner_and_tz(
    chat_id: &str,
    state: &ControlState,
) -> Result<(uuid::Uuid, Option<chrono_tz::Tz>)> {
    let g = state.mirror.read().await;
    let user_id = g
        .chats
        .get(chat_id)
        .map(|c| c.user_id)
        .ok_or_else(|| anyhow!("no chat with id {chat_id} (unknown or not yet synced)"))?;
    let tz = g
        .member_timezone(&user_id)
        .and_then(|s| s.parse::<chrono_tz::Tz>().ok());
    Ok((user_id, tz))
}

/// Daemon-side handler for `ScheduleMessage`. Resolves `chat_id → user_id` from
/// the mirror, encrypts `body` into the `messages.body` wire format, and PUTs a
/// fresh `scheduled_messages` row to `/api/writes` (inline — one encrypt + one
/// POST, no presign/upload). Backend stamps `user_id`/`created_at`.
async fn schedule_message(
    chat_id: &str,
    body: &str,
    deliver_at: Option<&str>,
    state: &ControlState,
) -> Result<uuid::Uuid> {
    let (user_id, tz) = chat_owner_and_tz(chat_id, state).await?;

    // Validate + canonicalize (the socket is its own entry point, so reject bad
    // values here too, not just at the CLI front-end). Absent stays NULL (queue).
    let deliver_at = normalize_deliver_at(deliver_at.map(str::to_string), tz)?;

    // Use `envelope::encode`, NOT `encrypt_field_b64`: the backend promotes this
    // body verbatim into a `sender='user'` message and the spawner runs
    // `envelope::decode` on user bodies, so it must be the `{text,attachments}`
    // JSON envelope — a raw field-encrypted string decrypts to a bare string and
    // fails the envelope parse, silently dropping the message.
    let key = state
        .keys
        .get(&user_id)
        .with_context(|| format!("no key for user {user_id}"))?;
    let body_ct = crate::envelope::encode(body, &key).context("encode scheduled message body")?;

    let id = uuid::Uuid::now_v7();
    // `deliver_at = None` serializes to explicit JSON null — the backend keys the
    // fire condition off its nullness (null = queue/agent-free, non-null = timed).
    let op = serde_json::json!({
        "op": "PUT",
        "table": "scheduled_messages",
        "id": id.to_string(),
        "data": {
            "chat_id": chat_id,
            "body": body_ct,
            "deliver_at": deliver_at,
        },
    });

    let url = format!("{}/api/writes", state.api_base_url.trim_end_matches('/'));
    let token = (state.fetch_token)()
        .await
        .context("fetch JWT for /api/writes")?;
    let resp = crate::net::send_reporting(
        "POST",
        &url,
        state
            .http
            .post(&url)
            .bearer_auth(token)
            .json(&serde_json::json!({ "ops": [op] }))
            .send(),
    )
    .await
    .context("POST /api/writes (schedule_message)")?;
    let status = resp.status();
    if !status.is_success() {
        let body = crate::net::capture_response_failure("POST", &url, resp).await;
        return Err(anyhow!("POST /api/writes {}: {}", status, body));
    }
    info!(chat_id, %id, deliver_at = ?deliver_at, "scheduled message via /api/writes");
    Ok(id)
}

/// The owner of `chat_id`, for a chat allowed to run watches: the command runs
/// as the spawner's OS user outside any sandbox, so a sandboxed member's chat
/// gets none.
fn watch_chat_owner(chat_id: &str, mirror: &Mirror, keys: &KeyStore) -> Result<uuid::Uuid> {
    let chat = mirror
        .chats
        .get(chat_id)
        .ok_or_else(|| anyhow!("no chat with id {chat_id} (unknown or not yet synced)"))?;
    if !mirror.can_watch(chat_id, keys) {
        return Err(anyhow!(
            "watches are not available in this chat (it is sandboxed or its user's key is gone)"
        ));
    }
    Ok(chat.user_id)
}

async fn create_watch(
    chat_id: &str,
    cmd: &str,
    cwd: &str,
    ttl: &str,
    description: &str,
    state: &ControlState,
) -> Result<WatchDef> {
    let watches = crate::watch::global()?;
    // Exclusive until the watch is registered and state.json holds its chat's
    // row: a revoke can't slip in between the access check and the watch it
    // would cancel, and the main loop's own state.json save can't overlap
    // this one. Without the row the next boot would remove the watch.
    let mirror = state.mirror.write().await;
    let user_id = watch_chat_owner(chat_id, &mirror, &state.keys)?;
    let def = watches.create(chat_id, user_id, cmd, cwd, ttl, description)?;
    if let Err(e) = mirror.save(&crate::state_path()) {
        warn!(error = %e, "failed to persist state.json");
    }
    Ok(def)
}

async fn list_watches(chat_id: &str, state: &ControlState) -> Result<Vec<WatchDef>> {
    watch_chat_owner(chat_id, &*state.mirror.read().await, &state.keys)?;
    let mut watches = crate::watch::global()?.list();
    watches.retain(|w| w.chat_id == chat_id);
    Ok(watches)
}

async fn delete_watch(chat_id: &str, description: &str, state: &ControlState) -> Result<WatchDef> {
    watch_chat_owner(chat_id, &*state.mirror.read().await, &state.keys)?;
    let watches = crate::watch::global()?;
    let missing = || anyhow!("this chat has no watch \"{description}\" (`watch list`)");
    let id = watches.find(chat_id, description).ok_or_else(missing)?.id;
    watches.remove(id, None).await.ok_or_else(missing)
}

/// Daemon-side handler for `ListScheduledMessages`: GET the chat's still-pending
/// rows, decrypt each body with the owner's key and render `deliver_at` into that
/// owner's local wall-clock. A row we can't decrypt is still returned (with
/// placeholder text) so the agent can cancel it — hiding it would strand the row.
async fn list_scheduled_messages(
    chat_id: &str,
    state: &ControlState,
) -> Result<Vec<ScheduledMessageInfo>> {
    #[derive(Deserialize)]
    struct Row {
        id: String,
        body: String,
        deliver_at: Option<String>,
    }
    #[derive(Deserialize)]
    struct Page {
        messages: Vec<Row>,
    }

    let (user_id, tz) = chat_owner_and_tz(chat_id, state).await?;
    let key = state.keys.get(&user_id).ok();

    let url = format!(
        "{}/api/scheduled?chat_id={chat_id}",
        state.api_base_url.trim_end_matches('/')
    );
    let token = (state.fetch_token)()
        .await
        .context("fetch JWT for /api/scheduled")?;
    let resp =
        crate::net::send_reporting("GET", &url, state.http.get(&url).bearer_auth(token).send())
            .await
            .context("GET /api/scheduled")?;
    let status = resp.status();
    if !status.is_success() {
        let body = crate::net::capture_response_failure("GET", &url, resp).await;
        // The handler's own 404s carry a reason; an empty one is a missing route.
        if status == reqwest::StatusCode::NOT_FOUND && body.is_empty() {
            return Err(anyhow!(
                "the Zucchini backend doesn't support listing scheduled messages yet"
            ));
        }
        return Err(anyhow!("GET /api/scheduled {}: {}", status, body));
    }
    let page: Page = resp.json().await.context("parse /api/scheduled response")?;

    Ok(page
        .messages
        .into_iter()
        .map(|row| {
            let envelope = key
                .as_deref()
                .and_then(|k| crate::envelope::decode(&row.body, k).ok());
            ScheduledMessageInfo {
                id: row.id,
                deliver_at: row.deliver_at.map(|at| local_wall_clock(&at, tz)),
                text: envelope
                    .as_ref()
                    .map_or_else(|| "<undecryptable>".to_string(), |env| env.text.clone()),
                attachments: envelope
                    .map(|env| env.attachments.into_iter().map(|a| a.name).collect())
                    .unwrap_or_default(),
            }
        })
        .collect())
}

/// Render a UTC RFC3339 instant as the owner's local wall-clock — the same shape
/// `schedule-message --at` accepts, so a listed time can be re-scheduled verbatim.
/// Falls back to the raw value when the zone is unknown or unparseable, which is
/// honest about the instant instead of guessing UTC is local.
///
/// In the repeated fall-back hour the offset is appended: a bare wall-clock would
/// re-schedule as its earlier occurrence. Only then, so the agent doesn't learn to
/// write offsets (and do DST math) itself.
fn local_wall_clock(rfc3339: &str, tz: Option<chrono_tz::Tz>) -> String {
    use chrono::TimeZone;
    match (chrono::DateTime::parse_from_rfc3339(rfc3339), tz) {
        (Ok(dt), Some(tz)) => {
            let local = dt.with_timezone(&tz);
            let ambiguous = matches!(
                tz.from_local_datetime(&local.naive_local()),
                chrono::offset::LocalResult::Ambiguous(..)
            );
            let fmt = if ambiguous {
                "%Y-%m-%dT%H:%M:%S%:z"
            } else {
                "%Y-%m-%dT%H:%M:%S"
            };
            local.format(fmt).to_string()
        }
        _ => rfc3339.to_string(),
    }
}

fn not_pending_error(id: &str) -> anyhow::Error {
    anyhow!("no pending scheduled message {id} in this chat — already delivered or cancelled")
}

/// Daemon-side handler for `CancelScheduledMessage`. The id must appear in THIS
/// chat's pending list before we delete it: that stops an agent in one chat from
/// cancelling another chat's message, and turns "already fired" into an honest
/// error rather than a silent success.
async fn cancel_scheduled_message(chat_id: &str, id: &str, state: &ControlState) -> Result<()> {
    let pending = list_scheduled_messages(chat_id, state).await?;
    if !pending.iter().any(|m| m.id == id) {
        return Err(not_pending_error(id));
    }

    let url = format!(
        "{}/api/scheduled/{id}",
        state.api_base_url.trim_end_matches('/')
    );
    let token = (state.fetch_token)()
        .await
        .context("fetch JWT for /api/scheduled")?;
    let resp = crate::net::send_reporting(
        "DELETE",
        &url,
        state.http.delete(&url).bearer_auth(token).send(),
    )
    .await
    .context("DELETE /api/scheduled")?;
    let status = resp.status();
    if status == reqwest::StatusCode::NOT_FOUND {
        return Err(not_pending_error(id));
    }
    if !status.is_success() {
        let body = crate::net::capture_response_failure("DELETE", &url, resp).await;
        return Err(anyhow!("DELETE /api/scheduled {}: {}", status, body));
    }
    info!(chat_id, %id, "cancelled scheduled message");
    Ok(())
}

// ===== CLI client side =====

/// One request/one response round-trip over the control socket: connect → write
/// newline-framed JSON → half-close → read the response line → bubble up
/// daemon-side `ok=false` errors. Shared by every CLI subcommand.
async fn send_control_request(req: &ControlRequest) -> Result<ControlResponse> {
    let sock = control_socket_path();
    let stream = sys::ipc_connect(&sock).await.with_context(|| {
        format!(
            "connect to {} — is zucchini-spawner running?",
            sock.display()
        )
    })?;
    let (read_half, mut write_half) = tokio::io::split(stream);
    let mut payload = serde_json::to_string(req).expect("serialize control request");
    payload.push('\n');
    write_half
        .write_all(payload.as_bytes())
        .await
        .context("write control request")?;
    write_half.shutdown().await.ok();

    let mut reader = BufReader::new(read_half);
    let mut line = String::new();
    reader
        .read_line(&mut line)
        .await
        .context("read control response")?;
    let resp: ControlResponse =
        serde_json::from_str(line.trim_end_matches('\n')).context("parse control response JSON")?;
    if !resp.ok {
        let error = resp.error.unwrap_or_else(|| "unknown error".to_string());
        // After a self-update the CLI can be newer than the daemon still running.
        if error.contains("unknown variant") {
            return Err(anyhow!(
                "the running zucchini-spawner daemon is older than this command and \
                 doesn't support it yet; it updates once no agent is busy ({error})"
            ));
        }
        return Err(anyhow!("{error}"));
    }
    Ok(resp)
}

/// Connect to the daemon's control socket and run one `attach-file` RPC.
/// Returns the (blob_key, name) on success; bubbles up daemon-side errors
/// (no-such-chat, upload failure) verbatim.
pub async fn attach_file_via_socket(chat_id: &str, path: &str) -> Result<(String, String)> {
    let resp = send_control_request(&ControlRequest::AttachFile {
        chat_id: chat_id.to_string(),
        path: path.to_string(),
    })
    .await?;
    let blob_key = resp
        .blob_key
        .ok_or_else(|| anyhow!("response missing blob_key"))?;
    let name = resp.name.ok_or_else(|| anyhow!("response missing name"))?;
    Ok((blob_key, name))
}

/// Connect to the daemon's control socket and run one (possibly batched)
/// `prune-context` RPC, returning the per-item eligible counts (parallel to
/// `items`, `0` = that item matched nothing). This RPC only pre-scans + queues a
/// `PruneRequest` per matching item, then returns cleanly so the CLI can exit 0.
/// The actual abort/rewrite/respawn is driven by the main loop when claude emits
/// the `prune-context` call's `tool_result` frame, folding the batch into ONE
/// restart that lands strictly after claude persists the call's result.
pub async fn prune_context_via_socket(
    chat_id: &str,
    items: Vec<PruneItem>,
    force: bool,
) -> Result<Vec<usize>> {
    let resp = send_control_request(&ControlRequest::PruneContext {
        chat_id: chat_id.to_string(),
        items,
        force,
    })
    .await?;
    Ok(resp.pruned_counts.unwrap_or_default())
}

/// Validate + canonicalize an optional `deliver_at` to a uniform UTC RFC3339.
///
/// `None` (flag omitted) stays `None` → NULL "queue, send when free" row. A
/// present value resolves via one of two paths:
///
///   * **Naive local wall-clock** (no offset/`Z`) — the agent contract, so it
///     never does timezone/DST math. Anchored in the user's IANA zone `tz` (DST
///     resolved by the tz database here): ambiguous fall-back hour → earliest
///     instant; nonexistent spring-forward hour → first valid instant after the
///     gap. Requires `tz`; without it we error rather than guess UTC.
///   * **Offset-bearing RFC3339** (`…Z` / `…+02:00`) — absolute instant, `tz`
///     irrelevant. Backwards tolerance for older agents / explicit offsets.
///
/// Anything else (natural language, empty `--at ""`) fails loudly naming the
/// value, so it can never silently become a misfiring row.
pub fn normalize_deliver_at(
    deliver_at: Option<String>,
    tz: Option<chrono_tz::Tz>,
) -> Result<Option<String>> {
    let Some(s) = deliver_at else { return Ok(None) };

    // 1) Offset-bearing RFC3339 → honor as an absolute instant (tz ignored).
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(&s) {
        return Ok(Some(dt.with_timezone(&chrono::Utc).to_rfc3339()));
    }

    // 2) Naive local wall-clock → anchor in the user's zone.
    if let Some(naive) = parse_naive_local(&s) {
        let tz = tz.ok_or_else(|| {
            anyhow!(
                "--at {s:?} is a local time but the user's timezone is unknown; \
                 pass an explicit offset (e.g. {s}Z) or omit --at to queue"
            )
        })?;
        return Ok(Some(zone_local_to_utc(naive, tz).to_rfc3339()));
    }

    Err(anyhow!(
        "--at {s:?} is not a valid timestamp (expected a local wall-clock like \
         2026-06-04T09:00:00, or RFC3339 with an offset); \
         omit --at to queue the message instead"
    ))
}

/// Parse a zoneless local datetime in the few shapes the agent might emit
/// (`T` or space separator, seconds optional).
fn parse_naive_local(s: &str) -> Option<chrono::NaiveDateTime> {
    use chrono::NaiveDateTime;
    const FORMATS: &[&str] = &[
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
    ];
    FORMATS
        .iter()
        .find_map(|f| NaiveDateTime::parse_from_str(s, f).ok())
}

/// Map a naive local wall-clock into a UTC instant using `tz`, resolving the
/// two DST edge cases deterministically (see [`normalize_deliver_at`]).
fn zone_local_to_utc(
    naive: chrono::NaiveDateTime,
    tz: chrono_tz::Tz,
) -> chrono::DateTime<chrono::Utc> {
    use chrono::offset::LocalResult;
    use chrono::TimeZone;
    match tz.from_local_datetime(&naive) {
        LocalResult::Single(dt) => dt.with_timezone(&chrono::Utc),
        // Fall-back hour occurs twice — pick the earliest (first) occurrence.
        LocalResult::Ambiguous(earliest, _latest) => earliest.with_timezone(&chrono::Utc),
        // Spring-forward gap: this wall-clock never occurs. The gap is 1h, so
        // the same clock time one hour later always lands in a valid window.
        LocalResult::None => {
            let bumped = naive + chrono::TimeDelta::hours(1);
            match tz.from_local_datetime(&bumped) {
                LocalResult::Single(dt) => dt.with_timezone(&chrono::Utc),
                LocalResult::Ambiguous(dt, _) => dt.with_timezone(&chrono::Utc),
                // Unreachable for real zones (no 2h+ gaps); fall back to a
                // literal UTC reading rather than panic.
                LocalResult::None => chrono::Utc.from_utc_datetime(&bumped),
            }
        }
    }
}

/// Run one `scheduled-messages create` RPC. Mirrors `attach_file_via_socket`: CLI never
/// touches K_user or the JWT — the daemon encrypts + POSTs. Returns the minted
/// `scheduled_messages.id`. `deliver_at` is forwarded raw and zoned daemon-side
/// (`normalize_deliver_at`), since only the daemon's mirror holds the user's tz.
pub async fn schedule_message_via_socket(
    chat_id: &str,
    body: &str,
    deliver_at: Option<String>,
) -> Result<String> {
    let resp = send_control_request(&ControlRequest::ScheduleMessage {
        chat_id: chat_id.to_string(),
        body: body.to_string(),
        deliver_at,
    })
    .await?;
    resp.scheduled_message_id
        .ok_or_else(|| anyhow!("response missing scheduled_message_id"))
}

/// Run one `scheduled-messages list` RPC. Bodies come back already decrypted and
/// times already zoned — the CLI holds neither K_user nor the member timezone.
pub async fn list_scheduled_messages_via_socket(
    chat_id: &str,
) -> Result<Vec<ScheduledMessageInfo>> {
    let resp = send_control_request(&ControlRequest::ListScheduledMessages {
        chat_id: chat_id.to_string(),
    })
    .await?;
    resp.scheduled_messages
        .ok_or_else(|| anyhow!("response missing scheduled_messages"))
}

/// Run one `scheduled-messages delete` RPC, returning the cancelled row's id.
/// An id that isn't pending in this chat comes back as a daemon-side error.
pub async fn cancel_scheduled_message_via_socket(chat_id: &str, id: &str) -> Result<String> {
    let resp = send_control_request(&ControlRequest::CancelScheduledMessage {
        chat_id: chat_id.to_string(),
        id: id.to_string(),
    })
    .await?;
    resp.scheduled_message_id
        .ok_or_else(|| anyhow!("response missing scheduled_message_id"))
}

async fn watch_request(req: ControlRequest) -> Result<Vec<WatchDef>> {
    send_control_request(&req)
        .await?
        .watches
        .ok_or_else(|| anyhow!("response missing watches"))
}

/// Run one `watch create` RPC: the daemon starts the command and keeps it.
pub async fn create_watch_via_socket(
    chat_id: &str,
    cmd: String,
    cwd: String,
    ttl: String,
    description: String,
) -> Result<WatchDef> {
    let req = ControlRequest::CreateWatch {
        chat_id: chat_id.to_string(),
        cmd,
        cwd,
        ttl,
        description,
    };
    watch_request(req)
        .await?
        .pop()
        .ok_or_else(|| anyhow!("response missing the watch"))
}

pub async fn list_watches_via_socket(chat_id: &str) -> Result<Vec<WatchDef>> {
    watch_request(ControlRequest::ListWatches {
        chat_id: chat_id.to_string(),
    })
    .await
}

pub async fn delete_watch_via_socket(chat_id: &str, description: String) -> Result<WatchDef> {
    watch_request(ControlRequest::DeleteWatch {
        chat_id: chat_id.to_string(),
        description,
    })
    .await?
    .pop()
    .ok_or_else(|| anyhow!("response missing the watch"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_watch_needs_an_unsandboxed_chat_whose_user_still_has_a_key() {
        use uuid::Uuid;
        let (owner, member) = (Uuid::now_v7(), Uuid::now_v7());
        let keys = KeyStore::with_keys([(owner, [0u8; 32]), (member, [0u8; 32])]);
        let mut mirror = Mirror::default();
        mirror.set_user_id(owner);
        mirror.upsert_member("row".into(), member, false);
        let chat_id = Uuid::now_v7().to_string();
        let row = serde_json::json!({
            "id": chat_id,
            "project_id": Uuid::now_v7().to_string(),
            "user_id": member.to_string(),
            "last_seq": 0,
            "agent_kind": "claude",
            "worktree": false,
        });
        mirror.upsert_chat(chat_id.clone(), &row.to_string(), &keys);

        assert_eq!(watch_chat_owner(&chat_id, &mirror, &keys).unwrap(), member);
        assert!(watch_chat_owner("no-such-chat", &mirror, &keys).is_err());
        // Soft-revoked: the member row stays unsandboxed, the key is gone.
        let owner_only = KeyStore::with_keys([(owner, [0u8; 32])]);
        assert!(watch_chat_owner(&chat_id, &mirror, &owner_only).is_err());
        mirror.upsert_member("row".into(), member, true);
        assert!(watch_chat_owner(&chat_id, &mirror, &keys).is_err());
    }

    fn utc(s: &str) -> chrono::DateTime<chrono::Utc> {
        chrono::DateTime::parse_from_rfc3339(s)
            .unwrap()
            .with_timezone(&chrono::Utc)
    }

    fn out_utc(deliver_at: &str, tz: Option<chrono_tz::Tz>) -> chrono::DateTime<chrono::Utc> {
        let out = normalize_deliver_at(Some(deliver_at.into()), tz)
            .unwrap()
            .unwrap();
        utc(&out)
    }

    #[test]
    fn normalize_deliver_at_absent_is_none() {
        // Flag omitted → NULL queue row.
        assert_eq!(normalize_deliver_at(None, None).unwrap(), None);
    }

    #[test]
    fn normalize_deliver_at_valid_rfc3339_canonicalizes() {
        // Offset-bearing input → absolute instant regardless of tz; `Z` and
        // `+02:00` forms land on the same instant.
        assert_eq!(
            out_utc("2026-06-04T09:00:00Z", None),
            utc("2026-06-04T09:00:00Z")
        );
        assert_eq!(
            out_utc("2026-06-04T11:00:00+02:00", None),
            utc("2026-06-04T09:00:00Z")
        );
    }

    #[test]
    fn normalize_deliver_at_naive_local_is_zoned() {
        // Bare local wall-clock anchored in the user's zone. June NY = EDT
        // (UTC−4) → 09:00 local = 13:00Z. Seconds-optional + space sep parse same.
        let ny: chrono_tz::Tz = "America/New_York".parse().unwrap();
        assert_eq!(
            out_utc("2026-06-04T09:00:00", Some(ny)),
            utc("2026-06-04T13:00:00Z")
        );
        assert_eq!(
            out_utc("2026-06-04T09:00", Some(ny)),
            utc("2026-06-04T13:00:00Z")
        );
        assert_eq!(
            out_utc("2026-06-04 09:00:00", Some(ny)),
            utc("2026-06-04T13:00:00Z")
        );

        // Reykjavik is UTC+0 year-round (no DST): the wall-clock IS the UTC
        // time. This is exactly the "looks like UTC but isn't UTC tz" case.
        let rvk: chrono_tz::Tz = "Atlantic/Reykjavik".parse().unwrap();
        assert_eq!(
            out_utc("2026-06-04T09:00:00", Some(rvk)),
            utc("2026-06-04T09:00:00Z")
        );
    }

    #[test]
    fn normalize_deliver_at_naive_without_tz_is_rejected() {
        // Naive local with no zone must error (never assume UTC); error names the
        // value + the offset escape.
        let err = normalize_deliver_at(Some("2026-06-04T09:00:00".into()), None).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("2026-06-04T09:00:00"), "got: {msg}");
        assert!(msg.contains("timezone"), "got: {msg}");
    }

    #[test]
    fn normalize_deliver_at_dst_spring_forward_gap_snaps_forward() {
        // 2026-03-08 02:30 doesn't exist in New York (clocks jump 02:00→03:00).
        // It must resolve deterministically to the first valid instant, not panic.
        let ny: chrono_tz::Tz = "America/New_York".parse().unwrap();
        // 03:30 EDT = 07:30Z (the bumped, valid reading).
        assert_eq!(
            out_utc("2026-03-08T02:30:00", Some(ny)),
            utc("2026-03-08T07:30:00Z")
        );
    }

    #[test]
    fn listed_time_round_trips_through_create_in_fall_back_hour() {
        // Lisbon falls back 02:00 WEST -> 01:00 WET on 2026-10-25, so 01:30 occurs twice.
        let lisbon: chrono_tz::Tz = "Europe/Lisbon".parse().unwrap();
        for (instant, listed) in [
            ("2026-10-25T00:30:00Z", "2026-10-25T01:30:00+01:00"),
            ("2026-10-25T01:30:00Z", "2026-10-25T01:30:00+00:00"),
            ("2026-10-25T12:00:00Z", "2026-10-25T12:00:00"),
        ] {
            let shown = local_wall_clock(instant, Some(lisbon));
            assert_eq!(shown, listed);
            assert_eq!(out_utc(&shown, Some(lisbon)), utc(instant));
        }
    }

    #[test]
    fn normalize_deliver_at_garbage_is_rejected() {
        // Natural language the agent might pass → loud error naming the value.
        let err = normalize_deliver_at(Some("tomorrow 9am".into()), None).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("tomorrow 9am"), "got: {msg}");
    }

    #[test]
    fn normalize_deliver_at_present_but_empty_is_rejected() {
        // Present-but-empty (`--at ""`) is NOT the queue sentinel — only an
        // omitted flag (`None`) is. Empty must fail loudly, not become NULL.
        assert!(normalize_deliver_at(Some(String::new()), None).is_err());
    }
}
