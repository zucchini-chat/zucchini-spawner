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
//!   - response: `{ "ok": true, "blob_key": "...", "name": "..." }\n`
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
use tokio::net::{UnixListener, UnixStream};
use tracing::{debug, error, info, warn};

use crate::agent::PendingAttachments;
use crate::blobs;
use crate::crypto::KeyStore;
use crate::envelope::EnvelopeAttachment;
use crate::state::SharedMirror;
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
}

/// Wire response. `ok=true` always carries `blob_key`+`name`; `ok=false`
/// always carries `error`. The unused fields are `None` and elided from the
/// JSON via `skip_serializing_if`. One struct (not an untagged enum) keeps
/// the CLI-side decode a single `serde_json::from_str` instead of poking
/// at `serde_json::Value`.
#[derive(Debug, Serialize, Deserialize)]
pub struct ControlResponse {
    pub ok: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blob_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl ControlResponse {
    fn ok(blob_key: String, name: String) -> Self {
        Self {
            ok: true,
            blob_key: Some(blob_key),
            name: Some(name),
            error: None,
        }
    }
    fn err(msg: impl Into<String>) -> Self {
        Self {
            ok: false,
            blob_key: None,
            name: None,
            error: Some(msg.into()),
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
}

/// Bind the listener and spawn the accept loop. Unlinks a stale socket from
/// a previous run before bind so a crash that left the inode behind doesn't
/// permanently break the RPC path (the bind would otherwise fail with
/// EADDRINUSE on every restart).
pub async fn start(state: ControlState) -> Result<()> {
    let path = control_socket_path();
    if path.exists() {
        // Best-effort: a *different* spawner process listening on the same
        // path would survive this remove since UDS bind replaces the inode
        // anyway, but the second `bind` below would then fail with EADDRINUSE
        // — surfaces as a startup error rather than silently fighting over
        // the socket.
        if let Err(e) = std::fs::remove_file(&path) {
            warn!(error = %e, path = %path.display(), "failed to remove stale control socket");
        }
    }
    let listener = UnixListener::bind(&path)
        .with_context(|| format!("bind control socket {}", path.display()))?;
    // 0o600 on the socket inode itself — the parent dir is already 0o700,
    // but explicit perms on the node defend against a future umask change
    // or a `chmod` that loosens the dir.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
    }
    info!(path = %path.display(), "control socket listening");

    tokio::spawn(async move {
        loop {
            match listener.accept().await {
                Ok((stream, _addr)) => {
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

async fn handle_conn(stream: UnixStream, state: ControlState) -> Result<()> {
    let (read_half, mut write_half) = stream.into_split();
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
    }
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
            tokio::task::spawn_blocking(move || crate::crypto::encrypt(&*key_for_task, &plaintext))
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

// ===== CLI client side =====

/// Connect to the daemon's control socket and run one `attach-file` RPC.
/// Returns the (blob_key, name) on success; bubbles up daemon-side errors
/// (no-such-chat, upload failure) verbatim.
pub async fn attach_file_via_socket(chat_id: &str, path: &str) -> Result<(String, String)> {
    let sock = control_socket_path();
    let stream = UnixStream::connect(&sock).await.with_context(|| {
        format!(
            "connect to {} — is zucchini-spawner running?",
            sock.display()
        )
    })?;
    let (read_half, mut write_half) = stream.into_split();
    let req = ControlRequest::AttachFile {
        chat_id: chat_id.to_string(),
        path: path.to_string(),
    };
    let mut payload = serde_json::to_string(&req).expect("serialize control request");
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
        return Err(anyhow!(
            "{}",
            resp.error.unwrap_or_else(|| "unknown error".to_string())
        ));
    }
    let blob_key = resp
        .blob_key
        .ok_or_else(|| anyhow!("response missing blob_key"))?;
    let name = resp.name.ok_or_else(|| anyhow!("response missing name"))?;
    Ok((blob_key, name))
}
