//! PowerSync sync-stream client (read-only).
//!
//! Opens `POST {base_url}/sync/stream`, streams NDJSON frames, maintains a per-bucket
//! cursor in memory, and forwards row operations + checkpoint snapshots to the caller
//! as `SyncEvent`s. Persistence of the cursor is the caller's concern (see `main.rs`
//! and `state::Mirror`), so the cursor stays consistent with whatever else the caller
//! persists (e.g. the projects mirror) in a single on-disk write.
//!
//! Scope is deliberately narrow: no SQLite, no CRUD upload, no conflict resolution.
//! The spawner only reads; agent replies are written through a separate API.
//!
//! Checksum validation is not yet implemented — the TCP/TLS layer catches corruption;
//! PowerSync's checksum is defense-in-depth we can add later.

use std::collections::HashMap;
use std::time::Duration;

use futures_util::{future::BoxFuture, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::io::AsyncBufReadExt;
use tokio::sync::mpsc;
use tokio_util::io::StreamReader;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

use crate::power::WakeSignal;

const TOKEN_REFRESH_THRESHOLD_SECS: i64 = 60;
const INITIAL_BACKOFF: Duration = Duration::from_secs(1);
const MAX_BACKOFF: Duration = Duration::from_secs(30);

/// Max silence on the stream — no frame of any kind — before we treat the connection
/// as dead and force a reconnect from the saved cursor.
///
/// Sizing is empirical (see the 2026-05-31 stall investigation). PowerSync sends a
/// keepalive frame every ~18-22s whenever the stream is otherwise idle (data frames
/// reset that timer, so keepalives only show up during quiet periods). So on a healthy
/// link SOME frame — data, checkpoint, or keepalive — always lands within ~25s: across
/// an overnight idle window the inter-frame distribution was tightly bunched under 25s
/// with a clean empty band at 45-60s. Above that band sit only genuine read-stalls:
/// the long-lived `POST /sync/stream` half-opens after a network blip (Wi-Fi/VPN/NAT)
/// and goes totally silent — observed at 60s, 103s, and the original 231s — while
/// short-lived `/api/writes` POSTs keep succeeding on separate sockets. 60s sits in the
/// empty band: above all healthy traffic (~3 missed keepalives), below every stall, so
/// it caps recovery at 60s without false-positive reconnects. Without this watchdog
/// `lines.next_line()` blocks until the OS finally tears the socket down — surfacing as
/// "sent but not delivered" until the network path heals on its own.
const FRAME_IDLE_TIMEOUT: Duration = Duration::from_secs(60);

/// OS-level TCP keepalive on the sync socket — defense in depth under the
/// application-level `FRAME_IDLE_TIMEOUT`. Probes a quiet connection so a truly
/// dead path surfaces as a read error/EOF (→ reconnect) instead of hanging.
const TCP_KEEPALIVE: Duration = Duration::from_secs(30);

// ---------- public API ----------

#[derive(Debug, Clone)]
pub enum SyncEvent {
    Put {
        table: String,
        id: String,
        /// Row columns as a JSON string (we request `raw_data: true` so data stays a string).
        data: String,
    },
    Remove {
        table: String,
        id: String,
    },
    /// Emitted on every `CheckpointComplete` frame. The caller should persist the
    /// snapshot atomically alongside any derived state (projects mirror, etc.).
    CheckpointComplete {
        buckets: HashMap<String, String>,
    },
}

pub type TokenFetcher =
    Box<dyn Fn() -> BoxFuture<'static, Result<String, anyhow::Error>> + Send + Sync>;

pub struct SyncConfig {
    pub base_url: String,
    pub client_id: String,
    pub initial_buckets: HashMap<String, String>,
    pub fetch_token: TokenFetcher,
    /// Fired when the system resumes from sleep; aborts the current sync stream so
    /// the outer loop reconnects immediately instead of waiting for TCP to time out.
    pub wake_signal: WakeSignal,
    /// Cancelled when `/auth/token` returns 410 (spawner permanently revoked).
    /// The run loop checks it after each failed connect and exits cleanly;
    /// main's revoked-handler runs the self-uninstall in parallel.
    pub revoked: CancellationToken,
}

enum ConnectExit {
    Eof,
    TokenRefresh,
    Wake,
    /// No frame (not even a keepalive) arrived within `FRAME_IDLE_TIMEOUT` —
    /// the read half is stalled. Reconnect immediately from the saved cursor.
    IdleTimeout,
}

pub fn start(config: SyncConfig) -> mpsc::Receiver<SyncEvent> {
    let (tx, rx) = mpsc::channel(256);
    tokio::spawn(async move {
        run(config, tx).await;
    });
    rx
}

// ---------- wire types ----------

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum SyncFrame {
    Checkpoint {
        checkpoint: Checkpoint,
    },
    CheckpointDiff {
        checkpoint_diff: CheckpointDiff,
    },
    Data {
        data: DataFrame,
    },
    CheckpointComplete {
        #[serde(rename = "checkpoint_complete")]
        _checkpoint_complete: serde::de::IgnoredAny,
    },
    Keepalive(Keepalive),
}

#[derive(Debug, Deserialize)]
struct Checkpoint {
    buckets: Vec<BucketCheckpoint>,
}

#[derive(Debug, Deserialize)]
struct CheckpointDiff {
    #[serde(default)]
    updated_buckets: Vec<BucketCheckpoint>,
    #[serde(default)]
    removed_buckets: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct BucketCheckpoint {
    bucket: String,
}

#[derive(Debug, Deserialize)]
struct DataFrame {
    bucket: String,
    next_after: String,
    data: Vec<OplogEntry>,
}

#[derive(Debug, Deserialize)]
struct OplogEntry {
    op: OpKind,
    object_type: Option<String>,
    object_id: Option<String>,
    /// With `raw_data: true` this is a JSON string, not a nested object.
    #[serde(default)]
    data: Option<String>,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "UPPERCASE")]
enum OpKind {
    Put,
    Remove,
    Move,
    Clear,
}

#[derive(Debug, Deserialize)]
struct Keepalive {
    token_expires_in: i64,
}

// ---------- request body ----------

#[derive(Debug, Serialize)]
struct StreamingSyncRequest {
    buckets: Vec<BucketRequest>,
    include_checksum: bool,
    raw_data: bool,
    client_id: String,
}

#[derive(Debug, Serialize)]
struct BucketRequest {
    name: String,
    after: String,
}

// ---------- main loop ----------

async fn run(config: SyncConfig, tx: mpsc::Sender<SyncEvent>) {
    let http = reqwest::Client::builder()
        .pool_idle_timeout(Duration::from_secs(60))
        .tcp_keepalive(TCP_KEEPALIVE)
        .build()
        .expect("reqwest client");

    let mut backoff = INITIAL_BACKOFF;
    let mut buckets = config.initial_buckets.clone();
    info!(buckets = buckets.len(), "starting sync stream");

    loop {
        let exit = connect_and_stream(&http, &config, &mut buckets, &tx).await;

        if tx.is_closed() {
            info!("receiver dropped, stopping sync");
            return;
        }

        match exit {
            Ok(reason) => {
                backoff = INITIAL_BACKOFF;
                match reason {
                    ConnectExit::Wake => {
                        info!("sync stream aborted by wake signal, reconnecting immediately");
                        continue;
                    }
                    ConnectExit::IdleTimeout => {
                        warn!(
                            timeout = ?FRAME_IDLE_TIMEOUT,
                            "no sync frame within idle timeout (stalled read half), reconnecting immediately"
                        );
                        continue;
                    }
                    ConnectExit::TokenRefresh => info!("sync stream ended for token refresh"),
                    ConnectExit::Eof => info!("sync stream ended cleanly"),
                }
            }
            Err(e) => {
                // Auth permanently revoked — main's revoked-handler is already
                // running the self-uninstall in parallel. Stop here so we don't
                // spam Sentry with 410 errors during the writer-drain window.
                if config.revoked.is_cancelled() {
                    info!(error = %e, "sync stream stopping (spawner revoked)");
                    return;
                }
                error!(error = %e, ?backoff, "sync stream failed, backing off");
            }
        }

        tokio::select! {
            _ = tokio::time::sleep(backoff) => {
                backoff = (backoff * 2).min(MAX_BACKOFF);
            }
            _ = config.wake_signal.notified() => {
                info!("wake signal during backoff, reconnecting immediately");
                backoff = INITIAL_BACKOFF;
            }
            _ = config.revoked.cancelled() => {
                info!("sync stream stopping (spawner revoked) during backoff");
                return;
            }
        }
    }
}

async fn connect_and_stream(
    http: &reqwest::Client,
    config: &SyncConfig,
    buckets: &mut HashMap<String, String>,
    tx: &mpsc::Sender<SyncEvent>,
) -> Result<ConnectExit, anyhow::Error> {
    let token = (config.fetch_token)().await?;

    let body = StreamingSyncRequest {
        buckets: buckets
            .iter()
            .map(|(name, after)| BucketRequest {
                name: name.clone(),
                after: after.clone(),
            })
            .collect(),
        include_checksum: true,
        raw_data: true,
        client_id: config.client_id.clone(),
    };

    let url = format!("{}/sync/stream", config.base_url.trim_end_matches('/'));
    debug!(%url, buckets = body.buckets.len(), "opening sync stream");

    let resp = http
        .post(&url)
        .header("Authorization", format!("Token {}", token))
        .header("Accept", "application/x-ndjson")
        .json(&body)
        .send()
        .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        anyhow::bail!("sync stream HTTP {}: {}", status, text);
    }

    let byte_stream = resp
        .bytes_stream()
        .map(|r| r.map_err(std::io::Error::other));
    let reader = StreamReader::new(byte_stream);
    let mut lines = tokio::io::BufReader::new(reader).lines();

    // Pin a single Notified future for the whole stream — re-creating it every
    // iteration would race wakes that fire between iterations.
    let wake = config.wake_signal.notified();
    tokio::pin!(wake);

    loop {
        let line = tokio::select! {
            biased;
            _ = &mut wake => return Ok(ConnectExit::Wake),
            // Bound the wait for the next frame. On a healthy link a keepalive
            // (or our 60s heartbeat's round-trip checkpoint) always lands inside
            // FRAME_IDLE_TIMEOUT; exceeding it means the read half has stalled,
            // so bail to the outer loop for an immediate reconnect from the
            // saved cursor rather than blocking indefinitely on a zombie socket.
            res = tokio::time::timeout(FRAME_IDLE_TIMEOUT, lines.next_line()) => match res {
                Ok(line) => line?,
                Err(_elapsed) => return Ok(ConnectExit::IdleTimeout),
            },
        };
        let line = match line {
            Some(l) => l,
            None => return Ok(ConnectExit::Eof),
        };
        if line.trim().is_empty() {
            continue;
        }

        let frame: SyncFrame = match serde_json::from_str(&line) {
            Ok(f) => f,
            Err(e) => {
                warn!(error = %e, line = %line, "failed to parse sync frame");
                continue;
            }
        };

        match frame {
            SyncFrame::Checkpoint { checkpoint } => {
                let keep: std::collections::HashSet<_> = checkpoint
                    .buckets
                    .iter()
                    .map(|b| &b.bucket)
                    .cloned()
                    .collect();
                buckets.retain(|name, _| keep.contains(name));
                for b in checkpoint.buckets {
                    buckets.entry(b.bucket).or_insert_with(|| "0".to_string());
                }
            }
            SyncFrame::CheckpointDiff { checkpoint_diff } => {
                for removed in checkpoint_diff.removed_buckets {
                    buckets.remove(&removed);
                }
                for b in checkpoint_diff.updated_buckets {
                    buckets.entry(b.bucket).or_insert_with(|| "0".to_string());
                }
            }
            SyncFrame::Data { data } => {
                for entry in data.data {
                    match entry.op {
                        OpKind::Put => {
                            if let (Some(table), Some(id), Some(row)) =
                                (entry.object_type, entry.object_id, entry.data)
                            {
                                if tx
                                    .send(SyncEvent::Put {
                                        table,
                                        id,
                                        data: row,
                                    })
                                    .await
                                    .is_err()
                                {
                                    return Ok(ConnectExit::Eof);
                                }
                            }
                        }
                        OpKind::Remove => {
                            if let (Some(table), Some(id)) = (entry.object_type, entry.object_id) {
                                if tx.send(SyncEvent::Remove { table, id }).await.is_err() {
                                    return Ok(ConnectExit::Eof);
                                }
                            }
                        }
                        OpKind::Move => {}
                        OpKind::Clear => {
                            buckets.insert(data.bucket.clone(), "0".to_string());
                        }
                    }
                }
                buckets.insert(data.bucket, data.next_after);
            }
            SyncFrame::CheckpointComplete { .. } => {
                if tx
                    .send(SyncEvent::CheckpointComplete {
                        buckets: buckets.clone(),
                    })
                    .await
                    .is_err()
                {
                    return Ok(ConnectExit::Eof);
                }
                debug!("checkpoint complete, emitted snapshot");
            }
            SyncFrame::Keepalive(k) => {
                debug!(token_expires_in = k.token_expires_in, "keepalive frame");
                if k.token_expires_in < TOKEN_REFRESH_THRESHOLD_SECS {
                    info!(
                        expires_in = k.token_expires_in,
                        "token near expiry, reconnecting with fresh token"
                    );
                    return Ok(ConnectExit::TokenRefresh);
                }
            }
        }
    }
}
