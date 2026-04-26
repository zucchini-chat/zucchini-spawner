//! PowerSync sync-stream client (read-only).
//!
//! Opens `POST {base_url}/sync/stream`, streams NDJSON frames, maintains a per-bucket
//! cursor in memory, and forwards row operations + checkpoint snapshots to the caller
//! as `SyncEvent`s. Persistence of the cursor is the caller's concern (see `main.rs`
//! + `state::Mirror`), so the cursor stays consistent with whatever else the caller
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
use tracing::{debug, error, info, warn};

const TOKEN_REFRESH_THRESHOLD_SECS: i64 = 60;
const INITIAL_BACKOFF: Duration = Duration::from_secs(1);
const MAX_BACKOFF: Duration = Duration::from_secs(30);

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
    Checkpoint { checkpoint: Checkpoint },
    CheckpointDiff { checkpoint_diff: CheckpointDiff },
    Data { data: DataFrame },
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
        .build()
        .expect("reqwest client");

    let mut backoff = INITIAL_BACKOFF;
    let mut buckets = config.initial_buckets.clone();
    info!(buckets = buckets.len(), "starting sync stream");

    loop {
        match connect_and_stream(&http, &config, &mut buckets, &tx).await {
            Ok(()) => {
                info!("sync stream ended cleanly");
                backoff = INITIAL_BACKOFF;
            }
            Err(e) => {
                error!(error = %e, ?backoff, "sync stream failed, backing off");
            }
        }

        if tx.is_closed() {
            info!("receiver dropped, stopping sync");
            return;
        }

        tokio::time::sleep(backoff).await;
        backoff = (backoff * 2).min(MAX_BACKOFF);
    }
}

async fn connect_and_stream(
    http: &reqwest::Client,
    config: &SyncConfig,
    buckets: &mut HashMap<String, String>,
    tx: &mpsc::Sender<SyncEvent>,
) -> Result<(), anyhow::Error> {
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

    let byte_stream = resp.bytes_stream().map(|r| r.map_err(std::io::Error::other));
    let reader = StreamReader::new(byte_stream);
    let mut lines = tokio::io::BufReader::new(reader).lines();

    while let Some(line) = lines.next_line().await? {
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
                let keep: std::collections::HashSet<_> =
                    checkpoint.buckets.iter().map(|b| &b.bucket).cloned().collect();
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
                                    .send(SyncEvent::Put { table, id, data: row })
                                    .await
                                    .is_err()
                                {
                                    return Ok(());
                                }
                            }
                        }
                        OpKind::Remove => {
                            if let (Some(table), Some(id)) = (entry.object_type, entry.object_id) {
                                if tx.send(SyncEvent::Remove { table, id }).await.is_err() {
                                    return Ok(());
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
                    .send(SyncEvent::CheckpointComplete { buckets: buckets.clone() })
                    .await
                    .is_err()
                {
                    return Ok(());
                }
                debug!("checkpoint complete, emitted snapshot");
            }
            SyncFrame::Keepalive(k) => {
                if k.token_expires_in < TOKEN_REFRESH_THRESHOLD_SECS {
                    info!(
                        expires_in = k.token_expires_in,
                        "token near expiry, reconnecting with fresh token"
                    );
                    return Ok(());
                }
            }
        }
    }

    Ok(())
}
