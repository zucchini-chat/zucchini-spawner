//! Tunnels: a public URL (`https://<id>.zuc.run`) for a local port, bound to
//! a chat like a watch. The daemon keeps one WebSocket per tunnel to the relay
//! (piko behind our queue proxy, `tunnel-relay/`); piko opens a yamux stream
//! per request, which is served here and forwarded to `localhost:<port>`.
//!
//! Each tunnel has its own Ed25519 key; its id is derived from the public key,
//! so the URL lives as long as the tunnel. The backend signs a ticket (a JWT
//! scoped to that id) after we prove possession of the key; piko checks it on
//! connect and cuts the connection at its `exp`. It is refreshed daily and the
//! connection replaced with the fresh one, so a tunnel lives for weeks.
//!
//! A tunnel lives in `~/.zucchini-spawner/tunnels/<id>.json` (key, latest
//! ticket, port, flags, chat). A removed tunnel's file is rewritten with
//! `removed_at` and kept until the relay has dropped its queue (`DELETE /queue`). `stopped/<chat_id>`
//! lists what a `/stop` removed until the chat's agent is told.

use std::collections::{HashMap, HashSet};
use std::convert::Infallible;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::task::{ready, Context as TaskContext, Poll};
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine};
use crypto_box::aead::{rand_core::RngCore, OsRng};
use ed25519_dalek::{Signer, SigningKey};
use futures::{AsyncRead, AsyncWrite, Sink, Stream};
use http_body_util::{combinators::BoxBody, BodyExt, Full};
use hyper::body::{Bytes, Incoming};
use hyper::header::{self, HeaderValue};
use hyper::{Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio_tungstenite::tungstenite::{self, client::IntoClientRequest, Message};
use tokio_util::compat::FuturesAsyncReadCompatExt;
use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn};
use uuid::Uuid;

use crate::watch::Changed;
use crate::writer::TokenFetcher;

const MAX_PER_CHAT: usize = 10;
const MAX_TUNNELS: usize = 50;

/// How long the replaced connection keeps serving what it already took.
const SWITCH_GRACE: Duration = Duration::from_secs(30);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
const LOCAL_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const MAX_BACKOFF: Duration = Duration::from_secs(300);
/// The relay forgets an id unseen this long, queue included.
const PURGE_GIVE_UP_SECS: i64 = 30 * 24 * 3600;
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);
const UNINSTALL_PURGE_WAIT: Duration = Duration::from_secs(10);
const CREATE_CONNECT_WAIT: Duration = Duration::from_secs(10);
const STOP_NOTES_DIR: &str = "stopped";
const TOKEN_PARAM: &str = "zt";
pub const UPSTREAM_DOWN: &str = "upstream-down";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Auth {
    Token { token: String },
    Basic { user: String, password: String },
}

impl Auth {
    pub fn generate(kind: &str) -> Result<Self> {
        match kind {
            "token" => Ok(Auth::Token {
                token: random_base32(32),
            }),
            "basic" => Ok(Auth::Basic {
                user: "zucchini".into(),
                password: random_base32(20),
            }),
            other => bail!("invalid --auth {other:?}: use token or basic"),
        }
    }

    fn describe(&self) -> String {
        match self {
            Auth::Token { token } => format!(
                "token {token} (send `Authorization: Bearer {token}`, or add `?{TOKEN_PARAM}={token}` to the URL in a browser or for a WebSocket)"
            ),
            Auth::Basic { user, password } => {
                format!("basic, user {user}, password {password} (browsers prompt for it)")
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Ticket {
    ticket: String,
    exp: i64,
    relay: String,
    url: String,
    #[serde(default)]
    fetched_at: i64,
}

/// When tickets are refreshed and connections replaced; short in tests.
#[derive(Clone, Copy)]
struct Timing {
    refresh_after_secs: i64,
    /// A connection is replaced this long before its ticket's `exp`, when
    /// piko cuts it.
    reconnect_before_exp_secs: i64,
    tick: Duration,
}

impl Default for Timing {
    fn default() -> Self {
        Self {
            refresh_after_secs: 24 * 3600,
            reconnect_before_exp_secs: 3600,
            tick: Duration::from_secs(60),
        }
    }
}

impl Ticket {
    fn refresh_due(&self, timing: &Timing) -> bool {
        unix_now() - self.fetched_at >= timing.refresh_after_secs || self.expiring(timing)
    }

    fn expiring(&self, timing: &Timing) -> bool {
        self.exp - unix_now() < timing.reconnect_before_exp_secs
    }

    /// `wss://host` → `https://host`: the queue proxy serves `/queue/*` there.
    fn relay_http(&self) -> String {
        let relay = self.relay.trim_end_matches('/');
        if let Some(rest) = relay.strip_prefix("wss://") {
            format!("https://{rest}")
        } else if let Some(rest) = relay.strip_prefix("ws://") {
            format!("http://{rest}")
        } else {
            relay.to_string()
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TunnelDef {
    pub id: String,
    pub chat_id: String,
    pub user_id: Uuid,
    pub port: u16,
    pub auth: Option<Auth>,
    pub keep_host: bool,
    pub no_queue: bool,
    /// Unix seconds.
    pub created_at: i64,
    /// Base64 Ed25519 seed; never leaves the machine.
    key: String,
    ticket: Ticket,
    /// Unix seconds; set once removed, while its queue purge is pending.
    #[serde(default)]
    removed_at: Option<i64>,
}

impl TunnelDef {
    pub fn url(&self) -> &str {
        &self.ticket.url
    }

    fn signing_key(&self) -> Result<SigningKey> {
        let seed: [u8; 32] = STANDARD
            .decode(&self.key)
            .ok()
            .and_then(|b| b.try_into().ok())
            .ok_or_else(|| anyhow!("tunnel {} has a corrupt key", self.id))?;
        Ok(SigningKey::from_bytes(&seed))
    }

    /// Its `chats.live_tasks` entry.
    pub fn live_task(&self) -> serde_json::Value {
        serde_json::json!({
            "i": self.id,
            "d": format!("{} → {}", self.port, self.url()),
            "k": "tunnel",
            "u": self.url(),
            "s0": self.created_at.saturating_mul(1000),
        })
    }

    fn info(&self, connected: bool) -> TunnelInfo {
        TunnelInfo {
            id: self.id.clone(),
            url: self.url().to_string(),
            port: self.port,
            auth: self.auth.clone(),
            keep_host: self.keep_host,
            no_queue: self.no_queue,
            connected,
        }
    }
}

/// A tunnel as the CLI sees it: no key, no ticket.
/// A tunnel with its key and ticket, not started yet.
pub struct Prepared {
    def: TunnelDef,
    /// [`Inner::stops`] of its chat before the ticket fetch.
    stops: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TunnelInfo {
    pub id: String,
    pub url: String,
    pub port: u16,
    pub auth: Option<Auth>,
    pub keep_host: bool,
    pub no_queue: bool,
    pub connected: bool,
}

impl TunnelInfo {
    pub fn summary(&self) -> String {
        let mut out = format!("{} → localhost:{}  (id {})\n", self.url, self.port, self.id);
        if let Some(auth) = &self.auth {
            out.push_str(&format!("auth: {}\n", auth.describe()));
        }
        if self.keep_host {
            out.push_str("keeps the Host header\n");
        }
        if self.no_queue {
            out.push_str("no offline queue\n");
        }
        if !self.connected {
            out.push_str("not connected to the relay yet, retrying\n");
        }
        out
    }
}

/// A message from the daemon to a tunnel's chat, delivered like a watch's.
pub struct Notice {
    pub chat_id: String,
    pub user_id: Uuid,
    /// Whom the chat's system row names.
    pub sender: String,
    /// Heads the turn the agent gets.
    pub label: String,
    pub text: String,
    /// Cancelled once the tunnel is removed: no turn after a `/stop`.
    pub cancelled: CancellationToken,
}

pub type Deliver =
    Box<dyn Fn(Notice) -> futures::future::BoxFuture<'static, Result<()>> + Send + Sync>;

/// Where tickets come from: the machine's own backend auth.
#[derive(Clone)]
pub struct Backend {
    pub http: reqwest::Client,
    pub api_base_url: String,
    pub fetch_token: Arc<TokenFetcher>,
}

impl Backend {
    async fn ticket(&self, key: &SigningKey) -> Result<Ticket> {
        let pubkey = key.verifying_key().to_bytes();
        let id = tunnel_id(&pubkey);
        let ts = unix_now();
        let sig = key.sign(format!("zucchini-tunnel-ticket|{id}|{ts}").as_bytes());
        let url = format!(
            "{}/api/tunnel/ticket",
            self.api_base_url.trim_end_matches('/')
        );
        let token = (self.fetch_token)()
            .await
            .context("fetch JWT for tunnel ticket")?;
        let resp = self
            .http
            .post(&url)
            .bearer_auth(token)
            .json(&serde_json::json!({
                "pubkey": STANDARD.encode(pubkey),
                "ts": ts,
                "sig": STANDARD.encode(sig.to_bytes()),
            }))
            .send()
            .await
            .context("POST /api/tunnel/ticket")?;
        let status = resp.status();
        if status == reqwest::StatusCode::NOT_FOUND {
            bail!("the Zucchini backend doesn't support tunnels yet");
        }
        if !status.is_success() {
            let body = crate::net::capture_response_failure("POST", &url, resp).await;
            bail!("POST /api/tunnel/ticket {status}: {body}");
        }
        #[derive(Deserialize)]
        struct Res {
            ticket: String,
            id: String,
            exp: i64,
            relay: String,
            url: String,
        }
        let res: Res = resp.json().await.context("parse tunnel ticket")?;
        if res.id != id {
            bail!("the backend issued a ticket for {}, not {id}", res.id);
        }
        Ok(Ticket {
            ticket: res.ticket,
            exp: res.exp,
            relay: res.relay,
            url: res.url,
            fetched_at: unix_now(),
        })
    }

    /// `POST /queue/hello` or `DELETE /queue` on the relay.
    async fn relay_call(
        &self,
        ticket: &Ticket,
        method: reqwest::Method,
        path: &str,
        body: Option<serde_json::Value>,
    ) -> Result<()> {
        let url = format!("{}{path}", ticket.relay_http());
        let mut req = self.http.request(method, &url).bearer_auth(&ticket.ticket);
        if let Some(body) = body {
            req = req.json(&body);
        }
        let resp = req.send().await.with_context(|| format!("relay {path}"))?;
        match resp.status().as_u16() {
            410 => return Err(Blocked.into()),
            401 | 403 => return Err(StaleTicket.into()),
            _ => {}
        }
        if !resp.status().is_success() {
            bail!("relay {path}: {}", resp.status());
        }
        Ok(())
    }
}

/// `base32(sha256(pubkey))[:26]`, lowercase: 128 bits, a valid DNS label.
pub fn tunnel_id(pubkey: &[u8]) -> String {
    let mut id = base32_lower(&Sha256::digest(pubkey));
    id.truncate(26);
    id
}

/// RFC 4648 base32 in lowercase, no padding.
fn base32_lower(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 32] = b"abcdefghijklmnopqrstuvwxyz234567";
    let mut out = String::new();
    let (mut acc, mut bits) = (0u32, 0);
    for &b in bytes {
        acc = (acc << 8) | b as u32;
        bits += 8;
        while bits >= 5 {
            bits -= 5;
            out.push(ALPHABET[((acc >> bits) & 31) as usize] as char);
        }
    }
    if bits > 0 {
        out.push(ALPHABET[((acc << (5 - bits)) & 31) as usize] as char);
    }
    out
}

fn random_base32(n: usize) -> String {
    let mut bytes = vec![0u8; n];
    OsRng.fill_bytes(&mut bytes);
    base32_lower(&bytes)
}

struct Entry {
    def: TunnelDef,
    connected: Arc<AtomicBool>,
    cancel: CancellationToken,
    task: tokio::task::JoinHandle<()>,
}

struct Inner {
    root: PathBuf,
    backend: Backend,
    deliver: Deliver,
    changed: Changed,
    timing: Timing,
    entries: Mutex<HashMap<String, Entry>>,
    /// Per chat, the `/stop`s since boot (`chat_stopped`).
    stops: Mutex<HashMap<String, u64>>,
    restored_admitted: CancellationToken,
    /// Set by `shutdown` (empty by `uninstall`): nothing starts after, and
    /// the tunnels it stopped keep their chats' rows in `state.json` and
    /// their `live_tasks` entries.
    stopped_at_shutdown: OnceLock<Vec<TunnelDef>>,
    /// Cancels the purges at shutdown; the next start resumes them.
    purges: CancellationToken,
    purge_tasks: Mutex<Vec<tokio::task::JoinHandle<()>>>,
}

#[derive(Clone)]
pub struct Tunnels(Arc<Inner>);

static GLOBAL: OnceLock<Tunnels> = OnceLock::new();

/// Load the tunnels on disk. The machine owner's connect at once with their
/// cached tickets (the owner can't lose access; a deleted chat's tunnel goes
/// at [`admit_restored`]); a member's wait for that access check. Pending
/// queue purges resume at once. Call once, holding the single-instance lock.
pub fn start(backend: Backend, deliver: Deliver, changed: Changed, owner: Option<Uuid>) {
    let root = crate::zucchini_spawner_dir().join("tunnels");
    let _ = GLOBAL.set(Tunnels::start(
        root,
        backend,
        deliver,
        changed,
        Timing::default(),
        owner,
    ));
}

pub fn global() -> Result<&'static Tunnels> {
    GLOBAL
        .get()
        .ok_or_else(|| anyhow!("tunnels are not running in this spawner"))
}

/// The first checkpoint is applied: the restored tunnels connect, except those
/// `allowed` refuses, which are removed.
pub fn admit_restored(allowed: impl Fn(&TunnelDef) -> bool) {
    if let Some(t) = GLOBAL.get() {
        t.admit_restored(allowed);
    }
}

/// The chats that own a tunnel; their rows are kept in `state.json`.
pub fn tunnel_chat_ids() -> HashSet<String> {
    GLOBAL
        .get()
        .map(|t| t.0.defs().into_iter().map(|d| d.chat_id).collect())
        .unwrap_or_default()
}

/// The chat's tunnels, oldest first: its first entries in `chats.live_tasks`.
pub fn running_in(chat_id: &str) -> Vec<TunnelDef> {
    GLOBAL
        .get()
        .map(|t| t.running_in(chat_id))
        .unwrap_or_default()
}

pub fn chat_deleted(chat_id: &str) {
    if let Some(t) = GLOBAL.get() {
        t.remove(|d| d.chat_id == chat_id);
        let _ = std::fs::remove_file(t.0.stop_note_path(chat_id));
    }
}

/// `/stop`: the chat's tunnels go; [`take_stop_note`] tells its agent later.
/// `true` = it had some.
pub fn chat_stopped(chat_id: &str) -> bool {
    GLOBAL.get().is_some_and(|t| t.chat_stopped(chat_id))
}

pub fn take_stop_note(chat_id: &str) -> Option<String> {
    GLOBAL.get()?.take_stop_note(chat_id)
}

/// The member was revoked or re-sandboxed: their tunnels go.
pub fn member_restricted(user_id: Uuid) {
    if let Some(t) = GLOBAL.get() {
        t.remove(|d| d.user_id == user_id);
    }
}

/// Close every connection; the tunnels stay on disk for the next start.
pub async fn shutdown() {
    if let Some(t) = GLOBAL.get() {
        t.shutdown().await;
    }
}

/// The machine is removed: every tunnel goes, and its relay queue gets a short
/// wait to be purged, as no next start will retry it.
pub async fn uninstall() {
    if let Some(t) = GLOBAL.get() {
        t.uninstall(UNINSTALL_PURGE_WAIT).await;
    }
}

impl Tunnels {
    fn start(
        root: PathBuf,
        backend: Backend,
        deliver: Deliver,
        changed: Changed,
        timing: Timing,
        owner: Option<Uuid>,
    ) -> Self {
        let inner = Arc::new(Inner {
            root,
            backend,
            deliver,
            changed,
            entries: Mutex::new(HashMap::new()),
            stops: Mutex::default(),
            timing,
            restored_admitted: CancellationToken::new(),
            stopped_at_shutdown: OnceLock::new(),
            purges: CancellationToken::new(),
            purge_tasks: Mutex::default(),
        });
        for path in json_files(&inner.root) {
            match load_def(&path) {
                Ok(def) if def.removed_at.is_some() => inner.purge_in_background(def, None),
                Ok(def) => {
                    let wait = owner != Some(def.user_id);
                    if let Err(e) = inner.launch(def, None, wait) {
                        warn!(error = %e, "tunnel not restored");
                    }
                }
                Err(e) => warn!(path = %path.display(), error = %e, "skipping unreadable tunnel"),
            }
        }
        Self(inner)
    }

    fn admit_restored(&self, allowed: impl Fn(&TunnelDef) -> bool) {
        if self.0.restored_admitted.is_cancelled() {
            return;
        }
        self.remove(|d| !allowed(d));
        self.0.restored_admitted.cancel();
        let chats: HashSet<String> = self.0.defs().into_iter().map(|d| d.chat_id).collect();
        for chat_id in chats {
            self.0.notify(&chat_id);
        }
    }

    fn running_in(&self, chat_id: &str) -> Vec<TunnelDef> {
        let mut defs = self.0.defs();
        defs.retain(|d| d.chat_id == chat_id);
        defs.sort_by(|a, b| (a.created_at, &a.id).cmp(&(b.created_at, &b.id)));
        defs
    }

    /// A new tunnel's key and ticket; [`register`](Self::register) starts it.
    /// The caller has checked the chat's access.
    pub async fn prepare(
        &self,
        chat_id: &str,
        user_id: Uuid,
        port: u16,
        auth: Option<Auth>,
        keep_host: bool,
        no_queue: bool,
    ) -> Result<Prepared> {
        if port == 0 {
            bail!("--port must be 1-65535");
        }
        let stops = self.0.stops(chat_id);
        check_caps(&self.0.entries(), chat_id)?;
        let mut seed = [0u8; 32];
        OsRng.fill_bytes(&mut seed);
        let key = SigningKey::from_bytes(&seed);
        let ticket = self.0.backend.ticket(&key).await?;
        let def = TunnelDef {
            id: tunnel_id(&key.verifying_key().to_bytes()),
            chat_id: chat_id.to_string(),
            user_id,
            port,
            auth,
            keep_host,
            no_queue,
            created_at: unix_now(),
            key: STANDARD.encode(seed),
            ticket,
            removed_at: None,
        };
        Ok(Prepared { def, stops })
    }

    /// Save, list and connect a prepared tunnel. The caller re-checks the
    /// chat's access and saves state.json with it, under the mirror's lock.
    pub fn register(&self, prepared: Prepared) -> Result<String> {
        let Prepared { def, stops } = prepared;
        self.0.launch(def.clone(), Some(stops), false)?;
        self.0.notify(&def.chat_id);
        info!(id = %def.id, chat_id = %def.chat_id, port = def.port, "tunnel created");
        Ok(def.id)
    }

    /// The tunnel once it connected, or after a short wait without.
    pub async fn wait_connected(&self, id: &str) -> Result<TunnelInfo> {
        let deadline = tokio::time::Instant::now() + CREATE_CONNECT_WAIT;
        loop {
            let info = {
                let entries = self.0.entries();
                let entry = entries
                    .get(id)
                    .ok_or_else(|| anyhow!("the tunnel was removed right after its creation"))?;
                entry.def.info(entry.connected.load(Ordering::Relaxed))
            };
            if info.connected || tokio::time::Instant::now() >= deadline {
                return Ok(info);
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    pub fn list(&self, chat_id: &str) -> Vec<TunnelInfo> {
        let connected: HashMap<String, bool> = self
            .0
            .entries()
            .values()
            .map(|e| (e.def.id.clone(), e.connected.load(Ordering::Relaxed)))
            .collect();
        self.running_in(chat_id)
            .iter()
            .map(|d| d.info(connected.get(&d.id).copied().unwrap_or(false)))
            .collect()
    }

    pub fn delete(&self, chat_id: &str, id: &str) -> Result<TunnelInfo> {
        let mut removed = self.remove(|d| d.chat_id == chat_id && d.id == id);
        let (def, saved) = removed
            .pop()
            .ok_or_else(|| anyhow!("this chat has no tunnel {id} (`tunnel list`)"))?;
        saved.context(
            "the tunnel is closed, but its removal could not be saved: \
             it comes back after a spawner restart",
        )?;
        Ok(def.info(false))
    }

    fn chat_stopped(&self, chat_id: &str) -> bool {
        // Before the removal: a create still fetching its ticket is refused.
        *self
            .0
            .stops
            .lock()
            .expect("tunnel stops mutex")
            .entry(chat_id.to_string())
            .or_default() += 1;
        let stopped = self.remove(|d| d.chat_id == chat_id);
        if stopped.is_empty() {
            return false;
        }
        let path = self.0.stop_note_path(chat_id);
        let mut note = std::fs::read_to_string(&path).unwrap_or_default();
        for (t, _) in &stopped {
            note.push_str(&format!("{} (port {})\n", t.url(), t.port));
        }
        let saved = std::fs::create_dir_all(self.0.root.join(STOP_NOTES_DIR))
            .and_then(|()| crate::atomic::atomic_write_private(&path, note.as_bytes()));
        if let Err(e) = saved {
            warn!(chat_id, error = %e, "failed to keep the tunnel stop note");
        }
        true
    }

    fn take_stop_note(&self, chat_id: &str) -> Option<String> {
        let path = self.0.stop_note_path(chat_id);
        let note = std::fs::read_to_string(&path).ok()?;
        let _ = std::fs::remove_file(&path);
        let tunnels: Vec<&str> = note.lines().collect();
        (!tunnels.is_empty()).then(|| {
            format!(
                "[zucchini] The user pressed Stop, which also removed this chat's tunnels: {}. \
                 Their URLs no longer work.",
                tunnels.join(", ")
            )
        })
    }

    /// Close the matching tunnels and purge their relay queues in the background.
    /// An `Err` = the removal isn't saved: the tunnel returns after a restart.
    fn remove(&self, pred: impl Fn(&TunnelDef) -> bool) -> Vec<(TunnelDef, Result<()>)> {
        let removed: Vec<(Entry, TunnelDef, Result<()>)> = {
            let mut entries = self.0.entries();
            let ids: Vec<String> = entries
                .values()
                .filter(|e| pred(&e.def))
                .map(|e| e.def.id.clone())
                .collect();
            let removed: Vec<Entry> = ids.iter().filter_map(|id| entries.remove(id)).collect();
            // Under the lock, so a ticket refresh can't write the file back.
            removed
                .into_iter()
                .map(|e| {
                    let (purge, saved) = self.0.mark_removed(&e.def);
                    (e, purge, saved)
                })
                .collect()
        };
        let mut defs = Vec::new();
        for (e, purge, saved) in removed {
            e.cancel.cancel();
            info!(id = %e.def.id, "tunnel removed");
            self.0.notify(&e.def.chat_id);
            self.0.purge_in_background(purge, Some(e.task));
            defs.push((e.def, saved));
        }
        defs
    }

    async fn uninstall(&self, wait: Duration) {
        // Before collecting: a create registering during the purge wait would
        // be closed unpurged by the shutdown below.
        let _ = self.0.stopped_at_shutdown.set(Vec::new());
        self.remove(|_| true);
        let purges = std::mem::take(&mut *self.0.purge_tasks.lock().expect("tunnel purges mutex"));
        if tokio::time::timeout(wait, futures::future::join_all(purges))
            .await
            .is_err()
        {
            warn!("tunnel queue purges did not finish before the uninstall");
        }
        self.shutdown().await;
    }

    async fn shutdown(&self) {
        let entries: Vec<Entry> = {
            let mut entries = self.0.entries();
            let defs = entries.values().map(|e| e.def.clone()).collect();
            let _ = self.0.stopped_at_shutdown.set(defs);
            entries.drain().map(|(_, e)| e).collect()
        };
        self.0.purges.cancel();
        for e in &entries {
            e.cancel.cancel();
        }
        let join = futures::future::join_all(entries.into_iter().map(|e| e.task));
        if tokio::time::timeout(SHUTDOWN_TIMEOUT, join).await.is_err() {
            warn!("tunnels did not close in time");
        }
    }
}

impl Inner {
    fn entries(&self) -> MutexGuard<'_, HashMap<String, Entry>> {
        self.entries.lock().expect("tunnel entries mutex")
    }

    fn defs(&self) -> Vec<TunnelDef> {
        let entries = self.entries();
        match self.stopped_at_shutdown.get() {
            Some(defs) => defs.clone(),
            None => entries.values().map(|e| e.def.clone()).collect(),
        }
    }

    fn notify(&self, chat_id: &str) {
        let _ = self.changed.send(chat_id.to_string());
    }

    fn def_path(&self, id: &str) -> PathBuf {
        self.root.join(format!("{id}.json"))
    }

    fn stop_note_path(&self, chat_id: &str) -> PathBuf {
        self.root.join(STOP_NOTES_DIR).join(chat_id)
    }

    /// Unsaved, the file stays as it was, so a restart before the purge
    /// lands reopens the tunnel rather than forgetting its queue.
    fn mark_removed(&self, def: &TunnelDef) -> (TunnelDef, Result<()>) {
        let def = TunnelDef {
            removed_at: Some(unix_now()),
            ..def.clone()
        };
        let saved = save_def(&self.def_path(&def.id), &def);
        if let Err(e) = &saved {
            error!(id = %def.id, error = %e, "failed to keep the tunnel's queue purge");
        }
        (def, saved)
    }

    /// Register and start a tunnel. A new one (`fresh`: the chat's stop count
    /// read before its ticket was fetched) is checked against the caps and
    /// refused if the chat was stopped meanwhile, all under the entries lock,
    /// and saved before it is listed.
    fn launch(
        self: &Arc<Self>,
        def: TunnelDef,
        fresh: Option<u64>,
        wait_for_admission: bool,
    ) -> Result<Arc<AtomicBool>> {
        let mut entries = self.entries();
        if self.stopped_at_shutdown.get().is_some() {
            bail!("the spawner is shutting down; create the tunnel again in a moment");
        }
        if let Some(stops) = fresh {
            if self.stops(&def.chat_id) != stops {
                bail!("the chat was stopped while the tunnel was being created");
            }
            check_caps(&entries, &def.chat_id)?;
            save_def(&self.def_path(&def.id), &def)?;
        }
        let cancel = CancellationToken::new();
        let connected = Arc::new(AtomicBool::new(false));
        let task = tokio::spawn(self.clone().run(
            def.clone(),
            connected.clone(),
            cancel.clone(),
            wait_for_admission,
        ));
        entries.insert(
            def.id.clone(),
            Entry {
                def,
                connected: connected.clone(),
                cancel,
                task,
            },
        );
        Ok(connected)
    }

    /// How often the chat was stopped since boot.
    fn stops(&self, chat_id: &str) -> u64 {
        let stops = self.stops.lock().expect("tunnel stops mutex");
        stops.get(chat_id).copied().unwrap_or_default()
    }

    /// Keep the tunnel connected until it is removed or the spawner stops.
    /// Ends once no hello is in flight, so the purge after it is the last word.
    async fn run(
        self: Arc<Self>,
        def: TunnelDef,
        connected: Arc<AtomicBool>,
        cancel: CancellationToken,
        wait_for_admission: bool,
    ) {
        let mut hellos = Hellos::default();
        self.keep_connected(def, connected, &cancel, wait_for_admission, &mut hellos)
            .await;
        hellos.join().await;
    }

    async fn keep_connected(
        self: &Arc<Self>,
        mut def: TunnelDef,
        connected: Arc<AtomicBool>,
        cancel: &CancellationToken,
        wait_for_admission: bool,
        hellos: &mut Hellos,
    ) {
        if wait_for_admission {
            tokio::select! {
                biased;
                () = cancel.cancelled() => return,
                () = self.restored_admitted.cancelled() => {}
            }
        }
        let mut backoff = Backoff::default();
        let mut stale_ticket = false;
        // The connection serving now. A refresh keeps it until the next one
        // is up, then drains it (make before break).
        let mut current: Option<Connection> = None;
        let serving = |c: &Option<Connection>| c.as_ref().is_some_and(|c| !c.served.is_finished());
        loop {
            if stale_ticket || def.ticket.refresh_due(&self.timing) {
                let Some(refreshed) = until_cancelled(cancel, self.refresh(&mut def)).await else {
                    return;
                };
                match refreshed {
                    Ok(()) => stale_ticket = false,
                    // The cached ticket still connects (backend outage).
                    Err(e) if !stale_ticket && !def.ticket.expiring(&self.timing) => {
                        warn!(id = %def.id, error = %e, "tunnel ticket refresh failed, using the cached one");
                    }
                    Err(e) => {
                        warn!(id = %def.id, error = %e, "tunnel ticket refresh failed");
                        connected.store(serving(&current), Ordering::Relaxed);
                        if !backoff.wait(cancel).await {
                            return;
                        }
                        continue;
                    }
                }
            }
            let Some(connection) = until_cancelled(cancel, connect(&def)).await else {
                return;
            };
            let ws = match connection {
                Ok(ws) => ws,
                Err(ConnectError::Blocked) => {
                    warn!(id = %def.id, "the relay blocked the tunnel");
                    self.end_blocked(&def, cancel).await;
                    return;
                }
                Err(e) => {
                    match e {
                        ConnectError::Refused(status) => {
                            warn!(id = %def.id, %status, "relay refused the tunnel ticket");
                            stale_ticket = true;
                        }
                        ConnectError::Other(e) => {
                            warn!(id = %def.id, error = %e, "tunnel connect failed")
                        }
                        ConnectError::Blocked => unreachable!(),
                    }
                    connected.store(serving(&current), Ordering::Relaxed);
                    if !backoff.wait(cancel).await {
                        return;
                    }
                    continue;
                }
            };
            backoff = Backoff::default();
            info!(id = %def.id, "tunnel connected");
            let new = Connection::start(ws, &def, cancel);
            let stop = cancel.child_token();
            hellos.start(stop.clone(), self.clone().hello(def.clone(), stop));
            if let Some(old) = current.replace(new) {
                old.retire.cancel();
            }
            connected.store(true, Ordering::Relaxed);
            let live = current.as_mut().expect("just set");
            let mut next_refresh = tokio::time::Instant::now();
            let refreshed = loop {
                tokio::select! {
                    () = cancel.cancelled() => {
                        let _ = (&mut live.served).await;
                        return;
                    }
                    _ = &mut live.served => break false,
                    () = tokio::time::sleep(self.timing.tick) => {
                        if !def.ticket.refresh_due(&self.timing) || tokio::time::Instant::now() < next_refresh {
                            continue;
                        }
                        let Some(refreshed) = until_cancelled(cancel, self.refresh(&mut def)).await else {
                            let _ = (&mut live.served).await;
                            return;
                        };
                        match refreshed {
                            Ok(()) => break true,
                            Err(e) => {
                                warn!(id = %def.id, error = %e, "tunnel ticket refresh failed, keeping the connection");
                                next_refresh = tokio::time::Instant::now() + MAX_BACKOFF;
                            }
                        }
                    }
                }
            };
            if !refreshed {
                hellos.stop();
                current = None;
                connected.store(false, Ordering::Relaxed);
                info!(id = %def.id, "tunnel disconnected");
                if !backoff.wait(cancel).await {
                    return;
                }
            }
        }
    }

    /// A fresh ticket, saved with the tunnel (unless it was removed meanwhile).
    async fn refresh(&self, def: &mut TunnelDef) -> Result<()> {
        def.ticket = self.backend.ticket(&def.signing_key()?).await?;
        let url_changed = {
            let mut entries = self.entries();
            let Some(entry) = entries.get_mut(&def.id) else {
                return Ok(());
            };
            save_def(&self.def_path(&def.id), def)?;
            let changed = entry.def.url() != def.url();
            entry.def = def.clone();
            changed
        };
        if url_changed {
            self.notify(&def.chat_id);
        }
        Ok(())
    }

    /// Tell the relay we're back, which also replays its queue. Retried until
    /// `session` (the connection) ends, but a request is never cut: the relay
    /// would still register it, maybe after the removal's purge.
    async fn hello(self: Arc<Self>, mut def: TunnelDef, session: CancellationToken) {
        let mut backoff = Backoff::default();
        let mut refreshed = false;
        while !session.is_cancelled() {
            let body = serde_json::json!({ "no_queue": def.no_queue });
            let call = self.backend.relay_call(
                &def.ticket,
                reqwest::Method::POST,
                "/queue/hello",
                Some(body),
            );
            match call.await {
                Ok(()) => return,
                // The relay cuts the connection too; the reconnect ends the tunnel.
                Err(e) if e.is::<Blocked>() => return,
                Err(e) if e.is::<StaleTicket>() && !refreshed => {
                    refreshed = true;
                    if until_cancelled(&session, self.refresh(&mut def))
                        .await
                        .is_none()
                    {
                        return;
                    }
                    continue;
                }
                Err(e) => warn!(id = %def.id, error = %e, "tunnel hello failed, retrying"),
            }
            if !backoff.wait(&session).await {
                return;
            }
        }
    }

    /// Tell the chat, then remove the tunnel (no purge). It stays listed and
    /// on disk until the chat took the notice, like a watch's end, so a
    /// restart before that only repeats it.
    async fn end_blocked(&self, def: &TunnelDef, cancel: &CancellationToken) {
        let mut backoff = Backoff::default();
        loop {
            let notice = Notice {
                chat_id: def.chat_id.clone(),
                user_id: def.user_id,
                sender: format!("tunnel {}", def.url()),
                label: format!("[zucchini tunnel {}]", def.url()),
                text: format!(
                    "The tunnel {} (port {}) was blocked by the relay and is removed.",
                    def.url(),
                    def.port
                ),
                cancelled: cancel.clone(),
            };
            let delivered = tokio::select! {
                () = cancel.cancelled() => return,
                r = (self.deliver)(notice) => r,
            };
            match delivered {
                Ok(()) => break,
                Err(e) => {
                    warn!(id = %def.id, error = %e, "tunnel notice not delivered, will retry")
                }
            }
            if !backoff.wait(cancel).await {
                return;
            }
        }
        if self.entries().remove(&def.id).is_some() {
            let _ = std::fs::remove_file(self.def_path(&def.id));
            self.notify(&def.chat_id);
            info!(id = %def.id, "blocked tunnel removed");
        }
    }

    /// `after`: the removed tunnel's task, whose last hello must land first.
    fn purge_in_background(
        self: &Arc<Self>,
        def: TunnelDef,
        after: Option<tokio::task::JoinHandle<()>>,
    ) {
        let inner = self.clone();
        let task = tokio::spawn(async move { inner.purge(def, after).await });
        let mut tasks = self.purge_tasks.lock().expect("tunnel purges mutex");
        tasks.retain(|t| !t.is_finished());
        tasks.push(task);
    }

    /// `DELETE /queue` until it lands; the file marked removed keeps it across
    /// restarts. Past the relay's own forget there is nothing left to purge.
    async fn purge(&self, mut def: TunnelDef, after: Option<tokio::task::JoinHandle<()>>) {
        let mut backoff = Backoff::default();
        let cancel = self.purges.clone();
        if let Some(task) = after {
            if until_cancelled(&cancel, task).await.is_none() {
                return;
            }
        }
        let mut fresh_ticket = false;
        loop {
            if unix_now() - def.removed_at.unwrap_or(0) > PURGE_GIVE_UP_SECS {
                break;
            }
            let attempt = async {
                if def.ticket.exp <= unix_now() + 60 || fresh_ticket {
                    fresh_ticket = false;
                    def.ticket = self.backend.ticket(&def.signing_key()?).await?;
                }
                self.backend
                    .relay_call(&def.ticket, reqwest::Method::DELETE, "/queue", None)
                    .await
            };
            let result = tokio::select! {
                () = cancel.cancelled() => return,
                r = attempt => r,
            };
            match result {
                // A blocked id's queue is the relay's to drop.
                Ok(()) => break,
                Err(e) if e.is::<Blocked>() => break,
                Err(e) => {
                    warn!(id = %def.id, error = %e, "tunnel queue purge failed, retrying");
                    fresh_ticket = e.is::<StaleTicket>();
                }
            }
            if !backoff.wait(&cancel).await {
                return;
            }
        }
        // An unlink a crash undoes only repeats the purge.
        match std::fs::remove_file(self.def_path(&def.id)) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
                error!(id = %def.id, error = %e, "removed tunnel's file stays, purging again at the next start")
            }
            _ => info!(id = %def.id, "tunnel queue purged"),
        }
    }
}

fn check_caps(entries: &HashMap<String, Entry>, chat_id: &str) -> Result<()> {
    if entries.len() >= MAX_TUNNELS {
        bail!("this machine already has {MAX_TUNNELS} tunnels, the most it takes");
    }
    if entries
        .values()
        .filter(|e| e.def.chat_id == chat_id)
        .count()
        >= MAX_PER_CHAT
    {
        bail!("this chat already has {MAX_PER_CHAT} tunnels, the most it takes; delete one first");
    }
    Ok(())
}

/// `None` = cancelled first.
async fn until_cancelled<T>(
    cancel: &CancellationToken,
    fut: impl std::future::Future<Output = T>,
) -> Option<T> {
    tokio::select! {
        () = cancel.cancelled() => None,
        out = fut => Some(out),
    }
}

/// A tunnel's hellos: the current connection's, and stopped ones whose
/// request may still be in flight.
#[derive(Default)]
struct Hellos {
    current: Option<CancellationToken>,
    tasks: tokio::task::JoinSet<()>,
}

impl Hellos {
    fn start(
        &mut self,
        stop: CancellationToken,
        hello: impl std::future::Future<Output = ()> + Send + 'static,
    ) {
        self.stop();
        while self.tasks.try_join_next().is_some() {}
        self.tasks.spawn(hello);
        self.current = Some(stop);
    }

    fn stop(&mut self) {
        if let Some(stop) = self.current.take() {
            stop.cancel();
        }
    }

    async fn join(mut self) {
        self.stop();
        while self.tasks.join_next().await.is_some() {}
    }
}

#[derive(Default)]
struct Backoff(Option<Duration>);

impl Backoff {
    /// `false` = cancelled.
    async fn wait(&mut self, cancel: &CancellationToken) -> bool {
        let delay = self
            .0
            .map_or(Duration::from_secs(1), |d| (d * 2).min(MAX_BACKOFF));
        self.0 = Some(delay);
        tokio::select! {
            () = cancel.cancelled() => false,
            () = tokio::time::sleep(delay) => true,
        }
    }
}

type WsStream =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

/// The relay answers `410 Gone` for a kill-listed id: retrying is pointless.
#[derive(Debug)]
struct Blocked;

impl std::fmt::Display for Blocked {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("blocked by the relay")
    }
}

impl std::error::Error for Blocked {}

/// The relay refused the ticket (401, 403): a fresh one may do.
#[derive(Debug)]
struct StaleTicket;

impl std::fmt::Display for StaleTicket {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("the relay refused the ticket")
    }
}

impl std::error::Error for StaleTicket {}

enum ConnectError {
    /// The relay refused the ticket (401, 403).
    Refused(u16),
    Blocked,
    Other(anyhow::Error),
}

/// piko's upstream WebSocket for the tunnel's endpoint.
async fn connect(def: &TunnelDef) -> Result<WsStream, ConnectError> {
    let url = format!(
        "{}/piko/v1/upstream/{}",
        def.ticket.relay.trim_end_matches('/'),
        def.id
    );
    let mut req = url
        .into_client_request()
        .map_err(|e| ConnectError::Other(e.into()))?;
    let bearer = HeaderValue::from_str(&format!("Bearer {}", def.ticket.ticket))
        .map_err(|e| ConnectError::Other(e.into()))?;
    req.headers_mut().insert(header::AUTHORIZATION, bearer);
    match tokio::time::timeout(CONNECT_TIMEOUT, tokio_tungstenite::connect_async(req)).await {
        Err(_) => Err(ConnectError::Other(anyhow!("timed out"))),
        Ok(Ok((ws, _))) => Ok(ws),
        Ok(Err(tungstenite::Error::Http(resp))) => match resp.status().as_u16() {
            410 => Err(ConnectError::Blocked),
            status @ (401 | 403) => Err(ConnectError::Refused(status)),
            status => Err(ConnectError::Other(anyhow!("relay answered {status}"))),
        },
        Ok(Err(e)) => Err(ConnectError::Other(e.into())),
    }
}

/// One upstream connection to the relay.
struct Connection {
    /// Cancelled = a newer connection took over: close once idle.
    retire: CancellationToken,
    served: tokio::task::JoinHandle<()>,
}

impl Connection {
    fn start(ws: WsStream, def: &TunnelDef, cancel: &CancellationToken) -> Self {
        let retire = CancellationToken::new();
        let local = Arc::new(Local::of(def));
        let served = tokio::spawn(serve(
            ws,
            local,
            cancel.clone(),
            retire.clone(),
            SWITCH_GRACE,
        ));
        Self { retire, served }
    }
}

/// Serve the yamux streams piko opens, one request each, until the
/// connection drops or `session` is cancelled. Once `retire` is cancelled
/// the connection closes as soon as no request or WebSocket is open on it
/// (piko then sends everything to the newer one), or after `grace` at the latest.
async fn serve(
    ws: WsStream,
    local: Arc<Local>,
    session: CancellationToken,
    retire: CancellationToken,
    grace: Duration,
) {
    let mut conn =
        yamux::Connection::new(WsIo::new(ws), yamux::Config::default(), yamux::Mode::Client);
    let in_flight = Arc::new(InFlight::default());
    let _close_local = in_flight.closed.clone().drop_guard();
    let mut retire_by: Option<tokio::time::Instant> = None;
    loop {
        if retire_by.is_some() && in_flight.count.load(Ordering::SeqCst) == 0 {
            break;
        }
        let next = tokio::select! {
            () = session.cancelled() => break,
            () = retire.cancelled(), if retire_by.is_none() => {
                retire_by = Some(tokio::time::Instant::now() + grace);
                continue;
            }
            () = in_flight.finished.notified() => continue,
            () = sleep_until(retire_by) => break,
            next = futures::future::poll_fn(|cx| conn.poll_next_inbound(cx)) => next,
        };
        let Some(Ok(stream)) = next else {
            return;
        };
        let (local, requests) = (local.clone(), in_flight.clone());
        let busy = in_flight.enter();
        in_flight.spawn(async move {
            let service =
                hyper::service::service_fn(move |req| proxy(req, local.clone(), requests.clone()));
            let served = hyper::server::conn::http1::Builder::new()
                .serve_connection(TokioIo::new(stream.compat()), service)
                .with_upgrades()
                .await;
            if let Err(e) = served {
                tracing::debug!(error = %e, "tunnel stream ended");
            }
            drop(busy);
        });
    }
    let close = futures::future::poll_fn(|cx| conn.poll_close(cx));
    let _ = tokio::time::timeout(Duration::from_secs(2), close).await;
}

/// Requests and upgraded connections open on one relay connection.
#[derive(Default)]
struct InFlight {
    count: std::sync::atomic::AtomicUsize,
    finished: tokio::sync::Notify,
    /// Cancelled when the relay connection is closed: its tasks drop their
    /// local sockets, which a closed yamux stream alone doesn't make them do.
    closed: CancellationToken,
}

impl InFlight {
    fn enter(self: &Arc<Self>) -> Busy {
        self.count.fetch_add(1, Ordering::SeqCst);
        Busy(self.clone())
    }

    fn spawn(&self, task: impl std::future::Future<Output = ()> + Send + 'static) {
        let closed = self.closed.clone();
        tokio::spawn(async move { until_cancelled(&closed, task).await });
    }
}

struct Busy(Arc<InFlight>);

impl Drop for Busy {
    fn drop(&mut self) {
        self.0.count.fetch_sub(1, Ordering::SeqCst);
        self.0.finished.notify_one();
    }
}

async fn sleep_until(at: Option<tokio::time::Instant>) {
    match at {
        Some(at) => tokio::time::sleep_until(at).await,
        None => std::future::pending().await,
    }
}

/// What a request needs from its tunnel.
struct Local {
    port: u16,
    auth: Option<Auth>,
    keep_host: bool,
    origin: String,
}

impl Local {
    fn of(def: &TunnelDef) -> Self {
        Self {
            port: def.port,
            auth: def.auth.clone(),
            keep_host: def.keep_host,
            origin: def.url().to_string(),
        }
    }
}

type Body = BoxBody<Bytes, hyper::Error>;

fn text_response(status: StatusCode, text: &str) -> Response<Body> {
    let body = Full::new(Bytes::from(format!("{text}\n")))
        .map_err(|never| match never {})
        .boxed();
    let mut resp = Response::new(body);
    *resp.status_mut() = status;
    resp.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/plain; charset=utf-8"),
    );
    resp
}

/// One request from the relay to `localhost:<port>`. Never takes the target
/// from the request.
async fn proxy(
    mut req: Request<Incoming>,
    local: Arc<Local>,
    in_flight: Arc<InFlight>,
) -> Result<Response<Body>, Infallible> {
    let piko: Vec<_> = req
        .headers()
        .keys()
        .filter(|k| k.as_str().starts_with("x-piko-"))
        .cloned()
        .collect();
    for name in piko {
        req.headers_mut().remove(name);
    }
    let set_cookie = match authorize(&mut req, &local) {
        Authorized::No => return Ok(unauthorized(local.auth.as_ref())),
        Authorized::CrossOrigin => {
            return Ok(text_response(
                StatusCode::FORBIDDEN,
                "cross-origin request authenticated only by credentials the browser keeps",
            ))
        }
        Authorized::Yes { set_cookie } => set_cookie,
        Authorized::Login { set_cookie } => {
            return Ok(login_redirect(&local.origin, req.uri(), set_cookie))
        }
    };
    if !local.keep_host {
        let host = HeaderValue::from_str(&format!("localhost:{}", local.port)).expect("ascii");
        if let Some(original) = req.headers_mut().insert(header::HOST, host) {
            req.headers_mut().insert("x-forwarded-host", original);
        }
    }
    let inbound_upgrade = req
        .headers()
        .contains_key(header::UPGRADE)
        .then(|| hyper::upgrade::on(&mut req));

    let Some(tcp) = connect_local(local.port).await else {
        return Ok(upstream_down(&format!(
            "nothing answers on the tunnel's local port {}",
            local.port
        )));
    };
    let sent = async {
        let (mut sender, conn) = hyper::client::conn::http1::handshake(TokioIo::new(tcp)).await?;
        in_flight.spawn(async move {
            let _ = conn.with_upgrades().await;
        });
        sender.send_request(req).await
    };
    let mut resp = match sent.await {
        Ok(resp) => resp,
        Err(e) => return Ok(upstream_down(&format!("local server error: {e}"))),
    };
    if resp.status() == StatusCode::SWITCHING_PROTOCOLS {
        if let Some(inbound) = inbound_upgrade {
            let outbound = hyper::upgrade::on(&mut resp);
            let busy = in_flight.enter();
            in_flight.spawn(async move {
                let _busy = busy;
                if let (Ok(a), Ok(b)) = tokio::join!(inbound, outbound) {
                    let _ =
                        tokio::io::copy_bidirectional(&mut TokioIo::new(a), &mut TokioIo::new(b))
                            .await;
                }
            });
        }
    }
    if let Some(cookie) = set_cookie {
        resp.headers_mut().append(header::SET_COOKIE, cookie);
    }
    Ok(resp.map(BodyExt::boxed))
}

/// A 502 the relay tells from the app's own: the request got no answer from
/// the local server (refused, reset, closed early), so the relay keeps it
/// queued for replay. The app may have seen part of it - at-least-once.
fn upstream_down(text: &str) -> Response<Body> {
    let mut resp = text_response(StatusCode::BAD_GATEWAY, text);
    resp.headers_mut()
        .insert("x-zucchini-tunnel", HeaderValue::from_static(UPSTREAM_DOWN));
    resp
}

/// `127.0.0.1`, else `::1` (dev servers that bind `localhost` may take either).
async fn connect_local(port: u16) -> Option<tokio::net::TcpStream> {
    for ip in ["127.0.0.1", "::1"] {
        let attempt = tokio::net::TcpStream::connect((ip, port));
        if let Ok(Ok(tcp)) = tokio::time::timeout(LOCAL_CONNECT_TIMEOUT, attempt).await {
            return Some(tcp);
        }
    }
    None
}

enum Authorized {
    No,
    /// `set_cookie`: the credential came in the query or Basic auth, so the
    /// browser keeps a cookie for the page's other requests.
    Yes {
        set_cookie: Option<HeaderValue>,
    },
    /// A credential the browser attaches by itself (the `zt` cookie or cached
    /// Basic auth) vouched for a mutation or upgrade from another origin.
    /// `zuc.run` is not on the Public Suffix List, so every tunnel is same-site
    /// to every other and SameSite=Lax doesn't keep a sibling tunnel's page
    /// from riding the cookie; Basic auth has no SameSite at all.
    CrossOrigin,
    /// A browser opened a `?zt=` link: send it to the same URL without the
    /// token, so it leaves the address bar, history and `Referer`s.
    Login {
        set_cookie: Option<HeaderValue>,
    },
}

/// Check `--auth` and strip our credentials, so the local server never sees them.
fn authorize(req: &mut Request<Incoming>, local: &Local) -> Authorized {
    let mut uri = std::mem::take(req.uri_mut());
    let method = req.method().clone();
    let (auth, origin) = (local.auth.as_ref(), local.origin.as_str());
    let authorized = authorize_parts(req.headers_mut(), &mut uri, &method, auth, origin);
    *req.uri_mut() = uri;
    authorized
}

fn authorize_parts(
    headers: &mut hyper::HeaderMap,
    uri: &mut hyper::Uri,
    method: &hyper::Method,
    auth: Option<&Auth>,
    origin: &str,
) -> Authorized {
    let Some(auth) = auth else {
        return Authorized::Yes { set_cookie: None };
    };
    let authorization = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    match auth {
        Auth::Basic { user, password } => {
            let pair = format!("{user}:{password}");
            let expected = format!("Basic {}", STANDARD.encode(&pair));
            // Safari never sends Basic credentials on a WebSocket handshake
            // (WebKit bug 80362), only cookies.
            let cookie_value = base32_lower(&Sha256::digest(format!("zucchini-basic|{pair}")));
            let by_header = ct_eq(&authorization, &expected);
            let by_cookie = strip_cookie_token(headers)
                .iter()
                .any(|t| ct_eq(t, &cookie_value));
            if !(by_header || by_cookie) {
                return Authorized::No;
            }
            if cross_origin_write(headers, method, origin) {
                return Authorized::CrossOrigin;
            }
            if by_header {
                headers.remove(header::AUTHORIZATION);
            }
            let set_cookie = (by_header && !by_cookie).then(|| token_cookie(&cookie_value));
            Authorized::Yes { set_cookie }
        }
        Auth::Token { token } => {
            let by_header = authorization
                .strip_prefix("Bearer ")
                .is_some_and(|t| ct_eq(t, token));
            if by_header {
                headers.remove(header::AUTHORIZATION);
            }
            let by_query = strip_query_token(uri).iter().any(|t| ct_eq(t, token));
            let by_cookie = strip_cookie_token(headers).iter().any(|t| ct_eq(t, token));
            if !(by_header || by_query || by_cookie) {
                return Authorized::No;
            }
            if !(by_header || by_query) && cross_origin_write(headers, method, origin) {
                return Authorized::CrossOrigin;
            }
            let set_cookie = (by_query && !by_cookie).then(|| token_cookie(token));
            if by_query && !by_header && top_level_navigation(headers, method) {
                return Authorized::Login { set_cookie };
            }
            Authorized::Yes { set_cookie }
        }
    }
}

fn token_cookie(value: &str) -> HeaderValue {
    HeaderValue::from_str(&format!(
        "{TOKEN_PARAM}={value}; Path=/; HttpOnly; Secure; SameSite=Lax"
    ))
    .expect("base32 token")
}

/// A non-GET/HEAD/OPTIONS request or a WebSocket upgrade that a browser sent
/// from a page of another origin. Cross-origin GETs pass: CORS keeps their
/// responses from the page, and links to the tunnel are top-level navigations.
/// WebSockets are exempt from CORS, so their upgrade (a GET) is checked.
fn cross_origin_write(headers: &hyper::HeaderMap, method: &hyper::Method, origin: &str) -> bool {
    let safe = matches!(
        *method,
        hyper::Method::GET | hyper::Method::HEAD | hyper::Method::OPTIONS
    );
    if safe && !headers.contains_key(header::UPGRADE) {
        return false;
    }
    let header = |name: &str| headers.get(name).and_then(|v| v.to_str().ok());
    let foreign_origin =
        header("origin").is_some_and(|o| !o.trim_end_matches('/').eq_ignore_ascii_case(origin));
    let foreign_site = matches!(header("sec-fetch-site"), Some("same-site" | "cross-site"));
    foreign_origin || foreign_site
}

/// Only a browser's own page load: API clients and webhooks (no
/// `Sec-Fetch-*`) may not follow a redirect, and a cross-site iframe would
/// drop the SameSite=Lax cookie and land on a 401.
fn top_level_navigation(headers: &hyper::HeaderMap, method: &hyper::Method) -> bool {
    let header = |name: &str| headers.get(name).and_then(|v| v.to_str().ok());
    matches!(*method, hyper::Method::GET | hyper::Method::HEAD)
        && header("sec-fetch-mode") == Some("navigate")
        && header("sec-fetch-dest") == Some("document")
}

/// `uri` is the request's, already without the token.
fn login_redirect(
    origin: &str,
    uri: &hyper::Uri,
    set_cookie: Option<HeaderValue>,
) -> Response<Body> {
    let mut resp = text_response(StatusCode::SEE_OTHER, "signed in");
    // Absolute, so a path like `//evil.example/` can't be read as another host.
    let path = uri.path_and_query().map_or("/", |pq| pq.as_str());
    let headers = resp.headers_mut();
    headers.insert(
        header::LOCATION,
        HeaderValue::from_str(&format!("{origin}{path}")).expect("from a uri"),
    );
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    if let Some(cookie) = set_cookie {
        headers.insert(header::SET_COOKIE, cookie);
    }
    resp
}

fn unauthorized(auth: Option<&Auth>) -> Response<Body> {
    let mut resp = text_response(StatusCode::UNAUTHORIZED, "this tunnel needs credentials");
    if matches!(auth, Some(Auth::Basic { .. })) {
        resp.headers_mut().insert(
            header::WWW_AUTHENTICATE,
            HeaderValue::from_static("Basic realm=\"zucchini tunnel\""),
        );
    }
    resp
}

/// Remove every `zt=` query parameter; their values.
fn strip_query_token(uri: &mut hyper::Uri) -> Vec<String> {
    let Some(query) = uri.query() else {
        return Vec::new();
    };
    let mut tokens = Vec::new();
    let kept: Vec<&str> = query
        .split('&')
        .filter(|pair| match pair.split_once('=') {
            Some((TOKEN_PARAM, value)) => {
                tokens.push(value.to_string());
                false
            }
            _ => *pair != TOKEN_PARAM,
        })
        .collect();
    if tokens.is_empty() {
        return tokens;
    }
    let path = uri.path();
    let pq = if kept.is_empty() {
        path.to_string()
    } else {
        format!("{path}?{}", kept.join("&"))
    };
    let mut parts = std::mem::take(uri).into_parts();
    parts.path_and_query = pq.parse().ok();
    *uri = hyper::Uri::from_parts(parts).unwrap_or_default();
    tokens
}

/// Remove every `zt` cookie from every `Cookie` header; their values.
fn strip_cookie_token(headers: &mut hyper::HeaderMap) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut kept = Vec::new();
    for value in headers.get_all(header::COOKIE) {
        let Ok(value) = value.to_str() else {
            kept.push(value.clone());
            continue;
        };
        let rest: Vec<&str> = value
            .split(';')
            .map(str::trim)
            .filter(|c| match c.split_once('=') {
                Some((TOKEN_PARAM, v)) => {
                    tokens.push(v.to_string());
                    false
                }
                _ => !c.is_empty(),
            })
            .collect();
        if !rest.is_empty() {
            kept.push(HeaderValue::from_str(&rest.join("; ")).expect("from a header value"));
        }
    }
    if tokens.is_empty() {
        return tokens;
    }
    headers.remove(header::COOKIE);
    for value in kept {
        headers.append(header::COOKIE, value);
    }
    tokens
}

fn ct_eq(a: &str, b: &str) -> bool {
    a.len() == b.len()
        && a.bytes()
            .zip(b.bytes())
            .fold(0, |acc, (x, y)| acc | (x ^ y))
            == 0
}

/// A WebSocket as a byte stream, one binary message per write - what piko's
/// Go side (`pkg/websocket.Conn`) speaks.
struct WsIo {
    ws: WsStream,
    buf: Bytes,
}

impl WsIo {
    fn new(ws: WsStream) -> Self {
        Self {
            ws,
            buf: Bytes::new(),
        }
    }
}

impl AsyncRead for WsIo {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut TaskContext<'_>,
        out: &mut [u8],
    ) -> Poll<std::io::Result<usize>> {
        loop {
            if !self.buf.is_empty() {
                let n = out.len().min(self.buf.len());
                out[..n].copy_from_slice(&self.buf.split_to(n));
                return Poll::Ready(Ok(n));
            }
            match ready!(Pin::new(&mut self.ws).poll_next(cx)) {
                Some(Ok(Message::Binary(data))) => self.buf = data,
                Some(Ok(Message::Close(_))) | None => return Poll::Ready(Ok(0)),
                Some(Ok(_)) => {}
                Some(Err(e)) => return Poll::Ready(Err(std::io::Error::other(e))),
            }
        }
    }
}

impl AsyncWrite for WsIo {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut TaskContext<'_>,
        data: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        ready!(Pin::new(&mut self.ws).poll_ready(cx)).map_err(std::io::Error::other)?;
        Pin::new(&mut self.ws)
            .start_send(Message::Binary(Bytes::copy_from_slice(data)))
            .map_err(std::io::Error::other)?;
        Poll::Ready(Ok(data.len()))
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.ws)
            .poll_flush(cx)
            .map_err(std::io::Error::other)
    }

    fn poll_close(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.ws)
            .poll_close(cx)
            .map_err(std::io::Error::other)
    }
}

fn json_files(dir: &Path) -> Vec<PathBuf> {
    std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect()
}

fn load_def(path: &Path) -> Result<TunnelDef> {
    Ok(serde_json::from_slice(&std::fs::read(path)?)?)
}

fn save_def(path: &Path, def: &TunnelDef) -> Result<()> {
    if let Some(dir) = path.parent() {
        crate::atomic::create_dir_durably(dir)?;
    }
    let json = serde_json::to_vec_pretty(def)?;
    crate::atomic::atomic_write_private(path, &json)
        .and_then(|()| crate::atomic::sync_parent(path))
        .with_context(|| format!("write tunnel {}", path.display()))
}

fn unix_now() -> i64 {
    chrono::Utc::now().timestamp()
}

#[cfg(test)]
impl TunnelDef {
    pub fn for_test(chat_id: &str, user_id: Uuid) -> Self {
        let id = random_base32(16)[..26].to_string();
        Self {
            ticket: Ticket {
                ticket: String::new(),
                exp: unix_now() + 7 * 86400,
                relay: "ws://127.0.0.1:9".into(),
                url: format!("https://{id}.zuc.run"),
                fetched_at: unix_now(),
            },
            id,
            chat_id: chat_id.to_string(),
            user_id,
            port: 8080,
            auth: None,
            keep_host: false,
            no_queue: false,
            created_at: 1_800_000_000,
            key: STANDARD.encode([7u8; 32]),
            removed_at: None,
        }
    }
}

/// Register a tunnel that connects nowhere, for the main loop's tests.
#[cfg(test)]
pub fn register_idle(def: TunnelDef) {
    let tunnels = GLOBAL.get_or_init(|| {
        let backend = Backend {
            http: reqwest::Client::new(),
            api_base_url: "http://127.0.0.1:9".into(),
            fetch_token: Arc::new(Box::new(|| Box::pin(async { bail!("no backend") }))),
        };
        let root = std::env::temp_dir().join(format!("zucchini-tunnel-none-{}", Uuid::now_v7()));
        let tunnels = Tunnels::start(
            root,
            backend,
            Box::new(|_| Box::pin(async { bail!("no chat takes it") })),
            tokio::sync::mpsc::unbounded_channel().0,
            Timing::default(),
            None,
        );
        tunnels.admit_restored(|_| true);
        tunnels
    });
    let entry = Entry {
        def,
        connected: Arc::default(),
        cancel: CancellationToken::new(),
        task: tokio::spawn(async {}),
    };
    tunnels.0.entries().insert(entry.def.id.clone(), entry);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_match_the_backends_derivation() {
        assert_eq!(base32_lower(b"foobar"), "mzxw6ytboi");
        assert_eq!(base32_lower(b"f"), "my");
        // `tunnel-relay/cmd/devticket -pubkey` and the backend agree on this one.
        let pubkey = STANDARD
            .decode("3oButUjqjfYwFAGg6i7/BRR4CqU4reHAIhBQ4/zu51w=")
            .unwrap();
        assert_eq!(tunnel_id(&pubkey), "bnw26ze2znzx4jjlz52gsy5y7u");
    }

    fn check(
        auth: &Auth,
        headers: &[(&str, &str)],
        uri: &str,
    ) -> (bool, hyper::HeaderMap, hyper::Uri, bool) {
        let (authorized, map, uri) = check_method(auth, "GET", headers, uri);
        let (ok, cookie) = match authorized {
            Authorized::Yes { set_cookie } => (true, set_cookie.is_some()),
            _ => (false, false),
        };
        (ok, map, uri, cookie)
    }

    fn check_method(
        auth: &Auth,
        method: &str,
        headers: &[(&str, &str)],
        uri: &str,
    ) -> (Authorized, hyper::HeaderMap, hyper::Uri) {
        let mut map = hyper::HeaderMap::new();
        for (k, v) in headers {
            map.append(
                hyper::header::HeaderName::from_bytes(k.as_bytes()).unwrap(),
                HeaderValue::from_str(v).unwrap(),
            );
        }
        let mut uri: hyper::Uri = uri.parse().unwrap();
        let method = method.parse().unwrap();
        let origin = "https://victim.zuc.run";
        let authorized = authorize_parts(&mut map, &mut uri, &method, Some(auth), origin);
        (authorized, map, uri)
    }

    #[test]
    fn browser_kept_credentials_do_not_vouch_for_cross_origin_writes_or_upgrades() {
        let auth = Auth::Token {
            token: "secret".into(),
        };
        let ok = |method, headers: &[(&str, &str)], uri| {
            let (authorized, _, _) = check_method(&auth, method, headers, uri);
            match authorized {
                Authorized::Yes { .. } | Authorized::Login { .. } => true,
                Authorized::CrossOrigin => false,
                Authorized::No => panic!("bad credentials"),
            }
        };
        let cookie = ("cookie", "zt=secret");
        let sibling = ("origin", "https://evil.zuc.run");
        let same_site = ("sec-fetch-site", "same-site");
        let ws = [("upgrade", "websocket"), ("connection", "Upgrade")];

        assert!(!ok("POST", &[cookie, sibling, same_site], "/a"));
        assert!(!ok("DELETE", &[cookie, same_site], "/a"));
        assert!(!ok("POST", &[cookie, ("origin", "null")], "/a"));
        let cross_ws = [
            cookie,
            ("origin", "https://evil.com"),
            ("sec-fetch-site", "cross-site"),
        ];
        assert!(!ok("GET", &[&cross_ws[..], &ws].concat(), "/ws"));

        let own = ("origin", "https://victim.zuc.run");
        let same_origin = ("sec-fetch-site", "same-origin");
        assert!(ok("POST", &[cookie, own, same_origin], "/a"));
        assert!(ok(
            "GET",
            &[&[cookie, own, same_origin][..], &ws].concat(),
            "/ws"
        ));
        assert!(ok("POST", &[cookie], "/a"));
        // Links from anywhere and subresource GETs keep working.
        let nav = [
            cookie,
            ("sec-fetch-site", "cross-site"),
            ("sec-fetch-mode", "navigate"),
        ];
        assert!(ok("GET", &nav, "/a"));
        assert!(ok("GET", &[cookie, sibling, same_site], "/a"));

        // An explicit credential is no ambient one.
        let bearer = ("authorization", "Bearer secret");
        assert!(ok("POST", &[bearer, cookie, sibling, same_site], "/a"));
        assert!(ok("POST", &[cookie, sibling, same_site], "/a?zt=secret"));
        let basic = Auth::Basic {
            user: "u".into(),
            password: "p".into(),
        };
        let good = format!("Basic {}", STANDARD.encode("u:p"));
        let pair = ("authorization", good.as_str());
        let basic_ok = |method, headers: &[(&str, &str)]| {
            let (authorized, _, _) = check_method(&basic, method, headers, "/a");
            match authorized {
                Authorized::Yes { .. } => true,
                Authorized::CrossOrigin => false,
                _ => panic!("bad credentials"),
            }
        };
        assert!(!basic_ok("POST", &[pair, sibling, same_site]));
        assert!(!basic_ok("PUT", &[pair, ("sec-fetch-site", "cross-site")]));
        assert!(!basic_ok("GET", &[&[pair, sibling][..], &ws].concat()));
        assert!(basic_ok("POST", &[pair, own, same_origin]));
        assert!(basic_ok("POST", &[pair]));
        assert!(basic_ok("GET", &[pair, sibling, same_site]));
    }

    #[test]
    fn a_token_is_accepted_from_header_query_or_cookie_and_never_forwarded() {
        let auth = Auth::Token {
            token: "secret".into(),
        };
        let (ok, headers, _, cookie) = check(&auth, &[("authorization", "Bearer secret")], "/a");
        assert!(ok && !cookie);
        assert!(headers.get("authorization").is_none());

        let (ok, _, uri, cookie) = check(&auth, &[], "/a?x=1&zt=secret&y=2");
        assert!(ok && cookie);
        assert_eq!(uri.to_string(), "/a?x=1&y=2");

        let (ok, headers, _, cookie) = check(&auth, &[("cookie", "a=1; zt=secret; b=2")], "/a");
        assert!(ok && !cookie);
        assert_eq!(headers.get("cookie").unwrap(), "a=1; b=2");

        let (ok, headers, uri, _) = check(
            &auth,
            &[("authorization", "Bearer wrong"), ("cookie", "zt=wrong")],
            "/a?zt=wrong",
        );
        assert!(!ok);
        assert!(headers.get("cookie").is_none() && uri.query().is_none());
        assert!(!check(&auth, &[], "/a").0);

        // Several `zt`s: any match lets it in, none is forwarded.
        let (ok, headers, _, _) = check(
            &auth,
            &[("cookie", "zt=old; a=1"), ("cookie", "zt=secret; zt=other")],
            "/a",
        );
        assert!(ok);
        assert_eq!(
            headers.get_all("cookie").iter().collect::<Vec<_>>(),
            ["a=1"]
        );
        let (ok, _, uri, _) = check(&auth, &[], "/a?zt=old&zt=secret&x=1");
        assert!(ok);
        assert_eq!(uri.to_string(), "/a?x=1");
        assert!(!check(&auth, &[("authorization", "Bearer secretx")], "/a").0);
    }

    #[test]
    fn a_browser_opening_a_token_link_is_sent_to_the_clean_url() {
        let auth = Auth::Token {
            token: "secret".into(),
        };
        let page = [
            ("sec-fetch-mode", "navigate"),
            ("sec-fetch-dest", "document"),
        ];
        let (authorized, _, uri) = check_method(&auth, "GET", &page, "/p?a=1&zt=secret&b=2");
        let Authorized::Login { set_cookie } = authorized else {
            panic!("not redirected");
        };
        let origin = "https://victim.zuc.run";
        let resp = login_redirect(origin, &uri, set_cookie);
        assert_eq!(resp.status(), StatusCode::SEE_OTHER);
        assert_eq!(
            resp.headers()["location"],
            "https://victim.zuc.run/p?a=1&b=2"
        );
        assert!(resp.headers()["set-cookie"]
            .to_str()
            .unwrap()
            .starts_with("zt=secret;"));
        let (authorized, _, uri) = check_method(&auth, "HEAD", &page, "/?zt=secret");
        assert!(matches!(authorized, Authorized::Login { .. }));
        let location =
            |uri: &hyper::Uri| login_redirect(origin, uri, None).headers()["location"].clone();
        assert_eq!(location(&uri), "https://victim.zuc.run/");
        // Never off the tunnel: a scheme-relative path stays a path.
        for path in ["//evil.example/?zt=secret", "/\\evil.example/?zt=secret"] {
            let (authorized, _, uri) = check_method(&auth, "GET", &page, path);
            assert!(matches!(authorized, Authorized::Login { .. }), "{path}");
            assert!(location(&uri)
                .to_str()
                .unwrap()
                .starts_with("https://victim.zuc.run/"));
        }
        let with_cookie = [&page[..], &[("cookie", "zt=secret")]].concat();
        let (authorized, _, _) = check_method(&auth, "GET", &with_cookie, "/?zt=secret");
        assert!(matches!(authorized, Authorized::Login { set_cookie: None }));

        // Everything else is served as before: API clients, webhooks, an
        // iframe, a WebSocket, a form post, a request that also has a Bearer.
        let iframe = [("sec-fetch-mode", "navigate"), ("sec-fetch-dest", "iframe")];
        let ws = [("sec-fetch-mode", "websocket"), ("sec-fetch-dest", "empty")];
        let bearer = [&page[..], &[("authorization", "Bearer secret")]].concat();
        for (method, headers) in [
            ("GET", &[][..]),
            ("GET", &iframe[..]),
            ("GET", &ws[..]),
            ("POST", &page[..]),
            ("GET", &bearer[..]),
        ] {
            let (authorized, _, _) = check_method(&auth, method, headers, "/p?zt=secret");
            assert!(
                matches!(authorized, Authorized::Yes { .. }),
                "{method} {headers:?}"
            );
        }
        let (authorized, _, _) = check_method(&auth, "GET", &with_cookie, "/p");
        assert!(matches!(authorized, Authorized::Yes { .. }));
    }

    #[test]
    fn basic_auth_needs_the_exact_pair() {
        let auth = Auth::Basic {
            user: "zucchini".into(),
            password: "pw".into(),
        };
        let good = format!("Basic {}", STANDARD.encode("zucchini:pw"));
        let (authorized, headers, _) = check_method(&auth, "GET", &[("authorization", &good)], "/");
        let Authorized::Yes {
            set_cookie: Some(cookie),
        } = authorized
        else {
            panic!("no cookie set");
        };
        assert!(headers.get("authorization").is_none());
        let bad = format!("Basic {}", STANDARD.encode("zucchini:px"));
        assert!(!check(&auth, &[("authorization", &bad)], "/").0);

        // The cookie alone lets a WebSocket in: Safari sends no Basic there.
        let cookie = cookie.to_str().unwrap().split(';').next().unwrap();
        let (ok, headers, _, set) = check(&auth, &[("cookie", cookie)], "/");
        assert!(ok && !set);
        assert!(headers.get("cookie").is_none());
        assert!(!check(&auth, &[("cookie", "zt=wrong")], "/").0);
        assert!(unauthorized(Some(&auth))
            .headers()
            .contains_key(header::WWW_AUTHENTICATE));
    }

    #[test]
    fn generated_credentials_are_long_and_random() {
        let (Ok(Auth::Token { token: a }), Ok(Auth::Token { token: b })) =
            (Auth::generate("token"), Auth::generate("token"))
        else {
            panic!("token auth");
        };
        assert_eq!(a.len(), 52);
        assert_ne!(a, b);
        assert!(Auth::generate("bearer").is_err());
    }

    #[test]
    fn the_queue_api_is_on_the_relay_host() {
        let ticket = |relay: &str| Ticket {
            ticket: String::new(),
            exp: 0,
            relay: relay.into(),
            url: String::new(),
            fetched_at: 0,
        };
        assert_eq!(
            ticket("wss://relay.zuc.run").relay_http(),
            "https://relay.zuc.run"
        );
        assert_eq!(
            ticket("ws://relay.localhost:8180/").relay_http(),
            "http://relay.localhost:8180"
        );
    }

    /// A backend on a local port answering every request with `status` and
    /// `body`; the request bodies it got.
    async fn fake_backend(
        status: u16,
        body: serde_json::Value,
    ) -> (Backend, Arc<Mutex<Vec<serde_json::Value>>>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let got: Arc<Mutex<Vec<serde_json::Value>>> = Arc::default();
        let seen = got.clone();
        tokio::spawn(async move {
            while let Ok((tcp, _)) = listener.accept().await {
                let (seen, body) = (seen.clone(), body.clone());
                let service = hyper::service::service_fn(move |req: Request<Incoming>| {
                    let (seen, body) = (seen.clone(), body.clone());
                    async move {
                        let bytes = req.into_body().collect().await.unwrap().to_bytes();
                        seen.lock()
                            .unwrap()
                            .push(serde_json::from_slice(&bytes).unwrap());
                        let mut resp = Response::new(Full::new(Bytes::from(body.to_string())));
                        *resp.status_mut() = StatusCode::from_u16(status).unwrap();
                        Ok::<_, Infallible>(resp)
                    }
                });
                tokio::spawn(
                    hyper::server::conn::http1::Builder::new()
                        .serve_connection(TokioIo::new(tcp), service),
                );
            }
        });
        let backend = Backend {
            http: reqwest::Client::new(),
            api_base_url: format!("http://{addr}"),
            fetch_token: Arc::new(Box::new(|| Box::pin(async { Ok("jwt".to_string()) }))),
        };
        (backend, got)
    }

    #[tokio::test]
    async fn a_ticket_request_proves_possession_of_the_key() {
        let key = SigningKey::from_bytes(&[3u8; 32]);
        let id = tunnel_id(&key.verifying_key().to_bytes());
        let res = serde_json::json!({
            "ticket": "t", "id": id, "exp": 1, "relay": "wss://relay.zuc.run", "url": format!("https://{id}.zuc.run"),
        });
        let (backend, got) = fake_backend(200, res).await;
        let ticket = backend.ticket(&key).await.unwrap();
        assert_eq!(
            (ticket.relay.as_str(), ticket.url),
            ("wss://relay.zuc.run", format!("https://{id}.zuc.run"))
        );

        let req = got.lock().unwrap()[0].clone();
        let pubkey: [u8; 32] = STANDARD
            .decode(req["pubkey"].as_str().unwrap())
            .unwrap()
            .try_into()
            .unwrap();
        let sig: [u8; 64] = STANDARD
            .decode(req["sig"].as_str().unwrap())
            .unwrap()
            .try_into()
            .unwrap();
        let message = format!("zucchini-tunnel-ticket|{id}|{}", req["ts"]);
        ed25519_dalek::VerifyingKey::from_bytes(&pubkey)
            .unwrap()
            .verify_strict(
                message.as_bytes(),
                &ed25519_dalek::Signature::from_bytes(&sig),
            )
            .unwrap();
        assert!((req["ts"].as_i64().unwrap() - unix_now()).abs() < 5);

        let (old_backend, _) = fake_backend(404, serde_json::json!({})).await;
        let err = old_backend.ticket(&key).await.unwrap_err().to_string();
        assert_eq!(err, "the Zucchini backend doesn't support tunnels yet");
    }

    /// A relay that answers 410 Gone to everything, upstream connect
    /// included; its URL and how many connections it took.
    async fn gone_relay() -> (String, Arc<std::sync::atomic::AtomicUsize>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let relay = format!("ws://{}", listener.local_addr().unwrap());
        let hits = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counted = hits.clone();
        tokio::spawn(async move {
            while let Ok((tcp, _)) = listener.accept().await {
                counted.fetch_add(1, Ordering::Relaxed);
                let service = hyper::service::service_fn(|_req: Request<Incoming>| async {
                    let mut resp = Response::new(Full::new(Bytes::new()));
                    *resp.status_mut() = StatusCode::GONE;
                    Ok::<_, Infallible>(resp)
                });
                tokio::spawn(
                    hyper::server::conn::http1::Builder::new()
                        .serve_connection(TokioIo::new(tcp), service),
                );
            }
        });
        (relay, hits)
    }

    type Notices = Arc<Mutex<Vec<(String, String)>>>;

    fn recorded_notices() -> (Deliver, Notices) {
        let notices: Notices = Arc::default();
        let got = notices.clone();
        let deliver: Deliver = Box::new(move |n| {
            got.lock().unwrap().push((n.sender, n.text));
            Box::pin(async { Ok(()) })
        });
        (deliver, notices)
    }

    fn no_backend() -> Backend {
        Backend {
            http: reqwest::Client::new(),
            api_base_url: "http://127.0.0.1:9".into(),
            fetch_token: Arc::new(Box::new(|| Box::pin(async { bail!("no backend") }))),
        }
    }

    fn fresh_root(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("zucchini-tunnel-{name}-{}", Uuid::now_v7()))
    }

    /// Unowned tunnels at `root`, restored ones admitted.
    fn admitted_tunnels(root: PathBuf, deliver: Deliver) -> Tunnels {
        let t = Tunnels::start(
            root,
            no_backend(),
            deliver,
            tokio::sync::mpsc::unbounded_channel().0,
            Timing::default(),
            None,
        );
        t.admit_restored(|_| true);
        t
    }

    /// Checks `done` up to 100 times, `interval_ms` apart.
    async fn poll_until(interval_ms: u64, mut done: impl FnMut() -> bool) {
        for _ in 0..100 {
            if done() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(interval_ms)).await;
        }
    }

    #[tokio::test]
    async fn a_create_meets_the_caps_and_a_stop_under_the_lock() {
        let t = admitted_tunnels(fresh_root("caps"), recorded_notices().0);
        let fresh = |t: &Tunnels| {
            let stops = t.0.stops("chat");
            t.0.launch(TunnelDef::for_test("chat", Uuid::nil()), Some(stops), false)
        };
        for _ in 0..MAX_PER_CHAT {
            fresh(&t).unwrap();
        }
        assert!(fresh(&t).is_err(), "over the per-chat cap");
        t.0.launch(TunnelDef::for_test("other", Uuid::nil()), Some(0), false)
            .unwrap();

        // A stop while the ticket was being fetched refuses the registration.
        let stops = t.0.stops("late");
        assert!(!t.chat_stopped("late"));
        let refused =
            t.0.launch(TunnelDef::for_test("late", Uuid::nil()), Some(stops), false);
        assert!(refused.is_err());
        assert!(t.running_in("late").is_empty());
        assert!(t
            .0
            .launch(
                TunnelDef::for_test("late", Uuid::nil()),
                Some(t.0.stops("late")),
                false
            )
            .is_ok());
        t.shutdown().await;
    }

    #[tokio::test]
    async fn the_owners_restored_tunnels_connect_before_the_first_checkpoint() {
        let (relay, _) = gone_relay().await;
        let root = fresh_root("restore");
        let owner = Uuid::now_v7();
        let member = Uuid::now_v7();
        for user in [owner, member] {
            let mut def = TunnelDef::for_test(&user.to_string(), user);
            def.ticket.relay = relay.clone();
            save_def(&root.join(format!("{}.json", def.id)), &def).unwrap();
        }
        let (deliver, notices) = recorded_notices();
        let unheard = tokio::sync::mpsc::unbounded_channel().0;
        let t = Tunnels::start(
            root,
            no_backend(),
            deliver,
            unheard,
            Timing::default(),
            Some(owner),
        );
        // The 410 relay ends whatever connects: only the owner's, so far.
        poll_until(50, || !notices.lock().unwrap().is_empty()).await;
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert_eq!(notices.lock().unwrap().len(), 1);
        assert!(t.running_in(&owner.to_string()).is_empty());
        assert_eq!(
            t.running_in(&member.to_string()).len(),
            1,
            "waits for its access check"
        );
        t.admit_restored(|_| true);
        poll_until(50, || notices.lock().unwrap().len() == 2).await;
        assert_eq!(notices.lock().unwrap().len(), 2);
        t.shutdown().await;
    }

    #[tokio::test]
    async fn a_blocked_tunnel_tells_its_chat_and_is_removed_without_retries() {
        let (relay, hits) = gone_relay().await;
        let (deliver, notices) = recorded_notices();
        let t = admitted_tunnels(fresh_root("blocked"), deliver);

        let mut def = TunnelDef::for_test("chat", Uuid::nil());
        def.ticket.relay = relay.clone();
        save_def(&t.0.def_path(&def.id), &def).unwrap();
        t.0.launch(def.clone(), None, false).unwrap();
        poll_until(50, || t.list("chat").is_empty()).await;
        assert!(t.list("chat").is_empty(), "removed");
        assert!(!t.0.def_path(&def.id).exists());
        let notices = notices.lock().unwrap().clone();
        assert_eq!(notices.len(), 1);
        assert_eq!(notices[0].0, format!("tunnel {}", def.url()));
        assert!(
            notices[0].1.contains("was blocked by the relay"),
            "{notices:?}"
        );
        tokio::time::sleep(Duration::from_millis(1500)).await;
        assert_eq!(hits.load(Ordering::Relaxed), 1, "no retries");

        // A pending purge that meets 410 stops too.
        let mut gone = TunnelDef::for_test("chat", Uuid::nil());
        gone.ticket.relay = relay;
        gone.removed_at = Some(unix_now());
        save_def(&t.0.def_path(&gone.id), &gone).unwrap();
        t.0.purge_in_background(gone.clone(), None);
        poll_until(50, || !t.0.def_path(&gone.id).exists()).await;
        assert!(!t.0.def_path(&gone.id).exists());
        assert_eq!(hits.load(Ordering::Relaxed), 2);
        t.shutdown().await;
    }

    /// A relay that accepts `DELETE /queue` and 404s the rest; its URL and
    /// how many purges it took.
    async fn purging_relay() -> (String, Arc<std::sync::atomic::AtomicUsize>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let relay = format!("ws://{}", listener.local_addr().unwrap());
        let purges = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counted = purges.clone();
        tokio::spawn(async move {
            while let Ok((tcp, _)) = listener.accept().await {
                let counted = counted.clone();
                let service = hyper::service::service_fn(move |req: Request<Incoming>| {
                    let purge =
                        req.method() == hyper::Method::DELETE && req.uri().path() == "/queue";
                    if purge {
                        counted.fetch_add(1, Ordering::Relaxed);
                    }
                    async move {
                        let mut resp = Response::new(Full::new(Bytes::new()));
                        if !purge {
                            *resp.status_mut() = StatusCode::NOT_FOUND;
                        }
                        Ok::<_, Infallible>(resp)
                    }
                });
                tokio::spawn(
                    hyper::server::conn::http1::Builder::new()
                        .serve_connection(TokioIo::new(tcp), service),
                );
            }
        });
        (relay, purges)
    }

    #[tokio::test]
    async fn a_deleted_tunnel_purges_its_relay_queue() {
        let (relay, purges) = purging_relay().await;
        let t = admitted_tunnels(fresh_root("purge"), recorded_notices().0);

        let mut def = TunnelDef::for_test("chat", Uuid::nil());
        def.ticket.relay = relay;
        t.0.launch(def.clone(), None, false).unwrap();
        t.delete("chat", &def.id).unwrap();
        poll_until(50, || !t.0.def_path(&def.id).exists()).await;
        assert!(!t.0.def_path(&def.id).exists(), "purge done");
        assert_eq!(purges.load(Ordering::Relaxed), 1, "relay queue purged");
        t.shutdown().await;
    }

    #[tokio::test]
    async fn an_uninstall_purges_every_relay_queue_before_it_returns() {
        let (relay, purges) = purging_relay().await;
        let root = fresh_root("uninstall");
        let mut pending = TunnelDef::for_test("old", Uuid::nil());
        pending.ticket.relay = relay.clone();
        pending.removed_at = Some(unix_now());
        save_def(&root.join(format!("{}.json", pending.id)), &pending).unwrap();
        let t = admitted_tunnels(root, recorded_notices().0);
        let mut def = TunnelDef::for_test("chat", Uuid::nil());
        def.ticket.relay = relay;
        t.0.launch(def, None, false).unwrap();

        t.uninstall(Duration::from_secs(10)).await;
        assert_eq!(purges.load(Ordering::Relaxed), 2, "both queues purged");

        let t = admitted_tunnels(fresh_root("uninstall-offline"), recorded_notices().0);
        let mut def = TunnelDef::for_test("chat", Uuid::nil());
        def.ticket.relay = "ws://127.0.0.1:1".into();
        t.0.launch(def, None, false).unwrap();
        let begun = std::time::Instant::now();
        let uninstall = tokio::spawn({
            let t = t.clone();
            async move { t.uninstall(Duration::from_millis(300)).await }
        });
        tokio::time::sleep(Duration::from_millis(100)).await;
        let late = TunnelDef::for_test("chat", Uuid::nil());
        assert!(
            t.0.launch(late, Some(0), false).is_err(),
            "no create registers during the purge wait"
        );
        uninstall.await.unwrap();
        assert!(
            begun.elapsed() < SHUTDOWN_TIMEOUT,
            "an offline relay doesn't hold it up"
        );
    }

    #[tokio::test]
    async fn an_unsaved_removal_keeps_the_tunnel_for_the_next_start() {
        let root = fresh_root("unsaved-purge");
        let start = || admitted_tunnels(root.clone(), recorded_notices().0);
        let t = start();
        let mut def = TunnelDef::for_test("chat", Uuid::nil());
        def.ticket.relay = "ws://127.0.0.1:1".into();
        save_def(&t.0.def_path(&def.id), &def).unwrap();
        t.0.launch(def.clone(), None, false).unwrap();
        let blocker = root.join(format!("{}.json.tmp", def.id));
        std::fs::create_dir_all(blocker.join("x")).unwrap();

        let err = t.delete("chat", &def.id).unwrap_err();
        assert!(
            format!("{err:#}").contains("comes back after a spawner restart"),
            "{err:#}"
        );
        assert!(t.list("chat").is_empty(), "closed locally");
        assert_eq!(
            load_def(&t.0.def_path(&def.id)).unwrap().removed_at,
            None,
            "active file kept"
        );
        t.shutdown().await;
        std::fs::remove_dir_all(blocker).unwrap();

        let t = start();
        assert_eq!(t.list("chat").len(), 1, "reopened, not forgotten");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode =
                |m| std::fs::set_permissions(&root, std::fs::Permissions::from_mode(m)).unwrap();
            mode(0o300);
            let unsynced = t.delete("chat", &def.id);
            mode(0o700);
            assert!(
                unsynced.is_err(),
                "a rename whose dir fsync failed is not saved"
            );
        }
        t.shutdown().await;
    }

    type RelayLog = Arc<Mutex<Vec<&'static str>>>;

    /// A relay whose first upstream connect is accepted (and closed after
    /// `close_after`, if set) and later ones refused. A hello registers
    /// `hello_delay` after it arrived, even if the spawner hung up meanwhile
    /// (the Go relay's handler runs on), then is answered `hello_status`.
    async fn hello_relay(
        hello_delay: Duration,
        hello_status: StatusCode,
        close_after: Option<Duration>,
    ) -> (String, RelayLog) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let relay = format!("ws://{}", listener.local_addr().unwrap());
        let log: RelayLog = Arc::default();
        let logged = log.clone();
        tokio::spawn(async move {
            let mut upstreams = 0;
            while let Ok((tcp, _)) = listener.accept().await {
                let mut method = [0u8; 1];
                let _ = tcp.peek(&mut method).await;
                if &method == b"G" {
                    upstreams += 1;
                    if upstreams > 1 {
                        continue;
                    }
                    tokio::spawn(async move {
                        let Ok(mut ws) = tokio_tungstenite::accept_async(tcp).await else {
                            return;
                        };
                        match close_after {
                            Some(after) => tokio::time::sleep(after).await,
                            None => {
                                while let Some(Ok(_)) = futures::StreamExt::next(&mut ws).await {}
                            }
                        }
                    });
                    continue;
                }
                let log = logged.clone();
                let service = hyper::service::service_fn(move |req: Request<Incoming>| {
                    let log = log.clone();
                    async move {
                        let mut resp = Response::new(Full::new(Bytes::new()));
                        if req.method() == hyper::Method::DELETE {
                            log.lock().unwrap().push("delete");
                            return Ok::<_, Infallible>(resp);
                        }
                        let registered = tokio::spawn(async move {
                            tokio::time::sleep(hello_delay).await;
                            log.lock().unwrap().push("hello");
                        });
                        let _ = registered.await;
                        *resp.status_mut() = hello_status;
                        Ok(resp)
                    }
                });
                tokio::spawn(
                    hyper::server::conn::http1::Builder::new()
                        .serve_connection(TokioIo::new(tcp), service),
                );
            }
        });
        (relay, log)
    }

    #[tokio::test]
    async fn a_hello_in_flight_lands_before_the_purge() {
        let delay = Duration::from_millis(300);
        let (relay, log) = hello_relay(delay, StatusCode::NO_CONTENT, None).await;
        let t = admitted_tunnels(fresh_root("hello-purge"), recorded_notices().0);
        let mut def = TunnelDef::for_test("chat", Uuid::nil());
        def.ticket.relay = relay;
        let connected = t.0.launch(def.clone(), None, false).unwrap();
        poll_until(20, || connected.load(Ordering::Relaxed)).await;
        assert!(connected.load(Ordering::Relaxed));
        t.delete("chat", &def.id).unwrap();
        poll_until(50, || !t.0.def_path(&def.id).exists()).await;
        assert_eq!(*log.lock().unwrap(), ["hello", "delete"]);
        t.shutdown().await;
    }

    #[tokio::test]
    async fn a_dropped_connection_stops_its_hello_retries() {
        let close = Some(Duration::from_millis(200));
        let (relay, log) = hello_relay(Duration::ZERO, StatusCode::BAD_GATEWAY, close).await;
        let t = admitted_tunnels(fresh_root("hello-drop"), recorded_notices().0);
        let mut def = TunnelDef::for_test("chat", Uuid::nil());
        def.ticket.relay = relay;
        t.0.launch(def, None, false).unwrap();
        // Unowned, the failing hello would retry at 1 s and 3 s.
        tokio::time::sleep(Duration::from_millis(3500)).await;
        assert_eq!(*log.lock().unwrap(), ["hello"]);
        t.shutdown().await;
    }

    #[test]
    fn a_tunnel_saved_by_the_previous_build_loads() {
        let saved = serde_json::json!({
            "id": "abc", "chat_id": "chat", "user_id": Uuid::nil(), "port": 8080,
            "auth": {"kind": "token", "token": "t"}, "keep_host": false, "no_queue": false,
            "created_at": 1, "key": STANDARD.encode([7u8; 32]),
            "ticket": {"ticket": "x", "exp": 2, "relay": "wss://relay.zuc.run", "url": "https://abc.zuc.run", "fetched_at": 1},
        });
        let path = fresh_root("legacy-def");
        std::fs::write(&path, saved.to_string()).unwrap();
        let def = load_def(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(def.removed_at, None);
        assert_eq!(Local::of(&def).origin, "https://abc.zuc.run");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_removed_tunnel_is_purged_not_reopened_until_its_file_is_gone() {
        use std::os::unix::fs::PermissionsExt;
        let root = fresh_root("stuck-removed");
        let start = || {
            Tunnels::start(
                root.clone(),
                no_backend(),
                recorded_notices().0,
                tokio::sync::mpsc::unbounded_channel().0,
                Timing::default(),
                Some(Uuid::nil()),
            )
        };
        let t = start();
        let removed = TunnelDef {
            removed_at: Some(0),
            ..TunnelDef::for_test("chat", Uuid::nil())
        };
        let path = t.0.def_path(&removed.id);
        save_def(&path, &removed).unwrap();
        let mode = |m| std::fs::set_permissions(&root, std::fs::Permissions::from_mode(m)).unwrap();
        mode(0o500);
        t.0.purge(removed.clone(), None).await;
        mode(0o700);
        assert!(path.exists(), "unlink refused");
        t.shutdown().await;

        let t = start();
        assert!(t.running_in("chat").is_empty(), "not reopened");
        poll_until(20, || !path.exists()).await;
        assert!(!path.exists(), "purge done");
        t.shutdown().await;
    }

    /// Status and `x-zucchini-tunnel` of a POST proxied to `port`.
    async fn proxied_post(port: u16) -> (StatusCode, Option<HeaderValue>) {
        let front = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = front.local_addr().unwrap();
        let local = Arc::new(Local {
            port,
            auth: None,
            keep_host: false,
            origin: String::new(),
        });
        tokio::spawn(async move {
            let (tcp, _) = front.accept().await.unwrap();
            let service =
                hyper::service::service_fn(move |req| proxy(req, local.clone(), Arc::default()));
            let _ = hyper::server::conn::http1::Builder::new()
                .serve_connection(TokioIo::new(tcp), service)
                .await;
        });
        let resp = reqwest::Client::new()
            .post(format!("http://{addr}/hook"))
            .body("{}")
            .send()
            .await
            .unwrap();
        (
            resp.status(),
            resp.headers().get("x-zucchini-tunnel").cloned(),
        )
    }

    /// A local server that reads the request head and then hangs up, or
    /// answers it with its own 502.
    async fn app(own_502: bool) -> u16 {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            while let Ok((mut tcp, _)) = listener.accept().await {
                let mut buf = [0; 4096];
                let _ = tcp.read(&mut buf).await;
                if own_502 {
                    let resp = "HTTP/1.1 502 Bad Gateway\r\ncontent-length: 0\r\n\r\n";
                    let _ = tcp.write_all(resp.as_bytes()).await;
                }
            }
        });
        port
    }

    #[tokio::test]
    async fn only_the_apps_own_502_is_its_answer() {
        let down = Some(HeaderValue::from_static(UPSTREAM_DOWN));
        let closed = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let refused = closed.local_addr().unwrap().port();
        drop(closed);
        assert_eq!(
            proxied_post(refused).await,
            (StatusCode::BAD_GATEWAY, down.clone())
        );
        let hangs_up = app(false).await;
        assert_eq!(
            proxied_post(hangs_up).await,
            (StatusCode::BAD_GATEWAY, down)
        );
        let own_502 = app(true).await;
        assert_eq!(proxied_post(own_502).await, (StatusCode::BAD_GATEWAY, None));
    }

    type RelayWs = tokio_tungstenite::WebSocketStream<tokio_util::compat::Compat<yamux::Stream>>;

    /// A relay connection `serve`d with `grace` to the local `port`, and a
    /// stream piko would open on it for one request.
    async fn stream_through_serve(
        port: u16,
        grace: Duration,
    ) -> (
        yamux::Stream,
        CancellationToken,
        CancellationToken,
        tokio::task::JoinHandle<()>,
    ) {
        let relay = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("ws://{}", relay.local_addr().unwrap());
        let (spawner_side, relay_side) =
            tokio::join!(tokio_tungstenite::connect_async(url), async {
                let (tcp, _) = relay.accept().await.unwrap();
                let tcp = tokio_tungstenite::MaybeTlsStream::Plain(tcp);
                tokio_tungstenite::accept_async(tcp).await.unwrap()
            });
        let local = Arc::new(Local {
            port,
            auth: None,
            keep_host: false,
            origin: String::new(),
        });
        let (session, retire) = (CancellationToken::new(), CancellationToken::new());
        let served = tokio::spawn(serve(
            spawner_side.unwrap().0,
            local,
            session.clone(),
            retire.clone(),
            grace,
        ));
        let mut relay = yamux::Connection::new(
            WsIo::new(relay_side),
            yamux::Config::default(),
            yamux::Mode::Server,
        );
        let stream = futures::future::poll_fn(|cx| relay.poll_new_outbound(cx))
            .await
            .unwrap();
        tokio::spawn(async move {
            while let Some(Ok(_)) = futures::future::poll_fn(|cx| relay.poll_next_inbound(cx)).await
            {
            }
        });
        (stream, session, retire, served)
    }

    /// A relay connection `serve`d with `grace`, and a WebSocket through it
    /// to a local echo server.
    async fn websocket_through_serve(
        grace: Duration,
    ) -> (RelayWs, CancellationToken, tokio::task::JoinHandle<()>) {
        use futures::{SinkExt, StreamExt};
        let echo = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = echo.local_addr().unwrap().port();
        tokio::spawn(async move {
            while let Ok((tcp, _)) = echo.accept().await {
                tokio::spawn(async move {
                    let mut ws = tokio_tungstenite::accept_async(tcp).await.unwrap();
                    while let Some(Ok(msg)) = ws.next().await {
                        if msg.is_text() && ws.send(msg).await.is_err() {
                            return;
                        }
                    }
                });
            }
        });
        let (stream, _, retire, served) = stream_through_serve(port, grace).await;
        let (ws, _) = tokio_tungstenite::client_async("ws://localhost/", stream.compat())
            .await
            .unwrap();
        (ws, retire, served)
    }

    /// A local server that reads a request, answers an upgrade with 101 and
    /// nothing else, then waits; it sends once the spawner closed its socket.
    async fn silent_local() -> (u16, tokio::sync::mpsc::UnboundedReceiver<()>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let (closed, rx) = tokio::sync::mpsc::unbounded_channel();
        tokio::spawn(async move {
            while let Ok((mut tcp, _)) = listener.accept().await {
                let closed = closed.clone();
                tokio::spawn(async move {
                    let mut head = Vec::new();
                    let mut buf = [0u8; 4096];
                    while !head.windows(4).any(|w| w == b"\r\n\r\n") {
                        match tcp.read(&mut buf).await {
                            Ok(0) | Err(_) => return,
                            Ok(n) => head.extend_from_slice(&buf[..n]),
                        }
                    }
                    if head.windows(8).any(|w| w.eq_ignore_ascii_case(b"upgrade:")) {
                        let switch = "HTTP/1.1 101 Switching Protocols\r\nConnection: upgrade\r\nUpgrade: x\r\n\r\n";
                        tcp.write_all(switch.as_bytes()).await.unwrap();
                    }
                    while !matches!(tcp.read(&mut buf).await, Ok(0) | Err(_)) {}
                    let _ = closed.send(());
                });
            }
        });
        (port, rx)
    }

    async fn send_head(stream: &mut tokio_util::compat::Compat<yamux::Stream>, head: &str) {
        use tokio::io::AsyncWriteExt;
        stream.write_all(head.as_bytes()).await.unwrap();
        stream.flush().await.unwrap();
    }

    #[tokio::test]
    async fn a_closed_connection_closes_its_local_sockets() {
        let unanswered = "POST /hook HTTP/1.1\r\nHost: x\r\nContent-Length: 0\r\n\r\n";
        let upgrade = "GET /ws HTTP/1.1\r\nHost: x\r\nConnection: upgrade\r\nUpgrade: x\r\n\r\n";
        for (head, stop) in [
            (unanswered, "stop"),
            (upgrade, "stop"),
            (unanswered, "retire"),
            (upgrade, "retire"),
        ] {
            let (port, mut closed) = silent_local().await;
            let (stream, session, retire, served) =
                stream_through_serve(port, Duration::from_millis(500)).await;
            let mut stream = stream.compat();
            send_head(&mut stream, head).await;
            tokio::time::sleep(Duration::from_millis(300)).await;
            assert!(closed.try_recv().is_err(), "{head:?} closed early");
            match stop {
                "stop" => session.cancel(),
                _ => retire.cancel(),
            }
            tokio::time::timeout(Duration::from_secs(5), served)
                .await
                .unwrap()
                .unwrap();
            tokio::time::timeout(Duration::from_secs(3), closed.recv())
                .await
                .unwrap_or_else(|_| panic!("{head:?} kept its local socket after {stop}"));
        }
    }

    async fn echoes(ws: &mut RelayWs) {
        use futures::{SinkExt, StreamExt};
        ws.send(Message::text("ping")).await.unwrap();
        assert_eq!(ws.next().await.unwrap().unwrap(), Message::text("ping"));
    }

    #[tokio::test]
    async fn a_retired_connection_serves_an_open_websocket_until_it_closes_or_the_grace_ends() {
        let (mut ws, retire, served) = websocket_through_serve(Duration::from_secs(30)).await;
        echoes(&mut ws).await;
        retire.cancel();
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert!(!served.is_finished(), "retired with a WebSocket open");
        echoes(&mut ws).await;
        ws.close(None).await.unwrap();
        drop(ws);
        tokio::time::timeout(Duration::from_secs(5), served)
            .await
            .expect("closes once the WebSocket did")
            .unwrap();

        let (mut ws, retire, served) = websocket_through_serve(Duration::from_secs(1)).await;
        echoes(&mut ws).await;
        retire.cancel();
        tokio::time::timeout(Duration::from_secs(5), served)
            .await
            .expect("closes when the grace ends, the WebSocket still open")
            .unwrap();
    }

    #[test]
    fn a_ticket_is_refreshed_daily_and_replaced_before_exp() {
        let now = unix_now();
        let mut t = Ticket {
            ticket: String::new(),
            exp: now + 7 * 86400,
            relay: String::new(),
            url: String::new(),
            fetched_at: now,
        };
        let timing = Timing::default();
        assert!(!t.refresh_due(&timing) && !t.expiring(&timing));
        t.fetched_at = now - 86400;
        assert!(t.refresh_due(&timing) && !t.expiring(&timing));
        t.fetched_at = now;
        t.exp = now + 60;
        assert!(t.refresh_due(&timing) && t.expiring(&timing));
    }
}

/// Against a relay (`tunnel-relay/dev/run.sh`) through the fake ticket
/// backend `tunnel-relay/cmd/devbackend` on 127.0.0.1:18190, which checks key
/// possession like the real one and signs with the relay's ticket key; how to
/// run them: `zucchini-spawner/CLAUDE.md` "Tunnels". Neither prints a ticket.
#[cfg(all(test, unix))]
mod relay_smoke {
    use super::*;
    use futures::{SinkExt, StreamExt};
    use tokio::net::TcpListener;

    const FAKE_BACKEND: &str = "http://127.0.0.1:18190";

    type Seen = Arc<Mutex<Vec<serde_json::Value>>>;

    fn tunnels(root: &Path, timing: Timing) -> Tunnels {
        let backend = Backend {
            http: reqwest::Client::new(),
            api_base_url: FAKE_BACKEND.into(),
            fetch_token: Arc::new(Box::new(|| Box::pin(async { Ok("jwt".to_string()) }))),
        };
        let tunnels = Tunnels::start(
            root.to_path_buf(),
            backend,
            Box::new(|n| Box::pin(async move { bail!("unexpected notice: {}", n.text) })),
            tokio::sync::mpsc::unbounded_channel().0,
            timing,
            None,
        );
        tunnels.admit_restored(|_| true);
        tunnels
    }

    fn fresh_root() -> PathBuf {
        std::env::temp_dir().join(format!("zucchini-tunnel-{}", Uuid::now_v7()))
    }

    async fn fake_backend(query: &str) {
        reqwest::get(format!("{FAKE_BACKEND}/control?{query}"))
            .await
            .unwrap();
    }

    /// An HTTP server that records each request and answers it with its JSON;
    /// `/sse` streams three events 300 ms apart.
    async fn local_server(listener: TcpListener, seen: Seen) {
        loop {
            let Ok((tcp, _)) = listener.accept().await else {
                return;
            };
            let seen = seen.clone();
            tokio::spawn(async move {
                let service = hyper::service::service_fn(move |req: Request<Incoming>| {
                    let seen = seen.clone();
                    async move {
                        let headers: serde_json::Map<String, serde_json::Value> = req
                            .headers()
                            .iter()
                            .map(|(k, v)| (k.to_string(), v.to_str().unwrap_or("").into()))
                            .collect();
                        let uri = req.uri().to_string();
                        let method = req.method().to_string();
                        let body = req
                            .into_body()
                            .collect()
                            .await
                            .map(|b| b.to_bytes())
                            .unwrap_or_default();
                        let record = serde_json::json!({
                            "method": method, "uri": uri, "headers": headers, "body": String::from_utf8_lossy(&body),
                        });
                        seen.lock().unwrap().push(record.clone());
                        if uri == "/sse" {
                            let (tx, rx) = futures::channel::mpsc::channel::<
                                Result<hyper::body::Frame<Bytes>, Infallible>,
                            >(4);
                            tokio::spawn(async move {
                                let mut tx = tx;
                                for i in 0..3 {
                                    let frame = hyper::body::Frame::data(Bytes::from(format!(
                                        "data: {i}\n\n"
                                    )));
                                    let _ = tx.send(Ok(frame)).await;
                                    tokio::time::sleep(Duration::from_millis(300)).await;
                                }
                            });
                            let mut resp =
                                Response::new(BodyExt::boxed(http_body_util::StreamBody::new(rx)));
                            resp.headers_mut().insert(
                                header::CONTENT_TYPE,
                                HeaderValue::from_static("text/event-stream"),
                            );
                            return Ok::<_, Infallible>(resp);
                        }
                        Ok(Response::new(
                            Full::new(Bytes::from(record.to_string())).boxed(),
                        ))
                    }
                });
                let _ = hyper::server::conn::http1::Builder::new()
                    .serve_connection(TokioIo::new(tcp), service)
                    .await;
            });
        }
    }

    async fn ws_echo(listener: TcpListener) {
        while let Ok((tcp, _)) = listener.accept().await {
            tokio::spawn(async move {
                let Ok(mut ws) = tokio_tungstenite::accept_async(tcp).await else {
                    return;
                };
                while let Some(Ok(msg)) = ws.next().await {
                    if msg.is_text() && ws.send(msg).await.is_err() {
                        return;
                    }
                }
            });
        }
    }

    async fn until(what: &str, secs: u64, cond: impl Fn() -> bool) {
        for _ in 0..secs * 10 {
            if cond() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        panic!("timed out waiting until {what}");
    }

    async fn create(t: &Tunnels, port: u16, auth: Option<&str>) -> TunnelInfo {
        let auth = auth.map(|a| Auth::generate(a).unwrap());
        let prepared = t
            .prepare("chat", Uuid::nil(), port, auth, false, false)
            .await
            .unwrap();
        let id = t.register(prepared).unwrap();
        let info = t.wait_connected(&id).await.unwrap();
        assert!(info.connected, "{} did not connect", info.id);
        info
    }

    async fn serve_local() -> (u16, Seen, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let seen: Seen = Arc::default();
        let server = tokio::spawn(local_server(listener, seen.clone()));
        (port, seen, server)
    }

    #[tokio::test]
    #[ignore = "needs a relay and the fake ticket backend"]
    async fn tunnel_relay_smoke() {
        let root = fresh_root();
        let t = tunnels(&root, Timing::default());
        let (port, seen, server) = serve_local().await;
        let ws_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let ws_port = ws_listener.local_addr().unwrap().port();
        tokio::spawn(ws_echo(ws_listener));

        let open = create(&t, port, None).await;
        let token = create(&t, port, Some("token")).await;
        let basic = create(&t, port, Some("basic")).await;
        let ws = create(&t, ws_port, None).await;
        let Some(Auth::Token { token: tok }) = token.auth.clone() else {
            panic!()
        };
        let Some(Auth::Basic { user, password }) = basic.auth.clone() else {
            panic!()
        };

        let http = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap();
        let get = |info: &TunnelInfo, path: &str| http.get(format!("{}{path}", info.url));
        let json = |r: reqwest::Response| async move {
            let status = r.status();
            let text = r.text().await.unwrap();
            assert_eq!(status, 200, "{text}");
            serde_json::from_str::<serde_json::Value>(&text).unwrap()
        };
        let host = |info: &TunnelInfo| info.url.split("://").nth(1).unwrap().to_string();

        // Host rewrite; no piko or spoofed relay headers reach the local server.
        let echo = json(
            get(&open, "/hello?a=1")
                .header("x-zucchini-replayed", "1")
                .header("x-piko-endpoint", "other")
                .send()
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(echo["uri"], "/hello?a=1");
        assert_eq!(echo["headers"]["host"], format!("localhost:{port}"));
        let no_port = host(&open).split(':').next().unwrap().to_string();
        assert_eq!(echo["headers"]["x-forwarded-host"], no_port);
        let names: Vec<&String> = echo["headers"].as_object().unwrap().keys().collect();
        assert!(
            !names
                .iter()
                .any(|k| k.starts_with("x-piko") || k.starts_with("x-zucchini")),
            "{names:?}"
        );
        println!("ok: host rewrite, header stripping");

        // --auth token / basic.
        assert_eq!(get(&token, "/").send().await.unwrap().status(), 401);
        let echo = json(get(&token, "/").bearer_auth(&tok).send().await.unwrap()).await;
        assert!(echo["headers"].get("authorization").is_none());
        let resp = get(&token, &format!("/p?zt={tok}&b=2"))
            .send()
            .await
            .unwrap();
        let cookie = resp.headers()["set-cookie"].to_str().unwrap().to_string();
        assert!(cookie.ends_with("; Path=/; HttpOnly; Secure; SameSite=Lax"));
        assert_eq!(json(resp).await["uri"], "/p?b=2");
        let echo = json(
            get(&token, "/")
                .header("cookie", format!("zt={tok}; a=1"))
                .send()
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(echo["headers"]["cookie"], "a=1");
        let resp = get(&basic, "/").send().await.unwrap();
        assert_eq!(resp.status(), 401);
        assert!(resp.headers().contains_key("www-authenticate"));
        json(
            get(&basic, "/")
                .basic_auth(user, Some(password))
                .send()
                .await
                .unwrap(),
        )
        .await;
        println!("ok: token and basic auth");

        // SSE arrives as it is written, not at the end.
        let started = std::time::Instant::now();
        let mut sse = get(&open, "/sse").send().await.unwrap().bytes_stream();
        let first = sse.next().await.unwrap().unwrap();
        assert!(first.starts_with(b"data: 0"));
        let first_at = started.elapsed();
        let mut rest = Vec::new();
        while let Some(chunk) = sse.next().await {
            rest.extend_from_slice(&chunk.unwrap());
        }
        assert!(String::from_utf8_lossy(&rest).contains("data: 2"));
        assert!(
            started.elapsed() - first_at >= Duration::from_millis(500),
            "streamed, not buffered"
        );
        println!("ok: SSE streamed (first event after {first_at:?})");

        // WebSocket passthrough.
        let ws_url = ws.url.replacen("http", "ws", 1) + "/ws";
        let (mut client, _) = tokio_tungstenite::connect_async(ws_url).await.unwrap();
        for text in ["ping", "pong"] {
            client.send(Message::text(text)).await.unwrap();
            assert_eq!(client.next().await.unwrap().unwrap(), Message::text(text));
        }
        println!("ok: WebSocket");

        // Local server down: GET gets our 502, POST is queued and replayed.
        server.abort();
        let _ = server.await;
        tokio::time::sleep(Duration::from_millis(200)).await;
        let resp = get(&open, "/").send().await.unwrap();
        let headers = format!("{:?}", resp.headers());
        let status = resp.status();
        let body: String = resp.text().await.unwrap().chars().take(300).collect();
        // The relay turns our upstream-down 502 into its own 503, which
        // Cloudflare passes through (it replaces a 502 with its own page).
        assert_eq!(status, 503, "{headers} {body}");
        assert!(
            !body.is_empty() && !body.starts_with("error code"),
            "{body}"
        );
        println!("upstream-down GET: 503 {body:?}");
        let resp = http
            .post(format!("{}/hook", open.url))
            .body("event-1")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200, "queued");
        let before = seen.lock().unwrap().len();
        let listener = TcpListener::bind(("127.0.0.1", port)).await.unwrap();
        let server = tokio::spawn(local_server(listener, seen.clone()));
        until("the queued request is replayed", 45, || {
            seen.lock().unwrap().len() > before
        })
        .await;
        let replayed = seen.lock().unwrap()[before].clone();
        assert_eq!(
            (replayed["uri"].as_str(), replayed["body"].as_str()),
            (Some("/hook"), Some("event-1"))
        );
        assert_eq!(replayed["headers"]["x-zucchini-replayed"], "1");
        assert!(replayed["headers"].get("x-zucchini-arrived").is_some());
        println!("ok: upstream-down 503, queued POST replayed");

        // Restart: the tunnels come back under the same URLs.
        t.shutdown().await;
        let t = tunnels(&root, Timing::default());
        assert_eq!(t.list("chat").len(), 4);
        until("restored", 30, || {
            t.list("chat").iter().all(|i| i.connected)
        })
        .await;
        tokio::time::sleep(Duration::from_secs(1)).await;
        let echo = json(
            get(&token, "/again")
                .bearer_auth(&tok)
                .send()
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(echo["uri"], "/again");
        println!("ok: restored after a restart");

        // Delete with a request in the relay's queue: once the DELETE /queue
        // landed the relay knows the id no more, so it answers 404 to any
        // request. A 503 would mean it still knows the id, disconnected, and
        // a POST would be queued (200).
        server.abort();
        let _ = server.await;
        let resp = http.post(format!("{}/hook", open.url)).body("pending");
        assert_eq!(resp.send().await.unwrap().status(), 200, "queued");
        for info in t.list("chat") {
            t.delete("chat", &info.id).unwrap();
        }
        until("the queues are purged", 60, || json_files(&root).is_empty()).await;
        for info in [&open, &token, &basic, &ws] {
            for _ in 0..50 {
                if get(info, "/").send().await.unwrap().status() == 404 {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
            assert_eq!(get(info, "/").send().await.unwrap().status(), 404);
            let resp = http.post(format!("{}/hook", info.url)).body("late");
            let status = resp.send().await.unwrap().status();
            assert_eq!(status, 404, "relay kept {}", info.id);
        }
        println!("ok: deleted, relay forgot the ids (404, POST not queued)");
        t.shutdown().await;
    }

    /// Deletes what earlier (failed) runs left in their temp roots, so the
    /// relay purges their queues.
    #[tokio::test]
    #[ignore = "cleanup after a failed relay test"]
    async fn tunnel_test_cleanup() {
        let roots = std::fs::read_dir(std::env::temp_dir())
            .unwrap()
            .flatten()
            .map(|e| e.path());
        for root in roots.filter(|p| {
            p.file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("zucchini-tunnel-0")
        }) {
            let t = tunnels(&root, Timing::default());
            for info in t.list("chat") {
                t.delete("chat", &info.id).unwrap();
            }
            until("purged", 60, || json_files(&root).is_empty()).await;
            t.shutdown().await;
            std::fs::remove_dir_all(&root).unwrap();
            println!("cleaned {}", root.display());
        }
    }

    /// With 150 s tickets: a POST stream across two refresh hand-overs and a
    /// piko disconnect at `exp` (refresh failing meanwhile). Every POST must
    /// get a 200 and reach the local server, live or replayed.
    #[tokio::test]
    #[ignore = "needs a relay and the fake ticket backend issuing 150 s tickets"]
    async fn tunnel_refresh_handover() {
        let timing = Timing {
            refresh_after_secs: 40,
            reconnect_before_exp_secs: 20,
            tick: Duration::from_secs(2),
        };
        let root = fresh_root();
        let t = tunnels(&root, timing);
        fake_backend("fail=0").await;
        let (port, seen, _server) = serve_local().await;
        let info = create(&t, port, None).await;
        tokio::time::sleep(Duration::from_secs(1)).await;

        let sent = Arc::new(Mutex::new(
            Vec::<(u32, u16, bool, std::time::Instant)>::new(),
        ));
        let stop = CancellationToken::new();
        let stream = {
            let (sent, stop, url) = (sent.clone(), stop.clone(), info.url.clone());
            tokio::spawn(async move {
                let http = reqwest::Client::new();
                for n in 0u32.. {
                    if stop.is_cancelled() {
                        return;
                    }
                    let resp = http
                        .post(format!("{url}/n"))
                        .body(n.to_string())
                        .send()
                        .await;
                    let (status, live) = match resp {
                        Ok(r) => {
                            let status = r.status().as_u16();
                            let live = r.text().await.unwrap_or_default().contains("\"uri\"");
                            (status, live)
                        }
                        Err(_) => (0, false),
                    };
                    sent.lock()
                        .unwrap()
                        .push((n, status, live, std::time::Instant::now()));
                    tokio::time::sleep(Duration::from_millis(300)).await;
                }
            })
        };
        let connected = |t: &Tunnels| t.list("chat").first().is_some_and(|i| i.connected);
        let fetched = |t: &Tunnels| t.running_in("chat")[0].ticket.fetched_at;

        // Two refreshes with their make-before-break hand-over.
        let mut refreshes = 0;
        let mut last = fetched(&t);
        while refreshes < 2 {
            until("a refresh", 90, || fetched(&t) != last).await;
            last = fetched(&t);
            refreshes += 1;
            println!("refresh {refreshes} done");
        }
        tokio::time::sleep(Duration::from_secs(35)).await;

        // Refresh fails: piko cuts the connection at the ticket's exp.
        fake_backend("fail=1").await;
        let exp = t.running_in("chat")[0].ticket.exp;
        until("piko disconnects at exp", 200, || !connected(&t)).await;
        let cut_at = std::time::Instant::now();
        println!("disconnected {}s after exp", unix_now() - exp);
        tokio::time::sleep(Duration::from_secs(5)).await;
        fake_backend("fail=0").await;
        until("reconnected", 120, || connected(&t)).await;
        println!("reconnected");
        tokio::time::sleep(Duration::from_secs(10)).await;
        stop.cancel();
        let _ = stream.await;

        let sent = sent.lock().unwrap().clone();
        // A request in flight when piko cuts the connection at exp may have
        // reached the local server, so the relay answers 502 instead of
        // queueing it (no double delivery); it must still have arrived.
        let near_cut = |at: &std::time::Instant| {
            let (a, b) = (*at, cut_at);
            (if a > b { a - b } else { b - a }) < Duration::from_secs(3)
        };
        let failed: Vec<_> = sent
            .iter()
            .filter(|(_, s, _, at)| *s != 200 && !(*s == 502 && near_cut(at)))
            .map(|(n, s, _, _)| (*n, *s))
            .collect();
        let cut: Vec<u32> = sent
            .iter()
            .filter(|(_, s, _, _)| *s != 200)
            .map(|(n, ..)| *n)
            .collect();
        let queued = sent.iter().filter(|(_, _, live, _)| !live).count();
        println!(
            "{} POSTs, {queued} answered by the queue, non-200 at the cut: {cut:?}",
            sent.len()
        );
        let arrived = || -> HashSet<u32> {
            seen.lock()
                .unwrap()
                .iter()
                .filter_map(|r| r["body"].as_str()?.parse().ok())
                .collect()
        };
        until("every POST arrived", 90, || {
            sent.iter().all(|(n, ..)| arrived().contains(n))
        })
        .await;
        assert!(failed.is_empty(), "{failed:?}");
        println!("ok: no POST lost across 2 hand-overs and an exp disconnect");

        t.delete("chat", &info.id).unwrap();
        until("the queue is purged", 60, || json_files(&root).is_empty()).await;
        t.shutdown().await;
    }
}
