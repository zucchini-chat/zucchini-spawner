//! Hermes Unix-socket server — multiplexes the single `hermes gateway run`
//! process and every per-turn trampoline client onto one socket.
//!
//! Topology
//! --------
//!
//! ```
//!                  spawner (this module owns the socket)
//!                  ┌───────────────────────────────────────────────┐
//!                  │  UnixListener bound at $ZUCCHINI_SPAWNER_SOCK │
//!                  │  accept loop fans each connection into a task │
//!                  └─┬───────────────────┬──────────────┬──────────┘
//!                    │                   │              │
//!         "hello"    │                   │ "turn" frame │
//!        (plugin)    │                   │ (trampoline) │
//!                    ▼                   ▼              ▼
//!         ┌──────────────────┐  ┌───────────────┐ ┌───────────────┐
//!         │ ROLE = Plugin    │  │ ROLE =        │ │ ROLE =        │
//!         │ outbound queue   │  │ Trampoline    │ │ Trampoline    │
//!         │ (turn / stop /   │  │ outbound q.   │ │ outbound q.   │
//!         │  ping / hello)   │  │ (envelopes    │ │ (envelopes    │
//!         │                  │  │  for chat X)  │ │  for chat Y)  │
//!         └────────▲─────────┘  └───────▲───────┘ └───────▲───────┘
//!                  │ envelope               │                 │
//!                  │ from plugin            │ turn frame      │ turn frame
//!                  │                        │ from trampoline │ from trampoline
//!                  │                        │   to plugin     │   to plugin
//!                  │ ┌──────────────────────┴─────────────────┴───┐
//!                  └─┤ ROUTER state                                │
//!                    │ • plugin connection (one)                   │
//!                    │ • trampoline registry: HashMap<chat_id,     │
//!                    │     UnboundedSender<envelope_line>>         │
//!                    │ • write_tx → writer (for proactive)          │
//!                    │ • mirror   → for proactive user_id lookup    │
//!                    └─────────────────────────────────────────────┘
//! ```
//!
//! Wire shape (verbatim from `~/.hermes/plugins/zucchini/adapter.py`):
//!
//!   Spawner → plugin (NOT chat-wrapped for control frames):
//!     `{"type":"hello"}`           — initial liveness ack request
//!     `{"type":"ping"}`            — periodic liveness probe
//!     `{"type":"turn","chat_id":...,...}` — forwarded from a trampoline
//!     `{"type":"stop","chat_id":...}`     — forwarded from a trampoline
//!
//!   Plugin → spawner:
//!     `{"type":"hello","version":"..."}`  — control reply
//!     `{"type":"pong"}`                   — control reply
//!     `{"chat_id":...,"proactive":bool,"event":{...claude-shape...}}` — turn output
//!
//! Routing decisions:
//!   - Control frames (`hello`, `pong`) from the plugin → consumed here,
//!     never forwarded.
//!   - Wrapped envelopes with `proactive: true` → routed via
//!     `send_agent_line(write_tx, mirror, chat_id, inner_event)` so the
//!     writer encrypts under the chat's K_user and posts to /api/writes.
//!     If the chat is unknown to the mirror we drop with a warn (same
//!     posture as `send_agent_line`).
//!   - Wrapped envelopes with `proactive: false` → looked up in the
//!     trampoline registry by `chat_id` and forwarded verbatim to that
//!     trampoline's outbound queue. If no trampoline is registered for the
//!     chat_id we drop with a warn — this is the "envelope arrived after
//!     the trampoline disconnected" race (shouldn't happen, the plugin
//!     emits the terminal envelope before closing).
//!
//! Connection identification:
//!   - First frame on a fresh connection identifies the role:
//!     - `{"type":"hello",...}` → Plugin role. Only one plugin connection
//!       is expected at a time; if a second arrives we close the older one
//!       (gateway restart raced with the previous instance shutting down).
//!     - `{"type":"turn","chat_id":<X>,...}` → Trampoline role with chat
//!       id X. Register, forward the turn frame to the plugin, then read
//!       further control frames (stop) on the same connection.
//!     - Anything else → close with a warn.
//!
//! Lifecycle:
//!   - Server is started by `main.rs` after `hermes_support::plugin_install` and before
//!     the main `select!` loop. It owns the supervisor of the
//!     `hermes gateway run` child process and restarts it with backoff on
//!     unexpected exit. On spawner shutdown the server task is cancelled
//!     and the gateway child is SIGTERM'd by `tokio::process::Child::kill`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::process::Command;
use tokio::sync::{mpsc, Mutex};
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use crate::adapter::parse_json_obj;
use crate::state::SharedMirror;
use crate::writer::WriteEvent;

/// Bounded outbound queue per connection. Each per-connection writer task
/// drains this and writes to the socket. Bounded because a stuck client
/// shouldn't blow up the spawner's RAM — back-pressure surfaces as a router
/// `try_send` warning and we drop the envelope rather than block routing.
const PER_CONN_QUEUE_CAPACITY: usize = 256;

/// Periodic ping to the plugin to detect a half-dead connection that hasn't
/// torn down at the TCP/Unix layer. Plugin reply (`pong`) is consumed in
/// the same read loop. We don't enforce a deadline on the pong — the plugin
/// always replies promptly in practice; if it ever doesn't, the
/// gateway-child watcher (`spawn_gateway_supervisor`) will notice the child
/// died and restart it.
const PLUGIN_PING_INTERVAL: Duration = Duration::from_secs(30);

/// Backoff for restarting the `hermes gateway run` child on unexpected
/// exit. We start at 1s and cap at 30s — a tight loop would hammer the
/// system if the gateway is misconfigured (e.g. missing auth.json). The
/// timer is reset to 1s after a successful "plugin connected" handshake.
const GATEWAY_RESTART_MIN: Duration = Duration::from_secs(1);
const GATEWAY_RESTART_MAX: Duration = Duration::from_secs(30);

/// Idle interval used when hermes isn't installed at all. Without this the
/// supervisor crash-loops `hermes gateway run` (which exits 127 instantly)
/// at the 30s backoff cap forever, flooding the journal with start/exit
/// pairs — ~190 lines/hour on a box that never had hermes. Instead we probe
/// PATH first and, while absent, re-probe at this slow cadence so the gateway
/// still auto-starts if the user installs hermes later, just without the
/// spam.
const HERMES_ABSENT_POLL_INTERVAL: Duration = Duration::from_secs(300);

/// Default path for the spawner-owned hermes socket, lives under the
/// spawner's 0700 dir. Overridable via the `ZUCCHINI_SPAWNER_SOCK` env var
/// for dev / tests; main.rs sets the env var on the gateway child + every
/// trampoline.
fn default_socket_path() -> PathBuf {
    crate::zucchini_spawner_dir().join("hermes.sock")
}

/// One handle per registered trampoline. We stash the unbounded sender side
/// of the outbound queue (envelopes from plugin to this trampoline). The
/// writer task on the connection drains the receiver side.
type TrampolineTx = mpsc::Sender<String>;

/// Router-shared state held under an async mutex (`Mutex<RouterState>`).
/// Mutex hold times are microseconds (one `HashMap::get` + one `try_send`)
/// so contention is irrelevant; we use an async mutex only so the type
/// composes with our async tasks without `await_holding_lock` warnings.
struct RouterState {
    plugin: Option<mpsc::Sender<String>>,
    trampolines: HashMap<String, TrampolineTx>,
}

impl RouterState {
    fn new() -> Self {
        Self {
            plugin: None,
            trampolines: HashMap::new(),
        }
    }
}

/// Public handle returned by `start`. Holds the cancellation token so
/// `main.rs` can stop the server on shutdown. The accept loop task watches
/// the token and tears down child + socket on cancel.
pub struct HermesSocketServer {
    pub cancel: CancellationToken,
    pub socket_path: PathBuf,
}

/// Start the hermes socket server + gateway supervisor. Returns a handle
/// with a cancellation token. `write_tx` is the writer channel (for
/// proactive envelopes); `mirror` is the chats/projects mirror (for the
/// user_id lookup inside `send_agent_line` parallel).
pub fn start(
    write_tx: mpsc::Sender<WriteEvent>,
    mirror: SharedMirror,
) -> Result<HermesSocketServer> {
    let socket_path = match std::env::var_os("ZUCCHINI_SPAWNER_SOCK") {
        Some(s) => PathBuf::from(s),
        None => default_socket_path(),
    };

    // Best-effort: remove a leftover socket file from a prior unclean
    // shutdown so `bind` doesn't fail with EADDRINUSE. Ignored if absent.
    if socket_path.exists() {
        if let Err(e) = std::fs::remove_file(&socket_path) {
            warn!(error = %e, path = %socket_path.display(), "stale hermes socket file remove failed");
        }
    }

    let listener = UnixListener::bind(&socket_path)
        .with_context(|| format!("bind hermes socket at {}", socket_path.display()))?;
    // chmod 0600 on the socket file so only the spawner UID can connect.
    // Best-effort: failure means the parent dir's 0700 inheritance is our
    // only auth signal (still strong on a single-user host).
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Err(e) =
            std::fs::set_permissions(&socket_path, std::fs::Permissions::from_mode(0o600))
        {
            warn!(error = %e, path = %socket_path.display(), "chmod 0600 hermes socket failed");
        }
    }
    info!(path = %socket_path.display(), "hermes socket server listening");

    let cancel = CancellationToken::new();
    let cancel_for_task = cancel.clone();
    let socket_path_for_env = socket_path.clone();
    let router = Arc::new(Mutex::new(RouterState::new()));
    let router_for_accept = router.clone();
    let write_tx_for_accept = write_tx.clone();
    let mirror_for_accept = mirror.clone();

    tokio::spawn(async move {
        // Spawn the gateway-child supervisor. It will (re)start
        // `hermes gateway run` on first call, and restart it with backoff
        // on unexpected exit. Failure to start once isn't fatal — the
        // accept loop still serves the socket; the plugin will appear once
        // the gateway eventually starts (e.g. after the user installs
        // hermes).
        let supervisor_cancel = cancel_for_task.clone();
        tokio::spawn(spawn_gateway_supervisor(
            socket_path_for_env,
            supervisor_cancel,
        ));

        run_accept_loop(
            listener,
            router_for_accept,
            write_tx_for_accept,
            mirror_for_accept,
            cancel_for_task,
        )
        .await;
    });

    Ok(HermesSocketServer {
        cancel,
        socket_path,
    })
}

/// Accept loop. On every accept, spawn a per-connection task that reads the
/// first frame to decide the role, then drives the role's read+write loops.
async fn run_accept_loop(
    listener: UnixListener,
    router: Arc<Mutex<RouterState>>,
    write_tx: mpsc::Sender<WriteEvent>,
    mirror: SharedMirror,
    cancel: CancellationToken,
) {
    loop {
        tokio::select! {
            _ = cancel.cancelled() => {
                info!("hermes socket server: shutdown");
                return;
            }
            res = listener.accept() => {
                match res {
                    Ok((stream, _addr)) => {
                        let router = router.clone();
                        let write_tx = write_tx.clone();
                        let mirror = mirror.clone();
                        let cancel = cancel.clone();
                        tokio::spawn(async move {
                            if let Err(e) = handle_connection(stream, router, write_tx, mirror, cancel).await {
                                debug!(error = %e, "hermes connection ended");
                            }
                        });
                    }
                    Err(e) => {
                        warn!(error = %e, "hermes socket accept error");
                        // Avoid a tight error loop if accept is failing.
                        tokio::time::sleep(Duration::from_millis(200)).await;
                    }
                }
            }
        }
    }
}

/// Per-connection task. Reads the first frame, decides the role, runs the
/// role's loops.
async fn handle_connection(
    stream: UnixStream,
    router: Arc<Mutex<RouterState>>,
    write_tx: mpsc::Sender<WriteEvent>,
    mirror: SharedMirror,
    cancel: CancellationToken,
) -> Result<()> {
    let (read_half, write_half) = stream.into_split();
    let mut reader = BufReader::new(read_half);

    // Read the first frame to decide the role.
    let mut buf = String::new();
    let n = reader
        .read_line(&mut buf)
        .await
        .context("read first frame")?;
    if n == 0 {
        return Err(anyhow!("connection closed before first frame"));
    }
    let trimmed = buf.trim_end_matches('\n');
    let Some(obj) = parse_json_obj(trimmed) else {
        warn!(line = %trimmed, "hermes connection first frame not a JSON object, closing");
        return Err(anyhow!("first frame not a JSON object"));
    };
    let frame_type = obj.get("type").and_then(|v| v.as_str()).unwrap_or("");

    match frame_type {
        "hello" => {
            info!("hermes plugin connected");
            run_plugin_connection(reader, write_half, router, write_tx, mirror, cancel, obj).await
        }
        "turn" => {
            let Some(chat_id) = obj
                .get("chat_id")
                .and_then(|v| v.as_str())
                .map(str::to_string)
            else {
                warn!("hermes trampoline connection: turn frame missing chat_id, closing");
                return Err(anyhow!("turn frame missing chat_id"));
            };
            info!(chat_id = %chat_id, "hermes trampoline connected");
            run_trampoline_connection(
                reader,
                write_half,
                router,
                cancel,
                chat_id,
                trimmed.to_string(),
            )
            .await
        }
        other => {
            warn!(ty = %other, "hermes connection unrecognised first frame, closing");
            Err(anyhow!("unrecognised first frame type"))
        }
    }
}

/// Plugin connection loop. The plugin connects out from inside the
/// `hermes gateway run` process; this is the single source of envelopes
/// (proactive + turn-driven). On reconnect we replace the old `plugin`
/// slot; any in-flight trampolines that were depending on the old
/// connection's plugin-side handler will see their envelopes drop until
/// the new plugin re-establishes (acceptable — the plugin's reconnect
/// path is slow and bounded by the backoff in adapter.py).
async fn run_plugin_connection(
    mut reader: BufReader<tokio::net::unix::OwnedReadHalf>,
    write_half: tokio::net::unix::OwnedWriteHalf,
    router: Arc<Mutex<RouterState>>,
    write_tx: mpsc::Sender<WriteEvent>,
    mirror: SharedMirror,
    cancel: CancellationToken,
    first_frame: Value,
) -> Result<()> {
    // Acknowledge the plugin's hello with our own hello so the plugin
    // confirms liveness on its end (per the wire-format spec). We send
    // hello after the plugin has already sent theirs — both sides exchange
    // hello frames as a handshake.
    let plugin_version = first_frame
        .get("version")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    info!(plugin_version, "hermes plugin handshake");

    // Per-connection outbound queue + writer task.
    let (out_tx, out_rx) = mpsc::channel::<String>(PER_CONN_QUEUE_CAPACITY);
    let cancel_for_writer = cancel.clone();
    let writer_handle = tokio::spawn(connection_writer(write_half, out_rx, cancel_for_writer));

    // Register as the plugin connection.
    {
        let mut g = router.lock().await;
        if g.plugin.replace(out_tx.clone()).is_some() {
            warn!("hermes plugin: replacing previous plugin connection (race?)");
        }
    }

    // Send our hello back. Best-effort — if the queue is full something is
    // very wrong, but the read loop below will still drive.
    let our_hello = serde_json::json!({
        "type": "hello",
        "spawner_version": env!("CARGO_PKG_VERSION"),
    })
    .to_string();
    if let Err(e) = out_tx.try_send(our_hello) {
        warn!(error = %e, "hermes plugin: hello queue full");
    }

    // Periodic ping. The plugin replies `pong` on its read loop; the reply
    // is consumed below. We don't enforce a deadline — `Child::wait` on
    // the gateway child is the real liveness signal.
    let cancel_for_ping = cancel.clone();
    let out_tx_for_ping = out_tx.clone();
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(PLUGIN_PING_INTERVAL);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        ticker.tick().await; // skip immediate first tick
        loop {
            tokio::select! {
                _ = cancel_for_ping.cancelled() => return,
                _ = ticker.tick() => {
                    let ping = r#"{"type":"ping"}"#.to_string();
                    if out_tx_for_ping.try_send(ping).is_err() {
                        // queue full or closed → connection is dead; stop pinging.
                        return;
                    }
                }
            }
        }
    });

    // Read loop. Each line is either a plugin control frame
    // (hello / pong — consume here) or a chat-wrapped envelope (route).
    let mut buf = String::new();
    let result: Result<()> = (async {
        loop {
            buf.clear();
            tokio::select! {
                _ = cancel.cancelled() => return Ok(()),
                r = reader.read_line(&mut buf) => {
                    let n = r.context("read from plugin")?;
                    if n == 0 {
                        debug!("hermes plugin closed connection");
                        return Ok(());
                    }
                    let trimmed = buf.trim_end_matches('\n');
                    if trimmed.is_empty() {
                        continue;
                    }
                    let Some(obj) = parse_json_obj(trimmed) else {
                        debug!(line = %trimmed, "hermes plugin: non-JSON frame, ignoring");
                        continue;
                    };
                    if let Some(ty) = obj.get("type").and_then(|v| v.as_str()) {
                        if ty == "hello" || ty == "pong" {
                            // Control reply, consume.
                            debug!(ty, "hermes plugin control frame");
                            continue;
                        }
                    }
                    // Otherwise it's an envelope. Route by chat_id and
                    // `proactive` flag.
                    route_plugin_envelope(&obj, trimmed, &router, &write_tx, &mirror).await;
                }
            }
        }
    })
    .await;

    // Clear our slot if we're still the registered plugin. Identity is
    // tested via `Sender::same_channel` — pointer-compare of `&out_tx` to
    // `&g.plugin.unwrap()` would compare stack/heap addresses of distinct
    // moved-from copies and always be false, defeating the cleanup.
    {
        let mut g = router.lock().await;
        let still_us = matches!(&g.plugin, Some(tx) if tx.same_channel(&out_tx));
        if still_us {
            g.plugin = None;
        }
    }

    drop(out_tx);
    let _ = writer_handle.await;
    result
}

/// Trampoline connection loop. Each trampoline connects per turn, sends ONE
/// `turn` frame, then may send a `stop` frame later (on user SIGINT). We
/// register it in the trampoline map under its `chat_id`, forward the
/// initial turn frame to the plugin, then forward any subsequent
/// `stop` frames; reverse-direction envelopes from the plugin land in our
/// outbound queue via the router.
async fn run_trampoline_connection(
    mut reader: BufReader<tokio::net::unix::OwnedReadHalf>,
    write_half: tokio::net::unix::OwnedWriteHalf,
    router: Arc<Mutex<RouterState>>,
    cancel: CancellationToken,
    chat_id: String,
    first_frame_line: String,
) -> Result<()> {
    let (out_tx, out_rx) = mpsc::channel::<String>(PER_CONN_QUEUE_CAPACITY);
    let cancel_for_writer = cancel.clone();
    let writer_handle = tokio::spawn(connection_writer(write_half, out_rx, cancel_for_writer));

    // Register the trampoline. If a stale entry exists (same chat_id was
    // mid-turn and didn't clean up), replace it — the new turn wins.
    {
        let mut g = router.lock().await;
        if let Some(prev) = g.trampolines.insert(chat_id.clone(), out_tx.clone()) {
            warn!(chat_id = %chat_id, "hermes trampoline: replacing prior registration");
            drop(prev);
        }
    }

    // Forward the turn frame to the plugin connection.
    forward_to_plugin(&router, &first_frame_line).await;

    // Read further frames from this trampoline. Only `stop` is expected
    // post-turn; anything else logs and gets forwarded too (the plugin's
    // read loop will warn on unknowns).
    let mut buf = String::new();
    let result: Result<()> = (async {
        loop {
            buf.clear();
            tokio::select! {
                _ = cancel.cancelled() => return Ok(()),
                r = reader.read_line(&mut buf) => {
                    let n = r.context("read from trampoline")?;
                    if n == 0 {
                        debug!(chat_id = %chat_id, "hermes trampoline closed connection");
                        return Ok(());
                    }
                    let trimmed = buf.trim_end_matches('\n').to_string();
                    if trimmed.is_empty() {
                        continue;
                    }
                    forward_to_plugin(&router, &trimmed).await;
                }
            }
        }
    })
    .await;

    // Cleanup: remove our registration if still ours. `Sender::same_channel`
    // identifies the underlying channel; pointer-compare of `&out_tx` to the
    // HashMap entry would always be false (distinct addresses), leaving stale
    // entries that block re-registration for this chat_id.
    {
        let mut g = router.lock().await;
        let still_ours = g
            .trampolines
            .get(&chat_id)
            .is_some_and(|tx| tx.same_channel(&out_tx));
        if still_ours {
            g.trampolines.remove(&chat_id);
        }
    }
    drop(out_tx);
    let _ = writer_handle.await;
    result
}

/// Drains a per-connection outbound queue onto the socket. One write +
/// flush per envelope so live streaming stays responsive.
async fn connection_writer(
    mut write_half: tokio::net::unix::OwnedWriteHalf,
    mut out_rx: mpsc::Receiver<String>,
    cancel: CancellationToken,
) {
    loop {
        tokio::select! {
            _ = cancel.cancelled() => return,
            msg = out_rx.recv() => {
                let Some(mut msg) = msg else {
                    // Sender dropped — connection done.
                    return;
                };
                if !msg.ends_with('\n') {
                    msg.push('\n');
                }
                if let Err(e) = write_half.write_all(msg.as_bytes()).await {
                    debug!(error = %e, "hermes connection writer: write failed (closing)");
                    return;
                }
                if let Err(e) = write_half.flush().await {
                    debug!(error = %e, "hermes connection writer: flush failed (closing)");
                    return;
                }
            }
        }
    }
}

/// Route a plugin-emitted envelope. Two outcomes:
///   - `proactive: true`  → writer path (encrypt + POST /api/writes).
///   - `proactive: false` → trampoline registry lookup by `chat_id`, then
///     forward the entire line verbatim (the trampoline strips the wrapper
///     on its end). Unmatched chat_id → drop with a warn.
async fn route_plugin_envelope(
    obj: &Value,
    raw_line: &str,
    router: &Arc<Mutex<RouterState>>,
    write_tx: &mpsc::Sender<WriteEvent>,
    mirror: &SharedMirror,
) {
    let Some(chat_id) = obj.get("chat_id").and_then(|v| v.as_str()) else {
        debug!(line = %raw_line, "hermes plugin envelope missing chat_id, dropping");
        return;
    };
    let proactive = obj
        .get("proactive")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    if proactive {
        // Proactive: route through the writer's existing agent-message
        // path. We need to extract the inner `event` and serialize it as
        // the message body — exactly what trampoline-driven assistant
        // frames look like in `messages.body`. iOS treats this as a
        // normal agent line and fires its push handler.
        let Some(event) = obj.get("event") else {
            debug!("hermes proactive envelope missing event, dropping");
            return;
        };
        // Pick the body: if `event` is already a full claude-shape
        // assistant envelope, persist it verbatim; otherwise, if it's a
        // raw string (someone called `BasePlatformAdapter.send(str)`),
        // wrap it into a synthetic assistant text envelope.
        let body = if event.is_object() {
            match serde_json::to_string(event) {
                Ok(s) => s,
                Err(e) => {
                    warn!(error = %e, "hermes proactive: serialize event failed");
                    return;
                }
            }
        } else if let Some(s) = event.as_str() {
            crate::adapters::hermes::synthesize_assistant_text(s)
        } else {
            debug!("hermes proactive event neither object nor string, dropping");
            return;
        };
        send_proactive_agent_line(write_tx, mirror, chat_id, body).await;
        return;
    }

    // Turn-driven envelope: look up the registered trampoline by chat_id.
    let tx_opt = {
        let g = router.lock().await;
        g.trampolines.get(chat_id).cloned()
    };
    match tx_opt {
        Some(tx) => {
            if let Err(e) = tx.try_send(raw_line.to_string()) {
                warn!(chat_id = %chat_id, error = %e, "hermes router: drop envelope, trampoline queue full/closed");
            }
        }
        None => {
            debug!(chat_id = %chat_id, "hermes router: no trampoline for envelope, dropping");
        }
    }
}

/// Forward a frame (turn / stop / whatever) from a trampoline to the plugin
/// connection. Drops the frame with a warn if there's no plugin connected
/// (gateway crashed, hasn't restarted yet) — the trampoline will see the
/// envelope idle timeout and exit non-zero.
async fn forward_to_plugin(router: &Arc<Mutex<RouterState>>, raw_line: &str) {
    let tx_opt = {
        let g = router.lock().await;
        g.plugin.clone()
    };
    match tx_opt {
        Some(tx) => {
            if let Err(e) = tx.try_send(raw_line.to_string()) {
                warn!(error = %e, "hermes router: drop frame, plugin queue full/closed");
            }
        }
        None => {
            warn!("hermes router: no plugin connected, dropping frame");
        }
    }
}

/// Mirror of `main::send_agent_line` for the proactive lane. We can't call
/// the main one directly without exposing it (it's `pub(crate)` already so
/// we could, but the proactive path is a slightly different shape:
/// `chat_id` here is `&str` straight off the wire, and we don't want to
/// drop on a missing chat — same warn posture is fine). Reuses the writer
/// channel and the mirror's user_id lookup.
async fn send_proactive_agent_line(
    write_tx: &mpsc::Sender<WriteEvent>,
    mirror: &SharedMirror,
    chat_id: &str,
    content: String,
) {
    let user_id = {
        let g = mirror.read().await;
        g.chats.get(chat_id).map(|c| c.user_id)
    };
    match user_id {
        Some(uid) => {
            let _ = write_tx
                .send(WriteEvent::agent_line(chat_id.to_string(), uid, content))
                .await;
        }
        None => {
            warn!(chat_id = %chat_id, "hermes proactive: chat not in mirror, dropping");
        }
    }
}

/// Supervise the `hermes gateway run` child process. Restart with bounded
/// exponential backoff. We don't model the plugin's connection state
/// directly — if the gateway exits, its socket connection drops, the
/// plugin slot in the router clears, and we restart the child; the plugin
/// will reconnect on its own once `hermes gateway run` is back up.
async fn spawn_gateway_supervisor(socket_path: PathBuf, cancel: CancellationToken) {
    let mut backoff = GATEWAY_RESTART_MIN;
    let mut logged_absent = false;
    loop {
        if cancel.is_cancelled() {
            return;
        }
        // Don't crash-loop the gateway when hermes isn't installed. Probe
        // PATH the same way the spawn resolves it (`-lic` login shell) and,
        // while absent, idle at a slow cadence — logging the transition just
        // once so the journal isn't flooded. Re-probing each tick means the
        // gateway starts automatically once hermes appears on PATH.
        if !crate::shell::binary_on_path("hermes").await {
            if !logged_absent {
                info!(
                    poll_secs = HERMES_ABSENT_POLL_INTERVAL.as_secs(),
                    "hermes not on PATH; gateway supervisor idle (re-probing)"
                );
                logged_absent = true;
            }
            tokio::select! {
                _ = cancel.cancelled() => return,
                _ = tokio::time::sleep(HERMES_ABSENT_POLL_INTERVAL) => {}
            }
            continue;
        }
        if logged_absent {
            info!("hermes now on PATH; starting gateway supervisor");
            logged_absent = false;
            backoff = GATEWAY_RESTART_MIN;
        }
        let started = std::time::Instant::now();
        match start_gateway_child(&socket_path).await {
            Ok(mut child) => {
                info!("hermes gateway run: started, pid={:?}", child.id());
                let exit_status = tokio::select! {
                    _ = cancel.cancelled() => {
                        let _ = child.kill().await;
                        return;
                    }
                    r = child.wait() => r,
                };
                match exit_status {
                    Ok(s) => info!(?s, "hermes gateway run exited"),
                    Err(e) => warn!(error = %e, "hermes gateway wait failed"),
                }
                // Reset backoff if the gateway stayed alive >= 30s (good
                // run); otherwise grow it.
                if started.elapsed() >= Duration::from_secs(30) {
                    backoff = GATEWAY_RESTART_MIN;
                }
            }
            Err(e) => {
                warn!(error = %e, "hermes gateway run: failed to start");
            }
        }
        tokio::select! {
            _ = cancel.cancelled() => return,
            _ = tokio::time::sleep(backoff) => {}
        }
        backoff = (backoff * 2).min(GATEWAY_RESTART_MAX);
    }
}

/// Launch `hermes gateway run` with `ZUCCHINI_SPAWNER_SOCK` baked in.
/// stdout/stderr are inherited so the operator sees gateway logs in the
/// spawner journal. Returns the child handle.
async fn start_gateway_child(socket_path: &Path) -> Result<tokio::process::Child> {
    // Run under the user's login shell so the plugin discovers hermes via
    // PATH the same way an interactive `hermes` invocation would. Same
    // convention as the agent supervisor (`agent.rs::default_spawn_fn`).
    let user_shell = crate::shell::user_login_shell();
    let mut cmd = Command::new(&user_shell);
    // `-lic` (login + interactive + command) — MUST match `agent.rs`'s
    // `default_spawn_fn` and `shell.rs`'s `binary_on_path` probe verbatim.
    // The `-i` is load-bearing, not cosmetic: many installers (hermes's own
    // `~/.local/bin`, asdf, mise) extend PATH only in `.zshrc`, which a
    // non-interactive login shell does NOT source. Drop the `-i` and the
    // probe still finds hermes (it uses `-lic`) while this launch can't —
    // `hermes gateway run` exits 127 and crash-loops, exactly the
    // installed-but-unspawnable skew `shell.rs` warns about.
    cmd.args(["-lic", "hermes gateway run"])
        .env("ZUCCHINI_SPAWNER_SOCK", socket_path)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    cmd.spawn()
        .with_context(|| "spawn hermes gateway run under login shell")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::Mirror;
    use uuid::Uuid;

    /// Build a synthetic plugin-emitted envelope JSON and verify the
    /// routing decision via `route_plugin_envelope`. Trampoline-bound
    /// envelopes land on the registered queue; proactive envelopes land on
    /// the writer channel.
    #[tokio::test]
    async fn proactive_envelope_routes_to_writer_when_chat_known() {
        let user_id = Uuid::now_v7();
        let chat_id = Uuid::now_v7().to_string();
        let mut mirror = Mirror::default();
        mirror.set_user_id(user_id);
        mirror.upsert_chat(
            chat_id.clone(),
            &serde_json::json!({
                "id": chat_id.clone(),
                "project_id": Uuid::now_v7().to_string(),
                "user_id": user_id.to_string(),
                "last_seq": 0,
                "agent_session_id": serde_json::Value::Null,
                "agent_kind": "hermes",
                "worktree": false,
            })
            .to_string(),
        );
        let mirror = Arc::new(tokio::sync::RwLock::new(mirror));

        let (write_tx, mut write_rx) = mpsc::channel::<WriteEvent>(64);
        let router = Arc::new(Mutex::new(RouterState::new()));

        let env = serde_json::json!({
            "chat_id": chat_id,
            "proactive": true,
            "event": "good morning",
        });
        let raw = env.to_string();
        route_plugin_envelope(&env, &raw, &router, &write_tx, &mirror).await;

        let got = write_rx.try_recv().expect("expected one write event");
        match got {
            WriteEvent::PutMessage {
                chat_id: cid,
                user_id: uid,
                content,
                sender,
                ..
            } => {
                assert_eq!(cid, chat_id);
                assert_eq!(uid, user_id);
                assert_eq!(sender, "agent");
                // Synthesised claude-shape assistant text envelope wrapping
                // "good morning".
                let v: Value = serde_json::from_str(&content).unwrap();
                assert_eq!(v["type"], "assistant");
                assert_eq!(v["message"]["content"][0]["text"], "good morning");
            }
            other => panic!("expected PutMessage, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn proactive_envelope_with_object_event_persisted_verbatim() {
        let user_id = Uuid::now_v7();
        let chat_id = Uuid::now_v7().to_string();
        let mut mirror = Mirror::default();
        mirror.set_user_id(user_id);
        mirror.upsert_chat(
            chat_id.clone(),
            &serde_json::json!({
                "id": chat_id.clone(),
                "project_id": Uuid::now_v7().to_string(),
                "user_id": user_id.to_string(),
                "last_seq": 0,
                "agent_session_id": serde_json::Value::Null,
                "agent_kind": "hermes",
                "worktree": false,
            })
            .to_string(),
        );
        let mirror = Arc::new(tokio::sync::RwLock::new(mirror));

        let (write_tx, mut write_rx) = mpsc::channel::<WriteEvent>(64);
        let router = Arc::new(Mutex::new(RouterState::new()));

        // Full claude-shape assistant envelope inside `event`.
        let inner = serde_json::json!({
            "type": "assistant",
            "message": {
                "content": [{"type":"text","text":"news of the day"}],
                "usage": {"input_tokens": 0, "output_tokens": 0}
            }
        });
        let env = serde_json::json!({
            "chat_id": chat_id,
            "proactive": true,
            "event": inner.clone(),
        });
        let raw = env.to_string();
        route_plugin_envelope(&env, &raw, &router, &write_tx, &mirror).await;

        let got = write_rx.try_recv().expect("expected one write event");
        match got {
            WriteEvent::PutMessage { content, .. } => {
                let v: Value = serde_json::from_str(&content).unwrap();
                assert_eq!(v["type"], "assistant");
                assert_eq!(v["message"]["content"][0]["text"], "news of the day");
            }
            other => panic!("expected PutMessage, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn proactive_envelope_dropped_for_unknown_chat() {
        let mirror = Arc::new(tokio::sync::RwLock::new(Mirror::default()));
        let (write_tx, mut write_rx) = mpsc::channel::<WriteEvent>(64);
        let router = Arc::new(Mutex::new(RouterState::new()));

        let env = serde_json::json!({
            "chat_id": "unknown",
            "proactive": true,
            "event": "hi",
        });
        let raw = env.to_string();
        route_plugin_envelope(&env, &raw, &router, &write_tx, &mirror).await;

        assert!(
            write_rx.try_recv().is_err(),
            "no write event when chat is unknown"
        );
    }

    #[tokio::test]
    async fn turn_driven_envelope_routes_to_registered_trampoline() {
        let mirror = Arc::new(tokio::sync::RwLock::new(Mirror::default()));
        let (write_tx, mut write_rx) = mpsc::channel::<WriteEvent>(64);
        let router = Arc::new(Mutex::new(RouterState::new()));

        // Register a fake trampoline for chat "c1".
        let (t_tx, mut t_rx) = mpsc::channel::<String>(8);
        {
            let mut g = router.lock().await;
            g.trampolines.insert("c1".to_string(), t_tx);
        }

        let env = serde_json::json!({
            "chat_id": "c1",
            "proactive": false,
            "event": {"type":"assistant"},
        });
        let raw = env.to_string();
        route_plugin_envelope(&env, &raw, &router, &write_tx, &mirror).await;

        let got = t_rx
            .try_recv()
            .expect("expected one envelope on trampoline queue");
        // Verbatim line (the trampoline strips the wrapper on its end).
        assert_eq!(got, raw);
        // No write event on the proactive path.
        assert!(write_rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn turn_driven_envelope_dropped_when_no_trampoline() {
        let mirror = Arc::new(tokio::sync::RwLock::new(Mirror::default()));
        let (write_tx, mut write_rx) = mpsc::channel::<WriteEvent>(64);
        let router = Arc::new(Mutex::new(RouterState::new()));

        let env = serde_json::json!({
            "chat_id": "nobody-home",
            "proactive": false,
            "event": {"type":"assistant"},
        });
        let raw = env.to_string();
        route_plugin_envelope(&env, &raw, &router, &write_tx, &mirror).await;

        assert!(write_rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn forward_to_plugin_when_registered() {
        let router = Arc::new(Mutex::new(RouterState::new()));
        let (p_tx, mut p_rx) = mpsc::channel::<String>(8);
        {
            let mut g = router.lock().await;
            g.plugin = Some(p_tx);
        }
        forward_to_plugin(&router, "{\"type\":\"turn\"}").await;
        let got = p_rx.try_recv().unwrap();
        assert_eq!(got, "{\"type\":\"turn\"}");
    }

    #[tokio::test]
    async fn forward_to_plugin_drops_when_none() {
        let router = Arc::new(Mutex::new(RouterState::new()));
        // No plugin registered — should silently drop, not panic.
        forward_to_plugin(&router, "{\"type\":\"turn\"}").await;
    }
}
