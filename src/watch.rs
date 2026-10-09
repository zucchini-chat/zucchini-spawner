//! Watches: shell commands the daemon runs for a chat until they exit or
//! their TTL passes. Each stdout line becomes a message into the chat (lines
//! within `Limits::batch` go as one); any exit of the command ends the watch.
//!
//! A watch lives in `~/.zucchini-spawner/watches/<id>/`:
//!   - `watch.json`  the definition; a watch found at boot is relaunched once
//!     the first checkpoint confirms its chat's access (`admit_restored`)
//!   - `state/`      the command's own directory (`ZUCCHINI_WATCH_STATE_DIR`),
//!     where a poller keeps its cursor across spawner restarts
//!   - `outbox/`     messages the chat hasn't taken yet, one file each
//!   - `stderr.log`  the current run's stderr; a failed run's end notice
//!     quotes its tail
//!
//! `watches/stopped/<chat_id>` lists the watches a `/stop` removed, until the
//! chat's agent is told on its next turn.
//!
//! A message is handed to the main loop (`main.rs::deliver_watch_message`),
//! which writes it as a system row and starts the chat's turn with it. The
//! directory outlives the command until the outbox is empty, so an end notice
//! the chat couldn't take yet still arrives.

use std::collections::{HashMap, HashSet, VecDeque};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufRead, AsyncBufReadExt, BufReader};
use tokio::sync::{mpsc, Notify};
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};
use uuid::Uuid;

use crate::sys;

const MAX_WATCHES: usize = 50;
/// A message longer than this is cut: every one is a user turn.
const MAX_MESSAGE_BYTES: usize = 8 * 1024;
/// How much of a failed run's stderr its end notice quotes.
const STDERR_TAIL_BYTES: u64 = 1024;
const MAX_STDERR_LOG_BYTES: u64 = 1024 * 1024;
/// How long a command's output is still read after it exited: a background
/// child that inherited stdout would otherwise keep the pipe open forever.
const EXIT_DRAIN: Duration = Duration::from_secs(1);
/// tokio's clock stands still while the machine sleeps, so the TTL is checked
/// against the wall clock on a tick.
const EXPIRY_TICK: Duration = Duration::from_secs(60);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(10);
/// Under the watches root, one file per chat: the watches a `/stop` removed
/// that its agent hasn't been told about yet.
const STOP_NOTES_DIR: &str = "stopped";
const STOP_NOTE_CMD_CHARS: usize = 200;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatchDef {
    pub id: Uuid,
    pub chat_id: String,
    pub user_id: Uuid,
    pub cmd: String,
    pub cwd: PathBuf,
    pub description: String,
    /// Unix seconds.
    pub created_at: i64,
    /// Unix seconds; `None` = no TTL.
    pub expires_at: Option<i64>,
    /// The command is over; only the outbox is left to deliver.
    #[serde(default)]
    pub ended: bool,
}

impl WatchDef {
    fn expired(&self) -> bool {
        self.expires_at.is_some_and(|at| unix_now() >= at)
    }

    /// Until the TTL passes; with none, far longer than any recheck.
    fn remaining(&self) -> Duration {
        match self.expires_at {
            Some(at) => Duration::from_secs((at - unix_now()).max(0) as u64),
            None => Duration::MAX,
        }
    }

    /// Heads the turn the agent gets.
    pub fn label(&self) -> String {
        format!("[zucchini watch \"{}\"]", self.description)
    }

    /// Whom the chat's system row names as the sender.
    pub fn sender_name(&self) -> String {
        format!("watch \"{}\"", self.description)
    }

    /// What `watch list` prints for it.
    pub fn summary(&self) -> String {
        let expires = match self.expires_at {
            None => "never".to_string(),
            Some(at) => format!("in {}", human_duration((at - unix_now()).max(0) as u64)),
        };
        let mut out = format!("== {}", self.description);
        if self.ended {
            out.push_str("  (ended, delivering its last messages)");
        }
        out.push_str(&format!(
            "\ncmd: {}\ncwd: {}\nexpires: {expires}\n",
            self.cmd,
            self.cwd.display(),
        ));
        out
    }
}

/// One message on its way to the watch's chat, offered again until taken.
pub struct Outgoing {
    pub watch: WatchDef,
    pub text: String,
    /// Cancelled once the watch is stopped or removed: a message still on its
    /// way to the main loop must start no turn after a `/stop`.
    pub cancelled: CancellationToken,
}

pub type Deliver = Box<dyn Fn(Outgoing) -> BoxFuture<'static, Result<()>> + Send + Sync>;

/// Gets the id of every chat whose running watches changed, so the main loop
/// rewrites its `chats.live_tasks`.
pub type Changed = mpsc::UnboundedSender<String>;

#[derive(Clone)]
pub struct Limits {
    /// Lines this close together go as one message.
    batch: Duration,
    /// More than `flood_max` messages within `flood_window` stops the watch.
    flood_max: usize,
    flood_window: Duration,
    delivery_retry: Duration,
    /// What a stopped command's process group gets between SIGTERM and SIGKILL.
    sigterm_grace: Duration,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            batch: Duration::from_millis(200),
            flood_max: 120,
            flood_window: Duration::from_secs(600),
            delivery_retry: Duration::from_secs(15),
            sigterm_grace: Duration::from_secs(5),
        }
    }
}

enum End {
    Exited(std::process::ExitStatus),
    /// The command never ran or its exit is unknown; holds what happened.
    Failed(String),
    Expired,
    Flood,
}

struct Entry {
    def: WatchDef,
    cancel: CancellationToken,
    task: tokio::task::JoinHandle<()>,
}

struct Inner {
    root: PathBuf,
    deliver: Deliver,
    changed: Changed,
    limits: Limits,
    entries: Mutex<HashMap<Uuid, Entry>>,
    /// Cancelled by `admit_restored`; the watches found on disk wait for it.
    restored_admitted: CancellationToken,
    /// Set by `shutdown`, under the `entries` lock: nothing is launched after,
    /// and the watches it stopped keep their chats' rows in `state.json` and
    /// their place in `chats.live_tasks`.
    stopped_at_shutdown: OnceLock<Vec<WatchDef>>,
    /// The background removals, which `shutdown` waits for: a removed watch's
    /// directory must not be left for the next start to run.
    removals: Mutex<Vec<tokio::task::JoinHandle<()>>>,
}

#[derive(Clone)]
pub struct Watches(Arc<Inner>);

static GLOBAL: OnceLock<Watches> = OnceLock::new();

pub fn default_root() -> PathBuf {
    crate::zucchini_spawner_dir().join("watches")
}

/// Load every watch on disk and make them reachable through [`global`]; they
/// run from [`admit_restored`] on. Call once, holding the single-instance
/// lock, after the orphan sweep.
pub fn start(deliver: Deliver, changed: Changed) {
    let watches = Watches::start(default_root(), deliver, changed, Limits::default());
    let _ = GLOBAL.set(watches);
}

/// The first checkpoint is applied, so access is known again: the watches
/// found on disk start, except those `allowed` refuses, which are removed
/// with no notice.
pub fn admit_restored(allowed: impl Fn(&WatchDef) -> bool) {
    if let Some(watches) = GLOBAL.get() {
        watches.admit_restored(allowed);
    }
}

pub fn global() -> Result<&'static Watches> {
    GLOBAL
        .get()
        .ok_or_else(|| anyhow!("watches are not running in this spawner"))
}

/// The chats that own a watch. Their rows must survive a restart (`state.rs`):
/// PowerSync won't re-stream them and a message needs the row to start a turn.
pub fn watched_chat_ids() -> HashSet<String> {
    GLOBAL
        .get()
        .map(|watches| chat_ids(&watches.0.defs()))
        .unwrap_or_default()
}

fn chat_ids(defs: &[WatchDef]) -> HashSet<String> {
    defs.iter().map(|d| d.chat_id.clone()).collect()
}

/// The chat's watches whose command is not over, oldest first: its entries in
/// `chats.live_tasks`.
pub fn running_in(chat_id: &str) -> Vec<WatchDef> {
    GLOBAL
        .get()
        .map(|watches| watches.running_in(chat_id))
        .unwrap_or_default()
}

/// The chat's row is gone: its watches go with it, with no notice.
pub fn chat_deleted(chat_id: &str) {
    if let Some(watches) = GLOBAL.get() {
        watches.remove_in_background(|d| d.chat_id == chat_id);
        let _ = std::fs::remove_file(watches.0.stop_note_path(chat_id));
    }
}

/// `/stop`: the chat's watches go, with no notice; [`take_stop_note`] tells
/// its agent later. `true` = it had running ones.
pub fn chat_stopped(chat_id: &str) -> bool {
    GLOBAL
        .get()
        .is_some_and(|watches| watches.chat_stopped(chat_id))
}

/// What stops removed from the chat since the last call, worded for its
/// agent's next turn.
pub fn take_stop_note(chat_id: &str) -> Option<String> {
    GLOBAL.get()?.take_stop_note(chat_id)
}

/// The member was revoked or their sandbox setting changed: a watch runs
/// outside any sandbox, so theirs are removed.
pub fn member_restricted(user_id: Uuid) {
    remove_in_background(|d| d.user_id == user_id);
}

fn remove_in_background(pred: impl Fn(&WatchDef) -> bool) {
    if let Some(watches) = GLOBAL.get() {
        watches.remove_in_background(pred);
    }
}

/// Stop every command; definitions and outboxes stay for the next start.
pub async fn shutdown() {
    if let Some(watches) = GLOBAL.get() {
        watches.shutdown().await;
    }
}

impl Watches {
    fn start(root: PathBuf, deliver: Deliver, changed: Changed, limits: Limits) -> Self {
        let inner = Arc::new(Inner {
            root,
            deliver,
            changed,
            limits,
            entries: Mutex::new(HashMap::new()),
            restored_admitted: CancellationToken::new(),
            stopped_at_shutdown: OnceLock::new(),
            removals: Mutex::new(Vec::new()),
        });
        for entry in std::fs::read_dir(&inner.root)
            .into_iter()
            .flatten()
            .flatten()
            .filter(|entry| entry.file_name() != STOP_NOTES_DIR)
        {
            match load_def(&entry.path()) {
                Ok(def) => {
                    inner.launch(def, true);
                }
                Err(e) => {
                    warn!(dir = %entry.path().display(), error = %e, "skipping unreadable watch")
                }
            }
        }
        Self(inner)
    }

    fn admit_restored(&self, allowed: impl Fn(&WatchDef) -> bool) {
        // Once: a watch created since passed the access check at its creation.
        if self.0.restored_admitted.is_cancelled() {
            return;
        }
        self.remove_in_background(|d| !allowed(d));
        self.0.restored_admitted.cancel();
        for chat_id in chat_ids(&self.0.defs()) {
            self.0.notify(&chat_id);
        }
    }

    fn running_in(&self, chat_id: &str) -> Vec<WatchDef> {
        let mut defs = self.0.defs();
        defs.retain(|d| d.chat_id == chat_id && !d.ended);
        defs.sort_by_key(|d| d.id);
        defs
    }

    fn chat_stopped(&self, chat_id: &str) -> bool {
        let stopped = self.running_in(chat_id);
        self.remove_in_background(|d| d.chat_id == chat_id);
        if stopped.is_empty() {
            return false;
        }
        // On disk: the next turn may come after a spawner restart.
        let path = self.0.stop_note_path(chat_id);
        let mut note = std::fs::read_to_string(&path).unwrap_or_default();
        for watch in &stopped {
            let cmd = watch.cmd.split_whitespace().collect::<Vec<_>>().join(" ");
            let cmd: String = cmd.chars().take(STOP_NOTE_CMD_CHARS).collect();
            note.push_str(&format!("\"{}\" ({cmd})\n", watch.description));
        }
        let saved = std::fs::create_dir_all(self.0.root.join(STOP_NOTES_DIR))
            .and_then(|()| crate::atomic::atomic_write_private(&path, note.as_bytes()));
        if let Err(e) = saved {
            warn!(chat_id, error = %e, "failed to keep the stop note");
        }
        true
    }

    fn take_stop_note(&self, chat_id: &str) -> Option<String> {
        let path = self.0.stop_note_path(chat_id);
        let note = std::fs::read_to_string(&path).ok()?;
        let _ = std::fs::remove_file(&path);
        let watches: Vec<&str> = note.lines().collect();
        (!watches.is_empty()).then(|| {
            format!(
                "[zucchini] The user pressed Stop, which also removed this chat's watches: {}. \
                 They are not running any more.",
                watches.join(", ")
            )
        })
    }

    /// Stop the matching watches now; their tasks are joined and their
    /// directories removed in the background, with no notice.
    fn remove_in_background(&self, pred: impl Fn(&WatchDef) -> bool) {
        let removed: Vec<Entry> = {
            let mut entries = self.0.entries();
            let ids: Vec<Uuid> = entries
                .values()
                .filter(|e| pred(&e.def))
                .map(|e| e.def.id)
                .collect();
            ids.iter().filter_map(|id| entries.remove(id)).collect()
        };
        if removed.is_empty() {
            return;
        }
        for entry in &removed {
            entry.cancel.cancel();
        }
        let watches = self.clone();
        let removal = tokio::spawn(async move {
            for entry in removed {
                watches.finish_removal(entry, None).await;
            }
        });
        let mut removals = self.0.removals.lock().expect("watch removals mutex");
        removals.retain(|r| !r.is_finished());
        removals.push(removal);
    }

    pub fn create(
        &self,
        chat_id: &str,
        user_id: Uuid,
        cmd: &str,
        cwd: &str,
        ttl: &str,
        description: &str,
    ) -> Result<WatchDef> {
        if cmd.trim().is_empty() {
            bail!("--cmd is empty");
        }
        let description = description.trim();
        if description.is_empty() {
            bail!("--description is empty");
        }
        let cwd = PathBuf::from(cwd);
        if !cwd.is_absolute() || !cwd.is_dir() {
            bail!("working directory {} is not a directory", cwd.display());
        }
        let ttl = parse_ttl(ttl)?;
        if self.list().len() >= MAX_WATCHES {
            bail!("this machine already runs {MAX_WATCHES} watches, the most it takes");
        }
        if self.find(chat_id, description).is_some() {
            bail!("this chat already has a watch \"{description}\"; delete it first or pick another description");
        }
        let now = unix_now();
        let def = WatchDef {
            id: Uuid::now_v7(),
            chat_id: chat_id.to_string(),
            user_id,
            cmd: cmd.to_string(),
            cwd,
            description: description.to_string(),
            created_at: now,
            expires_at: ttl.map(|t| now.saturating_add(t.as_secs() as i64)),
            ended: false,
        };
        let dir = self.0.dir(def.id);
        std::fs::create_dir_all(dir.join("state"))
            .and_then(|()| std::fs::create_dir_all(dir.join("outbox")))
            .with_context(|| format!("create {}", dir.display()))?;
        save_def(&dir, &def)?;
        if !self.0.launch(def.clone(), false) {
            remove_dir(&dir);
            bail!("the spawner is shutting down; create the watch again in a moment");
        }
        self.0.notify(chat_id);
        info!(id = %def.id, chat_id, cmd, "watch created");
        Ok(def)
    }

    pub fn list(&self) -> Vec<WatchDef> {
        let entries = self.0.entries();
        let mut defs: Vec<WatchDef> = entries.values().map(|e| e.def.clone()).collect();
        defs.sort_by_key(|d| d.id);
        defs
    }

    /// The chat's running watch of that description, its name within the chat.
    pub fn find(&self, chat_id: &str, description: &str) -> Option<WatchDef> {
        self.list()
            .into_iter()
            .find(|w| w.chat_id == chat_id && !w.ended && w.description == description.trim())
    }

    /// Stop and remove a watch. With `notice` its chat is told why, and the
    /// directory stays until that is delivered.
    pub async fn remove(&self, id: Uuid, notice: Option<&str>) -> Option<WatchDef> {
        let entry = self.0.entries().remove(&id)?;
        Some(self.finish_removal(entry, notice).await)
    }

    async fn finish_removal(&self, entry: Entry, notice: Option<&str>) -> WatchDef {
        let id = entry.def.id;
        entry.cancel.cancel();
        let _ = entry.task.await;
        let dir = self.0.dir(id);
        // The command may have ended on its own while we waited.
        let Ok(def) = load_def(&dir) else {
            self.0.notify(&entry.def.chat_id);
            return entry.def;
        };
        match notice {
            Some(text) if !def.ended => {
                let ended = self.0.end(&def, &Outbox::new(&dir), text);
                self.0.launch(ended, false);
            }
            _ => {
                remove_dir(&dir);
                self.0.notify(&def.chat_id);
            }
        }
        info!(%id, "watch removed");
        def
    }

    async fn shutdown(&self) {
        let entries: Vec<Entry> = {
            let mut entries = self.0.entries();
            let _ = self.0.stopped_at_shutdown.set(live_defs(&entries));
            entries.drain().map(|(_, e)| e).collect()
        };
        for e in &entries {
            e.cancel.cancel();
        }
        let removals = std::mem::take(&mut *self.0.removals.lock().expect("watch removals mutex"));
        let tasks = entries.into_iter().map(|e| e.task).chain(removals);
        let join = futures::future::join_all(tasks);
        if tokio::time::timeout(SHUTDOWN_TIMEOUT, join).await.is_err() {
            warn!("watches did not stop in time");
        }
    }
}

impl Inner {
    fn dir(&self, id: Uuid) -> PathBuf {
        self.root.join(id.to_string())
    }

    fn entries(&self) -> MutexGuard<'_, HashMap<Uuid, Entry>> {
        self.entries.lock().expect("watch entries mutex")
    }

    /// The registered watches that are not being removed; after `shutdown`,
    /// the ones it stopped.
    fn defs(&self) -> Vec<WatchDef> {
        let entries = self.entries();
        match self.stopped_at_shutdown.get() {
            Some(defs) => defs.clone(),
            None => live_defs(&entries),
        }
    }

    /// Sent when a watch starts counting for its chat (created, admitted at
    /// boot) and when it is gone for good. A watch that ends with a notice is
    /// gone once the chat took the notice: the turn that starts keeps the chat
    /// busy, so its `agent_running` drops once, not twice.
    fn notify(&self, chat_id: &str) {
        let _ = self.changed.send(chat_id.to_string());
    }

    fn stop_note_path(&self, chat_id: &str) -> PathBuf {
        self.root.join(STOP_NOTES_DIR).join(chat_id)
    }

    /// `restored` = found on disk at boot: it waits for `admit_restored`.
    /// `false` = the watches are shut down, nothing was started.
    fn launch(self: &Arc<Self>, def: WatchDef, restored: bool) -> bool {
        // Held across the spawn so the task can't finish and look for its entry
        // before it is inserted.
        let mut entries = self.entries();
        if self.stopped_at_shutdown.get().is_some() {
            return false;
        }
        let cancel = CancellationToken::new();
        let task = tokio::spawn(
            self.clone()
                .supervise(def.clone(), cancel.clone(), restored),
        );
        entries.insert(def.id, Entry { def, cancel, task });
        true
    }

    async fn supervise(self: Arc<Self>, def: WatchDef, cancel: CancellationToken, restored: bool) {
        if restored {
            tokio::select! {
                // A refused watch is cancelled before the others are admitted.
                biased;
                () = cancel.cancelled() => return,
                () = self.restored_admitted.cancelled() => {}
            }
        }
        let outbox = Outbox::new(&self.dir(def.id));
        let command = async {
            if !def.ended {
                let Some(end) = self.run(&def, &cancel, &outbox).await else {
                    return;
                };
                let text = match end {
                    End::Exited(status) if status.success() => "ended: the command exited with code 0. The watch is removed.".to_string(),
                    End::Exited(status) => self.failure_notice(&def, &match status.code() {
                        Some(code) => format!("exited with code {code}"),
                        None => "was killed by a signal".to_string(),
                    }),
                    End::Failed(what) => self.failure_notice(&def, &what),
                    End::Expired => "ended: its TTL passed and the watch is removed. Create it again if it is still needed.".to_string(),
                    End::Flood => format!(
                        "stopped and removed: it produced more than {} messages in {}. Make the command print one line per real event before creating it again.",
                        self.limits.flood_max,
                        human_duration(self.limits.flood_window.as_secs()),
                    ),
                };
                self.end(&def, &outbox, &text);
            }
            outbox.close();
        };
        // One delivery loop for the whole life of the watch, so the command's
        // end can't drop a delivery the main loop is still performing.
        let ((), drained) = tokio::join!(command, self.deliver_outbox(&def, &cancel, &outbox));
        if drained {
            remove_dir(&self.dir(def.id));
            self.entries().remove(&def.id);
            self.notify(&def.chat_id);
            info!(id = %def.id, "watch ended");
        }
    }

    /// Queue the final notice and mark the watch as only draining its outbox.
    fn end(&self, def: &WatchDef, outbox: &Outbox, notice: &str) -> WatchDef {
        outbox.push(notice);
        let ended = WatchDef {
            ended: true,
            ..def.clone()
        };
        if let Err(e) = save_def(&self.dir(def.id), &ended) {
            warn!(id = %def.id, error = %e, "failed to persist ended watch");
        }
        if let Some(entry) = self.entries().get_mut(&def.id) {
            entry.def.ended = true;
        }
        ended
    }

    /// The end notice of a command that `what` (e.g. "exited with code 3"),
    /// with the tail of its stderr: the log itself goes with the directory.
    fn failure_notice(&self, def: &WatchDef, what: &str) -> String {
        let mut text = format!("ended: the command {what}. The watch is removed.");
        let stderr = stderr_tail(&self.dir(def.id).join("stderr.log"));
        if !stderr.is_empty() {
            text.push_str(&format!(" The end of its stderr:\n{stderr}"));
        }
        text
    }

    /// Run the command until the watch is over. `None` = cancelled.
    async fn run(
        &self,
        def: &WatchDef,
        cancel: &CancellationToken,
        outbox: &Outbox,
    ) -> Option<End> {
        if def.expired() {
            return Some(End::Expired);
        }
        let sent = &mut VecDeque::new();
        let dir = self.dir(def.id);
        let mut cmd = sys::login_shell_command(&def.cmd);
        cmd.current_dir(&def.cwd)
            .env("ZUCCHINI_WATCH_STATE_DIR", dir.join("state"))
            .env("ZUCCHINI_WATCH_ID", def.id.to_string())
            .env("ZUCCHINI_CHAT_ID", &def.chat_id)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(match stderr_log(&dir) {
                Ok(f) => Stdio::from(f),
                Err(_) => Stdio::null(),
            });
        sys::configure_new_process_group(&mut cmd);
        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => return Some(End::Failed(format!("could not be started ({e})"))),
        };
        let group = match sys::ProcessGroup::adopt(&child) {
            Ok(g) => g,
            Err(e) => {
                let _ = child.kill().await;
                return Some(End::Failed(format!("could not be started ({e})")));
            }
        };
        let pgid = child.id().unwrap_or_default();
        crate::orphans::record(pgid);
        let mut stdout = BufReader::new(child.stdout.take().expect("piped stdout"));

        let mut batch = Batch::default();
        let mut line = Vec::new();
        let mut flush_at: Option<tokio::time::Instant> = None;
        let mut reading = true;
        let mut exited: Option<End> = None;
        let mut drain_until: Option<tokio::time::Instant> = None;
        // The TTL is rechecked on the wall clock at least every tick: a timer
        // doesn't count the machine's sleep.
        let next_check = || tokio::time::Instant::now() + def.remaining().min(EXPIRY_TICK);
        let mut check_at = next_check();
        let stopped = loop {
            if !reading && exited.is_some() {
                break exited;
            }
            tokio::select! {
                () = cancel.cancelled() => break None,
                () = tokio::time::sleep_until(check_at) => {
                    if def.expired() {
                        break Some(End::Expired);
                    }
                    trim_stderr_log(&dir);
                    check_at = next_check();
                }
                () = sleep_until(flush_at) => {
                    flush_at = None;
                    if !self.flush(outbox, &mut batch, sent) {
                        break Some(End::Flood);
                    }
                }
                status = child.wait(), if exited.is_none() => {
                    exited = Some(match status {
                        Ok(status) => End::Exited(status),
                        Err(e) => End::Failed(format!("could not be waited for ({e})")),
                    });
                    drain_until = Some(tokio::time::Instant::now() + EXIT_DRAIN);
                }
                () = sleep_until(drain_until), if reading => reading = false,
                // Cancel-safe: a partial line stays in `line` for the next call.
                read = read_line_capped(&mut stdout, &mut line), if reading => {
                    if read != LineRead::Partial {
                        let full = batch.push(&line);
                        line.clear();
                        if full.is_some_and(|text| !self.queue(outbox, &text, sent)) {
                            break Some(End::Flood);
                        }
                        if !batch.is_empty() && flush_at.is_none() {
                            flush_at = Some(tokio::time::Instant::now() + self.limits.batch);
                        }
                    }
                    reading = read != LineRead::Eof;
                }
            }
        };
        // What was read last still goes out, whatever stopped the run.
        let outcome = match (self.flush(outbox, &mut batch, sent), stopped) {
            (false, Some(End::Exited(_) | End::Failed(_))) => Some(End::Flood),
            (_, stopped) => stopped,
        };
        group.terminate(&mut child, self.limits.sigterm_grace).await;
        // Also after a clean exit: what the command left running in its group
        // must not outlive the watch.
        if pgid > 1 {
            sys::stop_process_group(pgid, self.limits.sigterm_grace).await;
        }
        crate::orphans::forget(pgid);
        outcome
    }

    /// Queue the batch as one message. `false` = the watch is flooding.
    fn flush(&self, outbox: &Outbox, batch: &mut Batch, sent: &mut VecDeque<Instant>) -> bool {
        batch
            .take()
            .is_none_or(|text| self.queue(outbox, &text, sent))
    }

    /// Queue one message. `false` = the watch is flooding and it is dropped.
    fn queue(&self, outbox: &Outbox, text: &str, sent: &mut VecDeque<Instant>) -> bool {
        let now = Instant::now();
        while sent
            .front()
            .is_some_and(|t| now.duration_since(*t) > self.limits.flood_window)
        {
            sent.pop_front();
        }
        if sent.len() >= self.limits.flood_max {
            return false;
        }
        sent.push_back(now);
        outbox.push(text);
        true
    }

    /// Send the oldest queued messages as one. `Ok(false)` = nothing queued.
    async fn deliver_next(
        &self,
        def: &WatchDef,
        cancel: &CancellationToken,
        outbox: &Outbox,
    ) -> Result<bool> {
        let Some((text, files)) = outbox.next()? else {
            return Ok(false);
        };
        (self.deliver)(Outgoing {
            watch: def.clone(),
            text,
            cancelled: cancel.clone(),
        })
        .await?;
        outbox.acknowledge(files);
        Ok(true)
    }

    /// Deliver the outbox as it fills, until it is closed and empty.
    /// `false` = cancelled first.
    async fn deliver_outbox(
        &self,
        def: &WatchDef,
        cancel: &CancellationToken,
        outbox: &Outbox,
    ) -> bool {
        while !cancel.is_cancelled() {
            // Read before the outbox, which gets nothing after it is set.
            let closed = outbox.closed.load(Ordering::Relaxed);
            let delivered = tokio::select! {
                // An acknowledgment that already arrived is not lost to a cancel.
                biased;
                delivered = self.deliver_next(def, cancel, outbox) => delivered,
                () = cancel.cancelled() => return false,
            };
            match delivered {
                Ok(true) => {}
                Ok(false) if closed => return true,
                Ok(false) => tokio::select! {
                    () = cancel.cancelled() => return false,
                    () = outbox.changed.notified() => {}
                },
                Err(e) => {
                    warn!(id = %def.id, error = %e, "watch message not delivered, will retry");
                    tokio::select! {
                        () = cancel.cancelled() => return false,
                        () = tokio::time::sleep(self.limits.delivery_retry) => {}
                    }
                }
            }
        }
        false
    }
}

fn live_defs(entries: &HashMap<Uuid, Entry>) -> Vec<WatchDef> {
    let live = entries.values().filter(|e| !e.cancel.is_cancelled());
    live.map(|e| e.def.clone()).collect()
}

#[cfg(test)]
impl WatchDef {
    pub fn for_test(chat_id: &str, user_id: Uuid, description: &str) -> Self {
        Self {
            id: Uuid::now_v7(),
            chat_id: chat_id.to_string(),
            user_id,
            cmd: "true".into(),
            cwd: "/tmp".into(),
            description: description.to_string(),
            created_at: 1_800_000_000,
            expires_at: None,
            ended: false,
        }
    }
}

/// Register a watch that runs nothing, for the main loop's tests.
#[cfg(test)]
pub fn register_idle(def: WatchDef) {
    let watches = GLOBAL.get_or_init(|| {
        let root = std::env::temp_dir().join("zucchini-watch-none");
        let deliver: Deliver = Box::new(|_| Box::pin(async { bail!("no chat takes it") }));
        let watches = Watches::start(
            root,
            deliver,
            mpsc::unbounded_channel().0,
            Limits::default(),
        );
        // Else another test's first checkpoint would refuse these watches.
        watches.admit_restored(|_| true);
        watches
    });
    let entry = Entry {
        def,
        cancel: CancellationToken::new(),
        task: tokio::spawn(async {}),
    };
    watches.0.entries().insert(entry.def.id, entry);
}

/// A watch's `outbox/`: the messages its chat hasn't taken yet, one file each,
/// named by a UUIDv7 so they sort oldest first.
struct Outbox {
    dir: PathBuf,
    /// Notified when a message is queued or the outbox closes.
    changed: Notify,
    /// Nothing more will be queued.
    closed: AtomicBool,
    /// Delivered files that couldn't be deleted: never offered again, or the
    /// chat would get them in a loop.
    undeletable: Mutex<HashSet<PathBuf>>,
}

impl Outbox {
    fn new(watch_dir: &Path) -> Self {
        Self {
            dir: watch_dir.join("outbox"),
            changed: Notify::new(),
            closed: AtomicBool::new(false),
            undeletable: Mutex::default(),
        }
    }

    fn push(&self, text: &str) {
        let path = self.dir.join(Uuid::now_v7().to_string());
        if let Err(e) = crate::atomic::atomic_write_private(&path, text.as_bytes()) {
            warn!(path = %path.display(), error = %e, "failed to queue watch message");
        }
        self.changed.notify_one();
    }

    fn close(&self) {
        self.closed.store(true, Ordering::Relaxed);
        self.changed.notify_one();
    }

    /// The oldest messages joined into one of at most `2 * MAX_MESSAGE_BYTES`,
    /// with the files to [`acknowledge`](Self::acknowledge) once it is
    /// delivered. `None` = nothing queued.
    fn next(&self) -> std::io::Result<Option<(String, Vec<PathBuf>)>> {
        let entries = match std::fs::read_dir(&self.dir) {
            Ok(entries) => entries,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e),
        };
        let mut queued = Vec::new();
        {
            let undeletable = self.undeletable.lock().expect("watch outbox mutex");
            for entry in entries {
                let entry = entry?;
                let is_message = entry.file_name().to_string_lossy().parse::<Uuid>().is_ok();
                if is_message && !undeletable.contains(&entry.path()) {
                    queued.push(entry.path());
                }
            }
        }
        queued.sort();
        let mut text = String::new();
        let mut files = Vec::new();
        for path in queued {
            let body = std::fs::read_to_string(&path)?;
            if !files.is_empty() {
                if text.len() + 1 + body.len() > 2 * MAX_MESSAGE_BYTES {
                    break;
                }
                text.push('\n');
            }
            text.push_str(&body);
            files.push(path);
        }
        Ok((!files.is_empty()).then_some((text, files)))
    }

    fn acknowledge(&self, files: Vec<PathBuf>) {
        for path in files {
            if let Err(e) = std::fs::remove_file(&path) {
                warn!(path = %path.display(), error = %e, "failed to delete a delivered watch message");
                self.undeletable
                    .lock()
                    .expect("watch outbox mutex")
                    .insert(path);
            }
        }
    }
}

/// A fresh `stderr.log` for a run, in append mode so that [`trim_stderr_log`]
/// can empty it under the command.
fn stderr_log(dir: &Path) -> std::io::Result<std::fs::File> {
    let path = dir.join("stderr.log");
    std::fs::File::create(&path)?;
    std::fs::File::options().append(true).open(path)
}

/// Only the end of a run's stderr is ever read, so a log past the cap starts
/// over rather than fill the disk under a long-lived watch.
fn trim_stderr_log(dir: &Path) {
    let path = dir.join("stderr.log");
    if std::fs::metadata(&path).is_ok_and(|m| m.len() > MAX_STDERR_LOG_BYTES) {
        if let Ok(file) = std::fs::File::options().write(true).open(&path) {
            let _ = file.set_len(0);
        }
    }
}

/// The last `STDERR_TAIL_BYTES` of a run's stderr, or nothing.
fn stderr_tail(path: &Path) -> String {
    let Ok(mut file) = std::fs::File::open(path) else {
        return String::new();
    };
    let len = file.metadata().map_or(0, |m| m.len());
    let mut tail = Vec::new();
    if file
        .seek(SeekFrom::Start(len.saturating_sub(STDERR_TAIL_BYTES)))
        .is_ok()
    {
        let _ = file.take(STDERR_TAIL_BYTES).read_to_end(&mut tail);
    }
    String::from_utf8_lossy(&tail).trim().to_string()
}

/// Lines waiting to go out as one message of at most `MAX_MESSAGE_BYTES`.
#[derive(Default)]
struct Batch {
    text: String,
    truncated: bool,
}

impl Batch {
    fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    /// Add a line. One the batch has no room for starts the next batch, and
    /// the full one is returned to be sent now.
    fn push(&mut self, raw: &[u8]) -> Option<String> {
        let line = String::from_utf8_lossy(raw);
        let line = line.trim_end_matches(['\n', '\r']);
        if line.trim().is_empty() {
            return None;
        }
        let full = if self.text.len() + 1 + line.len() > MAX_MESSAGE_BYTES {
            self.take()
        } else {
            None
        };
        // Replacing invalid UTF-8 can grow a line past the cap it was read with.
        let mut cut = line.len().min(MAX_MESSAGE_BYTES);
        while !line.is_char_boundary(cut) {
            cut -= 1;
        }
        // `read_line_capped` stops a line at the cap and discards the rest.
        self.truncated |= cut < line.len() || raw.len() >= MAX_MESSAGE_BYTES;
        if !self.text.is_empty() {
            self.text.push('\n');
        }
        self.text.push_str(&line[..cut]);
        full
    }

    fn take(&mut self) -> Option<String> {
        if self.text.is_empty() {
            return None;
        }
        let mut text = std::mem::take(&mut self.text);
        if std::mem::take(&mut self.truncated) {
            text.push_str("\n[output truncated]");
        }
        Some(text)
    }
}

#[derive(Debug, PartialEq)]
enum LineRead {
    Partial,
    Complete,
    Eof,
}

/// One read's worth of the current line into `line`, without its newline.
/// `line` stops growing at `MAX_MESSAGE_BYTES` and the rest of the line is
/// discarded; `Batch::push` marks a line that long as truncated.
/// Cancel-safe: nothing is consumed before the only await returns.
async fn read_line_capped(
    reader: &mut (impl AsyncBufRead + Unpin),
    line: &mut Vec<u8>,
) -> LineRead {
    let chunk = reader.fill_buf().await.unwrap_or_default();
    if chunk.is_empty() {
        return LineRead::Eof;
    }
    let newline = chunk.iter().position(|&b| b == b'\n');
    let content = &chunk[..newline.unwrap_or(chunk.len())];
    let room = MAX_MESSAGE_BYTES - line.len();
    line.extend_from_slice(&content[..content.len().min(room)]);
    let used = newline.map_or(chunk.len(), |at| at + 1);
    reader.consume(used);
    match newline {
        Some(_) => LineRead::Complete,
        None => LineRead::Partial,
    }
}

async fn sleep_until(at: Option<tokio::time::Instant>) {
    match at {
        Some(at) => tokio::time::sleep_until(at).await,
        None => std::future::pending().await,
    }
}

fn load_def(dir: &Path) -> Result<WatchDef> {
    let raw = std::fs::read(dir.join("watch.json"))?;
    Ok(serde_json::from_slice(&raw)?)
}

fn save_def(dir: &Path, def: &WatchDef) -> Result<()> {
    let json = serde_json::to_vec_pretty(def)?;
    crate::atomic::atomic_write_private(&dir.join("watch.json"), &json)
        .with_context(|| format!("write watch definition in {}", dir.display()))
}

fn remove_dir(dir: &Path) {
    if let Err(e) = std::fs::remove_dir_all(dir) {
        if e.kind() != std::io::ErrorKind::NotFound {
            warn!(dir = %dir.display(), error = %e, "failed to remove watch directory");
        }
    }
}

fn unix_now() -> i64 {
    chrono::Utc::now().timestamp()
}

/// `90s`, `30m`, `12h`, `7d`, `2w`, or `forever` (`None`).
pub fn parse_ttl(s: &str) -> Result<Option<Duration>> {
    let s = s.trim();
    if s == "forever" {
        return Ok(None);
    }
    let bad = || anyhow!("invalid --ttl {s:?}: use e.g. 30m, 12h, 7d, 2w or forever");
    let unit = s.chars().last().ok_or_else(bad)?;
    let per_unit = match unit {
        's' => 1,
        'm' => 60,
        'h' => 3600,
        'd' => 24 * 3600,
        'w' => 7 * 24 * 3600,
        _ => return Err(bad()),
    };
    let n: u64 = s[..s.len() - 1].parse().map_err(|_| bad())?;
    let secs = n
        .checked_mul(per_unit)
        .filter(|&secs| secs > 0)
        .ok_or_else(bad)?;
    Ok(Some(Duration::from_secs(secs)))
}

fn human_duration(secs: u64) -> String {
    match secs {
        s if s >= 2 * 24 * 3600 => format!("{}d", s / (24 * 3600)),
        s if s >= 2 * 3600 => format!("{}h", s / 3600),
        s if s >= 120 => format!("{}m", s / 60),
        s => format!("{s}s"),
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    type Sent = Arc<Mutex<Vec<String>>>;

    fn limits() -> Limits {
        Limits {
            batch: Duration::from_millis(100),
            flood_max: 3,
            flood_window: Duration::from_secs(600),
            delivery_retry: Duration::from_millis(50),
            sigterm_grace: Duration::from_millis(200),
        }
    }

    fn fresh_root(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("zucchini-watch-{name}-{}", Uuid::now_v7()));
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    /// Watches over a fresh root whose first `failing` deliveries error.
    fn watches(name: &str, failing: usize, limits: Limits) -> (Watches, Sent, PathBuf) {
        let root = fresh_root(name);
        let (deliver, sent) = recorder(failing);
        let watches = Watches::start(root.clone(), deliver, unheard(), limits);
        (watches, sent, root)
    }

    fn unheard() -> Changed {
        mpsc::unbounded_channel().0
    }

    async fn until(what: &str, cond: impl Fn() -> bool) {
        for _ in 0..400 {
            if cond() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        panic!("timed out waiting until {what}");
    }

    fn recorder(failing: usize) -> (Deliver, Sent) {
        let sent: Sent = Arc::default();
        let log = sent.clone();
        let left = Arc::new(Mutex::new(failing));
        let deliver: Deliver = Box::new(move |out| {
            let (log, left) = (log.clone(), left.clone());
            Box::pin(async move {
                let mut left = left.lock().unwrap();
                if *left > 0 {
                    *left -= 1;
                    bail!("backend unreachable");
                }
                log.lock().unwrap().push(out.text);
                Ok(())
            })
        });
        (deliver, sent)
    }

    fn create(w: &Watches, cmd: &str) -> WatchDef {
        let description = format!("test {}", w.list().len());
        w.create("chat", Uuid::nil(), cmd, "/tmp", "forever", &description)
            .unwrap()
    }

    #[tokio::test]
    async fn a_description_names_one_watch_per_chat() {
        let (w, _sent, _root) = watches("names", 0, limits());
        let make = |chat| w.create(chat, Uuid::nil(), "sleep 30", "/tmp", "1h", " mail ");
        let first = make("chat").unwrap();
        assert!(make("chat").is_err());
        make("other").unwrap();
        assert_eq!(w.find("chat", "mail").map(|d| d.id), Some(first.id));
        w.shutdown().await;
    }

    async fn until_gone(w: &Watches) {
        until_gone_but(w, 0).await;
    }

    #[tokio::test]
    async fn lines_are_batched_and_exit_0_ends_the_watch() {
        let (w, sent, root) = watches("batch", 0, limits());
        let def = create(&w, "echo one; echo two; sleep 0.5; echo three");
        until_gone(&w).await;

        // Messages queued while one is being sent go out together.
        let sent = sent.lock().unwrap().join("\n");
        assert_eq!(
            sent,
            "one\ntwo\nthree\nended: the command exited with code 0. The watch is removed."
        );
        assert!(!root.join(def.id.to_string()).exists());
    }

    #[tokio::test]
    async fn a_non_zero_exit_ends_the_watch_with_its_stderr() {
        let (w, sent, root) = watches("failed", 0, limits());
        let runs = root.join("runs");
        let def = create(
            &w,
            &format!("echo run >> {}; echo boom >&2; exit 3", runs.display()),
        );
        until_gone(&w).await;
        tokio::time::sleep(Duration::from_millis(300)).await;

        assert_eq!(
            sent.lock().unwrap().join("\n"),
            "ended: the command exited with code 3. The watch is removed. The end of its stderr:\nboom"
        );
        assert_eq!(
            std::fs::read_to_string(runs).unwrap(),
            "run\n",
            "no restart"
        );
        assert!(!root.join(def.id.to_string()).exists());
    }

    #[tokio::test]
    async fn a_command_killed_by_a_signal_ends_the_watch() {
        let (w, sent, _root) = watches("signal", 0, limits());
        create(&w, "kill -KILL $$");
        until_gone(&w).await;

        let sent = sent.lock().unwrap().join("\n");
        assert!(
            sent.starts_with("ended: the command was killed by a signal. The watch is removed."),
            "got: {sent}"
        );
    }

    #[tokio::test]
    async fn a_command_that_cannot_be_started_ends_the_watch() {
        let (w, _sent, root) = watches("unstartable", 0, limits());
        let cwd = fresh_root("unstartable-cwd");
        w.create(
            "chat",
            Uuid::nil(),
            "sleep 30",
            cwd.to_str().unwrap(),
            "1h",
            "test",
        )
        .unwrap();
        w.shutdown().await;
        std::fs::remove_dir(&cwd).unwrap();

        let (deliver, sent) = recorder(0);
        let w = Watches::start(root, deliver, unheard(), limits());
        w.admit_restored(|_| true);
        until_gone(&w).await;
        let sent = sent.lock().unwrap().join("\n");
        assert!(
            sent.starts_with("ended: the command could not be started (")
                && sent.ends_with("). The watch is removed."),
            "got: {sent}"
        );
    }

    #[tokio::test]
    async fn a_blank_description_or_a_bad_ttl_is_refused() {
        let (w, _sent, root) = watches("required", 0, limits());
        for (ttl, description) in [("1h", " "), ("", "test"), ("soon", "test")] {
            let refused = w.create("chat", Uuid::nil(), "sleep 30", "/tmp", ttl, description);
            assert!(refused.is_err(), "{ttl:?} {description:?} must be refused");
        }
        assert_eq!(std::fs::read_dir(&root).unwrap().count(), 0);
    }

    #[tokio::test]
    async fn a_flooding_watch_is_stopped_with_a_notice() {
        let (w, sent, _root) = watches("flood", 0, limits());
        create(&w, "while :; do echo tick; sleep 0.15; done");
        until_gone(&w).await;

        let sent = sent.lock().unwrap().join("\n");
        assert_eq!(sent.matches("tick").count(), 3, "got: {sent}");
        assert!(
            sent.contains("produced more than 3 messages"),
            "got: {sent}"
        );
    }

    #[tokio::test]
    async fn undelivered_messages_wait_on_disk_and_survive_a_restart() {
        let (w, sent, root) = watches("offline", usize::MAX, limits());
        let def = create(&w, "echo event; sleep 30");
        let outbox = root.join(def.id.to_string()).join("outbox");
        until("the message is queued", || {
            std::fs::read_dir(&outbox).unwrap().count() > 0
        })
        .await;
        w.shutdown().await;
        assert!(sent.lock().unwrap().is_empty());
        assert!(w.list().is_empty());

        let (deliver, sent) = recorder(0);
        let w = Watches::start(root, deliver, unheard(), limits());
        assert_eq!(w.list().len(), 1, "restored from disk");
        w.admit_restored(|_| true);
        until("the message is delivered", || {
            !sent.lock().unwrap().is_empty()
        })
        .await;
        assert_eq!(sent.lock().unwrap()[0], "event");

        w.remove(def.id, Some("removed with a notice.")).await;
        until_gone(&w).await;
        assert!(sent
            .lock()
            .unwrap()
            .last()
            .unwrap()
            .contains("removed with a notice."));
    }

    #[tokio::test]
    async fn an_expired_watch_ends_with_a_notice() {
        let (w, sent, root) = watches("ttl", 0, limits());
        let def = create(&w, "sleep 30");
        w.shutdown().await;
        let dir = root.join(def.id.to_string());
        save_def(
            &dir,
            &WatchDef {
                expires_at: Some(unix_now() - 1),
                ..def
            },
        )
        .unwrap();

        let (deliver, sent2) = recorder(0);
        let w = Watches::start(root, deliver, unheard(), limits());
        w.admit_restored(|_| true);
        until_gone(&w).await;
        assert!(sent.lock().unwrap().is_empty());
        assert!(sent2.lock().unwrap()[0].contains("its TTL passed"));
    }

    #[tokio::test]
    async fn a_restored_watch_runs_only_once_admitted() {
        let (w, _sent, root) = watches("admit", 0, limits());
        let ran = |chat: &str| root.join(format!("ran-{chat}"));
        for chat in ["kept", "refused"] {
            let cmd = format!("touch {}; sleep 30", ran(chat).display());
            w.create(chat, Uuid::nil(), &cmd, "/tmp", "forever", "test")
                .unwrap();
        }
        until("both ran", || {
            ran("kept").exists() && ran("refused").exists()
        })
        .await;
        w.shutdown().await;
        for chat in ["kept", "refused"] {
            std::fs::remove_file(ran(chat)).unwrap();
        }

        let w = Watches::start(root.clone(), recorder(0).0, unheard(), limits());
        tokio::time::sleep(Duration::from_millis(500)).await;
        assert_eq!(w.list().len(), 2);
        assert!(!ran("kept").exists(), "nothing runs before admission");

        w.admit_restored(|d| d.chat_id == "kept");
        until("the kept watch ran", || ran("kept").exists()).await;
        until("the refused watch is removed", || w.list().len() == 1).await;
        assert_eq!(w.list()[0].chat_id, "kept");
        assert!(!ran("refused").exists());
        assert_eq!(std::fs::read_dir(&root).unwrap().count(), 2);
        w.shutdown().await;
    }

    #[tokio::test]
    async fn a_delivery_in_flight_when_the_command_exits_is_not_repeated() {
        let sent: Sent = Arc::default();
        let log = sent.clone();
        let deliver: Deliver = Box::new(move |out| {
            let log = log.clone();
            // Like the main loop: performed whether or not the watch still waits.
            let performed = tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(800)).await;
                log.lock().unwrap().push(out.text);
            });
            Box::pin(async move { Ok(performed.await?) })
        });
        let w = Watches::start(fresh_root("in-flight"), deliver, unheard(), limits());
        create(&w, "echo one; sleep 0.4");
        until_gone(&w).await;

        let sent = sent.lock().unwrap();
        assert_eq!(sent.len(), 2, "got: {sent:?}");
        assert_eq!(sent[0], "one");
        assert!(sent[1].contains("exited with code 0"));
    }

    #[tokio::test]
    async fn what_the_command_leaves_in_its_group_is_killed() {
        let (w, sent, _root) = watches("leftover", 0, limits());
        create(
            &w,
            "(trap '' TERM; exec sleep 30) >/dev/null 2>&1 & echo $!",
        );
        until_gone(&w).await;

        let sent = sent.lock().unwrap().join("\n");
        let pid: u32 = sent.lines().next().unwrap().parse().unwrap();
        until("the leftover is gone", || !sys::process_exists(pid)).await;
    }

    #[tokio::test]
    async fn a_line_without_a_newline_is_capped() {
        let mut out = vec![b'x'; 5 * MAX_MESSAGE_BYTES];
        out.extend_from_slice(b"\nnext\n");
        let mut reader = BufReader::with_capacity(1024, out.as_slice());
        let mut line = Vec::new();
        let mut partial_reads = 0;
        while read_line_capped(&mut reader, &mut line).await == LineRead::Partial {
            partial_reads += 1;
            assert!(line.len() <= MAX_MESSAGE_BYTES);
        }
        assert!(partial_reads > 1);
        assert_eq!(line.len(), MAX_MESSAGE_BYTES);
        let mut batch = Batch::default();
        batch.push(&line);
        assert!(batch.take().unwrap().ends_with("\n[output truncated]"));

        line.clear();
        assert_eq!(
            read_line_capped(&mut reader, &mut line).await,
            LineRead::Complete
        );
        assert_eq!(line, b"next");
        assert_eq!(
            read_line_capped(&mut reader, &mut line).await,
            LineRead::Eof
        );
    }

    #[test]
    fn the_outbox_joins_messages_and_never_repeats_a_delivered_one() {
        use std::os::unix::fs::PermissionsExt;
        let root = fresh_root("outbox");
        let dir = root.join("outbox");
        std::fs::create_dir(&dir).unwrap();
        let outbox = Outbox::new(&root);
        assert!(outbox.next().unwrap().is_none());
        outbox.push("one");
        outbox.push("two");
        let (text, files) = outbox.next().unwrap().unwrap();
        assert_eq!(text, "one\ntwo");

        let set_mode = |mode| std::fs::set_permissions(&dir, PermissionsExt::from_mode(mode));
        set_mode(0o500).unwrap();
        outbox.acknowledge(files);
        assert!(outbox.next().unwrap().is_none(), "undeletable, not resent");
        set_mode(0o700).unwrap();

        outbox.push("three");
        let (text, files) = outbox.next().unwrap().unwrap();
        assert_eq!(text, "three");
        outbox.acknowledge(files);
        assert!(outbox.next().unwrap().is_none());
    }

    #[test]
    fn ttl_parsing() {
        assert_eq!(parse_ttl("30m").unwrap(), Some(Duration::from_secs(1800)));
        assert_eq!(
            parse_ttl("2w").unwrap(),
            Some(Duration::from_secs(14 * 86400))
        );
        assert_eq!(parse_ttl("forever").unwrap(), None);
        for bad in ["", "0h", "h", "3x", "1.5h", "-1d"] {
            assert!(parse_ttl(bad).is_err(), "{bad:?} must be rejected");
        }
    }

    #[test]
    fn a_full_batch_is_sent_whole_and_only_an_oversized_line_is_cut() {
        let mut batch = Batch::default();
        assert!(batch.push(b"  \n").is_none());
        assert!(batch.is_empty());

        let half = "x".repeat(MAX_MESSAGE_BYTES / 2);
        assert!(batch.push(half.as_bytes()).is_none());
        assert!(batch.push(b"fits\r\n").is_none());
        let full = batch.push(half.as_bytes()).unwrap();
        assert_eq!(full, format!("{half}\nfits"));

        let oversized = "é".repeat(MAX_MESSAGE_BYTES);
        assert_eq!(batch.push(oversized.as_bytes()), Some(half));
        let cut = batch.push(b"kept").unwrap();
        assert_eq!(cut.len(), MAX_MESSAGE_BYTES + "\n[output truncated]".len());
        assert!(cut.ends_with("\n[output truncated]"));
        assert_eq!(batch.take().unwrap(), "kept");
    }

    #[tokio::test]
    async fn a_burst_over_the_message_cap_goes_out_as_several_messages() {
        let (w, sent, _root) = watches("burst", 0, limits());
        create(
            &w,
            "for i in 1 2 3; do head -c 5000 /dev/zero | tr '\\0' Q; echo; done",
        );
        until_gone(&w).await;

        let sent = sent.lock().unwrap().join("\n");
        assert_eq!(sent.matches('Q').count(), 15000);
        assert!(!sent.contains("truncated"), "got: {sent}");
    }

    #[tokio::test]
    async fn nothing_is_created_after_shutdown() {
        let (w, _sent, root) = watches("stopped", 0, limits());
        w.shutdown().await;
        let refused = w.create("chat", Uuid::nil(), "sleep 30", "/tmp", "1h", "test");
        assert!(refused.is_err());
        assert_eq!(std::fs::read_dir(&root).unwrap().count(), 0);
    }

    #[tokio::test]
    async fn a_chats_running_watches_and_every_change_of_them_are_reported() {
        let (changed, changes) = mpsc::unbounded_channel();
        let w = Watches::start(fresh_root("changed"), recorder(0).0, changed, limits());
        let changes = std::cell::RefCell::new((changes, 0));
        let reported = || {
            let (changes, seen) = &mut *changes.borrow_mut();
            *seen += std::iter::from_fn(|| changes.try_recv().ok()).count();
            *seen
        };
        let running =
            |w: &Watches| -> Vec<Uuid> { w.running_in("chat").iter().map(|d| d.id).collect() };

        let kept = create(&w, "sleep 30");
        let ending = create(&w, "sleep 0.3");
        assert_eq!(running(&w), [kept.id, ending.id], "oldest first");
        assert!(w.running_in("another chat").is_empty());
        assert_eq!(reported(), 2);

        until("the command is over", || running(&w) == [kept.id]).await;
        until("its end is reported", || reported() == 3).await;

        let stopped = create(&w, "sleep 30");
        w.remove_in_background(|d| d.id == kept.id);
        assert_eq!(running(&w), [stopped.id], "not while it is being removed");
        until("the removal is reported", || reported() == 5).await;

        w.shutdown().await;
        assert_eq!(running(&w), [stopped.id], "the next start runs it again");
        assert_eq!(reported(), 5);
    }

    #[tokio::test]
    async fn a_stop_removes_the_chats_watches_and_keeps_a_note_for_its_agent() {
        let (w, sent, root) = watches("stop", 0, limits());
        create(&w, "sleep 30");
        assert!(w.take_stop_note("chat").is_none());

        assert!(w.chat_stopped("chat"));
        assert!(w.running_in("chat").is_empty());
        assert!(!w.chat_stopped("chat"), "nothing left to stop");
        // A shutdown right behind the stop still finishes the removal.
        w.shutdown().await;
        assert!(sent.lock().unwrap().is_empty(), "no notice");

        let restarted = Watches::start(root, recorder(0).0, unheard(), limits());
        assert!(restarted.list().is_empty(), "the notes are no watch");
        assert_eq!(
            restarted.take_stop_note("chat").unwrap(),
            "[zucchini] The user pressed Stop, which also removed this chat's watches: \
             \"test 0\" (sleep 30). They are not running any more."
        );
        assert!(restarted.take_stop_note("chat").is_none(), "told once");
    }

    async fn until_gone_but(w: &Watches, left: usize) {
        until("the watch is gone", || w.list().len() == left).await;
    }

    #[test]
    fn the_outbox_leaves_what_does_not_fit_one_message_queued() {
        let root = fresh_root("outbox-cap");
        std::fs::create_dir(root.join("outbox")).unwrap();
        let outbox = Outbox::new(&root);
        let body = "x".repeat(MAX_MESSAGE_BYTES - 1);
        for _ in 0..3 {
            outbox.push(&body);
        }
        let (text, files) = outbox.next().unwrap().unwrap();
        assert_eq!(text, format!("{body}\n{body}"));
        outbox.acknowledge(files);
        let (text, files) = outbox.next().unwrap().unwrap();
        assert_eq!(text, body);
        outbox.acknowledge(files);
        assert!(outbox.next().unwrap().is_none());
    }
}
