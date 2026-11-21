//! Self-update loop: polls a remote (or file://) version file, downloads the
//! matching binary next to the running one when the version changes, renames
//! it over the running binary, and lets launchd/systemd respawn with the new
//! version.
//!
//! Dev loop: set `SPAWNER_UPDATE_BASE_URL=file:///.../.zucchini-spawner/dev-update`
//! (written by `dev-watch.sh`) + `SPAWNER_UPDATE_CHECK_INTERVAL_SECS=2`.
//! Prod: defaults to `<PROD_BASE_URL>/install/`.
//!
//! CARGO_PKG_VERSION doesn't move on dev rebuilds, so we persist the last
//! applied version under `~/.zucchini-spawner/.applied-version` — otherwise the
//! loop would re-apply the same dev tag on every boot.

use std::path::Path;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tracing::{info, trace, warn};

const CURRENT_VERSION: &str = env!("CARGO_PKG_VERSION");
// deploy.sh greps every release binary for this and refuses to publish one that
// lacks it, so keep it referenced from runtime code or the linker drops it.
const RELEASE_MARKER: &str = concat!("zucchini-spawner-release:", env!("CARGO_PKG_VERSION"), ";");
const DEFAULT_CHECK_INTERVAL: Duration = Duration::from_secs(10 * 60);
const RETRY_WHEN_BUSY: Duration = Duration::from_secs(60);
const MAX_FAILED_UPDATE_WAIT: Duration = Duration::from_secs(6 * 60 * 60);
// The main loop waits on the download, so a stalled one must not hold it.
const DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(5 * 60);
const PROD_INSTALL_BASE: &str = "https://api.zucchini.chat/install";
const VERSION_FILENAME: &str = "zucchini-spawner-version.txt";

fn applied_version_path() -> std::path::PathBuf {
    crate::zucchini_spawner_dir().join(".applied-version")
}

fn last_applied_version() -> String {
    std::fs::read_to_string(applied_version_path())
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| CURRENT_VERSION.to_string())
}

fn base_url() -> String {
    std::env::var("SPAWNER_UPDATE_BASE_URL")
        .ok()
        .map(|s| s.trim_end_matches('/').to_string())
        .unwrap_or_else(|| PROD_INSTALL_BASE.to_string())
}

pub fn check_interval() -> Duration {
    std::env::var("SPAWNER_UPDATE_CHECK_INTERVAL_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .map(Duration::from_secs)
        .unwrap_or(DEFAULT_CHECK_INTERVAL)
}

fn platform_suffix() -> Result<String, String> {
    let (os, arch) = (std::env::consts::OS, std::env::consts::ARCH);
    let platform_os = match os {
        "macos" => "darwin",
        "linux" => "linux",
        "windows" => "windows",
        other => return Err(format!("unsupported os: {other}")),
    };
    // install.sh names darwin arm64 as "arm64"; linux keeps "aarch64".
    let platform_arch = match (platform_os, arch) {
        ("darwin", "aarch64") => "arm64",
        (_, "aarch64") => "aarch64",
        (_, "x86_64") => "x86_64",
        (_, other) => return Err(format!("unsupported arch: {other}")),
    };
    Ok(format!("{platform_os}-{platform_arch}"))
}

/// Fetch a URL's body bytes. Handles both `https://` (via the `reqwest` client
/// we already link — no external `curl` needed, so no `curl.exe`-on-PATH
/// dependency on Windows) and `file://` (the dev hot-reload loop; reqwest
/// doesn't speak file URLs). Non-2xx is an error, so a failed fetch never
/// surfaces an empty/garbage body that would look like a valid version change.
async fn fetch_bytes(url: &str) -> Result<Vec<u8>, String> {
    // `file://` is the author's dev-update loop only (macOS); strip the scheme
    // to a plain path. Prod always takes the reqwest branch below.
    if let Some(path) = url.strip_prefix("file://") {
        return tokio::fs::read(path)
            .await
            .map_err(|e| format!("read {path}: {e}"));
    }
    let resp = reqwest::get(url)
        .await
        .map_err(|e| format!("request {url} failed: {e}"))?;
    let status = resp.status();
    if !status.is_success() {
        return Err(format!("fetch {url} returned HTTP {status}"));
    }
    resp.bytes()
        .await
        .map(|b| b.to_vec())
        .map_err(|e| format!("read body of {url}: {e}"))
}

async fn check_version() -> Option<String> {
    let url = format!("{}/{}", base_url(), VERSION_FILENAME);
    let bytes = fetch_bytes(&url).await.ok()?;
    let version = String::from_utf8(bytes).ok()?.trim().to_string();
    if version.is_empty() {
        return None;
    }
    Some(version)
}

/// A new binary downloaded beside the running one. Dropping it deletes the
/// file, which is already gone once [`install`] moved it into place.
pub struct Staged {
    path: std::path::PathBuf,
    current_exe: std::path::PathBuf,
    version: String,
}

impl Drop for Staged {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

pub async fn download(remote_version: &str) -> Result<Staged, String> {
    let suffix = platform_suffix()?;
    let binary_name = format!("zucchini-spawner-{suffix}");
    let url = format!("{}/{binary_name}", base_url());
    let current_exe =
        std::env::current_exe().map_err(|e| format!("cannot find current exe: {e}"))?;

    let fetch = async {
        // A build published after the check replaced the binary too, so persist
        // the version as of now. Read before the binary: publishers write the
        // binary first, so this can lag what we install but never get ahead of it.
        let version = check_version()
            .await
            .unwrap_or_else(|| remote_version.to_string());
        info!(url = %url, %version, "downloading new spawner binary");
        fetch_bytes(&url).await.map(|bytes| (version, bytes))
    };
    let (version, bytes) = tokio::time::timeout(DOWNLOAD_TIMEOUT, fetch)
        .await
        .map_err(|_| format!("download of {url} timed out"))??;

    let staged = Staged {
        // Beside the exe: a rename across filesystems (a tmpfs /tmp) fails.
        path: current_exe.with_extension("new"),
        current_exe,
        version,
    };
    write_executable(&staged.path, &bytes).await?;
    Ok(staged)
}

/// Replaces the running binary. Unix renames the inode in place; Windows can't
/// overwrite a live .exe, so the windows seam renames the live exe aside to
/// `<exe>.old` first (swept on next start). Either way the supervisor (launchd /
/// systemd / the logon Scheduled Task) respawns from the new file once we exit.
pub fn install(staged: Staged) -> Result<(), String> {
    crate::sys::replace_running_exe(&staged.path, &staged.current_exe)
        .map_err(|e| format!("replace running exe failed: {e}"))?;

    if let Err(e) = std::fs::write(applied_version_path(), &staged.version) {
        warn!("failed to persist applied version: {e}");
    }

    Ok(())
}

async fn write_executable(path: &Path, bytes: &[u8]) -> Result<(), String> {
    tokio::fs::write(path, bytes)
        .await
        .map_err(|e| format!("write {}: {e}", path.display()))?;

    // Post-download prep is unix-only: Windows binaries need no execute bit and
    // carry no `com.apple.quarantine` xattr.
    #[cfg(unix)]
    {
        let status = tokio::process::Command::new("chmod")
            .arg("+x")
            .arg(path)
            .status()
            .await
            .map_err(|e| format!("chmod failed: {e}"))?;
        if !status.success() {
            return Err("chmod failed".to_string());
        }

        #[cfg(target_os = "macos")]
        if let Err(e) = tokio::process::Command::new("xattr")
            .arg("-c")
            .arg(path)
            .status()
            .await
        {
            warn!("xattr clear failed: {e}");
        }
    }

    Ok(())
}

/// The version waiting to be installed, and when a failed install of it may be
/// tried again: the wait doubles after each failure, up to 6 h.
pub struct UpdateRetry {
    pending: Option<String>,
    first_wait: Duration,
    wait: Duration,
    not_before: Option<Instant>,
}

impl UpdateRetry {
    pub fn new(first_wait: Duration) -> Self {
        let first_wait = first_wait.min(MAX_FAILED_UPDATE_WAIT);
        UpdateRetry {
            pending: None,
            first_wait,
            wait: first_wait,
            not_before: None,
        }
    }

    /// Due at once with the wait reset.
    pub fn offer(&mut self, version: String) {
        self.pending = Some(version);
        self.wait = self.first_wait;
        self.not_before = None;
    }

    pub fn due(&self, now: Instant) -> Option<&str> {
        self.pending
            .as_deref()
            .filter(|_| self.not_before.is_none_or(|t| now >= t))
    }

    /// Returns how long until the next try.
    pub fn failed(&mut self, now: Instant) -> Duration {
        let wait = self.wait;
        self.not_before = Some(now + wait);
        self.wait = (wait * 2).min(MAX_FAILED_UPDATE_WAIT);
        wait
    }
}

pub async fn run_update_loop(update_tx: mpsc::Sender<String>) -> ! {
    let base = base_url();
    let interval = check_interval();
    let mut last = last_applied_version();
    info!(
        embedded = CURRENT_VERSION,
        release_marker = RELEASE_MARKER,
        last_applied = %last,
        base = %base,
        interval_secs = interval.as_secs(),
        "update checker started"
    );

    loop {
        tokio::time::sleep(interval).await;

        match check_version().await {
            Some(remote) => {
                if remote != last {
                    info!(current = %last, remote = %remote, "version changed, requesting update");
                    // Announced once: the main loop retries a failed update itself.
                    let _ = update_tx.send(remote.clone()).await;
                    last = remote;
                } else {
                    trace!(current = %last, "version is up to date");
                }
            }
            None => {
                warn!(?RETRY_WHEN_BUSY, "failed to check for updates, retrying");
                tokio::time::sleep(RETRY_WHEN_BUSY).await;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIRST: Duration = Duration::from_secs(10 * 60);
    const MINUTE: Duration = Duration::from_secs(60);

    #[test]
    fn nothing_pending_is_never_due() {
        assert_eq!(UpdateRetry::new(FIRST).due(Instant::now()), None);
    }

    #[test]
    fn a_fresh_offer_is_due_at_once() {
        let mut retry = UpdateRetry::new(FIRST);
        retry.offer("4.0.1".into());
        assert_eq!(retry.due(Instant::now()), Some("4.0.1"));
    }

    #[test]
    fn a_failed_update_waits_before_the_next_try() {
        let t0 = Instant::now();
        let mut retry = UpdateRetry::new(FIRST);
        retry.offer("4.0.1".into());
        assert_eq!(retry.failed(t0), FIRST);
        assert_eq!(retry.due(t0 + 9 * MINUTE), None);
        assert_eq!(retry.due(t0 + FIRST), Some("4.0.1"));
    }

    #[test]
    fn the_wait_doubles_up_to_six_hours() {
        let t0 = Instant::now();
        let mut retry = UpdateRetry::new(FIRST);
        retry.offer("4.0.1".into());
        let minutes: Vec<u64> = (0..8).map(|_| retry.failed(t0).as_secs() / 60).collect();
        assert_eq!(minutes, [10, 20, 40, 80, 160, 320, 360, 360]);
    }

    #[test]
    fn a_different_version_resets_the_wait_and_is_due_at_once() {
        let t0 = Instant::now();
        let mut retry = UpdateRetry::new(FIRST);
        retry.offer("4.0.1".into());
        for _ in 0..3 {
            retry.failed(t0);
        }
        retry.offer("4.0.2".into());
        assert_eq!(retry.due(t0), Some("4.0.2"));
        assert_eq!(retry.failed(t0), FIRST);
    }

    #[test]
    fn a_first_wait_over_six_hours_is_clamped() {
        let mut retry = UpdateRetry::new(Duration::MAX);
        retry.offer("4.0.1".into());
        assert_eq!(retry.failed(Instant::now()), MAX_FAILED_UPDATE_WAIT);
    }
}
