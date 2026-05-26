//! Self-update loop: polls a remote (or file://) version file, downloads the
//! matching binary when the version changes, replaces the running binary,
//! and lets launchd/systemd respawn with the new version.
//!
//! Dev loop: set `SPAWNER_UPDATE_BASE_URL=file:///.../.zucchini-spawner/dev-update`
//! (written by `dev-watch.sh`) + `SPAWNER_UPDATE_CHECK_INTERVAL_SECS=2`.
//! Prod: defaults to `<PROD_BASE_URL>/install/`.
//!
//! CARGO_PKG_VERSION doesn't move on dev rebuilds, so we persist the last
//! applied version under `~/.zucchini-spawner/.applied-version` — otherwise the
//! loop would re-apply the same dev tag on every boot.

use std::time::Duration;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

const CURRENT_VERSION: &str = env!("CARGO_PKG_VERSION");
const DEFAULT_CHECK_INTERVAL: Duration = Duration::from_secs(10 * 60);
const RETRY_WHEN_BUSY: Duration = Duration::from_secs(60);
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

fn check_interval() -> Duration {
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

// `curl -f` makes any non-2xx exit non-zero so a failed fetch yields None,
// never an empty/garbage version that would look like a change. Using curl
// rather than reqwest so the same code path handles both `https://` and
// `file://` (reqwest doesn't speak file URLs).
async fn check_version() -> Option<String> {
    let url = format!("{}/{}", base_url(), VERSION_FILENAME);
    let output = tokio::process::Command::new("curl")
        .args(["-sf", &url])
        .output()
        .await
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let version = String::from_utf8(output.stdout).ok()?.trim().to_string();
    if version.is_empty() {
        return None;
    }
    Some(version)
}

pub async fn download_and_replace(remote_version: &str) -> Result<(), String> {
    let suffix = platform_suffix()?;
    let binary_name = format!("zucchini-spawner-{suffix}");
    let url = format!("{}/{binary_name}", base_url());
    let tmp_path = "/tmp/zucchini-spawner-new";

    info!(url = %url, "downloading new spawner binary");

    let status = tokio::process::Command::new("curl")
        .args(["-f", "-o", tmp_path, &url])
        .status()
        .await
        .map_err(|e| format!("curl failed: {e}"))?;
    if !status.success() {
        return Err("curl download failed".to_string());
    }

    let status = tokio::process::Command::new("chmod")
        .args(["+x", tmp_path])
        .status()
        .await
        .map_err(|e| format!("chmod failed: {e}"))?;
    if !status.success() {
        return Err("chmod failed".to_string());
    }

    #[cfg(target_os = "macos")]
    if let Err(e) = tokio::process::Command::new("xattr")
        .args(["-c", tmp_path])
        .status()
        .await
    {
        warn!("xattr clear failed: {e}");
    }

    let current_exe =
        std::env::current_exe().map_err(|e| format!("cannot find current exe: {e}"))?;

    let status = tokio::process::Command::new("mv")
        .args(["-f", tmp_path, current_exe.to_str().unwrap()])
        .status()
        .await
        .map_err(|e| format!("mv failed: {e}"))?;
    if !status.success() {
        return Err("mv replace failed".to_string());
    }

    if let Err(e) = std::fs::write(applied_version_path(), remote_version) {
        warn!("failed to persist applied version: {e}");
    }

    Ok(())
}

pub async fn run_update_loop(update_tx: mpsc::Sender<String>) {
    let base = base_url();
    let interval = check_interval();
    let mut last = last_applied_version();
    info!(
        embedded = CURRENT_VERSION,
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
                    // Update `last` regardless of send outcome so a stuck consumer
                    // doesn't make us re-announce on every tick. If a later apply
                    // fails, the next remote change re-triggers.
                    let _ = update_tx.send(remote.clone()).await;
                    last = remote;
                } else {
                    debug!(current = %last, "version is up to date");
                }
            }
            None => {
                warn!(?RETRY_WHEN_BUSY, "failed to check for updates, retrying");
                tokio::time::sleep(RETRY_WHEN_BUSY).await;
            }
        }
    }
}
