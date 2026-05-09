//! Best-effort self-uninstall. We can't tear down our own systemd/launchd
//! unit while running (the supervisor would respawn us), so hand cleanup off
//! to a detached `/bin/sh` child running the same `uninstall.sh` the user
//! script in the repo runs, parameterised for the daemon-mode flow:
//! non-interactive sudo, key purge, 2s pre-sleep, silent.

use std::process::Command;
use tracing::{info, warn};

const UNINSTALL_SCRIPT: &str = include_str!("../uninstall.sh");

pub fn spawn_detached_cleanup() {
    match Command::new("/bin/sh")
        .arg("-c")
        .arg(UNINSTALL_SCRIPT)
        .env("SUDO_CMD", "sudo -n")
        .env("REMOVE_KEY", "1")
        .env("PRE_SLEEP", "2")
        .env("SILENT", "1")
        // Detach: service managers SIGKILL our process group on exit.
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
    {
        Ok(child) => info!(pid = child.id(), "spawned detached self-uninstall"),
        Err(e) => warn!(error = %e, "failed to spawn self-uninstall script"),
    }
}
