//! Best-effort self-uninstall. We can't tear down our own systemd/launchd
//! unit while running (the supervisor would respawn us), so hand cleanup off
//! to a detached `/bin/sh` child running the same `uninstall.sh` the user
//! script in the repo runs, parameterised for the daemon-mode flow:
//! key purge, silent. With user-scope systemd on Linux and per-user launchd
//! on macOS the script needs no sudo.

use std::process::{Command, Stdio};
use tracing::{info, warn};

const UNINSTALL_SCRIPT: &str = include_str!("../uninstall.sh");

pub fn spawn_detached_cleanup() {
    #[cfg(target_os = "linux")]
    spawn_linux();
    #[cfg(target_os = "macos")]
    spawn_macos();
}

#[cfg(target_os = "linux")]
fn spawn_linux() {
    // The spawner runs inside a systemd --user unit's cgroup. A plain fork
    // would inherit that cgroup, and `systemctl --user stop` from inside the
    // cleanup script would SIGTERM the script alongside the spawner.
    // `systemd-run --user --no-block` launches the script in its own
    // transient unit (separate cgroup) so it survives the teardown.
    //
    // Stage the script under the install dir and invoke `/bin/sh <path>`
    // instead of `/bin/sh -c <content>`. systemd performs `${VAR}` expansion
    // on every ExecStart= argv element before execve — braced `${VAR}` unset
    // in the unit env collapses to empty (only `REMOVE_KEY`/`SILENT` are set
    // here), so embedding the script body as argv silently rewrote every
    // `${SERVICE_NAME}` in `uninstall.sh` and left the user unit file behind
    // in a `Restart=always` loop forever. Pulling the body off argv sidesteps
    // it. The script unlinks `INSTALL_DIR` at the end; that's fine — sh holds
    // the inode open until exit.
    let staged_path = crate::zucchini_spawner_dir().join("uninstall.sh");
    if let Err(e) = std::fs::write(&staged_path, UNINSTALL_SCRIPT) {
        warn!(error = %e, path = %staged_path.display(), "failed to stage uninstall script");
        return;
    }
    let Some(staged_path_str) = staged_path.to_str() else {
        warn!(path = %staged_path.display(), "uninstall script path is not valid utf-8");
        return;
    };

    // Clear any stale `failed` state from a prior cleanup attempt first:
    // --collect only reaps the unit *after* it reaches inactive/failed; it
    // doesn't bypass the unit-name uniqueness check at queue time, so a
    // wedged-failed previous run would make StartTransientUnit return
    // "already exists".
    let _ = Command::new("systemctl")
        .args([
            "--user",
            "reset-failed",
            "zucchini-spawner-uninstall.service",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();

    let mut cmd = Command::new("systemd-run");
    cmd.args([
        "--user",
        "--no-block",
        "--collect",
        "--unit=zucchini-spawner-uninstall",
        "--setenv=REMOVE_KEY=1",
        "--setenv=SILENT=1",
        "/bin/sh",
        staged_path_str,
    ])
    .stdin(Stdio::null())
    .stdout(Stdio::null())
    .stderr(Stdio::null());

    // We MUST wait for systemd-run to exit, even with --no-block. The flag
    // only short-circuits the wait for the unit to become active — the
    // StartTransientUnit D-Bus call itself is still synchronous and takes a
    // few ms. If we return from main() before systemd-run finishes that call,
    // systemd tears down our cgroup (KillMode=control-group by default) and
    // SIGTERMs the in-flight systemd-run process before the transient unit
    // is queued. The cleanup then never runs, Restart=always brings us back
    // 5s later, and we loop forever (16k Sentry events / 1342 restarts seen
    // on the e2e box). By the time systemd-run exits the unit is queued in
    // its own cgroup, immune to our teardown.
    match cmd.spawn() {
        Ok(mut child) => {
            let pid = child.id();
            info!(pid, "spawned detached self-uninstall, waiting for handoff");
            match child.wait() {
                Ok(status) => info!(pid, ?status, "self-uninstall handoff complete"),
                Err(e) => warn!(pid, error = %e, "failed to wait on self-uninstall child"),
            }
        }
        Err(e) => warn!(error = %e, "failed to spawn self-uninstall script"),
    }
}

#[cfg(target_os = "macos")]
fn spawn_macos() {
    // macOS: launchctl bootout signals only the unit's main PID; a forked
    // child is reparented to launchd and survives our exit, so a plain
    // fire-and-forget is correct here. We must NOT `wait()` on it — the
    // cleanup script itself calls `launchctl bootout` which kills us (the
    // service's main PID), and blocking on the child would deadlock us
    // until launchd's SIGKILL timeout fires.
    let mut cmd = Command::new("/bin/sh");
    cmd.arg("-c")
        .arg(UNINSTALL_SCRIPT)
        .env("REMOVE_KEY", "1")
        .env("SILENT", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());

    match cmd.spawn() {
        Ok(child) => info!(pid = child.id(), "spawned detached self-uninstall"),
        Err(e) => warn!(error = %e, "failed to spawn self-uninstall script"),
    }
}
