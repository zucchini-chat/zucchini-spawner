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
    let mut cmd = if cfg!(target_os = "linux") {
        // The spawner runs inside a systemd --user unit's cgroup. A plain
        // fork would inherit that cgroup, and `systemctl --user stop` from
        // inside the cleanup script would SIGTERM the script alongside the
        // spawner. `systemd-run --user --no-block` launches the script in
        // its own transient unit (separate cgroup) so it survives the
        // teardown.
        //
        // Clear any stale `failed` state from a prior cleanup attempt
        // first: --collect only reaps the unit *after* it reaches
        // inactive/failed; it doesn't bypass the unit-name uniqueness
        // check at queue time, so a wedged-failed previous run would make
        // StartTransientUnit return "already exists".
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

        let mut c = Command::new("systemd-run");
        c.args([
            "--user",
            "--no-block",
            "--collect",
            "--unit=zucchini-spawner-uninstall",
            "--setenv=REMOVE_KEY=1",
            "--setenv=SILENT=1",
            "/bin/sh",
            "-c",
            UNINSTALL_SCRIPT,
        ]);
        c
    } else {
        // macOS: launchctl bootout signals only the unit's main PID; a
        // forked child is reparented to launchd and survives the parent's
        // exit, so a plain fork is safe.
        let mut c = Command::new("/bin/sh");
        c.arg("-c")
            .arg(UNINSTALL_SCRIPT)
            .env("REMOVE_KEY", "1")
            .env("SILENT", "1");
        c
    };

    cmd.stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());

    match cmd.spawn() {
        Ok(child) => info!(pid = child.id(), "spawned detached self-uninstall"),
        Err(e) => warn!(error = %e, "failed to spawn self-uninstall script"),
    }
}
