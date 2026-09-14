//! Unix realizations of the platform seam (see `sys/mod.rs`).
#![cfg(unix)]

use std::io;
use std::os::unix::fs::PermissionsExt;
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

use nix::sys::signal::{self, Signal};
use nix::unistd::Pid;
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::net::{UnixListener, UnixStream};
use tokio::signal::unix::{signal as unix_signal, Signal as UnixSignal, SignalKind};
use tracing::{info, warn};

#[cfg(any(target_os = "macos", target_os = "linux"))]
use super::tree_pids;

// ---------------------------------------------------------------------------
// OS shutdown request
// ---------------------------------------------------------------------------

/// A resettable stream of "the OS asked us to terminate" events. On unix that
/// is SIGTERM **or** SIGINT (either should bring the daemon down gracefully);
/// `recv` resolves on whichever fires next, so a caller can `recv` it once (the
/// main daemon loop) or repeatedly (the hermes trampoline forwards a stop frame
/// on each). On Windows the same shape wraps the console control events.
pub struct ShutdownSignal {
    sigterm: UnixSignal,
    sigint: UnixSignal,
}

impl ShutdownSignal {
    /// Register the OS-termination handlers. Errors propagate (the daemon
    /// `.expect()`s; the trampoline `.ok()`s into an optional handler).
    pub fn new() -> io::Result<ShutdownSignal> {
        Ok(ShutdownSignal {
            sigterm: unix_signal(SignalKind::terminate())?,
            sigint: unix_signal(SignalKind::interrupt())?,
        })
    }

    /// Resolve when the next termination request arrives. `None` only if both
    /// underlying streams end (process teardown), matching `Signal::recv`.
    pub async fn recv(&mut self) -> Option<()> {
        tokio::select! {
            v = self.sigterm.recv() => v,
            v = self.sigint.recv() => v,
        }
    }
}

// ---------------------------------------------------------------------------
// Login-shell command invocation
// ---------------------------------------------------------------------------

/// The login-shell program used to run every agent/probe command string. On
/// unix that is `$USER_SHELL` (set by the installer to the user's real login
/// shell) or `/bin/sh`. Both the agent spawn and the install probe must resolve
/// PATH the same way — otherwise the probe says "installed" while the agent
/// can't find the binary, or vice versa.
pub fn login_shell_program() -> String {
    std::env::var("USER_SHELL").unwrap_or_else(|_| "/bin/sh".to_string())
}

/// Build a `Command` that runs `cmd_string` through the login shell with
/// `-lic` (login + interactive + command). The single seam every spawn site
/// funnels through — agent one-shot/resident spawns, the `command -v` probe,
/// the hermes gateway, the cursor status probe — so PATH (asdf, homebrew, mise,
/// …) resolves identically across all of them. The `-i` is load-bearing: many
/// installers extend PATH only in interactive rc files a non-interactive login
/// shell won't source. Callers chain `.stdout/.stderr/.stdin/.env/.current_dir`
/// and `configure_new_process_group` as needed.
pub fn login_shell_command(cmd_string: &str) -> tokio::process::Command {
    let mut cmd = tokio::process::Command::new(login_shell_program());
    cmd.arg("-lic").arg(cmd_string);
    cmd
}

// ---------------------------------------------------------------------------
// Process-group teardown
// ---------------------------------------------------------------------------

/// Put a to-be-spawned child into a fresh process group so the whole tree
/// (login shell + agent underneath it) can be signalled at once. On unix this
/// is `setpgid(0,0)` at exec via `Command::process_group(0)`; the group already
/// exists by the time `ProcessGroup::adopt` runs, so adoption is just a pid
/// capture with no syscall.
pub fn configure_new_process_group(cmd: &mut tokio::process::Command) {
    cmd.process_group(0);
}

/// Opaque handle to the spawned child's process group. On unix it is the PGID
/// (== the child's pid, since the child is the group leader). Stored alongside
/// the `Child` for the child's whole lifetime; teardown signals the group.
#[derive(Clone, Copy)]
pub struct ProcessGroup {
    pgid: Pid,
}

impl ProcessGroup {
    /// Capture the group of a just-spawned child. Errors only if the child has
    /// no pid (already reaped before we looked).
    pub fn adopt(child: &tokio::process::Child) -> io::Result<ProcessGroup> {
        let pid = child
            .id()
            .ok_or_else(|| io::Error::other("child has no pid (already reaped)"))?;
        Ok(ProcessGroup {
            pgid: Pid::from_raw(pid as i32),
        })
    }

    /// Graceful kill: SIGTERM the whole process group → wait up to `grace` →
    /// SIGKILL if still alive → wait. If the child has already been reaped
    /// (`child.id()` is `None`) fall back to `child.kill()` (a no-op in that
    /// case) to keep the API consistent.
    pub async fn terminate(&self, child: &mut tokio::process::Child, grace: Duration) {
        if child.id().is_some() {
            let _ = signal::killpg(self.pgid, Signal::SIGTERM);
            match tokio::time::timeout(grace, child.wait()).await {
                Ok(_) => info!("agent exited after SIGTERM"),
                Err(_) => {
                    warn!(
                        "agent did not exit in {:?}, sending SIGKILL to process group",
                        grace
                    );
                    let _ = signal::killpg(self.pgid, Signal::SIGKILL);
                    let _ = child.wait().await;
                }
            }
        } else {
            let _ = child.kill().await;
        }
    }

    /// Fast-path kill: SIGKILL the process group immediately, no grace period.
    /// Falls back to `child.kill()` if the child has no pid.
    pub async fn kill(&self, child: &mut tokio::process::Child) {
        if child.id().is_some() {
            let _ = signal::killpg(self.pgid, Signal::SIGKILL);
        } else {
            let _ = child.kill().await;
        }
    }
}

/// Start time of process `pid` in microseconds since the epoch, or `None` if
/// it's gone. The pid-reuse guard for the boot orphan sweep (`orphans.rs`),
/// which is macOS-only: on Linux systemd kills the service's whole cgroup.
#[cfg(target_os = "macos")]
pub fn process_start_time(pid: u32) -> Option<u64> {
    let mut info: libc::proc_bsdinfo = unsafe { std::mem::zeroed() };
    let size = std::mem::size_of::<libc::proc_bsdinfo>() as libc::c_int;
    let n = unsafe {
        libc::proc_pidinfo(
            pid as libc::c_int,
            libc::PROC_PIDTBSDINFO,
            0,
            (&mut info as *mut libc::proc_bsdinfo).cast(),
            size,
        )
    };
    (n == size).then(|| info.pbi_start_tvsec * 1_000_000 + info.pbi_start_tvusec)
}

#[cfg(not(target_os = "macos"))]
pub fn process_start_time(_pid: u32) -> Option<u64> {
    None
}

/// A process with this pid exists (zombies included).
pub fn process_exists(pid: u32) -> bool {
    signal::kill(Pid::from_raw(pid as i32), None) != Err(nix::errno::Errno::ESRCH)
}

/// SIGTERM the process group `pgid`.
pub fn terminate_process_group(pgid: u32) {
    let _ = signal::killpg(Pid::from_raw(pgid as i32), Signal::SIGTERM);
}

/// SIGKILL the process group `pgid`.
pub fn kill_process_group(pgid: u32) {
    let _ = signal::killpg(Pid::from_raw(pgid as i32), Signal::SIGKILL);
}

/// The process group `pgid` has a member (zombies included).
pub fn process_group_exists(pgid: u32) -> bool {
    signal::killpg(Pid::from_raw(pgid as i32), None) != Err(nix::errno::Errno::ESRCH)
}

// ---------------------------------------------------------------------------
// Memory measurement (warm-keep eviction, `warm.rs`)
// ---------------------------------------------------------------------------

/// Footprint (KB) of each root's whole process tree, or `None` if `ps` fails
/// or the OS has no footprint measure. Walks ppid links rather than the
/// process group: Playwright's browser runs in a group of its own.
#[cfg(any(target_os = "macos", target_os = "linux"))]
pub async fn process_tree_footprint_kb(roots: &[u32]) -> Option<Vec<u64>> {
    let out = tokio::process::Command::new("ps")
        .args(["-A", "-o", "pid=,ppid="])
        .output()
        .await
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let rows = parse_ps(&String::from_utf8_lossy(&out.stdout));
    let footprints = roots
        .iter()
        .map(|&root| {
            tree_pids(&rows, root)
                .into_iter()
                .filter_map(footprint_kb)
                .sum()
        })
        .collect();
    Some(footprints)
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub async fn process_tree_footprint_kb(_roots: &[u32]) -> Option<Vec<u64>> {
    None
}

/// `(pid, ppid)` per `ps -o pid=,ppid=` line.
fn parse_ps(out: &str) -> Vec<(u32, u32)> {
    out.lines()
        .filter_map(|line| {
            let fields: Vec<u32> = line
                .split_whitespace()
                .map(str::parse)
                .collect::<Result<_, _>>()
                .ok()?;
            match fields[..] {
                [pid, ppid] => Some((pid, ppid)),
                _ => None,
            }
        })
        .collect()
}

/// What killing `pid` frees: private memory, compressed pages included
/// (Activity Monitor's "Memory"). RSS would count the shared `claude` binary
/// once per process.
#[cfg(target_os = "macos")]
fn footprint_kb(pid: u32) -> Option<u64> {
    let mut info: libc::rusage_info_v2 = unsafe { std::mem::zeroed() };
    let rc = unsafe {
        libc::proc_pid_rusage(
            pid as libc::c_int,
            libc::RUSAGE_INFO_V2,
            (&mut info as *mut libc::rusage_info_v2).cast(),
        )
    };
    (rc == 0).then_some(info.ri_phys_footprint / 1024)
}

/// Proportional set size: shared pages split among the processes mapping them.
#[cfg(target_os = "linux")]
fn footprint_kb(pid: u32) -> Option<u64> {
    let rollup = std::fs::read_to_string(format!("/proc/{pid}/smaps_rollup")).ok()?;
    let line = rollup.lines().find(|l| l.starts_with("Pss:"))?;
    line["Pss:".len()..].split_whitespace().next()?.parse().ok()
}

/// Total RAM, and pressure at the kernel's warn level or worse (the level
/// Activity Monitor graphs). `None` if unreadable.
#[cfg(target_os = "macos")]
pub fn memory() -> Option<crate::warm::Memory> {
    const PRESSURE_WARN: i32 = 2;
    let total_bytes: u64 = sysctl_value(c"hw.memsize")?;
    let level: i32 = sysctl_value(c"kern.memorystatus_vm_pressure_level")?;
    Some(crate::warm::Memory {
        total_kb: total_bytes / 1024,
        pressure: level >= PRESSURE_WARN,
    })
}

#[cfg(target_os = "macos")]
fn sysctl_value<T: Default>(name: &std::ffi::CStr) -> Option<T> {
    let mut value = T::default();
    let mut size = std::mem::size_of::<T>();
    let rc = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            (&mut value as *mut T).cast(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    (rc == 0 && size == std::mem::size_of::<T>()).then_some(value)
}

/// Total RAM, and pressure below 10% `MemAvailable`. `None` if unreadable.
#[cfg(target_os = "linux")]
pub fn memory() -> Option<crate::warm::Memory> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    let field = |name: &str| -> Option<u64> {
        let line = meminfo.lines().find(|l| l.starts_with(name))?;
        line[name.len()..].split_whitespace().next()?.parse().ok()
    };
    let total_kb = field("MemTotal:")?;
    let available_kb = field("MemAvailable:")?;
    (total_kb > 0).then(|| crate::warm::Memory {
        total_kb,
        pressure: available_kb * 10 < total_kb,
    })
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub fn memory() -> Option<crate::warm::Memory> {
    None
}

// ---------------------------------------------------------------------------
// Single-instance lock
// ---------------------------------------------------------------------------

/// Held exclusive lock on the single-instance pidfile. Releasing happens on
/// drop (the kernel releases the `flock` when the fd closes), so the caller
/// MUST keep this bound for the daemon's whole lifetime.
pub struct InstanceLock {
    _file: std::fs::File,
}

/// Take an exclusive `flock` on `path` (the spawner pidfile). Returns
/// `Ok(Some(lock))` when acquired (and records our pid into the file so a
/// future contender can name us), `Ok(None)` when another live instance
/// already holds it, and `Err` on an I/O failure taking the lock (caller's
/// choice to continue unlocked).
///
/// `flock` is crash-safe: the kernel releases it when the holder exits, so
/// there's no stale-lock recovery. The lock fd is `O_CLOEXEC` (Rust std
/// default) so spawned agents don't inherit it and pin the lock past the
/// daemon's death.
pub fn acquire_single_instance_lock(path: &Path) -> io::Result<Option<InstanceLock>> {
    use std::io::Write;

    let file = std::fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        // Never truncate on open: a contender that opens the file but fails the
        // flock below must not erase the holder's recorded pid. We clear+rewrite
        // it via `set_len` only after we actually hold the lock.
        .truncate(false)
        .open(path)?;

    if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
        let err = io::Error::last_os_error();
        if err.raw_os_error() == Some(libc::EWOULDBLOCK) {
            return Ok(None);
        }
        return Err(err);
    }

    // Lock held — record our pid so a future contender can name us.
    let _ = file.set_len(0);
    let _ = (&file).write_all(format!("{}\n", std::process::id()).as_bytes());
    Ok(Some(InstanceLock { _file: file }))
}

// ---------------------------------------------------------------------------
// Permission hardening
// ---------------------------------------------------------------------------

/// Restrict a secret file to owner-only (`chmod 0600`).
pub fn harden_file(path: &Path) -> io::Result<()> {
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
}

/// Restrict a directory to owner-only (`chmod 0700`) so its filenames don't
/// leak to local cohorts on hosts with the default 022 umask.
pub fn harden_dir(path: &Path) -> io::Result<()> {
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))
}

// ---------------------------------------------------------------------------
// Self-update binary replacement
// ---------------------------------------------------------------------------

/// Move the freshly-downloaded `new_exe` over the running `current_exe`. On unix
/// a running binary's inode can be replaced while it executes (the kernel keeps
/// the open text mapping alive), so this is a plain atomic rename. The caller
/// relies on launchd/systemd to respawn from the new file on next start.
pub fn replace_running_exe(new_exe: &Path, current_exe: &Path) -> io::Result<()> {
    std::fs::rename(new_exe, current_exe)
}

/// Delete any `<exe>.old` parked by a prior update. Unix replaces the inode in
/// place, so there is never a leftover — this is a no-op kept only so the call
/// site (startup) stays `#[cfg]`-free (the Windows arm does the real work).
pub fn sweep_stale_exe() {}

/// launchd `KeepAlive` / systemd `Restart=always` already respawn us after any
/// exit; only the Windows Scheduled Task needs this set up at runtime.
pub fn ensure_supervisor_keepalive() {}

// ---------------------------------------------------------------------------
// Local IPC — Unix domain sockets (windows: named pipes, step 6)
// ---------------------------------------------------------------------------

/// A bound local-IPC listener. On unix this wraps a `UnixListener`; the
/// stale-socket cleanup + 0600 hardening folded into [`IpcListener::bind`] used
/// to be open-coded at each call site (control + hermes). On Windows (step 6)
/// the same `accept()`-shaped API hides the named-pipe "one server instance per
/// client" lifecycle behind this type.
pub struct IpcListener {
    inner: UnixListener,
}

impl IpcListener {
    /// Bind the listener at `path`: remove a stale socket left by an unclean
    /// shutdown (otherwise `bind` fails with `EADDRINUSE`), bind, then harden
    /// the inode to 0600. Hardening is best-effort — the parent dir is already
    /// 0700, so a failed chmod degrades to dir-inherited confinement, not an
    /// open socket.
    pub fn bind(path: &Path) -> io::Result<IpcListener> {
        if path.exists() {
            if let Err(e) = std::fs::remove_file(path) {
                warn!(error = %e, path = %path.display(), "failed to remove stale IPC socket");
            }
        }
        let inner = UnixListener::bind(path)?;
        if let Err(e) = harden_file(path) {
            warn!(error = %e, path = %path.display(), "chmod 0600 IPC socket failed");
        }
        Ok(IpcListener { inner })
    }

    /// Accept one client connection. On unix this is `UnixListener::accept`
    /// (bind-once / accept-many). On Windows the impl connects the current pipe
    /// instance and spins up the next one before returning.
    pub async fn accept(&self) -> io::Result<IpcStream> {
        let (stream, _addr) = self.inner.accept().await?;
        Ok(IpcStream { inner: stream })
    }
}

/// One connected duplex IPC channel. Wraps a `UnixStream`; implements
/// `AsyncRead + AsyncWrite + Unpin` by delegating to the inner stream so
/// callers can `tokio::io::split(stream)` generically (no UDS-specific
/// `into_split`/`OwnedReadHalf`).
pub struct IpcStream {
    inner: UnixStream,
}

/// Client dial: connect to a listener bound at `path`. On unix this is
/// `UnixStream::connect`.
pub async fn ipc_connect(path: &Path) -> io::Result<IpcStream> {
    let inner = UnixStream::connect(path).await?;
    Ok(IpcStream { inner })
}

impl AsyncRead for IpcStream {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_read(cx, buf)
    }
}

impl AsyncWrite for IpcStream {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.inner).poll_write(cx, buf)
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_flush(cx)
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_shutdown(cx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ps_rows_walk_the_process_tree() {
        let out = "    1     0\n  100     1\n  101   100\n  102   101\n  200     1\n  bogus line\n  300\n";
        let rows = parse_ps(out);
        assert_eq!(rows.len(), 5);
        assert_eq!(tree_pids(&rows, 100), vec![100, 101, 102]);
        assert_eq!(tree_pids(&rows, 999), vec![999]);
    }

    #[tokio::test]
    async fn measures_this_process_and_the_machine() {
        let kb = process_tree_footprint_kb(&[std::process::id()])
            .await
            .unwrap();
        assert!(kb[0] > 0);
        assert!(memory().unwrap().total_kb > 0);
    }
}
