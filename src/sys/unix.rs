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
