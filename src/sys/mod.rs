//! Platform-abstraction seam (mirrors std's `sys/pal/{unix,windows}`).
//!
//! The rest of the crate calls `sys::*` and never sees a `#[cfg]`. Only the
//! cfg-selected child module is compiled per target; both children expose the
//! SAME flat API (identical signatures) so a platform port is "fill in the
//! other file" and never touches call sites.
//!
//! The concerns isolated here, and their per-platform realizations:
//!
//! - **Process-group teardown** — unix: a new process group via `setpgid`
//!   (`Command::process_group(0)`) torn down with `killpg` SIGTERM→SIGKILL.
//!   windows: a kill-on-close Job Object torn down with `TerminateJobObject`
//!   (no SIGTERM, so the grace is spent waiting for stdin-EOF). The
//!   `ProcessGroup` value is captured at spawn time so the Windows path holds
//!   the job handle the same way the unix path holds the PGID; the caller
//!   stores it alongside the `Child` and it lives exactly as long as that
//!   `Child`.
//! - **Single-instance lock** — unix: `flock(LOCK_EX|LOCK_NB)` on a pidfile.
//!   windows: a non-blocking exclusive `LockFileEx` on the same pidfile (both
//!   release on handle close, so a crash frees the lock).
//! - **Secret-file / dir permission hardening** — unix: `chmod 0600`/`0700`.
//!   windows: a protected (inheritance-disabled) owner-only Win32 DACL via
//!   `SetNamedSecurityInfoW`, matching that intent.
//! - **Local IPC** — the control + hermes channels. unix: a `UnixListener`
//!   (`IpcListener`) with stale-socket cleanup + 0600 hardening folded into
//!   `bind`, and `UnixStream` (`IpcStream`) dialed via `ipc_connect`. windows:
//!   named pipes — `accept()` connects the current pipe instance and spins up
//!   the next, hiding the named-pipe lifecycle behind the unix-shaped
//!   bind-once/accept-many API. `IpcStream` is `AsyncRead + AsyncWrite` so call
//!   sites `tokio::io::split` it generically.

#[cfg(unix)]
mod unix;
#[cfg(unix)]
pub use unix::*;

#[cfg(windows)]
mod windows;
#[cfg(windows)]
pub use windows::*;
