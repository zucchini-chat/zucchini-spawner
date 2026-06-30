//! Windows realizations of the platform seam (see `sys/mod.rs`).
//!
//! Bodies are stubs for now — signatures match `sys/unix.rs` exactly so later
//! steps only fill bodies, never touch call sites. Not compiled on macOS.
#![cfg(windows)]

use std::hash::{Hash, Hasher};
use std::io;
use std::os::windows::ffi::OsStrExt;
use std::os::windows::io::{AsRawHandle, RawHandle};
use std::path::Path;
use std::pin::Pin;
use std::sync::Mutex;
use std::task::{Context, Poll};
use std::time::Duration;

use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::net::windows::named_pipe::{
    ClientOptions, NamedPipeClient, NamedPipeServer, ServerOptions,
};
use tokio::signal::windows::{ctrl_break, ctrl_c, ctrl_shutdown, CtrlBreak, CtrlC, CtrlShutdown};
use tracing::{info, warn};
use windows_sys::Win32::Foundation::{
    CloseHandle, LocalFree, ERROR_LOCK_VIOLATION, ERROR_PIPE_BUSY, GENERIC_ALL, HANDLE,
};
use windows_sys::Win32::Security::Authorization::{
    SetEntriesInAclW, SetNamedSecurityInfoW, EXPLICIT_ACCESS_W, NO_MULTIPLE_TRUSTEE, SET_ACCESS,
    SE_FILE_OBJECT, TRUSTEE_IS_SID, TRUSTEE_IS_USER, TRUSTEE_W,
};
use windows_sys::Win32::Security::{
    GetTokenInformation, TokenUser, ACL, DACL_SECURITY_INFORMATION, NO_INHERITANCE,
    PROTECTED_DACL_SECURITY_INFORMATION, PSID, TOKEN_QUERY, TOKEN_USER,
};
use windows_sys::Win32::Storage::FileSystem::{
    LockFileEx, LOCKFILE_EXCLUSIVE_LOCK, LOCKFILE_FAIL_IMMEDIATELY,
};
use windows_sys::Win32::System::JobObjects::{
    AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
    SetInformationJobObject, TerminateJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
};
use windows_sys::Win32::System::Threading::{
    GetCurrentProcess, OpenProcessToken, CREATE_NEW_PROCESS_GROUP, CREATE_NO_WINDOW,
};
use windows_sys::Win32::System::IO::OVERLAPPED;

// ---------------------------------------------------------------------------
// OS shutdown request
// ---------------------------------------------------------------------------

/// Windows counterpart of the unix SIGTERM/SIGINT seam: the console control
/// events that ask the process to stop. Ctrl-C and Ctrl-Break are the
/// interactive-console stops; Ctrl-Shutdown fires when the system is logging
/// out / shutting down (the closest analogue to systemd/launchd SIGTERM).
pub struct ShutdownSignal {
    ctrl_c: CtrlC,
    ctrl_break: CtrlBreak,
    ctrl_shutdown: CtrlShutdown,
}

impl ShutdownSignal {
    pub fn new() -> io::Result<ShutdownSignal> {
        Ok(ShutdownSignal {
            ctrl_c: ctrl_c()?,
            ctrl_break: ctrl_break()?,
            ctrl_shutdown: ctrl_shutdown()?,
        })
    }

    pub async fn recv(&mut self) -> Option<()> {
        tokio::select! {
            v = self.ctrl_c.recv() => v,
            v = self.ctrl_break.recv() => v,
            v = self.ctrl_shutdown.recv() => v,
        }
    }
}

// ---------------------------------------------------------------------------
// Login-shell command invocation
// ---------------------------------------------------------------------------

/// The login-shell program used to run every agent/probe command string.
///
/// The adapter command strings are POSIX bash (pipes, `&&`, single-quote
/// escaping, `cd … &&` chaining) — byte-identical to unix. On Windows these CLI
/// agents run under a POSIX bash (Git Bash / WSL), NOT cmd.exe, so we run the
/// exact same strings under `bash -lic`. ASSUMES a Git Bash (or compatible
/// POSIX bash) is installed; cmd.exe / PowerShell are not used.
///
/// Resolution order:
///   1. `$ZUCCHINI_BASH` env override (operator escape hatch / non-standard
///      install),
///   2. `bash` on PATH (covers WSL `bash`, a PATH-added Git Bash, scoop, etc.),
///   3. the standard Git Bash install locations.
///
/// Falls back to bare `"bash"` if none resolve, so the spawn surfaces a clear
/// "program not found" rather than a silent misconfiguration.
pub fn login_shell_program() -> String {
    if let Ok(p) = std::env::var("ZUCCHINI_BASH") {
        if !p.is_empty() {
            return p;
        }
    }
    if let Ok(p) = which::which("bash") {
        return p.to_string_lossy().into_owned();
    }
    for var in ["ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"] {
        if let Some(base) = std::env::var_os(var) {
            let mut candidate = std::path::PathBuf::from(base);
            if var == "LOCALAPPDATA" {
                candidate.push("Programs");
            }
            candidate.push("Git");
            candidate.push("bin");
            candidate.push("bash.exe");
            if candidate.exists() {
                return candidate.to_string_lossy().into_owned();
            }
        }
    }
    "bash".to_string()
}

/// Build a `Command` that runs `cmd_string` through bash with `-lic` (login +
/// interactive + command) — the same flag and the same byte-identical command
/// strings as unix (see [`login_shell_program`]). The single seam every spawn
/// site funnels through. Callers chain `.stdout/.stderr/.stdin/.env/
/// .current_dir` and `configure_new_process_group` as needed.
pub fn login_shell_command(cmd_string: &str) -> tokio::process::Command {
    let mut cmd = tokio::process::Command::new(login_shell_program());
    cmd.arg("-lic").arg(cmd_string);
    cmd
}

// ---------------------------------------------------------------------------
// Process-group teardown — Job Object (step 4)
// ---------------------------------------------------------------------------

/// Per-process creation flags so a background-service spawn doesn't flash a
/// console window. `CREATE_NO_WINDOW` suppresses the console the child would
/// otherwise allocate (the spawner runs as a Windows Service / detached, with
/// no console of its own to inherit). `CREATE_NEW_PROCESS_GROUP` matches the
/// unix `setpgid` intent of putting the child in its own group — but on Windows
/// the *killing* is done by the Job Object (assigned in `adopt`), not by the
/// process group; the new group only detaches the child from our Ctrl-C/Break.
pub fn configure_new_process_group(cmd: &mut tokio::process::Command) {
    cmd.creation_flags(CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP);
}

/// A `HANDLE` is a raw pointer (`*mut c_void`) and so is neither `Send` nor
/// `Sync`. `ProcessGroup` is captured at spawn time and held in the same task
/// that drives the child across `.await` points (and that future may be moved
/// between worker threads by tokio), exactly as the unix `Pid` is. A job
/// `HANDLE` is a kernel-object handle, not thread-affine — it is valid from any
/// thread — so transferring it across threads is sound. This wrapper asserts
/// that to the compiler; the contained handle is only ever used by the owning
/// `ProcessGroup` (`SetInformationJobObject`/`AssignProcessToJobObject`/
/// `TerminateJobObject`/`CloseHandle`), never aliased or shared.
///
/// Shared SAFETY invariant for the FFI calls below: `self.job.0` is the live
/// job handle we own (created in `adopt`, closed exactly once in `Drop`, never
/// closed twice). Sites that rely only on this cite `// SAFETY: FFI; see
/// JobHandle.`
struct JobHandle(HANDLE);
unsafe impl Send for JobHandle {}
unsafe impl Sync for JobHandle {}

/// Opaque handle to the spawned child's Job Object — the Windows analogue of
/// the unix process group. Stored alongside the `Child` for its whole lifetime;
/// teardown is `TerminateJobObject`. The job carries
/// `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`, so if the spawner process dies the OS
/// reaps the whole agent tree when our handle closes (the crash-safety analogue
/// of the unix process group dying with its leader).
pub struct ProcessGroup {
    job: JobHandle,
}

impl ProcessGroup {
    /// Create a kill-on-close Job Object and assign the just-spawned child to
    /// it. The child's raw handle is valid only until the child is reaped, so we
    /// assign it to the job here — synchronously, before returning and before
    /// any `.await` the caller might run that could reap the child. Errors if
    /// the child has already exited (`raw_handle()` is `None`, mirroring unix
    /// `child.id()`) or any Win32 call fails.
    pub fn adopt(child: &tokio::process::Child) -> io::Result<ProcessGroup> {
        let handle: RawHandle = child
            .raw_handle()
            .ok_or_else(|| io::Error::other("child has no handle (already reaped)"))?;

        // SAFETY: FFI. `CreateJobObjectW` with null attributes/name creates an
        // unnamed job we exclusively own; it returns a null `HANDLE` on failure.
        let job = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
        if job.is_null() {
            return Err(io::Error::last_os_error());
        }
        let job = JobHandle(job);

        let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { std::mem::zeroed() };
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        // SAFETY: FFI. `info` is a valid, fully-initialized struct of the size
        // we pass for the `JobObjectExtendedLimitInformation` class.
        let ok = unsafe {
            SetInformationJobObject(
                job.0,
                JobObjectExtendedLimitInformation,
                std::ptr::addr_of!(info).cast(),
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            )
        };
        if ok == 0 {
            let err = io::Error::last_os_error();
            unsafe { CloseHandle(job.0) };
            return Err(err);
        }

        // SAFETY: FFI. `handle` is the child's live process handle (checked
        // non-`None` above and not yet reaped on this synchronous path).
        let assigned = unsafe { AssignProcessToJobObject(job.0, handle as HANDLE) };
        if assigned == 0 {
            let err = io::Error::last_os_error();
            unsafe { CloseHandle(job.0) };
            return Err(err);
        }

        Ok(ProcessGroup { job })
    }

    /// Graceful kill — best-effort. There is no Windows SIGTERM, so this diverges
    /// from unix: by the time the caller invokes `terminate` it has already
    /// dropped the child's stdin (writer task in `agent.rs`), so a piped agent
    /// that honors stdin-EOF can exit on its own. We wait up to `grace` for that,
    /// and only `TerminateJobObject` (the hard SIGKILL analogue) if it hasn't
    /// exited — there's no soft signal to send in between. Then reap so no
    /// zombie. If the child is already reaped, fall back to `child.kill()` (a
    /// no-op), matching the unix shape.
    pub async fn terminate(&self, child: &mut tokio::process::Child, grace: Duration) {
        if child.raw_handle().is_some() {
            match tokio::time::timeout(grace, child.wait()).await {
                Ok(_) => info!("agent exited on its own (stdin EOF)"),
                Err(_) => {
                    warn!("agent did not exit in {:?}, terminating job object", grace);
                    // SAFETY: FFI; see JobHandle.
                    unsafe { TerminateJobObject(self.job.0, 1) };
                    let _ = child.wait().await;
                }
            }
        } else {
            let _ = child.kill().await;
        }
    }

    /// Fast-path kill: terminate the whole job tree immediately, no grace period
    /// (the unix SIGKILL-the-group analogue), then reap so no zombie. Falls back
    /// to `child.kill()` if the child is already reaped.
    pub async fn kill(&self, child: &mut tokio::process::Child) {
        if child.raw_handle().is_some() {
            // SAFETY: FFI; see JobHandle.
            unsafe { TerminateJobObject(self.job.0, 1) };
            let _ = child.wait().await;
        } else {
            let _ = child.kill().await;
        }
    }
}

impl Drop for ProcessGroup {
    fn drop(&mut self) {
        // Closing our only handle to a kill-on-close job reaps any survivors —
        // the crash-safety net even on a path that never calls terminate/kill.
        // SAFETY: FFI; see JobHandle (this is the one-and-only close).
        unsafe { CloseHandle(self.job.0) };
    }
}

// ---------------------------------------------------------------------------
// Single-instance lock — LockFileEx (step 7)
// ---------------------------------------------------------------------------

/// Held exclusive lock on the single-instance pidfile. The lock is a
/// byte-range `LockFileEx` on the open file; Windows releases it automatically
/// when the handle closes (on drop here, or when the process dies — including a
/// crash), so there's no stale-lock recovery, matching the unix `flock` shape.
/// The caller MUST keep this bound for the daemon's whole lifetime.
pub struct InstanceLock {
    _file: std::fs::File,
}

/// Take an exclusive `LockFileEx` lock on `path` (the spawner pidfile). Returns
/// `Ok(Some(lock))` when acquired (and records our pid into the file so a future
/// contender can name us), `Ok(None)` when another live instance already holds
/// it, and `Err` on an I/O failure (caller's choice to continue unlocked).
///
/// `LockFileEx` is the Windows analogue of the unix `flock(LOCK_EX|LOCK_NB)`:
/// `LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY` is the non-blocking
/// exclusive take, and the kernel drops the lock when the handle closes — no
/// explicit unlock needed for crash safety. We lock a single byte (offset 0,
/// length 1): a whole-file lock the contention check agrees on, never read or
/// written, so it doesn't collide with the pid bytes we record.
pub fn acquire_single_instance_lock(path: &Path) -> io::Result<Option<InstanceLock>> {
    use std::io::Write;

    let file = std::fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        // Never truncate on open: a contender that opens the file but fails the
        // lock below must not erase the holder's recorded pid. We clear+rewrite
        // it via `set_len` only after we actually hold the lock.
        .truncate(false)
        .open(path)?;

    let mut overlapped: OVERLAPPED = unsafe { std::mem::zeroed() };
    // SAFETY: FFI. `file`'s handle is live (just opened, kept alive by `file`);
    // `overlapped` is a valid zeroed struct we own for the duration of the call.
    let ok = unsafe {
        LockFileEx(
            file.as_raw_handle() as HANDLE,
            LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY,
            0,
            1,
            0,
            &mut overlapped,
        )
    };
    if ok == 0 {
        let err = io::Error::last_os_error();
        // On the fail-immediately path `LockFileEx` returns `ERROR_LOCK_VIOLATION`
        // when another holder has the lock (the unix `EWOULDBLOCK` analogue), so
        // report contention, not error.
        if err.raw_os_error() == Some(ERROR_LOCK_VIOLATION as i32) {
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
// Permission hardening — ACLs (step 7)
// ---------------------------------------------------------------------------

/// RAII wrapper for an ACL that `SetEntriesInAclW` allocated with `LocalAlloc`;
/// the caller must `LocalFree` it. Holding it in a guard frees it on every exit
/// path (early return, `?`) so the `unsafe` apply code below stays linear.
struct LocalAllocAcl(*mut ACL);

impl Drop for LocalAllocAcl {
    fn drop(&mut self) {
        // SAFETY: FFI. `self.0` is the live `SetEntriesInAclW` allocation we own
        // and free exactly once (on drop).
        unsafe { LocalFree(self.0.cast()) };
    }
}

/// Build a single-user DACL granting `GENERIC_ALL` to the current process
/// token's user SID, and nothing else. Returned ACL is `LocalAlloc`-owned (freed
/// by [`LocalAllocAcl`]). Shared by [`harden_file`]/[`harden_dir`] so the
/// owner-only intent is expressed once.
///
/// `token_buf` is the caller-owned backing storage for the `TOKEN_USER` the SID
/// points into; it must outlive every use of the returned ACL (the ACE holds the
/// SID pointer, which points into `token_buf`).
fn build_owner_only_dacl(token_buf: &mut Vec<u8>) -> io::Result<LocalAllocAcl> {
    let mut token: HANDLE = std::ptr::null_mut();
    // SAFETY: FFI. `GetCurrentProcess` returns a pseudo-handle that needs no
    // close; we open it for `TOKEN_QUERY` only.
    let ok = unsafe { OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token) };
    if ok == 0 {
        return Err(io::Error::last_os_error());
    }
    let _token_guard = TokenHandle(token);

    // First call sizes the buffer; second fills it once we've grown to fit.
    let mut needed: u32 = 0;
    // SAFETY: FFI. Passing null/0 to query the required length is the documented
    // two-call pattern; failure with a too-small buffer is expected here.
    unsafe { GetTokenInformation(token, TokenUser, std::ptr::null_mut(), 0, &mut needed) };
    if needed == 0 {
        return Err(io::Error::last_os_error());
    }
    token_buf.resize(needed as usize, 0);
    // SAFETY: FFI. `token_buf` now holds `needed` bytes; on success it's a valid
    // `TOKEN_USER` whose `User.Sid` points within `token_buf`.
    let ok = unsafe {
        GetTokenInformation(
            token,
            TokenUser,
            token_buf.as_mut_ptr().cast(),
            needed,
            &mut needed,
        )
    };
    if ok == 0 {
        return Err(io::Error::last_os_error());
    }
    let user_sid: PSID = unsafe { (*token_buf.as_ptr().cast::<TOKEN_USER>()).User.Sid };

    let explicit = EXPLICIT_ACCESS_W {
        grfAccessPermissions: GENERIC_ALL,
        grfAccessMode: SET_ACCESS,
        grfInheritance: NO_INHERITANCE,
        Trustee: TRUSTEE_W {
            pMultipleTrustee: std::ptr::null_mut(),
            MultipleTrusteeOperation: NO_MULTIPLE_TRUSTEE,
            TrusteeForm: TRUSTEE_IS_SID,
            TrusteeType: TRUSTEE_IS_USER,
            ptstrName: user_sid.cast(),
        },
    };

    let mut new_acl: *mut ACL = std::ptr::null_mut();
    // SAFETY: FFI. One explicit-access entry; `oldacl` null builds a fresh ACL.
    // `new_acl` receives a `LocalAlloc`-owned ACL we wrap in `LocalAllocAcl`.
    let rc = unsafe { SetEntriesInAclW(1, &explicit, std::ptr::null(), &mut new_acl) };
    if rc != 0 {
        return Err(io::Error::from_raw_os_error(rc as i32));
    }
    Ok(LocalAllocAcl(new_acl))
}

/// Apply a **protected** owner-only DACL to `path`. `PROTECTED_DACL_*` is the
/// load-bearing flag: it disables ACL inheritance so the parent dir's
/// (potentially permissive) ACEs don't merge back in — the protected DACL is the
/// whole story, granting only the current user. This is the Windows realization
/// of the unix `chmod 0600`/`0700` intent (a directory ACL works identically;
/// child inheritance is exactly what the protected flag turns off), so
/// `harden_file`/`harden_dir` share this one body.
fn apply_owner_only_dacl(path: &Path) -> io::Result<()> {
    let mut token_buf: Vec<u8> = Vec::new();
    let dacl = build_owner_only_dacl(&mut token_buf)?;

    let mut wide: Vec<u16> = path.as_os_str().encode_wide().collect();
    wide.push(0);

    // SAFETY: FFI. `wide` is a NUL-terminated UTF-16 path; `dacl.0` is the live
    // ACL owned by `dacl` (freed after this call returns); owner/group/SACL are
    // null because we set only the DACL.
    let rc = unsafe {
        SetNamedSecurityInfoW(
            wide.as_ptr(),
            SE_FILE_OBJECT,
            DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            dacl.0,
            std::ptr::null(),
        )
    };
    if rc != 0 {
        return Err(io::Error::from_raw_os_error(rc as i32));
    }
    Ok(())
}

/// RAII close for the process token handle opened in [`build_owner_only_dacl`].
struct TokenHandle(HANDLE);

impl Drop for TokenHandle {
    fn drop(&mut self) {
        // SAFETY: FFI. `self.0` is the token handle we opened and own.
        unsafe { CloseHandle(self.0) };
    }
}

/// Restrict a secret file to the current user only (the `chmod 0600` analogue).
pub fn harden_file(path: &Path) -> io::Result<()> {
    apply_owner_only_dacl(path)
}

/// Restrict a directory to the current user only (the `chmod 0700` analogue) so
/// its filenames don't leak to other local users.
pub fn harden_dir(path: &Path) -> io::Result<()> {
    apply_owner_only_dacl(path)
}

// ---------------------------------------------------------------------------
// Self-update binary replacement
// ---------------------------------------------------------------------------

/// Move the freshly-downloaded `new_exe` over the running `current_exe`.
///
/// Windows refuses to overwrite or unlink the image of a running process, so we
/// can't do the unix in-place rename. But Windows DOES allow *renaming* a
/// running .exe out of the way (the open image handle survives the rename), so:
/// rename the live exe to `<exe>.old`, then move the new one into its place. The
/// service manager (the logon Scheduled Task) respawns from the new file on next
/// start; the leftover `<exe>.old` is best-effort-deleted at startup by
/// [`sweep_stale_exe`].
///
/// UNTESTED on a real Windows box — the rename-running-exe dance is correct per
/// the documented semantics but has not been exercised here.
pub fn replace_running_exe(new_exe: &Path, current_exe: &Path) -> io::Result<()> {
    let old_exe = stale_exe_path(current_exe);
    // A `.old` from a prior update may still exist (locked if the previous
    // process hasn't exited). Best-effort remove; rename below fails loudly if
    // the path is genuinely unavailable.
    let _ = std::fs::remove_file(&old_exe);
    std::fs::rename(current_exe, &old_exe)?;
    std::fs::rename(new_exe, current_exe)
}

/// `<exe>.old`, the parking spot for the previous binary after an update.
fn stale_exe_path(current_exe: &Path) -> std::path::PathBuf {
    let mut p = current_exe.as_os_str().to_owned();
    p.push(".old");
    std::path::PathBuf::from(p)
}

/// Delete the `<exe>.old` left behind by a prior in-place update. Best-effort:
/// on the very first boot after an update the old image may still be unlinkable
/// for a moment, so a failure here is ignored and retried next start.
pub fn sweep_stale_exe() {
    if let Ok(current) = std::env::current_exe() {
        let _ = std::fs::remove_file(stale_exe_path(&current));
    }
}

// ---------------------------------------------------------------------------
// Local IPC — named pipes (step 6)
// ---------------------------------------------------------------------------

/// Map the filesystem `&Path` the callers pass (`~/.zucchini-spawner/
/// control.sock`, `hermes.sock`) to a Win32 pipe name. There is no pipe inode
/// in the filesystem, so server `bind` and client `ipc_connect` only agree if
/// they derive the SAME name from the SAME path — this is the single function
/// both go through.
///
/// Shape: `\\.\pipe\zucchini-spawner-<stem>-<hash>`. The `<stem>` (file stem)
/// keeps the name human-readable for debugging; the `<hash>` of the full path
/// string disambiguates two different runtime dirs that share a filename (e.g.
/// per-user installs both ending in `control.sock`) so they don't collide on a
/// flat, machine-global pipe namespace.
fn pipe_name(path: &Path) -> String {
    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    path.to_string_lossy().hash(&mut hasher);
    format!(
        "\\\\.\\pipe\\zucchini-spawner-{stem}-{:016x}",
        hasher.finish()
    )
}

/// A bound named-pipe listener. Mirrors the unix `UnixListener`-backed
/// `IpcListener` (bind-once / accept-many) over the named-pipe "one server
/// instance per client" model: `bind` creates the first instance and `accept`
/// connects the pending one while immediately creating the next, so back-to-back
/// clients never hit `ERROR_PIPE_BUSY`.
pub struct IpcListener {
    name: String,
    // `accept(&self)` must hand off the pending instance and stash a fresh one;
    // the seam signature is `&self`, so the swap needs interior mutability. The
    // critical section is a couple of synchronous moves, never held across the
    // `.connect().await`, so a std `Mutex` (no async lock) is enough.
    next: Mutex<Option<NamedPipeServer>>,
}

impl IpcListener {
    /// Create the first pipe-server instance. Unlike unix there is no filesystem
    /// socket inode, so no stale-file cleanup and no `harden_file`.
    ///
    /// Pipe ACL (residual exposure, accepted): we do NOT pass a custom
    /// `SECURITY_ATTRIBUTES` restricting the pipe DACL to the current user. A
    /// `\\.\pipe\…` instance created with the default security descriptor is
    /// reachable by other processes **in the same logon session** (cross-session
    /// access already needs admin/SeTcb, and `reject_remote_clients` is on by
    /// tokio default, so off-machine clients can't connect). That same-session
    /// surface is acceptable for our single-user desktop model: every process in
    /// the user's own session already runs as that user with the same rights as
    /// the spawner, so a same-user attacker gains nothing over reading the
    /// secret files (which `harden_file` now protects) directly. Wiring a
    /// retained SD/SA through `create_with_security_attributes_raw` on every
    /// instance recreate would add a dangling-pointer lifetime hazard for no gain
    /// against the threat we actually model (other *users* on the box).
    pub fn bind(path: &Path) -> io::Result<IpcListener> {
        let name = pipe_name(path);
        let first = ServerOptions::new()
            .first_pipe_instance(true)
            .create(&name)?;
        Ok(IpcListener {
            name,
            next: Mutex::new(Some(first)),
        })
    }

    /// Accept one client. Use the pre-staged server instance (or create one
    /// inline if a prior accept failed to restock the slot), wait for a client to
    /// connect, then best-effort restock the NEXT instance before returning —
    /// without an available instance waiting, a client connecting in the window
    /// before the next `accept` call would get `ERROR_PIPE_BUSY`.
    ///
    /// Panic-free and self-healing: the control/hermes accept loops log a
    /// transient error and call `accept` again, so this must never panic on an
    /// empty `next` slot. The slot is best-effort — if a restock ever fails (or a
    /// prior accept left it empty) the next `accept` simply creates an instance
    /// inline. A successfully-connected client is never dropped over a restock
    /// failure.
    pub async fn accept(&self) -> io::Result<IpcStream> {
        let server = match self.next.lock().expect("ipc accept mutex poisoned").take() {
            Some(s) => s,
            None => ServerOptions::new().create(&self.name)?,
        };

        server.connect().await?;

        if let Ok(next) = ServerOptions::new().create(&self.name) {
            *self.next.lock().expect("ipc accept mutex poisoned") = Some(next);
        }

        Ok(IpcStream {
            inner: PipeEnd::Server(server),
        })
    }
}

/// One connected duplex IPC channel — the server-side instance after a client
/// connected, or the client side after dialing. Implements `AsyncRead +
/// AsyncWrite + Unpin` so call sites `tokio::io::split` it generically, matching
/// the unix `UnixStream`-backed `IpcStream`.
pub struct IpcStream {
    inner: PipeEnd,
}

/// Both pipe ends speak the same `AsyncRead`/`AsyncWrite`; the enum lets the
/// trait impls delegate through one match instead of duplicating per side.
enum PipeEnd {
    Server(NamedPipeServer),
    Client(NamedPipeClient),
}

/// Client dial: open the same pipe name the server bound. When every server
/// instance is momentarily busy the OS returns `ERROR_PIPE_BUSY` until the
/// server loops back to its next `accept`; the documented client pattern is a
/// bounded retry with a short wait, so a connect racing the accept-recreate
/// window succeeds instead of failing spuriously. Other errors pass through.
///
/// No pipe-ACL restriction here either — see [`IpcListener::bind`] for the
/// same-session residual-exposure rationale; the DACL lives on the server side.
pub async fn ipc_connect(path: &Path) -> io::Result<IpcStream> {
    let name = pipe_name(path);
    const MAX_ATTEMPTS: u32 = 10;
    const RETRY_WAIT: Duration = Duration::from_millis(50);

    for attempt in 1..=MAX_ATTEMPTS {
        match ClientOptions::new().open(&name) {
            Ok(client) => {
                return Ok(IpcStream {
                    inner: PipeEnd::Client(client),
                })
            }
            Err(e)
                if e.raw_os_error() == Some(ERROR_PIPE_BUSY as i32) && attempt < MAX_ATTEMPTS =>
            {
                tokio::time::sleep(RETRY_WAIT).await;
            }
            Err(e) => return Err(e),
        }
    }
    unreachable!("retry loop returns on the final attempt")
}

impl AsyncRead for IpcStream {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        match &mut self.inner {
            PipeEnd::Server(s) => Pin::new(s).poll_read(cx, buf),
            PipeEnd::Client(c) => Pin::new(c).poll_read(cx, buf),
        }
    }
}

impl AsyncWrite for IpcStream {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        match &mut self.inner {
            PipeEnd::Server(s) => Pin::new(s).poll_write(cx, buf),
            PipeEnd::Client(c) => Pin::new(c).poll_write(cx, buf),
        }
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        match &mut self.inner {
            PipeEnd::Server(s) => Pin::new(s).poll_flush(cx),
            PipeEnd::Client(c) => Pin::new(c).poll_flush(cx),
        }
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        match &mut self.inner {
            PipeEnd::Server(s) => Pin::new(s).poll_shutdown(cx),
            PipeEnd::Client(c) => Pin::new(c).poll_shutdown(cx),
        }
    }
}
