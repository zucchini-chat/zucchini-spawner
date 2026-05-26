//! Atomic write of a private byte blob: tmp+chmod+fsync+rename, 0600 on unix.

use std::ffi::OsString;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};

use tracing::warn;

/// Caller is responsible for creating the parent dir.
pub fn atomic_write_private(path: &Path, contents: &[u8]) -> io::Result<()> {
    let tmp = tmp_sibling(path);

    let mut options = OpenOptions::new();
    // `truncate` is implied by `create_new` (open fails outright if the file
    // exists), so requesting it is redundant.
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        options.mode(0o600);
        // O_NOFOLLOW + O_EXCL (via create_new) defend against a same-UID
        // attacker pre-creating `<name>.tmp` as a symlink to an arbitrary
        // file: without these we'd follow the link and truncate+write the
        // secret into the target.
        options.custom_flags(libc::O_NOFOLLOW);
    }

    // Recover from a crashed prior write that left an orphan tmp behind —
    // otherwise create_new would refuse forever. NotFound is the happy path;
    // any other error (EACCES, etc.) should surface so we fail loudly rather
    // than silently fall through to an open that may itself misbehave.
    match fs::remove_file(&tmp) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) => return Err(e),
    }

    // Anything from here through `rename` must clean up `tmp` on error so
    // failures don't leave `<name>.tmp` orphans accumulating in the dir.
    let result = (|| -> io::Result<()> {
        let mut f = options.open(&tmp)?;
        f.write_all(contents)?;
        f.sync_all()?;
        drop(f);
        fs::rename(&tmp, path)
    })();
    if let Err(e) = result {
        let _ = fs::remove_file(&tmp);
        return Err(e);
    }

    // Best-effort parent-dir fsync so the rename is durable across crashes on
    // Linux (no-op on macOS; not available on Windows). The rename already
    // succeeded, so a fsync error here is informational — don't fail the write.
    // Warn on EIO/EACCES so loss-of-durability incidents are operator-visible
    // (K_machine files are unrecoverable if the rename evaporates on crash).
    #[cfg(unix)]
    if let Some(parent) = path.parent() {
        match std::fs::File::open(parent) {
            Ok(dir) => {
                if let Err(e) = dir.sync_all() {
                    warn!(error = %e, parent = %parent.display(), "parent-dir fsync failed; durability not guaranteed");
                }
            }
            Err(e) => {
                warn!(error = %e, parent = %parent.display(), "parent-dir open for fsync failed; durability not guaranteed")
            }
        }
    }

    Ok(())
}

/// Sibling tmp path that *appends* `.tmp` to the full filename rather than
/// replacing the extension — `state.json` → `state.json.tmp`, not `state.tmp`
/// — so recovery scanners keep the "half-written X" hint.
fn tmp_sibling(path: &Path) -> PathBuf {
    let mut name: OsString = path.file_name().map(OsString::from).unwrap_or_default();
    name.push(".tmp");
    path.with_file_name(name)
}
