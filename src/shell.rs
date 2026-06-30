use std::process::Stdio;

use tracing::warn;

use crate::sys;

/// The login-shell program (`$USER_SHELL`/`/bin/sh` on unix, a resolved bash on
/// Windows). Thin re-export of the `sys` seam's resolver, kept for the spawn
/// sites that log which shell they used; the actual invocation goes through
/// [`sys::login_shell_command`].
pub fn user_login_shell() -> String {
    sys::login_shell_program()
}

/// `command -v <binary>` through the login shell so PATH (asdf, homebrew, mise,
/// etc.) resolves the same way as the agent spawn. Returns `false` on any
/// failure with a warn — the caller treats absent as "not installed".
pub async fn binary_on_path(binary: &str) -> bool {
    let probe = format!("command -v {} >/dev/null 2>&1", binary);
    match sys::login_shell_command(&probe)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .await
    {
        Ok(status) => status.success(),
        Err(e) => {
            warn!(error = %e, binary, "failed to probe `command -v`");
            false
        }
    }
}
