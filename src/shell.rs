use std::process::Stdio;

use tracing::warn;

use crate::adapter::NativeSpawn;
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

/// Build a shell-free child command from a [`NativeSpawn`]: resolve `program`
/// on PATH (via the `which` crate — respects PATHEXT on Windows so `claude.exe`
/// is found), set argv + cwd, and, for a one-shot carrying a `prompt_stdin_file`,
/// open that file as the child's stdin (the native replacement for
/// `cat <file> | claude`). The caller still sets stdout/stderr + process-group
/// config + env. Errs (NotFound) if the program isn't on PATH, so the failure is
/// explicit rather than a cryptic "program not found".
pub fn native_command(spawn: &NativeSpawn) -> std::io::Result<tokio::process::Command> {
    let program = which::which(&spawn.program).map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("`{}` not found on PATH: {e}", spawn.program),
        )
    })?;
    let mut cmd = tokio::process::Command::new(program);
    cmd.args(&spawn.args);
    if let Some(cwd) = &spawn.cwd {
        cmd.current_dir(cwd);
    }
    if let Some(file) = &spawn.prompt_stdin_file {
        let f = std::fs::File::open(file)?;
        cmd.stdin(Stdio::from(f));
    }
    Ok(cmd)
}
