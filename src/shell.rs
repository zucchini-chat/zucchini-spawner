use std::process::Stdio;

use tokio::process::Command;
use tracing::warn;

/// Both the agent spawn and the claude-code install probe must resolve PATH
/// the same way — otherwise the probe says "installed" while the agent can't
/// find the binary, or vice versa.
pub fn user_login_shell() -> String {
    std::env::var("USER_SHELL").unwrap_or_else(|_| "/bin/sh".to_string())
}

/// `command -v <binary>` through the user's login shell so PATH (asdf,
/// homebrew, mise, etc.) resolves the same way as the agent spawn. Returns
/// `false` on any failure with a warn — the caller treats absent as
/// "not installed".
pub async fn binary_on_path(binary: &str) -> bool {
    let probe = format!("command -v {} >/dev/null 2>&1", binary);
    match Command::new(user_login_shell())
        .args(["-lic", &probe])
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
