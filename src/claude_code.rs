use std::path::PathBuf;
use std::process::Stdio;

use serde::Deserialize;
use tokio::process::Command;
use tracing::{debug, warn};

pub async fn is_installed() -> bool {
    match Command::new(crate::shell::user_login_shell())
        .args(["-lic", "command -v claude >/dev/null 2>&1"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .await
    {
        Ok(status) => status.success(),
        Err(e) => {
            warn!(error = %e, "failed to probe `command -v claude`");
            false
        }
    }
}

/// claude code stores OAuth state in `~/.claude.json` under `oauthAccount` —
/// cross-platform, regardless of where the actual token lives (macOS Keychain,
/// `~/.claude/.credentials.json` on Linux). Fallback covers older / non-Keychain
/// installs.
pub fn is_authenticated() -> bool {
    let Some(home) = std::env::var_os("HOME").map(PathBuf::from) else {
        return false;
    };

    #[derive(Deserialize)]
    struct ClaudeCfg {
        #[serde(default, rename = "oauthAccount")]
        oauth_account: Option<serde_json::Value>,
    }

    let cfg = home.join(".claude.json");
    if let Ok(bytes) = std::fs::read(&cfg) {
        match serde_json::from_slice::<ClaudeCfg>(&bytes) {
            Ok(c) if c.oauth_account.is_some() => return true,
            Ok(_) => {}
            Err(e) => debug!(error = %e, path = %cfg.display(), "claude config not parseable"),
        }
    }

    let creds = home.join(".claude").join(".credentials.json");
    std::fs::metadata(&creds).map(|m| m.len() > 0).unwrap_or(false)
}
