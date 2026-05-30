//! Embed-and-self-heal install of the hermes Python plugin into
//! `~/.hermes/plugins/zucchini/`.
//!
//! The three plugin files (`plugin.yaml`, `__init__.py`, `adapter.py`) are
//! `include_str!`'d at spawner compile time from `zucchini-spawner/hermes_plugin/`
//! and written to the install dir at startup with a content-byte-compare
//! against the embedded copy so an out-of-date on-disk version auto-heals.
//!
//! Trust model: the cosign-verified spawner binary writes the plugin file
//! verbatim, so the binary's signature transitively covers the plugin
//! payload (see `tmp/verifiable-spawner.md`). No separate plugin signature.
//! No separate update channel — plugin version bumps land alongside spawner
//! releases, guaranteed in-sync on first boot of a new build.
//!
//! Idempotent — safe to call any number of times. Returns `Ok(())` on
//! success; any non-recoverable filesystem error bubbles up so `main.rs`
//! can `warn!` + skip the hermes path (the rest of the spawner still
//! works without hermes).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use tracing::{debug, info, warn};

// Embedded plugin sources. Paths are relative to THIS file
// (zucchini-spawner/src/hermes_support/). The live tree drop-target is
// `zucchini-spawner/hermes_plugin/`, a sibling of `src/` so non-Rust files
// don't pollute rust-analyzer's tree.
const PLUGIN_YAML: &str = include_str!("../../hermes_plugin/plugin.yaml");
const PLUGIN_INIT: &str = include_str!("../../hermes_plugin/__init__.py");
const PLUGIN_ADAPTER: &str = include_str!("../../hermes_plugin/adapter.py");

/// Files to install under `~/.hermes/plugins/zucchini/`. Order doesn't
/// matter functionally — we install all of them before the success log.
const PLUGIN_FILES: &[(&str, &str)] = &[
    ("plugin.yaml", PLUGIN_YAML),
    ("__init__.py", PLUGIN_INIT),
    ("adapter.py", PLUGIN_ADAPTER),
];

/// `~/.hermes/plugins/zucchini/` — hermes' plugin discovery scans
/// `~/.hermes/plugins/`, one subdir per plugin.
fn plugin_install_dir() -> PathBuf {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    home.join(".hermes").join("plugins").join("zucchini")
}

/// Idempotent install / self-heal. Safe to call multiple times. Logs at
/// info when a file is written (first-install or refresh) and at debug
/// when the on-disk copy already matches.
pub fn ensure_hermes_plugin_installed() -> Result<()> {
    let dir = plugin_install_dir();

    // Create the parent chain if missing. `~/.hermes/plugins/` may not
    // exist on a host that's never run hermes — creating it ourselves is
    // harmless (hermes will scan it on next launch and discover us).
    std::fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;

    let mut changes = 0usize;
    for (name, embedded) in PLUGIN_FILES {
        let path = dir.join(name);
        match needs_write(&path, embedded.as_bytes()) {
            Ok(true) => {
                write_file(&path, embedded.as_bytes())
                    .with_context(|| format!("write {}", path.display()))?;
                info!(file = %path.display(), "hermes plugin: wrote/refreshed file");
                changes += 1;
            }
            Ok(false) => {
                debug!(file = %path.display(), "hermes plugin: file matches, skipping");
            }
            Err(e) => {
                // A stat error: treat as "needs write" since we can't
                // compare. write_file will surface any real perm error.
                warn!(file = %path.display(), error = %e, "hermes plugin: stat failed, rewriting");
                write_file(&path, embedded.as_bytes())
                    .with_context(|| format!("write {}", path.display()))?;
                changes += 1;
            }
        }
    }

    if changes == 0 {
        debug!(dir = %dir.display(), "hermes plugin: install up-to-date");
    } else {
        info!(dir = %dir.display(), changes, "hermes plugin: install/refresh complete");
    }
    Ok(())
}

/// True iff `path` is missing OR its on-disk bytes don't match `expected`.
/// Reads the full file (small — each plugin file is < 64 KB). A false
/// positive is fine here (we'd rewrite an identical file).
fn needs_write(path: &Path, expected: &[u8]) -> std::io::Result<bool> {
    match std::fs::read(path) {
        Ok(actual) => Ok(actual.as_slice() != expected),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(true),
        Err(e) => Err(e),
    }
}

/// Write `bytes` to `path` using the project's atomic-write helper so a
/// crash mid-write never leaves a half-installed plugin file (which would
/// crash hermes' import on next launch).
fn write_file(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    crate::atomic::atomic_write_private(path, bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static COUNTER: AtomicUsize = AtomicUsize::new(0);

    fn tmp_dir() -> PathBuf {
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let pid = std::process::id();
        let dir = std::env::temp_dir().join(format!("plugin-install-test-{pid}-{n}"));
        std::fs::create_dir_all(&dir).expect("create test tmp dir");
        dir
    }

    #[test]
    fn needs_write_missing_file_returns_true() {
        let dir = tmp_dir();
        let p = dir.join("missing.py");
        assert!(needs_write(&p, b"abc").unwrap());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn needs_write_matching_returns_false() {
        let dir = tmp_dir();
        let p = dir.join("a.py");
        std::fs::write(&p, b"abc").unwrap();
        assert!(!needs_write(&p, b"abc").unwrap());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn needs_write_mismatch_returns_true() {
        let dir = tmp_dir();
        let p = dir.join("a.py");
        std::fs::write(&p, b"old contents").unwrap();
        assert!(needs_write(&p, b"new contents").unwrap());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Embedded plugin source must include the wire-format spec heading.
    /// Drift guard against the live tree being edited without syncing the
    /// embedded copy.
    #[test]
    fn embedded_adapter_py_carries_wire_spec() {
        assert!(
            PLUGIN_ADAPTER.contains("ZUCCHINI_SPAWNER_SOCK"),
            "embedded adapter.py is missing the env var ref — hermes_plugin/adapter.py likely out of sync"
        );
    }

    #[test]
    fn embedded_plugin_yaml_carries_name() {
        assert!(
            PLUGIN_YAML.contains("name: zucchini-platform"),
            "embedded plugin.yaml is missing the name — hermes_plugin/plugin.yaml likely out of sync"
        );
    }
}
