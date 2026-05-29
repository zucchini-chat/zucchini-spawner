//! Cross-adapter importer helpers. Lives under `adapters/` (not as a top-level
//! `src/import_shared.rs`) because the only callers are the per-adapter
//! `import()` functions in `adapters/{claude,cursor}.rs` — keeping it in the
//! same module subtree keeps the import graph local.
//!
//! Project-id minting, label/title shaping, the synthetic-wrapper screen, and
//! the import-progress throttle all live here. The throttle (`ProgressThrottle`)
//! is shared because — despite each importer feeding it from a different source
//! (claude walks `~/.claude/projects/*`, cursor walks the sqlite `cursorDiskKV`
//! keys) — both reduce to the same `(done, total)` arithmetic: identical
//! rounding, the same 5% step gate, and the same `clamp(0, 100)` emission
//! contract with the dispatcher's per-kind rescaler. Keeping it in one place
//! stops that math from drifting and hands the future codex importer a ready
//! primitive.

use std::path::Path;

use uuid::Uuid;

use crate::adapter::ImportProgress;

// "zucchiniprojects" — fixed so re-imports converge on the same project ids
// across kinds AND across restarts. Both claude and cursor mint project ids
// from `(machine_id, project_path)` via this namespace; a project that has
// transcripts from both CLIs collapses to a single `projects` row.
const PROJECT_NS: Uuid = Uuid::from_bytes([
    0x7a, 0x75, 0x63, 0x63, 0x68, 0x69, 0x6e, 0x69, 0x70, 0x72, 0x6f, 0x6a, 0x65, 0x63, 0x74, 0x73,
]);

/// UUIDv5(`PROJECT_NS`, `machine_id || \0 || path`). The `\0` separator
/// prevents the rare collision where two `(machine, path)` pairs concatenate
/// to the same byte sequence (e.g. `(a, b\0c)` vs `(a\0b, c)`).
pub(crate) fn mint_project_id(machine_id: Uuid, path: &str) -> Uuid {
    Uuid::new_v5(
        &PROJECT_NS,
        &[machine_id.as_bytes().as_slice(), b"\0", path.as_bytes()].concat(),
    )
}

/// Strip a path down to its trailing component for use as the `projects.name`
/// label. Falls back to a caller-supplied placeholder when the path is empty
/// or has no resolvable basename (e.g. the "no project" sentinel used by the
/// cursor importer). Lifted from the claude importer so both adapters render
/// project labels the same way — keeps the "Importing…" sheet consistent.
pub(crate) fn basename_or(path: &str, fallback: &str) -> String {
    Path::new(path)
        .file_name()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or(fallback)
        .to_string()
}

/// Single-line title derived from a multi-line user prompt, capped at 100
/// chars. Cursor stores its own `composerData.name` (auto-titled by Cursor)
/// so this is only the fallback path; the claude importer uses the same
/// helper for the same reason.
pub(crate) fn collapse_title(text: &str) -> String {
    let collapsed = text.replace(['\r', '\n'], " ");
    collapsed.chars().take(100).collect()
}

/// User-content strings that claude-code wraps in these tags are synthetic —
/// either local CLI commands handled by the TUI (`<command-name>`,
/// `<local-command-stdout>`, `<local-command-stderr>`, `<local-command-caveat>`)
/// or harness-injected reminders/notifications (`<system-reminder>`,
/// `<task-notification>`). None of them are user-typed prompts, and the TUI
/// itself strips them before rendering. `<command-message>` is the contrast
/// case: it's how custom slash commands like `/simplify` introduce a real
/// prompt that gets sent to the model — keep those.
///
/// Shared by both importers: the claude importer needs it because these tags
/// appear in real transcripts; the cursor importer applies the same screen for
/// consistency even though Cursor bubbles haven't been observed to contain
/// them. A single source of truth keeps the prefix list from drifting between
/// the two.
pub(crate) fn is_synthetic_wrapper(s: &str) -> bool {
    const PREFIXES: &[&str] = &[
        "<local-command-caveat>",
        "<local-command-stdout>",
        "<local-command-stderr>",
        "<command-name>",
        "<system-reminder>",
        "<task-notification>",
    ];
    PREFIXES.iter().any(|p| s.starts_with(p))
}

/// 5%-step import-progress throttle, shared by every history importer.
///
/// Each emitted PATCH fans out via PowerSync to every connected client, so
/// firing one per imported item floods watch wakeups for negligible UX gain.
/// `step` only invokes `progress` when the rounded percent has advanced by at
/// least 5 points since the last emission.
///
/// Emission contract (kept identical across adapters so it can't drift):
/// - percent is `((done / total) * 100.0) as i32` — truncating, matching the
///   integer-floor each importer used inline;
/// - the value is clamped to `0..=100` before the `as u8` cast. For
///   `done <= total` the cast can't overflow, but the explicit clamp documents
///   the contract with the dispatcher's per-kind rescaler (which owns the final
///   100% / `"finished"` emission — see [`ImportProgress`]). Importers must not
///   emit 100 here for the `total == 0` short-circuit; they call `progress(100)`
///   directly in that branch instead of going through the throttle.
#[derive(Default)]
pub(crate) struct ProgressThrottle {
    last_pct: i32,
}

impl ProgressThrottle {
    pub(crate) fn new() -> Self {
        Self { last_pct: 0 }
    }

    /// Record that `done` of `total` items are complete and emit a throttled
    /// progress update if the percent has advanced at least 5 points.
    pub(crate) fn step(&mut self, done: usize, total: usize, progress: &ImportProgress) {
        let pct = ((done as f64 / total as f64) * 100.0) as i32;
        if pct >= self.last_pct + 5 {
            self.last_pct = pct;
            progress(pct.clamp(0, 100) as u8);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    fn recording() -> (ImportProgress, Arc<Mutex<Vec<u8>>>) {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&seen);
        let cb: ImportProgress = Box::new(move |p| sink.lock().unwrap().push(p));
        (cb, seen)
    }

    #[test]
    fn fires_every_five_percent_only() {
        let (cb, seen) = recording();
        let mut t = ProgressThrottle::new();
        // 20 items: 5% per item, so every item crosses a 5-point boundary.
        for i in 1..=20 {
            t.step(i, 20, &cb);
        }
        assert_eq!(
            *seen.lock().unwrap(),
            vec![5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
        );
    }

    #[test]
    fn coalesces_sub_step_increments() {
        let (cb, seen) = recording();
        let mut t = ProgressThrottle::new();
        // 100 items: 1% each. Should only fire on the multiples of 5.
        for i in 1..=100 {
            t.step(i, 100, &cb);
        }
        let fired = seen.lock().unwrap();
        assert_eq!(fired.len(), 20);
        assert_eq!(fired.first(), Some(&5));
        assert_eq!(fired.last(), Some(&100));
        assert!(fired.iter().all(|p| p % 5 == 0));
    }

    #[test]
    fn never_exceeds_one_hundred() {
        let (cb, seen) = recording();
        let mut t = ProgressThrottle::new();
        t.step(1, 1, &cb);
        assert_eq!(*seen.lock().unwrap(), vec![100]);
    }
}
