//! Cross-adapter importer helpers. Lives under `adapters/` (not as a top-level
//! `src/import_shared.rs`) because the only callers are the per-adapter
//! `import()` functions in `adapters/{claude,cursor}.rs` — keeping it in the
//! same module subtree keeps the import graph local.
//!
//! Today this is just project-id minting. The 5%-step progress throttle stays
//! inline in each adapter for now: each one drives progress from a different
//! source (claude walks `~/.claude/projects/*`, cursor walks the sqlite
//! `cursorDiskKV` keys), so there's no shared denominator to factor out yet.
//! Revisit when a third kind lands.

use std::path::Path;

use uuid::Uuid;

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
