//! Cross-adapter importer helpers. Lives under `adapters/` (not as a top-level
//! `src/import_shared.rs`) because the only callers are the per-adapter
//! `import()` functions in `adapters/{claude,cursor,codex,gemini}.rs` — keeping
//! it in the same module subtree keeps the import graph local.
//!
//! Project-id minting, label/title shaping, the synthetic-wrapper screen, the
//! RFC3339 timestamp parse, the user-message body envelope, and the
//! import-progress throttle all live here. The throttle (`ProgressThrottle`)
//! is shared because — despite each importer feeding it from a different source
//! (claude walks `~/.claude/projects/*`, cursor walks the sqlite `cursorDiskKV`
//! keys, codex walks `~/.codex/sessions/**`, gemini walks `~/.gemini/tmp/**`) —
//! all reduce to the same `(done, total)` arithmetic: identical rounding, the
//! same per-percent step gate, and the same `clamp(0, 100)` emission contract
//! with the dispatcher's per-kind rescaler. Keeping these in one place stops the
//! body/timestamp/progress logic from drifting between the four importers.

use std::path::Path;

use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::adapter::ImportProgress;
use crate::envelope::MessageEnvelope;

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
/// or has no resolvable basename. Lifted from the claude importer so both
/// adapters render project labels the same way — keeps the "Importing…" sheet
/// consistent.
pub(crate) fn basename_or(path: &str, fallback: &str) -> String {
    Path::new(path)
        .file_name()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or(fallback)
        .to_string()
}

/// Parse an RFC3339 timestamp string to UTC, e.g. `2025-10-03T00:16:55.629Z`.
/// Every importer's on-disk timestamps are RFC3339 (claude/gemini transcript
/// `timestamp`/`startTime`, codex rollout `timestamp`), so the
/// `parse_from_rfc3339 → with_timezone(Utc)` chain lived inline in each. Shared
/// here so the parse — and any future leniency (timezone handling, fallbacks) —
/// stays identical across adapters. Designed to drop into an `Option` chain via
/// `.and_then(parse_rfc3339_utc)`.
pub(crate) fn parse_rfc3339_utc(s: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(s)
        .ok()
        .map(|dt| dt.with_timezone(&Utc))
}

/// Serialize a plain user-typed prompt into the on-wire `messages.body`
/// envelope (`{text, attachments: []}`) the same way the live adapters do.
/// Imported user messages never carry attachments (the transcripts store only
/// text), so `attachments` is always empty. Every importer minted this exact
/// `MessageEnvelope` inline; sharing it keeps the body shape — and the
/// `expect` contract (a struct of owned `String`s is always serializable) —
/// identical across claude/cursor/codex/gemini.
pub(crate) fn user_message_body(text: &str) -> String {
    let env = MessageEnvelope {
        text: text.to_string(),
        attachments: Vec::new(),
    };
    serde_json::to_string(&env).expect("MessageEnvelope is always serializable")
}

/// One importable message ready to write: an already-shaped body string (a
/// `MessageEnvelope` JSON for user prompts, a claude-shape stream-json frame for
/// assistant text / tool_use), its sender, its `created_at`, and an optional
/// pre-minted id.
///
/// `id` is the only field that differs across importers: claude threads the
/// transcript entry uuid so `--continue`/`--resume` replays dedup in place;
/// codex/gemini leave it `None` (their native call/message ids aren't UUIDs) so
/// the writer mints `Uuid::now_v7()`. Everything downstream of building this
/// list — the PutChat-then-PutMessage emit — is identical, which is what
/// [`emit_chat`] consolidates.
pub(crate) struct ImportedMessage {
    pub(crate) id: Option<Uuid>,
    pub(crate) sender: &'static str,
    pub(crate) body: String,
    pub(crate) created_at: DateTime<Utc>,
}

/// One importable chat ready to write: its identity (`id`/`project_id`), its
/// title and `created_at`, and its already-shaped messages. Bundling these as
/// named fields (rather than positional [`emit_chat`] args) keeps the three
/// `Uuid`s — `id`, `project_id`, and the owner `user_id` — from being
/// transposable at the call site. Sibling of [`ImportedMessage`].
pub(crate) struct ImportedChat {
    pub(crate) id: Uuid,
    pub(crate) project_id: Uuid,
    pub(crate) title: String,
    pub(crate) created_at: DateTime<Utc>,
    pub(crate) messages: Vec<ImportedMessage>,
}

/// Emit one imported chat: a `PutChat` (which MUST land before its messages —
/// the backend has an FK on `messages.chat_id`) followed by one `PutMessage`
/// per message, all flagged `imported: true`. Send failures are dropped
/// silently (the writer logs them), matching every importer's prior inline loop.
///
/// Each message carries its own `created_at`; callers that need a monotonic
/// per-row bump (gemini's shared-timestamp records, cursor's timestamp-less
/// bubbles) compute it into `ImportedMessage.created_at` before calling, so that
/// caller-specific shaping stays out of this shared primitive.
pub(crate) async fn emit_chat(
    write_tx: &tokio::sync::mpsc::Sender<crate::writer::WriteEvent>,
    user_id: Uuid,
    chat: ImportedChat,
) {
    use crate::writer::WriteEvent;

    let _ = write_tx
        .send(WriteEvent::PutChat {
            id: chat.id,
            project_id: chat.project_id,
            user_id,
            title: chat.title,
            created_at: chat.created_at,
        })
        .await;

    let chat_id_str = chat.id.to_string();
    for msg in chat.messages {
        let _ = write_tx
            .send(WriteEvent::PutMessage {
                id: msg.id,
                chat_id: chat_id_str.clone(),
                user_id,
                sender: msg.sender,
                content: msg.body,
                created_at: Some(msg.created_at),
                imported: true,
            })
            .await;
    }
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
/// `<session_context>` is gemini-cli's analog: a synthetic first user message
/// the CLI injects to prime a resumed/continued session with prior context —
/// the user never typed it. The gemini importer screens it here so the same
/// prefix list stays the single source of truth across every adapter.
///
/// Shared by every importer: the claude importer needs it because these tags
/// appear in real transcripts; the cursor importer applies the same screen for
/// consistency even though Cursor bubbles haven't been observed to contain
/// them; the gemini importer relies on the `<session_context>` entry. A single
/// source of truth keeps the prefix list from drifting between them.
pub(crate) fn is_synthetic_wrapper(s: &str) -> bool {
    const PREFIXES: &[&str] = &[
        "<local-command-caveat>",
        "<local-command-stdout>",
        "<local-command-stderr>",
        "<command-name>",
        "<system-reminder>",
        "<task-notification>",
        "<session_context>",
    ];
    PREFIXES.iter().any(|p| s.starts_with(p))
}

/// Per-percent import-progress throttle, shared by every history importer.
///
/// The wire format is integer percent (`machines.claude_history_import_status`
/// = `running-{N}`, which iOS reads as `N / 100.0`), so one percent is the
/// finest step a client can render. `step` invokes `progress` whenever the
/// rounded percent advances at all — i.e. once per imported item until an
/// import exceeds ~100 items, beyond which multiple items share a percent and
/// coalesce into one emission. Emitting one PATCH per chat is cheap next to the
/// chat + message rows that chat's import already writes, and the dispatcher
/// dedups identical *rescaled* values before they hit the wire (see
/// [`ImportProgress`]), so a redundant same-percent emit here costs nothing.
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

    /// Record that `done` of `total` items are complete and emit a progress
    /// update if the percent has advanced since the last emission. `await`s the
    /// emit so a saturated writer channel applies backpressure rather than
    /// dropping the update (see [`ImportProgress`]).
    pub(crate) async fn step(&mut self, done: usize, total: usize, progress: &ImportProgress) {
        let pct = ((done as f64 / total as f64) * 100.0) as i32;
        if pct > self.last_pct {
            self.last_pct = pct;
            progress(pct.clamp(0, 100) as u8).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::future::BoxFuture;
    use std::sync::{Arc, Mutex};

    fn recording() -> (ImportProgress, Arc<Mutex<Vec<u8>>>) {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&seen);
        let cb: ImportProgress = Box::new(move |p| {
            let sink = Arc::clone(&sink);
            Box::pin(async move {
                sink.lock().unwrap().push(p);
            }) as BoxFuture<'static, ()>
        });
        (cb, seen)
    }

    #[tokio::test]
    async fn fires_once_per_item_when_each_advances_a_percent() {
        let (cb, seen) = recording();
        let mut t = ProgressThrottle::new();
        // 20 items: 5% per item, so every item advances the percent.
        for i in 1..=20 {
            t.step(i, 20, &cb).await;
        }
        assert_eq!(
            *seen.lock().unwrap(),
            vec![5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
        );
    }

    #[tokio::test]
    async fn fires_on_every_percent_step() {
        let (cb, seen) = recording();
        let mut t = ProgressThrottle::new();
        // 100 items: ~1% each — fires on essentially every integer percent.
        for i in 1..=100 {
            t.step(i, 100, &cb).await;
        }
        let fired = seen.lock().unwrap();
        // Strictly increasing (the `pct > last_pct` gate), reaching 100. A few
        // integer percents can be skipped by float truncation of
        // `done/total*100`, so assert the shape, not an exact 1..=100.
        assert!(fired.windows(2).all(|w| w[0] < w[1]), "strictly increasing");
        assert_eq!(fired.last(), Some(&100));
        assert!(fired.len() >= 98, "fires ~once per percent, got {}", fired.len());
    }

    #[tokio::test]
    async fn coalesces_items_that_share_a_percent() {
        let (cb, seen) = recording();
        let mut t = ProgressThrottle::new();
        // 1000 items: 0.1% each, so ~10 items collapse into each percent step.
        for i in 1..=1000 {
            t.step(i, 1000, &cb).await;
        }
        let fired = seen.lock().unwrap();
        assert!(fired.windows(2).all(|w| w[0] < w[1]), "strictly increasing");
        assert_eq!(fired.last(), Some(&100));
        assert!(
            (98..=100).contains(&fired.len()),
            "1000 items coalesce to ~100 emits, got {}",
            fired.len()
        );
    }

    #[tokio::test]
    async fn never_exceeds_one_hundred() {
        let (cb, seen) = recording();
        let mut t = ProgressThrottle::new();
        t.step(1, 1, &cb).await;
        assert_eq!(*seen.lock().unwrap(), vec![100]);
    }
}
