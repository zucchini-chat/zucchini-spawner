//! Chat title → claude session name.
//!
//! `claude --name <n>` registers the live process under `n` in
//! `~/.claude/sessions/<pid>.json`, which is what `ListAgents` lists and what
//! `SendMessage` resolves — so naming a session after its Zucchini chat lets
//! another agent on the machine address it ("cover letters writer") instead of
//! re-deriving its context.
//!
//! Renaming a chat reaches the LIVE process through an internal `/rename` turn
//! (`Supervisor::rename_resident`), not by editing that registry file: the file
//! alone leaves the process announcing its OLD name in the `from-name=` of
//! every peer message it sends.

/// Claude's own cap is far higher, but a name is a handle someone has to type
/// at another agent, and an un-renamed chat's title is the first 100 chars of
/// its first message.
const MAX_CHARS: usize = 40;

/// Sanitize a chat title into a claude session name. `None` when nothing
/// usable is left (blank title, attachment-only chat, failed decrypt) — the
/// caller then passes no `--name` and claude falls back to its derived
/// `<dir>-<hex>`.
pub fn from_title(title: &str) -> Option<String> {
    let mut out = String::new();
    let mut pending_space = false;
    for ch in title.chars() {
        // Claude collapses newlines and zero-width joiners to spaces itself;
        // doing it here keeps the name we record equal to the name it registers.
        if ch.is_whitespace() || ch.is_control() || is_zero_width(ch) {
            pending_space = !out.is_empty();
            continue;
        }
        if pending_space {
            out.push(' ');
            pending_space = false;
        }
        // A trailing `[xxxxxx]` is `SendMessage`'s disambiguation ref syntax, so
        // brackets anywhere in a name make it ambiguous to address.
        match ch {
            '[' => out.push('('),
            ']' => out.push(')'),
            _ => out.push(ch),
        }
    }
    // Claude's arg parser reads `--name -foo` as a flag, not a value.
    let trimmed = out.trim_start_matches(|c: char| c == '-' || c.is_whitespace());
    let name = truncate_on_word_boundary(trimmed, MAX_CHARS);
    (!name.is_empty()).then_some(name)
}

fn is_zero_width(ch: char) -> bool {
    matches!(ch, '\u{200b}'..='\u{200f}' | '\u{feff}' | '\u{2060}')
}

fn truncate_on_word_boundary(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let cut: String = s.chars().take(max_chars).collect();
    match cut.rsplit_once(' ') {
        // Only back up to the last word break if it doesn't cost most of the name.
        Some((head, _)) if head.chars().count() >= max_chars / 2 => head.to_string(),
        _ => cut.trim_end().to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_title_sanitizes() {
        assert_eq!(
            from_title("cover letters writer").as_deref(),
            Some("cover letters writer")
        );
        // Claude turns newlines/ZWJ into spaces; a name that differs from what
        // it registers would make our "already asked for" check never settle.
        assert_eq!(
            from_title("  fix\nthe\u{200d}tests  ").as_deref(),
            Some("fix the tests")
        );
        // Brackets are SendMessage's ref syntax.
        assert_eq!(
            from_title("deploy [prod]").as_deref(),
            Some("deploy (prod)")
        );
        // `--name -fix` would parse as a flag.
        assert_eq!(
            from_title("- fix the crash").as_deref(),
            Some("fix the crash")
        );
        // Dashes and the spaces between them go together, or a stripped dash
        // would just uncover the next one.
        assert_eq!(
            from_title("- --fix the crash").as_deref(),
            Some("fix the crash")
        );
        assert_eq!(
            from_title("привет 👋 мир").as_deref(),
            Some("привет 👋 мир")
        );
        assert_eq!(from_title("   ").as_deref(), None);
        assert_eq!(from_title("").as_deref(), None);
    }

    #[test]
    fn from_title_truncates_at_a_word_boundary() {
        let long = "please refactor the entire sync layer and also fix the tests";
        let name = from_title(long).expect("name");
        assert!(name.chars().count() <= MAX_CHARS, "{name:?}");
        assert_eq!(name, "please refactor the entire sync layer");
        // A single unbroken word can't be word-cut; it still has to fit.
        let blob = "a".repeat(120);
        assert_eq!(from_title(&blob).expect("name").chars().count(), MAX_CHARS);
    }
}
