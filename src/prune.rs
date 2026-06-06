//! Selective context forgetting ("prune-context") — shared core.
//!
//! Per-dialect transcript surgery lives in each adapter module
//! (`adapters/{claude,gemini,codex}.rs`), wired in through the [`PruneOps`]
//! table on each `AdapterDescriptor`. This module holds the dialect-agnostic
//! pieces: [`PruneRequest`], [`PruneStats`]/[`PRUNED_PLACEHOLDER`], the
//! empty-args predicate ([`args_value_is_empty`]), the timeline frame builder
//! ([`pruned_frame_json`]), and the [`PruneOps`] vtable.
//!
//! Flow: the agent, mid-turn, runs `zucchini-spawner prune-context
//! --tool-name <Tool> --args <needle> --summary <digest>` (`--chat-id` defaults to
//! the `ZUCCHINI_CHAT_ID` env var exported on the spawn). The control task
//! ([`crate::control`]) resolves the chat's [`PruneOps`] from its `AgentKind`,
//! locates the transcript (`find_session`), pre-scans (`count_matches`), and —
//! on ≥1 match — queues a [`PruneRequest`] for the main loop, then returns so the
//! CLI prints its summary and EXITS 0. The loop does NOT abort on the RPC; it
//! waits for claude to emit that `prune-context` call's own `tool_result` frame
//! (the `AgentEvent::ToolResult` cue). Only then — once claude has PERSISTED the
//! call's result (including the summary stdout) to the transcript — does the loop
//! abort the agent, call `prune` to rewrite the transcript, and respawn with
//! `--resume`. Sequencing the abort after the result lands is what lets the
//! resumed agent see its own prune call + summary in context, so it doesn't
//! re-issue the now-satisfied prune. (claude may fire one more API request with
//! the un-pruned context before the abort arrives; the abort kills it and the
//! respawn re-reads the rewritten transcript, so the freed context still takes
//! effect — at the cost of that one stray request.)
//!
//! Blank-in-place (never delete) is the contract every dialect pruner honors:
//! matched tool outputs are replaced with [`PRUNED_PLACEHOLDER`] but every line
//! and every id/threading field is preserved byte-for-byte. What this buys is
//! narrower than "keeps the prompt-cache prefix": preserving the `parentUuid`
//! threading chain is what stops the CLI's `--resume` from REJECTING or
//! collapsing the transcript (the real lesson of the delete-based regression
//! below — orphaning the chain dropped a 168k resume to 32k tokens). It does NOT
//! preserve cache HITS past the edit: Anthropic prompt caching is a prefix match,
//! so any changed byte invalidates the cache from that point on. Blanking an
//! output re-processes the whole transcript TAIL after it at ~full input price on
//! resume. That's why the default is last-only ([`last_only_target`]): pruning the
//! MOST RECENT output pushes the first-changed byte as far right as possible, so
//! the smallest possible tail is re-tokenized — the cache-optimal choice. The
//! "repeat to prune older ones" path and any `--args` aimed at an EARLY large
//! output invalidate the entire tail, and pruning an early output in a chat that
//! won't continue for many more turns can be net-negative on cost (the tail
//! re-processed > the window reclaimed). The instruction (`adapter.rs`) and the
//! reminder steer the agent toward recent outputs for this reason.

use std::collections::HashSet;
use std::io;
use std::path::{Path, PathBuf};

/// Carries the parsed prune request from the control task (which doesn't own
/// the `Supervisor`) to the main `select!` loop, which aborts the agent,
/// rewrites the jsonl, and respawns. `jsonl_path` is resolved control-side
/// (under the read guard's lookup) so the main loop doesn't re-glob.
#[derive(Debug, Clone)]
pub struct PruneRequest {
    pub jsonl_path: PathBuf,
    /// Which transcript dialect `jsonl_path` is in — selects the rewrite via
    /// `AgentKind::prune_ops()` (the per-adapter [`PruneOps::prune`]).
    pub agent_kind: crate::adapter::AgentKind,
    /// The CLAUDE-shape tool name to prune, or `""` for the "any tool" selector
    /// (match on `needle` alone — codex omits `--tool-name`).
    pub tool_name: String,
    /// The `--args` glob over argument VALUES; an empty string is the
    /// empty-arguments selector (match calls made with no arguments).
    pub needle: String,
    /// The agent's `--summary`: the task-relevant takeaway it still needs from the
    /// output it's dropping (NOT a recap of the whole output). Named `reason` on
    /// the wire (`ControlRequest::PruneContext`) and in logs for hot-reload
    /// compatibility; `--reason` is still an accepted alias.
    pub reason: String,
}

/// Shared park-table of pending [`PruneRequest`]s, keyed by `chat_id`. Replaces
/// the old `prune_tx`/`prune_rx` channel + main-loop-owned `HashMap`. The control
/// task writes synchronously inside the `prune-context` RPC — the write completes
/// before the RPC returns to the agent, hence before the agent's `tool_result`
/// persists, hence before the apply cue (`AgentEvent::ToolResult`) fires — and the
/// main loop's cue arm drains the chat's entry. Co-owning the table through a
/// shared lock (the same pattern [`crate::state::SharedMirror`] already uses
/// between these two tasks) removes the previous two-channels-into-one-`select!`
/// ordering hazard: the cue can no longer be processed before its request is
/// visible, so a prune can't be silently dropped or stranded for a later turn.
pub type PendingPrunes =
    std::sync::Arc<tokio::sync::Mutex<std::collections::HashMap<String, Vec<PruneRequest>>>>;

/// Per-adapter selective-forgetting vtable. Each adapter whose CLI owns a local
/// plaintext transcript the spawner can edit + the CLI re-reads on resume
/// (claude, gemini, codex) exports a `pub(crate) const PRUNE_OPS: PruneOps`
/// next to its dialect pruners and points its `AdapterDescriptor::prune` at it.
/// Adapters with no editable local transcript (cursor, hermes) leave
/// `AdapterDescriptor::prune` `None`.
///
/// All three callbacks operate purely on the filesystem/JSON; the orchestration
/// (abort → prune → respawn) lives in `main.rs` / `control.rs` and dispatches
/// through `AgentKind::prune_ops()`.
pub struct PruneOps {
    /// Locate the session transcript for the agent's self-generated session id
    /// (harvested into `chats.agent_session_id`), searching beneath `base` — the
    /// CLI's home dir resolved by [`crate::adapter::AgentKind::cli_home`] (e.g.
    /// `~/.codex`), so the resolver honors the CLI's relocation env var instead
    /// of hardcoding `$HOME/.<cli>`. `None` when no transcript for that id exists
    /// yet under `base`.
    pub find_session: fn(base: &Path, session_id: &str) -> Option<PathBuf>,
    /// Read-only pre-scan: how many ELIGIBLE tool calls match
    /// (`tool_name`, `needle`)? Eligible = matches whose paired output isn't
    /// already `[pruned]`. `needle` is the `--args` value glob, where `""`
    /// selects no-args calls. Run control-side before aborting — zero means we
    /// error back to the still-alive agent (it can retry with a better needle)
    /// instead of killing + respawning. The count also drives the CLI's "N
    /// remain" message; the prune itself only blanks the most recent match.
    pub count_matches: fn(&Path, &str, &str) -> io::Result<usize>,
    /// Rewrite the transcript in place for a WHOLE coalesced burst in ONE pass:
    /// `targets` is the list of `(tool_name, needle)` pairs from one chat's queued
    /// `PruneRequest`s (all sharing this `jsonl_path`). Blanks the last-only target
    /// of each, reproducing exactly what K separate single-target last-only passes
    /// produced (see [`rewrite_jsonl_batch_last_only`]) — same blanked set, same
    /// freed bytes — but with one read / one serialize / one fsync-pair instead of
    /// K. Returns the SUMMED [`PruneStats`] for the burst (`results_blanked` ≤ K).
    pub prune_batch: fn(&Path, &[PruneTarget]) -> io::Result<PruneStats>,
}

/// One `(tool_name, needle)` prune target — the per-item selector a queued
/// `PruneRequest` carries, passed in bulk to [`PruneOps::prune_batch`]. `tool_name`
/// is the CLAUDE-shape name (`""` = any tool); `needle` is the `--args` value glob
/// (`""` = no-args selector). Named alias so the batch `fn` signatures stay legible.
pub type PruneTarget = (String, String);

/// Is a tool call's arguments value "empty" (the call was made with no
/// arguments)? True for absent, JSON `null`, empty object `{}`, or empty array
/// `[]`. Predicate behind the `--args ""` selector. Parses and checks rather
/// than substring-matching `"{}"` (a nested empty object must not count).
/// Shared so claude/gemini (`Value` `input`/`args`) and codex (decodes its
/// encoded `arguments` string first) agree on what "empty" means.
pub(crate) fn args_value_is_empty(v: Option<&serde_json::Value>) -> bool {
    match v {
        None | Some(serde_json::Value::Null) => true,
        Some(serde_json::Value::Object(m)) => m.is_empty(),
        Some(serde_json::Value::Array(a)) => a.is_empty(),
        _ => false,
    }
}

/// Match `pattern`'s `*`-separated segments, in order, as substrings within a
/// SINGLE `leaf` string. Each non-empty segment must appear after the previous
/// one's end; empty segments (from leading/trailing `*` or a `**`) anchor
/// nothing and are skipped. No `*` in `pattern` ⇒ one segment ⇒ plain
/// substring. Case-sensitive. An all-empty pattern (`""`, `"*"`) matches any
/// leaf — but the empty needle never reaches here (it's the no-args selector,
/// handled before the glob is invoked). Shared so the dialect matchers can't
/// drift on glob semantics.
pub(crate) fn glob_leaf_match(leaf: &str, pattern: &str) -> bool {
    let mut cursor = 0usize;
    for segment in pattern.split('*') {
        if segment.is_empty() {
            continue;
        }
        match leaf[cursor..].find(segment) {
            Some(rel) => cursor += rel + segment.len(),
            None => return false,
        }
    }
    true
}

/// True if ANY string leaf of `value` matches [`glob_leaf_match`] against
/// `pattern`, recursing into object VALUES and array elements. Object KEYS are
/// never tested — `--args` globs the call's argument values, not their field
/// names (matching a key was the over-match bug). Shared by the dialect arg
/// matchers (claude `input`, gemini `args`, codex's decoded `arguments`).
pub(crate) fn value_glob_match(value: &serde_json::Value, pattern: &str) -> bool {
    match value {
        serde_json::Value::String(s) => glob_leaf_match(s, pattern),
        serde_json::Value::Array(items) => items.iter().any(|v| value_glob_match(v, pattern)),
        serde_json::Value::Object(map) => map.values().any(|v| value_glob_match(v, pattern)),
        // Numbers/bools/null carry no string leaf to glob against.
        _ => false,
    }
}

/// True if `leaf` is the agent's own `prune-context` CLI invocation. Such calls
/// must NEVER be eligible prune targets: at prune time the still-in-flight call's
/// own command line carries the same `--args` needle the user is pruning by, so
/// last-only would otherwise pick it (the newest match), blank its output-less
/// arguments, and spare the real tool output the user meant to drop
/// (`results_blanked` 0 → frame skipped → "nothing happened"). This bit the codex
/// adapter hardest — codex has no `Read` tool, so file reads AND the prune call
/// are both `shell`→`Bash` and collide on the same needle; claude/gemini hit it
/// too whenever the file was read via `Bash`. We key on BOTH the subcommand token
/// and its required `--args` flag so an unrelated command that merely mentions the
/// word (e.g. `grep prune-context …`) is NOT excluded — being conservative here
/// only ever LEAVES data in context, never drops the wrong thing.
fn leaf_is_prune_context_command(leaf: &str) -> bool {
    leaf.contains("prune-context") && leaf.contains("--args")
}

/// Recursive twin of [`value_glob_match`]: true if ANY string leaf of `value` is
/// a `prune-context` CLI invocation (see [`leaf_is_prune_context_command`]). Each
/// dialect's matcher calls this on a candidate call's decoded args (claude
/// `input`, gemini `args`, codex's decoded `arguments`) and skips the call when it
/// returns true, so a prune never targets itself.
pub(crate) fn value_is_prune_context_call(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::String(s) => leaf_is_prune_context_command(s),
        serde_json::Value::Array(items) => items.iter().any(value_is_prune_context_call),
        serde_json::Value::Object(map) => map.values().any(value_is_prune_context_call),
        _ => false,
    }
}

/// Shared last-only eligibility scan behind every dialect's `eligible_matches`.
/// Reads the transcript ONCE and, per parsed entry (blank/parse-failed lines
/// skipped), runs two injected closures:
///
///   * `collect_matched` — push the entry's matching tool-call ids onto the
///     ordered `Vec` (DOCUMENT ORDER preserved; an entry may push 0..n ids).
///   * `collect_pruned` — insert into the set every id whose paired output is
///     ALREADY the `[pruned]` placeholder.
///
/// Returns the ordered matched ids MINUS the already-pruned set ("ordered minus
/// pruned" — the combinator that was identical across claude/codex). De-dup of
/// the matched list is the caller's concern, but in practice each entry yields a
/// distinct id; the retain only drops already-pruned ids, never reorders. The
/// pruner takes `.last()` of this to blank the most-recent eligible match.
///
/// Dialect-agnostic by design — the closures own ALL transcript-shape knowledge,
/// so step-4 gemini plugs in by passing a `collect_matched` that recurses via its
/// `walk_objects` over matching `toolCall` ids and a `collect_pruned` that reads
/// `functionResponse.response.output == "[pruned]"`. No claude/codex struct
/// leaks into this signature.
pub(crate) fn select_eligible_ids(
    path: &Path,
    mut collect_matched: impl FnMut(&serde_json::Value, &mut Vec<String>),
    mut collect_pruned: impl FnMut(&serde_json::Value, &mut HashSet<String>),
) -> io::Result<Vec<String>> {
    let parsed = read_parsed_lines(path)?;
    let mut matched: Vec<String> = Vec::new();
    let mut already_pruned: HashSet<String> = HashSet::new();
    for (_, value) in parsed.iter() {
        let Some(entry) = value else { continue };
        collect_matched(entry, &mut matched);
        collect_pruned(entry, &mut already_pruned);
    }
    matched.retain(|id| !already_pruned.contains(id));
    Ok(matched)
}

/// "Pick the most-recent eligible match" → singleton (or empty) id set, the
/// last-only target every dialect pruner feeds to its existing blank pass(es).
/// `eligible` is the DOCUMENT-ORDER list from [`select_eligible_ids`]; we take
/// `.last()`. An empty input (TOCTOU after the control pre-check, or nothing
/// eligible) yields an empty set ⇒ the blank pass is a safe no-op
/// (`results_blanked` 0 → timeline frame skipped). Kept tiny + dialect-agnostic
/// so the blank step stays per-dialect; step-4 gemini reuses it unchanged.
/// TEST-ONLY: documents the last-only pick the batch driver inlines as
/// `.rev().find()`; pinned by `last_only_target_picks_last_or_empty`.
#[cfg(test)]
pub(crate) fn last_only_target(eligible: Vec<String>) -> HashSet<String> {
    eligible.last().cloned().into_iter().collect()
}

/// Placeholder replacing a pruned tool result's content. A plain string is the
/// simplest API-valid shape across every dialect (claude accepts a string for
/// `tool_result.content`; gemini/codex store the output as a string). Shared so
/// the three pruners can't drift on it.
pub(crate) const PRUNED_PLACEHOLDER: &str = "[pruned]";

/// Replace `map[key]` with the string `replacement`, idempotently. Returns
/// `None` when the field is absent or ALREADY equals `replacement` (no change —
/// keeps re-prune byte-stable so `--resume` keeps its prompt-cache prefix),
/// `Some(freed)` when actually replaced this call. `freed` = original
/// serialized length − replacement length − 2 JSON quotes, saturating at 0.
/// Shared by the 3 dialect pruners so the idempotency check and freed-byte
/// arithmetic can't drift.
pub(crate) fn blank_string_field(
    map: &mut serde_json::Map<String, serde_json::Value>,
    key: &str,
    replacement: &str,
) -> Option<usize> {
    let existing = map.get(key)?;
    if matches!(existing, serde_json::Value::String(s) if s == replacement) {
        return None;
    }
    let orig_len = existing.to_string().len();
    map.insert(
        key.into(),
        serde_json::Value::String(replacement.to_string()),
    );
    Some(orig_len.saturating_sub(replacement.len() + 2))
}

/// Tool-name gate shared by the gemini/codex `*_matches` predicates. `actual` is
/// the raw dialect tool name from the transcript; `want` is the CLAUDE-shape name
/// the user typed in `--tool-name`. `map` inverts the dialect's
/// persisted-name → claude-name table: it returns every raw dialect name `want`
/// fans out to (e.g. claude `Read` → gemini `["read_file","read_many_files"]`).
/// An EMPTY mapping means `want` isn't a known claude name, so it's matched
/// literally against `actual` (dialects forward unknown tools under their native
/// name, so a user can prune by the raw dialect name too). An EMPTY `want` is the
/// "any tool" selector — it matches every call, so the prune keys on the args
/// needle alone (codex omits `--tool-name`: its reads/greps/edits all funnel
/// through two generic shell tools, so a claude-shape name narrows nothing the
/// path doesn't). The args substring step stays dialect-specific and is NOT done
/// here. Shared by gemini/codex so the "mapped set OR literal" idiom can't drift.
pub(crate) fn tool_name_matches(
    actual: &str,
    want: &str,
    map: fn(&str) -> Vec<&'static str>,
) -> bool {
    if want.is_empty() {
        return true;
    }
    let mapped = map(want);
    if mapped.is_empty() {
        actual == want
    } else {
        mapped.contains(&actual)
    }
}

/// One parsed transcript line: the original verbatim string (kept byte-for-byte
/// for unchanged / blank / parse-failed lines) alongside its parsed JSON value
/// (`None` for blank or parse-failed lines).
type ParsedLine = (String, Option<serde_json::Value>);

/// Parse `path` line by line into [`ParsedLine`]s with the shared byte-stability
/// rules: trim trailing `\r`/`\n`; blank or parse-failed lines are kept verbatim
/// with `None` (never dropped); other lines parse via `serde_json::from_str`.
fn read_parsed_lines(path: &Path) -> io::Result<Vec<ParsedLine>> {
    let text = std::fs::read_to_string(path)?;
    let mut parsed: Vec<ParsedLine> = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.trim().is_empty() {
            parsed.push((trimmed.to_string(), None));
            continue;
        }
        match serde_json::from_str::<serde_json::Value>(trimmed) {
            Ok(v) => parsed.push((trimmed.to_string(), Some(v))),
            Err(_) => parsed.push((trimmed.to_string(), None)),
        }
    }
    Ok(parsed)
}

/// Single-read SINGLE-TARGET last-only prune driver — the one-round case of
/// [`rewrite_jsonl_batch_last_only`], which it delegates to. TEST-ONLY: each
/// adapter's test-only `prune_*_jsonl` calls it as the equivalence oracle the
/// batch path is pinned against; production prunes go through the batch driver so a
/// whole coalesced burst collapses to one read/write. Returns the same
/// `(PruneStats, final_entries)` shape so a dialect's fail-closed post-scan can run
/// on the EXACT state the driver decided.
#[cfg(test)]
pub(crate) fn rewrite_jsonl_last_only(
    path: &Path,
    collect_matched: impl FnMut(&serde_json::Value, &mut Vec<String>),
    collect_pruned: impl FnMut(&serde_json::Value, &mut HashSet<String>),
    blank: impl FnMut(&mut serde_json::Value, &HashSet<String>, &mut HashSet<String>) -> Option<usize>,
) -> io::Result<(PruneStats, Vec<Option<serde_json::Value>>)> {
    // A single-target prune is the degenerate one-round batch: one (matched
    // collector) target, last-only over the same parsed Vec. Routing it through
    // the batch driver keeps the two paths from drifting.
    let mut collect_matched = collect_matched;
    rewrite_jsonl_batch_last_only(
        path,
        1,
        |_idx, entry, matched| collect_matched(entry, matched),
        collect_pruned,
        blank,
    )
}

/// Single-read BATCH last-only prune driver. Reproduces, in ONE read + ONE write,
/// the EXACT result of running [`rewrite_jsonl_last_only`] `count` times in a row,
/// where each round targets its own `(tool_name, needle)` (selected by `idx`).
///
/// The old apply path looped K [`PruneOps::prune`] calls per chat, each doing a
/// full read + parse + atomic write (which `sync_all`s the file AND its parent
/// dir) — K reads, K serializes, K fsync-pairs on a possibly multi-MB transcript.
/// This collapses them to one.
///
/// Why one pass reproduces K last-only passes EXACTLY: each separate pass picks
/// `last()` of (its matches − ids whose paired output is ALREADY `[pruned]` on
/// disk). Running pass N sees pass N−1's freshly-written `[pruned]` placeholder,
/// which only ever flips an id's eligibility from eligible→excluded. In memory we
/// model that by subtracting BOTH the on-disk `already_pruned` set AND a running
/// `chosen` set of the ids picked in earlier rounds of THIS batch — so round N's
/// `last()` lands on the same id the on-disk pass N would have. (Two rounds with
/// the SAME needle therefore pick two DISTINCT successive matches, exactly as two
/// separate same-needle calls did.) After accumulating up to `count` target ids,
/// ONE [`blank_pass`] blanks them all and writes once.
///
/// `collect_matched(idx, entry, &mut Vec)` pushes the matching ids for target
/// `idx` (`0..count`) in DOCUMENT ORDER; `collect_pruned` / `blank` are the same
/// per-dialect closures the single-target driver takes — reused unchanged. Returns
/// one [`PruneStats`] for the whole batch (`results_blanked` = total outputs
/// blanked, `freed_bytes` summed) plus the post-blank `final_entries` for a
/// dialect's fail-closed post-scan.
pub(crate) fn rewrite_jsonl_batch_last_only(
    path: &Path,
    count: usize,
    mut collect_matched: impl FnMut(usize, &serde_json::Value, &mut Vec<String>),
    mut collect_pruned: impl FnMut(&serde_json::Value, &mut HashSet<String>),
    blank: impl FnMut(&mut serde_json::Value, &HashSet<String>, &mut HashSet<String>) -> Option<usize>,
) -> io::Result<(PruneStats, Vec<Option<serde_json::Value>>)> {
    let parsed = read_parsed_lines(path)?;

    // Ids whose paired output is ALREADY `[pruned]` on disk — excluded from every
    // round (same as `select_eligible_ids`'s retain). Scanned once.
    let mut already_pruned: HashSet<String> = HashSet::new();
    for (_, value) in parsed.iter() {
        let Some(entry) = value else { continue };
        collect_pruned(entry, &mut already_pruned);
    }

    // Run the pass-1 last-only selection `count` times IN MEMORY, accumulating the
    // chosen target ids. Each round re-derives its match list, subtracts the
    // on-disk already-pruned set AND the ids chosen in earlier rounds, and takes
    // `last()` — reproducing what `count` successive on-disk passes would pick.
    let mut chosen: HashSet<String> = HashSet::new();
    for idx in 0..count {
        let mut matched: Vec<String> = Vec::new();
        for (_, value) in parsed.iter() {
            let Some(entry) = value else { continue };
            collect_matched(idx, entry, &mut matched);
        }
        let pick = matched
            .into_iter()
            .rev()
            .find(|id| !already_pruned.contains(id) && !chosen.contains(id));
        if let Some(id) = pick {
            chosen.insert(id);
        }
        // No eligible match this round (TOCTOU, or the needle's matches were all
        // claimed by earlier rounds) → contributes nothing, exactly as a separate
        // pass that found 0 eligible would have blanked nothing.
    }

    blank_pass(path, parsed, &chosen, blank)
}

/// Pass 2 behind [`rewrite_jsonl_last_only`]: blank the outputs keyed by `target`,
/// re-serializing only the lines a blank touched (everything else byte-for-byte
/// verbatim), atomic-write the result, and assemble [`PruneStats`] + the final
/// parsed entries. Pulled out from the driver so the selection (pass 1) and the
/// write path (pass 2) stay separately testable.
fn blank_pass(
    path: &Path,
    parsed: Vec<ParsedLine>,
    target: &HashSet<String>,
    mut blank: impl FnMut(
        &mut serde_json::Value,
        &HashSet<String>,
        &mut HashSet<String>,
    ) -> Option<usize>,
) -> io::Result<(PruneStats, Vec<Option<serde_json::Value>>)> {
    let mut out_lines: Vec<String> = Vec::with_capacity(parsed.len());
    let mut final_entries: Vec<Option<serde_json::Value>> = Vec::with_capacity(parsed.len());
    let mut freed_bytes = 0usize;
    let mut outputs_blanked: HashSet<String> = HashSet::new();
    for (original, value) in parsed.into_iter() {
        let Some(mut entry) = value else {
            out_lines.push(original);
            final_entries.push(None);
            continue;
        };
        match blank(&mut entry, target, &mut outputs_blanked) {
            // Unchanged — write the original bytes back verbatim.
            None => out_lines.push(original),
            // Mutated: re-serialize even when `freed` is 0 (a tiny output shorter
            // than the placeholder still got blanked — don't drop the edit).
            Some(freed) => {
                freed_bytes += freed;
                match serde_json::to_string(&entry) {
                    Ok(s) => out_lines.push(s),
                    // Re-serialization of a value we just parsed shouldn't fail;
                    // if it does, keep the original rather than corrupt the file.
                    Err(_) => out_lines.push(original),
                }
            }
        }
        final_entries.push(Some(entry));
    }

    let mut blob = out_lines.join("\n");
    blob.push('\n');
    crate::atomic::atomic_write_private(path, blob.as_bytes())?;

    Ok((
        PruneStats {
            results_blanked: outputs_blanked.len(),
            freed_bytes,
        },
        final_entries,
    ))
}

/// What a dialect `prune` reports back. `results_blanked` = number of tool
/// outputs whose content we replaced (the user-facing count). `freed_bytes` =
/// approximate transcript size removed (summed serialized length of the blanked
/// content, minus placeholders); feeds a coarse `~Nk freed` token estimate in
/// the timeline frame — the precise drop shows in the context gauge on resume.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PruneStats {
    pub results_blanked: usize,
    pub freed_bytes: usize,
}

/// Coarse `freed_bytes → tokens` estimate (`/4`) shared by the timeline frame and
/// the respawn prompt so they never disagree.
pub fn freed_tokens(stats: PruneStats) -> usize {
    stats.freed_bytes / 4
}

/// Human one-liner shared by the timeline frame (`pruned_frame_json`) and the
/// post-prune respawn prompt (`pruned_respawn_prompt`): `pruned N tool output(s)
/// · ~Nk freed` (the `freed` clause is omitted when nothing measurable was freed).
pub fn prune_summary(stats: PruneStats) -> String {
    let n = stats.results_blanked;
    let noun = if n == 1 {
        "tool output"
    } else {
        "tool outputs"
    };
    let tokens = freed_tokens(stats);
    let freed = if tokens >= 1000 {
        format!(" · ~{}k freed", tokens / 1000)
    } else if tokens > 0 {
        format!(" · ~{tokens} freed")
    } else {
        String::new()
    };
    format!("pruned {n} {noun}{freed}")
}

/// Build the synthetic `system`/`context_pruned` frame inserted into the chat
/// after a successful prune (the describer renders `summary` as a dim system
/// line). Pre-formats the human string here — one place, not triplicated across
/// the iOS/Android describers. `freed_bytes / 4 ≈ tokens` is a coarse estimate.
pub fn pruned_frame_json(stats: PruneStats) -> String {
    serde_json::json!({
        "type": "system",
        "subtype": "context_pruned",
        "summary": prune_summary(stats),
        "count": stats.results_blanked,
        "freed_tokens": freed_tokens(stats),
    })
    .to_string()
}

/// Prompt fed to the agent we respawn right after a successful prune. The agent
/// was killed mid-`prune-context` (its shell call never returned a clean exit),
/// so without this it can't tell the prune SUCCEEDED from a genuine miss — and
/// re-running the now-satisfied prune returns "no … call found" (0 eligible),
/// which reads as failure and sends it into a retry loop (observed with gemini).
/// Stating the success explicitly is what breaks that loop; the agent can see the
/// `[pruned]` placeholder in its own resumed context, so we don't restate it.
pub fn pruned_respawn_prompt(stats: PruneStats) -> String {
    format!(
        "Context pruning succeeded — {}. Continue with the task.",
        prune_summary(stats)
    )
}

/// Reminder text the `PostToolUse` hook surfaces to claude (as a
/// `<system-reminder>`) after a large tool result, nudging the agent to prune
/// outputs it no longer needs. Single source of truth — the spawned
/// `--settings` hook command (`adapters/claude.rs`) and the hook handler
/// (`main.rs::prune_reminder_output`) both reference this; do not retype it.
pub const PRUNE_REMINDER_TEXT: &str =
    "don't forget to prune large outputs after you don't need them anymore";

/// Size gate for the `PostToolUse` prune reminder: only nudge when the
/// SERIALIZED `tool_response` is larger than this many bytes/chars. Small
/// results aren't worth a prune, so we stay quiet below the threshold to avoid
/// spamming a `<system-reminder>` after every trivial tool call.
pub const PRUNE_REMINDER_MIN_TOOL_RESPONSE_BYTES: usize = 1500;

/// Pure core of the `prune-reminder-hook` subcommand. Parses a claude
/// `PostToolUse` hook JSON `payload` from stdin, measures the SERIALIZED length
/// of its `tool_response` field (a string OR an object/array — serialize
/// whatever it is; absent ⇒ 0), and returns the exact `additionalContext` JSON
/// line to print when that length exceeds
/// [`PRUNE_REMINDER_MIN_TOOL_RESPONSE_BYTES`], else `None`.
///
/// Never errors: any parse failure (malformed/non-JSON payload) yields `None`,
/// so the hook prints nothing and exits 0 — a failing hook could disrupt
/// claude, so it must be inert on bad input. The emitted JSON is built via
/// serde_json so the apostrophe and quoting are escaped correctly, matching the
/// proven shape `{"hookSpecificOutput":{"hookEventName":"PostToolUse",
/// "additionalContext":"<text>"}}`.
pub fn prune_reminder_output(payload: &str) -> Option<String> {
    let parsed: serde_json::Value = serde_json::from_str(payload).ok()?;
    // Suppress the nudge for Task sub-agent tool calls. The `--settings` hook is
    // session-global, so it fires on sidechain (sub-agent) tool results too — but
    // a sub-agent must NOT prune: it has no `prune-context` how-to (that lives
    // only in the MAIN agent's `--append-system-prompt`), its context is
    // ephemeral (dies on return to the parent), and the `prune-context` CLI
    // resolves its target transcript by `chat_id` → a sub-agent acting on the
    // nudge would prune the PARENT's context. claude tags a sub-agent's
    // PostToolUse payload with `agent_id`/`agent_type` (main-agent calls carry
    // neither); `session_id`/`transcript_path` are SHARED with the parent and so
    // can't distinguish. Empirically verified against captured payloads. Gate on
    // `agent_id` — the most direct "this is a sub-agent" signal.
    if parsed.get("agent_id").is_some() {
        return None;
    }
    let len = match parsed.get("tool_response") {
        None | Some(serde_json::Value::Null) => 0,
        // A string's "serialized length" for the gate is its content length,
        // not its quoted JSON form — that's what the agent actually reads.
        Some(serde_json::Value::String(s)) => s.len(),
        // Objects/arrays/scalars: serialize and measure.
        Some(other) => other.to_string().len(),
    };
    if len > PRUNE_REMINDER_MIN_TOOL_RESPONSE_BYTES {
        Some(
            serde_json::json!({
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "additionalContext": PRUNE_REMINDER_TEXT,
                }
            })
            .to_string(),
        )
    } else {
        None
    }
}

/// Test helpers shared by every dialect pruner's test module, so the temp-file
/// pattern + jsonl reader aren't copy-pasted across the three.
#[cfg(test)]
pub(crate) mod test_util {
    use std::path::{Path, PathBuf};

    /// Owns a unique temp jsonl file under `std::env::temp_dir()` and deletes
    /// it on drop. No `tempfile` crate dependency (the spawner doesn't pull
    /// one in) — the same hand-rolled pattern `adapters/cursor.rs` tests use.
    pub(crate) struct TempJsonl {
        path: PathBuf,
    }

    impl TempJsonl {
        fn new(lines: &[&str]) -> Self {
            // Process-global monotonic counter: parallel tests can read IDENTICAL
            // `SystemTime::now()` nanos and collide on the same temp path, so the
            // counter guarantees a unique name regardless of clock resolution.
            use std::sync::atomic::{AtomicU64, Ordering};
            static SEQ: AtomicU64 = AtomicU64::new(0);
            let mut path = std::env::temp_dir();
            path.push(format!(
                "zucchini_prune_test_{}_{}_{}.jsonl",
                std::process::id(),
                SEQ.fetch_add(1, Ordering::Relaxed),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::write(&path, format!("{}\n", lines.join("\n"))).unwrap();
            Self { path }
        }

        pub(crate) fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TempJsonl {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    pub(crate) fn write_jsonl(lines: &[&str]) -> TempJsonl {
        TempJsonl::new(lines)
    }

    pub(crate) fn read_lines(path: &Path) -> Vec<serde_json::Value> {
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn args_value_is_empty_predicate() {
        use serde_json::json;
        // Absent / null / empty object / empty array → empty (the no-args call).
        assert!(args_value_is_empty(None));
        assert!(args_value_is_empty(Some(&json!(null))));
        assert!(args_value_is_empty(Some(&json!({}))));
        assert!(args_value_is_empty(Some(&json!([]))));
        // Any populated args → NOT empty (a with-args call is spared).
        assert!(!args_value_is_empty(Some(&json!({"query": "foo"}))));
        // A nested empty object does NOT make the whole args empty (this is why
        // the `--args ""` selector parses + checks, never substring-matches "{}").
        assert!(!args_value_is_empty(Some(&json!({"filter": {}}))));
        assert!(!args_value_is_empty(Some(&json!(["x"]))));
        assert!(!args_value_is_empty(Some(&json!("str"))));
    }

    #[test]
    fn tool_name_matches_empty_want_is_any_tool_selector() {
        fn no_map(_: &str) -> Vec<&'static str> {
            Vec::new()
        }
        // Empty `--tool-name` matches any actual tool name (prune on args alone) —
        // regardless of whether the map knows the name. This is what lets codex
        // omit `--tool-name`.
        assert!(tool_name_matches("shell", "", no_map));
        assert!(tool_name_matches("exec_command", "", no_map));
        assert!(tool_name_matches("read_file", "", no_map));
        // A non-empty want is still gated (empty map → literal compare).
        assert!(tool_name_matches("shell", "shell", no_map));
        assert!(!tool_name_matches("shell", "Bash", no_map));
    }

    #[test]
    fn glob_leaf_match_segments_in_order() {
        // No `*` → plain substring.
        assert!(glob_leaf_match("src/main.rs", "main"));
        assert!(!glob_leaf_match("src/main.rs", "nope"));
        // `*`-separated segments must appear IN ORDER within the one leaf.
        assert!(glob_leaf_match(
            "SELECT … analytics_186081460 … WHERE … 2026-05-25 … organic",
            "analytics_186081460*WHERE*2026-05-25*organic"
        ));
        // Same segments out of order → no match (ordering is load-bearing).
        assert!(!glob_leaf_match(
            "organic … WHERE … analytics_186081460",
            "analytics_186081460*WHERE"
        ));
        // A later segment must follow the previous one's END, not re-scan from 0.
        assert!(!glob_leaf_match("abXcd", "X*X"));
        assert!(glob_leaf_match("abXcdXef", "X*X"));
        // Leading/trailing `*` and `**` anchor nothing (empty segments skipped).
        assert!(glob_leaf_match("hello world", "*world*"));
        assert!(glob_leaf_match("hello world", "hello**world"));
    }

    #[test]
    fn value_glob_match_is_value_scoped_not_key_scoped() {
        use serde_json::json;
        // Matches a string value leaf.
        assert!(value_glob_match(
            &json!({"file_path": "src/junk.rs"}),
            "junk"
        ));
        // Recurses arrays and nested objects.
        assert!(value_glob_match(
            &json!({"q": ["a", {"deep": "needle-here"}]}),
            "needle"
        ));
        // A needle equal to a KEY name must NOT match — keys are never tested.
        assert!(!value_glob_match(
            &json!({"file_path": "src/main.rs"}),
            "file_path"
        ));
        // Non-string leaves (numbers/bools/null) carry nothing to glob.
        assert!(!value_glob_match(
            &json!({"limit": 100, "all": true}),
            "100"
        ));
        // Glob wildcard inside a value leaf.
        assert!(value_glob_match(
            &json!({"command": "grep -r foo src/ && echo bar"}),
            "grep*echo bar"
        ));
    }

    #[test]
    fn value_is_prune_context_call_detects_self_only() {
        use serde_json::json;
        // A real prune-context invocation (subcommand + required --args) anywhere
        // in the value tree → excluded.
        assert!(value_is_prune_context_call(&json!({
            "command": "\"$ZUCCHINI_SPAWNER_BIN\" prune-context --tool-name Bash --args \"*BACKLOG.md*\" --reason x"
        })));
        // codex shape: argv array, the CLI is one leaf.
        assert!(value_is_prune_context_call(&json!({
            "command": ["bash", "-lc", "zucchini-spawner prune-context --tool-name Read --args junk.rs"]
        })));
        // A normal file read that merely names a path → NOT a prune call.
        assert!(!value_is_prune_context_call(
            &json!({"command": "sed -n '1,220p' BACKLOG.md"})
        ));
        // Grepping the source for the word is conservative-safe: no --args flag,
        // so it is NOT treated as a prune call (stays prunable).
        assert!(!value_is_prune_context_call(
            &json!({"command": "grep -rn prune-context src/"})
        ));
    }

    #[test]
    fn select_eligible_ids_orders_and_subtracts_pruned() {
        use super::test_util::write_jsonl;
        // Three "calls" in document order: a, b, c. Output for `b` is already
        // `[pruned]`, so eligible = [a, c] preserving order.
        let f = write_jsonl(&[
            r#"{"kind":"call","id":"a"}"#,
            r#"{"kind":"call","id":"b"}"#,
            r#"{"kind":"output","id":"b","out":"[pruned]"}"#,
            r#"{"kind":"call","id":"c"}"#,
            r#"{"kind":"output","id":"c","out":"live"}"#,
        ]);
        let eligible = select_eligible_ids(
            f.path(),
            |entry, matched| {
                if entry.get("kind").and_then(|k| k.as_str()) == Some("call") {
                    if let Some(id) = entry.get("id").and_then(|i| i.as_str()) {
                        matched.push(id.to_string());
                    }
                }
            },
            |entry, pruned| {
                if entry.get("kind").and_then(|k| k.as_str()) == Some("output")
                    && entry.get("out").and_then(|o| o.as_str()) == Some(PRUNED_PLACEHOLDER)
                {
                    if let Some(id) = entry.get("id").and_then(|i| i.as_str()) {
                        pruned.insert(id.to_string());
                    }
                }
            },
        )
        .unwrap();
        assert_eq!(eligible, vec!["a".to_string(), "c".to_string()]);
    }

    #[test]
    fn select_eligible_ids_pruned_output_before_its_call() {
        use super::test_util::write_jsonl;
        // Order-independence: the `[pruned]` output for `x` appears BEFORE its
        // call line, yet `x` is still subtracted (collect_pruned runs on every
        // entry, retain happens after the full scan).
        let f = write_jsonl(&[
            r#"{"kind":"output","id":"x","out":"[pruned]"}"#,
            r#"{"kind":"call","id":"x"}"#,
            r#"{"kind":"call","id":"y"}"#,
        ]);
        let eligible = select_eligible_ids(
            f.path(),
            |entry, matched| {
                if entry.get("kind").and_then(|k| k.as_str()) == Some("call") {
                    if let Some(id) = entry.get("id").and_then(|i| i.as_str()) {
                        matched.push(id.to_string());
                    }
                }
            },
            |entry, pruned| {
                if entry.get("kind").and_then(|k| k.as_str()) == Some("output")
                    && entry.get("out").and_then(|o| o.as_str()) == Some(PRUNED_PLACEHOLDER)
                {
                    if let Some(id) = entry.get("id").and_then(|i| i.as_str()) {
                        pruned.insert(id.to_string());
                    }
                }
            },
        )
        .unwrap();
        assert_eq!(eligible, vec!["y".to_string()]);
    }

    #[test]
    fn last_only_target_picks_last_or_empty() {
        // Picks the greatest-document-position (last) id as a singleton.
        let set = last_only_target(vec!["a".to_string(), "b".to_string(), "c".to_string()]);
        assert_eq!(set.len(), 1);
        assert!(set.contains("c"));
        // Empty in → empty out (safe no-op for the blank pass).
        assert!(last_only_target(Vec::new()).is_empty());
        // Single in → that id.
        let one = last_only_target(vec!["only".to_string()]);
        assert_eq!(one.len(), 1);
        assert!(one.contains("only"));
    }

    #[test]
    fn pruned_frame_json_formats_summary() {
        // 1 output, ~12k freed → singular noun + `~Nk` rounding.
        let s = pruned_frame_json(PruneStats {
            results_blanked: 1,
            freed_bytes: 48_000, // /4 = 12_000 tokens → ~12k
        });
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["type"], "system");
        assert_eq!(v["subtype"], "context_pruned");
        assert_eq!(v["count"], 1);
        assert_eq!(v["freed_tokens"], 12_000);
        assert_eq!(v["summary"], "pruned 1 tool output · ~12k freed");

        // Plural noun + sub-1k freed → `~N` (no k).
        let s2 = pruned_frame_json(PruneStats {
            results_blanked: 3,
            freed_bytes: 2_000, // /4 = 500 tokens
        });
        let v2: serde_json::Value = serde_json::from_str(&s2).unwrap();
        assert_eq!(v2["summary"], "pruned 3 tool outputs · ~500 freed");

        // Zero freed → omit the freed clause entirely.
        let s3 = pruned_frame_json(PruneStats {
            results_blanked: 2,
            freed_bytes: 0,
        });
        let v3: serde_json::Value = serde_json::from_str(&s3).unwrap();
        assert_eq!(v3["summary"], "pruned 2 tool outputs");
    }

    #[test]
    fn respawn_prompt_states_success() {
        let p = pruned_respawn_prompt(PruneStats {
            results_blanked: 1,
            freed_bytes: 48_000,
        });
        // Carries the same human summary as the timeline frame …
        assert!(p.contains("pruned 1 tool output · ~12k freed"), "{p}");
        // … and explicitly tells the respawned agent the prune worked (the loop
        // guard this prompt exists to provide).
        assert!(p.contains("succeeded"), "{p}");
    }

    #[test]
    fn prune_reminder_fires_for_large_string_tool_response() {
        let big = "x".repeat(PRUNE_REMINDER_MIN_TOOL_RESPONSE_BYTES + 1);
        let payload = serde_json::json!({
            "hook_event_name": "PostToolUse",
            "tool_name": "Read",
            "tool_response": big,
        })
        .to_string();
        let out = prune_reminder_output(&payload).expect("should fire over threshold");
        // Emitted JSON carries the exact reminder text + the proven event-name shape.
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["hookSpecificOutput"]["hookEventName"], "PostToolUse");
        assert_eq!(
            v["hookSpecificOutput"]["additionalContext"],
            PRUNE_REMINDER_TEXT
        );
        assert!(out.contains(PRUNE_REMINDER_TEXT), "{out}");
    }

    #[test]
    fn prune_reminder_fires_for_large_object_tool_response() {
        // An object whose serialization exceeds the threshold also fires.
        let big_obj = serde_json::json!({
            "files": (0..200).map(|i| format!("path/to/file_{i}.rs")).collect::<Vec<_>>(),
        });
        let payload = serde_json::json!({ "tool_response": big_obj }).to_string();
        assert!(big_obj.to_string().len() > PRUNE_REMINDER_MIN_TOOL_RESPONSE_BYTES);
        assert!(prune_reminder_output(&payload).is_some());
    }

    #[test]
    fn prune_reminder_silent_for_small_tool_response() {
        let payload = serde_json::json!({ "tool_response": "tiny" }).to_string();
        assert!(prune_reminder_output(&payload).is_none());
    }

    #[test]
    fn prune_reminder_silent_for_missing_tool_response() {
        let payload = serde_json::json!({ "tool_name": "Bash" }).to_string();
        assert!(prune_reminder_output(&payload).is_none());
    }

    #[test]
    fn prune_reminder_silent_for_malformed_payload() {
        // Non-JSON garbage must not panic and must stay silent.
        assert!(prune_reminder_output("not json at all {[").is_none());
        assert!(prune_reminder_output("").is_none());
    }

    #[test]
    fn prune_reminder_silent_for_subagent_payload() {
        // A Task sub-agent's PostToolUse payload carries `agent_id`/`agent_type`
        // (the main agent's carries neither). Even with an over-threshold
        // `tool_response`, the nudge must stay silent — a sub-agent must not
        // prune (no how-to, ephemeral context, and a prune would hit the PARENT
        // transcript). Shape mirrors a captured real payload.
        let big = "x".repeat(PRUNE_REMINDER_MIN_TOOL_RESPONSE_BYTES + 1);
        let payload = serde_json::json!({
            "hook_event_name": "PostToolUse",
            "tool_name": "Read",
            "tool_response": big,
            // Parent and sub-agent SHARE these — they cannot distinguish.
            "session_id": "9fe8c87f-abfa-4f06-9527-23807d6bc871",
            "transcript_path": "/h/.claude/projects/p/9fe8c87f.jsonl",
            // The sub-agent markers — presence of `agent_id` is the gate.
            "agent_id": "ae75401ac3661b0a5",
            "agent_type": "general-purpose",
        })
        .to_string();
        assert!(
            prune_reminder_output(&payload).is_none(),
            "sub-agent payload must not fire the prune nudge"
        );

        // Same payload WITHOUT the sub-agent markers (a main-agent call) fires.
        let main = serde_json::json!({
            "hook_event_name": "PostToolUse",
            "tool_name": "Read",
            "tool_response": "x".repeat(PRUNE_REMINDER_MIN_TOOL_RESPONSE_BYTES + 1),
            "session_id": "9fe8c87f-abfa-4f06-9527-23807d6bc871",
            "transcript_path": "/h/.claude/projects/p/9fe8c87f.jsonl",
        })
        .to_string();
        assert!(
            prune_reminder_output(&main).is_some(),
            "main-agent payload (no agent_id) must still fire"
        );
    }

    #[test]
    fn prune_reminder_string_gate_is_strict_greater_than() {
        // The gate is `> PRUNE_REMINDER_MIN_TOOL_RESPONSE_BYTES`, measured against
        // a string's CONTENT length. Exactly at the threshold → silent; one byte
        // over → fires.
        let at = "x".repeat(PRUNE_REMINDER_MIN_TOOL_RESPONSE_BYTES);
        let at_payload = serde_json::json!({ "tool_response": at }).to_string();
        assert!(
            prune_reminder_output(&at_payload).is_none(),
            "exactly {PRUNE_REMINDER_MIN_TOOL_RESPONSE_BYTES} bytes must NOT fire"
        );

        let over = "x".repeat(PRUNE_REMINDER_MIN_TOOL_RESPONSE_BYTES + 1);
        let over_payload = serde_json::json!({ "tool_response": over }).to_string();
        assert!(
            prune_reminder_output(&over_payload).is_some(),
            "{} bytes must fire",
            PRUNE_REMINDER_MIN_TOOL_RESPONSE_BYTES + 1
        );
    }

    #[test]
    fn prune_reminder_handles_scalar_tool_response() {
        // A scalar `tool_response` (number / bool) exercises the `Some(other)`
        // serialize arm of `prune_reminder_output`: it must not panic, and its
        // short serialized form (`42`, `true`) is well under the threshold →
        // silent. (Tool responses are practically always strings or
        // objects/arrays, but the `Some(other)` arm must stay panic-free for any
        // JSON shape claude might hand the hook.)
        let num = serde_json::json!({ "tool_response": 42 }).to_string();
        assert!(prune_reminder_output(&num).is_none());
        let boolean = serde_json::json!({ "tool_response": true }).to_string();
        assert!(prune_reminder_output(&boolean).is_none());
    }
}
