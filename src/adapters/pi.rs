//! pi (`earendil-works/pi`, the `pi` coding-agent CLI) adapter. One-shot,
//! fork-per-turn like codex/gemini/cursor. Normalizes pi's `--mode json` JSONL
//! event stream into claude-shape envelopes on the wire so iOS's
//! `SpawnerMessageDescriber` (which only knows claude's wire format) renders
//! them WITHOUT any pi-specific branches — every pi tool call is remapped to the
//! equivalent claude tool name so iOS's `toolSummary` table picks it up as if it
//! were a claude `tool_use`. Single seam, same posture as `codex.rs`/`gemini.rs`.
//!
//! Spawn shape (verified against pi 0.79.9):
//!
//!   cd <proj> && cat <prompt-file> | pi -p --mode json --session-id <uuid> \
//!       [--model <m>] (--approve | --no-approve) --append-system-prompt <caps>
//!
//! Worktree (`ctx.worktree`): pi has NO `--worktree` flag, so the SPAWNER creates
//! the worktree itself and runs pi WITH THE WORKTREE AS cwd. The dir is
//! DETERMINISTIC from the chat id — `~/.zucchini-spawner/worktrees/<chat8>`
//! (`<chat8>` = first 8 chars of `chat_id`, same scheme as claude's worktree
//! name) — so it's recomputable every turn with no stored state, and it lives
//! OUTSIDE any repo so it never pollutes the parent's `git status`. Branch is
//! `zc-<chat8>`. The `git worktree add` runs ONLY on the first turn
//! (`agent_session_id.is_none()`); on resume we just cd into the recomputed path.
//! Because pi's cwd IS the worktree, the model naturally operates there, so we
//! deliberately DO NOT inject any "stay inside the worktree" containment block
//! (claude needs that only because its `--worktree` doesn't make cwd obvious).
//! A failed `git worktree add` propagates to stderr → the Supervisor surfaces it
//! as the turn's failure (and the next message retries, since a failed first turn
//! harvests no session id). Worktree only engages when `ctx.worktree && project
//! path is Some`; otherwise the cwd is `cd <proj>` (or nothing) as before:
//!
//!   git -C <proj> worktree add <wt> -b zc-<chat8> && cd <wt> && cat … | pi …  (first turn)
//!   cd <wt> && cat … | pi …                                                   (resume)
//!
//! Spawn shape details:
//!
//! - The prompt is piped via stdin (multi-MB attachment-laden bodies must not go
//!   through argv); `pi -p` with no positional prompt reads stdin as the prompt
//!   (verified).
//! - `--session-id <uuid>` is "use exact project session id, creating it if
//!   missing" — so the SAME flag both MINTS on the first turn and RESUMES on
//!   later turns. pi adopts the id verbatim and echoes it in the `session`
//!   header (unlike gemini, there's no ephemeral-fork id to dodge), so resume is
//!   exact. We mint a UUID per session and reuse the harvested
//!   `chats.agent_session_id` on resume.
//! - pi HAS `--append-system-prompt` (unlike gemini), so capabilities ride a
//!   real system prompt every turn (claude-style) rather than the first-message
//!   prepend path. `prompt_file_preamble` already defaults to `None` (so the
//!   capability block isn't double-injected); we additionally override
//!   `prompt_file_time_line` to `None` (its default returns the time line) so the
//!   time isn't prepended on top of the system-prompt path either.
//!
//! Frame mapping (pi `--mode json` → claude-shape), observed empirically:
//!
//! - `session` → `SessionIdHarvested` only (no Frame; matches claude's init-skip),
//!   FIRST turn only. The id equals the UUID we passed via `--session-id`.
//! - `message_end` role=assistant → for each content block IN ORDER: `text` →
//!   one claude assistant text envelope; `toolCall` → one claude tool_use
//!   envelope under the mapped claude tool name (`normalize_tool_use`);
//!   `thinking` → dropped (claude UI doesn't surface a separate reasoning bubble
//!   for non-claude adapters). Then the message's `usage` → `ContextTokens`
//!   (`input + cacheRead + cacheWrite` = the conversation actually sent to the
//!   model; deduped). We key off `message_end` (the FULL message) rather than the
//!   streamed `message_update` deltas so a reply lands as ONE immutable
//!   `messages` row per block instead of fragmenting into one bubble per delta
//!   (the message-frame invariant — see crate `CLAUDE.md`).
//! - `message_end` role=user / role=toolResult → dropped (echo of our prompt /
//!   tool output; claude UI shows tool_use only, infers the result — matches
//!   codex/gemini).
//! - `agent_end` → claude-shape success `result` envelope (carrying a
//!   spawner-measured `duration_ms` — pi's stream has no native turn-duration
//!   field, so we clock it from the first frame) + `Result`. This is the
//!   terminal event of a `-p` run (the model→tools→model loop is wrapped by one
//!   agent_start/agent_end). Hard errors (bad model, auth failure) print to
//!   STDERR and exit non-zero with NO stdout frame; the Supervisor surfaces that
//!   stderr as the turn's failure message, so we need no error-frame branch here.
//! - everything else (`agent_start`, `turn_*`, `message_start`, `message_update`,
//!   `tool_execution_*`, `queue_update`, `compaction_*`, `auto_retry_*`) → DROPPED
//!   (debug log). Unlike codex/gemini we do NOT default-forward unknown types:
//!   pi emits a rich set of internal lifecycle events and forwarding them would
//!   spam the chat with frames iOS can't render. New event types are dropped
//!   with a breadcrumb instead.
//!
//! Oversize frames (`line.len() > MAX_STREAM_FRAME_BYTES`): pi frames are NEVER
//! wire-compatible with iOS (we always transform), so forwarding a raw pi line
//! verbatim is wrong — iOS renders an unknown `{"type":"X"}` as the literal text
//! `[X]`. pi's lifecycle events EMBED full content (a >64KB `tool_execution_end`
//! / `turn_end` / `agent_end` is normal on a big turn), so the oversize path is
//! TYPE-AWARE: we cheaply classify via `extract_json_type` (no full parse) and
//! handle each like its in-band arm WITHOUT paying the parse cost. The critical
//! case is an oversize `agent_end` — it must still emit the synthesized
//! `result` frame + `Result` marker (a fixed, body-independent envelope), else
//! `has_result` stays false and the user sees "Agent interrupted" instead of
//! `[result: success]`. `message_end` (the only content-bearing case that needs
//! the body to transform) falls through to the normal full-parse arm; every
//! other type — and an unclassifiable line — is DROPPED with a breadcrumb
//! (raw pi JSON is never renderable, so dropping beats forwarding).
//!
//! Sandbox: pi headless has no per-tool approval prompt (it just runs its
//! read/bash/edit/write tools), so there is no strong sandbox lever. We gate the
//! project-trust flag on `!is_sandboxed` (`--approve` for owners, `--no-approve`
//! for sandboxed invitees so project-local extension/AGENTS.md files aren't
//! auto-trusted) — weaker than claude's, same thin-spawn posture as cursor's
//! sandbox caveat. Tool execution itself is not gated; communicated via UI
//! disclaimer, not a Zucchini-side wrapper.
//!
//! prune-context: SUPPORTED (`prune: Some(PRUNE_OPS)`). pi sessions are JSONL
//! under `~/.pi/agent/sessions/<canonical-cwd-mangled>/<ts>_<session-id>.jsonl`
//! (the filename embeds the session id; `find_session` walks recursively beneath
//! the base and matches `_<session_id>.jsonl`). Verified empirically that pi
//! `--resume` reloads an externally-edited transcript verbatim, so blank-in-place
//! forgetting works exactly like claude/gemini/codex: an assistant `toolCall`
//! block is the call, a paired `toolResult` line (matched by `toolCallId`) holds
//! the output we blank to `[pruned]`. The standing prune nudge rides
//! `--append-system-prompt` every turn (claude-shape `PRUNE_CONTEXT_INSTRUCTION`,
//! since pi maps tool names to claude names on the wire). See the prune section.
//!
//! Auth probe: `pi` on PATH + (`~/.pi/agent/auth.json` non-empty OR a provider
//! API key in env).
//!
//! History `import()` walks `~/.pi/agent/sessions/*/*.jsonl` (one file per
//! session). Each file's line 0 is `{"type":"session","id":...,"cwd":...}`: `cwd`
//! is the project path (mint the project id from it) and `id` is the session id →
//! the chat's `agent_session_id` AND the chat id (parsed as a UUID, like gemini).
//! Body lines are `{"type":"message","timestamp":...,"message":{role,content}}`:
//! `role=="user"` → a user prompt (text blocks concatenated); `role=="assistant"`
//! → assistant text + `toolCall` blocks (mapped via the live `normalize_tool_use`,
//! `thinking` dropped); `role=="toolResult"` → DROPPED (claude UI infers results
//! from the tool_use). Same claude-shape frames the live `handle_line` emits, so
//! imported and live rows render identically. See `import`.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use serde_json::{json, Value};
use smallvec::SmallVec;
use tokio::io::AsyncBufReadExt;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::adapter::{
    agent_capabilities_instructions, claude_assistant_text_envelope, claude_tool_use_envelope,
    current_time_in_tz_line, extract_json_type, file_nonempty, parse_json_obj,
    probe_with_blocking_auth, shell_escape, worktree_cwd_prefix, AdapterDescriptor, AgentAdapter,
    AgentEvent, AgentKind, ImportProgress, LastTokensDedup, TurnContext, MAX_STREAM_FRAME_BYTES,
    PRUNE_CONTEXT_INSTRUCTION,
};
use crate::adapters::import_shared::{
    basename_or, collapse_title, emit_chat, is_synthetic_wrapper, mint_project_id,
    parse_rfc3339_utc, user_message_body, ImportedChat, ImportedMessage, ProgressThrottle,
};
use crate::prune::{PruneOps, PruneStats, PruneTarget, PRUNED_PLACEHOLDER};
use crate::writer::WriteEvent;

/// Wired into `adapter::ADAPTERS`. `installed_col` / `authenticated_col` follow
/// the per-kind nullable-BOOLEAN pair convention (`gemini_*` 0039, `codex_*`
/// 0037, …); the matching `pi_*` columns land in migration 0046. `prune` is
/// wired (`PRUNE_OPS`) — pi `--resume` reloads an externally-edited transcript
/// verbatim, so blank-in-place forgetting works like claude/gemini/codex.
pub const DESCRIPTOR: AdapterDescriptor = AdapterDescriptor {
    kind: AgentKind::Pi,
    wire_name: "pi",
    installed_col: "pi_installed",
    authenticated_col: "pi_authenticated",
    make: make_boxed,
    probe: probe_boxed,
    import: import_boxed,
    prune: Some(PRUNE_OPS),
};

fn make_boxed() -> Box<dyn AgentAdapter> {
    Box::new(PiAdapter::new())
}

/// Per-turn state (adapter is constructed fresh per turn). `session_id` is the
/// UUID minted for this chat, passed via `--session-id` on the first turn;
/// `last_emitted_tokens` dedups a repeated context-token reading;
/// `resumed` gates the `session`-header harvest to the first turn only.
pub struct PiAdapter {
    session_id: Uuid,
    last_emitted_tokens: LastTokensDedup,
    resumed: bool,
    /// Wall-clock anchor for the turn's run time, set lazily on the FIRST
    /// observed stdout frame (so it excludes the shell-rc + pi cold-start spawn
    /// overhead that precedes any output). Elapsed at `agent_end` becomes the
    /// synthesized result frame's `duration_ms` → iOS/Android render
    /// `[result: success (Ns)]`. pi's own stream has no native duration field
    /// (unlike claude/cursor, which pass theirs through), so we measure it.
    started: Option<Instant>,
}

impl Default for PiAdapter {
    fn default() -> Self {
        Self {
            session_id: Uuid::now_v7(),
            last_emitted_tokens: LastTokensDedup::default(),
            resumed: false,
            started: None,
        }
    }
}

impl PiAdapter {
    pub fn new() -> Self {
        Self::default()
    }
}

impl AgentAdapter for PiAdapter {
    fn kind(&self) -> AgentKind {
        AgentKind::Pi
    }

    fn prepare_command(&mut self, ctx: &TurnContext<'_>) -> Result<String> {
        let mut cmd = String::new();

        // cwd selection. pi has NO `--worktree` flag, so when worktree mode is on
        // (and we have a project path) the SPAWNER creates the worktree itself and
        // runs pi WITH THE WORKTREE AS cwd — the model then naturally operates
        // there, so (unlike claude) no "stay inside the worktree" containment
        // guidance is needed. The entire worktree decision (deterministic dir
        // `~/.zucchini-spawner/worktrees/<chat8>`, branch `zc-<chat8>`, first-turn
        // `git worktree add` vs resume `cd`, plain `cd <proj>` when worktree is
        // off) lives in the shared `worktree_cwd_prefix` helper so codex/gemini/
        // hermes can adopt it with the same one-liner.
        cmd.push_str(&worktree_cwd_prefix(ctx));

        // Prompt piped via stdin — `pi -p` with no positional prompt reads stdin
        // as the prompt (verified). Keeps multi-MB attachment bodies out of argv.
        cmd.push_str(&format!(
            "cat {} | pi -p --mode json",
            shell_escape(&ctx.prompt_file.to_string_lossy()),
        ));

        // Session id. `--session-id` creates-if-missing, so the SAME flag mints
        // on turn 1 and resumes thereafter. On resume mark `self.resumed` so the
        // `session` header harvest is skipped (the id is already canonical).
        let sid = match ctx.agent_session_id {
            Some(s) => {
                self.resumed = true;
                s.to_string()
            }
            None => self.session_id.to_string(),
        };
        cmd.push_str(&format!(" --session-id {}", shell_escape(&sid)));

        // Verbatim pass-through of `chats.model` (migration 0035). pi uses
        // `--model` and accepts a `provider/id` form; we don't validate it — an
        // invalid value surfaces as a pi error on stderr (which the Supervisor
        // turns into the turn's failure message).
        if let Some(model) = ctx.model {
            cmd.push_str(&format!(" --model {}", shell_escape(model)));
        }

        // Sandbox → project-trust mapping. pi headless has no per-tool approval
        // gate, so this only governs whether project-local files (extensions,
        // AGENTS.md) are auto-trusted: owners (`!is_sandboxed`) → `--approve`,
        // sandboxed invitees → `--no-approve`. Weaker than claude's sandbox; the
        // gap is communicated via UI disclaimer (thin-spawn scope).
        if ctx.is_sandboxed {
            cmd.push_str(" --no-approve");
        } else {
            cmd.push_str(" --approve");
        }

        // Capabilities ride `--append-system-prompt` every turn (claude-style):
        // attach-file + schedule-message + the fresh per-turn time line, then the
        // standing prune-context nudge. The first
        // `agent_capabilities_instructions` arg is the worktree-containment block
        // and is INTENTIONALLY `None` here: when worktree mode is on, pi already
        // runs with the worktree AS cwd (see the cwd selection above), so the
        // "stay inside the worktree" guidance claude needs (its `--worktree`
        // doesn't make cwd obvious) is unnecessary for pi. The CLAUDE-shape
        // `PRUNE_CONTEXT_INSTRUCTION` is correct here: pi maps its tool names to
        // claude names on the wire AND the prune CLI takes claude-shape
        // `--tool-name`, so the Read/Bash examples match what the user types — the
        // nudge is appended after, exactly like claude.
        let mut sys = agent_capabilities_instructions(
            None,
            current_time_in_tz_line(ctx.user_timezone).as_deref(),
        );
        sys.push_str("\n\n");
        sys.push_str(PRUNE_CONTEXT_INSTRUCTION);
        cmd.push_str(&format!(" --append-system-prompt {}", shell_escape(&sys)));

        Ok(cmd)
    }

    /// Capabilities go via `--append-system-prompt` (built in `prepare_command`),
    /// not the first-message prepend path. `prompt_file_preamble` already defaults
    /// to `None`; override the time line to `None` too (its default prepends the
    /// current-time line) so nothing rides the prompt-file prepend path.
    fn prompt_file_time_line(&self, _ctx: &TurnContext<'_>) -> Option<String> {
        None
    }

    fn handle_line(&mut self, line: String) -> SmallVec<[AgentEvent; 2]> {
        let mut out: SmallVec<[AgentEvent; 2]> = SmallVec::new();

        // Anchor the turn clock on the FIRST frame (the `session` header in
        // practice), so the result's `duration_ms` reflects pi's run time and
        // not the spawn overhead that precedes any stdout.
        let started = *self.started.get_or_insert_with(Instant::now);

        // Oversize-frame fast-path. pi frames are NEVER iOS-wire-compatible (we
        // always transform), so forwarding a raw pi line verbatim is wrong — iOS
        // renders an unknown `{"type":"X"}` as literal `[X]`. The ONLY reason the
        // size cap exists is to skip a full `serde_json::Value` parse (heap
        // churn) on a multi-MB line. So classify CHEAPLY via `extract_json_type`
        // (no parse) and branch:
        if line.len() > MAX_STREAM_FRAME_BYTES {
            match extract_json_type(&line) {
                // CRITICAL: a >64KB agent_end (it embeds the final assistant
                // message + usage) must STILL emit the terminal result. The
                // envelope is fixed/body-independent, so zero parse needed —
                // dropping it here is what made the user see "Agent interrupted".
                Some("agent_end") => {
                    push_agent_end_terminal(&mut out, started.elapsed().as_millis() as i64);
                    return out;
                }
                // The only content-bearing oversize case: a >64KB assistant
                // message_end needs the (huge) body to transform into chat rows.
                // A >64KB assistant message is rare and correctness beats the
                // one-time heap blip, so DON'T early-return — fall through to the
                // normal full-parse `"message_end"` arm below.
                Some("message_end") => {}
                // Lifecycle (turn_*, message_start, tool_execution_*,
                // message_update, queue_update, compaction_*, auto_retry_*,
                // agent_start) + any unknown type → DROP. Forwarding raw pi JSON
                // is strictly worse than dropping (iOS can't render it).
                Some(other) => {
                    debug!(ty = %other, "pi oversize lifecycle frame dropped");
                    return out;
                }
                // Couldn't classify (no `"type"` near the start, escaped value,
                // …). Drop: raw pi JSON is never renderable, so a forward would
                // only surface garbage. Safe fallback.
                None => {
                    debug!("pi oversize frame without classifiable type dropped");
                    return out;
                }
            }
        }

        let Some(obj) = parse_json_obj(&line) else {
            // Non-JSON noise on stdout: drop (pi's stdout is pure JSONL; any
            // stray line is not renderable). pi errors land on stderr.
            debug!("pi non-json stdout line dropped");
            return out;
        };
        let Some(ty) = obj.get("type").and_then(|v| v.as_str()) else {
            debug!("pi json line without type, dropped");
            return out;
        };

        match ty {
            "session" => {
                // Harvest the session id → `chats.agent_session_id`; drop the
                // frame. FIRST turn only: it equals the UUID we minted, so this
                // persists the canonical resumable id. On resume we already know
                // the id (we passed it), so skip — re-harvesting the identical
                // value would only churn a redundant writeback.
                if !self.resumed {
                    if let Some(id) = obj.get("id").and_then(|v| v.as_str()) {
                        out.push(AgentEvent::SessionIdHarvested(id.to_string()));
                    } else {
                        debug!("pi session header without id");
                    }
                }
            }
            "message_end" => {
                let Some(msg) = obj.get("message") else {
                    debug!("pi message_end without message");
                    return out;
                };
                // Only assistant messages produce chat content. user (our prompt
                // echo) and toolResult (claude UI infers results) are dropped.
                if msg.get("role").and_then(|v| v.as_str()) != Some("assistant") {
                    return out;
                }
                if let Some(blocks) = msg.get("content").and_then(|v| v.as_array()) {
                    for block in blocks {
                        match block.get("type").and_then(|v| v.as_str()) {
                            Some("text") => {
                                let text = block.get("text").and_then(|v| v.as_str()).unwrap_or("");
                                if !text.is_empty() {
                                    out.push(AgentEvent::Frame(claude_assistant_text_envelope(
                                        text,
                                    )));
                                }
                            }
                            Some("toolCall") => {
                                if let Some(frame) = normalize_tool_use(block) {
                                    out.push(AgentEvent::Frame(frame));
                                }
                            }
                            // thinking / unknown blocks: dropped.
                            _ => {}
                        }
                    }
                }
                // Context tokens: what was actually sent to the model this turn =
                // fresh input + cached read + cached write. Deduped so a repeated
                // value across messages doesn't re-fire.
                if let Some(tokens) = context_tokens_from_usage(msg.get("usage")) {
                    if let Some(t) = self.last_emitted_tokens.observe(tokens) {
                        out.push(AgentEvent::ContextTokens(t));
                    }
                }
            }
            "agent_end" => {
                // Terminal — same fixed output as the oversize fast-path above.
                push_agent_end_terminal(&mut out, started.elapsed().as_millis() as i64);
            }
            other => {
                // pi's many internal lifecycle events (turn_*, message_start,
                // message_update deltas, tool_execution_*, queue_update,
                // compaction_*, auto_retry_*). Dropped — not renderable as chat
                // rows. Breadcrumb so a genuinely new content-bearing type is
                // noticed without spamming the chat.
                debug!(ty = %other, "pi event type dropped");
            }
        }

        out
    }
}

/// Terminal output for a pi `agent_end`: a claude-shape success `result`
/// envelope (so iOS renders `[result: success (Ns)]`) + the `Result` marker (so
/// the Supervisor latches `Done.has_result = true`). Body-independent except for
/// `duration_ms` (the spawner-measured turn run time, since pi's stream carries
/// no native duration) — emitted from BOTH the normal `"agent_end"` arm and the
/// oversize fast-path, so this is the single source (repo rule: never copy-paste
/// the result-frame block). iOS/Android render the `(Ns)` suffix from
/// `duration_ms`; absent/zero just yields `(0s)`.
fn push_agent_end_terminal(out: &mut SmallVec<[AgentEvent; 2]>, duration_ms: i64) {
    out.push(AgentEvent::Frame(
        json!({
            "type": "result",
            "subtype": "success",
            "is_error": false,
            "duration_ms": duration_ms,
        })
        .to_string(),
    ));
    out.push(AgentEvent::Result {
        origin_is_task: false,
    });
}

/// Context occupancy from pi's assistant `usage` block:
/// `input + cacheRead + cacheWrite` — the tokens actually sent to the model
/// (fresh + cached context), which mirrors claude's context counter. `output`
/// and `totalTokens` are excluded (output isn't context). Returns `None` when no
/// usage is present.
fn context_tokens_from_usage(usage: Option<&Value>) -> Option<i64> {
    let u = usage?;
    let get = |k: &str| u.get(k).and_then(|v| v.as_i64()).unwrap_or(0);
    let total = get("input") + get("cacheRead") + get("cacheWrite");
    (total > 0).then_some(total)
}

/// Maps one pi `toolCall` content block (`{type:toolCall, id, name, arguments}`)
/// to a claude-shape `tool_use` envelope under the equivalent claude tool name,
/// remapping argument keys to the ones iOS's `toolSummary` reads (e.g. pi `path`
/// → claude `file_path`). Unknown tools pass through under their own name with
/// verbatim arguments so nothing is silently lost. Returns `None` only when the
/// block has no usable id.
fn normalize_tool_use(block: &Value) -> Option<String> {
    let id = block.get("id").and_then(|v| v.as_str())?;
    let name = block.get("name").and_then(|v| v.as_str()).unwrap_or("");
    let args = block.get("arguments");

    // Helpers to lift a single string arg out of pi's `arguments` object.
    let arg = |k: &str| {
        args.and_then(|a| a.get(k))
            .and_then(|v| v.as_str())
            .unwrap_or("")
    };

    let (claude_name, input): (&str, Value) = match name {
        "read" => ("Read", json!({ "file_path": arg("path") })),
        "write" => (
            "Write",
            json!({ "file_path": arg("path"), "content": arg("content") }),
        ),
        "edit" => ("Edit", json!({ "file_path": arg("path") })),
        "bash" => ("Bash", json!({ "command": arg("command") })),
        "grep" => ("Grep", json!({ "pattern": arg("pattern") })),
        // pi `ls` has no claude twin; render it as a `Bash` `ls <path>` so iOS
        // shows a recognizable command line.
        "ls" => ("Bash", json!({ "command": format!("ls {}", arg("path")) })),
        "web_fetch" => ("WebFetch", json!({ "url": arg("url") })),
        "web_search" => ("WebSearch", json!({ "query": arg("query") })),
        // Unknown / extension tool: keep pi's own name + verbatim args.
        _ => (name, args.cloned().unwrap_or_else(|| json!({}))),
    };

    Some(claude_tool_use_envelope(id, claude_name, input))
}

/// Filesystem-only probe — same shape as codex/gemini. `installed = (pi on
/// PATH)`, `authenticated = (~/.pi/agent/auth.json non-empty OR a provider API
/// key in env)`. pi supports many providers; we check the common key names plus
/// the on-disk auth blob rather than enumerate every provider.
pub async fn probe() -> (bool, bool) {
    probe_with_blocking_auth("pi", is_authenticated).await
}

fn probe_boxed() -> futures::future::BoxFuture<'static, (bool, bool)> {
    Box::pin(probe())
}

fn is_authenticated() -> bool {
    // On-disk credential store written by pi's `/login` / `--api-key`.
    if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
        if file_nonempty(&home.join(".pi").join("agent").join("auth.json")) {
            return true;
        }
    }
    // Or a provider API key in the environment (pi reads these directly).
    const KEY_VARS: &[&str] = &[
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "OPENROUTER_API_KEY",
        "MOONSHOT_API_KEY",
    ];
    KEY_VARS
        .iter()
        .any(|k| std::env::var_os(k).is_some_and(|v| !v.is_empty()))
}

/// One-shot history import. Triggered once, immediately after a machine is
/// added; iOS blocks on the import-progress sheet so no live agent contends for
/// the writer channel (same contract as the gemini/claude importers).
///
/// pi stores one session per file at
/// `~/.pi/agent/sessions/<cwd-mangled>/<ts>_<session-id>.jsonl`. We do NOT
/// un-mangle the directory name — the real project path is line 0's `cwd` and the
/// resumable session id is line 0's `id`. We walk every `.jsonl` two levels down,
/// group sessions by their `cwd`, mint one project per path (shared
/// `mint_project_id`), and emit PutProject + (per session) PutChat/PutMessage.
pub(crate) async fn import(
    machine_id: Uuid,
    user_id: Uuid,
    write_tx: mpsc::Sender<WriteEvent>,
    progress: ImportProgress,
) -> Result<()> {
    let Some(sessions_dir) = pi_sessions_dir() else {
        info!("HOME not set, skipping pi history import");
        progress(100).await;
        return Ok(());
    };
    info!(path = %sessions_dir.display(), "scanning pi transcripts");

    // Collect every session `.jsonl` (one file per session) under the per-cwd
    // subdirs. Grouping by resolved project path happens after we read each
    // header (`cwd`), so first gather a flat file list. A missing dir (never run /
    // fresh install) surfaces as the `read_dir` NotFound arm below — early-out at
    // 100% so the dispatcher's per-kind slice closes cleanly.
    let mut files: Vec<std::path::PathBuf> = Vec::new();
    let outer = match std::fs::read_dir(&sessions_dir) {
        Ok(d) => d,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            info!(path = %sessions_dir.display(), "no ~/.pi/agent/sessions, nothing to import");
            progress(100).await;
            return Ok(());
        }
        Err(e) => return Err(e).with_context(|| format!("read_dir {}", sessions_dir.display())),
    };
    for project in outer.flatten() {
        if !project.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            continue;
        }
        let Ok(inner) = std::fs::read_dir(project.path()) else {
            continue;
        };
        for f in inner.flatten() {
            let p = f.path();
            if p.extension().and_then(|s| s.to_str()) == Some("jsonl") {
                files.push(p);
            }
        }
    }

    if files.is_empty() {
        info!("pi: no .jsonl transcripts found");
        progress(100).await;
        return Ok(());
    }
    // Stable order across re-runs (and for logs).
    files.sort();
    let total_sessions = files.len();
    info!(sessions = total_sessions, "starting pi import");

    // Emit each project's PutProject exactly once, the first time a session
    // resolves to that path (the shared `mint_project_id` is deterministic, so a
    // re-import converges on the same project row regardless).
    let mut seen_projects: BTreeMap<String, Uuid> = BTreeMap::new();
    let mut done_sessions: usize = 0;
    let mut throttle = ProgressThrottle::new();
    for jsonl in files {
        match import_session(&jsonl, machine_id, user_id, &write_tx, &mut seen_projects).await {
            Ok(()) => {}
            Err(e) => {
                warn!(file = %jsonl.display(), error = %e, "pi session import failed, skipping")
            }
        }
        done_sessions += 1;
        throttle
            .step(done_sessions, total_sessions, &progress)
            .await;
    }

    info!(sessions = done_sessions, "pi history import complete");
    Ok(())
}

fn pi_sessions_dir() -> Option<std::path::PathBuf> {
    let home = std::env::var_os("HOME")?;
    Some(
        std::path::PathBuf::from(home)
            .join(".pi")
            .join("agent")
            .join("sessions"),
    )
}

/// Parse one pi session `.jsonl`: line 0 (`type:"session"`) → project path
/// (`cwd`) + chat id/`agent_session_id` (`id`), body `type:"message"` lines →
/// kept frames (claude-shape, via the SAME `normalize_tool_use` the live adapter
/// uses), then emit PutProject (once per path) + PutChat + one PutMessage per
/// frame. Skips files whose header `id` isn't UUID-shaped or that yield no
/// messages.
///
/// `seen_projects` dedups the per-path PutProject across sessions sharing a `cwd`.
async fn import_session(
    jsonl: &std::path::Path,
    machine_id: Uuid,
    user_id: Uuid,
    write_tx: &mpsc::Sender<WriteEvent>,
    seen_projects: &mut BTreeMap<String, Uuid>,
) -> Result<()> {
    let file = tokio::fs::File::open(jsonl)
        .await
        .with_context(|| format!("open {}", jsonl.display()))?;
    let mut lines = tokio::io::BufReader::new(file).lines();

    // Line 0 = the session header carrying `id` (the session UUID → chat id +
    // agent_session_id) and `cwd` (the project path the agent ran in).
    let header_line = match lines.next_line().await? {
        Some(l) => l,
        None => return Ok(()), // empty file
    };
    let header: Value = serde_json::from_str(&header_line)
        .with_context(|| format!("session header not JSON: {}", jsonl.display()))?;
    if header.get("type").and_then(|v| v.as_str()) != Some("session") {
        return Ok(()); // not a session file (no header) — skip silently
    }
    let Some(session_id) = header.get("id").and_then(|v| v.as_str()) else {
        return Ok(());
    };
    let chat_id = Uuid::parse_str(session_id)
        .with_context(|| format!("session id is not a UUID: {session_id}"))?;
    let Some(cwd) = header.get("cwd").and_then(|v| v.as_str()) else {
        // Without a project path the chat can't be opened or resumed (the agent
        // spawns in the project's cwd) — skip, like the gemini importer.
        return Ok(());
    };
    let project_path = cwd.to_string();

    // Body lines → ordered (timestamp, sender, body) frames. user → one
    // MessageEnvelope; assistant → text + tool_use frames (thinking dropped);
    // toolResult → dropped (claude UI infers results).
    let mut emitted: Vec<(DateTime<Utc>, &'static str, String)> = Vec::new();
    let mut first_user_text: Option<String> = None;

    while let Some(line) = lines.next_line().await? {
        if line.is_empty() {
            continue;
        }
        let entry: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                warn!(error = %e, "pi: skipping malformed jsonl line");
                continue;
            }
        };
        // Only `message` lines carry chat content; `model_change`,
        // `thinking_level_change`, etc. are skipped.
        if entry.get("type").and_then(|v| v.as_str()) != Some("message") {
            continue;
        }
        let Some(msg) = entry.get("message") else {
            continue;
        };
        // Each message line carries an ISO `timestamp` (mirrors gemini's per-row
        // created-at). Fall back to now if absent.
        let ts = entry
            .get("timestamp")
            .and_then(|v| v.as_str())
            .and_then(parse_rfc3339_utc)
            .unwrap_or_else(Utc::now);

        match msg.get("role").and_then(|v| v.as_str()) {
            Some("user") => {
                let Some(text) = pi_user_prompt_text(msg) else {
                    continue; // synthetic / empty
                };
                if first_user_text.is_none() {
                    first_user_text = Some(text.clone());
                }
                emitted.push((ts, "user", user_message_body(&text)));
            }
            Some("assistant") => {
                let Some(blocks) = msg.get("content").and_then(|v| v.as_array()) else {
                    continue;
                };
                for block in blocks {
                    match block.get("type").and_then(|v| v.as_str()) {
                        Some("text") => {
                            let text = block.get("text").and_then(|v| v.as_str()).unwrap_or("");
                            if !text.is_empty() {
                                emitted.push((ts, "agent", claude_assistant_text_envelope(text)));
                            }
                        }
                        Some("toolCall") => {
                            // Shares the live adapter's pi→claude mapping table.
                            if let Some(frame) = normalize_tool_use(block) {
                                emitted.push((ts, "agent", frame));
                            }
                        }
                        // thinking / unknown blocks: dropped (matches live).
                        _ => {}
                    }
                }
            }
            // toolResult (claude UI infers results) / unknown role → drop.
            _ => {}
        }
    }

    if emitted.is_empty() {
        return Ok(());
    }

    // Emit the project once per resolved path (dedup across sessions). The
    // dispatcher's per-kind progress slice doesn't depend on PutProject order.
    let project_id = match seen_projects.get(&project_path) {
        Some(id) => *id,
        None => {
            let id = mint_project_id(machine_id, &project_path);
            let _ = write_tx
                .send(WriteEvent::PutProject {
                    id,
                    machine_id,
                    name: basename_or(&project_path, "project"),
                    path: project_path.clone(),
                })
                .await;
            seen_projects.insert(project_path, id);
            id
        }
    };

    let chat_created_at = emitted
        .first()
        .map(|(ts, _, _)| *ts)
        .unwrap_or_else(Utc::now);
    let chat_title = first_user_text
        .as_deref()
        .map(collapse_title)
        .unwrap_or_else(|| "Imported chat".to_string());

    // Several blocks of one assistant line share that line's `timestamp`, so bump
    // per-row by the emit index to keep `created_at` monotonic within the chat
    // (mirrors the gemini/cursor importers' multi-row-per-line shape). Message
    // ids are `None` — pi block ids aren't UUIDs and a line fans out into several
    // rows — so the writer mints them; the chat row is stable (chat id == session
    // id) and message rows are insert-once. PutChat + per-row PutMessage is shared
    // via `emit_chat`.
    let messages: Vec<ImportedMessage> = emitted
        .into_iter()
        .enumerate()
        .map(|(seq, (ts, sender, body))| ImportedMessage {
            id: None,
            sender,
            body,
            created_at: ts + ChronoDuration::milliseconds(seq as i64),
        })
        .collect();

    emit_chat(
        write_tx,
        user_id,
        ImportedChat {
            id: chat_id,
            project_id,
            title: chat_title,
            created_at: chat_created_at,
            messages,
        },
    )
    .await;

    Ok(())
}

/// Extract a real user prompt string from a pi `user` `message`, or `None` when
/// it's empty or a synthetic wrapper. pi's user `content` is an array of blocks;
/// a real prompt carries `{type:"text",text}` blocks, which we concatenate.
fn pi_user_prompt_text(msg: &Value) -> Option<String> {
    let blocks = msg.get("content").and_then(|c| c.as_array())?;
    let mut texts: Vec<&str> = Vec::new();
    for b in blocks {
        if b.get("type").and_then(|v| v.as_str()) == Some("text") {
            if let Some(t) = b.get("text").and_then(|v| v.as_str()) {
                texts.push(t);
            }
        }
    }
    if texts.is_empty() {
        return None;
    }
    let joined = texts.join("\n");
    if joined.trim().is_empty() || is_synthetic_wrapper(&joined) {
        return None;
    }
    Some(joined)
}

fn import_boxed(
    machine_id: Uuid,
    user_id: Uuid,
    write_tx: mpsc::Sender<WriteEvent>,
    progress: ImportProgress,
) -> futures::future::BoxFuture<'static, Result<()>> {
    Box::pin(import(machine_id, user_id, write_tx, progress))
}

// ===========================================================================
// Selective forgetting ("prune-context") — pi dialect. Shared contract: see
// `crate::prune`. Wired via `PRUNE_OPS` (→ `AdapterDescriptor::prune`).
//
// pi delta (vs codex's flat rollout): a pi session is a JSONL of
// `{"type":"message","id":...,"parentId":...,"message":{...}}` lines plus a
// few `message`-less header lines (`session`, `model_change`,
// `thinking_level_change`) we skip. The tool CALL and its RESULT live on
// SEPARATE message lines:
//   - assistant call line: `message.role=="assistant"`, `message.content[]`
//     holds one or more `{"type":"toolCall","id":"bash_0","name":...,
//     "arguments":{...}}` blocks. Each block's `id` is the pairing key.
//   - result line: `message.role=="toolResult"`, `message.toolCallId` == the
//     toolCall id, `message.content==[{"type":"text","text":"<output>"}]`.
// Pruning blanks the `text` leaf(s) of the matched result's content to
// `[pruned]`, preserving id/parentId/toolCallId/toolName byte-for-byte; pi
// `--resume` then reloads the edited transcript verbatim (verified). Document
// order + "ordered matched minus already-pruned" reuse the shared
// `crate::prune` core exactly like codex.

/// `crate::prune::PruneOps` for pi. Points the descriptor at this module's
/// dialect pruners.
pub(crate) const PRUNE_OPS: PruneOps = PruneOps {
    find_session: find_pi_session_jsonl,
    count_matches: count_pi_matches,
    prune_batch: prune_batch_pi_jsonl,
};

/// Locate the pi transcript for `session_id`. pi writes
/// `<base>/agent/sessions/<canonical-cwd-mangled>/<ts>_<session-id>.jsonl`, so
/// the FILENAME embeds the session id (`..._<session_id>.jsonl`). The cwd-dir
/// component is the canonicalized cwd (e.g. `/tmp`→`/private/tmp`), so we can't
/// assume it — we walk `<base>/agent/sessions/**` recursively (`base` from
/// `AgentKind::cli_home` = `~/.pi`) and match the first file whose name ends with
/// `_<session_id>.jsonl`. `session_id ↔ file` is 1:1 (resume APPENDS to the same
/// file), so first-match is correct.
fn find_pi_session_jsonl(base: &std::path::Path, session_id: &str) -> Option<std::path::PathBuf> {
    let sessions = base.join("agent").join("sessions");
    let suffix = format!("_{session_id}.jsonl");
    let mut stack = vec![sessions];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                stack.push(path);
                continue;
            }
            if path
                .file_name()
                .and_then(|s| s.to_str())
                .is_some_and(|n| n.ends_with(&suffix))
            {
                return Some(path);
            }
        }
    }
    None
}

/// Single source of truth for the pi-tool-name ↔ claude-tool-name mapping, as
/// `(pi name, claude name)` pairs. This is the INVERSE of the live adapter's
/// `normalize_tool_use` (pi→claude) — they MUST stay in lockstep: if a row here
/// drifts from a `match` arm there, a renamed pi tool would silently stop
/// matching and leak that tool's output past a prune. The forward map there:
/// read→Read, write→Write, edit→Edit, bash→Bash, grep→Grep, **ls→Bash**,
/// web_fetch→WebFetch, web_search→WebSearch. So claude `Bash` fans out to BOTH
/// `["bash","ls"]` (ls renders as a Bash `ls <path>` there). Unknown/extension
/// tools have no entry → `claude_to_pi_tool_names` returns empty → the shared
/// `tool_name_matches` falls back to a literal compare against the raw pi name
/// (correct: pi forwards unknown tools under their native name).
const PI_TOOL_NAME_MAP: &[(&str, &str)] = &[
    ("read", "Read"),
    ("write", "Write"),
    ("edit", "Edit"),
    ("bash", "Bash"),
    ("ls", "Bash"),
    ("grep", "Grep"),
    ("web_fetch", "WebFetch"),
    ("web_search", "WebSearch"),
];

/// Invert `PI_TOOL_NAME_MAP`: a CLAUDE-shape `--tool-name` → every raw pi tool
/// name that renders as it (one claude name can fan out, e.g. `Bash` →
/// `["bash","ls"]`). Empty result = unmapped name → caller matches it literally.
fn claude_to_pi_tool_names(claude_name: &str) -> Vec<&'static str> {
    PI_TOOL_NAME_MAP
        .iter()
        .filter(|(_, c)| *c == claude_name)
        .map(|(p, _)| *p)
        .collect()
}

/// Does a pi `toolCall` block (`name`, `arguments`) match (`tool_name`,
/// `needle`)? Gated on the name (`tool_name_matches` via `claude_to_pi_tool_names`);
/// a non-empty `needle` is a glob over `arguments` VALUE leaves
/// (`prune::value_glob_match` — never keys), and `needle == ""` is the empty-args
/// selector (`prune::args_value_is_empty`). The agent's own in-flight
/// prune-context call is excluded.
fn pi_tool_call_matches(name: &str, args: Option<&Value>, tool_name: &str, needle: &str) -> bool {
    crate::prune::value_args_tool_call_matches(
        name,
        args,
        tool_name,
        needle,
        claude_to_pi_tool_names,
    )
}

/// The `message` object of a persisted pi line whose `message.role == want_role`,
/// or `None` for header lines (no `message`) / a different role. Shared by the
/// collectors + blank pass so the line-shape check lives in one place.
fn pi_message_with_role<'a>(entry: &'a Value, want_role: &str) -> Option<&'a Value> {
    let msg = entry.get("message")?;
    (msg.get("role").and_then(|v| v.as_str()) == Some(want_role)).then_some(msg)
}

/// Matching toolCall `id`s in DOCUMENT ORDER, EXCLUDING any whose paired
/// `toolResult` content is already the `[pruned]` placeholder. A toolCall with no
/// result yet (in-flight) stays eligible. The "ordered matched minus
/// already-pruned" combinator (shared with claude/gemini/codex) lives in
/// [`crate::prune::select_eligible_ids`]; here we supply the pi-shape collectors
/// ([`pi_collect_matched`] / [`pi_collect_pruned`]) — the SAME two feed the
/// single-read apply driver, so count and apply can't drift.
fn eligible_matches(
    path: &std::path::Path,
    tool_name: &str,
    needle: &str,
) -> std::io::Result<Vec<String>> {
    crate::prune::select_eligible_ids(
        path,
        |entry, matched| pi_collect_matched(entry, tool_name, needle, matched),
        pi_collect_pruned,
    )
}

/// Pass-1 matched collector shared by `eligible_matches` (the control-side count
/// path) and the apply driver. For an assistant line, push the `id` of each
/// `toolCall` content block that matches (`tool_name`, `needle`) in DOCUMENT
/// ORDER. One assistant line can carry several toolCall blocks → several ids.
fn pi_collect_matched(entry: &Value, tool_name: &str, needle: &str, matched: &mut Vec<String>) {
    let Some(msg) = pi_message_with_role(entry, "assistant") else {
        return;
    };
    let Some(blocks) = msg.get("content").and_then(|v| v.as_array()) else {
        return;
    };
    for block in blocks {
        if block.get("type").and_then(|v| v.as_str()) != Some("toolCall") {
            continue;
        }
        let name = block.get("name").and_then(|v| v.as_str()).unwrap_or("");
        let args = block.get("arguments");
        if !pi_tool_call_matches(name, args, tool_name, needle) {
            continue;
        }
        if let Some(id) = block.get("id").and_then(|v| v.as_str()) {
            matched.push(id.to_string());
        }
    }
}

/// Pass-1 already-pruned collector shared by `eligible_matches` and the apply
/// driver. Marks a call ineligible when its `toolResult` line's content text is
/// already the `[pruned]` placeholder, keyed by `message.toolCallId`.
fn pi_collect_pruned(entry: &Value, already_pruned: &mut std::collections::HashSet<String>) {
    let Some(msg) = pi_message_with_role(entry, "toolResult") else {
        return;
    };
    let Some(id) = msg.get("toolCallId").and_then(|v| v.as_str()) else {
        return;
    };
    if pi_result_text_is_pruned(msg) {
        already_pruned.insert(id.to_string());
    }
}

/// True when EVERY text leaf of a `toolResult` `message.content` already equals
/// `[pruned]` (and there is at least one) — i.e. the result is fully pruned.
/// Mirrors the idempotency the blank pass relies on so a re-prune is a no-op.
fn pi_result_text_is_pruned(msg: &Value) -> bool {
    let Some(blocks) = msg.get("content").and_then(|v| v.as_array()) else {
        return false;
    };
    let mut saw_text = false;
    for block in blocks {
        if block.get("type").and_then(|v| v.as_str()) == Some("text") {
            saw_text = true;
            if block.get("text").and_then(|v| v.as_str()) != Some(PRUNED_PLACEHOLDER) {
                return false;
            }
        }
    }
    saw_text
}

/// Read-only pre-scan: how many ELIGIBLE matches (drives the zero-check and the
/// "N remain" CLI message). Zero → the control task errors back to the still-alive
/// agent instead of killing + respawning.
fn count_pi_matches(
    path: &std::path::Path,
    tool_name: &str,
    needle: &str,
) -> std::io::Result<usize> {
    Ok(eligible_matches(path, tool_name, needle)?.len())
}

/// Pass-2 helper: blank one parsed pi line in place. Only a `toolResult` line
/// whose `message.toolCallId` ∈ `pruned_ids` is touched — replace every `text`
/// leaf of its `message.content` with `[pruned]` via the shared
/// [`crate::prune::blank_string_field`], preserving id/parentId/toolCallId/
/// toolName byte-for-byte. Records the toolCallId in `outputs_blanked` (the
/// user-facing count counts outputs ACTUALLY dropped, not calls matched).
///
/// Returns `None` when the entry was left untouched, or `Some(freed_bytes)` when
/// ≥1 text leaf was blanked (`freed` may be `0` for a tiny output shorter than
/// the placeholder — the caller still re-serializes). An already-`[pruned]`
/// result returns `None` (idempotent re-prune), so a re-run reports
/// `results_blanked = 0` and skips the timeline frame (matches codex/claude).
fn blank_pi_entry(
    entry: &mut Value,
    pruned_ids: &std::collections::HashSet<String>,
    outputs_blanked: &mut std::collections::HashSet<String>,
) -> Option<usize> {
    let msg = entry.get_mut("message")?.as_object_mut()?;
    if msg.get("role").and_then(|v| v.as_str()) != Some("toolResult") {
        return None;
    }
    let call_id = msg.get("toolCallId").and_then(|v| v.as_str())?.to_string();
    if !pruned_ids.contains(&call_id) {
        return None;
    }
    let blocks = msg.get_mut("content").and_then(|v| v.as_array_mut())?;
    let mut freed = 0usize;
    let mut blanked_any = false;
    for block in blocks.iter_mut() {
        let Some(obj) = block.as_object_mut() else {
            continue;
        };
        if obj.get("type").and_then(|v| v.as_str()) != Some("text") {
            continue;
        }
        if let Some(f) = crate::prune::blank_string_field(obj, "text", PRUNED_PLACEHOLDER) {
            freed += f;
            blanked_any = true;
        }
    }
    if blanked_any {
        outputs_blanked.insert(call_id);
        Some(freed)
    } else {
        None
    }
}

/// Single-target pi rewrite, LAST-ONLY: blank the `toolResult` content of just the
/// MOST RECENT eligible match. The one-round case of [`prune_batch_pi_jsonl`];
/// kept TEST-ONLY as the equivalence oracle the batch path is pinned against.
/// Production goes through the batch fn. `PruneStats::results_blanked` is 0 or 1.
#[cfg(test)]
fn prune_pi_jsonl(
    path: &std::path::Path,
    tool_name: &str,
    needle: &str,
) -> std::io::Result<PruneStats> {
    let (stats, _final_entries) = crate::prune::rewrite_jsonl_last_only(
        path,
        |entry, matched| pi_collect_matched(entry, tool_name, needle, matched),
        pi_collect_pruned,
        blank_pi_entry,
    )?;
    Ok(stats)
}

/// Batch entry point behind `PRUNE_OPS::prune_batch`: blank the last-only target of
/// every `(tool_name, needle)` in `targets` in ONE read/write via the shared batch
/// driver ([`crate::prune::rewrite_jsonl_batch_last_only`]), reproducing exactly
/// what running [`prune_pi_jsonl`] once per target produced. Same pi collectors;
/// pi has no fail-closed post-scan (result outputs aren't duplicated across the
/// transcript like gemini's), so the final entries are ignored (mirrors codex).
fn prune_batch_pi_jsonl(
    path: &std::path::Path,
    targets: &[PruneTarget],
) -> std::io::Result<PruneStats> {
    let (stats, _final_entries) = crate::prune::rewrite_jsonl_batch_last_only(
        path,
        targets.len(),
        |idx, entry, matched| {
            let (tool_name, needle) = &targets[idx];
            pi_collect_matched(entry, tool_name, needle, matched)
        },
        pi_collect_pruned,
        blank_pi_entry,
    )?;
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use std::path::PathBuf;

    fn run(adapter: &mut PiAdapter, line: &str) -> Vec<String> {
        crate::adapter::stringify_events(adapter.handle_line(line.to_string()))
    }

    fn frame_value(event: &str) -> Value {
        crate::adapter::frame_value(event)
    }

    fn ctx<'a>(
        prompt_file: &'a std::path::Path,
        agent_session_id: Option<&'a str>,
        is_sandboxed: bool,
        model: Option<&'a str>,
    ) -> TurnContext<'a> {
        TurnContext::for_test(prompt_file, agent_session_id, is_sandboxed, model)
    }

    #[test]
    fn prepare_command_first_turn_mints_session_id_pipes_stdin_and_approves() {
        let mut a = PiAdapter::new();
        let prompt_file = PathBuf::from("/tmp/p.txt");
        let c = ctx(&prompt_file, None, false, None);
        let cmd = a.prepare_command(&c).unwrap();
        assert!(cmd.contains("cd '/tmp/proj' && "), "got: {}", cmd);
        assert!(
            cmd.contains("cat '/tmp/p.txt' | pi -p --mode json"),
            "got: {}",
            cmd
        );
        assert!(
            cmd.contains(&format!("--session-id '{}'", a.session_id)),
            "first turn mints the adapter's session id: {}",
            cmd
        );
        assert!(cmd.contains(" --approve"), "owner gets --approve: {}", cmd);
        assert!(!cmd.contains("--no-approve"), "got: {}", cmd);
        assert!(
            cmd.contains("--append-system-prompt "),
            "capabilities via system prompt: {}",
            cmd
        );
        assert!(!a.resumed, "first turn is not a resume");
    }

    #[test]
    fn prepare_command_resume_reuses_harvested_session_id() {
        let mut a = PiAdapter::new();
        let prompt_file = PathBuf::from("/tmp/p.txt");
        let c = ctx(
            &prompt_file,
            Some("11111111-2222-3333-4444-555555555555"),
            false,
            None,
        );
        let cmd = a.prepare_command(&c).unwrap();
        assert!(
            cmd.contains("--session-id '11111111-2222-3333-4444-555555555555'"),
            "resume reuses the harvested id: {}",
            cmd
        );
        assert!(a.resumed, "resume sets the harvest-skip flag");
    }

    #[test]
    fn prepare_command_sandboxed_invitee_uses_no_approve() {
        let mut a = PiAdapter::new();
        let prompt_file = PathBuf::from("/tmp/p.txt");
        let c = ctx(&prompt_file, None, true, None);
        let cmd = a.prepare_command(&c).unwrap();
        assert!(
            cmd.contains(" --no-approve"),
            "sandboxed gets --no-approve: {}",
            cmd
        );
        assert!(
            !cmd.contains(" --approve "),
            "must not also pass --approve: {}",
            cmd
        );
    }

    /// `for_test` hard-codes `worktree=false`, `chat_id="0000...0"`,
    /// `project_path=Some("/tmp/proj")`. For the worktree cases we struct-update
    /// `worktree` (the documented pattern) so `chat8`="00000000", branch
    /// "zc-00000000".
    const CHAT8: &str = "00000000";

    #[test]
    fn prepare_command_worktree_first_turn_creates_and_cds_into_worktree() {
        let mut a = PiAdapter::new();
        let prompt_file = PathBuf::from("/tmp/p.txt");
        let c = TurnContext {
            worktree: true,
            ..ctx(&prompt_file, None, false, None)
        };
        let cmd = a.prepare_command(&c).unwrap();

        let wt = crate::zucchini_spawner_dir().join("worktrees").join(CHAT8);
        // basename of the worktree dir == first 8 chars of the chat id.
        assert_eq!(
            wt.file_name().unwrap().to_str().unwrap(),
            CHAT8,
            "worktree dir basename is chat8"
        );
        let wt_esc = shell_escape(&wt.to_string_lossy());

        // First turn: `git -C <proj> worktree add <wt> -b zc-<chat8> && cd <wt>`.
        assert!(
            cmd.contains(&format!(
                "git -C '/tmp/proj' worktree add {wt_esc} -b zc-{CHAT8} && "
            )),
            "first turn emits worktree add with wt path + branch: {cmd}"
        );
        assert!(
            cmd.contains(&format!("cd {wt_esc} && cat '/tmp/p.txt' | pi -p")),
            "cd's into the worktree before piping the prompt: {cmd}"
        );
        // Must NOT also `cd` straight into the project.
        assert!(
            !cmd.contains("cd '/tmp/proj' && "),
            "worktree mode does not cd into the bare project: {cmd}"
        );
    }

    #[test]
    fn prepare_command_worktree_resume_cds_in_without_creating() {
        let mut a = PiAdapter::new();
        let prompt_file = PathBuf::from("/tmp/p.txt");
        let c = TurnContext {
            worktree: true,
            ..ctx(
                &prompt_file,
                Some("11111111-2222-3333-4444-555555555555"),
                false,
                None,
            )
        };
        let cmd = a.prepare_command(&c).unwrap();

        let wt = crate::zucchini_spawner_dir().join("worktrees").join(CHAT8);
        let wt_esc = shell_escape(&wt.to_string_lossy());

        assert!(
            !cmd.contains("worktree add"),
            "resume must NOT recreate the worktree: {cmd}"
        );
        assert!(
            cmd.contains(&format!("cd {wt_esc} && cat '/tmp/p.txt' | pi -p")),
            "resume cd's into the recomputed worktree path: {cmd}"
        );
    }

    #[test]
    fn prepare_command_no_worktree_cds_into_project_unchanged() {
        let mut a = PiAdapter::new();
        let prompt_file = PathBuf::from("/tmp/p.txt");
        // worktree=false (the for_test default) → today's behavior.
        let c = ctx(&prompt_file, None, false, None);
        let cmd = a.prepare_command(&c).unwrap();
        assert!(
            cmd.contains("cd '/tmp/proj' && cat '/tmp/p.txt' | pi -p"),
            "no-worktree cd's into the project: {cmd}"
        );
        assert!(
            !cmd.contains("worktree add"),
            "no-worktree never creates a worktree: {cmd}"
        );
    }

    #[test]
    fn prepare_command_model_pass_through() {
        let mut a = PiAdapter::new();
        let prompt_file = PathBuf::from("/tmp/p.txt");
        let c = ctx(&prompt_file, None, false, Some("anthropic/claude-opus-4"));
        let cmd = a.prepare_command(&c).unwrap();
        assert!(
            cmd.contains("--model 'anthropic/claude-opus-4'"),
            "got: {}",
            cmd
        );
    }

    #[test]
    fn session_header_harvests_id_first_turn_and_drops_frame() {
        let mut a = PiAdapter::new();
        let line = r#"{"type":"session","version":3,"id":"d26585d3-9680-42c4-9495-8af23c38a692","cwd":"/tmp"}"#;
        let events = run(&mut a, line);
        assert_eq!(
            events,
            vec!["SessionIdHarvested(d26585d3-9680-42c4-9495-8af23c38a692)"]
        );
    }

    #[test]
    fn session_header_ignored_on_resume() {
        let mut a = PiAdapter::new();
        a.resumed = true; // simulate a resume turn
        let line = r#"{"type":"session","version":3,"id":"d26585d3-9680-42c4-9495-8af23c38a692","cwd":"/tmp"}"#;
        let events = run(&mut a, line);
        assert!(
            events.is_empty(),
            "resume must not re-harvest: {:?}",
            events
        );
    }

    #[test]
    fn assistant_text_message_becomes_one_text_frame() {
        let mut a = PiAdapter::new();
        let line = r#"{"type":"message_end","message":{"role":"assistant","content":[{"type":"text","text":"hi there"}],"usage":{"input":10,"output":5,"cacheRead":0,"cacheWrite":0}}}"#;
        let events = run(&mut a, line);
        // text frame + context-tokens(10).
        assert_eq!(events.len(), 2, "got: {:?}", events);
        let v = frame_value(&events[0]);
        assert_eq!(v["type"], "assistant");
        assert_eq!(v["message"]["content"][0]["text"], "hi there");
        assert_eq!(events[1], "ContextTokens(10)");
    }

    #[test]
    fn assistant_toolcall_maps_to_claude_tool_use() {
        let mut a = PiAdapter::new();
        let line = r#"{"type":"message_end","message":{"role":"assistant","content":[{"type":"thinking","thinking":"hmm"},{"type":"toolCall","id":"tc1","name":"read","arguments":{"path":"note.txt","limit":100}}],"usage":{"input":177,"cacheRead":5632,"cacheWrite":0}}}"#;
        let events = run(&mut a, line);
        // thinking dropped, toolCall → Read tool_use, then context tokens.
        assert_eq!(events.len(), 2, "got: {:?}", events);
        let v = frame_value(&events[0]);
        assert_eq!(v["type"], "assistant");
        assert_eq!(v["message"]["content"][0]["type"], "tool_use");
        assert_eq!(v["message"]["content"][0]["name"], "Read");
        assert_eq!(v["message"]["content"][0]["input"]["file_path"], "note.txt");
        assert_eq!(v["message"]["content"][0]["id"], "tc1");
        assert_eq!(events[1], "ContextTokens(5809)");
    }

    #[test]
    fn bash_and_grep_tools_map_with_claude_keys() {
        let mut a = PiAdapter::new();
        let line = r#"{"type":"message_end","message":{"role":"assistant","content":[{"type":"toolCall","id":"b1","name":"bash","arguments":{"command":"echo hi"}},{"type":"toolCall","id":"g1","name":"grep","arguments":{"pattern":"foo"}}],"usage":{"input":1,"cacheRead":0,"cacheWrite":0}}}"#;
        let events = run(&mut a, line);
        let b = frame_value(&events[0]);
        assert_eq!(b["message"]["content"][0]["name"], "Bash");
        assert_eq!(b["message"]["content"][0]["input"]["command"], "echo hi");
        let g = frame_value(&events[1]);
        assert_eq!(g["message"]["content"][0]["name"], "Grep");
        assert_eq!(g["message"]["content"][0]["input"]["pattern"], "foo");
    }

    #[test]
    fn unknown_tool_passes_through_with_native_name() {
        let mut a = PiAdapter::new();
        let line = r#"{"type":"message_end","message":{"role":"assistant","content":[{"type":"toolCall","id":"x1","name":"some_extension_tool","arguments":{"foo":"bar"}}],"usage":{"input":1}}}"#;
        let events = run(&mut a, line);
        let v = frame_value(&events[0]);
        assert_eq!(v["message"]["content"][0]["name"], "some_extension_tool");
        assert_eq!(v["message"]["content"][0]["input"]["foo"], "bar");
    }

    #[test]
    fn user_and_toolresult_messages_dropped() {
        let mut a = PiAdapter::new();
        let user = r#"{"type":"message_end","message":{"role":"user","content":[{"type":"text","text":"my prompt"}]}}"#;
        assert!(run(&mut a, user).is_empty());
        let tr = r#"{"type":"message_end","message":{"role":"toolResult","content":[{"type":"text","text":"file contents"}]}}"#;
        assert!(run(&mut a, tr).is_empty());
    }

    #[test]
    fn agent_end_emits_result_frame_and_marker() {
        let mut a = PiAdapter::new();
        let line = r#"{"type":"agent_end","messages":[],"willRetry":false}"#;
        let events = run(&mut a, line);
        assert_eq!(events.len(), 2, "got: {:?}", events);
        let v = frame_value(&events[0]);
        assert_eq!(v["type"], "result");
        assert_eq!(v["subtype"], "success");
        assert_eq!(v["is_error"], false);
        // Spawner-measured run time → iOS/Android render `(Ns)`.
        assert!(v["duration_ms"].is_number(), "got: {v}");
        assert_eq!(events[1], "Result");
    }

    #[test]
    fn lifecycle_events_dropped() {
        let mut a = PiAdapter::new();
        for line in [
            r#"{"type":"agent_start"}"#,
            r#"{"type":"turn_start"}"#,
            r#"{"type":"message_start","message":{"role":"assistant","content":[]}}"#,
            r#"{"type":"message_update","message":{},"assistantMessageEvent":{}}"#,
            r#"{"type":"tool_execution_start","toolCallId":"t","toolName":"read","args":{}}"#,
            r#"{"type":"tool_execution_end","toolCallId":"t","toolName":"read","result":{},"isError":false}"#,
            r#"{"type":"turn_end","message":{},"toolResults":[]}"#,
        ] {
            assert!(run(&mut a, line).is_empty(), "should drop: {line}");
        }
    }

    #[test]
    fn oversize_agent_end_still_emits_result_and_marker() {
        // A >64KB agent_end (it embeds the final assistant message + usage) must
        // bypass the raw-forward and STILL produce the terminal result — else
        // `has_result` stays false and the user sees "Agent interrupted".
        let mut a = PiAdapter::new();
        let pad = "x".repeat(MAX_STREAM_FRAME_BYTES + 1024);
        let line = format!(r#"{{"type":"agent_end","pad":"{pad}"}}"#);
        assert!(line.len() > MAX_STREAM_FRAME_BYTES);
        let events = run(&mut a, &line);
        // Identical to a small agent_end: result frame + Result marker.
        assert_eq!(events.len(), 2, "got: {:?}", events);
        let v = frame_value(&events[0]);
        assert_eq!(v["type"], "result");
        assert_eq!(v["subtype"], "success");
        assert_eq!(v["is_error"], false);
        assert!(v["duration_ms"].is_number(), "got: {v}");
        assert_eq!(events[1], "Result");
    }

    #[test]
    fn oversize_lifecycle_frame_dropped_not_forwarded() {
        // A >64KB lifecycle frame (turn_end / tool_execution_end embed full tool
        // output) must DROP — never forward raw pi JSON (iOS would render `[X]`).
        let mut a = PiAdapter::new();
        let pad = "y".repeat(MAX_STREAM_FRAME_BYTES + 1024);
        for ty in ["turn_end", "tool_execution_end", "message_start"] {
            let line = format!(r#"{{"type":"{ty}","pad":"{pad}"}}"#);
            assert!(line.len() > MAX_STREAM_FRAME_BYTES);
            assert!(
                run(&mut a, &line).is_empty(),
                "oversize {ty} must be dropped, not forwarded"
            );
        }
        // Unclassifiable oversize line (no leading "type") → also dropped.
        let line = format!(r#"{{"pad":"{pad}"}}"#);
        assert!(line.len() > MAX_STREAM_FRAME_BYTES);
        assert!(run(&mut a, &line).is_empty());
    }

    #[test]
    fn repeated_context_tokens_dedup() {
        let mut a = PiAdapter::new();
        let line = r#"{"type":"message_end","message":{"role":"assistant","content":[],"usage":{"input":100,"cacheRead":0,"cacheWrite":0}}}"#;
        assert_eq!(run(&mut a, line), vec!["ContextTokens(100)"]);
        // Identical usage on a later message → no re-emit.
        assert!(run(&mut a, line).is_empty());
    }

    #[test]
    fn non_json_stdout_dropped() {
        let mut a = PiAdapter::new();
        assert!(run(&mut a, "not json").is_empty());
    }

    // ===== history import (pi dialect) ===================================
    mod import_tests {
        use super::super::{import, pi_user_prompt_text};
        use crate::adapter::ImportProgress;
        use crate::writer::WriteEvent;
        use serde_json::{json, Value};
        use tokio::sync::mpsc;
        use uuid::Uuid;

        /// Writes `lines` (joined by "\n") to a session .jsonl under `dir`.
        fn write_session(dir: &std::path::Path, name: &str, lines: &[Value]) {
            std::fs::create_dir_all(dir).unwrap();
            let body = lines
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join("\n");
            std::fs::write(dir.join(name), body).unwrap();
        }

        fn temp_home() -> std::path::PathBuf {
            let mut p = std::env::temp_dir();
            p.push(format!(
                "zucchini_pi_import_home_{}_{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::create_dir_all(&p).unwrap();
            p
        }

        #[test]
        fn user_prompt_concatenates_text_blocks_and_screens_synthetic() {
            let msg = json!({
                "role":"user",
                "content":[{"type":"text","text":"line one"},{"type":"text","text":"line two"}]
            });
            assert_eq!(
                pi_user_prompt_text(&msg).as_deref(),
                Some("line one\nline two")
            );
            // Synthetic wrapper (`<system-reminder>` etc.) screened out.
            let synthetic = json!({
                "role":"user",
                "content":[{"type":"text","text":"<system-reminder>noise</system-reminder>"}]
            });
            assert!(pi_user_prompt_text(&synthetic).is_none());
            // No text blocks → None.
            let empty = json!({"role":"user","content":[]});
            assert!(pi_user_prompt_text(&empty).is_none());
        }

        #[tokio::test]
        async fn end_to_end_import_over_fixture_tree() {
            let home = temp_home();
            // pi nests one session file under a mangled-cwd dir. The real project
            // path comes from the header `cwd`, NOT the dir name.
            let sessions = home
                .join(".pi")
                .join("agent")
                .join("sessions")
                .join("-private-tmp-proj");
            let sid = "019e91c1-8e30-7220-93e9-5c18e88a2595";
            write_session(
                &sessions,
                &format!("20260621_120000_{sid}.jsonl"),
                &[
                    // Line 0: session header → cwd (project path) + id (chat id).
                    json!({"type":"session","version":3,"id":sid,"cwd":"/tmp/proj"}),
                    // Non-message line → skipped.
                    json!({"type":"model_change","timestamp":"2026-06-21T12:00:00.500Z","model":"x"}),
                    // User prompt.
                    json!({"type":"message","id":"u1","parentId":null,"timestamp":"2026-06-21T12:00:01.000Z",
                        "message":{"role":"user","content":[{"type":"text","text":"read note then say done"}]}}),
                    // Assistant: thinking (drop) + text + a read toolCall.
                    json!({"type":"message","id":"a1","parentId":"u1","timestamp":"2026-06-21T12:00:02.000Z",
                    "message":{"role":"assistant","content":[
                        {"type":"thinking","thinking":"hmm"},
                        {"type":"text","text":"Let me read it."},
                        {"type":"toolCall","id":"tc1","name":"read","arguments":{"path":"note.txt"}}
                    ]}}),
                    // toolResult → DROPPED.
                    json!({"type":"message","id":"r1","parentId":"a1","timestamp":"2026-06-21T12:00:03.000Z",
                        "message":{"role":"toolResult","toolCallId":"tc1","toolName":"read",
                            "content":[{"type":"text","text":"BIG FILE BODY"}]}}),
                    // Final assistant text.
                    json!({"type":"message","id":"a2","parentId":"r1","timestamp":"2026-06-21T12:00:04.000Z",
                        "message":{"role":"assistant","content":[{"type":"text","text":"done"}]}}),
                ],
            );

            let (tx, mut rx) = mpsc::channel::<WriteEvent>(64);
            let prev_home = std::env::var_os("HOME");
            // SAFETY: env mutation is single-threaded per test process.
            unsafe {
                std::env::set_var("HOME", &home);
            }
            let result = import(
                Uuid::nil(),
                Uuid::nil(),
                tx,
                Box::new(|_| Box::pin(async {}) as futures::future::BoxFuture<'static, ()>)
                    as ImportProgress,
            )
            .await;
            // SAFETY: restore before assertions.
            unsafe {
                match prev_home {
                    Some(v) => std::env::set_var("HOME", v),
                    None => std::env::remove_var("HOME"),
                }
            }
            result.expect("import ok");

            let mut projects_seen = Vec::new();
            let mut chats_seen = Vec::new();
            let mut messages: Vec<(String, String)> = Vec::new(); // (sender, body)
            while let Ok(ev) = rx.try_recv() {
                match ev {
                    WriteEvent::PutProject { path, .. } => projects_seen.push(path),
                    WriteEvent::PutChat { id, title, .. } => chats_seen.push((id, title)),
                    WriteEvent::PutMessage {
                        sender, content, ..
                    } => messages.push((sender.to_string(), content)),
                    _ => {}
                }
            }
            let _ = home; // tempdir leaked deliberately (cheap; test teardown only)

            // Project path is the header `cwd`, not the mangled dir name.
            assert_eq!(projects_seen, vec!["/tmp/proj".to_string()]);
            // Chat id == the session id (verbatim, resume-consistent).
            assert_eq!(chats_seen.len(), 1);
            assert_eq!(chats_seen[0].0, Uuid::parse_str(sid).unwrap());
            // Title = first user message text.
            assert_eq!(chats_seen[0].1, "read note then say done");

            // Rows: user prompt, assistant text, read tool_use, final text = 4.
            // thinking dropped; toolResult dropped.
            assert_eq!(messages.len(), 4, "got: {:?}", messages);
            assert_eq!(messages[0].0, "user");
            assert!(messages[0].1.contains("read note then say done"));
            assert_eq!(messages[1].0, "agent");
            assert!(messages[1].1.contains("Let me read it."));
            // toolCall normalized to its claude name (read → Read, path → file_path).
            assert_eq!(messages[2].0, "agent");
            let tool: Value = serde_json::from_str(&messages[2].1).unwrap();
            assert_eq!(tool["message"]["content"][0]["type"], "tool_use");
            assert_eq!(tool["message"]["content"][0]["name"], "Read");
            assert_eq!(
                tool["message"]["content"][0]["input"]["file_path"],
                "note.txt"
            );
            assert_eq!(messages[3].0, "agent");
            assert!(messages[3].1.contains("done"));
            // toolResult output never imported.
            assert!(
                !messages.iter().any(|(_, b)| b.contains("BIG FILE BODY")),
                "toolResult must be dropped: {messages:?}"
            );
        }

        #[tokio::test]
        async fn missing_sessions_dir_reports_done_and_no_writes() {
            let home = temp_home(); // exists, but no ~/.pi/agent/sessions
            let (tx, mut rx) = mpsc::channel::<WriteEvent>(8);
            let prev_home = std::env::var_os("HOME");
            unsafe {
                std::env::set_var("HOME", &home);
            }
            let result = import(
                Uuid::nil(),
                Uuid::nil(),
                tx,
                Box::new(|_| Box::pin(async {}) as futures::future::BoxFuture<'static, ()>)
                    as ImportProgress,
            )
            .await;
            unsafe {
                match prev_home {
                    Some(v) => std::env::set_var("HOME", v),
                    None => std::env::remove_var("HOME"),
                }
            }
            result.expect("import ok on missing dir");
            assert!(rx.try_recv().is_err(), "no writes when dir absent");
        }
    }

    // ===== prune-context (pi dialect) ====================================
    mod prune {
        use super::super::{
            claude_to_pi_tool_names, count_pi_matches, find_pi_session_jsonl, prune_pi_jsonl,
            PI_TOOL_NAME_MAP,
        };
        use crate::prune::test_util::{read_lines, write_jsonl};

        /// A persisted pi assistant line carrying one `toolCall` content block.
        fn pi_call(id: &str, parent: &str, name: &str, args: serde_json::Value) -> String {
            serde_json::json!({
                "type": "message",
                "id": id,
                "parentId": parent,
                "timestamp": "2026-06-21T12:00:00.000Z",
                "message": {
                    "role": "assistant",
                    "content": [{ "type": "toolCall", "id": id, "name": name, "arguments": args }],
                }
            })
            .to_string()
        }

        /// A persisted pi `toolResult` line keyed by `toolCallId`, content one text
        /// block (`output`). `parent` is the threading parentId.
        fn pi_result(call_id: &str, parent: &str, tool_name: &str, output: &str) -> String {
            serde_json::json!({
                "type": "message",
                "id": format!("res_{call_id}"),
                "parentId": parent,
                "timestamp": "2026-06-21T12:00:01.000Z",
                "message": {
                    "role": "toolResult",
                    "toolCallId": call_id,
                    "toolName": tool_name,
                    "isError": false,
                    "content": [{ "type": "text", "text": output }],
                }
            })
            .to_string()
        }

        #[test]
        fn pi_tool_name_inverse_map_and_bash_fan_out() {
            // Lockstep guard: the inverse must agree with `normalize_tool_use`
            // (pi→claude). Each PI_TOOL_NAME_MAP entry inverts back to itself.
            for &(pi, claude) in PI_TOOL_NAME_MAP {
                assert!(
                    claude_to_pi_tool_names(claude).contains(&pi),
                    "claude_to_pi_tool_names({claude}) must contain {pi}",
                );
            }
            // Bash fans out to BOTH bash and ls (ls→Bash in normalize_tool_use).
            assert_eq!(claude_to_pi_tool_names("Bash"), vec!["bash", "ls"]);
            // Read/Edit are 1:1.
            assert_eq!(claude_to_pi_tool_names("Read"), vec!["read"]);
            assert_eq!(claude_to_pi_tool_names("Edit"), vec!["edit"]);
            // An unmapped / already-raw pi name yields empty → caller matches it
            // literally (forwarded unknown/extension tools stay prunable).
            assert!(claude_to_pi_tool_names("read").is_empty());
            assert!(claude_to_pi_tool_names("some_extension_tool").is_empty());
        }

        #[test]
        fn find_session_matches_by_id_suffix_recursively() {
            // pi embeds the session-id in the FILENAME and nests under a
            // canonical-cwd dir we can't predict — find_session walks
            // <base>/agent/sessions/** and matches `_<id>.jsonl`.
            let base = std::env::temp_dir().join(format!(
                "zucchini_pi_find_{}_{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            let sid = "019e91c1-8e30-7220-93e9-5c18e88a2595";
            // Canonicalized-cwd-style nested dir.
            let nested = base.join("agent").join("sessions").join("-private-tmp");
            std::fs::create_dir_all(&nested).unwrap();
            let want = nested.join(format!("20260621_120000_{sid}.jsonl"));
            std::fs::write(&want, "{}\n").unwrap();
            // A decoy with a different id must NOT match.
            std::fs::write(
                nested.join("20260621_110000_other-session-id.jsonl"),
                "{}\n",
            )
            .unwrap();

            let got = find_pi_session_jsonl(&base, sid).expect("should find by id suffix");
            assert_eq!(got, want);
            // Unknown id → None.
            assert!(find_pi_session_jsonl(&base, "no-such-id").is_none());
            let _ = std::fs::remove_dir_all(&base);
        }

        #[test]
        fn count_matches_maps_claude_name_and_excludes_already_pruned() {
            let f = write_jsonl(&[
                r#"{"type":"session","id":"019e91c1-8e30-7220-93e9-5c18e88a2595"}"#,
                &pi_call(
                    "bash_0",
                    "u0",
                    "bash",
                    serde_json::json!({"command":"cat junk.rs"}),
                ),
                &pi_result("bash_0", "bash_0", "bash", "BIG OUTPUT OF junk.rs"),
                &pi_call(
                    "ls_0",
                    "bash_0",
                    "ls",
                    serde_json::json!({"path":"junk_dir"}),
                ),
                &pi_result("ls_0", "ls_0", "ls", "junk_dir listing"),
                &pi_call(
                    "bash_1",
                    "ls_0",
                    "bash",
                    serde_json::json!({"command":"cat keep.rs"}),
                ),
                &pi_result("bash_1", "bash_1", "bash", "KEEP"),
            ]);
            // claude "Bash" fans out to bash + ls; needle "junk" → bash_0 and ls_0.
            assert_eq!(count_pi_matches(f.path(), "Bash", "junk").unwrap(), 2);
            // Narrow by the raw pi name too (literal fallback).
            assert_eq!(count_pi_matches(f.path(), "ls", "junk").unwrap(), 1);
            // No match.
            assert_eq!(count_pi_matches(f.path(), "Bash", "nope").unwrap(), 0);

            // Prune the most-recent "junk" match (ls_0) → bash_0 still eligible.
            let stats = prune_pi_jsonl(f.path(), "Bash", "junk").unwrap();
            assert_eq!(stats.results_blanked, 1);
            assert_eq!(count_pi_matches(f.path(), "Bash", "junk").unwrap(), 1);
        }

        #[test]
        fn empty_args_selector_matches_only_no_arg_calls() {
            let f = write_jsonl(&[
                r#"{"type":"session","id":"s"}"#,
                &pi_call("r0", "u0", "read", serde_json::json!({})),
                &pi_result("r0", "r0", "read", "NO-ARG OUTPUT"),
                &pi_call("r1", "r0", "read", serde_json::json!({"path":"x.rs"})),
                &pi_result("r1", "r1", "read", "WITH-ARG OUTPUT"),
            ]);
            // `--args ""` selects only the no-arg call (r0), spares the with-args one.
            assert_eq!(count_pi_matches(f.path(), "Read", "").unwrap(), 1);
            let stats = prune_pi_jsonl(f.path(), "Read", "").unwrap();
            assert_eq!(stats.results_blanked, 1);
            let lines = read_lines(f.path());
            assert_eq!(lines[2]["message"]["content"][0]["text"], "[pruned]");
            assert_eq!(lines[4]["message"]["content"][0]["text"], "WITH-ARG OUTPUT");
        }

        #[test]
        fn prune_context_self_call_is_excluded() {
            // The agent's own in-flight prune-context CLI call (run via bash) must
            // never be the prune target — otherwise last-only would blank ITS
            // output-less result and spare the real file read.
            let prune_cmd = r#""$ZUCCHINI_SPAWNER_BIN" prune-context --tool-name Bash --args "junk.rs" --summary x"#;
            let f = write_jsonl(&[
                r#"{"type":"session","id":"s"}"#,
                &pi_call(
                    "b0",
                    "u0",
                    "bash",
                    serde_json::json!({"command":"cat junk.rs"}),
                ),
                &pi_result("b0", "b0", "bash", "BULKY junk.rs BODY"),
                &pi_call(
                    "b1",
                    "b0",
                    "bash",
                    serde_json::json!({"command": prune_cmd}),
                ),
                &pi_result("b1", "b1", "bash", "pruned 1 tool output"),
            ]);
            // Only the real read (b0) matches "junk.rs"; the prune call is skipped
            // despite carrying the same needle on its command line.
            assert_eq!(count_pi_matches(f.path(), "Bash", "junk.rs").unwrap(), 1);
            let stats = prune_pi_jsonl(f.path(), "Bash", "junk.rs").unwrap();
            assert_eq!(stats.results_blanked, 1);
            let lines = read_lines(f.path());
            assert_eq!(lines[2]["message"]["content"][0]["text"], "[pruned]");
            // The prune call's own result is untouched.
            assert_eq!(
                lines[4]["message"]["content"][0]["text"],
                "pruned 1 tool output"
            );
        }

        #[test]
        fn glob_needle_matches_value_leaf_not_key() {
            let f = write_jsonl(&[
                r#"{"type":"session","id":"s"}"#,
                &pi_call(
                    "b0",
                    "u0",
                    "bash",
                    serde_json::json!({"command":"grep -r foo src/ && echo bar"}),
                ),
                &pi_result("b0", "b0", "bash", "GREP OUTPUT"),
            ]);
            // `*`-separated segments in order within one value leaf.
            assert_eq!(
                count_pi_matches(f.path(), "Bash", "grep*echo bar").unwrap(),
                1
            );
            // A needle equal to a KEY name ("command") must NOT match.
            assert_eq!(count_pi_matches(f.path(), "Bash", "command").unwrap(), 0);
        }

        #[test]
        fn full_blank_pass_blanks_result_and_preserves_threading_bytes() {
            // THE pairing + byte-stability test: the matched toolCall's paired
            // toolResult content is blanked to [pruned], while id/parentId/
            // toolCallId/toolName are preserved byte-for-byte. No line deleted; an
            // unmatched pair is untouched.
            let f = write_jsonl(&[
                r#"{"type":"session","id":"s"}"#,
                &pi_call("r0", "u0", "read", serde_json::json!({"path":"junk.rs"})),
                &pi_result("r0", "r0", "read", "BULKY FILE BODY OF junk.rs"),
                &pi_call("r1", "r0", "read", serde_json::json!({"path":"keep.rs"})),
                &pi_result("r1", "r1", "read", "KEEP BODY"),
            ]);
            let stats = prune_pi_jsonl(f.path(), "Read", "junk.rs").unwrap();
            assert_eq!(stats.results_blanked, 1);
            // No surviving copy of the bulky body.
            let raw = std::fs::read_to_string(f.path()).unwrap();
            assert!(
                !raw.contains("BULKY FILE BODY"),
                "stale output survived: {raw}"
            );
            let lines = read_lines(f.path());
            // Nothing deleted: session + 4 body lines.
            assert_eq!(lines.len(), 5);
            // r0 result blanked; threading fields preserved.
            assert_eq!(lines[2]["message"]["content"][0]["text"], "[pruned]");
            assert_eq!(lines[2]["id"], "res_r0");
            assert_eq!(lines[2]["parentId"], "r0");
            assert_eq!(lines[2]["message"]["toolCallId"], "r0");
            assert_eq!(lines[2]["message"]["toolName"], "read");
            // The call line itself is untouched (we only blank the result).
            assert_eq!(
                lines[1]["message"]["content"][0]["arguments"]["path"],
                "junk.rs"
            );
            // r1 (keep.rs) pair fully intact.
            assert_eq!(lines[4]["message"]["content"][0]["text"], "KEEP BODY");
        }

        #[test]
        fn re_prune_already_pruned_is_idempotent_noop() {
            // An already-[pruned] result reports results_blanked=0 (no timeline
            // frame), and the file stays byte-stable.
            let f = write_jsonl(&[
                r#"{"type":"session","id":"s"}"#,
                &pi_call("r0", "u0", "read", serde_json::json!({"path":"junk.rs"})),
                &pi_result("r0", "r0", "read", "[pruned]"),
            ]);
            // Already pruned → not eligible.
            assert_eq!(count_pi_matches(f.path(), "Read", "junk.rs").unwrap(), 0);
            let before = std::fs::read_to_string(f.path()).unwrap();
            let stats = prune_pi_jsonl(f.path(), "Read", "junk.rs").unwrap();
            assert_eq!(stats.results_blanked, 0);
            let after = std::fs::read_to_string(f.path()).unwrap();
            assert_eq!(before, after, "re-prune must be byte-stable");
        }
    }
}
