//! `zucchini-spawner hermes-turn` subcommand — the per-turn trampoline.
//!
//! Lifecycle
//! ---------
//!
//! 1. `HermesAdapter::prepare_command()` returns a shell command of the form
//!    `"$ZUCCHINI_SPAWNER_BIN" hermes-turn --chat-id=... --user-prompt-file=...
//!    [--yolo] [--project-path=...] [--model=...] [--resume=...]`.
//!    The supervisor invokes it under the user's login shell exactly like
//!    every other adapter; the long-running spawner process exports
//!    `ZUCCHINI_SPAWNER_SOCK=<server-socket-path>` into the spawn's child
//!    env so this trampoline knows where to dial.
//!
//! 2. `main.rs` routes the `hermes-turn` first arg to `run_hermes_turn(args)`
//!    here (parallel to `run_attach_file_cli`). This task:
//!
//!    - reads the user prompt from the file path passed as
//!      `--user-prompt-file` (the supervisor wrote it there before spawn
//!      and cleans up after `child.wait`; reading on connect-failure retry
//!      is safe because the cleanup runs after the trampoline exits)
//!    - opens `UnixStream::connect(ZUCCHINI_SPAWNER_SOCK)` against the
//!      spawner's socket server (`hermes_support::socket_server`)
//!    - sends ONE `{"type":"turn","chat_id":...,...}` frame as NDJSON
//!    - reads back envelopes (`{"chat_id","proactive","event"}`) line by
//!      line, validates `chat_id` matches ours, strips the wrapper, writes
//!      the inner `event` JSON + `\n` + flush to stdout
//!    - exits 0 on `event.type == "result"` with `is_error=false`, 1 on
//!      `is_error=true` or on socket drop / parse failure
//!    - on SIGINT (sent by the supervisor when the user taps Stop), forwards
//!      one `{"type":"stop","chat_id":...}` frame and keeps draining until
//!      the plugin closes the connection or emits the partial result
//!
//! The trampoline's stdout is byte-shaped to match what `claude
//! --output-format stream-json` emits per envelope, so `HermesAdapter::
//! handle_line` reuses every adapter.rs helper unchanged.
//!
//! Socket-architecture context: the spawner is the sole socket server. The
//! `hermes gateway run` process (started by main.rs) dials in as a long-lived
//! "plugin" client; this trampoline dials in as a short-lived "turn" client.
//! The spawner multiplexes — turn frames get forwarded from us to the plugin
//! connection, envelopes get fanned back to whichever trampoline's chat_id
//! matches. Proactive envelopes (cron-fired) never reach this code path —
//! the socket server routes them straight into the writer.

use std::env;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use serde_json::json;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use tracing::{debug, info, warn};

use crate::adapter::parse_json_obj;

/// REQUIRED env var. The spawner always sets this in the trampoline's child
/// env. Missing → programmer error (the spawn path forgot to wire it), we
/// exit non-zero with a clear stderr line; the supervisor synthesises an
/// `INTERRUPTED_RESULT` chat line.
const ENV_SOCK_PATH: &str = "ZUCCHINI_SPAWNER_SOCK";

/// Idle timeout — if the socket goes quiet for this long mid-turn (no
/// envelopes from the plugin), error out so the chat doesn't hang forever.
/// Hermes turns can legitimately run minutes (long shell commands, slow
/// model calls), so the cap is generous. The plugin emits at minimum a
/// `system/init` then a `result` envelope per turn; nothing observable for
/// 5 minutes is a stalled run.
const ENVELOPE_IDLE_TIMEOUT: Duration = Duration::from_secs(300);

/// Parsed `hermes-turn` argv. Hand-rolled (same convention as
/// `run_attach_file_cli` in main.rs) so we don't drag clap into the
/// dependency tree. Long-form flags only; argv is built by
/// `HermesAdapter::prepare_command` which we control end-to-end.
#[derive(Debug, Default)]
pub struct HermesTurnArgs {
    pub chat_id: Option<String>,
    pub yolo: bool,
    pub project_path: Option<String>,
    pub model: Option<String>,
    pub resume: Option<String>,
    /// Required. Supervisor writes the prompt to a temp file and passes the
    /// path here (parallel to codex's `<` stdin redirect — but file paths
    /// don't race with the supervisor's post-`child.wait` cleanup the way
    /// a stdin pipe would).
    pub user_prompt_file: Option<String>,
    /// Optional. For v1 we don't wire `channel_prompt` from the iOS side
    /// (no UI surface), but the flag is parsed so a future surface can land
    /// without re-touching this file. Treated as `None` if absent.
    pub channel_prompt_file: Option<String>,
    /// Repeatable: each `--attachment <abs-path>` adds one entry. For v1 we
    /// pass these straight through to the plugin's `attachments` field;
    /// the plugin currently appends a "Files attached:" footer to the user
    /// prompt rather than wiring them through to hermes' vision tools.
    pub attachments: Vec<String>,
}

impl HermesTurnArgs {
    pub fn parse(args: &[String]) -> Result<Self> {
        let mut out = HermesTurnArgs::default();
        let mut it = args.iter();
        while let Some(a) = it.next() {
            match a.as_str() {
                "--yolo" => out.yolo = true,
                "--chat-id" => out.chat_id = it.next().cloned(),
                "--project-path" => out.project_path = it.next().cloned(),
                "--model" => out.model = it.next().cloned(),
                "--resume" => out.resume = it.next().cloned(),
                "--user-prompt-file" => out.user_prompt_file = it.next().cloned(),
                "--channel-prompt-file" => out.channel_prompt_file = it.next().cloned(),
                "--attachment" => {
                    if let Some(p) = it.next().cloned() {
                        out.attachments.push(p);
                    }
                }
                s if s.starts_with("--chat-id=") => {
                    out.chat_id = Some(s["--chat-id=".len()..].to_string());
                }
                s if s.starts_with("--project-path=") => {
                    out.project_path = Some(s["--project-path=".len()..].to_string());
                }
                s if s.starts_with("--model=") => {
                    out.model = Some(s["--model=".len()..].to_string());
                }
                s if s.starts_with("--resume=") => {
                    out.resume = Some(s["--resume=".len()..].to_string());
                }
                s if s.starts_with("--user-prompt-file=") => {
                    out.user_prompt_file = Some(s["--user-prompt-file=".len()..].to_string());
                }
                s if s.starts_with("--channel-prompt-file=") => {
                    out.channel_prompt_file = Some(s["--channel-prompt-file=".len()..].to_string());
                }
                s if s.starts_with("--attachment=") => {
                    out.attachments.push(s["--attachment=".len()..].to_string());
                }
                other => return Err(anyhow!("unknown hermes-turn flag: {other}")),
            }
        }
        if out.chat_id.as_deref().unwrap_or("").is_empty() {
            return Err(anyhow!("--chat-id is required"));
        }
        if out.user_prompt_file.as_deref().unwrap_or("").is_empty() {
            return Err(anyhow!("--user-prompt-file is required"));
        }
        Ok(out)
    }
}

/// Entry point. Returns the process exit code (0 clean, 1 anything else
/// — the supervisor will surface non-zero as `Done.has_result = false` if
/// no result envelope was forwarded and synthesise the existing
/// INTERRUPTED_RESULT line). `i32` (not `ExitCode`) so the caller in
/// `main.rs` can hand it straight to `std::process::exit` without an
/// awkward `From<ExitCode>` conversion that `u8` doesn't implement.
pub async fn run_hermes_turn(args: HermesTurnArgs) -> i32 {
    match run_inner(args).await {
        Ok(()) => 0,
        Err(e) => {
            // Stderr only — stdout is the wire path back to the supervisor
            // and must stay claude-shape NDJSON. The supervisor's stderr
            // task in agent.rs buffers startup noise and warns on
            // post-startup lines, so this surfaces as a `agent stderr:`
            // log entry.
            eprintln!("hermes-turn: {e:#}");
            1
        }
    }
}

async fn run_inner(args: HermesTurnArgs) -> Result<()> {
    let chat_id = args.chat_id.clone().expect("validated in parse");
    let prompt_path = args.user_prompt_file.clone().expect("validated in parse");

    // Read the user prompt from disk. The supervisor wrote it there and
    // will remove it after our process exits (see agent.rs's post-`child.wait`
    // cleanup). Reading is sync-friendly: ~1-2 syscalls for typical prompts
    // (≤ a few KB).
    let user_prompt = tokio::fs::read_to_string(&prompt_path)
        .await
        .with_context(|| format!("read user prompt file {prompt_path}"))?;

    // Optional channel prompt — `None` (the absent case) maps to JSON null.
    let channel_prompt: Option<String> = match args.channel_prompt_file.as_deref() {
        Some(p) if !p.is_empty() => Some(
            tokio::fs::read_to_string(p)
                .await
                .with_context(|| format!("read channel prompt file {p}"))?,
        ),
        _ => None,
    };

    let sock_path = resolve_sock_path()
        .context("ZUCCHINI_SPAWNER_SOCK env var missing (spawner must set this on every spawn)")?;
    debug!(sock_path = %sock_path.display(), "hermes-turn connecting");

    let stream = UnixStream::connect(&sock_path)
        .await
        .with_context(|| format!("connect spawner socket at {}", sock_path.display()))?;
    let (read_half, mut write_half) = stream.into_split();
    let mut reader = BufReader::new(read_half);

    // Build and send the turn frame per the wire-format spec at the top of
    // `~/.hermes/plugins/zucchini/adapter.py`. The frame goes spawner-side
    // first, which routes it to the plugin connection by `chat_id`.
    let turn_frame = json!({
        "type": "turn",
        "chat_id": chat_id,
        "user_prompt": user_prompt,
        "project_path": args.project_path.clone().unwrap_or_default(),
        "yolo": args.yolo,
        "model": args.model,
        "channel_prompt": channel_prompt,
        "resume": args.resume,
        "attachments": args.attachments,
    });
    let mut line = serde_json::to_string(&turn_frame).expect("serialize turn frame");
    line.push('\n');
    write_half
        .write_all(line.as_bytes())
        .await
        .context("write turn frame")?;
    write_half.flush().await.context("flush turn frame")?;

    // SIGINT handler — the supervisor's `terminate_agent_process_group`
    // sends SIGTERM (5s grace before SIGKILL) on cancel; SIGINT is the
    // graceful interrupt path. We register early so a fast user-stop
    // between connect and the read loop is still caught.
    let mut sigint = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt()).ok();

    // Frame stream pump. One stdout write per inbound envelope, BufWriter so
    // the per-envelope write+flush cost stays low while still surfacing
    // each frame live to the supervisor (it reads with
    // `BufReader::lines()` — no aggregation, every flush is a row).
    let mut stdout = tokio::io::BufWriter::new(tokio::io::stdout());
    let mut buf = String::new();
    let mut stop_sent = false;
    loop {
        buf.clear();
        let read_fut = reader.read_line(&mut buf);
        let timeout = tokio::time::sleep(ENVELOPE_IDLE_TIMEOUT);
        tokio::pin!(read_fut);
        tokio::pin!(timeout);

        // We can't `select!` directly over `sigint` when it's `None` (no
        // handler registered) so branch on the option. The signal-arm
        // forwards a stop frame ONCE and keeps draining.
        let result = if let Some(sig) = sigint.as_mut() {
            tokio::select! {
                biased;
                _ = sig.recv(), if !stop_sent => {
                    info!("hermes-turn: SIGINT, forwarding stop frame");
                    let stop = json!({"type":"stop","chat_id":chat_id}).to_string() + "\n";
                    if let Err(e) = write_half.write_all(stop.as_bytes()).await {
                        warn!(error = %e, "hermes-turn: stop frame write failed");
                        break;
                    }
                    let _ = write_half.flush().await;
                    stop_sent = true;
                    continue;
                }
                _ = &mut timeout => Err(anyhow!("no envelopes for {:?}", ENVELOPE_IDLE_TIMEOUT)),
                r = &mut read_fut => r.map_err(|e| anyhow!("read envelope: {e}")),
            }
        } else {
            tokio::select! {
                _ = &mut timeout => Err(anyhow!("no envelopes for {:?}", ENVELOPE_IDLE_TIMEOUT)),
                r = &mut read_fut => r.map_err(|e| anyhow!("read envelope: {e}")),
            }
        };

        match result {
            Ok(0) => {
                // EOF before we ever saw a terminal envelope — surface as
                // an error so the supervisor lands on INTERRUPTED_RESULT.
                // The clean-end path returns Ok directly from inside the
                // terminal-envelope branch below, never reaching this
                // arm.
                debug!("hermes-turn: socket EOF before terminal");
                return Err(anyhow!("spawner closed socket before result"));
            }
            Ok(_) => {
                let trimmed = buf.trim_end_matches('\n');
                if trimmed.is_empty() {
                    continue;
                }
                let projected = match project_inbound_envelope(trimmed, &chat_id) {
                    EnvelopeOutcome::ForOurChat { inner, is_terminal } => {
                        Some((inner, is_terminal))
                    }
                    EnvelopeOutcome::ForeignChat => None,
                    EnvelopeOutcome::Proactive => {
                        // Should not happen: proactive envelopes are routed
                        // by the spawner's socket server straight into the
                        // writer, never fanned to a trampoline. Defensive
                        // drop — keep the trampoline robust against a future
                        // routing-bug.
                        debug!(
                            "hermes-turn: dropping proactive envelope (unexpected on turn path)"
                        );
                        None
                    }
                    EnvelopeOutcome::Malformed(reason) => {
                        warn!(reason, "hermes-turn: malformed envelope");
                        None
                    }
                };
                if let Some((inner_line, is_terminal)) = projected {
                    stdout
                        .write_all(inner_line.as_bytes())
                        .await
                        .context("write envelope to stdout")?;
                    stdout.write_all(b"\n").await.context("write newline")?;
                    stdout.flush().await.context("flush stdout")?;
                    if is_terminal {
                        debug!("hermes-turn: terminal envelope, exiting");
                        // Clean turn end — exit the loop directly. EOF
                        // following a terminal envelope is the expected
                        // path; we don't need to thread a `had_terminal`
                        // flag through the EOF arm because we never re-enter.
                        return Ok(());
                    }
                }
            }
            Err(e) => {
                return Err(e);
            }
        }
    }

    Ok(())
}

/// Outcome of demultiplexing one wrapped envelope from the spawner.
enum EnvelopeOutcome {
    /// Envelope is for our chat — emit `inner` to stdout. `is_terminal` is
    /// true for `result` and `error` events; caller exits after writing.
    ForOurChat { inner: String, is_terminal: bool },
    /// Envelope is for a different chat — the spawner shouldn't fan it
    /// here, but a routing race could send a strays; silently drop.
    ForeignChat,
    /// `proactive: true` envelope — never expected on a trampoline socket
    /// (the spawner routes those directly to the writer). Drop.
    Proactive,
    /// JSON parse failure, missing keys, wrong shape. Caller logs.
    Malformed(&'static str),
}

/// Strips the outer `{chat_id, proactive, event}` wrapper and returns the
/// inner `event` as a serialized line ready for stdout. `is_terminal` is
/// true when `event.type` is `result` or `error`.
fn project_inbound_envelope(line: &str, our_chat_id: &str) -> EnvelopeOutcome {
    let Some(obj) = parse_json_obj(line) else {
        return EnvelopeOutcome::Malformed("not a JSON object");
    };
    // Plugin liveness/control frames (`hello`, `pong`) are NOT chat-wrapped;
    // the spawner consumes them before they reach the trampoline. If one
    // somehow slips through, treat as malformed.
    let Some(env_chat) = obj.get("chat_id").and_then(|v| v.as_str()) else {
        return EnvelopeOutcome::Malformed("missing chat_id in envelope");
    };
    if env_chat != our_chat_id {
        return EnvelopeOutcome::ForeignChat;
    }
    let proactive = obj
        .get("proactive")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if proactive {
        return EnvelopeOutcome::Proactive;
    }
    let Some(event) = obj.get("event") else {
        return EnvelopeOutcome::Malformed("missing event in envelope");
    };
    let event_type = event.get("type").and_then(|v| v.as_str()).unwrap_or("");
    let is_terminal = matches!(event_type, "result" | "error");
    let serialized = match serde_json::to_string(event) {
        Ok(s) => s,
        Err(_) => return EnvelopeOutcome::Malformed("re-serialize event"),
    };
    EnvelopeOutcome::ForOurChat {
        inner: serialized,
        is_terminal,
    }
}

fn resolve_sock_path() -> Result<PathBuf> {
    let s = env::var(ENV_SOCK_PATH)
        .map_err(|_| anyhow!("ZUCCHINI_SPAWNER_SOCK env var is required for hermes-turn"))?;
    let p = PathBuf::from(s);
    if p.is_absolute() {
        Ok(p)
    } else {
        // Relative is unexpected but cheaply supported — anchor under $HOME.
        let home = env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."));
        Ok(home.join(p))
    }
}

/// Helper used by `EnvelopeOutcome` consumers in unit tests.
#[cfg(test)]
fn outcome_kind(out: &EnvelopeOutcome) -> &'static str {
    match out {
        EnvelopeOutcome::ForOurChat { .. } => "ours",
        EnvelopeOutcome::ForeignChat => "foreign",
        EnvelopeOutcome::Proactive => "proactive",
        EnvelopeOutcome::Malformed(_) => "malformed",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    #[test]
    fn parse_args_accepts_long_flags_with_equals() {
        let args = vec![
            "--chat-id=abc".to_string(),
            "--yolo".to_string(),
            "--project-path=/tmp/proj".to_string(),
            "--model=opus".to_string(),
            "--resume=20260528_214220_e6100c".to_string(),
            "--user-prompt-file=/tmp/p.txt".to_string(),
            "--attachment=/tmp/a.png".to_string(),
            "--attachment=/tmp/b.txt".to_string(),
        ];
        let parsed = HermesTurnArgs::parse(&args).unwrap();
        assert_eq!(parsed.chat_id.as_deref(), Some("abc"));
        assert!(parsed.yolo);
        assert_eq!(parsed.project_path.as_deref(), Some("/tmp/proj"));
        assert_eq!(parsed.model.as_deref(), Some("opus"));
        assert_eq!(parsed.resume.as_deref(), Some("20260528_214220_e6100c"));
        assert_eq!(parsed.user_prompt_file.as_deref(), Some("/tmp/p.txt"));
        assert_eq!(parsed.attachments, vec!["/tmp/a.png", "/tmp/b.txt"]);
    }

    #[test]
    fn parse_args_accepts_long_flags_with_space() {
        let args = vec![
            "--chat-id".to_string(),
            "abc".to_string(),
            "--user-prompt-file".to_string(),
            "/tmp/p.txt".to_string(),
            "--project-path".to_string(),
            "/tmp/proj".to_string(),
        ];
        let parsed = HermesTurnArgs::parse(&args).unwrap();
        assert_eq!(parsed.chat_id.as_deref(), Some("abc"));
        assert_eq!(parsed.user_prompt_file.as_deref(), Some("/tmp/p.txt"));
        assert_eq!(parsed.project_path.as_deref(), Some("/tmp/proj"));
        assert!(!parsed.yolo);
    }

    #[test]
    fn parse_args_missing_chat_id_errors() {
        let args = vec![
            "--yolo".to_string(),
            "--user-prompt-file=/tmp/p.txt".to_string(),
        ];
        let err = HermesTurnArgs::parse(&args).unwrap_err().to_string();
        assert!(err.contains("--chat-id"), "got: {err}");
    }

    #[test]
    fn parse_args_missing_user_prompt_file_errors() {
        let args = vec!["--chat-id=abc".to_string(), "--yolo".to_string()];
        let err = HermesTurnArgs::parse(&args).unwrap_err().to_string();
        assert!(err.contains("--user-prompt-file"), "got: {err}");
    }

    #[test]
    fn parse_args_unknown_flag_errors() {
        let args = vec![
            "--chat-id=abc".to_string(),
            "--user-prompt-file=/tmp/p.txt".to_string(),
            "--unknown=x".to_string(),
        ];
        assert!(HermesTurnArgs::parse(&args).is_err());
    }

    #[test]
    fn project_inbound_envelope_strips_wrapper_for_our_chat() {
        let line = r#"{"chat_id":"chat-1","proactive":false,"event":{"type":"assistant","message":{"content":[{"type":"text","text":"hi"}]}}}"#;
        let out = project_inbound_envelope(line, "chat-1");
        match out {
            EnvelopeOutcome::ForOurChat { inner, is_terminal } => {
                let v: Value = serde_json::from_str(&inner).unwrap();
                assert_eq!(v["type"], "assistant");
                assert_eq!(v["message"]["content"][0]["text"], "hi");
                assert!(!is_terminal);
            }
            other => panic!("expected ForOurChat, got {}", outcome_kind(&other)),
        }
    }

    #[test]
    fn project_inbound_envelope_detects_result_as_terminal() {
        let line = r#"{"chat_id":"c","proactive":false,"event":{"type":"result","subtype":"success","is_error":false}}"#;
        let out = project_inbound_envelope(line, "c");
        match out {
            EnvelopeOutcome::ForOurChat { is_terminal, .. } => assert!(is_terminal),
            other => panic!("expected ForOurChat, got {}", outcome_kind(&other)),
        }
    }

    #[test]
    fn project_inbound_envelope_detects_error_as_terminal() {
        let line = r#"{"chat_id":"c","proactive":false,"event":{"type":"error","message":"boom"}}"#;
        let out = project_inbound_envelope(line, "c");
        match out {
            EnvelopeOutcome::ForOurChat { is_terminal, .. } => assert!(is_terminal),
            other => panic!("expected ForOurChat, got {}", outcome_kind(&other)),
        }
    }

    #[test]
    fn project_inbound_envelope_foreign_chat_is_dropped() {
        let line = r#"{"chat_id":"other","proactive":false,"event":{"type":"assistant"}}"#;
        let out = project_inbound_envelope(line, "mine");
        assert_eq!(outcome_kind(&out), "foreign");
    }

    #[test]
    fn project_inbound_envelope_proactive_is_dropped() {
        let line = r#"{"chat_id":"mine","proactive":true,"event":{"type":"assistant"}}"#;
        let out = project_inbound_envelope(line, "mine");
        assert_eq!(outcome_kind(&out), "proactive");
    }

    #[test]
    fn project_inbound_envelope_non_json_is_malformed() {
        let out = project_inbound_envelope("not json", "mine");
        assert_eq!(outcome_kind(&out), "malformed");
    }

    #[test]
    fn project_inbound_envelope_missing_chat_id_is_malformed() {
        let line = r#"{"proactive":false,"event":{"type":"x"}}"#;
        let out = project_inbound_envelope(line, "mine");
        assert_eq!(outcome_kind(&out), "malformed");
    }

    #[test]
    fn project_inbound_envelope_missing_event_is_malformed() {
        let line = r#"{"chat_id":"mine","proactive":false}"#;
        let out = project_inbound_envelope(line, "mine");
        assert_eq!(outcome_kind(&out), "malformed");
    }
}
