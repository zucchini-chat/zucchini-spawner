//! Per-turn agent Supervisor. Owns the OS-side process lifecycle (spawn,
//! signal escalation on cancel, stderr buffering, startup watchdog, prompt
//! file write/cleanup, power assertion). The agent-specific bits — building
//! the CLI command, normalizing stdout frames into claude-shape envelopes,
//! harvesting session ids/usage/compact boundaries — live behind the
//! `AgentAdapter` trait in `adapter.rs` and the concrete adapters in
//! `adapters/`. Supervisor stays agent-agnostic; future adapters plug in
//! through the registry without touching this file.
//!
//! Each `spawn_agent` call constructs a fresh adapter, hands it to the spawned
//! task, and discards it when the turn ends.
use nix::sys::signal::{self, Signal};
use nix::unistd::Pid;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex as StdMutex,
};
use std::time::{Duration, SystemTime};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn};

use crate::adapter::{AgentEvent, AgentKind, TurnContext};
use crate::envelope::{EnvelopeAttachment, MessageEnvelope};

/// Per-chat pending-attachments registry: the agent-side `attach-file` flow
/// writes `EnvelopeAttachment`s through these unbounded senders, and the
/// per-turn supervisor task (the receiver-owner) drains them just before
/// forwarding the next assistant frame. Cleared on agent exit so a future
/// `attach-file` against a stale chat-id returns a "no running agent" error
/// rather than queueing forever.
///
/// Std `Mutex` because (a) hold times are microseconds (one `HashMap` get +
/// `try_send`), (b) the control-socket handler is the only writer and the
/// supervisor side only touches it at spawn/cleanup. No async work happens
/// under the guard so a `tokio::sync::Mutex` would be wasted async overhead.
pub type PendingAttachments =
    Arc<StdMutex<HashMap<String, mpsc::UnboundedSender<EnvelopeAttachment>>>>;

const AGENT_EXIT_TIMEOUT: Duration = Duration::from_secs(10);
const AGENT_STARTUP_TIMEOUT: Duration = Duration::from_secs(30);

pub enum AgentResponse {
    Line {
        topic: String,
        content: String,
    },
    ContextTokens {
        topic: String,
        tokens: i64,
    },
    /// Manual `/compact` or auto-compact completed; carries `compactMetadata.postTokens`.
    CompactBoundary {
        topic: String,
        post_tokens: i64,
    },
    /// First `system/init` frame from the agent on a fresh chat — carries the
    /// session id the agent generated for itself. Caller writes it to
    /// `chats.agent_session_id` so subsequent turns can resume it.
    SessionIdHarvested {
        topic: String,
        session_id: String,
    },
    Done {
        topic: String,
        has_result: bool,
    },
    /// The `prune-context` call's own result has persisted for `topic` — the main
    /// loop's cue to apply a queued prune (abort → rewrite → respawn). Call-keyed:
    /// the adapter only emits it for the prune call's own result, never a sibling's.
    /// Carries no body (the frame itself is skipped); a no-op unless a
    /// `PruneRequest` is pending for the chat. See `AgentEvent::ToolResult`.
    ToolResult {
        topic: String,
    },
}

impl AgentEvent {
    /// Maps one adapter-emitted event onto the supervisor's response channel.
    /// `AgentEvent::Result` is a supervisor-only signal — it flips `has_result`
    /// (so the eventual `AgentResponse::Done` carries the right flag) and
    /// returns `None`, suppressing wire emission. Every other variant maps
    /// 1:1 to an `AgentResponse` for the given chat topic.
    fn into_response(self, topic: &str, has_result: &mut bool) -> Option<AgentResponse> {
        Some(match self {
            AgentEvent::Frame(content) => AgentResponse::Line {
                topic: topic.to_string(),
                content,
            },
            AgentEvent::ContextTokens(tokens) => AgentResponse::ContextTokens {
                topic: topic.to_string(),
                tokens,
            },
            AgentEvent::CompactBoundary(post_tokens) => AgentResponse::CompactBoundary {
                topic: topic.to_string(),
                post_tokens,
            },
            AgentEvent::SessionIdHarvested(session_id) => AgentResponse::SessionIdHarvested {
                topic: topic.to_string(),
                session_id,
            },
            AgentEvent::Result => {
                *has_result = true;
                return None;
            }
            AgentEvent::ToolResult => AgentResponse::ToolResult {
                topic: topic.to_string(),
            },
        })
    }
}

/// One-shot input to `Supervisor::spawn_agent` — bundles every per-turn
/// parameter so `spawn_agent` is a single-argument call and the
/// `#[allow(clippy::too_many_arguments)]` waiver can stay off.
pub struct SpawnRequest {
    pub chat_id: String,
    pub prompt: String,
    pub project_path: Option<String>,
    pub worktree: bool,
    /// `Some(_)` to resume a prior session, `None` for a brand-new chat.
    pub agent_session_id: Option<String>,
    pub agent_kind: AgentKind,
    pub is_sandboxed: bool,
    /// `chats.model` — verbatim `--model <X>` pass-through to the CLI
    /// (migration 0035). Empty / blank values are filtered to `None` at
    /// the `main.rs` construction site so adapters can read `Some(_)`
    /// as "user picked a non-default model".
    pub model: Option<String>,
}

/// Per-spawn closure: owns the OS-side work (build Command, spawn, drive
/// stdout/stderr, emit `AgentResponse`s on `tx`, observe `token` for cancel,
/// return the driving `JoinHandle`). Default implementation is
/// `default_spawn_fn` which preserves the historic behavior; tests inject a
/// recorder via `Supervisor::with_spawn_fn` to capture `SpawnRequest`s
/// without ever touching `tokio::process::Command`.
///
/// `pending` is the per-chat `EnvelopeAttachment` mailbox the agent-side
/// `attach-file` flow pushes into; the per-turn spawn task registers its
/// receiver in it at spawn time and removes it on exit. Test spawn fns
/// ignore this — they never wrap frames.
pub type SpawnFn = Arc<
    dyn Fn(
            SpawnRequest,
            mpsc::Sender<AgentResponse>,
            CancellationToken,
            PendingAttachments,
        ) -> tokio::task::JoinHandle<()>
        + Send
        + Sync,
>;

pub struct Supervisor {
    agents: HashMap<String, (tokio::task::JoinHandle<()>, CancellationToken)>,
    response_tx: mpsc::Sender<AgentResponse>,
    spawn_fn: SpawnFn,
    /// Shared with the control-socket handler so an inbound `attach-file`
    /// RPC can look up the right per-chat mailbox by chat id.
    pending: PendingAttachments,
}

impl Supervisor {
    pub fn new(response_tx: mpsc::Sender<AgentResponse>) -> Self {
        Self::with_spawn_fn(response_tx, Arc::new(default_spawn_fn))
    }

    /// Inject a custom spawn implementation. Production callers go through
    /// `Supervisor::new` which wires this to `default_spawn_fn`. Tests pass a
    /// closure that records the `SpawnRequest` and returns a no-op JoinHandle.
    pub fn with_spawn_fn(response_tx: mpsc::Sender<AgentResponse>, spawn_fn: SpawnFn) -> Self {
        Self {
            agents: HashMap::new(),
            response_tx,
            spawn_fn,
            pending: Arc::new(StdMutex::new(HashMap::new())),
        }
    }

    /// Hand the control-socket task a clone of the pending-attachments
    /// registry so it can route `attach-file` RPC results to the right
    /// per-chat mailbox. Cheap `Arc` clone — same shared state.
    pub fn pending_attachments(&self) -> PendingAttachments {
        self.pending.clone()
    }

    pub async fn abort_agent(&mut self, topic: &str) -> bool {
        if let Some((handle, token)) = self.agents.remove(topic) {
            info!(topic = %topic, "aborting running agent");
            token.cancel();
            // Wait so the old agent process is fully dead before spawning a new one on the same session.
            await_agent_exit(handle, topic).await;
            true
        } else {
            false
        }
    }

    /// Check if there are no tracked agents.
    pub fn is_empty(&self) -> bool {
        self.agents.is_empty()
    }

    pub fn remove(&mut self, topic: &str) {
        self.agents.remove(topic);
    }

    pub fn is_running(&self, topic: &str) -> bool {
        self.agents
            .get(topic)
            .is_some_and(|(h, _)| !h.is_finished())
    }

    /// Constructs an adapter for `req.agent_kind` and spawns the per-turn task
    /// via `self.spawn_fn` (the default impl preserves the historic
    /// `tokio::process::Command` path — see `default_spawn_fn`).
    pub fn spawn_agent(&mut self, req: SpawnRequest) {
        let topic = req.chat_id.clone();
        let token = CancellationToken::new();
        let handle = (self.spawn_fn)(
            req,
            self.response_tx.clone(),
            token.clone(),
            self.pending.clone(),
        );
        self.agents.insert(topic, (handle, token));
    }

    pub fn cleanup(&mut self) {
        self.agents.retain(|_, (handle, _)| !handle.is_finished());
    }

    pub async fn shutdown_all(&mut self) {
        let agents: Vec<_> = self.agents.drain().collect();
        if agents.is_empty() {
            return;
        }
        info!("shutting down {} running agent(s)", agents.len());
        for (topic, (_, token)) in &agents {
            info!(topic = %topic, "cancelling agent");
            token.cancel();
        }
        for (topic, (handle, _)) in agents {
            await_agent_exit(handle, &topic).await;
        }
    }
}

/// Default spawn implementation — verbatim lift of the historic body of
/// `Supervisor::spawn_agent`. Builds a `tokio::process::Command` via the
/// adapter, drives stdout/stderr, and emits `AgentResponse`s onto `tx`. The
/// per-turn `CancellationToken` is observed for /stop and abort-then-respawn.
///
/// `pending` is the shared per-chat-id mailbox registry. The spawned task
/// installs its receiver into the map at startup (and removes it on exit)
/// so the control-socket handler can push `EnvelopeAttachment`s in via
/// `attach-file`; after each assistant text frame is forwarded, the
/// supervisor drains the receiver and — if non-empty — emits a separate
/// follow-up `AgentResponse::Line` whose body is an attachments-only
/// `MessageEnvelope`. iOS sees that as its own row and renders an
/// attachment-only bubble below the assistant text.
fn default_spawn_fn(
    req: SpawnRequest,
    tx: mpsc::Sender<AgentResponse>,
    token: CancellationToken,
    pending: PendingAttachments,
) -> tokio::task::JoinHandle<()> {
    let SpawnRequest {
        chat_id: topic,
        prompt,
        project_path,
        worktree,
        agent_session_id,
        agent_kind,
        is_sandboxed,
        model,
    } = req;
    let mut adapter = agent_kind.make_adapter();

    let topic_clone = topic;
    let token_clone = token;

    // Register the per-chat pending-attachments mailbox before the spawn so
    // a fast `attach-file` RPC issued right after the agent starts isn't
    // dropped on the floor (the agent could already be reading the prompt
    // file). Removed in the `Drop` arm below on every exit path (clean
    // finish, cancel, error) so a future RPC against a stale chat id fails
    // loud with "no running agent" rather than queueing forever.
    let (attach_tx, mut attach_rx) = mpsc::unbounded_channel::<EnvelopeAttachment>();
    {
        let mut guard = pending.lock().expect("PendingAttachments mutex");
        // If a prior turn for this chat is somehow still in the map (no-op
        // under the abort-then-respawn path because the prior task's RAII
        // guard already removed it), prefer the fresh sender — drops the
        // old one and its mailbox.
        guard.insert(topic_clone.clone(), attach_tx);
    }

    tokio::spawn(async move {
        // RAII guard so cancellation paths (early `return` after the
        // outer-cancel arm, error returns from prompt-file write, etc.)
        // still clean the registry entry. Cheap: one `HashMap::remove`
        // under a short-held std Mutex.
        struct PendingGuard {
            map: PendingAttachments,
            chat_id: String,
        }
        impl Drop for PendingGuard {
            fn drop(&mut self) {
                if let Ok(mut g) = self.map.lock() {
                    g.remove(&self.chat_id);
                }
            }
        }
        let _pending_guard = PendingGuard {
            map: pending,
            chat_id: topic_clone.clone(),
        };
        let _power_assertion = crate::power::AgentPowerAssertion::acquire();
        let kind = adapter.kind();
        info!(
            topic = %topic_clone,
            kind = ?kind,
            resume = agent_session_id.is_some(),
            project_path = ?project_path,
            worktree,
            sandbox = is_sandboxed,
            "spawning agent"
        );

        // First turn only (`agent_session_id.is_none()`): append the adapter's
        // prompt suffix after the user's text. See `first_turn_prompt_suffix`.
        let prompt = if agent_session_id.is_none() {
            match adapter.first_turn_prompt_suffix() {
                Some(suffix) => format!("{prompt}\n\n---\n{suffix}"),
                None => prompt,
            }
        } else {
            prompt
        };

        // Write prompt to a temp file so it never touches the shell command string
        let unique = format!(
            "{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        );
        let prompt_file = PathBuf::from(format!("/tmp/zucchini-prompt-{}.txt", unique));
        if let Err(e) = tokio::fs::write(&prompt_file, &prompt).await {
            error!("failed to write prompt file: {}", e);
            fail_agent(
                &tx,
                &topic_clone,
                format!("failed to write prompt file: {}", e),
            )
            .await;
            return;
        }

        let ctx = TurnContext {
            chat_id: &topic_clone,
            prompt_file: &prompt_file,
            project_path: project_path.as_deref(),
            worktree,
            agent_session_id: agent_session_id.as_deref(),
            is_sandboxed,
            model: model.as_deref(),
        };
        let cmd_string = match adapter.prepare_command(&ctx) {
            Ok(s) => s,
            Err(e) => {
                error!("adapter prepare_command failed: {}", e);
                let _ = tokio::fs::remove_file(&prompt_file).await;
                fail_agent(&tx, &topic_clone, format!("agent prepare failed: {}", e)).await;
                return;
            }
        };

        let user_shell = crate::shell::user_login_shell();
        info!(shell = %user_shell, kind = ?kind, "spawning agent via login shell");

        let mut cmd = Command::new(&user_shell);
        cmd.args(["-lic", &cmd_string])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .process_group(0); // new process group so we can kill shell + agent together
                               // Adapter system prompts reference both vars by name (see
                               // `adapters/claude.rs` / `adapters/cursor.rs` prompt strings) to tell
                               // the agent how to invoke `zucchini-spawner attach-file`. We export
                               // them on the spawn rather than relying on the user's shell rc so a
                               // stale PATH can't pick up the wrong binary — `current_exe()` is
                               // whatever launchd/systemd is actually running, which is what the
                               // RPC handler also listens on.
        cmd.env("ZUCCHINI_CHAT_ID", &topic_clone);
        if let Ok(exe) = std::env::current_exe() {
            cmd.env("ZUCCHINI_SPAWNER_BIN", exe);
        }
        let result = cmd.spawn();

        let mut child = match result {
            Ok(child) => child,
            Err(e) => {
                error!("failed to spawn agent: {}", e);
                let _ = tokio::fs::remove_file(&prompt_file).await;
                fail_agent(&tx, &topic_clone, format!("failed to spawn agent: {}", e)).await;
                return;
            }
        };

        // Shared flag: flipped to true once we receive the first stdout line from the agent.
        // The stderr task uses this to decide whether to buffer (startup noise) or warn (runtime).
        let agent_started = Arc::new(AtomicBool::new(false));
        let agent_started_stderr = agent_started.clone();

        // Read stderr in a separate task. Before the agent starts we buffer lines silently;
        // after startup any stderr is a genuine warning. We return the buffer so the main
        // task can report it to Sentry only if the agent never started.
        let stderr_handle = if let Some(stderr) = child.stderr.take() {
            let topic_for_stderr = topic_clone.clone();
            Some(tokio::spawn(async move {
                let reader = BufReader::new(stderr);
                let mut lines = reader.lines();
                let mut startup_buf: Vec<String> = Vec::new();
                while let Ok(Some(line)) = lines.next_line().await {
                    if agent_started_stderr.load(Ordering::Relaxed) {
                        warn!(topic = %topic_for_stderr, "agent stderr: {}", line);
                    } else if startup_buf.len() < 200 {
                        startup_buf.push(line);
                    }
                }
                startup_buf
            }))
        } else {
            None
        };

        let mut has_result = false;

        if let Some(stdout) = child.stdout.take() {
            let reader = BufReader::new(stdout);
            let mut lines = reader.lines();
            let startup_deadline = tokio::time::Instant::now() + AGENT_STARTUP_TIMEOUT;

            // `cancelled` is set by the inner per-send select when the outer
            // cancellation token fires mid-emit (slow writer back-pressures the mpsc
            // and a chatty turn can emit multiple events per line); we break out and
            // run the same SIGTERM/SIGKILL cleanup as the outer cancel arm.
            let mut cancelled_mid_send = false;
            loop {
                tokio::select! {
                    _ = token_clone.cancelled() => {
                        warn!(topic = %topic_clone, "agent cancelled, sending SIGTERM to process group");
                        terminate_agent_process_group(&mut child, &prompt_file, &topic_clone).await;
                        // Don't send Done — caller publishes INTERRUPTED_RESULT and has
                        // already removed our entry from the map.
                        return;
                    }
                    _ = tokio::time::sleep_until(startup_deadline), if !agent_started.load(Ordering::Relaxed) => {
                        error!(topic = %topic_clone, "agent produced no output within {:?}, killing", AGENT_STARTUP_TIMEOUT);
                        let _ = tx.send(AgentResponse::Line {
                            topic: topic_clone.clone(),
                            content: format!("Error: agent failed to start — no output within {:?}. Check shell configuration (~/.zshrc / ~/.bashrc).", AGENT_STARTUP_TIMEOUT),
                        }).await;
                        kill_agent_process_group(&mut child).await;
                        break;
                    }
                    line_result = lines.next_line() => {
                        match line_result {
                            Ok(Some(line)) => {
                                // First stdout line means the agent is alive — silences any
                                // later stderr buffering and stops the startup watchdog.
                                agent_started.store(true, Ordering::Relaxed);

                                let events = adapter.handle_line(line);
                                let mut channel_closed = false;
                                for ev in events {
                                    // Result is a supervisor-only signal — set the latch and
                                    // emit nothing on the wire. Every other event maps 1:1 to
                                    // an AgentResponse via `into_response`.
                                    let Some(resp) = ev.into_response(&topic_clone, &mut has_result) else {
                                        continue;
                                    };
                                    // Compute a follow-up attachment row BEFORE forwarding the
                                    // original — `attach_followup_for` peeks at the response
                                    // shape and drains the mailbox only when the line is an
                                    // assistant text frame with queued attachments. Held
                                    // over the original send so cancellation between the two
                                    // emits drops both cleanly.
                                    let followup = if let AgentResponse::Line { topic, content } = &resp {
                                        attach_followup_for(topic, content, &mut attach_rx)
                                    } else {
                                        None
                                    };
                                    // Nested select so a mid-loop cancellation (Stop
                                    // tapped while the bounded mpsc is full) is observed
                                    // immediately instead of waiting for the writer to
                                    // drain. `biased` keeps cancel polled first.
                                    tokio::select! {
                                        biased;
                                        _ = token_clone.cancelled() => {
                                            cancelled_mid_send = true;
                                            break;
                                        }
                                        send_res = tx.send(resp) => {
                                            if send_res.is_err() {
                                                channel_closed = true;
                                                break;
                                            }
                                        }
                                    }
                                    // Emit the attachment row immediately after the text frame
                                    // so iOS renders the paperclip pill bubble right below the
                                    // assistant text. Same cancellation contract as above.
                                    if let Some(followup) = followup {
                                        tokio::select! {
                                            biased;
                                            _ = token_clone.cancelled() => {
                                                cancelled_mid_send = true;
                                                break;
                                            }
                                            send_res = tx.send(followup) => {
                                                if send_res.is_err() {
                                                    channel_closed = true;
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                }
                                if cancelled_mid_send {
                                    break;
                                }
                                if channel_closed {
                                    warn!(topic = %topic_clone, "response channel closed, killing agent");
                                    let _ = child.kill().await;
                                    break;
                                }
                            }
                            Ok(None) => break,
                            Err(e) => {
                                error!(topic = %topic_clone, "error reading stdout: {}", e);
                                break;
                            }
                        }
                    }
                }
            }

            // Mid-send cancellation: same cleanup as the outer cancel arm
            // (SIGTERM → 5s wait → SIGKILL → wait → remove prompt file → return; no Done).
            if cancelled_mid_send {
                warn!(topic = %topic_clone, "agent cancelled mid-emit, sending SIGTERM to process group");
                terminate_agent_process_group(&mut child, &prompt_file, &topic_clone).await;
                return;
            }
        }

        if let Some(h) = stderr_handle {
            if let Ok(startup_buf) = h.await {
                if !agent_started.load(Ordering::Relaxed) && !startup_buf.is_empty() {
                    let stderr = startup_buf.join("\n");
                    error!(topic = %topic_clone, "agent failed to start. startup stderr:\n{}", stderr);
                    let _ = tx
                        .send(AgentResponse::Line {
                            topic: topic_clone.clone(),
                            content: format!("Error: agent failed to start.\n{}", stderr),
                        })
                        .await;
                }
            }
        }

        match child.wait().await {
            Ok(status) => info!(topic = %topic_clone, %status, "agent exited"),
            Err(e) => error!(topic = %topic_clone, "error waiting for agent: {}", e),
        }

        let _ = tokio::fs::remove_file(&prompt_file).await;

        // Post-turn context-token correction for adapters (codex) that read
        // occupancy from the on-disk transcript, now flushed after exit.
        // `None` (stream-sourced adapters, or a read miss) ⇒ gauge keeps its
        // last value, not zeroed.
        if let Some(tokens) = adapter.post_turn_context_tokens(agent_session_id.as_deref()) {
            let _ = tx
                .send(AgentResponse::ContextTokens {
                    topic: topic_clone.clone(),
                    tokens,
                })
                .await;
        }

        let _ = tx
            .send(AgentResponse::Done {
                topic: topic_clone,
                has_result,
            })
            .await;
    })
}

async fn await_agent_exit(handle: tokio::task::JoinHandle<()>, topic: &str) {
    match tokio::time::timeout(AGENT_EXIT_TIMEOUT, handle).await {
        Ok(_) => info!(topic = %topic, "agent exited"),
        Err(_) => warn!(topic = %topic, "agent did not exit in {:?}", AGENT_EXIT_TIMEOUT),
    }
}

/// Wait this long after SIGTERM before escalating to SIGKILL. Same value used
/// by the outer-cancel and mid-send-cancel arms.
const AGENT_SIGTERM_GRACE: Duration = Duration::from_secs(5);

/// Graceful kill: SIGTERM the whole process group → wait up to
/// `AGENT_SIGTERM_GRACE` → SIGKILL if still alive → wait → remove the prompt
/// file. Used by both the outer-cancel arm (token cancelled before the line
/// loop even gets a line) and the mid-send-cancel arm (token cancelled while
/// the line loop was draining events into the mpsc).
///
/// `process_group(0)` on spawn means `child.id()` IS the PGID, so a single
/// `killpg` reaches both the login shell and the agent process underneath it.
/// If `child.id()` is `None` (process already reaped) we fall back to
/// `child.kill()` which is a no-op in that case but keeps the API consistent.
async fn terminate_agent_process_group(
    child: &mut tokio::process::Child,
    prompt_file: &std::path::Path,
    topic: &str,
) {
    if let Some(pid) = child.id() {
        let pgid = Pid::from_raw(pid as i32);
        let _ = signal::killpg(pgid, Signal::SIGTERM);
        match tokio::time::timeout(AGENT_SIGTERM_GRACE, child.wait()).await {
            Ok(_) => info!(topic = %topic, "agent exited after SIGTERM"),
            Err(_) => {
                warn!(topic = %topic, "agent did not exit in {:?}, sending SIGKILL to process group", AGENT_SIGTERM_GRACE);
                let _ = signal::killpg(pgid, Signal::SIGKILL);
                let _ = child.wait().await;
            }
        }
    } else {
        let _ = child.kill().await;
    }
    let _ = tokio::fs::remove_file(prompt_file).await;
}

/// Fast-path kill: SIGKILL the process group immediately, no grace period.
/// Used only by the startup-deadline arm — if the agent has produced no output
/// in `AGENT_STARTUP_TIMEOUT` it's almost certainly hung on shell-rc init
/// rather than actively doing work, so SIGTERM grace is wasted wait. Caller
/// is responsible for removing the prompt file afterwards (the post-loop
/// cleanup path already does).
async fn kill_agent_process_group(child: &mut tokio::process::Child) {
    if let Some(pid) = child.id() {
        let pgid = Pid::from_raw(pid as i32);
        let _ = signal::killpg(pgid, Signal::SIGKILL);
    } else {
        let _ = child.kill().await;
    }
}

async fn fail_agent(tx: &mpsc::Sender<AgentResponse>, topic: &str, msg: String) {
    let _ = tx
        .send(AgentResponse::Line {
            topic: topic.to_string(),
            content: format!("Error: {}", msg),
        })
        .await;
    let _ = tx
        .send(AgentResponse::Done {
            topic: topic.to_string(),
            has_result: false,
        })
        .await;
}

/// True when `line` is a text-bearing assistant frame — i.e. an outer
/// `{"type":"assistant", ...}` whose `message.content[]` contains at least
/// one `{"type":"text", ...}` block. A tool_use-only assistant frame is
/// rejected: iOS renders those as system/tool rows that strip attachments,
/// so pinning a file there silently drops it. A mixed text + tool_use frame
/// still counts (the bubble carries the text).
///
/// Substring fast-rejects non-assistant frames (tool_result wraps in
/// `"type":"user"`, etc.) before doing the real `serde_json` parse. On parse
/// failure or unexpected shape we return false so attachments stay queued
/// for the next frame (no data loss).
fn is_assistant_text_frame(line: &str) -> bool {
    if !line.contains("\"type\":\"assistant\"") {
        return false;
    }
    let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
        return false;
    };
    let Some(content) = v
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_array())
    else {
        return false;
    };
    content
        .iter()
        .any(|b| b.get("type").and_then(|t| t.as_str()) == Some("text"))
}

/// If `content` is an assistant text frame AND `attach_rx` has any queued
/// attachments, build a follow-up `AgentResponse::Line` whose body is a
/// `MessageEnvelope { text: "", attachments }` JSON string. The caller emits
/// it immediately after the original text frame, producing a dedicated
/// "attachment-only" `messages` row that iOS renders as a paperclip pill
/// bubble below the assistant text. Returns `None` in every other case
/// (non-assistant frames, empty mailbox).
///
/// Why a separate row instead of wrapping the text frame: keeps the spawner's
/// "one stream-json frame per row, body never grows" invariant intact — the
/// attachment row is just another frame the spawner generated. The text-frame
/// body stays a verbatim claude-SDK frame, which is what `SpawnerMessageDescriber`
/// has always parsed, so there's no envelope-vs-raw branching on the iOS hot
/// path.
fn attach_followup_for(
    topic: &str,
    content: &str,
    attach_rx: &mut mpsc::UnboundedReceiver<EnvelopeAttachment>,
) -> Option<AgentResponse> {
    if !is_assistant_text_frame(content) {
        return None;
    }
    let mut attachments: Vec<EnvelopeAttachment> = Vec::new();
    while let Ok(att) = attach_rx.try_recv() {
        attachments.push(att);
    }
    if attachments.is_empty() {
        return None;
    }
    let n = attachments.len();
    let envelope = MessageEnvelope {
        text: String::new(),
        attachments,
    };
    match serde_json::to_string(&envelope) {
        Ok(s) => {
            info!(
                topic = %topic,
                n,
                "emitting follow-up attachment row with {} attachment(s)",
                n
            );
            Some(AgentResponse::Line {
                topic: topic.to_string(),
                content: s,
            })
        }
        Err(e) => {
            // Effectively unreachable — `MessageEnvelope` is a trivial
            // struct of plain types. Drop the attachments rather than
            // crashing the turn; the assistant text already went out.
            warn!(error = %e, "failed to serialize MessageEnvelope; dropping attachment row");
            None
        }
    }
}

#[cfg(test)]
mod attach_tests {
    use super::is_assistant_text_frame;

    #[test]
    fn text_only_assistant_frame_matches() {
        let f = r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"hi"}],"usage":{}}}"#;
        assert!(is_assistant_text_frame(f));
    }

    #[test]
    fn tool_use_only_assistant_frame_rejected() {
        let f = r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"Bash","input":{"command":"ls"}}],"usage":{}}}"#;
        assert!(!is_assistant_text_frame(f));
    }

    #[test]
    fn mixed_text_and_tool_use_matches() {
        let f = r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"let me check"},{"type":"tool_use","id":"t1","name":"Bash","input":{"command":"ls"}}],"usage":{}}}"#;
        assert!(is_assistant_text_frame(f));
    }

    #[test]
    fn user_tool_result_frame_rejected() {
        let f = r#"{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"ok"}]}}"#;
        assert!(!is_assistant_text_frame(f));
    }

    #[test]
    fn malformed_json_rejected() {
        assert!(!is_assistant_text_frame(r#"{"type":"assistant","#));
    }
}
