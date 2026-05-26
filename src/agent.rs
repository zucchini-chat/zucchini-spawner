//! Per-turn agent Supervisor. Owns the OS-side process lifecycle (spawn,
//! signal escalation on cancel, stderr buffering, startup watchdog, prompt
//! file write/cleanup, power assertion). The agent-specific bits — building
//! the CLI command, normalizing stdout frames into claude-shape envelopes,
//! harvesting session ids/usage/compact boundaries — live behind the
//! `AgentAdapter` trait in `adapter.rs` and the concrete adapters in
//! `adapters/`. Supervisor stays agent-agnostic; future adapters (codex,
//! hermes, gemini) plug in without touching this file.
//!
//! Each `spawn_agent` call constructs a fresh adapter, hands it to the spawned
//! task, and discards it when the turn ends.
use nix::sys::signal::{self, Signal};
use nix::unistd::Pid;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::{Duration, SystemTime};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn};

use crate::adapter::{AgentEvent, AgentKind, TurnContext};

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
pub type SpawnFn = Arc<
    dyn Fn(
            SpawnRequest,
            mpsc::Sender<AgentResponse>,
            CancellationToken,
        ) -> tokio::task::JoinHandle<()>
        + Send
        + Sync,
>;

pub struct Supervisor {
    agents: HashMap<String, (tokio::task::JoinHandle<()>, CancellationToken)>,
    response_tx: mpsc::Sender<AgentResponse>,
    spawn_fn: SpawnFn,
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
        }
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
        let handle = (self.spawn_fn)(req, self.response_tx.clone(), token.clone());
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
fn default_spawn_fn(
    req: SpawnRequest,
    tx: mpsc::Sender<AgentResponse>,
    token: CancellationToken,
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

    tokio::spawn(async move {
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
