use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use nix::sys::signal::{self, Signal};
use nix::unistd::Pid;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn};
use std::time::{Duration, SystemTime};

const AGENT_EXIT_TIMEOUT: Duration = Duration::from_secs(10);
const AGENT_STARTUP_TIMEOUT: Duration = Duration::from_secs(30);

pub enum AgentResponse {
    Line { topic: String, content: String },
    Done { topic: String, has_result: bool },
}

/// Claude stores session transcripts at `~/.claude/projects/<encoded-path>/<session-id>.jsonl`,
/// where `encoded-path` is the absolute project path with every `/` turned into `-`. The file's
/// existence is our source of truth for "is this a first message" vs "resume an existing session":
/// passing `--session-id <uuid>` on an existing session fails ("Session ID already in use"), and
/// passing `--resume <uuid>` on a fresh one fails too. Checking the file removes the need to track
/// that bit in the mirror (and so it survives spawner restarts without extra persistence).
fn claude_session_file(project_path: &str, chat_id: &str) -> PathBuf {
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
    let encoded = project_path.replace('/', "-");
    home.join(".claude").join("projects").join(encoded).join(format!("{chat_id}.jsonl"))
}

pub struct Supervisor {
    agents: HashMap<String, (tokio::task::JoinHandle<()>, CancellationToken)>,
    response_tx: mpsc::Sender<AgentResponse>,
}

impl Supervisor {
    pub fn new(response_tx: mpsc::Sender<AgentResponse>) -> Self {
        Self {
            agents: HashMap::new(),
            response_tx,
        }
    }

    pub async fn abort_agent(&mut self, topic: &str) -> bool {
        if let Some((handle, token)) = self.agents.remove(topic) {
            info!(topic = %topic, "aborting running agent");
            token.cancel();
            // Wait so the old claude process is fully dead before spawning a new one on the same session.
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
        self.agents.get(topic).is_some_and(|(h, _)| !h.is_finished())
    }

    pub fn spawn_agent(&mut self, topic: String, prompt: String, project_path: Option<String>) {
        let tx = self.response_tx.clone();
        let topic_clone = topic.clone();
        let token = CancellationToken::new();
        let token_clone = token.clone();

        let handle = tokio::spawn(async move {
            // chat_id doubles as the claude session id — on the first message we create the
            // session with `--session-id <topic>`, on later messages we `--resume <topic>`.
            let is_resume = project_path
                .as_deref()
                .map(|pp| claude_session_file(pp, &topic_clone).exists())
                .unwrap_or(false);
            info!(topic = %topic_clone, resume = is_resume, project_path = ?project_path, "spawning claude agent");

            // Write prompt to a temp file so it never touches the shell command string
            let unique = format!("{}-{}", std::process::id(), SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH).unwrap_or_default().as_nanos());
            let prompt_file = format!("/tmp/zucchini-prompt-{}.txt", unique);
            if let Err(e) = tokio::fs::write(&prompt_file, &prompt).await {
                error!("failed to write prompt file: {}", e);
                let _ = tx.send(AgentResponse::Line {
                    topic: topic_clone.clone(),
                    content: format!("Error: failed to write prompt file: {}", e),
                }).await;
                let _ = tx.send(AgentResponse::Done { topic: topic_clone, has_result: false }).await;
                return;
            }

            let mut claude_cmd = String::new();
            if let Some(ref pp) = project_path {
                claude_cmd.push_str(&format!("cd {} && ", shell_escape(pp)));
            }
            claude_cmd.push_str(&format!("cat {} | claude", shell_escape(&prompt_file)));
            let session_flag = if is_resume { "--resume" } else { "--session-id" };
            claude_cmd.push_str(&format!(" {} {}", session_flag, shell_escape(&topic_clone)));
            claude_cmd.push_str(" --print --verbose --output-format stream-json --dangerously-skip-permissions");

            let user_shell = std::env::var("USER_SHELL").unwrap_or_else(|_| "/bin/sh".to_string());
            info!(shell = %user_shell, "spawning claude via login shell");

            let mut cmd = Command::new(&user_shell);
            cmd.args(["-lic", &claude_cmd])
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .process_group(0); // new process group so we can kill shell + claude together
            let result = cmd.spawn();

            let mut child = match result {
                Ok(child) => child,
                Err(e) => {
                    error!("failed to spawn claude: {}", e);
                    let _ = tokio::fs::remove_file(&prompt_file).await;
                    let _ = tx.send(AgentResponse::Line {
                        topic: topic_clone.clone(),
                        content: format!("Error: failed to spawn claude: {}", e),
                    }).await;
                    let _ = tx.send(AgentResponse::Done {
                        topic: topic_clone,
                        has_result: false,
                    }).await;
                    return;
                }
            };

            // Shared flag: flipped to true once we receive the first valid session JSON from claude.
            // The stderr task uses this to decide whether to buffer (startup noise) or warn (runtime).
            let claude_started = Arc::new(AtomicBool::new(false));
            let claude_started_stderr = claude_started.clone();

            // Read stderr in a separate task. Before claude starts we buffer lines silently;
            // after startup any stderr is a genuine warning. We return the buffer so the main
            // task can report it to Sentry only if claude never started.
            let stderr_handle = if let Some(stderr) = child.stderr.take() {
                let topic_for_stderr = topic_clone.clone();
                Some(tokio::spawn(async move {
                    let reader = BufReader::new(stderr);
                    let mut lines = reader.lines();
                    let mut startup_buf: Vec<String> = Vec::new();
                    while let Ok(Some(line)) = lines.next_line().await {
                        if claude_started_stderr.load(Ordering::Relaxed) {
                            warn!(topic = %topic_for_stderr, "claude stderr: {}", line);
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

                loop {
                    tokio::select! {
                        _ = token_clone.cancelled() => {
                            warn!(topic = %topic_clone, "agent cancelled, sending SIGTERM to process group");
                            if let Some(pid) = child.id() {
                                // child PID == PGID because we used process_group(0)
                                let pgid = Pid::from_raw(pid as i32);
                                let _ = signal::killpg(pgid, Signal::SIGTERM);
                                match tokio::time::timeout(
                                    std::time::Duration::from_secs(5),
                                    child.wait(),
                                ).await {
                                    Ok(_) => info!(topic = %topic_clone, "agent exited after SIGTERM"),
                                    Err(_) => {
                                        warn!(topic = %topic_clone, "agent did not exit in 5s, sending SIGKILL to process group");
                                        let _ = signal::killpg(pgid, Signal::SIGKILL);
                                        let _ = child.wait().await;
                                    }
                                }
                            } else {
                                let _ = child.kill().await;
                            }
                            let _ = tokio::fs::remove_file(&prompt_file).await;
                            // Don't send Done — caller publishes INTERRUPTED_RESULT and has
                            // already removed our entry from the map.
                            return;
                        }
                        _ = tokio::time::sleep_until(startup_deadline), if !claude_started.load(Ordering::Relaxed) => {
                            error!(topic = %topic_clone, "agent produced no output within {:?}, killing", AGENT_STARTUP_TIMEOUT);
                            let _ = tx.send(AgentResponse::Line {
                                topic: topic_clone.clone(),
                                content: format!("Error: agent failed to start — no output within {:?}. Check shell configuration (~/.zshrc / ~/.bashrc).", AGENT_STARTUP_TIMEOUT),
                            }).await;
                            if let Some(pid) = child.id() {
                                let pgid = Pid::from_raw(pid as i32);
                                let _ = signal::killpg(pgid, Signal::SIGKILL);
                            } else {
                                let _ = child.kill().await;
                            }
                            break;
                        }
                        line_result = lines.next_line() => {
                            match line_result {
                                Ok(Some(line)) => {
                                    // First stdout line means claude is alive — silences any
                                    // later stderr buffering and stops the startup watchdog.
                                    claude_started.store(true, Ordering::Relaxed);

                                    // Filter heavy messages (tool results, rate limits) without
                                    // a full JSON parse — they can be multi-MB of base64/file content.
                                    let mut skip = false;
                                    if line.starts_with('{') {
                                        if line.contains("\"type\":\"user\"") || line.contains("\"type\":\"rate_limit_event\"") {
                                            skip = true;
                                        } else if !has_result && line.contains("\"type\":\"result\"") {
                                            has_result = true;
                                        }
                                    }

                                    if !skip
                                        && tx.send(AgentResponse::Line {
                                            topic: topic_clone.clone(),
                                            content: line,
                                        }).await.is_err()
                                    {
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
            }

            if let Some(h) = stderr_handle {
                if let Ok(startup_buf) = h.await {
                    if !claude_started.load(Ordering::Relaxed) && !startup_buf.is_empty() {
                        let stderr = startup_buf.join("\n");
                        error!(topic = %topic_clone, "claude failed to start. startup stderr:\n{}", stderr);
                        let _ = tx.send(AgentResponse::Line {
                            topic: topic_clone.clone(),
                            content: format!("Error: claude failed to start.\n{}", stderr),
                        }).await;
                    }
                }
            }

            match child.wait().await {
                Ok(status) => info!(topic = %topic_clone, %status, "claude agent exited"),
                Err(e) => error!(topic = %topic_clone, "error waiting for claude: {}", e),
            }

            let _ = tokio::fs::remove_file(&prompt_file).await;

            let _ = tx.send(AgentResponse::Done {
                topic: topic_clone,
                has_result,
            }).await;
        });

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

async fn await_agent_exit(handle: tokio::task::JoinHandle<()>, topic: &str) {
    match tokio::time::timeout(AGENT_EXIT_TIMEOUT, handle).await {
        Ok(_) => info!(topic = %topic, "agent exited"),
        Err(_) => warn!(topic = %topic, "agent did not exit in {:?}", AGENT_EXIT_TIMEOUT),
    }
}

fn shell_escape(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}
