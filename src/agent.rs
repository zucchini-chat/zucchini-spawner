use std::collections::HashMap;
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use nix::sys::signal::{self, Signal};
use nix::unistd::Pid;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};
use std::time::{Duration, SystemTime};

const AGENT_EXIT_TIMEOUT: Duration = Duration::from_secs(10);
const AGENT_STARTUP_TIMEOUT: Duration = Duration::from_secs(30);
/// Cap to skip full-line scans on multi-MB tool_result frames; usage frames are tiny.
const MAX_USAGE_FRAME_BYTES: usize = 65_536;

pub enum AgentResponse {
    Line { topic: String, content: String },
    ContextTokens { topic: String, tokens: i64 },
    /// Manual `/compact` or auto-compact completed; carries `compactMetadata.postTokens`.
    CompactBoundary { topic: String, post_tokens: i64 },
    Done { topic: String, has_result: bool },
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

    pub fn spawn_agent(
        &mut self,
        topic: String,
        prompt: String,
        project_path: Option<String>,
        worktree: bool,
        is_resume: bool,
    ) {
        let tx = self.response_tx.clone();
        let topic_clone = topic.clone();
        let token = CancellationToken::new();
        let token_clone = token.clone();

        let handle = tokio::spawn(async move {
            let _power_assertion = crate::power::AgentPowerAssertion::acquire();
            info!(topic = %topic_clone, resume = is_resume, project_path = ?project_path, worktree, "spawning claude agent");

            // Write prompt to a temp file so it never touches the shell command string
            let unique = format!("{}-{}", std::process::id(), SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH).unwrap_or_default().as_nanos());
            let prompt_file = format!("/tmp/zucchini-prompt-{}.txt", unique);
            if let Err(e) = tokio::fs::write(&prompt_file, &prompt).await {
                error!("failed to write prompt file: {}", e);
                fail_agent(&tx, &topic_clone, format!("failed to write prompt file: {}", e)).await;
                return;
            }

            let mut claude_cmd = String::new();
            if let Some(ref pp) = project_path {
                claude_cmd.push_str(&format!("cd {} && ", shell_escape(pp)));
            }
            claude_cmd.push_str(&format!("cat {} | claude", shell_escape(&prompt_file)));
            // chat_id doubles as the claude session id: first message creates the
            // session with --session-id, every subsequent message resumes it.
            let session_flag = if is_resume { "--resume" } else { "--session-id" };
            claude_cmd.push_str(&format!(" {} {}", session_flag, shell_escape(&topic_clone)));
            // claude --print is one-shot; once the result frame emits, the process
            // exits and the spawner stops reading stdout. Background work has no
            // way to reach the chat afterwards, so steer agents away from it.
            let mut sys = String::from(
                "You are spawned via a harness, no background subagents will wake you when finished, use subagents with `run_in_background: false` only."
            );

            if worktree {
                // Use the chat_id prefix so the worktree directory name stays short.
                let worktree_name: String = topic_clone.chars().take(8).collect();
                claude_cmd.push_str(&format!(" --worktree {}", shell_escape(&worktree_name)));
                // Plug the path-containment hole in --worktree: the harness chdirs into
                // the worktree but doesn't tell the agent (or its subagents) to stay
                // there, so absolute paths into the parent repo "just work" and edits
                // leak out. Inject the absolute worktree path with an explicit rule.
                if let Some(ref pp) = project_path {
                    let worktree_abs = format!("{}/.claude/worktrees/{}", pp.trim_end_matches('/'), worktree_name);
                    sys.push_str(&format!(
                        "\n\nWorktree: {}\nParent repo: {} (do not touch unless the user explicitly asks).\nKeep all edits and Bash commands inside the worktree. If a path under the parent repo appears in context, rewrite it to the worktree before calling Edit/Write/Bash. When delegating via Task, repeat this rule and the worktree path — subagents don't inherit it.",
                        worktree_abs, pp
                    ));
                }
            }
            claude_cmd.push_str(&format!(" --append-system-prompt {}", shell_escape(&sys)));
            claude_cmd.push_str(" --print --verbose --output-format stream-json --dangerously-skip-permissions --disallowedTools AskUserQuestion");

            let user_shell = crate::shell::user_login_shell();
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
                    fail_agent(&tx, &topic_clone, format!("failed to spawn claude: {}", e)).await;
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
            // Per-turn usage often repeats across consecutive frames (e.g. a
            // thinking frame and the text frame after it). Dedupe so the
            // writer doesn't fire a redundant /api/writes PATCH on each one.
            let mut last_emitted_tokens: Option<i64> = None;

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

                                    // Pre-skip-filter: thinking-only frames also carry usage.
                                    if line.len() < MAX_USAGE_FRAME_BYTES
                                        && line.starts_with('{')
                                        && line.contains("\"type\":\"assistant\"")
                                    {
                                        if let Some(tokens) = parse_assistant_usage(&line) {
                                            if last_emitted_tokens != Some(tokens) {
                                                last_emitted_tokens = Some(tokens);
                                                let _ = tx.send(AgentResponse::ContextTokens {
                                                    topic: topic_clone.clone(),
                                                    tokens,
                                                }).await;
                                            }
                                        }
                                    }

                                    // Skip frames the UI never renders so they don't flicker
                                    // the chat-list preview to empty between visible messages.
                                    // Substring match: tool_result bodies can be multi-MB, so
                                    // we avoid a full JSON parse on every line. Filtered:
                                    //   stream_event                — per-token deltas, high-freq
                                    //   system (non-status)         — init/shutdown frames
                                    //   user, rate_limit_event      — never shown
                                    //   assistant with only thinking — SpawnerMessageDescriber
                                    //     returns nil for these, so they contribute nothing to
                                    //     ChatView and would otherwise clobber `last_agent_body`
                                    //     with unreadable content (chat list falls back to
                                    //     "Thinking…" even though prior text/tool_use exists).
                                    //     JSON-string-escaping renders \"type\":\"text\" as
                                    //     `\"type\":\"text\"` in the source, so the substring
                                    //     below only matches structural block types.
                                    let mut skip = false;
                                    if line.starts_with('{') {
                                        if line.contains("\"type\":\"system\"")
                                            && !line.contains("\"subtype\":\"status\"")
                                        {
                                            // System frames are skipped, but compact_boundary
                                            // carries postTokens we need — harvest it here
                                            // instead of dropping the line silently.
                                            if line.contains("\"subtype\":\"compact_boundary\"") {
                                                if let Some(post_tokens) = parse_compact_post_tokens(&line) {
                                                    let _ = tx.send(AgentResponse::CompactBoundary {
                                                        topic: topic_clone.clone(),
                                                        post_tokens,
                                                    }).await;
                                                }
                                            }
                                            skip = true;
                                        } else if line.contains("\"type\":\"stream_event\"")
                                            || line.contains("\"type\":\"user\"")
                                            || line.contains("\"type\":\"rate_limit_event\"")
                                            || (line.contains("\"type\":\"assistant\"")
                                                && !line.contains("\"type\":\"text\"")
                                                && !line.contains("\"type\":\"tool_use\""))
                                        {
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

async fn fail_agent(tx: &mpsc::Sender<AgentResponse>, topic: &str, msg: String) {
    let _ = tx.send(AgentResponse::Line {
        topic: topic.to_string(),
        content: format!("Error: {}", msg),
    }).await;
    let _ = tx.send(AgentResponse::Done {
        topic: topic.to_string(),
        has_result: false,
    }).await;
}

/// Cumulative for the current turn — caller overwrites `chats.context_tokens`
/// with each emission. Uses a narrow Deserialize struct so serde skips the rest
/// of the frame (text blocks, tool calls) without allocating it.
fn parse_assistant_usage(line: &str) -> Option<i64> {
    #[derive(serde::Deserialize)]
    struct Frame { message: Message }
    #[derive(serde::Deserialize)]
    struct Message { usage: Usage }
    #[derive(serde::Deserialize, Default)]
    struct Usage {
        #[serde(default)] input_tokens: i64,
        #[serde(default)] cache_creation_input_tokens: i64,
        #[serde(default)] cache_read_input_tokens: i64,
    }
    match serde_json::from_str::<Frame>(line) {
        Ok(f) => Some(f.message.usage.input_tokens
            + f.message.usage.cache_creation_input_tokens
            + f.message.usage.cache_read_input_tokens),
        Err(e) => {
            debug!("failed to parse assistant frame for usage: {}", e);
            None
        }
    }
}

/// Reads `compactMetadata.postTokens` from a `compact_boundary` system frame.
/// Narrow Deserialize struct so serde skips the rest of the frame without allocating it.
fn parse_compact_post_tokens(line: &str) -> Option<i64> {
    #[derive(serde::Deserialize)]
    struct Frame { #[serde(rename = "compactMetadata")] metadata: Metadata }
    #[derive(serde::Deserialize)]
    struct Metadata { #[serde(rename = "postTokens")] post_tokens: i64 }
    match serde_json::from_str::<Frame>(line) {
        Ok(f) => Some(f.metadata.post_tokens),
        Err(e) => {
            debug!("failed to parse compact_boundary frame: {}", e);
            None
        }
    }
}
