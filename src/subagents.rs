//! Holding a background subagent in its turn while background subagents it
//! launched still run. Under `claude --print` a subagent that ends its turn
//! counts as done even then: its interim text reaches the main session as its
//! report, and its children's reports land in the main session after it. Our
//! `SubagentStop` hook (`subagent-stop-hook`, wired in `adapters/claude.rs`)
//! blocks that stop and has the subagent run `subagents wait`, which returns
//! as soon as something for it arrives: a child's report or a message. Claude
//! hands those over only at a tool boundary, so the subagent has to be in a
//! tool call that ends when they come.

use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use crate::agent::{LiveSessions, LiveTask, SessionState};

/// Under the Bash tool's default 2-minute timeout, past which claude moves the
/// command to the background.
pub const WAIT_TIMEOUT: Duration = Duration::from_secs(110);
/// How long the hook's check waits for the reader to catch up with the tasks
/// claude listed as running.
pub const HOOK_CATCH_UP: Duration = Duration::from_secs(2);
/// The hook has 10 s in `--settings`; a stuck daemon must not use them up.
const HOOK_RPC_TIMEOUT: Duration = Duration::from_secs(5);

const WAIT_USAGE: &str = "usage: zucchini-spawner subagents wait --agent <ID>
Waits while background subagents you launched run. Run it with the Bash tool in
the foreground when a hook tells you to. It returns when one of them finishes,
when a message to you arrives, or after about 2 minutes, and prints why. What
arrived reaches you right after it returns.";

/// A wait's line for a message delivered to its subagent.
pub(crate) const MESSAGE_ARRIVED: &str =
    "A message to you arrived. It reaches you right after this command.";

/// A wait's line for the end of a background subagent its subagent launched.
pub(crate) fn child_ended(label: &str) -> String {
    format!("Your background subagent \"{label}\" finished. Its report reaches you right after this command.")
}

fn session_of(sessions: &LiveSessions, chat_id: &str) -> Option<Arc<StdMutex<SessionState>>> {
    sessions
        .lock()
        .expect("LiveSessions mutex")
        .get(chat_id)
        .cloned()
}

fn is_live_subagent(st: &SessionState, agent_id: &str) -> bool {
    st.live_tasks
        .iter()
        .any(|t| t.id == agent_id && t.is_subagent())
}

/// The running background subagents `agent_id` launched.
fn children<'a>(st: &'a SessionState, agent_id: &'a str) -> impl Iterator<Item = &'a LiveTask> {
    st.live_tasks
        .iter()
        .filter(move |t| t.is_subagent() && t.parent_task_id.as_deref() == Some(agent_id))
}

/// `children`, quoted; empty if none.
fn running_children(st: &SessionState, agent_id: &str) -> String {
    let labels: Vec<String> = children(st, agent_id)
        .map(|t| format!("\"{}\"", t.label()))
        .collect();
    labels.join(", ")
}

/// Why subagent `agent_id` of `chat_id` must not end its turn yet: background
/// subagents it launched still run. `None` lets it stop. `running` are the
/// subagents claude listed as running when the hook fired: the reader may not
/// have read their `task_started` yet, so it is given up to `catch_up` to.
pub async fn stop_reason(
    sessions: &LiveSessions,
    chat_id: &str,
    agent_id: &str,
    running: &[String],
    catch_up: Duration,
) -> Option<String> {
    let session = session_of(sessions, chat_id)?;
    let events = Arc::clone(&session.lock().expect("SessionState mutex").subagent_events);
    let deadline = tokio::time::Instant::now() + catch_up;
    loop {
        let notified = events.notified();
        tokio::pin!(notified);
        notified.as_mut().enable();
        let known = {
            let st = session.lock().expect("SessionState mutex");
            st.closed
                || running
                    .iter()
                    .all(|id| st.live_tasks.iter().any(|t| &t.id == id))
        };
        if known || tokio::time::timeout_at(deadline, notified).await.is_err() {
            break;
        }
    }
    let st = session.lock().expect("SessionState mutex");
    let children = running_children(&st, agent_id);
    if st.closed || !is_live_subagent(&st, agent_id) || children.is_empty() {
        return None;
    }
    Some(format!(
        "Background subagents you launched are still running: {children}. Their reports reach \
         you only while your turn runs, so do not end it yet. Run \
         `\"$ZUCCHINI_SPAWNER_BIN\" subagents wait --agent {agent_id}` with the Bash tool, in the \
         foreground (not in the background). It returns when one of them finishes or a message \
         to you arrives, and you get the report or message right after it returns. Then go on, \
         and end your turn once they are all done."
    ))
}

/// What a wait returns: the line for its caller, and whether that line is a
/// pending wake it took, to put back if the caller is gone (`put_back`).
pub struct Waited {
    pub line: String,
    pub took_wake: bool,
}

/// A wake a wait took for `agent_id` of `chat_id`.
#[derive(Debug)]
pub struct TakenWake {
    pub chat_id: String,
    pub agent_id: String,
    pub line: String,
}

/// Waits until there is something for subagent `agent_id` of `chat_id` to read
/// and says what, or why there is nothing to wait for. A newer wait for the
/// same subagent ends this one, which then takes no wake: its caller was
/// killed or moved to the background.
pub async fn wait(
    sessions: &LiveSessions,
    chat_id: &str,
    agent_id: &str,
    timeout: Duration,
) -> Waited {
    let Some(session) = session_of(sessions, chat_id) else {
        return said("This chat has no running agent session, so there is nothing to wait for.");
    };
    let (events, generation) = {
        let mut st = session.lock().expect("SessionState mutex");
        // No entry for an id that is no subagent: only its end would drop it.
        let generation = if is_live_subagent(&st, agent_id) {
            let waits = st.subagent_waits.entry(agent_id.to_string()).or_default();
            *waits += 1;
            *waits
        } else {
            0
        };
        (Arc::clone(&st.subagent_events), generation)
    };
    events.notify_waiters();
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        // Enabled before the check, so an event between the two still wakes us.
        let notified = events.notified();
        tokio::pin!(notified);
        notified.as_mut().enable();
        let current = session_of(sessions, chat_id).is_some_and(|s| Arc::ptr_eq(&s, &session));
        let waited = wake(
            &mut session.lock().expect("SessionState mutex"),
            agent_id,
            current,
            generation,
        );
        if let Some(waited) = waited {
            return waited;
        }
        if tokio::time::timeout_at(deadline, notified).await.is_err() {
            let st = session.lock().expect("SessionState mutex");
            return said(format!(
                "Still running after {} s: {}. Run this command again to keep waiting.",
                timeout.as_secs(),
                running_children(&st, agent_id)
            ));
        }
    }
}

fn said(line: impl Into<String>) -> Waited {
    Waited {
        line: line.into(),
        took_wake: false,
    }
}

/// Why the wait number `generation` of `agent_id` returns now, if it does.
/// Takes its pending wake. `current`: the session is still the chat's.
fn wake(st: &mut SessionState, agent_id: &str, current: bool, generation: u64) -> Option<Waited> {
    if st.closed || !current {
        return Some(said("The agent session of this chat ended."));
    }
    if !is_live_subagent(st, agent_id) {
        return Some(said(format!(
            "{agent_id} is not a running background subagent of this chat."
        )));
    }
    if st.subagent_waits.get(agent_id) != Some(&generation) {
        return Some(said("This wait was replaced by a newer one."));
    }
    if let Some(line) = st.subagent_wakes.remove(agent_id) {
        return Some(Waited {
            line,
            took_wake: true,
        });
    }
    if children(st, agent_id).next().is_none() {
        return Some(said(
            "None of the background subagents you launched is still running.",
        ));
    }
    None
}

/// Gives a taken wake back when the answer that carried it never reached the
/// wait's caller, unless a newer wake is already pending.
pub fn put_back(sessions: &LiveSessions, wake: TakenWake) {
    let Some(session) = session_of(sessions, &wake.chat_id) else {
        return;
    };
    let mut st = session.lock().expect("SessionState mutex");
    if is_live_subagent(&st, &wake.agent_id) {
        st.subagent_wakes.entry(wake.agent_id).or_insert(wake.line);
        st.subagent_events.notify_waiters();
    }
}

/// `subagent-stop-hook`: claude's `SubagentStop` hook. Prints a `block`
/// decision while the stopping subagent has background subagents running.
/// Fails open: on any error it prints nothing. Always exits 0.
pub async fn run_stop_hook_cli(chat_id: Option<String>) {
    use std::io::Read;
    let mut payload = String::new();
    let _ = std::io::stdin().read_to_string(&mut payload);
    let hook: serde_json::Value = serde_json::from_str(&payload).unwrap_or_default();
    let agent_id = hook["agent_id"].as_str();
    if let (Some(chat_id), Some(agent_id)) = (chat_id, agent_id) {
        if let Some(running) = other_running_subagents(&hook, agent_id) {
            let check = crate::control::subagent_stop_via_socket(&chat_id, agent_id, running);
            if let Ok(Ok(Some(reason))) = tokio::time::timeout(HOOK_RPC_TIMEOUT, check).await {
                println!(
                    "{}",
                    serde_json::json!({ "decision": "block", "reason": reason })
                );
            }
        }
    }
    std::process::exit(0);
}

/// The subagents other than `agent_id` that the hook input's `background_tasks`
/// (claude's own list when the hook fired) shows running. `None` when it shows
/// none, so `agent_id` has no child to wait for; an empty list when the CLI
/// sends no such field.
fn other_running_subagents(hook: &serde_json::Value, agent_id: &str) -> Option<Vec<String>> {
    let Some(tasks) = hook["background_tasks"].as_array() else {
        return Some(Vec::new());
    };
    let running: Vec<String> = tasks
        .iter()
        .filter(|t| t["type"] == "subagent" && t["status"] == "running" && t["id"] != agent_id)
        .filter_map(|t| t["id"].as_str().map(str::to_string))
        .collect();
    (!running.is_empty()).then_some(running)
}

/// `subagents wait --agent <id>`. Fails open: an error returns at once, exit 0,
/// with a line saying so, since the hook will let the subagent stop then too.
pub async fn run_cli(args: &[String], chat_id: Option<String>) {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        crate::usage_and_exit(WAIT_USAGE, 0);
    }
    let mut agent_id = None;
    let mut it = args.iter().skip(1);
    while let Some(a) = it.next() {
        if let Some(v) = crate::take_flag(a, "--agent", &mut it) {
            agent_id = Some(v);
        } else {
            crate::usage_and_exit(WAIT_USAGE, 2);
        }
    }
    let (Some("wait"), Some(agent_id)) = (args.first().map(String::as_str), agent_id) else {
        crate::usage_and_exit(WAIT_USAGE, 2);
    };
    let result = match chat_id {
        Some(chat_id) => {
            let wait = crate::control::wait_subagent_via_socket(&chat_id, &agent_id);
            tokio::time::timeout(WAIT_TIMEOUT + Duration::from_secs(5), wait)
                .await
                .unwrap_or_else(|_| Err(anyhow::anyhow!("no answer")))
        }
        None => Err(anyhow::anyhow!("ZUCCHINI_CHAT_ID is not set")),
    };
    match result {
        Ok(why) => println!("{why}"),
        Err(e) => println!("Could not wait for your subagents ({e:#}). Go on without waiting."),
    }
    std::process::exit(0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::AgentEvent;
    use crate::agent::{reduce, subagent_started, task_started};
    use std::collections::HashMap;

    const CHAT: &str = "chat";

    fn child_of(id: &str, parent: &str) -> AgentEvent {
        let mut ev = subagent_started(id);
        if let AgentEvent::TaskStarted { parent_task_id, .. } = &mut ev {
            *parent_task_id = Some(parent.into());
        }
        ev
    }

    fn named(id: &str, given: &str) -> AgentEvent {
        let mut ev = subagent_started(id);
        if let AgentEvent::TaskStarted { name, .. } = &mut ev {
            *name = Some(given.into());
        }
        ev
    }

    fn sessions_with(events: &[AgentEvent]) -> (LiveSessions, Arc<StdMutex<SessionState>>) {
        let mut st = SessionState::default();
        for ev in events {
            reduce(&mut st, ev);
        }
        let session = Arc::new(StdMutex::new(st));
        let sessions: LiveSessions = Arc::new(StdMutex::new(HashMap::from([(
            CHAT.to_string(),
            Arc::clone(&session),
        )])));
        (sessions, session)
    }

    fn apply(session: &Arc<StdMutex<SessionState>>, ev: AgentEvent) {
        reduce(&mut session.lock().unwrap(), &ev);
    }

    /// B launched C: C's end goes to B, not the main session, while B runs;
    /// once B is gone it is the main session's again.
    #[test]
    fn a_nested_subagents_end_owes_the_main_session_only_without_its_parent() {
        let mut held = SessionState::default();
        reduce(&mut held, &subagent_started("B"));
        reduce(&mut held, &child_of("C", "B"));
        assert_eq!(
            held.live_tasks[1].parent_task_id.as_deref(),
            Some("B"),
            "parent kept"
        );
        reduce(&mut held, &AgentEvent::TaskFinished("C".into()));
        assert!(held.owed_notifications.is_empty(), "{held:?}");
        assert_eq!(
            held.subagent_wakes.get("B"),
            Some(&child_ended("C description"))
        );

        let mut orphaned = SessionState::default();
        reduce(&mut orphaned, &subagent_started("B"));
        reduce(&mut orphaned, &child_of("C", "B"));
        reduce(&mut orphaned, &AgentEvent::TaskFinished("B".into()));
        reduce(&mut orphaned, &AgentEvent::TaskFinished("C".into()));
        assert_eq!(orphaned.owed_notifications, vec!["B", "C"]);
        assert!(orphaned.subagent_wakes.is_empty(), "nothing left for B");
    }

    /// A subagent's shells and monitors never hold it: they can run forever.
    /// The hook's check with nothing to catch up on.
    async fn stop(sessions: &LiveSessions, chat_id: &str, agent_id: &str) -> Option<String> {
        stop_reason(sessions, chat_id, agent_id, &[], Duration::ZERO).await
    }

    #[tokio::test]
    async fn only_subagent_children_hold_a_subagent() {
        let mut shell = task_started("shell");
        if let AgentEvent::TaskStarted { parent_task_id, .. } = &mut shell {
            *parent_task_id = Some("B".into());
        }
        let (sessions, session) = sessions_with(&[subagent_started("B"), shell]);
        assert_eq!(stop(&sessions, CHAT, "B").await, None);
        apply(&session, AgentEvent::TaskFinished("shell".into()));
        assert!(
            session.lock().unwrap().subagent_wakes.is_empty(),
            "a shell's end wakes nobody"
        );
    }

    #[tokio::test]
    async fn the_hook_blocks_a_subagent_while_its_subagents_run() {
        let (sessions, session) = sessions_with(&[
            subagent_started("B"),
            child_of("C", "B"),
            child_of("D", "B"),
            subagent_started("E"),
        ]);
        let reason = stop(&sessions, CHAT, "B").await.expect("blocks");
        assert!(
            reason.contains("\"C description\", \"D description\""),
            "{reason}"
        );
        assert!(
            reason.contains("`\"$ZUCCHINI_SPAWNER_BIN\" subagents wait --agent B`"),
            "{reason}"
        );
        assert!(reason.contains("in the foreground"), "{reason}");

        assert_eq!(stop(&sessions, CHAT, "E").await, None, "no children");
        assert_eq!(stop(&sessions, CHAT, "C").await, None, "no children");
        assert_eq!(stop(&sessions, "other", "B").await, None, "unknown chat");
        assert_eq!(stop(&sessions, CHAT, "ghost").await, None, "unknown agent");
        apply(&session, AgentEvent::TaskFinished("C".into()));
        apply(&session, AgentEvent::TaskFinished("D".into()));
        assert_eq!(stop(&sessions, CHAT, "B").await, None, "children done");
        session.lock().unwrap().closed = true;
        apply(&session, child_of("F", "B"));
        assert_eq!(stop(&sessions, CHAT, "B").await, None, "session closed");
    }

    /// claude may run the hook before the reader has read the child's
    /// `task_started`: the check waits for the subagents claude listed.
    #[tokio::test]
    async fn the_hook_waits_for_the_reader_to_know_the_listed_subagents() {
        let (sessions, session) = sessions_with(&[subagent_started("B")]);
        let check = {
            let sessions = Arc::clone(&sessions);
            tokio::spawn(
                async move { stop_reason(&sessions, CHAT, "B", &["C".into()], SHORT).await },
            )
        };
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!check.is_finished());
        apply(&session, child_of("C", "B"));
        let reason = tokio::time::timeout(Duration::from_secs(1), check)
            .await
            .expect("decided once C is known")
            .unwrap();
        assert!(reason.is_some_and(|r| r.contains("\"C description\"")));

        let started = tokio::time::Instant::now();
        let ghost = ["ghost".to_string()];
        let unseen = stop_reason(&sessions, CHAT, "B", &ghost, Duration::from_millis(100)).await;
        assert!(unseen.is_some(), "decides on what it knows");
        assert!(
            started.elapsed() >= Duration::from_millis(100),
            "after the bound"
        );
    }

    /// Probe shape (`tmp/nested-probe`): `background_tasks` lists every
    /// background task of the session, the stopping subagent included.
    #[test]
    fn the_hook_asks_only_while_claude_lists_another_running_subagent() {
        let hook = |tasks: serde_json::Value| serde_json::json!({"agent_id": "B", "background_tasks": tasks});
        let b = serde_json::json!({"id": "B", "type": "subagent", "status": "running", "description": "B"});
        let c = serde_json::json!({"id": "C", "type": "subagent", "status": "running", "description": "C"});
        let shell = serde_json::json!({"id": "s", "type": "shell", "status": "running", "command": "sleep 60"});
        let done = serde_json::json!({"id": "D", "type": "subagent", "status": "completed"});

        let none = hook(serde_json::json!([b, shell, done]));
        assert_eq!(other_running_subagents(&none, "B"), None);
        let with_c = hook(serde_json::json!([b, c, shell]));
        assert_eq!(
            other_running_subagents(&with_c, "B"),
            Some(vec!["C".into()])
        );
        let old_cli = serde_json::json!({"agent_id": "B"});
        assert_eq!(other_running_subagents(&old_cli, "B"), Some(vec![]));
    }

    const SHORT: Duration = Duration::from_secs(5);

    /// Starts a wait for B and gives it time to register.
    async fn waiting(sessions: &LiveSessions) -> tokio::task::JoinHandle<String> {
        let sessions = Arc::clone(sessions);
        let handle = tokio::spawn(async move { wait(&sessions, CHAT, "B", SHORT).await.line });
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!handle.is_finished(), "{:?}", handle.await);
        handle
    }

    async fn returned(handle: tokio::task::JoinHandle<String>) -> String {
        tokio::time::timeout(Duration::from_secs(1), handle)
            .await
            .expect("the wait returned")
            .unwrap()
    }

    fn b_with_children() -> (LiveSessions, Arc<StdMutex<SessionState>>) {
        sessions_with(&[
            subagent_started("B"),
            child_of("C", "B"),
            child_of("D", "B"),
        ])
    }

    #[tokio::test]
    async fn a_wait_returns_when_a_child_ends() {
        let (sessions, session) = b_with_children();
        let handle = waiting(&sessions).await;
        apply(&session, AgentEvent::TaskFinished("C".into()));
        let why = returned(handle).await;
        assert!(why.contains("\"C description\" finished"), "{why}");
        assert!(session.lock().unwrap().subagent_wakes.is_empty(), "taken");
    }

    #[tokio::test]
    async fn a_wait_returns_on_a_message_by_id() {
        let (sessions, session) = b_with_children();
        let handle = waiting(&sessions).await;
        apply(&session, AgentEvent::SendMessageTo("main".into()));
        apply(&session, AgentEvent::SendMessageTo("C".into()));
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!handle.is_finished(), "a message to someone else");
        apply(&session, AgentEvent::SendMessageTo("B".into()));
        assert!(returned(handle).await.contains("A message to you arrived"));
    }

    #[tokio::test]
    async fn a_wait_returns_on_a_message_by_name() {
        let (sessions, session) = sessions_with(&[named("B", "researcher"), child_of("C", "B")]);
        let handle = waiting(&sessions).await;
        apply(
            &session,
            AgentEvent::SendMessageTo("researcher [x7k2]".into()),
        );
        assert!(returned(handle).await.contains("A message to you arrived"));
    }

    #[tokio::test]
    async fn a_wait_returns_when_nothing_is_left_to_wait_for() {
        let (sessions, session) = sessions_with(&[subagent_started("B")]);
        let childless = wait(&sessions, CHAT, "B", SHORT).await.line;
        assert!(
            childless.contains("None of the background subagents"),
            "{childless}"
        );

        apply(&session, child_of("C", "B"));
        let handle = waiting(&sessions).await;
        apply(&session, AgentEvent::TaskFinished("B".into()));
        let gone = returned(handle).await;
        assert!(gone.contains("not a running background subagent"), "{gone}");

        let unknown = wait(&sessions, "other", "B", SHORT).await.line;
        assert!(unknown.contains("no running agent session"), "{unknown}");

        let ghost = wait(&sessions, CHAT, "ghost", SHORT).await.line;
        assert!(
            ghost.contains("not a running background subagent"),
            "{ghost}"
        );
        assert!(
            session.lock().unwrap().subagent_waits.is_empty(),
            "nothing kept for an id that is no subagent, or for B after its end"
        );
    }

    #[tokio::test]
    async fn a_wait_returns_when_the_session_closes() {
        let (sessions, session) = b_with_children();
        let handle = waiting(&sessions).await;
        crate::agent::close_session(&session);
        assert!(returned(handle)
            .await
            .contains("session of this chat ended"));
    }

    #[tokio::test]
    async fn a_wait_returns_when_the_session_is_replaced() {
        let (sessions, session) = b_with_children();
        let handle = waiting(&sessions).await;
        let replacement = Arc::new(StdMutex::new(SessionState::default()));
        sessions.lock().unwrap().insert(CHAT.into(), replacement);
        session.lock().unwrap().subagent_events.notify_waiters();
        assert!(returned(handle)
            .await
            .contains("session of this chat ended"));
    }

    /// A bare name goes to the latest subagent that has it.
    #[tokio::test]
    async fn a_message_by_name_wakes_the_latest_subagent_with_it() {
        let (sessions, session) = sessions_with(&[
            named("old", "researcher"),
            named("B", "researcher"),
            child_of("C", "B"),
        ]);
        apply(&session, AgentEvent::SendMessageTo("researcher".into()));
        let wakes = session.lock().unwrap().subagent_wakes.clone();
        assert!(
            wakes.contains_key("B") && !wakes.contains_key("old"),
            "{wakes:?}"
        );
        let why = wait(&sessions, CHAT, "B", SHORT).await.line;
        assert_eq!(why, MESSAGE_ARRIVED);
    }

    /// A wait claude gave up on (its Bash call timed out) must not take a
    /// wake from the wait that replaced it.
    #[tokio::test]
    async fn a_newer_wait_replaces_an_older_one_without_its_wake() {
        let (sessions, session) = b_with_children();
        let older = waiting(&sessions).await;
        let newer = {
            let sessions = Arc::clone(&sessions);
            tokio::spawn(async move { wait(&sessions, CHAT, "B", SHORT).await })
        };
        assert!(returned(older).await.contains("replaced by a newer one"));
        apply(&session, AgentEvent::TaskFinished("C".into()));
        let waited = tokio::time::timeout(Duration::from_secs(1), newer)
            .await
            .expect("the newer wait returned")
            .unwrap();
        assert!(waited.took_wake, "{}", waited.line);
        assert_eq!(waited.line, child_ended("C description"));
    }

    /// An event no wait was there for is not lost: the next wait returns on it
    /// at once, and only that one.
    #[tokio::test]
    async fn a_wait_returns_at_once_on_an_earlier_event() {
        let (sessions, session) = b_with_children();
        apply(&session, AgentEvent::SendMessageTo("B".into()));
        let why = wait(&sessions, CHAT, "B", SHORT).await.line;
        assert!(why.contains("A message to you arrived"), "{why}");
        let handle = waiting(&sessions).await;
        handle.abort();
    }

    #[tokio::test]
    async fn a_wait_gives_up_after_its_timeout() {
        let (sessions, _session) = b_with_children();
        let why = wait(&sessions, CHAT, "B", Duration::from_millis(100))
            .await
            .line;
        assert_eq!(
            why,
            "Still running after 0 s: \"C description\", \"D description\". Run this command again to keep waiting."
        );
    }
}

/// Opt-in end-to-end run against the real claude CLI (real model calls):
/// `cargo build && cargo test nested_subagents_smoke -- --ignored --nocapture`.
/// A resident main session launches background subagent B, B launches
/// background subagent C and ends its turn; the hook holds B until C's report
/// reaches it, and a message to B while it waits wakes it at once. Raw stdout
/// lands in `/tmp/nested-smoke/`.
#[cfg(test)]
mod cli_smoke {
    use super::*;
    use crate::adapter::AgentKind;
    use crate::agent::{AgentResponse, SpawnRequest, Supervisor, STDOUT_TAP};
    use crate::user_prompt::UserSystemPrompt;
    use serde_json::Value;
    use std::time::Instant;

    const DIR: &str = "/tmp/nested-smoke";
    const CHAT: &str = "5c0de000-0000-4000-8000-0000000000b2";

    fn req(prompt: &str) -> SpawnRequest {
        SpawnRequest {
            chat_id: CHAT.to_string(),
            prompt: prompt.to_string(),
            project_path: Some(format!("{DIR}/project")),
            worktree: false,
            agent_session_id: None,
            agent_kind: AgentKind::Claude,
            is_sandboxed: false,
            model: Some("sonnet".to_string()),
            user_timezone: None,
            session_name: None,
            user_system_prompt: UserSystemPrompt::default(),
        }
    }

    fn prompts() -> (String, &'static str) {
        let c = "Run this Bash command in the foreground (not in the background): \
                 perl -e 'sleep 45'; echo C-DONE. Then reply with exactly: C-REPORT <the command output>";
        let b = format!(
            "Launch ONE subagent with the Agent tool: subagent_type general-purpose, \
             run_in_background: true, description \"C\", prompt: \"{c}\". After launching C, end \
             your turn at once with the text B-WAITING. Do not poll or sleep on your own unless a \
             hook tells you to. When C's report reaches you, reply with B-FINAL followed by C's \
             report. If you get a message asking you to reply, do what it says."
        );
        let main = format!(
            "This is a test of nested background subagents. Do exactly this:\n\
             1. Launch ONE subagent with the Agent tool: subagent_type general-purpose, \
             run_in_background: true, description \"B\", prompt: \"{b}\"\n\
             2. End your turn with the text MAIN-WAITING.\n\
             3. When B's report arrives, reply MAIN-GOT followed by B's report verbatim.\n\
             4. If a report from any other agent reaches you, reply RELAY-SEEN <agent description> \
             <report> and do NOT forward it to anyone."
        );
        let ping = "Use SendMessage to tell agent B exactly this: 'PING. Reply to main via \
                    SendMessage with the word ACK and the sum 7700+31, then keep waiting for C.' \
                    Then end your turn.";
        (main, ping)
    }

    fn texts(frame: &Value) -> impl Iterator<Item = String> + '_ {
        blocks(frame)
            .filter(|b| b["type"] == "text")
            .filter_map(|b| b["text"].as_str().map(str::to_string))
    }

    fn blocks(frame: &Value) -> impl Iterator<Item = &Value> {
        frame["message"]["content"].as_array().into_iter().flatten()
    }

    #[derive(Default)]
    struct Run {
        b: Option<(String, String)>,
        c: Option<String>,
        blocks: Vec<f64>,
        waits: Vec<(String, f64)>,
        /// `(call id, time, text)` of each wait's result.
        wait_returns: Vec<(String, f64, String)>,
        ping: Option<f64>,
        /// Main received B's ACK (its peer turn).
        ack: Option<f64>,
        c_end: Option<f64>,
        b_final: Option<String>,
        main_texts: Vec<String>,
        done: Option<f64>,
        picked: Vec<String>,
        session_id: Option<String>,
    }

    impl Run {
        fn observe(&mut self, t: f64, line: &str, f: &Value) {
            let parent = f["parent_tool_use_id"].as_str();
            let b_call = self.b.as_ref().map(|(_, call)| call.as_str());
            let b_id = self.b.as_ref().map(|(id, _)| id.clone());
            match (f["type"].as_str(), f["subtype"].as_str()) {
                (Some("system"), Some("task_started")) if f["task_type"] == "local_agent" => {
                    let id = f["task_id"].as_str().unwrap_or_default().to_string();
                    match f["parent_task_id"].as_str() {
                        None => {
                            let call = f["tool_use_id"].as_str().unwrap_or_default();
                            self.b.get_or_insert((id, call.to_string()));
                        }
                        Some(p) if Some(p) == b_id.as_deref() => self.c = Some(id),
                        Some(_) => {}
                    }
                }
                (Some("system"), Some("task_notification" | "task_updated"))
                    if self.c.is_some()
                        && f["task_id"].as_str() == self.c.as_deref()
                        && self.c_end.is_none() =>
                {
                    if f["subtype"] == "task_notification" || f["patch"]["status"] == "completed" {
                        self.c_end = Some(t);
                    }
                }
                (Some("user"), _) if parent.is_some() && parent == b_call => {
                    for b in blocks(f) {
                        if b["type"] == "text"
                            && b["text"]
                                .as_str()
                                .is_some_and(|s| s.starts_with("Stop hook feedback"))
                        {
                            self.pick(line, self.blocks.is_empty());
                            self.blocks.push(t);
                        }
                        let id = b["tool_use_id"].as_str().unwrap_or_default();
                        if b["type"] == "tool_result" && self.waits.iter().any(|(w, _)| w == id) {
                            self.pick(line, self.wait_returns.is_empty());
                            let text = b["content"].as_str().unwrap_or_default();
                            self.wait_returns.push((id.into(), t, text.into()));
                        }
                    }
                    if f["message"]["content"]
                        .as_str()
                        .is_some_and(|s| s.starts_with("Stop hook feedback"))
                    {
                        self.pick(line, self.blocks.is_empty());
                        self.blocks.push(t);
                    }
                }
                (Some("assistant"), _) if parent.is_some() && parent == b_call => {
                    for b in blocks(f) {
                        let input = b["input"].to_string();
                        if b["name"] == "Bash" && input.contains("subagents wait") {
                            self.pick(line, self.waits.is_empty());
                            self.waits
                                .push((b["id"].as_str().unwrap_or_default().into(), t));
                        }
                    }
                    if let Some(text) = texts(f).find(|s| s.contains("B-FINAL")) {
                        self.b_final = Some(text);
                    }
                }
                (Some("assistant"), _) if parent.is_none() => {
                    for b in blocks(f) {
                        if b["name"] == "SendMessage" && b["input"].to_string().contains("PING") {
                            self.ping.get_or_insert(t);
                        }
                    }
                    self.main_texts.extend(texts(f));
                }
                (Some("result"), _) if self.main_texts.iter().any(|s| s.contains("MAIN-GOT")) => {
                    self.done = Some(t);
                }
                (Some("user"), _)
                    if f["origin"]["kind"] == "peer"
                        && f["origin"]["senderTaskId"].as_str() == b_id.as_deref()
                        && f["origin"]["body"]
                            .as_str()
                            .is_some_and(|s| s.contains("ACK")) =>
                {
                    self.ack.get_or_insert(t);
                }
                (Some("system"), Some("init")) => {
                    let sid = f["session_id"].as_str().unwrap_or_default();
                    self.session_id.get_or_insert(sid.to_string());
                }
                _ => {}
            }
        }

        fn pick(&mut self, line: &str, first: bool) {
            if first {
                self.picked.push(line.to_string());
            }
        }

        /// When and with what the wait already running at `event` returned.
        fn wake_of(&self, event: Option<f64>) -> Option<(f64, &str)> {
            let event = event?;
            let running = self.waits.iter().filter(|(_, start)| *start < event);
            running
                .flat_map(|(id, _)| {
                    self.wait_returns
                        .iter()
                        .filter(move |(call, ..)| call == id)
                })
                .find(|(_, at, _)| *at >= event)
                .map(|(_, at, text)| (*at, text.as_str()))
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "drives the real claude CLI"]
    async fn nested_subagents_smoke() {
        let exe = std::env::current_exe().unwrap();
        let bin = exe
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("zucchini-spawner");
        assert!(bin.exists(), "run `cargo build` first: {}", bin.display());
        // We may run inside a claude session ourselves; the hook must reach
        // this test's daemon, not the live one.
        for (key, _) in std::env::vars() {
            if key.starts_with("CLAUDE") || key.starts_with("ZUCCHINI") || key == "AI_AGENT" {
                std::env::remove_var(key);
            }
        }
        let _ = std::fs::remove_dir_all(DIR);
        std::fs::create_dir_all(format!("{DIR}/project")).unwrap();
        std::env::set_var("ZUCCHINI_CONTROL_SOCKET", format!("{DIR}/control.sock"));
        std::env::set_var("CLAUDE_CODE_SUBAGENT_MODEL", "sonnet");
        std::env::set_var("CLAUDE_CODE_EFFORT_LEVEL", "low");

        let (tx, mut rx) = tokio::sync::mpsc::channel::<AgentResponse>(1024);
        let mut sup = Supervisor::new(tx);
        crate::control::start(crate::control::test_state(sup.live_sessions())).unwrap();

        let (main_prompt, ping) = prompts();
        *STDOUT_TAP.lock().unwrap() = Some(Vec::new());
        let t0 = Instant::now();
        sup.spawn_agent(req(&main_prompt));
        let mut run = Run::default();
        let mut raw = Vec::new();
        let mut pinged = false;
        while run.done.is_none() {
            assert!(t0.elapsed() < Duration::from_secs(480), "timed out");
            tokio::select! {
                _ = rx.recv() => {}
                () = tokio::time::sleep(Duration::from_millis(200)) => {}
            }
            let new: Vec<_> = STDOUT_TAP
                .lock()
                .unwrap()
                .as_mut()
                .unwrap()
                .drain(..)
                .collect();
            for (at, line) in new {
                let t = at.duration_since(t0).as_secs_f64();
                if let Ok(f) = serde_json::from_str::<Value>(&line) {
                    if f["type"] != "stream_event" {
                        run.observe(t, &line, &f);
                        eprintln!("{t:6.1} {}", line.chars().take(200).collect::<String>());
                    }
                }
                raw.push(line);
            }
            if !pinged && !run.waits.is_empty() {
                pinged = true;
                eprintln!("{:6.1} >>> PING turn", t0.elapsed().as_secs_f64());
                assert!(sup.send_user_turn(&req(ping)));
            }
        }
        *STDOUT_TAP.lock().unwrap() = None;
        sup.shutdown_all().await;
        std::fs::write(format!("{DIR}/stream.jsonl"), raw.join("\n")).unwrap();
        std::fs::write(format!("{DIR}/picked.jsonl"), run.picked.join("\n")).unwrap();

        let ping_wake = run.wake_of(run.ping);
        let c_wake = run.wake_of(run.c_end);
        eprintln!("== timings (s since start)");
        eprintln!(
            "first block {:?}, blocks {:?}",
            run.blocks.first(),
            run.blocks
        );
        eprintln!("waits {:?}", run.waits);
        eprintln!("wait returns {:?}", run.wait_returns);
        eprintln!(
            "ping: SendMessage {:?} -> wait returns {ping_wake:?} -> ACK {:?}",
            run.ping, run.ack
        );
        eprintln!("C ends {:?} -> wait returns {c_wake:?}", run.c_end);
        eprintln!("total {:?}", run.done);
        eprintln!("B: {:?}", run.b_final);
        eprintln!("main: {:?}", run.main_texts);

        assert!(
            !run.blocks.is_empty() && !run.waits.is_empty(),
            "B was held"
        );
        let b_final = run.b_final.as_deref().expect("B's final report");
        assert!(b_final.contains("C-REPORT"), "{b_final}");
        assert!(
            !run.main_texts.iter().any(|s| s.contains("RELAY-SEEN")),
            "main saw only B's report: {:?}",
            run.main_texts
        );
        let got = run
            .main_texts
            .iter()
            .find(|s| s.contains("MAIN-GOT"))
            .unwrap();
        assert!(
            got.contains("B-FINAL") && !got.contains("B-WAITING"),
            "{got}"
        );
        let ping_at = run.ping.expect("main pinged B");
        let ack_at = run.ack.expect("main got B's ACK");
        assert!(ack_at - ping_at < 30.0, "B answered within seconds");
        let woke = |wake: Option<(f64, &str)>, event: f64, line: &str| {
            wake.is_some_and(|(at, text)| at - event < 5.0 && text.contains(line))
        };
        assert!(
            woke(ping_wake, ping_at, MESSAGE_ARRIVED),
            "the PING woke B's running wait: {ping_wake:?}"
        );
        let c_end = run.c_end.expect("C ended");
        assert!(
            woke(c_wake, c_end, &child_ended("C")),
            "C's end woke B's running wait: {c_wake:?}"
        );

        // What main was given, whatever the models wrote.
        let sid = run.session_id.as_deref().expect("main's session id");
        let find = AgentKind::Claude.prune_ops().unwrap().find_session;
        let path = find(&AgentKind::Claude.cli_home().unwrap(), sid).expect("main's transcript");
        let transcript = std::fs::read_to_string(&path).unwrap();
        let (b_id, c_id) = (&run.b.as_ref().unwrap().0, run.c.as_deref().unwrap());
        let b_given = notified_in(&transcript, b_id);
        assert!(
            b_given.iter().any(|t| t == "user"),
            "main was given B's report: {b_given:?} in {}",
            path.display()
        );
        let c_given = notified_in(&transcript, c_id);
        assert!(
            c_given.is_empty(),
            "main was given nothing of C: {c_given:?} in {}",
            path.display()
        );
    }

    /// The types of main-thread transcript entries that carry task `id`'s
    /// notification. Queue operations don't count: claude queues a nested
    /// subagent's notification on the session's queue, and its parent takes it.
    fn notified_in(transcript: &str, id: &str) -> Vec<String> {
        let tag = format!("<task-id>{id}</task-id>");
        transcript
            .lines()
            .filter(|l| l.contains(&tag))
            .filter_map(|l| serde_json::from_str::<Value>(l).ok())
            .filter(|e| e["type"] != "queue-operation" && e["isSidechain"] != true)
            .map(|e| e["type"].as_str().unwrap_or_default().to_string())
            .collect()
    }
}
