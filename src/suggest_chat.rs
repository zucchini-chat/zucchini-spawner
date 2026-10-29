//! `zucchini-spawner suggest-chat`: an agent suggests a new chat. The suggestion
//! is a `chat_suggestion` system row in the calling chat; the app offers it and,
//! if the user accepts, opens its new-chat screen filled with these values.
//! Nothing starts here. Wire format: `tmp/suggest-chat-spec.md`.

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};

use anyhow::{anyhow, bail, Result};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::adapter::AgentKind;
use crate::envelope::EnvelopeAttachment;
use crate::state::Mirror;
use crate::user_prompt::{UserSystemPrompt, MAX_FILES_BYTES};

pub const MAX_PER_TURN: usize = 5;

/// The backend's bounds on `chats.auto_compact_tokens`.
const AUTO_COMPACT_TOKENS: std::ops::RangeInclusive<i64> = 20_000..=2_000_000;

/// The flags as the agent passed them; `None` = omitted. `--attach` paths are
/// absolute (the CLI resolves them against its own directory).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct SuggestChatRequest {
    pub message: String,
    pub project: Option<String>,
    pub agent: Option<String>,
    pub model: Option<String>,
    pub worktree: bool,
    pub title: Option<String>,
    pub system_prompt: Option<String>,
    pub system_prompt_files: Option<String>,
    pub always_on: bool,
    pub background: bool,
    pub auto_compact: Option<String>,
    pub attach: Vec<String>,
}

/// The suggestion as the row carries it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatSuggestion {
    pub message: String,
    pub attachments: Vec<EnvelopeAttachment>,
    pub project_path: String,
    pub agent_kind: String,
    pub model: Option<String>,
    pub worktree: bool,
    pub title: Option<String>,
    pub system_prompt: Option<String>,
    pub system_prompt_files: Option<String>,
    pub always_on: bool,
    pub background: bool,
    pub auto_compact_tokens: Option<i64>,
}

impl ChatSuggestion {
    /// The row body: a system frame, which clients that don't know the
    /// subtype hide. No top-level `text`: clients read such a body as a
    /// message envelope.
    pub fn frame_json(&self) -> String {
        #[derive(Serialize)]
        struct Frame<'a> {
            r#type: &'static str,
            subtype: &'static str,
            #[serde(flatten)]
            suggestion: &'a ChatSuggestion,
        }
        serde_json::to_string(&Frame {
            r#type: "system",
            subtype: "chat_suggestion",
            suggestion: self,
        })
        .expect("serialize chat suggestion")
    }
}

/// Checks `req` against what this machine has, files aside (`check_files`), and
/// returns the chat's user with the suggestion, its attachments still to be
/// uploaded. `installed`: the agent kinds this machine's probes found.
pub fn validate(
    req: &SuggestChatRequest,
    chat_id: &str,
    mirror: &Mirror,
    installed: &[AgentKind],
) -> Result<(Uuid, ChatSuggestion)> {
    let chat = mirror
        .chats
        .get(chat_id)
        .ok_or_else(|| anyhow!("no chat with id {chat_id} (unknown or not yet synced)"))?;
    let message = req.message.trim();
    if message.is_empty() {
        bail!("--message is empty: give the new agent its first message");
    }
    let project_path = match req.project.as_deref() {
        Some(path) => machine_project(path, mirror)?,
        None => mirror
            .projects
            .get(&chat.project_id)
            .cloned()
            .ok_or_else(|| anyhow!("this chat's project isn't synced yet; pass --project"))?,
    };
    let agent_kind = match req.agent.as_deref() {
        Some(agent) => installed_agent(agent, installed)?,
        None => chat.agent_kind,
    };
    let model = match (&req.agent, &req.model) {
        (None, None) => chat.model.clone(),
        _ => non_blank(&req.model),
    };
    let adapter = agent_kind.make_adapter();

    let system_prompt = req
        .system_prompt
        .as_deref()
        .map(crate::user_prompt::agent_text)
        .transpose()
        .map_err(|e| anyhow!("--system-prompt: {e}"))?
        .flatten();
    let system_prompt_files = non_blank(&req.system_prompt_files);
    let files_size = system_prompt_files.as_deref().map_or(0, str::len);
    if files_size > MAX_FILES_BYTES {
        bail!("--system-prompt-files takes {files_size} bytes, the limit is {MAX_FILES_BYTES}");
    }
    if (system_prompt.is_some() || system_prompt_files.is_some())
        && !mirror.takes_user_system_prompt(chat_id, agent_kind)
    {
        bail!("--system-prompt and --system-prompt-files are for claude only, and not in a sandboxed chat");
    }

    if req.always_on {
        if !adapter.is_resident() {
            bail!("--always-on is for claude and codex only");
        }
        if !mirror.is_owner(chat.user_id) {
            bail!("--always-on is for the machine owner only");
        }
    }
    let auto_compact_tokens = req
        .auto_compact
        .as_deref()
        .map(parse_auto_compact)
        .transpose()?;
    if auto_compact_tokens.is_some() && !adapter.is_resident() {
        bail!("--auto-compact is for claude and codex only");
    }

    Ok((
        chat.user_id,
        ChatSuggestion {
            message: message.to_string(),
            attachments: Vec::new(),
            project_path,
            agent_kind: agent_kind.descriptor().wire_name.to_string(),
            model,
            worktree: req.worktree,
            title: non_blank(&req.title),
            system_prompt,
            system_prompt_files,
            always_on: req.always_on,
            background: req.background,
            auto_compact_tokens,
        },
    ))
}

/// What `validate` leaves to the filesystem, which may block (a stalled network
/// mount), so callers run it outside the mirror lock: the system prompt
/// renders against the suggested project and every `--attach` is a file.
pub fn check_files(suggestion: &ChatSuggestion, attach: &[String]) -> Result<()> {
    UserSystemPrompt {
        files: suggestion.system_prompt_files.clone(),
        text: suggestion.system_prompt.clone(),
        undecryptable: false,
    }
    .render(Some(&suggestion.project_path))
    .map_err(|e| anyhow!(e))?;
    for path in attach {
        if !std::fs::metadata(path).is_ok_and(|m| m.is_file()) {
            bail!("--attach {path}: not a file");
        }
    }
    Ok(())
}

fn non_blank(value: &Option<String>) -> Option<String> {
    value
        .as_deref()
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(str::to_string)
}

/// The path of a project on this machine as the app stores it; `path` may
/// differ in trailing or doubled separators.
fn machine_project(path: &str, mirror: &Mirror) -> Result<String> {
    if let Some(known) = mirror
        .projects
        .values()
        .find(|known| Path::new(known.as_str()) == Path::new(path.trim()))
    {
        return Ok(known.clone());
    }
    let mut paths: Vec<&str> = mirror.projects.values().map(String::as_str).collect();
    paths.sort_unstable();
    paths.dedup();
    if paths.is_empty() {
        bail!("--project {path}: this machine has no projects");
    }
    bail!(
        "--project {path}: not a project on this machine; its projects: {}",
        paths.join(", ")
    )
}

fn installed_agent(agent: &str, installed: &[AgentKind]) -> Result<AgentKind> {
    if let Some(kind) = AgentKind::parse(agent.trim()).filter(|k| installed.contains(k)) {
        return Ok(kind);
    }
    let names: Vec<&str> = installed.iter().map(|k| k.descriptor().wire_name).collect();
    bail!(
        "--agent {agent}: not installed on this machine; installed: {}",
        if names.is_empty() {
            "none".to_string()
        } else {
            names.join(", ")
        }
    )
}

/// `200000`, `200k`, `1.5M`.
fn parse_auto_compact(value: &str) -> Result<i64> {
    let value = value.trim();
    let (number, unit) = match value.char_indices().last() {
        Some((i, 'k' | 'K')) => (&value[..i], 1e3),
        Some((i, 'm' | 'M')) => (&value[..i], 1e6),
        _ => (value, 1.0),
    };
    number
        .parse::<f64>()
        .ok()
        .map(|n| (n * unit).round())
        .filter(|n| n.is_finite())
        .map(|n| n as i64)
        .filter(|n| AUTO_COMPACT_TOKENS.contains(n))
        .ok_or_else(|| {
            anyhow!("--auto-compact {value}: expected a token count from 20k to 2M, e.g. 200k")
        })
}

/// Suggestions per chat since its last `result` row.
fn turn_counts() -> MutexGuard<'static, HashMap<String, Turn>> {
    static COUNTS: OnceLock<Mutex<HashMap<String, Turn>>> = OnceLock::new();
    COUNTS
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

struct Turn {
    /// Unique across turns, so a slot taken before the chat's last `result`
    /// row never touches the turn after it.
    id: u64,
    slots: usize,
}

/// One suggestion counted against its chat's turn. Dropped without `commit`
/// (the suggestion failed), it gives the slot back.
#[must_use = "dropping the slot gives it back"]
pub struct TurnSlot {
    chat_id: String,
    turn: u64,
    committed: bool,
}

impl TurnSlot {
    /// Runs `queue` (the row's enqueue) unless the turn has ended meanwhile.
    /// Under the same lock as `agent_row_written`, which runs before a
    /// `result` row is enqueued, so the row can't land after it.
    pub fn commit(mut self, queue: impl FnOnce()) -> Result<()> {
        let turns = turn_counts();
        if turns.get(&self.chat_id).is_none_or(|t| t.id != self.turn) {
            bail!("the turn ended before the suggestion was ready; nothing was suggested");
        }
        queue();
        self.committed = true;
        Ok(())
    }
}

impl Drop for TurnSlot {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        let mut turns = turn_counts();
        if let Some(turn) = turns.get_mut(&self.chat_id).filter(|t| t.id == self.turn) {
            turn.slots -= 1;
            if turn.slots == 0 {
                turns.remove(&self.chat_id);
            }
        }
    }
}

/// Holds one of the chat's slots for the current turn while a suggestion is
/// under way.
pub fn take_turn_slot(chat_id: &str) -> Result<TurnSlot> {
    static NEXT_TURN: AtomicU64 = AtomicU64::new(0);
    let mut turns = turn_counts();
    let turn = turns.entry(chat_id.to_string()).or_insert_with(|| Turn {
        id: NEXT_TURN.fetch_add(1, Ordering::Relaxed),
        slots: 0,
    });
    if turn.slots >= MAX_PER_TURN {
        bail!("this turn already suggested {MAX_PER_TURN} chats, the most one turn may");
    }
    turn.slots += 1;
    Ok(TurnSlot {
        chat_id: chat_id.to_string(),
        turn: turn.id,
        committed: false,
    })
}

/// Every agent row of a chat passes here: a `result` row ends its turn.
pub fn agent_row_written(chat_id: &str, content: &str) {
    #[derive(Deserialize)]
    struct Frame {
        r#type: Option<String>,
    }
    let mut counts = turn_counts();
    if counts.contains_key(chat_id)
        && serde_json::from_str::<Frame>(content)
            .is_ok_and(|f| f.r#type.as_deref() == Some("result"))
    {
        counts.remove(chat_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::KeyStore;

    struct Fixture {
        mirror: Mirror,
        keys: KeyStore,
        owner: Uuid,
        member: Uuid,
        dir: std::path::PathBuf,
    }

    impl Fixture {
        fn new() -> Self {
            let (owner, member) = (Uuid::now_v7(), Uuid::now_v7());
            let keys = KeyStore::with_keys([(owner, [1u8; 32]), (member, [2u8; 32])]);
            let mut mirror = Mirror::default();
            mirror.set_user_id(owner);
            let dir = std::env::temp_dir().join(format!("zucchini-suggest-{}", Uuid::now_v7()));
            std::fs::create_dir_all(&dir).unwrap();
            Self {
                mirror,
                keys,
                owner,
                member,
                dir,
            }
        }

        fn project(&mut self, path: &str) -> String {
            let id = Uuid::now_v7().to_string();
            let row = serde_json::json!({ "path": path });
            self.mirror.upsert_project(id.clone(), &row.to_string());
            id
        }

        fn chat(
            &mut self,
            user: Uuid,
            project_id: &str,
            kind: &str,
            model: Option<&str>,
        ) -> String {
            let chat_id = Uuid::now_v7().to_string();
            let row = serde_json::json!({
                "id": chat_id,
                "project_id": project_id,
                "user_id": user.to_string(),
                "last_seq": 0,
                "agent_kind": kind,
                "model": model,
                "worktree": false,
            });
            self.mirror
                .upsert_chat(chat_id.clone(), &row.to_string(), &self.keys);
            chat_id
        }

        fn file(&self, name: &str, content: &str) -> String {
            let path = self.dir.join(name);
            std::fs::write(&path, content).unwrap();
            path.to_string_lossy().into_owned()
        }

        fn validate(&self, chat_id: &str, req: SuggestChatRequest) -> Result<ChatSuggestion> {
            let installed = [AgentKind::Claude, AgentKind::Codex];
            let (_, s) = validate(&req, chat_id, &self.mirror, &installed)?;
            check_files(&s, &req.attach)?;
            Ok(s)
        }

        fn error(&self, chat_id: &str, req: SuggestChatRequest) -> String {
            format!("{:#}", self.validate(chat_id, req).unwrap_err())
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    fn message(text: &str) -> SuggestChatRequest {
        SuggestChatRequest {
            message: text.to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn defaults_come_from_the_calling_chat() {
        let mut f = Fixture::new();
        let project = f.project("/work/app");
        let chat = f.chat(f.owner, &project, "claude", Some("opus"));

        let s = f
            .validate(&chat, message("  fix the login bug \n"))
            .unwrap();
        assert_eq!(
            s,
            ChatSuggestion {
                message: "fix the login bug".to_string(),
                attachments: Vec::new(),
                project_path: "/work/app".to_string(),
                agent_kind: "claude".to_string(),
                model: Some("opus".to_string()),
                worktree: false,
                title: None,
                system_prompt: None,
                system_prompt_files: None,
                always_on: false,
                background: false,
                auto_compact_tokens: None,
            }
        );
        let other_model = SuggestChatRequest {
            model: Some("sonnet".to_string()),
            ..message("m")
        };
        assert_eq!(
            f.validate(&chat, other_model).unwrap().model.as_deref(),
            Some("sonnet")
        );
        let other_agent = SuggestChatRequest {
            agent: Some("codex".to_string()),
            ..message("m")
        };
        let s = f.validate(&chat, other_agent).unwrap();
        assert_eq!((s.agent_kind.as_str(), s.model), ("codex", None));
    }

    #[test]
    fn a_project_is_any_path_on_the_machine_as_the_app_stores_it() {
        let mut f = Fixture::new();
        let project = f.project("/work/app");
        f.project("/work/site");
        let chat = f.chat(f.member, &project, "claude", None);
        let req = |path: &str| SuggestChatRequest {
            project: Some(path.to_string()),
            ..message("m")
        };

        assert_eq!(
            f.validate(&chat, req("/work/site/")).unwrap().project_path,
            "/work/site"
        );
        assert_eq!(
            f.error(&chat, req("/work/other")),
            "--project /work/other: not a project on this machine; its projects: /work/app, /work/site"
        );
    }

    #[test]
    fn bad_values_fail_with_a_reason() {
        let mut f = Fixture::new();
        let project = f.project("/work/app");
        let claude = f.chat(f.owner, &project, "claude", None);
        let codex = f.chat(f.owner, &project, "codex", None);
        let cursor = f.chat(f.owner, &project, "cursor", None);
        f.mirror.upsert_member("row".into(), f.member, true);
        let sandboxed = f.chat(f.member, &project, "claude", None);
        let rules = f.file("rules.md", "be terse");

        assert!(f
            .error(&claude, message(" \n"))
            .starts_with("--message is empty"));
        assert!(f
            .error("no-such-chat", message("m"))
            .starts_with("no chat with id"));
        assert_eq!(
            f.error(
                &claude,
                SuggestChatRequest {
                    agent: Some("cursor".to_string()),
                    ..message("m")
                }
            ),
            "--agent cursor: not installed on this machine; installed: claude, codex"
        );

        let prompt = |text: &str, files: &str| SuggestChatRequest {
            system_prompt: Some(text.to_string()),
            system_prompt_files: Some(files.to_string()),
            ..message("m")
        };
        assert!(f.validate(&claude, prompt("be terse", &rules)).is_ok());
        for chat in [&codex, &sandboxed] {
            assert!(f
                .error(chat, prompt("be terse", ""))
                .contains("claude only"));
        }
        let over = "x".repeat(crate::user_prompt::MAX_TEXT_BYTES + 1);
        assert!(f
            .error(&claude, prompt(&over, ""))
            .starts_with("--system-prompt: the text takes"));
        assert!(f
            .error(&claude, prompt("", "missing.md"))
            .starts_with("system prompt file `missing.md`"));
        let files = "a.md,".repeat(MAX_FILES_BYTES / 5 + 1);
        assert!(f
            .error(&claude, prompt("", &files))
            .starts_with("--system-prompt-files takes 4005 bytes"));

        let always_on = SuggestChatRequest {
            always_on: true,
            ..message("m")
        };
        assert!(f.validate(&claude, always_on.clone()).is_ok());
        assert!(f
            .error(&cursor, always_on.clone())
            .contains("claude and codex only"));
        assert!(f
            .error(&sandboxed, always_on)
            .contains("machine owner only"));

        let compact = |value: &str| SuggestChatRequest {
            auto_compact: Some(value.to_string()),
            ..message("m")
        };
        assert_eq!(
            f.validate(&codex, compact("200k"))
                .unwrap()
                .auto_compact_tokens,
            Some(200_000)
        );
        for bad in ["10k", "3M", "lots", ""] {
            assert!(
                f.error(&claude, compact(bad)).contains("from 20k to 2M"),
                "{bad:?}"
            );
        }
        assert!(f
            .error(&cursor, compact("200k"))
            .contains("claude and codex only"));

        let attach = SuggestChatRequest {
            attach: vec![rules, f.dir.to_string_lossy().into_owned()],
            ..message("m")
        };
        assert!(f.error(&claude, attach).ends_with(": not a file"));
    }

    #[test]
    fn auto_compact_takes_suffixes() {
        assert_eq!(parse_auto_compact("20000").unwrap(), 20_000);
        assert_eq!(parse_auto_compact("50k").unwrap(), 50_000);
        assert_eq!(parse_auto_compact(" 1.5M ").unwrap(), 1_500_000);
        assert_eq!(parse_auto_compact("2m").unwrap(), 2_000_000);
        assert!(parse_auto_compact("19999").is_err());
        assert!(parse_auto_compact("k").is_err());
    }

    #[test]
    fn the_row_is_a_sealed_agent_system_frame() {
        let mut f = Fixture::new();
        let project = f.project("/work/app");
        let chat = f.chat(f.owner, &project, "claude", None);
        let mut s = f
            .validate(
                &chat,
                SuggestChatRequest {
                    title: Some("Login bug".to_string()),
                    always_on: true,
                    auto_compact: Some("200k".to_string()),
                    ..message("fix it")
                },
            )
            .unwrap();
        let blob_key = Uuid::now_v7();
        s.attachments.push(EnvelopeAttachment {
            blob_key,
            size: 12,
            name: "log.txt".to_string(),
        });

        let row = crate::writer::WriteEvent::agent_line(chat.clone(), f.owner, s.frame_json());
        let op = crate::writer::encode_event(&row, &f.keys).expect("op");
        assert_eq!((op.op, op.table), ("PUT", "messages"));
        let data = op.data.expect("data");
        assert_eq!(data["sender"], "agent");
        let key = f.keys.get(&f.owner).unwrap();
        let body = crate::crypto::decrypt_field_b64(&key, data["body"].as_str().unwrap()).unwrap();
        let frame: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(
            frame,
            serde_json::json!({
                "type": "system",
                "subtype": "chat_suggestion",
                "message": "fix it",
                "attachments": [{ "blob_key": blob_key.to_string(), "size": 12, "name": "log.txt" }],
                "project_path": "/work/app",
                "agent_kind": "claude",
                "model": null,
                "worktree": false,
                "title": "Login bug",
                "system_prompt": null,
                "system_prompt_files": null,
                "always_on": true,
                "background": false,
                "auto_compact_tokens": 200000,
            })
        );
    }

    fn suggest(chat: &str) {
        take_turn_slot(chat).unwrap().commit(|| {}).unwrap();
    }

    #[test]
    fn a_result_row_ends_the_turn_and_its_cap() {
        let chat = Uuid::now_v7().to_string();
        for _ in 0..MAX_PER_TURN {
            suggest(&chat);
        }
        assert!(take_turn_slot(&chat).is_err());
        agent_row_written(&chat, r#"{"type":"assistant","message":{}}"#);
        agent_row_written(&chat, "Error: not json");
        assert!(take_turn_slot(&chat).is_err());
        agent_row_written(&chat, crate::adapter::INTERRUPTED_RESULT);
        assert!(take_turn_slot(&chat).is_ok());
    }

    #[test]
    fn only_a_committed_suggestion_counts() {
        let chat = Uuid::now_v7().to_string();
        for _ in 0..MAX_PER_TURN {
            drop(take_turn_slot(&chat).unwrap());
        }
        let under_way: Vec<TurnSlot> = (0..MAX_PER_TURN)
            .map(|_| take_turn_slot(&chat).unwrap())
            .collect();
        assert!(take_turn_slot(&chat).is_err());
        drop(under_way);
        assert!(!turn_counts().contains_key(&chat));
        suggest(&chat);
        let under_way: Vec<TurnSlot> = (1..MAX_PER_TURN)
            .map(|_| take_turn_slot(&chat).unwrap())
            .collect();
        for slot in under_way {
            slot.commit(|| {}).unwrap();
        }
        assert!(take_turn_slot(&chat).is_err());
    }

    #[test]
    fn a_failed_suggestion_gives_its_slot_back_to_its_own_turn_only() {
        let chat = Uuid::now_v7().to_string();
        let first_turn: Vec<TurnSlot> = (0..MAX_PER_TURN)
            .map(|_| take_turn_slot(&chat).unwrap())
            .collect();
        agent_row_written(&chat, crate::adapter::INTERRUPTED_RESULT);
        for _ in 0..MAX_PER_TURN {
            suggest(&chat);
        }
        drop(first_turn);
        assert!(take_turn_slot(&chat).is_err());
    }

    #[test]
    fn a_suggestion_its_turn_outlived_is_not_queued() {
        let chat = Uuid::now_v7().to_string();
        let slot = take_turn_slot(&chat).unwrap();
        agent_row_written(&chat, crate::adapter::INTERRUPTED_RESULT);
        let mut queued = false;
        let err = slot.commit(|| queued = true).unwrap_err();
        assert!(!queued);
        assert!(err.to_string().starts_with("the turn ended"));
        for _ in 0..MAX_PER_TURN {
            suggest(&chat);
        }
        assert!(take_turn_slot(&chat).is_err());
    }
}
