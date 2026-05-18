//! In-memory mirror of the chats/projects rows streamed from PowerSync, plus the
//! per-bucket sync cursors. Projects + cursors are persisted to `state.json`; chats
//! are transient because PowerSync only delivers ops after the saved cursor and we'd
//! miss any chat created before the last restart otherwise.
//!
//! Chats don't need persistence: they're re-derived from messages that flow through
//! the sync stream after the cursor, and their Claude session is reconstructed when
//! the agent emits its first `session_id`.
//!
//! `buckets` is a HashMap because PowerSync's sync protocol is per-bucket, not
//! per-connection — each bucket has its own op_id cursor. Today Zucchini has one
//! bucket per user, but the planned hot/cold partial-sync split and any future
//! machine-scoped bucket will each get their own cursor. It is **not** about
//! sharing data across users; this app is strictly one-user-to-many-machines.

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use tracing::warn;
use uuid::Uuid;

#[allow(dead_code)]
pub struct ChatState {
    pub id: String,
    pub user_id: Uuid,
    pub project_id: String,
    pub worktree: bool,
    /// `chats.last_seq` from Postgres: the seq of the most recent message in
    /// this chat. Used to skip replayed historical user messages — see
    /// `handle_message_put` in main.rs.
    pub last_seq: i64,
}

#[derive(Default, Serialize, Deserialize)]
pub struct Mirror {
    #[serde(skip)]
    pub chats: HashMap<String, ChatState>,
    #[serde(default)]
    pub projects: HashMap<String, String>,
    #[serde(default)]
    pub buckets: HashMap<String, String>,
    /// Harvested from the first `machines` PUT in the `by_machine` bucket;
    /// `None` until that lands. Needed to scope `key_<user_id>` lookups.
    #[serde(default)]
    pub user_id: Option<Uuid>,
    /// Re-streamed on every boot, so not persisted. The `set_import_status`
    /// change-detection guard is what keeps re-emissions of the same status
    /// (heartbeat-driven, ~every 10s) from re-firing the importer.
    #[serde(skip)]
    pub claude_history_import_status: Option<String>,
}

impl Mirror {
    pub fn upsert_chat(&mut self, id: String, row_json: &str) {
        let Some(v) = parse_row_json(row_json, "chat", &id) else { return };
        let Some(project_id) = v.get("project_id").and_then(|p| p.as_str()) else {
            warn!(chat_id = %id, "chat row missing project_id");
            return;
        };
        let Some(user_id) = v
            .get("user_id")
            .and_then(|p| p.as_str())
            .and_then(|s| Uuid::parse_str(s).ok())
        else {
            warn!(chat_id = %id, "chat row missing or invalid user_id");
            return;
        };

        let worktree = crate::json_pg_bool(v.get("worktree"));

        let last_seq = v.get("last_seq").and_then(crate::json_to_i64).unwrap_or(0);

        self.chats.insert(
            id.clone(),
            ChatState {
                id,
                user_id,
                project_id: project_id.to_string(),
                worktree,
                last_seq,
            },
        );
    }

    /// Returns true if `user_id` transitioned from `None` to `Some` — i.e. this
    /// is the first machines row we've ever seen for this spawner. The machines
    /// row is re-emitted on every heartbeat, so subsequent calls with the same
    /// `uid` are no-ops. A `uid` change (re-pair under the same machine_id) is
    /// not expected — the spawner token is per-(machine, user) and a re-pair
    /// goes through 410-from-/auth/token + self-uninstall; treat any change as
    /// a no-op rather than overwriting silently.
    pub fn set_user_id(&mut self, uid: Uuid) -> bool {
        if self.user_id.is_some() {
            return false;
        }
        self.user_id = Some(uid);
        true
    }

    /// Returns true if the status actually changed. The machines row is re-emitted
    /// every ~30s on heartbeats, so most calls during a spawner's lifetime are no-ops.
    pub fn set_import_status(&mut self, status: Option<&str>) -> bool {
        if self.claude_history_import_status.as_deref() == status {
            return false;
        }
        self.claude_history_import_status = status.map(str::to_string);
        true
    }

    pub fn upsert_project(&mut self, id: String, row_json: &str) {
        let Some(v) = parse_row_json(row_json, "project", &id) else { return };
        let Some(path) = v.get("path").and_then(|f| f.as_str()) else {
            warn!(project_id = %id, "project row missing path");
            return;
        };
        self.projects.insert(id, path.to_string());
    }

    pub fn remove_chat(&mut self, id: &str) {
        self.chats.remove(id);
    }

    pub fn remove_project(&mut self, id: &str) {
        self.projects.remove(id);
    }

    pub fn load(path: &Path) -> Self {
        match std::fs::read(path) {
            Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_else(|e| {
                warn!(error = %e, "failed to parse state file, starting fresh");
                Mirror::default()
            }),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Mirror::default(),
            Err(e) => {
                warn!(error = %e, "failed to read state file, starting fresh");
                Mirror::default()
            }
        }
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = path.with_extension("json.tmp");
        let bytes = serde_json::to_vec_pretty(self).expect("serialize mirror");
        std::fs::write(&tmp, bytes)?;
        std::fs::rename(&tmp, path)
    }
}

pub(crate) fn parse_row_json(row_json: &str, table: &'static str, id: &str) -> Option<serde_json::Value> {
    match serde_json::from_str(row_json) {
        Ok(v) => Some(v),
        Err(e) => {
            warn!(error = %e, table, %id, "failed to parse row");
            None
        }
    }
}
