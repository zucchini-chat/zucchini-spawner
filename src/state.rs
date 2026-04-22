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

#[allow(dead_code)]
pub struct ChatState {
    pub id: String,
    pub project_id: String,
    pub last_processed_seq: i64,
}

#[derive(Default, Serialize, Deserialize)]
pub struct Mirror {
    #[serde(skip)]
    pub chats: HashMap<String, ChatState>,
    #[serde(default)]
    pub projects: HashMap<String, String>,
    #[serde(default)]
    pub buckets: HashMap<String, String>,
}

impl Mirror {
    pub fn upsert_chat(&mut self, id: String, row_json: &str) {
        let v: serde_json::Value = match serde_json::from_str(row_json) {
            Ok(v) => v,
            Err(e) => {
                warn!(error = %e, chat_id = %id, "failed to parse chat row");
                return;
            }
        };
        let Some(project_id) = v.get("project_id").and_then(|p| p.as_str()) else {
            warn!(chat_id = %id, "chat row missing project_id");
            return;
        };

        let last_processed_seq = self
            .chats
            .get(&id)
            .map(|c| c.last_processed_seq)
            .unwrap_or(0);

        self.chats.insert(
            id.clone(),
            ChatState {
                id,
                project_id: project_id.to_string(),
                last_processed_seq,
            },
        );
    }

    pub fn upsert_project(&mut self, id: String, row_json: &str) {
        let v: serde_json::Value = match serde_json::from_str(row_json) {
            Ok(v) => v,
            Err(e) => {
                warn!(error = %e, project_id = %id, "failed to parse project row");
                return;
            }
        };
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
