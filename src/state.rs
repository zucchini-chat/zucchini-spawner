//! In-memory mirror of the chats/projects rows streamed from PowerSync, plus the
//! per-bucket sync cursors. Projects + cursors are persisted to `state.json`; chats
//! are transient because PowerSync only delivers ops after the saved cursor and we'd
//! miss any chat created before the last restart otherwise.

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use tracing::warn;
use uuid::Uuid;

pub struct ChatState {
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
    #[serde(default)]
    pub spawner_pubkey: Option<String>,
    /// Persisted so `member_is_sandboxed` doesn't fail open after a restart
    /// (PowerSync resumes from the saved cursor and won't re-emit historical
    /// `machine_users` rows).
    #[serde(default)]
    members: HashMap<Uuid, MemberInfo>,
}

#[derive(Serialize, Deserialize)]
struct MemberInfo {
    /// Empty default keeps the rest of `members` intact on a partial-corruption
    /// parse; an empty row_id can't match a real `machine_users.id` UUID.
    #[serde(default)]
    row_id: String,
    /// Fail-closed default; only fires on serde-default during state.json parse.
    #[serde(default = "default_true")]
    is_sandboxed: bool,
    /// Last sealed_blob base64 we successfully unsealed for this user. Used to
    /// short-circuit re-emissions of the same `machine_users` row on heartbeat
    /// fan-out (the X25519 unseal + key file write are otherwise re-run every
    /// tick). `None` until the first sealed_blob lands.
    #[serde(default)]
    last_sealed_blob: Option<String>,
}

fn default_true() -> bool {
    true
}

impl Mirror {
    pub fn upsert_chat(&mut self, id: String, row_json: &str) {
        let Some(v) = parse_row_json(row_json, "chat", &id) else { return };
        let Some(project_id) = v.get("project_id").and_then(|p| p.as_str()) else {
            warn!(chat_id = %id, "chat row missing project_id");
            return;
        };
        let Some(user_id) = crate::parse_uuid_field(&v, "user_id") else {
            warn!(chat_id = %id, "chat row missing or invalid user_id");
            return;
        };

        let worktree = crate::json_pg_bool(v.get("worktree"));

        let last_seq = v.get("last_seq").and_then(crate::json_to_i64).unwrap_or(0);

        self.chats.insert(
            id,
            ChatState {
                user_id,
                project_id: project_id.to_string(),
                worktree,
                last_seq,
            },
        );
    }

    /// Returns true iff the stored `user_id` materially changed (None→Some, or
    /// Some(a)→Some(b) where a != b). The machines row is re-emitted on every
    /// heartbeat, so same-uid calls are no-ops. A uid CHANGE is unexpected
    /// (the spawner token is per-(machine, user) and a re-pair normally goes
    /// through 410-from-/auth/token + self-uninstall), but legacy installs
    /// that upgraded the spawner in-place without re-running uninstall.sh may
    /// carry a stale state.json from the prior owner. Refusing the new uid
    /// would leave `mirror.user_id` stuck on the old owner and mis-classify
    /// the actual owner as a member via `is_owner`. Log and accept instead.
    pub fn set_user_id(&mut self, uid: Uuid) -> bool {
        match self.user_id {
            Some(existing) if existing == uid => false,
            Some(existing) => {
                warn!(
                    %existing,
                    new = %uid,
                    "mirror.user_id changed; accepting new owner (likely stale state.json from prior pairing)"
                );
                // Drop prior owner's members so downstream `is_owner` /
                // `has_member` decisions don't classify them under the new uid.
                self.members.clear();
                self.user_id = Some(uid);
                true
            }
            None => {
                self.user_id = Some(uid);
                true
            }
        }
    }

    pub fn is_owner(&self, uid: Uuid) -> bool {
        self.user_id == Some(uid)
    }

    pub fn set_spawner_pubkey(&mut self, pubkey: Option<&str>) -> bool {
        if self.spawner_pubkey.as_deref() == pubkey {
            return false;
        }
        self.spawner_pubkey = pubkey.map(str::to_string);
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

    /// Membership rows fan out on every heartbeat tick, so most calls are
    /// no-ops on an existing entry. Returns true iff the entry was inserted
    /// or its row_id/is_sandboxed materially changed.
    pub fn upsert_member(&mut self, row_id: String, user_id: Uuid, is_sandboxed: bool) -> bool {
        if row_id.is_empty() {
            return false;
        }
        if let Some(existing) = self.members.get_mut(&user_id) {
            if existing.row_id == row_id && existing.is_sandboxed == is_sandboxed {
                return false;
            }
            existing.row_id = row_id;
            existing.is_sandboxed = is_sandboxed;
            return true;
        }
        self.members.insert(
            user_id,
            MemberInfo { row_id, is_sandboxed, last_sealed_blob: None },
        );
        true
    }

    /// True if we've already unsealed this exact sealed_blob for this user.
    /// Used to short-circuit the X25519 unseal + key-file write on heartbeat
    /// re-emissions. False on a missing entry is safe: the caller will then
    /// run the unseal path and `record_sealed_blob` will no-op if upsert
    /// somehow hasn't happened — at worst we re-unseal next tick.
    pub fn member_sealed_blob_matches(&self, user_id: &Uuid, sealed_b64: &str) -> bool {
        self.members
            .get(user_id)
            .and_then(|m| m.last_sealed_blob.as_deref())
            == Some(sealed_b64)
    }

    /// Cache the sealed_blob after a successful unseal+persist so the next
    /// heartbeat short-circuits. No-op if the member entry is missing.
    pub fn record_sealed_blob(&mut self, user_id: &Uuid, sealed_b64: &str) {
        if let Some(info) = self.members.get_mut(user_id) {
            info.last_sealed_blob = Some(sealed_b64.to_string());
        }
    }

    pub fn remove_member(&mut self, row_id: &str) -> Option<Uuid> {
        let user_id = self.user_for_row_id(row_id)?;
        self.members.remove(&user_id);
        Some(user_id)
    }

    /// Look up the user_id a `machine_users` row maps to without mutating.
    /// Callers that need to special-case ownership (e.g. refuse REMOVE on the
    /// owner's row) use this to peek before `remove_member`.
    pub fn user_for_row_id(&self, row_id: &str) -> Option<Uuid> {
        if row_id.is_empty() {
            return None;
        }
        self.members
            .iter()
            .find(|(_, m)| !m.row_id.is_empty() && m.row_id == row_id)
            .map(|(uid, _)| *uid)
    }

    pub fn member_is_sandboxed(&self, user_id: &Uuid) -> Option<bool> {
        self.members.get(user_id).map(|m| m.is_sandboxed)
    }

    pub fn has_member(&self, user_id: &Uuid) -> bool {
        self.members.contains_key(user_id)
    }

    /// Drop the cached sealed_blob (e.g. after a server-side soft-revoke that
    /// patches `sealed_blob` to NULL/empty) so the next non-empty PUT is
    /// treated as fresh and re-unsealed. Keeps the member row for telemetry.
    /// Returns true iff a cached blob was actually cleared.
    pub fn clear_sealed_blob(&mut self, user_id: &Uuid) -> bool {
        match self.members.get_mut(user_id) {
            Some(info) if info.last_sealed_blob.is_some() => {
                info.last_sealed_blob = None;
                true
            }
            _ => false,
        }
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
        let bytes = serde_json::to_vec_pretty(self).expect("serialize mirror");
        crate::atomic::atomic_write_private(path, &bytes)
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
