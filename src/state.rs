//! In-memory mirror of the chats/projects rows streamed from PowerSync, plus the
//! per-bucket sync cursors. Projects + cursors are persisted to `state.json`; chats
//! are transient because PowerSync only delivers ops after the saved cursor and we'd
//! miss any chat created before the last restart otherwise.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tracing::{error, warn};
use uuid::Uuid;

use crate::adapter::AgentKind;

/// Shared handle to `Mirror`. The control socket task (see `control.rs`) needs
/// to read chat → user_id projections at the same time the main loop is
/// mutating `Mirror` from `handle_sync_event` / `handle_agent_response`. Using
/// `tokio::sync::RwLock` is deliberate: `handle_sync_event` `.await`s under
/// the write guard (decryption, R2 downloads, writer-channel sends), and a
/// `std::sync::RwLock` would deadlock + trip `clippy::await_holding_lock`.
pub type SharedMirror = Arc<tokio::sync::RwLock<Mirror>>;

pub struct ChatState {
    pub user_id: Uuid,
    pub project_id: String,
    pub worktree: bool,
    /// `chats.last_seq` from Postgres: the seq of the most recent message in
    /// this chat. Used to skip replayed historical user messages — see
    /// `handle_message_put` in main.rs.
    pub last_seq: i64,
    /// `chats.agent_session_id` from Postgres. `None` on the first turn of a
    /// freshly created chat; populated on the first stdout frame from the
    /// agent (harvested from its `system/init` frame, via `set_agent_session_id`)
    /// or backfilled from `id::text` for pre-migration rows. When `Some(s)`, the
    /// spawner resumes via `--resume s`; when `None`, the agent generates a fresh
    /// session id. `upsert_chat` stores the incoming value verbatim (including a
    /// stale NULL); the locally-harvested id is preserved across such a re-stream
    /// by the checkpoint-window restore in main.rs (`SyncEvent::CheckpointComplete`).
    pub agent_session_id: Option<String>,
    /// `chats.agent_kind` from Postgres. Defaults to `AgentKind::Claude`
    /// when the column is absent (chats synced before the column was
    /// added) — forward-compat fallback matches the Postgres DEFAULT.
    pub agent_kind: AgentKind,
    /// `chats.model` from Postgres (migration 0035). Verbatim `--model <X>`
    /// pass-through; the empty-string / NULL → `None` filter lives at the
    /// `SpawnRequest` construction site in `main.rs` (so adapter logic can
    /// stay `if let Some(m) = ctx.model { ... }`). Column absent → `None`,
    /// same as NULL.
    pub model: Option<String>,
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
    /// CSV of `AgentKind` wire names the user selected via the iOS import
    /// modal's checkboxes (e.g. "claude" / "cursor" / "codex" /
    /// "claude,cursor,codex").
    /// `None` means the column is absent or NULL — older iOS without the
    /// checkbox UI, or an older backend without migration 0034. The
    /// dispatcher in main.rs falls back to `AgentKind::ALL` in that case so
    /// the historic "all supported kinds" behavior is preserved.
    #[serde(skip)]
    pub claude_history_import_kinds: Option<String>,
    #[serde(default)]
    pub spawner_pubkey: Option<String>,
    /// Persisted so `member_is_sandboxed` doesn't fail open after a restart
    /// (PowerSync resumes from the saved cursor and won't re-emit historical
    /// `machine_users` rows).
    #[serde(default)]
    members: HashMap<Uuid, MemberInfo>,
    /// One-shot latch for the install-time `machine_users.agents` seed
    /// (`seed_initial_agents_if_pending` in main.rs). `serde(default)` FALSE
    /// for a pre-upgrade state.json is deliberate: the cohort stranded with a
    /// NULL roster gets one healing attempt post-upgrade; non-NULL rosters
    /// drain it on the backend's seed-only guard (`WHERE agents IS NULL`).
    #[serde(default)]
    pub initial_agents_seed_done: bool,
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
    /// Mirror of `machine_users.timezone` (migration 0040) — IANA id of the
    /// member's most-recently-active device, or `None` (NULL / older client).
    /// Consulted at spawn to inject the current local time (`current_time_in_tz_line`)
    /// and zone naive `schedule-message --at` (`control::normalize_deliver_at`).
    ///
    /// PERSISTED. The `by_machine` bucket resumes incrementally
    /// from the saved cursor, so an unchanged `machine_users` row is never
    /// re-streamed after a restart — and nothing bumps it per turn (chats dodge
    /// this via the per-message `last_seq` UPDATE). So a once-set tz would
    /// otherwise be lost on the first restart and stay `None` forever, breaking the
    /// prompt time line + naive `--at`. Accept the staleness window (a peer's tz
    /// change while offline self-corrects on the next row change) to survive restarts.
    #[serde(default)]
    timezone: Option<String>,
}

fn default_true() -> bool {
    true
}

impl Mirror {
    pub fn upsert_chat(&mut self, id: String, row_json: &str) {
        let Some(v) = parse_row_json(row_json, "chat", &id) else {
            return;
        };
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

        let incoming_agent_session_id = v
            .get("agent_session_id")
            .and_then(|x| x.as_str())
            .map(str::to_string);

        // Strict parse when the column is present. Pre-migration rows
        // synced before the `agent_kind` column existed have no such
        // field — fall through to `AgentKind::Claude` (matches the
        // Postgres DEFAULT). A present-but-unrecognized value (e.g. `"codex"` from
        // the backend whitelist before its adapter ships) escalates to an
        // `error!` so it lands in Sentry — without an adapter the chat is
        // dead (subsequent message handlers checking `mirror.chats.get(...)`
        // log-and-bail on the missing chat row, leaving the iOS UI stuck
        // showing `agent_running=true` with no reply). Visibility here is
        // the only signal that the backend whitelist has drifted ahead of
        // the spawner's adapter set; the real fix is either tightening the
        // whitelist in `backend/src/writes.rs::validate_agent_kind` or
        // shipping the missing adapter.
        let agent_kind = match v.get("agent_kind").and_then(|x| x.as_str()) {
            None => AgentKind::Claude,
            Some(raw) => match AgentKind::parse(raw) {
                Some(k) => k,
                None => {
                    error!(
                        chat_id = %id,
                        agent_kind = %raw,
                        "unsupported agent_kind: backend whitelist accepted a value this spawner has no adapter for; chat will hang with no reply until a supported PUT lands or this spawner upgrades"
                    );
                    return;
                }
            },
        };

        // Store the incoming `agent_session_id` verbatim (including None). The
        // harvested-id-survives-a-stale-NULL race is now handled at apply time
        // by the checkpoint-window restore in main.rs
        // (`SyncEvent::CheckpointComplete`): it captures the local id before
        // upsert and restores it when the applied row landed NULL.

        // `chats.model` is migration 0035. Pre-migration rows omit the column
        // entirely; an empty string also lands as `None` so the
        // `SpawnRequest` construction site has a single shape to handle
        // (NULL and "" are indistinguishable on the wire for our purposes).
        let model = v
            .get("model")
            .and_then(|x| x.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_string);

        self.chats.insert(
            id,
            ChatState {
                user_id,
                project_id: project_id.to_string(),
                worktree,
                last_seq,
                agent_session_id: incoming_agent_session_id,
                agent_kind,
                model,
            },
        );
    }

    /// Stash the harvested session id locally so a fast-followup user message
    /// in the same chat doesn't race the writer/PowerSync round-trip and
    /// re-spawn the agent without `--resume`. Idempotent on subsequent harvests
    /// (keeps the first id).
    pub fn set_agent_session_id(&mut self, chat_id: &str, session_id: String) {
        if let Some(c) = self.chats.get_mut(chat_id) {
            if c.agent_session_id.is_none() {
                c.agent_session_id = Some(session_id);
            }
        }
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

    /// The owner's own `machine_users` row id once it has streamed in (`None`
    /// until its `row_id` is non-empty). Gates `seed_initial_agents_if_pending`.
    pub fn owner_row(&self) -> Option<String> {
        let uid = self.user_id?;
        let m = self.members.get(&uid)?;
        if m.row_id.is_empty() {
            return None;
        }
        Some(m.row_id.clone())
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

    /// Stash the latest `claude_history_import_kinds` CSV streamed from the
    /// `machines` row. No change-detection — the dispatcher reads this
    /// directly when `ImportRequested` fires; mid-import column changes don't
    /// retro-affect the in-flight run.
    pub fn set_import_kinds(&mut self, kinds: Option<&str>) {
        self.claude_history_import_kinds = kinds.map(str::to_string);
    }

    /// Parse `claude_history_import_kinds` into the closed adapter set the
    /// dispatcher iterates. `None` (column absent / NULL — older iOS without
    /// the checkbox UI) falls back to `AgentKind::ALL` so the historic
    /// "all supported kinds" behavior is preserved. Unknown / unparseable
    /// entries are dropped with a warn so a forwards-compat backend whitelist drift
    /// can't break the importer; if every entry is dropped we also fall
    /// back to `ALL` so the user still gets something.
    pub fn parsed_import_kinds(&self) -> Vec<AgentKind> {
        let Some(csv) = self.claude_history_import_kinds.as_deref() else {
            return AgentKind::ALL.to_vec();
        };
        let mut out: Vec<AgentKind> = Vec::new();
        for part in csv.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            match AgentKind::parse(part) {
                Some(k) => {
                    if !out.contains(&k) {
                        out.push(k);
                    }
                }
                None => {
                    warn!(kind = %part, "unknown agent kind in claude_history_import_kinds, dropping");
                }
            }
        }
        if out.is_empty() {
            warn!(
                csv,
                "claude_history_import_kinds had no known kinds; falling back to all"
            );
            return AgentKind::ALL.to_vec();
        }
        out
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
            MemberInfo {
                row_id,
                is_sandboxed,
                last_sealed_blob: None,
                timezone: None,
            },
        );
        true
    }

    /// Stash the raw `machine_users.timezone` IANA id (migration 0040). `None`
    /// clears (NULL). No-op if the member entry doesn't exist yet (row_id lands
    /// first via `upsert_member`).
    pub fn set_member_timezone(&mut self, user_id: &Uuid, timezone: Option<String>) {
        if let Some(info) = self.members.get_mut(user_id) {
            info.timezone = timezone;
        }
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

    /// Cached `machine_users.timezone` IANA id (migration 0040). `None` = no
    /// member entry OR NULL column; both mean "no tz hint". Mirrors
    /// `member_is_sandboxed`.
    pub fn member_timezone(&self, user_id: &Uuid) -> Option<&str> {
        self.members
            .get(user_id)
            .and_then(|m| m.timezone.as_deref())
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
        let Some(v) = parse_row_json(row_json, "project", &id) else {
            return;
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
        let bytes = serde_json::to_vec_pretty(self).expect("serialize mirror");
        crate::atomic::atomic_write_private(path, &bytes)
    }
}

pub(crate) fn parse_row_json(
    row_json: &str,
    table: &'static str,
    id: &str,
) -> Option<serde_json::Value> {
    match serde_json::from_str(row_json) {
        Ok(v) => Some(v),
        Err(e) => {
            warn!(error = %e, table, %id, "failed to parse row");
            None
        }
    }
}
