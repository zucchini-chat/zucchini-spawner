//! In-memory mirror of the chats/projects rows streamed from PowerSync, plus the
//! per-bucket sync cursors. Projects + cursors are persisted to `state.json`; of
//! the chats only the always-on and watched ones are (title and system prompt sealed as synced), so
//! boot can start their residents although PowerSync resumes from the saved cursor.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tracing::{error, warn};
use uuid::Uuid;

use crate::adapter::AgentKind;
use crate::user_prompt::UserSystemPrompt;

/// Shared handle to `Mirror`. The control socket task (see `control.rs`) needs
/// to read chat → user_id projections at the same time the main loop is
/// mutating `Mirror` from `handle_sync_event` / `handle_agent_response`. Using
/// `tokio::sync::RwLock` is deliberate: `handle_sync_event` `.await`s under
/// the write guard (decryption, R2 downloads, writer-channel sends), and a
/// `std::sync::RwLock` would deadlock + trip `clippy::await_holding_lock`.
pub type SharedMirror = Arc<tokio::sync::RwLock<Mirror>>;

#[derive(Default, Serialize, Deserialize)]
#[serde(default)]
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
    /// `chats.title` as synced: K_machine-sealed under the chat OWNER's key,
    /// like `messages.body`. Kept sealed so state.json carries no plaintext, yet
    /// a chat started at boot (before its row re-streams) gets its name.
    pub sealed_title: Option<String>,
    /// `sealed_title` decrypted. `None` when the chat has no title or the
    /// owner's key isn't unsealed yet — a chats row re-streams on every
    /// writeback, so a miss heals on the next one. Feeds `claude --name` via
    /// `session_name::from_title`.
    #[serde(skip)]
    pub title: Option<String>,
    /// `chats.pinned_at` (migration 0014) is non-NULL: the user pinned this chat
    /// to the top of the list. A warm resident of a pinned chat is evicted last
    /// and on a longer TTL (`warm.rs`).
    pub pinned: bool,
    /// `chats.always_on` (migration 0052): the chat's resident is never evicted
    /// and is started whenever missing (`main.rs::reconcile_always_on`).
    pub always_on: bool,
    /// `chats.auto_compact_tokens` (migration 0054): compact the resident once
    /// its context reaches this many tokens (`SessionState::after_turn`).
    pub auto_compact_tokens: Option<i64>,
    /// `chats.system_prompt_files` and `chats.system_prompt` (migration 0055)
    /// as synced: K_machine-sealed under the chat owner's key. Kept sealed so
    /// state.json carries no plaintext, yet an always-on chat started at boot
    /// (before its row re-streams) gets its prompt (`unseal_loaded_chats`).
    pub system_prompt_files: Option<String>,
    pub system_prompt: Option<String>,
    /// The two columns decrypted; a spawn knob, read through
    /// `Mirror::user_system_prompt`. Empty for an adapter that
    /// takes no user system prompt; a column that fails to decrypt marks it
    /// `undecryptable`, which fails the spawn.
    #[serde(skip)]
    pub user_system_prompt: UserSystemPrompt,
}

impl ChatState {
    fn unseal_title(&mut self, keys: &crate::crypto::KeyStore) {
        self.title = self.sealed_title.as_deref().and_then(|sealed| {
            let key = keys.get_existing(&self.user_id)?;
            crate::crypto::decrypt_field_b64(&key, sealed)
        });
    }

    fn unseal_system_prompt(&mut self, chat_id: &str, keys: &crate::crypto::KeyStore) {
        if !self.agent_kind.make_adapter().takes_user_system_prompt() {
            return;
        }
        let key = keys.get_existing(&self.user_id);
        let mut undecryptable = false;
        let mut unseal = |column: &str, sealed: &Option<String>| {
            let sealed = sealed.as_deref().filter(|s| !s.is_empty())?;
            let plain = key
                .as_ref()
                .and_then(|k| crate::crypto::decrypt_field_b64(k, sealed));
            if plain.is_none() {
                warn!(chat_id = %chat_id, column, "can't decrypt the chat's system prompt");
                undecryptable = true;
            }
            plain.filter(|p| !p.trim().is_empty())
        };
        let files = unseal("system_prompt_files", &self.system_prompt_files);
        let text = unseal("system_prompt", &self.system_prompt);
        self.user_system_prompt = UserSystemPrompt {
            files,
            text,
            undecryptable,
        };
    }
}

#[derive(Default, Serialize, Deserialize)]
pub struct Mirror {
    #[serde(default, serialize_with = "serialize_kept_chats")]
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
    /// Per chat, the highest user-message seq this machine handled — the
    /// replay guard in `handle_message_put`. Missing (pre-upgrade state.json,
    /// or no message handled yet) falls back to the `chats.last_seq` guard.
    #[serde(default)]
    handled_user_seqs: HashMap<String, i64>,
    /// Chats whose last-synced row had `agent_running`, `agent_alive` or
    /// `live_tasks` set.
    /// Persisted because PowerSync resumes from the saved cursor and won't
    /// re-stream those rows after a restart, when every such flag is stale:
    /// no agent outlives the spawner (`orphans.rs`).
    #[serde(default)]
    flagged_chats: HashSet<String>,
    /// `first_checkpoint` already ran this process.
    #[serde(skip)]
    booted: bool,
    /// Per always-on chat, starts of its resident no reconcile pass has seen
    /// alive yet (`main.rs::reconcile_always_on`).
    #[serde(skip)]
    pub always_on_starts: HashMap<String, u32>,
}

fn serialize_kept_chats<S: serde::Serializer>(
    chats: &HashMap<String, ChatState>,
    s: S,
) -> Result<S::Ok, S::Error> {
    let mut watched = crate::watch::watched_chat_ids();
    watched.extend(crate::tunnel::tunnel_chat_ids());
    s.collect_map(
        chats
            .iter()
            .filter(|(id, c)| c.always_on || watched.contains(*id)),
    )
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
    pub fn upsert_chat(&mut self, id: String, row_json: &str, keys: &crate::crypto::KeyStore) {
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

        // `live_tasks` joins the flags: a teardown that missed its clear leaves a
        // task blob on a row whose flags are already false, and nothing else
        // would ever reset it.
        let has_live_tasks = v.get("live_tasks").is_some_and(|t| !t.is_null());
        if crate::json_pg_bool(v.get("agent_running"))
            || crate::json_pg_bool(v.get("agent_alive"))
            || has_live_tasks
        {
            self.flagged_chats.insert(id.clone());
        } else {
            self.flagged_chats.remove(&id);
        }

        let sealed = |column: &str| v.get(column).and_then(|x| x.as_str()).map(str::to_string);
        let mut chat = ChatState {
            user_id,
            project_id: project_id.to_string(),
            worktree,
            last_seq,
            agent_session_id: incoming_agent_session_id,
            agent_kind,
            model,
            sealed_title: sealed("title").filter(|s| !s.is_empty()),
            title: None,
            pinned: v.get("pinned_at").is_some_and(|p| !p.is_null()),
            always_on: crate::json_pg_bool(v.get("always_on")),
            auto_compact_tokens: v.get("auto_compact_tokens").and_then(crate::json_to_i64),
            system_prompt_files: sealed("system_prompt_files"),
            system_prompt: sealed("system_prompt"),
            user_system_prompt: UserSystemPrompt::default(),
        };
        chat.unseal_title(keys);
        chat.unseal_system_prompt(&id, keys);
        self.chats.insert(id, chat);
    }

    /// Decrypt the titles and system prompts of the chats loaded from state.json.
    pub fn unseal_loaded_chats(&mut self, keys: &crate::crypto::KeyStore) {
        for (id, chat) in &mut self.chats {
            chat.unseal_title(keys);
            chat.unseal_system_prompt(id, keys);
        }
    }

    /// Chats the user pinned; their warm residents are evicted last (`warm.rs`).
    pub fn pinned_chats(&self) -> HashSet<String> {
        self.chat_ids(|c| c.pinned).into_iter().collect()
    }

    /// Always-on chats whose warm residents are never evicted (`warm.rs`).
    /// Only the machine owner's: a member must not pin the owner's machine.
    pub fn always_on_chats(&self) -> HashSet<String> {
        self.chats
            .keys()
            .filter(|id| self.is_always_on(id))
            .cloned()
            .collect()
    }

    pub fn is_always_on(&self, chat_id: &str) -> bool {
        self.chats
            .get(chat_id)
            .is_some_and(|c| c.always_on && self.is_owner(c.user_id))
    }

    /// The user system prompt `chat_id` spawns with. A sandboxed member's chat
    /// gets none, like an adapter that takes none: the files are read with
    /// the spawner's own access, outside claude's permission gating. A member
    /// whose row hasn't synced counts as sandboxed.
    pub fn user_system_prompt(&self, chat_id: &str) -> UserSystemPrompt {
        self.chats
            .get(chat_id)
            .filter(|_| self.is_unsandboxed(chat_id))
            .map(|c| c.user_system_prompt.clone())
            .unwrap_or_default()
    }

    /// A `kind` agent in `chat_id`'s place would spawn with a user system
    /// prompt (`user_system_prompt`).
    pub fn takes_user_system_prompt(&self, chat_id: &str, kind: AgentKind) -> bool {
        kind.make_adapter().takes_user_system_prompt() && self.is_unsandboxed(chat_id)
    }

    /// The chat's agent runs with the spawner's own access: its user is the
    /// owner or an unsandboxed member. An unknown chat, or a member whose row
    /// hasn't synced, counts as sandboxed.
    pub fn is_unsandboxed(&self, chat_id: &str) -> bool {
        self.chats.get(chat_id).is_some_and(|c| {
            self.is_owner(c.user_id) || self.member_is_sandboxed(&c.user_id) == Some(false)
        })
    }

    /// The chat may own a watch, whose command runs unsandboxed: its user is
    /// unsandboxed and still has a key here. A soft-revoked member keeps their
    /// `machine_users` row, sandbox bit included, and loses only the key.
    pub fn can_watch(&self, chat_id: &str, keys: &crate::crypto::KeyStore) -> bool {
        self.is_unsandboxed(chat_id)
            && self
                .chats
                .get(chat_id)
                .is_some_and(|c| keys.get_existing(&c.user_id).is_some())
    }

    /// True on the first call per process.
    pub fn first_checkpoint(&mut self) -> bool {
        !std::mem::replace(&mut self.booted, true)
    }

    /// `first_checkpoint` ran: the boot window is applied.
    pub fn booted(&self) -> bool {
        self.booted
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

    pub fn chat_ids(&self, pred: impl Fn(&ChatState) -> bool) -> Vec<String> {
        self.chats
            .iter()
            .filter(|(_, c)| pred(c))
            .map(|(id, _)| id.clone())
            .collect()
    }

    pub fn remove_chat(&mut self, id: &str) {
        self.chats.remove(id);
        self.handled_user_seqs.remove(id);
        self.flagged_chats.remove(id);
    }

    /// The flagged chats; at boot all their flags are stale. They stay flagged
    /// until their reset rows sync back, so a restart before the writer
    /// flushes resets them again.
    pub fn flagged_chat_ids(&self) -> Vec<String> {
        self.flagged_chats.iter().cloned().collect()
    }

    pub fn handled_user_seq(&self, chat_id: &str) -> Option<i64> {
        self.handled_user_seqs.get(chat_id).copied()
    }

    pub fn set_handled_user_seq(&mut self, chat_id: &str, seq: i64) {
        let handled = self
            .handled_user_seqs
            .entry(chat_id.to_string())
            .or_insert(seq);
        *handled = (*handled).max(seq);
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
