//! Symmetric crypto for PowerSync ciphertext fields.
//!
//! Per-user 32-byte secret stored as base64 in `~/.zucchini-spawner/key_<user_id>`.
//! `install.sh` writes the file. `KeyStore::get` also migrates a legacy single-user
//! `~/.zucchini-spawner/key` (pre per-user keys) into the new path via a one-shot
//! in-place rename on first lookup.
//!
//! Naming: these bytes ARE K_user from the spawner's point of view. Content keys are
//! per (user, machine), but a spawner is pinned to a single machine — so its axis
//! of variation is the user, hence `key_<user_id>` and the `KUser` type name. A
//! machine can host many users (shared-machines feature), so the filename is keyed
//! by user_id, not a single global `key`. iOS calls the same bytes "K_machine"
//! because on that side a user has many machines; both sides are right for their
//! own axis. Don't rename to "K_machine" here.
//!
//! Nonce convention: first 24 bytes of the ciphertext are the
//! XChaCha20-Poly1305 nonce, the remainder is the AEAD ciphertext.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Context, Result};
use base64::{engine::general_purpose::STANDARD as B64, Engine as _};
use chacha20poly1305::{
    aead::{Aead, KeyInit, OsRng},
    AeadCore, Key, XChaCha20Poly1305, XNonce,
};
use tracing::info;
use uuid::Uuid;

#[derive(Clone)]
pub struct KUser([u8; 32]);

impl KUser {
    fn load_from(path: &std::path::Path) -> Result<Self> {
        let contents = fs::read_to_string(path)
            .with_context(|| format!("read key file {}", path.display()))?;
        let decoded = B64.decode(contents.trim()).context("key is not valid base64")?;
        let bytes: [u8; 32] = decoded
            .try_into()
            .map_err(|v: Vec<u8>| anyhow!("key must decode to 32 bytes, got {}", v.len()))?;
        Ok(KUser(bytes))
    }
}

/// Per-user key cache. Reads `key_<user_id>` lazily on first lookup;
/// renames the legacy single-user `key` file into place if present.
/// `None` entries are negative-cache misses — synced rows for un-enrolled
/// users would otherwise re-stat both `key_<user_id>` and `legacy_key`
/// per agent stream frame.
pub struct KeyStore {
    cache: Mutex<HashMap<Uuid, Option<Arc<KUser>>>>,
}

impl KeyStore {
    pub fn new() -> Self {
        Self { cache: Mutex::new(HashMap::new()) }
    }

    pub fn get(&self, user_id: &Uuid) -> Result<Arc<KUser>> {
        // Single lock window: spawner has 1-2 uids per process lifetime and the
        // fs I/O fires at most once per uid, so brief blocking under the guard
        // is fine. Read/parse errors are NOT cached so a fixed file can succeed
        // on retry; only the "no file exists" case is negative-cached.
        let mut cache = self.cache.lock().expect("KeyStore mutex");
        if let Some(entry) = cache.get(user_id) {
            return entry
                .clone()
                .ok_or_else(|| anyhow!("no key for user {}", user_id));
        }
        let dir = PathBuf::from(std::env::var_os("HOME").ok_or_else(|| anyhow!("HOME not set"))?)
            .join(".zucchini-spawner");
        let per_user = dir.join(format!("key_{}", user_id));
        if !per_user.exists() {
            let legacy = dir.join("key");
            if legacy.exists() {
                // Atomic rename within the same filesystem — no half-state.
                fs::rename(&legacy, &per_user).with_context(|| {
                    format!("migrate legacy key {} -> {}", legacy.display(), per_user.display())
                })?;
                info!(
                    user_id = %user_id,
                    from = %legacy.display(),
                    to = %per_user.display(),
                    "migrated legacy spawner key to per-user file"
                );
            } else {
                cache.insert(*user_id, None);
                return Err(anyhow!(
                    "no key for user {} ({} not found)",
                    user_id,
                    per_user.display()
                ));
            }
        }
        let arc = Arc::new(KUser::load_from(&per_user)?);
        cache.insert(*user_id, Some(arc.clone()));
        Ok(arc)
    }
}

impl Default for KeyStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Encrypt a plaintext string into PowerSync's BYTEA ciphertext convention:
/// 24-byte nonce prefix + AEAD ciphertext. Returns the raw bytes — callers base64
/// them for the JSON wire format.
pub fn encrypt(key: &KUser, plaintext: &[u8]) -> Vec<u8> {
    let cipher = XChaCha20Poly1305::new(Key::from_slice(&key.0));
    let nonce = XChaCha20Poly1305::generate_nonce(&mut OsRng);
    let mut out = Vec::with_capacity(24 + plaintext.len() + 16);
    out.extend_from_slice(nonce.as_slice());
    let ct = cipher.encrypt(&nonce, plaintext).expect("aead encrypt");
    out.extend_from_slice(&ct);
    out
}

pub fn encrypt_field_b64(key: &KUser, plaintext: &str) -> String {
    B64.encode(encrypt(key, plaintext.as_bytes()))
}

/// Decrypt a `nonce‖ct‖tag` blob. Used for both message-envelope JSON
/// (caller `serde_json::from_slice`s the bytes) and binary R2 blobs.
pub fn decrypt_bytes(key: &KUser, ciphertext: &[u8]) -> Option<Vec<u8>> {
    if ciphertext.len() < 24 {
        return None;
    }
    let (nonce_bytes, body) = ciphertext.split_at(24);
    let cipher = XChaCha20Poly1305::new(Key::from_slice(&key.0));
    cipher.decrypt(XNonce::from_slice(nonce_bytes), body).ok()
}
