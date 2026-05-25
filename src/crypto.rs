//! Symmetric crypto for PowerSync ciphertext fields.
//!
//! Per-user 32-byte secret stored as base64 in `~/.zucchini-spawner/key_<user_id>`.
//! Ciphertext layout: 24-byte XChaCha20-Poly1305 nonce prefix, then AEAD ciphertext.
//!
//! Naming: these bytes ARE K_user from the spawner's point of view (a spawner is
//! pinned to one machine, so its axis of variation is the user). iOS calls the same
//! bytes "K_machine" because on that side a user has many machines. Both names are
//! right for their own axis — don't rename to K_machine here.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Context, Result};
use base64::{engine::general_purpose::STANDARD as B64, Engine as _};
use chacha20poly1305::{
    aead::{Aead, KeyInit, OsRng},
    AeadCore, Key, XChaCha20Poly1305, XNonce,
};
use tracing::info;
use uuid::Uuid;
use zeroize::{Zeroize, Zeroizing};

/// Read a file containing exactly base64-of-32-bytes (the layout used for both
/// `key_<user_id>` and `x25519_secret`). Both the `String` and the decoded
/// `Vec<u8>` are wrapped so the secret plaintext doesn't sit in freed-slot heap
/// once this returns; the final `[u8;32]` is returned inside a `Zeroizing` so
/// the caller's stack slot is scrubbed too. `NotFound` propagates as
/// `io::ErrorKind::NotFound` so callers can match on it.
pub fn read_b64_32(path: &Path) -> std::io::Result<Zeroizing<[u8; 32]>> {
    let contents: Zeroizing<String> = Zeroizing::new(fs::read_to_string(path)?);
    let decoded: Zeroizing<Vec<u8>> = Zeroizing::new(
        B64.decode(contents.trim())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?,
    );
    if decoded.len() != 32 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("expected 32 bytes, got {}", decoded.len()),
        ));
    }
    // Build the [u8;32] directly inside the Zeroizing wrapper — no intermediate
    // stack array, so there's no extra slot to scrub manually.
    let mut out: Zeroizing<[u8; 32]> = Zeroizing::new([0u8; 32]);
    out.copy_from_slice(&decoded);
    Ok(out)
}

/// Encode raw bytes as base64 into a `Zeroizing<String>` so the encoded form
/// doesn't linger in freed-slot heap once persisted. Used for both
/// `key_<user_id>` and `x25519_secret`.
pub fn encode_b64_zeroized(bytes: &[u8]) -> Zeroizing<String> {
    let cap = bytes.len().div_ceil(3) * 4;
    let mut s: Zeroizing<String> = Zeroizing::new(String::with_capacity(cap));
    B64.encode_string(bytes, &mut s);
    // If a future caller passes a non-3-multiple length, the encoded string
    // would grow past `cap` and `String::push_str` would reallocate, leaving
    // a non-zeroized copy in freed-slot heap. Fail loudly in debug builds.
    debug_assert_eq!(s.len(), cap, "encode_b64_zeroized: encoded length exceeded preallocated capacity");
    s
}

/// `Drop` scrubs the cached 32-byte K_user when the last `Arc<KUser>` in
/// `KeyStore::cache` is dropped (revocation via `forget`, process exit).
/// Without it the secret would linger in the allocator's freed-slot pool until
/// reuse — exposed via Sentry heap dumps, /proc/<pid>/maps for a same-uid
/// attacker, or swap.
///
/// No `Clone`: shared access goes through `Arc<KUser>`, so a bare clone would
/// produce a sibling array with its own scrub timing diverging from the Arc.
pub struct KUser([u8; 32]);

impl Drop for KUser {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}

impl KUser {
    fn load_from(path: &std::path::Path) -> Result<Self> {
        let bytes = read_b64_32(path)
            .with_context(|| format!("read key file {}", path.display()))?;
        // `*bytes` deref-copies the [u8;32] out of `Zeroizing` into a fresh
        // stack slot owned by `KUser`. The `Zeroizing` wrapper scrubs its
        // own copy on drop here; the new slot is then scrubbed by `KUser::drop`
        // when the last Arc<KUser> is released.
        Ok(KUser(*bytes))
    }
}

/// Single owner for the `key_<user_id>` filename convention.
pub(crate) fn user_key_path(user_id: &Uuid) -> PathBuf {
    crate::zucchini_spawner_dir().join(format!("key_{}", user_id))
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

    pub fn forget(&self, user_id: &Uuid) {
        let mut cache = self.cache.lock().expect("KeyStore mutex");
        cache.remove(user_id);
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
        let per_user = user_key_path(user_id);
        match KUser::load_from(&per_user) {
            Ok(k) => {
                let arc = Arc::new(k);
                cache.insert(*user_id, Some(arc.clone()));
                Ok(arc)
            }
            Err(e) => {
                // Negative-cache only the "no file" case; parse/IO errors bubble
                // uncached so a fixed file can succeed on retry. `with_context`
                // wraps the io::Error as the chain root, so walk the chain.
                let not_found = e
                    .chain()
                    .filter_map(|c| c.downcast_ref::<std::io::Error>())
                    .any(|io| io.kind() == std::io::ErrorKind::NotFound);
                if !not_found {
                    return Err(e);
                }
                // Legacy-key migration: in the pre per-user-keys era a single
                // `~/.zucchini-spawner/key` held the owner's K_user. On first
                // lookup after upgrade, rename it into the per-user path so
                // subsequent calls take the fast path above. Gated to the
                // NotFound arm so the `legacy.exists()` syscall doesn't run
                // on every hot-path `get()` once migration is complete.
                if try_migrate_legacy_key(user_id, &per_user)? {
                    let k = KUser::load_from(&per_user)?;
                    let arc = Arc::new(k);
                    cache.insert(*user_id, Some(arc.clone()));
                    return Ok(arc);
                }
                cache.insert(*user_id, None);
                Err(anyhow!(
                    "no key for user {} ({} not found)",
                    user_id,
                    per_user.display()
                ))
            }
        }
    }
}

/// Returns Ok(true) iff a legacy `key` file was present and successfully
/// renamed into `per_user`. Ok(false) means no legacy file existed (most
/// installs). Errors propagate so a half-completed migration is visible.
fn try_migrate_legacy_key(user_id: &Uuid, per_user: &Path) -> Result<bool> {
    let legacy = crate::zucchini_spawner_dir().join("key");
    if !legacy.exists() {
        return Ok(false);
    }
    // chmod before rename so a chmod failure leaves the file at the legacy
    // path for the next call to retry — chmodding after the rename would
    // strand it at its legacy mode if the next entry short-circuits.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&legacy, fs::Permissions::from_mode(0o600))
            .with_context(|| format!("chmod 0600 on legacy key {}", legacy.display()))?;
    }
    // Atomic rename within the same filesystem — no half-state.
    fs::rename(&legacy, per_user).with_context(|| {
        format!("migrate legacy key {} -> {}", legacy.display(), per_user.display())
    })?;
    info!(
        user_id = %user_id,
        from = %legacy.display(),
        to = %per_user.display(),
        "migrated legacy spawner key to per-user file"
    );
    Ok(true)
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
