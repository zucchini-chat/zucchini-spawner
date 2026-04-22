//! Dev-only symmetric crypto for PowerSync ciphertext fields.
//!
//! Loads a 32-byte key from `~/.zucchini-spawner/dev_key` (base64) — user-level, shared with the apps.
//! Nonce convention: first 24 bytes
//! of the ciphertext are the XChaCha20-Poly1305 nonce, the remainder is the AEAD ciphertext.
//! Real key management (K_user, pairing) will replace this later.

use std::fs;
use std::path::PathBuf;

use base64::{engine::general_purpose::STANDARD as B64, Engine as _};
use chacha20poly1305::{
    aead::{Aead, KeyInit, OsRng},
    AeadCore, Key, XChaCha20Poly1305, XNonce,
};
use tracing::warn;

pub struct DevKey([u8; 32]);

impl DevKey {
    pub fn load_or_warn() -> Option<Self> {
        let path = key_path()?;
        let contents = match fs::read_to_string(&path) {
            Ok(s) => s,
            Err(e) => {
                warn!(
                    error = %e,
                    path = %path.display(),
                    "no dev key found; ciphertext fields will be passed as utf-8-lossy bytes"
                );
                return None;
            }
        };
        let decoded = match B64.decode(contents.trim()) {
            Ok(b) => b,
            Err(e) => {
                warn!(error = %e, "dev key is not valid base64");
                return None;
            }
        };
        if decoded.len() != 32 {
            warn!(len = decoded.len(), "dev key must decode to exactly 32 bytes");
            return None;
        }
        let mut key = [0u8; 32];
        key.copy_from_slice(&decoded);
        Some(DevKey(key))
    }
}

fn key_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    Some(PathBuf::from(home).join(".zucchini-spawner").join("dev_key"))
}

/// Encrypt a plaintext string into PowerSync's BYTEA ciphertext convention:
/// 24-byte nonce prefix + AEAD ciphertext. Returns the raw bytes — callers base64
/// them for the JSON wire format.
pub fn encrypt(key: &DevKey, plaintext: &[u8]) -> Vec<u8> {
    let cipher = XChaCha20Poly1305::new(Key::from_slice(&key.0));
    let nonce = XChaCha20Poly1305::generate_nonce(&mut OsRng);
    let mut out = Vec::with_capacity(24 + plaintext.len() + 16);
    out.extend_from_slice(nonce.as_slice());
    let ct = cipher.encrypt(&nonce, plaintext).expect("aead encrypt");
    out.extend_from_slice(&ct);
    out
}

/// Encrypt and base64-encode. When no key is available (dev fallback),
/// falls back to raw UTF-8 bytes so the end-to-end path still works.
pub fn encrypt_field_b64(key: Option<&DevKey>, plaintext: &str) -> String {
    let bytes = match key {
        Some(k) => encrypt(k, plaintext.as_bytes()),
        None => plaintext.as_bytes().to_vec(),
    };
    B64.encode(&bytes)
}

pub fn decrypt(key: &DevKey, ciphertext: &[u8]) -> Option<String> {
    if ciphertext.len() < 24 {
        return None;
    }
    let (nonce_bytes, body) = ciphertext.split_at(24);
    let cipher = XChaCha20Poly1305::new(Key::from_slice(&key.0));
    let plaintext = cipher.decrypt(XNonce::from_slice(nonce_bytes), body).ok()?;
    String::from_utf8(plaintext).ok()
}

/// Decode PowerSync's string representation of a BYTEA column (base64) and decrypt it.
/// Without a dev key, returns the raw ciphertext bytes as a UTF-8-lossy string so
/// the spawner still works end-to-end without crypto.
pub fn decrypt_field(key: Option<&DevKey>, field: &serde_json::Value) -> Option<String> {
    let s = field.as_str()?;
    let bytes = B64.decode(s).ok()?;
    match key {
        Some(k) => decrypt(k, &bytes),
        None => Some(String::from_utf8_lossy(&bytes).into_owned()),
    }
}
