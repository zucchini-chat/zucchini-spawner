//! Symmetric crypto for PowerSync ciphertext fields.
//!
//! Loads K_user (32-byte secret) from `~/.zucchini-spawner/key` (base64) — user-level, shared with the apps.
//! Nonce convention: first 24 bytes
//! of the ciphertext are the XChaCha20-Poly1305 nonce, the remainder is the AEAD ciphertext.
//! Pairing (proper key transfer between devices) will replace the file-on-disk later.

use std::fs;
use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use base64::{engine::general_purpose::STANDARD as B64, Engine as _};
use chacha20poly1305::{
    aead::{Aead, KeyInit, OsRng},
    AeadCore, Key, XChaCha20Poly1305, XNonce,
};

#[derive(Clone)]
pub struct KUser([u8; 32]);

impl KUser {
    pub fn load() -> Result<Self> {
        let path = key_path().ok_or_else(|| anyhow!("HOME not set"))?;
        let contents = fs::read_to_string(&path)
            .with_context(|| format!("read key file {}", path.display()))?;
        let decoded = B64.decode(contents.trim()).context("key is not valid base64")?;
        let bytes: [u8; 32] = decoded
            .try_into()
            .map_err(|v: Vec<u8>| anyhow!("key must decode to 32 bytes, got {}", v.len()))?;
        Ok(KUser(bytes))
    }
}

fn key_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    Some(PathBuf::from(home).join(".zucchini-spawner").join("key"))
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
