//! E2E message body envelope.
//!
//! `messages.body` ciphertext decrypts to JSON of this shape:
//!
//! ```json
//! { "text": "...", "attachments": [ { "blob_key": "...", "name": "..." } ] }
//! ```

use anyhow::{anyhow, Context, Result};
use base64::{engine::general_purpose::STANDARD as B64, Engine as _};
use serde::Deserialize;
use uuid::Uuid;

use crate::crypto::{self, DevKey};

#[derive(Debug, Deserialize)]
pub struct MessageEnvelope {
    pub text: String,
    #[serde(default)]
    pub attachments: Vec<EnvelopeAttachment>,
}

#[derive(Debug, Deserialize)]
pub struct EnvelopeAttachment {
    pub blob_key: Uuid,
    pub name: String,
}

pub fn decode(body_b64: &str, key: &DevKey) -> Result<MessageEnvelope> {
    let bytes = B64.decode(body_b64).context("body is not valid base64")?;
    let plaintext = crypto::decrypt_bytes(key, &bytes)
        .ok_or_else(|| anyhow!("AEAD decrypt failed for messages.body"))?;
    serde_json::from_slice(&plaintext).context("parse message envelope JSON")
}
