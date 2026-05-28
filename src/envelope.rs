//! E2E message body envelope.
//!
//! `messages.body` ciphertext decrypts to JSON of this shape:
//!
//! ```json
//! { "text": "...", "attachments": [ { "blob_key": "...", "size": 123, "name": "..." } ] }
//! ```
//!
//! Same wire shape is reused for agent-sent attachments (see
//! `zucchini-spawner attach-file`) but in a SEPARATE follow-up `messages`
//! row: the assistant text frame keeps its raw claude-SDK stream-json body
//! unchanged, and the spawner emits an additional row whose body is a
//! `MessageEnvelope { text: "", attachments }`. iOS renders that row as an
//! attachment-only bubble below the assistant text. The split preserves the
//! "one frame per row, body never grows" invariant and keeps older iOS
//! builds — which don't decode envelopes on agent bodies and would otherwise
//! drop the row — at the same loss surface as silently dropping (no
//! mid-bubble corruption).

use anyhow::{anyhow, Context, Result};
use base64::{engine::general_purpose::STANDARD as B64, Engine as _};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::crypto::{self, KUser};

#[derive(Debug, Serialize, Deserialize)]
pub struct MessageEnvelope {
    pub text: String,
    #[serde(default)]
    pub attachments: Vec<EnvelopeAttachment>,
}

/// `size` matches the iOS-side `MessageEnvelope.Attachment.size` (plaintext
/// byte length). `#[serde(default)]` for inbound decode tolerates older user
/// messages that didn't carry the field; outbound encode always writes it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvelopeAttachment {
    pub blob_key: Uuid,
    #[serde(default)]
    pub size: i64,
    pub name: String,
}

pub fn decode(body_b64: &str, key: &KUser) -> Result<MessageEnvelope> {
    let bytes = B64.decode(body_b64).context("body is not valid base64")?;
    let plaintext = crypto::decrypt_bytes(key, &bytes)
        .ok_or_else(|| anyhow!("AEAD decrypt failed for messages.body"))?;
    serde_json::from_slice(&plaintext).context("parse message envelope JSON")
}
