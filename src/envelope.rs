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

/// Encrypt a text-only body into the `messages.body` wire format (base64 of
/// `nonce ‖ ct ‖ tag` over the envelope JSON). Inverse of `decode`.
///
/// Use for any body landing in a `sender='user'` row — notably a
/// `scheduled_messages` body, which the backend promotes verbatim into a user
/// message and the spawner THEN `decode`s. So it must be the `{text,attachments}`
/// envelope, not a raw field-encrypted string (`crypto::encrypt_field_b64`),
/// which decrypts to a bare string and fails `decode`'s JSON parse. (Agent-sent
/// frames are never `decode`d, so they stay raw field-encrypted.)
pub fn encode(text: &str, key: &KUser) -> Result<String> {
    let envelope = MessageEnvelope {
        text: text.to_string(),
        attachments: Vec::new(),
    };
    let json = serde_json::to_vec(&envelope).context("serialize message envelope JSON")?;
    Ok(B64.encode(crypto::encrypt(key, &json)))
}

pub fn decode(body_b64: &str, key: &KUser) -> Result<MessageEnvelope> {
    let bytes = B64.decode(body_b64).context("body is not valid base64")?;
    let plaintext = crypto::decrypt_bytes(key, &bytes)
        .ok_or_else(|| anyhow!("AEAD decrypt failed for messages.body"))?;
    serde_json::from_slice(&plaintext).context("parse message envelope JSON")
}

#[cfg(test)]
mod tests {
    use super::*;

    // `encode` must produce exactly what `decode` consumes (the promotion
    // wire-format contract). The original bug encrypted the raw body, so the
    // promoted `sender='user'` row failed `decode` and was silently dropped.
    #[test]
    fn encode_decode_roundtrips() {
        let key = KUser::from_bytes([7u8; 32]);
        let body = "⏰ scheduled body with unicode";
        let ct = encode(body, &key).expect("encode");
        let env = decode(&ct, &key).expect("decode");
        assert_eq!(env.text, body);
        assert!(env.attachments.is_empty());
    }

    // A raw field-encrypted string (the old, buggy path) must NOT decode as an
    // envelope — guards against silently reintroducing the format mismatch.
    #[test]
    fn raw_field_encrypted_body_fails_decode() {
        let key = KUser::from_bytes([9u8; 32]);
        let raw = crypto::encrypt_field_b64(&key, "plain body, no envelope");
        assert!(decode(&raw, &key).is_err());
    }
}
