use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use base64::{engine::general_purpose::STANDARD as B64, Engine as _};
use crypto_box::aead::OsRng;
use crypto_box::{PublicKey, SecretKey};
use tracing::info;
use zeroize::{Zeroize, Zeroizing};

use crate::atomic::atomic_write_private;
use crate::crypto::{encode_b64_zeroized, read_b64_32};
use crate::zucchini_spawner_dir;

const SECRET_FILENAME: &str = "x25519_secret";

/// Cache the parsed `SecretKey` directly: `crypto_box::SecretKey` impls
/// `ZeroizeOnDrop`, so the long-term spawner identity isn't left around in
/// freed heap. `SecretKey::from(&[u8;32])` is a deref-copy of the bytes, so we
/// don't want to re-construct it from a bare `[u8;32]` on every call site.
pub fn load_or_generate_secret() -> Result<SecretKey> {
    let path = secret_path();
    match read_b64_32(&path) {
        Ok(bytes) => Ok(SecretKey::from(*bytes)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            let secret = SecretKey::generate(&mut OsRng);
            persist_secret(&secret)?;
            info!(path = %path.display(), "generated fresh x25519 spawner keypair");
            Ok(secret)
        }
        Err(e) => Err(e).with_context(|| format!("read x25519 secret {}", path.display())),
    }
}

pub fn public_key_b64(secret: &SecretKey) -> String {
    let pk: PublicKey = secret.public_key();
    B64.encode(pk.as_bytes())
}

/// Upper bound for the decoded sealed_blob: a libsodium sealedbox over a
/// 32-byte payload is exactly 80 bytes (32 ephemeral pk + 32 ct + 16 tag).
/// Cap an order of magnitude above so we don't bind to the exact size, but
/// reject obviously-oversized blobs before handing them to `unseal`.
const MAX_SEALED_BLOB_BYTES: usize = 128;

pub fn open_sealed(secret: &SecretKey, sealed_b64: &str) -> Result<Zeroizing<Vec<u8>>> {
    let blob = B64
        .decode(sealed_b64.trim())
        .context("sealed_blob is not valid base64")?;
    if blob.len() > MAX_SEALED_BLOB_BYTES {
        return Err(anyhow!(
            "sealed_blob too large: {} bytes (cap {})",
            blob.len(),
            MAX_SEALED_BLOB_BYTES
        ));
    }
    secret
        .unseal(&blob)
        .map(Zeroizing::new)
        .map_err(|e| anyhow!("sealedbox open failed: {:?}", e))
}

fn secret_path() -> PathBuf {
    zucchini_spawner_dir().join(SECRET_FILENAME)
}

fn persist_secret(secret: &SecretKey) -> Result<()> {
    // `main()` already ran `create_dir_all` + `ensure_spawner_dir_private` on
    // the spawner dir before calling `load_or_generate_secret`, so skip that.
    let final_path = secret_path();
    let encoded: Zeroizing<String> = {
        // `to_bytes()` returns a bare `[u8;32]` with no Zeroizing wrapper, and
        // `encode_b64_zeroized` borrows it through `&bytes` rather than moving
        // — so the explicit `bytes.zeroize()` is the only thing scrubbing
        // the raw private-key bytes on the stack here.
        let mut bytes = secret.to_bytes();
        let enc = encode_b64_zeroized(&bytes);
        bytes.zeroize();
        enc
    };
    atomic_write_private(&final_path, encoded.as_bytes())
        .with_context(|| format!("write x25519 secret {}", final_path.display()))?;

    #[cfg(not(unix))]
    tracing::warn!(
        path = %final_path.display(),
        "x25519 secret written without 0600 perms (non-unix target)"
    );

    Ok(())
}
