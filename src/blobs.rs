//! Spawner-side attachment download + decrypt.
//!
//! Flow per blob: POST /api/blobs/download-url → presigned R2 GET → decrypt with
//! K_user → write to `~/.zucchini-spawner/attachments/<uuid><ext>`. Files persist
//! for the life of the process (TTL/GC is out of MVP scope per attachements_plan.md).

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use futures_util::future::try_join_all;
use serde::{Deserialize, Serialize};
use tracing::info;
use uuid::Uuid;

use crate::crypto::{self, KUser};
use crate::envelope::EnvelopeAttachment;
use crate::writer::TokenFetcher;

fn attachment_dir() -> PathBuf {
    crate::zucchini_spawner_dir().join("attachments")
}

pub struct BlobDownloader {
    http: reqwest::Client,
    download_url_endpoint: String,
    fetch_token: TokenFetcher,
}

#[derive(Serialize)]
struct DownloadUrlReq<'a> {
    blob_key: &'a Uuid,
}

#[derive(Deserialize)]
struct DownloadUrlRes {
    url: String,
}

pub struct DownloadedAttachment {
    pub path: PathBuf,
    pub name: String,
}

impl BlobDownloader {
    pub fn new(api_base_url: &str, fetch_token: TokenFetcher) -> Self {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .expect("reqwest client");
        Self {
            http,
            download_url_endpoint: format!(
                "{}/api/blobs/download-url",
                api_base_url.trim_end_matches('/')
            ),
            fetch_token,
        }
    }

    /// Download + decrypt every attachment in parallel, returning local paths in
    /// input order. Errors propagate so the caller can decide to skip the message
    /// rather than hand claude a prompt that points at non-existent files.
    pub async fn fetch_all(
        &self,
        attachments: &[EnvelopeAttachment],
        key: &KUser,
    ) -> Result<Vec<DownloadedAttachment>> {
        if attachments.is_empty() {
            return Ok(Vec::new());
        }
        let dir = attachment_dir();
        tokio::fs::create_dir_all(&dir)
            .await
            .with_context(|| format!("create attachments dir {}", dir.display()))?;

        try_join_all(attachments.iter().map(|att| {
            let path = dir.join(local_filename(&att.blob_key, &att.name));
            async move {
                // The updater guard keeps the spawner from exiting mid-fetch, so
                // a half-written file from a crash isn't a realistic concern.
                if !path.exists() {
                    self.download_one(&att.blob_key, &path, key).await?;
                }
                Ok::<_, anyhow::Error>(DownloadedAttachment { path, name: att.name.clone() })
            }
        }))
        .await
    }

    async fn download_one(
        &self,
        blob_key: &Uuid,
        dest: &Path,
        key: &KUser,
    ) -> Result<()> {
        let token = (self.fetch_token)().await?;
        let resp = self
            .http
            .post(&self.download_url_endpoint)
            .bearer_auth(&token)
            .json(&DownloadUrlReq { blob_key })
            .send()
            .await
            .context("POST /api/blobs/download-url")?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow!("/api/blobs/download-url {}: {}", status, body));
        }
        let presigned: DownloadUrlRes = resp.json().await.context("parse DownloadUrlRes")?;

        let blob = self
            .http
            .get(&presigned.url)
            .send()
            .await
            .context("GET presigned R2 url")?;
        let blob_status = blob.status();
        if !blob_status.is_success() {
            let body = blob.text().await.unwrap_or_default();
            return Err(anyhow!("R2 GET {}: {}", blob_status, body));
        }
        let ciphertext = blob.bytes().await.context("read R2 body")?;

        let plaintext = crypto::decrypt_bytes(key, &ciphertext)
            .ok_or_else(|| anyhow!("AEAD decrypt failed for blob {}", blob_key))?;

        tokio::fs::write(dest, &plaintext)
            .await
            .with_context(|| format!("write attachment {}", dest.display()))?;
        info!(blob_key = %blob_key, path = %dest.display(), size = plaintext.len(), "downloaded attachment");
        Ok(())
    }
}

/// `<uuid><ext>` — UUID stem keeps writes collision-free across messages, the
/// extension preserves whatever claude needs to recognise the file type.
fn local_filename(blob_key: &Uuid, name: &str) -> String {
    let ext = Path::new(name)
        .extension()
        .and_then(|e| e.to_str())
        .filter(|e| !e.is_empty())
        .map(|e| format!(".{e}"))
        .unwrap_or_default();
    format!("{blob_key}{ext}")
}

/// Build the prompt claude receives. Annotates each on-disk path with the
/// envelope's original `name` so claude sees something meaningful. Drops the
/// `[User's message:]` section for image-only messages (empty `text`) so we
/// don't emit a dangling header.
pub fn build_prompt(text: &str, attachments: &[DownloadedAttachment]) -> String {
    if attachments.is_empty() {
        return text.to_string();
    }
    let mut out = String::from("[The user attached the following files:]\n");
    for a in attachments {
        // Sanitise name before splicing — collapse control chars (incl. newlines)
        // to spaces. Defensive prompt-injection guard; harmless in single-user MVP.
        let safe_name: String = a
            .name
            .chars()
            .map(|c| if c.is_control() { ' ' } else { c })
            .collect();
        out.push_str(&format!(
            "- {} (original name: {})\n",
            a.path.display(),
            safe_name
        ));
    }
    if !text.is_empty() {
        out.push_str("\n[User's message:]\n");
        out.push_str(text);
    }
    out
}
