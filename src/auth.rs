//! Token source for the spawner.
//!
//! Two modes:
//! - **Prod**: exchange the long-lived `ZUCCHINI_SPAWNER_TOKEN` for a short-lived
//!   RS256 JWT at `POST /auth/token`. Cache it until ~60s before expiry.
//! - **Dev**: use the pre-minted JWT in `ZUCCHINI_DEV_JWT` verbatim (no /auth/token call).

use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use chrono::{DateTime, Utc};
use reqwest::StatusCode;
use serde::Deserialize;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

const REFRESH_SLACK_SECS: i64 = 60;

#[derive(Deserialize)]
struct TokenRes {
    token: String,
    expires_at: DateTime<Utc>,
}

struct Cache {
    token: String,
    expires_at: DateTime<Utc>,
}

pub struct AuthClient {
    http: reqwest::Client,
    endpoint: String,
    spawner_token: String,
    cache: Mutex<Option<Cache>>,
    /// Cancelled when /auth/token returns 410; main loop awaits this to self-uninstall.
    revoked: CancellationToken,
}

impl AuthClient {
    pub fn new(api_base_url: &str, spawner_token: String) -> Self {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(15))
            .build()
            .expect("reqwest client");
        Self {
            http,
            endpoint: format!("{}/auth/token", api_base_url.trim_end_matches('/')),
            spawner_token,
            cache: Mutex::new(None),
            revoked: CancellationToken::new(),
        }
    }

    /// Shared cancellation token the main loop awaits. Cancelled when fetch_jwt sees 410.
    pub fn revoked_signal(&self) -> CancellationToken {
        self.revoked.clone()
    }

    pub async fn fetch_jwt(&self) -> Result<String> {
        let mut guard = self.cache.lock().await;
        let now = Utc::now();
        if let Some(c) = guard.as_ref() {
            if c.expires_at > now + chrono::Duration::seconds(REFRESH_SLACK_SECS) {
                return Ok(c.token.clone());
            }
        }

        let resp = self
            .http
            .post(&self.endpoint)
            .bearer_auth(&self.spawner_token)
            .send()
            .await
            .context("POST /auth/token")?;
        let status = resp.status();
        if status == StatusCode::GONE {
            // Drop cached JWT so a concurrent fetcher doesn't reuse an orphaned token.
            *guard = None;
            self.revoked.cancel();
            return Err(anyhow!("/auth/token 410 Gone — spawner revoked"));
        }
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow!("/auth/token {}: {}", status, body));
        }
        let parsed: TokenRes = resp.json().await.context("parse TokenRes")?;
        *guard = Some(Cache {
            token: parsed.token.clone(),
            expires_at: parsed.expires_at,
        });
        Ok(parsed.token)
    }
}

pub fn token_fetcher(
    client: Arc<AuthClient>,
) -> Box<dyn Fn() -> futures_util::future::BoxFuture<'static, Result<String>> + Send + Sync> {
    Box::new(move || {
        let c = client.clone();
        Box::pin(async move { c.fetch_jwt().await })
    })
}
