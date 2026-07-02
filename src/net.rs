//! Central Sentry reporting for failed outbound HTTP requests.
//!
//! Every outbound request in the spawner (backend `/api/writes`, `/auth/token`,
//! blob presign, R2 PUT/GET, sync stream) funnels its failure branch through
//! [`send_reporting`] (transport failures) and [`capture_response_failure`]
//! (non-2xx responses), so no request fails silently. The event shape is
//! kept byte-for-byte in step with the Qt client (`native-qt/src/Net/
//! SentryClient.cpp`) and the iOS/Android reporters so all platforms group into
//! the same issues in the shared `app` Sentry project.
//!
//! 🚨 The URL query string is DROPPED before anything is recorded — presigned R2
//! URLs carry a signature there and it must never reach Sentry. Only host+path
//! are kept.

use std::borrow::Cow;

use sentry::protocol::{Event, Level, Value};

/// Longest response-body prefix we attach to the event.
const BODY_SNIPPET_LEN: usize = 500;

/// The two real failure shapes a request can fail with.
enum Failure<'a> {
    /// No HTTP status was ever received (connect/DNS/TLS/timeout).
    Transport(&'a str),
    /// A completed but non-2xx response.
    Response { status: u16, body: &'a str },
}

/// Report a failed HTTP request to Sentry. No-op when the SDK isn't enabled.
///
/// Trigger (per the shared cross-platform contract): any non-2xx response OR a
/// transport failure. Call once, at the lowest shared layer, per failure.
fn capture_http_failure(method: &str, url: &str, failure: Failure) {
    let (host, norm_path) = split_host_and_normalized_path(url);
    let host_path = format!("{host}{norm_path}");

    let level = match failure {
        Failure::Transport(_) => Level::Error,
        Failure::Response { status, .. } if status >= 500 => Level::Error,
        Failure::Response { .. } => Level::Warning,
    };

    let message = match failure {
        Failure::Transport(err) => format!("HTTP {method} {host_path} — {err}"),
        Failure::Response { status, .. } => format!("HTTP {status} {method} {host_path}"),
    };

    let mut tags = sentry::protocol::Map::new();
    tags.insert("http.method".to_string(), method.to_string());
    tags.insert("http.host".to_string(), host.clone());
    if let Failure::Response { status, .. } = failure {
        tags.insert("http.status".to_string(), status.to_string());
    }

    let fingerprint_status: Cow<str> = match failure {
        Failure::Transport(_) => Cow::Borrowed("transport"),
        Failure::Response { status, .. } => Cow::Owned(status.to_string()),
    };
    let fingerprint = vec![
        Cow::Borrowed("http"),
        Cow::Owned(method.to_string()),
        Cow::Owned(host_path.clone()),
        fingerprint_status,
    ];

    let mut extra = sentry::protocol::Map::new();
    extra.insert("method".to_string(), Value::from(method));
    extra.insert("url".to_string(), Value::from(host_path));
    match failure {
        Failure::Transport(err) => {
            extra.insert("transport_error".to_string(), Value::from(err));
        }
        Failure::Response { status, body } => {
            extra.insert("status_code".to_string(), Value::from(status));
            extra.insert(
                "body".to_string(),
                Value::from(body.chars().take(BODY_SNIPPET_LEN).collect::<String>()),
            );
        }
    }

    sentry::capture_event(Event {
        level,
        message: Some(message),
        fingerprint: Cow::Owned(fingerprint),
        tags,
        extra,
        ..Default::default()
    });
}

/// Await a request future, reporting a transport failure to Sentry before
/// returning it. Collapses the duplicated `match … { Err(e) => { report; return
/// } }` boilerplate — the caller adds its own `?`/`.context(…)`.
pub async fn send_reporting(
    method: &str,
    url: &str,
    fut: impl std::future::Future<Output = reqwest::Result<reqwest::Response>>,
) -> reqwest::Result<reqwest::Response> {
    fut.await.inspect_err(|e| {
        capture_http_failure(method, url, Failure::Transport(&e.to_string()));
    })
}

/// Report a failed HTTP *response* (non-2xx). Reads the body itself (consuming
/// the response) and clamps the snippet — the one call needed by every
/// "presign / write returned non-2xx" branch.
pub async fn capture_response_failure(method: &str, url: &str, resp: reqwest::Response) -> String {
    let status = resp.status().as_u16();
    let body = resp.text().await.unwrap_or_default();
    capture_http_failure(
        method,
        url,
        Failure::Response {
            status,
            body: &body,
        },
    );
    body
}

/// Split a URL into `(host, normalized_path)`, dropping scheme, userinfo, port,
/// query and fragment. Path segments that look like an id — a UUID, a long hex
/// blob (>=16 hex chars), or an all-digit segment — collapse to `:id` so one
/// endpoint groups into one Sentry issue regardless of the concrete id.
fn split_host_and_normalized_path(url: &str) -> (String, String) {
    match reqwest::Url::parse(url) {
        Ok(parsed) => {
            let host = parsed.host_str().unwrap_or("").to_string();
            (host, normalize_path(parsed.path()))
        }
        // Not a valid absolute URL — fall back to treating the whole string as a
        // path so we still record *something* useful (never the raw query).
        Err(_) => {
            let no_query = url.split(['?', '#']).next().unwrap_or(url);
            ("".to_string(), normalize_path(no_query))
        }
    }
}

fn normalize_path(path: &str) -> String {
    let normalized: Vec<&str> = path
        .split('/')
        .map(|seg| if is_id_segment(seg) { ":id" } else { seg })
        .collect();
    normalized.join("/")
}

fn is_id_segment(seg: &str) -> bool {
    if seg.is_empty() {
        return false;
    }
    let all_digits = seg.chars().all(|c| c.is_ascii_digit());
    let hex_len = seg.len() >= 16 && seg.chars().all(|c| c.is_ascii_hexdigit());
    all_digits || hex_len || is_uuid(seg)
}

fn is_uuid(seg: &str) -> bool {
    let bytes = seg.as_bytes();
    if bytes.len() != 36 {
        return false;
    }
    bytes.iter().enumerate().all(|(i, &b)| {
        if matches!(i, 8 | 13 | 18 | 23) {
            b == b'-'
        } else {
            b.is_ascii_hexdigit()
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drops_query_and_normalizes_ids() {
        let (host, path) = split_host_and_normalized_path(
            "https://r2.example.com/blobs/123e4567-e89b-12d3-a456-426614174000?X-Amz-Signature=secret",
        );
        assert_eq!(host, "r2.example.com");
        assert_eq!(path, "/blobs/:id");
    }

    #[test]
    fn collapses_hex_and_digit_segments() {
        let (_, path) =
            split_host_and_normalized_path("https://api.zucchini.chat/v1/deadbeefdeadbeef99/42");
        assert_eq!(path, "/v1/:id/:id");
    }

    #[test]
    fn keeps_short_words() {
        let (host, path) = split_host_and_normalized_path("https://api.zucchini.chat/api/writes");
        assert_eq!(host, "api.zucchini.chat");
        assert_eq!(path, "/api/writes");
    }
}
