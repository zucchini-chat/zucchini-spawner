//! Central Sentry reporting for failed outbound HTTP requests.
//!
//! Every outbound request in the spawner (backend `/api/writes`, `/auth/token`,
//! blob presign, R2 PUT/GET, sync stream) reports a non-2xx response through
//! [`capture_response_failure`]. Transport failures (offline, timeout, DNS) are
//! not reported: retry loops on a sleeping laptop turned them into thousands of
//! events. The event shape is
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

/// Report a failed HTTP *response* (non-2xx) to Sentry and return its body.
/// Reads the body itself (consuming the response) and clamps the snippet. No-op
/// reporting when the SDK isn't enabled.
pub async fn capture_response_failure(method: &str, url: &str, resp: reqwest::Response) -> String {
    let status = resp.status().as_u16();
    let body = resp.text().await.unwrap_or_default();
    let (host, norm_path) = split_host_and_normalized_path(url);
    let host_path = format!("{host}{norm_path}");

    let mut tags = sentry::protocol::Map::new();
    tags.insert("http.method".to_string(), method.to_string());
    tags.insert("http.host".to_string(), host);
    tags.insert("http.status".to_string(), status.to_string());

    let fingerprint = vec![
        Cow::Borrowed("http"),
        Cow::Owned(method.to_string()),
        Cow::Owned(host_path.clone()),
        Cow::Owned(status.to_string()),
    ];

    let mut extra = sentry::protocol::Map::new();
    extra.insert("method".to_string(), Value::from(method));
    extra.insert("url".to_string(), Value::from(host_path.clone()));
    extra.insert("status_code".to_string(), Value::from(status));
    extra.insert(
        "body".to_string(),
        Value::from(body.chars().take(BODY_SNIPPET_LEN).collect::<String>()),
    );

    sentry::capture_event(Event {
        level: if status >= 500 {
            Level::Error
        } else {
            Level::Warning
        },
        message: Some(format!("HTTP {status} {method} {host_path}")),
        fingerprint: Cow::Owned(fingerprint),
        tags,
        extra,
        ..Default::default()
    });
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
