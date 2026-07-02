//! Single source of the HTTP `User-Agent` string every reqwest client in the
//! spawner sends, so requests identify themselves (and don't blend in with the
//! native-qt sync FFI, which also uses reqwest with the bare default UA).
//!
//! Shape (shared across all Zucchini clients — iOS/Android/Qt use the same):
//! `Zucchini-<Client>/<appVersion> (<detail; semicolon-separated>)`.
//! For the spawner: `Zucchini-Spawner/0.5.0 (macos; aarch64)`.

use std::sync::OnceLock;

/// The spawner's `User-Agent` header value, e.g. `Zucchini-Spawner/3.2.1 (macos; aarch64)`.
///
/// `std::env::consts::{OS,ARCH}` are consts (not literals) so `concat!` can't
/// build this — cache it in a `OnceLock` instead (built once, no deps).
pub fn user_agent() -> &'static str {
    static UA: OnceLock<String> = OnceLock::new();
    UA.get_or_init(|| {
        format!(
            "Zucchini-Spawner/{} ({}; {})",
            env!("CARGO_PKG_VERSION"),
            std::env::consts::OS,
            std::env::consts::ARCH,
        )
    })
}
