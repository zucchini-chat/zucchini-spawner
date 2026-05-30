//! Hermes-specific infrastructure shared by the `adapters::hermes` adapter.
//!
//! Unlike claude/codex/cursor — which run as short-lived per-turn subprocesses
//! and need no spawner-resident state — hermes runs as a single long-lived
//! `hermes gateway run` process supervised by the spawner. That supervisor
//! plus its Unix-socket multiplexer plus the embedded-Python-plugin self-heal
//! plus the `hermes-turn` trampoline live here. The `AgentAdapter` impl
//! itself stays in `adapters/hermes.rs` to match the per-adapter file layout
//! the other coding agents use.

pub mod plugin_install;
pub mod socket_server;
pub mod trampoline;
