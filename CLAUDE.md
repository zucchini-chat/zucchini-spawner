# zucchini-spawner

Rust. PowerSync sync-stream client (read-only NDJSON) + writeback. Spawns claude code on incoming user messages. `cargo check` / `cargo run`.

Installed via `install.sh`, which enrolls a Machine and runs a heartbeat. Runtime dir: `~/.zucchini-spawner/` (config, sync cursor, `key`). Dev JWT via `ZUCCHINI_DEV_JWT`.

In prod the installer registers a **per-user service** (`SERVICE_NAME=chat.zucchini.spawner`) — `systemctl --user` on Linux (loginctl linger enabled so it survives logout), launchd LaunchAgent on macOS. Read logs with `journalctl --user -u chat.zucchini.spawner -f` (Linux); on the e2e box (`beprod`) that's the source of truth for what the spawner actually did. The install/build/ship mechanics are simpler to read in `install.sh`, `build-releases.sh`, and `uninstall.sh` than to mirror here.

## Message-frame invariant

Consumes claude's `--output-format stream-json` and writes one `messages` row per JSON frame (whole frames, not token deltas — bodies are final at insert, never grow; attachments are encoded inside the body envelope so they're frozen alongside it). Combined with append-only ordering by `seq`, `(count, last.id)` is a stable identity for a chat's message array — guards like `ChatMessagesList.ApplyFingerprint` depend on this; widen them before adding edits, deletes, or out-of-envelope attachment hydration.

## Scope of responsibility

**Thin spawn layer — launches `claude` with our flags and that's it.** Claude's own config (`~/.claude/settings.json`, project `.claude/settings.json`, MCP allowlist, sandbox policy) is the user's responsibility, not Zucchini's. Features that touch how claude runs (e.g. the machines-sharing sandbox toggle) flip a flag on the spawn — they don't curate config or ship policy defaults; scope-of-responsibility gaps are communicated via UI disclaimers, not Zucchini-side wrappers.

## Hot reload (author's machine)

Two launchd agents — see `tmp/spawner-dev-watcher.md`. `WatchPaths` only covers `src/`, so after a `Cargo.toml` bump `touch zucchini-spawner/src/main.rs` to force a rebuild.

## Public GitHub mirror

Mirrored to `github.com/zucchini-chat/zucchini-spawner` — the only Zucchini code that runs on user machines, so the only one we open-source. Push-only via `git subtree`; remote `spawner-public` uses SSH alias `github_zucchini_admin`. Plan + supply-chain rationale (cosign-verified releases via Sigstore, dual update channel `users.spawner_source`): `tmp/verifiable-spawner.md`.

**Sync the mirror after committing spawner changes.** From monorepo root:

```
git subtree split --prefix=zucchini-spawner --branch=spawner-export && git push spawner-public spawner-export:main
```

Subtree split reads committed history only, so a dirty working tree is fine. Required before tagging a release once cosign verification ships — the public commit at the tag is what the Sigstore signature binds to; without the sync, the release can't be reproduced or verified.
