# zucchini-spawner

Rust. PowerSync sync-stream client (read-only NDJSON) + writeback. Spawns claude code on incoming user messages. `cargo check` / `cargo run`.

Installed via `install.sh`, which enrolls a Machine and runs a heartbeat. Runtime dir: `~/.zucchini-spawner/` (config, sync cursor, `key`). Dev JWT via `ZUCCHINI_DEV_JWT`.

In prod the installer registers a **per-user service** (`SERVICE_NAME=chat.zucchini.spawner`) — `systemctl --user` on Linux (loginctl linger enabled so it survives logout), launchd LaunchAgent on macOS. Read logs with `journalctl --user -u chat.zucchini.spawner -f` (Linux); on the e2e box (`beprod`) that's the source of truth for what the spawner actually did. The install/build/ship mechanics are simpler to read in `install.sh`, `build-releases.sh`, and `uninstall.sh` than to mirror here.

## Message-frame invariant

Consumes claude's `--output-format stream-json` and writes one `messages` row per JSON frame (whole frames, not token deltas — bodies are final at insert, never grow; attachments are encoded inside the body envelope so they're frozen alongside it). Combined with append-only ordering by `seq`, `(count, last.id)` is a stable identity for a chat's message array — guards like `ChatMessagesList.ApplyFingerprint` depend on this; widen them before adding edits, deletes, or out-of-envelope attachment hydration.

## Resident lifecycle, background tasks & the prune live-task guard

The resident `claude` process lives only while a foreground turn **or** an in-process background task / Monitor is active — that's the one way a resident differs from a one-shot adapter (one process can outlive its foreground turn to host background work until it finishes). It's SIGTERM'd at the **turn boundary**: the reader emits the `running=false` edge only on a `result` frame with no live tasks (**NOT** on a bare `TaskFinished` — a monitor/task completion triggers a continuation turn that ends with its own `result`, so the edge defers to the real boundary), and `main.rs` calls `abort_agent` on that edge. There is **no** periodic idle-reaper — the old `reap_if_idle`/`reap_idle` were removed because they raced monitor-fired continuation turns and killed them mid-flight. Every new message respawns a FRESH `claude --resume <agent_session_id>` (no stdin reuse). Both `/stop` and interrupt-then-send hard-abort the process group, killing any armed monitors (`--resume` can't re-arm in-process tasks); we publish our own `interrupted` result frame in both.

**Running a Monitor / background task is perfectly fine** — it just keeps the resident warm until it completes. The only interaction to know about is with `prune-context`: applying a prune hard-restarts the resident (abort → rewrite jsonl → respawn-with-`--resume`), and the in-process task/Monitor runtime is **not** restored by `--resume`, so a prune would kill anything in flight. Therefore `prune-context` **REFUSES** when the session has live tasks (`live_tasks` non-empty) and tells the agent to wait or pass `--force`. The check reads a shared `live_sessions` directory (`agent.rs` `LiveSessions` — the same `SessionState` Arcs the reader mutates) at RPC time so the error reaches the agent synchronously (exit 1). A forced prune (or a same-instant race) names the terminated tasks in the respawn prompt so the kill is never silent.

## Scope of responsibility

**Thin spawn layer — launches `claude` with our flags and that's it.** Claude's own config (`~/.claude/settings.json`, project `.claude/settings.json`, MCP allowlist, sandbox policy) is the user's responsibility, not Zucchini's. Features that touch how claude runs (e.g. the machines-sharing sandbox toggle) flip a flag on the spawn — they don't curate config or ship policy defaults; scope-of-responsibility gaps are communicated via UI disclaimers, not Zucchini-side wrappers.

**One sanctioned exception: the spawner injects a `PostToolUse` hook via `claude --settings` (in `adapters/claude.rs` `prepare_command`).** The hook command is `"$ZUCCHINI_SPAWNER_BIN" prune-reminder-hook` (a self-contained CLI subcommand in `main.rs`, no jq/script-file dependency — Rust parses the PostToolUse payload from stdin); after any tool result whose serialized `tool_response` exceeds `PRUNE_REMINDER_MIN_TOOL_RESPONSE_BYTES` (`prune.rs`) it returns `additionalContext` nudging claude to `prune-context` the output. This reinforces `PRUNE_CONTEXT_INSTRUCTION` at the recency frontier — claude deprioritizes the one-time `--append-system-prompt` standing order (codex/gemini comply from it; claude didn't), so the reminder is re-surfaced right after each heavy result. This is config-injection, which the rule above otherwise forbids — it's sanctioned only because it reinforces *our own* prune feature, not user policy. It stays safe because `--settings` **merges** with (never replaces) the user's `~/.claude/settings.json` / project hooks (verified empirically), so user-configured hooks survive the spawn; the hook never errors or writes partial output (always exits 0). Don't extend this into curating other claude config.

## Hot reload (author's machine)

Two launchd agents — see `tmp/spawner-dev-watcher.md`. `WatchPaths` only covers `src/`, so after a `Cargo.toml` bump `touch zucchini-spawner/src/main.rs` to force a rebuild.

## Public GitHub mirror

Mirrored to `github.com/zucchini-chat/zucchini-spawner` — the only Zucchini code that runs on user machines, so the only one we open-source. Push-only via `git subtree`; remote `spawner-public` uses SSH alias `github_zucchini_admin`. Plan + supply-chain rationale (cosign-verified releases via Sigstore, dual update channel `users.spawner_source`): `tmp/verifiable-spawner.md`.

**Sync the mirror after committing spawner changes.** From monorepo root:

```
git subtree split --prefix=zucchini-spawner --branch=spawner-export && git push spawner-public spawner-export:main
```

Subtree split reads committed history only, so a dirty working tree is fine. Required before tagging a release once cosign verification ships — the public commit at the tag is what the Sigstore signature binds to; without the sync, the release can't be reproduced or verified.
