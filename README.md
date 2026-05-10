# zucchini-spawner

The Rust daemon that runs Claude Code agents on a developer's machine in response to messages from the [Zucchini Chat](https://zucchini.chat) iOS app.

## How it fits in

Zucchini is a chat-style remote control for coding agents. The iOS app talks to a backend (PowerSync + Postgres + a Rust auth service); the backend syncs end-to-end-encrypted messages to the spawner running on the machine that hosts the chat's project. The spawner decrypts the message, forks `claude` in the project's working directory, and streams the response back as encrypted message rows.

```
iOS app  ──►  backend  ──►  spawner (this repo)  ──►  claude CLI
                                       ▲
                                       └─ runs on your Mac or Linux box
```

The iOS app, the backend, and the website are closed-source. The spawner is the only Zucchini code that runs on your machine — and it's open here.

## Why this repo is open

The spawner has access to your Claude credentials, your source trees, your ssh keys, and the user encryption key (`K_user`) that decrypts your message bodies. The most consequential trust decision in the Zucchini stack is "what code is this binary actually running."

This repo is the source of truth. The plan is for releases to be built and signed entirely inside GitHub Actions using [Sigstore](https://www.sigstore.dev/) keyless OIDC signing — the signing certificate bound to the public workflow file at the tagged commit, recorded in the [Rekor](https://docs.sigstore.dev/logging/overview/) public transparency log. The spawner's autoupdater (`src/updater.rs`) will verify the signature against the pinned workflow identity before installing, so a release that wasn't produced by this exact public workflow is rejected. See [Status](#status) for what's shipped vs. planned.

## Platforms

| OS | Architectures |
|---|---|
| macOS | arm64, x86_64 |
| Linux | x86_64, aarch64 |

Windows is not currently planned. It would depend on Claude Code's native Windows support, which is outside this project's control.

## Building

```sh
cargo build --release
```

Default target is your host. Released binaries are built per-target on GitHub-hosted runners.

## Threat model — current state

Honest snapshot of what the spawner does and doesn't defend against today:

- **K_user transfer at install:** the install command copied from the iOS app currently carries `K_user` as an argument. This leaks momentarily through clipboard, shell history, and process arguments. **Planned:** SAS-verified ECDH pairing — the install flow displays a 9-digit verification code on both the spawner host and a device that already has `K_user`; the human compares the codes; `K_user` transfers wrapped to an ephemeral X25519 key. Server sees only ciphertext.
- **K_user at rest:** stored as a 0600 file at `~/.zucchini-spawner/key`. Same posture as `~/.ssh/id_ed25519`, kubeconfig, `.npmrc` tokens.
- **K_user at runtime:** decrypted in process memory while the spawner is running. A malicious binary that the spawner trusts to install would have full access — which is why the cosign verification (planned) is the load-bearing supply-chain mitigation.
- **Backend operator:** sees ciphertext and metadata only; cannot read message bodies. **Today**, can ship a malicious spawner update (the autoupdater currently does no signature verification on the binary it downloads). **After cosign verification ships**, can't ship code to your machine if you're on the `public` update channel. A separate `internal` channel exists for development and the project's e2e test, opt-in per user.

## Status

This repo is being opened in stages. Current state:

- ✅ Source mirrored to this public repo
- ⏳ Sigstore-signed release pipeline (GitHub Actions `release.yml`)
- ⏳ Updater signature verification (`src/updater.rs`)
- ⏳ Per-user channel selection (`public` vs `internal`)
- ⏳ SAS-verified pairing flow

## Contributing

Bug reports and patches welcome. The spawner is small (~3k lines of Rust); no formal contribution guide yet — open an issue or PR. Note that this repo is a one-way mirror of a directory in a private monorepo; PRs are reviewed here and cherry-picked back upstream.

## License

MIT — see [LICENSE](LICENSE).
