#!/bin/sh
# Build the spawner and stage it where the running spawner's updater will pick
# it up. The running spawner polls `file://~/.zucchini-spawner/dev-update/` via
# its updater loop (see the local launchd plist for env), drains in-flight
# writes, then mv's the new binary into place. No signals, no live-binary
# truncation — launchd respawns after the exec exits.
set -eu

cd "$(dirname "$0")"

DEST="${HOME}/.zucchini-spawner/dev-update"
mkdir -p "$DEST"

# Map Rust's arch triple to the install.sh naming (darwin-arm64 vs darwin-aarch64).
OS_NAME="$(uname -s)"
case "$OS_NAME" in
  Darwin) PLATFORM_OS="darwin" ;;
  Linux)  PLATFORM_OS="linux" ;;
  *) echo "Unsupported OS: $OS_NAME" >&2; exit 1 ;;
esac

HOST_ARCH="$(uname -m)"
case "$PLATFORM_OS:$HOST_ARCH" in
  darwin:arm64|darwin:aarch64)   RUST_TARGET="aarch64-apple-darwin"; BIN_SUFFIX="darwin-arm64" ;;
  darwin:x86_64)                 RUST_TARGET="x86_64-apple-darwin";  BIN_SUFFIX="darwin-x86_64" ;;
  linux:aarch64|linux:arm64)     RUST_TARGET="aarch64-unknown-linux-gnu"; BIN_SUFFIX="linux-aarch64" ;;
  linux:x86_64|linux:amd64)      RUST_TARGET="x86_64-unknown-linux-gnu";  BIN_SUFFIX="linux-x86_64" ;;
  *) echo "Unsupported host: $PLATFORM_OS/$HOST_ARCH" >&2; exit 1 ;;
esac

BIN_NAME="zucchini-spawner-${BIN_SUFFIX}"

cargo build --release --target "$RUST_TARGET"

install -m 755 "target/${RUST_TARGET}/release/zucchini-spawner" "$DEST/${BIN_NAME}.new"
xattr -c "$DEST/${BIN_NAME}.new" 2>/dev/null || true

# Codesign darwin builds with Developer ID + hardened runtime so TCC's
# designated requirement (identifier + team ID) stays stable across rebuilds.
# Cargo's default ad-hoc signing has a CDHash-only DR — every rebuild shifts
# the hash and macOS re-prompts for folder access on each auto-update.
if [ "$PLATFORM_OS" = "darwin" ] && command -v codesign >/dev/null 2>&1; then
  codesign --sign "Developer ID Application: Hayaku Tech Limited (UGHY643XCA)" \
    --options runtime --timestamp --force "$DEST/${BIN_NAME}.new"
fi

mv -f "$DEST/${BIN_NAME}.new" "$DEST/${BIN_NAME}"

# Monotonic per-build version — the spawner compares against last-applied
# (persisted across restarts) so each rebuild triggers exactly one update.
VERSION="$(awk -F'"' '/^version =/ {print $2; exit}' Cargo.toml)-dev.$(date +%s)"
printf '%s\n' "$VERSION" > "$DEST/zucchini-spawner-version.txt"

echo ">>> Staged $VERSION ($BIN_NAME) in $DEST"
