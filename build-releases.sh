#!/usr/bin/env bash
set -euo pipefail

# Build zucchini-spawner binaries for all supported platforms.
#
# Native target (aarch64-apple-darwin) is built directly.
# Other targets use `cross` (Docker-based):
#   cargo install cross --git https://github.com/cross-rs/cross
#
# Usage:
#   ./build-releases.sh            # build all 4 targets
#   ./build-releases.sh --native   # build only native (fast, no Docker)
#
# Output: releases/ — binaries + zucchini-spawner-version.txt (from Cargo.toml).
# macOS binaries are codesigned with Developer ID so launchd TCC grants survive
# rebuilds (stable designated requirement = identifier + team ID).

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/releases"

NATIVE_ONLY=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --native) NATIVE_ONLY=true; shift ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "$OUTPUT_DIR"
# Wipe any stragglers from prior runs (esp. --native leftovers) so the scp glob
# in deploy.sh never ships a mix of fresh + stale arch binaries.
rm -f "${OUTPUT_DIR}"/zucchini-spawner-*

NATIVE_TARGET="aarch64-apple-darwin"

# install.sh / install.ps1 name darwin arm64 as "arm64", linux keeps "aarch64",
# windows is "windows-x86_64" (see updater.rs::platform_suffix).
#
# Windows (x86_64-pc-windows-msvc) cross-compiles from this macOS/Linux box only
# with an MSVC-compatible toolchain that `cross` does NOT provide out of the box.
# Build it from a Windows CI runner, or cross with `cargo-xwin`
# (`cargo install cargo-xwin`; downloads the MSVC CRT + Windows SDK headers):
#   cargo xwin build --release --target x86_64-pc-windows-msvc
# This script's generic else-branch uses `cross`, which will fail for the
# windows target — see build_target's windows special-case below.
ALL_TARGETS="
aarch64-apple-darwin:zucchini-spawner-darwin-arm64
x86_64-apple-darwin:zucchini-spawner-darwin-x86_64
aarch64-unknown-linux-gnu:zucchini-spawner-linux-aarch64
x86_64-unknown-linux-gnu:zucchini-spawner-linux-x86_64
x86_64-pc-windows-msvc:zucchini-spawner-windows-x86_64
"

build_target() {
  local rust_target="$1"
  local output_name="$2"

  echo "Building ${output_name} (${rust_target})..."

  # Windows MSVC artifact carries a .exe extension; the others don't.
  local artifact="zucchini-spawner"
  if [[ "$rust_target" == *"-windows-"* ]]; then
    artifact="zucchini-spawner.exe"
  fi

  if [[ "$rust_target" == *"-apple-darwin" ]]; then
    if ! rustup target list --installed 2>/dev/null | grep -q "^${rust_target}$"; then
      echo "  Installing rust target ${rust_target}..."
      rustup target add "$rust_target" >/dev/null
    fi
    cargo build --release --target "$rust_target" --manifest-path "${SCRIPT_DIR}/Cargo.toml" --quiet
  elif [[ "$rust_target" == *"-windows-msvc" ]]; then
    # MSVC target: `cross` can't build it. Use cargo-xwin, which bundles the
    # MSVC CRT + Windows SDK so it cross-links from macOS/Linux. On a Windows
    # CI runner `cargo build --target x86_64-pc-windows-msvc` works directly —
    # swap this branch for that there.
    if ! command -v cargo-xwin >/dev/null; then
      echo "ERROR: 'cargo-xwin' not found (needed for the windows MSVC target)." >&2
      echo "  Install with: cargo install cargo-xwin" >&2
      echo "  Or run this on a Windows CI runner with the native MSVC toolchain." >&2
      exit 1
    fi
    cargo xwin build --release --target "$rust_target" --manifest-path "${SCRIPT_DIR}/Cargo.toml" --quiet
  else
    if ! command -v cross >/dev/null; then
      echo "ERROR: 'cross' not found. Install with:" >&2
      echo "  cargo install cross --git https://github.com/cross-rs/cross" >&2
      exit 1
    fi
    cross build --release --target "$rust_target" --manifest-path "${SCRIPT_DIR}/Cargo.toml" --quiet
  fi

  cp "${SCRIPT_DIR}/target/${rust_target}/release/${artifact}" "${OUTPUT_DIR}/${output_name}"
  echo "  -> ${OUTPUT_DIR}/${output_name}"

  # TODO(windows): Authenticode-sign the .exe (signtool / osslsigncode) with the
  # code-signing cert before publishing, so SmartScreen doesn't flag the
  # download. The macOS codesign step below is the analogue; Windows signing is
  # NOT implemented here.
}

for entry in $ALL_TARGETS; do
  rust_target="${entry%%:*}"
  output_name="${entry##*:}"

  if [[ "$NATIVE_ONLY" == true && "$rust_target" != "$NATIVE_TARGET" ]]; then
    continue
  fi

  build_target "$rust_target" "$output_name"
done

# Sign macOS binaries with Developer ID + hardened runtime. Stable designated
# requirement (identifier + team ID) keeps TCC folder-access grants across
# rebuilds. Notarization is only needed for distribution, not launchd.
if command -v codesign >/dev/null; then
  for darwin_bin in "${OUTPUT_DIR}"/zucchini-spawner-darwin-*; do
    [[ -f "$darwin_bin" ]] || continue
    echo "Signing $(basename "$darwin_bin") with Developer ID..."
    codesign --sign "Developer ID Application: Hayaku Tech Limited (UGHY643XCA)" \
      --options runtime --timestamp --force "$darwin_bin"
  done
fi

# Version file, read by updater.rs to decide whether to self-update.
VERSION=$(awk -F'"' '/^version = /{print $2; exit}' "${SCRIPT_DIR}/Cargo.toml")
echo "$VERSION" > "${OUTPUT_DIR}/zucchini-spawner-version.txt"
echo "Wrote zucchini-spawner-version.txt (v${VERSION})"

echo ""
echo "Builds complete:"
ls -lh "$OUTPUT_DIR"/zucchini-spawner-*
