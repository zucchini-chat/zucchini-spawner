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
# Wipe stragglers from prior runs (esp. --native leftovers) so the scp glob in
# deploy.sh never ships a mix of fresh + stale arch binaries — but PRESERVE a
# hand-supplied zucchini-spawner-windows-x86_64. Windows is built + Authenticode-
# signed on a Windows box and copied in here; this script never rebuilds it (see
# build_target's windows branch), so the wipe must not delete it.
find "${OUTPUT_DIR}" -maxdepth 1 -type f -name 'zucchini-spawner-*' \
  ! -name 'zucchini-spawner-windows-x86_64' -delete

NATIVE_TARGET="aarch64-apple-darwin"

# install.sh / install.ps1 name darwin arm64 as "arm64", linux keeps "aarch64",
# windows is "windows-x86_64" (see updater.rs::platform_suffix).
#
# Windows (x86_64-pc-windows-msvc) is NOT built here. It's built AND Authenticode-
# signed on a Windows machine (the signing cert lives there — signing must happen
# on that box anyway), then copied into
#   releases/zucchini-spawner-windows-x86_64
# before this script runs. build_target's windows branch only VERIFIES it's
# present (warns, does not fail, if missing) and the wipe above preserves it, so
# this Mac/Linux run builds darwin + linux and ships the Windows binary alongside
# them untouched. On the Windows box, ONE script does build + sign + stage (+
# optional scp to this Mac): zucchini-spawner/build-windows.ps1 (signing via
# Azure Trusted Signing, sign-windows.ps1).
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

  # Windows is not built on this box — it's built + Authenticode-signed on a
  # Windows machine and hand-copied into releases/ (see the ALL_TARGETS note).
  # Verify it's present; warn (don't fail) if not, so a Mac-only run still ships
  # darwin + linux. The wipe above already preserved any existing copy.
  if [[ "$rust_target" == *"-windows-"* ]]; then
    if [[ -f "${OUTPUT_DIR}/${output_name}" ]]; then
      echo "Using prebuilt ${output_name} (built + signed on Windows)"
    else
      echo "WARNING: ${OUTPUT_DIR}/${output_name} not found — Windows binary will NOT ship." >&2
      echo "         Build + Authenticode-sign it on a Windows box, copy it there, then re-run." >&2
    fi
    return 0
  fi

  echo "Building ${output_name} (${rust_target})..."

  if [[ "$rust_target" == *"-apple-darwin" ]]; then
    if ! rustup target list --installed 2>/dev/null | grep -q "^${rust_target}$"; then
      echo "  Installing rust target ${rust_target}..."
      rustup target add "$rust_target" >/dev/null
    fi
    cargo build --release --target "$rust_target" --manifest-path "${SCRIPT_DIR}/Cargo.toml" --quiet
  else
    if ! command -v cross >/dev/null; then
      echo "ERROR: 'cross' not found. Install with:" >&2
      echo "  cargo install cross --git https://github.com/cross-rs/cross" >&2
      exit 1
    fi
    cross build --release --target "$rust_target" --manifest-path "${SCRIPT_DIR}/Cargo.toml" --quiet
  fi

  cp "${SCRIPT_DIR}/target/${rust_target}/release/zucchini-spawner" "${OUTPUT_DIR}/${output_name}"
  echo "  -> ${OUTPUT_DIR}/${output_name}"
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
