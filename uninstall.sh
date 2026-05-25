#!/bin/sh
set -eu

# Zucchini spawner uninstaller. Removes the launchd/systemd service, binary,
# config, and (optionally) the per-user K_user keys. Two modes:
#
# - User-driven: ./uninstall.sh — keeps the owner's per-user K_user files
#   (`key_<user_id>` and the legacy `key`) so the owner doesn't have to
#   re-paste / re-wrap on re-install. Everything else is purged: spawner
#   identity (`x25519_secret`), enrollment metadata (`config.env`, `state.json`),
#   sync cursor, logs. Note for SHARED machines: wiping `x25519_secret` means
#   every previously-invited member's `machine_users.sealed_blob` (which iOS
#   sealed against the old spawner pubkey) becomes undecryptable — they need
#   re-invite. That's fine in practice: uninstall also wipes `config.env`, so
#   re-install enrolls as a NEW `machine_id` with fresh server-side
#   `machine_users` rows anyway, and the old sealed_blobs are orphaned with
#   the old machine row regardless.
# - Spawner self-uninstall (DELETE /api/account): REMOVE_KEY=1 SILENT=1
#   ./uninstall.sh — wipes everything including the keys. Both paths run as
#   the user; user-scope systemd / per-user launchd need no sudo.
#
# Usage: ./uninstall.sh

INSTALL_DIR="${HOME}/.zucchini-spawner"
SERVICE_NAME="chat.zucchini.spawner"
REMOVE_KEY="${REMOVE_KEY:-0}"
SILENT="${SILENT:-0}"

if [ "$SILENT" = "1" ]; then
  exec >/dev/null 2>&1
fi

# Mirror of backend-powersync/backend/static/install.sh OS case — if you add/remove a case here, update the
# other file. Can't share via `source`: install.sh runs before any spawner
# files exist on the user's box, and uninstall.sh must work even when the
# spawner binary is broken or missing.
OS="$(uname -s)"
case "$OS" in
  Darwin) PLATFORM="darwin" ;;
  Linux)  PLATFORM="linux" ;;
  *) echo "Unsupported OS: $OS" >&2; exit 1 ;;
esac

# ---------- stop + remove service ----------

if [ "$PLATFORM" = "darwin" ]; then
  PLIST_PATH="${HOME}/Library/LaunchAgents/${SERVICE_NAME}.plist"
  echo "Removing launchd service ${SERVICE_NAME}..."
  launchctl bootout "gui/$(id -u)/${SERVICE_NAME}" 2>/dev/null || true
  rm -f "$PLIST_PATH"

elif [ "$PLATFORM" = "linux" ]; then
  USER_UNIT_PATH="${HOME}/.config/systemd/user/${SERVICE_NAME}.service"
  echo "Removing systemd --user service ${SERVICE_NAME}..."
  systemctl --user stop "${SERVICE_NAME}.service" 2>/dev/null || true
  systemctl --user disable "${SERVICE_NAME}.service" >/dev/null 2>&1 || true
  rm -f "$USER_UNIT_PATH"
  systemctl --user daemon-reload 2>/dev/null || true
  systemctl --user reset-failed "${SERVICE_NAME}.service" 2>/dev/null || true
fi

# ---------- remove install dir ----------

if [ -d "$INSTALL_DIR" ]; then
  if [ "$REMOVE_KEY" = "1" ]; then
    rm -rf "$INSTALL_DIR"
    echo "Removed ${INSTALL_DIR}"
  else
    rm -rf "$INSTALL_DIR/bin"
    rm -f "$INSTALL_DIR/config.env" "$INSTALL_DIR/sync_cursor.json" "$INSTALL_DIR/spawner.log"
    # Spawner identity (x25519_secret) is minted per-enrollment; keeping it
    # across uninstall would just leak the long-term private key. state.json
    # carries stale user_id / spawner_pubkey / project paths from the prior
    # enrollment — also worthless post-uninstall.
    rm -f "$INSTALL_DIR/x25519_secret" "$INSTALL_DIR/state.json"
    # Sweep atomic_write_private leftovers (`<name>.tmp`) so a crash
    # mid-write doesn't leave orphan half-files next to the live keys.
    # Enumerated to match the surrounding `rm -f` pattern — `key_*.tmp`
    # stays as a glob since the per-user uuid is unknown at uninstall.
    rm -f \
      "$INSTALL_DIR/config.env.tmp" \
      "$INSTALL_DIR/state.json.tmp" \
      "$INSTALL_DIR/x25519_secret.tmp"
    rm -f "$INSTALL_DIR"/key_*.tmp 2>/dev/null || true
    if [ -z "$(ls -A "$INSTALL_DIR" 2>/dev/null)" ]; then
      rmdir "$INSTALL_DIR"
      echo "Removed ${INSTALL_DIR}"
    else
      echo "Cleaned ${INSTALL_DIR} (kept key / other user files)"
    fi
  fi
else
  echo "${INSTALL_DIR} not found, skipping."
fi

echo ""
echo "Done. zucchini-spawner has been uninstalled."
