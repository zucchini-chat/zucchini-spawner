#!/bin/sh
set -eu

# Zucchini spawner uninstaller. Removes the launchd/systemd service, binary,
# config, and (optionally) the user-level key. Two modes:
#
# - User-driven: ./uninstall.sh — keeps ~/.zucchini-spawner/key.
# - Spawner self-uninstall (DELETE /api/account): REMOVE_KEY=1 SILENT=1
#   ./uninstall.sh — wipes the key too. Both paths run as the user; user-scope
#   systemd / per-user launchd need no sudo.
#
# Usage: ./uninstall.sh

INSTALL_DIR="${HOME}/.zucchini-spawner"
SERVICE_NAME="chat.zucchini.spawner"
REMOVE_KEY="${REMOVE_KEY:-0}"
SILENT="${SILENT:-0}"

if [ "$SILENT" = "1" ]; then
  exec >/dev/null 2>&1
fi

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
