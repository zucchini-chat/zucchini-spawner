#!/bin/sh
set -eu

# Zucchini spawner uninstaller. Removes the launchd/systemd service, binary,
# config, and (optionally) the user-level key. Two modes:
#
# - User-driven: ./uninstall.sh — interactive sudo, keeps ~/.zucchini-spawner/key.
# - Spawner self-uninstall (DELETE /api/account): SUDO_CMD="sudo -n" REMOVE_KEY=1
#   PRE_SLEEP=2 SILENT=1 ./uninstall.sh — non-interactive, wipes the key too,
#   and sleeps first so the parent spawner exits before service-manager teardown
#   (otherwise launchd/systemd would just respawn the binary we're trying to remove).
#
# Usage: ./uninstall.sh

INSTALL_DIR="${HOME}/.zucchini-spawner"
SERVICE_NAME="chat.zucchini.spawner"
SUDO_CMD="${SUDO_CMD:-sudo}"
REMOVE_KEY="${REMOVE_KEY:-0}"
PRE_SLEEP="${PRE_SLEEP:-0}"
SILENT="${SILENT:-0}"

if [ "$SILENT" = "1" ]; then
  exec >/dev/null 2>&1
fi

if [ "$PRE_SLEEP" -gt 0 ]; then
  sleep "$PRE_SLEEP"
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
  LINUX_SERVICE="${SERVICE_NAME}-$(whoami)"
  UNIT_PATH="/etc/systemd/system/${LINUX_SERVICE}.service"

  if [ -f "$UNIT_PATH" ]; then
    if $SUDO_CMD true 2>/dev/null; then
      echo "Removing systemd service ${LINUX_SERVICE}..."
      $SUDO_CMD systemctl stop "${LINUX_SERVICE}.service" 2>/dev/null || true
      $SUDO_CMD systemctl disable "${LINUX_SERVICE}.service" 2>/dev/null || true
      $SUDO_CMD rm -f "$UNIT_PATH"
      $SUDO_CMD systemctl daemon-reload 2>/dev/null || true
    else
      echo "sudo not available; skipping systemd teardown (unit will fail to restart)" >&2
    fi
  else
    echo "systemd unit ${UNIT_PATH} not found, skipping."
  fi
fi

# ---------- remove install dir ----------

if [ -d "$INSTALL_DIR" ]; then
  if [ "$REMOVE_KEY" = "1" ]; then
    rm -rf "$INSTALL_DIR"
    echo "Removed ${INSTALL_DIR}"
  else
    rm -rf "$INSTALL_DIR/bin"
    rm -f "$INSTALL_DIR/config.env" "$INSTALL_DIR/sync_cursor.json" "$INSTALL_DIR/spawner.log"
    # User-level key (shared with apps) is left in place. Drop the dir if empty.
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
