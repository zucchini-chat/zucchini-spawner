#!/bin/sh
set -eu

# Zucchini spawner uninstaller.
# Removes the launchd/systemd service, binary, config, and logs installed
# by install.sh. Leaves ~/.zucchini-spawner/dev_key alone (it's user-level,
# shared with the apps).
#
# Usage: ./uninstall.sh

INSTALL_DIR="${HOME}/.zucchini-spawner"
SERVICE_NAME="chat.zucchini.spawner"

OS="$(uname -s)"
case "$OS" in
  Darwin) PLATFORM="darwin" ;;
  Linux)  PLATFORM="linux" ;;
  *) echo "Unsupported OS: $OS" >&2; exit 1 ;;
esac

# ---------- stop + remove service ----------

if [ "$PLATFORM" = "darwin" ]; then
  PLIST_PATH="${HOME}/Library/LaunchAgents/${SERVICE_NAME}.plist"

  if launchctl print "gui/$(id -u)/${SERVICE_NAME}" >/dev/null 2>&1; then
    echo "Stopping launchd service ${SERVICE_NAME}..."
    launchctl bootout "gui/$(id -u)/${SERVICE_NAME}" 2>/dev/null || true
  else
    echo "launchd service ${SERVICE_NAME} not loaded, skipping stop."
  fi

  if [ -f "$PLIST_PATH" ]; then
    rm -f "$PLIST_PATH"
    echo "Removed ${PLIST_PATH}"
  fi

elif [ "$PLATFORM" = "linux" ]; then
  LINUX_SERVICE="zucchini-spawner-$(whoami)"
  UNIT_PATH="/etc/systemd/system/${LINUX_SERVICE}.service"

  if [ -f "$UNIT_PATH" ]; then
    echo "sudo is required to remove the systemd service."
    if sudo systemctl is-active "${LINUX_SERVICE}.service" >/dev/null 2>&1; then
      sudo systemctl stop "${LINUX_SERVICE}.service"
    fi
    if sudo systemctl is-enabled "${LINUX_SERVICE}.service" >/dev/null 2>&1; then
      sudo systemctl disable "${LINUX_SERVICE}.service" >/dev/null 2>&1 || true
    fi
    sudo rm -f "$UNIT_PATH"
    sudo systemctl daemon-reload
    echo "Removed systemd service ${LINUX_SERVICE}"
  else
    echo "systemd unit ${UNIT_PATH} not found, skipping."
  fi
fi

# ---------- remove install dir (preserving dev_key) ----------

if [ -d "$INSTALL_DIR" ]; then
  rm -rf "$INSTALL_DIR/bin"
  rm -f "$INSTALL_DIR/config.env" "$INSTALL_DIR/sync_cursor.json" "$INSTALL_DIR/spawner.log"
  # Only the user-level dev_key (shared with the apps) should remain. If nothing
  # else is left, drop the dir too so we don't leave an empty ~/.zucchini-spawner.
  if [ -z "$(ls -A "$INSTALL_DIR" 2>/dev/null)" ]; then
    rmdir "$INSTALL_DIR"
    echo "Removed ${INSTALL_DIR}"
  else
    echo "Cleaned ${INSTALL_DIR} (kept dev_key / other user files)"
  fi
else
  echo "${INSTALL_DIR} not found, skipping."
fi

echo ""
echo "Done. zucchini-spawner has been uninstalled."
