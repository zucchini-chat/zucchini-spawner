#Requires -Version 5.1
$ErrorActionPreference = 'Stop'

# Zucchini spawner uninstaller (Windows). Removes the logon Scheduled Task,
# binary, config, and (optionally) the per-user K_user keys. Mirrors the
# POSIX uninstall.sh contract exactly:
#
# - User-driven: .\uninstall.ps1 — keeps the owner's per-user K_user files
#   (`key_<user_id>` and the legacy `key`) so re-install needs no re-paste.
#   Everything else is purged (spawner identity `x25519_secret`, enrollment
#   metadata `config.env`/`state.json`, sync cursor, logs).
# - Spawner self-uninstall (DELETE /api/account): REMOVE_KEY=1 SILENT=1 — wipes
#   everything including the keys. Driven by env vars so the Rust self-uninstall
#   path (src/uninstall.rs) and the user-run path share one script, matching
#   uninstall.sh's REMOVE_KEY/SILENT contract.
#
# Autostart is a per-user logon Scheduled Task (not a Windows Service) — see
# install.ps1 header for the rationale.

# Keep these in lockstep with install.ps1 and uninstall.sh.
$InstallDir  = Join-Path $env:USERPROFILE '.zucchini-spawner'
$ServiceName = 'chat.zucchini.spawner'

$RemoveKey = ($env:REMOVE_KEY -eq '1')
$Silent    = ($env:SILENT -eq '1')

function Info($msg) {
  if (-not $Silent) { Write-Host $msg }
}

# ---------- stop + remove the scheduled task ----------

Info "Removing scheduled task ${ServiceName}..."
# `Stop-ScheduledTask` is best-effort: the running instance is THIS process'
# parent when invoked from the self-uninstall path, so it may already be gone.
& schtasks.exe /End /TN $ServiceName *> $null
& schtasks.exe /Delete /TN $ServiceName /F *> $null

# ---------- remove install dir ----------

if (Test-Path -LiteralPath $InstallDir) {
  if ($RemoveKey) {
    Remove-Item -LiteralPath $InstallDir -Recurse -Force -ErrorAction SilentlyContinue
    Info "Removed ${InstallDir}"
  } else {
    # Preserve key_* / legacy `key`; purge everything else. Same file set as
    # uninstall.sh's non-REMOVE_KEY branch.
    Remove-Item -LiteralPath (Join-Path $InstallDir 'bin') -Recurse -Force -ErrorAction SilentlyContinue
    $purge = @(
      'config.env', 'sync_cursor.json', 'spawner.log',
      'x25519_secret', 'state.json',
      'config.env.tmp', 'state.json.tmp', 'x25519_secret.tmp'
    )
    foreach ($name in $purge) {
      Remove-Item -LiteralPath (Join-Path $InstallDir $name) -Force -ErrorAction SilentlyContinue
    }
    Get-ChildItem -LiteralPath $InstallDir -Filter 'key_*.tmp' -File -ErrorAction SilentlyContinue |
      Remove-Item -Force -ErrorAction SilentlyContinue

    $remaining = @(Get-ChildItem -LiteralPath $InstallDir -Force -ErrorAction SilentlyContinue)
    if ($remaining.Count -eq 0) {
      Remove-Item -LiteralPath $InstallDir -Force -ErrorAction SilentlyContinue
      Info "Removed ${InstallDir}"
    } else {
      Info "Cleaned ${InstallDir} (kept key / other user files)"
    }
  }
} else {
  Info "${InstallDir} not found, skipping."
}

Info ''
Info 'Done. zucchini-spawner has been uninstalled.'
