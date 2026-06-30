#Requires -Version 5.1
$ErrorActionPreference = 'Stop'

# Zucchini spawner installer (Windows).
#
# Usage (PowerShell):
#   $env:ZUCCHINI_CODE='<enrollment-code>'; $env:ZUCCHINI_KEY='<base64-K_machine>'
#   irm https://api.zucchini.chat/install/install.ps1 | iex
#
# or invoked directly with params:
#   .\install.ps1 -Code <enrollment-code> -Key <base64-K_machine>
#
# `irm | iex` can't forward script parameters, so the canonical web flow passes
# the enrollment code + key via the ZUCCHINI_CODE / ZUCCHINI_KEY env vars; the
# -Code/-Key params are the direct-invocation convenience. This mirrors the
# `curl | sh -s -- --code … --key …` POSIX flow (install.sh).
#
# ---------------------------------------------------------------------------
# Autostart: a per-user logon Scheduled Task, NOT a Windows Service.
#
# A Windows Service runs in session 0 with the LocalSystem (or a service)
# profile — it does NOT get the interactive user's PATH or environment, where
# Git Bash, node, claude, etc. live. The spawner runs adapter commands under
# bash (Git Bash/WSL) and resolves the CLIs off the user PATH, so it must run
# in the user context. A logon-triggered Scheduled Task does exactly that: it
# launches in the interactive user's session with their environment, which is
# the direct analogue of the per-user launchd LaunchAgent (macOS) and
# `systemctl --user` unit (Linux). Trade-off vs. those: this task runs only
# while the user is logged on (no "run whether logged on or not", which would
# need stored credentials + admin) — the same practical scope as a LaunchAgent
# and as `systemctl --user` WITHOUT loginctl linger.
# ---------------------------------------------------------------------------

param(
  [string]$Code = $env:ZUCCHINI_CODE,
  [string]$Key  = $env:ZUCCHINI_KEY
)

# Keep these in lockstep with uninstall.ps1 and install.sh / uninstall.sh.
$BaseUrl     = 'https://api.zucchini.chat'
$InstallDir  = Join-Path $env:USERPROFILE '.zucchini-spawner'
$BinDir      = Join-Path $InstallDir 'bin'
$ConfigFile  = Join-Path $InstallDir 'config.env'
$ServiceName = 'chat.zucchini.spawner'

# ---------- parse + validate args ----------

if ([string]::IsNullOrEmpty($Code) -or [string]::IsNullOrEmpty($Key)) {
  Write-Error 'Error: -Code and -Key are both required (or set ZUCCHINI_CODE / ZUCCHINI_KEY).'
  exit 1
}

# Fail before mutating any autostart state.
try {
  $keyLen = [Convert]::FromBase64String($Key).Length
} catch {
  Write-Error 'Error: -Key must be valid base64 of a 32-byte K_machine.'
  exit 1
}
if ($keyLen -ne 32) {
  Write-Error "Error: -Key must be base64 of a 32-byte K_machine (decoded $keyLen bytes)."
  exit 1
}

# ---------- detect platform ----------
# Only x86_64 Windows is published today; arm64 Windows would need its own
# release artifact (see build-releases.sh) — guard so we fail loud, not with a
# silent x86 download under emulation.

$arch = $env:PROCESSOR_ARCHITECTURE
if ($arch -eq 'AMD64' -or $arch -eq 'x86') {
  $platformArch = 'x86_64'
} else {
  Write-Error "Unsupported architecture: $arch (only x86_64 Windows is published)."
  exit 1
}
$binaryName = "zucchini-spawner-windows-$platformArch"

# ---------- remove a previous install (idempotent re-run) ----------

& schtasks.exe /End /TN $ServiceName *> $null
& schtasks.exe /Delete /TN $ServiceName /F *> $null

# ---------- exchange enrollment code for spawner token ----------

Write-Host 'Enrolling machine...'
$response = Invoke-RestMethod -Method Post -Uri "$BaseUrl/auth/spawner/complete" `
  -ContentType 'application/json' -Body (@{ code = $Code } | ConvertTo-Json -Compress)

$machineId    = $response.machine_id
$userId       = $response.user_id
$spawnerToken = $response.spawner_token

if ([string]::IsNullOrEmpty($machineId) -or [string]::IsNullOrEmpty($userId) -or [string]::IsNullOrEmpty($spawnerToken)) {
  Write-Error "Error: enrollment failed. Server response: $($response | ConvertTo-Json -Compress)"
  exit 1
}
$keyFile = Join-Path $InstallDir "key_$userId"
Write-Host "Enrolled as machine $machineId"

# ---------- install binary ----------

New-Item -ItemType Directory -Force -Path $BinDir | Out-Null
$exePath = Join-Path $BinDir 'zucchini-spawner.exe'

Write-Host "Downloading $binaryName..."
$tmpBin = Join-Path $BinDir ("zucchini-spawner.tmp.{0}.exe" -f $PID)
Invoke-WebRequest -Uri "$BaseUrl/install/$binaryName" -OutFile $tmpBin
Move-Item -LiteralPath $tmpBin -Destination $exePath -Force

# ---------- write config + K_user key ----------
# config.env is the same KEY=VALUE file Linux feeds to systemd's
# EnvironmentFile. A Scheduled Task can't load an env file natively, so the
# generated launcher (below) parses it into the process environment before
# exec'ing the spawner — keeping config.env the single source of truth across
# all three platforms (the spawner reads ZUCCHINI_* purely from its env).
# Hardening to an owner-only ACL is also done by the spawner itself
# (sys::harden_file) on first run; restrict here too as defense in depth.

$config = @(
  "ZUCCHINI_MACHINE_ID=$machineId",
  "ZUCCHINI_USER_ID=$userId",
  "ZUCCHINI_SPAWNER_TOKEN=$spawnerToken"
) -join "`n"
Set-Content -LiteralPath $ConfigFile -Value $config -NoNewline -Encoding ascii
Set-Content -LiteralPath $keyFile -Value $Key -NoNewline -Encoding ascii

function Restrict-ToOwner($path) {
  # Same protected-DACL / owner-only intent as sys::harden_file (windows.rs);
  # keep the two in lockstep if the access model changes.
  # Disable inheritance, drop inherited ACEs, grant only the current user.
  & icacls.exe $path /inheritance:r *> $null
  & icacls.exe $path /grant:r "$($env:USERNAME):F" *> $null
}
Restrict-ToOwner $ConfigFile
Restrict-ToOwner $keyFile

# ---------- write the launcher (env-file loader + log redirect) ----------
# Loads config.env into the environment, then runs the spawner with stdout+stderr
# appended to spawner.log (the launchd StandardOutPath / journald analogue).
$launcherPath = Join-Path $InstallDir 'run.ps1'
$logPath = Join-Path $InstallDir 'spawner.log'
$launcher = @"
`$ErrorActionPreference = 'Stop'
Get-Content -LiteralPath '$ConfigFile' | ForEach-Object {
  if (`$_ -match '^\s*([^=#]+?)\s*=\s*(.*)$') {
    [Environment]::SetEnvironmentVariable(`$matches[1], `$matches[2], 'Process')
  }
}
& '$exePath' *>> '$logPath'
"@
Set-Content -LiteralPath $launcherPath -Value $launcher -Encoding ascii

# ---------- register the logon Scheduled Task ----------

Write-Host 'Registering logon Scheduled Task...'
$action  = New-ScheduledTaskAction -Execute 'powershell.exe' `
  -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$launcherPath`""
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" `
  -LogonType Interactive -RunLevel Highest
# Settings tuned to the launchd KeepAlive / systemd Restart=always intent:
# restart on failure, never auto-expire, allow start on battery.
$settings = New-ScheduledTaskSettingsSet `
  -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
  -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) `
  -ExecutionTimeLimit (New-TimeSpan -Seconds 0)

Register-ScheduledTask -TaskName $ServiceName -Action $action -Trigger $trigger `
  -Principal $principal -Settings $settings -Force | Out-Null

# ---------- start it now ----------

Start-ScheduledTask -TaskName $ServiceName

Write-Host ''
Write-Host "Installed as Scheduled Task $ServiceName"
Write-Host "  Logs:  $InstallDir\spawner.log  (if the spawner is configured to log to file)"
Write-Host "  Stop:  schtasks /End /TN $ServiceName"
Write-Host "  Start: schtasks /Run /TN $ServiceName"
Write-Host ''
Write-Host 'Done. The spawner is running and connected.'
