#Requires -Version 5.1
<#
  Build + sign + stage the Windows zucchini-spawner release binary.

  Runs on a Windows box. Produces releases\zucchini-spawner-windows-x86_64 — the
  EXACT artifact name deploy.sh and install.ps1 expect (see build-releases.sh) —
  Authenticode-signed via Azure Trusted Signing (sign-windows.ps1). Optionally
  scp's it to the Mac that runs backend-powersync/deploy.sh; that script builds
  the darwin/linux binaries and ships this Windows one alongside them untouched
  (build-releases.sh preserves a hand-supplied windows binary).

  Usage:
    .\build-windows.ps1              # build + sign + stage into releases\
    .\build-windows.ps1 -Push        # also scp to $env:ZUCCHINI_MAC_RELEASES_DEST
    .\build-windows.ps1 -SkipSign    # skip the signing step (still stages)

  Env:
    ZUCCHINI_MAC_RELEASES_DEST   scp destination DIRECTORY for -Push, e.g.
                                 user@mac:/Users/ivan/src/zucchini-chat/zucchini-spawner/releases/
    (signing vars: see sign-windows.ps1)
#>
param([switch]$Push, [switch]$SkipSign)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Root = $PSScriptRoot
$Target = 'x86_64-pc-windows-msvc'
$ArtifactName = 'zucchini-spawner-windows-x86_64'
$ReleasesDir = Join-Path $Root 'releases'

# cargo lives under ~\.cargo\bin and may not be on a fresh shell's PATH.
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    $cargoBin = Join-Path $env:USERPROFILE '.cargo\bin'
    if (Test-Path (Join-Path $cargoBin 'cargo.exe')) { $env:PATH = "$cargoBin;$env:PATH" }
}
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    throw 'cargo not found. Install Rust (rustup) or add ~\.cargo\bin to PATH.'
}

Write-Host "==> cargo build --release --target $Target"
Push-Location $Root
try {
    cargo build --release --target $Target
    if ($LASTEXITCODE -ne 0) { throw "cargo build failed (exit $LASTEXITCODE)" }
}
finally { Pop-Location }

$builtExe = Join-Path $Root "target\$Target\release\zucchini-spawner.exe"
if (-not (Test-Path $builtExe)) { throw "build reported success but $builtExe is missing" }

New-Item -ItemType Directory -Force -Path $ReleasesDir | Out-Null
$staged = Join-Path $ReleasesDir $ArtifactName
Copy-Item -LiteralPath $builtExe -Destination $staged -Force

if (-not $SkipSign) {
    . (Join-Path $Root 'sign-windows.ps1')
    Invoke-ZucchiniSign -Files @($staged)
}

$verMatch = Select-String -Path (Join-Path $Root 'Cargo.toml') -Pattern '^version\s*=\s*"([^"]+)"' | Select-Object -First 1
$ver = if ($verMatch) { $verMatch.Matches[0].Groups[1].Value } else { '?' }
Write-Host ''
Write-Host "==> Staged $ArtifactName (v$ver):"
Write-Host "    $staged"

if ($Push) {
    $dest = $env:ZUCCHINI_MAC_RELEASES_DEST
    if (-not $dest) {
        throw "-Push needs ZUCCHINI_MAC_RELEASES_DEST (scp dest DIR, e.g. user@mac:/…/zucchini-spawner/releases/)"
    }
    Write-Host "==> scp -> $dest"
    scp $staged $dest
    if ($LASTEXITCODE -ne 0) { throw "scp failed (exit $LASTEXITCODE)" }
    Write-Host '    Sent. On the Mac, run backend-powersync/deploy.sh (it builds darwin/linux and ships this too).'
}
else {
    Write-Host "    Next: copy it to <mac>:.../zucchini-spawner/releases/$ArtifactName"
    Write-Host '    (or re-run with -Push after setting ZUCCHINI_MAC_RELEASES_DEST).'
}
