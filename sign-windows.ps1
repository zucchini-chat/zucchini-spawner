#Requires -Version 5.1
<#
  Zucchini Windows code-signing helper — Azure Trusted Signing.

  Dot-source it to get the Invoke-ZucchiniSign function, or run it directly to
  sign one or more files:
      .\sign-windows.ps1 path\to\a.exe path\to\b.exe

  Configure Azure Trusted Signing with env vars:
      ZUCCHINI_AZURE_SIGN_ENDPOINT   e.g. https://eus.codesigning.azure.net/
      ZUCCHINI_AZURE_SIGN_ACCOUNT    Trusted Signing account name
      ZUCCHINI_AZURE_SIGN_PROFILE    certificate profile name
  Azure auth uses DefaultAzureCredential: `az login`, or the
  AZURE_CLIENT_ID / AZURE_TENANT_ID / AZURE_CLIENT_SECRET service-principal
  vars, or a managed identity.

  Optional:
      ZUCCHINI_AZURE_SIGN_DLIB       Azure.CodeSigning.Dlib.dll path (auto-found
                                     from the Microsoft.Trusted.Signing.Client
                                     NuGet package if unset)
      ZUCCHINI_AZURE_SIGN_TIMESTAMP  RFC3161 URL (default
                                     http://timestamp.acs.microsoft.com)
      ZUCCHINI_SIGNTOOL              signtool.exe (default: newest Win SDK x64)
      ZUCCHINI_REQUIRE_SIGNING=1     make missing config a hard ERROR (releases/CI)

  Until the three ENDPOINT/ACCOUNT/PROFILE vars are set, signing is SKIPPED with
  a warning so local builds still succeed — the binaries are just unsigned
  (SmartScreen shows "unknown publisher"). Azure Trusted Signing is LIVE as of
  Aug 2026 (account hayakutech / profile zucchini / endpoint neu; the vars are
  persisted in this box's user env). NOTE the signing dlib is .NET-hosted, so the
  box needs the .NET 8 runtime — without it signtool exits 3 with no output.

  Keep in lockstep with the twin copy native-qt/sign-windows.ps1 — two copies
  exist because this zucchini-spawner/ tree is published to a standalone public
  GitHub mirror (git subtree) and must not reference files outside its own tree.
#>
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Files)

Set-StrictMode -Version Latest

function Get-ZucchiniSigntool {
    if ($env:ZUCCHINI_SIGNTOOL) {
        if (Test-Path $env:ZUCCHINI_SIGNTOOL) { return $env:ZUCCHINI_SIGNTOOL }
        throw "ZUCCHINI_SIGNTOOL set but not found: $env:ZUCCHINI_SIGNTOOL"
    }
    $roots = @("${env:ProgramFiles(x86)}\Windows Kits\10\bin", "$env:ProgramFiles\Windows Kits\10\bin")
    $best = $roots |
        Where-Object { Test-Path $_ } |
        ForEach-Object { Get-ChildItem -Path $_ -Recurse -Filter signtool.exe -ErrorAction SilentlyContinue } |
        Where-Object { $_.FullName -match '\\x64\\signtool\.exe$' } |
        Sort-Object FullName -Descending |
        Select-Object -First 1
    if (-not $best) { throw "signtool.exe not found; set ZUCCHINI_SIGNTOOL to its path." }
    return $best.FullName
}

function Resolve-ZucchiniDlib {
    if ($env:ZUCCHINI_AZURE_SIGN_DLIB) {
        if (Test-Path $env:ZUCCHINI_AZURE_SIGN_DLIB) { return $env:ZUCCHINI_AZURE_SIGN_DLIB }
        throw "ZUCCHINI_AZURE_SIGN_DLIB set but not found: $env:ZUCCHINI_AZURE_SIGN_DLIB"
    }
    $roots = @(
        (Join-Path $env:USERPROFILE '.nuget\packages\microsoft.trusted.signing.client'),
        (Join-Path $env:LOCALAPPDATA 'Microsoft\TrustedSigning')
    )
    foreach ($root in $roots) {
        if (Test-Path $root) {
            $hit = Get-ChildItem -Path $root -Recurse -Filter 'Azure.CodeSigning.Dlib.dll' -ErrorAction SilentlyContinue |
                Sort-Object FullName -Descending | Select-Object -First 1
            if ($hit) { return $hit.FullName }
        }
    }
    return $null
}

function Invoke-ZucchiniSign {
    param([Parameter(Mandatory = $true)][string[]]$Files)

    $endpoint = $env:ZUCCHINI_AZURE_SIGN_ENDPOINT
    $account = $env:ZUCCHINI_AZURE_SIGN_ACCOUNT
    $certProfile = $env:ZUCCHINI_AZURE_SIGN_PROFILE
    $require = ($env:ZUCCHINI_REQUIRE_SIGNING -eq '1')

    if (-not $endpoint -or -not $account -or -not $certProfile) {
        $msg = 'Azure Trusted Signing not configured (set ZUCCHINI_AZURE_SIGN_ENDPOINT / _ACCOUNT / _PROFILE).'
        if ($require) { throw "$msg  (ZUCCHINI_REQUIRE_SIGNING=1 refuses to ship unsigned.)" }
        Write-Warning "$msg`n         Skipping signing — output is UNSIGNED (SmartScreen 'unknown publisher')."
        return
    }

    $dlib = Resolve-ZucchiniDlib
    if (-not $dlib) {
        throw "Azure.CodeSigning.Dlib.dll not found. Install the NuGet package 'Microsoft.Trusted.Signing.Client' (or set ZUCCHINI_AZURE_SIGN_DLIB)."
    }
    $signtool = Get-ZucchiniSigntool
    $timestamp = if ($env:ZUCCHINI_AZURE_SIGN_TIMESTAMP) { $env:ZUCCHINI_AZURE_SIGN_TIMESTAMP } else { 'http://timestamp.acs.microsoft.com' }

    $meta = [ordered]@{
        Endpoint               = $endpoint
        CodeSigningAccountName = $account
        CertificateProfileName = $certProfile
    } | ConvertTo-Json
    $metaPath = Join-Path ([System.IO.Path]::GetTempPath()) "zucchini-acs-$PID.json"
    Set-Content -LiteralPath $metaPath -Value $meta -Encoding ascii
    try {
        foreach ($f in $Files) {
            if (-not (Test-Path $f)) { throw "sign: file not found: $f" }
            Write-Host "==> Signing $f (Azure Trusted Signing)"
            & $signtool sign /v /fd SHA256 /tr $timestamp /td SHA256 /dlib $dlib /dmdf $metaPath $f
            if ($LASTEXITCODE -ne 0) { throw "signtool failed (exit $LASTEXITCODE) for $f" }
        }
    }
    finally {
        Remove-Item -LiteralPath $metaPath -Force -ErrorAction SilentlyContinue
    }
}

# Run directly (not dot-sourced) with file arguments → sign them.
if ($MyInvocation.InvocationName -ne '.' -and $Files) {
    Invoke-ZucchiniSign -Files $Files
}
