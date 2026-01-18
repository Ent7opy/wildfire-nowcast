param(
  [Parameter(Position = 0, Mandatory = $true)]
  [ValidateSet("init", "plan", "run", "status")]
  [string]$Command,

  [string]$TaskFile
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptPath = $MyInvocation.MyCommand.Definition
if (-not $scriptPath) {
    $scriptPath = Resolve-Path $MyInvocation.MyCommand.Path
}
$scriptDir = Split-Path $scriptPath -Parent
$binDir = Join-Path $scriptDir "bin"

if (Get-Command "jq" -ErrorAction SilentlyContinue) {
    try {
        jq --version | Out-Null
    } catch {
        Write-Host "jq found but broken"
    }
}

if (-not (Test-Path $binDir)) {
    New-Item -ItemType Directory -Path $binDir | Out-Null
}

$jqPath = Join-Path $binDir "jq.exe"
if (-not (Test-Path $jqPath)) {
    Write-Host "jq not found. Downloading..."
    $url = "https://github.com/jqlang/jq/releases/download/jq-1.7.1/jq-windows-amd64.exe"
    try {
        $wc = New-Object System.Net.WebClient
        $wc.DownloadFile($url, $jqPath)
        Write-Host "Installed jq to $jqPath"
    }
    catch {
        Write-Error "Failed to download jq: $_"
        exit 1
    }
}

$env:PATH = "$binDir;" + $env:PATH

if ($TaskFile) {
  $env:RALPH_TASK_FILE = $TaskFile
}

if (-not $env:RALPH_MODEL) {
  $env:RALPH_MODEL = "auto"
}

$bash = $env:RALPH_BASH
if (-not $bash) {
  $bash = (Get-Command bash -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue)
}

if ($bash -and ($bash -match '\\Windows\\System32\\bash\.exe$')) {
  $bash = $null
}

if (-not $bash) {
  $pf = $env:ProgramFiles
  if ((-not $pf) -or ($pf -match '^/')) { $pf = [Environment]::GetFolderPath("ProgramFiles") }
  $pf86 = ${env:ProgramFiles(x86)}
  if ((-not $pf86) -or ($pf86 -match '^/')) { $pf86 = [Environment]::GetFolderPath("ProgramFilesX86") }

  $candidates = @(
    (Join-Path $pf "Git\bin\bash.exe"),
    (Join-Path $pf "Git\usr\bin\bash.exe"),
    (Join-Path $pf86 "Git\bin\bash.exe"),
    (Join-Path $pf86 "Git\usr\bin\bash.exe")
  ) | Where-Object { $_ -and (Test-Path $_) }

  if (@($candidates).Count -gt 0) {
    $bash = @($candidates)[0]
  }
}

if (-not $bash) {
  $git = (Get-Command git -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue)
  if ($git) {
    $gitDir = Split-Path $git -Parent
    $gitRoot = Split-Path $gitDir -Parent
    $gitCandidates = @(
      (Join-Path $gitRoot "bin\bash.exe"),
      (Join-Path $gitRoot "usr\bin\bash.exe")
    ) | Where-Object { $_ -and (Test-Path $_) }

    if (@($gitCandidates).Count -gt 0) {
      $bash = @($gitCandidates)[0]
    }
  }
}

if (-not $bash) {
  Write-Error "bash not found. Install Git for Windows."
  exit 1
}

& $bash ".ralph/ralph.sh" $Command
exit $LASTEXITCODE
