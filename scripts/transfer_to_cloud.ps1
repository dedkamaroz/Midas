<#
.SYNOPSIS
    Push everything not in git (secrets, MBO feature parquets, prior KB learnings)
    to a cloud pod where the repo has already been cloned.

.DESCRIPTION
    Companion to docs/CLOUD_HANDOVER.md §5. Run AFTER you have:
      1. Spun up the pod (RunPod GPU/CPU pod, etc.)
      2. SSH'd in and run the steps in §4 of the handover doc:
           - apt installs, venv creation, `pip install -e .[nq]`
           - `git clone https://github.com/dedkamaroz/Midas.git /workspace/Midas`
      3. Confirmed `~/.ssh/config` has a Host entry whose alias matches -RemoteHost.

    This script then transfers, idempotently:
      - .env                  (API keys; NEVER in git)
      - feature parquets      (~56 MB; 14-col 1s bars, current MNQ panel)
      - knowledge/learnings/  (optional, default ON; prior discovery learnings)

.PARAMETER RemoteHost
    SSH alias (from ~/.ssh/config) or user@host. Default: runpod-mnq

.PARAMETER RemoteRoot
    Path on the remote where the repo lives. Default: /workspace/Midas

.PARAMETER LocalFeaturesDir
    Local path to the MNQ feature parquets.

.PARAMETER SkipLearnings
    Skip transferring prior discovery learnings (saves a few seconds; fresh KB).

.EXAMPLE
    .\scripts\transfer_to_cloud.ps1

.EXAMPLE
    .\scripts\transfer_to_cloud.ps1 -RemoteHost runpod-other -SkipLearnings
#>

[CmdletBinding()]
param(
    [string]$RemoteHost      = "runpod-mnq",
    [string]$RemoteRoot      = "/workspace/Midas",
    [string]$LocalFeaturesDir = "D:\File Transfer\FX Trading\market_data\features\GLBX.MDP3\MNQ.c.0\bar_1s",
    [switch]$SkipLearnings
)

$ErrorActionPreference = "Stop"

# --- Resolve repo root from script location ---
$RepoRoot      = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$EnvFile       = Join-Path $RepoRoot ".env"
$LearningsDir  = Join-Path $RepoRoot "midas-kb-mnq\knowledge\learnings"

Write-Host ""
Write-Host "=== Midas: local -> cloud transfer ===" -ForegroundColor Cyan
Write-Host "  Repo root : $RepoRoot"
Write-Host "  Remote    : ${RemoteHost}:${RemoteRoot}"
Write-Host "  Features  : $LocalFeaturesDir"
Write-Host "  Learnings : $(if ($SkipLearnings) { 'SKIPPED' } else { $LearningsDir })"
Write-Host ""

# --- 1. SSH reachability ---
Write-Host "[1/5] Verifying SSH to $RemoteHost ..." -ForegroundColor Cyan
$probe = & ssh -o ConnectTimeout=8 -o BatchMode=yes $RemoteHost "echo ok" 2>&1
if ($LASTEXITCODE -ne 0 -or $probe -notmatch "ok") {
    Write-Error "SSH to '$RemoteHost' failed. Check ~/.ssh/config and that the pod is running.`n  Output: $probe"
    exit 1
}
Write-Host "      ok" -ForegroundColor Green

# --- 2. Local file checks ---
Write-Host "[2/5] Checking local files ..." -ForegroundColor Cyan
if (-not (Test-Path $EnvFile)) {
    Write-Error ".env not found at $EnvFile — copy one in with DATABENTO_API_KEY and ANTHROPIC_MIDAS_API_KEY before re-running."
    exit 1
}
if (-not (Test-Path $LocalFeaturesDir)) {
    Write-Error "Feature parquets not found at $LocalFeaturesDir"
    exit 1
}
$parquets = Get-ChildItem $LocalFeaturesDir -Filter "*.parquet"
if ($parquets.Count -eq 0) {
    Write-Error "No .parquet files in $LocalFeaturesDir"
    exit 1
}
$totalMB = [math]::Round((($parquets | Measure-Object Length -Sum).Sum / 1MB), 1)
Write-Host "      .env:      $EnvFile"
Write-Host "      parquets:  $($parquets.Count) files, $totalMB MB"
if (-not $SkipLearnings) {
    if (Test-Path $LearningsDir) {
        $learningCount = (Get-ChildItem $LearningsDir -Recurse -File | Measure-Object).Count
        Write-Host "      learnings: $learningCount files under $LearningsDir"
    } else {
        Write-Host "      learnings: (none locally — will skip)" -ForegroundColor Yellow
    }
}

# --- 3. Remote directory tree ---
Write-Host "[3/5] Creating remote directory tree ..." -ForegroundColor Cyan
$mkdirs = @(
    "$RemoteRoot/market_data/features/GLBX.MDP3/MNQ.c.0",
    "$RemoteRoot/midas-kb-mnq/knowledge"
) -join " "
& ssh $RemoteHost "mkdir -p $mkdirs"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Remote mkdir failed"
    exit 1
}
Write-Host "      ok" -ForegroundColor Green

# --- 4. Transfer ---
Write-Host "[4/5] Transferring ..." -ForegroundColor Cyan

# 4a. .env
Write-Host "      .env -> ${RemoteHost}:${RemoteRoot}/.env"
& scp -q $EnvFile "${RemoteHost}:${RemoteRoot}/.env"
if ($LASTEXITCODE -ne 0) { Write-Error "scp .env failed"; exit 1 }
& ssh $RemoteHost "chmod 600 $RemoteRoot/.env"

# 4b. Feature parquets (transfer the bar_1s directory itself into MNQ.c.0/)
Write-Host "      parquets -> ${RemoteHost}:${RemoteRoot}/market_data/features/GLBX.MDP3/MNQ.c.0/bar_1s/"
& scp -q -r $LocalFeaturesDir "${RemoteHost}:${RemoteRoot}/market_data/features/GLBX.MDP3/MNQ.c.0/"
if ($LASTEXITCODE -ne 0) { Write-Error "scp parquets failed"; exit 1 }

# 4c. Learnings (optional)
if (-not $SkipLearnings -and (Test-Path $LearningsDir)) {
    Write-Host "      learnings -> ${RemoteHost}:${RemoteRoot}/midas-kb-mnq/knowledge/learnings/"
    & scp -q -r $LearningsDir "${RemoteHost}:${RemoteRoot}/midas-kb-mnq/knowledge/"
    if ($LASTEXITCODE -ne 0) { Write-Warning "scp learnings failed (non-fatal — discovery will rebuild)" }
}
Write-Host "      transfer complete" -ForegroundColor Green

# --- 5. Verify remote state ---
Write-Host "[5/5] Verifying remote ..." -ForegroundColor Cyan
$verifyCmd = @"
echo --- .env ---;
ls -la $RemoteRoot/.env 2>/dev/null | awk '{print \$1, \$5, \$9}';
echo --- parquets ---;
ls $RemoteRoot/market_data/features/GLBX.MDP3/MNQ.c.0/bar_1s/ 2>/dev/null | head -3;
echo \"  (total: \$(ls $RemoteRoot/market_data/features/GLBX.MDP3/MNQ.c.0/bar_1s/ 2>/dev/null | wc -l) files)\";
echo --- learnings ---;
ls $RemoteRoot/midas-kb-mnq/knowledge/learnings/ 2>/dev/null | head -3 || echo '(none)';
"@
& ssh $RemoteHost $verifyCmd

Write-Host ""
Write-Host "Done. Next on the cloud machine:" -ForegroundColor Green
Write-Host "  cd $RemoteRoot && source .venv/bin/activate"
Write-Host "  python scripts/run_discovery.py --provider anthropic --iters 5 --goals 6"
Write-Host ""
