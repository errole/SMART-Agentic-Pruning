# Runs (or resumes) a CAP-Prune experiment matrix until every run is finished.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_matrix.ps1 -Config configs\base.yaml
#
# The supervisor already restarts crashed or hung workers. This outer loop adds
# one more layer: if the supervisor process itself dies, it is started again,
# and every run resumes from its last checkpoint. Ctrl+C stops cleanly.
param(
    [string]$Config = "configs\base.yaml",
    [int]$MaxLoops = 50,
    [int]$PauseSeconds = 60
)
$ErrorActionPreference = "Continue"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
if (Test-Path "$root\.venv\Scripts\Activate.ps1") { & "$root\.venv\Scripts\Activate.ps1" }

for ($i = 1; $i -le $MaxLoops; $i++) {
    Write-Host "[$(Get-Date -Format s)] supervisor pass $i for $Config"
    python -m capprune supervise --config $Config
    $code = $LASTEXITCODE
    if ($code -eq 0)  { Write-Host "All runs complete. Next: python -m capprune report --config $Config"; exit 0 }
    if ($code -eq 75) { Write-Host "Stopped by user; re-run this script to continue."; exit 75 }
    if ($code -eq 77) { Write-Host "Another supervisor is already running this matrix."; exit 77 }
    Write-Host "Supervisor exited with code $code; restarting in $PauseSeconds s (runs resume from checkpoints)."
    Start-Sleep -Seconds $PauseSeconds
}
exit 1
