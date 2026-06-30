#!/usr/bin/env pwsh
# Experiment-matrix runner (Phase C / Task M14). Runs the CPU-feasible Tier 0-1 models (+ the
# CPU Mamba S6 fallback) over the walk-forward folds, each writing its own experiments/<run_id>/.
# GPU-only combos (official Mamba, foundation fine-tune) are skipped-and-logged on a CPU profile.
# Usage:  pwsh -File scripts/run_experiments.ps1 [-Task leading] [-H 1] [-Smoke]
param(
    [string]$Task = "leading",
    [int]$H = 1,
    [int]$Seed = 1414,
    [switch]$Smoke
)
$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

$models = @("momentum", "dlinear", "patchtst", "itransformer", "mamba", "chronos")
foreach ($m in $models) {
    Write-Host "== running $m ($Task H$H) =="
    $cmd = @("scripts/run_experiment.py", "--model", $m, "--task", $Task, "--H", $H, "--seed", $Seed)
    if ($Smoke) { $cmd += "--smoke" }
    & $py @cmd
}
Write-Host "matrix complete"
