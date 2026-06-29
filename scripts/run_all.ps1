#!/usr/bin/env pwsh
# STUB orchestrator (Phase 0 / S11). Phase E (W11) OWNS the final scripts/run_all.ps1 with the
# full six-stage flow (download -> panel -> train -> eval -> figures -> paper) and the
# -SkipGpu / -RunId / -Seed / -DryRun flags. Each per-stage script is currently a stub that
# exits 0, so this runs end-to-end from day one and each phase fills in its stub.
$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

Write-Host "== Phase 0: environment check =="
& $py scripts/check_env.py

Write-Host "== Phase A/B: data acquisition =="
& $py scripts/download_ntl.py
& $py scripts/download_finance.py
& $py scripts/download_macro.py

Write-Host "== Phase B: panel + splits =="
& $py scripts/build_panel.py

Write-Host "== Phase C: experiments =="
foreach ($m in @("momentum", "dlinear", "patchtst", "itransformer", "mamba")) {
    & $py scripts/run_experiment.py --model $m --task leading --H 1   # mamba auto-skips on CPU
}

Write-Host "== Phase D: evaluation + stats =="
& $py scripts/run_experiment.py --eval-all

Write-Host "ALL DONE"
