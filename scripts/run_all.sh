#!/usr/bin/env bash
# STUB orchestrator (Phase 0 / S11). Phase E (W11) OWNS the final scripts/run_all.sh with the
# full six-stage flow and the GPU opt-in. Mirrors run_all.ps1 for WSL2/Colab/Linux.
set -euo pipefail

py=".venv/Scripts/python.exe"
[ -x "$py" ] || py=".venv/bin/python"
[ -x "$py" ] || py="python"

"$py" scripts/check_env.py
"$py" scripts/download_ntl.py
"$py" scripts/download_finance.py
"$py" scripts/download_macro.py
"$py" scripts/build_panel.py
for m in momentum dlinear patchtst itransformer mamba; do
  "$py" scripts/run_experiment.py --model "$m" --task leading --H 1
done
"$py" scripts/run_experiment.py --eval-all
echo "ALL DONE"
