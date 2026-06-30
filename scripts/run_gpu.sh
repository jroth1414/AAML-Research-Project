#!/usr/bin/env bash
# V100 / CUDA-Linux GPU tier (Profile 2): runs the experiments that are deferred on Windows/CPU —
# the OFFICIAL Mamba fused kernel (un-defers H4) and GPU foundation references. On a Volta (V100)
# + Linux box this "just works" (no Blackwell bleeding-edge); mamba-ssm/causal-conv1d build with
# standard CUDA wheels.
#
# Prereqs on the GPU box:
#   1. core env:  python -m venv .venv && source .venv/bin/activate
#                 pip install -r requirements.txt && pip install -e .
#   2. processed data present (small): copy data/processed/*.parquet + data/processed/series_registry.parquet
#      from the main box, OR rebuild with creds: python scripts/build_panel.py --config configs/panel.yaml
#
# Then:  bash scripts/run_gpu.sh
# Afterwards, copy experiments/mamba__*/ (and any foundation runs) back to the main box and re-run
# scripts/analyze_results.py — H4 is then DECIDED on the official kernel and the stores update.
set -euo pipefail
py="${PYTHON:-python}"

echo "== installing GPU extras (mamba-ssm + causal-conv1d build on CUDA+Linux) =="
$py -m pip install -r requirements-extras.txt --no-build-isolation || \
  $py -m pip install -r requirements-extras.txt

echo "== capability check (expect profile=gpu_full, cuda=True, mamba_ssm=True) =="
$py scripts/check_env.py

echo "== OFFICIAL Mamba kernel (manifest mamba_impl=official) — leading H=1, multi-seed =="
for s in 1414 1415 1416; do
  $py scripts/run_experiment.py --model mamba --task leading --H 1 --seed "$s"
done

echo "== foundation references (GPU-accelerated zero-shot; fine-tune is future work) =="
for m in chronos moirai timesfm; do
  $py scripts/run_experiment.py --model "$m" --task leading --H 1 --variant zeroshot || true
done

echo
echo "DONE. Next: copy experiments/{mamba,moirai,timesfm}__*/ back to the main box, then run:"
echo "  python scripts/merge_results.py        # fold GPU result rows into experiments/results_store.parquet"
echo "  python scripts/analyze_results.py --experiments-dir experiments"
echo "  -> H4 verdict is now decided on the official kernel (no longer 'deferred')."
