"""STUB — implemented in Phase C (docs/devplan/04-models-and-training.md, tasks M13) and the
eval driver in Phase D. Parses its CLI, logs a TODO, exits 0 so scripts/run_all runs end-to-end.

Honors the capability gate: a request for a GPU-only model on a CPU profile logs a skip and
still exits 0 (the skip-and-log contract), so the stub already models the real behavior.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ntl_etf.utils.capabilities import detect  # noqa: E402
from ntl_etf.utils.logging import get_logger  # noqa: E402

log = get_logger("run_experiment")

GPU_ONLY_MODELS = {"mamba"}  # official kernel; foundation finetune handled in Phase C/M


def main() -> int:
    ap = argparse.ArgumentParser(description="Run one experiment (STUB).")
    ap.add_argument("--model", default="momentum")
    ap.add_argument("--task", default="leading")
    ap.add_argument("--H", type=int, default=1)
    ap.add_argument("--config", default=None)
    ap.add_argument("--variant", default="scratch")
    ap.add_argument("--seed", type=int, default=1414)
    ap.add_argument("--eval-all", action="store_true")
    args, _ = ap.parse_known_args()
    caps = detect()
    if args.model in GPU_ONLY_MODELS and not caps.mamba_ssm:
        log.warning(
            "model=%s requires the official mamba-ssm/CUDA kernel; profile=%s -> skip-and-log "
            "(Phase C uses the CPU S6 fallback). Stub no-op.",
            args.model,
            caps.profile,
        )
        return 0
    log.info("STUB run_experiment: implemented in Phase C/D (M13). args=%s", vars(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
