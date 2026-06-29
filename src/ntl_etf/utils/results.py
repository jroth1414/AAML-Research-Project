"""Lightweight, appendable, GPU-mergeable metrics store.

This is the AUTHORITATIVE long/tidy metrics table (DEVPLAN Appendix A.4):
``experiments/results_store.parquet`` (+ a ``.csv`` mirror), one metric value per row.
Phase E (eval/results.py) builds richer aggregation on top of this same schema;
``scripts/merge_results.py`` dedups GPU-profile rows back into it on ``config_hash``.
``status`` records the Mamba/foundation skip-and-log contract ("skipped").
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

RESULTS_STORE_PARQUET = Path("experiments/results_store.parquet")
RESULTS_STORE_CSV = Path("experiments/results_store.csv")

# Appendix A.4 superset schema (extends E1 with the columns the GPU-merge dedup needs).
RESULT_COLUMNS = [
    "run_id",
    "model",
    "variant",
    "task",
    "scope",  # ETF ticker or "POOLED"
    "fold",  # or -1 for cross-fold aggregate
    "stratum",  # all|single_region|multi_region|disruption|stable|pretrained|from_scratch
    "metric",  # mse|mae|dir_acc|dir_acc_pvalue|sharpe_gross|sharpe_net|nowcast_r2|pearson|n_obs
    "value",
    "ci_low",
    "ci_high",
    "status",  # ok|skipped|failed
    "profile",  # windows_cpu|gpu_full|...
    "seed",
    "git_sha",
    "config_hash",  # dedup key for the GPU-merge (merge_results.py)
]

# Dedup key used by scripts/merge_results.py when merging GPU-profile rows back in.
DEDUP_KEYS = [
    "run_id",
    "model",
    "variant",
    "task",
    "scope",
    "fold",
    "stratum",
    "metric",
    "config_hash",
]


def append_results(
    rows: list[dict],
    parquet_path: os.PathLike = RESULTS_STORE_PARQUET,
    csv_path: os.PathLike = RESULTS_STORE_CSV,
) -> Path:
    """Append metric rows to the results store (parquet + csv). Returns the parquet path."""
    df = pd.DataFrame(rows)
    for c in RESULT_COLUMNS:
        if c not in df.columns:
            df[c] = None
    df = df[RESULT_COLUMNS]
    pq = Path(parquet_path)
    pq.parent.mkdir(parents=True, exist_ok=True)
    if pq.exists():
        df = pd.concat([pd.read_parquet(pq), df], ignore_index=True)
    df.to_parquet(pq, index=False)
    df.to_csv(csv_path, index=False)
    return pq


def read_results(parquet_path: os.PathLike = RESULTS_STORE_PARQUET) -> pd.DataFrame:
    return pd.read_parquet(parquet_path)
