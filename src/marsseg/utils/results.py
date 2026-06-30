"""Lightweight, appendable, long/tidy metrics store.

The canonical metrics table ``experiments/results_store.parquet`` (+ a ``.csv`` mirror), one metric
value per row. The evaluation aggregation builds on this same schema; ``scripts/merge_results.py``
dedups rows produced on a GPU profile back into it on ``config_hash``. ``status`` records the
skip-and-log contract for gated models ("skipped").
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

RESULTS_STORE_PARQUET = Path("experiments/results_store.parquet")
RESULTS_STORE_CSV = Path("experiments/results_store.csv")

# One metric value per row (long/tidy). Extra columns let GPU-profile rows merge back cleanly.
RESULT_COLUMNS = [
    "run_id",
    "model",  # baseline|unet|deeplabv3plus|segformer|dinov3_sat|sam
    "backbone",  # resnet34|efficientnet-b0|mit-b0|vitl16-sat493m|...
    "variant",  # pretrained|scratch|zeroshot|finetuned
    "scope",  # 'ALL' (overall) or a class name (per-class)
    "stratum",  # all|per_class|in_rover|cross_rover|pretrained|scratch
    "metric",  # miou|iou|pixel_acc|boundary_f1|n
    "value",
    "ci_low",
    "ci_high",
    "status",  # ok|skipped|failed
    "profile",  # windows_cpu|gpu_full
    "seed",
    "git_sha",
    "config_hash",  # dedup key for merging GPU-profile rows
]

# Dedup key used by scripts/merge_results.py when merging GPU-profile rows back in.
DEDUP_KEYS = [
    "run_id",
    "model",
    "backbone",
    "variant",
    "scope",
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
