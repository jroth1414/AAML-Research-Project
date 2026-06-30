"""Prediction loader + canonical metrics store + defensive alignment audit (Phase D / E1, E2).

Reads the A.3 ``predictions.parquet`` files, validates the schema, and persists the A.4 long/tidy
metrics store ``experiments/results_store.parquet``. The alignment audit (E2) re-verifies, at
evaluation time, that no leakage slipped through (month-end target dates, no NaN, and — when the run
manifest is present — ``scaler_fit_on == 'train'``).
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import pandas as pd

REQUIRED_PRED_COLS = [
    "model",
    "variant",
    "pretrained",
    "task",
    "target_kind",
    "etf",
    "horizon",
    "fold",
    "split",
    "date",
    "y_true",
    "y_pred",
    "seed",
]

RESULTS_SCHEMA = [
    "run_id",
    "model",
    "variant",
    "task",
    "scope",
    "fold",
    "stratum",
    "metric",
    "value",
    "ci_low",
    "ci_high",
    "status",
    "profile",
    "seed",
    "git_sha",
    "config_hash",
]


def load_predictions(
    experiments_dir: str = "experiments", run_ids: list[str] | None = None
) -> pd.DataFrame:
    """Read+concat all predictions.parquet, assert schema, dedup, sort by date. Raises on problems."""
    paths = []
    for p in sorted(glob.glob(str(Path(experiments_dir) / "*" / "predictions.parquet"))):
        rid = Path(p).parent.name
        if run_ids is None or rid in run_ids:
            paths.append((rid, p))
    if not paths:
        raise FileNotFoundError(f"no predictions.parquet under {experiments_dir}")
    frames = []
    for rid, p in paths:
        df = pd.read_parquet(p)
        missing = [c for c in REQUIRED_PRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{p} missing prediction columns {missing}")
        df["run_id"] = rid
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    if out[["y_true", "y_pred"]].isna().any().any():
        raise ValueError("NaN in y_true/y_pred")
    key = ["model", "variant", "task", "etf", "fold", "date", "seed", "horizon"]
    dups = out.duplicated(key).sum()
    if dups:
        raise ValueError(f"{dups} duplicate prediction keys {key}")
    return out.sort_values(["model", "task", "etf", "date"]).reset_index(drop=True)


def audit_alignment(preds: pd.DataFrame, experiments_dir: str = "experiments") -> dict:
    """Defensive re-check (E2): month-end target dates, no NaN, scaler_fit_on=='train' in manifests."""
    res = {}
    res["no_nan"] = "pass" if not preds[["y_true", "y_pred"]].isna().any().any() else "fail"
    res["target_month_end"] = "pass" if bool(preds["date"].dt.is_month_end.all()) else "fail"
    scaler_ok = True
    for rid in preds["run_id"].unique():
        m = Path(experiments_dir) / rid / "manifest.json"
        if m.exists():
            man = json.loads(m.read_text(encoding="utf-8"))
            if man.get("scaler_fit_on") not in ("train", None):
                scaler_ok = False
    res["scaler_fit_on_train"] = "pass" if scaler_ok else "fail"
    res["all_pass"] = all(v == "pass" for v in res.values())
    return res


def write_results_store(
    df_long: pd.DataFrame, out_path: str = "experiments/results_store.parquet"
) -> Path:
    out = df_long.copy()
    for c in RESULTS_SCHEMA:
        if c not in out.columns:
            out[c] = None
    out = out[RESULTS_SCHEMA]
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(p, index=False)
    out.to_csv(p.with_suffix(".csv"), index=False)
    return p


def read_results_store(path: str = "experiments/results_store.parquet") -> pd.DataFrame:
    return pd.read_parquet(path)
