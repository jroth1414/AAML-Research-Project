"""Walk-forward (rolling-origin) CV splits + train-only normalization (Phase B / Task P6).

Expanding-window by default: min 60-month train, a 12-month validation tail carved from the
pre-test region, then a 12-month out-of-sample test block; step 12 months (dense step 1 only for
the final evaluation). With ~144 months this yields ~6 folds at step 12 (documented; motivates
pooling across folds for DM power). Normalization stats are fit on the TRAIN split ONLY and never
refit on val/test — the central leakage guard audited in P8 (L1).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    fold_id: int
    train_dates: list
    val_dates: list
    test_dates: list
    norm: dict = field(default_factory=dict)  # series_id -> (mu, sigma); set train-only

    def as_record(self) -> dict:
        return {
            "fold_id": self.fold_id,
            "train_start": str(self.train_dates[0]),
            "train_end": str(self.train_dates[-1]),
            "val_start": str(self.val_dates[0]),
            "val_end": str(self.val_dates[-1]),
            "test_start": str(self.test_dates[0]),
            "test_end": str(self.test_dates[-1]),
            "n_train": len(self.train_dates),
            "n_val": len(self.val_dates),
            "n_test": len(self.test_dates),
        }


def walk_forward_splits(dates: pd.DatetimeIndex, cfg: dict) -> list[Fold]:
    """Ordered list of folds. cfg keys (under ``walk_forward``): min_train_months, val_months,
    test_months, step_months, expanding."""
    wf = cfg.get("walk_forward", cfg)
    min_train = int(wf.get("min_train_months", 60))
    val_m = int(wf.get("val_months", 12))
    test_m = int(wf.get("test_months", 12))
    step = int(wf.get("step_months", 12))
    expanding = bool(wf.get("expanding", True))
    dates = pd.DatetimeIndex(sorted(pd.DatetimeIndex(dates)))

    folds: list[Fold] = []
    k = 0
    while True:
        split_point = min_train + val_m + k * step
        test_end = split_point + test_m
        if test_end > len(dates):
            break
        train_lo = 0 if expanding else max(0, split_point - val_m - min_train)
        train = list(dates[train_lo : split_point - val_m])
        val = list(dates[split_point - val_m : split_point])
        test = list(dates[split_point:test_end])
        # Invariants
        assert len(train) >= min_train
        assert train[-1] < val[0] < test[0]
        folds.append(Fold(fold_id=k, train_dates=train, val_dates=val, test_dates=test))
        k += 1
    return folds


def fit_norm_stats(values_by_series: dict, train_dates) -> dict:
    """Per-series (mu, sigma) computed on TRAIN months ONLY. ``values_by_series`` maps series_id
    -> pd.Series indexed by date. sigma floored at 1e-8 to avoid divide-by-zero."""
    train_idx = pd.DatetimeIndex(train_dates)
    out = {}
    for sid, s in values_by_series.items():
        v = s.reindex(train_idx).to_numpy(dtype="float64")
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        mu = float(np.mean(v))
        sigma = float(np.std(v))
        out[sid] = (mu, max(sigma, 1e-8))
    return out


def apply_norm(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return (np.asarray(x, dtype="float64") - mu) / sigma


def write_folds_manifest(folds: list[Fold], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    recs = [f.as_record() for f in folds]
    path.write_text(json.dumps({"n_folds": len(folds), "folds": recs}, indent=2), encoding="utf-8")
    return path


def fold_to_dict(fold: Fold) -> dict:
    d = asdict(fold)
    d["train_dates"] = [str(x) for x in fold.train_dates]
    d["val_dates"] = [str(x) for x in fold.val_dates]
    d["test_dates"] = [str(x) for x in fold.test_dates]
    return d
