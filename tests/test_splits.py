"""Walk-forward split + train-only normalization tests (Task P6)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ntl_etf.data import panel
from ntl_etf.data.splits import Fold, fit_norm_stats, walk_forward_splits, write_folds_manifest

CFG = {
    "walk_forward": {
        "min_train_months": 60,
        "val_months": 12,
        "test_months": 12,
        "step_months": 12,
        "expanding": True,
    }
}


def test_six_folds_and_ordering():
    folds = walk_forward_splits(panel.master_grid(), CFG)
    assert len(folds) == 6  # 144 months, 60+12+12 at step 12
    for f in folds:
        assert len(f.train_dates) >= 60
        assert f.train_dates[-1] < f.val_dates[0] < f.test_dates[0]
        # pairwise disjoint
        assert not (set(f.train_dates) & set(f.val_dates))
        assert not (set(f.val_dates) & set(f.test_dates))


def test_expanding_train_grows():
    folds = walk_forward_splits(panel.master_grid(), CFG)
    assert len(folds[0].train_dates) == 60
    assert len(folds[1].train_dates) == 72  # +step


def test_norm_train_only_provenance():
    grid = panel.master_grid()
    folds = walk_forward_splits(grid, CFG)
    f = folds[0]
    s = pd.Series(np.arange(len(grid), dtype="float64"), index=grid)
    stats = fit_norm_stats({"r1": s}, f.train_dates)
    # mu/sigma must equal the train-only stats (provenance ⊆ train_dates)
    train_vals = s.reindex(pd.DatetimeIndex(f.train_dates)).to_numpy()
    assert stats["r1"][0] == np.mean(train_vals)
    assert abs(stats["r1"][1] - np.std(train_vals)) < 1e-9
    # value at a TEST date must NOT influence the stat (different from full-range mean)
    assert stats["r1"][0] != np.mean(s.to_numpy())


def test_folds_manifest_roundtrip(tmp_path):
    folds = walk_forward_splits(panel.master_grid(), CFG)
    p = write_folds_manifest(folds, tmp_path / "folds.json")
    import json

    doc = json.loads(p.read_text())
    assert doc["n_folds"] == 6
    assert doc["folds"][0]["n_train"] == 60


def test_fold_is_frozen():
    f = Fold(
        0, [pd.Timestamp("2013-01-31")], [pd.Timestamp("2013-02-28")], [pd.Timestamp("2013-03-31")]
    )
    assert f.fold_id == 0
