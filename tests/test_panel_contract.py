"""Tensor contract tests for the PanelDataset (Task P9). Synthetic; no network."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ntl_etf.data import panel
from ntl_etf.data.splits import walk_forward_splits

CFG = {
    "walk_forward": {
        "min_train_months": 60,
        "val_months": 12,
        "test_months": 12,
        "step_months": 12,
        "expanding": True,
    },
    "release_lag_months": 1,
}


def _build(view, horizon=1):
    grid = panel.master_grid()
    rng = np.random.default_rng(1)
    regions = ["r0", "r1", "r2"]  # all XLI -> a multi-region group for the variate view
    ntl_long = pd.concat(
        [
            pd.DataFrame(
                {
                    "date": grid,
                    "region_id": r,
                    "sector": "XLI",
                    "ntl_value": rng.normal(10, 2, len(grid)),
                }
            )
            for r in regions
        ],
        ignore_index=True,
    )
    ntl_wide = panel.pivot_ntl_wide(ntl_long)
    returns = pd.DataFrame({s: rng.normal(0, 0.04, len(grid)) for s in panel.SPDR}, index=grid)
    ip = returns[["XLI"]]
    kept = pd.DataFrame({"region_id": regions, "sector": "XLI", "kept": True, "forced_keep": False})
    reg = panel.build_series_registry(kept)
    folds = walk_forward_splits(grid, CFG)
    spec = panel.WindowSpec(12, horizon, "leading", view)
    norms = panel.compute_fold_norms(ntl_wide, returns, ip, reg, folds[0], spec)
    anchors = [
        a
        for a in panel.build_anchors(reg, ntl_wide, returns, ip, spec, folds[0], CFG)
        if a["split"] == "train"
    ]
    return panel.PanelDataset(anchors, ntl_wide, reg, norms, spec), spec


def test_ci_tensor_contract():
    ds, spec = _build("ci", horizon=1)
    s = ds[0]
    assert tuple(s["x"].shape) == (12, 1)
    assert tuple(s["y"].shape) == (1,)
    assert str(s["x"].dtype) == "torch.float32"
    assert str(s["sector_id"].dtype) == "torch.int64"
    assert not bool(s["x"].isnan().any()) and not bool(s["y"].isnan().any())


def test_ci_horizon3():
    ds, _ = _build("ci", horizon=3)
    assert tuple(ds[0]["y"].shape) == (3,)


def test_variate_tensor_contract():
    ds, _ = _build("variate", horizon=1)
    s = ds[0]
    assert tuple(s["x"].shape) == (12, 3)
    assert tuple(s["var_mask"].shape) == (3,)
    assert int(s["var_mask"].sum()) >= 2  # variate view requires >=2 valid variates
    assert not bool(s["x"].isnan().any())


def test_dataloader_batches():
    ds, _ = _build("ci")
    dl = panel.make_dataloader(ds, batch_size=8)
    batch = next(iter(dl))
    assert batch["x"].shape[0] == 8
    assert tuple(batch["x"].shape[1:]) == (12, 1)
