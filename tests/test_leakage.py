"""Executable leakage audit with negative controls (Task P8). All synthetic; no network.

Proves the five invariants pass on a clean panel AND that deliberately-broken variants flip the
corresponding invariant to 'fail' (the asserts have teeth).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ntl_etf.data import panel
from ntl_etf.data.splits import Fold, fit_norm_stats, walk_forward_splits

CFG = {
    "walk_forward": {
        "min_train_months": 60,
        "val_months": 12,
        "test_months": 12,
        "step_months": 12,
        "expanding": True,
    },
    "release_lag_months": 1,
    "audit_lookback": 12,
}


def _synth():
    grid = panel.master_grid()
    rng = np.random.default_rng(0)
    regions = ["r0", "r1", "r2", "r3"]
    sectors = ["XLE", "XLI", "XLK"]
    ntl_long = pd.concat(
        [
            pd.DataFrame(
                {
                    "date": grid,
                    "region_id": r,
                    "sector": sectors[i % 3],
                    "ntl_value": rng.normal(10, 2, len(grid)),
                }
            )
            for i, r in enumerate(regions)
        ],
        ignore_index=True,
    )
    ntl_wide = panel.pivot_ntl_wide(ntl_long)
    returns = pd.DataFrame({s: rng.normal(0, 0.04, len(grid)) for s in panel.SPDR}, index=grid)
    ip = returns[["XLE", "XLI", "XLK"]]
    kept = pd.DataFrame(
        {
            "region_id": regions,
            "sector": [sectors[i % 3] for i in range(4)],
            "kept": True,
            "forced_keep": False,
        }
    )
    registry = panel.build_series_registry(kept)
    folds = walk_forward_splits(grid, CFG)
    return ntl_wide, returns, ip, registry, folds


def test_audit_all_pass():
    nw, ret, ip, reg, folds = _synth()
    res = panel.audit_panel(folds, nw, ret, ip, reg, CFG, pd.Timestamp("2017-12-31"))
    assert res["all_pass"] is True
    assert res["_n_anchors_audited"] > 0
    for k in (
        "L1_norm_train_only",
        "L2_release_lag_forward",
        "L3_no_nan_window",
        "L4_screen_no_test",
        "L5_temporal_ordering",
    ):
        assert res[k] == "pass"


def test_negative_L4_screen_lookahead():
    nw, ret, ip, reg, folds = _synth()
    res = panel.audit_panel(folds, nw, ret, ip, reg, CFG, pd.Timestamp("2021-12-31"))
    assert res["L4_screen_no_test"] == "fail"
    assert res["all_pass"] is False


def test_negative_L2_release_lag_zero():
    nw, ret, ip, reg, folds = _synth()
    bad = {**CFG, "release_lag_months": 0}
    res = panel.audit_panel(folds, nw, ret, ip, reg, bad, pd.Timestamp("2017-12-31"))
    assert res["L2_release_lag_forward"] == "fail"
    assert res["all_pass"] is False


def test_negative_L5_bad_folds():
    nw, ret, ip, reg, folds = _synth()
    f = folds[0]
    bad = [Fold(0, f.train_dates + f.val_dates, f.val_dates, f.test_dates)]  # train overlaps val
    res = panel.audit_panel(bad, nw, ret, ip, reg, CFG, pd.Timestamp("2017-12-31"))
    assert res["L5_temporal_ordering"] == "fail"


def test_L1_train_only_discriminates():
    nw, _, _, reg, folds = _synth()
    r0 = reg["region_id"].iloc[0]
    train = fit_norm_stats({r0: nw[r0]}, pd.DatetimeIndex(folds[0].train_dates))
    full = fit_norm_stats({r0: nw[r0]}, nw.index)
    # train-only stats genuinely differ from full-range -> the L1 discriminator has teeth
    assert abs(train[r0][0] - full[r0][0]) > 1e-9
