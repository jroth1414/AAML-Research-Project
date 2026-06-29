"""Offline panel-loader tests (Task P2). Synthetic parquets in tmp; no real data/network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ntl_etf.data import panel


def test_master_grid():
    g = panel.master_grid()
    assert len(g) == 144
    assert g.is_month_end.all()
    assert g.tz is None
    assert g[0] == pd.Timestamp("2013-01-31") and g[-1] == pd.Timestamp("2024-12-31")


def _write_etf(tmp):
    dates = panel.master_grid()
    rows = []
    rng = np.random.default_rng(0)
    for t in panel.SPDR:
        ret = rng.normal(0, 0.04, len(dates)).astype("float32")
        if t == "XLC":  # ragged: NaN before 2018-06
            ret[dates < pd.Timestamp("2018-06-30")] = np.nan
        rows.append(
            pd.DataFrame({"ticker": t, "date": dates, "log_return": ret, "momentum_12m": ret * 12})
        )
    p = tmp / "etf.parquet"
    pd.concat(rows, ignore_index=True).to_parquet(p)
    return p


def test_load_etf_returns_and_ragged(tmp_path):
    wide = panel.load_etf_returns(_write_etf(tmp_path))
    assert wide.shape == (144, 11)
    assert list(wide.columns) == panel.SPDR
    assert wide.loc[:"2018-05", "XLC"].isna().all()


def test_schema_error_on_missing_col(tmp_path):
    p = tmp_path / "bad.parquet"
    pd.DataFrame({"ticker": ["XLE"], "date": [pd.Timestamp("2013-01-31")]}).to_parquet(p)
    with pytest.raises(panel.SchemaError):
        panel.load_etf_returns(p)


def test_load_vix(tmp_path):
    dates = panel.master_grid()
    p = tmp_path / "vix.parquet"
    pd.DataFrame(
        {
            "date": dates,
            "vix_mean": np.full(144, 18.0, "float32"),
            "vix_max": np.full(144, 22.0, "float32"),
            "disruption_flag": np.zeros(144, bool),
        }
    ).to_parquet(p)
    vix = panel.load_vix(p)
    assert list(vix.columns) == ["vix_mean", "vix_max", "disruption_flag"]
    assert len(vix) == 144


def test_build_valid_masks(tmp_path):
    etf = panel.load_etf_returns(_write_etf(tmp_path))
    ntl = pd.DataFrame(
        {
            "date": list(panel.master_grid()[:24]) * 2,
            "region_id": ["r1"] * 24 + ["r2"] * 24,
            "sector": ["XLI"] * 48,
            "ntl_value": np.arange(48.0, dtype="float32"),
        }
    )
    masks = panel.build_valid_masks(etf, ntl)
    assert set(masks["kind"]) == {"etf", "region_feat"}
    assert (masks["first_valid"] <= masks["last_valid"]).all()


def _wide_returns():
    grid = panel.master_grid()
    return pd.DataFrame({t: np.arange(len(grid), dtype="float64") for t in panel.SPDR}, index=grid)


def test_align_targets_strictly_forward():
    ret = _wide_returns()
    ip = ret[["XLI"]]
    cfg = {"release_lag_months": 1}
    spec = panel.WindowSpec(12, 1, "leading")
    y, meta = panel.align_targets(pd.Timestamp("2018-03-31"), ret, ip, "XLE", spec, cfg)
    assert meta["target_dates"][0] == pd.Timestamp("2018-04-30")  # strictly t+1
    assert meta["target_dates"][0] > pd.Timestamp("2018-03-31")
    # horizon 3 -> three consecutive forward months
    y3, m3 = panel.align_targets(
        pd.Timestamp("2018-03-31"), ret, ip, "XLE", panel.WindowSpec(12, 3, "leading"), cfg
    )
    assert len(m3["target_dates"]) == 3


def test_align_targets_none_at_series_end():
    ret = _wide_returns()
    spec = panel.WindowSpec(12, 1, "leading")
    # last grid month has no t+1 -> None
    assert panel.align_targets(panel.master_grid()[-1], ret, ret, "XLE", spec, {}) is None


def test_screen_forced_keep_and_no_lookahead():
    grid = panel.master_grid()
    rng = np.random.default_rng(0)
    ntl = pd.concat(
        [
            pd.DataFrame(
                {
                    "date": grid,
                    "region_id": rid,
                    "sector": "XLI",
                    "ntl_value": rng.normal(0, 1, len(grid)),
                }
            )
            for rid in ["pearl_river_delta", "weak_region"]
        ],
        ignore_index=True,
    )
    ret = _wide_returns()
    regions = [
        {"id": "pearl_river_delta", "candidate_sectors": ["XLI"]},
        {"id": "weak_region", "candidate_sectors": ["XLI"]},
    ]
    cfg = {
        "screen_warmup": ["2013-01", "2017-12"],
        "screen_rho_min": 0.99,
        "walk_forward": {"min_train_months": 60, "val_months": 12},
    }
    scr = panel.screen_pairs(
        ntl, ret, ret[["XLI"]], regions, cfg, forced_pairs={("pearl_river_delta", "XLI")}
    )
    assert scr.set_index("region_id").loc["pearl_river_delta", "kept"]  # forced
    assert scr.set_index("region_id").loc["pearl_river_delta", "forced_keep"]


def test_screen_raises_on_lookahead():
    ret = _wide_returns()
    ntl = pd.DataFrame(
        {"date": panel.master_grid(), "region_id": "r", "sector": "XLI", "ntl_value": 1.0}
    )
    regions = [{"id": "r", "candidate_sectors": ["XLI"]}]
    # warmup ending in 2024 overlaps test periods -> must raise
    cfg = {
        "screen_warmup": ["2013-01", "2024-12"],
        "screen_rho_min": 0.15,
        "walk_forward": {"min_train_months": 60, "val_months": 12},
    }
    with pytest.raises(AssertionError):
        panel.screen_pairs(ntl, ret, ret[["XLI"]], regions, cfg)
