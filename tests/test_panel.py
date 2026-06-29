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
