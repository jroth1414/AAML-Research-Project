"""Offline macro transform tests + credential-gated FRED smoke (Task F11)."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from ntl_etf.data import macro


def test_dlog():
    r = macro.dlog(pd.Series([100.0, 101.0, 103.0]))
    assert np.isnan(r.iloc[0])
    assert r.iloc[1] == pytest.approx(0.00995033, abs=1e-5)
    assert r.iloc[2] == pytest.approx(0.01960847, abs=1e-5)


def test_stl_reduces_seasonality():
    t = np.arange(120)
    seasonal = 10 * np.sin(2 * np.pi * t / 12)
    raw = pd.Series(
        100 + 0.1 * t + seasonal + np.random.default_rng(0).normal(0, 0.5, 120),
        index=pd.date_range("2013-01-31", periods=120, freq="ME"),
    )
    sa = macro.stl_deseasonalize(raw, period=12)

    def acf12(x):
        x = np.asarray(x.dropna())
        x = x - x.mean()
        return abs(np.corrcoef(x[:-12], x[12:])[0, 1])

    assert acf12(sa) < acf12(raw)


def test_add_macro_transforms_no_inf():
    df = pd.DataFrame(
        {"date": pd.date_range("2013-01-31", periods=30, freq="ME"), "value": np.arange(1, 31.0)}
    )
    out = macro.add_macro_transforms(df)
    assert {"value_sa", "value_dlog", "value_diff"} <= set(out.columns)
    assert np.isnan(out["value_dlog"].iloc[0]) and np.isnan(out["value_diff"].iloc[0])
    assert np.isfinite(out["value_dlog"].iloc[1:]).all()


def test_to_month_end_macro():
    s = pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2013-01-01", "2013-02-01", "2013-03-01"]))
    me = macro.to_month_end_macro(s)
    assert me.index.is_month_end.all()
    assert len(me) == 3


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("FRED_API_KEY"), reason="no FRED key")
def test_fred_smoke():
    fred = macro.get_fred_client()
    s = macro.fetch_series(fred, "INDPRO", "2024-01-01", "2024-03-31")
    assert len(s.dropna()) >= 2
    assert (s.dropna() > 0).all()
