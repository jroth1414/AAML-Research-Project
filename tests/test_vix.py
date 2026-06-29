"""Offline VIX monthly + disruption-flag tests (Task F11)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ntl_etf.data.macro import build_vix_monthly


def _daily_const(value, month="2020-03"):
    idx = pd.date_range(f"{month}-01", f"{month}-31", freq="D")
    return pd.Series(np.full(len(idx), float(value)), index=idx)


def test_disruption_flag_threshold():
    assert bool(build_vix_monthly(_daily_const(30)).loc[0, "disruption_flag"]) is True
    assert bool(build_vix_monthly(_daily_const(15)).loc[0, "disruption_flag"]) is False
    # boundary: strictly greater-than, so mean exactly 25 is NOT a disruption month
    assert bool(build_vix_monthly(_daily_const(25)).loc[0, "disruption_flag"]) is False


def test_vix_columns_and_dtype():
    df = build_vix_monthly(_daily_const(30))
    assert list(df.columns) == ["date", "vix_mean", "vix_max", "disruption_flag"]
    assert df["disruption_flag"].dtype == bool
    assert not df["vix_mean"].isna().any()
    assert df.loc[0, "vix_max"] >= df.loc[0, "vix_mean"]


def test_vix_max_differs_from_mean():
    idx = pd.date_range("2020-03-01", "2020-03-31", freq="D")
    s = pd.Series(np.linspace(20, 40, len(idx)), index=idx)
    df = build_vix_monthly(s)
    assert df.loc[0, "vix_max"] > df.loc[0, "vix_mean"]
