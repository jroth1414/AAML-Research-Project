"""Offline finance math tests (Task F11). No network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ntl_etf.data import finance


def _monthly(prices, start="2013-01-31"):
    idx = pd.date_range(start, periods=len(prices), freq="ME")
    return pd.Series(prices, index=idx, dtype="float64")


def test_log_return_values():
    r = finance.log_returns(_monthly([100, 110, 99]))
    assert np.isnan(r.iloc[0])
    assert r.iloc[1] == pytest.approx(0.0953102, abs=1e-5)
    assert r.iloc[2] == pytest.approx(-0.1053605, abs=1e-5)


def test_momentum_constant_growth():
    # 1% log growth per month for 13 months -> momentum_12m at month index 12 == 0.12
    prices = np.exp(0.01 * np.arange(13))
    mom = finance.momentum_12m(_monthly(prices), lookback=12)
    assert mom.iloc[12] == pytest.approx(0.12, abs=1e-6)
    assert mom.iloc[:12].isna().all()


def test_month_end_resample():
    daily = pd.Series(
        range(1, 90), index=pd.date_range("2013-01-01", periods=89, freq="D"), dtype="float64"
    )
    me = finance.to_month_end(daily)
    assert me.index.is_month_end.all()
    assert me.index.tz is None
    assert len(me) == 3  # Jan, Feb, Mar


def test_no_ffill_on_gap():
    s = _monthly([100, np.nan, 110])
    r = finance.log_returns(s)
    assert np.isnan(r.iloc[1]) and np.isnan(r.iloc[2])  # gap not bridged


def test_inception_mask():
    # synthetic ragged ETF starting 2018-06: first valid RETURN is the following month
    daily = pd.Series(
        100.0 + np.arange(400),
        index=pd.date_range("2018-06-01", periods=400, freq="D"),
        dtype="float64",
    )
    frame = finance.build_ticker_frame(daily, "XLC", "communication")
    valid = frame.loc[frame["valid"], "date"]
    assert valid.min() >= pd.Timestamp("2018-07-31")
    assert "valid" in frame.columns
