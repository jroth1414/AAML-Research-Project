"""FRED macro (industrial production) + CBOE VIX acquisition and stationarity transforms (Phase A.2).

Outputs use a tz-naive month-END DatetimeIndex (Appendix A.1). The model-ready stationary IP form is
``value_dlog`` (causal); STL ``value_sa`` is NON-causal and is plotting/descriptive only — it must
never reach the model feature matrix (Phase B leakage audit asserts no ``*_sa`` column enters
``PanelDataset.x``, Risk R13). VIX table carries ``vix_max`` per Appendix A.2 (Risk R19).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

ME = "ME"


def _me() -> str:
    try:
        pd.Series(dtype="float64", index=pd.DatetimeIndex([])).resample(ME)
        return ME
    except (ValueError, KeyError):  # pragma: no cover
        return "M"


# --------------------------------------------------------------------------------------
# F5 — FRED client + fetch + month-end align
# --------------------------------------------------------------------------------------
def get_fred_client(api_key: str | None = None):
    """Return a fredapi.Fred client. Reads FRED_API_KEY from env if api_key is None."""
    from fredapi import Fred

    key = api_key or os.environ.get("FRED_API_KEY")
    if not key:
        raise RuntimeError("set FRED_API_KEY in .env (see .env.example)")
    return Fred(api_key=key)


def fetch_series(fred, series_id: str, start: str, end: str) -> pd.Series:
    """get_series then validate non-empty; re-raise with the offending id on failure."""
    try:
        s = fred.get_series(series_id, observation_start=start, observation_end=end)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"FRED fetch failed for series_id={series_id!r}: {exc}") from exc
    if s is None or len(s.dropna()) == 0:
        raise RuntimeError(f"FRED series {series_id!r} returned no observations")
    s.name = series_id
    return s


def to_month_end_macro(s: pd.Series) -> pd.Series:
    """Re-stamp first-of-month FRED dates to month-end; assert <=1 obs/month."""
    s = s.copy()
    s.index = pd.to_datetime(s.index)
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    out = s.resample(_me()).last()
    return out


# --------------------------------------------------------------------------------------
# F7 — stationarity transforms
# --------------------------------------------------------------------------------------
def dlog(s: pd.Series) -> pd.Series:
    """Month-over-month log change Delta ln(s) (causal; primary stationary IP form)."""
    return np.log(s.astype("float64")).diff()


def first_diff(s: pd.Series) -> pd.Series:
    """First difference Delta s (causal alt)."""
    return s.astype("float64").diff()


def stl_deseasonalize(s: pd.Series, period: int = 12, robust: bool = True) -> pd.Series:
    """STL seasonally-adjusted level = trend + resid (NON-causal; descriptive/plotting only).

    Requires >= 2*period observations. Falls back to the raw series if too short.
    """
    from statsmodels.tsa.seasonal import STL

    s = s.astype("float64")
    valid = s.dropna()
    if len(valid) < 2 * period:
        return s.copy()
    res = STL(valid, period=period, robust=robust).fit()
    sa = res.trend + res.resid
    return sa.reindex(s.index)


def add_macro_transforms(
    df: pd.DataFrame, value_col: str = "value", period: int = 12
) -> pd.DataFrame:
    """Add value_sa (descriptive), value_dlog (primary causal), value_diff (causal alt)."""
    out = df.copy()
    s = pd.Series(out[value_col].to_numpy(), index=pd.to_datetime(out["date"]))
    out["value_sa"] = stl_deseasonalize(s, period=period).to_numpy()
    out["value_dlog"] = dlog(s).to_numpy()
    out["value_diff"] = first_diff(s).to_numpy()
    return out


# --------------------------------------------------------------------------------------
# F9 — VIX monthly + disruption flag (Appendix A.2: date, vix_mean, vix_max, disruption_flag)
# --------------------------------------------------------------------------------------
def build_vix_monthly(vix_daily: pd.Series, threshold: float = 25.0) -> pd.DataFrame:
    """Monthly VIX table from daily closes: [date(month-end), vix_mean, vix_max, disruption_flag].

    vix_mean/vix_max are the calendar-month mean/max of daily closes; disruption_flag = mean > thr.
    """
    s = vix_daily.astype("float64").copy()
    s.index = pd.to_datetime(s.index)
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    alias = _me()
    mean = s.resample(alias).mean()
    mx = s.resample(alias).max()
    df = pd.DataFrame(
        {
            "date": mean.index,
            "vix_mean": mean.to_numpy().astype("float32"),
            "vix_max": mx.to_numpy().astype("float32"),
        }
    )
    df["disruption_flag"] = (df["vix_mean"] > threshold).astype(bool)
    return df
