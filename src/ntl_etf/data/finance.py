"""ETF financial targets: month-end log returns + 12-month momentum, ragged-history safe (Phase A.2).

Pure, side-effect-free math (no download). All outputs use a tz-naive month-END DatetimeIndex
(DEVPLAN Appendix A.1). Prices are NEVER forward-filled before differencing (that would manufacture
zero returns); genuine gaps stay NaN. ETF log returns are already stationary and must never be
differenced again (Phase F10 transform registry).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ME = "ME"  # pandas month-end alias (pandas >= 2.2; falls back to "M" on older)


def _resample_alias() -> str:
    try:
        pd.Series(dtype="float64", index=pd.DatetimeIndex([])).resample(ME)
        return ME
    except (ValueError, KeyError):  # pragma: no cover - very old pandas
        return "M"


def to_month_end(prices: pd.Series, how: str = "last") -> pd.Series:
    """Resample (daily or finer) adj-close prices to month-end using the LAST obs in each month.

    Index becomes tz-naive month-end Timestamps. Months with no observation -> NaN (NOT ffilled).
    """
    s = prices.copy()
    if isinstance(s.index, pd.DatetimeIndex) and s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    out = s.resample(_resample_alias()).last()
    out.index = out.index.tz_localize(None) if out.index.tz is not None else out.index
    return out


def log_returns(month_end_prices: pd.Series) -> pd.Series:
    """Monthly log return r_t = ln(P_t) - ln(P_{t-1}) = np.log(p).diff(). First month NaN; no fill.

    A return straddling a NaN month stays NaN (the gap is not bridged).
    """
    p = month_end_prices.astype("float64")
    pos = p.dropna()
    if not (pos > 0).all():
        raise ValueError("non-positive prices encountered; cannot take log")
    return np.log(p).diff()


def momentum_12m(month_end_prices: pd.Series, lookback: int = 12) -> pd.Series:
    """12-month trailing time-series momentum signal: mom_t = ln(P_t) - ln(P_{t-lookback}).

    Uses only information available AT month t (trailing) -> a same-month feature, not a label.
    Requires ``lookback`` prior observations, so the first ``lookback`` rows are NaN.
    """
    lp = np.log(month_end_prices.astype("float64"))
    return lp - lp.shift(lookback)


def first_valid_month(month_end_prices: pd.Series) -> pd.Timestamp:
    """First month-end with a non-NaN price."""
    return month_end_prices.first_valid_index()


def build_ticker_frame(
    prices_daily: pd.Series, ticker: str, sector: str, lookback: int = 12
) -> pd.DataFrame:
    """Tidy long frame for ONE ticker: [ticker, sector, date, adj_close, log_return, momentum_12m]."""
    me = to_month_end(prices_daily)
    df = pd.DataFrame(
        {
            "ticker": ticker,
            "sector": sector,
            "date": me.index,
            "adj_close": me.to_numpy(),
            "log_return": log_returns(me).to_numpy(),
            "momentum_12m": momentum_12m(me, lookback).to_numpy(),
        }
    )
    return add_validity_mask(df)


def add_validity_mask(ticker_frame: pd.DataFrame) -> pd.DataFrame:
    """Add boolean ``valid`` = finite log_return. Does NOT drop rows (panel needs the full grid)."""
    out = ticker_frame.copy()
    out["valid"] = np.isfinite(out["log_return"].to_numpy())
    return out
