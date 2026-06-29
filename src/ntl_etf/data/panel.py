"""Panel assembly: thin source loaders onto the canonical month-end grid (Phase B / Task P2).

All loaders validate columns against the data contract (P9) and reindex onto the master month-END
grid (Appendix A.1: 2013-01-31 .. 2024-12-31, 144 months, tz-naive). Genuine gaps stay NaN —
never forward-filled across pre-inception gaps (handled by valid-date masks in P4/P7).

Pair-screen (P3), target alignment (P5), windowing/registry (P4/P7) live in sibling modules /
later edits; this file currently holds the loaders + the master grid + valid masks.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

MASTER_START = "2013-01-31"
MASTER_END = "2024-12-31"
SPDR = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]


class SchemaError(ValueError):
    """Raised when a loaded artifact deviates from the P9 data contract."""


def master_grid(start: str = MASTER_START, end: str = MASTER_END) -> pd.DatetimeIndex:
    """The canonical tz-naive month-end grid (144 months for the default range)."""
    return pd.date_range(start, end, freq="ME")


def _require_cols(df: pd.DataFrame, cols, name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SchemaError(f"{name} missing columns {missing}; has {list(df.columns)}")


def _to_month_end(s: pd.Series) -> pd.Series:
    d = pd.to_datetime(s)
    if getattr(d.dt, "tz", None) is not None:
        d = d.dt.tz_localize(None)
    return d


def sector_to_ticker(map_path: str = "configs/sector_fred_map.yaml") -> dict:
    """sector-name -> SPDR ticker (from the F6-owned map); plus 'overall' -> INDPRO control."""
    doc = yaml.safe_load(Path(map_path).read_text(encoding="utf-8")) or {}
    out = {row["sector"]: row["ticker"] for row in doc.get("sectors", [])}
    if doc.get("control"):
        out[doc["control"]["sector"]] = doc["control"]["ticker"]
    return out


def load_ntl_features(
    path: str = "data/processed/ntl_features.parquet", feature: str = "ntl_mean"
) -> pd.DataFrame:
    """Long NTL: [date, region_id, sector, ntl_value] for the chosen primary feature, month-end."""
    df = pd.read_parquet(path)
    _require_cols(df, ["region", "sector", "date", feature], "ntl_features")
    out = pd.DataFrame(
        {
            "date": _to_month_end(df["date"]),
            "region_id": df["region"].astype(str),
            "sector": df["sector"].astype(str),
            "ntl_value": df[feature].astype("float32"),
        }
    )
    if out.duplicated(["date", "region_id"]).any():
        raise SchemaError("duplicate (date, region_id) in ntl_features")
    return out


def load_etf_returns(path: str = "data/processed/etf_returns.parquet") -> pd.DataFrame:
    """Wide month-end ETF log returns: index=date (master grid), 11 ticker columns."""
    df = pd.read_parquet(path)
    _require_cols(df, ["ticker", "date", "log_return"], "etf_returns")
    df = df.copy()
    df["date"] = _to_month_end(df["date"])
    wide = df.pivot(index="date", columns="ticker", values="log_return").reindex(master_grid())
    missing = [t for t in SPDR if t not in wide.columns]
    if missing:
        raise SchemaError(f"etf_returns missing tickers {missing}")
    return wide[SPDR].astype("float32")


def load_etf_momentum(path: str = "data/processed/etf_returns.parquet") -> pd.DataFrame:
    """Wide month-end 12m momentum: index=date (master grid), 11 ticker columns."""
    df = pd.read_parquet(path)
    _require_cols(df, ["ticker", "date", "momentum_12m"], "etf_returns")
    df = df.copy()
    df["date"] = _to_month_end(df["date"])
    wide = df.pivot(index="date", columns="ticker", values="momentum_12m").reindex(master_grid())
    return wide[[t for t in SPDR if t in wide.columns]].astype("float32")


def load_macro_ip(
    path: str = "data/processed/macro_ip.parquet",
    target_col: str = "value_dlog",
    map_path: str = "configs/sector_fred_map.yaml",
) -> pd.DataFrame:
    """Wide month-end nowcast target (per ticker): index=date, columns=eligible tickers + INDPRO."""
    df = pd.read_parquet(path)
    _require_cols(df, ["sector", "date", target_col], "macro_ip")
    df = df.copy()
    df["date"] = _to_month_end(df["date"])
    s2t = sector_to_ticker(map_path)
    df["ticker"] = df["sector"].map(s2t)
    df = df.dropna(subset=["ticker"])
    wide = df.pivot(index="date", columns="ticker", values=target_col).reindex(master_grid())
    return wide.astype("float32")


def load_vix(path: str = "data/processed/vix_monthly.parquet") -> pd.DataFrame:
    """VIX monthly: index=date (master grid), columns [vix_mean, vix_max, disruption_flag]."""
    df = pd.read_parquet(path)
    _require_cols(df, ["date", "vix_mean", "vix_max", "disruption_flag"], "vix_monthly")
    df = df.copy()
    df["date"] = _to_month_end(df["date"])
    out = df.set_index("date")[["vix_mean", "vix_max", "disruption_flag"]].reindex(master_grid())
    return out


def build_valid_masks(etf_wide: pd.DataFrame, ntl_long: pd.DataFrame) -> pd.DataFrame:
    """One row per series with first_valid/last_valid: ETF tickers + (region_id) NTL series."""
    rows = []
    for t in etf_wide.columns:
        s = etf_wide[t].dropna()
        if len(s):
            rows.append(
                {
                    "series_id": t,
                    "kind": "etf",
                    "first_valid": s.index.min(),
                    "last_valid": s.index.max(),
                }
            )
    for rid, g in ntl_long.dropna(subset=["ntl_value"]).groupby("region_id"):
        rows.append(
            {
                "series_id": rid,
                "kind": "region_feat",
                "first_valid": g["date"].min(),
                "last_valid": g["date"].max(),
            }
        )
    return pd.DataFrame(rows)
