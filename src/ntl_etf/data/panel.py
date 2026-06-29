"""Panel assembly: thin source loaders onto the canonical month-end grid (Phase B / Task P2).

All loaders validate columns against the data contract (P9) and reindex onto the master month-END
grid (Appendix A.1: 2013-01-31 .. 2024-12-31, 144 months, tz-naive). Genuine gaps stay NaN —
never forward-filled across pre-inception gaps (handled by valid-date masks in P4/P7).

Pair-screen (P3), target alignment (P5), windowing/registry (P4/P7) live in sibling modules /
later edits; this file currently holds the loaders + the master grid + valid masks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

MASTER_START = "2013-01-31"
MASTER_END = "2024-12-31"
SPDR = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]


@dataclass(frozen=True)
class WindowSpec:
    lookback: int
    horizon: int
    task: str  # 'leading' | 'nowcast'
    view: str = "ci"  # 'ci' | 'variate'


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


# --------------------------------------------------------------------------------------
# P5 — release-lag / publication-lag target alignment (the central leakage guard)
# --------------------------------------------------------------------------------------
def align_targets(
    origin_t: pd.Timestamp,
    returns_wide: pd.DataFrame,
    ip_wide: pd.DataFrame,
    sector: str,
    spec: WindowSpec,
    cfg: dict,
) -> tuple[np.ndarray, dict] | None:
    """Return (y(H,), meta) for one anchor, or None if the target span is unavailable/NaN.

    leading:  target months = [t + release_lag + k for k in range(H)] (strictly forward).
    nowcast:  target months = [t + k for k in range(H)] (contemporaneous); as_of = t + 1.
    """
    lag = int(cfg.get("release_lag_months", 1))
    grid = returns_wide.index
    if origin_t not in grid:
        return None
    i = grid.get_loc(origin_t)
    if spec.task == "leading":
        idx = [i + lag + k for k in range(spec.horizon)]
        if max(idx) >= len(grid) or sector not in returns_wide.columns:
            return None
        target_dates = [grid[j] for j in idx]
        y = returns_wide.loc[target_dates, sector].to_numpy(dtype="float64")
        as_of = grid[i + lag]
        if min(target_dates) <= origin_t:  # strict-forward invariant (audited in P8/L2)
            raise AssertionError("leading target not strictly forward of origin")
    elif spec.task == "nowcast":
        idx = [i + k for k in range(spec.horizon)]
        if max(idx) >= len(grid) or sector not in ip_wide.columns:
            return None
        target_dates = [grid[j] for j in idx]
        y = ip_wide.loc[target_dates, sector].to_numpy(dtype="float64")
        as_of = grid[i + 1] if i + 1 < len(grid) else grid[i]
    else:
        raise ValueError(f"unknown task {spec.task!r}")
    if not np.isfinite(y).all():
        return None
    return y, {"target_dates": target_dates, "as_of_date": as_of}


# --------------------------------------------------------------------------------------
# P3 — region->sector PRE-SCREEN correlation gate (no look-ahead)
# --------------------------------------------------------------------------------------
def screen_pairs(
    ntl_long: pd.DataFrame,
    returns_wide: pd.DataFrame,
    ip_wide: pd.DataFrame,
    regions: list[dict],
    cfg: dict,
    forced_pairs: set[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """One row per (region_id, sector) candidate pair, screened ONLY on cfg['screen_warmup'].

    Keeps a pair if |spearman| >= screen_rho_min on EITHER task, OR it is a forced (H2/H3) pair.
    Asserts the screen window ends strictly before the first fold's test month (no look-ahead).
    """
    from scipy.stats import pearsonr, spearmanr

    warm = cfg.get("screen_warmup", ["2013-01", "2017-12"])
    w0 = pd.Timestamp(warm[0]) + pd.offsets.MonthEnd(0)
    w1 = pd.Timestamp(warm[1]) + pd.offsets.MonthEnd(0)
    rho_min = float(cfg.get("screen_rho_min", 0.15))
    lag = int(cfg.get("release_lag_months", 1))
    forced_pairs = forced_pairs or set()

    rows = []
    for region in regions:
        rid = region["id"]
        ntl_s = ntl_long[ntl_long["region_id"] == rid].set_index("date")["ntl_value"].sort_index()
        for sector in region.get("candidate_sectors", []):
            forced = (rid, sector) in forced_pairs
            best = {"pearson": np.nan, "spearman": np.nan, "n_obs": 0, "task_screened": ""}
            for task, tgt in (("leading", returns_wide), ("nowcast", ip_wide)):
                if sector not in tgt.columns:
                    continue
                y = tgt[sector].shift(-lag) if task == "leading" else tgt[sector]
                df = pd.DataFrame({"x": ntl_s, "y": y}).loc[w0:w1].dropna()
                if len(df) < 8 or df["x"].nunique() <= 1 or df["y"].nunique() <= 1:
                    continue  # too few obs or a constant series -> correlation undefined
                sp = spearmanr(df["x"], df["y"]).statistic
                pe = pearsonr(df["x"], df["y"]).statistic
                if np.isnan(best["spearman"]) or abs(sp) > abs(best["spearman"]):
                    best = {
                        "pearson": float(pe),
                        "spearman": float(sp),
                        "n_obs": int(len(df)),
                        "task_screened": task,
                    }
            kept = forced or (not np.isnan(best["spearman"]) and abs(best["spearman"]) >= rho_min)
            drop_reason = "" if kept else ("low_|spearman|" if best["n_obs"] else "no_screen_obs")
            rows.append(
                {
                    "region_id": rid,
                    "sector": sector,
                    **best,
                    "kept": bool(kept),
                    "forced_keep": bool(forced),
                    "drop_reason": drop_reason,
                }
            )
    out = pd.DataFrame(rows)
    # No-look-ahead guarantee: warmup ends strictly before the first test month.
    first_test = pd.Timestamp(MASTER_START) + pd.offsets.MonthEnd(
        int(cfg.get("walk_forward", {}).get("min_train_months", 60))
        + int(cfg.get("walk_forward", {}).get("val_months", 12))
    )
    if not (w1 < first_test):
        raise AssertionError(
            f"screen warmup end {w1.date()} not before first test {first_test.date()}"
        )
    return out
