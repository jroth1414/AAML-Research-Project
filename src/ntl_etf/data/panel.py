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
        # NOTE: strict-forward (min(target) > origin) is enforced by the P8 leakage audit (L2),
        # not raised here, so the audit can *report* a tampered release_lag=0 as a failure.
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


# --------------------------------------------------------------------------------------
# P4 / P7 — series registry, windowing engine, and the dual-view PanelDataset
# --------------------------------------------------------------------------------------
def pivot_ntl_wide(ntl_long: pd.DataFrame) -> pd.DataFrame:
    """Wide NTL: index=master grid, columns=region_id, values=ntl_value (primary feature)."""
    w = ntl_long.pivot_table(index="date", columns="region_id", values="ntl_value")
    return w.reindex(master_grid())


def build_series_registry(kept_pairs: pd.DataFrame) -> pd.DataFrame:
    """One series per kept (region_id, sector) pair, with stable ids + variate grouping.

    Columns: series_idx, region_id, sector, sector_group_id, variate_pos, forced_keep.
    """
    kept = kept_pairs[kept_pairs["kept"]].copy().reset_index(drop=True)
    kept = kept.sort_values(["sector", "region_id"]).reset_index(drop=True)
    kept["series_idx"] = range(len(kept))
    sector_groups = {s: g for g, s in enumerate(sorted(kept["sector"].unique()))}
    kept["sector_group_id"] = kept["sector"].map(sector_groups)
    kept["variate_pos"] = kept.groupby("sector").cumcount()
    return kept[
        ["series_idx", "region_id", "sector", "sector_group_id", "variate_pos", "forced_keep"]
    ]


def compute_fold_norms(
    ntl_wide: pd.DataFrame,
    returns_wide: pd.DataFrame,
    ip_wide: pd.DataFrame,
    registry: pd.DataFrame,
    fold,
    spec: WindowSpec,
) -> dict:
    """Train-only (mu, sigma): X per region_id, y per sector. Provenance ⊆ fold.train_dates."""
    from .splits import fit_norm_stats

    train = pd.DatetimeIndex(fold.train_dates)
    x_series = {r: ntl_wide[r] for r in registry["region_id"].unique() if r in ntl_wide.columns}
    x_norm = fit_norm_stats(x_series, train)
    tgt = returns_wide if spec.task == "leading" else ip_wide
    y_series = {s: tgt[s] for s in registry["sector"].unique() if s in tgt.columns}
    y_norm = fit_norm_stats(y_series, train)
    return {"x": x_norm, "y": y_norm}


def build_anchors(
    registry: pd.DataFrame,
    ntl_wide: pd.DataFrame,
    returns_wide: pd.DataFrame,
    ip_wide: pd.DataFrame,
    spec: WindowSpec,
    fold,
    cfg: dict,
) -> list[dict]:
    """Admissible (series|group, origin_t, split) anchors for one fold.

    Requires the full lookback [t-L+1..t] of the NTL input to be valid (non-NaN) AND the target
    span (from align_targets) to be available. Split assignment uses the OUTCOME month (as_of),
    never the input month — so a window whose input is in train but whose label is in test is
    excluded from train.
    """
    grid = ntl_wide.index
    L = spec.lookback
    split_of = {}
    for sp, dates in (
        ("train", fold.train_dates),
        ("val", fold.val_dates),
        ("test", fold.test_dates),
    ):
        for d in dates:
            split_of[pd.Timestamp(d)] = sp

    def _ci_anchors(reg_rows):
        out = []
        for _, row in reg_rows.iterrows():
            r, s = row["region_id"], row["sector"]
            if r not in ntl_wide.columns:
                continue
            xv = ntl_wide[r]
            for i in range(L - 1, len(grid)):
                t = grid[i]
                window = xv.iloc[i - L + 1 : i + 1]
                if window.isna().any():
                    continue
                aligned = align_targets(t, returns_wide, ip_wide, s, spec, cfg)
                if aligned is None:
                    continue
                y, meta = aligned
                split = split_of.get(pd.Timestamp(meta["as_of_date"]))
                if split is None:
                    continue
                out.append(
                    {
                        "view": "ci",
                        "series_idx": int(row["series_idx"]),
                        "region_id": r,
                        "sector": s,
                        "origin_t": t,
                        "split": split,
                        "y": y,
                        "as_of": meta["as_of_date"],
                        "target_dates": meta["target_dates"],
                    }
                )
        return out

    if spec.view == "ci":
        return _ci_anchors(registry)

    # variate view: per sector-group, gather region series (>=2 valid variates)
    out = []
    for (sector, gid), grp in registry.groupby(["sector", "sector_group_id"]):
        regions_in = [
            r for r in grp.sort_values("variate_pos")["region_id"] if r in ntl_wide.columns
        ]
        if len(regions_in) < 2:
            continue
        for i in range(L - 1, len(grid)):
            t = grid[i]
            mask = []
            for r in regions_in:
                w = ntl_wide[r].iloc[i - L + 1 : i + 1]
                mask.append(bool(not w.isna().any()))
            if sum(mask) < 2:
                continue
            aligned = align_targets(t, returns_wide, ip_wide, sector, spec, cfg)
            if aligned is None:
                continue
            y, meta = aligned
            split = split_of.get(pd.Timestamp(meta["as_of_date"]))
            if split is None:
                continue
            out.append(
                {
                    "view": "variate",
                    "group_idx": int(gid),
                    "sector": sector,
                    "region_ids": tuple(regions_in),
                    "var_mask": tuple(mask),
                    "origin_t": t,
                    "split": split,
                    "y": y,
                    "as_of": meta["as_of_date"],
                    "target_dates": meta["target_dates"],
                }
            )
    return out


SECTOR_ID = {t: i for i, t in enumerate(SPDR)}


def _std(arr, mu_sigma):
    mu, sigma = mu_sigma
    return (np.asarray(arr, dtype="float32") - mu) / sigma


class PanelDataset:
    """torch Dataset yielding the P9 tensor contract for CI or variate views.

    X (NTL lookback) is standardized per region with TRAIN-only stats; y (target) is standardized
    per sector with TRAIN-only stats and ``(y_mu, y_sigma)`` is carried for de-standardized metrics.
    Implemented without subclassing torch.utils.data.Dataset at import time (torch is imported
    lazily in __getitem__) so the module imports even where torch is heavy.
    """

    def __init__(self, anchors, ntl_wide, registry, norms, spec: WindowSpec):
        self.anchors = anchors
        self.ntl_wide = ntl_wide
        self.registry = registry
        self.norms = norms
        self.spec = spec
        self.region_id_idx = {r: i for i, r in enumerate(sorted(ntl_wide.columns))}
        self.grid = ntl_wide.index
        self._pos = {d: i for i, d in enumerate(self.grid)}
        # variate view: pad every sample to a global max-V so batches stack; var_mask + zero-fill
        # exclude invalid/padded variates from attention (M6/P4 reconciliation).
        self.max_v = (
            max((len(a["region_ids"]) for a in anchors), default=1) if spec.view == "variate" else 1
        )

    def __len__(self):
        return len(self.anchors)

    def _window(self, region, t):
        i = self._pos[pd.Timestamp(t)]
        return self.ntl_wide[region].iloc[i - self.spec.lookback + 1 : i + 1].to_numpy("float32")

    def __getitem__(self, idx):
        import torch

        a = self.anchors[idx]
        L, H = self.spec.lookback, self.spec.horizon
        sector = a["sector"]
        y_ms = self.norms["y"].get(sector, (0.0, 1.0))
        y_std = _std(a["y"], y_ms)
        epoch_month = int(self._pos[pd.Timestamp(a["origin_t"])])
        base = {
            "y": torch.tensor(y_std, dtype=torch.float32).reshape(H),
            "y_mu": torch.tensor(y_ms[0], dtype=torch.float32),
            "y_sigma": torch.tensor(y_ms[1], dtype=torch.float32),
            "sector_id": torch.tensor(SECTOR_ID.get(sector, -1), dtype=torch.long),
            "origin_date": torch.tensor(epoch_month, dtype=torch.long),
        }
        if a["view"] == "ci":
            r = a["region_id"]
            x = _std(self._window(r, a["origin_t"]), self.norms["x"].get(r, (0.0, 1.0)))
            base.update(
                {
                    "x": torch.tensor(x, dtype=torch.float32).reshape(L, 1),
                    "region_id": torch.tensor(self.region_id_idx.get(r, -1), dtype=torch.long),
                    "feature_id": torch.tensor(0, dtype=torch.long),
                    "series_idx": torch.tensor(a["series_idx"], dtype=torch.long),
                }
            )
        else:  # variate (padded to self.max_v with zero columns + var_mask=False)
            regions = list(a["region_ids"])
            vmask = list(a["var_mask"])
            cols, ids, mask = [], [], []
            for r, valid in zip(regions, vmask, strict=True):
                cols.append(
                    _std(self._window(r, a["origin_t"]), self.norms["x"].get(r, (0.0, 1.0)))
                    if valid
                    else np.zeros(L, dtype="float32")
                )
                ids.append(self.region_id_idx.get(r, -1))
                mask.append(bool(valid))
            while len(cols) < self.max_v:  # pad to global max-V
                cols.append(np.zeros(L, dtype="float32"))
                ids.append(-1)
                mask.append(False)
            x = np.stack(cols, axis=1)  # (L, max_v)
            base.update(
                {
                    "x": torch.tensor(x, dtype=torch.float32).reshape(L, self.max_v),
                    "region_ids": torch.tensor(ids, dtype=torch.long),
                    "var_mask": torch.tensor(mask, dtype=torch.bool),
                    "group_idx": torch.tensor(a["group_idx"], dtype=torch.long),
                }
            )
        return base


def make_dataloader(
    dataset: PanelDataset, batch_size: int = 32, shuffle: bool = False, seed: int = 1414
):
    """Build a deterministic DataLoader (num_workers=0 for Windows-safe determinism)."""
    import torch
    from torch.utils.data import DataLoader

    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, generator=g, drop_last=False
    )


# --------------------------------------------------------------------------------------
# P8 — executable leakage audit (the five invariants)
# --------------------------------------------------------------------------------------
def audit_panel(
    folds,
    ntl_wide: pd.DataFrame,
    returns_wide: pd.DataFrame,
    ip_wide: pd.DataFrame,
    registry: pd.DataFrame,
    cfg: dict,
    screen_warmup_end: pd.Timestamp,
) -> dict:
    """Mechanically prove the five leakage invariants. Returns {Lk: 'pass'|'fail', ...}.

    L1 norm train-only · L2 leading strictly forward · L3 no NaN window · L4 screen no test data ·
    L5 temporal fold ordering. Designed to be FAILABLE: a tampered input (release_lag=0, norm fit
    on the full range, screen warmup into the test era) flips the corresponding invariant.
    """
    from .splits import fit_norm_stats

    res = {}
    spec = WindowSpec(int(cfg.get("audit_lookback", 12)), 1, "leading", "ci")

    # L5 temporal ordering
    l5 = all(max(f.train_dates) < min(f.val_dates) < min(f.test_dates) for f in folds)
    res["L5_temporal_ordering"] = "pass" if l5 else "fail"

    # L4 screen used no test data
    first_test = min(min(f.test_dates) for f in folds)
    res["L4_screen_no_test"] = "pass" if screen_warmup_end < first_test else "fail"

    # L1 norm provenance train-only: stats must equal a TRAIN-only recompute (and differ from full)
    f0 = folds[0]
    norms = compute_fold_norms(ntl_wide, returns_wide, ip_wide, registry, f0, spec)
    region0 = registry["region_id"].iloc[0]
    train_only = fit_norm_stats({region0: ntl_wide[region0]}, pd.DatetimeIndex(f0.train_dates))
    full = fit_norm_stats({region0: ntl_wide[region0]}, ntl_wide.index)
    l1 = (
        region0 in norms["x"]
        and abs(norms["x"][region0][0] - train_only[region0][0]) < 1e-9
        and abs(norms["x"][region0][0] - full[region0][0]) > 1e-12
    )
    res["L1_norm_train_only"] = "pass" if l1 else "fail"

    # L2/L3 over built anchors (across folds)
    l2 = l3 = True
    n_anchors = 0
    for f in folds:
        anchors = build_anchors(registry, ntl_wide, returns_wide, ip_wide, spec, f, cfg)
        n_anchors += len(anchors)
        for a in anchors:
            if min(a["target_dates"]) <= a["origin_t"]:
                l2 = False
            i = ntl_wide.index.get_loc(a["origin_t"])
            win = ntl_wide[a["region_id"]].iloc[i - spec.lookback + 1 : i + 1]
            if win.isna().any() or not np.isfinite(a["y"]).all():
                l3 = False
    res["L2_release_lag_forward"] = "pass" if (l2 and n_anchors > 0) else "fail"
    res["L3_no_nan_window"] = "pass" if (l3 and n_anchors > 0) else "fail"
    res["_n_anchors_audited"] = n_anchors
    res["all_pass"] = all(v == "pass" for k, v in res.items() if k.startswith("L"))
    return res
