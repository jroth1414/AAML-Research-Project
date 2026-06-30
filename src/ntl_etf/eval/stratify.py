"""Hypothesis-aligned strata + cross-fold aggregation (Phase D / Task E6).

Strata: all, single_region (H3), multi_region (H2), disruption/stable (H4, VIX>25),
pretrained/from_scratch (H6). Region class is PRE-REGISTERED structural metadata from
``regions.yaml`` hypothesis_pairs (not learned): XLI -> multi_region, XLE -> single_region.
Aggregation reports the metric on POOLED observations plus a fold-dispersion CI for MSE.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from .metrics import compute_all_point_metrics

STRATA = [
    "all",
    "single_region",
    "multi_region",
    "disruption",
    "stable",
    "pretrained",
    "from_scratch",
]


def add_strata(
    preds: pd.DataFrame,
    regions_yaml: str = "configs/regions.yaml",
    vix_path: str = "data/processed/vix_monthly.parquet",
) -> pd.DataFrame:
    """Add region_class + disruption columns (pretrained already present). Asserts XLE/XLI classes."""
    doc = yaml.safe_load(open(regions_yaml, encoding="utf-8"))
    hp = doc.get("hypothesis_pairs", {})
    multi = {hp.get("H2_multiregion", {}).get("sector")}
    single = {hp.get("H3_singleregion", {}).get("sector")}
    out = preds.copy()
    out["region_class"] = out["etf"].apply(
        lambda e: "single_region" if e in single else ("multi_region" if e in multi else "other")
    )
    assert (
        out.loc[out.etf == "XLE", "region_class"].eq("single_region").all()
        or (out.etf == "XLE").sum() == 0
    )
    assert (
        out.loc[out.etf == "XLI", "region_class"].eq("multi_region").all()
        or (out.etf == "XLI").sum() == 0
    )
    vix = pd.read_parquet(vix_path)[["date", "disruption_flag"]]
    vix["date"] = pd.to_datetime(vix["date"])
    out = out.merge(vix, on="date", how="left")
    out["disruption"] = out["disruption_flag"].fillna(False).astype(bool)
    return out


def _filter(df: pd.DataFrame, stratum: str) -> pd.DataFrame:
    if stratum == "all":
        return df
    if stratum in ("single_region", "multi_region"):
        return df[df["region_class"] == stratum]
    if stratum == "disruption":
        return df[df["disruption"]]
    if stratum == "stable":
        return df[~df["disruption"]]
    if stratum == "pretrained":
        return df[df["pretrained"]]
    if stratum == "from_scratch":
        return df[~df["pretrained"]]
    return df.iloc[0:0]


def aggregate(preds: pd.DataFrame, strata=None, scopes=("POOLED", "per_etf")) -> pd.DataFrame:
    """Return long results-store rows (one metric value per row) per (model,task,scope,stratum)."""
    strata = strata or STRATA
    rows = []
    for (model, variant, task), g0 in preds.groupby(["model", "variant", "task"]):
        for stratum in strata:
            g = _filter(g0, stratum)
            if len(g) == 0:
                continue
            scope_groups = [("POOLED", g)] if "POOLED" in scopes else []
            if "per_etf" in scopes:
                scope_groups += [(etf, ge) for etf, ge in g.groupby("etf")]
            for scope, gs in scope_groups:
                if len(gs) < 2:
                    continue
                metrics = compute_all_point_metrics(
                    gs["y_true"].to_numpy(), gs["y_pred"].to_numpy(), task=task
                )
                # fold-dispersion CI for MSE
                fold_mse = gs.groupby("fold").apply(
                    lambda x: float(np.mean((x["y_true"] - x["y_pred"]) ** 2)), include_groups=False
                )
                lo = hi = np.nan
                if len(fold_mse) >= 2:
                    se = float(fold_mse.std(ddof=1) / np.sqrt(len(fold_mse)))
                    lo, hi = float(fold_mse.mean() - 1.96 * se), float(fold_mse.mean() + 1.96 * se)
                for metric, value in metrics.items():
                    rows.append(
                        {
                            "model": model,
                            "variant": variant,
                            "task": task,
                            "scope": scope,
                            "fold": -1,
                            "stratum": stratum,
                            "metric": metric,
                            "value": value,
                            "ci_low": lo if metric == "mse" else np.nan,
                            "ci_high": hi if metric == "mse" else np.nan,
                            "seed": int(gs["seed"].iloc[0]),
                        }
                    )
    return pd.DataFrame(rows)
