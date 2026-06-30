"""Diebold-Mariano test with HAC variance + small-sample correction (Phase D / Task E4).

Pairwise DM on the loss differential with a Newey-West (HAC) long-run variance and the
Harvey-Leybourne-Newbold (HLN, 1997) small-sample correction, referenced to a Student-t(T-1). Per
Risk R9 the truncation lag is DATA-DRIVEN even at H=1 (``floor(4*(T/100)^(2/9))``), not assumed zero
— pooling/overlap induce autocorrelation. Negative mean_diff => model A better.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class DMResult:
    stat: float  # HLN-corrected statistic DM*
    pvalue: float  # two-sided, t(T-1)
    mean_diff: float  # d-bar (negative => model A better)
    n: int  # realized T
    horizon: int
    loss: str  # "mse" | "mae"
    better: str | None = None  # "A" | "B" | None (filled by the caller post-correction)


def newey_west_lag(T: int, horizon: int = 1) -> int:
    """Data-driven NW truncation lag = max(h-1, floor(4*(T/100)^(2/9))) (Risk R9)."""
    if T <= 1:
        return 0
    data_driven = int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0)))
    return max(horizon - 1, data_driven)


def _loss(e, loss: str):
    e = np.asarray(e, float)
    return e**2 if loss == "mse" else np.abs(e)


def dm_from_diff(d, horizon: int = 1, loss: str = "mse") -> DMResult:
    """DM test on a precomputed loss-differential series ``d`` (negative mean => model A better).

    Use this for date-clustered pooling: pass the per-date mean loss differential so cross-sectional
    correlation across ETFs at a date is collapsed before the HAC+HLN test (Risk R9)."""
    d = np.asarray(d, float)
    T = len(d)
    if T <= horizon:
        return DMResult(
            float("nan"),
            float("nan"),
            float(np.mean(d)) if T else float("nan"),
            T,
            horizon,
            loss,
            None,
        )
    dbar = float(np.mean(d))
    dc = d - dbar
    gamma0 = float(np.mean(dc * dc))
    m = newey_west_lag(T, horizon)
    lrv = gamma0
    for k in range(1, m + 1):
        lrv += 2.0 * float(np.mean(dc[k:] * dc[:-k]))
    if lrv <= 0:
        return DMResult(0.0, 1.0, dbar, T, horizon, loss, None)
    dm = dbar / np.sqrt(lrv / T)
    hln = np.sqrt((T + 1 - 2 * horizon + horizon * (horizon - 1) / T) / T)
    dm_star = dm * hln
    pvalue = float(2 * stats.t.cdf(-abs(dm_star), df=T - 1))
    return DMResult(float(dm_star), pvalue, dbar, T, horizon, loss, None)


def diebold_mariano(e_a, e_b, horizon: int = 1, loss: str = "mse") -> DMResult:
    """e_a, e_b are forecast ERROR arrays (y_true - y_pred), aligned, equal-length."""
    e_a, e_b = np.asarray(e_a, float), np.asarray(e_b, float)
    d = _loss(e_a, loss) - _loss(e_b, loss)
    T = len(d)
    if T <= horizon:
        return DMResult(
            float("nan"),
            float("nan"),
            float(np.mean(d)) if T else float("nan"),
            T,
            horizon,
            loss,
            None,
        )
    dbar = float(np.mean(d))
    dc = d - dbar
    gamma0 = float(np.mean(dc * dc))
    m = newey_west_lag(T, horizon)
    lrv = gamma0
    for k in range(1, m + 1):
        gk = float(np.mean(dc[k:] * dc[:-k]))
        lrv += 2.0 * gk
    if lrv <= 0:  # degenerate (e.g. identical forecasts) -> no evidence
        return DMResult(0.0, 1.0, dbar, T, horizon, loss, None)
    dm = dbar / np.sqrt(lrv / T)
    hln = np.sqrt((T + 1 - 2 * horizon + horizon * (horizon - 1) / T) / T)
    dm_star = dm * hln
    pvalue = float(2 * stats.t.cdf(-abs(dm_star), df=T - 1))
    return DMResult(float(dm_star), pvalue, dbar, T, horizon, loss, None)


def apply_correction(pvals, method: str = "holm", alpha: float = 0.10):
    """Wrap statsmodels multipletests; returns (reject: np.ndarray, p_adj: np.ndarray)."""
    from statsmodels.stats.multitest import multipletests

    pvals = np.asarray(pvals, float)
    if len(pvals) == 0:
        return np.array([], bool), np.array([], float)
    reject, p_adj, _, _ = multipletests(pvals, alpha=alpha, method=method)
    return reject, p_adj


def _filter_stratum(preds, task, horizon, stratum):
    import pandas as pd  # noqa: F401

    df = preds[(preds["task"] == task) & (preds["horizon"] == horizon)]
    if stratum in ("multi_region", "single_region"):
        return df[df["region_class"] == stratum]
    if stratum == "disruption":
        return df[df["disruption"]]
    if stratum == "stable":
        return df[~df["disruption"]]
    return df


def run_dm_suite(preds, families, alpha: float = 0.10, loss: str = "mse"):
    """Execute every pre-registered family/pair with date-clustered DM + Holm/BH within family.

    For each pair, align the two models on common (etf, date), date-cluster the loss differential
    (mean per date), and run the HAC+HLN DM. ``win`` is the better model iff Holm-significant.
    """
    import pandas as pd

    rows, fam_idx = [], []
    for fam in families:
        strata = (
            ["multi_region", "single_region"]
            if fam["stratum"] == "by_region_class"
            else [fam["stratum"]]
        )
        idxs = []
        for stratum in strata:
            sub = _filter_stratum(preds, fam["task"], fam["horizon"], stratum)
            for a, b in fam["pairs"]:
                da = sub[sub["model"] == a][["etf", "date", "y_true", "y_pred"]]
                db = sub[sub["model"] == b][["etf", "date", "y_true", "y_pred"]]
                merged = da.merge(db, on=["etf", "date"], suffixes=("_a", "_b"))
                rec = {
                    "family": fam["name"],
                    "hypothesis": fam["hypothesis"],
                    "task": fam["task"],
                    "scope": fam.get("scope", "POOLED"),
                    "stratum": stratum,
                    "horizon": fam["horizon"],
                    "model_a": a,
                    "model_b": b,
                    "loss": loss,
                    "dm_stat": float("nan"),
                    "p_raw": float("nan"),
                    "p_holm": float("nan"),
                    "p_bh": float("nan"),
                    "mean_diff": float("nan"),
                    "n": 0,
                    "win": "none",
                }
                if len(merged) >= 3:
                    la = _loss(merged["y_true_a"] - merged["y_pred_a"], loss)
                    lb = _loss(merged["y_true_b"] - merged["y_pred_b"], loss)
                    d_date = (
                        pd.DataFrame({"date": merged["date"].to_numpy(), "d": la - lb})
                        .groupby("date")["d"]
                        .mean()
                        .sort_index()
                        .to_numpy()
                    )
                    r = dm_from_diff(d_date, fam["horizon"], loss)
                    rec.update(dm_stat=r.stat, p_raw=r.pvalue, mean_diff=r.mean_diff, n=r.n)
                rows.append(rec)
                idxs.append(len(rows) - 1)
        fam_idx.append(idxs)

    df = pd.DataFrame(rows)
    for idxs in fam_idx:  # Holm + BH within each family
        valid = [i for i in idxs if not np.isnan(df.at[i, "p_raw"])]
        if not valid:
            continue
        pv = df.loc[valid, "p_raw"].to_numpy()
        _, p_holm = apply_correction(pv, "holm", alpha)
        _, p_bh = apply_correction(pv, "fdr_bh", alpha)
        for j, i in enumerate(valid):
            df.at[i, "p_holm"] = float(p_holm[j])
            df.at[i, "p_bh"] = float(p_bh[j])
            if p_holm[j] < alpha:
                df.at[i, "win"] = (
                    df.at[i, "model_a"] if df.at[i, "mean_diff"] < 0 else df.at[i, "model_b"]
                )
    return df
