"""Point, directional, trading, and nowcast metrics (Phase D / Task E3).

Pure functions over numpy arrays; no global state. All exact formulas are unit-tested against
hand-computed fixtures. Conventions: sign(0)=+1; OOS R^2 uses the mean of the TEST targets as the
baseline; Sharpe annualizes monthly returns with sqrt(12); long/short cost is 10 bps one-way.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def mse(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean((y - p) ** 2))


def mae(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean(np.abs(y - p)))


def _sign(a, zero_sign=1):
    s = np.sign(a)
    s[s == 0] = zero_sign
    return s


def directional_accuracy(y, p, zero_sign: int = 1, exclude_zero_target: bool = False) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    if exclude_zero_target:
        keep = y != 0
        y, p = y[keep], p[keep]
    if len(y) == 0:
        return float("nan")
    return float(np.mean(_sign(p, zero_sign) == _sign(y, zero_sign)))


def directional_accuracy_pvalue(y, p, zero_sign: int = 1) -> float:
    """One-sided binomial test that directional accuracy > 0.50 (Risk R24)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    if len(y) == 0:
        return float("nan")
    hits = int(np.sum(_sign(p, zero_sign) == _sign(y, zero_sign)))
    return float(stats.binomtest(hits, len(y), 0.5, alternative="greater").pvalue)


def strategy_returns(
    y_true, y_pred, cost_one_way: float = 0.0010, prev_pos: int | None = None
) -> dict:
    """Long/short strategy: pos=+1 if pred>0 else -1; gross r=pos*y_true; net subtracts cost*turnover.

    Turnover_i = |pos_i - pos_{i-1}| in {0,2}; first period uses ``prev_pos`` (or |pos_0| if None,
    i.e. an entry from flat). Returns positions/turnover/gross/net arrays.
    """
    y_true = np.asarray(y_true, float)
    pos = np.where(np.asarray(y_pred, float) > 0, 1.0, -1.0)
    prev = np.empty_like(pos)
    prev[0] = 0.0 if prev_pos is None else float(prev_pos)
    prev[1:] = pos[:-1]
    turnover = np.abs(pos - prev)
    gross = pos * y_true
    net = gross - cost_one_way * turnover
    return {"gross": gross, "net": net, "positions": pos, "turnover": turnover}


def annualized_sharpe(returns, periods_per_year: int = 12) -> float:
    r = np.asarray(returns, float)
    if len(r) < 2:
        return float("nan")
    sd = np.std(r, ddof=1)
    if not np.isfinite(sd) or sd < 1e-12:  # ~constant returns (float-noise tolerant)
        return float("nan")
    return float(np.sqrt(periods_per_year) * np.mean(r) / sd)


def nowcast_r2(y, p) -> float:
    """OOS R^2 vs the mean of the TEST targets (can be negative; sign kept)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def pearson_corr(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    if len(y) < 2 or np.std(y) == 0 or np.std(p) == 0:
        return float("nan")
    return float(stats.pearsonr(p, y).statistic)


def compute_all_point_metrics(
    y_true, y_pred, *, task: str = "leading", cost_one_way: float = 0.0010
) -> dict:
    """All scalar metrics for one (model, task, scope, stratum) group."""
    out = {
        "mse": mse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "dir_acc": directional_accuracy(y_true, y_pred),
        "dir_acc_pvalue": directional_accuracy_pvalue(y_true, y_pred),
        "pearson": pearson_corr(y_true, y_pred),
        "n_obs": int(len(np.asarray(y_true))),
    }
    if task == "leading":
        sr = strategy_returns(y_true, y_pred, cost_one_way=cost_one_way)
        out["sharpe_gross"] = annualized_sharpe(sr["gross"])
        out["sharpe_net"] = annualized_sharpe(sr["net"])
    else:
        out["nowcast_r2"] = nowcast_r2(y_true, y_pred)
    return out
