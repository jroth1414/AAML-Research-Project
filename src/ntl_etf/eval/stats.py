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
