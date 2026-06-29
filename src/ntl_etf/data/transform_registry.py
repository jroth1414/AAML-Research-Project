"""Calendar-alignment + transform registry (Phase A.2 / Task F10).

The single source of truth for WHICH transform feeds WHICH task, so Phase B/C consume the right
columns and never double-transform. Alignment rule (Appendix A.1): all series share one tz-naive
month-end grid; left-join to the master grid; never forward-fill across pre-inception gaps.

Leading task: target = ETF log_return at month t+1 (release-lag, RELEASE_LAG_MONTHS=1); NTL feature
is month t (lag 1). Nowcast task: target = IP value_dlog at month t; NTL feature is month t (lag 0).
ETF log returns are already stationary and must NOT be differenced again.
"""

from __future__ import annotations

TRANSFORM_REGISTRY = {
    "etf_return": {
        "leading": {
            "col": "log_return",
            "transform": "none",
            "note": "already stationary; DO NOT difference",
        },
        "nowcast": {"col": None, "note": "returns are the LEADING target only"},
    },
    "etf_momentum": {
        "leading": {
            "col": "momentum_12m",
            "transform": "trailing_12m",
            "note": "same-month trailing feature; baseline signal",
        },
    },
    "ip_target": {
        "nowcast": {
            "col": "value_dlog",
            "transform": "dlog",
            "causal": True,
            "note": "model target/feature stationary form",
        },
        "descriptive": {
            "col": "value_sa",
            "transform": "STL",
            "causal": False,
            "note": "plots only; never enters the model feature matrix (Risk R13)",
        },
    },
    "ntl_feature": {
        "leading": {
            "transform": "dlog_or_diff",
            "lag": 1,
            "causal": True,
            "note": "NTL month t -> predict return month t+1 (release-lag, F12)",
        },
        "nowcast": {
            "transform": "dlog_or_diff",
            "lag": 0,
            "causal": True,
            "note": "contemporaneous NTL allowed for nowcast (no forward target)",
        },
    },
}

# Documented alignment constants (consumed by Phase B).
CALENDAR = "tz-naive month-end DatetimeIndex; left-join to master grid; no ffill"
LEADING_LAG_MONTHS = 1
NOWCAST_LAG_MONTHS = 0
MASTER_GRID_START = "2013-01-31"
MASTER_GRID_END = "2024-12-31"
