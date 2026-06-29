"""Shared pytest fixtures. CI must NEVER call yfinance/FRED/Earthdata — all fixtures are
synthetic and offline. Network/credential-gated tests are marked and skipped by default.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def fake_config() -> dict:
    """A minimal experiment-config-shaped dict."""
    return {
        "run": {"name": "test", "seed": 1414, "task": "leading", "horizon": 1},
        "data": {"study_start": "2013-01-01", "study_end": "2024-12-31"},
        "split": {"scheme": "walk_forward", "min_train_months": 60, "step_months": 12},
        "model": {"name": "momentum", "params": {}},
        "train": {"epochs": 1, "batch_size": 16, "lr": 1e-3, "normalize": "train_only"},
        "eval": {"metrics": ["mse"], "dm_test": True, "alpha": 0.10},
        "tracking": {"mlflow": False, "wandb": False},
    }


@pytest.fixture
def fake_panel() -> pd.DataFrame:
    """A tiny month-end panel: a handful of series over ~24 months (offline)."""
    dates = pd.date_range("2018-01-31", periods=24, freq="ME")
    rows = []
    rng = np.random.default_rng(0)
    for series_idx, (sector, region) in enumerate(
        [("XLE", "permian_basin"), ("XLI", "pearl_river_delta"), ("XLI", "yangtze_river_delta")]
    ):
        ntl = np.cumsum(rng.normal(0, 1, len(dates))) + 100.0
        ret = rng.normal(0, 0.04, len(dates))
        rows.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "sector": sector,
                    "region_id": region,
                    "series_idx": series_idx,
                    "ntl_value": ntl.astype("float32"),
                    "target_leading": ret.astype("float32"),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)
