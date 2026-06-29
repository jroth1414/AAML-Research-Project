"""Tests for the metrics results store + no-op tracker (S8 / Appendix A.4)."""

from __future__ import annotations

import pandas as pd

from ntl_etf.utils.results import RESULT_COLUMNS, append_results
from ntl_etf.utils.tracking import tracker


def _row(metric, value, fold):
    return {
        "run_id": "r1",
        "model": "momentum",
        "variant": "scratch",
        "task": "leading",
        "scope": "POOLED",
        "fold": fold,
        "stratum": "all",
        "metric": metric,
        "value": value,
        "status": "ok",
        "profile": "windows_cpu",
        "seed": 1414,
        "git_sha": "abc",
        "config_hash": "h1",
    }


def test_append_accumulates_and_schema(tmp_path):
    pq = tmp_path / "results_store.parquet"
    csv = tmp_path / "results_store.csv"
    append_results([_row("mse", 0.1, 0), _row("mae", 0.2, 0)], pq, csv)
    append_results([_row("mse", 0.15, 1)], pq, csv)
    df = pd.read_parquet(pq)
    assert list(df.columns) == RESULT_COLUMNS
    assert len(df) == 3
    assert pq.exists() and csv.exists()


def test_tracker_noop():
    with tracker({}, "x") as h:
        assert h is None
