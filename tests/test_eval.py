"""Eval results-loader, stratify, DM-suite, and verdict tests (Tasks E1,E2,E6,E7,E8,E12)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ntl_etf.eval import prereg, results, stats, verdict
from ntl_etf.models.base import PRED_COLUMNS, write_predictions


def _pred_rows(model, etf="XLE", n=12, mse_scale=0.04):
    rng = np.random.default_rng(abs(hash(model)) % 2**32)
    dates = pd.date_range("2019-01-31", periods=n, freq="ME")
    yt = rng.normal(0, 0.04, n)
    return [
        {
            "model": model,
            "variant": "scratch",
            "pretrained": False,
            "task": "leading",
            "target_kind": "return",
            "etf": etf,
            "horizon": 1,
            "fold": 0,
            "split": "test",
            "date": d,
            "y_true": float(yt[i]),
            "y_pred": float(yt[i] + rng.normal(0, mse_scale)),
            "seed": 1414,
        }
        for i, d in enumerate(dates)
    ]


def test_load_predictions_dedup_raises(tmp_path):
    rows = _pred_rows("momentum")
    write_predictions(rows + rows, tmp_path / "momentum__x")  # duplicates
    with pytest.raises(ValueError):
        results.load_predictions(str(tmp_path))


def test_load_predictions_ok(tmp_path):
    write_predictions(_pred_rows("momentum"), tmp_path / "momentum__x")
    df = results.load_predictions(str(tmp_path))
    assert list(df.columns)[: len(PRED_COLUMNS)] == PRED_COLUMNS or "run_id" in df.columns
    assert len(df) == 12


def _fab_results(patchtst_mse, mom_mse=0.005, dlin_mse=0.004, dir_acc=0.6, dir_p=0.01):
    rows = []

    def add(model, metric, value, stratum="all"):
        rows.append(
            {
                "model": model,
                "variant": "scratch",
                "task": "leading",
                "scope": "POOLED",
                "fold": -1,
                "stratum": stratum,
                "metric": metric,
                "value": value,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "seed": 1414,
            }
        )

    add("patchtst", "mse", patchtst_mse)
    add("patchtst", "dir_acc", dir_acc)
    add("patchtst", "dir_acc_pvalue", dir_p)
    add("momentum", "mse", mom_mse)
    add("dlinear", "mse", dlin_mse)
    return pd.DataFrame(rows)


def _fab_dm(win):
    rows = []
    for b in ["momentum", "dlinear"]:
        rows.append(
            {
                "family": "A_signal_existence",
                "hypothesis": "H1",
                "task": "leading",
                "scope": "POOLED",
                "stratum": "all",
                "horizon": 1,
                "model_a": "patchtst",
                "model_b": b,
                "loss": "mse",
                "dm_stat": -3.0,
                "p_raw": 0.001,
                "p_holm": 0.002,
                "p_bh": 0.002,
                "mean_diff": -0.001,
                "n": 60,
                "win": win,
            }
        )
    return pd.DataFrame(rows)


def test_verdict_h1_support():
    v = verdict.decide_hypotheses(_fab_results(0.001), _fab_dm("patchtst"), prereg)
    assert v["H1"]["verdict"] == "support"
    assert v["H1"]["winner"] == "patchtst"


def test_verdict_h0_holds():
    v = verdict.decide_hypotheses(_fab_results(0.006), _fab_dm("none"), prereg)
    assert v["H1"]["verdict"] == "reject"
    assert "H0 holds" in v["H0_note"]


def test_verdict_deferred_h6():
    v = verdict.decide_hypotheses(_fab_results(0.001), _fab_dm("patchtst"), prereg)
    assert v["H6a"]["verdict"] == "deferred"
    assert v["H6b"]["verdict"] == "deferred"


def test_dm_suite_runs():
    preds = pd.concat(
        [
            pd.DataFrame(_pred_rows("patchtst", mse_scale=0.001)),
            pd.DataFrame(_pred_rows("momentum", mse_scale=0.05)),
        ],
        ignore_index=True,
    )
    preds["region_class"] = "all"
    preds["disruption"] = False
    fam = [
        {
            "name": "A_signal_existence",
            "hypothesis": "H1",
            "task": "leading",
            "horizon": 1,
            "stratum": "all",
            "scope": "POOLED",
            "pairs": [("patchtst", "momentum")],
        }
    ]
    dm = stats.run_dm_suite(preds, fam)
    assert len(dm) == 1 and "win" in dm.columns
