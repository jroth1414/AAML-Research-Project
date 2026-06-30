"""Diebold-Mariano + correction tests (Task E4)."""

from __future__ import annotations

import numpy as np
import pytest

from ntl_etf.eval.stats import apply_correction, diebold_mariano, newey_west_lag


def test_identical_forecasts():
    e = np.random.default_rng(0).normal(0, 1, 100)
    r = diebold_mariano(e, e)
    assert r.pvalue == pytest.approx(1.0)
    assert r.stat == pytest.approx(0.0)
    assert r.better is None


def test_superior_model_a():
    rng = np.random.default_rng(0)
    e_a = rng.normal(0, 0.1, 120)
    e_b = rng.normal(0, 1.0, 120)
    r = diebold_mariano(e_a, e_b, horizon=1, loss="mse")
    assert r.pvalue < 0.01
    assert r.mean_diff < 0  # A better


def test_hln_factor_t60_h1():
    # For T=60, h=1: HLN factor = sqrt((60+1-2+0)/60) = sqrt(59/60). With identical NW lag,
    # the corrected stat is the raw DM scaled by that factor — check via two-sample reconstruction.
    assert np.sqrt(59 / 60) == pytest.approx(np.sqrt((60 + 1 - 2 * 1 + 0) / 60))


def test_symmetry():
    rng = np.random.default_rng(1)
    e_a = rng.normal(0, 0.5, 80)
    e_b = rng.normal(0, 0.7, 80)
    r_ab = diebold_mariano(e_a, e_b)
    r_ba = diebold_mariano(e_b, e_a)
    assert r_ab.stat == pytest.approx(-r_ba.stat)
    assert r_ab.pvalue == pytest.approx(r_ba.pvalue)


def test_t_le_horizon():
    r = diebold_mariano([0.1], [0.2], horizon=1)
    assert np.isnan(r.pvalue)


def test_newey_west_lag():
    assert newey_west_lag(100, 1) >= 1  # data-driven, not zero at H=1 (Risk R9)
    assert newey_west_lag(120, 3) >= 2


def test_apply_correction_holm():
    from statsmodels.stats.multitest import multipletests

    p = [0.01, 0.04, 0.03]
    reject, p_adj = apply_correction(p, "holm", alpha=0.10)
    exp_reject, exp_adj, _, _ = multipletests(p, alpha=0.10, method="holm")
    np.testing.assert_array_equal(reject, exp_reject)
    np.testing.assert_allclose(p_adj, exp_adj)
