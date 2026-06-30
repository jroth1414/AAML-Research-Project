"""Hand-computed metric tests (Task E3)."""

from __future__ import annotations

import numpy as np
import pytest

from ntl_etf.eval import metrics as M


def test_dir_acc():
    assert M.directional_accuracy([1, -1, 2, -2], [0.5, -0.5, 1, 1]) == pytest.approx(0.75)


def test_mse_mae():
    assert M.mse([0, 2], [0, 0]) == pytest.approx(2.0)
    assert M.mae([0, 2], [0, 0]) == pytest.approx(1.0)


def test_sharpe():
    assert np.isnan(M.annualized_sharpe([0.01] * 12))  # std 0
    r = [0.01, 0.02, 0.03]
    assert M.annualized_sharpe(r, 12) == pytest.approx(np.sqrt(12) * np.mean(r) / np.std(r, ddof=1))


def test_strategy_turnover_cost():
    sr = M.strategy_returns([0.10, 0.10, -0.10], [1, -1, -1], cost_one_way=0.0010)
    np.testing.assert_array_equal(sr["positions"], [1, -1, -1])
    np.testing.assert_array_equal(sr["turnover"], [1, 2, 0])
    np.testing.assert_allclose(sr["gross"], [0.10, -0.10, 0.10])
    np.testing.assert_allclose(sr["net"], [0.099, -0.102, 0.10])


def test_nowcast_r2():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert M.nowcast_r2(y, y) == pytest.approx(1.0)
    assert M.nowcast_r2(y, [y.mean()] * 4) == pytest.approx(0.0)
    assert M.nowcast_r2(y, [10, 10, 10, 10]) < 0


def test_pearson():
    assert M.pearson_corr([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert np.isnan(M.pearson_corr([1, 1, 1], [1, 2, 3]))


def test_dir_acc_pvalue():
    # all correct -> strong evidence dir_acc > 0.5
    assert M.directional_accuracy_pvalue([1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]) < 0.05
    # exactly half -> not significant
    assert M.directional_accuracy_pvalue([1, 1, -1, -1], [1, -1, 1, -1]) > 0.3
