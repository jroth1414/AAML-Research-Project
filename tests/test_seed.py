"""Tests for deterministic seeding (S9)."""

from __future__ import annotations

import numpy as np

from ntl_etf.utils.seed import DEFAULT_SEED, SEED_SET, set_seed


def test_numpy_reproducible():
    set_seed(1)
    a = np.random.rand(3)
    set_seed(1)
    b = np.random.rand(3)
    np.testing.assert_array_equal(a, b)


def test_torch_reproducible():
    import torch

    set_seed(1)
    a = torch.rand(3)
    set_seed(1)
    b = torch.rand(3)
    assert torch.equal(a, b)


def test_seed_constants():
    assert DEFAULT_SEED == 1414
    assert SEED_SET == [1414, 1415, 1416, 1417, 1418]
