"""Model shape, overfit, and determinism tests (Tasks M15-M17) + base contract (M1)."""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from ntl_etf.models.base import PRED_COLUMNS, ModelConfig, write_predictions
from ntl_etf.models.dlinear import dlinear_factory
from ntl_etf.models.itransformer import itransformer_factory
from ntl_etf.models.mamba import mamba_factory, mamba_impl
from ntl_etf.models.patchtst import patchtst_factory
from ntl_etf.utils.seed import set_seed

CI_FACTORIES = {"dlinear": dlinear_factory, "patchtst": patchtst_factory, "mamba": mamba_factory}


def _cfg(H=1, **extra):
    return ModelConfig(name="m", L=12, H=H, extra=extra)


@pytest.mark.parametrize("name", list(CI_FACTORIES))
@pytest.mark.parametrize("H", [1, 3])
def test_ci_shapes(name, H):
    mod = CI_FACTORIES[name](_cfg(H))
    out = mod(torch.randn(8, 12, 1))
    assert tuple(out.shape) == (8, H)


@pytest.mark.parametrize("H", [1, 3])
def test_itransformer_shape(H):
    mod = itransformer_factory(_cfg(H))
    out = mod(torch.randn(8, 12, 3), torch.ones(8, 3, dtype=torch.bool))
    assert tuple(out.shape) == (8, H)


def _overfit(mod, x, y, steps=1200, lr=1e-2, mask=None):
    opt = torch.optim.AdamW(mod.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    last = None
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(mod(x) if mask is None else mod(x, mask), y)
        loss.backward()
        opt.step()
        last = float(loss.item())
    return last


@pytest.mark.parametrize("name", ["dlinear", "patchtst"])
def test_overfit_ci(name):
    set_seed(0)
    x = torch.randn(8, 12, 1)
    y = torch.randn(8, 1)
    assert _overfit(CI_FACTORIES[name](_cfg(dropout=0.0)), x, y) < 1e-2


def test_overfit_mamba_fallback():
    set_seed(0)
    assert mamba_impl() == "fallback"  # CPU profile
    x = torch.randn(8, 12, 1)
    y = torch.randn(8, 1)
    assert _overfit(mamba_factory(_cfg()), x, y, steps=600) < 5e-2


def test_overfit_itransformer():
    set_seed(0)
    mod = itransformer_factory(_cfg(dropout=0.0))
    x = torch.randn(8, 12, 3)
    mask = torch.ones(8, 3, dtype=torch.bool)
    y = torch.randn(8, 1)
    assert _overfit(mod, x, y, mask=mask) < 1e-2


@pytest.mark.parametrize("name", list(CI_FACTORIES))
def test_determinism(name):
    set_seed(1414)
    a = CI_FACTORIES[name](_cfg())(torch.ones(4, 12, 1))
    set_seed(1414)
    b = CI_FACTORIES[name](_cfg())(torch.ones(4, 12, 1))
    assert torch.allclose(a, b, atol=1e-6)


def test_base_contract(tmp_path):
    mc = ModelConfig(name="patchtst", task="leading", variant="pretrained")
    assert mc.target_kind == "return" and mc.pretrained is True
    rows = [
        {
            "model": "m",
            "variant": "scratch",
            "pretrained": False,
            "task": "leading",
            "target_kind": "return",
            "etf": "XLE",
            "horizon": 1,
            "fold": 0,
            "split": "test",
            "date": pd.Timestamp("2019-01-31"),
            "y_true": 0.01,
            "y_pred": 0.02,
            "seed": 1414,
        }
    ]
    p = write_predictions(rows, tmp_path / "run")
    df = pd.read_parquet(p)
    assert list(df.columns) == PRED_COLUMNS
