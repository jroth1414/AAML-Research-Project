"""Masked-pretraining tests (Task M8)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ntl_etf.models import patchtst, pretrain
from ntl_etf.models.base import ModelConfig


def _ntl_wide(n_regions=4, n_months=120):
    idx = pd.date_range("2013-01-31", periods=n_months, freq="ME")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {f"r{i}": np.cumsum(rng.normal(0, 1, n_months)) + 50 for i in range(n_regions)}, index=idx
    )


def test_unlabeled_windows_shape():
    X = pretrain.build_unlabeled_windows(_ntl_wide(), L=12)
    assert X.ndim == 3 and X.shape[1:] == (12, 1)
    assert np.isfinite(X).all()


def test_pretraining_reduces_recon_loss(tmp_path):
    mc = ModelConfig(name="patchtst", L=12, H=1, extra={"patch_len": 6, "d_model": 32})
    res = pretrain.pretrain_patchtst(_ntl_wide(), mc, str(tmp_path / "p.pt"), steps=300, seed=0)
    assert res["recon_loss"][-1] < res["recon_loss"][0]  # learning happens


def test_param_parity_scratch_vs_pretrained(tmp_path):
    mc = ModelConfig(name="patchtst", L=12, H=1, extra={"patch_len": 6, "d_model": 32})
    ckpt = str(tmp_path / "p.pt")
    pretrain.pretrain_patchtst(_ntl_wide(), mc, ckpt, steps=50, seed=0)
    scratch = patchtst.patchtst_factory(mc)
    mc.extra["pretrained_ckpt"] = ckpt
    pre = pretrain.pretrained_patchtst_factory(mc)
    assert sum(p.numel() for p in scratch.parameters()) == sum(p.numel() for p in pre.parameters())
