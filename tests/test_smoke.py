"""Scaffold smoke tests (MS0): package imports, capability detection, seeding, manifest."""

from __future__ import annotations

import json

import numpy as np

from marsseg.utils.capabilities import detect
from marsseg.utils.manifest import write_manifest
from marsseg.utils.seed import set_seed


def test_package_imports():
    import marsseg

    assert marsseg.__version__ == "0.1.0"


def test_capabilities_profile():
    caps = detect()
    assert caps.profile in ("windows_cpu", "gpu_full")
    # boolean flags present
    for f in ("cuda", "smp", "transformers", "sam", "timm"):
        assert isinstance(getattr(caps, f), bool)


def test_seed_reproducible():
    set_seed(1414)
    a = np.random.rand(5)
    set_seed(1414)
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)


def test_manifest_superset(tmp_path):
    p = write_manifest(tmp_path / "run", {"k": "v"}, seed=1414, model="unet", task="seg")
    m = json.loads(p.read_text(encoding="utf-8"))
    assert m["seed"] == 1414 and m["model"] == "unet"
    assert "git_sha" in m and "capabilities" in m and "config_hash" in m
