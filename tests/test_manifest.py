"""Tests for the run-manifest superset writer (S7 / Appendix A.5)."""

from __future__ import annotations

import json

from ntl_etf.utils.manifest import write_manifest


def test_write_manifest_superset(tmp_path):
    run_dir = tmp_path / "run_x"
    out = write_manifest(run_dir, {"a": 1}, 7, model="momentum", variant="scratch")
    m = json.loads(out.read_text(encoding="utf-8"))
    # git_sha present (sha or "UNKNOWN"); seed echoed; capabilities + packages populated
    assert m["git_sha"]  # non-empty
    assert m["seed"] == 7
    assert isinstance(m["capabilities"], dict)
    assert "numpy" in m["packages"] and "torch" in m["packages"]
    assert isinstance(m["git_dirty"], bool)
    # A.5 required reconciliation keys
    assert m["scaler_fit_on"] == "train"
    assert "mamba_impl" in m
    assert "gpu_stages_skipped" in m
    assert m["config_hash"]  # stable hash present
    assert m["model"] == "momentum" and m["variant"] == "scratch"
