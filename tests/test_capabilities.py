"""Tests for capability detection + skip-and-log contract (S4)."""

from __future__ import annotations

import importlib.util
import logging

from ntl_etf.utils.capabilities import Capabilities, as_dict, detect


def test_detect_returns_capabilities_no_raise():
    caps = detect()
    assert isinstance(caps, Capabilities)
    assert caps.profile in {"windows_cpu", "gpu_full", "unknown"}


def test_mamba_flag_matches_definition():
    import torch

    expected = (importlib.util.find_spec("mamba_ssm") is not None) and torch.cuda.is_available()
    assert detect().mamba_ssm == expected


def test_as_dict_has_profile():
    d = as_dict()
    assert "profile" in d and "cuda" in d and "mamba_ssm" in d


def _fake_run_gated_model(caps: Capabilities, logger: logging.Logger) -> dict:
    """A tiny stand-in for the Phase-D skip-and-log contract."""
    if not caps.mamba_ssm:
        logger.warning("mamba_ssm unavailable on profile=%s; skipping", caps.profile)
        return {"status": "skipped", "reason": "mamba_ssm_unavailable"}
    return {"status": "ok"}


def test_skip_and_log_contract(caplog):
    caps = detect()
    with caplog.at_level(logging.WARNING):
        row = _fake_run_gated_model(caps, logging.getLogger("test"))
    if not caps.mamba_ssm:
        assert row["status"] == "skipped"
        assert any("unavailable" in r.message for r in caplog.records)
