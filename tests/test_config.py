"""Tests for the config loader + secret handling (S5)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from ntl_etf.utils import config as cfgmod

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_load_yaml_and_override(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        textwrap.dedent("""
            train:
              lr: 1.0e-3
              batch: 16
            model:
              name: patchtst
            """),
        encoding="utf-8",
    )
    cfg = cfgmod.load_yaml(p)
    cfg = cfgmod.apply_overrides(cfg, ["train.lr=2e-4", "model.name=mamba"])
    assert cfg["train"]["lr"] == pytest.approx(2e-4)
    assert isinstance(cfg["train"]["lr"], float)
    assert cfg["train"]["batch"] == 16  # untouched by override (deep merge / partial set)
    assert cfg["model"]["name"] == "mamba"


def test_deep_merge_base_then_main(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("a:\n  x: 1\n  y: 2\n", encoding="utf-8")
    main = tmp_path / "main.yaml"
    main.write_text("a:\n  y: 99\nb: 3\n", encoding="utf-8")
    cfg = cfgmod.load_config(main, base_paths=[str(base)])
    assert cfg["a"]["x"] == 1  # preserved from base
    assert cfg["a"]["y"] == 99  # overridden by main
    assert cfg["b"] == 3


def test_require_secret_raises(monkeypatch):
    monkeypatch.delenv("DOES_NOT_EXIST", raising=False)
    with pytest.raises(RuntimeError) as exc:
        cfgmod.require_secret("DOES_NOT_EXIST")
    assert "DOES_NOT_EXIST" in str(exc.value)
    assert ".env.example" in str(exc.value)


def test_env_example_lists_secrets():
    txt = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    for key in [
        "EARTHDATA_TOKEN",
        "BLACKMARBLE_TOKEN",
        "EARTHDATA_USERNAME",
        "EARTHDATA_PASSWORD",
        "FRED_API_KEY",
    ]:
        assert key in txt


def test_experiment_schema_keys():
    cfg = cfgmod.load_yaml(REPO_ROOT / "configs" / "experiment.example.yaml")
    for key in ["run", "data", "split", "model", "train", "eval", "tracking"]:
        assert key in cfg
