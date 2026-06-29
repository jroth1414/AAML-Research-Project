"""Imports every lightweight ntl_etf.utils submodule + logging-handler invariant (S9/S10)."""

from __future__ import annotations

import importlib

import ntl_etf
from ntl_etf.utils.logging import get_logger

UTILS_MODULES = [
    "ntl_etf.utils.capabilities",
    "ntl_etf.utils.config",
    "ntl_etf.utils.manifest",
    "ntl_etf.utils.results",
    "ntl_etf.utils.tracking",
    "ntl_etf.utils.seed",
    "ntl_etf.utils.logging",
]


def test_version():
    assert ntl_etf.__version__ == "0.1.0"


def test_import_utils_submodules():
    for m in UTILS_MODULES:
        importlib.import_module(m)


def test_logger_no_duplicate_handlers():
    a = get_logger("ntl_etf_test_dup")
    b = get_logger("ntl_etf_test_dup")
    assert a is b
    assert len(a.handlers) == 1
