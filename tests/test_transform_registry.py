"""Transform-registry contract tests (Task F10)."""

from __future__ import annotations

from ntl_etf.data.transform_registry import (
    LEADING_LAG_MONTHS,
    NOWCAST_LAG_MONTHS,
    TRANSFORM_REGISTRY,
)


def test_registry_imports_and_no_double_diff():
    assert TRANSFORM_REGISTRY["etf_return"]["leading"]["transform"] == "none"


def test_ntl_feature_lags():
    assert TRANSFORM_REGISTRY["ntl_feature"]["leading"]["lag"] == 1
    assert TRANSFORM_REGISTRY["ntl_feature"]["nowcast"]["lag"] == 0
    assert LEADING_LAG_MONTHS == 1 and NOWCAST_LAG_MONTHS == 0


def test_ip_target_causal_form():
    assert TRANSFORM_REGISTRY["ip_target"]["nowcast"]["col"] == "value_dlog"
    assert TRANSFORM_REGISTRY["ip_target"]["nowcast"]["causal"] is True
    assert TRANSFORM_REGISTRY["ip_target"]["descriptive"]["causal"] is False
