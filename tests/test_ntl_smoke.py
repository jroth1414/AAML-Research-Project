"""Credential-gated NTL integration smoke (Task N10). Skipped without Earthdata credentials and
excluded from the default pytest run (``-m "not integration"``)."""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.integration


def _has_creds() -> bool:
    from dotenv import dotenv_values

    env = {**dotenv_values(".env"), **os.environ}
    return bool((env.get("EARTHDATA_TOKEN") or env.get("BLACKMARBLE_TOKEN") or "").strip())


@pytest.mark.skipif(not _has_creds(), reason="no Earthdata credentials")
def test_download_one_month_small_roi(tmp_path):
    import yaml

    from ntl_etf.data.ntl import build_ntl_features
    from ntl_etf.utils.config import load_env

    load_env()
    # single small single-tile ROI (Singapore port), mid-mission month
    region = {
        "id": "singapore_port",
        "name": "Singapore Port",
        "bbox": [103.6, 1.2, 104.1, 1.5],
        "candidate_sectors": ["XLE"],
        "rationale": "smoke",
    }
    regions_yaml = tmp_path / "regions.yaml"
    regions_yaml.write_text(
        yaml.safe_dump({"schema_version": 1, "regions": [region]}), encoding="utf-8"
    )
    df = build_ntl_features(
        regions_yaml=regions_yaml,
        ntl_cfg="configs/ntl.yaml",
        start="2020-06",
        end="2020-06",
        out_parquet=tmp_path / "smoke.parquet",
        manifest=tmp_path / "manifest.json",
    )
    assert len(df) == 1
    assert df.loc[0, "n_valid_pixels"] > 0
    assert df.loc[0, "ntl_mean"] > 0
    assert df.loc[0, "data_release_date"] > df.loc[0, "date"]
