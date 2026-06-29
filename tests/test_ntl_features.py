"""Offline NTL feature-math, masking, geometry, and release-lag tests (Task N9). Zero network."""

from __future__ import annotations

import numpy as np
import pytest

from ntl_etf.data import ntl


def _clean_2x3():
    # composite [[1,2,fill],[3,4,5]]; quality marks the '5' (at [1,2]) as poor (flag 1)
    comp = np.array([[1.0, 2.0, 65535.0], [3.0, 4.0, 5.0]])
    qual = np.array([[0, 0, 0], [0, 0, 1]])
    return ntl.clean_radiance_array(comp, qual, fill_values=(65535,), drop_quality_flags=(1, 2))


def test_masking_drops_fill_and_bad_quality():
    clean = _clean_2x3()
    assert clean.count() == 4  # {1,2,3,4}
    assert set(clean.compressed().tolist()) == {1.0, 2.0, 3.0, 4.0}


def test_feature_math():
    feats = ntl.compute_ntl_features(_clean_2x3(), {"lit_threshold": 2.5})
    assert feats["ntl_mean"] == pytest.approx(2.5)
    assert feats["ntl_median"] == pytest.approx(2.5)
    assert feats["ntl_std"] == pytest.approx(1.1180339887, abs=1e-3)  # population std
    assert feats["ntl_sum"] == pytest.approx(10.0)
    assert feats["n_valid_pixels"] == 4
    assert feats["frac_masked"] == pytest.approx(2 / 6)
    assert feats["ntl_lit_count"] == 2  # {3,4} > 2.5
    assert feats["ntl_lit_frac"] == pytest.approx(0.5)


def test_scale_offset_applied():
    comp = np.array([[10.0, 20.0]])
    clean = ntl.clean_radiance_array(comp, None, scale=0.1, offset=0.0)
    feats = ntl.compute_ntl_features(clean)
    assert feats["ntl_mean"] == pytest.approx(1.5)  # (1.0 + 2.0)/2
    assert feats["ntl_sum"] == pytest.approx(3.0)


def test_all_masked_returns_nan():
    comp = np.full((2, 2), 65535.0)
    clean = ntl.clean_radiance_array(comp, None, fill_values=(65535,))
    feats = ntl.compute_ntl_features(clean)
    assert np.isnan(feats["ntl_mean"])
    assert feats["frac_masked"] == 1.0
    assert feats["n_valid_pixels"] == 0


def test_release_lag():
    s = ntl.stamp_release_date(2013, 3, lag_days=45)
    assert s["date"].year == 2013 and s["date"].month == 3 and s["date"].day == 31
    assert s["data_release_date"].month == 5  # 03-31 + 45d -> mid-May
    assert s["data_release_date"] > s["date"]


def test_determinism():
    a = ntl.compute_ntl_features(_clean_2x3(), {"lit_threshold": 2.5})
    b = ntl.compute_ntl_features(_clean_2x3(), {"lit_threshold": 2.5})
    assert a == b


# --- geometry helpers ---
def test_tile_hv_from_name():
    assert ntl.tile_hv_from_name("VNP46A3.A2020153.h26v06.001.h5") == (26, 6)


def test_tiles_for_bbox_single():
    # Singapore (~103.8E, 1.3N) -> h=28 (-180+10*28=100..110), v=8 (90-10*9=0..10)
    assert ntl.tiles_for_bbox([103.6, 1.2, 104.1, 1.5]) == [(28, 8)]


def test_granule_month():
    # A2020153 = day-of-year 153 of 2020 -> June 1 (monthly composite stamp)
    assert ntl.granule_month("VNP46A3.A2020153.h28v08.002.h5") == (2020, 6)
    assert ntl.granule_month("VNP46A3.A2020122.h28v08.002.h5") == (2020, 5)


def test_quality_default_keeps_all_retrieved_classes():
    # VERIFIED behaviour: keep quality {0,1,2}; only 255 (no-retrieval) is dropped by default.
    comp = np.array([[10.0, 20.0, 30.0, 40.0]])
    qual = np.array([[0, 1, 2, 255]])
    clean = ntl.clean_radiance_array(
        comp, qual, drop_quality_flags=ntl.DROP_QUALITY_FLAGS, fill_values=(-999.9,)
    )
    assert clean.count() == 3  # 255 dropped, {0,1,2} kept
    assert set(clean.compressed().tolist()) == {10.0, 20.0, 30.0}


def test_clip_to_bbox():
    from rasterio.transform import from_origin

    data = np.arange(100, dtype="float32").reshape(10, 10)
    tr = from_origin(0, 10, 1, 1)  # covers lon[0,10], lat[0,10], 1deg/px
    sub, _ = ntl.clip_array_to_bbox(data, tr, [2, 2, 5, 5])
    assert sub.shape == (3, 3)
    np.testing.assert_array_equal(sub, data[5:8, 2:5])


def test_mosaic_two_tiles():
    from rasterio.transform import from_origin

    a = np.ones((10, 10), "float32")
    b = np.full((10, 10), 2.0, "float32")
    tr_a = from_origin(0, 10, 1, 1)
    tr_b = from_origin(10, 10, 1, 1)  # adjacent to the east
    mosaic, _ = ntl.mosaic_tiles([(a, tr_a), (b, tr_b)])
    assert mosaic.shape == (10, 20)
    assert mosaic[0, 0] == 1.0 and mosaic[0, -1] == 2.0


def test_resolve_token(monkeypatch):
    monkeypatch.delenv("EARTHDATA_TOKEN", raising=False)
    monkeypatch.delenv("BLACKMARBLE_TOKEN", raising=False)
    with pytest.raises(RuntimeError):
        ntl.resolve_earthdata_token()
    monkeypatch.setenv("EARTHDATA_TOKEN", "abc123")
    assert ntl.resolve_earthdata_token() == "abc123"
