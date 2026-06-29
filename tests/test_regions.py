"""Validate configs/regions.yaml against the unified Appendix A.6 schema (Task N4)."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SPDR = {"XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"}


def _load():
    return yaml.safe_load((REPO_ROOT / "configs" / "regions.yaml").read_text(encoding="utf-8"))


def test_schema_and_fields():
    doc = _load()
    regions = doc["regions"]
    ids = [r["id"] for r in regions]
    assert len(ids) == len(set(ids)), "region ids must be unique"
    for r in regions:
        for key in ("id", "name", "bbox", "candidate_sectors", "rationale"):
            assert key in r, f"{r.get('id')} missing {key}"
        assert isinstance(r["candidate_sectors"], list) and r["candidate_sectors"]
        assert set(r["candidate_sectors"]) <= SPDR


def test_bbox_sanity():
    for r in _load()["regions"]:
        w, s, e, n = r["bbox"]
        assert w < e and s < n, f"{r['id']} bbox not W<E,S<N"
        assert -180 <= w <= 180 and -180 <= e <= 180
        assert -90 <= s <= 90 and -90 <= n <= 90


def test_full_sector_coverage_and_multiregion():
    regions = _load()["regions"]
    covered = set()
    counts = {t: 0 for t in SPDR}
    for r in regions:
        for sec in r["candidate_sectors"]:
            covered.add(sec)
            counts[sec] += 1
    assert covered == SPDR, f"missing sectors: {SPDR - covered}"
    # XLI/XLE/XLY/XLK/XLV must have >=2 candidate regions so H2 is not n=1 (Risk R7)
    for t in ("XLI", "XLE", "XLY", "XLK", "XLV"):
        assert counts[t] >= 2, f"{t} needs >=2 regions, has {counts[t]}"


def test_hypothesis_pairs_present():
    doc = _load()
    hp = doc["hypothesis_pairs"]
    assert hp["H2_multiregion"]["sector"] == "XLI"
    assert len(hp["H2_multiregion"]["regions"]) >= 2
    assert hp["H3_singleregion"]["sector"] == "XLE"
    assert hp["H3_singleregion"]["regions"] == ["permian_basin"]


def test_anchors_present():
    anchors = [r for r in _load()["regions"] if r.get("anchor")]
    assert len(anchors) >= 5
    ids = {r["id"] for r in anchors}
    assert {"pearl_river_delta", "yangtze_river_delta", "permian_basin"} <= ids
