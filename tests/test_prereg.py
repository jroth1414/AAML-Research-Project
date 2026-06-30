"""Pre-registration tests (Task E5)."""

from __future__ import annotations

from ntl_etf.eval import prereg


def test_alpha_and_families():
    assert prereg.ALPHA == 0.10
    assert len(prereg.FAMILIES) >= 4


def test_h1_family_has_six_pairs():
    fam_a = next(f for f in prereg.FAMILIES if f["name"] == "A_signal_existence")
    # {patchtst, itransformer, mamba} x {momentum, dlinear} = 6 pairs
    assert len(fam_a["pairs"]) == 6


def test_pairs_reference_known_models():
    known = set(prereg.DL_MODELS) | {"momentum", "patchtst_pretrained", "chronos"}
    for fam in prereg.FAMILIES:
        for a, b in fam["pairs"]:
            assert a in known and b in known


def test_hypotheses_yaml_has_all_keys():
    h = prereg.load_hypotheses()["hypotheses"]
    for k in ["H0", "H1", "H2", "H3", "H4", "H5", "H6a", "H6b"]:
        assert k in h and "statement" in h[k]
