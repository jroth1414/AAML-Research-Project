"""Pre-registration: load the frozen H0-H6 rules + comparison families (Phase D / Task E5).

The decision thresholds live in ``configs/hypotheses.yaml`` (owner E); this module loads them and
defines the pre-registered pairwise comparison FAMILIES so the agent cannot data-snoop. Changing
these after first results requires a logged config bump (experiments/PREREG.md).
"""

from __future__ import annotations

from pathlib import Path

import yaml

HYP_PATH = "configs/hypotheses.yaml"


def load_hypotheses(path: str = HYP_PATH) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


_H = load_hypotheses()
ALPHA = float(_H["alpha"])
PRIMARY_LOSS = _H["primary_loss"]
DL_MODELS = list(_H["dl_models"])
BASELINES_FOR_H1 = list(_H["baselines_for_h1"])
CORRECTION_PRIMARY = _H["correction_primary"]
CORRECTION_SECONDARY = _H["correction_secondary"]

# Pre-registered families: list of pairwise comparisons. Each pair is (model_a, model_b); the test
# is run on the intersection of dates within (task, scope, stratum, horizon). win => Holm-significant.
FAMILIES = [
    {
        "name": "A_signal_existence",
        "hypothesis": "H1",
        "task": "leading",
        "horizon": 1,
        "stratum": "all",
        "scope": "POOLED",
        "pairs": [
            (dl, b) for dl in ["patchtst", "itransformer", "mamba"] for b in BASELINES_FOR_H1
        ],
    },
    {
        "name": "B_architecture",
        "hypothesis": "H2/H3",
        "task": "leading",
        "horizon": 1,
        "stratum": "by_region_class",  # multi_region (H2) and single_region (H3)
        "scope": "POOLED",
        "pairs": [("itransformer", "patchtst")],
    },
    {
        "name": "C_disruption",
        "hypothesis": "H4",
        "task": "leading",
        "horizon": 1,
        "stratum": "disruption",
        "scope": "POOLED",
        "pairs": [("mamba", "patchtst"), ("mamba", "itransformer")],
    },
    {
        "name": "D_transfer",
        "hypothesis": "H6a",
        "task": "leading",
        "horizon": 1,
        "stratum": "all",
        "scope": "POOLED",
        "pairs": [("patchtst_pretrained", "patchtst")],
    },
    {
        # H6b reference: foundation (Chronos zero-shot on target return history) vs the naive
        # return-history baseline (momentum). Both are NO-NTL forecasters (Risk R23 split).
        "name": "E_foundation",
        "hypothesis": "H6b",
        "task": "leading",
        "horizon": 1,
        "stratum": "all",
        "scope": "POOLED",
        "pairs": [("chronos", "momentum")],
    },
]
