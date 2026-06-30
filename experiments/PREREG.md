# Pre-Registration Freeze (Phase D / Task E12)

This freezes the comparison families and the H0–H6 decision rules used by `analyze_results.py`
**before** the headline results are interpreted. The decision thresholds live in
[`configs/hypotheses.yaml`](../configs/hypotheses.yaml) (owner: Phase E5) and are loaded by
`src/ntl_etf/eval/prereg.py`; `analyze_results.py` may not retroactively pick favorable cuts.

- **Significance level:** α = 0.10. **Primary loss:** squared error (MSE).
- **Primary correction:** Holm (FWER) *within each family*; **secondary (reported):** Benjamini–Hochberg (FDR).
- **A win** requires the Holm-adjusted p < 0.10 **and** the effect direction matching the hypothesis.
- **Diebold–Mariano:** data-driven Newey–West HAC lag even at H=1 (`floor(4·(T/100)^(2/9))`), HLN
  small-sample correction, referenced to t(T−1); pooled comparisons are **date-clustered** (mean loss
  differential per date) so cross-sectional correlation across ETFs is collapsed (Risk R9). Realized
  T is recorded per test; T < 30 is flagged underpowered.
- **Directional accuracy** uses a one-sided binomial test that dir-acc > 0.50 (Risk R24).

## Pre-registered families

- **A — H1 signal existence** (`leading`, H=1, POOLED, stratum `all`): each of
  {patchtst, itransformer, mamba} vs each of {momentum, dlinear} — 6 pairs.
- **B — H2/H3 architecture** (`leading`, H=1): iTransformer vs PatchTST on `multi_region` (H2) and
  `single_region` (H3) strata.
- **C — H4 disruption** (`leading`, H=1, stratum `disruption`): mamba vs patchtst, mamba vs itransformer.
- **D — H6a transfer** (`leading`, H=1): patchtst_pretrained vs patchtst.

## Decision rules

The exact, frozen H0–H6 statements and thresholds are in `configs/hypotheses.yaml`. Honest-reporting
rules: if no DL model significantly beats momentum after Holm, **H0 holds and is stated explicitly**;
`deferred` (not `reject`) is used when a comparison could not run on this hardware — Mamba's official
fused kernel (the CPU S6 fallback is tagged `mamba_impl=fallback`, so H4 is `deferred`), foundation
models (CPU/extras absent), and absent NTL-masked-pretrained variants (H6a, Tier 2).

_Frozen 2026-06-29. Changes after this point require a logged config bump recorded here._
