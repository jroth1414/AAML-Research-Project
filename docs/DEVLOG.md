# Development Log

A running, dated trail of what was built, the decisions made, deviations from the per-phase
plans (and why), problems hit and how they were resolved, and verification evidence. This is
the raw material for the final paper's Methodology, Experimental-Setup, and Limitations
sections, and for the reproducibility appendix. Newest phase appended at the bottom.

Conventions: dates are absolute; task IDs (S#, N#, F#, P#, M#, E#, W#) reference the per-phase
plans under `docs/devplan/`. "Appendix A" refers to the normative contracts in `DEVPLAN.md`.

---

## Phase 0 — Setup (tasks S1–S12) · 2026-06-28/29 · branch `phase0-setup` → merged to `main`

**Goal.** Stand up the canonical repo scaffold, a reproducible Python 3.11 CPU environment, the
shared `ntl_etf` utility layer, an offline test suite + CI, runner stubs, and the git workflow —
the foundation every later phase writes into.

**Environment (verified).** Python 3.11.9 at the pinned path; torch 2.4.1+cpu; numpy 2.0.2,
pandas 2.2.3, pyarrow 17.0.0, rasterio 1.4.4, geopandas 1.1.4, statsmodels 0.14.6. Platform
win32 → capability profile `windows_cpu` (CUDA absent). `latexmk` present (paper can build).

**What was built.** Canonical tree + `ntl_etf` package (`__version__=0.1.0`); split
requirements (`requirements.txt` core / `requirements-extras.txt` GPU-only / `requirements.lock.txt`
frozen, 107 pkgs); utils: `capabilities.py`, `config.py`, `manifest.py`, `results.py`,
`tracking.py`, `seed.py`, `logging.py`; `.env.example`, `.gitignore` (+ `.gitattributes`),
`pyproject.toml` (ruff/black/pytest); stub configs; 18 offline tests + GitHub Actions CI
(`.github/workflows/ci.yml`, Linux+Windows, core deps, no network); `scripts/check_env.py`,
`merge_results.py`, and stub `download_*/build_panel/run_experiment` + `run_all.{ps1,sh}`.

**Key decisions & deviations (carry into the paper / reproducibility notes):**

1. **NTL backend = `earthaccess`, not `blackmarblepy` (deviation; Risk R5 realized).**
   `blackmarblepy` transitively requires `ipywidgets`, whose JupyterLab labextension static
   files have very long names; combined with this repo's deep install path they exceed the
   Windows 260-char `MAX_PATH` limit, so `pip install` fails with an `OSError` when long-path
   support is off (default) and the user is not an administrator. The plan anticipated NTL
   install fragility (R5) and provides `earthaccess` as a backend-agnostic fallback (Task N2)
   with the same downstream interface. Resolution: `earthaccess` is in core `requirements.txt`
   and is the default `download_backend` on Windows/CPU; `blackmarblepy` moved to optional
   `requirements-ntl.txt` with documented workarounds (enable long paths / short venv path /
   Linux-WSL2). Verified `earthaccess` resolves with **no** jupyter deps; `blackmarblepy` does
   pull them (dry-run dependency report).

2. **Seed reconciliation (Appendix A / Risk R17).** Project default seed = **1414**; multi-seed
   set `[1414,1415,1416,1417,1418]`; evaluation bootstrap uses purpose-seed `0`. The draft
   values 1337/42 in some per-phase files are superseded.

3. **Binding contracts honored up front:** month-end tz-naive dates (A.1); A.4 results-store
   superset schema in `utils/results.py`; A.5 manifest superset (`git_sha`, `scaler_fit_on`,
   `mamba_impl`, `gpu_stages_skipped`) in `utils/manifest.py`; capability-gated skip-and-log for
   Mamba/foundation.

**Verification (DoD gate, all green).** `pytest -q` 18 passed (no network); `ruff check .` clean;
`black --check .` clean; `scripts/check_env.py` → `profile=windows_cpu`, `core OK`, exit 0;
`scripts/run_all.ps1` end-to-end over stubs exit 0 with the `mamba` step correctly
skip-and-logged; `pip check` clean; extras not installed; RESEARCH.MD byte-identical to baseline
SHA `181361D2…E84E31`.

**Open items flagged to the user.**
- The repo's pre-existing root commit `c82157f` (predates this build) carries a
  `Co-Authored-By: Claude` trailer. All Phase-0 commits are clean; scrubbing the trailer would
  rewrite history back to the RESEARCH.MD-adding root commit — deferred to the user's decision.

**Git.** 12 task-ID-prefixed commits on `phase0-setup`; merged to `main` via `--no-ff`
("Merge Phase 0 (setup): …"). Credentials confirmed present in `.env` (`EARTHDATA_TOKEN`,
`FRED_API_KEY`) → Phase A can do real downloads.

---

## Phase A — Data acquisition (N + F) · 2026-06-29 · branch `phaseA-data-ntl`

Built Phase A.1 (NTL, tasks N1–N10) and Phase A.2 (finance/macro, tasks F1–F12) on one branch
(combined, since the long background NTL download reads the working tree). 51 offline tests +
2 live integration smokes (FRED + NTL) green; ruff/black clean.

### A.1 — NTL (VNP46A3) findings, all verified against a real tile (2020-06, h28v08)
The N3 "TO-VERIFY" items were resolved empirically by downloading a real Singapore tile and a
real 2-tile Yangtze region:
- **Auth:** `earthaccess.login(strategy="environment")` works with the 672-char bearer
  `EARTHDATA_TOKEN`. No username/password needed. Login cached once per process.
- **HDF5 path:** the grid is `//HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data_Fields/<layer>` (not the
  older `VNP_Grid_DNB`); the reader matches the layer suffix, so it is version-robust.
- **Georeferencing:** subdataset has an identity transform, so we build the affine from the
  10°×10° Black Marble geographic tile grid (h→lon, v→lat; 2400×2400 px; 15 arc-sec). Validated
  single-tile (Singapore) and 2-tile mosaic+clip (Yangtze, h29v05+h30v05).
- **Fill value:** the v2 composite fill is **-999.9** (float, in the BAND tags, not dataset
  tags) — NOT 65535 as an early draft assumed. Reader now merges band tags + GDAL nodata; fill
  detection is float-safe (np.isclose). scale_factor=1.0, offset=0.
- **Quality flags (the big one):** for the MONTHLY Snow_Free composite, class 0 is rare
  (~1.6k px/tile), the bulk is classes 1 & 2, and 255 = no-retrieval. The composite is already
  gap-filled and marks true no-data with its own -999.9 fill. So we keep quality {0,1,2} and drop
  only 255. (Dropping {1,2} as the N3 draft assumed discarded ~all pixels → n_valid=0; fixed.)
- **Month selection:** a CMR bbox+temporal query for month M can also return the adjacent
  month's monthly composite, so we filter granules by the `.A{YYYY}{DOY}.` date token to keep
  only the target month.
- **Sanity:** Singapore 2020-06 → ntl_mean 38.0, lit_frac 0.97, frac_masked 0.06 (just ocean);
  Yangtze → ntl_mean 8.2, frac_masked 0.008. Both plausible.
- **Release-lag note (flag for Phase B):** the conservative 45-day stamp (`data_release_date =
  month_end(M) + 45d`) lands in early **M+2** for 31-day months (e.g. June→Aug 14), whereas the
  canonical Phase-B alignment constant is `RELEASE_LAG_MONTHS = 1` (target = M+1, Appendix). The
  stamp is a diagnostic; Phase B's `lag_decision.md` reconciles whether the real VNP46A3 latency
  (historically ~2–4 weeks) supports lag 1 or requires 2.
- **Scale reality:** tiles are ~28 MB. The full 28-region × 144-month pull **completed** in the
  background (~3.75 h): `data/processed/ntl_features.parquet` has **4032 rows** (28×144), release-lag
  holds on every row, `frac_masked` mean 0.3% (only 4 region-months > 0.5, Risk R27 candidates),
  0 NaN. 4 region-months failed on a first pass (network) and succeeded on the resumable re-run
  (manifest-driven). **Disk footprint:** the raw HDF5 cache under `data/raw/ntl/` is **~178 GB**
  (2448 tiles), leaving ~112 GB free. The cache is gitignored and only needed for re-extraction;
  it can be purged to reclaim disk (re-download is ~3.75 h). **Purged after extraction** (user
  approved) — reclaimed ~178 GB (free 112 → 290 GB); `ntl_features.parquet` + the manifest remain.

### A.2 — Finance/Macro (real data downloaded & validated)
- **ETF (yfinance):** 11 tickers, month-end tz-naive log returns + 12m momentum. Ragged
  inceptions validated exactly: XLC first return **2018-07-31**, XLRE **2015-11-30** (others
  2013-02). XLK 2020-03 return −9.0% (COVID crash). No forward-fill across gaps.
- **FRED IP:** all 8 nowcast-eligible series + INDPRO control resolved on their **primary** IDs
  (IPMAN, IPDMAT, IPG211S, IPUTIL, IPG3254S, IPNCONGD, IPDCONGD, IPG334S) — every TO-VERIFY id
  exists; no fallbacks needed. XLF/XLC/XLRE correctly excluded (no IP analog). 144 obs each.
- **VIX:** `VIXCLS` → monthly mean/max + disruption_flag (>25); 13 disruption months over
  2013–2024; 2020-03 flagged, 2017 calm. Table carries `vix_max` (Appendix A.2 / Risk R19).
- **Contracts:** `transform_registry.py` pins lag/causal rules; `financial_macro_manifest.json`
  records the release-lag + alignment contract Phase B enforces.

**Verification.** `pytest -q` 51 passed (no network); `pytest -m integration` 2 passed (live FRED
INDPRO + live NTL Singapore); ruff/black clean. Real artifacts written:
`data/processed/{etf_returns,macro_ip,vix_monthly}.parquet` + manifests (all gitignored).

**Git.** Task-ID-prefixed commits (N4, N1-N8, N9-N10, F1/F6, F2-F3, F5/F7/F9, F10, F4/F8/F12,
F11) on `phaseA-data-ntl`.

---

## Phase B — Panel, walk-forward splits, leakage audit (P1–P10) · 2026-06-29 · branch `phaseB-panel-splits`

Built the global channel-independent panel, the dual CI/variate dataset, walk-forward CV, and the
**executable leakage audit** — the project's central safety gate. 74 offline tests green;
ruff/black clean. **The real `build_panel.py` passes the No-Go gate on actual data.**

**Design decisions (carry into the paper):**
- **Primary NTL feature = `ntl_mean`** (area-normalized). `ntl_sum` (sum-of-lights) scales with the
  ROI bbox area, which varies enormously across regions (tiny Singapore vs huge Central Valley), so
  it is not comparable in a shared-weight global pool (Risk R25); `ntl_sum/lit_frac/p90` are kept as
  QA diagnostics only.
- **Month-END tz-naive grid** everywhere (Appendix A.1); loaders left-join onto it, NaN for gaps,
  no ffill. Ragged ETF guards verified (XLRE NaN < 2015-11, XLC < 2018-06).
- **Walk-forward:** expanding 60-month train + 12 val + 12 test, step 12 → **6 folds** on 144 months
  (documented; motivates cross-fold pooling for DM power). Normalization is **train-only**, fit per
  region (X) and per sector (y), de-standardized for metrics.
- **Release-lag alignment (P5):** NTL month t → ETF return t+1 (strictly forward, `RELEASE_LAG=1`).
  `align_targets` builds the H-step forward target; the strict-forward property is *enforced by the
  leakage audit* (so a tampered `release_lag=0` is reported as a failure rather than crashing).
- **No-look-ahead pair screen (P3):** correlation screen on the 2013–2017 warmup ONLY (ends before
  the first fold's test); H2/H3 pairs force-kept. On real data **12 of 28 candidate pairs kept**
  across **7 sectors** (XLE, XLI, XLK, XLP, XLU, XLV, XLY); XLB/XLC/XLF/XLRE had no region clearing
  |ρ|≥0.15 on the warmup and were not force-kept — an honest limitation (those ETFs lack a screened
  NTL predictor; H1 is evaluated on the 7 with signal). Multi-region sectors for H2: XLE(3), XLI(3),
  XLV(2) — so H2 is not n=1 (Risk R7).

**Executable leakage audit (P8 — the No-Go gate).** Five invariants, each with a negative control
proving the assert has teeth: L1 norm train-only (vs a full-range fit), L2 release-lag strictly
forward (vs `lag=0`), L3 no-NaN window, L4 screen used no test data (vs warmup into the test era),
L5 temporal fold ordering (vs train∩val). **On the real panel: all five PASS** (7344 anchors audited
across 6 folds); `experiments/manifests/leakage_audit.json` records it.

**Real artifacts written:** `data/processed/panel_long.parquet` (1728 rows = 12 series × 144 months,
P9 schema), `series_registry.parquet` (12 series), `data/interim/{pair_screen_manifest.csv,
folds_manifest.json}`. **CI-view anchors (L=12,H=1,leading) = 1584 > 1000** (soft floors per R18).

**Verification.** `pytest -q` 74 passed (no network); `build_panel.py` exits 0 with the audit ALL
PASS; ruff/black clean. **Git:** commits P1-P2, P6, P3/P5, P4/P7, P8/P9, P10 on `phaseB-panel-splits`.

---

## Phase C — Models + training (M1–M17 core) · 2026-06-29 · branch `phaseC-models`

Built the full model layer + deterministic Trainer + the experiment runner; all six models run on
CPU and produce the Appendix-A.3 `predictions.parquet` contract on the real panel. 90 offline tests
green; ruff/black clean.

**Models (all validated on real data):**
- **momentum** (M2): trailing-12m mean of the ETF's own returns (the H0/H1 yardstick; uses return
  history, not NTL). 864 contract-valid predictions.
- **DLinear** (M3): moving-avg decomposition + per-component linear heads on the CI NTL window
  (26 params).
- **PatchTST** (M5): clean-room patch attention on the CI NTL window (67.8k params). *Clean-room
  note:* implemented from the published architecture rather than vendoring the thuml library — more
  self-contained, no heavy external dep, same math (documented for the paper).
- **iTransformer** (M6): clean-room cross-region (variate) attention; adapted to predict the
  exogenous ETF return via masked-mean pooling over valid NTL region tokens (72k params, variate
  view, 216 multi-region predictions).
- **Mamba** (M7): capability-gated. On this CPU box it runs the **pure-PyTorch S6 fallback** (faithful
  sequential selective scan), tagged `mamba_impl=fallback` in the manifest so Phase D reports H4
  honestly (never as the official kernel). 864 predictions.
- **foundation** (M10, chronos/moirai/timesfm): **skip-and-log** — unavailable on the CPU profile
  (extras only), writes `skipped.json`, exits 0. Verified.

**Trainer (M9):** one deterministic loop for every torch model — seeded, AdamW, MSE on STANDARDIZED
targets, grad clipping, early stopping on val with best-weight restore, CPU-first. The PanelDataset
carries train-only `(y_mu, y_sigma)`; predictions are de-standardized to return units (leakage guard).

**Variate batching fix (M6/P4 reconciliation):** sectors have different variate counts (XLE/XLI=3,
XLV=2), so variate samples are padded to a global max-V with `var_mask` and zero-filled
invalid/padded variates (excluded from attention + the pool). This is the batchable resolution of
the M6-padding vs P4-gather alternatives.

**Preliminary signal (NOT a verdict — default HP, no tuning; significance decided in Phase D).**
Pooled 1-month-return test MSE / directional accuracy: PatchTST 0.004894 / 0.528, DLinear 0.004950 /
0.552, Mamba(fallback) 0.005046 / 0.519, **momentum 0.005262 / 0.513**, iTransformer 0.005724 / 0.514
(iTransformer is on the 216 multi-region samples only, not pool-comparable). So the NTL-based PatchTST
and DLinear edge out the momentum baseline on pooled MSE — but the gaps are tiny, HP is untuned, and
with ~6 folds power is low; whether any edge is DM-significant after Holm correction (H1) is the
Phase-D question. Monthly returns are near-unpredictable, so H0 may well hold — to be reported honestly.

**Remaining Phase C (Tier 2-3, for later):** M8 masked pretraining + pretrained variants (needed for
H6a; until then H6a is `deferred`), M12 HP search (currently default HP), official Mamba + foundation
fine-tune (GPU only). The core models make **H1/H2/H3/H4 testable in Phase D**.

**Verification.** `pytest -q` 90 passed (shapes/overfit/determinism + contract, no network);
all six models produce contract-valid runs (or skip-and-log); ruff/black clean. **Git:** commits
M1, M2/M3, M9, M4-M6, M7/M10, M13/M15-M17 on `phaseC-models`.

---

## Phase D — Evaluation, statistics, hypothesis verdicts (E1–E12) · 2026-06-29 · branch `phaseD-eval-stats`

Built the full pre-registered evaluation: metrics, the Diebold-Mariano test, strata, the DM suite
with multiple-comparison correction, and the honest H0-H6 verdict. The **real evaluation ran
end-to-end** on the actual predictions. 108 offline tests green; ruff/black clean.

**The headline result (honest, and the scientifically correct one):**
- Loaded 2232 predictions (5 models); E2 alignment audit ALL PASS (no NaN, month-end targets,
  `scaler_fit_on=train` in every manifest).
- DM suite: 11 pre-registered comparisons; **only 1 Holm-significant** result — in the disruption
  stratum, PatchTST beats Mamba(fallback) (p_holm=0.067, n=13).
- **H1: reject. H2: reject. H3: reject. H4/H5/H6a/H6b: deferred. H0 HOLDS** — *no DL model
  significantly beats the 12-month momentum baseline on pooled 1-month-return MSE after Holm
  correction (with directional significance).* The point-estimate MSE edges from Phase C (PatchTST/
  DLinear < momentum) are **not significant** once tested properly with the data-driven HAC lag,
  HLN correction, and date-clustered pooling across ETFs. This is the expected outcome for
  near-random monthly returns and is reported plainly, not buried.
- **Deferred (not reject), with reasons:** H4 — Mamba ran via the CPU S6 fallback, not the official
  fused kernel (Risk R6); H5 — no nowcast-task runs in this store; H6a — no NTL-masked-pretrained
  variant (Tier 2).

**Addendum — H6b enabled (Chronos zero-shot on CPU).** `chronos-forecasting` was lumped with the
GPU-only extras and skip-and-logged, but Chronos zero-shot is actually CPU-feasible (M10). Installed
`chronos-forecasting` (torch pin intact), fixed `foundation.py` for the 2.3.0 API
(`BaseChronosPipeline` / `predict_quantiles(inputs,...)`, model `amazon/chronos-bolt-small`, pipeline
cached + per-(sector,date) memoized), and ran the zero-shot reference (504 predictions). Added a
pre-registered family E (Chronos vs momentum — both NO-NTL return-history forecasters, Risk R23
split) and the H6b verdict logic. **Result: H6b = support** — Chronos beats momentum at next-month
return prediction (mean_diff −0.000192, **p_holm = 0.049**, n=72). The foundation prior has *modest*
value over naive momentum; it says nothing about NTL (it never sees NTL). Reproducing H6b needs
`pip install chronos-forecasting` (CPU-ok); the run skip-and-logs if absent. Mamba's official kernel
and TimesFM/Moirai fine-tune stay correctly deferred (GPU).

**Statistical rigor (the project's whole point):**
- **DM (E4):** data-driven Newey-West lag even at H=1 (`floor(4·(T/100)^(2/9))`, Risk R9), HLN
  small-sample correction, t(T-1); **date-clustered pooling** (mean loss differential per date) so
  cross-sectional correlation across ETFs is collapsed; realized T recorded, T<30 flagged.
- **Corrections (E5/E7):** Holm (FWER, primary) + Benjamini-Hochberg (FDR, reported) within each
  pre-registered family at α=0.10; a "win" needs Holm-p<0.10 AND the right direction.
- **Directional accuracy** has a one-sided binomial test that dir-acc>0.50 (Risk R24).
- **Pre-registration frozen** in `configs/hypotheses.yaml` (owner E) + `experiments/PREREG.md`;
  `prereg.py` loads it so the driver cannot data-snoop.

**Artifacts (real):** `experiments/{results_store.parquet+csv, dm_results.parquet+csv,
hypotheses_verdict.json}`; `paper/tables/{main_results.md+tex, dm_family_a.md,
hypotheses_verdict.md}`; `paper/figures/{metric_bars_mse, metric_bars_dir_acc,
dm_significance_A}.{png,pdf}`.

**Honest limitations carried to the paper:** ~6 folds (low DM power); the screen kept 7/11 sectors
(H1 evaluated on those with signal); H2/H3 strata are thin (XLE single / XLI multi per
pre-registration); Mamba is the CPU fallback, not the official kernel.

**Verification.** `pytest -q` 108 passed (no network); `analyze_results.py` exits 0 with the audit
ALL PASS and a complete H0-H6 verdict; ruff/black clean. **Git:** per-task commits E3, E4, E5, E1/E2,
E6, E7/E8, E9, E11, E12 on `phaseD-eval-stats`.
