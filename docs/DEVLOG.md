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
