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
