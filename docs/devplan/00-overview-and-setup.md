# Phase 0 — Overview, Environment, Reproducibility, Tooling

> File: `docs/devplan/00-overview-and-setup.md` · Task-ID prefix: **S** · Phase order: **Phase 0 → A → B → C → D → E**
> This is the foundation phase. Every later phase assumes the repo structure, environment, config loader, seeding, results store, and git workflow defined here already exist.

---

## 0. Orientation for the Executing Agent

**You are an autonomous coding agent.** There is no human to ask. When this document says "verify," run the stated command and read the output. When it says **TO-VERIFY**, the exact external fact could not be pinned down at planning time — run the exact check given and adopt whatever the check returns.

**What this project is.** Research project *"Nighttime Light Emissions as an Economic Proxy for Sector ETF Forecasting: PatchTST vs iTransformer vs Mamba."* You will build an end-to-end pipeline that downloads NASA VIIRS Black Marble monthly nighttime-light (NTL) composites, aligns them (with a strict release-lag guard) to the 11 SPDR sector ETFs and FRED industrial-production series, builds one global channel-independent panel, trains baselines + three deep models + transfer-learning variants under walk-forward CV, runs pre-registered statistical tests of hypotheses H1–H6, and writes the paper.

**How the plan files relate.** The master index is `DEVPLAN.md` (written by the editor, not by you). The per-phase files and their task prefixes:

| Phase | File | Prefix | Scope |
|------|------|--------|-------|
| 0 | `docs/devplan/00-overview-and-setup.md` | **S** | environment, repo bootstrap, reproducibility, tooling, git workflow (**this file**) |
| A | `docs/devplan/01-data-ntl.md` | **N** | Black Marble VNP46A3 download + raster feature extraction |
| B | `docs/devplan/02-data-financial-macro.md` | **F** | yfinance ETF returns, FRED industrial production, VIX |
| C | `docs/devplan/03-panel-and-splits.md` | **P** | region→sector pairing, global panel, windowing, walk-forward splits, leakage audit |
| D | `docs/devplan/04-models-and-training.md` | **M** | baselines + PatchTST + iTransformer + Mamba + pretraining + foundation init + trainer + HP search |
| E | `docs/devplan/05-evaluation-and-stats.md` | **E** | metrics, Diebold–Mariano, stratified analyses, hypothesis decision rules, plots |
| W | `docs/devplan/06-paper-and-deliverables.md` | **W** | paper structure mapped to rubric, deliverables, reproducibility, limitations |

**Dependency order.** Do Phase 0 fully first; nothing else runs without it. Then A and B may proceed in parallel (both depend only on Phase 0). C depends on A + B. D depends on C. E depends on D. W depends on E. Cross-phase dependencies are cited by task ID (e.g. "depends on **P3**").

**Golden rules enforced from this phase onward:** config-driven; seed-controlled; leakage-safe (normalization fit on train only; NTL release-lag enforced); CPU-by-default with Mamba/foundation work gated behind a capability flag; **RESEARCH.MD is never modified**; **Claude is never listed as an author or co-author** anywhere (commits, paper, docs).

---

## S1 — Repository bootstrap (preserve RESEARCH.MD, create canonical structure)

**Objective.** Clone the repo and materialize the canonical directory tree without touching `RESEARCH.MD`.

**Dependencies.** None.

**Actions.**
1. Clone and inspect:
   ```powershell
   git clone https://github.com/jroth1414/AAML-Research-Project.git
   cd AAML-Research-Project
   git log --oneline -5
   Get-ChildItem
   ```
   Confirm `RESEARCH.MD` exists. **Do not edit, rename, move, or reformat it at any point.** Record its current SHA-256 so you can later prove it is untouched:
   ```powershell
   (Get-FileHash RESEARCH.MD -Algorithm SHA256).Hash | Out-File scratch_research_hash.txt
   ```
   (Keep `scratch_research_hash.txt` outside the repo or delete it before committing; it is a check artifact, not a deliverable.)
2. Create the directory tree exactly as in the canonical structure. PowerShell:
   ```powershell
   $dirs = @(
     "docs/devplan","configs",
     "data/raw","data/interim","data/processed","data/external",
     "src/ntl_etf/data","src/ntl_etf/models","src/ntl_etf/train","src/ntl_etf/eval","src/ntl_etf/utils",
     "scripts","tests","notebooks","experiments","paper"
   )
   $dirs | ForEach-Object { New-Item -ItemType Directory -Force -Path $_ | Out-Null }
   ```
   bash:
   ```bash
   mkdir -p docs/devplan configs data/{raw,interim,processed,external} \
     src/ntl_etf/{data,models,train,eval,utils} scripts tests notebooks experiments paper
   ```
3. Add package `__init__.py` files so `ntl_etf` is importable:
   ```powershell
   $pkgs = @("src/ntl_etf","src/ntl_etf/data","src/ntl_etf/models","src/ntl_etf/train","src/ntl_etf/eval","src/ntl_etf/utils")
   foreach ($p in $pkgs) { if (-not (Test-Path "$p/__init__.py")) { New-Item -ItemType File "$p/__init__.py" | Out-Null } }
   ```
   Put a version string in the top-level package init:
   ```python
   # src/ntl_etf/__init__.py
   __version__ = "0.1.0"
   ```
4. Add a `.gitkeep` to each `data/*` and `experiments/` subfolder so the empty (gitignored) directories exist on a fresh clone:
   ```powershell
   "data/raw","data/interim","data/processed","data/external","experiments" | ForEach-Object { New-Item -ItemType File "$_/.gitkeep" -Force | Out-Null }
   ```

**Files created/modified.** The full tree above; `src/ntl_etf/**/__init__.py`; `data/**/.gitkeep`, `experiments/.gitkeep`.

**Deliverables.** Canonical tree present; `RESEARCH.MD` untouched.

**Acceptance Criteria.**
- `Test-Path RESEARCH.MD` is `True` and `(Get-FileHash RESEARCH.MD -Algorithm SHA256).Hash` equals the value in `scratch_research_hash.txt`.
- All directories in the canonical structure exist (`Get-ChildItem -Recurse -Directory | Measure-Object` ≥ 18 dirs under repo root).
- `git status` shows only new untracked files; `RESEARCH.MD` is **not** listed as modified.

---

## S2 — Python 3.11 virtual environment

**Objective.** Create and activate an isolated venv using the pinned Windows Python 3.11.

**Dependencies.** S1.

**Actions.**
1. Confirm the interpreter:
   ```powershell
   & "C:/Users/Admin/AppData/Local/Programs/Python/Python311/python.exe" --version   # expect Python 3.11.x
   ```
2. Create the venv at repo root (`.venv` is gitignored — see S6):
   ```powershell
   & "C:/Users/Admin/AppData/Local/Programs/Python/Python311/python.exe" -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip setuptools wheel
   ```
   bash (Git Bash):
   ```bash
   "C:/Users/Admin/AppData/Local/Programs/Python/Python311/python.exe" -m venv .venv
   source .venv/Scripts/activate
   python -m pip install --upgrade pip setuptools wheel
   ```
   > If `Activate.ps1` is blocked by execution policy, run once per shell: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.

**Deliverables.** Working `.venv` with Python 3.11.

**Acceptance Criteria.**
- After activation, `python -c "import sys; print(sys.version)"` prints 3.11.x and `python -c "import sys; print(sys.prefix)"` points inside `.venv`.
- `pip --version` runs without error.

---

## S3 — Dependency pinning: core (CPU) vs. extras (GPU)

**Objective.** Split dependencies so the **core** set installs cleanly on Windows/CPU and powers everything except Mamba and heavy foundation fine-tuning; the **extras** set holds GPU-only / heavy packages installed only in a GPU profile.

**Dependencies.** S2.

**Actions.**

1. Create `requirements.txt` (core — must install on native Windows/CPU). Pin to known-good ranges; the agent should resolve exact versions and freeze them (step 4).
   ```
   # requirements.txt  (CORE — Windows/CPU clean install)
   numpy>=1.26,<2.1
   pandas>=2.1,<2.3
   pyarrow>=15,<17
   scipy>=1.11
   scikit-learn>=1.4
   statsmodels>=0.14
   pyyaml>=6.0
   python-dotenv>=1.0
   tqdm>=4.66
   matplotlib>=3.8
   seaborn>=0.13
   # geo / raster (see Windows wheel note below)
   rasterio>=1.3,<1.4
   geopandas>=0.14
   shapely>=2.0
   xarray>=2024.1
   # data sources
   yfinance>=0.2.40
   fredapi>=0.5.2
   blackmarblepy>=2024.1
   # deep learning (CPU build of torch is fine for core)
   torch>=2.2,<2.5
   # tooling
   ruff>=0.5
   black>=24.4
   pytest>=8.0
   ```
   > **Windows wheel gotchas (handle proactively):**
   > - **rasterio / GDAL / shapely / pyarrow** ship manylinux+Windows binary wheels on PyPI for CPython 3.11. Install via `pip` only — do **not** pull in conda. If `pip install rasterio` fails to find a wheel, do **not** attempt a source build (needs GDAL toolchain). Instead pin `rasterio==1.3.10` (or the latest 1.3.x that has a `cp311-win_amd64` wheel — verify with `pip index versions rasterio`).
   > - Install `numpy` and `pyarrow` *before* `rasterio`/`geopandas` so the resolver doesn't downgrade them.
   > - Keep `numpy<2.1` until you confirm every binary dep (rasterio, torch, pyarrow) has a NumPy-2 ABI wheel; mismatches surface as `_ARRAY_API not found` errors at import.
   > - **torch on CPU:** `pip install torch` installs the CPU wheel by default on Windows. That is correct for the core profile.

2. Create `requirements-extras.txt` (GPU-only / heavy; **do not** install in the Windows/CPU profile):
   ```
   # requirements-extras.txt  (GPU / Linux / WSL2 / Colab ONLY — see S4)
   mamba-ssm>=2.2          # requires CUDA + Linux build toolchain; will NOT install on native Windows
   causal-conv1d>=1.4      # dependency of mamba-ssm; CUDA kernels
   triton>=2.3             # mamba-ssm runtime dep; Linux/CUDA
   chronos-forecasting>=1.4
   uni2ts>=1.2             # Moirai
   timesfm[torch]>=1.2
   einops>=0.8
   ```
   Add a header comment at the top of the file:
   ```
   # NOTE: Install ONLY in a CUDA Linux/WSL2/Colab environment (Execution Profile 2 or 3).
   # On native Windows/CPU these will fail to build (mamba-ssm/causal-conv1d need nvcc + Linux).
   # See docs/devplan/00-overview-and-setup.md S4.
   ```

3. Install core and verify imports:
   ```powershell
   pip install -r requirements.txt
   python -c "import numpy,pandas,pyarrow,rasterio,geopandas,xarray,yfinance,fredapi,blackmarble,sklearn,statsmodels,torch; print('core OK', torch.__version__, 'cuda', torch.cuda.is_available())"
   ```
   > If any single core package has no Windows wheel, pin it down a minor version and re-resolve; never fall back to a source build that needs a C/GDAL toolchain.

4. Freeze the resolved core environment for reproducibility:
   ```powershell
   pip freeze > requirements.lock.txt
   ```

**Files created/modified.** `requirements.txt`, `requirements-extras.txt`, `requirements.lock.txt`.

**Deliverables.** Core deps installed and importable on Windows/CPU; extras file staged for GPU profiles; lockfile.

**Acceptance Criteria.**
- The verification one-liner in step 3 prints `core OK ...` with no `ImportError`.
- `pip check` reports no broken requirements.
- `requirements-extras.txt` is **not** installed in the Windows profile (confirm: `python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('mamba_ssm') is None else 1)"` exits 0 on Windows/CPU — absence is expected and correct).

---

## S4 — Mamba/Windows reality: capability flag + three execution profiles

**Objective.** Make the pipeline detect its own capabilities and degrade gracefully: every non-Mamba part runs CPU-only; Mamba and heavy foundation work are gated behind a flag and deferred to a GPU profile **without ever hard-crashing**.

**Dependencies.** S2; consumes config from S5; used by every Phase D/E task that touches Mamba or foundation models.

**Background (verified).** `mamba-ssm` and its deps `causal-conv1d` and `triton` require CUDA + Linux build toolchains and **do not install cleanly on native Windows/CPU** (state-spaces/mamba issue #662; PRs adding partial Windows support exist but are unreliable). Therefore Mamba experiments must be **optional and skippable**. Foundation models (Chronos, Moirai/uni2ts, TimesFM) install on CPU but are GPU-preferred and large; treat fine-tuning as GPU-only, with CPU **zero-shot** as the feasible fallback.

**Actions.**

1. Create `src/ntl_etf/utils/capabilities.py`:
   ```python
   from __future__ import annotations
   import importlib.util
   from dataclasses import dataclass, asdict

   def _can_import(name: str) -> bool:
       return importlib.util.find_spec(name) is not None

   def has_cuda() -> bool:
       try:
           import torch
           return bool(torch.cuda.is_available())
       except Exception:
           return False

   def has_mamba_ssm() -> bool:
       # importable AND CUDA present (mamba-ssm kernels are CUDA-only)
       return _can_import("mamba_ssm") and has_cuda()

   def has_foundation(pkg: str) -> bool:
       # pkg in {"chronos", "uni2ts", "timesfm"}
       return _can_import(pkg)

   @dataclass(frozen=True)
   class Capabilities:
       cuda: bool
       mamba_ssm: bool
       chronos: bool
       uni2ts: bool
       timesfm: bool
       profile: str  # "windows_cpu" | "gpu_full" | "unknown"

   def detect() -> "Capabilities":
       cuda = has_cuda()
       caps = dict(
           cuda=cuda,
           mamba_ssm=has_mamba_ssm(),
           chronos=has_foundation("chronos"),
           uni2ts=has_foundation("uni2ts"),
           timesfm=has_foundation("timesfm"),
       )
       profile = "gpu_full" if (cuda and caps["mamba_ssm"]) else ("windows_cpu" if not cuda else "unknown")
       return Capabilities(profile=profile, **caps)

   def as_dict() -> dict:
       return asdict(detect())
   ```
2. Mandate the **skip-and-log** contract for every model that may be unavailable: any trainer/experiment that requests Mamba or a foundation model on a profile lacking it must **log a WARNING, record a `status="skipped"` row in the results store (S8), and continue** — never raise. Phase D/E tasks reference this contract.
3. Add a CLI capability report `scripts/check_env.py` that prints `detect()` plus torch/CUDA/key package versions, used at the start of `run_all` (S9) and CI smoke (S10).

**Three execution profiles** (record the active one in every run manifest, S7):

| Profile | Where | Installs | Runs | Mamba | Foundation models |
|--------|-------|----------|------|-------|-------------------|
| **1. Windows/CPU** (default) | this machine | `requirements.txt` | data → panel → baselines → PatchTST → iTransformer → DLinear → pretraining (small) → all eval/stats **except** Mamba rows | **skipped** (logged) | zero-shot CPU only (slow; optional) |
| **2. WSL2+CUDA or cloud GPU** | Linux + NVIDIA | core + `requirements-extras.txt` | **everything** incl. Mamba + foundation fine-tune | full | full |
| **3. Colab notebook** | hosted GPU | core + extras in-notebook | only the GPU-only experiments (Mamba, foundation fine-tune) | full | full |

**Merging GPU results back (Profiles 2/3 → main results store).** GPU runs write the same per-run artifacts as Profile 1: a manifest JSON (S7) + result rows (S8) under `experiments/<run_id>/`. To merge: copy the `experiments/<run_id>/` directory back into the repo's `experiments/` (or commit it from the GPU environment), then run `scripts/merge_results.py` which appends/dedupes rows into `experiments/results.parquet` keyed by `(run_id, model, hypothesis, fold)`. Dedup rule: identical key + identical config hash ⇒ keep the most recent `timestamp`. The Colab notebook (Phase D) must download its `experiments/<run_id>/` as a zip for the agent to unzip into the repo.

**Files created/modified.** `src/ntl_etf/utils/capabilities.py`, `scripts/check_env.py`, `scripts/merge_results.py` (stub here; finalized in Phase E).

**Deliverables.** Capability detection; documented profiles; skip-and-log contract; result-merge path.

**Acceptance Criteria.**
- `python scripts/check_env.py` runs on Windows/CPU and prints `profile=windows_cpu`, `mamba_ssm=False`, `cuda=False` **without raising**.
- Unit test `tests/test_capabilities.py`: `detect()` returns a `Capabilities` with `mamba_ssm == (find_spec('mamba_ssm') is not None and torch.cuda.is_available())`; asserts no exception when `mamba_ssm` is absent.
- A stub call to a "skipped" model path logs a WARNING and returns a `status="skipped"` row (verified in Phase D; the contract is asserted here via a tiny fake in `tests/test_capabilities.py`).

---

## S5 — Secrets and config: `.env.example`, config loader, experiment schema

**Objective.** Centralize secrets (never committed) and provide a thin YAML config loader plus the experiment-config schema all phases share.

**Dependencies.** S2; `python-dotenv`, `pyyaml` from S3.

**Background (verified).** `blackmarblepy` reads a NASA Earthdata **bearer token** from env var `BLACKMARBLE_TOKEN` (recommended) or accepts `token=` directly; FRED via `fredapi.Fred(api_key=...)`. Earthdata downloads may alternatively use a `.netrc` with username/password.

**Actions.**

1. Create `.env.example` (committed; the real `.env` is gitignored):
   ```dotenv
   # .env.example  — copy to .env and fill in; .env is gitignored, NEVER commit it.
   # NASA Earthdata (Black Marble). Preferred: a bearer token from https://urs.earthdata.nasa.gov/ (Generate Token).
   EARTHDATA_TOKEN=
   BLACKMARBLE_TOKEN=        # set equal to EARTHDATA_TOKEN; blackmarblepy reads this name
   # Alternative to a token: username/password used to write ~/.netrc for Earthdata
   EARTHDATA_USERNAME=
   EARTHDATA_PASSWORD=
   # FRED
   FRED_API_KEY=
   # Optional experiment tracking (leave blank to disable)
   MLFLOW_TRACKING_URI=
   WANDB_API_KEY=
   WANDB_MODE=offline
   ```
2. Create `src/ntl_etf/utils/config.py` — a Hydra-lite loader: load `.env`, load + merge YAML, allow `key=value` CLI overrides, return a dot-accessible mapping. No heavy deps.
   ```python
   from __future__ import annotations
   import os, copy
   from pathlib import Path
   from typing import Any, Mapping
   import yaml
   from dotenv import load_dotenv

   REPO_ROOT = Path(__file__).resolve().parents[3]

   def load_env(dotenv_path: str | os.PathLike | None = None) -> None:
       load_dotenv(dotenv_path or (REPO_ROOT / ".env"), override=False)

   def _deep_merge(a: dict, b: Mapping) -> dict:
       out = copy.deepcopy(a)
       for k, v in b.items():
           out[k] = _deep_merge(out[k], v) if isinstance(out.get(k), dict) and isinstance(v, Mapping) else v
       return out

   def load_yaml(path: str | os.PathLike) -> dict:
       with open(path, "r", encoding="utf-8") as f:
           return yaml.safe_load(f) or {}

   def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
       # overrides like ["train.lr=1e-3", "model.name=patchtst"]
       out = copy.deepcopy(cfg)
       for ov in overrides:
           key, _, raw = ov.partition("=")
           val = yaml.safe_load(raw)  # auto-types ints/floats/bools/lists
           d = out
           *parents, leaf = key.split(".")
           for p in parents:
               d = d.setdefault(p, {})
           d[leaf] = val
       return out

   def require_secret(name: str) -> str:
       v = os.environ.get(name)
       if not v:
           raise RuntimeError(f"Missing required secret env var: {name}. "
                              f"Copy .env.example to .env and fill it in.")
       return v

   def load_config(yaml_path: str | os.PathLike, overrides: list[str] | None = None,
                   base_paths: list[str] | None = None) -> dict:
       load_env()
       cfg: dict = {}
       for bp in (base_paths or []):
           cfg = _deep_merge(cfg, load_yaml(bp))
       cfg = _deep_merge(cfg, load_yaml(yaml_path))
       return apply_overrides(cfg, overrides or [])
   ```
3. Define the **experiment-config schema** as a committed template at `configs/experiment.example.yaml`. Phase D/E produce concrete configs under `configs/` following this shape:
   ```yaml
   # configs/experiment.example.yaml  — canonical experiment-config schema
   run:
     name: "patchtst_leading_h1"
     seed: 1337
     task: "leading"            # "leading" (forward 1-3m returns) | "nowcast" (contemporaneous IP)
     horizon: 1                 # forecast horizon in months (1,2,3)
     profile: "auto"            # auto-detected; recorded in manifest
   data:
     panel_path: "data/processed/panel.parquet"
     study_start: "2013-01-01"
     study_end: "2024-12-31"
     ntl_release_lag_months: 1  # CRITICAL: month-t NTL predicts month t+1 (Phase C enforces)
   split:
     scheme: "walk_forward"     # rolling origin
     min_train_months: 60
     step_months: 1
   model:
     name: "patchtst"           # momentum | dlinear | patchtst | itransformer | mamba | foundation
     channel_independent: true
     params: {}                 # model-specific (Phase D)
   train:
     epochs: 50
     batch_size: 64
     lr: 1.0e-3
     normalize: "train_only"    # CRITICAL: fit scaler on train split only (Phase C/D guard)
     early_stopping_patience: 10
   eval:
     metrics: ["mse","mae","directional_accuracy"]
     dm_test: true              # Diebold-Mariano (Phase E)
     transaction_cost_bps: 10   # one-way; Sharpe gross & net (Phase E)
   tracking:
     mlflow: false
     wandb: false
   ```
4. Create stub config files that later phases fill in (so paths exist and are referenced consistently): `configs/regions.yaml`, `configs/sector_fred_map.yaml`. Add a one-line header comment in each: `# Populated in Phase A/B; schema owned by docs/devplan/01 & 02.`

**Files created/modified.** `.env.example`, `src/ntl_etf/utils/config.py`, `configs/experiment.example.yaml`, `configs/regions.yaml`, `configs/sector_fred_map.yaml`.

**Deliverables.** Secret handling + config loader + experiment schema.

**Acceptance Criteria.**
- `tests/test_config.py`: writes a temp YAML, loads it, applies override `["train.lr=2e-4"]`, asserts `cfg["train"]["lr"] == 2e-4` (float) and deep-merge works.
- `require_secret("DOES_NOT_EXIST")` raises `RuntimeError` with the helpful message.
- `.env.example` contains `EARTHDATA_TOKEN`, `BLACKMARBLE_TOKEN`, `EARTHDATA_USERNAME`, `EARTHDATA_PASSWORD`, `FRED_API_KEY`.
- Loading `configs/experiment.example.yaml` returns a dict with the keys `run, data, split, model, train, eval, tracking`.

---

## S6 — `.gitignore` and data/secret hygiene

**Objective.** Guarantee data, secrets, venv, checkpoints, caches, and run outputs are never committed (manifests excepted).

**Dependencies.** S1.

**Actions.** Create `.gitignore`:
```gitignore
# secrets / env
.env
.env.*
!.env.example
.netrc

# python
__pycache__/
*.py[cod]
.venv/
venv/
*.egg-info/
.pytest_cache/
.ruff_cache/
.ipynb_checkpoints/

# data (gitignored entirely; keep .gitkeep placeholders)
data/raw/*
data/interim/*
data/processed/*
data/external/*
!data/**/.gitkeep

# model checkpoints / large artifacts
*.pt
*.pth
*.ckpt
*.safetensors
*.h5
*.hdf5
*.tif
*.tiff

# experiment outputs: ignore everything EXCEPT manifests + the aggregated results store
experiments/*
!experiments/.gitkeep
!experiments/**/manifest.json
!experiments/results.csv
!experiments/results.parquet

# tracking
mlruns/
wandb/

# os / editor
.DS_Store
Thumbs.db
.vscode/
.idea/
```
> Rationale: the negated patterns keep `manifest.json` and the aggregated results store under version control (so GPU-profile results merged back in S4 are reviewable) while excluding bulky checkpoints/rasters.

**Deliverables.** `.gitignore`.

**Acceptance Criteria.**
- `git check-ignore -v .env data/raw/x.tif .venv/x experiments/run1/model.pt` reports each as ignored.
- `git check-ignore .env.example experiments/run1/manifest.json` returns **nothing** (i.e. they are tracked).
- After creating a dummy `.env`, `git status --porcelain` does not list `.env`.

---

## S7 — Run manifest (config + git SHA + versions + seed)

**Objective.** Every run writes a self-describing manifest enabling exact reproduction.

**Dependencies.** S4 (capabilities), S5 (config), S8 (results store dir layout).

**Actions.** Create `src/ntl_etf/utils/manifest.py`:
```python
from __future__ import annotations
import json, platform, subprocess, sys, time
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
from .capabilities import as_dict as caps_dict

_TRACK = ["numpy","pandas","pyarrow","scikit-learn","scipy","statsmodels",
          "torch","rasterio","geopandas","xarray","yfinance","fredapi",
          "blackmarblepy","mamba-ssm","chronos-forecasting","uni2ts","timesfm"]

def _ver(p):
    try: return version(p)
    except PackageNotFoundError: return None

def _git_sha() -> str:
    try:
        return subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip()
    except Exception:
        return "UNKNOWN"

def _git_dirty() -> bool:
    try:
        return bool(subprocess.check_output(["git","status","--porcelain"], text=True).strip())
    except Exception:
        return False

def write_manifest(run_dir: str | Path, config: dict, seed: int) -> Path:
    run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    m = {
        "run_id": run_dir.name,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "python": sys.version,
        "platform": platform.platform(),
        "seed": seed,
        "capabilities": caps_dict(),
        "packages": {p: _ver(p) for p in _TRACK},
        "config": config,
    }
    out = run_dir / "manifest.json"
    out.write_text(json.dumps(m, indent=2), encoding="utf-8")
    return out
```

**Deliverables.** `src/ntl_etf/utils/manifest.py`.

**Acceptance Criteria.**
- `tests/test_manifest.py`: `write_manifest(tmp, {"a":1}, 7)` creates `manifest.json` whose parsed JSON has non-null `git_sha` (or `"UNKNOWN"`), `seed==7`, a `capabilities` dict, and a `packages` map containing `numpy` and `torch`.
- `git_dirty` is a bool.

---

## S8 — Lightweight results store (+ optional MLflow/W&B hook)

**Objective.** One canonical, appendable, GPU-mergeable results table; optional experiment-tracker hook that no-ops when disabled.

**Dependencies.** S5, S7.

**Actions.**
1. Create `src/ntl_etf/utils/results.py` with the **fixed result-row schema** (every phase writes rows in this shape):
   ```python
   from __future__ import annotations
   import os
   from pathlib import Path
   import pandas as pd

   RESULTS_PARQUET = Path("experiments/results.parquet")
   RESULTS_CSV = Path("experiments/results.csv")

   RESULT_COLUMNS = [
       "run_id","timestamp_utc","model","task","horizon","hypothesis",
       "fold","train_start","train_end","test_month",
       "metric","value","status","seed","git_sha","config_hash","profile",
   ]

   def append_results(rows: list[dict],
                      parquet_path: os.PathLike = RESULTS_PARQUET,
                      csv_path: os.PathLike = RESULTS_CSV) -> None:
       df = pd.DataFrame(rows)
       for c in RESULT_COLUMNS:
           if c not in df.columns: df[c] = None
       df = df[RESULT_COLUMNS]
       pq = Path(parquet_path); pq.parent.mkdir(parents=True, exist_ok=True)
       if pq.exists():
           df = pd.concat([pd.read_parquet(pq), df], ignore_index=True)
       df.to_parquet(pq, index=False)
       df.to_csv(csv_path, index=False)
   ```
   > `status ∈ {"ok","skipped","failed"}` — "skipped" is how the Mamba/foundation skip-and-log contract (S4) records itself. `config_hash` = stable hash of the resolved config (used by the S4 merge dedup).
2. Create `src/ntl_etf/utils/tracking.py` — a context-manager tracker that wraps MLflow or W&B if enabled in config and otherwise does nothing:
   ```python
   from contextlib import contextmanager

   @contextmanager
   def tracker(cfg: dict, run_name: str):
       use_mlflow = cfg.get("tracking", {}).get("mlflow", False)
       use_wandb = cfg.get("tracking", {}).get("wandb", False)
       handle = None
       try:
           if use_mlflow:
               import mlflow; mlflow.start_run(run_name=run_name); handle = ("mlflow", mlflow)
           elif use_wandb:
               import wandb; handle = ("wandb", wandb.init(name=run_name, mode="offline"))
           yield handle
       finally:
           if handle and handle[0] == "mlflow": handle[1].end_run()
           elif handle and handle[0] == "wandb": handle[1].finish()
   ```
   > MLflow/W&B are **optional** — not in `requirements.txt`. Guard imports; if unavailable or disabled, the tracker is a no-op. Document: install `mlflow` only if `tracking.mlflow: true`.

**Deliverables.** `src/ntl_etf/utils/results.py`, `src/ntl_etf/utils/tracking.py`.

**Acceptance Criteria.**
- `tests/test_results.py`: append two batches; reload parquet; assert columns equal `RESULT_COLUMNS` and row count accumulates; both `results.parquet` and `results.csv` exist.
- `tracker({}, "x")` enters/exits with no error and no extra deps (no-op path).

---

## S9 — Logging, seeding, and the package install

**Objective.** Standard logging, deterministic seeding (with documented limits), and an editable install of `ntl_etf`.

**Dependencies.** S2, S3.

**Actions.**
1. `pyproject.toml` (editable, src-layout) + tool config:
   ```toml
   [build-system]
   requires = ["setuptools>=68", "wheel"]
   build-backend = "setuptools.build_meta"

   [project]
   name = "ntl_etf"
   version = "0.1.0"
   description = "Nighttime-light NTL economic proxy for SPDR sector ETF forecasting (JHU EN.705.742)."
   requires-python = ">=3.11,<3.12"
   dynamic = ["dependencies"]

   [tool.setuptools.packages.find]
   where = ["src"]

   [tool.setuptools.dynamic]
   dependencies = {file = ["requirements.txt"]}

   [tool.ruff]
   line-length = 100
   target-version = "py311"
   [tool.ruff.lint]
   select = ["E","F","I","UP","B"]
   ignore = ["E501"]   # black/line-length handled separately

   [tool.black]
   line-length = 100
   target-version = ["py311"]

   [tool.pytest.ini_options]
   testpaths = ["tests"]
   addopts = "-q"
   ```
   Install editable:
   ```powershell
   pip install -e .
   python -c "import ntl_etf; print(ntl_etf.__version__)"
   ```
2. `src/ntl_etf/utils/logging.py`:
   ```python
   import logging, sys

   def get_logger(name: str = "ntl_etf", level: int = logging.INFO) -> logging.Logger:
       logger = logging.getLogger(name)
       if not logger.handlers:
           h = logging.StreamHandler(sys.stdout)
           h.setFormatter(logging.Formatter(
               "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
               datefmt="%Y-%m-%d %H:%M:%S"))
           logger.addHandler(h); logger.setLevel(level); logger.propagate = False
       return logger
   ```
3. `src/ntl_etf/utils/seed.py`:
   ```python
   from __future__ import annotations
   import os, random
   import numpy as np

   def set_seed(seed: int = 1337, deterministic: bool = True) -> int:
       os.environ["PYTHONHASHSEED"] = str(seed)
       random.seed(seed); np.random.seed(seed)
       try:
           import torch
           torch.manual_seed(seed)
           if torch.cuda.is_available():
               torch.cuda.manual_seed_all(seed)
           if deterministic:
               torch.backends.cudnn.deterministic = True
               torch.backends.cudnn.benchmark = False
               # Best-effort full determinism; some ops have no deterministic kernel.
               os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
               try: torch.use_deterministic_algorithms(True, warn_only=True)
               except Exception: pass
       except ImportError:
           pass
       return seed
   ```
   > **Documented nondeterminism limits:** exact bit-reproducibility is guaranteed only within the *same* hardware + library versions. CUDA atomics, some attention/conv kernels, and multi-threaded BLAS can still vary slightly. We therefore (a) set seeds + deterministic flags, (b) record seed + versions + git SHA in the manifest (S7), and (c) report results as means over a fixed seed set where stochasticity matters (Phase D/E). `warn_only=True` keeps runs from crashing on ops lacking deterministic kernels.

**Files created/modified.** `pyproject.toml`, `src/ntl_etf/utils/logging.py`, `src/ntl_etf/utils/seed.py`.

**Deliverables.** Editable package + logging + seeding.

**Acceptance Criteria.**
- `pip install -e .` succeeds; `import ntl_etf` works from any cwd; `ntl_etf.__version__ == "0.1.0"`.
- `tests/test_seed.py`: `set_seed(1)` then `np.random.rand(3)` equals a second `set_seed(1)` + `np.random.rand(3)`; same for `torch.rand(3)` if torch present.
- `get_logger()` returns a logger with exactly one handler when called twice (no duplicate handlers).

---

## S10 — Dev tooling, tests scaffold, and CI

**Objective.** Lint/format config (S9 covers config; this wires the commands + CI), a tiny test suite with fixtures, and a GitHub Actions workflow that lints + tests on CPU with **no data download**.

**Dependencies.** S9.

**Actions.**
1. Create tiny synthetic fixtures so tests never need network/secrets: `tests/conftest.py` builds a small fake panel DataFrame (a handful of series × ~24 months) and a fake config dict. **CI must never call yfinance/FRED/Earthdata.**
2. Add the foundational tests referenced above: `tests/test_config.py`, `tests/test_seed.py`, `tests/test_capabilities.py`, `tests/test_manifest.py`, `tests/test_results.py`, plus `tests/test_import.py` (imports every `ntl_etf` submodule that has no heavy deps).
3. GitHub Actions workflow `.github/workflows/ci.yml` (Windows + Linux matrix, core deps only):
   ```yaml
   name: ci
   on:
     push: { branches: ["**"] }
     pull_request: { branches: [main] }
   jobs:
     lint-test:
       strategy:
         fail-fast: false
         matrix:
           os: [ubuntu-latest, windows-latest]
       runs-on: ${{ matrix.os }}
       steps:
         - uses: actions/checkout@v4
         - uses: actions/setup-python@v5
           with: { python-version: "3.11" }
         - name: Install core deps
           run: |
             python -m pip install --upgrade pip
             pip install -r requirements.txt
             pip install -e .
         - name: Ruff
           run: ruff check .
         - name: Black (check)
           run: black --check .
         - name: Pytest (CPU, no network)
           env: { NTL_ETF_NO_NETWORK: "1" }
           run: pytest -q
   ```
   > CI installs only `requirements.txt` (core). It must **not** install `requirements-extras.txt` (mamba-ssm would fail). Mamba/foundation code paths are import-guarded (S4) so the suite passes without them. Tests honor `NTL_ETF_NO_NETWORK=1` by skipping any network-touching test.
4. Local commands (document in README, S11):
   ```powershell
   ruff check .; black --check .; pytest -q
   ```
5. **Optional** pre-commit: add `.pre-commit-config.yaml` running ruff + black; note it is optional and not required for CI to pass.

**Files created/modified.** `tests/conftest.py`, `tests/test_*.py`, `.github/workflows/ci.yml`, optional `.pre-commit-config.yaml`.

**Deliverables.** Test suite + CI.

**Acceptance Criteria.**
- `pytest -q` passes locally on Windows/CPU with zero network calls (verify by running with the network disabled or `NTL_ETF_NO_NETWORK=1`).
- `ruff check .` and `black --check .` exit 0.
- CI workflow is syntactically valid YAML and, when pushed, the `lint-test` job passes on both `ubuntu-latest` and `windows-latest` without installing extras.

---

## S11 — Pipeline runners (`run_all.ps1` + bash) and README

**Objective.** A single command runs the whole pipeline in dependency order; README documents setup, profiles, and reproduction.

**Dependencies.** S2–S10; orchestrates scripts owned by later phases (`download_*`, `build_panel`, `run_experiment`).

**Actions.**
1. `scripts/run_all.ps1` — ordered, fail-fast, capability-aware:
   ```powershell
   #!/usr/bin/env pwsh
   $ErrorActionPreference = "Stop"
   Write-Host "== Phase 0: environment check =="
   python scripts/check_env.py
   Write-Host "== Phase A/B: data acquisition =="
   python scripts/download_ntl.py        # N-tasks
   python scripts/download_finance.py    # F-tasks
   python scripts/download_macro.py       # F-tasks
   Write-Host "== Phase C: panel + splits =="
   python scripts/build_panel.py          # P-tasks
   Write-Host "== Phase D: experiments =="
   python scripts/run_experiment.py --config configs/momentum.yaml
   python scripts/run_experiment.py --config configs/dlinear.yaml
   python scripts/run_experiment.py --config configs/patchtst.yaml
   python scripts/run_experiment.py --config configs/itransformer.yaml
   python scripts/run_experiment.py --config configs/mamba.yaml   # auto-skips on Windows/CPU (S4)
   Write-Host "== Phase E: evaluation + stats =="
   python scripts/run_experiment.py --eval-all
   Write-Host "ALL DONE"
   ```
2. `scripts/run_all.sh` — same order for WSL2/Colab; this profile additionally installs extras and runs Mamba/foundation:
   ```bash
   #!/usr/bin/env bash
   set -euo pipefail
   python scripts/check_env.py
   python scripts/download_ntl.py
   python scripts/download_finance.py
   python scripts/download_macro.py
   python scripts/build_panel.py
   for cfg in momentum dlinear patchtst itransformer mamba; do
     python scripts/run_experiment.py --config "configs/${cfg}.yaml"
   done
   python scripts/run_experiment.py --eval-all
   echo "ALL DONE"
   ```
   > The scripts referenced (`download_*`, `build_panel`, `run_experiment`) are created in Phases A–E. In Phase 0, create **stub** versions that parse args, log "TODO: implemented in Phase X", and exit 0, so `run_all` is end-to-end runnable from day one and each phase fills its stub. Each runner step is idempotent and re-runnable.
3. `README.md` — concise: project one-paragraph summary; quickstart (clone → venv → `pip install -r requirements.txt -e .` → copy `.env.example` to `.env`); the three execution profiles table (link to this file); how to get an Earthdata token + FRED key; how to run `scripts/run_all.ps1`; where results land (`experiments/results.parquet`); and a pointer to `DEVPLAN.md` / `docs/devplan/`. **Do not** restate or duplicate `RESEARCH.MD`.

**Files created/modified.** `scripts/run_all.ps1`, `scripts/run_all.sh`, stub `scripts/{download_ntl,download_finance,download_macro,build_panel,run_experiment,check_env,merge_results}.py`, `README.md`.

**Deliverables.** One-command pipeline runner (both shells) + README.

**Acceptance Criteria.**
- On Windows/CPU, `scripts/run_all.ps1` runs end-to-end against stubs without raising (each stub exits 0; Mamba step logs a skip).
- `scripts/check_env.py` is invoked first and prints the active profile.
- README quickstart, copy-pasted, reproduces a working environment; it links to this file and never modifies `RESEARCH.MD`.

---

## S12 — Git workflow, commit cadence, authorship policy

**Objective.** Define branch naming, commit cadence with task IDs, RESEARCH.MD protection, and the strict authorship rule.

**Dependencies.** S1.

**Actions.**

1. **Branch per phase.** Work on a feature branch off `main`; never commit directly to `main` for substantive work. Naming:
   ```
   phase0-setup        phaseA-data-ntl     phaseB-data-finmacro
   phaseC-panel-splits phaseD-models       phaseE-eval-stats     phaseW-paper
   ```
   ```powershell
   git switch -c phase0-setup
   ```
2. **Commit cadence.** One focused commit per **completed task**, with the task ID as the subject prefix. Examples:
   ```
   S1: bootstrap repo structure, preserve RESEARCH.MD
   S3: pin core (CPU) and extras (GPU) requirements
   S9: editable ntl_etf package + seeding + logging
   ```
   Commit only when the task's Acceptance Criteria pass. Keep commits scoped (don't mix tasks).
3. **Configure the project author identity** (John Roth) for this repo before committing:
   ```powershell
   git config user.name "John Roth"
   git config user.email "<project email on file>"
   ```
   > Verify against the existing repo history (`git log -1 --format='%an <%ae>'`) and match it. Use the user's real identity, never a placeholder.
4. **Authorship policy (hard rule).** Do **NOT** add Claude (or any AI) as an author or co-author **anywhere**: no `Co-Authored-By:` trailer in commits, no author line in the paper, no acknowledgement crediting an AI as a contributor. This overrides any default commit-message convention. Commit messages describe *what changed*, attributed solely to the project author.
   ```powershell
   git commit -m "S1: bootstrap repo structure, preserve RESEARCH.MD"
   ```
   (No trailers. Plain subject + optional body.)
5. **Protect RESEARCH.MD.** Never stage it. Before each commit:
   ```powershell
   if (git diff --cached --name-only | Select-String -Quiet '^RESEARCH\.MD$') {
       throw "RESEARCH.MD must not be modified or staged."
   }
   ```
   Add this check to the optional pre-commit hook. After all phases, confirm `RESEARCH.MD` SHA-256 equals the value captured in S1.
6. **Merging phases.** Open a PR per phase branch into `main`; CI (S10) must be green before merge. GPU-profile results (S4) are committed from the GPU environment on a `results-gpu-<date>` branch or copied in and merged via `merge_results.py`.

**Deliverables.** Documented git workflow (this section is the spec; the agent executes it).

**Acceptance Criteria.**
- Branch `phase0-setup` exists; commits use the `S#:` prefix.
- `git log --format='%an <%ae>%n%b' | Select-String -Pattern 'Claude|Co-Authored-By|Anthropic'` returns **nothing**.
- `git config user.name` / `user.email` match the existing repo history identity (John Roth), not a placeholder.
- `RESEARCH.MD` never appears in any commit's changed files (`git log --name-only -- RESEARCH.MD` shows only the original commit, if any).

---

## Phase 0 — Definition of Done

Phase 0 is complete when **all** of the following hold:

- [ ] **S1** Canonical repo tree exists; `RESEARCH.MD` present and byte-identical to its original SHA-256.
- [ ] **S2** `.venv` on Python 3.11 created and activatable in both PowerShell and bash.
- [ ] **S3** `requirements.txt` (core) installs cleanly on Windows/CPU; `requirements-extras.txt` staged but not installed; `requirements.lock.txt` frozen; `pip check` clean; core import one-liner prints `core OK`.
- [ ] **S4** `capabilities.detect()` works and reports `profile=windows_cpu` on this machine without raising; three execution profiles + GPU-result merge path documented; skip-and-log contract defined.
- [ ] **S5** `.env.example` lists all required secrets; config loader + experiment schema in place; `require_secret` raises helpfully; `.env` is real-secret-only and gitignored.
- [ ] **S6** `.gitignore` ignores data/secrets/venv/checkpoints/caches; keeps manifests + results store tracked (verified via `git check-ignore`).
- [ ] **S7** `write_manifest` produces a JSON with git SHA, seed, capabilities, package versions, and full config.
- [ ] **S8** `append_results` writes the fixed-schema `results.parquet`/`results.csv`; tracker no-ops when disabled.
- [ ] **S9** `pip install -e .` works; `import ntl_etf` from any cwd; seeding is reproducible with documented limits; logging has no duplicate handlers.
- [ ] **S10** `ruff`, `black --check`, and `pytest` all pass on Windows/CPU with **no network**; CI workflow valid and green on the Linux+Windows matrix using core deps only.
- [ ] **S11** `scripts/run_all.ps1` and `scripts/run_all.sh` run end-to-end over stubs without error; README quickstart reproduces the environment; stubs exist for every later-phase script.
- [ ] **S12** `phase0-setup` branch + `S#:`-prefixed commits; author identity is the project owner; **no Claude/AI author or co-author anywhere**; `RESEARCH.MD` never staged.

---

### Notes / TO-VERIFY for the executing agent

- **blackmarblepy API surface (TO-VERIFY).** Recent versions expose **both** a procedural API (`from blackmarble import bm_raster, bm_extract`) taking `product_id` and `token`, and a class-based API (`from blackmarble import BlackMarble, Product`; `bm.raster(gdf, product_id=Product.VNP46A3, date_range=...)`). After installing, run `python -c "import blackmarble, inspect; print([n for n in dir(blackmarble) if not n.startswith('_')])"` and adopt whichever names exist; Phase A (N-tasks) pins the exact call. The bearer token is read from env var **`BLACKMARBLE_TOKEN`**.
- **VNP46A3 variable name (TO-VERIFY).** The default composite variable differs by version (e.g. `NearNadir_Composite_Snow_Free` vs `AllAngle_Composite_Snow_Free`). Phase A enumerates the actual variables from a downloaded sample and selects one explicitly; do not hard-code here.
- **mamba-ssm install (verified).** Will not build on native Windows/CPU — never add it to `requirements.txt`; only Profile 2/3 install `requirements-extras.txt`. Confirmed via state-spaces/mamba issue #662.
- **Foundation packages (verified).** `pip install chronos-forecasting`, `pip install uni2ts` (Moirai), `pip install timesfm[torch]` — CPU-installable, GPU-preferred; treat fine-tuning as GPU-only with CPU zero-shot fallback.

Sources consulted: [BlackMarblePy API](https://worldbank.github.io/blackmarblepy/api/blackmarble.html), [blackmarblepy on PyPI](https://pypi.org/project/blackmarblepy/), [state-spaces/mamba issue #662 (Windows install)](https://github.com/state-spaces/mamba/issues/662), [mamba-ssm on PyPI](https://pypi.org/project/mamba-ssm/), [chronos-forecasting on PyPI](https://pypi.org/project/chronos-forecasting/), [SalesforceAIResearch/uni2ts](https://github.com/SalesforceAIResearch/uni2ts).
