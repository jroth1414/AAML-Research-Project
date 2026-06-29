# Phase C — Models, Pretraining, Transfer Learning, Training Harness, HP Search

This file specifies every model, the training harness, and the hyperparameter search for the
project. All models implement ONE common interface (`BaseForecaster`) so the evaluation phase
(Phase E) is model-agnostic. Build order is **baselines first** (they are the H0/H1 yardstick and
exercise the full prediction-artifact contract end-to-end on CPU), then the two Transformers, then
Mamba, then transfer learning, then foundation models.

> **Cross-phase contract.** Phase B (prefix P) produces the global panel, the windowed tensors, and
> the walk-forward splits. Phase E (prefix E) consumes the *prediction artifact* this phase writes.
> Phase 0 (prefix S) provides the seed utility, config loader, logging, and the **capability flags**
> (`HAS_CUDA`, `HAS_MAMBA_SSM`, `HAS_FOUNDATION`). Do not duplicate those; import them.

## Conventions used in this file

- **Two tasks** (both regression):
  - `leading` — predict FORWARD log return at horizon H ∈ {1,3} months for an ETF.
  - `nowcast` — predict CONTEMPORANEOUS sector industrial-production growth (H is the alignment
    offset defined in Phase F/P; the model still emits a scalar per series per step).
- **Two data views** (defined in Phase P, task P-windowing):
  - **CI (channel-independent)** view: each (region/feature → series) is an independent univariate
    window of shape `(L, 1)`; the global panel stacks all series into the batch dimension. Used by
    PatchTST, Mamba, DLinear, momentum, and all from-scratch/pretrained variants.
  - **Variate view**: a multivariate window of shape `(L, C)` where the C channels are the regions
    feeding ONE target series at a step. Used by iTransformer (cross-region attention).
- **L** = input lookback length ∈ {6,12,24}. **H** = horizon ∈ {1,3}. **Csrc** = number of source
  channels (regions/features) in the variate view for a given target.
- All randomness goes through `utils/seed.py::seed_everything(seed)` (Phase S). Default `seed=1414`.

---

## The prediction artifact (the contract every model must satisfy)

Every `predict()` returns predictions and every experiment run writes them to disk in ONE format so
Phase E never special-cases a model. Define it once here.

`experiments/<run_id>/predictions.parquet` with EXACTLY these columns:

| column        | dtype          | meaning                                                        |
|---------------|----------------|---------------------------------------------------------------|
| `series_id`   | str            | e.g. `XLE` (leading) or `XLE_IP` (nowcast); matches panel id  |
| `task`        | str            | `leading` or `nowcast`                                         |
| `horizon`     | int            | H (months)                                                     |
| `date`        | datetime64[ns] | the timestamp the prediction is FOR (target month)            |
| `y_true`      | float64        | realized target (NaN allowed only for the final live month)   |
| `y_pred`      | float64        | point prediction                                              |
| `fold`        | int            | walk-forward origin index (from Phase P splits)               |
| `split`       | str            | `test` for evaluated rows (val/train rows may be written too) |
| `model`       | str            | model name, e.g. `patchtst`                                   |
| `variant`     | str            | `scratch`, `pretrained`, `zeroshot`, or `finetuned`           |

Each run also writes `experiments/<run_id>/manifest.json` (see M11) and `config.snapshot.yaml`.

Phase E asserts these columns and dtypes; treat them as frozen.

---

## M1 — Common model interface (`BaseForecaster`)

**Objective.** Define the abstract base class all models implement, plus shared utilities (config
binding, standardization handling, artifact assembly), so downstream code targets one interface.

**Dependencies.** S (seed/config/logging utils). Phase P windowing/splits API (column names + the
`PanelDataset` shape contract) — if P is not yet final, code against the documented shapes above and
add a `TODO(P)` assert.

**Actions.**
1. Create `src/ntl_etf/models/base.py`.
2. Define a frozen-ish config dataclass and the ABC:

```python
# src/ntl_etf/models/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np, pandas as pd

@dataclass
class ModelConfig:
    name: str
    task: str               # "leading" | "nowcast"
    L: int = 12             # lookback
    H: int = 1              # horizon
    seed: int = 1414
    # standardization stats are fit on TRAIN ONLY and injected by the trainer (never refit here)
    x_mean: np.ndarray | None = None
    x_std:  np.ndarray | None = None
    y_mean: float | None = None
    y_std:  float | None = None
    extra: dict = field(default_factory=dict)   # model-specific knobs (patch_len, d_model, ...)

class BaseForecaster(ABC):
    """All models implement this. Deep models subclass DeepForecaster (M9)."""
    def __init__(self, config: ModelConfig):
        self.config = config
        self.is_fit = False

    @abstractmethod
    def fit(self, train_view, val_view=None) -> "BaseForecaster": ...
    @abstractmethod
    def predict(self, view) -> np.ndarray:
        """Return point predictions in ORIGINAL (de-standardized) target units, shape (N,)."""
    @abstractmethod
    def save(self, path: str | Path) -> None: ...
    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "BaseForecaster": ...

    # shared helper used by every model to build the artifact rows for one split
    def to_artifact(self, view, y_pred: np.ndarray, *, split: str, variant: str) -> pd.DataFrame: ...
```

3. `to_artifact` reads `series_id/date/fold/y_true` carried on the view (Phase P attaches these as
   aligned metadata arrays) and emits a DataFrame with the exact columns/dtypes above.
4. Standardization rule (leakage guard): models NEVER fit scalers. The trainer fits
   `x_mean/x_std/y_mean/y_std` on the TRAIN fold only and writes them into `ModelConfig`; `predict`
   de-standardizes with `y_pred * y_std + y_mean`. Document this in the class docstring.

**Deliverables.** `src/ntl_etf/models/base.py`.

**Acceptance criteria.**
- `pytest tests/test_base.py` passes: instantiating a trivial subclass and calling `to_artifact`
  yields a DataFrame whose `.columns.tolist()` equals the frozen list and whose dtypes match.
- `ModelConfig` round-trips through `utils/config.py` YAML load/dump unchanged.

---

## M2 — Momentum baseline (`momentum.py`)

**Objective.** Implement the 12-month time-series-momentum baseline = H0/H1 yardstick.

**Dependencies.** M1.

**Actions.**
1. Create `src/ntl_etf/models/momentum.py` → `class MomentumForecaster(BaseForecaster)`.
2. Prediction rule for the `leading` task: predicted next-month return = trailing 12-month MEAN of
   realized monthly log returns ending at the lookback origin. For H=3, predict the same per-month
   mean for each of the 3 steps (or the cumulative 3-month sum if Phase P targets cumulative
   returns — read the target definition from the view metadata and match it; assert which one).
3. `fit` is a no-op except recording that L≥12 is required (if L<12, use all available months and
   log a warning). No parameters, no scaling needed; ignore injected scalers.
4. For `nowcast`, momentum predicts the trailing 12-month mean IP growth (random-walk-in-drift).

**Deliverables.** `src/ntl_etf/models/momentum.py`.

**Acceptance criteria.**
- On a synthetic series with constant +1% monthly return, predicted return ≈ 0.01 (atol 1e-6).
- Runs on the full real panel for `leading`, H=1, CPU, in < 5 s and writes a valid
  `predictions.parquet` (column/dtype assert from M1 passes).
- Directional accuracy computed in Phase E is well-defined (no NaN preds on test rows).

---

## M3 — DLinear baseline (`dlinear.py`)

**Objective.** Implement DLinear: moving-average trend/seasonal decomposition + per-component linear
maps. Strong, fast CPU baseline; second member of the H1 comparison set.

**Dependencies.** M1.

**Actions.**
1. Create `src/ntl_etf/models/dlinear.py` → `class DLinearForecaster(DeepForecaster)` (uses the M9
   trainer; it is a tiny torch module).
2. Module: input `(B, L, 1)` CI view. Decompose with a moving-average kernel
   `kernel_size = configs.extra.get("moving_avg", 25)` (clamp ≤ L; for monthly data with small L use
   `min(L, 13)` and record it). `trend = AvgPool1d(kernel, stride=1, padding=...)`,
   `seasonal = x - trend`. Two `nn.Linear(L, H)` heads (one per component), summed. Channel-shared
   weights (single global model) per the channel-independent design.
3. Standardize inputs with injected `x_mean/x_std`; de-standardize output.

```python
class _DLinear(nn.Module):
    def __init__(self, L, H, kernel):
        super().__init__()
        self.decomp = SeriesDecomp(kernel)
        self.lin_trend = nn.Linear(L, H)
        self.lin_seasonal = nn.Linear(L, H)
    def forward(self, x):                 # x: (B, L, 1)
        s, t = self.decomp(x.squeeze(-1)) # (B, L) each
        return (self.lin_seasonal(s) + self.lin_trend(t)).unsqueeze(-1)  # (B, H, 1)
```

**Deliverables.** `src/ntl_etf/models/dlinear.py`.

**Acceptance criteria.**
- Overfit test (M16): on 4 fixed batches, train MSE drops below 1e-3 within 300 steps.
- Full-panel `leading` H=1 run completes on CPU < 2 min and writes a valid artifact.
- Determinism test (M17) passes.

---

## M4 — Vendoring strategy for thuml models (PatchTST + iTransformer)

**Objective.** Decide and record HOW the Transformer code enters the repo. **Vendor** the specific
model files; do NOT depend on the whole `Time-Series-Library` at runtime.

**Dependencies.** S (license note convention).

**Facts (verified).** `thuml/Time-Series-Library` is **MIT-licensed** (vendoring with attribution is
permitted). Its models take a single `configs` namespace; PatchTST reads
`task_name, seq_len, pred_len, enc_in, d_model, n_heads, e_layers, d_ff, dropout, activation,
factor` plus `patch_len`/`stride` (defaults 16/8). `forward(x_enc, x_mark_enc, x_dec, x_mark_dec,
mask=None)` expects `x_enc` of shape `(B, seq_len, n_vars)` and returns forecast
`(B, pred_len, n_vars)`. iTransformer reads the same `configs` set (no `patch_len`; uses `factor`,
`use_norm`) and inverts the sequence so attention is across variates.

**Actions.**
1. Create `src/ntl_etf/models/_vendor/` with `__init__.py`, `LICENSE.thuml` (copy of the MIT text),
   and `README.md` stating: source repo URL, commit SHA pinned at vendor time, files copied, and
   "minimal edits: removed unused task branches; no algorithmic change."
2. Vendor the minimal dependency closure for PatchTST and iTransformer:
   `layers/Transformer_EncDec.py`, `layers/SelfAttention_Family.py`, `layers/Embed.py`,
   `layers/PatchTST_layers` (RevIN/patch embed), and `models/PatchTST.py`, `models/iTransformer.py`.
   Strip branches for tasks we do not use (imputation/anomaly/classification) to reduce surface.
3. Add a tiny `SimpleNamespace`-style `Configs` builder that maps our `ModelConfig.extra` knobs onto
   the attribute names the vendored modules expect.
4. Pin the vendored commit SHA in `_vendor/README.md` and in `requirements.txt` as a comment
   (the library is NOT a runtime dependency).

**Deliverables.** `src/ntl_etf/models/_vendor/{...}`, `LICENSE.thuml`, `_vendor/README.md`.

**Acceptance criteria.**
- `python -c "from ntl_etf.models._vendor.PatchTST import Model"` imports with no reference to the
  external library installed.
- `_vendor/README.md` contains a concrete commit SHA (not "latest").
- `pip uninstall`-ing any thuml package (it is never installed) does not affect import.

---

## M5 — PatchTST wrapper (`patchtst.py`)

**Objective.** Wrap the vendored PatchTST to `BaseForecaster`, channel-independent global model.

**Dependencies.** M1, M4, M9 (trainer).

**Actions.**
1. Create `src/ntl_etf/models/patchtst.py` → `class PatchTSTForecaster(DeepForecaster)`.
2. Feed the **CI view** `(B, L, 1)` to `x_enc`; pass `x_mark_enc=None`-equivalent zeros (the vendored
   forecast path tolerates absent time marks; if not, build a zeros tensor of the right shape). Take
   the forecast output `(B, H, 1)` and reduce to the H scalars Phase E expects.
3. Config knobs surfaced via `ModelConfig.extra` (with the proposal ranges as defaults for HP search):

   | knob        | type | search range            | maps to              |
   |-------------|------|-------------------------|----------------------|
   | `patch_len` | int  | {4, 6, 12}              | `configs.patch_len`  |
   | `stride`    | int  | {patch_len//2, patch_len}| `configs.stride`    |
   | `d_model`   | int  | {64, 128}               | `configs.d_model`    |
   | `n_heads`   | int  | {4, 8}                  | `configs.n_heads`    |
   | `e_layers`  | int  | {2, 3}                  | `configs.e_layers`   |
   | `d_ff`      | int  | {128, 256}              | `configs.d_ff`       |
   | `dropout`   | float| {0.1, 0.2, 0.3}         | `configs.dropout`    |

4. Guard: require `patch_len ≤ L`; if `L=6` skip `patch_len=12` configs (log + skip, do not crash).
5. Single set of weights shared across ALL series (channel independence): the series dimension is
   folded into the batch; never make weights per-series.

**Deliverables.** `src/ntl_etf/models/patchtst.py`.

**Acceptance criteria.**
- Shape test (M15): input `(8, 12, 1)` → prediction `(8,)` for H=1, `(8,3)` for H=3.
- Overfit test (M16) passes on 4 batches.
- `leading` H=1 full-panel CPU run completes (rough budget: < 30 min on CPU for one config).

---

## M6 — iTransformer wrapper (`itransformer.py`)

**Objective.** Wrap vendored iTransformer to `BaseForecaster` using the **variate view** so attention
operates across regions (this is the architectural lever for H2/H3).

**Dependencies.** M1, M4, M9.

**Actions.**
1. Create `src/ntl_etf/models/itransformer.py` → `class ITransformerForecaster(DeepForecaster)`.
2. Feed the **variate view** `(B, L, Csrc)` so `enc_in = Csrc` (number of source regions for the
   target). Because `Csrc` varies by target ETF, set `enc_in` to the **max** Csrc across the panel
   and zero-pad shorter targets, OR train one global model with a fixed superset of region channels
   and a channel mask. Choose padding+mask; record the mask construction in the wrapper and assert no
   target attends to a region it was not paired with in Phase P screening (leakage guard).
3. Config knobs (proposal ranges as HP defaults):

   | knob       | type  | search range      | maps to            |
   |------------|-------|-------------------|--------------------|
   | `d_model`  | int   | {64, 128, 256}    | `configs.d_model`  |
   | `e_layers` | int   | {2, 3, 4}         | `configs.e_layers` |
   | `n_heads`  | int   | {4, 8}            | `configs.n_heads`  |
   | `d_ff`     | int   | {128, 256, 512}   | `configs.d_ff`     |
   | `dropout`  | float | {0.1, 0.2, 0.3}   | `configs.dropout`  |
   | `use_norm` | bool  | {true}            | `configs.use_norm` |

4. Output: iTransformer forecasts all C variates; select the TARGET channel's H predictions only.

**Deliverables.** `src/ntl_etf/models/itransformer.py`.

**Acceptance criteria.**
- Shape test: input `(8, 12, Csrc)` → target-channel prediction `(8,)` (H=1).
- Padding-mask assert: a unit test feeds a target with `Csrc=2` into a panel padded to `Csrc_max=5`
  and confirms padded channels contribute zero (gradient w.r.t. padded inputs is 0).
- Overfit + determinism tests pass; full `leading` H=1 CPU run completes.

---

## M7 — Mamba (`mamba.py`) with capability gating and CPU fallback

**Objective.** Provide a selective state-space (S6) model for the architecture comparison (H4),
with the official GPU kernel as the primary path and a CPU-safe pure-PyTorch fallback so the
comparison can run even without CUDA. NEVER crash the suite when CUDA/mamba-ssm is absent.

**Dependencies.** M1, M9, S (capability flags).

**Facts (verified).** `mamba-ssm` and its deps `causal-conv1d`/`triton` require CUDA + a Linux/WSL
build toolchain; native-Windows CPU install fails (missing/incompatible `triton`, `nvcc` not found).
Recommended install (on the GPU/WSL profile only):
`pip install mamba-ssm[causal-conv1d] --no-build-isolation`.

**Actions.**
1. Create `src/ntl_etf/models/mamba.py` → `class MambaForecaster(DeepForecaster)`.
2. **Path selection at construction time:**
   - If `S.HAS_MAMBA_SSM and S.HAS_CUDA`: build blocks from `mamba_ssm.Mamba` (official). Set
     `variant_impl="official"`.
   - Else: build the pure-PyTorch S6 fallback (below). Set `variant_impl="fallback"`. Log a WARNING
     and record `impl` in the manifest so Phase E can flag that Mamba results used the fallback.
   - If a config explicitly demands `require_official=true` and the official path is unavailable:
     do NOT crash — write a `skipped.json` with reason `"mamba_ssm_unavailable"` and return cleanly
     so `run_all` continues.
3. **Architecture (both paths):** input CI view `(B, L, 1)` → `Linear(1, d_model)` → `depth` Mamba
   blocks (each: RMSNorm → Mamba/S6 mixer → residual) → take last-step hidden → `Linear(d_model, H)`.
4. **Config knobs (proposal ranges):**

   | knob       | type | search range   | notes                          |
   |------------|------|----------------|--------------------------------|
   | `d_state`  | int  | {16, 32, 64}   | SSM state size                 |
   | `d_model`  | int  | {64, 128}      | channel dim                    |
   | `expand`   | int  | {2}            | block expansion                |
   | `depth`    | int  | {2, 3, 4}      | number of Mamba blocks         |
   | `dropout`  | float| {0.1, 0.2}     | post-block dropout             |

5. **Pure-PyTorch S6 fallback outline** (clearly labeled; numerically faithful to the S6 recurrence,
   NOT the official fused selective-scan kernel — expect speed/precision differences):

```python
class _S6Fallback(nn.Module):
    """CPU-safe selective SSM. Sequential scan (slow but correct). Not the fused kernel."""
    def __init__(self, d_model, d_state, expand):
        super().__init__()
        d_inner = expand * d_model
        self.in_proj  = nn.Linear(d_model, 2 * d_inner)        # x and gate z
        self.conv1d   = nn.Conv1d(d_inner, d_inner, 4, groups=d_inner, padding=3)
        self.x_proj   = nn.Linear(d_inner, d_state * 2 + 1)    # B, C, dt (selective)
        self.dt_proj  = nn.Linear(1, d_inner)
        self.A_log    = nn.Parameter(torch.log(torch.arange(1, d_state+1).float()
                                               .repeat(d_inner, 1)))   # (d_inner, d_state)
        self.D        = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model)
    def forward(self, u):                       # u: (B, L, d_model)
        # 1) in_proj -> x, z ; causal conv + SiLU on x
        # 2) per step t: dt = softplus(dt_proj(...)); discretize A,B via ZOH;
        #    h = exp(dt*A)*h + dt*B*x_t ; y_t = (C·h) + D*x_t        # sequential loop over L
        # 3) y = y * SiLU(z) ; return out_proj(y)
        ...
```

   Keep the loop in pure torch (vectorize over B and d_inner; iterate only over L which is ≤ 24).

**Deliverables.** `src/ntl_etf/models/mamba.py`.

**Acceptance criteria.**
- On a CPU-only machine: constructing `MambaForecaster` selects the fallback, logs the warning, and
  the shape test `(8,12,1)→(8,)` passes; overfit test passes (fallback can memorize 4 batches).
- With `require_official=true` and no CUDA: run writes `skipped.json` with the recorded reason and
  exits 0 (a `run_all` integration check confirms the suite continues).
- Determinism test passes for the fallback (same seed → identical output, CPU).

---

## M8 — Self-supervised pretraining (`pretrain.py`) + paired scratch/pretrained variants

**Objective.** Implement PatchTST-style masked-reconstruction pretraining on the FULL unlabeled NTL
corpus across all regions, then fine-tune a supervised head. Produce, for each deep model, BOTH a
`pretrained` and an identically-sized `scratch` variant so Phase D can test H6.

**Dependencies.** M5, M6, M7, M9, Phase N (NTL corpus), Phase P (unlabeled windowing).

**Actions.**
1. Create `src/ntl_etf/models/pretrain.py`.
2. **Pretraining corpus.** Use ALL region NTL series as univariate CI windows — including regions
   not paired to any sector — with NO target labels. Leakage guard: pretraining may use the full
   history (it is unlabeled and self-supervised), BUT fine-tuning + evaluation still obey the
   walk-forward train-only fit and the VNP46A3 release-lag alignment from Phase P. State this
   explicitly in the module docstring.
3. **Masked objective.** Randomly mask a fraction `mask_ratio ∈ {0.4}` of patches (PatchTST scheme)
   or of timesteps (for Mamba/DLinear which have no patches); reconstruct masked positions; loss =
   MSE on masked positions only.
4. **Shared encoder, swappable head.** `class MaskedPretrainer` wraps a model's encoder, attaches a
   reconstruction head for pretraining, then swaps in the forecasting head for fine-tuning. The
   encoder weights transfer; the head is reinitialized.
5. **Variant production.** Provide `build_variant(model_name, cfg, variant)`:
   - `variant="scratch"` → random init, train head+encoder on the supervised task.
   - `variant="pretrained"` → load pretrained encoder, fine-tune end-to-end (smaller lr on encoder
     optional). Both variants use the IDENTICAL architecture/param count so H6 is a fair test.
6. Save pretrained encoders to `experiments/_pretrained/<model>/<hash>.pt` keyed by a config hash so
   repeated fine-tunes reuse them.

**Deliverables.** `src/ntl_etf/models/pretrain.py`, pretrained checkpoints under `experiments/_pretrained/`.

**Acceptance criteria.**
- Pretraining one model on the corpus for a few hundred steps reduces masked-reconstruction MSE
  monotonically (logged), on CPU, in a smoke run (`--smoke` limits steps).
- `count_parameters(scratch) == count_parameters(pretrained)` for every model (assert in a test).
- A pretrained run and a scratch run both write valid artifacts with `variant` set correctly.

---

## M9 — Unified Trainer (`train/trainer.py`)

**Objective.** One config-driven training loop for every torch model: seeding, optional AMP,
gradient clipping, early stopping on the validation fold, checkpointing, deterministic dataloading,
structured logging, and artifact + manifest writing.

**Dependencies.** M1, S (seed/logging/config). Phase P `PanelDataset` / split API.

**Actions.**
1. Create `src/ntl_etf/train/trainer.py` and `src/ntl_etf/models/base.py::DeepForecaster`
   (a `BaseForecaster` subclass that owns an `nn.Module` and delegates `fit/predict` to the Trainer).
2. Trainer responsibilities:
   - `seed_everything(cfg.seed)`; set `torch.use_deterministic_algorithms(True)` where supported,
     `torch.backends.cudnn.deterministic=True`, `cudnn.benchmark=False`; set
     `CUBLAS_WORKSPACE_CONFIG=:4096:8` for deterministic CUDA matmul; DataLoader with
     `num_workers=0` (Windows-safe) and a seeded `generator`.
   - Fit `x_mean/x_std/y_mean/y_std` on the TRAIN fold ONLY; inject into `ModelConfig` (leakage
     guard — never recompute on val/test).
   - Loss = MSE on standardized targets. Optimizer AdamW; cosine or plateau LR schedule.
   - **AMP** only if `HAS_CUDA` and `cfg.amp`; otherwise full fp32 (CPU).
   - **Gradient clipping** `clip_grad_norm_(params, cfg.clip=1.0)`.
   - **Early stopping** on val MSE, `patience=cfg.patience (default 15)`, restore best weights.
   - **Checkpointing** best + last to `experiments/<run_id>/ckpt/`.
   - Structured per-epoch logs (JSONL) via `utils/logging.py`.
3. On completion, build the prediction artifact for the TEST split (and optionally val/train),
   write `predictions.parquet`, `manifest.json` (M11), and `config.snapshot.yaml`.

```python
class Trainer:
    def __init__(self, cfg: ModelConfig, train_cfg: TrainConfig): ...
    def fit(self, module: nn.Module, train_view, val_view) -> dict: ...   # returns history
    def predict(self, module: nn.Module, view) -> np.ndarray: ...
```

**Deliverables.** `src/ntl_etf/train/trainer.py`; `DeepForecaster` in `base.py`.

**Acceptance criteria.**
- Determinism: two `fit` calls, same seed/config, produce bit-identical val curves and identical
  test predictions (M17).
- Early stopping triggers on a synthetic overfitting run (history shows restored best epoch).
- Runs CPU-only with `amp=False` and GPU with `amp=True` (guarded by `HAS_CUDA`).
- Writes a complete `experiments/<run_id>/` directory with all four artifacts.

---

## M10 — Foundation models (`foundation.py`): zero-shot first, fine-tune optional

**Objective.** Wrap Chronos / Moirai (uni2ts) / TimesFM for (a) CPU-feasible zero-shot reference
forecasts and (b) OPTIONAL GPU fine-tuning, staged as a later milestone. The project must be
complete (H1–H5 testable) even if fine-tuning never runs.

**Dependencies.** M1, S (capability flags), Phase P windowing.

**Facts (verified).**
- **Chronos:** `pip install chronos-forecasting`; OS-independent, Python ≥3.10, runs on CPU.
  `from chronos import BaseChronosPipeline` / `ChronosPipeline`;
  `pipe = ChronosPipeline.from_pretrained("amazon/chronos-bolt-small", device_map="cpu")`; forecast
  via `pipe.predict(context=tensor, prediction_length=H)` (quantiles via `predict_quantiles`).
  Use the small/bolt variants on CPU. (Newer `Chronos2Pipeline`/`amazon/chronos-2` exists; prefer a
  pinned small model for CPU determinism — TO-VERIFY the exact small-model id at install time with
  `python -c "from chronos import ChronosPipeline"` and the model card.)
- **Moirai:** `pip install uni2ts`; load `MoiraiForecast`/`MoiraiModule.from_pretrained(
  "Salesforce/moirai-1.1-R-small")`; inference goes through GluonTS predictors. Small model is
  CPU-feasible for short contexts.
- **TimesFM:** `pip install timesfm[torch]`; `timesfm.TimesFm(...)` then `forecast(...)`. Larger;
  prefer GPU. Provide a CPU zero-shot attempt but allow `skipped.json` if it OOMs/too slow.

**Actions.**
1. Create `src/ntl_etf/models/foundation.py` with a wrapper per model implementing `BaseForecaster`,
   each producing artifacts with `variant="zeroshot"` (and `"finetuned"` on the GPU profile).
2. Input adaptation: foundation models take the raw univariate target history as the context (NOT
   the NTL features — they are general TS models). For `leading`, the context is the ETF return
   history; for `nowcast`, the IP history. Make this explicit: foundation models serve as a
   "no-NTL" reference, useful framing for H1/H5 discussion.
3. Respect the same walk-forward alignment: context ends at the last allowed month per the Phase P
   split; predict H ahead; map to artifact rows.
4. Gate behind `S.HAS_FOUNDATION` (set if the packages import). If a package is missing or a model
   OOMs on CPU: write `skipped.json` with the reason; never crash.
5. Fine-tuning (`finetune.py` path inside `foundation.py`) is an OPTIONAL milestone gated behind
   `HAS_CUDA`; document the command but mark it `OPTIONAL — GPU profile`.

**Deliverables.** `src/ntl_etf/models/foundation.py`.

**Acceptance criteria.**
- Zero-shot Chronos (bolt/small) produces a valid `leading` H=1 artifact on CPU for at least the
  longest-history ETF (e.g. XLE), in a smoke run.
- Missing-package path writes `skipped.json` and exits 0.
- `variant` column correctly reads `zeroshot`.

---

## M11 — Run manifest schema

**Objective.** Standardize the per-run manifest so runs are reproducible and Phase E/W can audit.

**Dependencies.** M9.

**Actions.** `experiments/<run_id>/manifest.json` MUST contain:

```json
{
  "run_id": "patchtst__leading__H1__L12__seed1414__<8charhash>",
  "model": "patchtst", "variant": "scratch", "task": "leading",
  "L": 12, "H": 1, "seed": 1414,
  "hp": { "patch_len": 6, "d_model": 128, "lr": 5e-4, "batch": 32, "dropout": 0.2 },
  "n_params": 123456, "device": "cpu", "amp": false,
  "mamba_impl": null,
  "train_window_months": 60, "n_folds": 84,
  "git_commit": "<sha>", "python": "3.11.x", "torch": "x.y.z",
  "started": "ISO8601", "finished": "ISO8601", "duration_sec": 0.0,
  "val_metric": {"mse": 0.0}, "selected_on": "val",
  "data_hashes": {"panel": "<sha256>", "splits": "<sha256>"}
}
```

**Deliverables.** `manifest.json` writer in `utils/io.py` + Trainer integration.

**Acceptance criteria.** Every run emits a manifest validating against a JSON schema in
`tests/manifest.schema.json`; `git_commit` and `data_hashes` are populated (not null).

---

## M12 — HP search (`train/hpsearch.py`)

**Objective.** Grid search over the proposal ranges; selection ONLY on the validation fold; the best
validated config per model is the single config evaluated on test.

**Dependencies.** M5, M6, M7, M9, M11. Phase P (val fold definition).

**Actions.**
1. Create `src/ntl_etf/train/hpsearch.py`.
2. **Shared grid (all deep models):**

   | hyperparam | values            |
   |------------|-------------------|
   | `L`        | {6, 12, 24}       |
   | `H`        | {1, 3}            |
   | `lr`       | {1e-4, 5e-4, 1e-3}|
   | `batch`    | {16, 32}          |
   | `dropout`  | {0.1, 0.2, 0.3}   |

   Plus the model-specific knobs from M5/M6/M7. Note H is part of the experiment spec, not tuned away
   — search L/lr/batch/dropout/model-knobs **separately per (task, H)**.
3. **Selection rule (leakage guard):** rank configs by mean validation MSE across walk-forward folds
   ONLY. Never look at test. Persist the full grid's val scores to
   `experiments/_hpsearch/<model>__<task>__H<h>.csv` and the winner to `..._best.yaml`.
4. **Multiple-comparison hygiene note for the agent:** HP search inflates apparent performance. Keep
   the grid modest, do selection on val only, and let Phase E apply the pre-registered DM tests with
   the agreed multiple-comparison correction (do NOT re-tune after seeing test). Record this in the
   manifest as `selected_on: "val"`.
5. Provide `--smoke` (1–2 configs, few epochs) and `--full` modes.

```bash
# PowerShell
& $PY -m ntl_etf.train.hpsearch --model patchtst --task leading --H 1 --mode full
# bash
$PY -m ntl_etf.train.hpsearch --model patchtst --task leading --H 1 --mode full
```

**Deliverables.** `src/ntl_etf/train/hpsearch.py`, `experiments/_hpsearch/*`.

**Acceptance criteria.**
- Produces `<model>__<task>__H<h>_best.yaml`; the chosen config's val MSE equals the grid minimum.
- A unit test confirms the selector never reads any row with `split=="test"`.

---

## M13 — Single-experiment runner (`scripts/run_experiment.py`)

**Objective.** Run ONE model + ONE resolved config end-to-end and write the run directory.

**Dependencies.** M2, M3, M5, M6, M7, M8, M9, M10, M11.

**Actions.**
1. Create `scripts/run_experiment.py`. CLI:

```bash
& $PY scripts/run_experiment.py --model patchtst --task leading --H 1 \
     --config experiments/_hpsearch/patchtst__leading__H1_best.yaml \
     --variant scratch --seed 1414
```

2. Resolve config → build model via a `MODEL_REGISTRY` dict
   (`{"momentum":..., "dlinear":..., "patchtst":..., "itransformer":..., "mamba":...,
   "chronos":..., "moirai":..., "timesfm":...}`) → fit (walk-forward via Phase P splits) →
   write artifact + manifest. Compute and print the `run_id`.
3. Honor capability flags; on a gated-unavailable model, write `skipped.json` and exit 0.

**Deliverables.** `scripts/run_experiment.py`.

**Acceptance criteria.**
- `--model momentum --task leading --H 1` produces a complete run dir on CPU in seconds.
- Unknown `--model` exits non-zero with a clear message; gated-unavailable model exits 0 with skip.

---

## M14 — Full-matrix orchestration (`scripts/run_all.ps1`)

**Objective.** Orchestrate the entire experiment matrix in a compute-aware order so CPU results land
early and GPU-only work is deferred.

**Dependencies.** M12, M13.

**Actions.**
1. Create `scripts/run_all.ps1` (and mirror logic in `scripts/run_all.sh` for bash) that iterates:
   tasks × H × models × variants, calling hpsearch then run_experiment, skipping gated-unavailable
   combos. Order by tier (see M18). Accept `-Tier cpu|gpu|all` and `-Smoke`.
2. Aggregate all `predictions.parquet` into `experiments/_index.csv` (one row per run with manifest
   fields) for Phase E.

```powershell
# PowerShell
pwsh scripts/run_all.ps1 -Tier cpu        # baselines + PatchTST + iTransformer + Chronos zero-shot
pwsh scripts/run_all.ps1 -Tier gpu        # Mamba (official) + foundation fine-tune (GPU/WSL)
```

**Deliverables.** `scripts/run_all.ps1`, `scripts/run_all.sh`, `experiments/_index.csv`.

**Acceptance criteria.**
- `-Tier cpu -Smoke` completes on a CPU-only Windows box with zero crashes; `_index.csv` lists every
  CPU run and marks gated GPU runs as `skipped`.

---

## M15 — Shape tests

**Objective.** Prove every model maps the documented input view to the documented output shape.

**Dependencies.** M2–M7, M10.

**Actions.** `tests/test_shapes.py`: for each model, feed random tensors of the correct view and
assert output shapes for H ∈ {1,3}: CI models accept `(B, L, 1)`; iTransformer accepts
`(B, L, Csrc)`; outputs reduce to `(B,)` (H=1) or `(B,3)` (H=3).

**Acceptance criteria.** All shape assertions pass for every available model (gated models skipped
via `pytest.importorskip`/flag).

---

## M16 — Tiny-fixture overfit test

**Objective.** Each trainable model can memorize a few batches (sanity that learning works).

**Dependencies.** M3, M5, M6, M7, M9.

**Actions.** `tests/test_overfit.py`: fixed 4 batches, train to ≥300 steps, assert final train MSE
< 1e-3 (relax to < 1e-2 for the Mamba CPU fallback). Use `seed=0`.

**Acceptance criteria.** Test passes on CPU for DLinear, PatchTST, iTransformer, Mamba-fallback.

---

## M17 — Determinism test

**Objective.** Same seed + same config ⇒ identical outputs.

**Dependencies.** M9, M13.

**Actions.** `tests/test_determinism.py`: run the same `run_experiment` config twice with
`seed=1414`; assert `predictions.parquet` `y_pred` columns are equal
(`np.testing.assert_array_equal`) and manifests agree on `n_params`/`val_metric`.

**Acceptance criteria.** Passes on CPU for DLinear and PatchTST at minimum (the determinism-critical
core); document any model where bitwise determinism is not achievable and assert `allclose(atol=1e-6)`
instead, with the reason recorded.

---

## M18 — Compute-budget note and execution ordering

**Objective.** Tell the agent what runs where and in what order so the project produces results
incrementally without blocking on GPU.

**Dependencies.** none (planning note); referenced by M14.

**Guidance.**

| Tier | Runs                                                                 | Where        | Notes |
|------|----------------------------------------------------------------------|--------------|-------|
| 0    | momentum, DLinear (both tasks, H∈{1,3})                              | CPU          | minutes; establishes H0/H1 yardstick first |
| 1    | PatchTST, iTransformer (scratch), Chronos/Moirai zero-shot          | CPU          | hours; H1/H2/H3 become testable |
| 2    | pretraining + pretrained variants of PatchTST/iTransformer (H6)     | CPU (slow) / GPU | run on GPU if available, else overnight CPU smoke→full |
| 3    | Mamba official kernel, foundation-model FINE-TUNING                 | GPU / WSL2   | gated; deferred; fallback Mamba can run on CPU for the architecture comparison |

**Mamba fallback path for the GPU-less environment (record in README and `00-overview-and-setup.md`):**
1. **Preferred:** WSL2 + CUDA, then `pip install mamba-ssm[causal-conv1d] --no-build-isolation`.
2. **Alternative:** Google Colab / cloud GPU notebook (`notebooks/`), export checkpoints back to
   `experiments/`.
3. **Always-available:** the pure-PyTorch S6 fallback (M7), CPU-runnable, clearly labeled as not the
   fused kernel; Phase E must footnote that Mamba results may be fallback-derived.

**Acceptance criteria.** `scripts/run_all.ps1 -Tier cpu` yields complete Tier 0–1 results (and Tier
2 if time permits) with NO GPU; Tier 3 is cleanly skipped-with-reason on CPU-only hosts.

---

## Pitfalls checklist (the agent must actively handle these)

- **Mamba on Windows/CPU:** never `pip install mamba-ssm` on native Windows in the CPU profile — it
  fails on `triton`/`nvcc`. Gate strictly behind `HAS_MAMBA_SSM and HAS_CUDA`; default to fallback.
- **Ragged ETF histories:** XLRE/XLC inception is later than 2013 (verify inception dates in Phase F);
  the trainer must tolerate series with fewer folds and never pad with look-ahead values. The global
  panel folds the series into the batch, so short series simply contribute fewer rows.
- **Leakage:** scalers fit on TRAIN only (M9); pretraining is unlabeled but fine-tune/eval still obey
  walk-forward + VNP46A3 release-lag alignment (Phase P); iTransformer channel mask must forbid
  unscreened region→target attention (M6).
- **Small-sample overfitting:** ~144 months/series; keep models small (prefer the lower ends of the
  d_model/depth grids), use dropout, early stopping, and weight decay; report params per model (M11).
- **Multiple comparisons:** modest grid, val-only selection, no post-hoc re-tuning; DM tests +
  correction live in Phase E. Record `selected_on: "val"` in every manifest.
- **Non-determinism:** seed everything, `num_workers=0` on Windows, set
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`, document any model that needs `allclose` instead of exact.
- **Foundation-model size/OOM on CPU:** prefer bolt/small Chronos and small Moirai; allow
  `skipped.json` for TimesFM on CPU; pin model ids and verify the exact id at install (TO-VERIFY in
  M10).
- **License:** vendored thuml code carries its MIT `LICENSE.thuml` + pinned SHA; do NOT add Claude as
  an author anywhere in code, manifests, or the paper.

## Phase-C deliverables summary

`src/ntl_etf/models/{base,momentum,dlinear,patchtst,itransformer,mamba,pretrain,foundation}.py`,
`src/ntl_etf/models/_vendor/{...}`, `src/ntl_etf/train/{trainer,hpsearch}.py`,
`scripts/{run_experiment.py,run_all.ps1,run_all.sh}`,
`tests/{test_base,test_shapes,test_overfit,test_determinism}.py`, `tests/manifest.schema.json`,
and the `experiments/` artifact/manifest contract consumed by Phase E.
