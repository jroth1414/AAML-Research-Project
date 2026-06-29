# Phase D -- Evaluation, Significance Testing, and Stratified Hypothesis Analysis

This phase consumes the per-fold predictions written by Phase M (model training, prefix `M`) and the
walk-forward splits / panel from Phase P (prefix `P`), and turns them into the pre-registered,
leakage-safe, statistically-defensible results that the paper (Phase W, prefix `W`) reports. Nothing
here trains a model; everything here is deterministic given the prediction files. All comparisons are
**pre-registered** below so the autonomous agent cannot retroactively pick favorable cuts.

Hard rules for this phase:

- **Wins are only claimed when Diebold-Mariano (DM) significant** after multiple-comparison correction
  at `alpha = 0.10` (family-wise / FDR controlled). Lower mean MSE alone is *not* a win.
- **Report H0 honestly.** If no deep-learning (DL) model significantly beats the 12-month momentum
  baseline, the hypotheses-verdict summary must say so explicitly. Do not bury it.
- **No look-ahead anywhere.** Every metric is computed from predictions that Phase P/M already aligned
  with the VNP46A3 release lag (NTL of month *t* predicts return of month *t+1*). This phase never
  re-aligns dates; it only validates the alignment via an audit (E2) and then aggregates.
- **Everything reproducible from the results store.** Figures and tables are regenerated from the tidy
  results parquet, never hand-edited.

---

## Inputs this phase assumes (cross-phase dependencies)

| What | Produced by | Path / contract |
|---|---|---|
| Per-fold predictions | M (trainer) | `experiments/<run_id>/predictions/*.parquet` |
| Run manifest (seeds, config hash, model, task) | M / S | `experiments/<run_id>/manifest.json` |
| Walk-forward fold definitions | P (`P*` splits) | `data/processed/folds.parquet` |
| Region->sector pairing + ETF region-count | P | `configs/regions.yaml`, `data/processed/panel_meta.parquet` |
| VIX monthly + disruption flag (`vix_mean > 25`) | F (`F*`) | `data/processed/vix_monthly.parquet` |
| Sector IP (nowcast target) | F | `data/processed/macro_ip.parquet` |
| Seed / config / io / logging utils | S (`S*`) | `src/ntl_etf/utils/*` |

### Canonical prediction-file schema (contract with Phase M)

Phase M MUST write one tidy parquet per `(run_id)` (or per model/task) with **exactly** these columns.
E1 begins by asserting this schema; if it fails, stop and report the mismatch (do not silently coerce).

| column | dtype | meaning |
|---|---|---|
| `model` | str | one of `momentum`, `dlinear`, `patchtst`, `itransformer`, `mamba`, `patchtst_pretrained`, `chronos`, `moirai`, `timesfm` |
| `task` | str | `leading` (forward log return) or `nowcast` (contemporaneous IP) |
| `etf` | str | `XLB`,`XLC`,`XLE`,`XLF`,`XLI`,`XLK`,`XLP`,`XLRE`,`XLU`,`XLV`,`XLY` |
| `fold` | int | walk-forward fold index (rolling origin), from P |
| `date` | datetime64[ns] | the **target** month being predicted (month-end) |
| `y_true` | float | realized value: forward 1-month log return (`leading`) or contemporaneous IP value/log-change (`nowcast`) |
| `y_pred` | float | model prediction for the same target |
| `horizon` | int | forecast horizon in months (default 1; 2 and 3 reserved) |
| `pretrained` | bool | True if weights were pretrained/foundation-initialized (H6) |
| `seed` | int | RNG seed for this run |

> **Nowcast target form (TO-VERIFY against Phase F output).** Phase F may emit IP either as a level
> index or as month-over-month log change. E-phase code reads `configs/experiment.yaml:nowcast_target_form`
> (`level` | `logchange`) and computes R^2 on whatever Phase F standardized on. Assert the two phases
> agree; if `panel_meta` lacks this key, fail loudly. Returns for the `leading` task are always log returns.

---

## E1 -- Results-store schema, loader, and tidy long-format store

**Objective.** Define the canonical results schema, load+validate all prediction files, and persist a
single tidy results store keyed by `(model, task, etf_or_pooled, fold, stratum, metric)`.

**Dependencies:** M (predictions), P (folds), F (vix, ip). No other E task; this is the foundation.

**Actions.**
1. Create `src/ntl_etf/eval/results.py` with the prediction loader and the results-store writer.
2. Define the **results-store schema** (long/tidy — one metric value per row):

   | column | dtype | notes |
   |---|---|---|
   | `model` | str | as above |
   | `task` | str | `leading` / `nowcast` |
   | `scope` | str | individual ETF ticker, or `POOLED` |
   | `fold` | int | walk-forward fold, or `-1` for cross-fold aggregate rows |
   | `stratum` | str | `all`, `single_region`, `multi_region`, `disruption`, `stable`, `pretrained`, `from_scratch` |
   | `metric` | str | `mse`,`mae`,`dir_acc`,`sharpe_gross`,`sharpe_net`,`nowcast_r2`,`pearson`,`n_obs` |
   | `value` | float | the metric value |
   | `ci_low` | float | lower bound of dispersion/CI (NaN for raw per-fold rows) |
   | `ci_high` | float | upper bound (NaN for per-fold rows) |
   | `run_id` | str | provenance back to `experiments/<run_id>` |

3. Loader contract:
```python
# src/ntl_etf/eval/results.py
import pandas as pd

REQUIRED_PRED_COLS = ["model","task","etf","fold","date","y_true","y_pred",
                      "horizon","pretrained","seed"]

def load_predictions(experiments_dir: str, run_ids: list[str] | None = None) -> pd.DataFrame:
    """Read+concat all prediction parquets, assert schema, dedup on
    (model,task,etf,fold,date,seed,horizon), sort by date. Raise on missing cols,
    NaN in y_true/y_pred, or duplicate keys."""

def write_results_store(df_long: pd.DataFrame, out_path: str) -> None:
    """Validate against RESULTS_SCHEMA, write parquet + a CSV mirror."""

def read_results_store(path: str) -> pd.DataFrame: ...
```
4. Output paths: `experiments/results_store.parquet` and `experiments/results_store.csv`.

**Pitfalls to guard.**
- **Ragged ETF histories** (XLC inception 2018-06, XLRE 2015-10). Folds for those ETFs start late; the
  loader must NOT fabricate rows. Assert that each ETF's earliest `date` matches `panel_meta` inception.
- **Duplicate seeds.** If multiple seeds exist per key, keep all rows; aggregation (E6) averages over
  seeds *and* folds and records `n_seeds`.

**Deliverables.** `src/ntl_etf/eval/results.py`; `experiments/results_store.parquet` (+ `.csv`).

**Acceptance criteria.**
- `pytest tests/test_results.py::test_schema_roundtrip` passes (write then read yields identical frame).
- Loading the predictions raises `ValueError` if any required column is missing (covered by a test that
  drops a column).
- `load_predictions` on a fixture with a duplicate `(model,task,etf,fold,date,seed,horizon)` raises.
- For the real run: number of `leading` rows per ETF equals `(n_months_after_inception - 1)` summed
  over folds (the `-1` is the release-lag drop); assert this against `folds.parquet`.

---

## E2 -- Leakage / alignment audit (defensive re-check)

**Objective.** Independently re-verify, at evaluation time, that NTL release-lag alignment was respected
and that no fold's prediction uses information dated on/after its target month.

**Dependencies:** E1, P (leakage audit in `P*`), N (release-lag metadata).

**Actions.**
1. Add `audit_alignment(preds, folds, panel_meta) -> dict` to `src/ntl_etf/eval/results.py`.
2. Checks (each returns pass/fail + offending rows):
   - **Release-lag:** for `leading`, the feature month feeding each `date` is `date - 1 month` at the
     latest. Cross-check against `panel_meta` which records `feature_month` per `(etf,date)`; assert
     `feature_month <= date - 1 month` for every row (E-phase reads, does not recompute, the mapping).
   - **Train/test disjointness:** for each fold, `max(train_date) < min(test_date)`.
   - **Standardization provenance:** assert `manifest.json` records `scaler_fit_on == "train"` for every
     run (Phase M writes this; if absent, FAIL).
3. Write `experiments/<run_id>/audit_alignment.json` per run and a pooled `experiments/audit_summary.json`.

**Acceptance criteria.**
- `audit_alignment` returns `{"release_lag": "pass", "disjoint": "pass", "scaler": "pass"}` for the real
  runs; any fail aborts E-phase with a nonzero exit and the offending rows printed.
- `tests/test_audit.py` includes a synthetic leaked frame (`feature_month == date`) and asserts the audit
  flags it.

---

## E3 -- `metrics.py`: point, directional, trading, and nowcast metrics

**Objective.** Implement every scalar metric with exact, unit-tested formulas.

**Dependencies:** E1.

**File:** `src/ntl_etf/eval/metrics.py`. Pure functions over numpy arrays; no global state.

### Formulas (exact)

Let `y` be realized values and `p` predictions, length `n`.

- **MSE** = `(1/n) * Σ (y_i - p_i)^2`
- **MAE** = `(1/n) * Σ |y_i - p_i|`
- **Directional accuracy** (sign hit rate), chance = 0.50:
  `dir_acc = (1/n) * Σ 1[ sign(p_i) == sign(y_i) ]`. **Zeros:** define `sign(0)=+1` for both prediction
  and target (document this; a flat truly-zero return is rare in monthly data). Optionally exclude exact
  `y_i == 0` and record `n_used`; default is *include with sign(+1)*.
- **Long/short strategy return** (the only strategy we trade): position
  `pos_i = +1 if p_i > 0 else -1`. Strategy gross return in month *i*: `r_i = pos_i * y_true_i`, where
  `y_true_i` is the realized **log return** of the ETF that month.
  - Net of costs: turnover `τ_i = |pos_i - pos_{i-1}|` (∈ {0,2}); first period `τ_0 = |pos_0|` (= entry
    cost). Cost per unit of turnover = `c = 0.0010` (10 bps one-way). Net return
    `r^net_i = r_i - c * τ_i`. (A flip from -1 to +1 trades 2 units => 20 bps round-trip, matching
    "10 bps one-way".)
- **Annualized Sharpe** (risk-free = 0), monthly returns -> annual:
  `Sharpe = sqrt(12) * mean(r) / std(r, ddof=1)`. Compute `sharpe_gross` from `r_i` and `sharpe_net`
  from `r^net_i`. If `std == 0` or `n < 2`, return `NaN` (record, do not crash).
- **Nowcast R^2** (predicted vs actual contemporaneous IP), out-of-sample (vs the *realized mean of the
  test targets*, not the train mean — report the standard OOS R^2 and document it):
  `R2 = 1 - Σ(y_i - p_i)^2 / Σ(y_i - ȳ)^2`, `ȳ = mean(y)`. (R^2 can be negative; keep the sign.)
- **Pearson correlation** of `p` vs `y`: `pearson = scipy.stats.pearsonr(p, y).statistic`; return `NaN`
  if either array is constant.

### Signatures
```python
# src/ntl_etf/eval/metrics.py
import numpy as np

def mse(y, p) -> float: ...
def mae(y, p) -> float: ...
def directional_accuracy(y, p, zero_sign: int = 1, exclude_zero_target: bool = False) -> float: ...

def strategy_returns(y_true, y_pred, cost_one_way: float = 0.0010,
                     prev_pos: int | None = None) -> dict:
    """Returns {'gross': np.ndarray, 'net': np.ndarray, 'positions': np.ndarray,
               'turnover': np.ndarray}. prev_pos carries the position across fold
               boundaries when chaining a continuous equity curve (default None => flat start)."""

def annualized_sharpe(returns, periods_per_year: int = 12) -> float: ...
def nowcast_r2(y, p) -> float: ...
def pearson_corr(y, p) -> float: ...

def compute_all_point_metrics(df_group) -> dict:
    """Given a sub-frame for one (model,task,scope,stratum), return all metrics as a dict."""
```

**Pitfalls.**
- **Cost direction across folds.** Per-fold metrics start flat (`prev_pos=None`). The *continuous equity
  curve* figure (E9) chains positions across the whole pooled timeline, so it passes `prev_pos`.
- **Sharpe annualization** uses `sqrt(12)` exactly; do not use 252 (these are monthly returns).
- **R^2 baseline choice** is OOS mean of test targets; record it in the metric docstring so the paper is
  consistent. Do not silently switch to train-mean R^2.

**Deliverables.** `src/ntl_etf/eval/metrics.py`.

**Acceptance criteria (hand-computed fixtures in `tests/test_metrics.py`).**
- `y=[1,-1,2,-2]`, `p=[0.5,-0.5,1,1]` => `dir_acc == 0.75` (last sign wrong).
- MSE/MAE on `y=[0,2]`, `p=[0,0]` => `mse==2.0`, `mae==1.0`.
- Sharpe: returns `[0.01]*12` => `mean=0.01,std=0 => NaN`; returns alternating `[0.02,-0.01]*6` checked
  against a hand value; `annualized_sharpe([0.01,0.02,0.03], 12)` matches `sqrt(12)*mean/std(ddof=1)`.
- Turnover/cost: `y_true=[0.10,0.10,-0.10]`, `y_pred=[1,-1,-1]` => positions `[+1,-1,-1]`,
  turnover `[1,2,0]`, gross `[0.10,-0.10,0.10]`, net `[0.10-0.001, -0.10-0.002, 0.10-0.0]`.
- `nowcast_r2(y, y) == 1.0`; `nowcast_r2(y, [ȳ]*n) == 0.0`; a worse-than-mean prediction yields R^2 < 0.
- `pearson_corr` of perfectly correlated arrays == 1.0; constant array => `NaN`.

---

## E4 -- `stats.py`: Diebold-Mariano with HAC variance + small-sample correction

**Objective.** Implement the pairwise DM test on the loss differential with Newey-West (HAC) long-run
variance and the Harvey-Leybourne-Newbold (HLN, 1997) small-sample correction, returning a t-referenced
p-value. Small test sets per fold make the correction mandatory.

**Dependencies:** E3 (loss functions), E1.

**File:** `src/ntl_etf/eval/stats.py`.

### Method (exact)

For two models A and B over aligned targets, with per-observation loss `L(e) = e^2` (squared error;
default loss for MSE-based comparisons) or `L(e) = |e|` (absolute) selectable:

1. Loss differential `d_i = L(e^A_i) - L(e^B_i)`, `i = 1..T`. Negative mean => A better.
2. Sample mean `d̄ = mean(d)`.
3. **HAC (Newey-West) long-run variance** of `d̄`:
   `γ_0 + 2 Σ_{k=1}^{h-1} γ_k`, where `γ_k = (1/T) Σ_{i=k+1}^{T} (d_i - d̄)(d_{i-k} - d̄)`, and `h` is the
   forecast horizon (truncation lag = `h - 1`; for `h = 1` this reduces to `γ_0`, i.e. plain variance).
   `DM = d̄ / sqrt( (1/T) * (γ_0 + 2 Σ_{k=1}^{h-1} γ_k) )`.
4. **HLN small-sample correction** (confirmed formula): multiply DM by
   `sqrt( (T + 1 - 2h + h(h-1)/T) / T )`, then compare the corrected statistic to a Student-t with
   `T - 1` degrees of freedom (two-sided).
   `DM* = DM * sqrt((T + 1 - 2h + h(h-1)/T)/T)`; `p = 2 * t.cdf(-|DM*|, df=T-1)`.

### Signature
```python
# src/ntl_etf/eval/stats.py
from dataclasses import dataclass
import numpy as np

@dataclass
class DMResult:
    stat: float          # HLN-corrected statistic DM*
    pvalue: float        # two-sided, t(T-1)
    mean_diff: float     # d̄ (negative => model A better)
    n: int               # T
    horizon: int
    loss: str            # "mse" | "mae"
    better: str | None   # "A","B", or None if p>=alpha (filled by caller post-correction)

def diebold_mariano(e_a, e_b, horizon: int = 1, loss: str = "mse",
                    alternative: str = "two-sided") -> DMResult:
    """e_a, e_b are forecast ERROR arrays (y_true - y_pred), aligned and equal-length.
    Returns HLN-corrected DM with Newey-West variance referenced to t(T-1)."""
```

**Edge cases.** If `d` is (numerically) constant zero (identical forecasts) => variance 0 =>
`DM = 0`, `pvalue = 1.0`, `better = None`. If `T <= horizon` => return `pvalue = NaN` and skip the pair
(record reason). Never raise on degenerate input; record and continue.

**Deliverables.** `src/ntl_etf/eval/stats.py` with `diebold_mariano`, `DMResult`, and the
pre-registration constant block (E5) imported here.

**Acceptance criteria (`tests/test_stats.py`).**
- **Identical forecasts** (`e_a == e_b`) => `pvalue == 1.0`, `stat == 0.0`, `better is None` (DM sanity).
- A clearly-superior model (A error ~N(0,0.1), B error ~N(0,1), fixed seed, T=120) yields `pvalue < 0.01`
  and `mean_diff < 0` (A better).
- For `h=1`, HAC variance equals `np.var(d, ddof=0)` (assert within 1e-12).
- HLN factor for `T=60,h=1` equals `sqrt((60+1-2+0)/60)=sqrt(59/60)`; assert numerically.
- Symmetry: `diebold_mariano(e_a,e_b)` stat == `-diebold_mariano(e_b,e_a)` stat; pvalues equal.

---

## E5 -- Pre-registration block: exact comparisons, losses, alpha, and corrections

**Objective.** Hard-code the *complete, frozen* list of pairwise comparisons and the correction policy so
the agent cannot data-snoop. This block is the single source of truth referenced by E7/E8/E10.

**Dependencies:** E4.

**File:** `src/ntl_etf/eval/prereg.py` (importable constants) + mirrored human-readable section in this
doc. Changing it after first results requires a logged config bump.

### Pre-registered families and tests

DL models: `dlinear`, `patchtst`, `itransformer`, `mamba` (and pretrained/foundation variants for H6).
Baselines: `momentum` (12-month TSMOM) and `dlinear` is treated as a **DL baseline** but for H1 the
"two baselines to beat" are **`momentum` and `dlinear`**.

**Loss:** squared-error (`loss="mse"`) for all return/IP DM tests (matches the headline MSE metric). A
secondary absolute-error DM family is computed and reported but NOT used for verdicts.

**Family A -- H1 signal existence (per ETF and POOLED, `leading`, horizon=1):**
each of `{patchtst, itransformer, mamba}` vs each of `{momentum, dlinear}`. (6 pairs x scope.)

**Family B -- H2/H3 architecture (per the named ETFs, `leading`, horizon=1):**
`itransformer` vs `patchtst` on every ETF; verdicts focus on H2 ETFs (multi-region, e.g. `XLI`) and H3
ETFs (single-dominant-region, e.g. `XLE`), as classified by E6.

**Family C -- H4 disruption (POOLED + per ETF, `leading`, horizon=1, stratum=`disruption`):**
`mamba` vs `patchtst`, `mamba` vs `itransformer`.

**Family D -- H6 transfer (paired model, `leading`, horizon=1):**
`patchtst_pretrained` vs `patchtst`; and each foundation model (`chronos`,`moirai`,`timesfm`, if run) vs
its from-scratch size-matched counterpart as declared in `manifest.json`.

**Family E -- H5 task contrast** is NOT a DM test (different targets/units); it compares `nowcast_r2`
vs `leading` pseudo-R^2 (E6 computes a leading R^2 for symmetry) — decision rule in E8.

### Multiple-comparison policy

- Report **raw DM p-values always.**
- Apply **Holm** (FWER) *within each family* as the primary correction, and **Benjamini-Hochberg
  `fdr_bh`** (FDR) as a secondary report, both via `statsmodels.stats.multitest.multipletests` at
  `alpha = 0.10`.
- A comparison is a **"win"** only if the Holm-adjusted p-value `< 0.10` **and** the effect direction
  matches the hypothesis. BH-adjusted p is reported alongside for transparency.
- Pooled-vs-per-ETF: H1 verdict uses POOLED first; per-ETF results are reported as a supporting heatmap,
  not as separate hypothesis tests (avoids inflating the family with 11x duplication).

```python
# src/ntl_etf/eval/prereg.py
ALPHA = 0.10
PRIMARY_LOSS = "mse"
DL_MODELS = ["dlinear","patchtst","itransformer","mamba"]
BASELINES_FOR_H1 = ["momentum","dlinear"]
CORRECTION_PRIMARY = "holm"
CORRECTION_SECONDARY = "fdr_bh"
# FAMILIES: list[dict(name, task, scope_set, pairs, stratum, horizon)] — frozen.
```

**Acceptance criteria.**
- `tests/test_prereg.py` asserts `ALPHA == 0.10`, the family list is non-empty, every pair references
  only known model names, and the H1 family contains exactly the 6 DL-vs-baseline pairs.
- `apply_correction(pvals, method)` wraps `multipletests` and returns `(reject, p_adj)`; tested that
  Holm on `[0.01,0.04,0.03]` matches statsmodels output.

---

## E6 -- `stratify.py`: hypothesis-aligned cuts + cross-fold aggregation

**Objective.** Build every stratum used by H1-H6 and aggregate metrics across walk-forward folds (and
seeds) with a mean and a dispersion/CI, both per-ETF and pooled.

**Dependencies:** E1, E3, F (vix flag), P (region counts).

**File:** `src/ntl_etf/eval/stratify.py`.

### Strata definitions

| Stratum | Rule | Hypothesis |
|---|---|---|
| `single_region` | ETF mapped to exactly 1 screened region in `regions.yaml` | H3 |
| `multi_region` | ETF mapped to >= 2 screened regions | H2 |
| `disruption` | target month has `vix_mean > 25` | H4 |
| `stable` | target month has `vix_mean <= 25` | H4 |
| `pretrained` | rows with `pretrained == True` | H6 |
| `from_scratch` | rows with `pretrained == False` | H6 |
| `all` | no filter | H1, H5 |

> **Region-count source (no look-ahead).** `single`/`multi` is structural metadata from `regions.yaml`
> (set during Phase P's pre-screen), not learned from outcomes. Assert the classification of the named
> ETFs: `XLE -> single_region` (Permian Basin), `XLI -> multi_region` (Pearl River + Yangtze River
> Delta). If `regions.yaml` disagrees, FAIL and report — the hypotheses are pinned to these.

### Aggregation

- For each `(model, task, scope, stratum, metric)`: pool the per-fold/per-seed *observations* and compute
  the metric on the pooled errors **as well as** the fold-wise metrics. Report:
  - `value` = metric on pooled observations (primary; avoids tiny-fold instability), and
  - `ci_low/ci_high` = mean +/- 1.96 * SE across folds (fold-level dispersion), or a **block bootstrap**
    CI (1000 resamples over folds, fixed seed) when `n_folds < 8`. Record which was used in a `ci_method`
    note column or sidecar.
- **Pooled scope** concatenates errors across all 11 ETFs (channel-independent global panel => this is the
  natural unit and the headline H1 number).

### Signatures
```python
# src/ntl_etf/eval/stratify.py
def add_strata(preds: pd.DataFrame, regions_yaml: str, vix: pd.DataFrame) -> pd.DataFrame:
    """Adds boolean/category columns: region_class, disruption, (pretrained already present).
    Asserts XLE single, XLI multi."""

def aggregate(preds: pd.DataFrame, metric_fns, strata: list[str],
              scopes=("POOLED","per_etf"), n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Returns the long results-store frame (E1 schema) with value + CI per group."""
```

**Pitfalls.**
- **Empty strata.** Some ETFs have zero disruption months in their (short) history; emit a row with
  `value=NaN, n_obs=0` rather than dropping — the heatmap must show the gap. H4 verdict ignores
  `n_obs < 12` cells.
- **Multiple comparisons via strata** is controlled in E5 (verdicts only use pre-registered families);
  stratified tables are descriptive.

**Acceptance criteria.**
- `tests/test_stratify.py`: synthetic preds with known VIX produce exactly the expected disruption/stable
  row counts; `add_strata` asserts XLE=single, XLI=multi (test fails if regions.yaml flips them).
- `aggregate` output validates against the E1 `RESULTS_SCHEMA`; CI columns finite where `n_obs >= 2`.
- Pooled `n_obs` equals the sum of per-ETF `n_obs` for the same `(model,task,stratum,metric)`.

---

## E7 -- Run all pre-registered DM tests + corrections; persist test store

**Objective.** Execute every comparison in the E5 pre-registration, apply Holm + BH within each family,
and write a tidy DM-results store.

**Dependencies:** E4, E5, E6.

**Actions.**
1. Add `run_dm_suite(preds, prereg, alpha) -> pd.DataFrame` to `src/ntl_etf/eval/stats.py`.
2. For each family/pair: align the two models' errors on the **intersection of `date`** within the
   relevant `(task, scope, stratum, horizon)`; require equal length and identical date index (assert).
   Compute `diebold_mariano`, collect raw p-values per family, apply Holm and BH, set `better`.
3. DM-store schema: `family, hypothesis, task, scope, stratum, horizon, model_a, model_b, loss, dm_stat,
   p_raw, p_holm, p_bh, mean_diff, n, win` (`win` ∈ {A,B,none}).
4. Write `experiments/dm_results.parquet` (+ `.csv`).

**Acceptance criteria.**
- `tests/test_dm_suite.py`: on a fixture where model A is engineered to dominate B in family A, the suite
  marks `win == "A"` with `p_holm < 0.10`; on identical forecasts, `win == "none"`, `p_holm == 1.0`.
- Every family's `p_holm` and `p_bh` are >= corresponding `p_raw` (monotonic sanity).
- Date-alignment assertion fires (test) when two models have mismatched dates.

---

## E8 -- Hypothesis decision rules + verdict summary

**Objective.** Encode the precise, pre-registered numeric criteria for H1-H6, evaluate them from the
results + DM stores, and emit a machine- and human-readable verdict (including honest H0 reporting).

**Dependencies:** E6 (metrics), E7 (DM), E5 (policy).

**File:** `src/ntl_etf/eval/verdict.py`.

### Decision rules (frozen)

| ID | Statement | Support criterion (ALL must hold) | Reject / H0 |
|---|---|---|---|
| **H1** | Signal exists | >=1 of `{patchtst,itransformer,mamba}` has POOLED `leading` MSE lower than **both** `momentum` and `dlinear`, **and** that model's `dir_acc > 0.50`, **and** its DM win vs **both** baselines is Holm-significant (`p_holm < 0.10`, `win` favors the DL model) | Else **H0 holds**: state that no DL model significantly beats momentum; report the best raw MSE delta and its (insignificant) p_holm |
| **H2** | iTransformer > PatchTST on multi-region ETFs | On `multi_region` H2 ETFs (e.g. XLI), `itransformer` MSE < `patchtst` MSE **and** DM `win == itransformer` Holm-significant (Family B) | Reject if not significant or direction reversed |
| **H3** | PatchTST > iTransformer on single-region ETFs | On `single_region` H3 ETFs (e.g. XLE), `patchtst` MSE < `itransformer` MSE **and** DM `win == patchtst` Holm-significant | Reject otherwise |
| **H4** | Mamba >= both Transformers in disruption | In `stratum=disruption`, `mamba` MSE <= both `patchtst` and `itransformer` MSE **and** (DM not significantly worse: `not (win in {patchtst,itransformer})`) vs each; "strong support" if Mamba DM-*beats* at least one | Reject if Mamba is DM-significantly worse than either |
| **H5** | Nowcast R^2 >> leading R^2 | POOLED `nowcast_r2` exceeds POOLED `leading` pseudo-R^2 by `>= 0.10` absolute **and** `nowcast_r2 > 0` while leading pseudo-R^2 ~ 0 | Reject if gap < 0.10 |
| **H6** | Pretrained/foundation-init > from-scratch | Paired comparison: `pretrained` variant POOLED `leading` MSE < from-scratch counterpart **and** DM `win` favors pretrained Holm-significant (Family D) | Reject otherwise; if foundation models not run on this hardware, mark H6 `deferred` (not reject) and say so |

Notes:
- **"Leading pseudo-R^2"** for H5 uses the same OOS-R^2 formula on the `leading` task vs the OOS mean
  return (expected near 0). Document that returns and IP are different units; H5 is about *relative*
  predictability, reported with both R^2 values side by side and the caveat in the paper.
- **Effect sizes accompany every verdict** (MSE deltas, R^2 values, Sharpe), not just p-values.

### Signature + outputs
```python
# src/ntl_etf/eval/verdict.py
def decide_hypotheses(results_store, dm_results, prereg) -> dict:
    """Returns {'H1':{'verdict':'support'|'reject'|'deferred','evidence':{...}}, ...,
                'H0_note': str}."""
```
- Write `experiments/hypotheses_verdict.json` and a Markdown `paper/tables/hypotheses_verdict.md`.

**Acceptance criteria.**
- `tests/test_verdict.py`: a fabricated results+DM store where a DL model dominates yields
  `H1.verdict == "support"`; a store where nothing beats momentum yields `H1.verdict == "reject"` and a
  non-empty `H0_note`.
- H6 with no foundation/pretrained rows yields `verdict == "deferred"` (never silently "reject").
- Verdict JSON validates against a small schema (all H1-H6 keys present, verdict in allowed set).

---

## E9 -- `plots.py`: publication figures from the results store

**Objective.** Render every paper figure deterministically from the results/DM/predictions stores into
`paper/figures/`. matplotlib/seaborn; fixed style; no interactive state.

**Dependencies:** E1, E3, E6, E7.

**File:** `src/ntl_etf/eval/plots.py`. Each function takes a store path (or frame) + `out_dir` and
returns the written path; sets `np.random`/seed where any jitter is used.

### Required figures

| Function | Figure | Source |
|---|---|---|
| `plot_pred_vs_actual(...)` | predicted vs actual scatter + 45deg line, per task; faceted by a few key ETFs | predictions |
| `plot_metric_bars(...)` | grouped bars: MSE / dir_acc / Sharpe per model (POOLED), error bars = fold CI | results_store |
| `plot_stratified_heatmap(...)` | heatmap model x ETF of MSE (and of dir_acc), one per task | results_store (per_etf) |
| `plot_disruption_vs_stable(...)` | side-by-side bars of MSE per model in disruption vs stable (H4) | results_store strata |
| `plot_equity_curves(...)` | cumulative long/short equity per model, **gross and net** overlaid; chained across folds via `prev_pos` | predictions -> metrics.strategy_returns |
| `plot_dm_significance(...)` | matrix/heatmap of Holm-adjusted DM p-values for Family A (stars for wins) | dm_results |

Conventions: 300 DPI PNG **and** vector PDF for each; consistent model color map in a module constant;
titles include task + stratum; figures never read raw data folders (only stores), so they are
reproducible.

**Acceptance criteria.**
- `tests/test_plots.py` (smoke): each function writes a non-empty `.png` and `.pdf` to a tmp dir from a
  tiny fixture store without error; returned paths exist and size > 1 KB.
- Equity-curve net line is <= gross line at the final point (costs reduce cumulative return); asserted on
  a fixture with >0 turnover.

---

## E10 -- Publication tables (Markdown + LaTeX) from the results store

**Objective.** Generate the paper's tables programmatically so numbers in the paper always match the
store.

**Dependencies:** E6, E7, E8.

**File:** add `render_tables(...)` to `src/ntl_etf/eval/results.py` (or `eval/tables.py`).

**Tables to emit (to `paper/tables/`, each as `.md` and `.tex`):**
1. `main_results` -- POOLED MSE, MAE, dir_acc, Sharpe(gross/net) per model x task, with fold CIs.
2. `per_etf_mse` -- model x ETF MSE (leading), best-per-ETF bolded.
3. `dm_family_a` -- DM stats + raw/Holm/BH p-values for H1 family, wins starred.
4. `nowcast_vs_leading` -- R^2 and Pearson, nowcast vs leading, per model (H5).
5. `hypotheses_verdict` -- H1-H6 verdict + key effect size + corrected p (mirrors E8 JSON).

LaTeX via `pandas.DataFrame.to_latex(..., float_format="%.4f", escape=True)`; Markdown via
`DataFrame.to_markdown`. Round consistently (4 dp for losses/R^2, 2 dp for Sharpe, 1 dp for dir_acc %).

**Acceptance criteria.**
- `tests/test_tables.py`: rendered LaTeX compiles-clean enough to pass a regex check (no stray `nan` that
  isn't intended; numeric cells match the store values for a fixture).
- Bolding logic test: in `per_etf_mse`, exactly one model is bolded per ETF column (the min).

---

## E11 -- Driver `scripts/analyze_results.py` (single eval entry point)

**Objective.** One command consumes `experiments/<run_id>/predictions/*` and emits the full results
store, DM store, all tables, all figures, and the hypotheses verdict.

**Dependencies:** E1-E10.

**Actions / pipeline order:** load+validate (E1) -> audit alignment (E2; abort on fail) -> add strata
(E6) -> aggregate metrics (E3/E6) -> write results store (E1) -> run DM suite + corrections (E7) ->
decide hypotheses (E8) -> render tables (E10) -> render figures (E9) -> print a console summary of
verdicts.

**CLI.**
```powershell
# PowerShell
& C:/Users/Admin/AppData/Local/Programs/Python/Python311/python.exe `
  scripts/analyze_results.py --experiments-dir experiments --out-dir experiments `
  --figures-dir paper/figures --tables-dir paper/tables --alpha 0.10 --seed 0
```
```bash
# Git Bash
python scripts/analyze_results.py --experiments-dir experiments --out-dir experiments \
  --figures-dir paper/figures --tables-dir paper/tables --alpha 0.10 --seed 0
```
Flags: `--run-ids` (subset), `--task {leading,nowcast,both}` (default both), `--loss {mse,mae}` (default
mse), `--skip-figures` (CI speed). Logs via `src/ntl_etf/utils/logging.py`; reads alpha/loss defaults from
`configs/experiment.yaml` and records the resolved config + git SHA in `experiments/analyze_manifest.json`.

**Acceptance criteria.**
- End-to-end on a small synthetic `experiments/` fixture (committed under `tests/fixtures/mini_experiments/`)
  produces: `results_store.parquet`, `dm_results.parquet`, `hypotheses_verdict.json`, all 6 figure PNGs,
  all 5 tables, exit code 0.
- Re-running with the same seed produces byte-identical `results_store.parquet` and identical verdict JSON
  (determinism check in `tests/test_analyze_e2e.py`).
- If E2 alignment audit fails on the fixture (a deliberately leaked variant), the driver exits nonzero and
  writes nothing downstream.

---

## E12 -- Test suite + pre-registration freeze record

**Objective.** Consolidate the phase's tests and freeze the pre-registration before any real results are
generated.

**Dependencies:** E3-E11.

**Actions.**
1. Ensure `tests/` contains: `test_results.py`, `test_audit.py`, `test_metrics.py`, `test_stats.py`,
   `test_prereg.py`, `test_stratify.py`, `test_dm_suite.py`, `test_verdict.py`, `test_plots.py`,
   `test_tables.py`, `test_analyze_e2e.py`.
2. Write `experiments/PREREG.md` (timestamped, git-SHA-stamped snapshot of E5 + E8 rules) **before** the
   first real `analyze_results.py` run; the paper cites this as evidence the tests were pre-registered.
3. Add a `tests/fixtures/` package with the mini experiments + hand-computed metric fixtures.

**Acceptance criteria.**
- `pytest tests/ -q` passes on CPU with no GPU and no network (all fixtures local).
- `experiments/PREREG.md` exists, contains `ALPHA=0.10`, the full family list, and the H1-H6 rules, and is
  committed in a commit *prior* to any commit that adds real `results_store.parquet`.
- A coverage check: every metric in `metrics.py` and `diebold_mariano` has at least one hand-computed
  assertion (not just smoke).

---

## Cross-references and honesty checklist (for the agent)

- Predictions come from **M** (`experiments/<run_id>/predictions`), folds from **P** (`folds.parquet`),
  VIX flag and IP from **F**, region classification from **P/`regions.yaml`**, utils from **S**.
- The paper (**W**) pulls every number from `paper/tables/*` and every figure from `paper/figures/*`; W
  must not restate numbers by hand.
- **Claim wins only when DM-significant after Holm.** Report raw and corrected p-values. If **H0 holds**
  (nothing beats momentum), the verdict JSON, the `hypotheses_verdict` table, and the paper's results
  section must say so plainly, with effect sizes, rather than emphasizing insignificant point-estimate
  differences.
- **Deferred != rejected.** If Mamba (Windows/CUDA, see Phase M fallback) or foundation models were not
  run on this hardware, H4/H6 are reported `deferred` with the reason, and the corresponding rows are
  absent from the DM families (not counted as losses).

> **TO-VERIFY at runtime (cheap asserts, no external calls):** (1) `nowcast_target_form` agreement
> between F and this phase; (2) `XLE`/`XLI` region classification in `regions.yaml`; (3) presence of
> `scaler_fit_on=="train"` in every run manifest; (4) statsmodels `multipletests` is importable
> (`from statsmodels.stats.multitest import multipletests`) and `scipy.stats.t` / `pearsonr` available.
