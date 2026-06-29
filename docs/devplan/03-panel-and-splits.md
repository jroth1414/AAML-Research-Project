# Phase B — Panel Assembly, Windowing, Walk-Forward Splits, and Leakage Audit

`docs/devplan/03-panel-and-splits.md` · Task-ID prefix **P**

This phase turns the per-source tables produced in Phase N (NTL) and Phase F (finance/macro) into a single **global, channel-independent panel** of windowed training samples, plus a **walk-forward split generator** and an **executable leakage audit**. It is the seam between data and models: Phase M codes its `Dataset`/`DataLoader` consumption against the **DATA CONTRACT** defined here (task P9), so every column name, dtype, index, and tensor shape in this file is normative.

> **Design north star.** The single most important overfitting mitigation in this project is *global pooling*: one shared-weight model trained over thousands of windows drawn from all 11 sectors × all kept regions, rather than 11 tiny per-sector models. Every abstraction below is built to produce that pool while staying strictly leakage-safe.

---

## 0. Conventions, units, and assumed inputs

**Time index.** All series are monthly, indexed by a `pandas.PeriodIndex(freq="M")` internally and serialized as a first-of-month `Timestamp` (`date`, dtype `datetime64[ns]`, normalized to day 1). The canonical study range is **2013-01 .. 2024-12** (≈144 months). NTL begins effectively 2012-04 (VNP46A3 availability), but the labeled study window starts 2013-01 to give lookback room.

**Symbols.** `L` = lookback length (months), `H` = forecast horizon (months), `V` = number of region variates in a multivariate (per-sector) group, `S` = number of sectors (11), `N` = number of kept region series.

**Tasks (target modes).**
| Mode | Target | Alignment rule |
|---|---|---|
| `leading` | forward ETF log return | NTL month `t` predicts return of month `t+1+LAG_RET` (release-lag enforced, P5) |
| `nowcast` | contemporaneous sector industrial production (IP), as YoY or MoM log-change | NTL month `t` aligns to IP of the SAME month `t`, but the IP value used must itself respect IP's publication lag (P5) |

**Upstream deliverables consumed (cross-phase dependencies).**

| Artifact | Produced by | Path |
|---|---|---|
| NTL monthly region-feature table | Phase N (≈N6) | `data/processed/ntl_features.parquet` |
| ETF monthly log-return table | Phase F (≈F3) | `data/processed/etf_returns.parquet` |
| Sector IP / activity table | Phase F (≈F5) | `data/processed/macro_ip.parquet` |
| VIX monthly table (mean + max) | Phase F (≈F6) | `data/processed/vix_monthly.parquet` |
| Region definitions | Phase N (≈N1) / configs | `configs/regions.yaml` |
| Seed/util helpers | Phase S | `src/ntl_etf/utils/{seed,io,config,logging}.py` |

If a consumed artifact's exact column names differ from the contract in P9, **adapt in a thin loader (P2) — do not silently rename downstream**; raise a clear error listing missing columns.

---

## P1 — Author `regions.yaml` and `sector_fred_map.yaml` (config schema)

**Objective.** Define the canonical region→sector candidate mapping and the sector→FRED-IP linkage as version-controlled YAML so the pipeline is config-driven.

**Dependencies.** Phase S setup (repo + `configs/`); Phase N region list; Phase F FRED series selection.

**Actions.**
1. Create/confirm `configs/regions.yaml`. Each region entry carries an id, a human name, a bounding geometry reference (resolved in Phase N), and the **candidate** sectors it may proxy. Hypotheses H2/H3 require these specific multi-vs-single-region cases, so encode them explicitly.
2. Create `configs/sector_fred_map.yaml` mapping each of the 11 SPDR tickers to (a) its FRED IP/activity series id(s) for the nowcast target and (b) a fallback flag for sectors with no clean industrial-production analog (XLF, XLK, XLC, XLRE, XLP partial). Mark those `ip_analog: weak` and supply the chosen proxy series + a `TO-VERIFY` note.

**Files.** `configs/regions.yaml`, `configs/sector_fred_map.yaml`.

```yaml
# configs/regions.yaml  (excerpt — Phase N owns geometry resolution)
schema_version: 1
regions:
  - id: pearl_river_delta
    name: Pearl River Delta
    bbox: [112.5, 21.5, 114.7, 23.6]      # lon_min, lat_min, lon_max, lat_max  (TO-VERIFY in N)
    candidate_sectors: [XLI, XLK, XLY]
  - id: yangtze_river_delta
    name: Yangtze River Delta
    bbox: [118.5, 29.8, 122.2, 32.5]
    candidate_sectors: [XLI, XLK, XLY]
  - id: permian_basin
    name: Permian Basin
    bbox: [-104.5, 29.5, -100.5, 33.5]
    candidate_sectors: [XLE]              # single-dominant-region case for H3
  - id: ruhr_valley
    name: Ruhr Valley
    bbox: [6.6, 51.2, 7.7, 51.7]
    candidate_sectors: [XLI, XLB]
  # ... remaining regions
features:                                  # NTL feature columns produced in Phase N, per region
  - sum_radiance
  - mean_radiance
  - lit_area_frac        # fraction of pixels above a fixed radiance threshold
hypothesis_pairs:                          # used by E-phase to slice results
  H2_multiregion:  { sector: XLI, regions: [pearl_river_delta, yangtze_river_delta] }
  H3_singleregion: { sector: XLE, regions: [permian_basin] }
```

```yaml
# configs/sector_fred_map.yaml  (excerpt — FRED ids are TO-VERIFY, see P1 acceptance)
schema_version: 1
nowcast_transform: yoy_log        # one of {yoy_log, mom_log}; fit nothing, pure transform
sectors:
  XLI: { fred_ip: [IPMAN],         ip_analog: strong }   # Industrial Production: Manufacturing
  XLB: { fred_ip: [IPMINE, IPG211S], ip_analog: strong } # mining/materials components
  XLE: { fred_ip: [IPG211111CN, IPMINE], ip_analog: strong }
  XLU: { fred_ip: [IPUTIL],        ip_analog: strong }   # Industrial Production: Utilities
  XLP: { fred_ip: [IPNCONGD],      ip_analog: medium }   # nondurable consumer goods
  XLY: { fred_ip: [IPDCONGD],      ip_analog: medium }   # durable consumer goods
  XLV: { fred_ip: [IPG3254S],      ip_analog: weak }     # pharma prep mfg (partial proxy)
  XLK: { fred_ip: [IPG334S],       ip_analog: weak }     # computer/electronic product mfg
  XLF: { fred_ip: [],              ip_analog: none, proxy: [INDPRO] }   # no IP analog -> total IP fallback
  XLC: { fred_ip: [],              ip_analog: none, proxy: [INDPRO] }
  XLRE:{ fred_ip: [],              ip_analog: none, proxy: [INDPRO] }
```

**Deliverables.** Both YAML files committed, loadable by `utils/config.py`.

**Acceptance criteria.**
- `python -c "from ntl_etf.utils.config import load_yaml; load_yaml('configs/regions.yaml'); load_yaml('configs/sector_fred_map.yaml')"` runs without error.
- A test asserts every ticker in the 11-SPDR set appears exactly once in `sector_fred_map.yaml`, and every `fred_ip` / `proxy` id used is non-empty for sectors not flagged `ip_analog: none`.
- **TO-VERIFY check (the agent must run this, do not trust the placeholders):** call `fredapi.Fred.get_series_info(<id>)` for each id; assert HTTP 200 and `frequency_short == "M"`. Replace any id that 404s and log the substitution to `data/interim/fred_id_resolution.json`. Known-stable anchors: `INDPRO` (total IP), `IPMAN` (manufacturing), `IPUTIL` (utilities), `IPMINE` (mining) — use these as fallbacks.

---

## P2 — Thin source loaders + alignment to the canonical monthly index

**Objective.** Provide one function per upstream artifact that loads it, validates columns against the DATA CONTRACT (P9), reindexes to the canonical monthly `PeriodIndex`, and returns tidy frames.

**Dependencies.** P1; Phase N/F deliverables.

**Actions.**
1. In `src/ntl_etf/data/panel.py`, implement loaders that read the parquet artifacts, coerce `date` to month-start `datetime64[ns]`, set/validate index, and assert no duplicate `(date, region|sector)` keys.
2. Reindex each series to the full canonical month range; leave genuine gaps as `NaN` (do **not** forward-fill across pre-inception gaps — that is handled by valid-date masks in P4).
3. Emit a per-series **valid-date mask** table from observed (non-NaN) spans of each ETF and each region feature.

**Files.** `src/ntl_etf/data/panel.py` (loaders), writes `data/interim/valid_masks.parquet`.

```python
# src/ntl_etf/data/panel.py  (signatures)
def load_ntl_features(path="data/processed/ntl_features.parquet") -> pd.DataFrame: ...
    # returns long: [date, region_id, feature, value]  -> validated & reindexed
def load_etf_returns(path="data/processed/etf_returns.parquet") -> pd.DataFrame: ...
    # returns wide: index=date, columns=11 tickers, values=monthly log return
def load_macro_ip(path="data/processed/macro_ip.parquet", cfg=...) -> pd.DataFrame: ...
    # returns wide: index=date, columns=11 tickers, values=transformed sector IP (nowcast target)
def load_vix(path="data/processed/vix_monthly.parquet") -> pd.DataFrame: ...
    # returns wide: index=date, columns=[vix_mean, vix_max]
def build_valid_masks(...) -> pd.DataFrame: ...
    # returns: [series_id, kind{etf|region_feat}, first_valid, last_valid]
```

**Deliverables.** Loader functions; `data/interim/valid_masks.parquet`.

**Acceptance criteria.**
- Loading any artifact whose columns deviate from the contract raises `SchemaError` naming the missing/extra columns.
- A test asserts `etf_returns` has exactly 11 columns and that `XLRE` is all-`NaN` before `2015-10` and `XLC` is all-`NaN` before `2018-06` (ragged-start guard — confirmed inception dates: XLRE 2015-10-07, XLC 2018-06-18; the other 9 trade since 1998 so are fully populated across the study range).
- `valid_masks.parquet` has one row per series with `first_valid <= last_valid` and dates within the canonical range.

---

## P3 — Region→sector PRE-SCREEN correlation gate (no look-ahead)

**Objective.** For every candidate `(region feature, sector)` pair from `regions.yaml`, compute a correlation screen on a **fixed warmup window only**, keep pairs passing a documented threshold/sign rule, and record kept/dropped pairs to a manifest.

**Dependencies.** P1, P2; **leakage rule** shared with P5/P8.

**Why this matters / pitfall.** Selecting pairs by their correlation with the target is itself a form of training. If the screen sees any test data, the *selection* leaks future information even if model training does not. We therefore screen **only on a pre-study warmup window** that ends strictly before the first fold's test period.

**Actions.**
1. Define `SCREEN_WARMUP = ("2013-01", "2017-12")` (60 months) — this is *also* guaranteed to be inside the first walk-forward training window, so the screen never touches any test month. Make it a config value `configs/panel.yaml: screen_warmup`.
2. For each candidate pair, build the **release-lag-aligned** leading series (NTL month `t` vs ETF return `t+1+LAG_RET`) and the contemporaneous nowcast series, restricted to `SCREEN_WARMUP` and to months where both sides are valid (mask-aware).
3. Compute Spearman and Pearson correlation. **Keep rule:** retain the pair if `|spearman| >= SCREEN_RHO_MIN` (default `0.15`) on EITHER task, OR if the pair is an explicit `hypothesis_pairs` member (H2/H3 pairs are *forced-keep* so the hypotheses remain testable regardless of screen noise — record them with `forced_keep: true`). Optional sign rule: drop pairs whose sign flips between Spearman and Pearson AND whose `|rho| < 0.10` (treat as noise).
4. Adjust for multiple comparisons in the *reporting* (not the keep gate): store BH-FDR-adjusted p-values in the manifest so Phase E can cite screen selectivity honestly.
5. Write the manifest and emit the kept-pair set used by P4.

**Files.** `src/ntl_etf/data/panel.py` (`screen_pairs`), `configs/panel.yaml`, writes `data/interim/pair_screen_manifest.csv` and `configs/pairs_kept.yaml`.

```python
def screen_pairs(ntl, etf, ip, masks, cfg) -> pd.DataFrame:
    """Returns one row per (region_id, feature, sector) with:
       n_obs_screen, pearson, spearman, p_value, p_value_bh,
       kept(bool), forced_keep(bool), drop_reason(str|''), task_screened.
       Uses ONLY cfg.screen_warmup months. Asserts max(screen month) < first test month."""
```

**Deliverables.** `pair_screen_manifest.csv`; `pairs_kept.yaml` (the working pair set for the panel).

**Acceptance criteria.**
- Manifest contains every candidate pair (kept and dropped) with a non-empty `drop_reason` for every dropped pair.
- A test asserts `screen_pairs` raises if any screen observation has a `date` >= the first fold's test start (computed via P6) — i.e., the screen is provably look-ahead-free.
- `pairs_kept.yaml` is non-empty and includes the H2 (XLI: PRD+YRD) and H3 (XLE: Permian) pairs with `forced_keep: true`.
- Re-running with a fixed seed reproduces the manifest byte-for-byte (deterministic).

---

## P4 — Build the global panel + the dual dataset abstraction (CI and variate views)

**Objective.** Assemble all kept series into one global structure, and define the dataset abstraction that yields BOTH a **channel-independent (CI) per-series view** (for PatchTST, Mamba, DLinear) and a **per-sector multivariate (variate) view** (for iTransformer), from the same underlying tensors.

**Dependencies.** P2, P3.

**Conceptual model (state this precisely for Phase M).**
- A **series** is one `(region_id, feature, sector)` triple's NTL input channel. Multiple series can map to the same sector (multi-region sectors).
- **CI view** (PatchTST/Mamba/DLinear): every series is an independent univariate training sequence; the model has **shared weights across all series and all sectors** (the global pool). A sample is `X:(L,1)`, `y:(H,)`.
- **Variate view** (iTransformer): for a given sector at a given time origin, gather the set of `V` region series mapped to that sector into one multivariate sample `X:(L,V)`; the model embeds **each variate (region) as a token** and attends across regions (confirmed iTransformer formulation: time points of each series → one variate token; attention is cross-variate; FFN is per-variate-temporal). Target `y:(H,)` is that sector's ETF return or IP.
- **Static covariates** attached to every sample for conditioning: `sector_id` (0..10), `region_id` (categorical), `feature_id` (categorical). These let one global model distinguish series without breaking channel-independence.

**Actions.**
1. Build a tidy long **panel frame** keyed `(date, sector, region_id, feature)` with columns for the NTL input value, the leading target (lag-aligned), and the nowcast target. Persist it.
2. Build a `SeriesRegistry`: for each series an integer `series_idx`, its `(sector, region_id, feature)`, valid span (from P4 masks), and `sector_group_id` (which multivariate sector group it belongs to).
3. Implement `PanelDataset(torch.utils.data.Dataset)` parameterized by `(L, H, task, view)`; precompute the list of admissible `(series_or_group, origin_t)` sample anchors once, filtered by validity (P5/P7).
4. Implement `make_dataloader(...)` building train/val/test loaders given a fold's index sets (P6) and per-fold normalization stats (P6).

**Files.** `src/ntl_etf/data/panel.py`, writes `data/processed/panel_long.parquet` and `data/processed/series_registry.parquet`.

```python
@dataclass
class WindowSpec: lookback:int; horizon:int; task:str   # 'leading'|'nowcast'
                  ; view:str                              # 'ci'|'variate'

class PanelDataset(Dataset):
    def __init__(self, panel, registry, spec, fold, norm_stats): ...
    def __len__(self) -> int: ...
    def __getitem__(self, i) -> dict:
        # CI view returns:
        #   {'x':(L,1) f32, 'y':(H,) f32, 'sector_id':(), 'region_id':(),
        #    'feature_id':(), 'series_idx':(), 'origin_date':()}
        # variate view returns:
        #   {'x':(L,V) f32, 'y':(H,) f32, 'sector_id':(), 'region_ids':(V,),
        #    'var_mask':(V,) bool, 'group_idx':(), 'origin_date':()}
        ...
```

**Pitfalls handled.** Ragged region availability inside a sector group → `var_mask` flags variates that are invalid at this origin (iTransformer must ignore masked tokens in attention; document this requirement to Phase M). Variate ordering within a group is **fixed and recorded** in the registry so token positions are stable across folds.

**Deliverables.** `panel_long.parquet`; `series_registry.parquet`; `PanelDataset`; `make_dataloader`.

**Acceptance criteria.**
- `len(PanelDataset(..., view='ci'))` ≫ `len(... view='variate')` (CI explodes per-series → the intended large pool; assert CI sample count > 1000 for `L=12,H=1,leading` on the full study range).
- For one fixed anchor, the CI sample's `x` equals the corresponding column of the variate sample's `x` (consistency test).
- Every emitted `x` has shape `(L,1)` (CI) or `(L,V)` (variate); every `y` has shape `(H,)`; no `NaN` in any emitted tensor.

---

## P5 — Release-lag & publication-lag alignment (the central leakage guard)

**Objective.** Implement the feature/target alignment that enforces VNP46A3's release lag for the leading task and IP's publication lag for the nowcast task, so a NTL month can never be used to predict a same-or-earlier outcome it could not have informed in real time.

**Dependencies.** P2, P4.

**Facts (verified / TO-VERIFY).**
- VNP46A3 (monthly composite) is released **~30–45 days after month end**. Concretely: NTL for month `t` is generally not available until late in month `t+1`. Therefore for the **leading** task, NTL month `t` may only predict ETF returns of month `t+1` onward → `LAG_RET = 0` extra months beyond the natural `+1` (i.e., target month = `t+1`). Set `RELEASE_LAG_MONTHS = 1` and align target month = `t + RELEASE_LAG_MONTHS` for `H=1` (and `t+1 .. t+H` for multi-horizon). Make it a config constant `configs/panel.yaml: release_lag_months: 1` with a comment citing the 30–45 day lag. **TO-VERIFY:** the agent should confirm in Phase N the actual file-availability dates and bump to `2` if any month's composite lands after the start of `t+1`; record the decision in `data/interim/lag_decision.md`.
- FRED IP publication: a month's IP is published mid-following-month and **revised**. For the **nowcast** task the target is contemporaneous IP of month `t`, but the *value used* must be the vintage available given NTL's own availability. Since both NTL(`t`) and a usable IP(`t`) estimate are available by ~`t+1`, the nowcast pairing (NTL `t` ↔ IP `t`) is admissible **only when the sample's prediction is timestamped at `t+1`**. Record `NOWCAST_AS_OF = t + 1`. Do not attempt true real-time IP vintages unless ALFRED vintage data is wired in Phase F; otherwise add a limitation note (P9, W-phase).

**Actions.**
1. Centralize alignment in one function so the rule exists in exactly one place (auditable).
2. For leading: `target[t] = etf_return[t + release_lag_months ... t + release_lag_months + H - 1]`.
3. For nowcast: `target[t] = ip[t ... t + H - 1]`, with the sample's effective `as_of = t + 1` stored for the audit.
4. Drop any anchor whose required target months exceed the series' valid span.

**Files.** `src/ntl_etf/data/panel.py` (`align_targets`), `configs/panel.yaml`.

```python
def align_targets(ntl_month_t, returns, ip, spec, cfg) -> tuple[np.ndarray, dict]:
    """Returns (y(H,), meta) where meta carries target_dates and as_of_date.
       leading:  target_dates = [t + release_lag + k for k in range(H)]
       nowcast:  target_dates = [t + k for k in range(H)]; as_of = t+1
       Asserts min(target_date) > t for leading (strict forward)."""
```

**Deliverables.** `align_targets`; `lag_decision.md`.

**Acceptance criteria.**
- Unit test (fixtures): for `leading, H=1`, the target date attached to NTL month `2018-03` is `2018-04` (strictly later), never `2018-03` or earlier.
- Test asserts `release_lag_months >= 1` is read from config and that setting it to `0` makes the leakage audit (P8) **fail loudly**.
- For nowcast, every sample's stored `as_of_date == origin_date + 1 month`.

---

## P6 — Walk-forward split generator with per-fold, per-series normalization

**Objective.** Implement rolling-origin walk-forward CV: minimum 60-month training window, 1-month step, a validation tail carved from train for HP selection, and a next out-of-sample test block — yielding train/val/test index sets per fold, plus normalization stats fit on **train only**, per series.

**Dependencies.** P4, P5.

**Actions.**
1. Implement `walk_forward_splits(dates, cfg)` producing an ordered list of folds. Defaults (config `configs/panel.yaml`): `min_train_months: 60`, `val_months: 12`, `test_months: 12`, `step_months: 12` (use `step_months: 1` only for the final dense evaluation; HP search uses the coarse step to control cost). Expanding-window by default (`expanding: true`); a `rolling` option keeps the train window fixed at `min_train_months`.
2. Each fold defines disjoint month sets: `train` (first `min_train_months+ k*step .. ` minus `val` tail), `val` (the last `val_months` of the pre-test region), `test` (the next `test_months`). Enforce `max(train) < min(val) < min(test)`.
3. Compute normalization stats **per series, on train months only**: mean/std of the NTL input per `series_idx`; targets standardized per `(sector, task)` on train. Apply to val/test. Store stats in the fold object; never refit.
4. Provide a deterministic fold manifest written to disk for the audit and for Phase E reproducibility.

**Files.** `src/ntl_etf/data/splits.py`, writes `data/interim/folds_manifest.json`.

```python
@dataclass(frozen=True)
class Fold:
    fold_id:int; train_dates:list; val_dates:list; test_dates:list
    norm: dict   # series_idx -> (mu, sigma) for X ; (sector,task) -> (mu,sigma) for y

def walk_forward_splits(dates: pd.PeriodIndex, cfg) -> list[Fold]: ...
def fit_norm_stats(panel, train_dates, registry) -> dict: ...   # TRAIN ONLY
def apply_norm(x, y, fold, keys) -> tuple[np.ndarray, np.ndarray]: ...
```

**Pitfalls handled.** Small samples → with ≈144 months, a 60-month train + 12 val + 12 test leaves few folds; document the resulting fold count (~6 at step 12, up to ~60 at step 1) and warn Phase M/E that per-fold test blocks are small (motivates pooling across folds for DM tests). Series whose valid span starts after a fold's train window (XLC, XLRE) get **per-fold availability**: they enter only in folds where they have ≥ `min_series_history` (config, default 24) train months; recorded in the manifest.

**Deliverables.** `walk_forward_splits`; `folds_manifest.json`.

**Acceptance criteria.**
- Test asserts for every fold: `len(train_dates) >= 60`, `max(train) < min(val) < min(test)`, and the three sets are pairwise disjoint.
- Test asserts every per-series norm stat's source months are a subset of that fold's `train_dates` (cross-checked against `fit_norm_stats` provenance log).
- `folds_manifest.json` round-trips: reloading reproduces identical fold boundaries.
- Foundation-model note: when `cfg.preprocess == "foundation_zeroshot"`, normalization is **skipped/per-model** (Chronos/Moirai/TimesFM scale internally); `apply_norm` returns inputs unchanged and the manifest records `norm: "model_internal"`. Document this so Phase M does not double-normalize.

---

## P7 — Windowing engine (anchors, validity, ragged-gap exclusion)

**Objective.** Generate the admissible sample anchors `(series_or_group, origin_t)` for given `(L, H, task, view, fold)`, guaranteeing no window crosses a pre-inception gap or an invalid region span.

**Dependencies.** P4, P5, P6.

**Actions.**
1. For each candidate origin `t`, require the full lookback `[t-L+1 .. t]` of the input AND the full target span (from P5) to be valid (non-NaN, within the series' mask). For the variate view, require each *unmasked* variate to be valid over the lookback; variates valid for only part are excluded from the group at that origin (reflected in `var_mask`); a group with `< 2` valid variates falls back to being skipped for iTransformer (it degenerates to CI).
2. Partition anchors by fold/split using the **origin's prediction timestamp** (leading: `t+release_lag`; nowcast: `as_of=t+1`) — an anchor belongs to whichever split contains its *outcome/decision* month, never its input month. This prevents a window whose input is in train but whose label is in test from leaking.
3. Cache anchor lists per `(spec, fold)` to disk for speed and auditability.

**Files.** `src/ntl_etf/data/panel.py` (`build_anchors`), cache under `data/interim/anchors/`.

```python
def build_anchors(registry, masks, spec, fold) -> list[Anchor]:
    """Anchor = (series_idx|group_idx, origin_t, split). Excludes any anchor whose
       lookback or target span hits an invalid/NaN month. Split assignment uses the
       OUTCOME month, not the input month."""
```

**Deliverables.** `build_anchors`; cached anchor lists.

**Acceptance criteria.**
- Test: no emitted window for XLRE has any input or target month before 2015-10; none for XLC before 2018-06.
- Test: an anchor whose outcome month is in `test` never appears in the train loader, even if its lookback months are entirely in train.
- Test: total anchor counts per fold are logged; assert train anchors > 0 and test anchors > 0 for every fold.

---

## P8 — Executable leakage audit (CI-gated)

**Objective.** A concrete checklist plus a `pytest` module that mechanically proves the five leakage invariants on fixtures and on a small real-data slice, run in CI.

**Dependencies.** P3, P5, P6, P7.

**The five invariants (each is one or more asserts).**

| # | Invariant | Assert |
|---|---|---|
| L1 | No normalization stat from val/test | every `Fold.norm` stat's provenance months ⊆ `train_dates` |
| L2 | Release-lag present (leading) | for every leading anchor, `min(target_date) > origin_date` (strictly forward by ≥ `release_lag_months`) |
| L3 | No window spans a pre-inception gap | every emitted `x`/`y` window lies fully inside the series' `[first_valid,last_valid]`; no `NaN` |
| L4 | Correlation pre-screen used no test data | `max(screen month) < min(first-fold test month)`; manifest `screen_warmup` end < earliest test start |
| L5 | Temporal ordering per fold | `max(train) < min(val) < min(test)`; anchor split assignment uses outcome month |

**Actions.**
1. Build tiny synthetic fixtures in `tests/fixtures/` (3 sectors, 4 regions, ~80 months, one ragged series) with known answers.
2. Implement `tests/test_leakage.py` covering L1–L5; add a `tests/test_panel_contract.py` validating P9 schemas.
3. Add a programmatic checklist function `audit_panel(...)` that runs all invariants on the real folds and writes `experiments/manifests/leakage_audit.json` (pass/fail per invariant). Wire it into `run_all.ps1` as a gate (abort the pipeline on any failure).
4. Add to CI (Phase S workflow): the test job must run `pytest tests/test_leakage.py tests/test_panel_contract.py`.

**Files.** `tests/test_leakage.py`, `tests/test_panel_contract.py`, `tests/fixtures/`, `src/ntl_etf/data/splits.py` (`audit_panel`), `experiments/manifests/leakage_audit.json`.

```bash
# bash
pytest -q tests/test_leakage.py tests/test_panel_contract.py
```
```powershell
# PowerShell
& $env:PY -m pytest -q tests/test_leakage.py tests/test_panel_contract.py
```

**Deliverables.** Passing audit suite; `leakage_audit.json` with all five invariants `pass`.

**Acceptance criteria.**
- All five invariants pass on fixtures AND on the real panel.
- **Negative controls:** the suite includes deliberately-broken variants (set `release_lag_months=0`; fit norm on full range; screen on full range) and asserts each makes the corresponding invariant **fail** — proving the asserts have teeth.
- `audit_panel` returns nonzero exit / raises when any invariant fails, halting `run_all`.

---

## P9 — DATA CONTRACT (normative schemas for Phase C)

**Objective.** Freeze the exact schema of every processed table and every tensor handed to models, so Phase M codes against a stable interface.

**Dependencies.** P2, P4, P5, P6, P7.

**Processed table schemas.**

`data/processed/panel_long.parquet`
| column | dtype | notes |
|---|---|---|
| `date` | `datetime64[ns]` | month-start; canonical index |
| `sector` | `category` | one of 11 SPDR tickers |
| `region_id` | `category` | from `regions.yaml` |
| `feature` | `category` | NTL feature name |
| `series_idx` | `int32` | stable id, matches registry |
| `ntl_value` | `float32` | raw (un-normalized) NTL input for month `date` |
| `target_leading` | `float32` | lag-aligned forward ETF log return (NaN if unavailable) |
| `target_nowcast` | `float32` | transformed sector IP (NaN if unavailable) |
| `as_of_leading` | `datetime64[ns]` | decision month = `date + release_lag` |
| `as_of_nowcast` | `datetime64[ns]` | `date + 1 month` |
| `valid` | `bool` | row is inside series mask |

`data/processed/series_registry.parquet`: `series_idx:int32, sector:category, region_id:category, feature:category, sector_group_id:int16, variate_pos:int16, first_valid:datetime64, last_valid:datetime64, forced_keep:bool`.

`data/processed/etf_returns.parquet`: index `date:datetime64`, 11 `float32` columns (tickers), monthly log returns.
`data/processed/macro_ip.parquet`: index `date:datetime64`, 11 `float32` columns, transformed IP per `nowcast_transform`.
`data/processed/vix_monthly.parquet`: index `date:datetime64`, columns `vix_mean:float32, vix_max:float32` (drives H4 disruption stratification in Phase E).

**Tensor contract (output of `DataLoader`, batch size `B`).**

| Field | CI view shape/dtype | Variate view shape/dtype |
|---|---|---|
| `x` | `(B, L, 1)` f32, normalized | `(B, L, V)` f32, normalized |
| `y` | `(B, H)` f32, normalized | `(B, H)` f32, normalized |
| `sector_id` | `(B,)` int64 | `(B,)` int64 |
| `region_id` | `(B,)` int64 | — |
| `region_ids` | — | `(B, V)` int64 |
| `var_mask` | — | `(B, V)` bool (True = valid variate) |
| `feature_id` | `(B,)` int64 | — |
| `series_idx` / `group_idx` | `(B,)` int64 | `(B,)` int64 |
| `origin_date` | `(B,)` int64 (epoch-month) | `(B,)` int64 |
| `y_norm` | `(mu,sigma)` per item for inverse-transform at eval | same |

**Inverse-transform contract.** Phase E metrics are computed in **original return/IP units**; `make_dataloader` attaches per-sample `(mu, sigma)` for `y` so predictions can be de-standardized. Document the exact call `eval.metrics.denorm(pred, mu, sigma)`.

**Deliverables.** This contract section; a `tests/test_panel_contract.py` enforcing it.

**Acceptance criteria.**
- `test_panel_contract.py` loads each processed table and asserts column names + dtypes exactly match the tables above (fails on any drift).
- A loader smoke test pulls one batch in each view and asserts every field's shape/dtype matches the tensor contract; asserts no `NaN`; asserts `var_mask.sum(dim=1) >= 2` for every variate-view item.

---

## P10 — Panel build CLI and pipeline wiring

**Objective.** A single reproducible entry point that runs P2→P9 end to end, seeded and logged, producing all processed artifacts and the leakage audit.

**Dependencies.** P1–P9; Phase S `seed`/`io`/`logging`.

**Actions.**
1. Implement `scripts/build_panel.py` orchestrating: load sources → screen pairs (P3) → build panel + registry (P4) → align targets (P5) → generate folds (P6) → build anchors (P7) → run leakage audit (P8) → write contract-conformant outputs (P9).
2. Set seeds via `utils/seed.set_all_seeds(cfg.seed)`; write a run manifest (config hash, git SHA, package versions, row counts) to `experiments/manifests/panel_build_<timestamp>.json`.
3. Make it config-driven (`--config configs/panel.yaml`) with overridable `L`, `H`, `task`, `view`.

**Files.** `scripts/build_panel.py`, `configs/panel.yaml`; appends a step to `scripts/run_all.ps1`.

```powershell
# PowerShell
& $env:PY scripts/build_panel.py --config configs/panel.yaml
```
```bash
# bash
python scripts/build_panel.py --config configs/panel.yaml
```

```yaml
# configs/panel.yaml (excerpt)
seed: 1414
study_start: 2013-01
study_end:   2024-12
screen_warmup: [2013-01, 2017-12]
screen_rho_min: 0.15
release_lag_months: 1
nowcast_transform: yoy_log
lookbacks: [6, 12, 24]
horizons:  [1, 3]
walk_forward: { min_train_months: 60, val_months: 12, test_months: 12,
                step_months: 12, expanding: true, dense_step_months: 1 }
min_series_history: 24
```

**Deliverables.** `build_panel.py`; all `data/processed/*` artifacts; `panel_build_*.json` manifest; leakage audit passing.

**Acceptance criteria.**
- `python scripts/build_panel.py --config configs/panel.yaml` exits 0, writes all four processed parquet files + registry + folds manifest + leakage audit, and the audit reports all five invariants `pass`.
- Run manifest records git SHA, seed, package versions, and per-table row counts; two runs with the same config + seed produce identical row counts and an identical `pair_screen_manifest.csv`.
- Total CI-view anchor count for the default `(L=12,H=1,leading)` config is logged and > 1000 (confirms the global pool is large enough to justify shared-weight training).

---

## Cross-phase interface summary

| This phase provides | Consumed by |
|---|---|
| `PanelDataset` (CI + variate views), `make_dataloader` | M (all model training: PatchTST/Mamba/DLinear use CI; iTransformer uses variate) |
| `walk_forward_splits` / `Fold` objects + per-fold norm | M (training loop), E (per-fold OOS predictions) |
| Tensor + table DATA CONTRACT (P9) | M (model I/O), E (denorm + metrics) |
| `vix_monthly` + `as_of` columns | E (H4 disruption stratification, H5 nowcast-vs-leading) |
| `pair_screen_manifest.csv`, `folds_manifest.json`, `leakage_audit.json` | W (paper: methods, reproducibility, limitations) |

**Open TO-VERIFY items the agent must close before declaring Phase B done.**
1. Confirm actual VNP46A3 file-availability dates in Phase N; bump `release_lag_months` to `2` if any composite arrives after the start of month `t+1` (record in `lag_decision.md`).
2. Resolve every FRED id in `sector_fred_map.yaml` against the live FRED API; substitute and log any 404s.
3. Decide whether ALFRED real-time IP vintages are feasible for the nowcast `as_of`; if not, record the simplification as a limitation for the W-phase paper.
