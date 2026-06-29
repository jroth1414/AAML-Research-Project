# DEVPLAN — Nighttime Light Emissions as an Economic Proxy for Sector ETF Forecasting: PatchTST vs iTransformer vs Mamba

**Master development plan (index + normative contracts) for an autonomous coding agent.** This project tests whether monthly NASA VIIRS Black Marble nighttime-light (NTL) intensity over economically-linked world regions predicts (a) **forward 1–3 month log returns** of the 11 SPDR sector ETFs (the *leading* task) and (b) **contemporaneous sector industrial production** (the *coincident nowcast* task). It compares three deep architectures — **PatchTST** (channel-independent patch attention), **iTransformer** (cross-region/variate attention), and **Mamba** (selective state-space S6) — against a 12-month time-series-momentum baseline and DLinear, adds masked self-supervised pretraining and time-series foundation-model initialization (Chronos/Moirai/TimesFM), and evaluates everything under leakage-safe walk-forward cross-validation with pre-registered Diebold–Mariano significance testing. This document is the **single entry point**: it states how the agent works the plan, the guiding principles, the canonical repo structure, the phase/milestone overview, the dependency graph, a consolidated risk register, the hypothesis traceability map, the rubric mapping, and a week-by-week schedule with go/no-go gates. **It also defines the normative cross-phase Data & Artifact Contracts (Appendix A) that supersede any per-phase wording that conflicts with it.** The detailed step-by-step tasks live in the six per-phase files under [`docs/devplan/`](docs/devplan/).

> **Authoritative-conflict rule.** Where a per-phase file disagrees with this DEVPLAN (a column name, a date convention, a file path, who owns a config), **DEVPLAN.md wins**. The reviewer-driven reconciliations in Appendix A and the Risk Register are binding. If the agent finds a contradiction not resolved here, it must (1) follow Appendix A if covered, else (2) prefer the owning phase named in the Ownership table, and (3) record the resolution in the run manifest.

---

## 1. How an agent should use this plan

**Task-ID prefix scheme.** Every task in every phase has a stable ID = a one-letter phase prefix + a number. Cite dependencies by ID across phases.

| Prefix | Phase | Doc |
|---|---|---|
| **S** | Phase 0 — overview, environment, reproducibility, tooling, git | [`docs/devplan/00-overview-and-setup.md`](docs/devplan/00-overview-and-setup.md) |
| **N** | Phase A.1 — NTL acquisition + raster feature extraction | [`docs/devplan/01-data-ntl.md`](docs/devplan/01-data-ntl.md) |
| **F** | Phase A.2 — ETF returns, FRED IP, VIX | [`docs/devplan/02-data-financial-macro.md`](docs/devplan/02-data-financial-macro.md) |
| **P** | Phase B — panel, pairing, windowing, walk-forward splits, leakage audit | [`docs/devplan/03-panel-and-splits.md`](docs/devplan/03-panel-and-splits.md) |
| **M** | Phase C — baselines + 3 deep models + pretraining + foundation init + trainer + HP search | [`docs/devplan/04-models-and-training.md`](docs/devplan/04-models-and-training.md) |
| **E** | Phase D — metrics, Diebold–Mariano, stratified analyses, hypothesis decisions, plots | [`docs/devplan/05-evaluation-and-stats.md`](docs/devplan/05-evaluation-and-stats.md) |
| **W** | Phase E — paper, rubric mapping, reproducibility, deliverables | [`docs/devplan/06-paper-and-deliverables.md`](docs/devplan/06-paper-and-deliverables.md) |

**Phase dependency order.** `S` → (`N` ∥ `F`) → `P` → `M` → `E` → `W`. Phases N and F run in parallel after S. Do not start a downstream phase until its upstream **Definition of Done** gate (Section 5) passes.

**Work task-by-task, gated by acceptance criteria.** Execute tasks in ID order within a phase (respecting cross-phase dependencies). A task is **done only when its Acceptance Criteria are proven** — by the specific test, assert, shape/column check, or numeric sanity check stated in the task. Do not advance to a dependent task until the current one passes. Commit one focused commit per completed task, prefixed with the task ID (e.g. `P6: walk-forward split generator`). Every run records its git SHA, seed, capability profile, and config hash in a manifest (Appendix A.5).

**When uncertain about an external fact**, run the exact `TO-VERIFY` check named in the task and adopt what it returns — never fabricate API names, FRED series IDs, package versions, or citations.

---

## 2. Guiding principles

### 2.1 Reproducibility & seeding
- One config-driven pipeline; one editable package (`ntl_etf`); one results store; one manifest schema (Appendix A.5).
- **Project default seed = `1414`.** The multi-seed averaging set is **`[1414, 1415, 1416, 1417, 1418]`**, defined once in `configs/experiment.yaml` and referenced everywhere. The evaluation bootstrap uses a *distinct* purpose-seed `0` (labeled as such, not the project default). All four "default seed" values that drifted across draft phases (1337, 42, 1414, 0) are reconciled to this rule — see Risk R17.
- `set_seed` sets Python/NumPy/PyTorch seeds + deterministic flags (`CUBLAS_WORKSPACE_CONFIG=:4096:8`, `use_deterministic_algorithms(True, warn_only=True)`, `num_workers=0` on Windows). Bit-exact determinism is guaranteed only within identical hardware + library versions; where stochasticity remains, report means over the seed set and record seed + versions + git SHA in the manifest.

### 2.2 Leakage safety (the project's defining constraint)
- **Release-lag rule (single canonical statement; supersedes all per-phase wording).** The NTL feature timestamp is the **end of month M**. The VNP46A3 composite for month M is released ~30–45 days after month-end, i.e. during month M+1, **before the end of M+1**. Therefore:
  - **Leading task:** NTL(M) may predict the ETF log return realized at **end of month M+1** (`H=1`), and `M+2…M+H` for longer horizons. The target month index for `H=1` is **M+1** — *not* M, and *not* M+2. There is exactly one forward month, equal to the release lag; there is no additional double-lag. `RELEASE_LAG_MONTHS = 1`. (TO-VERIFY in N: if any composite lands after the day needed to act on the M+1 return, bump to 2 and record in `data/interim/lag_decision.md`.)
  - **Nowcast task:** NTL(M) aligns to IP(M) (contemporaneous), but the sample's decision timestamp is `as_of = M+1` because both NTL(M) and a usable IP(M) estimate are only available by ~M+1. This is an *as-if-released-simultaneously* coincident estimate, **not** a true real-time nowcast — flagged as a limitation (FRED revisions/vintages; Risk R12).
  - **Invariant (audited in P8/L2 and re-checked in E2):** for every leading sample, `min(target_date) >= origin_date + 1 month` (strictly forward).
- **Train-only normalization.** All scalers/standardization stats are fit on the **train split of each fold only** and applied to val/test; models never fit scalers. The fold object carries per-series `(mu, sigma)`; predictions are de-standardized for metrics. Every run manifest records `scaler_fit_on: "train"`; E2 fails any run missing it (Risk R16).
- **No-look-ahead pair screen.** The region→sector correlation pre-screen runs only on a fixed warmup window (2013-01..2017-12) that ends strictly before the first fold's test period; hypothesis pairs (H2/H3) are force-kept regardless of screen noise.
- **No `*_sa` columns reach models.** STL seasonal adjustment is non-causal; `value_sa` is plotting-only. P and M assert no column ending `_sa` enters the model feature matrix (Risk R13). The causal `value_dlog` is the model-ready stationary IP form.
- **Pretraining leakage stance (decided, consistent across P/M/E/W).** Masked self-supervised pretraining uses the **full unlabeled NTL history** (no labels, standard-in-the-literature relaxation). This is an explicit, documented limitation: the shared encoder sees test-period NTL *distributions* (never targets). Fine-tuning and evaluation still obey per-fold train-only fitting and the release-lag alignment. Fold-bounded pretraining is noted as a cleaner-but-deferred alternative. The W limitations section states this plainly (Risk R8).

### 2.3 Config-driven design
- Behavior comes from YAML under `configs/`, not hard-coded constants. **Single-owner rule for shared config files** (Risk R2; Ownership table below): `configs/regions.yaml` (owner N), `configs/sector_fred_map.yaml` (owner F), `configs/hypotheses.yaml` (owner E), `configs/data.yaml`/`configs/panel.yaml`/`configs/experiment.yaml` per their phase. Non-owners **load and validate**, never re-author.

### 2.4 CPU-first / GPU-deferred execution
- The whole pipeline runs **CPU-only by default**; Mamba (official `mamba-ssm`) and foundation-model fine-tuning are gated behind a capability flag and deferred to a GPU profile, **without ever hard-crashing** the CPU run (skip-and-log → `status="skipped"` row).
- **mamba-ssm caveat (verified).** `mamba-ssm` and its deps `causal-conv1d`/`triton` require CUDA + Linux build toolchains and **do not install on native Windows/CPU**. They live only in `requirements-extras.txt`, installed only in the GPU profile. A pure-PyTorch S6 fallback (M7) runs Mamba on CPU but is **not** the fused kernel; H4 verdicts from the fallback are tagged `support (fallback impl)` or `deferred`, never presented as official Mamba (Risk R6/R14).
- **NTL/geo import fragility.** `blackmarblepy`/`earthaccess` and the heavy geo stack are **not** in the import-critical core smoke test; their imports are lazy (only inside `download_ntl` paths) so a missing Windows wheel never fails the whole environment gate (Risk R5).
- Three execution profiles (Windows/CPU default; WSL2+CUDA or cloud GPU; Colab) with a result-merge path back into the canonical store.

---

## 3. Canonical repo structure

Every phase uses these exact paths.

```
AAML-Research-Project/
  RESEARCH.MD                      # PRESERVE — rubric + checklist; never edited or staged
  DEVPLAN.md                       # this master index (Appendix A holds the binding contracts)
  README.md
  requirements.txt  requirements-extras.txt  requirements.lock.txt
  pyproject.toml  .gitignore  .env.example
  docs/devplan/                    # per-phase plan files (00..06)
  docs/LICENSING_AND_ETHICS.md  docs/RUBRIC_MAPPING.md  docs/DELIVERABLES_CHECKLIST.md
  configs/                         # regions.yaml(N) sector_fred_map.yaml(F) hypotheses.yaml(E)
                                   # data.yaml panel.yaml experiment.yaml + model YAMLs
  data/  raw/ interim/ processed/ external/   # all gitignored (.gitkeep only)
  src/ntl_etf/
    data/   ntl.py finance.py macro.py panel.py splits.py transform_registry.py
    models/ base.py patchtst.py itransformer.py mamba.py dlinear.py momentum.py
            pretrain.py foundation.py _vendor/
    train/  trainer.py hpsearch.py
    eval/   metrics.py stats.py stratify.py plots.py results.py prereg.py verdict.py
    utils/  seed.py io.py logging.py config.py capabilities.py manifest.py results.py
  scripts/   download_ntl.py download_finance.py download_macro.py build_panel.py
             run_experiment.py run_experiments.ps1 analyze_results.py
             collect_paper_assets.py check_env.py merge_results.py check_deliverables.py
             run_all.ps1 run_all.sh        # OWNER = W11 (full pipeline); others call into it
  tests/   notebooks/   experiments/ (run outputs; gitignored except manifests + results store)
  paper/
```

---

## 4. Phase / milestone overview

| Phase | Doc | Task IDs | Key deliverables | Definition of Done gate |
|---|---|---|---|---|
| **0 — Setup** | [`docs/devplan/00-overview-and-setup.md`](docs/devplan/00-overview-and-setup.md) | S1–S12 | repo tree (RESEARCH.MD preserved), Python 3.11 venv, core/extras requirements + lockfile, `capabilities.detect()`, `.env.example` + config loader, `.gitignore`, manifest + results-store helpers, seeding/logging, CI, run-all stubs, git workflow | `pytest`/`ruff`/`black` green CPU-only no-network; `check_env.py` prints `profile=windows_cpu` without raising; core import smoke prints `core OK` (no geo/NTL/mamba in it); RESEARCH.MD byte-identical; no AI author anywhere |
| **A.1 — NTL** | [`docs/devplan/01-data-ntl.md`](docs/devplan/01-data-ntl.md) | N1–N10 | Earthdata auth, `regions.yaml` (owner), VNP46A3 download adapter + quality masking, per-region monthly features, **release-lag stamp**, `data/processed/ntl_features.parquet` + manifest | offline feature-math tests pass with zero network; credential-gated smoke test downloads 1 ROI×month with positive `ntl_mean`; release-lag invariant holds on every row; `regions.yaml` validates against the unified schema (A.6) |
| **A.2 — Finance/Macro** | [`docs/devplan/02-data-financial-macro.md`](docs/devplan/02-data-financial-macro.md) | F1–F12 | `data.yaml`, **`sector_fred_map.yaml` (owner)**, ETF month-end log returns + momentum (ragged-history safe), FRED IP (eligibility-tiered) + transforms, VIX monthly (`vix_mean`,`vix_max`,`disruption_flag`), alignment/transform registry, manifest | offline tests pass no-key/no-network; live smoke (key-gated) fetches INDPRO; **month-end tz-naive** index on every table; XLC first valid ≥ 2018-07, XLRE ≥ 2015-11; 8 nowcast-eligible sectors + INDPRO control |
| **B — Panel & Splits** | [`docs/devplan/03-panel-and-splits.md`](docs/devplan/03-panel-and-splits.md) | P1–P10 | consume regions/FRED configs, source loaders (month-end), pair pre-screen, global panel (CI + variate views), release-lag alignment, walk-forward folds + per-fold train-only norm, windowing, **executable leakage audit**, **DATA CONTRACT** | `build_panel.py` exits 0; leakage audit all five invariants `pass` with negative controls that *fail* on tampering; panel/tensor schemas match Appendix A; CI-view anchor count logged (soft floors, not a hard >1000 crash — Risk R18) |
| **C — Models & Training** | [`docs/devplan/04-models-and-training.md`](docs/devplan/04-models-and-training.md) | M1–M18 | `BaseForecaster`, momentum, DLinear, vendored PatchTST + iTransformer, Mamba (gated + CPU S6 fallback), masked pretraining + scratch/pretrained variants, foundation wrappers (zero-shot CPU / finetune GPU), Trainer, HP search, single-run + matrix runners, **`predictions.parquet` per Appendix A.3** | shape/overfit/determinism tests pass CPU; momentum+DLinear+PatchTST+iTransformer produce contract-valid predictions; Mamba/foundation skip-and-log cleanly on CPU; every manifest records `scaler_fit_on`, `seed`, `mamba_impl`, `variant`, `git_sha` |
| **D — Eval & Stats** | [`docs/devplan/05-evaluation-and-stats.md`](docs/devplan/05-evaluation-and-stats.md) | E1–E12 | results store (Appendix A.4), defensive alignment audit, metrics, DM test (data-driven HAC + HLN + date-clustered pooling), **`hypotheses.yaml` (owner)** + prereg + verdict, stratified analyses, figures, tables, `analyze_results.py`, PREREG freeze | end-to-end on fixture `experiments/`; DM records realized `T` per test and flags `T<30` underpowered; binomial test on directional accuracy; verdicts honest about H0/deferred; reproducible byte-identical store |
| **E — Paper & Deliverables** | [`docs/devplan/06-paper-and-deliverables.md`](docs/devplan/06-paper-and-deliverables.md) | W1–W15 | paper skeleton + sections mapped to rubric, block diagram, results binding (`results.lock.json`), README, licensing/ethics note, **single-owner `run_all`**, env freeze + manifest tests, rubric mapping, deliverables checklist, milestone schedule | `main.pdf` builds with all sections; `check_deliverables.py` exits 0; clean-checkout `run_all -SkipGpu` reproduces figures/tables; all H1–H6 verdicts + explicit H0 decision present; no secrets, RESEARCH.MD unchanged |

---

## 5. Dependency graph / critical path

```
S (setup)
├─→ N (NTL data)  ─┐
└─→ F (fin/macro) ─┴─→ P (panel + splits + leakage audit)
                          └─→ M (models + training)
                                   └─→ E (eval + stats)
                                            └─→ W (paper + deliverables)
```

**Critical path:** `S → (N,F) → P → M → E → W`. The longest chain runs through the deep-model training and evaluation; everything else feeds it.

**Parallelism the agent should exploit.** After S: N and F are independent (different sources, different modules) — run them concurrently. Within W, scaffolding tasks (W1 skeleton, W9 README stub, W10 licensing, W12 manifest helpers) can be created early (Week 1) so later phases write into the right places; prose sections (W2–W8) and results binding (W7) come after E.

**Gating discipline.** P must not start until **both** N's and F's DoD gates pass (P joins their outputs). M must not start until P's leakage audit is green. E consumes only M's `predictions.parquet` + P's folds. W binds numbers only from E's stores (never hand-typed).

**Front-loaded CPU value.** Tiers (M18): Tier 0 momentum+DLinear (minutes) → Tier 1 PatchTST+iTransformer scratch + foundation zero-shot (hours, CPU) → Tier 2 pretraining/pretrained variants → Tier 3 GPU-only Mamba official + foundation finetune. H1/H2/H3/H5 become testable after Tiers 0–1 on CPU; H4/H6 may be `deferred` if no GPU.

---

## 6. Consolidated risk register

Folds in the reviewer's cross-cutting risks and every critical/high finding. Each row names the owning task(s) that must implement the mitigation. (Detailed normative contracts for the consistency risks are in Appendix A.)

| ID | Risk | Sev | Impact | Mitigation | Owner task(s) |
|---|---|---|---|---|---|
| **R1** | **Date-index convention conflict** (month-start in P9 vs month-end in F/M/E) | Critical | left-join of month-end tables onto a month-start panel → all-NaN joins; pipeline silently empty | **Canonical = month-END, tz-naive (Appendix A.1).** P9 + all of P updated to month-end; build_panel asserts every joined table's index equals the master month-end grid | P1,P2,P9; F2 (already month-end); M,E read A.1 |
| **R2** | **Shared-config ownership conflicts** (`regions.yaml` N vs P; `sector_fred_map.yaml` F vs P; `hypotheses.yaml` cited by W, created by none) | Critical | two contradictory specs/bboxes/series for one file; agent cannot resolve | **Single owner per file (Section 2.3 + Appendix A.6):** N owns regions.yaml (unified schema), F owns sector_fred_map.yaml, **E creates `configs/hypotheses.yaml`** and prereg.py loads it. Non-owners load+validate only | N4 (regions); F6 (fred map); E5 (hypotheses.yaml); P1→consume |
| **R3** | **Prediction-artifact contract mismatch** (M `predictions.parquet` w/ `series_id` vs E `predictions/*.parquet` dir w/ `etf`,`pretrained`,`seed`) | Critical | E1 strict schema assert fails on every real M output | **One contract (Appendix A.3):** single file `experiments/<run_id>/predictions.parquet`; columns = `model,variant,task,etf,target_kind,horizon,fold,split,date,y_true,y_pred,pretrained,seed`; drop `series_id`/`_IP` suffix in favor of `etf`+`target_kind` | M0/M9 write; E1 read |
| **R4** | **Results-store schema/name collision** (`results.parquet` S8 vs `results_store.parquet` E1) | Critical | two incompatible canonical metrics tables; merge dedup keys missing | **E's `experiments/results_store.parquet` is authoritative (Appendix A.4).** Extend its schema with `status,profile,git_sha,config_hash` so the GPU-merge dedup works; .gitignore tracks `results_store.parquet`; `merge_results.py` keys on existing columns | S8→repurpose; E1 schema; S4 merge |
| **R5** | **Windows/CPU install fragility** (blackmarblepy/earthaccess + geo stack in import-critical core smoke) | High | one missing cp311 wheel breaks the whole env gate, not just NTL | Lazy-import `blackmarble`/`earthaccess` inside download paths; **remove them from S3 core import smoke**; add `earthaccess` to requirements; verify real PyPI version floor (don't assert `2024.1`) | S3; N2 |
| **R6** | **Mamba/H4 provenance** (CPU S6 fallback ≠ official fused kernel) | High/Med | H4 verdict from fallback misrepresents "official Mamba" | Tag `mamba_impl` in manifest; E reports H4 from fallback as `support (fallback impl)` or `deferred`; cap/​warn on fallback pool size to avoid an overnight CPU run that never finishes | M7; E8 |
| **R7** | **iTransformer variate view may collapse to <2 variates** after screen + ragged histories → H2/H3 untestable | High | H2 reduces to n=1 sector (XLI); padding scheme conflict (M6 global-superset vs P4 per-group gather) | **One batching scheme = P4 per-sector-group gather + `var_mask`** (drop M6 global padding). Add ≥2–3 multi-region sectors to regions.yaml so H2 isn't n=1; E reports the stratum n per region-class | P4,P7; M6; N4; E6/E8 |
| **R8** | **Pretraining leakage stance** (full-history self-supervision sees test-period NTL distribution) | High | unstated leak undermines H6 fairness | **Decided (Section 2.2):** accept full-history *unlabeled* pretraining as a documented limitation; fine-tune/eval stay fold-bounded; W states it. Consistent across P/M/E/W | M8; W8/W10 |
| **R9** | **Statistical power / DM independence** (~6 folds at step 12; dense step-1 overlapping, serially+cross-sectionally correlated test points; HAC lag h−1=0 at H=1 assumes no autocorrelation) | High | low DM power; nominal α ≠ true α | (a) data-driven Newey-West lag even at H=1: `floor(4*(T/100)^(2/9))`; (b) record realized `T` per DM test, flag `T<30` underpowered; (c) when pooling across the 11 ETFs at a date, **cluster/block DM variance by date**, not iid (etf,date); (d) explicit power/limitation statement | E4,E7,E8; W8 |
| **R10** | **Ragged-ETF screen window empty** (XLC starts 2018-06 → zero observations in 2013-2017 warmup) | High | XLC pairs screened on empty window, silently dropped | Force-keep ragged ETFs' hypothesis pairs; flag screen `n_obs`; never silently drop | P3 |
| **R11** | **Headline Sharpe portfolio aggregation unspecified** (11 per-ETF long/short strategies → one gross/net Sharpe?) | Med (missing topic) | net-Sharpe sensitivity to 10bps cost under monthly sign-flipping not analyzed | Define equal-weight portfolio of the 11 single-ETF strategies for the headline Sharpe; report per-ETF too; chain positions for the equity curve; analyze net vs gross under cost | E3,E6,E9 |
| **R12** | **FRED revisions/vintages** (nowcast uses final revised IP = look-ahead vs true real-time) | Med (missing) | H5 validity caveat | Document as limitation; ALFRED real-time vintages optional/deferred; if not wired, record simplification | F (ALFRED note); P5; W10 |
| **R13** | **STL `value_sa` could leak into models** (non-causal) | Med | hidden look-ahead via seasonal adjustment | Keep `value_sa` plotting-only; model uses causal `value_dlog`; **assert no `*_sa` column enters `PanelDataset.x`** in P leakage audit | F7; P8 |
| **R14** | **CPU S6 fallback runtime** (sequential scan over thousands of global windows) | Med | overnight CPU run silently never finishes | Cap fallback pool size or warn on projected runtime; H4 from fallback tagged accordingly | M7 |
| **R15** | **`regions.yaml` schema/bbox conflict** (N single-`sector` vs P `candidate_sectors` list; differing bboxes) | Med | two schemas + two coordinate sets for one file | **Unified schema (Appendix A.6):** N owns geometry; adopt `candidate_sectors` list + `anchor`/`rationale`/`hypothesis`; one bbox per region; `features:`/`hypothesis_pairs:` blocks where E/P expect them | N4; P1→consume |
| **R16** | **`scaler_fit_on` missing from manifest** → E2 audit fails every run | Low | defensive audit blocks all runs | Add `scaler_fit_on:"train"` to manifest schema + Trainer write_manifest | M11; A.5 |
| **R17** | **Seed inconsistency** (1337/42/1414/0 across phases) | Low | reviewer confusion; ambiguous multi-seed set | One default `1414` + set `[1414..1418]` in configs; bootstrap seed `0` labeled distinct | S5,S9; all phases via A |
| **R18** | **Hard `>1000` anchor assert** could crash build_panel under aggressive screen | Low | pipeline halts on a soft condition | Replace with logged count + soft floor (warn `<500`, fail only `<100`); show kept_series × usable_origins arithmetic | P4 |
| **R19** | **VIX column contract mismatch** (P expects `vix_max`; F emits `date,vix_mean,disruption_flag`) | Med | P/E read a column F never produced | **Reconcile to F's output + add `vix_max` (Appendix A.2):** F9 emits `date,vix_mean,vix_max,disruption_flag`; P2/P9/E6 read these | F9; P2,P9; E6 |
| **R20** | **Orchestration collision** (`run_all.ps1` defined 3× in S11/M14/W11) | Med | three files at one path with different flags | **W11 owns `run_all.ps1`** (six stages + paper). S11 makes a stub W supersedes; M14's experiment-matrix logic moves to `scripts/run_experiments.ps1` | W11; S11 stub; M14→rename |
| **R21** | **Manifest schema divergence** (S7 `git_sha` vs W12 `git_commit`, three field sets) | Low | reproducibility test validates the wrong shape | **One superset manifest in S (Appendix A.5);** M and W *extend* it; standardize on `git_sha` | S7; M11; W12 |
| **R22** | **H5 metric harshness** (OOS R² on differenced returns ~0/negative; absolute gap not scale-comparable) | Med | H5 unreachable for reasons unrelated to coincidence | Report nowcast R² on YoY-log (smoother) **and** MoM; compare each model's nowcast vs leading R² **paired**; add contemporaneous-vs-lagged correlation robustness check; document negative OOS R² is expected for returns | E8; W8 |
| **R23** | **H6 conflates two transfer tests** (NTL-masked-pretrain uses NTL; foundation models never see NTL) | Low | paper could wrongly claim foundation benefits from NTL pretraining | Split: **H6a** = NTL-masked-pretrained vs from-scratch (core, fair); **H6b** = foundation zero-shot/finetune on target history (reference). E Family D distinguishes them | E5,E8; M10; W8 |
| **R24** | **No directional-accuracy significance test** | Med (missing) | a 0.55 dir-acc could be non-significant | Add a binomial/sign test that dir_acc > 0.50; H1 requires it significant, not just >0.50 point estimate | E3,E8 |
| **R25** | **NTL feature multicollinearity** (mean/median/sum/p90/lit_count/lit_frac highly collinear) | Med (missing) | redundant CI series inflate the pool + the comparison family | Choose a primary feature set for the model-facing channels; document which features feed models vs which are QA-only; do not pool every redundant feature as a separate series | P1/P4; W5 |
| **R26** | **No compute-time budget cap** (large HP grid × folds × tasks → hundreds of CPU-hours) | Med (missing) | autonomous run intractable | Coarse step (step_months=12) for HP search; modest grids; total-runtime estimate + hard early-abort/time cap in the matrix runner | M12; M18; W11 |
| **R27** | **NTL coverage gaps** (high-latitude winter, persistent cloud → high `frac_masked`) | Med (missing) | unusable region-months silently averaged | Define min coverage per region-month (e.g. drop if `frac_masked > 0.5`); invalidate a region series if too many months drop; record in panel build | N6/N8; P2 |
| **R28** | **Common-market-factor confound** (sector returns ≈ market beta) | Med (missing) | "signal" may be market beta, not sector-specific NTL info | Add a robustness target: sector-minus-SPY (market-excess) returns; report whether signal survives factor-neutralization | F (excess-return option); E8; W8 |

---

## 7. Task → hypothesis traceability

Maps which tasks/experiments produce the evidence for each hypothesis and the null. Decision rules and thresholds are the single source of truth in **`configs/hypotheses.yaml`** (owner E5); `prereg.py` loads them; the paper text is asserted byte-identical to the YAML (W12).

| Hypothesis | Statement (decision rule lives in `configs/hypotheses.yaml`) | Producing tasks | Deciding tasks |
|---|---|---|---|
| **H0** (null) | No DL model significantly beats the 12-month momentum baseline | M2 (momentum), M3/M5/M6/M7 (DL) | E7 (DM), E8 (verdict — reported honestly if it holds) |
| **H1** | ≥1 DL model beats **both** baselines on 1-month return MSE (DM-significant, Holm) **and** dir_acc significantly > 0.50 (binomial) | M2,M3,M5,M6,M7; P (leading panel) | E3 (metrics+binomial, R24), E4/E7 (DM Family A), E8 |
| **H2** | iTransformer > PatchTST on **multi-region** ETFs (e.g. XLI: Pearl River + Yangtze Delta) | M5,M6; P4 variate view (R7); N4 ≥2 multi-region sectors | E6 (region-class strata + n), E7 (Family B), E8 |
| **H3** | PatchTST > iTransformer on **single-dominant-region** ETFs (e.g. XLE: Permian) | M5,M6; P3 force-keep | E6, E7 (Family B), E8 |
| **H4** | Mamba ≥ both Transformers during disruption (months with mean VIX > 25) | M7 (gated; `mamba_impl` tagged R6/R14); F9 (vix_max/flag, A.2) | E6 (disruption stratum), E7 (Family C), E8 (fallback→`support (fallback impl)`/`deferred`) |
| **H5** | Nowcast R² >> leading R² (NTL primarily coincident); reported on YoY-log + MoM, **paired** per model, + corr robustness (R22) | M (nowcast runs); F6 eligible sectors; P5 nowcast alignment | E3/E6 (R²), E8 (gap rule + caveats) |
| **H6a** | NTL-masked-pretrained > identically-sized from-scratch (same NTL inputs — core, fair) | M8 (pretrain + scratch/pretrained variants, equal params) | E7 (Family D, H6a), E8 |
| **H6b** | Foundation zero-shot/finetune on target history vs from-scratch (reference; foundation never sees NTL) (R23) | M10 (Chronos/Moirai/TimesFM; CPU zero-shot / GPU finetune) | E7 (Family D, H6b), E8 (`deferred` if GPU absent) |

---

## 8. Deliverables → course-rubric mapping (summary)

Full traceability table is **`docs/RUBRIC_MAPPING.md`** + paper appendix, owned by [`docs/devplan/06-paper-and-deliverables.md`](docs/devplan/06-paper-and-deliverables.md) (W13). Summary:

| Rubric criterion (RESEARCH.MD) | Earned by | Primary artifacts |
|---|---|---|
| **INTRO** — topic + goals + ML problem | W2 | `paper/sections/00-abstract.tex`, `01-introduction.tex`; `configs/hypotheses.yaml` |
| **HYPOTHESES & METHOD** — algorithm/math/code soundness + block diagram | W4, W6 | `03-methodology.tex`, `assets/block-diagram.tex` + `figures/block-diagram.{pdf,png}`; `train/trainer.py`, `utils/seed.py` |
| **RESEARCH** — related work | W3 | `02-related-work.tex`, `paper/refs.bib` |
| **APPLICATION** — empirical runs + metrics + HP search + statistics + problem↔solution map | W5, W7 | `04-data.tex`, `05-experimental-setup.tex`, `06-results.tex`; `results_store.parquet`, `dm_results.parquet`, figures/tables; `results.lock.json` |
| **WHAT IS LEARNED** — conclusions answering why | W8, W10 | `07-discussion.tex`, `08-conclusion.tex`; `docs/LICENSING_AND_ETHICS.md` |

---

## 9. Milestone schedule with go/no-go gates (relative weeks)

Front-loads CPU-feasible results; GPU-only work (Mamba official, foundation finetune) is an enhancement, never a prerequisite for a complete, gradeable deliverable.

| Week | Focus | Tasks | Go/No-Go gate |
|---|---|---|---|
| **1** | Setup + data acquisition + paper scaffold | S1–S12; N1 (auth) + start N downloads; F1 + F downloads; W1 (skeleton), W9 (README stub), W10 (licensing), W12 (manifest/env scaffold) | Repo bootstraps; `pytest`/`ruff`/`black` green CPU no-network; secrets resolve; ≥1 VNP46A3 month + full ETF/FRED/VIX downloaded; `requirements.lock.txt` written. **No-Go if** any downloader fails auth or core env smoke fails. |
| **2** | Panel + baselines (CPU) | N feature extraction (N5–N8); P1–P10 panel + folds + **leakage audit**; M1–M3 baselines; W5 (Data/Setup draft), W2 (Intro draft) | Panel parquet built to Appendix-A schemas; **leakage audit all five invariants pass** (negative controls fail on tampering); momentum + DLinear produce contract-valid walk-forward predictions. **No-Go if** leakage audit fails. |
| **3** | CPU deep models | M4–M6 PatchTST + iTransformer (CPU) + M7 S6 CPU fallback; M8 pretraining (smoke); M12 HP search (coarse step); W4 Methodology + W6 block diagram | PatchTST & iTransformer train and beat ≥1 baseline on ≥1 ETF; variate view yields ≥2 variates for the H2 sectors (R7); block diagram renders. **No-Go if** deep models fail the persistence sanity floor or H2 stratum is empty. |
| **4** | Evaluation + stats (CPU) | E1–E12 metrics, **DM (HAC data-driven lag + HLN + date-clustered)**, binomial dir-acc test, stratified (VIX>25) analyses, plots; W7 asset collection; W3 Related Work | Leaderboard + DM matrix + ≥4 hypothesis figures generated; H1/H2/H3/H5 evaluable CPU-only; realized `T` recorded per DM test with underpowered flags. **No-Go if** DM tests or multiple-comparison correction missing. |
| **5** | GPU-only experiments (deferred) | M7 Mamba official (WSL2/Colab) for H4; M10 foundation finetune for H6b; re-run E with full model set | GPU path reproduced **or** explicitly `deferred` with CPU S6 Mamba + CPU zero-shot foundation substituted and limitation noted (`gpu_stages_skipped` in manifest). **GO either way** if the substitution is documented. |
| **6** | Paper assembly + verdicts | W7 refresh from final run; W8 Discussion verdicts + Conclusion; W13 rubric mapping; finalize W9 README, W11 `run_all` | All H1–H6 have verdicts; explicit H0 decision stated; `main.pdf` builds with all sections; rubric mapping complete with every artifact existing. **No-Go if** any rubric criterion lacks a mapped artifact. |
| **7** | Reproducibility hardening + DoD | W11 end-to-end on clean checkout; W12 test suite; W14 DoD; W10 finalize | `check_deliverables.py` exits 0 on automatable checks; clean-checkout `run_all -SkipGpu` reproduces figures; DoD all-checked; no secrets; RESEARCH.MD unchanged. **No-Go = project not done.** |

**Enforceable ordering rule.** Never block the CPU pipeline on GPU work. If no GPU is available, complete Weeks 1–4 and 6–7 with the CPU S6 fallback and CPU zero-shot foundation models substituted, mark `manifest.gpu_stages_skipped`, set H4/H6b verdicts to `deferred` (never `reject`), and ship a complete deliverable.

---

## Appendix A — Data & Artifact Contracts (NORMATIVE)

This appendix is binding. Every phase cites it instead of re-declaring. Conflicts are resolved in favor of this appendix.

### A.1 Canonical time index
- **All monthly tables and the panel use a `date` column that is a month-END `pandas.Timestamp`, tz-naive, normalized to midnight.** No month-start, no `PeriodIndex` serialized as first-of-month.
- The master grid is `2013-01-31 … 2024-12-31` (144 month-ends).
- Loaders left-join each source onto this grid; genuine gaps stay `NaN` (never forward-filled across pre-inception gaps).
- `build_panel.py` asserts every joined table's index equals the master month-end grid; `df.date.dt.is_month_end.all()` and `df.date.dt.tz is None` hold everywhere. (Resolves R1; updates P9 + all of Phase P from month-start to month-end.)

### A.2 VIX table — `data/processed/vix_monthly.parquet`
| column | dtype | meaning |
|---|---|---|
| `date` | datetime64[ns] | month-end, tz-naive |
| `vix_mean` | float32 | mean of daily VIX closes in the month |
| `vix_max` | float32 | max daily VIX close in the month |
| `disruption_flag` | bool | `vix_mean > 25.0` |

F9 emits all four columns; P2/P9 and E6 read them. H4 strata use `disruption_flag` (or recompute from `vix_mean` + threshold consistently). (Resolves R19.)

### A.3 Prediction artifact — `experiments/<run_id>/predictions.parquet` (SINGLE FILE)
Exact columns and dtypes; E1 asserts this and accepts nothing else.

| column | dtype | meaning |
|---|---|---|
| `model` | str | `momentum`,`dlinear`,`patchtst`,`itransformer`,`mamba`,`chronos`,`moirai`,`timesfm` |
| `variant` | str | `scratch`,`pretrained`,`zeroshot`,`finetuned` |
| `pretrained` | bool | derived: `variant in {pretrained,finetuned}` |
| `task` | str | `leading` or `nowcast` |
| `target_kind` | str | the target series identity (replaces the `_IP` suffix), e.g. `return` / `ip` |
| `etf` | str | one of the 11 SPDR tickers (replaces `series_id`) |
| `horizon` | int | H in months |
| `fold` | int | walk-forward fold index |
| `split` | str | `test` (val/train rows optional) |
| `date` | datetime64[ns] | the **target** month (month-end, A.1) |
| `y_true` | float64 | realized target (de-standardized units) |
| `y_pred` | float64 | prediction (de-standardized units) |
| `seed` | int | RNG seed |

M0/M9 write exactly this single file; E1 reads `experiments/<run_id>/predictions.parquet` (not a directory). (Resolves R3.)

### A.4 Metrics store — `experiments/results_store.parquet` (AUTHORITATIVE) + `.csv`
Long/tidy, one metric value per row. Extends E1's schema with the columns the GPU-merge dedup needs.

| column | dtype | notes |
|---|---|---|
| `run_id` | str | provenance |
| `model` | str | |
| `variant` | str | |
| `task` | str | |
| `scope` | str | ETF ticker or `POOLED` |
| `fold` | int | or `-1` for cross-fold aggregate |
| `stratum` | str | `all`,`single_region`,`multi_region`,`disruption`,`stable`,`pretrained`,`from_scratch` |
| `metric` | str | `mse`,`mae`,`dir_acc`,`dir_acc_pvalue`,`sharpe_gross`,`sharpe_net`,`nowcast_r2`,`pearson`,`n_obs` |
| `value` | float | |
| `ci_low` / `ci_high` | float | dispersion/CI (NaN for raw per-fold rows) |
| `status` | str | `ok`,`skipped`,`failed` |
| `profile` | str | `windows_cpu`,`gpu_full`,… |
| `seed` | int | |
| `git_sha` | str | |
| `config_hash` | str | dedup key for S4 merge |

`merge_results.py` dedups on `(run_id, model, variant, task, scope, fold, stratum, metric, config_hash)`. `.gitignore` tracks `experiments/results_store.parquet` (+ `.csv`) and `experiments/**/manifest.json`. S8's old `results.parquet` is repurposed/merged into this. (Resolves R4.)

### A.5 Run manifest superset — `experiments/<run_id>/manifest.json`
Defined once in `src/ntl_etf/utils/manifest.py` (Phase S); M and W **extend**, never redefine. Git key spelling standardized to **`git_sha`**.

Required keys: `run_id, timestamp_utc, git_sha, git_dirty, python, platform, seed, profile, capabilities, packages, config, config_hash, scaler_fit_on ("train"), model, variant, task, horizon, n_folds, mamba_impl (null|"official"|"fallback"), data_hashes {panel, splits}, stages_completed, gpu_stages_skipped`. (Resolves R16, R21.)

### A.6 `configs/regions.yaml` — unified schema (owner N4; P consumes)
One entry per region; geometry owned by N, but a region may proxy multiple sectors.

```yaml
schema_version: 1
regions:
  - id: permian_basin
    name: Permian Basin
    bbox: [W, S, E, N]          # ONE coordinate set per region (reconcile N/P duplicates)
    shapefile: null              # optional upgrade under data/external/
    anchor: true
    candidate_sectors: [XLE]     # list form (a region can proxy >1 sector)
    rationale: "TX/NM oil basin; gas-flaring NTL"
    hypothesis: H3
features:                        # NTL feature columns the panel may use (R25: choose primary set)
  - ntl_sum
  - ntl_mean
  - ntl_lit_frac
hypothesis_pairs:
  H2_multiregion:  { sector: XLI, regions: [pearl_river_delta, yangtze_river_delta] }
  H3_singleregion: { sector: XLE, regions: [permian_basin] }
```
Every SPDR ticker appears in ≥1 region's `candidate_sectors`; ≥2–3 sectors have ≥2 plausible regions so H2 is not n=1 (R7). N4's and P1's acceptance tests both assert this single schema. (Resolves R15.)

### A.7 Orchestration ownership
`scripts/run_all.ps1` / `run_all.sh` have a **single owner: W11** (stages download→panel→train→eval→figures→paper; flags `-SkipGpu`/`-RunId`/`-Seed`/`-DryRun`). S11 creates a stub W11 supersedes. M14's experiment-matrix logic lives in `scripts/run_experiments.ps1`, invoked by W11's train stage. (Resolves R20.)
