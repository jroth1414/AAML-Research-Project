# Phase E — Paper, Rubric Mapping, Reproducibility & Deliverables

`docs/devplan/06-paper-and-deliverables.md` · Task-ID prefix **W**

This phase converts the executed pipeline into the graded artifact: a rubric-aligned paper, a reproducibility surface (README + one-command orchestration + environment freeze + run manifests), a definition-of-done checklist, a licensing/ethics note, and a relative-week milestone schedule with go/no-go gates. It is the *last* phase to produce content but its scaffolding (W1–W4, W9, W12) should be created early so other phases write into the right places.

This document depends on outputs from every other phase. Cross-phase task IDs are referenced as: **S** (setup, `00-overview-and-setup.md`), **N** (NTL data, `01`), **F** (financial/macro, `02`), **P** (panel/splits, `03`), **M** (models/training, `04`), **E-stats** (evaluation/stats, `05`). To avoid prefix collision with this phase's own **E**-numbered tasks, evaluation-phase tasks are written **E5.x** (the `05-` file) and this phase's tasks are **W*n***.

---

## Conventions for all W tasks

- All paths are repo-relative to `AAML-Research-Project/`. The autonomous agent runs on Windows 11; give PowerShell first, bash second where they differ.
- The paper lives in `paper/`. Source-of-truth figures and tables are produced by Phase E5 (`src/ntl_etf/eval/plots.py`, `eval/metrics.py`, `eval/stats.py`) and written to `experiments/<run_id>/figures/` and `experiments/<run_id>/tables/`. The paper **imports** those artifacts; it never recomputes numbers.
- Never invent citations. The bibliography is the proposal bibliography (Phase S records it at `paper/refs.bib`). If a reference is missing from the proposal, mark `% TO-VERIFY: missing from proposal bibliography` rather than fabricating.
- Do **not** name Claude/any AI as an author, acknowledgee, or contributor anywhere in `paper/`, `README.md`, manifests, or commit metadata authorship fields.
- Preserve `RESEARCH.MD` verbatim. It is the rubric source and must not be edited by any W task.

---

## W1 — Create the `paper/` skeleton (LaTeX primary, Markdown drafting mirror)

**Objective.** Stand up the paper directory with a compilable LaTeX shell and a parallel Markdown draft so prose can be written CPU-only before final results exist.

**Dependencies.** S1 (repo bootstrap), S-bib (proposal bibliography captured at `paper/refs.bib`).

**Actions.**
1. Create `paper/` with the structure below. LaTeX is the *primary* deliverable (PDF is graded); Markdown is a drafting convenience that can be authored and reviewed without a TeX toolchain.
2. Choose a self-contained, common class so the agent does not depend on a specific template: `article` with `\usepackage{...}` for `graphicx, booktabs, amsmath, amssymb, hyperref, natbib, geometry, subcaption, siunitx`. Do **not** assume a proprietary venue template.
3. Write one `\input{}`-ed file per section so sections can be drafted independently and merged by `main.tex`.
4. Provide a `latexmkrc` and a `Makefile`/PowerShell build that compiles to `paper/build/main.pdf`. If no TeX distribution is installed, the build must fail with a clear, actionable message (see W11) — it must NOT silently produce nothing.
5. Mirror each LaTeX section as a `.md` file under `paper/markdown/` with identical headings, for early prose drafting and for reviewers who cannot build TeX.

**Files to create.**
```
paper/
  main.tex
  refs.bib                      # produced by S-bib; W1 only references it
  latexmkrc
  build.ps1   build.sh
  sections/
    00-abstract.tex
    01-introduction.tex
    02-related-work.tex
    03-methodology.tex
    04-data.tex
    05-experimental-setup.tex
    06-results.tex
    07-discussion.tex
    08-conclusion.tex
  figures/                      # symlinked/copied from experiments/<run_id>/figures (W7)
  tables/                       # copied from experiments/<run_id>/tables (W7)
  markdown/                     # 00..08 .md mirrors
  assets/
    block-diagram.tex           # TikZ source for the architecture block diagram (W6)
```

**`main.tex` skeleton.**
```latex
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx,booktabs,amsmath,amssymb,hyperref,natbib,subcaption,siunitx}
\graphicspath{{figures/}}
\title{Nighttime Light Emissions as an Economic Proxy for Sector ETF
Forecasting: PatchTST vs.\ iTransformer vs.\ Mamba}
\author{John Roth\\ JHU EN.705.742 Advanced Applied Machine Learning}
\date{\today}
\begin{document}\maketitle
\input{sections/00-abstract}
\input{sections/01-introduction}
\input{sections/02-related-work}
\input{sections/03-methodology}
\input{sections/04-data}
\input{sections/05-experimental-setup}
\input{sections/06-results}
\input{sections/07-discussion}
\input{sections/08-conclusion}
\bibliographystyle{plainnat}
\bibliography{refs}
\end{document}
```

**Build commands.**
```powershell
# PowerShell
pwsh -File paper/build.ps1          # wraps: latexmk -pdf -outdir=build paper/main.tex
```
```bash
# bash
bash paper/build.sh                 # wraps: latexmk -pdf -outdir=build paper/main.tex
```

**Deliverables.** Compilable LaTeX skeleton (placeholder prose allowed), Markdown mirror, build scripts.

**Acceptance Criteria.**
- `pwsh -File paper/build.ps1` produces `paper/build/main.pdf` with all 9 sections present (or fails with the explicit "TeX not installed" message from W11).
- `paper/markdown/` contains 9 `.md` files whose H1/H2 headings match the LaTeX section titles 1:1.
- `RESEARCH.MD` is unchanged (`git diff --stat RESEARCH.MD` is empty).

---

## W2 — Section content spec: Abstract, Introduction & Goals

**Objective.** Specify exactly what goes in the front matter so the rubric INTRO criterion is unambiguously earned.

**Dependencies.** W1.

**Actions.** Write `sections/00-abstract.tex` and `sections/01-introduction.tex` to contain, in order:
1. **Abstract** (≤250 words): the core claim (monthly VIIRS NTL over economically-linked regions predicts forward 1–3 month log returns of the 11 SPDR sector ETFs and nowcasts contemporaneous sector industrial production), the three architectures + two baselines + transfer-learning regime, the walk-forward protocol, and a one-line headline result (filled from W7).
2. **Introduction & Goals** must explicitly state, each as its own labeled paragraph so a grader can find them:
   - *Research topic*: NTL as an economic proxy for sector forecasting.
   - *Goals*: the LEADING task (forward returns) and the COINCIDENT NOWCAST task (industrial production), and the comparative-architecture question.
   - *ML problem formalization*: multivariate time-series forecasting / regression + directional classification on a channel-independent global panel; define inputs (region NTL features + lagged target context), outputs (h∈{1,2,3}-month log return; contemporaneous IP), loss (MSE; auxiliary directional metric).
   - *Why it matters*: alternative-data economic signal, architecture-suitability question (cross-variate vs. channel-independent vs. state-space).
3. End the Introduction with an explicit enumerated list of hypotheses **H1–H6** and the null **H0**, copied to match Phase E5 decision rules verbatim (single source of truth: reference `configs/hypotheses.yaml` from E5).

**Files.** `paper/sections/00-abstract.tex`, `paper/sections/01-introduction.tex` (+ markdown mirrors).

**Deliverables.** Drafted front matter with all rubric INTRO elements labeled.

**Acceptance Criteria.**
- A grep over `01-introduction.tex` finds the literal tokens `Research topic`, `Goals`, `ML problem`, and `H1`…`H6` and `H0`.
- Hypotheses text is byte-identical to the canonical statements in `configs/hypotheses.yaml` (assert via a test in W12 that diffs the two).

---

## W3 — Section content spec: Related Work

**Objective.** Earn the RESEARCH rubric criterion with a related-work section grounded only in the proposal bibliography.

**Dependencies.** W1, S-bib.

**Actions.** Write `sections/02-related-work.tex` organized into four themed subsections, each citing only entries present in `paper/refs.bib`:
1. **Nighttime lights as economic proxy** (NTL→GDP/economic-activity literature).
2. **Time-series transformers** (PatchTST, iTransformer and the channel-independence vs. cross-variate debate).
3. **State-space sequence models** (Mamba/S6 and selective SSMs for forecasting).
4. **Time-series foundation models & self-supervised pretraining** (Chronos, Moirai, TimesFM, masked pretraining).
End with a one-paragraph **gap statement**: no prior work jointly (a) uses NTL for sector-ETF forecasting and (b) compares these three architecture families under a leakage-controlled walk-forward protocol.

**Files.** `paper/sections/02-related-work.tex` (+ mirror).

**Deliverables.** Related-work section with explicit gap statement.

**Acceptance Criteria.**
- Every `\cite{...}` key resolves to an entry in `refs.bib` (W12 runs a citation-key linter; zero undefined references in the `latexmk` log).
- The four themes and the gap paragraph are all present.

---

## W4 — Section content spec: Methodology (math + transfer learning + baselines)

**Objective.** Earn the HYPOTHESES & METHOD rubric criterion: correct algorithm/math, code-soundness narrative, and the block diagram hook.

**Dependencies.** W1, W6 (block diagram), and method definitions from M (model tasks) and P (panel/windowing).

**Actions.** Write `sections/03-methodology.tex` with these required subsections:
1. **Problem formalization.** Define the global panel notation: series index *s* over (region→sector) pairs, window length *L*, horizon *h*, the channel-independent shared-weight setup `f_θ: ℝ^{L×C} → ℝ^h`, the leakage-safe feature alignment (month-*t* NTL → predict month-*(t+1)* return, citing the VNP46A3 release lag, cross-ref P leakage audit and N release-lag handling).
2. **Architectures with math.** One subsection each:
   - *PatchTST*: patching, channel independence, the attention over patches; give the patch/stride and the transformer block equations.
   - *iTransformer*: inverted embedding (variates-as-tokens), cross-variate/cross-region attention; equations.
   - *Mamba (S6)*: the selective state-space recurrence `h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t`, `y_t = C_t h_t`, with input-dependent (selective) Δ, A, B, C; note the discretization.
   - A `\ref{}` to the block diagram figure (W6).
3. **Transfer-learning regime.** Masked self-supervised pretraining on the unlabeled NTL corpus, then fine-tuning; foundation-model initialization (Chronos/Moirai/TimesFM zero-shot and fine-tuned). Map to M pretraining/foundation tasks.
4. **Baselines.** 12-month time-series momentum (the H0 reference) and DLinear; give their exact formulas.
5. **Code-soundness paragraph.** State that all models share one trainer/config (`src/ntl_etf/train/trainer.py`), one metrics module, fixed seeds (`src/ntl_etf/utils/seed.py`), and train-split-only normalization — these are the claims the rubric "code soundness" sub-criterion checks; each links to the implementing task ID.

**Files.** `paper/sections/03-methodology.tex` (+ mirror).

**Deliverables.** Methodology section with all four architecture/baseline maths, transfer regime, and the diagram reference.

**Acceptance Criteria.**
- Each architecture subsection contains at least one numbered equation (`\begin{equation}`), and the Mamba selective-SSM recurrence appears.
- The block-diagram `\ref` resolves (no `??` in the PDF).
- The leakage guard (month-*t*→month-*(t+1)*) and train-only normalization are stated in prose with cross-refs to P and M task IDs.

---

## W5 — Section content spec: Data + Experimental Setup

**Objective.** Document data provenance and the exact experimental protocol so APPLICATION rubric "proper problem↔solution association" and reproducibility are visible in the paper.

**Dependencies.** N (NTL), F (finance/macro), P (panel/splits), M (HP search), E5 (protocol).

**Actions.**
1. **`sections/04-data.tex`.** Tabulate: VIIRS VNP46A3 (~500 m / 15 arc-second monthly composite, 2013-01…2024-12), the region set (`configs/regions.yaml`), the 11 ETFs with inception/history notes and how ragged histories are handled (cross-ref F), FRED sector IP series (`configs/sector_fred_map.yaml`) with a note on sectors lacking a clean IP analog (cross-ref F), and VIX for disruption stratification. State the study window (~144 monthly obs/series) and the region→sector pre-screen correlation check (cross-ref P), explicitly noting the screen uses no look-ahead.
2. **`sections/05-experimental-setup.tex`.** Specify: walk-forward rolling-origin CV (≥60-month min train window, 1-month step), the HP-search protocol and grid (cross-ref M HP-search), the leakage controls (release-lag alignment, train-only normalization, no-look-ahead screen — list each as a guard), seed policy, and the gross/net Sharpe convention (10 bps one-way cost). Add a small table mapping each hypothesis H1–H6 to the metric and the figure/table that decides it (this is the problem↔solution association the rubric rewards).

**Files.** `paper/sections/04-data.tex`, `paper/sections/05-experimental-setup.tex` (+ mirrors).

**Deliverables.** Data and setup sections with provenance tables and the hypothesis→artifact map table.

**Acceptance Criteria.**
- Data section lists all 11 ETF tickers (XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV, XLY) and references both config files by path.
- Setup section contains a table with one row per H1–H6, each naming a concrete metric and a figure/table ID.
- The three leakage guards are each named explicitly.

---

## W6 — Architecture block diagram (TikZ → PDF/PNG)

**Objective.** Produce the single block diagram required by the HYPOTHESES & METHOD rubric criterion for the top mark.

**Dependencies.** W1; conceptual model defs from M.

**Actions.**
1. Author `paper/assets/block-diagram.tex` as standalone TikZ showing the end-to-end flow: *VNP46A3 rasters → region NTL feature extraction → release-lag alignment → global channel-independent panel → {PatchTST | iTransformer | Mamba} shared-weight encoder + baselines → forward-return head & nowcast head → walk-forward eval → DM tests / hypotheses.*
2. Compile it standalone to `paper/figures/block-diagram.pdf` and also export `paper/figures/block-diagram.png` (for the Markdown mirror and README). If TeX is unavailable, provide a fallback: a `matplotlib`-drawn diagram via `src/ntl_etf/eval/plots.py::plot_block_diagram()` writing the same two files, so the diagram exists CPU-only without TeX.
3. Reference it from `03-methodology.tex` (W4) and embed the PNG in `paper/markdown/03-methodology.md` and `README.md`.

**Files.** `paper/assets/block-diagram.tex`, `paper/figures/block-diagram.{pdf,png}`, fallback in `src/ntl_etf/eval/plots.py`.

**Deliverables.** Block diagram in both PDF and PNG.

**Acceptance Criteria.**
- Both `block-diagram.pdf` and `block-diagram.png` exist and are non-empty.
- The diagram shows all three deep models, both baselines, the two task heads, and the release-lag/leakage-guard node.

---

## W7 — Results ingestion: bind paper figures/tables to a frozen run

**Objective.** Make Results reproducible-by-reference: the paper renders artifacts emitted by Phase E5 for one canonical `run_id`, never hand-typed numbers.

**Dependencies.** E5.x (metrics, stats, plots, stratify), M (trained runs), W12 (manifest convention).

**Actions.**
1. Define a canonical results pointer file `paper/results.lock.json` containing the `run_id` (or set of run_ids) the paper reports, the git commit, and the absolute/relative path to `experiments/<run_id>/`.
2. Write `scripts/collect_paper_assets.py` that, given `results.lock.json`, copies `experiments/<run_id>/figures/*` → `paper/figures/` and `experiments/<run_id>/tables/*` (LaTeX `booktabs` `.tex` and `.csv`) → `paper/tables/`, and emits `paper/figures/MANIFEST.json` listing every asset with its source path + sha256.
3. Write `sections/06-results.tex` to `\input{}`/`\includegraphics{}` those copied artifacts only. Required result objects (all generated by E5, named here so W7 can assert they arrived):
   - **Table: main leaderboard** — per-model 1-month-return MSE, directional accuracy, nowcast R², gross/net Sharpe.
   - **Table: Diebold–Mariano** matrix (each DL model vs. each baseline) with p-values and the multiple-comparison correction note.
   - **Figure: per-hypothesis** panels (e.g., H2 iTransformer vs PatchTST on multi-region ETFs; H3 reverse on single-region; H4 disruption-month stratification VIX>25; H5 nowcast-R² vs leading-R²; H6 pretrained vs from-scratch).
   - **Figure: walk-forward** error-over-origin and an example forecast overlay.

**Files.** `paper/results.lock.json`, `scripts/collect_paper_assets.py`, `paper/sections/06-results.tex` (+ mirror), `paper/figures/MANIFEST.json`.

**CLI.**
```powershell
C:/Users/Admin/AppData/Local/Programs/Python/Python311/python.exe scripts/collect_paper_assets.py --lock paper/results.lock.json
```
```bash
python scripts/collect_paper_assets.py --lock paper/results.lock.json
```

**Deliverables.** Asset-collection script, populated `paper/figures` & `paper/tables`, results section wired to them.

**Acceptance Criteria.**
- After running `collect_paper_assets.py`, every `\includegraphics`/`\input` target in `06-results.tex` exists on disk (W12 test asserts no missing graphics in the `latexmk` log).
- `paper/figures/MANIFEST.json` lists ≥1 leaderboard table, ≥1 DM table, ≥4 hypothesis figures, with sha256 hashes matching the copied files.
- No numeric literal in `06-results.tex` that should be a result (heuristic check in W12: results section contains `\input`/`\includegraphics` and no hardcoded `0.` decimal performance numbers outside table files).

---

## W8 — Section content spec: Discussion (H1–H6 verdicts) + Conclusion

**Objective.** Earn the WHAT IS LEARNED rubric criterion: per-hypothesis verdicts tied to evidence, an architecture-difference explanation, limitations, and a "why it mattered" conclusion.

**Dependencies.** W7, E5 decision rules (`configs/hypotheses.yaml`), W10 (limitations note).

**Actions.**
1. **`sections/07-discussion.tex`.** One labeled paragraph per hypothesis **H1…H6**, each stating: the pre-registered decision rule, the observed statistic (referencing the W7 table/figure), and the verdict (Supported / Not supported / Inconclusive). Then a synthesis paragraph: *why architectures differ* (cross-variate iTransformer vs single-region PatchTST per H2/H3; Mamba under disruption per H4; coincident-vs-leading per H5; transfer benefit per H6). Then the **limitations** paragraph (pull from W10): small sample (~144 obs), multiple-testing inflation and the correction used, assumed-and-screened NTL→ETF causal linkage, data vendor caveats.
2. **`sections/08-conclusion.tex`.** State what was learned, whether H0 was rejected, the practical implication for alternative-data forecasting, and concrete future work (more regions, daily/weekly NTL, larger foundation-model fine-tuning on GPU).

**Files.** `paper/sections/07-discussion.tex`, `paper/sections/08-conclusion.tex` (+ mirrors).

**Deliverables.** Discussion with six verdicts + synthesis + limitations; conclusion answering "why."

**Acceptance Criteria.**
- grep finds exactly six hypothesis verdict paragraphs (`H1`…`H6`), each containing one of the literal verdict tokens `Supported`, `Not supported`, or `Inconclusive`.
- Conclusion explicitly states the H0 decision.
- Limitations paragraph names: small-sample, multiple-testing, assumed causal linkage, and data caveats.

---

## W9 — README.md: setup, data access, full regeneration

**Objective.** A single README that lets a fresh machine reproduce everything, including secrets handling and the GPU-only boundary.

**Dependencies.** S1 (env), N/F (downloaders), P/M/E5 (pipeline), W7 (asset collection), W11 (orchestration).

**Actions.** Write `README.md` with these sections (exact headings):
1. **Overview** — one paragraph + the block-diagram PNG.
2. **Requirements** — Python 3.11 at `C:/Users/Admin/AppData/Local/Programs/Python/Python311/python.exe`; OS notes; the explicit statement that the full pipeline runs **CPU-only except Mamba and foundation-model fine-tuning**.
3. **Install** — venv creation + `pip install -r requirements.txt` (PowerShell and bash), plus the optional `requirements-gpu.txt` for the GPU/WSL path.
4. **Secrets** — copy `.env.example`→`.env` (gitignored); set `EARTHDATA_TOKEN` (or `.netrc`) and `FRED_API_KEY`. Link to where to obtain each (NASA Earthdata login; FRED API key page). State that `.env`, `.netrc`, and `data/` are gitignored and must never be committed.
5. **Reproduce everything** — the single orchestration entry (W11): `run_all.ps1` / `run_all.sh`, with the stage list download→panel→train→eval→figures→paper and how to run a CPU-only subset (`-SkipGpu`).
6. **GPU-only parts** — exactly which steps need GPU (Mamba via mamba-ssm; foundation-model fine-tuning) and the three reproduction paths: WSL2+CUDA, Colab/cloud GPU, or the pure-PyTorch CPU S6 fallback behind the capability flag (cross-ref M Mamba task). State that CPU-only zero-shot foundation inference is supported.
7. **Outputs** — where results, manifests, figures, and the paper PDF land.
8. **Citation & licensing** — link to the licensing/ethics note (W10).

**Files.** `README.md` (repo root).

**Deliverables.** Complete README.

**Acceptance Criteria.**
- README documents all six pipeline stages in order and names `run_all.ps1`.
- README states the CPU-only-vs-GPU boundary and the three Mamba/foundation reproduction paths.
- README never prints a real secret value; it references env-var names only (W12 test greps README for `EARTHDATA_TOKEN`/`FRED_API_KEY` patterns that look like actual keys and fails if found).

---

## W10 — Limitations, data-licensing & ethics note

**Objective.** A standalone, citable note covering data terms and research caveats (feeds W8 limitations and README citation section).

**Dependencies.** None blocking; finalize after W7.

**Actions.** Write `docs/LICENSING_AND_ETHICS.md` covering, with the verified facts below:

| Source | Terms (verified) | Required action in repo/paper |
|---|---|---|
| **NASA VIIRS Black Marble VNP46A3** | Distributed by NASA LAADS DAAC under EOSDIS open Data Use & Citation Guidance — openly shared without restriction; cite with the dataset DOI. | Cite dataset DOI `10.5067/VIIRS/VNP46A3.002` (verify exact collection/version at download time, N task) and acknowledge NASA EOSDIS in `paper/refs.bib` + Data section. |
| **yfinance / Yahoo Finance** | Unofficial endpoints; intended for personal/research/educational use, **not** commercial use; consult Yahoo Terms of Service for data-use rights. | State in limitations that data is via the unofficial `yfinance` API for academic research only; do not redistribute raw Yahoo data; results are research, not investment advice. |
| **FRED (St. Louis Fed)** | FRED API Terms of Use; cite series with FRED attribution and original source (e.g., "Source: BLS via FRED"); do not redistribute third-party proprietary content. | Attribute each FRED series as "Source: <origin> via FRED" in the Data section; do not commit raw FRED proprietary series beyond what terms allow (keep in gitignored `data/`). |
| **CBOE VIX** | Used only for disruption stratification (VIX>25); attribute source. | One-line attribution in Data section. |

Also include the **research caveats**: (a) the NTL→ETF causal linkage is *assumed and pre-screened*, not established — the correlation pre-screen reduces but does not eliminate spurious selection; (b) small sample (~144 monthly obs/series) limits power and invites overfitting — mitigated by the global channel-independent panel and walk-forward CV; (c) multiple-comparison inflation across 6 hypotheses × multiple ETFs — mitigated by pre-registered DM tests and the family-wise correction from E5; (d) results are not investment advice.

**Files.** `docs/LICENSING_AND_ETHICS.md`.

**Deliverables.** Licensing/ethics note.

**Acceptance Criteria.**
- File names all four data sources with their terms and the required attribution string for each.
- File states all four research caveats (assumed causal link, small sample, multiple testing, not-investment-advice).
- The VNP46A3 DOI string and the "Source: ... via FRED" attribution convention are present.

---

## W11 — One-command orchestration: `run_all.ps1` / `run_all.sh`

**Objective.** A single reproducibility entry point that runs download → panel → train → eval → figures → paper in order, CPU-safe by default, with a GPU opt-in.

**Dependencies.** Downloaders (N1/F1), `scripts/build_panel.py` (P), `scripts/run_experiment.py` (M), E5 figure/table generation, W7 asset collection, W1 paper build.

**Actions.**
1. Write `scripts/run_all.ps1` and `scripts/run_all.sh` orchestrating the stages by calling the existing per-stage scripts (do not duplicate logic). Stages, in order:
   1. `download_ntl.py` → `download_finance.py` → `download_macro.py`
   2. `build_panel.py`
   3. `run_experiment.py` for baselines + CPU-feasible deep models
   4. (GPU-gated) Mamba + foundation-model fine-tuning, skipped unless `-Gpu`/`--gpu`
   5. E5 eval (metrics, DM stats, stratified analyses, plots)
   6. `collect_paper_assets.py` then paper build (`build.ps1`)
2. Flags: `-SkipDownload` (use cached `data/`), `-Gpu`/`-SkipGpu` (default skip), `-RunId <id>`, `-Seed <n>`. Each stage logs to `experiments/<run_id>/logs/<stage>.log` and writes/updates the run manifest (W12).
3. Fail-fast: any nonzero stage exit stops the run with the failing stage name. The TeX-missing case (W1) is a *soft* failure — report it but still leave all figures/tables/manifests in place so the run is otherwise reproducible.
4. Guard: assert required env vars (`EARTHDATA_TOKEN`/`.netrc`, `FRED_API_KEY`) exist before the download stage; print a clear remediation message pointing to README §Secrets if missing.

**Files.** `scripts/run_all.ps1`, `scripts/run_all.sh`.

**CLI.**
```powershell
pwsh -File scripts/run_all.ps1 -RunId paper_main -Seed 1414 -SkipGpu
```
```bash
bash scripts/run_all.sh --run-id paper_main --seed 1414 --skip-gpu
```

**Deliverables.** Cross-platform orchestration scripts.

**Acceptance Criteria.**
- A dry-run mode (`-DryRun`) prints the exact ordered stage commands without executing; W12 asserts the six stages appear in order.
- With GPU skipped and `data/` cached, the script runs baselines+CPU deep models end-to-end and produces `experiments/<run_id>/figures`, `.../tables`, and (TeX permitting) `paper/build/main.pdf`.
- Missing secrets cause a clear pre-download abort, not a mid-run crash.

---

## W12 — Environment freeze, run-manifest convention, and reproducibility tests

**Objective.** Pin the environment, define the run manifest, and add automated checks that the paper/repo are internally consistent and leakage-safe at the deliverable level.

**Dependencies.** S1 (`requirements.txt`/`pyproject.toml`), all phases (manifests are written by trainers/eval).

**Actions.**
1. **Environment freeze.** After a clean install, write the exact resolved versions to `requirements.lock.txt` via `pip freeze`, and record interpreter + platform + key library versions (torch, numpy, pandas, rasterio, statsmodels, mamba-ssm if present, chronos/uni2ts/timesfm if present) into `experiments/<run_id>/env.json`. Keep `requirements.txt` (loose, with the GPU extras split into `requirements-gpu.txt`) plus `requirements.lock.txt` (exact).
   ```powershell
   C:/Users/Admin/AppData/Local/Programs/Python/Python311/python.exe -m pip freeze > requirements.lock.txt
   ```
2. **Run-manifest convention.** Standardize `experiments/<run_id>/manifest.json` (single schema reused by M and E5):
   ```json
   {
     "run_id": "paper_main",
     "git_commit": "<sha>",
     "created_utc": "<iso8601>",
     "seed": 1414,
     "device": "cpu",
     "config_files": {"experiment": "configs/exp_main.yaml", "regions": "configs/regions.yaml"},
     "config_sha256": {"configs/exp_main.yaml": "<sha>"},
     "data_inputs": {"panel": "data/processed/panel.parquet", "panel_sha256": "<sha>"},
     "stages_completed": ["download","panel","train","eval","figures"],
     "metrics_summary_path": "experiments/paper_main/tables/leaderboard.csv",
     "env_path": "experiments/paper_main/env.json",
     "gpu_stages_skipped": ["mamba","foundation_finetune"]
   }
   ```
   Provide `src/ntl_etf/utils/io.py::write_manifest(run_id, **fields)` and `read_manifest(run_id)`; every stage appends to `stages_completed`.
3. **Reproducibility / consistency tests** in `tests/test_deliverables.py`:
   - Hypothesis text in `01-introduction.tex` matches `configs/hypotheses.yaml` (W2).
   - Every `\cite` key in `paper/sections/*.tex` exists in `refs.bib`; no `\includegraphics`/`\input` target is missing (parse `paper/build/main.log` for `Missing` / undefined references) (W3, W7).
   - `paper/figures/MANIFEST.json` sha256 values match files on disk (W7).
   - README documents all six `run_all` stages and the GPU boundary (W9).
   - No secret-shaped strings committed: grep tracked files for plausible Earthdata bearer tokens / FRED 32-hex keys; fail if any match (W9, S secrets).
   - `RESEARCH.MD` byte-unchanged vs. the committed baseline.
   - Manifest schema: required keys present, `git_commit` non-empty, `seed` recorded (S reproducibility).

**Files.** `requirements.lock.txt`, `requirements-gpu.txt`, `src/ntl_etf/utils/io.py` (manifest helpers), `tests/test_deliverables.py`, `experiments/<run_id>/env.json` (generated).

**Deliverables.** Frozen env, manifest schema + helpers, deliverable test suite.

**Acceptance Criteria.**
- `pytest tests/test_deliverables.py` passes on a completed `paper_main` run.
- `requirements.lock.txt` exists with pinned `==` versions; `env.json` records interpreter + platform + torch/numpy/pandas versions.
- The secret-scan test passes (no secrets) and the `RESEARCH.MD`-unchanged test passes.

---

## W13 — Rubric mapping table (top-mark traceability)

**Objective.** Produce the explicit mapping from each RESEARCH.MD rubric criterion to the deliverable + paper section + repo artifact that earns the top (26 pt) mark. This table is itself a deliverable (placed in the paper appendix and the README).

**Dependencies.** W2–W12.

**Actions.** Create `docs/RUBRIC_MAPPING.md` and embed it as a paper appendix `paper/sections/09-appendix-rubric.tex`. Use this table (fill artifact paths as files are produced):

| Rubric criterion (RESEARCH.MD) | What top mark requires | Paper section earning it | Repo artifact(s) | Owning task |
|---|---|---|---|---|
| **INTRO** — topic + goals + ML problem | Topic, both task goals, and a formal ML-problem statement, all explicit | §1 Introduction & Goals (labeled paragraphs); Abstract | `sections/00-abstract.tex`, `01-introduction.tex`; `configs/hypotheses.yaml` | W2 |
| **HYPOTHESES & METHOD** — algorithm/math/code soundness + block diagram | Correct math for all 3 architectures + 2 baselines, transfer regime, the block diagram, and a code-soundness statement (shared trainer, seeds, train-only norm) | §3 Methodology; Fig. block diagram | `sections/03-methodology.tex`, `assets/block-diagram.tex`, `figures/block-diagram.pdf`; `src/ntl_etf/train/trainer.py`, `utils/seed.py` | W4, W6 |
| **RESEARCH** — related work | Four themed, correctly-cited related-work areas + a gap statement | §2 Related Work | `sections/02-related-work.tex`, `refs.bib` | W3 |
| **APPLICATION** — empirical runs + metrics + HP search + statistics + problem↔solution association | Leaderboard, DM significance tests w/ correction, HP-search description, walk-forward protocol, and the H→metric→artifact map | §4 Data, §5 Experimental Setup, §6 Results | `sections/04-data.tex`, `05-experimental-setup.tex`, `06-results.tex`; `experiments/<run_id>/tables/leaderboard.csv`, `.../dm_tests.csv`, figures; `results.lock.json` | W5, W7 |
| **WHAT IS LEARNED** — conclusions answering why | Six per-hypothesis verdicts tied to evidence, architecture-difference synthesis, limitations, and a "why it matters" conclusion | §7 Discussion, §8 Conclusion | `sections/07-discussion.tex`, `08-conclusion.tex`; `docs/LICENSING_AND_ETHICS.md` | W8, W10 |

**Files.** `docs/RUBRIC_MAPPING.md`, `paper/sections/09-appendix-rubric.tex` (add `\input` to `main.tex`).

**Deliverables.** Rubric-mapping table in repo and paper.

**Acceptance Criteria.**
- The table has one row per rubric criterion in RESEARCH.MD; W12 asserts every row's named artifact path exists.
- The appendix renders in `main.pdf`.

---

## W14 — Deliverables checklist (definition of done for the whole project)

**Objective.** A single authoritative DoD the agent can self-verify against before declaring the project complete.

**Dependencies.** All W tasks and all upstream phases.

**Actions.** Create `docs/DELIVERABLES_CHECKLIST.md`. The project is **done** only when every box is checked:

```
[ ] Code package: src/ntl_etf/ importable; `pytest tests/` green (data, models, eval, deliverables).
[ ] Configs present & valid: regions.yaml, sector_fred_map.yaml, hypotheses.yaml, model/experiment YAMLs.
[ ] Data build scripts run: download_ntl/finance/macro.py + build_panel.py produce data/processed/panel.parquet.
[ ] Results store: experiments/<run_id>/ with manifest.json, env.json, logs/, tables/, figures/.
[ ] All paper figures & tables generated by Phase E5 and collected into paper/figures, paper/tables (W7 MANIFEST verified).
[ ] Paper source compiles: paper/build/main.pdf with all 9 sections + rubric appendix.
[ ] Block diagram present (PDF + PNG).
[ ] README.md complete: setup, secrets, one-command reproduction, GPU boundary.
[ ] run_all.ps1 / run_all.sh reproduce download->panel->train->eval->figures->paper.
[ ] Environment frozen: requirements.txt + requirements-gpu.txt + requirements.lock.txt + env.json.
[ ] Run/model manifests written for the reported run(s); seeds recorded.
[ ] Licensing & ethics note (docs/LICENSING_AND_ETHICS.md) covers NASA/yfinance/FRED/VIX + 4 research caveats.
[ ] Rubric mapping table (docs/RUBRIC_MAPPING.md) — every criterion mapped, every artifact exists.
[ ] GPU-only parts documented with 3 reproduction paths; CPU-only path verified end-to-end.
[ ] No AI listed as author anywhere; RESEARCH.MD unchanged; no secrets committed; data/ gitignored.
[ ] Hypotheses H1-H6 each have a recorded verdict; H0 decision stated in Conclusion.
```

**Files.** `docs/DELIVERABLES_CHECKLIST.md`.

**Deliverables.** The DoD checklist.

**Acceptance Criteria.**
- A script `scripts/check_deliverables.py` programmatically verifies each checkbox it can (file existence, `pytest` exit, manifest keys, RESEARCH.MD unchanged, no-secrets, PDF built) and prints a pass/fail per line; exits nonzero if any automatable check fails.
- Manual-only items (e.g., prose-quality verdicts present) are flagged for human/grader review but their *presence* is asserted (grep).

---

## W15 — Milestone schedule with go/no-go gates (relative weeks)

**Objective.** Order the work to front-load CPU-feasible results and defer GPU-only experiments, with explicit gates so the autonomous agent knows when to proceed vs. stop and fix.

**Dependencies.** Integrates all phases; this is the master ordering for the agent.

**Schedule (relative weeks; each gate is GO only if its criteria pass, else fix before advancing).**

| Week | Focus | Tasks (this phase + cross-phase) | Go/No-Go gate |
|---|---|---|---|
| **1** | Setup + data acquisition | S1–S*; N1 NTL download; F1 finance/macro download; **W1** paper skeleton, **W9** README stub, **W12** manifest+env scaffold, **W10** licensing note | Repo bootstraps; secrets resolve; ≥1 month of VNP46A3 + full ETF/FRED/VIX series downloaded; `requirements.lock.txt` written. **No-Go if** any downloader fails auth. |
| **2** | Panel + baselines (CPU) | N feature extraction; P panel + walk-forward splits + **leakage audit**; M baselines (momentum, DLinear); **W5** Data/Setup draft, **W2** Intro draft | Panel parquet built; leakage audit passes (no month-*t* feature predicts ≤month-*t* target); baselines produce walk-forward metrics. **No-Go if** leakage audit fails. |
| **3** | CPU deep models | M PatchTST + iTransformer (CPU) + pure-PyTorch S6 CPU fallback; M HP search; **W4** Methodology + **W6** block diagram | PatchTST & iTransformer train and beat at least one baseline on ≥1 ETF; block diagram renders. **No-Go if** deep models do not exceed naive/persistence sanity floor. |
| **4** | Evaluation + stats (CPU) | E5 metrics, DM tests, stratified (VIX>25) analyses, plots; **W7** asset collection; **W3** Related Work | Leaderboard + DM matrix + ≥4 hypothesis figures generated; H1/H2/H3/H5 evaluable CPU-only. **No-Go if** DM tests not produced or multiple-comparison correction missing. |
| **5** | GPU-only experiments (deferred) | M Mamba via mamba-ssm (WSL2/Colab) for H4; foundation-model fine-tuning (Chronos/Moirai/TimesFM) for H6; re-run E5 with full model set | GPU path documented + reproduced, **or** explicitly marked deferred with CPU zero-shot foundation results + CPU S6 Mamba results substituting, and limitation noted. Gate is GO either way if the substitution is documented. |
| **6** | Paper assembly + verdicts | **W7** refresh from final run; **W8** Discussion verdicts + Conclusion; **W13** rubric mapping; finalize **W9** README, **W11** run_all | All six hypotheses have verdicts; `main.pdf` builds with all sections; rubric mapping complete. **No-Go if** any rubric criterion lacks a mapped artifact. |
| **7** | Reproducibility hardening + DoD | **W11** end-to-end run on clean checkout; **W12** test suite green; **W14** DoD; **W10** finalize | `scripts/check_deliverables.py` exits 0 on automatable checks; clean-checkout `run_all` reproduces figures; DoD all-checked. **No-Go = project not done.** |

**Ordering rule for the agent.** Never block the CPU pipeline on GPU work. If the GPU environment is unavailable, complete Weeks 1–4 and 6–7 with the CPU S6 fallback and CPU zero-shot foundation models substituted for the GPU variants, record this in `manifest.json.gpu_stages_skipped` and the limitations section, and still ship a complete, gradeable deliverable. GPU experiments (Week 5) are an enhancement, not a prerequisite for done.

**Files.** Add this schedule to `docs/devplan/06-paper-and-deliverables.md` (this file) and summarize in `README.md` §Reproduce.

**Acceptance Criteria.**
- The schedule covers all W tasks W1–W14 and references each phase's gating output.
- Each week has an explicit, checkable go/no-go condition.
- The CPU-first / GPU-deferred ordering is stated as an enforceable rule.

---

## Cross-phase dependency summary (for the editor's DEVPLAN.md index)

| This phase needs | From | Used by |
|---|---|---|
| Proposal bibliography → `paper/refs.bib` | S | W1, W3, W4 |
| `configs/hypotheses.yaml` (canonical H0–H6 + decision rules) | E5 | W2, W8, W13 |
| Region/sector/FRED configs | P, F | W5, W10 |
| Release-lag + leakage guards | N, P | W4, W5 |
| Trained runs + run manifests | M | W7, W12 |
| Metrics, DM tests, stratified plots → `experiments/<run_id>/{tables,figures}` | E5 | W7, W8 |
| Per-stage scripts (download/build_panel/run_experiment) | N, F, P, M | W11 |
| GPU/Mamba/foundation capability flags & fallbacks | S, M | W9, W15 |

---

### External facts verified (so the agent need not re-check)

- **VNP46A3** is the VIIRS/NPP monthly nighttime-lights L3 product, ~15 arc-second grid, distributed by NASA **LAADS DAAC** under **EOSDIS** open Data Use & Citation Guidance (openly shared, citation requested); cite the dataset DOI (e.g. `10.5067/VIIRS/VNP46A3.002` — confirm exact collection/version string at download time in the N task).
- **yfinance** uses unofficial Yahoo Finance endpoints; intended for **personal/research/educational** use, **not** commercial; raw data should not be redistributed — state as a research-only caveat.
- **FRED API** Terms of Use require **attribution** ("Source: \<origin\> via FRED") and prohibit redistributing third-party proprietary content — attribute each series and keep raw data in gitignored `data/`.

Sources:
- [NASA Earthdata VNP46A3 catalog](https://www.earthdata.nasa.gov/data/catalog/laads-vnp46a3-2)
- [LAADS DAAC Data Use & Citation Policies (PDF)](https://modaps.modaps.eosdis.nasa.gov/services/faq/LAADS_Data-Use_Citation_Policies.pdf)
- [yfinance on PyPI](https://pypi.org/project/yfinance/)
- [Yahoo Finance Terms (product/finance)](https://legal.yahoo.com/us/en/yahoo/terms/product-atos/finance/index.html)
- [FRED API Terms of Use](https://fred.stlouisfed.org/docs/api/terms_of_use.html)
