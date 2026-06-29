# AAML Research Project — Nighttime Lights as an Economic Proxy for Sector ETF Forecasting

**Course:** JHU EN.705.742 — Advanced Applied Machine Learning · **Author:** John Roth

This repository hosts a graduate research project investigating whether monthly **NASA VIIRS Black Marble nighttime-light (NTL)** intensity over economically-linked world regions can forecast:

- **Leading task** — forward 1–3 month log returns of the 11 SPDR sector ETFs (XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV, XLY), and
- **Coincident nowcast task** — contemporaneous sector industrial production (FRED).

It compares three sequence architectures with different inductive biases — **PatchTST** (temporal patch attention), **iTransformer** (cross-region/variate attention), and **Mamba** (selective state-space) — against **12-month momentum** and **DLinear** baselines, with masked self-supervised pretraining and time-series foundation-model initialization (Chronos / Moirai / TimesFM), all under leakage-safe walk-forward cross-validation and pre-registered Diebold–Mariano significance testing.

## Current status: development plan

The repository is at the **planning stage**. The complete, executable build plan lives here:

- **[`DEVPLAN.md`](DEVPLAN.md)** — master plan: guiding principles, canonical repo structure, phase/milestone overview, dependency graph, consolidated risk register, hypothesis traceability (H0–H6), rubric mapping, week-by-week schedule, and the **normative Data & Artifact Contracts (Appendix A)**. Start here.
- **[`docs/devplan/`](docs/devplan/)** — the detailed, task-by-task per-phase plans:
  - [`00-overview-and-setup.md`](docs/devplan/00-overview-and-setup.md) — environment, reproducibility, tooling, git workflow (tasks `S*`)
  - [`01-data-ntl.md`](docs/devplan/01-data-ntl.md) — NASA Black Marble VNP46A3 download + raster feature extraction (`N*`)
  - [`02-data-financial-macro.md`](docs/devplan/02-data-financial-macro.md) — yfinance ETF returns, FRED industrial production, VIX (`F*`)
  - [`03-panel-and-splits.md`](docs/devplan/03-panel-and-splits.md) — global panel, windowing, walk-forward splits, leakage audit (`P*`)
  - [`04-models-and-training.md`](docs/devplan/04-models-and-training.md) — baselines + 3 deep models + transfer learning + training harness (`M*`)
  - [`05-evaluation-and-stats.md`](docs/devplan/05-evaluation-and-stats.md) — metrics, significance testing, stratified hypothesis analysis (`E*`)
  - [`06-paper-and-deliverables.md`](docs/devplan/06-paper-and-deliverables.md) — paper, rubric mapping, reproducibility, deliverables (`W*`)
  - [`REVIEW.md`](docs/devplan/REVIEW.md) — provenance record of the adversarial review used to harden the plan.
- **[`RESEARCH.MD`](RESEARCH.MD)** — the course research-process checklist and grading rubric the final deliverable must satisfy.

The plan is written for an autonomous coding agent to execute task-by-task, with explicit acceptance criteria gating each step. Setup, build, and reproduction instructions will be filled in here as the implementation is built out per Phase 0 (`S*`) and Phase E (`W9`).

## Research hypotheses (summary)

| | Hypothesis |
|---|---|
| **H0** (null) | No deep model beats the 12-month momentum baseline on forward returns. |
| **H1** | At least one deep model beats both baselines on 1-month return prediction (MSE + directional accuracy). |
| **H2** | iTransformer > PatchTST on multi-region ETFs (e.g. XLI: Pearl River + Yangtze deltas). |
| **H3** | PatchTST > iTransformer on single-dominant-region ETFs (e.g. XLE: Permian Basin). |
| **H4** | Mamba ≥ both Transformers during disruption months (mean VIX > 25). |
| **H5** | NTL nowcasts contemporaneous industrial production far better than it leads returns. |
| **H6** | Pretrained / foundation-initialized models beat identically-sized from-scratch models. |
