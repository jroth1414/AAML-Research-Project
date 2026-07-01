# Development Log — marsseg (Mars terrain segmentation)

Running, dated trail of decisions, problems + fixes, and verification evidence — the raw material for
the paper's methodology / limitations / reproducibility sections. Newest entry at the bottom.

---

## MS0 — Project scaffold · branch `mars-terrain-segmentation`

Stood up the `marsseg` package and the reproducibility backbone: config-driven runs, run manifests,
a long/tidy results store, deterministic seeding, capability detection, logging, CI, and a
`check_env.py` gate. CV stack installs clean on Windows/CPU (torch 2.4.1+cpu, torchvision 0.19.1+cpu,
`segmentation-models-pytorch` 0.5.0, transformers, albumentations, opencv, scikit-image);
`check_env.py` → `profile=windows_cpu`, `core OK`.

**Compute split.** Segmentation training is GPU-bound. The full pipeline + smoke tests run CPU-only;
full model training targets the V100 (16 GB) via `scripts/run_gpu.sh`. The local RTX 5070 Ti
(Blackwell) is left unused for training — its bleeding-edge toolchain isn't worth fighting when a
well-supported V100 is available.

## MS1 — AI4Mars data pipeline

Built the AI4Mars acquisition + dataset layer: a resumable, MD5-verified Zenodo downloader
(`scripts/download_data.py`); an index builder over the merged-0.6 layout
(`msl/images/edr` + `msl/labels/{train,test}`, MER for the cross-rover test); a `SegDataset` that
pairs each image with its label map, replicates grayscale to 3 channels, and preserves the **255 =
ignore** label (unlabeled / rover-self / >30 m range) through albumentations transforms; **by-image**
train/val splits (a frame never crosses splits); and per-class pixel counts for the class-weighted
loss. Offline fixture tests cover the index, the dataset item + ignore handling, split disjointness,
and class counts.

**Verified facts (AI4Mars merged-0.6, Zenodo).** Labels are pixel values **0–3**
(soil/bedrock/sand/big_rock), **255 = NULL/ignore**; MSL (Curiosity) ≈ 16,064 train + 322
expert-labeled test images; MER (Opportunity/Spirit) supports the cross-rover test. The 16 GB
dataset is downloaded to `data/raw/ai4mars` (gitignored).

**Next:** MS2 — model zoo (baseline U-Net; smp U-Net / DeepLabV3+ with ImageNet encoders; SegFormer;
DINOv3-SAT + SAM foundation references) + the segmentation Trainer (class-weighted CE + Dice,
`ignore_index`, AMP on GPU).

## DEVPLAN hardening — adversarial cold-start pass

Rewrote `DEVPLAN.md` to be **adversarially hardened**: self-contained and mechanically executable so any
agent with no prior context can reach the same results and the same H0–H5 verdicts. Method was an
11-agent red-team fan-out — six adversarial lenses (structure/contract drift, data/artifact contracts,
environment & reproducibility, protocol ambiguity, guardrails, gate executability) → synthesis → three
**cold-start verifier** agents that tried to execute the rewrite blind → reconcile. It surfaced 71
findings (35 blocker) and 22 residual cold-start blockers, all folded in. Every load-bearing claim was
re-verified by hand against the repo (RESEARCH.MD SHA-256, image/label counts, `seed.py` API, the
results-store/manifest schemas, pyproject markers).

The rewrite pins every load-bearing constant and adds sections the plan lacked: a cold-start status +
resume block, non-negotiable operating rules (no-AI-author + commit-msg hook, RESEARCH.MD SHA, `.env`,
gitignore policy, seed=1414, per-task-commit/per-phase-branch/pause-at-gate), exact env/setup + V100
handoff, the verified on-disk data layout, frozen protocol constants, the single-source-of-truth
paired-bootstrap spec (§5.6) + deterministic H4 rule (§5.7) + H5 partial-GPU rule (§5.8) +
canonical-run selection (§5.9), the artifact contracts (results store, manifest, and a **newly-defined
segmentation predictions contract**), a corrected repo tree, per-phase gates as runnable commands, and
a hypothesis→evidence decision table.

**Defect discovered (MS1 REOPENED).** The adversarial pass found a real, silent bug the synthetic-fixture
unit tests never exercised: `data/ai4mars.py::build_index` returns an **EMPTY MSL training index** and an
**empty MER index** against the actual nested `merged-0.6` layout. Verified empirically:
`build_index('data/raw/ai4mars')` → MSL `train=0, test=322`; `rover='mer'` → `train=0, test=0`. Root
causes: (A) MSL images are under `msl/ncam/images/edr` and labels under `msl/ncam/labels/train`, but
`find_dir`'s `rglob` fallback binds "train" to the wrong camera (`msl/mcam/labels/train`, since `mcam`
sorts before `ncam`) → ncam images paired against mcam label stems → 0 matches; (B) MER JPGs live under
`mer/images/{eff,test}` (not directly in `mer/images/`) and gold labels carry `_<digits>_T<digits>_merged`
suffixes the stem normalizer doesn't strip. Also: `build_index` records carry no `name`, so the
paired-bootstrap cross-model join key is unusable, and the gold-dir picker hard-selects `min1` while the
protocol pins `min3` (counts coincide at 322/204, so a green count doesn't prove the right set). Exact
fixes are specified in DEVPLAN §4.1/§4.3/§5.4 as hard prerequisites of the first training run.

**Next:** MS2 — build `models/foundation.py`, then MS3 (`configs/*`, `eval/*`, `run_experiment.py`,
`analyze_results.py`, `hypotheses.yaml`, `PREREG.md`) — but first apply the §4.1/§4.3 `build_index`
fixes and re-verify the count assertions (16064 / 322 / 204, MER train empty).
