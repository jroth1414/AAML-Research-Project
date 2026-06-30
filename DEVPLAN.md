# DEVPLAN — Terrain-Aware Semantic Segmentation for Mars Rover Drivability (AI4Mars)

**Master development plan for the `marsseg` project (JHU EN.705.742, Advanced Applied ML).**
This project builds and rigorously evaluates deep semantic-segmentation models that label Martian
terrain — *soil, bedrock, sand, big rock* — from rover camera images, the core perception task for
**autonomous drivability** assessment. We compare CNN (U-Net, DeepLabV3+) vs transformer (SegFormer)
architectures, measure the value of **ImageNet transfer**, test **cross-rover generalization**
(Curiosity → Opportunity/Spirit), and benchmark a **foundation-model reference** (SAM / DINOv3
pretrained on satellite imagery), all under a leakage-safe, pre-registered, significance-tested
protocol.

---

## 1. Why this project (and why it will yield a positive result)

Terrain class is **directly observable in the pixels**, so deep segmentation models *will* beat
naive baselines — a meaningful positive result is expected. The interesting questions are not
*whether* ML works but **what drives performance** — architecture family, transfer, and robustness to
domain shift — and whether a model trained on one rover **generalizes** to another. This is a
positive, visual, societally-relevant study (it underpins real Mars-rover autonomy).

## 2. Research question & ML problem

**RQ.** For Mars terrain segmentation, what drives accuracy — architecture (CNN vs transformer),
encoder transfer (ImageNet vs from-scratch), label-noise handling — and does a model generalize
**across rovers/cameras**?

**ML problem.** Multi-class **semantic segmentation**: map a grayscale Navcam image
`x ∈ ℝ^{H×W}` to a per-pixel label `y ∈ {soil, bedrock, sand, big_rock}^{H×W}`, with an **ignore
class** for unlabeled / rover-self / out-of-range (>30 m) pixels excluded from loss and metrics. Loss
= class-weighted cross-entropy + Dice on valid pixels; primary metric = **mean IoU**.

## 3. Hypotheses (frozen in `configs/hypotheses.yaml`; significance at α = 0.10, Holm-corrected)

| ID | Statement | Decided by |
|---|---|---|
| **H0** | No deep model significantly beats a simple baseline at mIoU. | reported honestly if H1 fails |
| **H1** | ≥1 deep model (U-Net / DeepLab / SegFormer) beats the baseline on mIoU (paired-bootstrap significant). | Family A |
| **H2** | ImageNet-pretrained encoder > identical from-scratch (transfer helps). | Family B (paired) |
| **H3** | SegFormer (transformer) vs U-Net/DeepLab (CNN): which wins overall and **per class** (hypothesis: transformer favors large homogeneous soil/sand; CNN favors small `big_rock` boundaries). | Family C + per-class strata |
| **H4** | A model generalizes across rovers: train Curiosity (MSL), test Opportunity/Spirit (MER) with a bounded mIoU drop; pretraining/augmentation narrow the gap. | Family D (cross-rover stratum) |
| **H5** | A foundation model is a useful reference vs from-scratch: **DINOv3 ViT-L pretrained on Earth satellite imagery (SAT-493M)** as a frozen backbone + trained head, and SAM zero-shot. Tests cross-planet/cross-viewpoint transfer. | Family E (gated → V100) |

## 4. Method

- **Baseline:** a tiny from-scratch U-Net (few filters) — the H0/H1 yardstick.
- **CNN:** `segmentation-models-pytorch` U-Net + DeepLabV3+ with ResNet/EfficientNet encoders
  (ImageNet vs random init for H2).
- **Transformer:** SegFormer (MiT-B0/B2 via `transformers`).
- **Foundation (H5, gated):** **DINOv3 ViT-L/16 pretrained on SAT-493M** (Earth-satellite RGB; gated
  HF weights `facebook/dinov3-vitl16-pretrain-sat493m`, needs an HF token + license accept) as a
  frozen backbone with a trained decoder head; plus SAM zero-shot mask proposals mapped to terrain.
  Loaded via `transformers`; skip-and-log if the weights/GPU are absent. ViT-7B SAT is too large for
  the V100 (16 GB), so the distilled ViT-L variant is the target.
- **Trainer:** one config-driven loop — seeding + determinism, class-weighted CE + Dice with
  `ignore_index`, AdamW + cosine LR, early stopping on val mIoU, AMP on CUDA, checkpointing,
  manifest. Grayscale→3-channel for ImageNet encoders.

## 5. Data & protocol (AI4Mars)

- **Source.** NASA AI4Mars (open): ~35K rover images, ~326K crowdsourced labels (each image ~10
  annotators). MSL (Curiosity) subset ≈ 16,064 train + 322 **expert-labeled** test; MER
  (Opportunity/Spirit) subset for the cross-rover test (H4). 4 terrain classes + ignore masks.
- **Leakage guards.** Split **by image** (never patch-leak a frame across train/val);
  the official **expert test set** is the held-out evaluation (no peeking); **ignore masks**
  (NULL/unlabeled, rover-self, >30 m range) are excluded from both loss and metrics; train-only
  computation of any normalization stats; every run records seed + git SHA + config hash.
- **Imbalance.** Soil dominates → class-weighted loss + report per-class IoU (not just mIoU).
- **Augmentation.** Horizontal flip + photometric jitter (preserve the horizon → no vertical flip);
  small scale/crop. Verified not to cross the ignore mask.

## 6. Metrics & significance

mean IoU (primary), per-class IoU, pixel accuracy, boundary-F1. Dispersion = bootstrap CI over test
**images**. Model-vs-model significance = **paired bootstrap** over images (and **McNemar** on
per-pixel correctness for a secondary view), **Holm-corrected within each family**. Wins are claimed
only when significant — H0 reported honestly otherwise.

## 7. Canonical repo structure

```
src/marsseg/
  data/    ai4mars.py (download+index)  dataset.py (SegDataset, masks, splits)  transforms.py
  models/  base.py  unet.py  smp_models.py  segformer.py  foundation.py  registry.py
  train/   trainer.py
  eval/    metrics.py  stats.py  prereg.py  aggregate.py  verdict.py  plots.py
  utils/   seed.py config.py manifest.py results.py logging.py capabilities.py  (REUSED)
configs/   data.yaml  models/*.yaml  hypotheses.yaml
scripts/   download_data.py  run_experiment.py  analyze_results.py  run_gpu.sh  check_env.py
tests/  paper/  docs/DEVLOG.md  experiments/ (gitignored except manifests + results store)
```

## 8. Phases & gates

| Phase | Tasks | Gate |
|---|---|---|
| **MS0 — setup** | restructure, CV deps, utils adapt, DEVPLAN/README/CI, import smoke | `pytest`/`ruff`/`black` green CPU; `check_env` prints profile; import OK |
| **MS1 — data** | AI4Mars download + index, `SegDataset` (masks, by-image splits), offline fixture tests | dataset yields valid (image, mask) with ignore handled; splits disjoint by image |
| **MS2 — models** | baseline + U-Net/DeepLab/SegFormer + foundation wrappers + Trainer | shape/overfit/determinism tests pass CPU; contract-valid predictions |
| **MS3 — eval** | metrics + paired significance + strata + `hypotheses.yaml` + verdict + analyze | end-to-end on a fixture run; honest H0/deferred verdicts |
| **MS4 — runs** | CPU smoke (subset) green; **V100** full training + cross-rover + foundation; merge-back | smoke contract-valid; `run_gpu.sh` documented; gate green |
| **MS5 — paper** | rubric-aligned paper + overlay figures + results binding + deliverables checklist | `main.pdf` builds; all H1–H5 verdicts + explicit H0; RESEARCH.MD unchanged |

## 9. Rubric mapping (RESEARCH.MD)

INTRO → §1–2 (topic, drivability goal, segmentation ML problem). HYPOTHESES & METHOD → §3–4 +
block diagram (encoders/decoders, loss, ignore mask). RESEARCH → related work (AI4Mars, MarsSeg,
U-Net, DeepLab, SegFormer, SAM/DINOv2). APPLICATION → mIoU leaderboard, per-class + cross-rover,
transfer ablation, paired-significance tables, augmentation/HP study. WHAT IS LEARNED → which
architecture wins where, the value of transfer, the cross-rover gap, and drivability implications.

## 10. Compute

CPU (now): full pipeline + tiny smoke (subset of images, few steps). **V100 (incoming):** full
training of all architectures, cross-rover, and the SAM/DINOv2 foundation reference — `run_gpu.sh`
makes it a one-command job whose outputs merge back into the canonical store and re-decide the
gated hypotheses.
