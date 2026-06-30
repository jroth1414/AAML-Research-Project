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
