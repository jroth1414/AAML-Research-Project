"""AI4Mars dataset constants + (image, label) index builder (Phase MS1).

Label PNGs are single-channel with terrain pixel values 0-3; 255 = NULL/ignore (unlabeled,
rover-self, and >30 m range are merged into 255 in the merged-0.6 release). The merged layout is
roughly:  <root>/msl/images/edr/*.JPG  +  <root>/msl/labels/train/*.png  +
<root>/msl/labels/test/masked-gold-min*-*agree/*.png ; MER (Opportunity/Spirit) under <root>/mer/.
Exact subpaths are discovered by globbing (robust to release layout — verified at extraction).
"""

from __future__ import annotations

from pathlib import Path

CLASSES = ["soil", "bedrock", "sand", "big_rock"]
NUM_CLASSES = 4
IGNORE_INDEX = 255  # NULL / unlabeled / rover-self / >30 m range
# RGB colors for segmentation overlays (paper figures).
CLASS_COLORS = {
    0: (205, 133, 63),  # soil — tan
    1: (112, 128, 144),  # bedrock — slate
    2: (238, 214, 175),  # sand — pale
    3: (139, 0, 0),  # big rock — dark red
}
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG")


def _stem(p: Path) -> str:
    return p.stem


def find_dir(root: Path, *candidates: str) -> Path | None:
    """Return the first existing subdir among candidates, else search by name."""
    for c in candidates:
        if (root / c).is_dir():
            return root / c
    # fall back: recursive search for a dir whose path ends with the last candidate
    name = candidates[-1].split("/")[-1]
    for d in root.rglob(name):
        if d.is_dir():
            return d
    return None


def _match_labels(image_dir: Path, label_dir: Path) -> list[dict]:
    """Pair images with labels by filename stem (labels may carry a suffix)."""
    images = {}
    for ext in IMAGE_EXTS:
        for p in image_dir.glob(f"*{ext}"):
            images.setdefault(_stem(p), p)
    pairs = []
    for lab in label_dir.glob("*.png"):
        stem = _stem(lab)
        img = images.get(stem)
        if img is None:  # labels sometimes append a suffix like "_merged"/"_label"
            base = stem.split("_merged")[0].split("_label")[0]
            img = images.get(base)
        if img is not None:
            pairs.append({"image": str(img), "label": str(lab)})
    return pairs


def build_index(root: str | Path, rover: str = "msl") -> dict:
    """Index (image, label) pairs for a rover. Returns {'train': [...], 'test': [...]}.

    ``rover`` in {'msl' (Curiosity), 'mer' (Opportunity/Spirit)}.
    """
    root = Path(root)
    base = find_dir(root, "ai4mars-dataset-merged-0.6", ".") or root
    rdir = find_dir(base, rover) or (base / rover)
    img_dir = find_dir(rdir, "images/edr", "images", "edr")
    train_lab = find_dir(rdir, "labels/train", "train")
    out = {"train": [], "test": []}
    if img_dir and train_lab:
        out["train"] = _match_labels(img_dir, train_lab)
    # expert/gold test labels live under labels/test/<masked-gold-min*-*agree>/
    test_root = find_dir(rdir, "labels/test", "test")
    if img_dir and test_root:
        sub = sorted([d for d in test_root.glob("masked-gold-*") if d.is_dir()])
        chosen = sub[0] if sub else test_root  # prefer the most permissive gold set
        out["test"] = _match_labels(img_dir, chosen)
    return out
