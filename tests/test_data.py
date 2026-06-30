"""Offline AI4Mars data-pipeline tests on a synthetic fixture tree (Phase MS1). No network."""

from __future__ import annotations

import cv2
import numpy as np

from marsseg.data import ai4mars
from marsseg.data.dataset import SegDataset, class_pixel_counts, make_splits
from marsseg.data.transforms import eval_transform


def _make_tree(root, n=4, size=32):
    img_dir = root / "ai4mars-dataset-merged-0.6" / "msl" / "images" / "edr"
    train_dir = root / "ai4mars-dataset-merged-0.6" / "msl" / "labels" / "train"
    test_dir = (
        root
        / "ai4mars-dataset-merged-0.6"
        / "msl"
        / "labels"
        / "test"
        / "masked-gold-min1-100agree"
    )
    for d in (img_dir, train_dir, test_dir):
        d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        cv2.imwrite(str(img_dir / f"img{i}.JPG"), rng.integers(0, 255, (size, size), dtype="uint8"))
        m = rng.integers(0, 4, (size, size), dtype="uint8")
        m[0, 0] = 255  # an ignore pixel
        cv2.imwrite(str(train_dir / f"img{i}.png"), m)
    cv2.imwrite(str(test_dir / "img0.png"), rng.integers(0, 4, (size, size), dtype="uint8"))
    return root


def test_build_index(tmp_path):
    root = _make_tree(tmp_path)
    idx = ai4mars.build_index(root, "msl")
    assert len(idx["train"]) == 4
    assert len(idx["test"]) == 1
    assert all("image" in r and "label" in r for r in idx["train"])


def test_dataset_item_and_ignore(tmp_path):
    root = _make_tree(tmp_path)
    idx = ai4mars.build_index(root, "msl")
    ds = SegDataset(idx["train"], eval_transform(size=32))
    item = ds[0]
    assert tuple(item["image"].shape) == (3, 32, 32)
    assert tuple(item["mask"].shape) == (32, 32)
    assert str(item["mask"].dtype) == "torch.int64"
    vals = set(item["mask"].unique().tolist())
    assert vals <= {0, 1, 2, 3, ai4mars.IGNORE_INDEX}
    assert ai4mars.IGNORE_INDEX in vals  # the ignore pixel survived the transform


def test_splits_disjoint_by_image(tmp_path):
    root = _make_tree(tmp_path, n=10)
    idx = ai4mars.build_index(root, "msl")
    sp = make_splits(idx["train"], val_frac=0.3, seed=1414)
    train_imgs = {r["image"] for r in sp["train"]}
    val_imgs = {r["image"] for r in sp["val"]}
    assert train_imgs.isdisjoint(val_imgs)
    assert len(sp["train"]) + len(sp["val"]) == len(idx["train"])


def test_class_pixel_counts(tmp_path):
    root = _make_tree(tmp_path)
    idx = ai4mars.build_index(root, "msl")
    counts = class_pixel_counts(idx["train"], num_classes=4)
    assert counts.sum() > 0 and len(counts) == 4
