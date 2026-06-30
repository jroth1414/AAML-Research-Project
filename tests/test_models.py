"""Model zoo + loss + Lightning smoke tests (MS2). CPU, no network (scratch variants only)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from marsseg.models.zoo import build_model
from marsseg.train.loss import CombinedLoss


@pytest.mark.parametrize(
    "name,kw",
    [
        ("baseline", {}),
        ("unet", {"pretrained": False}),
        ("deeplabv3plus", {"pretrained": False}),
        ("segformer", {"backbone": "b0", "pretrained": False}),
    ],
)
def test_zoo_forward_shapes(name, kw):
    m = build_model(name, num_classes=4, **kw)
    out = m(torch.randn(2, 3, 64, 64))
    assert tuple(out.shape) == (2, 4, 64, 64)


def test_unknown_model():
    with pytest.raises(ValueError):
        build_model("nope")


def test_combined_loss_ignore():
    logits = torch.randn(2, 4, 32, 32, requires_grad=True)
    target = torch.randint(0, 4, (2, 32, 32))
    target[:, 0, :] = 255  # an ignore row
    loss = CombinedLoss(class_weights=[1, 2, 3, 4], ignore_index=255)(logits, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None


def _make_records(root, n=4, size=64):
    img_dir, lab_dir = root / "img", root / "lab"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    recs = []
    for i in range(n):
        ip, lp = img_dir / f"{i}.png", lab_dir / f"{i}.png"
        cv2.imwrite(str(ip), rng.integers(0, 255, (size, size), dtype="uint8"))
        m = rng.integers(0, 4, (size, size), dtype="uint8")
        m[0, 0] = 255
        cv2.imwrite(str(lp), m)
        recs.append({"image": str(ip), "label": str(lp), "name": str(i)})
    return recs


def test_lightning_fast_dev_run(tmp_path):
    import lightning as L

    from marsseg.train.lit import SegDataModule, SegLitModule

    recs = _make_records(tmp_path, n=4, size=64)
    dm = SegDataModule(recs[:3], recs[3:], batch_size=2, size=64)
    model = SegLitModule(model_name="baseline", max_epochs=1, pretrained=False)
    trainer = L.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(model, dm)  # 1 train + 1 val batch; must complete without error
