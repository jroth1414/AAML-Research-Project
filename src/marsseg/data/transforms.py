"""Albumentations transforms for AI4Mars segmentation (Phase MS1).

Geometric ops apply nearest-neighbour to the mask automatically (labels are not interpolated).
Augmentation preserves the horizon (horizontal flip + photometric only — NO vertical flip).
Grayscale images are replicated to 3 channels and ImageNet-normalized for pretrained encoders.
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def train_transform(size: int = 512) -> A.Compose:
    return A.Compose(
        [
            A.Resize(size, size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def eval_transform(size: int = 512) -> A.Compose:
    return A.Compose(
        [
            A.Resize(size, size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
