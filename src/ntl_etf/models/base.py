"""Common model interface + the prediction-artifact contract (Phase C / Task M1).

Every model implements ``BaseForecaster`` so Phase D is model-agnostic. Deep models subclass
``DeepForecaster`` (owns an ``nn.Module``, delegates fit/predict to the Trainer). ``predict`` returns
predictions in ORIGINAL (de-standardized) target units. Models NEVER fit scalers — the Trainer fits
train-only stats and the PanelDataset carries ``(y_mu, y_sigma)`` per sample for de-standardization
(the leakage guard).

The single prediction artifact is ``experiments/<run_id>/predictions.parquet`` with EXACTLY the
columns in ``PRED_COLUMNS`` (DEVPLAN Appendix A.3).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

PRED_COLUMNS = [
    "model",
    "variant",
    "pretrained",
    "task",
    "target_kind",  # 'return' (leading) | 'ip' (nowcast)
    "etf",
    "horizon",
    "fold",
    "split",
    "date",  # the TARGET month (month-end)
    "y_true",
    "y_pred",
    "seed",
]

PRETRAINED_VARIANTS = {"pretrained", "finetuned"}


@dataclass
class ModelConfig:
    name: str
    task: str = "leading"  # 'leading' | 'nowcast'
    L: int = 12
    H: int = 1
    seed: int = 1414
    variant: str = "scratch"  # scratch | pretrained | zeroshot | finetuned
    extra: dict = field(default_factory=dict)

    @property
    def target_kind(self) -> str:
        return "return" if self.task == "leading" else "ip"

    @property
    def pretrained(self) -> bool:
        return self.variant in PRETRAINED_VARIANTS


class BaseForecaster(ABC):
    """All models implement this. ``predict`` returns de-standardized target-unit predictions."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.is_fit = False

    @abstractmethod
    def fit(self, train_ds, val_ds=None) -> BaseForecaster: ...

    @abstractmethod
    def predict(self, dataset) -> np.ndarray:
        """Point predictions in ORIGINAL (de-standardized) target units, aligned to dataset order."""

    def to_rows(self, dataset, y_pred: np.ndarray, *, fold: int, split: str) -> list[dict]:
        """Build A.3 prediction rows from a dataset's anchors + aligned predictions."""
        rows = []
        for a, yp in zip(dataset.anchors, y_pred, strict=True):
            rows.append(
                {
                    "model": self.config.name,
                    "variant": self.config.variant,
                    "pretrained": self.config.pretrained,
                    "task": self.config.task,
                    "target_kind": self.config.target_kind,
                    "etf": a["sector"],
                    "horizon": self.config.H,
                    "fold": fold,
                    "split": split,
                    "date": pd.Timestamp(a["target_dates"][0]),
                    "y_true": float(np.asarray(a["y"]).ravel()[0]),
                    "y_pred": float(np.asarray(yp).ravel()[0]),
                    "seed": self.config.seed,
                }
            )
        return rows


class DeepForecaster(BaseForecaster):
    """A BaseForecaster that owns an ``nn.Module`` and delegates fit/predict to the Trainer."""

    def __init__(self, config: ModelConfig, module_factory, train_cfg=None):
        super().__init__(config)
        self.module_factory = module_factory  # callable(ModelConfig) -> nn.Module
        self.train_cfg = train_cfg
        self.module = None
        self.trainer = None
        self.history = None

    def fit(self, train_ds, val_ds=None) -> DeepForecaster:
        from ..train.trainer import TrainConfig, Trainer

        self.module = self.module_factory(self.config)
        tc = self.train_cfg or TrainConfig(seed=self.config.seed)
        self.trainer = Trainer(tc)
        self.history = self.trainer.fit(self.module, train_ds, val_ds)
        self.is_fit = True
        return self

    def predict(self, dataset) -> np.ndarray:
        return self.trainer.predict(self.module, dataset)  # (N, H) de-standardized

    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.module.parameters())) if self.module else 0


def write_predictions(rows: list[dict], run_dir: str | Path) -> Path:
    """Write the single ``predictions.parquet`` (A.3 schema) for a run."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    for c in PRED_COLUMNS:
        if c not in df.columns:
            df[c] = None
    df = df[PRED_COLUMNS]
    out = run_dir / "predictions.parquet"
    df.to_parquet(out, index=False)
    return out
