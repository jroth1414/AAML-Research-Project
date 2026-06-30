"""Time-series foundation models: zero-shot CPU / fine-tune GPU (Phase C / Task M10).

Foundation models (Chronos / Moirai / TimesFM) take the univariate TARGET history as context (NOT
the NTL features) — a deliberate "no-NTL" reference for the H6b discussion. They are gated behind
the capability flags: if the package is unavailable (the default on the Windows/CPU profile, where
they live in requirements-extras.txt), the forecaster raises ``FoundationUnavailable`` so the runner
writes a ``skipped.json`` and continues — never crashing the suite (the skip-and-log contract).
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd

from .base import BaseForecaster, ModelConfig

_PKG = {"chronos": "chronos", "moirai": "uni2ts", "timesfm": "timesfm"}


class FoundationUnavailable(RuntimeError):
    """Raised when a foundation model's package/hardware is unavailable (skip-and-log)."""


def foundation_available(name: str) -> bool:
    pkg = _PKG.get(name, name)
    return importlib.util.find_spec(pkg) is not None


class FoundationForecaster(BaseForecaster):
    """Zero-shot wrapper over the target's own history. Gated; skips cleanly when unavailable."""

    def __init__(self, config: ModelConfig, target_wide: pd.DataFrame):
        super().__init__(config)
        self.name = config.extra.get("foundation", config.name)
        self.target_wide = target_wide
        self._pos = {d: i for i, d in enumerate(target_wide.index)}
        if not foundation_available(self.name):
            raise FoundationUnavailable(
                f"{self.name} ({_PKG.get(self.name)}) not installed; install requirements-extras.txt "
                "on a GPU/Linux profile. Skipping (skip-and-log)."
            )

    def fit(self, train_ds=None, val_ds=None) -> FoundationForecaster:
        self.is_fit = True
        return self

    def predict(self, dataset) -> np.ndarray:  # pragma: no cover - requires extras
        import torch
        from chronos import ChronosPipeline  # type: ignore

        pipe = ChronosPipeline.from_pretrained("amazon/chronos-bolt-small", device_map="cpu")
        preds = []
        for a in dataset.anchors:
            t = pd.Timestamp(a["origin_t"])
            s = a["sector"]
            i = self._pos[t]
            ctx = self.target_wide[s].iloc[: i + 1].dropna().to_numpy()
            fc = pipe.predict(torch.tensor(ctx).unsqueeze(0), prediction_length=self.config.H)
            preds.append(np.asarray(fc).reshape(-1)[: self.config.H])
        return np.asarray(preds, dtype="float64")
