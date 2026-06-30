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
        self._pipe = None
        if not foundation_available(self.name):
            raise FoundationUnavailable(
                f"{self.name} ({_PKG.get(self.name)}) not installed. Chronos is CPU-installable "
                "(`pip install chronos-forecasting`); TimesFM/Moirai prefer GPU (requirements-extras.txt)."
            )

    def fit(self, train_ds=None, val_ds=None) -> FoundationForecaster:
        self.is_fit = True
        return self

    def _pipeline(self):  # pragma: no cover - requires the chronos package + a model download
        if self._pipe is None:
            from chronos import BaseChronosPipeline  # type: ignore

            model = self.config.extra.get("chronos_model", "amazon/chronos-bolt-small")
            self._pipe = BaseChronosPipeline.from_pretrained(model, device_map="cpu")
        return self._pipe

    def predict(self, dataset) -> np.ndarray:  # pragma: no cover - requires extras
        import torch

        pipe = self._pipeline()
        h = self.config.H
        preds, cache = [], {}
        # context = the target's own history up to (and including) the origin month t -> forecast t+1
        # (a leakage-safe, no-NTL time-series reference; Chronos normalizes scale internally).
        # The forecast depends only on (sector, origin) -> cache to skip redundant region anchors.
        for a in dataset.anchors:
            s = a["sector"]
            i = self._pos[pd.Timestamp(a["origin_t"])]
            key = (s, i)
            if key not in cache:
                ctx = self.target_wide[s].iloc[: i + 1].dropna().to_numpy()
                if len(ctx) < 2:
                    cache[key] = [0.0] * h
                else:
                    _, mean = pipe.predict_quantiles(
                        torch.tensor(ctx, dtype=torch.float32),
                        prediction_length=h,
                        quantile_levels=[0.5],
                    )
                    cache[key] = np.asarray(mean).reshape(-1)[:h].tolist()
            preds.append(cache[key])
        return np.asarray(preds, dtype="float64")
