"""12-month time-series momentum baseline = the H0/H1 yardstick (Phase C / Task M2).

Prediction = trailing ``lookback``-month MEAN of the target's own realized values ending at the
lookback origin (a random-walk-in-drift forecast). Uses ONLY information available at month t, so
it is leakage-safe. No parameters, no scaling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseForecaster, ModelConfig


class MomentumForecaster(BaseForecaster):
    def __init__(self, config: ModelConfig, target_wide: pd.DataFrame, lookback: int = 12):
        super().__init__(config)
        self.target_wide = target_wide  # returns_wide (leading) or ip_wide (nowcast)
        self.lookback = lookback
        self._pos = {d: i for i, d in enumerate(target_wide.index)}

    def fit(self, train_ds=None, val_ds=None) -> MomentumForecaster:
        self.is_fit = True
        return self

    def predict(self, dataset) -> np.ndarray:
        preds = []
        for a in dataset.anchors:
            t = pd.Timestamp(a["origin_t"])
            s = a["sector"]
            pred = 0.0
            if s in self.target_wide.columns and t in self._pos:
                i = self._pos[t]
                win = self.target_wide[s].iloc[max(0, i - self.lookback + 1) : i + 1].dropna()
                pred = float(win.mean()) if len(win) else 0.0
            preds.append([pred] * self.config.H)
        return np.asarray(preds, dtype="float64")
