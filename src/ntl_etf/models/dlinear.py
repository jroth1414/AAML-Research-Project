"""DLinear baseline: moving-average trend/seasonal decomposition + per-component linear maps
(Phase C / Task M3). A strong, fast CPU baseline operating on the channel-independent NTL window.
Channel-shared weights (one global model). Architecture follows Zeng et al. 2023 (clean-room).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import ModelConfig


class _MovingAvg(nn.Module):
    """Moving average with replication padding (keeps the sequence length)."""

    def __init__(self, kernel: int):
        super().__init__()
        self.kernel = kernel
        self.avg = nn.AvgPool1d(kernel_size=kernel, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, L)
        pad = (self.kernel - 1) // 2
        front = x[:, :1].repeat(1, pad)
        end = x[:, -1:].repeat(1, pad)
        xp = torch.cat([front, x, end], dim=1)
        return self.avg(xp.unsqueeze(1)).squeeze(1)


class _SeriesDecomp(nn.Module):
    def __init__(self, kernel: int):
        super().__init__()
        self.moving_avg = _MovingAvg(kernel)

    def forward(self, x: torch.Tensor):  # x: (B, L)
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class _DLinear(nn.Module):
    def __init__(self, L: int, H: int, kernel: int):
        super().__init__()
        self.decomp = _SeriesDecomp(kernel)
        self.lin_seasonal = nn.Linear(L, H)
        self.lin_trend = nn.Linear(L, H)

    def forward(self, x: torch.Tensor, var_mask=None) -> torch.Tensor:  # x: (B, L, 1) -> (B, H)
        s, t = self.decomp(x.squeeze(-1))
        return self.lin_seasonal(s) + self.lin_trend(t)


def dlinear_factory(config: ModelConfig) -> nn.Module:
    L, H = config.L, config.H
    kernel = int(config.extra.get("moving_avg", min(L, 13)))
    kernel = min(kernel, L)
    if kernel % 2 == 0:  # keep odd for symmetric replication padding
        kernel = max(1, kernel - 1)
    return _DLinear(L, H, kernel)
