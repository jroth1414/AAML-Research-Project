"""PatchTST: channel-independent patch attention (Phase C / Tasks M4-M5).

Clean-room minimal implementation of the PatchTST architecture (Nie et al. 2023): the univariate
NTL lookback is split into patches, each patch is linearly embedded into a token, a Transformer
encoder attends over the patch tokens, and a linear head maps the flattened patch representations to
the H-step forecast. Channel independence = one shared-weight model over all series (the series
dimension is folded into the batch).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import ModelConfig


class _PatchTST(nn.Module):
    def __init__(self, L, H, patch_len, stride, d_model, n_heads, e_layers, d_ff, dropout):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = max(1, (L - patch_len) // stride + 1)
        self.embed = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=e_layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.num_patches * d_model, H)

    def forward(self, x: torch.Tensor, var_mask=None) -> torch.Tensor:  # x: (B, L, 1) -> (B, H)
        x = x.squeeze(-1)  # (B, L)
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)  # (B, P, patch_len)
        tok = self.embed(patches) + self.pos[:, : patches.size(1)]  # (B, P, d_model)
        z = self.encoder(self.dropout(tok))  # (B, P, d_model)
        return self.head(z.reshape(z.size(0), -1))  # (B, H)


def patchtst_factory(config: ModelConfig) -> nn.Module:
    e = config.extra
    L, H = config.L, config.H
    patch_len = int(e.get("patch_len", min(6, L)))
    patch_len = min(patch_len, L)
    stride = int(e.get("stride", max(1, patch_len // 2)))
    return _PatchTST(
        L=L,
        H=H,
        patch_len=patch_len,
        stride=stride,
        d_model=int(e.get("d_model", 64)),
        n_heads=int(e.get("n_heads", 4)),
        e_layers=int(e.get("e_layers", 2)),
        d_ff=int(e.get("d_ff", 128)),
        dropout=float(e.get("dropout", 0.2)),
    )
