"""iTransformer: cross-variate (cross-region) attention (Phase C / Tasks M4, M6).

Clean-room minimal implementation of the iTransformer idea (Liu et al. 2024): each variate's full
lookback series is embedded into ONE token, and attention operates ACROSS variates (regions) rather
than across time — the architectural lever for H2/H3. Adaptation for this project: the variates are
NTL region series and the prediction target is the (exogenous) sector ETF return, so after
cross-region attention we masked-mean-pool the valid variate tokens (``var_mask``) and apply an MLP
head to produce the H-step return forecast. Padded/invalid variates are excluded from both attention
and the pool (so a target never attends to a region it was not paired with — leakage guard).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import ModelConfig


class _ITransformer(nn.Module):
    def __init__(self, L, H, d_model, n_heads, e_layers, d_ff, dropout):
        super().__init__()
        self.embed = nn.Linear(L, d_model)  # each variate's L-series -> one token
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=e_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, H)
        )

    def forward(self, x: torch.Tensor, var_mask=None) -> torch.Tensor:  # x: (B, L, V) -> (B, H)
        tok = self.embed(x.transpose(1, 2))  # (B, V, d_model) — one token per variate
        if var_mask is not None:
            pad = ~var_mask  # True where invalid -> excluded from attention
            z = self.encoder(tok, src_key_padding_mask=pad)
            m = var_mask.unsqueeze(-1).float()  # (B, V, 1)
            pooled = (z * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)  # masked mean over variates
        else:
            z = self.encoder(tok)
            pooled = z.mean(dim=1)
        return self.head(pooled)  # (B, H)


def itransformer_factory(config: ModelConfig) -> nn.Module:
    e = config.extra
    return _ITransformer(
        L=config.L,
        H=config.H,
        d_model=int(e.get("d_model", 64)),
        n_heads=int(e.get("n_heads", 4)),
        e_layers=int(e.get("e_layers", 2)),
        d_ff=int(e.get("d_ff", 128)),
        dropout=float(e.get("dropout", 0.2)),
    )
