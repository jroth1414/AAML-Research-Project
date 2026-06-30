"""Masked self-supervised pretraining on the unlabeled NTL corpus (Phase C / Task M8).

PatchTST-style masked-patch reconstruction over ALL region NTL series (no labels). The pretrained
embed+encoder weights transfer to the supervised PatchTST; the forecasting head is reinitialized, so
the `pretrained` and `scratch` variants have IDENTICAL architecture/param count — a fair H6a test.

Leakage stance (documented, Risk R8): pretraining uses the full unlabeled NTL history (a standard,
no-labels relaxation). Fine-tuning + evaluation still obey the per-fold train-only fit and the
VNP46A3 release-lag alignment.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .base import ModelConfig
from .patchtst import _PatchTST, patchtst_factory


class _MaskedPatchTST(nn.Module):
    """Reuses the supervised PatchTST embed+encoder; adds a per-patch reconstruction head."""

    def __init__(self, L, patch_len, stride, d_model, n_heads, e_layers, d_ff, dropout, mask_ratio):
        super().__init__()
        self.patch_len, self.stride, self.mask_ratio = patch_len, stride, mask_ratio
        self.num_patches = max(1, (L - patch_len) // stride + 1)
        self.embed = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=e_layers)
        self.mask_token = nn.Parameter(torch.zeros(d_model))
        self.recon = nn.Linear(d_model, patch_len)

    def forward(self, x):  # x: (B, L, 1)
        patches = x.squeeze(-1).unfold(1, self.patch_len, self.stride)  # (B, P, patch_len)
        tok = self.embed(patches) + self.pos[:, : patches.size(1)]
        mask = (torch.rand(tok.shape[:2], device=x.device) < self.mask_ratio).unsqueeze(-1)
        tok_in = torch.where(mask, self.mask_token, tok)
        recon = self.recon(self.encoder(tok_in))  # (B, P, patch_len)
        return recon, patches, mask.squeeze(-1)


def build_unlabeled_windows(ntl_wide, L: int) -> np.ndarray:
    """All (region, origin) NTL lookback windows, standardized per region on the full history."""
    cols = []
    for r in ntl_wide.columns:
        s = ntl_wide[r].to_numpy("float64")
        mu, sd = np.nanmean(s), np.nanstd(s)
        sd = sd if sd > 1e-8 else 1.0
        for i in range(L - 1, len(s)):
            w = s[i - L + 1 : i + 1]
            if np.isfinite(w).all():
                cols.append(((w - mu) / sd).astype("float32"))
    return np.asarray(cols, dtype="float32")[:, :, None]  # (N, L, 1)


def pretrain_patchtst(
    ntl_wide,
    config: ModelConfig,
    out_ckpt: str,
    *,
    steps: int = 800,
    batch_size: int = 64,
    lr: float = 1e-3,
    mask_ratio: float = 0.4,
    seed: int = 1414,
) -> dict:
    """Pretrain the masked PatchTST and save the transferable embed+encoder+pos weights."""
    from ..utils.seed import set_seed

    set_seed(seed)
    e = config.extra
    patch_len = min(int(e.get("patch_len", 6)), config.L)
    stride = int(e.get("stride", max(1, patch_len // 2)))
    model = _MaskedPatchTST(
        L=config.L,
        patch_len=patch_len,
        stride=stride,
        d_model=int(e.get("d_model", 64)),
        n_heads=int(e.get("n_heads", 4)),
        e_layers=int(e.get("e_layers", 2)),
        d_ff=int(e.get("d_ff", 128)),
        dropout=float(e.get("dropout", 0.2)),
        mask_ratio=mask_ratio,
    )
    X = torch.tensor(build_unlabeled_windows(ntl_wide, config.L))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    history = []
    model.train()
    for step in range(steps):
        idx = torch.randint(0, len(X), (min(batch_size, len(X)),), generator=g)
        recon, patches, mask = model(X[idx])
        m = mask.unsqueeze(-1).float()
        loss = ((recon - patches) ** 2 * m).sum() / m.sum().clamp(min=1.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            history.append(float(loss.item()))
    Path(out_ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "embed": model.embed.state_dict(),
            "pos": model.pos.detach(),
            "encoder": model.encoder.state_dict(),
        },
        out_ckpt,
    )
    return {"recon_loss": history, "n_windows": len(X), "ckpt": out_ckpt}


def pretrained_patchtst_factory(config: ModelConfig) -> nn.Module:
    """Build a supervised PatchTST and load the pretrained embed/pos/encoder (head stays random)."""
    module: _PatchTST = patchtst_factory(config)
    ckpt = config.extra.get("pretrained_ckpt")
    if ckpt and Path(ckpt).exists():
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        try:
            module.embed.load_state_dict(state["embed"])
            module.encoder.load_state_dict(state["encoder"])
            with torch.no_grad():
                if module.pos.shape == state["pos"].shape:
                    module.pos.copy_(state["pos"])
        except (RuntimeError, KeyError):
            pass  # shape mismatch -> fall back to scratch init (logged by caller)
    return module
