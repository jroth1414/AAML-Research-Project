"""Mamba selective state-space (S6) model with capability gating + CPU fallback (Phase C / Task M7).

On a GPU profile with ``mamba-ssm`` + CUDA, the official fused kernel is used (impl='official').
Otherwise a pure-PyTorch S6 fallback runs on CPU (impl='fallback', tagged in the manifest so Phase D
reports H4 as 'support (fallback impl)' or 'deferred' — never as the official kernel). NEVER crashes
when CUDA/mamba-ssm is absent. The fallback is a faithful sequential selective scan (slow, not the
fused kernel); keep L small (<=24) so CPU runtime is bounded (Risk R14).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.capabilities import detect
from .base import ModelConfig


class _S6Fallback(nn.Module):
    """CPU-safe selective SSM (sequential scan). Not the fused mamba-ssm kernel."""

    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        d_inner = expand * d_model
        self.d_inner, self.d_state = d_inner, d_state
        self.in_proj = nn.Linear(d_model, 2 * d_inner)
        self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=4, groups=d_inner, padding=3)
        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1)
        self.dt_proj = nn.Linear(1, d_inner)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1).float()).repeat(d_inner, 1)
        )
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model)

    def forward(self, u: torch.Tensor) -> torch.Tensor:  # u: (B, L, d_model)
        b, length, _ = u.shape
        x, z = self.in_proj(u).chunk(2, dim=-1)  # each (B, L, d_inner)
        x = self.conv1d(x.transpose(1, 2))[..., :length].transpose(1, 2)
        x = F.silu(x)
        dbl = self.x_proj(x)  # (B, L, 2*d_state + 1)
        dt, b_mat, c_mat = torch.split(dbl, [1, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))  # (B, L, d_inner)
        a = -torch.exp(self.A_log)  # (d_inner, d_state)
        h = torch.zeros(b, self.d_inner, self.d_state, device=u.device, dtype=u.dtype)
        ys = []
        for t in range(length):
            da = torch.exp(dt[:, t, :].unsqueeze(-1) * a)  # (B, d_inner, d_state)
            db = dt[:, t, :].unsqueeze(-1) * b_mat[:, t, :].unsqueeze(1)
            h = da * h + db * x[:, t, :].unsqueeze(-1)
            ys.append((h * c_mat[:, t, :].unsqueeze(1)).sum(-1))  # (B, d_inner)
        y = torch.stack(ys, dim=1) + x * self.D
        return self.out_proj(y * F.silu(z))  # (B, L, d_model)


class _MambaModel(nn.Module):
    def __init__(self, L, H, d_model, d_state, expand, depth, dropout, use_official=False):
        super().__init__()
        self.embed = nn.Linear(1, d_model)
        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(depth):
            self.norms.append(nn.LayerNorm(d_model))
            if use_official:  # pragma: no cover - GPU-only path
                from mamba_ssm import Mamba

                self.blocks.append(Mamba(d_model=d_model, d_state=d_state, expand=expand))
            else:
                self.blocks.append(_S6Fallback(d_model, d_state, expand))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, H)

    def forward(self, x: torch.Tensor, var_mask=None) -> torch.Tensor:  # (B, L, 1) -> (B, H)
        h = self.embed(x)
        for norm, block in zip(self.norms, self.blocks, strict=True):
            h = h + self.dropout(block(norm(h)))
        return self.head(h[:, -1, :])  # last-step hidden -> H


def mamba_impl() -> str:
    caps = detect()
    return "official" if (caps.mamba_ssm and caps.cuda) else "fallback"


def mamba_factory(config: ModelConfig) -> nn.Module:
    e = config.extra
    impl = mamba_impl()
    config.extra["mamba_impl"] = impl  # recorded in the run manifest (Risk R6)
    return _MambaModel(
        L=config.L,
        H=config.H,
        d_model=int(e.get("d_model", 64)),
        d_state=int(e.get("d_state", 16)),
        expand=int(e.get("expand", 2)),
        depth=int(e.get("depth", 2)),
        dropout=float(e.get("dropout", 0.1)),
        use_official=(impl == "official"),
    )
