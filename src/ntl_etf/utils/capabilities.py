"""Capability detection: let the pipeline discover its own hardware/library profile.

The whole pipeline runs CPU-only by default. Mamba (official ``mamba-ssm`` fused kernel)
and foundation-model fine-tuning are gated behind these flags and degrade gracefully
(skip-and-log) on a profile that lacks them — they never hard-crash the run.

``mamba-ssm`` kernels are CUDA-only, so ``has_mamba_ssm`` requires BOTH the package to be
importable AND CUDA to be present.
"""

from __future__ import annotations

import importlib.util
from dataclasses import asdict, dataclass


def _can_import(name: str) -> bool:
    """True if ``name`` is importable without actually importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def has_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def has_mamba_ssm() -> bool:
    # importable AND CUDA present (mamba-ssm kernels are CUDA-only)
    return _can_import("mamba_ssm") and has_cuda()


def has_foundation(pkg: str) -> bool:
    """pkg in {"chronos", "uni2ts", "timesfm"}."""
    return _can_import(pkg)


@dataclass(frozen=True)
class Capabilities:
    cuda: bool
    mamba_ssm: bool
    chronos: bool
    uni2ts: bool
    timesfm: bool
    profile: str  # "windows_cpu" | "gpu_full" | "unknown"


def detect() -> Capabilities:
    cuda = has_cuda()
    caps = dict(
        cuda=cuda,
        mamba_ssm=has_mamba_ssm(),
        chronos=has_foundation("chronos"),
        uni2ts=has_foundation("uni2ts"),
        timesfm=has_foundation("timesfm"),
    )
    if cuda and caps["mamba_ssm"]:
        profile = "gpu_full"
    elif not cuda:
        profile = "windows_cpu"
    else:
        profile = "unknown"
    return Capabilities(profile=profile, **caps)


def as_dict() -> dict:
    return asdict(detect())
