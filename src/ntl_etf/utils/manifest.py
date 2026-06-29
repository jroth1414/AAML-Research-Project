"""Run manifest — the single self-describing record that makes a run reproducible.

This is the ONE superset manifest schema (DEVPLAN Appendix A.5). Phases M and W *extend* it
with extra fields via ``**extra``; they never redefine it. The git key is spelled ``git_sha``
everywhere (Risk R21). Every run records ``scaler_fit_on`` so the Phase E defensive audit
(E2) can pass (Risk R16).
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .capabilities import as_dict as caps_dict

# Packages whose versions are worth recording for reproducibility.
_TRACK = [
    "numpy",
    "pandas",
    "pyarrow",
    "scikit-learn",
    "scipy",
    "statsmodels",
    "torch",
    "rasterio",
    "geopandas",
    "xarray",
    "yfinance",
    "fredapi",
    "blackmarblepy",
    "earthaccess",
    "mamba-ssm",
    "chronos-forecasting",
    "uni2ts",
    "timesfm",
]


def _ver(p: str) -> str | None:
    try:
        return version(p)
    except PackageNotFoundError:
        return None


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "UNKNOWN"


def _git_dirty() -> bool:
    try:
        return bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except Exception:
        return False


def config_hash(config: dict) -> str:
    """Stable sha256 of the resolved config (used by the Phase E GPU-merge dedup, A.4)."""
    blob = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def write_manifest(
    run_dir: str | Path,
    config: dict,
    seed: int,
    *,
    profile: str | None = None,
    scaler_fit_on: str = "train",
    model: str | None = None,
    variant: str | None = None,
    task: str | None = None,
    horizon: int | None = None,
    n_folds: int | None = None,
    mamba_impl: str | None = None,
    data_hashes: dict | None = None,
    stages_completed: list[str] | None = None,
    gpu_stages_skipped: list[str] | None = None,
    **extra,
) -> Path:
    """Write ``<run_dir>/manifest.json`` (the A.5 superset). Returns the path."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    caps = caps_dict()
    m = {
        "run_id": run_dir.name,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "python": sys.version,
        "platform": platform.platform(),
        "seed": seed,
        "profile": profile or caps.get("profile", "unknown"),
        "capabilities": caps,
        "packages": {p: _ver(p) for p in _TRACK},
        "config": config,
        "config_hash": config_hash(config),
        "scaler_fit_on": scaler_fit_on,
        "model": model,
        "variant": variant,
        "task": task,
        "horizon": horizon,
        "n_folds": n_folds,
        "mamba_impl": mamba_impl,
        "data_hashes": data_hashes or {},
        "stages_completed": stages_completed or [],
        "gpu_stages_skipped": gpu_stages_skipped or [],
    }
    m.update(extra)
    out = run_dir / "manifest.json"
    out.write_text(json.dumps(m, indent=2, default=str), encoding="utf-8")
    return out
