"""CLI capability/environment report. Used at the start of run_all and CI smoke.

Prints the detected capability profile + key package versions. The core import smoke here
deliberately EXCLUDES the geo/NTL backends (rasterio is checked, but blackmarble/earthaccess
are NOT) so a missing NTL wheel never fails this gate (Risk R5). Must run on Windows/CPU and
print ``profile=windows_cpu`` without raising.
"""

from __future__ import annotations

import importlib
import sys

# Add src/ to path so this runs without an editable install too.
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ntl_etf.utils.capabilities import detect  # noqa: E402

# Truly-universal core packages whose absence should fail the env gate.
# NOTE: blackmarble / earthaccess are intentionally NOT here (lazy NTL imports, Risk R5).
CORE_SMOKE = [
    "numpy",
    "pandas",
    "pyarrow",
    "scipy",
    "sklearn",
    "statsmodels",
    "matplotlib",
    "rasterio",
    "geopandas",
    "xarray",
    "yfinance",
    "fredapi",
    "torch",
]


def _ver(mod: str) -> str:
    try:
        m = importlib.import_module(mod)
        return getattr(m, "__version__", "?")
    except Exception as exc:  # pragma: no cover - reported, not raised
        return f"MISSING ({exc.__class__.__name__})"


def main() -> int:
    caps = detect()
    print(f"profile={caps.profile}")
    print(f"cuda={caps.cuda}")
    print(f"mamba_ssm={caps.mamba_ssm}")
    print(f"chronos={caps.chronos} uni2ts={caps.uni2ts} timesfm={caps.timesfm}")
    print("--- core package versions ---")
    missing = []
    for mod in CORE_SMOKE:
        v = _ver(mod)
        if v.startswith("MISSING"):
            missing.append(mod)
        print(f"  {mod}: {v}")
    if missing:
        # Report but do not raise on optional-ish; raise only if a hard-core pkg is gone.
        print(f"WARNING: missing core packages: {missing}", file=sys.stderr)
        return 1
    print("core OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
