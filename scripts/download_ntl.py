"""STUB — implemented in Phase A.1 (docs/devplan/01-data-ntl.md, tasks N1-N10).

Parses its CLI, logs a TODO, and exits 0 so scripts/run_all is end-to-end runnable from day
one. Phase A.1 replaces the body with the real Black Marble VNP46A3 download + feature build.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ntl_etf.utils.logging import get_logger  # noqa: E402

log = get_logger("download_ntl")


def main() -> int:
    ap = argparse.ArgumentParser(description="Download NTL VNP46A3 features (STUB).")
    ap.add_argument("--regions", default="configs/regions.yaml")
    ap.add_argument("--config", default="configs/ntl.yaml")
    ap.add_argument("--start", default="2013-01")
    ap.add_argument("--end", default="2024-12")
    ap.add_argument("--backend", default="earthaccess")
    ap.add_argument("--out", default="data/processed/ntl_features.parquet")
    ap.add_argument("--dry-run", action="store_true")
    args, _ = ap.parse_known_args()
    log.info("STUB download_ntl: implemented in Phase A.1 (N1-N10). args=%s", vars(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
