"""STUB — implemented in Phase A.2 (docs/devplan/02-data-financial-macro.md, tasks F5-F9).

Parses its CLI, logs a TODO, exits 0 so scripts/run_all runs end-to-end. Phase A.2 replaces
the body with the real FRED IP + VIX download and stationarity transforms.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ntl_etf.utils.logging import get_logger  # noqa: E402

log = get_logger("download_macro")


def main() -> int:
    ap = argparse.ArgumentParser(description="Download FRED IP + VIX (STUB).")
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--map", default="configs/sector_fred_map.yaml")
    ap.add_argument("--out", default="data/processed/macro_ip.parquet")
    ap.add_argument("--vix-out", default="data/processed/vix_monthly.parquet")
    ap.add_argument("--start", default="2013-01-01")
    ap.add_argument("--end", default="2024-12-31")
    args, _ = ap.parse_known_args()
    log.info("STUB download_macro: implemented in Phase A.2 (F5-F9). args=%s", vars(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
