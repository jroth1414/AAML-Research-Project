"""Download NASA Black Marble VNP46A3 NTL composites and build the per-region monthly feature
table (Phase A.1 / Tasks N1-N8). Idempotent and resumable via the manifest.

Examples
--------
PowerShell:
  python scripts/download_ntl.py --regions configs/regions.yaml --config configs/ntl.yaml \
      --start 2013-01 --end 2024-12
  python scripts/download_ntl.py --region singapore_port --start 2020-06 --end 2020-06   # 1-ROI smoke
  python scripts/download_ntl.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ntl_etf.data.ntl import build_ntl_features  # noqa: E402
from ntl_etf.utils.config import load_env  # noqa: E402
from ntl_etf.utils.logging import get_logger  # noqa: E402
from ntl_etf.utils.seed import set_seed  # noqa: E402

log = get_logger("download_ntl")


def main() -> int:
    ap = argparse.ArgumentParser(description="Download VNP46A3 NTL features.")
    ap.add_argument("--regions", default="configs/regions.yaml")
    ap.add_argument("--config", default="configs/ntl.yaml")
    ap.add_argument("--start", default="2013-01")
    ap.add_argument("--end", default="2024-12")
    ap.add_argument("--out", default="data/processed/ntl_features.parquet")
    ap.add_argument("--manifest", default="data/interim/ntl/manifest.json")
    ap.add_argument("--region", default=None, help="restrict to a single region id")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    load_env()
    set_seed()  # no randomness here, but keep determinism flags consistent
    df = build_ntl_features(
        regions_yaml=args.regions,
        ntl_cfg=args.config,
        start=args.start,
        end=args.end,
        out_parquet=args.out,
        manifest=args.manifest,
        only_region=args.region,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        log.info("dry-run complete: %d planned tasks", len(df))
        return 0
    if df is None or getattr(df, "empty", True):
        log.warning("no rows produced (check credentials / network / region-months)")
        return 0
    log.info("done: %d rows", len(df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
