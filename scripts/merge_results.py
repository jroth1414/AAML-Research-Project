"""Merge GPU-profile result rows back into the canonical metrics store (Appendix A.4).

GPU runs (Profiles 2/3) write the same ``experiments/<run_id>/`` artifacts as Profile 1.
This script appends/dedups their rows into ``experiments/results_store.parquet`` keyed on
``DEDUP_KEYS`` (… + config_hash); on a key collision the most recent ``timestamp`` wins.

Stub created in Phase S; the full Phase-E driver may extend it. Safe to run anytime.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from ntl_etf.utils.logging import get_logger  # noqa: E402
from ntl_etf.utils.results import (  # noqa: E402
    DEDUP_KEYS,
    RESULT_COLUMNS,
    RESULTS_STORE_CSV,
    RESULTS_STORE_PARQUET,
)

log = get_logger("merge_results")


def merge(incoming_globs: list[str]) -> int:
    frames = []
    if Path(RESULTS_STORE_PARQUET).exists():
        frames.append(pd.read_parquet(RESULTS_STORE_PARQUET))
    for g in incoming_globs:
        for p in Path().glob(g):
            if p.suffix == ".parquet":
                frames.append(pd.read_parquet(p))
            elif p.suffix == ".csv":
                frames.append(pd.read_csv(p))
    if not frames:
        log.warning("nothing to merge")
        return 0
    df = pd.concat(frames, ignore_index=True)
    for c in RESULT_COLUMNS:
        if c not in df.columns:
            df[c] = None
    keys = [k for k in DEDUP_KEYS if k in df.columns]
    df = df.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)
    Path(RESULTS_STORE_PARQUET).parent.mkdir(parents=True, exist_ok=True)
    df[RESULT_COLUMNS].to_parquet(RESULTS_STORE_PARQUET, index=False)
    df[RESULT_COLUMNS].to_csv(RESULTS_STORE_CSV, index=False)
    log.info("merged store has %d rows -> %s", len(df), RESULTS_STORE_PARQUET)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge result rows into the canonical store.")
    ap.add_argument(
        "--incoming",
        nargs="*",
        default=["experiments/**/results_store.parquet"],
        help="glob(s) of incoming result parquet/csv files to merge in",
    )
    args = ap.parse_args()
    return merge(args.incoming)


if __name__ == "__main__":
    raise SystemExit(main())
