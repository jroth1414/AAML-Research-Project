"""STUB — implemented in Phase B (docs/devplan/03-panel-and-splits.md, tasks P1-P10).

Parses its CLI, logs a TODO, exits 0 so scripts/run_all runs end-to-end. Phase B replaces the
body with the real panel build + walk-forward splits + executable leakage audit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ntl_etf.utils.logging import get_logger  # noqa: E402

log = get_logger("build_panel")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build global panel + folds + leakage audit (STUB).")
    ap.add_argument("--config", default="configs/panel.yaml")
    args, _ = ap.parse_known_args()
    log.info("STUB build_panel: implemented in Phase B (P1-P10). args=%s", vars(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
