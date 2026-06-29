"""Build the global panel + walk-forward folds + executable leakage audit (Phase B / Task P10).

Orchestrates P2->P9: load sources -> screen pairs (P3) -> series registry (P4) -> walk-forward
folds (P6) -> leakage audit (P8, abort on any failure) -> write contract-conformant artifacts
(P9). Seeded, logged, manifested. CPU-only, no network.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from ntl_etf.data import panel  # noqa: E402
from ntl_etf.data.splits import walk_forward_splits, write_folds_manifest  # noqa: E402
from ntl_etf.utils.logging import get_logger  # noqa: E402
from ntl_etf.utils.manifest import write_manifest  # noqa: E402
from ntl_etf.utils.seed import set_seed  # noqa: E402

log = get_logger("build_panel")


def _forced_pairs(regions_doc: dict) -> set:
    pairs = set()
    for hp in (regions_doc.get("hypothesis_pairs") or {}).values():
        for r in hp.get("regions", []):
            pairs.add((r, hp["sector"]))
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the global panel + folds + leakage audit.")
    ap.add_argument("--config", default="configs/panel.yaml")
    ap.add_argument("--regions", default="configs/regions.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    regions_doc = yaml.safe_load(Path(args.regions).read_text(encoding="utf-8"))
    set_seed(int(cfg.get("seed", 1414)))
    feature = cfg.get("ntl_feature_primary", "ntl_mean")
    lag = int(cfg.get("release_lag_months", 1))

    # --- load sources onto the master grid (P2) ---
    ntl = panel.load_ntl_features(feature=feature)
    returns = panel.load_etf_returns()
    ip = panel.load_macro_ip(target_col=cfg.get("nowcast_target_col", "value_dlog"))
    grid = panel.master_grid()
    ntl_wide = panel.pivot_ntl_wide(ntl)

    # --- pair pre-screen (P3) ---
    forced = _forced_pairs(regions_doc)
    screen = panel.screen_pairs(ntl, returns, ip, regions_doc["regions"], cfg, forced)
    Path("data/interim").mkdir(parents=True, exist_ok=True)
    screen.to_csv("data/interim/pair_screen_manifest.csv", index=False)
    registry = panel.build_series_registry(screen)
    log.info(
        "screen: %d pairs, %d kept (%d forced); %d series, %d sectors",
        len(screen),
        int(screen["kept"].sum()),
        int(screen["forced_keep"].sum()),
        len(registry),
        registry["sector"].nunique(),
    )

    # --- walk-forward folds (P6) ---
    folds = walk_forward_splits(grid, cfg)
    Path("data/interim").mkdir(parents=True, exist_ok=True)
    write_folds_manifest(folds, "data/interim/folds_manifest.json")
    log.info(
        "walk-forward: %d folds (min_train=%d)",
        len(folds),
        cfg.get("walk_forward", {}).get("min_train_months", 60),
    )

    # --- executable leakage audit (P8) — abort on any failure ---
    warm_end = pd.Timestamp(
        cfg.get("screen_warmup", ["2013-01", "2017-12"])[1]
    ) + pd.offsets.MonthEnd(0)
    audit = panel.audit_panel(folds, ntl_wide, returns, ip, registry, cfg, warm_end)
    Path("experiments/manifests").mkdir(parents=True, exist_ok=True)
    Path("experiments/manifests/leakage_audit.json").write_text(
        json.dumps(audit, indent=2, default=str), encoding="utf-8"
    )
    log.info("leakage audit: %s", {k: v for k, v in audit.items() if k.startswith("L")})
    if not audit["all_pass"]:
        log.error("LEAKAGE AUDIT FAILED — aborting panel build (No-Go gate).")
        return 2

    # --- CI anchor count over the full study (soft floors, Risk R18) ---
    from ntl_etf.data.splits import Fold

    full = Fold(0, list(grid), [grid[-1]], [grid[-1]])
    spec = panel.WindowSpec(12, 1, "leading", "ci")
    ci_anchors = panel.build_anchors(registry, ntl_wide, returns, ip, spec, full, cfg)
    n_ci = len(ci_anchors)
    log.info(
        "CI-view anchors (L=12,H=1,leading, full study): %d " "(kept_series=%d x usable_origins)",
        n_ci,
        len(registry),
    )
    if n_ci < int(cfg.get("anchor_fail_floor", 100)):
        log.error("CI anchor count %d < fail floor — aborting", n_ci)
        return 3
    if n_ci < int(cfg.get("anchor_warn_floor", 500)):
        log.warning("CI anchor count %d below warn floor %d", n_ci, cfg.get("anchor_warn_floor"))

    # --- write contract artifacts (P9) ---
    pos = {d: i for i, d in enumerate(grid)}
    rows = []
    for _, sr in registry.iterrows():
        r, s, sidx = sr["region_id"], sr["sector"], int(sr["series_idx"])
        xv = ntl_wide[r] if r in ntl_wide.columns else None
        for t in grid:
            i = pos[t]
            ntl_val = float(xv.loc[t]) if xv is not None and pd.notna(xv.loc[t]) else float("nan")
            tl = (
                float(returns[s].iloc[i + lag])
                if (s in returns.columns and i + lag < len(grid))
                else float("nan")
            )
            tn = float(ip[s].iloc[i]) if s in ip.columns else float("nan")
            rows.append(
                {
                    "date": t,
                    "sector": s,
                    "region_id": r,
                    "feature": feature,
                    "series_idx": sidx,
                    "ntl_value": ntl_val,
                    "target_leading": tl,
                    "target_nowcast": tn,
                    "as_of_leading": grid[i + lag] if i + lag < len(grid) else pd.NaT,
                    "as_of_nowcast": grid[i + 1] if i + 1 < len(grid) else pd.NaT,
                    "valid": bool(pd.notna(ntl_val)),
                }
            )
    panel_long = pd.DataFrame(rows)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    panel_long.to_parquet("data/processed/panel_long.parquet", index=False)
    registry.to_parquet("data/processed/series_registry.parquet", index=False)
    log.info(
        "wrote panel_long (%d rows) + series_registry (%d series)", len(panel_long), len(registry)
    )

    # --- run manifest ---
    write_manifest(
        "experiments/manifests/panel_build",
        cfg,
        int(cfg.get("seed", 1414)),
        task="panel_build",
        n_folds=len(folds),
        data_hashes={
            "n_panel_rows": len(panel_long),
            "n_ci_anchors": n_ci,
            "n_series": len(registry),
        },
        stages_completed=["load", "screen", "folds", "audit", "write"],
    )
    log.info("panel build complete; leakage audit ALL PASS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
