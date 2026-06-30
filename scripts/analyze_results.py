"""Single evaluation entry point (Phase D / Task E11).

Pipeline: load+validate predictions (E1) -> alignment audit (E2; abort on fail) -> add strata (E6)
-> aggregate metrics -> write results store (E1) -> DM suite + Holm/BH (E7) -> decide hypotheses
(E8) -> render tables (E10) + figures (E9) -> print a console verdict summary. Deterministic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ntl_etf.eval import prereg, results, stats, stratify, verdict  # noqa: E402
from ntl_etf.utils.logging import get_logger  # noqa: E402
from ntl_etf.utils.seed import set_seed  # noqa: E402

log = get_logger("analyze")


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze predictions -> metrics, DM, verdicts.")
    ap.add_argument("--experiments-dir", default="experiments")
    ap.add_argument("--tables-dir", default="paper/tables")
    ap.add_argument("--figures-dir", default="paper/figures")
    ap.add_argument("--alpha", type=float, default=prereg.ALPHA)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-figures", action="store_true")
    args = ap.parse_args()
    set_seed(args.seed)

    preds = results.load_predictions(args.experiments_dir)
    log.info("loaded %d predictions from %d models", len(preds), preds["model"].nunique())

    audit = results.audit_alignment(preds, args.experiments_dir)
    log.info("alignment audit: %s", audit)
    if not audit["all_pass"]:
        log.error("alignment audit FAILED — aborting eval")
        return 2

    preds = stratify.add_strata(preds)
    agg = stratify.aggregate(preds)
    results.write_results_store(agg, str(Path(args.experiments_dir) / "results_store.parquet"))
    log.info("results store: %d metric rows", len(agg))

    dm = stats.run_dm_suite(preds, prereg.FAMILIES, alpha=args.alpha)
    dm.to_parquet(Path(args.experiments_dir) / "dm_results.parquet", index=False)
    dm.to_csv(Path(args.experiments_dir) / "dm_results.csv", index=False)
    log.info(
        "DM suite: %d comparisons, %d Holm-significant wins",
        len(dm),
        int((dm["win"] != "none").sum()),
    )

    # H4 is decided only on the official Mamba kernel; read mamba_impl from its run manifest.
    mamba_impl = "fallback"
    for mdir in Path(args.experiments_dir).glob("mamba__*"):
        mf = mdir / "manifest.json"
        if mf.exists():
            impl = json.loads(mf.read_text(encoding="utf-8")).get("mamba_impl")
            if impl:
                mamba_impl = impl
    log.info("mamba_impl detected: %s", mamba_impl)
    verdicts = verdict.decide_hypotheses(agg, dm, prereg, mamba_impl=mamba_impl)
    (Path(args.experiments_dir) / "hypotheses_verdict.json").write_text(
        json.dumps(verdicts, indent=2, default=str), encoding="utf-8"
    )

    _render_tables(agg, dm, verdicts, args.tables_dir)
    if not args.skip_figures:
        _render_figures(agg, dm, args.figures_dir)

    print("\n===== HYPOTHESIS VERDICTS =====")
    for h in ["H1", "H2", "H3", "H4", "H5", "H6a", "H6b"]:
        v = verdicts.get(h, {})
        print(f"  {h}: {v.get('verdict'):<26} {v.get('reason', v.get('winner', ''))}")
    print(f"  H0: {verdicts['H0_note']}")
    return 0


def _render_tables(agg, dm, verdicts, tables_dir):
    import pandas as pd

    tdir = Path(tables_dir)
    tdir.mkdir(parents=True, exist_ok=True)
    lead = agg[(agg.task == "leading") & (agg.scope == "POOLED") & (agg.stratum == "all")]
    main = lead.pivot_table(index="model", columns="metric", values="value")
    for col, fn in [("md", "to_markdown"), ("tex", "to_latex")]:
        (tdir / f"main_results.{col}").write_text(getattr(main.round(4), fn)(), encoding="utf-8")
    (tdir / "dm_family_a.md").write_text(
        dm[dm.family == "A_signal_existence"][
            ["model_a", "model_b", "stratum", "dm_stat", "p_raw", "p_holm", "win", "n"]
        ]
        .round(4)
        .to_markdown(index=False),
        encoding="utf-8",
    )
    rows = [
        {
            "hypothesis": h,
            **{k: v for k, v in verdicts[h].items() if k in ("verdict", "reason", "winner")},
        }
        for h in ["H1", "H2", "H3", "H4", "H5", "H6a", "H6b"]
    ]
    pd.DataFrame(rows).to_markdown(tdir / "hypotheses_verdict.md", index=False)


def _render_figures(agg, dm, figures_dir):
    from ntl_etf.eval import plots

    plots.plot_metric_bars(agg, figures_dir, "mse")
    plots.plot_metric_bars(agg, figures_dir, "dir_acc")
    plots.plot_dm_significance(dm, figures_dir, "A_signal_existence")


if __name__ == "__main__":
    raise SystemExit(main())
