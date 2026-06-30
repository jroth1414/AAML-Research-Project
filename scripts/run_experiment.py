"""Run ONE model + config end-to-end over the walk-forward folds and write the run directory
(Phase C / Task M13): predictions.parquet (A.3), manifest.json (A.5), and skipped.json when a
capability-gated model is unavailable. CPU-first; honors the skip-and-log contract.
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
from ntl_etf.data.splits import walk_forward_splits  # noqa: E402
from ntl_etf.models.base import ModelConfig, write_predictions  # noqa: E402
from ntl_etf.utils.config import load_env  # noqa: E402
from ntl_etf.utils.logging import get_logger  # noqa: E402
from ntl_etf.utils.manifest import write_manifest  # noqa: E402
from ntl_etf.utils.seed import set_seed  # noqa: E402

log = get_logger("run_experiment")
DEEP = {"dlinear", "patchtst", "itransformer", "mamba"}
FOUNDATION = {"chronos", "moirai", "timesfm"}


def _build_forecaster(model: str, mc: ModelConfig, returns, ip, train_cfg):
    from ntl_etf.models.base import DeepForecaster

    target_wide = returns if mc.task == "leading" else ip
    if model == "momentum":
        from ntl_etf.models.momentum import MomentumForecaster

        return MomentumForecaster(mc, target_wide)
    if model == "dlinear":
        from ntl_etf.models.dlinear import dlinear_factory

        return DeepForecaster(mc, dlinear_factory, train_cfg)
    if model == "patchtst":
        from ntl_etf.models.patchtst import patchtst_factory

        return DeepForecaster(mc, patchtst_factory, train_cfg)
    if model == "itransformer":
        from ntl_etf.models.itransformer import itransformer_factory

        return DeepForecaster(mc, itransformer_factory, train_cfg)
    if model == "mamba":
        from ntl_etf.models.mamba import mamba_factory

        return DeepForecaster(mc, mamba_factory, train_cfg)
    if model == "patchtst_pretrained":
        from ntl_etf.models.pretrain import pretrained_patchtst_factory

        return DeepForecaster(mc, pretrained_patchtst_factory, train_cfg)
    if model in FOUNDATION:
        from ntl_etf.models.foundation import FoundationForecaster

        mc.extra["foundation"] = model
        return FoundationForecaster(mc, target_wide)
    raise SystemExit(f"unknown model {model!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run one experiment.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", default="leading", choices=["leading", "nowcast"])
    ap.add_argument("--H", type=int, default=1)
    ap.add_argument("--L", type=int, default=12)
    ap.add_argument("--variant", default="scratch")
    ap.add_argument("--seed", type=int, default=1414)
    ap.add_argument("--config", default="configs/panel.yaml")
    ap.add_argument("--smoke", action="store_true", help="few folds/epochs for a quick check")
    ap.add_argument("--out", default=None, help="experiments/<run_id> dir (auto if omitted)")
    args = ap.parse_args()
    load_env()
    set_seed(args.seed)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    feature = cfg.get("ntl_feature_primary", "ntl_mean")
    run_id = f"{args.model}__{args.task}__H{args.H}__L{args.L}__seed{args.seed}__{args.variant}"
    run_dir = Path(args.out) if args.out else Path("experiments") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    mc = ModelConfig(
        name=args.model,
        task=args.task,
        L=args.L,
        H=args.H,
        seed=args.seed,
        variant=args.variant,
        extra=dict(cfg.get("model_extra", {})),
    )

    # --- data context ---
    ntl = panel.load_ntl_features(feature=feature)
    returns = panel.load_etf_returns()
    ip = panel.load_macro_ip(target_col=cfg.get("nowcast_target_col", "value_dlog"))
    ntl_wide = panel.pivot_ntl_wide(ntl)
    registry = pd.read_parquet("data/processed/series_registry.parquet")
    grid = panel.master_grid()
    folds = walk_forward_splits(grid, cfg)
    if args.smoke:
        folds = folds[:1]

    # H6a: ensure the masked-pretrained PatchTST encoder exists, then fine-tune from it.
    if args.model == "patchtst_pretrained":
        from ntl_etf.models.pretrain import pretrain_patchtst

        ckpt = "experiments/_pretrained/patchtst.pt"
        if not Path(ckpt).exists():
            log.info("pretraining PatchTST encoder on the unlabeled NTL corpus (M8)...")
            pretrain_patchtst(ntl_wide, mc, ckpt, steps=30 if args.smoke else 800, seed=args.seed)
        mc.extra["pretrained_ckpt"] = ckpt

    from ntl_etf.train.trainer import TrainConfig

    tcfg = TrainConfig(
        epochs=3 if args.smoke else 40,
        batch_size=32,
        lr=5e-4,
        seed=args.seed,
        max_steps=30 if args.smoke else None,
    )

    # --- build the forecaster (skip-and-log on capability-gated unavailability) ---
    try:
        forecaster = _build_forecaster(args.model, mc, returns, ip, tcfg)
    except Exception as exc:  # FoundationUnavailable etc.
        from ntl_etf.models.foundation import FoundationUnavailable

        if isinstance(exc, FoundationUnavailable):
            log.warning("%s unavailable: %s", args.model, exc)
            (run_dir / "skipped.json").write_text(
                json.dumps({"model": args.model, "reason": str(exc)}, indent=2), encoding="utf-8"
            )
            return 0
        raise

    view = "variate" if args.model == "itransformer" else "ci"
    spec = panel.WindowSpec(args.L, args.H, args.task, view)

    rows = []
    mamba_impl = None
    for fold in folds:
        norms = panel.compute_fold_norms(ntl_wide, returns, ip, registry, fold, spec)
        anchors = panel.build_anchors(registry, ntl_wide, returns, ip, spec, fold, cfg)
        by = {sp: [a for a in anchors if a["split"] == sp] for sp in ("train", "val", "test")}
        if not by["test"]:
            continue
        train_ds = panel.PanelDataset(by["train"], ntl_wide, registry, norms, spec)
        val_ds = panel.PanelDataset(by["val"], ntl_wide, registry, norms, spec)
        test_ds = panel.PanelDataset(by["test"], ntl_wide, registry, norms, spec)
        forecaster.fit(train_ds, val_ds)
        y_pred = forecaster.predict(test_ds)
        rows.extend(forecaster.to_rows(test_ds, y_pred, fold=fold.fold_id, split="test"))
        mamba_impl = mc.extra.get("mamba_impl", mamba_impl)
        log.info("fold %d: %d test predictions", fold.fold_id, len(by["test"]))

    # Ensemble per-region CI predictions into one per (sector, date, fold): multiple region series
    # map to the same ETF, so average y_pred (y_true is identical for a sector at a date) -> A.3.
    if rows:
        gdf = pd.DataFrame(rows)
        keys = [
            "model",
            "variant",
            "pretrained",
            "task",
            "target_kind",
            "etf",
            "horizon",
            "fold",
            "split",
            "date",
            "seed",
        ]
        rows = (
            gdf.groupby(keys, as_index=False)
            .agg(y_true=("y_true", "first"), y_pred=("y_pred", "mean"))
            .to_dict("records")
        )
    write_predictions(rows, run_dir)
    n_params = forecaster.n_params() if hasattr(forecaster, "n_params") else 0
    write_manifest(
        run_dir,
        cfg,
        args.seed,
        model=args.model,
        variant=args.variant,
        task=args.task,
        horizon=args.H,
        n_folds=len(folds),
        mamba_impl=mamba_impl,
        n_params=n_params,
        n_predictions=len(rows),
    )
    log.info("run %s complete: %d predictions, n_params=%d", run_id, len(rows), n_params)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
