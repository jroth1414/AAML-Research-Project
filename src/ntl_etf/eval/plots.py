"""Publication figures from the results/DM stores (Phase D / Task E9).

Deterministic matplotlib (Agg backend, no interactive state); each function writes a 300-DPI PNG +
a vector PDF and returns the PNG path. Figures read only the stores, never raw data folders.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _save(fig, out_dir, name) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    return png


def plot_metric_bars(results, out_dir, metric="mse", task="leading") -> Path:
    """Grouped bar of POOLED `metric` per model (stratum=all) with fold-CI error bars."""
    sub = results[
        (results["task"] == task)
        & (results["scope"] == "POOLED")
        & (results["stratum"] == "all")
        & (results["metric"] == metric)
    ].sort_values("value")
    fig, ax = plt.subplots(figsize=(7, 4))
    yerr = None
    if metric == "mse" and sub["ci_low"].notna().any():
        yerr = [
            (sub["value"] - sub["ci_low"]).clip(lower=0).fillna(0),
            (sub["ci_high"] - sub["value"]).clip(lower=0).fillna(0),
        ]
    ax.bar(sub["model"], sub["value"], yerr=yerr, capsize=4, color="#4C72B0")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Pooled {metric.upper()} per model ({task}, H=1)")
    ax.tick_params(axis="x", rotation=30)
    return _save(fig, out_dir, f"metric_bars_{metric}_{task}")


def plot_dm_significance(dm, out_dir, family="A_signal_existence") -> Path:
    """Heatmap of Holm-adjusted DM p-values for one family (stars where a win is significant)."""
    sub = dm[dm["family"] == family].copy()
    fig, ax = plt.subplots(figsize=(6, 4))
    if len(sub):
        sub["pair"] = sub["model_a"] + " vs " + sub["model_b"]
        vals = sub["p_holm"].fillna(1.0).to_numpy().reshape(-1, 1)
        ax.imshow(vals, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(sub["pair"])
        ax.set_xticks([0])
        ax.set_xticklabels(["p_holm"])
        for i, (_, r) in enumerate(sub.iterrows()):
            star = "*" if (r["win"] not in (None, "none")) else ""
            ax.text(0, i, f"{r['p_holm']:.3f}{star}", ha="center", va="center")
    ax.set_title(f"DM Holm-adjusted p-values: {family}")
    return _save(fig, out_dir, f"dm_significance_{family}")
