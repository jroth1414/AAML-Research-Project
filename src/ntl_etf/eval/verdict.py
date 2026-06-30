"""Hypothesis decision rules + honest verdict summary (Phase D / Task E8).

Encodes the pre-registered H1-H6 criteria, evaluates them from the results + DM stores, and emits a
machine- and human-readable verdict. H0 is reported honestly (if no DL model significantly beats
momentum, say so). 'deferred' (not 'reject') is used when a comparison could not run on this
hardware (Mamba official kernel, foundation models, or absent pretrained variants).
"""

from __future__ import annotations

ALLOWED = {"support", "support (fallback impl)", "reject", "deferred"}


def _pooled(results, model, metric, stratum="all", task="leading"):
    m = results[
        (results["model"] == model)
        & (results["task"] == task)
        & (results["scope"] == "POOLED")
        & (results["stratum"] == stratum)
        & (results["metric"] == metric)
    ]
    return float(m["value"].iloc[0]) if len(m) else None


def _dm(dm, family, a, b, stratum=None):
    q = dm[(dm["family"] == family) & (dm["model_a"] == a) & (dm["model_b"] == b)]
    if stratum is not None:
        q = q[q["stratum"] == stratum]
    return q.iloc[0].to_dict() if len(q) else None


def decide_hypotheses(results, dm, prereg, mamba_impl: str = "fallback") -> dict:
    alpha = prereg.ALPHA
    out = {}

    # --- H1: a DL model beats BOTH baselines (MSE DM-sig + dir_acc>0.5 binomial) ---
    h1_support, h1_winner, h1_ev = False, None, []
    for dl in ["patchtst", "itransformer", "mamba"]:
        mse_dl = _pooled(results, dl, "mse")
        da = _pooled(results, dl, "dir_acc")
        da_p = _pooled(results, dl, "dir_acc_pvalue")
        mse_mom, mse_dlin = _pooled(results, "momentum", "mse"), _pooled(results, "dlinear", "mse")
        if None in (mse_dl, mse_mom, mse_dlin, da, da_p):
            continue
        beats_mse = mse_dl < mse_mom and mse_dl < mse_dlin
        dm_mom = _dm(dm, "A_signal_existence", dl, "momentum")
        dm_dlin = _dm(dm, "A_signal_existence", dl, "dlinear")
        dm_ok = bool(dm_mom and dm_dlin and dm_mom["win"] == dl and dm_dlin["win"] == dl)
        ev = {
            "model": dl,
            "mse": mse_dl,
            "mse_momentum": mse_mom,
            "mse_dlinear": mse_dlin,
            "dir_acc": da,
            "dir_acc_pvalue": da_p,
            "beats_both_mse": beats_mse,
            "dm_holm_beats_both": dm_ok,
        }
        h1_ev.append(ev)
        if beats_mse and da > 0.50 and da_p < alpha and dm_ok:
            h1_support, h1_winner = True, dl
    out["H1"] = {
        "verdict": "support" if h1_support else "reject",
        "winner": h1_winner,
        "evidence": h1_ev,
    }
    out["H0_note"] = (
        f"H1 supported by {h1_winner}; H0 rejected."
        if h1_support
        else "No DL model significantly beats the 12-month momentum baseline on pooled 1-month "
        "return MSE after Holm correction (with directional significance) — H0 holds. "
        "Point-estimate MSE deltas are reported but are not DM-significant."
    )

    # --- H2 / H3: architecture by region class (Family B) ---
    dm_h2 = _dm(dm, "B_architecture", "itransformer", "patchtst", stratum="multi_region")
    dm_h3 = _dm(dm, "B_architecture", "itransformer", "patchtst", stratum="single_region")
    out["H2"] = _arch_verdict(dm_h2, want_winner="itransformer")
    out["H3"] = _arch_verdict(dm_h3, want_winner="patchtst")

    # --- H4: Mamba >= both Transformers in disruption (Family C). On the CPU S6 fallback this is
    # DEFERRED (Risk R6); on the official kernel (mamba_impl='official', e.g. the V100) it is decided.
    ev_h4 = {
        "mamba_mse_disruption": _pooled(results, "mamba", "mse", "disruption"),
        "patchtst_mse_disruption": _pooled(results, "patchtst", "mse", "disruption"),
        "itransformer_mse_disruption": _pooled(results, "itransformer", "mse", "disruption"),
        "mamba_impl": mamba_impl,
    }
    if mamba_impl != "official":
        out["H4"] = {
            "verdict": "deferred",
            "reason": "Mamba ran via the CPU pure-PyTorch S6 fallback, not the official fused kernel "
            "(Risk R6/R14). Re-run on a CUDA GPU (mamba_impl='official') to decide H4.",
            "evidence": ev_h4,
        }
    else:
        c1 = _dm(dm, "C_disruption", "mamba", "patchtst", stratum="disruption")
        c2 = _dm(dm, "C_disruption", "mamba", "itransformer", stratum="disruption")
        worse = any(c and c.get("win") in ("patchtst", "itransformer") for c in (c1, c2))
        beats_one = any(c and c.get("win") == "mamba" for c in (c1, c2))
        out["H4"] = {
            "verdict": "reject" if worse else ("support" if beats_one else "support"),
            "note": "official Mamba kernel" + (" (beats >=1 Transformer)" if beats_one else ""),
            "evidence": ev_h4,
        }

    # --- H5: nowcast OOS R^2 >> leading OOS R^2 (gap >= 0.10) ---
    now = _pooled(results, "patchtst", "oos_r2", task="nowcast")
    lead = _pooled(results, "patchtst", "oos_r2", task="leading")
    if now is None:
        out["H5"] = {"verdict": "deferred", "reason": "no nowcast-task runs in this store."}
    else:
        gap = now - (lead if lead is not None else 0.0)
        out["H5"] = {
            "verdict": "support" if (gap >= 0.10 and now > 0) else "reject",
            "nowcast_r2": now,
            "leading_r2": lead,
            "gap": gap,
            "note": "OOS R^2 on contemporaneous IP (nowcast) vs forward returns (leading), paired on PatchTST.",
        }

    # --- H6a: NTL-masked-pretrained vs from-scratch PatchTST (Family D, identical params) ---
    dm_h6a = _dm(dm, "D_transfer", "patchtst_pretrained", "patchtst")
    if dm_h6a is None or dm_h6a.get("n", 0) < 3:
        out["H6a"] = {
            "verdict": "deferred",
            "reason": "no NTL-masked-pretrained variant run (Tier 2).",
        }
    else:
        out["H6a"] = {
            "verdict": "support" if dm_h6a["win"] == "patchtst_pretrained" else "reject",
            "win": dm_h6a["win"],
            "p_holm": dm_h6a["p_holm"],
            "n": dm_h6a["n"],
            "note": "NTL-masked-pretrained vs from-scratch PatchTST (identical architecture/params).",
        }
    dm_h6b = _dm(dm, "E_foundation", "chronos", "momentum")
    if dm_h6b is None or dm_h6b.get("n", 0) < 3:
        out["H6b"] = {
            "verdict": "deferred",
            "reason": "no foundation (Chronos) runs in this store.",
        }
    else:
        out["H6b"] = {
            "verdict": "support" if dm_h6b["win"] == "chronos" else "reject",
            "win": dm_h6b["win"],
            "p_holm": dm_h6b["p_holm"],
            "n": dm_h6b["n"],
            "note": "Chronos zero-shot vs momentum (no-NTL return-history reference; R23 H6b).",
        }

    for k, v in out.items():
        if isinstance(v, dict) and "verdict" in v:
            assert v["verdict"] in ALLOWED, (k, v["verdict"])
    return out


def _arch_verdict(dm_row, want_winner: str) -> dict:
    if dm_row is None or dm_row.get("n", 0) < 3:
        return {
            "verdict": "deferred",
            "reason": "insufficient paired observations in this stratum.",
        }
    win = dm_row["win"]
    return {
        "verdict": "support" if win == want_winner else "reject",
        "dm_stat": dm_row["dm_stat"],
        "p_holm": dm_row["p_holm"],
        "win": win,
        "n": dm_row["n"],
    }
