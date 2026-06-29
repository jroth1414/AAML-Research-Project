"""Download FRED sector industrial production + CBOE VIX, align to month-end, attach stationarity
transforms, and write the macro/VIX parquets + the Phase-B leakage/alignment manifest
(Phase A.2 / Tasks F5-F12).

Outputs:
  data/processed/macro_ip.parquet            (long: sector, fred_series_id, date, value, transforms)
  data/processed/macro_series_resolved.json  (which series id each sector actually used)
  data/processed/vix_monthly.parquet         (date, vix_mean, vix_max, disruption_flag)
  data/processed/financial_macro_manifest.json (release-lag + alignment contract for Phase B)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from ntl_etf.data.macro import (  # noqa: E402
    add_macro_transforms,
    build_vix_monthly,
    fetch_series,
    get_fred_client,
    to_month_end_macro,
)
from ntl_etf.utils.config import load_env  # noqa: E402
from ntl_etf.utils.logging import get_logger  # noqa: E402
from ntl_etf.utils.seed import set_seed  # noqa: E402

log = get_logger("download_macro")


def _resolve_and_fetch(fred, entry, start, end):
    """Try fred_series_id then fallback_chain; return (series, resolved_id) or (None, None)."""
    candidates = []
    if entry.get("fred_series_id"):
        candidates.append(entry["fred_series_id"])
    for sid in entry.get("fallback_chain", []) or []:
        if sid not in candidates:
            candidates.append(sid)
    for sid in candidates:
        try:
            s = fetch_series(fred, sid, start, end)
            return to_month_end_macro(s), sid
        except Exception as exc:  # noqa: BLE001
            log.warning("series %s failed (%s); trying next", sid, str(exc)[:80])
    return None, None


def main() -> int:
    ap = argparse.ArgumentParser(description="Download FRED IP + VIX.")
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--map", default="configs/sector_fred_map.yaml")
    ap.add_argument("--out", default="data/processed/macro_ip.parquet")
    ap.add_argument("--vix-out", default="data/processed/vix_monthly.parquet")
    ap.add_argument("--start", default="2013-01-01")
    ap.add_argument("--end", default="2024-12-31")
    args = ap.parse_args()
    load_env()
    set_seed()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    smap = yaml.safe_load(Path(args.map).read_text(encoding="utf-8"))
    fred = get_fred_client()

    rows = []
    resolved = {}
    entries = []
    if smap.get("control"):
        entries.append({**smap["control"], "nowcast_eligible": True})
    entries += smap.get("sectors", [])

    for entry in entries:
        if not entry.get("fred_series_id") and not entry.get("fallback_chain"):
            log.info("skip %s (no IP analog; nowcast-ineligible)", entry["ticker"])
            continue
        s, rid = _resolve_and_fetch(fred, entry, args.start, args.end)
        if s is None:
            log.warning("could not resolve any series for %s", entry["ticker"])
            resolved[entry["ticker"]] = None
            continue
        resolved[entry["ticker"]] = rid
        df = pd.DataFrame({"date": pd.to_datetime(s.index), "value": s.to_numpy()})
        df = add_macro_transforms(df)
        df.insert(0, "sector", entry.get("sector", entry["ticker"].lower()))
        df.insert(1, "fred_series_id", rid)
        df["tier"] = entry.get("tier", "")
        df["nowcast_eligible"] = bool(entry.get("nowcast_eligible", False))
        rows.append(df)
        log.info("%s -> %s: %d obs", entry["ticker"], rid, len(s.dropna()))
        time.sleep(0.2)  # be gentle with the FRED API

    macro = pd.concat(rows, ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    macro.to_parquet(args.out, index=False)
    Path("data/processed/macro_series_resolved.json").write_text(
        json.dumps(resolved, indent=2), encoding="utf-8"
    )
    log.info(
        "wrote macro_ip: %d rows, %d sectors -> %s", len(macro), macro["sector"].nunique(), args.out
    )

    # --- VIX (F9) ---
    vix_id = cfg.get("vix", {}).get("fred_series_id", "VIXCLS")
    thr = float(cfg.get("vix", {}).get("disruption_threshold", 25.0))
    vix_daily = fetch_series(fred, vix_id, args.start, args.end)
    vix = build_vix_monthly(vix_daily, threshold=thr)
    Path(args.vix_out).parent.mkdir(parents=True, exist_ok=True)
    vix.to_parquet(args.vix_out, index=False)
    log.info(
        "wrote vix_monthly: %d rows, %d disruption months -> %s",
        len(vix),
        int(vix["disruption_flag"].sum()),
        args.vix_out,
    )

    # --- F12 manifest (release-lag + alignment contract for Phase B) ---
    manifest = {
        "study": cfg["study"],
        "macro": {
            "source": "fredapi",
            "resolved_series": resolved,
            "nowcast_eligible_sectors": [
                e.get("sector") for e in entries if e.get("nowcast_eligible")
            ],
            "nowcast_excluded_sectors": ["financials", "communication", "real_estate"],
        },
        "vix": {"source": f"FRED:{vix_id}", "threshold": thr, "agg": "monthly_mean_of_daily_close"},
        "alignment": {
            "calendar": "tz-naive month-end DatetimeIndex; left-join to master grid; no ffill",
            "leading_lag_months": 1,
            "nowcast_lag_months": 0,
            "etf_return_transform": "none (already stationary)",
            "ip_target_transform": "value_dlog (causal)",
            "stl_policy": "value_sa is descriptive only; never enters model features",
        },
        "seeds": {"global_seed": 1414},
    }
    Path("data/processed/financial_macro_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    log.info("wrote financial_macro_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
