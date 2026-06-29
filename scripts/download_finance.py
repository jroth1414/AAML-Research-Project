"""Download the 11 SPDR sector ETFs via yfinance and build the month-end log-return + momentum
table (Phase A.2 / Tasks F2-F4). Writes data/processed/etf_returns.parquet.

Uses daily auto-adjusted closes resampled to month-end in our own code (more robust than
interval='1mo'). yfinance with auto_adjust=True returns split/dividend-adjusted 'Close' (no
'Adj Close' column) — that is the total-return-adjusted price we want.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from ntl_etf.data.finance import build_ticker_frame  # noqa: E402
from ntl_etf.utils.logging import get_logger  # noqa: E402
from ntl_etf.utils.seed import set_seed  # noqa: E402

log = get_logger("download_finance")


def _ticker_sector_map(map_path: str) -> dict:
    doc = yaml.safe_load(Path(map_path).read_text(encoding="utf-8")) or {}
    out = {}
    for row in doc.get("sectors", []):
        out[row["ticker"]] = row.get("sector", row["ticker"].lower())
    return out


def _download(tickers, start, end, attempts=3):
    import yfinance as yf

    last = None
    for i in range(attempts):
        try:
            raw = yf.download(
                tickers,
                start=start,
                end=end,
                interval="1d",
                auto_adjust=True,
                group_by="ticker",
                progress=False,
                threads=True,
            )
            if raw is not None and len(raw) > 0:
                return raw
        except Exception as exc:  # noqa: BLE001
            last = exc
        time.sleep(2**i)
    raise RuntimeError(f"yfinance download failed after {attempts} attempts: {last}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Download ETF returns.")
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--map", default="configs/sector_fred_map.yaml")
    ap.add_argument("--out", default="data/processed/etf_returns.parquet")
    ap.add_argument("--start", default="2013-01-01")
    ap.add_argument("--end", default="2024-12-31")
    args = ap.parse_args()
    set_seed()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    tickers = cfg["tickers"]
    lookback = int(cfg.get("momentum", {}).get("lookback_months", 12))
    sectors = _ticker_sector_map(args.map)

    raw = _download(tickers, args.start, args.end)
    frames = []
    first_valid = {}
    for t in tickers:
        if isinstance(raw.columns, pd.MultiIndex):
            if t not in raw.columns.get_level_values(0):
                log.warning("no data for %s", t)
                continue
            px = raw[t]["Close"].dropna()
        else:
            px = raw["Close"].dropna()
        if px.empty:
            log.warning("empty price series for %s", t)
            continue
        frame = build_ticker_frame(px, t, sectors.get(t, t.lower()), lookback=lookback)
        fv = frame.loc[frame["log_return"].notna(), "date"]
        first_valid[t] = str(fv.min().date()) if len(fv) else None
        frames.append(frame)
        log.info("%s: %d months, first valid return %s", t, len(frame), first_valid[t])

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    log.info("wrote %d rows (%d tickers) -> %s", len(df), df["ticker"].nunique(), args.out)
    log.info("first valid months: %s", first_valid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
