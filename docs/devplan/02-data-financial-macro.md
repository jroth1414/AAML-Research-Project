# Phase A.2 — Financial (ETF) & Macro (FRED, VIX) Acquisition

> Per-phase plan file: `docs/devplan/02-data-financial-macro.md`. Task-ID prefix: **F**.
> Audience: an autonomous coding agent, no human in the loop. Imperative voice. Every external fact that could drift over time is marked **TO-VERIFY** with the exact check to run.

This phase acquires the **financial target** (11 SPDR sector ETF monthly log returns + a 12-month momentum baseline feature), the **coincident nowcast target** (FRED sector industrial-production / activity series), and the **disruption stratifier** (CBOE VIX monthly mean + `disruption_flag`). It produces tidy, month-end-indexed parquet files that Phase P (`docs/devplan/03-panel-and-splits.md`) joins into the global panel, and it defines the stationarity transforms and the release-lag alignment contract that Phase P must enforce.

This phase does **not** build the panel, the windowing, or the walk-forward splits (Phase P), and does **not** touch NTL rasters (Phase N, `docs/devplan/01-data-ntl.md`). It only references their task IDs as dependencies/consumers.

---

## 0. Conventions used in this phase

| Convention | Rule |
|---|---|
| Index | Every output series uses a `date` column that is a **month-end** `pd.Timestamp`, timezone-naive (tz-free), normalized to midnight. Use pandas period-end (`MS`→`M` resample with `"ME"` label; see F2). |
| Canonical ticker set | `XLB XLC XLE XLF XLI XLK XLP XLRE XLU XLV XLY` (11 SPDR Select Sector funds). |
| Sector key | A lowercase short name per ticker, defined once in `configs/sector_fred_map.yaml` (F6) and reused everywhere. |
| Storage | Parquet via `pyarrow`, written by `src/ntl_etf/utils/io.py` helpers (created in Phase S, prefix S; if not yet present, create a minimal `write_parquet`/`read_parquet` here and let S supersede it). |
| Secrets | `FRED_API_KEY` read from environment / `.env` (gitignored). Never hard-code. yfinance needs no key. |
| Determinism | No randomness in this phase except optional jitter in tests; still call `src/ntl_etf/utils/seed.py:set_global_seed(seed)` at the top of each script for consistency. |
| Leakage stance | Returns and momentum are **target-side** and contemporaneous-feature-side artifacts; the release-lag guard (NTL_t → return_{t+1}) is applied at **panel build** (Phase P), but this phase records the metadata Phase P needs (F12). |

---

## Task index

| ID | Objective | Depends on |
|---|---|---|
| F1 | Add config files: ticker list, sector→FRED mapping, VIX threshold | S (repo bootstrap) |
| F2 | `finance.py`: month-end resample + log-return + 12m momentum core | F1 |
| F3 | `finance.py`: ragged-history handling (inception masks, no price ffill) | F2 |
| F4 | `scripts/download_finance.py` CLI → ETF parquet | F2, F3 |
| F5 | `macro.py`: FRED client wrapper + series fetch + month-end align | F1 |
| F6 | Finalize sector→FRED IP/activity mapping with honest coverage notes | F1, F5 |
| F7 | `macro.py`: stationarity transforms (STL deseasonalize + first diff) | F5, F6 |
| F8 | `scripts/download_macro.py` CLI → FRED IP parquet | F5, F6, F7 |
| F9 | `macro.py` + script: VIX fetch, monthly mean, `disruption_flag` | F5 |
| F10 | Calendar-alignment + transform registry (which transform feeds which task) | F2, F7, F9 |
| F11 | Offline unit tests on fixtures (returns, momentum, resample, flag, STL) | F2, F3, F7, F9 |
| F12 | Release-lag + leakage contract doc + machine-readable manifest for Phase P | F4, F8, F9, F10 |

---

## F1 — Config: tickers, sector keys, VIX threshold

**Objective.** Create config files that every downstream task reads, so tickers/series IDs/thresholds are never hard-coded.

**Dependencies.** S (repo bootstrap, `configs/` exists, `src/ntl_etf/utils/config.py` loader exists; if the loader is missing, add a thin `load_yaml(path)->dict` here).

**Actions.**
1. Create `configs/sector_fred_map.yaml` with one entry per ticker (full mapping finalized in F6; create a stub now with the ticker/sector keys and leave `fred_series_id`/`nowcast_eligible` placeholders).
2. Add a top-level finance/macro config block to `configs/experiment.yaml` (or a new `configs/data.yaml`) with the study range and VIX threshold.

**Files.**
- `configs/sector_fred_map.yaml` (create)
- `configs/data.yaml` (create)

**`configs/data.yaml` content:**
```yaml
study:
  start: "2013-01-01"      # study window start (panel; series may begin later)
  end:   "2024-12-31"      # study window end
  freq:  "ME"              # pandas month-end alias
tickers: [XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV, XLY]
vix:
  fred_series_id: "VIXCLS" # daily CBOE VIX on FRED (TO-VERIFY: fred.get_series('VIXCLS') returns daily series)
  yf_symbol: "^VIX"        # fallback source via yfinance
  disruption_threshold: 25.0   # monthly mean VIX > 25 => disruption month (H4)
momentum:
  lookback_months: 12      # 12-month time-series momentum signal
transforms:
  stl_period: 12           # monthly seasonality
  stl_robust: true
```

**Deliverables.** Two committed config files; `load_yaml` round-trips both.

**Acceptance criteria.**
- `python -c "import yaml,sys; d=yaml.safe_load(open('configs/data.yaml')); assert len(d['tickers'])==11 and d['vix']['disruption_threshold']==25.0"` exits 0.
- `sector_fred_map.yaml` has exactly 11 ticker keys equal to `data.yaml['tickers']`.

---

## F2 — `finance.py`: month-end resample, log returns, 12-month momentum

**Objective.** Implement the pure, side-effect-free core math for monthly returns and the momentum feature, independent of any download.

**Dependencies.** F1.

**Actions.**
1. Create `src/ntl_etf/data/finance.py`.
2. Implement the functions below. All accept a DataFrame already indexed by a `DatetimeIndex` of **daily or monthly** adjusted-close prices and return month-end-aligned outputs.

**File.** `src/ntl_etf/data/finance.py` (create)

**Key signatures.**
```python
import numpy as np
import pandas as pd

def to_month_end(prices: pd.Series, how: str = "last") -> pd.Series:
    """Resample a (daily or finer) adj-close price Series to month-end.
    Uses LAST observation in each calendar month: prices.resample('ME').last().
    Index becomes tz-naive month-end Timestamps. Months with no observation -> NaN (NOT ffilled)."""

def log_returns(month_end_prices: pd.Series) -> pd.Series:
    """Monthly log return r_t = ln(P_t) - ln(P_{t-1}).
    Implemented as np.log(p).diff(). First month is NaN. Do NOT fill.
    GUARD: if two consecutive prices straddle a NaN month, the return is NaN (gap not bridged)."""

def momentum_12m(month_end_prices: pd.Series, lookback: int = 12) -> pd.Series:
    """12-month time-series momentum SIGNAL feature, computed from prices:
       mom_t = ln(P_t) - ln(P_{t-lookback}) = sum of trailing `lookback` monthly log returns.
    Uses only information available AT month t (trailing), so it is a same-month feature,
    NOT a forward-looking label. Requires `lookback` prior observations -> first `lookback` rows NaN."""

def build_ticker_frame(prices_daily: pd.Series, ticker: str, sector: str,
                       lookback: int = 12) -> pd.DataFrame:
    """Assemble tidy long frame for ONE ticker:
       columns = [ticker, sector, date, adj_close, log_return, momentum_12m].
    `adj_close` is the month-end price; date is the month-end index reset to a column."""
```

**Implementation notes / pitfalls.**
- **Resample label.** Use `resample("ME")` (month-end). On older pandas (<2.2) `"ME"` is `"M"`; detect pandas version and fall back to `"M"` if `"ME"` raises (`TO-VERIFY: pandas>=2.2 in requirements -> "ME" valid`).
- **`np.log` of non-positive** prices is invalid; assert `(month_end_prices.dropna() > 0).all()` before logging.
- Momentum from prices (not from summing returns) is more robust to interior NaNs only when the endpoints exist; document that interior NaNs propagate to NaN momentum, which is the desired conservative behavior.

**Deliverables.** `finance.py` with the four functions.

**Acceptance criteria.** (covered by F11 tests)
- `log_returns(pd.Series([100,110,99]))` ≈ `[NaN, 0.09531, -0.10536]` (atol 1e-5).
- `momentum_12m` on a 13-month constant-growth (1% log/month) series returns `0.12` at month 13 (atol 1e-6) and NaN for months 1–12.
- `to_month_end` of a daily series returns one row per calendar month, index dtype is tz-naive datetime64, all timestamps are month-ends.

---

## F3 — Ragged-history handling (inception masks, no forward-fill)

**Objective.** Correctly handle ETFs whose history starts after 2013-01 so the panel never trains on synthetic pre-inception data.

**Dependencies.** F2.

**Verified inception facts** (used to sanity-check downloaded data; do **not** hard-code as the source of truth — derive masks from the data itself):

| Ticker | Inception | First full month of returns |
|---|---|---|
| XLC | **2018-06-18** (TO-VERIFY: yfinance first row ≈ 2018-06) | 2018-07 |
| XLRE | **2015-10-07** (TO-VERIFY: yfinance first row ≈ 2015-10) | 2015-11 |
| XLB XLE XLF XLI XLK XLP XLU XLV XLY | 1998-12 (pre-dates study) | 2013-01 |

> Source check the agent must run once and log: print `prices.first_valid_index()` per ticker; assert XLC ≥ `2018-06-01` and XLRE ≥ `2015-10-01`. If yfinance returns earlier rows (rare back-fill artifacts), trim to the verified inception month.

**Actions.**
1. Add a per-series validity mask: a row is valid only where `adj_close` is non-NaN **and** the date ≥ the series' first valid month.
2. **Never forward-fill prices** before computing returns (forward-filling would manufacture zero returns). The first real return is the month after the first two valid month-end prices.
3. Emit a `valid` boolean column in the tidy frame (True where `log_return` is finite). Phase P uses this to exclude pre-inception windows.
4. Record per-ticker `first_valid_month` and `n_valid_months` into the manifest (F12).

**File.** `src/ntl_etf/data/finance.py` (modify): add `add_validity_mask(df) -> df` and `first_valid_month(prices) -> pd.Timestamp`.

**Signatures.**
```python
def first_valid_month(month_end_prices: pd.Series) -> pd.Timestamp: ...
def add_validity_mask(ticker_frame: pd.DataFrame) -> pd.DataFrame:
    """Adds boolean column `valid` = np.isfinite(log_return). Does NOT drop rows
    (panel needs the full month grid); Phase P filters on `valid`."""
```

**Pitfalls.**
- Do **not** reindex shorter-history tickers onto the full 2013–2024 grid with `ffill`; reindex with `NaN` fill so missing months stay missing.
- The momentum feature for XLC is NaN until 12 months after its inception (≈ 2019-06); for XLRE until ≈ 2016-10. Phase P must drop those windows; this is expected, not a bug.

**Deliverables.** Validity-aware tidy frame builder.

**Acceptance criteria.**
- After download, `df[df.ticker=="XLC"].dropna(subset=["log_return"]).date.min()` ≥ `2018-07-31`.
- For full-history tickers, `valid` is True for all of 2014-01 … 2024-12 (allowing the first 13 months to be NaN for momentum).
- No two consecutive identical prices appear from accidental ffill (assert no run of exact-equal `adj_close` longer than data justifies — soft check, warn only).

---

## F4 — `scripts/download_finance.py` CLI → ETF parquet

**Objective.** End-to-end script: download all 11 tickers, build tidy frame, write parquet.

**Dependencies.** F2, F3.

**Actions.**
1. Create `scripts/download_finance.py`.
2. Download monthly-or-daily adjusted close via yfinance, resample to month-end, build per-ticker frames, concatenate, write parquet.
3. **yfinance adjusted-close caveat (verified):** with `auto_adjust=True` (current default) the returned `Close` is split/dividend-adjusted and there is **no** `Adj Close` column; with `auto_adjust=False` you get raw `Close` plus `Adj Close`. **Use `auto_adjust=True` and treat `Close` as the adjusted close** (this is the total-return-adjusted price we want for log returns). Record `auto_adjust=True` in the manifest so the choice is auditable. (TO-VERIFY: `yf.download("XLK", period="6mo", interval="1mo").columns` — confirm `Close` present, `Adj Close` absent under installed yfinance.)
4. Download **daily** (`interval="1d"`) then resample to month-end in our code (more robust than `interval="1mo"`, which can mis-stamp the last partial month); fall back to `interval="1mo"` if daily download is rate-limited.

**File.** `scripts/download_finance.py` (create)

**CLI (PowerShell):**
```powershell
& C:/Users/Admin/AppData/Local/Programs/Python/Python311/python.exe scripts/download_finance.py `
  --config configs/data.yaml `
  --out data/processed/etf_returns.parquet `
  --start 2013-01-01 --end 2024-12-31
```
**CLI (bash):**
```bash
python scripts/download_finance.py \
  --config configs/data.yaml \
  --out data/processed/etf_returns.parquet \
  --start 2013-01-01 --end 2024-12-31
```

**Core download snippet:**
```python
import yfinance as yf
raw = yf.download(tickers, start=start, end=end, interval="1d",
                  auto_adjust=True, group_by="ticker", progress=False, threads=True)
# per ticker: px = raw[ticker]["Close"].dropna()
#             me = to_month_end(px); frame = build_ticker_frame(me, ticker, sector)
```

**Pitfalls.**
- yfinance occasionally returns a single-level column frame for one ticker vs multi-index for many — handle both shapes (`if isinstance(raw.columns, pd.MultiIndex)`).
- Network flakiness: wrap in retry (3 attempts, exponential backoff). On total failure exit non-zero with a clear message.
- yfinance prints noisy warnings; set `progress=False` and capture logger.

**Deliverables.** `data/processed/etf_returns.parquet` (gitignored under `data/`).

**Output schema (long/tidy):**
| column | dtype | notes |
|---|---|---|
| `ticker` | str | one of the 11 |
| `sector` | str | from config |
| `date` | datetime64[ns] | month-end, tz-naive |
| `adj_close` | float | month-end adjusted close (yfinance `Close`, auto_adjust) |
| `log_return` | float | NaN at series start |
| `momentum_12m` | float | NaN for first 12 valid months |
| `valid` | bool | finite log_return |

**Acceptance criteria.**
- Parquet exists; `df.ticker.nunique()==11`; columns exactly the schema above.
- `df.date.dt.is_month_end.all()` is True; `df.date.dt.tz is None`.
- Row count ≈ 11 × number-of-months minus ragged gaps; full-history tickers have ≥ 144 monthly rows for 2013–2024.
- Spot-check: XLK 2020-03 `log_return` < 0 (COVID crash) and XLK 2020-04 `log_return` > 0 (rebound) — sign sanity, warn (don't fail) if violated since values are real.

---

## F5 — `macro.py`: FRED client wrapper + series fetch + month-end align

**Objective.** A thin, testable FRED wrapper that fetches a series, validates it, and aligns it to month-end.

**Dependencies.** F1.

**Actions.**
1. Create `src/ntl_etf/data/macro.py`.
2. Read `FRED_API_KEY` from env (`os.environ["FRED_API_KEY"]`); raise a clear, actionable error if missing (point to `.env.example`).
3. Wrap `fredapi.Fred.get_series(series_id, observation_start, observation_end)` → returns a `pd.Series` indexed by date.
4. FRED monthly IP series are stamped on the **first of the month** (`MS`). Re-stamp to **month-end** so they align with ETF returns: `s.resample("ME").last()` (each FRED monthly obs maps to its month-end). Verify exactly one observation per month after re-stamp.

**File.** `src/ntl_etf/data/macro.py` (create)

**Signatures.**
```python
from fredapi import Fred

def get_fred_client(api_key: str | None = None) -> Fred:
    """Reads FRED_API_KEY from env if api_key is None. Raises RuntimeError with
    'set FRED_API_KEY in .env (see .env.example)' if absent."""

def fetch_series(fred: Fred, series_id: str,
                 start: str, end: str) -> pd.Series:
    """get_series then validate non-empty; raise on unknown series id."""

def to_month_end_macro(s: pd.Series) -> pd.Series:
    """Re-stamp first-of-month FRED dates to month-end; assert <=1 obs/month."""
```

**Pitfalls.**
- `fredapi` raises a generic exception for a bad series ID; catch and re-raise with the offending ID so the agent can fix the mapping.
- Some IP series have a one-month **publication lag** (latest months provisional / NaN); that is acceptable for the nowcast target but record `last_valid_index` per series.
- Distinguish seasonally-adjusted (e.g., `IPMAN`) from NSA variants; **prefer SA series for the raw target**, then ALSO compute our own STL on a chosen series for the deseasonalized column (F7). Document which we use.

**Deliverables.** FRED wrapper module.

**Acceptance criteria.** (live parts behind credential guard in F11)
- With a valid key, `fetch_series(fred,"INDPRO","2013-01-01","2024-12-31")` returns ≥ 140 monthly values, all > 0.
- `to_month_end_macro` output index is all month-ends, one per month.

---

## F6 — Finalize sector → FRED IP/activity mapping (honest coverage)

**Objective.** Provide a concrete, FRED-verified mapping from each SPDR sector to a coincident activity series, and explicitly mark which sectors are **eligible** for the H5 nowcast task and which are only **best-effort proxies**.

**Dependencies.** F1, F5.

**Verified FRED series IDs** (each confirmed to exist on fred.stlouisfed.org; TO-VERIFY by `fetch_series` smoke test in F8):

| Ticker | Sector key | FRED series | Series title | Nowcast eligibility |
|---|---|---|---|---|
| XLI | industrials | **IPMAN** | IP: Manufacturing (NAICS) | **Eligible — clean** |
| XLB | materials | **IPDMAT** (+ **IPNMAT** as 2nd channel) | IP: Durable / Nondurable Goods Materials | **Eligible — good** |
| XLE | energy | **IPG211S** (primary) / **IPMINE** (broad) | IP: Mining: Oil & Gas Extraction (NAICS 211) / Mining total | **Eligible — good** |
| XLU | utilities | **IPUTIL** | IP: Electric & Gas Utilities | **Eligible — clean** |
| XLV | health | **IPG3254S** | IP: Pharma & Medicine (NAICS 3254) | **Partial — pharma-only proxy** (misses services/insurers) |
| XLP | staples | **IPNMAN** (nondurable mfg) or **IPB54200S** (business supplies) | IP: Nondurable Manufacturing | **Partial — proxy** |
| XLY | consumer disc. | **RSXFS** (Advance Retail Sales ex-food svcs) | Retail sales (activity, not IP) | **Partial — activity proxy, not IP** |
| XLK | technology | **IPG334S** (Computer & Electronic Product mfg, NAICS 334) | IP: Computer/Electronic Products | **Partial — hardware-only; misses software** |
| XLC | communication | **—** (no clean IP); candidate **IPG334S**/info-sector activity | n/a | **Ineligible — exclude from H5 (caveat)** |
| XLF | financials | **—** (no IP analog) | n/a | **Ineligible — exclude from H5 (caveat)** |
| XLRE | real estate | **—** (no IP analog); candidate `HOUST` (housing starts) as weak proxy | n/a | **Ineligible — exclude from H5 (caveat)** |

> **Overall-economy control:** also fetch **INDPRO** (IP: Total Index) as a baseline/control nowcast target available for all sectors.
> Series-ID existence is verified for: INDPRO, IPMAN, IPMINE, IPG211S, IPUTIL, IPDMAT, IPNMAT, IPG3254S (search-confirmed). `IPG334S`, `IPNMAN`, `RSXFS`, `IPB54200S`, `HOUST`, `IPG211S` exact ID spelling — **TO-VERIFY** by smoke fetch in F8; if any 404s, fall back per the rule below.

**Decision rule the plan adopts (state explicitly in the paper, W-phase):**
- **H5 (nowcast R² >> leading-return R²) is evaluated only on the eligible/partial-but-defensible sectors:** `{XLI, XLB, XLE, XLU, XLV, XLP, XLY, XLK}` plus the `INDPRO` overall control.
- **XLF, XLC, XLRE are excluded from the nowcast (H5) task** for lack of an industrial-production analog. They remain fully in the **leading-return (H1–H4)** task. Document this asymmetry as a known limitation.
- For any series ID that fails to fetch, fall back to the next-broadest valid series (e.g., `IPG211S`→`IPMINE`→`IPMAN`) and record the substitution in the manifest.

**Actions.**
1. Fill `configs/sector_fred_map.yaml` with the table above, including a per-row `nowcast_eligible: true|false`, `tier: clean|good|partial|ineligible`, optional `secondary_series`, and a one-line `caveat`.
2. Add `fallback_chain` lists where a substitution rule applies.

**File.** `configs/sector_fred_map.yaml` (modify/finalize)

**Example YAML rows:**
```yaml
- ticker: XLI
  sector: industrials
  fred_series_id: IPMAN
  nowcast_eligible: true
  tier: clean
  caveat: "Manufacturing IP is a strong coincident analog for industrials."
- ticker: XLE
  sector: energy
  fred_series_id: IPG211S
  fallback_chain: [IPG211S, IPMINE, IPB50089S]   # oil&gas -> mining -> energy total
  secondary_series: IPMINE
  nowcast_eligible: true
  tier: good
  caveat: "Oil & gas extraction IP; broad mining as fallback."
- ticker: XLF
  sector: financials
  fred_series_id: null
  nowcast_eligible: false
  tier: ineligible
  caveat: "No industrial-production analog; excluded from H5 nowcast."
```

**Deliverables.** Finalized mapping config + the explicit eligible-sector list.

**Acceptance criteria.**
- `nowcast_eligible: true` count == 8 (the eligible/partial set), `false` count == 3 (XLF, XLC, XLRE).
- Every eligible row has a non-null `fred_series_id`.
- A loader function `load_sector_fred_map()` returns a dict keyed by ticker with these fields.

---

## F7 — Stationarity transforms (STL deseasonalize + first difference)

**Objective.** Produce stationary versions of the IP target (and provide the same utilities NTL Phase N will reuse), keeping **both raw and transformed** columns and documenting which feeds which task.

**Dependencies.** F5, F6.

**Actions.**
1. In `macro.py`, implement STL deseasonalization (`statsmodels.tsa.seasonal.STL`, `period=12`, `robust=True`) and first-differencing / log-differencing utilities.
2. For each IP series produce these columns alongside `value` (raw):
   - `value_sa`: STL seasonally-adjusted level = `trend + resid` (i.e., raw minus seasonal). (If the FRED series is already SA, STL seasonal should be near-zero — log this.)
   - `value_dlog`: month-over-month log change `Δln(value)` (primary **stationary** form for modeling).
   - `value_diff`: first difference `Δvalue` (for series where log is inappropriate, e.g., can be ≤0 — IP indices are >0 so log is fine; keep diff as alt).
3. **Do not over-difference.** IP levels are trending/seasonal → use `value_dlog`. ETF **log returns are already stationary → never difference them again** (enforced in F10 registry).

**File.** `src/ntl_etf/data/macro.py` (modify): add transform functions.

**Signatures.**
```python
from statsmodels.tsa.seasonal import STL

def stl_deseasonalize(s: pd.Series, period: int = 12, robust: bool = True) -> pd.Series:
    """Return SA level = trend + resid. Requires >= 2*period observations.
    GUARD: fit STL on full available history but the panel uses TRAIN-fit transforms
    where parameters must be train-only (see F10 leakage note)."""

def dlog(s: pd.Series) -> pd.Series:        # ln(s).diff()
def first_diff(s: pd.Series) -> pd.Series:  # s.diff()

def add_macro_transforms(df: pd.DataFrame, value_col="value",
                         period=12) -> pd.DataFrame:
    """Add value_sa, value_dlog, value_diff columns."""
```

**Leakage guard (critical).** STL fit on the *entire* series uses future data, which would leak in walk-forward CV. The transform **parameters** are deterministic only for differencing (`Δlog`, which is causal). STL is **not strictly causal**. Therefore:
- Treat `value_dlog`/`value_diff` (causal) as the **model-ready stationary inputs**.
- Treat `value_sa` (STL) as **exploratory/plotting only**, OR have Phase P re-fit STL **inside each walk-forward train window** if STL-SA is used as a model input. Record this rule in F10 and F12 so Phase P enforces it. **Default: model uses `value_dlog`; STL-SA is descriptive.**

**Deliverables.** Transform utilities + documented causal-vs-noncausal policy.

**Acceptance criteria.** (F11 tests)
- `dlog(pd.Series([100,101,103]))` ≈ `[NaN, 0.00995, 0.01961]` (atol 1e-5).
- On a synthetic `trend + sin(2π t/12) + noise` series, `stl_deseasonalize` reduces the lag-12 autocorrelation of the seasonal component markedly (assert seasonal range ≥ noise range; SA series lag-12 |acf| < raw lag-12 |acf|).
- `add_macro_transforms` yields no infinities; first row of diff/dlog is NaN.

---

## F8 — `scripts/download_macro.py` CLI → FRED IP parquet

**Objective.** Fetch all mapped FRED series (eligible + control), align to month-end, attach transforms, write tidy parquet; verify every series ID resolves (live).

**Dependencies.** F5, F6, F7.

**Actions.**
1. Create `scripts/download_macro.py`.
2. Load `sector_fred_map.yaml`; for each row with a non-null `fred_series_id`, fetch via F5, applying `fallback_chain` on failure; also fetch `INDPRO` control.
3. Align to month-end (F5), attach transforms (F7), tag `sector`, `fred_series_id`, `tier`, `nowcast_eligible`.
4. Write long parquet; write a small `macro_series_resolved.json` recording which ID each sector actually used (after fallbacks) for the manifest (F12).

**File.** `scripts/download_macro.py` (create)

**CLI (PowerShell):**
```powershell
$env:FRED_API_KEY = (Get-Content .env | Select-String '^FRED_API_KEY=' ).ToString().Split('=')[1]
& C:/Users/Admin/AppData/Local/Programs/Python/Python311/python.exe scripts/download_macro.py `
  --config configs/data.yaml --map configs/sector_fred_map.yaml `
  --out data/processed/macro_ip.parquet --start 2013-01-01 --end 2024-12-31
```
**CLI (bash):**
```bash
set -a; source .env; set +a
python scripts/download_macro.py \
  --config configs/data.yaml --map configs/sector_fred_map.yaml \
  --out data/processed/macro_ip.parquet --start 2013-01-01 --end 2024-12-31
```

**Output schema:**
| column | dtype | notes |
|---|---|---|
| `sector` | str | maps to ticker via config |
| `fred_series_id` | str | resolved ID actually used |
| `date` | datetime64[ns] | month-end, tz-naive |
| `value` | float | raw IP/activity level |
| `value_sa` | float | STL-SA level (descriptive; see F7 guard) |
| `value_dlog` | float | causal stationary input (primary) |
| `value_diff` | float | causal alt |
| `tier` | str | clean/good/partial |
| `nowcast_eligible` | bool | true for the 8 eligible sectors + control |

**Pitfalls.**
- If `FRED_API_KEY` missing → exit non-zero with the `.env.example` hint (no traceback spam).
- A fallback substitution must be **logged** and reflected in `fred_series_id` and `macro_series_resolved.json`.
- Provisional latest months: keep them but mark `value` NaN where FRED returns NaN; do not impute.

**Deliverables.** `data/processed/macro_ip.parquet` + `data/processed/macro_series_resolved.json`.

**Acceptance criteria.**
- Parquet has ≥ 9 distinct `sector` values (8 eligible + `overall/INDPRO` control).
- Every `value` is > 0 where non-NaN; `date` all month-ends, tz-naive.
- `IPMAN`, `IPUTIL`, `INDPRO` each return ≥ 140 monthly rows for 2013–2024.
- `macro_series_resolved.json` lists no unresolved (null) IDs for `nowcast_eligible==true` rows.

---

## F9 — VIX fetch, monthly mean, disruption flag

**Objective.** Build the H4 disruption stratifier: month-end series of mean VIX + boolean `disruption_flag` (monthly mean VIX > 25).

**Dependencies.** F5 (FRED path) and/or F4 (yfinance path).

**Actions.**
1. Primary source: **FRED `VIXCLS`** (daily CBOE VIX; verified to exist). Fallback: yfinance `^VIX` daily `Close`.
2. Compute the **calendar-month mean of daily closes**, stamp to month-end.
3. Compute `disruption_flag = vix_mean > threshold` (threshold from `configs/data.yaml`, default 25.0).
4. Write tidy parquet.

**File.** `src/ntl_etf/data/macro.py` (add `build_vix_monthly`) + extend `scripts/download_macro.py` (or a `--vix-out` flag) to emit the VIX parquet.

**Signature.**
```python
def build_vix_monthly(vix_daily: pd.Series, threshold: float = 25.0) -> pd.DataFrame:
    """vix_daily: daily VIX close (DatetimeIndex). Returns columns
       [date(month-end), vix_mean, disruption_flag(bool)].
       vix_mean = vix_daily.resample('ME').mean()."""
```

**CLI addition (bash):**
```bash
python scripts/download_macro.py --vix-out data/processed/vix_monthly.parquet \
  --start 2013-01-01 --end 2024-12-31
```

**Output schema:**
| column | dtype | notes |
|---|---|---|
| `date` | datetime64[ns] | month-end, tz-naive |
| `vix_mean` | float | mean of daily VIX closes in month |
| `disruption_flag` | bool | `vix_mean > threshold` |

**Pitfalls.**
- FRED `VIXCLS` has occasional NaN (holidays) — `resample("ME").mean()` ignores NaN by default; fine.
- Use **daily-mean** within month (not month-end snapshot) per the proposal ("monthly average VIX"); a single month-end reading would be noisier.
- yfinance `^VIX` returns `Close` under `auto_adjust=True`; VIX has no dividends/splits so adjustment is a no-op — `Close` is correct.

**Deliverables.** `data/processed/vix_monthly.parquet`.

**Acceptance criteria.**
- 144 monthly rows for 2013-01 … 2024-12 (allowing newest provisional).
- Known disruption months flagged True: **2020-03** (COVID, mean VIX ≫ 25) and at least one of 2020-04. (TO-VERIFY numerically; 2020-03 mean VIX is well above 25.)
- Calm months flagged False: e.g., 2017 months (VIX historically low) all `disruption_flag==False`.
- `disruption_flag` dtype is bool; no NaN in `vix_mean` for non-provisional months.

---

## F10 — Calendar alignment + transform registry

**Objective.** Centralize (a) the month-end alignment contract and (b) a machine-readable registry stating **which transform feeds which task**, so Phase P and Phase M consume the right columns and never double-transform.

**Dependencies.** F2, F7, F9.

**Actions.**
1. Create `src/ntl_etf/data/transform_registry.py` (or a section of `panel.py` referenced here) exporting a dict the panel reads.
2. Encode the alignment rules and transform-per-task table below.

**File.** `src/ntl_etf/data/transform_registry.py` (create; Phase P imports it)

**Transform registry (the contract):**
```python
TRANSFORM_REGISTRY = {
    # series_role : {task : column_to_use, "stationary": bool, "causal": bool}
    "etf_return": {
        "leading":  {"col": "log_return",   "transform": "none", "note": "already stationary; DO NOT difference"},
        "nowcast":  {"col": None,            "note": "returns are the LEADING target only"},
    },
    "etf_momentum": {
        "leading":  {"col": "momentum_12m", "transform": "trailing_12m", "note": "same-month trailing feature; baseline signal"},
    },
    "ip_target": {
        "nowcast":  {"col": "value_dlog",   "transform": "dlog", "causal": True,
                     "note": "model target/feature stationary form"},
        "descriptive": {"col": "value_sa",  "transform": "STL", "causal": False,
                        "note": "plots only OR re-fit per train window in Phase P"},
    },
    "ntl_feature": {  # produced in Phase N; alignment rule lives here for one source of truth
        "leading":  {"transform": "dlog_or_diff", "lag": 1, "causal": True,
                     "note": "NTL month t -> predict return month t+1 (release-lag, F12)"},
        "nowcast":  {"transform": "dlog_or_diff", "lag": 0, "causal": True,
                     "note": "contemporaneous NTL allowed for nowcast (no forward target)"},
    },
}
```

**Alignment rules (documented constants).**
- All series share one tz-naive month-end `DatetimeIndex`. Phase P builds the master month grid `2013-01-31 … 2024-12-31` and **left-joins** each series; missing → NaN (never ffill).
- **Leading task:** target is `log_return` at month `t+1`; NTL feature is month `t` (lag 1). This is the release-lag guard (formalized in F12). Momentum feature uses month `t` (trailing, same-month).
- **Nowcast task:** target is `ip_target.value_dlog` at month `t`; NTL feature is month `t` (contemporaneous, lag 0) — permitted because there is no forward target to leak into.
- **Normalization** (mean/std, min/max) is **out of scope here** and must be fit **train-only** in Phase M/P. This phase emits raw + causal-stationary columns only.

**Deliverables.** Importable registry module + the written alignment contract.

**Acceptance criteria.**
- `from ntl_etf.data.transform_registry import TRANSFORM_REGISTRY` imports cleanly.
- `TRANSFORM_REGISTRY["etf_return"]["leading"]["transform"] == "none"` (asserts no double-differencing of returns).
- `TRANSFORM_REGISTRY["ntl_feature"]["leading"]["lag"] == 1` and `["nowcast"]["lag"] == 0`.

---

## F11 — Offline unit tests on fixtures + credential-guarded smoke test

**Objective.** Prove the math and alignment with deterministic, network-free tests; gate live tests on credentials.

**Dependencies.** F2, F3, F7, F9.

**Actions.**
1. Create `tests/test_finance.py`, `tests/test_macro.py`, `tests/test_vix.py` using `pytest`.
2. Build tiny in-memory fixtures (no I/O). Use the constructed values in earlier acceptance criteria.
3. Add a `@pytest.mark.skipif(not os.getenv("FRED_API_KEY"), reason="no FRED key")` live smoke test that fetches `INDPRO` (3 rows) and asserts shape; likewise a yfinance smoke test marked `network` (skipped in CI by default).

**Files.** `tests/test_finance.py`, `tests/test_macro.py`, `tests/test_vix.py` (create)

**Required offline test cases.**
| Test | Assertion |
|---|---|
| `test_log_return_values` | `log_returns([100,110,99])` ≈ `[nan,0.09531,-0.10536]` |
| `test_momentum_constant_growth` | 1%/mo log growth → `momentum_12m[12]==0.12`±1e-6; first 12 NaN |
| `test_month_end_resample` | daily fixture → one row/month, all `is_month_end`, tz-naive |
| `test_no_ffill_on_gap` | a NaN month between prices → adjacent return is NaN, not bridged |
| `test_inception_mask_xlc` | synthetic XLC starting 2018-06 → first valid return ≥ 2018-07; pre-rows NaN |
| `test_dlog` | `dlog([100,101,103])` ≈ `[nan,0.00995,0.01961]` |
| `test_stl_reduces_seasonality` | SA series lag-12 |acf| < raw lag-12 |acf| on synthetic seasonal data |
| `test_disruption_flag` | `build_vix_monthly` with a month mean 30 → flag True; mean 15 → False; boundary 25 → False (strict `>`) |
| `test_disruption_flag_dtype` | output `disruption_flag.dtype == bool`, no NaN in `vix_mean` |
| `test_registry_no_double_diff` | `TRANSFORM_REGISTRY["etf_return"]["leading"]["transform"]=="none"` |

**Live (guarded) tests.**
- `test_fred_smoke` (skip if no `FRED_API_KEY`): `fetch_series(fred,"INDPRO","2024-01-01","2024-03-31")` → ≥ 2 rows, all > 0.
- `test_yf_smoke` (mark `network`): `yf.download("XLK", period="3mo", interval="1mo")` non-empty, has `Close`.

**Deliverables.** Passing `pytest` suite (offline subset green with no network/key).

**Acceptance criteria.**
- `pytest tests/test_finance.py tests/test_macro.py tests/test_vix.py -q` passes with no network and no `FRED_API_KEY` (live tests skipped, not failed).
- Coverage includes every function in `finance.py`/`macro.py` listed above (verify via `pytest --cov=src/ntl_etf/data` ≥ 80% on these two modules).

---

## F12 — Release-lag + leakage contract + manifest for Phase P

**Objective.** Emit the explicit, machine-readable contract Phase P (`docs/devplan/03-panel-and-splits.md`) must enforce, and a manifest summarizing what was downloaded.

**Dependencies.** F4, F8, F9, F10.

**Actions.**
1. Write `data/processed/financial_macro_manifest.json` capturing provenance and the alignment contract.
2. Document (here and in the manifest) the **release-lag rule** the panel must apply.

**Release-lag contract (the rule Phase P enforces; sourced from Phase A.1 / `docs/devplan/01-data-ntl.md`):**
- VNP46A3 for month `t` is published ~30–45 days after month-end, so it is **not knowable** until well into month `t+1`.
- **Leading task:** NTL features for month `t` may predict ETF `log_return` at month **`t+1`** (and 2–3 month horizons `t+2`, `t+3`). The panel must shift the NTL feature forward by ≥ 1 month relative to the return target. **No contemporaneous NTL on the leading task.**
- **Nowcast task:** NTL for month `t` may be used to nowcast IP `value_dlog` at month `t` (contemporaneous), because the IP target itself is also only published with a lag — there is no *forward* target being predicted, so no look-ahead. State that the nowcast is an *as-if-released-simultaneously* coincident estimate, not a real-time one; flag this as a limitation in W-phase.
- Returns and momentum carry **no extra lag** beyond the above (returns are the target; momentum is a same-month trailing feature).

**`financial_macro_manifest.json` fields:**
```json
{
  "generated_utc": "<iso8601>",
  "study": {"start": "2013-01-01", "end": "2024-12-31", "freq": "ME"},
  "etf": {
    "source": "yfinance", "auto_adjust": true, "interval": "1d->ME",
    "tickers": ["XLB","XLC","XLE","XLF","XLI","XLK","XLP","XLRE","XLU","XLV","XLY"],
    "first_valid_month": {"XLC": "2018-07-31", "XLRE": "2015-11-30", "...": "2013-..."}
  },
  "macro": {
    "source": "fredapi", "resolved_series": "<contents of macro_series_resolved.json>",
    "nowcast_eligible_sectors": ["industrials","materials","energy","utilities","health","staples","consumer_disc","technology","overall"],
    "nowcast_excluded_sectors": ["financials","communication","real_estate"]
  },
  "vix": {"source": "FRED:VIXCLS", "threshold": 25.0, "agg": "monthly_mean_of_daily_close"},
  "alignment": {
    "calendar": "tz-naive month-end DatetimeIndex; left-join to master grid; no ffill",
    "leading_lag_months": 1,
    "nowcast_lag_months": 0,
    "etf_return_transform": "none (already stationary)",
    "ip_target_transform": "value_dlog (causal)",
    "stl_policy": "STL-SA is descriptive OR refit per train window in Phase P"
  },
  "seeds": {"global_seed": 42}
}
```

**Deliverables.** `data/processed/financial_macro_manifest.json` (and a short prose copy of the contract appended to this doc's appendix in the repo).

**Acceptance criteria.**
- Manifest validates as JSON; `alignment.leading_lag_months==1`, `alignment.nowcast_lag_months==0`.
- `nowcast_excluded_sectors` == `["financials","communication","real_estate"]`.
- `etf.first_valid_month["XLC"]` ≥ `2018-07-31`, `["XLRE"]` ≥ `2015-11-30`.
- Phase P task that builds the panel (P-prefixed) lists F12 as a dependency and reads this manifest.

---

## Appendix A — Files created/modified in this phase

| Path | Task | Kind |
|---|---|---|
| `configs/data.yaml` | F1 | create |
| `configs/sector_fred_map.yaml` | F1, F6 | create/finalize |
| `src/ntl_etf/data/finance.py` | F2, F3 | create |
| `src/ntl_etf/data/macro.py` | F5, F7, F9 | create |
| `src/ntl_etf/data/transform_registry.py` | F10 | create |
| `scripts/download_finance.py` | F4 | create |
| `scripts/download_macro.py` | F8, F9 | create |
| `tests/test_finance.py` | F11 | create |
| `tests/test_macro.py` | F11 | create |
| `tests/test_vix.py` | F11 | create |
| `data/processed/etf_returns.parquet` | F4 | output (gitignored) |
| `data/processed/macro_ip.parquet` | F8 | output (gitignored) |
| `data/processed/macro_series_resolved.json` | F8 | output |
| `data/processed/vix_monthly.parquet` | F9 | output (gitignored) |
| `data/processed/financial_macro_manifest.json` | F12 | output |

## Appendix B — TO-VERIFY checklist (run once, log results)

1. `yf.download("XLK", period="6mo", interval="1mo").columns` → confirm `Close` present, `Adj Close` absent under `auto_adjust=True` (default).
2. `prices.first_valid_index()` for XLC ≈ 2018-06, XLRE ≈ 2015-10 (matches verified inception 2018-06-18 / 2015-10-07).
3. `fetch_series(fred, sid, ...)` for each of: INDPRO, IPMAN, IPUTIL, IPMINE, IPG211S, IPDMAT, IPNMAT, IPG3254S (confirmed to exist) **and** the not-yet-confirmed `IPG334S`, `IPNMAN`, `RSXFS`, `IPB54200S`, `HOUST` — apply `fallback_chain` on any 404.
4. `fred.get_series("VIXCLS")` returns a daily series spanning ≥ 2013–2024.
5. pandas ≥ 2.2 so `"ME"` resample alias is valid; else fall back to `"M"`.
