# Phase A.1 — NTL Acquisition & Raster Feature Extraction

File: `docs/devplan/01-data-ntl.md`
Task-ID prefix: **N**

This phase acquires NASA Black Marble **VNP46A3** monthly nighttime-light (NTL) composites, extracts per-region monthly radiance features, and writes a tidy feature table that downstream phases consume. Every task is config-driven, cached, and leakage-aware. The single most important leakage guard in this phase is the **`data_release_date` column** (Task N7), which encodes the ~30–45 day VNP46A3 release lag so that Phase P/F can align features without look-ahead.

---

## Phase invariants (apply to every task)

| Concern | Rule |
|---|---|
| Study range | NTL months **2013-01 through 2024-12** inclusive (144 months). VNP46A3 begins 2012-01; the first study year is 2013 to skip early-mission quality gaps. |
| Raw data | Raw HDF5 tiles under `data/raw/ntl/` (gitignored). |
| Intermediate | Mosaicked/clipped GeoTIFFs and per-tile cache under `data/interim/ntl/` (gitignored). |
| Processed | Final feature table at `data/processed/ntl_features.parquet` (gitignored; only its schema/manifest is tracked). |
| Secrets | Earthdata bearer token from env var `EARTHDATA_TOKEN` (preferred) or `.netrc`; never hard-coded, never committed. See Task N1. |
| Determinism | No randomness in this phase. Extraction is pure given inputs; assert byte-stable feature outputs across re-runs in tests. |
| Units | VNP46A3 radiance is in **nW·cm⁻²·sr⁻¹**; v2.0 stores it already scaled (`scale_factor=1.0`). Do not re-scale. Document the unit in every output. |

---

## N1 — Earthdata authentication & credential plumbing

**Objective:** Make a NASA Earthdata bearer token available to the NTL pipeline via environment, with a `.netrc` fallback, all gitignored.

**Dependencies:** S2 (repo bootstrap, `.gitignore`, `.env.example`), S5 (config loader `src/ntl_etf/utils/config.py`).

**Actions:**
1. Document the manual one-time human/agent step in this file (the autonomous agent must perform it before any download): register/login at `https://urs.earthdata.nasa.gov`, then **Generate Token** under the user profile → copy the bearer token (valid ~60 days).
2. Add `EARTHDATA_TOKEN=` and `EARTHDATA_USER=` / `EARTHDATA_PASS=` placeholders to `.env.example` (modify file from S2).
3. Implement credential resolution in `src/ntl_etf/data/ntl.py`:

```python
def resolve_earthdata_token() -> str:
    """Return a NASA Earthdata bearer token.
    Order: env EARTHDATA_TOKEN -> ~/.netrc (machine urs.earthdata.nasa.gov).
    Raises RuntimeError with a remediation message if none found."""
```
4. For the `.netrc` fallback, generate `~/.netrc` (or `%USERPROFILE%\_netrc` on Windows) from `EARTHDATA_USER`/`EARTHDATA_PASS` if the token is absent; restrict perms (`chmod 600`) on POSIX. Note Windows uses `_netrc` (no leading dot).
5. Confirm `.env`, `.netrc`, `_netrc` are in `.gitignore`.

**Deliverables:** `resolve_earthdata_token()` in `ntl.py`; `.env.example` placeholders.

**Acceptance Criteria:**
- `resolve_earthdata_token()` returns the token when `EARTHDATA_TOKEN` is set; raises a clear `RuntimeError` (mentioning the Earthdata token URL) when no credential source exists.
- `git check-ignore .env .netrc _netrc` returns all three (test asserts via `pathlib`/`subprocess`).
- No token literal appears anywhere under `src/` or `configs/` (test greps for a token-like pattern and asserts none).

---

## N2 — Download method decision (blackmarblepy vs earthaccess raw tiles)

**Objective:** Select and justify the VNP46A3 download/mosaic method; define a primary path and a fallback.

**Dependencies:** N1.

**Decision (RECOMMENDED — primary): `blackmarblepy`.**

Justification (verified against the World Bank BlackMarblePy docs):
- `blackmarblepy` (`pip install blackmarblepy`, import as `import blackmarble as bm`) automates the entire chain for VNP46A3: it resolves which 10°×10° tiles cover a region of interest (a GeoDataFrame), downloads the raw HDF5 from NASA LAADS DAAC, converts HDF-EOS5 → georeferenced raster, mosaics adjacent tiles, and returns an **`xarray.Dataset`** (`bm.raster`) or a **zonal-stats `pandas.DataFrame`** (`bm.extract`). It accepts a bearer `token=` and skips already-downloaded files (`output_skip_if_exists=True`). This removes ~80% of the bespoke tile-grid/mosaic code we would otherwise write.
- Default variable for VNP46A3 is `NearNadir_Composite_Snow_Free` — exactly the layer we want (Task N4).
- We will primarily use **`bm.raster(...)`** (not `bm.extract`) so we control the quality masking and feature math ourselves with rasterio/numpy (Tasks N5–N6) for full transparency and reproducibility, and so the unit tests in N9 cover *our* math rather than a black box.

Verified primary API signature:
```python
bm.raster(
    gdf,                              # GeoDataFrame ROI (EPSG:4326)
    product_id="VNP46A3",
    date_range=[date(2013,1,1), ...], # list of monthly anchor dates (use day=1)
    token=<bearer>,
    variable="NearNadir_Composite_Snow_Free",   # default for VNP46A3
    drop_values_by_quality_flag=[],   # we mask ourselves; see N5
    output_directory=Path("data/raw/ntl"),
    output_skip_if_exists=True,
) -> xarray.Dataset
```

**FALLBACK (secondary): `earthaccess` + raw HDF5.**
If `blackmarblepy` fails to install, breaks on a NASA endpoint change, or its mosaic logic misbehaves:
- Use `earthaccess` (`pip install earthaccess`; `earthaccess.login(strategy="environment")` reads `EARTHDATA_TOKEN`/`EARTHDATA_USERNAME`/`EARTHDATA_PASSWORD`).
- `earthaccess.search_data(short_name="VNP46A3", version="2", temporal=(...), bounding_box=(W,S,E,N))` → `earthaccess.download(results, "data/raw/ntl")`.
- Then open each HDF-EOS5 with `rasterio` (subdataset path form: `HDF5:"<file>"://HDFEOS/GRIDS/VNP_Grid_DNB/Data_Fields/NearNadir_Composite_Snow_Free`) or `h5py`, attach the tile's lat/lon grid, mosaic with `rasterio.merge.merge`. The HDF tile grid id is in the filename (`...h##v##...`).
- The fallback gives us the exact same downstream interface (a clipped masked array per region-month), so N5–N7 are method-agnostic.

**Actions:**
1. Implement a thin adapter in `src/ntl_etf/data/ntl.py` exposing one internal function regardless of backend:
```python
def fetch_region_month_raster(gdf_row, month: date, token: str,
                              backend: str = "blackmarblepy",
                              raw_dir: Path = Path("data/raw/ntl"),
                              ) -> "xarray.Dataset | dict":
    """Return the NearNadir_Composite_Snow_Free DataArray AND its quality + num-obs
    DataArrays, clipped to the row's bbox, for one month. backend in
    {'blackmarblepy','earthaccess'}. Caches raw HDF5; skips re-download."""
```
2. Record the chosen backend in `configs/ntl.yaml` (key `download_backend`), default `blackmarblepy`.

**Deliverables:** `fetch_region_month_raster()` adapter; `configs/ntl.yaml` with `download_backend`.

**Acceptance Criteria:**
- `pip install blackmarblepy` succeeds in the project venv on Windows/CPU (record version in `requirements.txt`); `import blackmarble` works (test, skipped if not installed).
- Adapter returns, for a synthetic stub backend, an object exposing the composite, quality, and num-obs arrays plus an affine/transform (unit-tested with a fake backend, no network).

---

## N3 — Quality layers & VNP46A3 product facts (reference, no code)

**Objective:** Pin down the exact science-dataset layers, quality semantics, scale, and fill values so N5–N6 mask correctly.

**Dependencies:** N2.

**VNP46A3 facts (verified; mark any uncertain item TO-VERIFY against the product file itself):**

| Item | Value | Source / verification |
|---|---|---|
| Product | VNP46A3 — VIIRS/NPP Lunar BRDF-Adjusted Nighttime Lights **Monthly** L3, 15 arc-second (~500 m) global, Collection v2.0 | LAADS DAAC product page |
| Temporal start | 2012-01 (monthly). **Use 2013-01 onward.** | NASA Earthdata catalog |
| Tile layout | 10°×10° HDF-EOS5 tiles on `h##v##` sinusoidal-style Black Marble grid; a region may span ≥2 tiles → mosaic. | Black Marble User Guide v1.2 / C2.0 |
| Primary radiance layer | `NearNadir_Composite_Snow_Free` (gap-filled, snow-free, near-nadir composite) | blackmarblepy default for VNP46A3 |
| Std-dev layer | `NearNadir_Composite_Snow_Free_Std` (optional extra) | User Guide |
| Num-observations layer | `NearNadir_Composite_Snow_Free_Num` (count of obs in composite) | User Guide |
| Quality layer | `NearNadir_Composite_Snow_Free_Quality` (the **Mandatory_Quality_Flag** for the monthly composite) | User Guide |
| Quality values (monthly) | `0` = good-quality (≥ several valid obs); `1` = poor-quality / few observations (use with caution); `2` = gap-filled / no retrieval. **TO-VERIFY**: open one downloaded tile and print the unique values + the layer's `flag_values`/`flag_meanings` HDF attributes before trusting this mapping. | User Guide C2.0 (semantics) |
| Scale / offset | v2.0: `scale_factor = 1.0`, `add_offset = 0.0` (radiance already in nW·cm⁻²·sr⁻¹). **Still read the attribute at runtime** and apply it generically rather than assuming. | blackmarblepy dataset attrs |
| Fill / no-data | `_FillValue = 65535` (uint) in raw HDF; blackmarblepy surfaces it as `NaN`. Treat both `== _FillValue` and `NaN` as no-data. | Product metadata / blackmarblepy |
| Units | nW·cm⁻²·sr⁻¹ | User Guide |

**Actions:**
1. Encode these as constants in `src/ntl_etf/data/ntl.py`:
```python
VNP46A3_VAR = "NearNadir_Composite_Snow_Free"
VNP46A3_QF  = "NearNadir_Composite_Snow_Free_Quality"
VNP46A3_NUM = "NearNadir_Composite_Snow_Free_Num"
GOOD_QUALITY_FLAGS = (0,)          # default keep-set; configurable
DROP_QUALITY_FLAGS = (1, 2)        # poor + gap-filled dropped by default
FILL_VALUES = (65535,)             # plus np.isnan
RADIANCE_UNITS = "nW/cm^2/sr"
```
2. Add a one-time **attribute audit** helper that prints `scale_factor`, `add_offset`, `_FillValue`, and quality `flag_values`/`flag_meanings` from the first downloaded tile and writes them to `data/interim/ntl/_layer_attrs.json`. The pipeline must read scale/offset/fill from this audit, not hard-code them.

**Deliverables:** Constants block; `audit_layer_attributes()` writing `_layer_attrs.json`.

**Acceptance Criteria:**
- Constants importable; `audit_layer_attributes()` on a synthetic HDF/array fixture returns the expected attr dict (unit test).
- A `# TO-VERIFY` comment in code instructs confirming the quality mapping against `flag_meanings` on first real download; the smoke test (N10) asserts the real tile's quality layer contains only values in `{0,1,2}`.

---

## N4 — ROI catalog: `configs/regions.yaml`

**Objective:** Produce a concrete, hypothesis-driven region→sector mapping with bounding boxes, covering all 11 SPDR sectors and including the anchor cases.

**Dependencies:** S5 (config loader).

**Actions:** Create `configs/regions.yaml`. Schema per region:
```yaml
- name: pearl_river_delta
  sector: XLI
  anchor: true                 # core hypothesis region
  geometry: bbox               # 'bbox' | 'shapefile'
  bbox: [112.5, 22.0, 114.6, 23.7]   # [min_lon, min_lat, max_lon, max_lat] (EPSG:4326)
  shapefile: null              # path under data/external/ when geometry: shapefile
  rationale: "Shenzhen/Guangzhou/Dongguan manufacturing cluster"
  hypothesis: H2
```

Notes the agent must record at the top of the file:
- **All pairings are HYPOTHESES.** They are validated/pruned by the Phase P correlation pre-screen (task `P-screen`) with strict no-look-ahead. Bounding boxes are approximate; refine only with shapefiles when a bbox is too coarse (e.g. the Permian Basin is a polygon, not a rectangle — start with a bbox, optionally upgrade to a shapefile in `data/external/`).
- All coordinates are decimal degrees, lon/lat order `[W, S, E, N]`.

**Starter catalog (use these values; plausible/approximate):**

| name | sector | bbox `[W,S,E,N]` | anchor | rationale |
|---|---|---|---|---|
| pearl_river_delta | XLI | `[112.5, 22.0, 114.6, 23.7]` | ✔ | Shenzhen–Guangzhou–Dongguan manufacturing (H2) |
| yangtze_river_delta | XLI | `[119.8, 30.4, 122.2, 32.2]` | ✔ | Shanghai–Suzhou–Ningbo industrial cluster (H2) |
| permian_basin | XLE | `[-104.5, 31.0, -100.8, 33.0]` | ✔ | TX/NM oil basin, gas flaring NTL (H3) |
| us_gulf_coast_refining | XLE | `[-95.6, 28.8, -91.0, 30.4]` | ✔ | Houston–Port Arthur–Lake Charles refinery corridor |
| atlanta_metro | XLY | `[-84.7, 33.5, -84.0, 34.1]` | ✔ | Metro retail/consumer activity |
| phoenix_metro | XLY | `[-112.4, 33.2, -111.6, 33.8]` | ✔ | Sunbelt retail growth |
| dallas_ftworth_metro | XLY | `[-97.5, 32.5, -96.5, 33.2]` | ✔ | Metro retail/consumer |
| ruhr_rhine_industrial | XLB | `[6.4, 50.8, 7.7, 51.6]` | | Ruhr chemicals/materials (BASF region nearby) |
| antofagasta_copper | XLB | `[-70.6, -24.3, -68.5, -22.5]` | | Chilean copper mining belt (materials) |
| wall_street_midtown_nyc | XLF | `[-74.05, 40.6, -73.85, 40.85]` | | Financial-district activity proxy |
| city_of_london | XLF | `[-0.20, 51.45, 0.05, 51.58]` | | London financial core |
| silicon_valley | XLK | `[-122.2, 37.2, -121.7, 37.55]` | | Bay Area tech corridor |
| seattle_redmond_tech | XLK | `[-122.4, 47.4, -122.0, 47.8]` | | Seattle/Redmond tech (H3 single-dominant) |
| shenzhen_tech_hub | XLK | `[113.8, 22.5, 114.3, 22.8]` | | Hardware/electronics manufacturing-tech |
| three_gorges_yangtze_hydro | XLU | `[110.5, 30.6, 111.5, 31.2]` | | Major hydro generation region |
| ercot_houston_load | XLU | `[-95.8, 29.5, -95.0, 30.1]` | | Electric-load/utility demand metro |
| la_long_beach_port | XLI | `[-118.4, 33.6, -118.0, 33.9]` | | Port-complex throughput (industrials/logistics) |
| singapore_port | XLE | `[103.6, 1.2, 104.1, 1.5]` | | Major bunkering/refining/shipping hub |
| rotterdam_port | XLE | `[3.9, 51.85, 4.6, 52.0]` | | Europe's largest oil/petrochemical port |
| houston_med_center | XLV | `[-95.41, 29.69, -95.38, 29.72]` | | Texas Medical Center (healthcare hub) |
| boston_longwood_med | XLV | `[-71.12, 42.32, -71.09, 42.35]` | | Longwood medical/pharma cluster |
| nj_pharma_corridor | XLV | `[-74.7, 40.3, -74.2, 40.7]` | | NJ pharmaceutical manufacturing belt |
| amazon_midwest_fulfillment | XLP | `[-90.4, 38.5, -89.9, 38.9]` | | Consumer-staples distribution/logistics (proxy) |
| central_valley_agri | XLP | `[-121.0, 36.3, -119.2, 37.6]` | | CA Central Valley food production (staples) |
| dallas_data_center_alley | XLC | `[-97.05, 32.9, -96.7, 33.2]` | | Comms/data-center density (DFW) |
| northern_virginia_data_centers | XLC | `[-77.55, 38.9, -77.3, 39.1]` | | Ashburn "Data Center Alley" (XLC comms) |
| sunbelt_housing_construction | XLRE | `[-112.4, 33.2, -111.6, 33.8]` | | Phoenix real-estate/construction expansion |
| atlanta_cre | XLRE | `[-84.55, 33.7, -84.3, 33.95]` | | Atlanta commercial real-estate core |

Coverage check (must hold): every sector in {XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV, XLY} appears ≥1×; XLI/XLE/XLY/XLK/XLV have ≥2 candidates.

**Deliverables:** `configs/regions.yaml` (~28 regions, all 11 sectors covered, anchors flagged).

**Acceptance Criteria:**
- A loader test parses the YAML, asserts each entry has `name, sector, geometry, bbox|shapefile, rationale`; asserts unique `name`s; asserts `sector ∈ the 11`; asserts `set(sectors) == {11 SPDR tickers}`.
- bbox sanity: `W<E`, `S<N`, all lon∈[-180,180], lat∈[-90,90] (test asserts for every row).
- ≥5 anchors present including all six named anchor regions above.

---

## N5 — Raster open, mosaic, clip, and quality-mask

**Objective:** Turn the per-region-month raster from N2 into a clean masked radiance array (fill + bad-quality removed), handling multi-tile mosaics and bbox/shapefile clipping.

**Dependencies:** N2, N3, N4.

**Actions:** Implement in `src/ntl_etf/data/ntl.py`:
```python
def load_clean_radiance(region: dict, month: date, token: str, cfg: dict
                        ) -> "numpy.ma.MaskedArray":
    """1) fetch composite+quality+num arrays (N2);
       2) mosaic tiles if region spans >1 tile (rasterio.merge.merge);
       3) clip to region bbox (rasterio.windows / .mask.mask for shapefile);
       4) apply scale_factor/add_offset from layer attrs (N3);
       5) mask fill values and NaN;
       6) mask pixels whose quality flag in cfg['drop_quality_flags'];
       returns a masked float32 array (nW/cm^2/sr) of clipped radiance."""
```
Implementation details:
- Mosaic: when `bm.raster` already mosaics, this is a no-op; in the `earthaccess` fallback call `rasterio.merge.merge([...])`.
- Clip bbox: build a `rasterio.windows.from_bounds(*bbox, transform=...)` window; for shapefiles use `rasterio.mask.mask(geometry, crop=True)` after `geopandas.read_file`.
- Masking order matters: apply scale/offset, then mask `data == fill` and `np.isnan(data)`, then mask `quality ∈ drop_quality_flags`. Combine into one boolean mask.
- Negative radiance (rare instrument artifact) → clip to 0 or mask per `cfg['clip_negative']` (default: mask). Record the choice.

**Deliverables:** `load_clean_radiance()`; optional cached clipped GeoTIFF written to `data/interim/ntl/<region>/<YYYY-MM>.tif` (controlled by `cfg['cache_interim']`).

**Acceptance Criteria:**
- On a synthetic 2-tile fixture straddling a tile boundary, mosaic+clip returns the correct stitched window shape (unit test, no network).
- Pixels with fill value and with quality∈{1,2} are masked; `masked_array.count()` equals the number of good pixels in the fixture (unit test).
- Scale/offset application verified: a fixture with `scale_factor=0.1` yields values 10× the raw integers (unit test).

---

## N6 — Per-region-per-month feature computation

**Objective:** Reduce each clean masked array to the monthly feature vector.

**Dependencies:** N5.

**Features (computed over unmasked/good pixels only):**

| column | definition | notes |
|---|---|---|
| `ntl_mean` | mean radiance | core |
| `ntl_median` | median radiance | robust core |
| `ntl_std` | std radiance | dispersion |
| `ntl_sum` | sum of radiance ("sum of lights", SoL) | total economic luminance proxy |
| `ntl_p90` | 90th percentile radiance | bright-core intensity |
| `ntl_lit_count` | # pixels with radiance > `cfg['lit_threshold']` (default 0.5 nW/cm²/sr) | lit-area proxy |
| `ntl_lit_frac` | `ntl_lit_count / n_valid_pixels` | normalizes for ROI size/coverage |
| `n_valid_pixels` | count of good pixels after masking | data-quality/coverage diagnostic |
| `frac_masked` | masked pixels / total pixels | QA: high values flag unreliable months |

**Actions:** Implement:
```python
def compute_ntl_features(clean: "numpy.ma.MaskedArray", cfg: dict) -> dict:
    """Return the feature dict above. Uses np.ma reductions so masked
    (fill/bad-quality) pixels are excluded. If n_valid_pixels == 0,
    return all-NaN feature row and frac_masked=1.0 (do NOT raise)."""
```
- Use `np.ma.mean/median/std/sum`, `np.percentile(clean.compressed(), 90)`.
- `lit_threshold` configurable in `configs/ntl.yaml`.
- Guard: empty/all-masked month → NaN features, `frac_masked=1.0` (Phase P decides whether to drop).

**Deliverables:** `compute_ntl_features()`.

**Acceptance Criteria (numeric, unit-tested on a hand-computed fixture):**
- For array `[[1,2,fill],[3,4,bad]]` (fill masked, one bad-quality masked) over good values `{1,2,3,4}`: `ntl_mean==2.5`, `ntl_median==2.5`, `ntl_std==pytest.approx(1.118, abs=1e-3)` (population std, `ddof=0`), `ntl_sum==10`, `n_valid_pixels==4`, `frac_masked==pytest.approx(2/6)`.
- `ntl_lit_count` with threshold 2.5 over `{1,2,3,4}` == 2; `ntl_lit_frac == 0.5`.
- All-masked input → every feature NaN, `frac_masked==1.0`, no exception.

---

## N7 — Release-lag stamp (`data_release_date`) — leakage guard

**Objective:** Attach the realistic VNP46A3 availability date to each month so Phase F/P can prevent look-ahead. **This is the critical leakage guard of the phase.**

**Dependencies:** N6.

**Rule:** VNP46A3 for calendar month *M* is released ~30–45 days after *M* ends. Model the release as the **last day of month M+1** by default (a conservative ~30–60 day lag covering the published range), configurable via `cfg['release_lag_days']`.

**Actions:** Implement:
```python
def stamp_release_date(year: int, month: int, lag_days: int = 45) -> dict:
    """Return {'date': <month-end of M>, 'data_release_date': <month-end of M> + lag_days}.
    Default policy: data_release_date = month_end(M) + 45 days, which lands in M+1.
    This means month M's NTL may only be used to predict M+1 (or later) returns."""
```
- `date` = month-end of M (the observation month).
- `data_release_date` = `month_end(M) + lag_days` (default 45).
- Document explicitly in this file: **Phase P/F MUST require `data_release_date <= prediction_decision_date`**; i.e. NTL of month M predicts the return of month M+1, never the return of month M (which is the H5 nowcast caveat: even the coincident nowcast can only use M's NTL after release, so the nowcast is technically a *late* nowcast — note this for Phase E/W).

**Deliverables:** `stamp_release_date()`; a prose note in this doc that Phase P consumes `data_release_date`.

**Acceptance Criteria:**
- For 2013-03 with `lag_days=45`: `date==2013-03-31`, `data_release_date` falls in 2013-05 (i.e. `> 2013-04-30`) — assert `data_release_date.month==5` (unit test).
- For every output row, `data_release_date > date` strictly (test asserts over the full table).

---

## N8 — Download/extraction CLI & feature-table assembly

**Objective:** Orchestrate N2/N5/N6/N7 over all regions × all 144 months into the processed parquet, with caching, retry/backoff, and a manifest.

**Dependencies:** N1–N7.

**Actions:**
1. Build the orchestrator entry point in `src/ntl_etf/data/ntl.py`:
```python
def build_ntl_features(regions_yaml: Path, ntl_cfg: Path,
                       start="2013-01", end="2024-12",
                       out_parquet=Path("data/processed/ntl_features.parquet"),
                       manifest=Path("data/interim/ntl/manifest.json")) -> "pandas.DataFrame":
    """For each region × month: ensure raw tile cached (skip if present),
    load_clean_radiance -> compute_ntl_features -> stamp_release_date.
    Append row; update manifest. Idempotent & resumable."""
```
2. **Caching / resume:** before download, check raw HDF5 presence and check the manifest; skip completed (region,month). `output_skip_if_exists=True` on `bm.raster`.
3. **Retry/backoff:** wrap network calls in retry (e.g. `tenacity` or hand-rolled) — max 5 attempts, exponential backoff (2,4,8,16,32 s), retry on transient HTTP/timeout; surface 401/403 immediately (auth, not transient) with a token-refresh hint.
4. **Manifest** (`data/interim/ntl/manifest.json`): list of `{region, year, month, tiles:[h##v##...], status:'ok'|'failed'|'empty', raw_files:[...], fetched_at}`. This is the only tracked NTL artifact (tracked via a committed copy under `experiments/` if Phase S requires; raw data stays gitignored).
5. **CLI** `scripts/download_ntl.py` (argparse): `--regions configs/regions.yaml --config configs/ntl.yaml --start 2013-01 --end 2024-12 --backend blackmarblepy --out data/processed/ntl_features.parquet [--region <name>] [--dry-run]`.

Commands:
```powershell
# PowerShell
$env:EARTHDATA_TOKEN = (Get-Content .env | Select-String '^EARTHDATA_TOKEN=' | %{ $_ -replace '^EARTHDATA_TOKEN=','' })
C:/Users/Admin/AppData/Local/Programs/Python/Python311/python.exe scripts/download_ntl.py --regions configs/regions.yaml --config configs/ntl.yaml --start 2013-01 --end 2024-12
```
```bash
# Git Bash
export EARTHDATA_TOKEN=$(grep '^EARTHDATA_TOKEN=' .env | cut -d= -f2-)
python scripts/download_ntl.py --regions configs/regions.yaml --config configs/ntl.yaml --start 2013-01 --end 2024-12
```
6. **Output schema** of `ntl_features.parquet` (one row per region×month):

| column | dtype | note |
|---|---|---|
| `region` | str | from regions.yaml |
| `sector` | str | SPDR ticker |
| `year` | int | |
| `month` | int | 1–12 |
| `date` | date | month-end of observation month M |
| `data_release_date` | date | N7 lag stamp (leakage key) |
| `ntl_mean, ntl_median, ntl_std, ntl_sum, ntl_p90, ntl_lit_count, ntl_lit_frac, n_valid_pixels, frac_masked` | float | N6 features |
| `radiance_units` | str | `"nW/cm^2/sr"` |
| `ntl_backend` | str | provenance |

**Deliverables:** `build_ntl_features()`; `scripts/download_ntl.py`; `data/processed/ntl_features.parquet`; `manifest.json`.

**Acceptance Criteria:**
- `--dry-run` prints the planned (region,month) work list and exits 0 with no network (test invokes it).
- Re-running the CLI after a complete run downloads nothing (manifest shows all `ok`; test asserts download function not called on second pass, e.g. via monkeypatch counter).
- Final parquet has exactly `len(regions) × 144` rows (minus any documented `status:'empty'` months), the schema columns above, no duplicate `(region, year, month)` (test asserts).
- `(date, data_release_date)` invariant holds on every row; `region`/`sector` consistent with `regions.yaml` (test asserts via join).

---

## N9 — Unit tests: feature math & masking (no network)

**Objective:** Prove extraction math and masking are correct without any download.

**Dependencies:** N5, N6, N7.

**Actions:** Create `tests/test_ntl_features.py`:
1. **Synthetic GeoTIFF fixture**: write a small (e.g. 4×4) GeoTIFF to a tmp path with `rasterio` using a known transform/CRS (EPSG:4326), a companion quality array, and embedded fill values. Optionally test with a plain `numpy` array path too.
2. Tests:
   - `test_masking_drops_fill_and_bad_quality` — fill + quality∈{1,2} excluded from valid count.
   - `test_feature_math` — exact asserts from N6 acceptance (mean/median/std `ddof=0`/sum/p90/lit_count/lit_frac/frac_masked).
   - `test_scale_offset_applied` — `scale_factor`/`add_offset` from attrs applied once, correctly.
   - `test_all_masked_returns_nan` — empty month → NaN features, `frac_masked==1.0`, no raise.
   - `test_release_lag` — N7 invariants.
   - `test_clip_to_bbox` — clipping a fixture to a sub-window returns the expected sub-array shape and values.
   - `test_mosaic_two_tiles` — two adjacent fixtures merge into the expected stitched array.
   - `test_determinism` — running `compute_ntl_features` twice on the same input yields identical dict (byte-stable).

**Deliverables:** `tests/test_ntl_features.py` (all offline).

**Acceptance Criteria:** `pytest tests/test_ntl_features.py -q` passes with **zero** network access (enforce via a fixture that monkeypatches the download adapter to raise if called). All numeric asserts use tight tolerances (`abs<=1e-3`).

---

## N10 — Integration smoke test (credential-gated)

**Objective:** Prove the real download+extract path works end-to-end on one tiny ROI for one month.

**Dependencies:** N1–N8.

**Actions:** Create `tests/test_ntl_smoke.py`:
```python
@pytest.mark.skipif(not os.getenv("EARTHDATA_TOKEN") and not netrc_present(),
                    reason="no Earthdata credentials")
@pytest.mark.integration
def test_download_one_month_small_roi(tmp_path):
    # Singapore port bbox (small, single-tile), month 2020-06
    df = build_ntl_features(regions_yaml=<single-region tmp yaml>,
                            ntl_cfg=<cfg>, start="2020-06", end="2020-06",
                            out_parquet=tmp_path/"smoke.parquet",
                            manifest=tmp_path/"manifest.json")
    assert len(df) == 1
    assert df.loc[0,"n_valid_pixels"] > 0
    assert df.loc[0,"ntl_mean"] > 0
    assert df.loc[0,"data_release_date"] > df.loc[0,"date"]
    # verify quality layer only had values in {0,1,2} (N3 TO-VERIFY)
```
- Use a small single-tile ROI (e.g. `singapore_port`) and a mid-mission month (2020-06) to avoid early-mission gaps.
- Mark `@pytest.mark.integration`; default `pytest` run excludes it (configure marker in `pyproject.toml`, S-phase).

**Deliverables:** `tests/test_ntl_smoke.py`; registered `integration` marker.

**Acceptance Criteria:**
- With credentials present, the test downloads one tile, produces a 1-row table with positive `ntl_mean`/`n_valid_pixels`, satisfies the release-lag invariant, and confirms the real quality layer's unique values ⊆ {0,1,2} (resolving the N3 TO-VERIFY).
- With no credentials, the test **skips** (not fails); CI on a no-secret runner stays green.

---

## Real-world pitfalls & how this phase handles them

| Pitfall | Handling |
|---|---|
| Earthdata token expires (~60 days) mid-run | 401/403 surfaced immediately with a "regenerate token at urs.earthdata.nasa.gov" message (N8 retry logic distinguishes auth vs transient). |
| Region spans 2+ tiles | Mosaic via `bm.raster` (auto) or `rasterio.merge.merge` fallback (N5); covered by `test_mosaic_two_tiles`. |
| Permian/basin shapes are polygons, not rectangles | Start with bbox; allow `geometry: shapefile` upgrade in `regions.yaml` (N4) + `rasterio.mask.mask` path (N5). |
| Early-mission quality gaps (2012, early 2013) | Study starts 2013-01; `frac_masked`/`n_valid_pixels` columns flag low-coverage months for Phase P to drop. |
| Snow-covered northern winters reduce snow-free obs | Use snow-free composite (intended); high `frac_masked` winter months flagged, not silently averaged. |
| Hard-coded scale/fill assumptions drift across collections | Read `scale_factor`/`add_offset`/`_FillValue`/`flag_meanings` at runtime from the audited attrs (N3), never assume. |
| Look-ahead leakage from using month-M NTL to predict month-M return | `data_release_date` stamp (N7) forces M→M+1 alignment downstream; enforced again in Phase P leakage audit. |
| Re-download cost / interrupted runs | Idempotent, manifest-driven, `output_skip_if_exists=True` (N8); `test` asserts no re-download on second pass. |
| `blackmarblepy` install/endpoint breakage on Windows | `earthaccess` raw-HDF5 fallback behind `download_backend` flag (N2); downstream code is backend-agnostic. |
| Secrets accidentally committed | `.env`/`.netrc`/`_netrc` gitignored; N1 test greps `src/`,`configs/` for token-like strings. |

## Outputs consumed by later phases
- `data/processed/ntl_features.parquet` → **Phase P** (panel build, region→sector correlation pre-screen `P-screen`, walk-forward splits) and **Phase F** (alignment join on `data_release_date`).
- `configs/regions.yaml` → **Phase P** (pairing universe) and **Phase E** (multi-region vs single-region stratification for H2/H3).
- `data/interim/ntl/manifest.json` and `_layer_attrs.json` → **Phase W** (reproducibility/data-provenance appendix).
