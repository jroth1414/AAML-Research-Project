"""NASA Black Marble VNP46A3 nighttime-light acquisition + raster feature extraction (Phase A.1).

Design: the feature math and quality masking are PURE functions over numpy arrays (fully
unit-tested offline, no network). The download/georeferencing path lazy-imports the heavy geo
+ NTL stack (``earthaccess``, ``rasterio``) only inside the IO functions, so a missing NTL wheel
never breaks the core environment (Risk R5). Default backend on Windows/CPU is ``earthaccess``;
``blackmarblepy`` is an optional alternate (see requirements-ntl.txt).

The single most important leakage guard here is :func:`stamp_release_date` (Task N7), which encodes
the ~30-45 day VNP46A3 release lag so Phase B/F can align features without look-ahead.

Units: VNP46A3 radiance is nW/cm^2/sr; Collection 2.0 stores it already scaled (scale_factor=1.0).
"""

from __future__ import annotations

import calendar
import json
import os
import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np

# --- VNP46A3 product facts / constants (Task N3) ---
VNP46A3_VAR = "NearNadir_Composite_Snow_Free"
VNP46A3_QF = "NearNadir_Composite_Snow_Free_Quality"
VNP46A3_NUM = "NearNadir_Composite_Snow_Free_Num"
VNP46A3_STD = "NearNadir_Composite_Snow_Free_Std"
# VERIFIED on a real VNP46A3 v2 tile (2020-06, h28v08): for the MONTHLY Snow_Free composite the
# Mandatory_Quality_Flag is dominated by classes 1 and 2 (class 0 is rare: ~1.6k px / tile), and
# 255 = no-retrieval. The composite is already gap-filled and marks true no-data with its own
# _FillValue (-999.9), so we keep quality {0,1,2} as valid radiance and drop only 255. (Dropping
# {1,2} as an earlier draft assumed would discard ~all pixels — N3 TO-VERIFY resolved here.)
GOOD_QUALITY_FLAGS = (0, 1, 2)  # all retrieved classes are valid for the monthly composite
DROP_QUALITY_FLAGS = (255,)  # no-retrieval only (composite _FillValue handles the rest)
FILL_VALUES = (-999.9, 65535)  # VNP46A3 v2 composite fill is -999.9; 65535 kept as a fallback
RADIANCE_UNITS = "nW/cm^2/sr"

# Black Marble HDF-EOS5 geographic tile grid: 36 cols x 18 rows of 10 deg x 10 deg tiles.
_TILE_DEG = 10.0
_HV_RE = re.compile(r"h(\d{2})v(\d{2})")
_DATE_RE = re.compile(r"\.A(\d{4})(\d{3})\.")  # VNP46A3.A{YYYY}{DOY}. (composite month start)

FEATURE_COLUMNS = [
    "ntl_mean",
    "ntl_median",
    "ntl_std",
    "ntl_sum",
    "ntl_p90",
    "ntl_lit_count",
    "ntl_lit_frac",
    "n_valid_pixels",
    "frac_masked",
]


# --------------------------------------------------------------------------------------
# N1 — Earthdata credential resolution
# --------------------------------------------------------------------------------------
def resolve_earthdata_token() -> str:
    """Return a NASA Earthdata bearer token.

    Order: env ``EARTHDATA_TOKEN`` -> ``BLACKMARBLE_TOKEN``. Raises ``RuntimeError`` with a
    remediation message (Earthdata token URL) if neither is set.
    """
    for name in ("EARTHDATA_TOKEN", "BLACKMARBLE_TOKEN"):
        v = os.environ.get(name)
        if v and v.strip():
            return v.strip()
    raise RuntimeError(
        "No Earthdata bearer token found. Set EARTHDATA_TOKEN in .env (copy .env.example). "
        "Generate one at https://urs.earthdata.nasa.gov/ (profile -> Generate Token)."
    )


# --------------------------------------------------------------------------------------
# N7 — Release-lag stamp (the leakage guard)
# --------------------------------------------------------------------------------------
def _month_end(year: int, month: int) -> date:
    return date(year, month, calendar.monthrange(year, month)[1])


def stamp_release_date(year: int, month: int, lag_days: int = 45) -> dict:
    """Return ``{'date': month_end(M), 'data_release_date': month_end(M) + lag_days}``.

    VNP46A3 for month M is released ~30-45 days after month-end, i.e. during M+1. Phase B/F MUST
    require ``data_release_date <= prediction_decision_date`` so month-M NTL only predicts M+1 (or
    later) returns. ``data_release_date`` is strictly later than ``date``.
    """
    d = _month_end(year, month)
    return {"date": d, "data_release_date": d + timedelta(days=lag_days)}


# --------------------------------------------------------------------------------------
# N3 — layer attribute audit
# --------------------------------------------------------------------------------------
def audit_layer_attributes(attrs: dict, out_path: str | os.PathLike | None = None) -> dict:
    """Normalize and persist the science-dataset attributes read from the first real tile.

    ``attrs`` is a mapping that may contain scale_factor/add_offset/_FillValue/flag_values/
    flag_meanings. Returns a normalized dict; writes it to ``out_path`` (JSON) if given. The
    pipeline reads scale/offset/fill from this audit rather than hard-coding them.
    """

    def _num(x, default):
        try:
            return float(x)
        except (TypeError, ValueError):
            return default

    norm = {
        "scale_factor": _num(attrs.get("scale_factor", attrs.get("scale", 1.0)), 1.0),
        "add_offset": _num(attrs.get("add_offset", attrs.get("offset", 0.0)), 0.0),
        # VNP46A3 v2 fill is -999.9 (float, band-level); older assumptions used 65535.
        "fill_value": _num(attrs.get("_FillValue", FILL_VALUES[0]), FILL_VALUES[0]),
        "flag_values": list(attrs.get("flag_values", [0, 1, 2])),
        "flag_meanings": attrs.get("flag_meanings", "good poor gap_filled"),
        "units": attrs.get("units", RADIANCE_UNITS),
    }
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(norm, indent=2, default=str), encoding="utf-8")
    return norm


# --------------------------------------------------------------------------------------
# N5 — quality masking (pure)
# --------------------------------------------------------------------------------------
def clean_radiance_array(
    composite: np.ndarray,
    quality: np.ndarray | None,
    *,
    scale: float = 1.0,
    offset: float = 0.0,
    fill_values=FILL_VALUES,
    drop_quality_flags=DROP_QUALITY_FLAGS,
    clip_negative: str = "mask",
) -> np.ma.MaskedArray:
    """Return a masked float32 radiance array (nW/cm^2/sr) with fill + bad-quality pixels masked.

    Fill detection is on the RAW composite (before scaling); scale/offset are then applied to the
    valid pixels. ``clip_negative`` in {"mask","zero"} controls rare negative-radiance artifacts.
    """
    raw = np.asarray(composite, dtype="float64")
    invalid = np.isnan(raw)
    for fv in fill_values:
        invalid |= np.isclose(raw, float(fv))  # float-safe (VNP46A3 fill is -999.9)
    if quality is not None:
        q = np.asarray(quality)
        bad_q = np.zeros(q.shape, dtype=bool)
        for f in drop_quality_flags:
            bad_q |= q == f
        invalid |= bad_q
    data = raw * scale + offset
    if clip_negative == "mask":
        invalid |= data < 0
    elif clip_negative == "zero":
        data = np.where(data < 0, 0.0, data)
    return np.ma.MaskedArray(data.astype("float32"), mask=invalid, fill_value=np.nan)


# --------------------------------------------------------------------------------------
# N6 — per-region-per-month features (pure)
# --------------------------------------------------------------------------------------
def compute_ntl_features(clean: np.ma.MaskedArray, cfg: dict | None = None) -> dict:
    """Reduce a clean masked radiance array to the monthly feature dict (FEATURE_COLUMNS).

    Masked (fill/bad-quality) pixels are excluded. If no valid pixels remain, returns all-NaN
    features with ``frac_masked == 1.0`` (does NOT raise).
    """
    cfg = cfg or {}
    lit_threshold = float(cfg.get("lit_threshold", 0.5))
    total = int(np.asarray(clean).size)
    valid = clean.compressed() if np.ma.isMaskedArray(clean) else np.asarray(clean).ravel()
    n_valid = int(valid.size)
    if n_valid == 0:
        feats = {c: float("nan") for c in FEATURE_COLUMNS}
        feats["ntl_lit_count"] = float("nan")
        feats["n_valid_pixels"] = 0.0
        feats["frac_masked"] = 1.0
        return feats
    lit_count = int((valid > lit_threshold).sum())
    return {
        "ntl_mean": float(np.mean(valid)),
        "ntl_median": float(np.median(valid)),
        "ntl_std": float(np.std(valid)),  # population std (ddof=0)
        "ntl_sum": float(np.sum(valid)),
        "ntl_p90": float(np.percentile(valid, 90)),
        "ntl_lit_count": float(lit_count),
        "ntl_lit_frac": float(lit_count / n_valid),
        "n_valid_pixels": float(n_valid),
        "frac_masked": float((total - n_valid) / total) if total else 1.0,
    }


# --------------------------------------------------------------------------------------
# Black Marble tile geometry (pure)
# --------------------------------------------------------------------------------------
def tile_hv_from_name(filename: str) -> tuple[int, int]:
    """Extract (h, v) tile indices from a Black Marble granule filename (``...h##v##...``)."""
    m = _HV_RE.search(str(filename))
    if not m:
        raise ValueError(f"no h##v## tile id found in {filename!r}")
    return int(m.group(1)), int(m.group(2))


def granule_month(filename: str) -> tuple[int, int]:
    """Return the (year, month) of a VNP46A3 monthly granule from its ``.A{YYYY}{DOY}.`` token.

    The composite is stamped with the first day of the month, so this maps the day-of-year back
    to a calendar month — used to keep only the granule for the target month (a bbox/temporal CMR
    query can return the adjacent month's composite too).
    """
    m = _DATE_RE.search(str(filename))
    if not m:
        raise ValueError(f"no .A{{YYYY}}{{DOY}}. date token in {filename!r}")
    year, doy = int(m.group(1)), int(m.group(2))
    d = date(year, 1, 1) + timedelta(days=doy - 1)
    return d.year, d.month


def tile_transform(h: int, v: int, ncols: int, nrows: int):
    """Affine transform for a 10x10 deg Black Marble geographic tile (lazy rasterio import)."""
    from rasterio.transform import from_origin

    west = -180.0 + _TILE_DEG * h
    north = 90.0 - _TILE_DEG * v
    return from_origin(west, north, _TILE_DEG / ncols, _TILE_DEG / nrows)


def tiles_for_bbox(bbox) -> list[tuple[int, int]]:
    """Return the list of (h, v) tiles covering ``bbox`` = [W, S, E, N]."""
    w, s, e, n = bbox
    h0 = int((w + 180.0) // _TILE_DEG)
    h1 = int((e + 180.0) // _TILE_DEG)
    v0 = int((90.0 - n) // _TILE_DEG)
    v1 = int((90.0 - s) // _TILE_DEG)
    return [(h, v) for v in range(v0, v1 + 1) for h in range(h0, h1 + 1)]


# --------------------------------------------------------------------------------------
# N5 (IO) — clip + mosaic
# --------------------------------------------------------------------------------------
def clip_array_to_bbox(data: np.ndarray, transform, bbox) -> tuple[np.ndarray, object]:
    """Clip a 2D array to ``bbox`` = [W, S, E, N] using its affine ``transform``.

    Returns (clipped_array, clipped_transform). Lazy rasterio import.
    """
    from rasterio.windows import from_bounds
    from rasterio.windows import transform as window_transform

    w, s, e, n = bbox
    win = from_bounds(w, s, e, n, transform=transform).round_offsets().round_lengths()
    r0 = max(0, int(win.row_off))
    c0 = max(0, int(win.col_off))
    r1 = min(data.shape[0], int(win.row_off + win.height))
    c1 = min(data.shape[1], int(win.col_off + win.width))
    sub = data[r0:r1, c0:c1]
    new_win = type(win)(col_off=c0, row_off=r0, width=c1 - c0, height=r1 - r0)
    return sub, window_transform(new_win, transform)


def mosaic_tiles(tiles: list[tuple[np.ndarray, object]], nodata=np.nan):
    """Mosaic a list of (array, transform) tiles into one (array, transform) via rasterio.merge.

    Lazy import. Each array is 2D; builds in-memory single-band datasets and merges.
    """
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.merge import merge as rio_merge

    datasets = []
    memfiles = []
    try:
        for arr, tr in tiles:
            arr2 = np.asarray(arr, dtype="float32")
            mf = MemoryFile()
            memfiles.append(mf)
            ds = mf.open(
                driver="GTiff",
                height=arr2.shape[0],
                width=arr2.shape[1],
                count=1,
                dtype="float32",
                crs="EPSG:4326",
                transform=tr,
                nodata=nodata,
            )
            ds.write(arr2, 1)
            datasets.append(ds)
        mosaic, out_transform = rio_merge(datasets, nodata=nodata)
        return mosaic[0], out_transform
    finally:
        for ds in datasets:
            ds.close()
        for mf in memfiles:
            mf.close()
        del rasterio  # keep linter quiet; import used above


# --------------------------------------------------------------------------------------
# N2 / N5 (IO) — real download + georeference (earthaccess backend)
# --------------------------------------------------------------------------------------
_LOGGED_IN = False


def _earthaccess_login() -> None:
    """Authenticate earthaccess using the bearer token once per process (lazy import)."""
    global _LOGGED_IN
    if _LOGGED_IN:
        return
    import earthaccess

    token = resolve_earthdata_token()
    # earthaccess reads EARTHDATA_TOKEN for the "environment" strategy in recent versions.
    os.environ["EARTHDATA_TOKEN"] = token
    try:
        earthaccess.login(strategy="environment")
    except Exception:
        # fall back to a netrc strategy if username/password are present
        earthaccess.login(strategy="netrc")
    _LOGGED_IN = True


def _read_subdataset(hdf_path: str, layer: str) -> tuple[np.ndarray, dict]:
    """Open one HDF-EOS5 layer with rasterio; return (array, attrs). Lazy import.

    Picks the subdataset whose name ends with ``/Data_Fields/<layer>`` regardless of grid name
    (the grid name has drifted across collections — TO-VERIFY pinned at first real download).
    """
    import rasterio

    with rasterio.open(hdf_path) as src:
        subs = list(src.subdatasets)
    target = None
    for sd in subs:
        if sd.rstrip().endswith(f"Data_Fields/{layer}") or sd.endswith(layer):
            target = sd
            break
    if target is None:
        raise RuntimeError(
            f"layer {layer!r} not found among subdatasets of {hdf_path}: {subs[:6]}..."
        )
    with rasterio.open(target) as src:
        arr = src.read(1).astype("float64")
        # scale_factor/add_offset/_FillValue live in the BAND tags for VNP46A3 v2, not the
        # dataset tags. Merge both; fall back to the GDAL nodata for the fill value.
        attrs = {**src.tags(), **src.tags(1)}
        if src.nodata is not None:
            attrs.setdefault("_FillValue", src.nodata)
    return arr, attrs


def fetch_region_month_raster(
    region: dict,
    month: date,
    cfg: dict,
    raw_dir: str | os.PathLike = "data/raw/ntl",
) -> dict:
    """Download (cached) the VNP46A3 tiles covering ``region['bbox']`` for ``month`` and return
    georeferenced composite + quality arrays per tile.

    Returns ``{'tiles': [{'h','v','composite','quality','transform','attrs'}], 'backend': str}``.
    Network path; earthaccess backend. Skips already-downloaded files.
    """
    backend = cfg.get("download_backend", "earthaccess")
    if backend != "earthaccess":  # pragma: no cover - alternate backend
        raise NotImplementedError(
            f"download_backend={backend!r} not available here; use 'earthaccess' "
            "(blackmarblepy is optional, see requirements-ntl.txt)."
        )
    import earthaccess

    _earthaccess_login()
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    w, s, e, n = region["bbox"]
    results = earthaccess.search_data(
        short_name=cfg.get("product", "VNP46A3"),
        version=str(cfg.get("version", "2")),
        temporal=(f"{month:%Y-%m-01}", f"{month:%Y-%m-%d}"),
        bounding_box=(w, s, e, n),
    )
    # Keep only granules whose composite month matches the target (a CMR temporal query can
    # also return the adjacent month's monthly composite).
    target = (month.year, month.month)
    matching = []
    for g in results:
        try:
            links = g.data_links()
            name = links[0].split("/")[-1] if links else str(g)
            if granule_month(name) == target:
                matching.append(g)
        except (ValueError, IndexError, KeyError):
            matching.append(g)  # keep if the date token is unparseable
    files = earthaccess.download(matching or results, str(raw_dir))
    layer, qf = VNP46A3_VAR, VNP46A3_QF
    tiles = []
    for f in files:
        f = str(f)
        if not f.lower().endswith((".h5", ".hdf5")):
            continue
        h, v = tile_hv_from_name(f)
        comp, attrs = _read_subdataset(f, layer)
        try:
            qual, _ = _read_subdataset(f, qf)
        except RuntimeError:
            qual = None
        tr = tile_transform(h, v, comp.shape[1], comp.shape[0])
        tiles.append(
            {"h": h, "v": v, "composite": comp, "quality": qual, "transform": tr, "attrs": attrs}
        )
    return {"tiles": tiles, "backend": backend}


def load_clean_radiance(region: dict, month: date, cfg: dict) -> np.ma.MaskedArray:
    """Fetch -> mosaic (if >1 tile) -> clip to bbox -> quality-mask. Returns masked radiance."""
    fetched = fetch_region_month_raster(
        region, month, cfg, raw_dir=cfg.get("raw_dir", "data/raw/ntl")
    )
    tiles = fetched["tiles"]
    if not tiles:
        return np.ma.MaskedArray(np.zeros((1, 1), "float32"), mask=True)
    attrs = audit_layer_attributes(tiles[0]["attrs"])
    if len(tiles) == 1:
        comp, tr = tiles[0]["composite"], tiles[0]["transform"]
        qual = tiles[0]["quality"]
    else:
        comp, tr = mosaic_tiles([(t["composite"], t["transform"]) for t in tiles])
        qual, _ = (
            mosaic_tiles([(t["quality"], t["transform"]) for t in tiles])
            if all(t["quality"] is not None for t in tiles)
            else (None, None)
        )
    comp_c, tr_c = clip_array_to_bbox(comp, tr, region["bbox"])
    qual_c = clip_array_to_bbox(qual, tr, region["bbox"])[0] if qual is not None else None
    # Use the fill value AUDITED from the tile (VNP46A3 v2 = -999.9), plus any config fallbacks.
    fills = (attrs["fill_value"], *cfg.get("fill_values", []))
    return clean_radiance_array(
        comp_c,
        qual_c,
        scale=attrs["scale_factor"],
        offset=attrs["add_offset"],
        fill_values=fills,
        drop_quality_flags=tuple(cfg.get("drop_quality_flags", DROP_QUALITY_FLAGS)),
        clip_negative=cfg.get("clip_negative", "mask"),
    )


# --------------------------------------------------------------------------------------
# N8 — orchestrator + feature-table assembly
# --------------------------------------------------------------------------------------
def month_range(start: str, end: str) -> list[tuple[int, int]]:
    """Inclusive list of (year, month) from ``start``/``end`` formatted ``YYYY-MM``."""
    sy, sm = (int(x) for x in start.split("-")[:2])
    ey, em = (int(x) for x in end.split("-")[:2])
    out = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        out.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def _load_manifest(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"entries": {}}


def _retry_load(region: dict, mdate: date, cfg: dict, attempts, backoff):
    """Call load_clean_radiance with bounded retry/backoff; auth errors (401/403) raise fast."""
    import time

    last = None
    for i in range(attempts):
        try:
            return load_clean_radiance(region, mdate, cfg)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            if "401" in msg or "403" in msg or "token" in msg.lower():
                raise RuntimeError(
                    f"Earthdata auth error ({msg}). Regenerate EARTHDATA_TOKEN at "
                    "https://urs.earthdata.nasa.gov/ and update .env."
                ) from exc
            last = exc
            if i < attempts - 1:
                time.sleep(backoff[min(i, len(backoff) - 1)])
    raise RuntimeError(f"failed after {attempts} attempts: {last}")


def build_ntl_features(
    regions_yaml: str | os.PathLike = "configs/regions.yaml",
    ntl_cfg: str | os.PathLike = "configs/ntl.yaml",
    start: str = "2013-01",
    end: str = "2024-12",
    out_parquet: str | os.PathLike = "data/processed/ntl_features.parquet",
    manifest: str | os.PathLike = "data/interim/ntl/manifest.json",
    only_region: str | None = None,
    dry_run: bool = False,
):
    """For each region x month: cached download -> clean -> features -> release-lag stamp.

    Idempotent & resumable (skips (region, month) already 'ok' in the manifest). Returns the
    assembled DataFrame and writes ``out_parquet`` (+ manifest). With ``dry_run`` prints the
    planned work list and returns it without any network access.
    """
    import pandas as pd
    import yaml

    from ..utils.logging import get_logger

    log = get_logger("ntl")
    cfg = yaml.safe_load(Path(ntl_cfg).read_text(encoding="utf-8")) or {}
    regions = (yaml.safe_load(Path(regions_yaml).read_text(encoding="utf-8")) or {}).get(
        "regions", []
    )
    if only_region:
        regions = [r for r in regions if r["id"] == only_region]
    months = month_range(start, end)
    work = [(r, ym) for r in regions for ym in months]
    if dry_run:
        for r, (y, m) in work:
            log.info("PLAN %s %04d-%02d", r["id"], y, m)
        log.info("dry-run: %d (region, month) tasks", len(work))
        return work

    man_path = Path(manifest)
    man = _load_manifest(man_path)
    attempts = int(cfg.get("retry_max_attempts", 5))
    backoff = cfg.get("retry_backoff_seconds", [2, 4, 8, 16, 32])
    rows = []
    for r, (y, m) in work:
        key = f"{r['id']}:{y:04d}-{m:02d}"
        if man["entries"].get(key, {}).get("status") == "ok":
            cached = man["entries"][key].get("row")
            if cached:
                rows.append(cached)
            continue
        stamp = stamp_release_date(y, m, int(cfg.get("release_lag_days", 45)))
        try:
            clean = _retry_load(r, _month_end(y, m), cfg, attempts, backoff)
            feats = compute_ntl_features(clean, cfg)
            status = "ok" if feats["n_valid_pixels"] and feats["n_valid_pixels"] > 0 else "empty"
        except Exception as exc:  # noqa: BLE001
            log.warning("FAILED %s: %s", key, exc)
            man["entries"][key] = {"status": "failed", "error": str(exc)}
            man_path.parent.mkdir(parents=True, exist_ok=True)
            man_path.write_text(json.dumps(man, indent=2, default=str), encoding="utf-8")
            continue
        sector = (r.get("candidate_sectors") or [None])[0]
        row = {
            "region": r["id"],
            "sector": sector,
            "year": y,
            "month": m,
            "date": stamp["date"],
            "data_release_date": stamp["data_release_date"],
            **feats,
            "radiance_units": RADIANCE_UNITS,
            "ntl_backend": cfg.get("download_backend", "earthaccess"),
        }
        rows.append(row)
        man["entries"][key] = {"status": status, "row": row}
        man_path.parent.mkdir(parents=True, exist_ok=True)
        man_path.write_text(json.dumps(man, indent=2, default=str), encoding="utf-8")
        log.info(
            "OK %s ntl_mean=%.3f n_valid=%d", key, feats["ntl_mean"], int(feats["n_valid_pixels"])
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df["data_release_date"] = pd.to_datetime(df["data_release_date"])
        Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_parquet, index=False)
        log.info("wrote %d rows -> %s", len(df), out_parquet)
    return df
