import numpy as np
import lightkurve as lk
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

TESS_PLATE_SCALE_ARCSEC = 21.0
MAX_TPF_FRAMES = 200


def _find_cached_tpf(tic_id: str, sector: str | None) -> Path | None:
    """Look for a cached TPF FITS file, avoiding a MAST search."""
    try:
        from config import settings
        cache_dir = settings.mast_cache_dir / "mastDownload" / "TESS"
        if not cache_dir.exists():
            cache_dir = Path(lk.config.get_cache_dir()) / "mastDownload" / "TESS"
    except Exception:
        return None
    if not cache_dir.exists():
        return None
    tic_padded = tic_id.zfill(16)
    sector_str = f"s{str(sector).zfill(4)}" if sector and sector != "all" else None
    for d in cache_dir.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        if tic_padded not in name:
            continue
        if sector_str and sector_str not in name:
            continue
        for f in d.glob("*_tp.fits"):
            return f
    return None


def _extract_tpf_pixels(tpf) -> dict:
    """Extract pixel data from a TPF for frontend animation, downsampled to MAX_TPF_FRAMES."""
    flux_cube = tpf.flux.value
    n_cadences, n_rows, n_cols = flux_cube.shape
    time_arr = tpf.time.value

    step = max(1, n_cadences // MAX_TPF_FRAMES)
    idx = np.arange(0, n_cadences, step)
    time_ds = time_arr[idx]
    flux_ds = flux_cube[idx]

    nan_mask = np.isnan(flux_ds)
    flux_ds = np.where(nan_mask, 0.0, flux_ds)

    aperture = None
    try:
        aperture = tpf.pipeline_mask.astype(int).tolist()
    except Exception:
        pass

    flux_list = np.round(flux_ds, 2).tolist()

    return {
        "available": True,
        "time": np.round(time_ds, 6).tolist(),
        "flux": flux_list,
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "n_frames": len(idx),
        "aperture_mask": aperture,
        "column": int(tpf.column),
        "row": int(tpf.row),
    }


def compute_centroid(tic_id: str, sector: str = "all",
                     prefetched_tp_pairs: list[tuple[str, str]] | None = None) -> dict:
    """
    Download TPF and compute flux-weighted centroid displacement.
    Shifts > 1 pixel (~21 arcsec) during a dip indicate background contamination.

    If prefetched_tp_pairs is provided, skips the MAST search and downloads
    directly from S3.
    """
    tpf = None

    if prefetched_tp_pairs:
        try:
            from pipeline.s3_fetch import download_single_fits_sync
            from config import settings as _settings

            s3_url, fname = prefetched_tp_pairs[0]
            cache_dir = _settings.mast_cache_dir / "s3_cache"
            local_path = download_single_fits_sync(s3_url, fname, cache_dir)
            if local_path:
                tpf = lk.TessTargetPixelFile(str(local_path))
        except Exception as e:
            logger.warning("Prefetched TPF download failed for TIC %s: %s", tic_id, e)
            tpf = None

    if tpf is None:
        cached_path = _find_cached_tpf(tic_id, sector)
        if cached_path:
            try:
                tpf = lk.TessTargetPixelFile(str(cached_path))
            except Exception:
                tpf = None

    if tpf is None:
        search_sector = None if sector == "all" else int(sector)
        try:
            tpf_results = lk.search_targetpixelfile(
                f"TIC {tic_id}",
                mission="TESS",
                sector=search_sector,
                author="SPOC",
            )
        except Exception as e:
            logger.warning(f"TPF search failed for TIC {tic_id}: {e}")
            return {"available": False}

        if len(tpf_results) == 0:
            return {"available": False}

        try:
            from pipeline.s3_fetch import extract_s3_urls_from_search, download_single_fits_sync
            from config import settings as _settings

            pairs = extract_s3_urls_from_search(tpf_results)
            pairs = [(u, f) for u, f in pairs if f.endswith("_tp.fits")]
            if pairs:
                s3_url, fname = pairs[0]
                cache_dir = _settings.mast_cache_dir / "s3_cache"
                local_path = download_single_fits_sync(s3_url, fname, cache_dir)
                if local_path:
                    tpf = lk.TessTargetPixelFile(str(local_path))
        except Exception as e:
            logger.debug("S3 TPF download failed for TIC %s (%s), trying lightkurve", tic_id, e)
            tpf = None

        if tpf is None:
            try:
                tpf = tpf_results[0].download(cache=True)
            except Exception as e:
                logger.warning(f"TPF download failed for TIC {tic_id}: {e}")
                return {"available": False}

    try:
        centroid_col, centroid_row = tpf.estimate_centroids(aperture_mask="pipeline")
    except Exception:
        try:
            centroid_col, centroid_row = tpf.estimate_centroids()
        except Exception as e:
            logger.warning(f"Centroid estimation failed: {e}")
            return {"available": False}

    time = tpf.time.value.tolist()
    col = centroid_col.value.tolist()
    row = centroid_row.value.tolist()

    col_arr = np.array(col)
    row_arr = np.array(row)
    valid = ~(np.isnan(col_arr) | np.isnan(row_arr))
    col_clean = col_arr[valid]
    row_clean = row_arr[valid]

    if len(col_clean) == 0:
        return {"available": False}

    col_baseline = float(np.median(col_clean))
    row_baseline = float(np.median(row_clean))

    col_disp = col_arr - col_baseline
    row_disp = row_arr - row_baseline
    displacement_pixels = np.sqrt(col_disp**2 + row_disp**2)
    displacement_arcsec = displacement_pixels * TESS_PLATE_SCALE_ARCSEC

    max_shift_arcsec = float(np.nanmax(displacement_arcsec))
    rms_shift_arcsec = float(np.sqrt(np.nanmean(displacement_arcsec**2)))
    shift_flagged = float(np.nanmax(displacement_pixels)) > 1.0

    try:
        tpf_pixels = _extract_tpf_pixels(tpf)
    except Exception as e:
        logger.warning(f"TPF pixel extraction failed: {e}")
        tpf_pixels = {"available": False}

    del tpf

    return {
        "available": True,
        "time": time,
        "col": col_arr.tolist(),
        "row": row_arr.tolist(),
        "col_baseline": col_baseline,
        "row_baseline": row_baseline,
        "displacement_arcsec": displacement_arcsec.tolist(),
        "max_shift_arcsec": round(max_shift_arcsec, 2),
        "rms_shift_arcsec": round(rms_shift_arcsec, 2),
        "shift_flagged": bool(shift_flagged),
        "tpf_pixels": tpf_pixels,
    }


def centroid_shift_during_event(
    event_time_center: float, event_duration_hours: float, centroid_result: dict
) -> float:
    if not centroid_result.get("available"):
        return -1.0

    time_arr = np.array(centroid_result["time"])
    disp_arr = np.array(centroid_result["displacement_arcsec"])

    half_dur_days = (event_duration_hours / 2.0) / 24.0
    in_event = np.abs(time_arr - event_time_center) < half_dur_days

    if not in_event.any():
        return 0.0

    return round(float(np.nanmax(disp_arr[in_event])), 2)
