import gc
import shutil
import lightkurve as lk
import numpy as np
import os
import logging
from config import settings

logger = logging.getLogger(__name__)


def clear_query_caches() -> None:
    """Free in-memory caches that lightkurve / astroquery accumulate."""
    for _clear in (
        _clear_mast_caches,
        _clear_irsa_cache,
        _clear_simbad_cache,
        _clear_lightkurve_cache,
    ):
        try:
            _clear()
        except Exception:
            pass
    gc.collect()


def _clear_mast_caches() -> None:
    from astroquery.mast import Observations, Catalogs
    for svc in (Observations, Catalogs):
        for attr in ("_cache", "_service_api_connection"):
            obj = getattr(svc, attr, None)
            if obj is not None and hasattr(obj, "clear"):
                obj.clear()
        if hasattr(svc, "clear_cache"):
            svc.clear_cache()


def _clear_irsa_cache() -> None:
    from astroquery.ipac.irsa import Irsa
    for attr in ("_cache", "_service_api_connection"):
        obj = getattr(Irsa, attr, None)
        if obj is not None and hasattr(obj, "clear"):
            obj.clear()
    if hasattr(Irsa, "clear_cache"):
        Irsa.clear_cache()


def _clear_simbad_cache() -> None:
    from astroquery.simbad import Simbad
    for attr in ("_cache", "_service_api_connection"):
        obj = getattr(Simbad, attr, None)
        if obj is not None and hasattr(obj, "clear"):
            obj.clear()
    if hasattr(Simbad, "clear_cache"):
        Simbad.clear_cache()


def _clear_lightkurve_cache() -> None:
    import lightkurve
    for attr in ("_cache", "_search_cache"):
        obj = getattr(lightkurve, attr, None)
        if obj is not None and hasattr(obj, "clear"):
            obj.clear()
    sr_mod = getattr(lightkurve, "search", None)
    if sr_mod:
        for attr in ("_cache", "_search_cache", "cache"):
            obj = getattr(sr_mod, attr, None)
            if obj is not None and hasattr(obj, "clear"):
                obj.clear()

def _sync_lk_cache_dir() -> None:
    cache_path = str(settings.mast_cache_dir)
    os.environ["LIGHTKURVE_CACHE_DIR"] = cache_path
    try:
        lk.conf.cache_dir = cache_path
    except Exception:
        pass


_sync_lk_cache_dir()


def clear_mast_downloads() -> None:
    """Delete downloaded FITS under the Lightkurve/MAST tree to reclaim disk space."""
    mast_dir = settings.mast_cache_dir / "mastDownload"
    if mast_dir.exists():
        try:
            shutil.rmtree(mast_dir, ignore_errors=True)
            logger.info("Cleared MAST download cache: %s", mast_dir)
        except Exception as e:
            logger.warning("Could not fully clear MAST downloads %s: %s", mast_dir, e)


def normalize_tic_id(raw) -> str:
    """
    Lightkurve / MAST sometimes return target_name as ndarray, list, or str.
    str(np.array([id])) -> \"['123']\" which breaks search_lightcurve(\"TIC ...\").
    """
    if raw is None:
        raise ValueError("Missing TIC ID from search result")

    if hasattr(raw, "item") and not isinstance(raw, (str, bytes)):
        try:
            raw = raw.item()
        except Exception:
            pass

    if isinstance(raw, (list, tuple)):
        if not raw:
            raise ValueError("Empty TIC ID list from search result")
        raw = raw[0]
        if hasattr(raw, "item"):
            try:
                raw = raw.item()
            except Exception:
                pass

    if isinstance(raw, np.ndarray):
        if raw.size == 0:
            raise ValueError("Empty TIC ID array from search result")
        raw = raw.flat[0].item() if hasattr(raw.flat[0], "item") else int(raw.flat[0])

    s = str(raw).strip()
    # str(list) or str ndarray repr
    if s.startswith("[") and "]" in s:
        inner = s.strip("[]").replace("'", "").replace('"', "").split(",")[0].strip()
        s = inner
    if s.upper().startswith("TIC"):
        s = s[3:].strip()

    digits = "".join(c for c in s if c.isdigit())
    if len(digits) >= 6:
        return digits
    if s.isdigit():
        return s
    raise ValueError(f"Could not parse TIC ID from: {raw!r}")


def resolve_target(identifier: str) -> dict:
    """
    Accepts a TIC ID like 'TIC 234994474' or a name like 'K2-18'.
    Returns basic stellar parameters from the TESS Input Catalog.

    If the identifier is already a numeric TIC ID (with or without the
    "TIC" prefix), skip the MAST search entirely -- saves ~10-15s per target.
    """
    clean = identifier.strip()
    stripped = clean.upper().replace("TIC", "").strip()
    if stripped.isdigit() and len(stripped) >= 6:
        return {
            "tic_id": stripped,
            "common_name": None,
            "available_sectors": [],
        }

    _sync_lk_cache_dir()
    results = lk.search_lightcurve(identifier, mission="TESS")
    if len(results) == 0:
        raise ValueError(f"No TESS data found for: {identifier}")

    tic_id = normalize_tic_id(results[0].target_name)
    id_as_tic = clean.replace(" ", "").upper() == f"TIC{tic_id}"
    common_name = None if (clean == tic_id or id_as_tic) else identifier

    return {
        "tic_id": tic_id,
        "common_name": common_name,
        "available_sectors": list(set(results.table["sequence_number"].tolist())),
    }


def _download_via_s3(results, tic_id: str) -> lk.LightCurveCollection | None:
    """Try to download light curve FITS from S3 in parallel, returning a LightCurveCollection.

    Returns None if S3 download fails or yields no files, so the caller can fall back
    to the standard lightkurve download path.
    """
    try:
        from pipeline.s3_fetch import extract_s3_urls_from_search, download_fits_parallel_sync

        all_pairs = extract_s3_urls_from_search(results)
        pairs = [(u, f) for u, f in all_pairs if f.endswith("_lc.fits")]
        if not pairs:
            logger.info("No S3 LC URLs for TIC %s (%d total products, 0 matched _lc.fits), falling back to lightkurve", tic_id, len(all_pairs))
            return None

        cache_dir = settings.mast_cache_dir / "s3_cache"
        paths = download_fits_parallel_sync(pairs, cache_dir)
        if not paths:
            logger.warning("S3 download returned no files for TIC %s, falling back", tic_id)
            return None

        lcs = []
        for p in paths:
            try:
                obj = lk.read(str(p))
                if hasattr(obj, "normalize"):
                    lcs.append(obj)
                else:
                    logger.debug("Skipping non-LC file: %s", p.name)
            except Exception as e:
                logger.debug("Failed to read %s: %s", p.name, e)
        if not lcs:
            return None

        logger.info("S3 parallel download: %d/%d files for TIC %s", len(lcs), len(pairs), tic_id)
        return lk.LightCurveCollection(lcs)
    except Exception as e:
        logger.warning("S3 download failed for TIC %s (%s), falling back to lightkurve", tic_id, e)
        return None


def fetch_light_curve(tic_id: str, sector: str = "all",
                      prefetched_lc_pairs: list[tuple[str, str]] | None = None) -> dict:
    """
    Downloads TESS light curve data, preferring direct S3 download with
    parallel fetching. Falls back to lightkurve's MAST download if S3 fails.

    If prefetched_lc_pairs is provided (from a batch sector prefetch), the
    MAST search is skipped entirely and S3 URLs are used directly.
    """
    _sync_lk_cache_dir()
    tic_id = normalize_tic_id(tic_id)

    lc_collection = None

    if prefetched_lc_pairs:
        from pipeline.s3_fetch import download_fits_parallel_sync
        cache_dir = settings.mast_cache_dir / "s3_cache"
        paths = download_fits_parallel_sync(prefetched_lc_pairs, cache_dir)
        if paths:
            lcs = []
            for p in paths:
                try:
                    obj = lk.read(str(p))
                    if isinstance(obj, lk.LightCurve):
                        lcs.append(obj)
                except Exception:
                    pass
            if lcs:
                lc_collection = lk.LightCurveCollection(lcs)
                logger.info("Prefetch S3: %d LC files for TIC %s", len(lcs), tic_id)

    if lc_collection is None:
        search_sector = None if sector == "all" else int(sector)

        results = lk.search_lightcurve(
            f"TIC {tic_id}",
            mission="TESS",
            author="SPOC",
            sector=search_sector,
            exptime=120,
        )

        if len(results) == 0:
            results = lk.search_lightcurve(
                f"TIC {tic_id}",
                mission="TESS",
                sector=search_sector,
            )

        if len(results) == 0:
            raise ValueError(f"No light curve data available for TIC {tic_id}")

        logger.info(f"Downloading {len(results)} sector(s) for TIC {tic_id} (MAST search)")

        lc_collection = _download_via_s3(results, tic_id)
        if lc_collection is None:
            logger.info("Using lightkurve download for TIC %s", tic_id)
            lc_collection = results.download_all(cache=True)

    only_lcs = [item for item in lc_collection if isinstance(item, lk.LightCurve)]
    if not only_lcs:
        raise ValueError(f"No light curve objects found for TIC {tic_id}")
    lc_collection = lk.LightCurveCollection(only_lcs)

    lc = lc_collection.stitch()
    lc = lc.remove_nans()

    sector_count = len(only_lcs)
    data = {
        "time": lc.time.value.tolist(),
        "flux": lc.flux.value.tolist(),
        "flux_err": lc.flux_err.value.tolist(),
        "sector_count": sector_count,
    }
    del lc_collection, lc
    return data
