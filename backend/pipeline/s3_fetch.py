"""
Direct S3 download for TESS FITS files from the public stpubdata bucket.

The MAST download API at mast.stsci.edu/api/v0.1/Download/file redirects (307)
to S3 objects at stpubdata.s3.us-east-1.amazonaws.com. By downloading directly
from S3 we skip the MAST intermediary and can parallelize fetches with httpx.

S3 path structure for light curves and TPFs:
  tess/public/tid/s{sctr}/{tid1}/{tid2}/{tid3}/{tid4}/{filename}

Where the TIC ID is zero-padded to 16 digits and split into 4-digit groups.
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

S3_BASE = "https://stpubdata.s3.us-east-1.amazonaws.com"
MAST_DL_BASE = "https://mast.stsci.edu/api/v0.1/Download/file"

MAX_CONCURRENT = 6
DOWNLOAD_TIMEOUT = 120.0
MAX_RETRIES = 1


def tic_to_tid_path(tic_id: str) -> str:
    """Convert a numeric TIC ID to the 4-group directory path used on S3.

    Example: '261136679' -> '0000/0002/6113/6679'
    """
    padded = tic_id.zfill(16)
    return f"{padded[0:4]}/{padded[4:8]}/{padded[8:12]}/{padded[12:16]}"


def _product_filename_to_s3_url(filename: str) -> str | None:
    """Build a full S3 URL from a TESS product filename.

    Expects filenames like:
      tess2018206045859-s0001-0000000261136679-0120-s_lc.fits
      tess2018206045859-s0001-0000000261136679-0120-s_tp.fits

    Returns None if the filename doesn't match the expected pattern.
    """
    m = re.match(
        r"(tess\d+-s(\d{4})-(\d{16})-\d+-[a-z]_(?:lc|tp)\.fits)",
        filename,
    )
    if not m:
        return None
    fname = m.group(1)
    sector = m.group(2)
    tid = m.group(3)
    tid_path = f"{tid[0:4]}/{tid[4:8]}/{tid[8:12]}/{tid[12:16]}"
    return f"{S3_BASE}/tess/public/tid/s{sector}/{tid_path}/{fname}"


def extract_s3_urls_from_search(search_result) -> list[tuple[str, str]]:
    """Extract (s3_url, filename) pairs from a lightkurve SearchResult.

    Tries the 'productFilename' column first (most reliable), then falls back
    to parsing 'dataURI'. Returns only entries where a valid S3 URL could be
    constructed.
    """
    table = search_result.table
    pairs: list[tuple[str, str]] = []

    filenames: list[str] = []
    for col_name in ("productFilename", "dataURI"):
        if col_name in table.colnames:
            for val in table[col_name]:
                s = str(val).strip()
                if col_name == "dataURI":
                    s = s.rsplit("/", 1)[-1]
                filenames.append(s)
            break

    if not filenames:
        logger.info("No product filenames found in search result (cols: %s)", table.colnames)
        return pairs

    for fname in filenames:
        url = _product_filename_to_s3_url(fname)
        if url:
            pairs.append((url, fname))
        else:
            logger.info("Filename did not match S3 pattern: %s", fname)

    return pairs


def _mast_fallback_url(filename: str) -> str:
    """Build a MAST download API URL as fallback (follows 307 -> S3)."""
    return f"{MAST_DL_BASE}?uri=mast:TESS/product/{filename}"


async def _download_one(
    client: httpx.AsyncClient,
    url: str,
    dest: Path,
    filename: str,
    semaphore: asyncio.Semaphore,
) -> Path | None:
    """Download a single FITS file with retry and MAST fallback."""
    if dest.exists() and dest.stat().st_size > 1000:
        logger.debug("S3 cache hit: %s", dest.name)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")

    urls_to_try = [url, _mast_fallback_url(filename)]

    for attempt_url in urls_to_try:
        for attempt in range(MAX_RETRIES + 1):
            try:
                async with semaphore:
                    async with client.stream("GET", attempt_url, follow_redirects=True) as resp:
                        if resp.status_code == 404:
                            break
                        resp.raise_for_status()
                        with open(tmp, "wb") as f:
                            async for chunk in resp.aiter_bytes(chunk_size=65536):
                                f.write(chunk)
                tmp.rename(dest)
                logger.debug("Downloaded %s (%s)", dest.name, "S3" if attempt_url == url else "MAST fallback")
                return dest
            except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.ConnectError, OSError) as e:
                if tmp.exists():
                    tmp.unlink(missing_ok=True)
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(1.0 * (attempt + 1))
                    continue
                logger.debug("Failed %s (attempt %d): %s", attempt_url, attempt + 1, e)
                break

    return None


async def download_fits_parallel(
    url_filename_pairs: list[tuple[str, str]],
    cache_dir: Path,
    max_concurrent: int = MAX_CONCURRENT,
) -> list[Path]:
    """Download multiple FITS files in parallel from S3.

    Returns list of local Paths for successfully downloaded files.
    Order matches input; failed downloads are omitted.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    timeout = httpx.Timeout(DOWNLOAD_TIMEOUT, connect=15.0)
    limits = httpx.Limits(max_connections=max_concurrent + 2, max_keepalive_connections=max_concurrent)

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        tasks = []
        for url, filename in url_filename_pairs:
            dest = cache_dir / filename
            tasks.append(_download_one(client, url, dest, filename, semaphore))

        results = await asyncio.gather(*tasks, return_exceptions=True)

    paths: list[Path] = []
    for r in results:
        if isinstance(r, Path):
            paths.append(r)
        elif isinstance(r, Exception):
            logger.warning("Parallel download task failed: %s", r)
    return paths


def download_fits_parallel_sync(
    url_filename_pairs: list[tuple[str, str]],
    cache_dir: Path,
    max_concurrent: int = MAX_CONCURRENT,
) -> list[Path]:
    """Synchronous wrapper around the async parallel downloader.

    Safe to call from sync code (creates a new event loop if needed).
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        new_loop = asyncio.new_event_loop()
        try:
            return new_loop.run_until_complete(
                download_fits_parallel(url_filename_pairs, cache_dir, max_concurrent)
            )
        finally:
            new_loop.close()
    else:
        return asyncio.run(
            download_fits_parallel(url_filename_pairs, cache_dir, max_concurrent)
        )


async def download_single_fits(
    url: str,
    filename: str,
    cache_dir: Path,
) -> Path | None:
    """Download a single FITS file from S3 (async)."""
    semaphore = asyncio.Semaphore(1)
    timeout = httpx.Timeout(DOWNLOAD_TIMEOUT, connect=15.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        return await _download_one(client, url, cache_dir / filename, filename, semaphore)


def download_single_fits_sync(
    url: str,
    filename: str,
    cache_dir: Path,
) -> Path | None:
    """Synchronous wrapper for single-file S3 download."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        new_loop = asyncio.new_event_loop()
        try:
            return new_loop.run_until_complete(download_single_fits(url, filename, cache_dir))
        finally:
            new_loop.close()
    else:
        return asyncio.run(download_single_fits(url, filename, cache_dir))


# ---------------------------------------------------------------------------
# Batch sector product prefetch
# ---------------------------------------------------------------------------

def prefetch_sector_products(sector: int) -> dict[str, dict[str, list[tuple[str, str]]]]:
    """Batch-query MAST for ALL product filenames in a sector.

    Returns a dict keyed by TIC ID:
      { "12345": { "lc": [(s3_url, filename), ...], "tp": [(s3_url, filename), ...] } }

    This replaces per-target lk.search_lightcurve() calls during sector scans,
    eliminating ~20s of MAST API latency per target.
    """
    from astroquery.mast import Observations

    logger.info("Prefetching product list for sector %d from MAST...", sector)

    obs = Observations.query_criteria(
        obs_collection="TESS",
        dataproduct_type="timeseries",
        sequence_number=sector,
    )
    if obs is None or len(obs) == 0:
        logger.warning("No observations for sector %d", sector)
        return {}

    logger.info("Got %d observations, fetching product lists...", len(obs))
    products = Observations.get_product_list(obs)
    if products is None or len(products) == 0:
        logger.warning("No products returned for sector %d", sector)
        return {}

    logger.info("Got %d products, building lookup...", len(products))

    result: dict[str, dict[str, list[tuple[str, str]]]] = {}

    fname_col = "productFilename" if "productFilename" in products.colnames else None
    if not fname_col:
        for col in ("dataURI", "dataproduct_type"):
            if col in products.colnames:
                fname_col = col
                break
    if not fname_col:
        logger.warning("No filename column in product table")
        return {}

    for row in products:
        raw = str(row[fname_col]).strip()
        if fname_col == "dataURI":
            raw = raw.rsplit("/", 1)[-1]

        url = _product_filename_to_s3_url(raw)
        if not url:
            continue

        m = re.search(r"-(\d{16})-", raw)
        if not m:
            continue
        tic = str(int(m.group(1)))

        if raw.endswith("_lc.fits"):
            kind = "lc"
        elif raw.endswith("_tp.fits"):
            kind = "tp"
        else:
            continue

        entry = result.setdefault(tic, {"lc": [], "tp": []})
        entry[kind].append((url, raw))

    logger.info("Prefetch complete: %d TICs with products", len(result))
    return result
