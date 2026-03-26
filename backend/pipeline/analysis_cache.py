"""
In-memory TTL cache for analysis chart payloads.

Heavy per-point arrays are not written to disk. Data is kept only while the
server process is running and expires after CHART_CACHE_TTL_SECONDS (default
10 minutes) so memory stays bounded.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

MAX_CHART_POINTS = 4000
MAX_PERIODOGRAM_POINTS = 2000

# How long chart JSON stays in RAM after a pipeline completes
CHART_CACHE_TTL_SECONDS = 600.0

_lock = threading.Lock()
_store: dict[int, tuple[float, dict[str, Any]]] = {}


def _prune_locked(now: float | None = None) -> None:
    """Remove expired entries; caller must hold _lock."""
    t = now if now is not None else time.monotonic()
    dead = [aid for aid, (ts, _) in _store.items() if t - ts > CHART_CACHE_TTL_SECONDS]
    for aid in dead:
        del _store[aid]
    if dead:
        logger.debug("Pruned %d expired chart cache entries", len(dead))


def put(analysis_id: int, data: dict[str, Any]) -> None:
    """Store chart data for this analysis (replaces any prior entry)."""
    with _lock:
        _prune_locked()
        _store[analysis_id] = (time.monotonic(), data)
        logger.info("Chart data cached in memory for analysis %d (TTL %ds)", analysis_id, int(CHART_CACHE_TTL_SECONDS))


def get(analysis_id: int) -> dict[str, Any] | None:
    """Return chart data if present and not expired."""
    with _lock:
        now = time.monotonic()
        _prune_locked(now)
        item = _store.get(analysis_id)
        if not item:
            return None
        ts, data = item
        if now - ts > CHART_CACHE_TTL_SECONDS:
            del _store[analysis_id]
            return None
        return data


def evict(analysis_id: int) -> bool:
    """Remove one analysis from the chart store."""
    with _lock:
        if analysis_id in _store:
            del _store[analysis_id]
            return True
    return False


def clear_all() -> int:
    """Clear all in-memory chart entries. Returns number removed."""
    with _lock:
        n = len(_store)
        _store.clear()
        return n


def _downsample_timeseries(data: dict, max_points: int = MAX_CHART_POINTS) -> dict:
    """Downsample parallel arrays in a dict to max_points."""
    if not data:
        return data
    first_key = next((k for k in data if isinstance(data.get(k), list)), None)
    if first_key is None:
        return data
    n = len(data[first_key])
    if n <= max_points:
        return data
    step = max(1, n // max_points)
    result = {}
    for k, v in data.items():
        if isinstance(v, list) and len(v) == n:
            result[k] = v[::step]
        else:
            result[k] = v
    return result


def _downsample_periodogram(data: dict) -> dict:
    if not data or not data.get("period"):
        return data
    n = len(data["period"])
    if n <= MAX_PERIODOGRAM_POINTS:
        return data
    step = max(1, n // MAX_PERIODOGRAM_POINTS)
    return {
        **data,
        "period": data["period"][::step],
        "power": data["power"][::step],
    }


def _downsample_centroid(data: dict) -> dict:
    if not data or not data.get("available") or not data.get("time"):
        return data
    n = len(data["time"])
    if n <= MAX_CHART_POINTS:
        return data
    step = max(1, n // MAX_CHART_POINTS)
    result = dict(data)
    for key in ("time", "col", "row", "displacement_arcsec"):
        if key in result and isinstance(result[key], list) and len(result[key]) == n:
            result[key] = result[key][::step]
    return result


def prepare_chart_data(
    raw_flux: dict,
    detrended_flux: dict,
    score_timeline: dict,
    periodogram: dict,
    wavelet: dict,
    centroid: dict,
    tpf_data: dict,
) -> dict:
    """Downsample and package all chart data for the in-memory store."""
    return {
        "raw_flux": _downsample_timeseries(raw_flux),
        "detrended_flux": _downsample_timeseries(detrended_flux),
        "score_timeline": _downsample_timeseries(score_timeline),
        "periodogram": _downsample_periodogram(periodogram),
        "wavelet": wavelet,
        "centroid": _downsample_centroid(centroid),
        "tpf_data": tpf_data,
    }
