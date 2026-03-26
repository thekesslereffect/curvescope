"""
Per-TIC training data cache.

Stores preprocessed window arrays as lightweight .npz files so MAST
downloads + detrending only happen once per target.  Files live under
``{data_dir}/training_cache/{tic_id}.npz``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from config import settings

logger = logging.getLogger(__name__)


def _cache_dir() -> Path:
    d = settings.training_cache_dir
    d.mkdir(parents=True, exist_ok=True)
    return d


def _path_for(tic_id: str) -> Path:
    clean = tic_id.strip().replace(" ", "")
    if clean.upper().startswith("TIC"):
        clean = clean[3:].strip()
    return _cache_dir() / f"{clean}.npz"


def has(tic_id: str) -> bool:
    return _path_for(tic_id).exists()


def load(tic_id: str) -> np.ndarray | None:
    """Return cached windows array, or None if not cached."""
    p = _path_for(tic_id)
    if not p.exists():
        return None
    try:
        data = np.load(p)
        windows = data["windows"]
        logger.debug("Cache hit for TIC %s (%d windows)", tic_id, len(windows))
        return windows
    except Exception as e:
        logger.warning("Corrupt cache for TIC %s, removing: %s", tic_id, e)
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass
        return None


def save(tic_id: str, windows: np.ndarray) -> None:
    """Persist preprocessed windows for a target."""
    p = _path_for(tic_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, windows=windows)
    logger.info("Cached %d windows for TIC %s → %s", len(windows), tic_id, p.name)


def evict(tic_id: str) -> bool:
    """Remove cached data for one target. Returns True if a file was deleted."""
    p = _path_for(tic_id)
    if p.exists():
        try:
            p.unlink()
            logger.info("Evicted training cache for TIC %s", tic_id)
            return True
        except OSError as e:
            logger.warning("Could not evict cache for TIC %s: %s", tic_id, e)
    return False


def evict_many(tic_ids: list[str]) -> int:
    """Remove cached data for multiple targets. Returns count deleted."""
    return sum(1 for t in tic_ids if evict(t))


def clear_all() -> int:
    """Delete every cached training file. Returns count deleted."""
    d = _cache_dir()
    count = 0
    for f in d.glob("*.npz"):
        try:
            f.unlink()
            count += 1
        except OSError:
            pass
    if count:
        logger.info("Cleared %d training cache files", count)
    return count


def list_cached() -> list[str]:
    """Return TIC IDs that have cached data."""
    d = _cache_dir()
    if not d.exists():
        return []
    return [f.stem for f in sorted(d.glob("*.npz"))]


def cache_size_bytes() -> int:
    """Total disk usage of the training cache."""
    d = _cache_dir()
    if not d.exists():
        return 0
    return sum(f.stat().st_size for f in d.glob("*.npz"))
