"""
Settings & training router.

IMPORTANT: This module must NOT import torch / lightkurve / pipeline.train
at the top level. Those imports take 30-60 s on Windows and block the entire
uvicorn startup, making *all* endpoints unreachable until they finish.
Instead we import them lazily inside the functions that need them.
"""

import logging
import os
import threading
from pathlib import Path

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import func

from config import settings, write_data_dir_to_env, load_training_targets, save_training_targets
from db.database import SessionLocal, recreate_engine
from db import models

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

# ---------------------------------------------------------------------------
# Lazy import helpers  (torch / lightkurve / pipeline.train)
# ---------------------------------------------------------------------------

_train_mod = None
_train_mod_lock = threading.Lock()


def _get_train_module():
    """Import pipeline.train on first use so server startup isn't blocked."""
    global _train_mod
    if _train_mod is not None:
        return _train_mod
    with _train_mod_lock:
        if _train_mod is not None:
            return _train_mod
        logger.info("Lazy-loading pipeline.train (torch + lightkurve)…")
        from pipeline import train as mod
        _train_mod = mod
        logger.info("pipeline.train loaded")
        return mod


# ---------------------------------------------------------------------------
# GPU info — probed in a background daemon thread; never blocks HTTP.
# ---------------------------------------------------------------------------

_gpu_cache: dict = {
    "cuda_available": False,
    "name": None,
    "vram_total_bytes": None,
    "vram_free_bytes": None,
    "ready": False,
}
_gpu_lock = threading.Lock()


def _probe_gpu() -> None:
    result: dict = {
        "cuda_available": False,
        "name": None,
        "vram_total_bytes": None,
        "vram_free_bytes": None,
        "ready": True,
    }
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
        result["cuda_available"] = cuda
        if cuda:
            result["name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            result["vram_total_bytes"] = int(props.total_memory)
            try:
                free, _ = torch.cuda.mem_get_info(0)
                result["vram_free_bytes"] = int(free)
            except Exception:
                pass
    except Exception as exc:
        logger.warning("GPU probe failed: %s", exc)
    with _gpu_lock:
        _gpu_cache.update(result)
    logger.info("GPU probe done: cuda=%s name=%s", result["cuda_available"], result.get("name"))


threading.Thread(target=_probe_gpu, daemon=True, name="gpu-probe").start()


def _get_gpu_info() -> dict:
    with _gpu_lock:
        return dict(_gpu_cache)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size if path.exists() else 0
    except OSError:
        return 0


# ---------------------------------------------------------------------------
# Settings endpoints
# ---------------------------------------------------------------------------

@router.get("/settings")
def get_settings_view():
    """
    Returns quickly (<1 s). No heavy imports, no directory walks.
    """
    logger.debug("GET /settings — start")

    db = SessionLocal()
    try:
        n_targets = db.query(func.count(models.Target.id)).scalar() or 0
        n_analyses = db.query(func.count(models.Analysis.id)).scalar() or 0
        n_events = db.query(func.count(models.FlaggedEvent.id)).scalar() or 0
    finally:
        db.close()

    logger.debug("GET /settings — db done")

    db_file = settings.data_dir / "tess_anomaly.db"
    resp = {
        "data_dir": str(settings.data_dir),
        "mast_cache_dir": str(settings.mast_cache_dir),
        "model_weights_dir": str(settings.model_weights_dir),
        "database_url": settings.database_url,
        "database_size_bytes": _file_size(db_file),
        "model_weights_exist": settings.model_weights_path.exists(),
        "model_stats_exist": settings.model_stats_path.exists(),
        "counts": {
            "targets": int(n_targets),
            "analyses": int(n_analyses),
            "events": int(n_events),
        },
        "gpu": _get_gpu_info(),
    }
    logger.debug("GET /settings — responding")
    return resp


# ---------------------------------------------------------------------------
# Data-dir management
# ---------------------------------------------------------------------------

class DataDirBody(BaseModel):
    path: str = Field(..., min_length=1)


@router.put("/settings/data-dir")
def put_data_dir(body: DataDirBody):
    raw = body.path.strip()
    # Block obvious path-traversal attempts
    for component in raw.replace("\\", "/").split("/"):
        if component == "..":
            raise HTTPException(status_code=400, detail="Path must not contain '..' components")
    new_path = Path(raw).expanduser().resolve()
    try:
        new_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise HTTPException(status_code=400, detail=f"Cannot create directory: {e}") from e

    write_data_dir_to_env(str(new_path))
    settings.reload_from_env()
    os.environ["LIGHTKURVE_CACHE_DIR"] = str(settings.mast_cache_dir)
    try:
        from pipeline import fetch as fetch_mod
        fetch_mod._sync_lk_cache_dir()
    except Exception:
        pass

    recreate_engine()
    from db.models import Base
    from db.database import engine
    Base.metadata.create_all(bind=engine)

    return {
        "ok": True,
        "data_dir": str(settings.data_dir),
        "message": "DATA_DIR updated; database engine reconnected to new location.",
    }


# ---------------------------------------------------------------------------
# Training  (all pipeline.train access is lazy)
# ---------------------------------------------------------------------------

@router.get("/train/defaults")
def train_defaults():
    t = _get_train_module()
    targets = load_training_targets()
    return {
        "epochs": t.DEFAULT_EPOCHS,
        "batch_size": t.DEFAULT_BATCH_SIZE,
        "learning_rate": t.DEFAULT_LEARNING_RATE,
        "max_targets": None,
        "available_training_targets": len(targets),
    }


@router.get("/train/targets")
def get_training_targets():
    targets = load_training_targets()
    return {"count": len(targets), "targets": targets}


class AddTargetsBody(BaseModel):
    targets: list[dict] = Field(..., min_length=1)


@router.post("/train/targets")
def add_training_targets(body: AddTargetsBody):
    existing = load_training_targets()
    existing_ids = {t["tic_id"] for t in existing}
    added = 0
    for t in body.targets:
        tic = str(t.get("tic_id", "")).strip()
        if not tic or tic in existing_ids:
            continue
        existing.append({
            "tic_id": tic,
            "anomaly_score": t.get("anomaly_score"),
            "source": t.get("source", "scan"),
        })
        existing_ids.add(tic)
        added += 1
    save_training_targets(existing)
    return {"ok": True, "added": added, "total": len(existing)}


class RemoveTargetsBody(BaseModel):
    tic_ids: list[str] = Field(..., min_length=1)


@router.delete("/train/targets")
def remove_training_targets(body: RemoveTargetsBody):
    existing = load_training_targets()
    remove_set = set(body.tic_ids)
    filtered = [t for t in existing if t["tic_id"] not in remove_set]
    removed = len(existing) - len(filtered)
    save_training_targets(filtered)

    from pipeline import training_cache
    evicted = training_cache.evict_many(list(remove_set))
    if evicted:
        logger.info("Evicted %d training cache files for removed targets", evicted)

    return {"ok": True, "removed": removed, "total": len(filtered)}


@router.post("/train/targets/import-quiet")
def import_quiet_stars(max_score: float = 0.25, max_flags: int = 0, limit: int = 200):
    """Auto-import quiet stars from completed scan analyses into the training targets file."""
    db = SessionLocal()
    try:
        rows = (
            db.query(models.Target.tic_id, models.Analysis.anomaly_score)
            .join(models.Analysis, models.Analysis.target_id == models.Target.id)
            .filter(
                models.Analysis.status == models.AnalysisStatus.complete,
                models.Analysis.anomaly_score <= max_score,
                models.Analysis.flag_count <= max_flags,
            )
            .order_by(models.Analysis.anomaly_score.asc())
            .limit(limit)
            .all()
        )
    finally:
        db.close()

    existing = load_training_targets()
    existing_ids = {t["tic_id"] for t in existing}
    added = 0
    for tic_id, score in rows:
        if tic_id not in existing_ids:
            existing.append({
                "tic_id": tic_id,
                "anomaly_score": round(float(score), 4) if score is not None else None,
                "source": "scan",
            })
            existing_ids.add(tic_id)
            added += 1
    save_training_targets(existing)
    logger.info("Imported %d quiet stars into training targets (total %d)", added, len(existing))
    return {"ok": True, "added": added, "total": len(existing), "scanned": len(rows)}


@router.get("/train/quiet-stars")
def get_quiet_stars(
    max_score: float = 0.25,
    max_flags: int = 0,
    limit: int = 200,
):
    """Return TIC IDs of completed analyses with low anomaly scores and zero/few flags."""
    db = SessionLocal()
    try:
        rows = (
            db.query(models.Target.tic_id, models.Analysis.anomaly_score, models.Analysis.flag_count)
            .join(models.Analysis, models.Analysis.target_id == models.Target.id)
            .filter(
                models.Analysis.status == models.AnalysisStatus.complete,
                models.Analysis.anomaly_score <= max_score,
                models.Analysis.flag_count <= max_flags,
            )
            .order_by(models.Analysis.anomaly_score.asc())
            .limit(limit)
            .all()
        )
        stars = [
            {"tic_id": r[0], "anomaly_score": round(float(r[1]), 4), "flag_count": r[2]}
            for r in rows
        ]
        return {"count": len(stars), "max_score": max_score, "max_flags": max_flags, "stars": stars}
    finally:
        db.close()


class TrainStartBody(BaseModel):
    epochs: int = Field(default=50, ge=1, le=500)
    batch_size: int = Field(default=64, ge=4, le=512)
    learning_rate: float = Field(default=1e-3, ge=1e-7, le=1.0)
    max_targets: int | None = Field(default=None)
    use_quiet_stars: bool = Field(default=False, description="Auto-import quiet stars from scan data")
    quiet_max_score: float = Field(default=0.25, ge=0.0, le=1.0)
    quiet_max_flags: int = Field(default=0, ge=0)


@router.post("/train")
def post_train(body: TrainStartBody = Body(default_factory=TrainStartBody)):
    t = _get_train_module()

    custom_tics = None
    if body.use_quiet_stars:
        db = SessionLocal()
        try:
            rows = (
                db.query(models.Target.tic_id)
                .join(models.Analysis, models.Analysis.target_id == models.Target.id)
                .filter(
                    models.Analysis.status == models.AnalysisStatus.complete,
                    models.Analysis.anomaly_score <= body.quiet_max_score,
                    models.Analysis.flag_count <= body.quiet_max_flags,
                )
                .order_by(models.Analysis.anomaly_score.asc())
                .limit(200)
                .all()
            )
            custom_tics = [r[0] for r in rows]
        finally:
            db.close()
        if not custom_tics:
            raise HTTPException(
                status_code=422,
                detail=f"No quiet stars found (score <= {body.quiet_max_score}, flags <= {body.quiet_max_flags}). Run a scan first.",
            )
        logger.info("Training with %d quiet stars from scan data", len(custom_tics))

    ok, msg = t.run_training_async(
        {
            "epochs": body.epochs,
            "batch_size": body.batch_size,
            "learning_rate": body.learning_rate,
            "max_targets": body.max_targets,
            "custom_tics": custom_tics,
        }
    )
    if not ok:
        raise HTTPException(status_code=409, detail=msg)
    source = f"{len(custom_tics)} quiet stars from scans" if custom_tics else "built-in targets"
    return {"ok": True, "message": f"Training started ({source})"}


@router.get("/train/cache")
def get_training_cache_info():
    """Return stats about the training data cache."""
    from pipeline import training_cache
    cached = training_cache.list_cached()
    return {
        "cached_targets": len(cached),
        "tic_ids": cached,
        "size_bytes": training_cache.cache_size_bytes(),
        "cache_dir": str(settings.training_cache_dir),
    }


@router.delete("/train/cache")
def clear_training_cache():
    """Delete all cached training data, forcing re-fetch on next training run."""
    from pipeline import training_cache
    removed = training_cache.clear_all()
    return {"ok": True, "cleared": removed}


@router.get("/train/status")
def train_status():
    t = _get_train_module()
    return t.get_training_status()
