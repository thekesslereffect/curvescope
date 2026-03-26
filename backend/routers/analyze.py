"""
Analyze router.

Pipeline modules (torch, lightkurve, scipy…) are imported lazily inside
run_pipeline / get_latest_analysis so the server starts instantly.
"""

import gc
import logging
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import and_, cast, exists, func, or_, Float
from sqlalchemy.orm import Session

from db.database import get_db
from db import models
from config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


class AnalyzeRequest(BaseModel):
    identifier: str
    sector: str = "all"


class AnalyzeResponse(BaseModel):
    analysis_id: int
    status: str


def _serialize_analysis(analysis: models.Analysis) -> dict[str, Any]:
    from pipeline.analysis_cache import get as chart_store_get

    chart = chart_store_get(analysis.id)

    result: dict[str, Any] = {
        "id": analysis.id,
        "status": analysis.status.value if analysis.status else "unknown",
        "sector": analysis.sector,
        "anomaly_score": analysis.anomaly_score,
        "known_period": analysis.known_period,
        "flag_count": analysis.flag_count,
        "raw_flux": chart["raw_flux"] if chart else analysis.raw_flux,
        "detrended_flux": chart["detrended_flux"] if chart else analysis.detrended_flux,
        "score_timeline": chart["score_timeline"] if chart else analysis.score_timeline,
        "periodogram": chart["periodogram"] if chart else analysis.periodogram,
        "wavelet": chart["wavelet"] if chart else analysis.wavelet,
        "centroid": chart["centroid"] if chart else analysis.centroid,
        "tpf_data": chart["tpf_data"] if chart else getattr(analysis, "tpf_data", None),
        "technosignature": analysis.technosignature,
        "error_message": analysis.error_message,
        "created_at": analysis.created_at.isoformat() if analysis.created_at else None,
        "target": None,
        "events": [],
    }

    if analysis.target:
        result["target"] = {
            "id": analysis.target.id,
            "tic_id": analysis.target.tic_id,
            "common_name": analysis.target.common_name,
            "ra": analysis.target.ra,
            "dec": analysis.target.dec,
            "magnitude": analysis.target.magnitude,
            "stellar_type": analysis.target.stellar_type,
        }

    for ev in analysis.events:
        result["events"].append({
            "id": ev.id,
            "event_type": ev.event_type.value if ev.event_type else "unknown",
            "time_center": ev.time_center,
            "duration_hours": ev.duration_hours,
            "depth_ppm": ev.depth_ppm,
            "anomaly_score": ev.anomaly_score,
            "confidence": ev.confidence,
            "notes": ev.notes,
            "centroid_shift_arcsec": ev.centroid_shift_arcsec,
            "systematic_match": ev.systematic_match,
        })

    return result


def _serialize_analysis_summary(analysis: models.Analysis) -> dict[str, Any]:
    techno = 0.0
    if analysis.technosignature and isinstance(analysis.technosignature, dict):
        techno = float(analysis.technosignature.get("composite_score") or 0)

    tic_id = analysis.target.tic_id if analysis.target else None
    common_name = analysis.target.common_name if analysis.target else None

    return {
        "id": analysis.id,
        "status": analysis.status.value if analysis.status else "unknown",
        "sector": analysis.sector,
        "anomaly_score": analysis.anomaly_score,
        "technosignature_score": round(techno, 4),
        "known_period": analysis.known_period,
        "flag_count": analysis.flag_count or 0,
        "tic_id": tic_id,
        "common_name": common_name,
        "created_at": analysis.created_at.isoformat() if analysis.created_at else None,
        "error_message": analysis.error_message,
    }


@router.post("/analyze", response_model=AnalyzeResponse)
def start_analysis(req: AnalyzeRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    analysis = models.Analysis(status=models.AnalysisStatus.pending, sector=req.sector)
    db.add(analysis)
    db.commit()
    db.refresh(analysis)

    background_tasks.add_task(run_pipeline, analysis.id, req.identifier, req.sector)
    return {"analysis_id": analysis.id, "status": "pending"}


@router.get("/analysis/latest")
def get_latest_analysis(identifier: str = Query(..., min_length=1), db: Session = Depends(get_db)):
    from pipeline import fetch

    s = identifier.strip()
    target = None
    digits_only = "".join(c for c in s if c.isdigit())
    if len(digits_only) >= 6:
        target = db.query(models.Target).filter_by(tic_id=digits_only).first()
    if not target:
        try:
            target_info = fetch.resolve_target(s)
            tic_id = target_info["tic_id"]
            target = db.query(models.Target).filter_by(tic_id=tic_id).first()
        except Exception as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
    if not target:
        raise HTTPException(status_code=404, detail="No analyses for this target yet")

    analysis = (
        db.query(models.Analysis)
        .filter(
            models.Analysis.target_id == target.id,
            models.Analysis.status == models.AnalysisStatus.complete,
        )
        .order_by(models.Analysis.created_at.desc())
        .first()
    )
    if not analysis:
        raise HTTPException(status_code=404, detail="No completed analysis for this target")

    return _serialize_analysis(analysis)


@router.get("/analysis/{analysis_id}")
def get_analysis(analysis_id: int, db: Session = Depends(get_db)):
    analysis = db.query(models.Analysis).filter_by(id=analysis_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return _serialize_analysis(analysis)


@router.get("/analyses")
def list_analyses(
    db: Session = Depends(get_db),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    sort_by: str = Query("anomaly_score"),
    event_type: str | None = Query(None),
    min_score: float = Query(0.0),
    search: str | None = Query(None),
    include_failed: bool = Query(False),
):
    if sort_by not in ("anomaly_score", "technosignature_score", "created_at"):
        raise HTTPException(status_code=400, detail="sort_by must be anomaly_score, technosignature_score, or created_at")

    q = db.query(models.Analysis).join(models.Target, models.Analysis.target_id == models.Target.id, isouter=True)

    if not include_failed:
        q = q.filter(models.Analysis.status == models.AnalysisStatus.complete)
    else:
        q = q.filter(
            models.Analysis.status.in_(
                [models.AnalysisStatus.complete, models.AnalysisStatus.failed]
            )
        )

    if min_score > 0:
        q = q.filter(models.Analysis.anomaly_score >= min_score)

    if search:
        s = search.strip()
        s_digits = "".join(c for c in s if c.isdigit())
        conds = []
        if s_digits:
            conds.append(models.Target.tic_id.contains(s_digits))
        if len(s) >= 2:
            conds.append(models.Target.common_name.ilike(f"%{s}%"))
        if conds:
            q = q.filter(or_(*conds))

    if event_type:
        try:
            et = models.EventType[event_type.strip().lower()]
        except KeyError:
            raise HTTPException(status_code=400, detail=f"Invalid event_type: {event_type}") from None
        q = q.filter(
            exists().where(
                and_(
                    models.FlaggedEvent.analysis_id == models.Analysis.id,
                    models.FlaggedEvent.event_type == et,
                )
            )
        )

    techno_expr = cast(
        func.json_extract(models.Analysis.technosignature, "$.composite_score"),
        Float,
    )

    if sort_by == "anomaly_score":
        q = q.order_by(models.Analysis.anomaly_score.desc().nullslast(), models.Analysis.id.desc())
    elif sort_by == "technosignature_score":
        q = q.order_by(techno_expr.desc().nullslast(), models.Analysis.id.desc())
    else:
        q = q.order_by(models.Analysis.created_at.desc().nullslast(), models.Analysis.id.desc())

    total = q.count()
    offset = (page - 1) * page_size
    rows = q.offset(offset).limit(page_size).all()

    summaries = [_serialize_analysis_summary(a) for a in rows]

    high_ts = (
        db.query(func.max(techno_expr)).filter(models.Analysis.status == models.AnalysisStatus.complete).scalar()
    )
    anomaly_high = (
        db.query(func.max(models.Analysis.anomaly_score))
        .filter(models.Analysis.status == models.AnalysisStatus.complete)
        .scalar()
    )
    interesting = (
        db.query(func.count(models.FlaggedEvent.id))
        .join(models.Analysis, models.FlaggedEvent.analysis_id == models.Analysis.id)
        .filter(
            models.Analysis.status == models.AnalysisStatus.complete,
            models.FlaggedEvent.event_type.not_in([
                models.EventType.systematic,
                models.EventType.contamination,
                models.EventType.eclipsing_binary,
                models.EventType.stellar_variability,
            ]),
        )
        .scalar()
        or 0
    )

    return {
        "items": summaries,
        "total": total,
        "page": page,
        "page_size": page_size,
        "summary": {
            "total_complete": db.query(func.count(models.Analysis.id))
            .filter(models.Analysis.status == models.AnalysisStatus.complete)
            .scalar()
            or 0,
            "interesting_event_count": int(interesting),
            "max_technosignature": float(high_ts) if high_ts is not None else 0.0,
            "max_anomaly_score": float(anomaly_high) if anomaly_high is not None else 0.0,
        },
    }


def _jsonable(obj):
    """Recursively convert numpy types so json.dumps (SQLAlchemy JSON columns) won't choke."""
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


class PipelineStopped(Exception):
    """Raised when a scan stop is requested mid-pipeline."""
    pass


def _cleanup_superseded(db, new_analysis: models.Analysis) -> None:
    """Delete older analyses for the same target+sector, keeping only the latest."""
    if not new_analysis.target_id:
        return
    from pipeline.analysis_cache import evict
    old = (
        db.query(models.Analysis)
        .filter(
            models.Analysis.target_id == new_analysis.target_id,
            models.Analysis.sector == new_analysis.sector,
            models.Analysis.id != new_analysis.id,
        )
        .all()
    )
    if not old:
        return
    for a in old:
        db.query(models.FlaggedEvent).filter(models.FlaggedEvent.analysis_id == a.id).delete()
        try:
            evict(a.id)
        except Exception:
            pass
        db.delete(a)
    db.commit()
    logger.info("Cleaned up %d superseded analysis(es) for target_id=%s sector=%s",
                len(old), new_analysis.target_id, new_analysis.sector)


def run_pipeline(analysis_id: int, identifier: str, sector: str,
                 on_phase=None, stop_check=None, include_charts: bool = True,
                 prefetched_products: dict | None = None):
    from db.database import SessionLocal
    from pipeline import fetch, clean, autoencoder, periodogram, wavelet, centroid, classifier, technosignature
    from pipeline.analysis_cache import prepare_chart_data, put as chart_store_put

    def _phase(name: str):
        if stop_check and stop_check():
            raise PipelineStopped("Stopped by user")
        logger.info(f"[{analysis_id}] {name}")
        if on_phase:
            on_phase(name)

    db = SessionLocal()

    analysis = db.query(models.Analysis).filter_by(id=analysis_id).first()
    if not analysis:
        db.close()
        return

    try:
        analysis.status = models.AnalysisStatus.running
        db.commit()

        _phase(f"Resolving target: {identifier}")
        target_info = fetch.resolve_target(identifier)

        tic_id = target_info["tic_id"]
        common_name = target_info.get("common_name")
        del target_info

        lc_pairs = None
        tp_pairs = None
        if prefetched_products and tic_id in prefetched_products:
            lc_pairs = prefetched_products[tic_id].get("lc")
            tp_pairs = prefetched_products[tic_id].get("tp")

        _phase(f"Downloading light curve for TIC {tic_id}")
        raw_data = fetch.fetch_light_curve(tic_id, sector, prefetched_lc_pairs=lc_pairs)
        fetch.clear_query_caches()

        target = db.query(models.Target).filter_by(tic_id=tic_id).first()
        if not target:
            target = models.Target(tic_id=tic_id, common_name=common_name)
            db.add(target)
            db.commit()
            db.refresh(target)
        analysis.target_id = target.id

        time_list = raw_data["time"]
        raw_flux_list = raw_data["flux"]
        del raw_data

        import threading

        _phase("Cleaning & detrending")
        centroid_result_holder: list = []
        centroid_error_holder: list = []

        def _bg_centroid():
            try:
                centroid_result_holder.append(
                    centroid.compute_centroid(tic_id, sector, prefetched_tp_pairs=tp_pairs)
                )
            except Exception as e:
                centroid_error_holder.append(e)

        tpf_thread = threading.Thread(target=_bg_centroid, daemon=True, name=f"tpf-{tic_id}")
        tpf_thread.start()

        flux_norm = clean.normalize_flux(raw_flux_list)
        del raw_flux_list
        flux_clean, _ = clean.remove_outliers(flux_norm)
        flux_detrended = clean.detrend_flux(time_list, flux_clean)
        del flux_clean

        _phase("Autoencoder scoring")
        score_result = autoencoder.score_light_curve(
            flux_detrended, str(settings.model_weights_path)
        )

        _phase("BLS periodogram")
        period_result = periodogram.run_bls(time_list, flux_detrended)

        _phase("Wavelet transform")
        wavelet_result = wavelet.run_wavelet(time_list, flux_detrended)

        _phase("Centroid analysis (waiting for TPF)")
        tpf_thread.join(timeout=300)
        if centroid_error_holder:
            logger.warning("[%s] Centroid failed in background: %s", analysis_id, centroid_error_holder[0])
            centroid_result = {"available": False}
            tpf_pixels = {"available": False}
        elif centroid_result_holder:
            centroid_result = centroid_result_holder[0]
            tpf_pixels = centroid_result.pop("tpf_pixels", {"available": False})
        else:
            logger.warning("[%s] Centroid thread timed out", analysis_id)
            centroid_result = {"available": False}
            tpf_pixels = {"available": False}
        fetch.clear_query_caches()

        _phase("Classifying events")
        events_list = classifier.find_dip_events(
            time_list,
            flux_detrended,
            score_result["score_per_point"],
            wavelet_result=wavelet_result,
            centroid_result=centroid_result,
            bls_result=period_result,
        )
        events_list, selected_period = classifier.analyze_event_ensemble(
            events_list,
            period_result,
            score_result["score_per_point"],
            time_range=(float(time_list[0]), float(time_list[-1])),
        )

        _phase("Technosignature catalogs")
        unknown_events = [e for e in events_list if e["event_type"] == "unknown"]
        techno_result = technosignature.analyze(
            tic_id=tic_id,
            time=time_list,
            flux=flux_detrended,
            unknown_events=unknown_events,
            period_result=period_result,
        )
        del unknown_events
        fetch.clear_query_caches()

        if include_charts:
            _phase("Preparing chart data")
            chart_data = prepare_chart_data(
                raw_flux=_jsonable({"time": time_list, "flux": flux_norm}),
                detrended_flux=_jsonable({"time": time_list, "flux": flux_detrended}),
                score_timeline=_jsonable({"time": time_list, "score": score_result["score_per_point"]}),
                periodogram=_jsonable({"period": period_result["periods"], "power": period_result["powers"]}),
                wavelet=_jsonable(wavelet_result),
                centroid=_jsonable(centroid_result),
                tpf_data=_jsonable(tpf_pixels),
            )
            chart_store_put(analysis_id, chart_data)
            del chart_data
            analysis.has_chart_data = True
        else:
            analysis.has_chart_data = False

        del flux_norm, flux_detrended, wavelet_result, centroid_result, tpf_pixels

        analysis.raw_flux = None
        analysis.detrended_flux = None
        analysis.score_timeline = None
        analysis.periodogram = None
        analysis.wavelet = None
        analysis.centroid = None
        analysis.tpf_data = None
        analysis.technosignature = _jsonable(techno_result)
        analysis.anomaly_score = float(score_result["combined_score"])
        bp = period_result.get("best_period_days") or 0
        analysis.known_period = (
            float(selected_period)
            if selected_period is not None
            else (float(bp) if bp > 0 else None)
        )
        analysis.flag_count = len([e for e in events_list if e["event_type"] not in ("systematic", "contamination")])
        del techno_result, score_result, period_result, time_list

        for ev in events_list:
            safe = _jsonable(ev)
            db.add(models.FlaggedEvent(
                analysis_id=analysis.id,
                event_type=safe["event_type"],
                time_center=safe["time_center"],
                duration_hours=safe["duration_hours"],
                depth_ppm=safe["depth_ppm"],
                anomaly_score=safe["anomaly_score"],
                confidence=safe["confidence"],
                notes=safe["notes"],
                centroid_shift_arcsec=safe.get("centroid_shift_arcsec", -1.0),
                systematic_match=safe.get("systematic_match"),
            ))
        del events_list

        analysis.status = models.AnalysisStatus.complete
        db.commit()
        logger.info(f"[{analysis_id}] Complete")

        _cleanup_superseded(db, analysis)

    except PipelineStopped:
        raise
    except Exception as e:
        logger.error(f"[{analysis_id}] Pipeline failed: {e}", exc_info=True)
        try:
            db.rollback()
            analysis = db.query(models.Analysis).filter_by(id=analysis_id).first()
            if analysis:
                analysis.status = models.AnalysisStatus.failed
                analysis.error_message = str(e)[:2000]
                db.commit()
        except Exception:
            logger.error(f"[{analysis_id}] Could not persist failure status", exc_info=True)
    finally:
        try:
            fetch.clear_mast_downloads()
        except Exception:
            logger.warning("[%s] MAST download cleanup failed", analysis_id, exc_info=True)
        db.close()
        gc.collect()


@router.delete("/analyses")
def delete_all_analyses(db: Session = Depends(get_db)):
    """Remove every analysis and its flagged events. Targets are kept."""
    from pipeline.analysis_cache import clear_all
    ev_count = db.query(models.FlaggedEvent).delete()
    an_count = db.query(models.Analysis).delete()
    db.commit()
    mem_cleared = clear_all()
    logger.info("Purged %d analyses, %d events, %d in-memory chart entries", an_count, ev_count, mem_cleared)
    return {"ok": True, "deleted_analyses": an_count, "deleted_events": ev_count}
