import csv
import io
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from db.database import get_db
from db import models

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/export")


def _ts_score(analysis: models.Analysis) -> float:
    ts = analysis.technosignature
    if ts and isinstance(ts, dict):
        return float(ts.get("composite_score") or 0)
    return 0.0


def _csv_response(text: str, filename: str) -> StreamingResponse:
    return StreamingResponse(
        iter([text]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/analyses")
def export_analyses(
    db: Session = Depends(get_db),
    sector: str | None = Query(None, description="Filter by sector"),
    min_score: float = Query(0.0),
):
    q = (
        db.query(models.Analysis)
        .join(models.Target, models.Analysis.target_id == models.Target.id, isouter=True)
        .filter(models.Analysis.status == models.AnalysisStatus.complete)
    )
    if sector:
        q = q.filter(models.Analysis.sector == sector)
    if min_score > 0:
        q = q.filter(models.Analysis.anomaly_score >= min_score)

    rows = q.order_by(models.Analysis.anomaly_score.desc()).all()

    fields = [
        "tic_id", "common_name", "sector", "anomaly_score",
        "technosignature_score", "known_period", "flag_count",
        "ra", "dec", "magnitude", "stellar_type", "created_at",
    ]

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    for a in rows:
        t = a.target
        writer.writerow({
            "tic_id": t.tic_id if t else "",
            "common_name": t.common_name or "" if t else "",
            "sector": a.sector or "",
            "anomaly_score": round(a.anomaly_score, 6) if a.anomaly_score is not None else "",
            "technosignature_score": round(_ts_score(a), 6),
            "known_period": round(a.known_period, 6) if a.known_period is not None else "",
            "flag_count": a.flag_count or 0,
            "ra": round(t.ra, 6) if t and t.ra is not None else "",
            "dec": round(t.dec, 6) if t and t.dec is not None else "",
            "magnitude": round(t.magnitude, 3) if t and t.magnitude is not None else "",
            "stellar_type": t.stellar_type or "" if t else "",
            "created_at": a.created_at.isoformat() if a.created_at else "",
        })

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    sector_tag = f"_sector{sector}" if sector else ""
    return _csv_response(buf.getvalue(), f"curvescope_analyses{sector_tag}_{stamp}.csv")


@router.get("/events")
def export_events(
    db: Session = Depends(get_db),
    sector: str | None = Query(None, description="Filter by sector"),
    event_type: str | None = Query(None),
):
    q = (
        db.query(models.FlaggedEvent)
        .join(models.Analysis, models.FlaggedEvent.analysis_id == models.Analysis.id)
        .join(models.Target, models.Analysis.target_id == models.Target.id, isouter=True)
        .filter(models.Analysis.status == models.AnalysisStatus.complete)
    )
    if sector:
        q = q.filter(models.Analysis.sector == sector)
    if event_type:
        try:
            et = models.EventType[event_type.strip().lower()]
        except KeyError:
            pass
        else:
            q = q.filter(models.FlaggedEvent.event_type == et)

    events = q.order_by(models.FlaggedEvent.anomaly_score.desc()).all()

    fields = [
        "tic_id", "sector", "event_type", "time_center_btjd",
        "duration_hours", "depth_ppm", "anomaly_score", "confidence",
        "centroid_shift_arcsec", "systematic_match", "notes",
    ]

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    for ev in events:
        a = ev.analysis
        t = a.target if a else None
        writer.writerow({
            "tic_id": t.tic_id if t else "",
            "sector": a.sector if a else "",
            "event_type": ev.event_type.value if ev.event_type else "",
            "time_center_btjd": round(ev.time_center, 6) if ev.time_center is not None else "",
            "duration_hours": round(ev.duration_hours, 4) if ev.duration_hours is not None else "",
            "depth_ppm": round(ev.depth_ppm, 2) if ev.depth_ppm is not None else "",
            "anomaly_score": round(ev.anomaly_score, 6) if ev.anomaly_score is not None else "",
            "confidence": round(ev.confidence, 4) if ev.confidence is not None else "",
            "centroid_shift_arcsec": round(ev.centroid_shift_arcsec, 4) if ev.centroid_shift_arcsec not in (None, -1.0) else "",
            "systematic_match": ev.systematic_match or "",
            "notes": ev.notes or "",
        })

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    sector_tag = f"_sector{sector}" if sector else ""
    return _csv_response(buf.getvalue(), f"curvescope_events{sector_tag}_{stamp}.csv")
