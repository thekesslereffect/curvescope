from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from db.database import get_db
from db import models

router = APIRouter(prefix="/api")


@router.get("/events")
def list_events(
    event_type: str = Query(None),
    min_score: float = Query(0.0),
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
):
    q = db.query(models.FlaggedEvent).filter(models.FlaggedEvent.anomaly_score >= min_score)
    if event_type:
        try:
            et = models.EventType[event_type.strip().lower()]
        except KeyError:
            raise HTTPException(status_code=400, detail=f"Invalid event_type: {event_type}") from None
        q = q.filter(models.FlaggedEvent.event_type == et)
    q = q.order_by(models.FlaggedEvent.anomaly_score.desc()).limit(limit)

    return [
        {
            "id": ev.id,
            "analysis_id": ev.analysis_id,
            "event_type": ev.event_type.value if ev.event_type else None,
            "time_center": ev.time_center,
            "duration_hours": ev.duration_hours,
            "depth_ppm": ev.depth_ppm,
            "anomaly_score": ev.anomaly_score,
            "confidence": ev.confidence,
            "notes": ev.notes,
            "centroid_shift_arcsec": ev.centroid_shift_arcsec,
            "systematic_match": ev.systematic_match,
        }
        for ev in q.all()
    ]
