from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from db.database import get_db
from db import models

router = APIRouter(prefix="/api")


@router.get("/targets")
def list_targets(db: Session = Depends(get_db)):
    targets = db.query(models.Target).order_by(models.Target.created_at.desc()).all()
    return [
        {
            "id": t.id,
            "tic_id": t.tic_id,
            "common_name": t.common_name,
            "ra": t.ra,
            "dec": t.dec,
            "magnitude": t.magnitude,
            "stellar_type": t.stellar_type,
            "analysis_count": len(t.analyses),
        }
        for t in targets
    ]


@router.get("/targets/{target_id}")
def get_target(target_id: int, db: Session = Depends(get_db)):
    target = db.query(models.Target).filter_by(id=target_id).first()
    if not target:
        raise HTTPException(status_code=404, detail="Target not found")
    return {
        "id": target.id,
        "tic_id": target.tic_id,
        "common_name": target.common_name,
        "ra": target.ra,
        "dec": target.dec,
        "analyses": [
            {"id": a.id, "status": a.status.value, "anomaly_score": a.anomaly_score, "flag_count": a.flag_count}
            for a in target.analyses
        ],
    }
