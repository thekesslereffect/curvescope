from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api")


def _tess_sector_range():
    """TESS primary + extended; adjust upper bound as mission continues."""
    return list(range(1, 110))


@router.get("/sectors")
def list_sectors():
    return {"sectors": _tess_sector_range(), "note": "Use sequence_number / sector index from TESS data releases"}


class ScanStartRequest(BaseModel):
    sector: int = Field(..., ge=1, le=200)
    limit: int | None = Field(None, description="Max targets to process (None = all in sector)")
    skip_existing: bool = Field(False, description="Skip TICs that already have a completed analysis")


def _scanner():
    from pipeline.scanner import SectorScanner
    return SectorScanner.instance()


@router.post("/scan/start")
def scan_start(body: ScanStartRequest):
    scanner = _scanner()
    ok, msg = scanner.start(body.sector, limit=body.limit, skip_existing=body.skip_existing)
    if not ok:
        raise HTTPException(status_code=409, detail=msg)
    return {"ok": True, "message": msg}


@router.get("/scan/status")
def scan_status():
    return _scanner().get_state()


@router.post("/scan/stop")
def scan_stop():
    ok, msg = _scanner().stop()
    return {"ok": ok, "message": msg}
