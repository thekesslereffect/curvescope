"""
Sector-wide batch scanning: discover TIC IDs in a TESS sector and run the pipeline for each.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


def get_sector_targets(sector: int, limit: int | None = None) -> list[str]:
    """
    Return unique TIC IDs (numeric strings) that have TESS light-curve products in the given sector.
    Uses MAST Observations (astroquery).
    """
    from astroquery.mast import Observations

    logger.info(f"Querying MAST for TESS sector {sector} targets...")
    obs = Observations.query_criteria(
        obs_collection="TESS",
        dataproduct_type="timeseries",
        sequence_number=sector,
    )
    if obs is None or len(obs) == 0:
        logger.warning(f"No observations returned for sector {sector}")
        return []

    col = "target_name" if "target_name" in obs.colnames else None
    if col is None:
        for c in ("target_name", "objID", "obs_id"):
            if c in obs.colnames:
                col = c
                break
    if col is None:
        logger.error("Could not find target column in MAST table")
        return []

    tics: set[str] = set()
    for row in obs:
        name = str(row[col]).strip()
        if not name or name == "None":
            continue
        if name.upper().startswith("TIC"):
            parts = name.replace("TIC", "").strip().split()
            tid = parts[0] if parts else name
        else:
            tid = name.split()[0] if name else name
        if tid.isdigit():
            tics.add(tid)

    ordered = sorted(tics, key=lambda x: int(x))
    if limit is not None:
        ordered = ordered[:limit]
    logger.info(f"Sector {sector}: {len(ordered)} unique TIC targets")
    return ordered


@dataclass
class ScanState:
    running: bool = False
    stop_requested: bool = False
    sector: int | None = None
    total: int = 0
    completed: int = 0
    skipped: int = 0
    current_tic: str | None = None
    current_phase: str = ""
    errors: list[dict[str, str]] = field(default_factory=list)
    results_preview: list[dict[str, Any]] = field(default_factory=list)
    message: str = ""


class SectorScanner:
    _instance: SectorScanner | None = None
    _lock = threading.Lock()

    def __init__(self):
        self._state = ScanState()
        self._thread: threading.Thread | None = None

    @classmethod
    def instance(cls) -> SectorScanner:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def get_state(self) -> dict[str, Any]:
        with self._lock:
            return {
                "running": self._state.running,
                "sector": self._state.sector,
                "total": self._state.total,
                "completed": self._state.completed,
                "skipped": self._state.skipped,
                "current_tic": self._state.current_tic,
                "current_phase": self._state.current_phase,
                "errors": list(self._state.errors[-50:]),
                "results_preview": list(self._state.results_preview[:30]),
                "message": self._state.message,
            }

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self._state.current_phase = phase

    def start(self, sector: int, limit: int | None = None, skip_existing: bool = False) -> tuple[bool, str]:
        with self._lock:
            if self._state.running:
                return False, "A scan is already running"
            self._state = ScanState(
                running=True,
                sector=sector,
                message=f"Starting sector {sector}...",
            )
            self._state.stop_requested = False

        self._thread = threading.Thread(
            target=self._run_loop,
            args=(sector, limit, skip_existing),
            daemon=True,
            name=f"sector-scan-{sector}",
        )
        self._thread.start()
        return True, "Scan started"

    def stop(self) -> tuple[bool, str]:
        with self._lock:
            if not self._state.running:
                return False, "No scan running"
            self._state.stop_requested = True
        return True, "Stop requested"

    def _run_loop(self, sector: int, limit: int | None, skip_existing: bool = False) -> None:
        from db.database import SessionLocal
        from db import models
        from routers.analyze import run_pipeline, PipelineStopped
        from pipeline.fetch import clear_query_caches

        try:
            with self._lock:
                self._state.current_phase = "Querying MAST for sector targets..."

            tics = get_sector_targets(sector, limit=limit)

            product_cache: dict = {}
            if tics:
                try:
                    with self._lock:
                        self._state.current_phase = f"Prefetching product list for {len(tics)} targets..."
                    from pipeline.s3_fetch import prefetch_sector_products
                    product_cache = prefetch_sector_products(sector)
                    logger.info("Prefetched products for %d TICs", len(product_cache))
                except Exception as e:
                    logger.warning("Product prefetch failed (%s), will use per-target MAST search", e)
                    product_cache = {}

            if skip_existing and tics:
                db = SessionLocal()
                try:
                    rows = (
                        db.query(models.Target.tic_id)
                        .join(models.Analysis, models.Analysis.target_id == models.Target.id)
                        .filter(
                            models.Analysis.status.in_([
                                models.AnalysisStatus.complete,
                                models.AnalysisStatus.running,
                                models.AnalysisStatus.pending,
                            ])
                        )
                        .all()
                    )
                    existing = {r[0] for r in rows}
                    before = len(tics)
                    tics = [t for t in tics if t not in existing]
                    skipped = before - len(tics)
                    logger.info("Skip existing: %d already analyzed or in-progress, %d remaining", skipped, len(tics))
                finally:
                    db.close()
                with self._lock:
                    self._state.skipped = skipped

            with self._lock:
                self._state.total = len(tics)
                self._state.message = f"Scanning {len(tics)} targets in sector {sector}"
                if skip_existing and self._state.skipped:
                    self._state.message += f" ({self._state.skipped} already done, skipped)"

            if not tics:
                with self._lock:
                    self._state.running = False
                    self._state.total = 0
                    if skip_existing:
                        self._state.message = "All targets in this sector already analyzed"
                    else:
                        self._state.message = "No targets found for this sector (MAST returned empty)"
                return

            def _is_stop_requested() -> bool:
                with self._lock:
                    return self._state.stop_requested

            for tic in tics:
                with self._lock:
                    if self._state.stop_requested:
                        self._state.message = "Stopped by user"
                        break
                    self._state.current_tic = tic
                    self._state.current_phase = ""

                db = SessionLocal()
                try:
                    analysis = models.Analysis(
                        status=models.AnalysisStatus.pending,
                        sector=str(sector),
                    )
                    db.add(analysis)
                    db.commit()
                    db.refresh(analysis)
                    aid = analysis.id
                finally:
                    db.close()

                try:
                    run_pipeline(
                        aid, f"TIC {tic}", str(sector),
                        on_phase=self.set_phase,
                        stop_check=_is_stop_requested,
                        include_charts=False,
                        prefetched_products=product_cache,
                    )
                except PipelineStopped:
                    logger.info("Scan stopped by user during TIC %s", tic)
                    with self._lock:
                        self._state.message = "Stopped by user"
                    break
                except Exception as e:
                    logger.exception("Pipeline error for TIC %s", tic)
                    with self._lock:
                        self._state.errors.append({"tic": tic, "error": str(e)[:300]})
                        if len(self._state.errors) > 100:
                            self._state.errors = self._state.errors[-50:]
                        self._state.completed += 1
                else:
                    preview = _fetch_analysis_preview(aid)
                    with self._lock:
                        self._state.completed += 1
                        if preview:
                            self._state.results_preview.append(preview)
                            self._state.results_preview.sort(
                                key=lambda x: (
                                    x.get("anomaly_score") or 0,
                                    x.get("technosignature_score") or 0,
                                ),
                                reverse=True,
                            )
                            self._state.results_preview = self._state.results_preview[:30]
                finally:
                    clear_query_caches()

            with self._lock:
                self._state.running = False
                self._state.current_tic = None
                if not self._state.stop_requested and self._state.message != "Stopped by user":
                    self._state.message = "Scan finished"
        except Exception as e:
            logger.exception("Sector scan failed")
            with self._lock:
                self._state.running = False
                self._state.message = f"Scan error: {e}"
                self._state.errors.append({"tic": "", "error": str(e)})


def _fetch_analysis_preview(analysis_id: int) -> dict[str, Any] | None:
    from db.database import SessionLocal
    from db import models

    db = SessionLocal()
    try:
        a = db.query(models.Analysis).filter_by(id=analysis_id).first()
        if not a or a.status != models.AnalysisStatus.complete:
            return None
        tic = a.target.tic_id if a.target else None
        techno = 0.0
        if a.technosignature and isinstance(a.technosignature, dict):
            techno = float(a.technosignature.get("composite_score") or 0)
        return {
            "analysis_id": a.id,
            "tic_id": tic,
            "anomaly_score": a.anomaly_score,
            "technosignature_score": round(techno, 4),
            "flag_count": a.flag_count,
        }
    finally:
        db.close()
