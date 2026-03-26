import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from db.database import engine
from db.models import Base
from routers import analyze, targets, events, scan, settings as settings_router
from brand import APP_DESCRIPTION, APP_NAME, APP_TAGLINE

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

NOISY_PATHS = {"/api/scan/status", "/api/health", "/api/train/status"}


class _QuietAccessFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(p in msg for p in NOISY_PATHS)


logging.getLogger("uvicorn.access").addFilter(_QuietAccessFilter())
logging.getLogger("lightkurve").setLevel(logging.WARNING)
logging.getLogger("routers.analyze").setLevel(logging.WARNING)
logging.getLogger("pipeline").setLevel(logging.WARNING)

@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    _migrate_add_missing_columns()
    _cleanup_stale_analyses()
    w_ok = settings.model_weights_path.exists()
    s_ok = settings.model_stats_path.exists()
    logging.getLogger(__name__).info(
        f"{APP_NAME} started | data_dir={settings.data_dir} | "
        f"model_weights={'present' if w_ok else 'missing'} | "
        f"model_stats={'present' if s_ok else 'missing'} | "
        f"weights_path={settings.model_weights_path}"
    )
    yield


app = FastAPI(
    title=f"{APP_NAME} · {APP_TAGLINE}",
    version="1.0",
    description=APP_DESCRIPTION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze.router)
app.include_router(targets.router)
app.include_router(events.router)
app.include_router(scan.router)
app.include_router(settings_router.router)


def _migrate_add_missing_columns():
    """Add columns that exist in the ORM model but not yet in the SQLite table."""
    import sqlalchemy as sa
    inspector = sa.inspect(engine)
    for table_name, table in Base.metadata.tables.items():
        if not inspector.has_table(table_name):
            continue
        existing = {col["name"] for col in inspector.get_columns(table_name)}
        for col in table.columns:
            if col.name not in existing:
                col_type = col.type.compile(engine.dialect)
                stmt = f'ALTER TABLE "{table_name}" ADD COLUMN "{col.name}" {col_type}'
                with engine.begin() as conn:
                    conn.execute(sa.text(stmt))
                logging.getLogger(__name__).info("Migrated: added column %s.%s (%s)", table_name, col.name, col_type)


def _cleanup_stale_analyses():
    """Clean up analyses left in bad states from crashes."""
    from db.database import SessionLocal
    from db.models import Analysis, AnalysisStatus, FlaggedEvent
    log = logging.getLogger(__name__)
    db = SessionLocal()
    try:
        orphans = db.query(Analysis).filter(Analysis.target_id == None).all()  # noqa: E711
        if orphans:
            for a in orphans:
                db.query(FlaggedEvent).filter(FlaggedEvent.analysis_id == a.id).delete()
                db.delete(a)
            db.commit()
            log.info("Deleted %d orphan analyses (no target linked)", len(orphans))

        stale = (
            db.query(Analysis)
            .filter(Analysis.status.in_([AnalysisStatus.pending, AnalysisStatus.running]))
            .all()
        )
        for a in stale:
            a.status = AnalysisStatus.failed
            a.error_message = "Interrupted by server restart"
        if stale:
            db.commit()
            log.info("Marked %d stale analyses as failed", len(stale))
    finally:
        db.close()


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "app": APP_NAME,
        "tagline": APP_TAGLINE,
        "data_dir": str(settings.data_dir),
    }
