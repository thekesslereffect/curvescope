from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from config import settings


def _make_engine_for_url(database_url: str):
    eng = create_engine(
        database_url,
        connect_args={"check_same_thread": False} if "sqlite" in database_url else {},
        echo=False,
    )
    if "sqlite" in database_url:

        @event.listens_for(eng, "connect")
        def _set_sqlite_pragma(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return eng


engine = _make_engine_for_url(settings.database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def recreate_engine() -> None:
    """Recreate SQLAlchemy engine after config paths change (e.g. new DATA_DIR)."""
    global engine, SessionLocal
    from config import settings as cfg

    engine.dispose(close=True)
    engine = _make_engine_for_url(cfg.database_url)
    SessionLocal.configure(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
