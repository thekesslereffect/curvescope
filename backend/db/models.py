from sqlalchemy import Boolean, Column, Integer, String, Float, DateTime, JSON, ForeignKey, Enum as SAEnum
from sqlalchemy.orm import relationship, DeclarativeBase
from datetime import datetime, timezone
import enum


class Base(DeclarativeBase):
    pass


class AnalysisStatus(str, enum.Enum):
    pending = "pending"
    running = "running"
    complete = "complete"
    failed = "failed"


class EventType(str, enum.Enum):
    transit = "transit"
    asymmetric = "asymmetric"
    depth_anomaly = "depth_anomaly"
    non_periodic = "non_periodic"
    exocomet = "exocomet"
    stellar_flare = "stellar_flare"
    stellar_spot = "stellar_spot"
    eclipsing_binary = "eclipsing_binary"
    stellar_variability = "stellar_variability"
    systematic = "systematic"
    contamination = "contamination"
    unknown = "unknown"


class Target(Base):
    __tablename__ = "targets"
    id = Column(Integer, primary_key=True)
    tic_id = Column(String, unique=True, nullable=False, index=True)
    common_name = Column(String)
    ra = Column(Float)
    dec = Column(Float)
    magnitude = Column(Float)
    stellar_type = Column(String)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    analyses = relationship("Analysis", back_populates="target")


class Analysis(Base):
    __tablename__ = "analyses"
    id = Column(Integer, primary_key=True)
    target_id = Column(Integer, ForeignKey("targets.id"), nullable=True, index=True)
    sector = Column(String)
    status = Column(SAEnum(AnalysisStatus), default=AnalysisStatus.pending, index=True)
    anomaly_score = Column(Float)
    known_period = Column(Float)
    flag_count = Column(Integer, default=0)
    raw_flux = Column(JSON)
    detrended_flux = Column(JSON)
    score_timeline = Column(JSON)
    periodogram = Column(JSON)
    wavelet = Column(JSON)
    centroid = Column(JSON)
    tpf_data = Column(JSON)
    technosignature = Column(JSON)
    has_chart_data = Column(Boolean, default=False)
    error_message = Column(String, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    target = relationship("Target", back_populates="analyses")
    events = relationship("FlaggedEvent", back_populates="analysis", cascade="all, delete-orphan")


class FlaggedEvent(Base):
    __tablename__ = "flagged_events"
    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey("analyses.id"), index=True)
    event_type = Column(SAEnum(EventType), index=True)
    time_center = Column(Float)
    duration_hours = Column(Float)
    depth_ppm = Column(Float)
    anomaly_score = Column(Float)
    confidence = Column(Float)
    notes = Column(String)
    centroid_shift_arcsec = Column(Float, default=-1.0)
    systematic_match = Column(String, nullable=True)
    analysis = relationship("Analysis", back_populates="events")
