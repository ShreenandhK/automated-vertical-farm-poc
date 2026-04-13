"""
data_logger.py — SQLAlchemy Data Layer

Persists every inference result (local or cloud) to a SQLite database so
we have a permanent audit trail of what the system analysed and decided.

Why SQLite + SQLAlchemy?
────────────────────────
• SQLite needs zero infrastructure — perfect for a POC running on a
  Raspberry Pi or in a Docker container with a volume-mounted .db file.
• SQLAlchemy gives us Python objects instead of raw SQL strings, which
  reduces the chance of SQL injection and makes the code more readable.
• We use **synchronous** SQLAlchemy (not async) because the POC runs a
  single Uvicorn worker.  All DB calls are offloaded to a threadpool
  via `asyncio.to_thread()` in main.py so the ASGI event loop is never
  blocked (CLAUDE.md §6 Async & Blocking I/O).

Design decisions
────────────────
• The `AnalysisRecord` table stores one row per image analysis — both
  the AI prediction and metadata (source, timestamp, image path).
• `create_tables()` is called once at server startup from the lifespan
  hook.  It uses `create_all(checkfirst=True)` so it's safe to call
  repeatedly without dropping existing data.
• `log_analysis()` is the single public function that main.py calls.
  It creates its own session, commits, and closes — no session leaking.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from sqlalchemy import Column, DateTime, Float, Integer, String, Boolean, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# ──────────────────────────────────────────────
# Environment & logging
# ──────────────────────────────────────────────
load_dotenv()
logger = logging.getLogger(__name__)

# Read the database URL from .env.  The triple-slash in
# "sqlite:///./server/plant_monitor.db" means a relative path —
# this maps cleanly to a Docker volume mount later (CLAUDE.md §6).
DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./server/plant_monitor.db")

# ──────────────────────────────────────────────
# SQLAlchemy setup
# ──────────────────────────────────────────────
# `check_same_thread=False` is required for SQLite when the engine is
# shared across threads (which happens when we offload DB writes via
# asyncio.to_thread).  This is safe for our single-writer POC.
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,  # Set to True to see raw SQL in logs (noisy but useful for debugging)
)

# sessionmaker produces a factory — calling SessionLocal() creates a
# fresh session.  Each call to log_analysis() gets its own session to
# avoid cross-request leaks.
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

# Base class for all ORM models in this project.
Base = declarative_base()


# ──────────────────────────────────────────────
# ORM Models
# ──────────────────────────────────────────────
class AnalysisRecord(Base):
    """One row per image analysis — the permanent audit trail.

    Every time the /ingest/image endpoint processes an image, a record
    is written here with the prediction, confidence, source (local vs.
    cloud), and whether a disease was detected.  This lets a farmer
    review historical trends or export data for agronomists.
    """
    __tablename__ = "analysis_records"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Which image was analysed — the filename links back to the saved
    # JPEG in IMAGE_SAVE_DIR so a human can cross-reference.
    filename = Column(String, nullable=False)
    image_url = Column(String, nullable=False)

    # AI prediction results
    label = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    source = Column(String, nullable=False)  # "local", "cloud", or "local (cloud unavailable)"
    description = Column(String, default="")

    # Quick flag so we can query "show me all disease detections" without
    # parsing the label string every time.
    is_disease = Column(Boolean, nullable=False, default=False)

    # UTC timestamp — always use timezone-aware datetimes to avoid
    # ambiguity when the system runs across time zones.
    timestamp = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    def __repr__(self) -> str:
        return (
            f"<AnalysisRecord id={self.id} label='{self.label}' "
            f"confidence={self.confidence:.4f} source='{self.source}'>"
        )


# ──────────────────────────────────────────────
# Table creation
# ──────────────────────────────────────────────
def create_tables() -> None:
    """Create all ORM tables if they don't already exist.

    Called once during FastAPI lifespan startup.  Uses checkfirst=True
    internally so it's idempotent — safe to call on every server boot
    without losing existing data.
    """
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables verified/created at: %s", DATABASE_URL)


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────
def log_analysis(
    filename: str,
    image_url: str,
    label: str,
    confidence: float,
    source: str,
    is_disease: bool,
    description: str = "",
    timestamp: str | None = None,
) -> AnalysisRecord:
    """Persist an image analysis result to the database.

    Parameters
    ----------
    filename : str
        Saved JPEG filename (e.g., "abc123.jpg").
    image_url : str
        URL path the dashboard uses to load the thumbnail.
    label : str
        Predicted class name from the AI model.
    confidence : float
        Softmax probability (0.0–1.0) for the predicted class.
    source : str
        Which AI produced this result ("local" or "cloud").
    is_disease : bool
        Whether the label indicates a plant disease.
    description : str
        Optional plain-English explanation (from cloud fallback).
    timestamp : str | None
        ISO-format UTC timestamp.  If None, the current time is used.

    Returns
    -------
    AnalysisRecord
        The newly created database record (with .id populated).
    """
    # Parse the ISO timestamp if provided, otherwise use current UTC time.
    ts = (
        datetime.fromisoformat(timestamp) if timestamp
        else datetime.now(timezone.utc)
    )

    record = AnalysisRecord(
        filename=filename,
        image_url=image_url,
        label=label,
        confidence=confidence,
        source=source,
        description=description,
        is_disease=is_disease,
        timestamp=ts,
    )

    # Each write gets its own session — open, commit, close.
    # This prevents session leaks across concurrent requests.
    session = SessionLocal()
    try:
        session.add(record)
        session.commit()
        session.refresh(record)  # Populate the auto-generated .id
        logger.info("Logged analysis → id=%d, label='%s', source='%s'", record.id, label, source)
        return record
    except Exception:
        session.rollback()
        logger.exception("Failed to log analysis result to database.")
        raise
    finally:
        session.close()
