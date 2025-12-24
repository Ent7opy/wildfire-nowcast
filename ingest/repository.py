"""Database helpers for FIRMS ingestion."""

from __future__ import annotations

from typing import Sequence

from sqlalchemy import JSON, bindparam, create_engine, text
from sqlalchemy.engine import Engine

from api.config import settings as api_settings
from ingest.models import DetectionRecord

_engine: Engine | None = None


def get_engine() -> Engine:
    """Create (or memoize) the SQLAlchemy engine."""
    global _engine
    if _engine is None:
        _engine = create_engine(api_settings.database_url, pool_pre_ping=True, future=True)
    return _engine


def create_ingest_batch(
    source: str,
    source_uri: str,
    area: str,
    day_range: int,
    metadata_extra: dict | None = None,
) -> int:
    """Insert a new ingest batch row and return its ID."""
    metadata = {"area": area, "day_range": day_range}
    if metadata_extra:
        metadata.update(metadata_extra)
    stmt = text(
        """
        INSERT INTO ingest_batches (
            source,
            source_uri,
            started_at,
            status,
            "metadata",
            records_fetched,
            records_inserted,
            records_skipped_duplicates
        )
        VALUES (
            :source,
            :source_uri,
            NOW(),
            'running',
            :metadata,
            0,
            0,
            0
        )
        RETURNING id
        """
    ).bindparams(bindparam("metadata", type_=JSON))
    with get_engine().begin() as conn:
        result = conn.execute(
            stmt,
            {
                "source": source,
                "source_uri": source_uri,
                "metadata": metadata,
            },
        )
        batch_id = result.scalar_one()
    return int(batch_id)


def finalize_ingest_batch(
    batch_id: int,
    *,
    status: str,
    fetched: int,
    inserted: int,
    skipped: int,
) -> None:
    """Update ingest batch metrics and mark completion."""
    stmt = text(
        """
        UPDATE ingest_batches
        SET
            completed_at = NOW(),
            status = :status,
            record_count = :inserted,
            records_fetched = :fetched,
            records_inserted = :inserted,
            records_skipped_duplicates = :skipped
        WHERE id = :batch_id
        """
    )
    with get_engine().begin() as conn:
        conn.execute(
            stmt,
            {
                "batch_id": batch_id,
                "status": status,
                "fetched": fetched,
                "inserted": inserted,
                "skipped": skipped,
            },
        )


def insert_detections(detections: Sequence[DetectionRecord]) -> int:
    """Bulk insert detections and return the number of inserted rows."""
    if not detections:
        return 0

    insert_stmt = text(
        """
        INSERT INTO fire_detections (
            geom,
            lat,
            lon,
            acq_time,
            sensor,
            source,
            confidence,
            brightness,
            bright_t31,
            frp,
            scan,
            track,
            raw_properties,
            ingest_batch_id,
            dedupe_hash
        )
        VALUES (
            ST_SetSRID(ST_MakePoint(:lon, :lat), 4326),
            :lat,
            :lon,
            :acq_time,
            :sensor,
            :source,
            :confidence,
            :brightness,
            :bright_t31,
            :frp,
            :scan,
            :track,
            :raw_properties,
            :ingest_batch_id,
            :dedupe_hash
        )
        ON CONFLICT (source, dedupe_hash) DO NOTHING
        """
    ).bindparams(bindparam("raw_properties", type_=JSON))
    parameters = [record.to_parameters() for record in detections]

    with get_engine().begin() as conn:
        result = conn.execute(insert_stmt, parameters)
        inserted = result.rowcount or 0

    return inserted


