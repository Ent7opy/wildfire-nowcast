"""Database helpers for FIRMS ingestion."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

from sqlalchemy import JSON, bindparam, create_engine, text
from sqlalchemy.engine import Connection, Engine

from api.config import settings as api_settings
from ingest.models import DetectionRecord

_engine: Engine | None = None


REQUIRED_SCORING_COLUMNS: tuple[str, ...] = (
    "confidence_score",
    "false_source_masked",
    "persistence_score",
    "landcover_score",
    "weather_score",
    "fire_likelihood",
)

REQUIRED_DENOISER_COLUMNS: tuple[str, ...] = (
    "denoised_score",
    "is_noise",
)

REQUIRED_DENOISER_V2_COLUMNS: tuple[str, ...] = (
    "front_id",
    "event_id",
    "event_score",
    "denoiser_decision",
    "review_required",
)

_ALLOWED_COMPLETENESS_COLUMNS: set[str] = (
    set(REQUIRED_SCORING_COLUMNS) | set(REQUIRED_DENOISER_COLUMNS) | set(REQUIRED_DENOISER_V2_COLUMNS)
)


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


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
    conn: Connection | None = None,
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
    params = {
        "batch_id": batch_id,
        "status": status,
        "fetched": fetched,
        "inserted": inserted,
        "skipped": skipped,
    }
    if conn is not None:
        conn.execute(stmt, params)
    else:
        with get_engine().begin() as new_conn:
            new_conn.execute(stmt, params)


def get_ingest_watermark(source: str, area_key: str) -> dict | None:
    """Fetch the current ingestion watermark for a source and area."""
    stmt = text(
        """
        SELECT
            source,
            area_key,
            last_acq_time_utc,
            last_batch_id,
            updated_at
        FROM ingest_watermarks
        WHERE source = :source
          AND area_key = :area_key
        """
    )
    with get_engine().begin() as conn:
        row = conn.execute(
            stmt,
            {
                "source": source,
                "area_key": area_key,
            },
        ).mappings().first()

    if row is None:
        return None

    payload = dict(row)
    payload["last_acq_time_utc"] = _as_utc(payload.get("last_acq_time_utc"))
    payload["updated_at"] = _as_utc(payload.get("updated_at"))
    return payload


def advance_ingest_watermark(
    *,
    source: str,
    area_key: str,
    last_acq_time_utc: datetime,
    last_batch_id: int,
    conn: Connection | None = None,
) -> None:
    """Advance the source+area ingestion watermark after successful batch completion."""
    stmt = text(
        """
        INSERT INTO ingest_watermarks (
            source,
            area_key,
            last_acq_time_utc,
            last_batch_id,
            updated_at
        )
        VALUES (
            :source,
            :area_key,
            :last_acq_time_utc,
            :last_batch_id,
            NOW()
        )
        ON CONFLICT (source, area_key)
        DO UPDATE SET
            last_acq_time_utc = GREATEST(
                COALESCE(ingest_watermarks.last_acq_time_utc, EXCLUDED.last_acq_time_utc),
                EXCLUDED.last_acq_time_utc
            ),
            last_batch_id = EXCLUDED.last_batch_id,
            updated_at = NOW()
        """
    )
    params = {
        "source": source,
        "area_key": area_key,
        "last_acq_time_utc": _as_utc(last_acq_time_utc),
        "last_batch_id": int(last_batch_id),
    }
    if conn is not None:
        conn.execute(stmt, params)
    else:
        with get_engine().begin() as new_conn:
            new_conn.execute(stmt, params)


def insert_detections(
    detections: Sequence[DetectionRecord],
    *,
    conn: Connection | None = None,
) -> int:
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
            confidence_score,
            brightness,
            bright_t31,
            frp,
            scan,
            track,
            raw_properties,
            ingest_batch_id,
            dedupe_hash,
            is_archive
        )
        VALUES (
            ST_SetSRID(ST_MakePoint(:lon, :lat), 4326),
            :lat,
            :lon,
            :acq_time,
            :sensor,
            :source,
            :confidence,
            :confidence_score,
            :brightness,
            :bright_t31,
            :frp,
            :scan,
            :track,
            :raw_properties,
            :ingest_batch_id,
            :dedupe_hash,
            :is_archive
        )
        ON CONFLICT (source, dedupe_hash) DO NOTHING
        """
    ).bindparams(bindparam("raw_properties", type_=JSON))
    parameters = [record.to_parameters() for record in detections]

    def _execute(active_conn: Connection) -> int:
        result = active_conn.execute(insert_stmt, parameters)
        return result.rowcount or 0

    if conn is not None:
        return _execute(conn)

    with get_engine().begin() as new_conn:
        return _execute(new_conn)


def count_unscored_likelihood_for_batch(
    batch_id: int,
    *,
    conn: Connection | None = None,
) -> int:
    """Count detections in a batch that still have NULL fire_likelihood."""
    stmt = text(
        """
        SELECT COUNT(*) AS n
        FROM fire_detections
        WHERE ingest_batch_id = :batch_id
          AND fire_likelihood IS NULL
        """
    )

    def _execute(active_conn: Connection) -> int:
        value = active_conn.execute(stmt, {"batch_id": int(batch_id)}).scalar_one()
        return int(value or 0)

    if conn is not None:
        return _execute(conn)

    with get_engine().begin() as new_conn:
        return _execute(new_conn)


def count_rows_with_null_columns_for_batch(
    batch_id: int,
    *,
    columns: Iterable[str],
    exclude_source_like: str | None = None,
    conn: Connection | None = None,
) -> int:
    """Count detections in a batch where any required column is NULL.

    `columns` must be a subset of `_ALLOWED_COMPLETENESS_COLUMNS`.
    """
    checked_columns = tuple(dict.fromkeys(str(col).strip() for col in columns if str(col).strip()))
    if not checked_columns:
        raise ValueError("columns must be non-empty")
    unsupported = [col for col in checked_columns if col not in _ALLOWED_COMPLETENESS_COLUMNS]
    if unsupported:
        raise ValueError(f"Unsupported completeness columns: {unsupported}")

    null_predicate = " OR ".join(f"{col} IS NULL" for col in checked_columns)
    source_predicate = ""
    params: dict[str, object] = {"batch_id": int(batch_id)}
    if exclude_source_like:
        source_predicate = "AND (source IS NULL OR source NOT LIKE :exclude_source_like)"
        params["exclude_source_like"] = str(exclude_source_like)

    stmt = text(
        f"""
        SELECT COUNT(*) AS n
        FROM fire_detections
        WHERE ingest_batch_id = :batch_id
          {source_predicate}
          AND ({null_predicate})
        """
    )

    def _execute(active_conn: Connection) -> int:
        value = active_conn.execute(stmt, params).scalar_one()
        return int(value or 0)

    if conn is not None:
        return _execute(conn)

    with get_engine().begin() as new_conn:
        return _execute(new_conn)


def count_detections_for_batch(
    batch_id: int,
    *,
    conn: Connection | None = None,
) -> int:
    """Count detections tied to a specific ingest batch."""
    stmt = text(
        """
        SELECT COUNT(*) AS n
        FROM fire_detections
        WHERE ingest_batch_id = :batch_id
        """
    )

    def _execute(active_conn: Connection) -> int:
        value = active_conn.execute(stmt, {"batch_id": int(batch_id)}).scalar_one()
        return int(value or 0)

    if conn is not None:
        return _execute(conn)

    with get_engine().begin() as new_conn:
        return _execute(new_conn)


def list_batches_with_unscored_likelihood(limit: int = 5) -> list[int]:
    """Return recent ingest batch IDs containing rows with NULL fire_likelihood."""
    stmt = text(
        """
        SELECT ingest_batch_id
        FROM fire_detections
        WHERE ingest_batch_id IS NOT NULL
          AND fire_likelihood IS NULL
        GROUP BY ingest_batch_id
        ORDER BY ingest_batch_id DESC
        LIMIT :limit
        """
    )
    with get_engine().begin() as conn:
        rows = conn.execute(stmt, {"limit": max(1, int(limit))}).scalars().all()
    return [int(row) for row in rows if row is not None]


def list_batches_with_undenoised_detections(limit: int = 5) -> list[int]:
    """Return recent ingest batch IDs containing rows with NULL denoiser_decision."""
    stmt = text(
        """
        SELECT ingest_batch_id
        FROM fire_detections
        WHERE ingest_batch_id IS NOT NULL
          AND denoiser_decision IS NULL
        GROUP BY ingest_batch_id
        ORDER BY ingest_batch_id DESC
        LIMIT :limit
        """
    )
    with get_engine().begin() as conn:
        rows = conn.execute(stmt, {"limit": max(1, int(limit))}).scalars().all()
    return [int(row) for row in rows if row is not None]


def list_all_watermarks() -> list[dict[str, Any]]:
    """Return all rows from ingest_watermarks for startup validation."""
    stmt = text(
        """
        SELECT source, area_key, last_acq_time_utc
        FROM ingest_watermarks
        ORDER BY source, area_key
        """
    )
    with get_engine().begin() as conn:
        rows = conn.execute(stmt).mappings().all()
    result = []
    for row in rows:
        payload = dict(row)
        payload["last_acq_time_utc"] = _as_utc(payload.get("last_acq_time_utc"))
        result.append(payload)
    return result


def expire_stale_running_batches(
    max_age_minutes: int = 120,
    *,
    conn: Connection | None = None,
) -> list[dict[str, Any]]:
    """Mark zombie 'running' batches older than *max_age_minutes* as 'failed'.

    Returns a list of dicts ``{"id": …, "source": …, "started_at": …}`` for
    every batch that was expired so callers can log a per-batch warning.
    """
    select_stmt = text(
        """
        SELECT id, source, started_at
        FROM ingest_batches
        WHERE status = 'running'
          AND started_at < NOW() - MAKE_INTERVAL(mins => :max_age)
        ORDER BY id
        """
    )
    update_stmt = text(
        """
        UPDATE ingest_batches
        SET status = 'failed', completed_at = NOW()
        WHERE status = 'running'
          AND started_at < NOW() - MAKE_INTERVAL(mins => :max_age)
        """
    )

    def _execute(active_conn: Connection) -> list[dict[str, Any]]:
        rows = active_conn.execute(select_stmt, {"max_age": int(max_age_minutes)}).mappings().all()
        expired = [dict(r) for r in rows]
        if expired:
            active_conn.execute(update_stmt, {"max_age": int(max_age_minutes)})
        return expired

    if conn is not None:
        return _execute(conn)

    with get_engine().begin() as new_conn:
        return _execute(new_conn)


def reset_ingest_watermark(*, source: str, area_key: str) -> None:
    """Reset a watermark's last_acq_time_utc to NULL (triggers bootstrap on next ingest)."""
    stmt = text(
        """
        UPDATE ingest_watermarks
        SET last_acq_time_utc = NULL, updated_at = NOW()
        WHERE source = :source AND area_key = :area_key
        """
    )
    with get_engine().begin() as conn:
        conn.execute(stmt, {"source": source, "area_key": area_key})


def delete_detections_for_batch(
    batch_id: int,
    *,
    conn: Connection | None = None,
) -> int:
    """Delete all detections tied to a specific ingest batch."""
    stmt = text(
        """
        DELETE FROM fire_detections
        WHERE ingest_batch_id = :batch_id
        """
    )

    def _execute(active_conn: Connection) -> int:
        result = active_conn.execute(stmt, {"batch_id": int(batch_id)})
        return int(result.rowcount or 0)

    if conn is not None:
        return _execute(conn)

    with get_engine().begin() as new_conn:
        return _execute(new_conn)


def count_consecutive_no_data_batches(
    source: str,
    area_key: str,
    threshold: int = 3,
) -> int:
    """Count consecutive no_data batches for a source+area combo (most recent first).

    Queries the ingest_batches table for the most recent batches with matching
    source and area_key (from metadata). Returns the count of consecutive batches
    with status='no_data', stopping when a non-no_data batch is encountered or
    the threshold is exceeded.

    Args:
        source: The ingest source (e.g., 'VIIRS_SNPP_NRT')
        area_key: The normalized area bounding box
        threshold: Return early if count exceeds this value

    Returns:
        The number of consecutive no_data batches (0 if the most recent is not no_data)
    """
    select_stmt = text(
        """
        SELECT status
        FROM ingest_batches
        WHERE source = :source
          AND metadata->>'area_key' = :area_key
        ORDER BY completed_at DESC
        LIMIT :threshold_limit
        """
    )

    with get_engine().begin() as conn:
        rows = conn.execute(
            select_stmt,
            {
                "source": source,
                "area_key": area_key,
                "threshold_limit": threshold + 1,
            },
        ).scalars().all()

    consecutive = 0
    for status in rows:
        if status == "no_data":
            consecutive += 1
        else:
            break

    return consecutive
