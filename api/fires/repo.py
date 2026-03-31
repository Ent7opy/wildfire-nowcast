"""DB queries for fire detections."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Iterable, Literal, TYPE_CHECKING

from sqlalchemy import text, column as sa_column
from sqlalchemy.sql.expression import TextClause

if TYPE_CHECKING:
    from sqlalchemy.engine import Connection

from api.config import settings
from api.pagination import encode_cursor, decode_cursor, build_page
from api.db import get_engine, get_async_engine
from api.fires.scoring import compute_fire_likelihood
from api.fires.scoring_pipeline import (
    _log_step,
    run_scoring_stage,
    FalseSourceMaskingStrategy,
    LandcoverScoringStrategy,
    PersistenceScoringStrategy,
    WeatherScoringStrategy,
)

BBox = tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)
LOGGER = logging.getLogger(__name__)

_SPATIAL_QUERY_TIMEOUT = text("SET LOCAL statement_timeout = '5000ms'")

# Stateless singletons — strategies hold no mutable state so one instance suffices.
_FALSE_SOURCE_STRATEGY = FalseSourceMaskingStrategy()
_PERSISTENCE_STRATEGY = PersistenceScoringStrategy()
_LANDCOVER_STRATEGY = LandcoverScoringStrategy()
_WEATHER_STRATEGY = WeatherScoringStrategy()


def validate_bbox(bbox: BBox) -> None:
    """Validate that a bbox has valid coordinate ordering.
    
    Args:
        bbox: (min_lon, min_lat, max_lon, max_lat)
        
    Raises:
        ValueError: If min >= max for either dimension.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    if min_lon >= max_lon:
        raise ValueError(f"min_lon ({min_lon}) must be less than max_lon ({max_lon})")
    if min_lat >= max_lat:
        raise ValueError(f"min_lat ({min_lat}) must be less than max_lat ({max_lat})")



# Keep this list tight to avoid SQL injection when constructing SELECT clauses.
_ALLOWED_COLUMNS: dict[str, str] = {
    "id": "id",
    "acq_time": "acq_time",
    "lat": "lat",
    "lon": "lon",
    "frp": "frp",
    "confidence": "confidence",
    "confidence_score": "confidence_score",
    "persistence_score": "persistence_score",
    "landcover_score": "landcover_score",
    "weather_score": "weather_score",
    "brightness": "brightness",
    "bright_t31": "bright_t31",
    "scan": "scan",
    "track": "track",
    "sensor": "sensor",
    "source": "source",
    "is_noise": "is_noise",
    "denoised_score": "denoised_score",
    "false_source_masked": "false_source_masked",
    "fire_likelihood": "fire_likelihood",
    "event_id": "event_id",
    "front_id": "front_id",
    "event_score": "event_score",
    "denoiser_decision": "denoiser_decision",
    "review_required": "review_required",
    "denoiser_model_id": "denoiser_model_id",
    "denoiser_scored_at": "denoiser_scored_at",
}


def _build_detections_query(
    bbox: BBox,
    start_time: datetime,
    end_time: datetime,
    *,
    columns: Iterable[str] = ("lat", "lon", "acq_time"),
    limit: int | None = None,
    order: Literal["asc", "desc"] = "asc",
    include_noise: bool = False,
    include_masked: bool = False,
    min_confidence: float | None = None,
    min_fire_likelihood: float | None = None,
    cursor: str | None = None,
    offset: int | None = None,
) -> tuple[TextClause, dict[str, object], int]:
    """Build the SQL statement and params for listing fire detections.

    Returns ``(stmt, params, page_limit)`` where *page_limit* is the
    validated limit value to pass to :func:`build_page`.
    """
    min_lon, min_lat, max_lon, max_lat = bbox

    cols = list(columns)
    if not cols:
        raise ValueError("columns must be non-empty.")

    # acq_time and id are required for keyset cursor construction; ensure they are
    # always selected regardless of the caller-supplied column list.
    for _required in ("id", "acq_time"):
        if _required not in cols:
            cols.append(_required)

    # Build SELECT clause using SQLAlchemy column objects for safety
    # This avoids SQL injection even if whitelist is bypassed in future
    select_parts: list[str] = []
    for c in cols:
        if c not in _ALLOWED_COLUMNS:
            raise ValueError(f"Unsupported column: {c}")
        # Use SQLAlchemy's column() to properly quote identifiers
        col_obj = sa_column(_ALLOWED_COLUMNS[c])
        select_parts.append(f"{col_obj} AS {sa_column(c)}")
    select_sql = ",\n            ".join(select_parts)

    if order not in ("asc", "desc"):
        raise ValueError("order must be 'asc' or 'desc'.")

    # Noise filter: default to excluding detections explicitly marked as noise.
    # We use "IS NOT TRUE" to include NULLs (detections not yet scored).
    noise_predicate = ""
    if not include_noise:
        noise_predicate = "AND is_noise IS NOT TRUE"

    # Masked filter: default to excluding detections near industrial sources.
    # We use "IS NOT TRUE" to include NULLs (detections not yet checked).
    masked_predicate = ""
    if not include_masked:
        masked_predicate = "AND false_source_masked IS NOT TRUE"

    confidence_predicate = ""
    if min_confidence is not None:
        # Include NULL confidence values when filtering (NULL means unknown, not 0)
        confidence_predicate = "AND (confidence IS NULL OR confidence >= :min_confidence)"

    likelihood_predicate = ""
    if min_fire_likelihood is not None:
        # Include NULL likelihood values when filtering (NULL means not yet scored)
        likelihood_predicate = "AND (fire_likelihood IS NULL OR fire_likelihood >= :min_fire_likelihood)"

    # Apply default limit for pagination if not specified
    # This prevents unbounded queries that could cause memory issues
    if limit is None:
        limit = 1000
    if limit <= 0 or limit > 10000:
        raise ValueError("limit must be between 1 and 10000.")

    # Ensure datetimes are timezone-aware (UTC) for database queries
    if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=timezone.utc)
    elif start_time.tzinfo != timezone.utc:
        start_time = start_time.astimezone(timezone.utc)

    if end_time.tzinfo is None:
        end_time = end_time.replace(tzinfo=timezone.utc)
    elif end_time.tzinfo != timezone.utc:
        end_time = end_time.astimezone(timezone.utc)

    # Cursor-based pagination: keyset on (acq_time, id) for stable, index-friendly pagination.
    # Cursor encodes {"t": acq_time_iso, "id": int_id}.
    cursor_acq_time: datetime | None = None
    cursor_id: int | None = None
    cursor_predicate = ""
    if cursor is not None:
        parsed = decode_cursor(cursor)
        cursor_acq_time = parsed["t"]
        cursor_id = int(parsed["id"])
        if order == "asc":
            cursor_predicate = (
                "AND (acq_time > :cursor_acq_time "
                "OR (acq_time = :cursor_acq_time AND id > :cursor_id))"
            )
        else:
            cursor_predicate = (
                "AND (acq_time < :cursor_acq_time "
                "OR (acq_time = :cursor_acq_time AND id < :cursor_id))"
            )

    # Deprecated offset path: slow on large tables, kept for backward compatibility.
    offset_sql = ""
    if cursor is None and offset is not None:
        if offset < 0:
            raise ValueError("offset must be >= 0.")
        offset_sql = "\n        OFFSET :offset"

    params: dict[str, object] = {
        "start_time": start_time,
        "end_time": end_time,
        "min_lon": float(min_lon),
        "min_lat": float(min_lat),
        "max_lon": float(max_lon),
        "max_lat": float(max_lat),
        "limit": int(limit) + 1,  # Fetch one extra to determine if there's a next page
    }
    if min_confidence is not None:
        params["min_confidence"] = float(min_confidence)
    if min_fire_likelihood is not None:
        params["min_fire_likelihood"] = float(min_fire_likelihood)
    if cursor_acq_time is not None and cursor_id is not None:
        params["cursor_acq_time"] = cursor_acq_time
        params["cursor_id"] = cursor_id
    if cursor is None and offset is not None:
        params["offset"] = int(offset)

    stmt = text(
        f"""
        SELECT
            {select_sql}
        FROM fire_detections
        WHERE acq_time BETWEEN :start_time AND :end_time
          AND geom && ST_MakeEnvelope(:min_lon, :min_lat, :max_lon, :max_lat, 4326)
          AND ST_Intersects(geom, ST_MakeEnvelope(:min_lon, :min_lat, :max_lon, :max_lat, 4326))
          {noise_predicate}
          {masked_predicate}
          {confidence_predicate}
          {likelihood_predicate}
          {cursor_predicate}
        ORDER BY acq_time {order}, id {order}
        LIMIT :limit{offset_sql}
        """
    )

    return stmt, params, limit


def list_fire_detections_bbox_time(
    bbox: BBox,
    start_time: datetime,
    end_time: datetime,
    *,
    columns: Iterable[str] = ("lat", "lon", "acq_time"),
    limit: int | None = None,
    order: Literal["asc", "desc"] = "asc",
    include_noise: bool = False,
    include_masked: bool = False,
    min_confidence: float | None = None,
    min_fire_likelihood: float | None = None,
    cursor: str | None = None,
    offset: int | None = None,
) -> dict:
    """List fire detections in a lon/lat bbox and acquisition time window.

    Supports cursor-based pagination for efficient retrieval of large result sets.

    Notes
    - Time filter uses `BETWEEN` (inclusive bounds).
    - Spatial filter uses GiST index-friendly predicates:
      `geom && envelope` plus `ST_Intersects(geom, envelope)`.
    - Denoiser: By default, filters out rows where `is_noise` is TRUE.
    - False-source masking: By default, filters out rows where `false_source_masked` is TRUE.
    - Filtering: min_confidence filters FIRMS confidence (0-100), min_fire_likelihood filters
      composite likelihood score (0-1). Both include NULL values (not yet scored).
    - Pagination: Pass next_cursor from a previous response as cursor to get the next page.
      Returns at most `limit` rows per request (default: 1000, max: 10000).
    """
    stmt, params, page_limit = _build_detections_query(
        bbox, start_time, end_time,
        columns=columns, limit=limit, order=order,
        include_noise=include_noise, include_masked=include_masked,
        min_confidence=min_confidence, min_fire_likelihood=min_fire_likelihood,
        cursor=cursor, offset=offset,
    )

    with get_engine().begin() as conn:
        conn.execute(_SPATIAL_QUERY_TIMEOUT)
        result = conn.execute(stmt, params)
        rows = result.mappings().all()

    return build_page(
        rows, page_limit,
        cursor_fn=lambda r: encode_cursor(t=r["acq_time"], id=r["id"]),
    )


async def async_list_fire_detections_bbox_time(
    bbox: BBox,
    start_time: datetime,
    end_time: datetime,
    *,
    columns: Iterable[str] = ("lat", "lon", "acq_time"),
    limit: int | None = None,
    order: Literal["asc", "desc"] = "asc",
    include_noise: bool = False,
    include_masked: bool = False,
    min_confidence: float | None = None,
    min_fire_likelihood: float | None = None,
    cursor: str | None = None,
    offset: int | None = None,
) -> dict:
    """Async variant of :func:`list_fire_detections_bbox_time`."""
    stmt, params, page_limit = _build_detections_query(
        bbox, start_time, end_time,
        columns=columns, limit=limit, order=order,
        include_noise=include_noise, include_masked=include_masked,
        min_confidence=min_confidence, min_fire_likelihood=min_fire_likelihood,
        cursor=cursor, offset=offset,
    )

    async with get_async_engine().begin() as conn:
        await conn.execute(_SPATIAL_QUERY_TIMEOUT)
        result = await conn.execute(stmt, params)
        rows = result.mappings().all()

    return build_page(
        rows, page_limit,
        cursor_fn=lambda r: encode_cursor(t=r["acq_time"], id=r["id"]),
    )


_DETECTION_BY_ID_SQL = text("""
    SELECT
        id, lat, lon, acq_time,
        confidence, brightness, bright_t31, frp,
        sensor, source,
        confidence_score, persistence_score, landcover_score, weather_score,
        false_source_masked, fire_likelihood,
        denoised_score, is_noise, event_id, event_score,
        denoiser_decision, review_required
    FROM fire_detections
    WHERE id = :detection_id
""")


def get_fire_detection_by_id(detection_id: int) -> dict | None:
    """Fetch a single fire detection by primary key.

    Returns a dict of detection attributes or ``None`` if not found.
    """
    with get_engine().connect() as conn:
        row = conn.execute(_DETECTION_BY_ID_SQL, {"detection_id": detection_id}).mappings().first()

    if not row:
        return None
    return dict(row)


async def async_get_fire_detection_by_id(detection_id: int) -> dict | None:
    """Async variant of :func:`get_fire_detection_by_id`."""
    async with get_async_engine().connect() as conn:
        row = (
            await conn.execute(_DETECTION_BY_ID_SQL, {"detection_id": detection_id})
        ).mappings().first()

    if not row:
        return None
    return dict(row)


def update_false_source_masking(batch_id: int, conn: Connection | None = None) -> int:
    """Update false_source_masked column for detections in a batch.

    Returns the number of detections marked as masked.
    """
    return run_scoring_stage(batch_id, _FALSE_SOURCE_STRATEGY, conn)


def update_persistence_scores(batch_id: int, conn: Connection | None = None) -> int:
    """Update persistence_score column for detections in a batch.

    Returns the number of detections with scores updated.
    """
    return run_scoring_stage(batch_id, _PERSISTENCE_STRATEGY, conn)


def update_landcover_scores(batch_id: int, conn: Connection | None = None) -> int:
    """Update landcover_score column for detections in a batch.

    Returns the number of detections with scores updated.
    """
    return run_scoring_stage(batch_id, _LANDCOVER_STRATEGY, conn)


def update_weather_scores(batch_id: int, conn: Connection | None = None) -> int:
    """Update weather_score column for detections in a batch.

    Returns the number of detections with scores updated.
    """
    return run_scoring_stage(batch_id, _WEATHER_STRATEGY, conn)


def update_fire_likelihood(batch_id: int, conn: Connection | None = None) -> int:
    """Update fire_likelihood column for detections in a batch.

    Queries detections from the batch with all component scores, uses compute_fire_likelihood()
    to compute composite fire likelihood, and updates the fire_likelihood column.

    Args:
        batch_id: The ingest batch ID to process
        conn: Optional existing database connection to use. If provided, this
            connection will be used for the operation and no new connection
            will be opened. This is useful for batching multiple scoring updates
            within a single transaction to avoid connection pool exhaustion.

    Returns:
        Number of detections with likelihood updated
    """
    def _execute(conn: Connection) -> int:
        # Query detections with all component scores
        stmt = text("""
            SELECT 
                id, 
                confidence_score, 
                persistence_score, 
                landcover_score, 
                weather_score, 
                false_source_masked
            FROM fire_detections
            WHERE ingest_batch_id = :batch_id
        """)
        started = time.perf_counter()
        result = conn.execute(stmt, {"batch_id": batch_id})
        rows = result.mappings().all()
        _log_step("likelihood.fetch_batch", started, batch_id=batch_id, rows=int(len(rows)))

        if not rows:
            return 0

        # Compute fire likelihood for each detection
        # NULL handling is centralized in compute_fire_likelihood() - pass None values directly
        started = time.perf_counter()
        params = []
        for row in rows:
            likelihood = compute_fire_likelihood(
                confidence_score=float(row["confidence_score"]) if row["confidence_score"] is not None else 0.5,
                persistence_score=float(row["persistence_score"]) if row["persistence_score"] is not None else None,
                landcover_score=float(row["landcover_score"]) if row["landcover_score"] is not None else None,
                weather_score=float(row["weather_score"]) if row["weather_score"] is not None else None,
                false_source_masked=bool(row["false_source_masked"]) if row["false_source_masked"] is not None else False,
            )
            params.append({"detection_id": row["id"], "likelihood": likelihood})
        _log_step("likelihood.compute_scores", started, batch_id=batch_id, rows=int(len(params)))

        # Update fire_detections table with fire likelihood
        update_stmt = text("""
            UPDATE fire_detections
            SET fire_likelihood = :likelihood
            WHERE id = :detection_id
        """)

        started = time.perf_counter()
        conn.execute(update_stmt, params)
        _log_step("likelihood.update_batch", started, batch_id=batch_id, rows=int(len(params)))

        return len(params)

    if conn is not None:
        return _execute(conn)
    else:
        with get_engine().begin() as new_conn:
            return _execute(new_conn)



def update_all_scoring_for_batch(
    batch_id: int,
    conn: Connection | None = None,
) -> dict[str, int]:
    """Update all scoring columns for a batch within a single transaction.
    
    This function wraps all scoring updates (false source masking, persistence,
    landcover, weather, and fire likelihood) in a single database transaction.
    This ensures atomicity and prevents connection pool exhaustion during batch
    processing.
    
    Args:
        batch_id: The ingest batch ID to process
        
    Returns:
        Dictionary with counts for each scoring type:
        {
            "masked_count": int,
            "persistence_count": int,
            "landcover_count": int,
            "weather_count": int,
            "likelihood_count": int,
        }
        
    Raises:
        Exception: If any scoring update fails, the entire transaction is rolled back
    """
    def _execute(active_conn: Connection) -> dict[str, int]:
        started_total = time.perf_counter()
        masked_count = update_false_source_masking(batch_id, conn=active_conn)
        persistence_count = update_persistence_scores(batch_id, conn=active_conn)
        landcover_count = update_landcover_scores(batch_id, conn=active_conn)
        weather_count = update_weather_scores(batch_id, conn=active_conn)
        likelihood_count = update_fire_likelihood(batch_id, conn=active_conn)
        _log_step(
            "scoring.update_all",
            started_total,
            batch_id=batch_id,
            rows=int(likelihood_count),
        )

        return {
            "masked_count": masked_count,
            "persistence_count": persistence_count,
            "landcover_count": landcover_count,
            "weather_count": weather_count,
            "likelihood_count": likelihood_count,
        }

    if conn is not None:
        return _execute(conn)

    with get_engine().begin() as new_conn:
        return _execute(new_conn)


def _build_events_query(
    bbox: BBox,
    start_time: datetime,
    end_time: datetime,
    *,
    min_event_score: float | None = None,
    include_review_required: bool = True,
    limit: int = 1000,
    cursor: str | None = None,
    offset: int | None = None,
) -> tuple[TextClause, dict[str, object], int]:
    """Build the SQL statement and params for listing fire events.

    Returns ``(stmt, params, page_limit)``.
    """
    min_lon, min_lat, max_lon, max_lat = bbox

    if limit <= 0 or limit > 10000:
        raise ValueError("limit must be between 1 and 10000.")
    if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=timezone.utc)
    else:
        start_time = start_time.astimezone(timezone.utc)
    if end_time.tzinfo is None:
        end_time = end_time.replace(tzinfo=timezone.utc)
    else:
        end_time = end_time.astimezone(timezone.utc)

    review_predicate = ""
    if not include_review_required:
        review_predicate = "AND review_required IS NOT TRUE"

    score_predicate = ""
    # Cursor keyset pagination on (COALESCE(start_time, end_time) DESC, event_id DESC).
    # Cursor encodes {"t": iso_datetime_or_null, "id": event_id_string}.
    cursor_predicate = ""
    if cursor is not None:
        parsed = decode_cursor(cursor)
        cursor_event_time: datetime | None = parsed.get("t")
        cursor_event_id: str = str(parsed["id"])
        if cursor_event_time is not None:
            cursor_predicate = (
                "AND (COALESCE(start_time, end_time) < :cursor_time "
                "OR COALESCE(start_time, end_time) IS NULL "
                "OR (COALESCE(start_time, end_time) = :cursor_time AND event_id < :cursor_id))"
            )
        else:
            # Already in the NULL bucket; only advance by event_id.
            cursor_predicate = (
                "AND COALESCE(start_time, end_time) IS NULL "
                "AND event_id < :cursor_id"
            )

    # Deprecated offset path.
    offset_sql = ""
    if cursor is None and offset is not None:
        if offset < 0:
            raise ValueError("offset must be >= 0.")
        offset_sql = "\n            OFFSET :offset"

    params: dict[str, object] = {
        "start_time": start_time,
        "end_time": end_time,
        "min_lon": float(min_lon),
        "min_lat": float(min_lat),
        "max_lon": float(max_lon),
        "max_lat": float(max_lat),
        "limit": int(limit) + 1,  # fetch one extra to detect has_more
    }
    if min_event_score is not None:
        score_predicate = "AND (event_score IS NULL OR event_score >= :min_event_score)"
        params["min_event_score"] = float(min_event_score)
    if cursor is not None:
        if cursor_event_time is not None:
            params["cursor_time"] = cursor_event_time
        params["cursor_id"] = cursor_event_id
    if cursor is None and offset is not None:
        params["offset"] = int(offset)

    stmt = text(
        f"""
        WITH selected_events AS (
            SELECT
                event_id,
                source,
                sensor,
                start_time,
                end_time,
                detection_count,
                front_count,
                event_score,
                denoiser_decision,
                review_required,
                geom_source,
                geom_method,
                geom_quality,
                authority_profile,
                authoritative_perimeter_id,
                geom
            FROM fire_events
            WHERE start_time <= :end_time
              AND end_time >= :start_time
              AND geom && ST_MakeEnvelope(:min_lon, :min_lat, :max_lon, :max_lat, 4326)
              AND ST_Intersects(geom, ST_MakeEnvelope(:min_lon, :min_lat, :max_lon, :max_lat, 4326))
              {review_predicate}
              {score_predicate}
              {cursor_predicate}
            ORDER BY COALESCE(start_time, end_time) DESC, event_id DESC
            LIMIT :limit{offset_sql}
        )
        SELECT
            e.event_id,
            e.source,
            e.sensor,
            e.start_time,
            e.end_time,
            e.detection_count,
            e.front_count,
            e.event_score,
            e.denoiser_decision,
            e.review_required,
            e.geom_source,
            e.geom_method,
            e.geom_quality,
            e.authority_profile,
            e.authoritative_perimeter_id,
            intensity.frp_max,
            intensity.frp_mean,
            intensity.brightness_max,
            intensity.brightness_mean,
            ST_AsGeoJSON(e.geom) AS geom_geojson,
            ST_X(ST_Centroid(e.geom)) AS lon,
            ST_Y(ST_Centroid(e.geom)) AS lat,
            rgc.location_name,
            rgc.country_name AS country,
            rgc.admin1_name
        FROM selected_events e
        LEFT JOIN LATERAL (
            SELECT
                MAX(fd.frp) AS frp_max,
                AVG(fd.frp) FILTER (WHERE fd.frp IS NOT NULL) AS frp_mean,
                MAX(fd.brightness) AS brightness_max,
                AVG(fd.brightness) FILTER (WHERE fd.brightness IS NOT NULL) AS brightness_mean
            FROM fire_detections fd
            WHERE fd.event_id = e.event_id
        ) intensity ON TRUE
        LEFT JOIN reverse_geocode_cache rgc ON
            rgc.provider = :geocoding_provider
            AND rgc.cached_lat = ROUND(CAST(ST_Y(ST_Centroid(e.geom)) AS NUMERIC), :geocoding_precision)
            AND rgc.cached_lon = ROUND(CAST(ST_X(ST_Centroid(e.geom)) AS NUMERIC), :geocoding_precision)
            AND rgc.expires_at > NOW()
        ORDER BY COALESCE(e.start_time, e.end_time) DESC, e.event_id DESC
        """
    )

    params["geocoding_provider"] = settings.geocoding_provider.strip().lower()
    params["geocoding_precision"] = settings.geocoding_cache_precision

    return stmt, params, limit


def list_fire_events_bbox_time(
    bbox: BBox,
    start_time: datetime,
    end_time: datetime,
    *,
    min_event_score: float | None = None,
    include_review_required: bool = True,
    limit: int = 1000,
    cursor: str | None = None,
    offset: int | None = None,
) -> dict:
    """List denoiser events in a bbox/time window.

    Returns a dict with keys ``data``, ``next_cursor``, ``has_more``, and ``limit``.
    Order is COALESCE(start_time, end_time) DESC, event_id DESC.
    """
    stmt, params, page_limit = _build_events_query(
        bbox, start_time, end_time,
        min_event_score=min_event_score,
        include_review_required=include_review_required,
        limit=limit, cursor=cursor, offset=offset,
    )

    with get_engine().begin() as conn:
        conn.execute(_SPATIAL_QUERY_TIMEOUT)
        rows = conn.execute(stmt, params).mappings().all()

    return build_page(
        rows, page_limit,
        cursor_fn=lambda r: encode_cursor(
            t=r.get("start_time") or r.get("end_time"), id=r["event_id"]
        ),
    )


async def async_list_fire_events_bbox_time(
    bbox: BBox,
    start_time: datetime,
    end_time: datetime,
    *,
    min_event_score: float | None = None,
    include_review_required: bool = True,
    limit: int = 1000,
    cursor: str | None = None,
    offset: int | None = None,
) -> dict:
    """Async variant of :func:`list_fire_events_bbox_time`."""
    stmt, params, page_limit = _build_events_query(
        bbox, start_time, end_time,
        min_event_score=min_event_score,
        include_review_required=include_review_required,
        limit=limit, cursor=cursor, offset=offset,
    )

    async with get_async_engine().begin() as conn:
        await conn.execute(_SPATIAL_QUERY_TIMEOUT)
        rows = (await conn.execute(stmt, params)).mappings().all()

    return build_page(
        rows, page_limit,
        cursor_fn=lambda r: encode_cursor(
            t=r.get("start_time") or r.get("end_time"), id=r["event_id"]
        ),
    )


def _build_fronts_query(
    bbox: BBox,
    start_time: datetime,
    end_time: datetime,
    *,
    min_event_score: float | None = None,
    include_review_required: bool = True,
    limit: int = 2000,
    cursor: str | None = None,
    offset: int | None = None,
) -> tuple[TextClause, dict[str, object], int]:
    """Build the SQL statement and params for listing fire fronts.

    Returns ``(stmt, params, effective_limit)``.
    """
    min_lon, min_lat, max_lon, max_lat = bbox

    if limit <= 0 or limit > 10000:
        raise ValueError("limit must be between 1 and 10000.")
    # Keep this endpoint bounded to protect API responsiveness under global windows.
    effective_limit = min(int(limit), 800)
    if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=timezone.utc)
    else:
        start_time = start_time.astimezone(timezone.utc)
    if end_time.tzinfo is None:
        end_time = end_time.replace(tzinfo=timezone.utc)
    else:
        end_time = end_time.astimezone(timezone.utc)

    review_predicate = ""
    if not include_review_required:
        review_predicate = "AND fe.review_required IS NOT TRUE"

    # Cursor keyset pagination on (COALESCE(overpass_end, overpass_start) DESC NULLS LAST, front_id DESC).
    # Cursor encodes {"t": iso_datetime_or_null, "id": front_id_string}.
    cursor_predicate = ""
    if cursor is not None:
        parsed = decode_cursor(cursor)
        cursor_front_time: datetime | None = parsed.get("t")
        cursor_front_id: str = str(parsed["id"])
        if cursor_front_time is not None:
            cursor_predicate = (
                "AND (COALESCE(overpass_end, overpass_start) < :cursor_time "
                "OR COALESCE(overpass_end, overpass_start) IS NULL "
                "OR (COALESCE(overpass_end, overpass_start) = :cursor_time "
                "AND front_id < :cursor_id))"
            )
        else:
            cursor_predicate = (
                "AND COALESCE(overpass_end, overpass_start) IS NULL "
                "AND front_id < :cursor_id"
            )

    # Deprecated offset path.
    offset_sql = ""
    if cursor is None and offset is not None:
        if offset < 0:
            raise ValueError("offset must be >= 0.")
        offset_sql = "\n            OFFSET :offset"

    score_predicate = ""
    params: dict[str, object] = {
        "start_time": start_time,
        "end_time": end_time,
        "min_lon": float(min_lon),
        "min_lat": float(min_lat),
        "max_lon": float(max_lon),
        "max_lat": float(max_lat),
        "limit": effective_limit + 1,  # fetch one extra to detect has_more
    }
    if min_event_score is not None:
        score_predicate = "AND (fe.event_score IS NULL OR fe.event_score >= :min_event_score)"
        params["min_event_score"] = float(min_event_score)
    if cursor is not None:
        if cursor_front_time is not None:
            params["cursor_time"] = cursor_front_time
        params["cursor_id"] = cursor_front_id
    if cursor is None and offset is not None:
        params["offset"] = int(offset)

    stmt = text(
        f"""
        WITH candidate_events AS (
            SELECT
                fe.event_id,
                fe.event_score,
                fe.denoiser_decision,
                fe.review_required,
                fe.start_time,
                fe.end_time
            FROM fire_events fe
            WHERE fe.start_time <= :end_time
              AND fe.end_time >= :start_time
              {review_predicate}
              {score_predicate}
        ),
        linked_fronts AS (
            SELECT DISTINCT
                fem.front_id,
                ce.event_id,
                ce.event_score,
                ce.denoiser_decision,
                ce.review_required,
                ce.start_time,
                ce.end_time
            FROM fire_event_memberships fem
            JOIN candidate_events ce ON ce.event_id = fem.event_id
            WHERE fem.front_id IS NOT NULL
        ),
        ranked_fronts AS (
            SELECT
                ff.front_id,
                ff.source,
                ff.sensor,
                ff.overpass_start,
                ff.overpass_end,
                ff.detection_count,
                ff.frp_max,
                ff.frp_mean,
                ff.confidence_max,
                ff.geom_source,
                ff.geom_method,
                ff.geom_quality,
                ff.authority_profile,
                ff.authoritative_perimeter_id,
                lf.event_id,
                lf.event_score,
                lf.denoiser_decision,
                lf.review_required,
                ff.geom,
                ROW_NUMBER() OVER (
                    PARTITION BY ff.front_id
                    ORDER BY COALESCE(lf.end_time, lf.start_time) DESC NULLS LAST, lf.event_id DESC
                ) AS front_rank
            FROM linked_fronts lf
            JOIN fire_fronts ff ON ff.front_id = lf.front_id
            WHERE ff.geom IS NOT NULL
              AND ff.geom && ST_MakeEnvelope(:min_lon, :min_lat, :max_lon, :max_lat, 4326)
              AND ST_Intersects(ff.geom, ST_MakeEnvelope(:min_lon, :min_lat, :max_lon, :max_lat, 4326))
        ),
        selected_fronts AS (
            SELECT
                front_id,
                source,
                sensor,
                overpass_start,
                overpass_end,
                detection_count,
                frp_max,
                frp_mean,
                confidence_max,
                geom_source,
                geom_method,
                geom_quality,
                authority_profile,
                authoritative_perimeter_id,
                event_id,
                event_score,
                denoiser_decision,
                review_required,
                geom
            FROM ranked_fronts
            WHERE front_rank = 1
              {cursor_predicate}
            ORDER BY COALESCE(overpass_end, overpass_start) DESC NULLS LAST, front_id DESC
            LIMIT :limit{offset_sql}
        )
        SELECT
            sf.front_id,
            sf.source,
            sf.sensor,
            sf.overpass_start,
            sf.overpass_end,
            sf.detection_count,
            sf.frp_max,
            sf.frp_mean,
            sf.confidence_max,
            sf.geom_source,
            sf.geom_method,
            sf.geom_quality,
            sf.authority_profile,
            sf.authoritative_perimeter_id,
            sf.event_id,
            sf.event_score,
            sf.denoiser_decision,
            sf.review_required,
            ST_X(ST_Centroid(sf.geom)) AS lon,
            ST_Y(ST_Centroid(sf.geom)) AS lat,
            ST_AsGeoJSON(sf.geom) AS geom_geojson
        FROM selected_fronts sf
        """
    )

    return stmt, params, effective_limit


def list_fire_fronts_bbox_time(
    bbox: BBox,
    start_time: datetime,
    end_time: datetime,
    *,
    min_event_score: float | None = None,
    include_review_required: bool = True,
    limit: int = 2000,
    cursor: str | None = None,
    offset: int | None = None,
) -> dict:
    """List denoiser fire fronts in a bbox/time window.

    Returns a dict with keys ``data``, ``next_cursor``, ``has_more``, and ``limit``.
    Order is COALESCE(overpass_end, overpass_start) DESC NULLS LAST, front_id DESC.
    """
    stmt, params, page_limit = _build_fronts_query(
        bbox, start_time, end_time,
        min_event_score=min_event_score,
        include_review_required=include_review_required,
        limit=limit, cursor=cursor, offset=offset,
    )

    with get_engine().begin() as conn:
        # Keep request latency bounded so map interactions do not starve API health checks.
        conn.execute(_SPATIAL_QUERY_TIMEOUT)
        rows = conn.execute(stmt, params).mappings().all()

    return build_page(
        rows, page_limit,
        cursor_fn=lambda r: encode_cursor(
            t=r.get("overpass_end") or r.get("overpass_start"), id=r["front_id"]
        ),
    )


async def async_list_fire_fronts_bbox_time(
    bbox: BBox,
    start_time: datetime,
    end_time: datetime,
    *,
    min_event_score: float | None = None,
    include_review_required: bool = True,
    limit: int = 2000,
    cursor: str | None = None,
    offset: int | None = None,
) -> dict:
    """Async variant of :func:`list_fire_fronts_bbox_time`."""
    stmt, params, page_limit = _build_fronts_query(
        bbox, start_time, end_time,
        min_event_score=min_event_score,
        include_review_required=include_review_required,
        limit=limit, cursor=cursor, offset=offset,
    )

    async with get_async_engine().begin() as conn:
        await conn.execute(_SPATIAL_QUERY_TIMEOUT)
        rows = (await conn.execute(stmt, params)).mappings().all()

    return build_page(
        rows, page_limit,
        cursor_fn=lambda r: encode_cursor(
            t=r.get("overpass_end") or r.get("overpass_start"), id=r["front_id"]
        ),
    )


def get_fire_front_by_id(
    front_id: str,
    *,
    buffer_km: float = 0.0,
) -> dict | None:
    """Get a single fire front and an optional buffered bbox envelope."""
    if not front_id:
        raise ValueError("front_id must be non-empty.")
    if buffer_km < 0:
        raise ValueError("buffer_km must be >= 0.")

    stmt = text(
        """
        WITH selected AS (
            SELECT
                ff.front_id,
                ff.source,
                ff.sensor,
                ff.overpass_start,
                ff.overpass_end,
                ff.detection_count,
                ff.frp_max,
                ff.frp_mean,
                ff.confidence_max,
                ff.geom
            FROM fire_fronts ff
            WHERE ff.front_id = :front_id
            LIMIT 1
        ),
        expanded AS (
            SELECT
                s.*,
                ST_Envelope(
                    CASE
                        WHEN :buffer_m > 0
                            THEN ST_Buffer(s.geom::geography, :buffer_m)::geometry
                        ELSE s.geom
                    END
                ) AS geom_envelope
            FROM selected s
        )
        SELECT
            e.front_id,
            e.source,
            e.sensor,
            e.overpass_start,
            e.overpass_end,
            e.detection_count,
            e.frp_max,
            e.frp_mean,
            e.confidence_max,
            ST_AsGeoJSON(e.geom) AS geom_geojson,
            ST_XMin(e.geom_envelope) AS bbox_min_lon,
            ST_YMin(e.geom_envelope) AS bbox_min_lat,
            ST_XMax(e.geom_envelope) AS bbox_max_lon,
            ST_YMax(e.geom_envelope) AS bbox_max_lat
        FROM expanded e
        """
    )

    with get_engine().begin() as conn:
        row = conn.execute(
            stmt,
            {
                "front_id": str(front_id),
                "buffer_m": float(buffer_km) * 1000.0,
            },
        ).mappings().first()
    return dict(row) if row else None


def get_latest_denoiser_gate_report() -> dict | None:
    """Return latest gate report written by v2 evaluation."""
    stmt = text(
        """
        SELECT run_id, model_id, status, gate_report_json, evaluated_at
        FROM denoiser_eval_runs
        WHERE gate_report_json IS NOT NULL
        ORDER BY evaluated_at DESC
        LIMIT 1
        """
    )
    with get_engine().begin() as conn:
        row = conn.execute(stmt).mappings().first()
    return dict(row) if row else None


def get_latest_denoiser_coverage_status(authority_profile: str = "wfigs_us") -> dict | None:
    """Return latest authoritative coverage ingest status + active mask summary."""
    run_stmt = text(
        """
        WITH run_candidates AS (
            SELECT
                air.run_id,
                air.source_profile,
                air.source_uri,
                air.source_layer,
                air.status,
                air.started_at,
                air.finished_at,
                air.source_last_edit,
                air.records_fetched,
                air.records_upserted,
                air.records_skipped,
                air.http_429_count,
                air.max_backoff_seconds
            FROM authoritative_perimeter_ingest_runs air
            JOIN perimeter_coverage_masks pcm
              ON pcm.run_id = air.run_id
            WHERE pcm.authority_profile = :authority_profile
              AND pcm.is_active
              AND air.status = 'succeeded'
            UNION
            SELECT
                run_id,
                source_profile,
                source_uri,
                source_layer,
                status,
                started_at,
                finished_at,
                source_last_edit,
                records_fetched,
                records_upserted,
                records_skipped,
                http_429_count,
                max_backoff_seconds
            FROM authoritative_perimeter_ingest_runs
            WHERE source_profile = :authority_profile
              AND status = 'succeeded'
        )
        SELECT
            run_id,
            source_profile,
            source_uri,
            source_layer,
            status,
            started_at,
            finished_at,
            source_last_edit,
            records_fetched,
            records_upserted,
            records_skipped,
            http_429_count,
            max_backoff_seconds
        FROM run_candidates
        ORDER BY finished_at DESC NULLS LAST, started_at DESC
        LIMIT 1
        """
    )
    mask_stmt = text(
        """
        SELECT
            COUNT(*) AS active_mask_count,
            (ARRAY_AGG(mask_id ORDER BY mask_id))[1:20] AS sample_mask_ids,
            MIN(valid_from) AS min_valid_from,
            MAX(valid_to) AS max_valid_to
        FROM perimeter_coverage_masks
        WHERE is_active
          AND authority_profile = :authority_profile
        """
    )
    with get_engine().begin() as conn:
        run_row = conn.execute(run_stmt, {"authority_profile": authority_profile}).mappings().first()
        if run_row is None:
            return None
        mask_row = conn.execute(mask_stmt, {"authority_profile": authority_profile}).mappings().first()

    payload = dict(run_row)
    if mask_row is not None:
        payload["active_mask_count"] = int(mask_row["active_mask_count"] or 0)
        payload["sample_mask_ids"] = list(mask_row["sample_mask_ids"] or [])
        payload["min_valid_from"] = mask_row["min_valid_from"]
        payload["max_valid_to"] = mask_row["max_valid_to"]
    else:
        payload["active_mask_count"] = 0
        payload["sample_mask_ids"] = []
        payload["min_valid_from"] = None
        payload["max_valid_to"] = None
    return payload


def get_latest_denoiser_industrial_coverage_status(
    source_profile: str | None = None,
    policy_version: str | None = None,
) -> dict | None:
    """Return latest authoritative industrial ingest + policy/no-go coverage summary."""
    run_filter = ""
    params: dict[str, object] = {}
    if source_profile:
        run_filter = "AND source_profile = :source_profile"
        params["source_profile"] = source_profile

    run_stmt = text(
        f"""
        SELECT
            run_id,
            source_profile,
            status,
            started_at,
            finished_at,
            records_fetched,
            records_upserted,
            records_skipped,
            source_uri,
            source_version,
            metrics_json
        FROM authoritative_industrial_ingest_runs
        WHERE status = 'succeeded'
          {run_filter}
        ORDER BY finished_at DESC NULLS LAST, started_at DESC
        LIMIT 1
        """
    )

    policy_stmt = text(
        """
        SELECT
            policy_version,
            strict_no_go,
            gold_buffer_m,
            silver_buffer_min_m,
            silver_buffer_max_m,
            active_from,
            active_to
        FROM industrial_mask_policies
        WHERE (
                :policy_version IS NOT NULL
                AND policy_version = :policy_version
              )
           OR (
                :policy_version IS NULL
                AND (active_to IS NULL OR active_to > NOW())
              )
        ORDER BY active_from DESC, policy_version DESC
        LIMIT 1
        """
    )

    source_stats_stmt = text(
        """
        SELECT
            COUNT(*) FILTER (WHERE COALESCE(is_active, TRUE)) AS active_sources,
            COUNT(*) FILTER (WHERE COALESCE(is_active, TRUE) AND authority_tier = 'gold') AS gold_sources,
            COUNT(*) FILTER (WHERE COALESCE(is_active, TRUE) AND authority_tier = 'silver') AS silver_sources,
            COUNT(*) FILTER (WHERE COALESCE(is_active, TRUE) AND authority_tier = 'blocked') AS blocked_sources,
            COUNT(DISTINCT country_iso3) FILTER (WHERE COALESCE(is_active, TRUE)) AS active_countries
        FROM industrial_sources
        WHERE (:source_profile IS NULL OR source_profile = :source_profile)
        """
    )

    no_go_stmt = text(
        """
        SELECT COUNT(*) AS active_no_go_zones
        FROM industrial_no_go_zones
        WHERE is_active
          AND (:policy_version IS NULL OR policy_version = :policy_version)
        """
    )

    profile_breakdown_stmt = text(
        """
        SELECT source_profile, COUNT(*) AS n
        FROM industrial_sources
        WHERE COALESCE(is_active, TRUE)
        GROUP BY source_profile
        ORDER BY COUNT(*) DESC, source_profile
        LIMIT 25
        """
    )

    try:
        with get_engine().begin() as conn:
            latest_run = conn.execute(run_stmt, params).mappings().first()
            if latest_run is None:
                return None

            policy_row = conn.execute(
                policy_stmt,
                {"policy_version": policy_version},
            ).mappings().first()

            stats_row = conn.execute(
                source_stats_stmt,
                {"source_profile": source_profile},
            ).mappings().first()

            effective_policy_version = policy_version
            if not effective_policy_version and policy_row is not None:
                effective_policy_version = str(policy_row["policy_version"])

            no_go_row = conn.execute(
                no_go_stmt,
                {"policy_version": effective_policy_version},
            ).mappings().first()

            profile_rows = conn.execute(profile_breakdown_stmt).mappings().all()
    except Exception:
        return None

    payload = {
        "latest_run": dict(latest_run),
        "policy": dict(policy_row) if policy_row is not None else None,
        "source_profile_filter": source_profile,
        "source_stats": {
            "active_sources": int((stats_row or {}).get("active_sources") or 0),
            "gold_sources": int((stats_row or {}).get("gold_sources") or 0),
            "silver_sources": int((stats_row or {}).get("silver_sources") or 0),
            "blocked_sources": int((stats_row or {}).get("blocked_sources") or 0),
            "active_countries": int((stats_row or {}).get("active_countries") or 0),
        },
        "active_no_go_zones": int((no_go_row or {}).get("active_no_go_zones") or 0),
        "active_profiles": [
            {"source_profile": str(row["source_profile"]), "count": int(row["n"])}
            for row in profile_rows
            if row.get("source_profile")
        ],
    }
    return payload


def list_recent_denoiser_drift(limit: int = 50) -> list[dict]:
    """Return recent drift metric rows."""
    stmt = text(
        """
        SELECT
            id,
            model_id,
            metric_name,
            metric_value,
            threshold_value,
            window_start,
            window_end,
            triggered_rollback,
            payload_json,
            created_at
        FROM denoiser_drift_metrics
        ORDER BY created_at DESC, id DESC
        LIMIT :limit
        """
    )
    with get_engine().begin() as conn:
        rows = conn.execute(stmt, {"limit": max(1, int(limit))}).mappings().all()
    return [dict(r) for r in rows]


_COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def _format_nearest_place(name: str | None, dist_km: float | None, bearing_deg: float | None) -> str | None:
    if name is None or dist_km is None:
        return None
    idx = round((float(bearing_deg or 0) % 360) / 45) % 8
    return f"{int(dist_km)} km {_COMPASS[idx]} of {name}"


def list_denoiser_review_queue(limit: int = 200, status: str = "open") -> list[dict]:
    """List denoiser review queue rows enriched with location context."""
    stmt = text(
        """
        WITH limited_queue AS (
            SELECT
                id,
                event_id,
                fire_detection_id,
                reason,
                severity,
                status,
                payload_json,
                resolved_by,
                resolved_notes,
                resolved_at,
                created_at,
                updated_at
            FROM denoiser_review_queue
            WHERE status = :status
            ORDER BY created_at DESC, id DESC
            LIMIT :limit
        ),
        event_centroids AS (
            SELECT fe.event_id,
                   ST_Centroid(fe.geom)            AS centroid,
                   ST_Centroid(fe.geom)::geography AS centroid_geog
            FROM   fire_events fe
            INNER JOIN limited_queue lq ON lq.event_id = fe.event_id
        )
        SELECT
            lq.*,
            ST_Y(ec.centroid)                                                AS centroid_lat,
            ST_X(ec.centroid)                                                AS centroid_lon,
            UPPER(rgc.raw_payload->'address'->>'country_code')               AS country_code,
            rgc.admin1_name                                                  AS region_name,
            np.name                                                          AS _np_name,
            ROUND(
                ST_Distance(ec.centroid_geog, np.geom::geography) / 1000.0
            )                                                                AS _np_dist_km,
            degrees(
                ST_Azimuth(np.geom::geography, ec.centroid_geog)
            )                                                                AS _np_bearing_deg,
            lc.terrain_label
        FROM limited_queue lq
        LEFT JOIN event_centroids ec ON ec.event_id = lq.event_id
        LEFT JOIN reverse_geocode_cache rgc
            ON  ec.centroid IS NOT NULL
            AND rgc.provider      = :geocoding_provider
            AND rgc.cached_lat    = ROUND(CAST(ST_Y(ec.centroid) AS NUMERIC), :geocoding_precision)
            AND rgc.cached_lon    = ROUND(CAST(ST_X(ec.centroid) AS NUMERIC), :geocoding_precision)
            AND rgc.expires_at    > NOW()
        LEFT JOIN LATERAL (
            SELECT pp.name, pp.geom
            FROM   ne_populated_places pp
            WHERE  ec.centroid IS NOT NULL
              AND  ST_DWithin(ec.centroid_geog, pp.geom::geography, 200000)
            ORDER BY pp.geom <-> ec.centroid
            LIMIT 1
        ) np ON TRUE
        LEFT JOIN LATERAL (
            SELECT MODE() WITHIN GROUP (ORDER BY fd.landcover_label) AS terrain_label
            FROM   fire_detections fd
            WHERE  fd.event_id        = lq.event_id
              AND  fd.landcover_label IS NOT NULL
        ) lc ON TRUE
        """
    )
    params = {
        "status": str(status),
        "limit": max(1, int(limit)),
        "geocoding_provider": settings.geocoding_provider.strip().lower(),
        "geocoding_precision": settings.geocoding_cache_precision,
    }
    with get_engine().begin() as conn:
        conn.execute(_SPATIAL_QUERY_TIMEOUT)
        rows = conn.execute(stmt, params).mappings().all()

    result = []
    for r in rows:
        row = dict(r)
        row["nearest_place"] = _format_nearest_place(
            row.pop("_np_name"),
            row.pop("_np_dist_km"),
            row.pop("_np_bearing_deg"),
        )
        result.append(row)
    return result


def _build_reason_summary(
    reason: str,
    payload: dict | None,
    terrain_label: str | None,
) -> str:
    """Build a plain-language explanation of why this event was flagged."""
    p = payload or {}
    frp = p.get("frp_max")
    score = p.get("event_score")
    terrain = terrain_label or "unknown terrain"

    if reason == "fail_closed_hard_bypass":
        frp_str = f"{frp:.0f} MW " if frp is not None else ""
        return (
            f"Flagged automatically: {frp_str}fire radiative power in {terrain}. "
            "High-confidence FIRMS detection. Treated as confirmed fire until reviewed."
        )

    if reason == "fail_closed_or_uncertainty":
        score_str = f"{score:.2f}" if score is not None else "unknown"
        frp_note = (
            "Low FRP suggests possible industrial or agricultural burn."
            if frp is not None and frp < 50
            else "High FRP — likely a real fire needing confirmation."
            if frp is not None
            else "Insufficient FRP data to determine burn type."
        )
        return (
            f"Model score was {score_str} — right at the decision boundary "
            f"(threshold: 0.45–0.55). {frp_note}"
        )

    return f"Flagged for operator review (reason: {reason})."


def get_review_event_detail(event_id: str) -> dict | None:
    """Return decision-panel data for a single review queue event.

    Returns None if the event is not found in the review queue.
    Collects: reason summary, weather context, nearby fires (100 km / 48 h),
    location history (5 km / 30 d).
    """
    from api.core.weather import get_weather_context_for_point

    # ── 1. Fetch event centroid, reason, payload, terrain ──────────────────
    base_stmt = text(
        """
        SELECT
            drq.reason,
            drq.payload_json,
            ST_Y(ST_Centroid(fe.geom)) AS centroid_lat,
            ST_X(ST_Centroid(fe.geom)) AS centroid_lon,
            MODE() WITHIN GROUP (ORDER BY fd.landcover_label) AS terrain_label
        FROM denoiser_review_queue drq
        JOIN fire_events fe ON fe.event_id = drq.event_id
        LEFT JOIN fire_detections fd
            ON fd.event_id = drq.event_id AND fd.landcover_label IS NOT NULL
        WHERE drq.event_id = :event_id
        GROUP BY drq.reason, drq.payload_json, fe.geom
        LIMIT 1
        """
    )

    # ── 2. Nearby fires (100 km / 48 h) ───────────────────────────────────
    # LEFT JOIN aggregates FRP per event once; avoids a correlated subquery per row.
    nearby_stmt = text(
        """
        SELECT
            COUNT(fe.event_id)                                              AS cnt,
            MAX(fd_agg.total_frp)                                          AS max_frp,
            MIN(
                ST_Distance(
                    ST_Centroid(fe.geom)::geography,
                    ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography
                ) / 1000.0
            )                                                               AS nearest_km
        FROM fire_events fe
        LEFT JOIN (
            SELECT event_id, SUM(frp) AS total_frp
            FROM fire_detections
            GROUP BY event_id
        ) fd_agg ON fd_agg.event_id = fe.event_id
        WHERE fe.review_required IS NOT TRUE
          AND (fe.denoiser_decision IS NULL OR fe.denoiser_decision != 'drop')
          AND fe.end_time >= NOW() - INTERVAL '48 hours'
          AND fe.event_id != :event_id
          AND ST_DWithin(
                ST_Centroid(fe.geom)::geography,
                ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography,
                100000
              )
        """
    )

    # ── 3. Location history (5 km / 30 d) ─────────────────────────────────
    history_stmt = text(
        """
        SELECT
            drq.resolved_notes,
            COUNT(*) AS cnt
        FROM denoiser_review_queue drq
        JOIN fire_events fe ON fe.event_id = drq.event_id
        WHERE drq.event_id != :event_id
          AND drq.created_at >= NOW() - INTERVAL '30 days'
          AND ST_DWithin(
                ST_Centroid(fe.geom)::geography,
                ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography,
                5000
              )
        GROUP BY drq.resolved_notes
        """
    )

    with get_engine().begin() as conn:
        conn.execute(_SPATIAL_QUERY_TIMEOUT)

        base_row = conn.execute(base_stmt, {"event_id": event_id}).mappings().first()
        if base_row is None:
            return None

        lat = base_row["centroid_lat"]
        lon = base_row["centroid_lon"]
        coords = {"lat": lat, "lon": lon, "event_id": event_id}

        nearby_row = conn.execute(nearby_stmt, coords).mappings().first()
        history_rows = conn.execute(history_stmt, coords).mappings().all()

    history_confirmed = 0
    history_noise = 0
    history_other = 0
    for h in history_rows:
        notes = (h["resolved_notes"] or "").lower()
        if "confirmed" in notes or notes == "confirmed_fire":
            history_confirmed += int(h["cnt"])
        elif "noise" in notes or notes == "marked_noise":
            history_noise += int(h["cnt"])
        else:
            history_other += int(h["cnt"])
    history_flagged = history_confirmed + history_noise + history_other

    weather = None
    if lat is not None and lon is not None:
        try:
            weather = get_weather_context_for_point(
                lat=lat,
                lon=lon,
                ref_time=datetime.now(timezone.utc),
            )
        except Exception:
            weather = None

    wind_speed_kmh: float | None = None
    wind_dir: float | None = None
    rh: float | None = None
    temp: float | None = None
    if weather:
        ws = weather.get("wind_speed_ms")
        wind_speed_kmh = round(ws * 3.6, 1) if ws is not None else None
        wind_dir = weather.get("wind_direction_deg")
        rh = weather.get("relative_humidity_pct")
        temp = weather.get("temperature_c")

    payload = base_row["payload_json"] or {}

    reason_summary = _build_reason_summary(
        reason=base_row["reason"],
        payload=payload,
        terrain_label=base_row["terrain_label"],
    )

    return {
        "reason_summary": reason_summary,
        "centroid_lat": lat,
        "centroid_lon": lon,
        "wind_speed_kmh": wind_speed_kmh,
        "wind_direction_deg": wind_dir,
        "relative_humidity_pct": rh,
        "temperature_c": temp,
        "nearby_fires_count": int(nearby_row["cnt"]) if nearby_row else 0,
        "nearby_fires_max_frp_mw": float(nearby_row["max_frp"]) if nearby_row and nearby_row["max_frp"] is not None else None,
        "nearby_fires_nearest_km": float(nearby_row["nearest_km"]) if nearby_row and nearby_row["nearest_km"] is not None else None,
        "location_history_flagged": history_flagged,
        "location_history_confirmed": history_confirmed,
        "location_history_noise": history_noise,
    }


def resolve_denoiser_review_event(
    event_id: str,
    *,
    resolved_by: str,
    resolved_notes: str | None = None,
) -> int:
    """Resolve all open review items for an event."""
    stmt = text(
        """
        UPDATE denoiser_review_queue
        SET
            status = 'resolved',
            resolved_by = :resolved_by,
            resolved_notes = :resolved_notes,
            resolved_at = NOW(),
            updated_at = NOW()
        WHERE event_id = :event_id
          AND status = 'open'
        """
    )
    with get_engine().begin() as conn:
        result = conn.execute(
            stmt,
            {
                "event_id": event_id,
                "resolved_by": resolved_by,
                "resolved_notes": resolved_notes,
            },
        )
    return int(result.rowcount or 0)
