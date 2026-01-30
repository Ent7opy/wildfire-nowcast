"""DB queries for fire detections."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Literal, TYPE_CHECKING

from sqlalchemy import text, column as sa_column

if TYPE_CHECKING:
    from sqlalchemy.engine import Connection

from api.db import get_engine
from api.fires.scoring import (
    compute_fire_likelihood,
    compute_persistence_scores,
    compute_weather_plausibility_scores,
    mask_false_sources,
)

BBox = tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)


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
}


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
) -> list[dict]:
    """List fire detections in a lon/lat bbox and acquisition time window.

    Notes
    - Time filter uses `BETWEEN` (inclusive bounds).
    - Spatial filter uses GiST index-friendly predicates:
      `geom && envelope` plus `ST_Intersects(geom, envelope)`.
    - Denoiser: By default, filters out rows where `is_noise` is TRUE.
    - False-source masking: By default, filters out rows where `false_source_masked` is TRUE.
    - Filtering: min_confidence filters FIRMS confidence (0-100), min_fire_likelihood filters
      composite likelihood score (0-1). Both include NULL values (not yet scored).
    """

    min_lon, min_lat, max_lon, max_lat = bbox

    cols = list(columns)
    if not cols:
        raise ValueError("columns must be non-empty.")

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

    limit_sql = ""
    # Ensure datetimes are timezone-aware (UTC) for database queries
    if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=timezone.utc)
    elif start_time.tzinfo != timezone.utc:
        start_time = start_time.astimezone(timezone.utc)
    
    if end_time.tzinfo is None:
        end_time = end_time.replace(tzinfo=timezone.utc)
    elif end_time.tzinfo != timezone.utc:
        end_time = end_time.astimezone(timezone.utc)
    
    params: dict[str, object] = {
        "start_time": start_time,
        "end_time": end_time,
        "min_lon": float(min_lon),
        "min_lat": float(min_lat),
        "max_lon": float(max_lon),
        "max_lat": float(max_lat),
    }
    if min_confidence is not None:
        params["min_confidence"] = float(min_confidence)
    if min_fire_likelihood is not None:
        params["min_fire_likelihood"] = float(min_fire_likelihood)
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive.")
        limit_sql = "\n        LIMIT :limit"
        params["limit"] = int(limit)

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
        ORDER BY acq_time {order}
        {limit_sql}
        """
    )

    with get_engine().begin() as conn:
        result = conn.execute(stmt, params)
        rows = result.mappings().all()
    return [dict(r) for r in rows]


def update_false_source_masking(batch_id: int, conn: Connection | None = None) -> int:
    """Update false_source_masked column for detections in a batch.

    Queries detections from the batch, uses mask_false_sources() to identify
    detections near industrial sources, and updates the false_source_masked column.

    Args:
        batch_id: The ingest batch ID to process
        conn: Optional existing database connection to use. If provided, this
            connection will be used for the operation and no new connection
            will be opened. This is useful for batching multiple scoring updates
            within a single transaction to avoid connection pool exhaustion.

    Returns:
        Number of detections marked as masked
    """
    # Query detections from the batch
    stmt = text("""
        SELECT id, lat, lon
        FROM fire_detections
        WHERE ingest_batch_id = :batch_id
    """)

    def _execute(conn: Connection) -> int:
        result = conn.execute(stmt, {"batch_id": batch_id})
        rows = result.mappings().all()

        detections = [dict(r) for r in rows]
        if not detections:
            return 0

        # Compute masking results
        masked_results = mask_false_sources(detections)

        # Update fire_detections table with masking results
        update_stmt = text("""
            UPDATE fire_detections
            SET false_source_masked = :masked
            WHERE id = :detection_id
        """)

        params = [
            {"detection_id": det_id, "masked": is_masked}
            for det_id, is_masked in masked_results.items()
        ]

        conn.execute(update_stmt, params)

        # Count how many were marked as masked
        return sum(1 for is_masked in masked_results.values() if is_masked)

    if conn is not None:
        return _execute(conn)
    else:
        with get_engine().begin() as new_conn:
            return _execute(new_conn)


def update_persistence_scores(batch_id: int, conn: Connection | None = None) -> int:
    """Update persistence_score column for detections in a batch.

    Queries detections from the batch, uses compute_persistence_scores()
    to compute spatial-temporal clustering scores, and updates the persistence_score column.

    Args:
        batch_id: The ingest batch ID to process
        conn: Optional existing database connection to use. If provided, this
            connection will be used for the operation and no new connection
            will be opened. This is useful for batching multiple scoring updates
            within a single transaction to avoid connection pool exhaustion.

    Returns:
        Number of detections with scores updated
    """
    def _execute(conn: Connection) -> int:
        # Query detections from the batch with required fields
        stmt = text("""
            SELECT id, lat, lon, acq_time, sensor
            FROM fire_detections
            WHERE ingest_batch_id = :batch_id
        """)

        result = conn.execute(stmt, {"batch_id": batch_id})
        rows = result.mappings().all()

        detections = [dict(r) for r in rows]
        if not detections:
            return 0

        # Compute persistence scores
        persistence_scores = compute_persistence_scores(detections)

        # Update fire_detections table with persistence scores
        update_stmt = text("""
            UPDATE fire_detections
            SET persistence_score = :score
            WHERE id = :detection_id
        """)

        params = [
            {"detection_id": det_id, "score": score}
            for det_id, score in persistence_scores.items()
        ]

        conn.execute(update_stmt, params)

        return len(persistence_scores)

    if conn is not None:
        return _execute(conn)
    else:
        with get_engine().begin() as new_conn:
            return _execute(new_conn)


def update_landcover_scores(batch_id: int, conn: Connection | None = None) -> int:
    """Update landcover_score column for detections in a batch.

    Queries detections from the batch, uses compute_landcover_scores()
    to compute land-cover plausibility scores, and updates the landcover_score column.

    Args:
        batch_id: The ingest batch ID to process
        conn: Optional existing database connection to use. If provided, this
            connection will be used for the operation and no new connection
            will be opened. This is useful for batching multiple scoring updates
            within a single transaction to avoid connection pool exhaustion.

    Returns:
        Number of detections with scores updated
    """
    def _execute(conn: Connection) -> int:
        # Query detections from the batch with required fields
        stmt = text("""
            SELECT id, lat, lon
            FROM fire_detections
            WHERE ingest_batch_id = :batch_id
        """)

        result = conn.execute(stmt, {"batch_id": batch_id})
        rows = result.mappings().all()

        detections = [dict(r) for r in rows]
        if not detections:
            return 0

        # Import landcover module
        from api.fires.landcover import compute_landcover_scores

        # Compute landcover scores
        landcover_scores = compute_landcover_scores(detections)

        # Update fire_detections table with landcover scores
        update_stmt = text("""
            UPDATE fire_detections
            SET landcover_score = :score
            WHERE id = :detection_id
        """)

        params = [
            {"detection_id": det_id, "score": score}
            for det_id, score in landcover_scores.items()
        ]

        conn.execute(update_stmt, params)

        return len(landcover_scores)

    if conn is not None:
        return _execute(conn)
    else:
        with get_engine().begin() as new_conn:
            return _execute(new_conn)


def update_weather_scores(batch_id: int, conn: Connection | None = None) -> int:
    """Update weather_score column for detections in a batch.

    Queries detections from the batch, uses compute_weather_plausibility_scores()
    to compute weather plausibility scores, and updates the weather_score column.

    Args:
        batch_id: The ingest batch ID to process
        conn: Optional existing database connection to use. If provided, this
            connection will be used for the operation and no new connection
            will be opened. This is useful for batching multiple scoring updates
            within a single transaction to avoid connection pool exhaustion.

    Returns:
        Number of detections with scores updated
    """
    def _execute(conn: Connection) -> int:
        # Query detections from the batch with required fields
        stmt = text("""
            SELECT id, lat, lon, acq_time
            FROM fire_detections
            WHERE ingest_batch_id = :batch_id
        """)

        result = conn.execute(stmt, {"batch_id": batch_id})
        rows = result.mappings().all()

        detections = [dict(r) for r in rows]
        if not detections:
            return 0

        # Compute weather plausibility scores
        weather_scores = compute_weather_plausibility_scores(detections)

        # Update fire_detections table with weather scores
        update_stmt = text("""
            UPDATE fire_detections
            SET weather_score = :score
            WHERE id = :detection_id
        """)

        params = [
            {"detection_id": det_id, "score": score}
            for det_id, score in weather_scores.items()
        ]

        conn.execute(update_stmt, params)

        return len(weather_scores)

    if conn is not None:
        return _execute(conn)
    else:
        with get_engine().begin() as new_conn:
            return _execute(new_conn)


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

        result = conn.execute(stmt, {"batch_id": batch_id})
        rows = result.mappings().all()

        if not rows:
            return 0

        # Compute fire likelihood for each detection
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

        # Update fire_detections table with fire likelihood
        update_stmt = text("""
            UPDATE fire_detections
            SET fire_likelihood = :likelihood
            WHERE id = :detection_id
        """)

        conn.execute(update_stmt, params)

        return len(params)

    if conn is not None:
        return _execute(conn)
    else:
        with get_engine().begin() as new_conn:
            return _execute(new_conn)



def update_all_scoring_for_batch(batch_id: int) -> dict[str, int]:
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
    with get_engine().begin() as conn:
        masked_count = update_false_source_masking(batch_id, conn=conn)
        persistence_count = update_persistence_scores(batch_id, conn=conn)
        landcover_count = update_landcover_scores(batch_id, conn=conn)
        weather_count = update_weather_scores(batch_id, conn=conn)
        likelihood_count = update_fire_likelihood(batch_id, conn=conn)
    
    return {
        "masked_count": masked_count,
        "persistence_count": persistence_count,
        "landcover_count": landcover_count,
        "weather_count": weather_count,
        "likelihood_count": likelihood_count,
    }
