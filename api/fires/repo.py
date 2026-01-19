"""DB queries for fire detections."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Literal

from sqlalchemy import text

from api.db import get_engine

BBox = tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)

# Keep this list tight to avoid SQL injection when constructing SELECT clauses.
_ALLOWED_COLUMNS: dict[str, str] = {
    "id": "id",
    "acq_time": "acq_time",
    "lat": "lat",
    "lon": "lon",
    "frp": "frp",
    "confidence": "confidence",
    "confidence_score": "confidence_score",
    "brightness": "brightness",
    "bright_t31": "bright_t31",
    "scan": "scan",
    "track": "track",
    "sensor": "sensor",
    "source": "source",
    "is_noise": "is_noise",
    "denoised_score": "denoised_score",
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
    min_confidence: float | None = None,
) -> list[dict]:
    """List fire detections in a lon/lat bbox and acquisition time window.

    Notes
    - Time filter uses `BETWEEN` (inclusive bounds).
    - Spatial filter uses GiST index-friendly predicates:
      `geom && envelope` plus `ST_Intersects(geom, envelope)`.
    - Denoiser: By default, filters out rows where `is_noise` is TRUE.
    """

    min_lon, min_lat, max_lon, max_lat = bbox

    cols = list(columns)
    if not cols:
        raise ValueError("columns must be non-empty.")

    select_parts: list[str] = []
    for c in cols:
        if c not in _ALLOWED_COLUMNS:
            raise ValueError(f"Unsupported column: {c}")
        expr = _ALLOWED_COLUMNS[c]
        select_parts.append(f"{expr} AS {c}")
    select_sql = ",\n            ".join(select_parts)

    if order not in ("asc", "desc"):
        raise ValueError("order must be 'asc' or 'desc'.")

    # Noise filter: default to excluding detections explicitly marked as noise.
    # We use "IS NOT TRUE" to include NULLs (detections not yet scored).
    noise_predicate = ""
    if not include_noise:
        noise_predicate = "AND is_noise IS NOT TRUE"

    confidence_predicate = ""
    if min_confidence is not None:
        # Include NULL confidence values when filtering (NULL means unknown, not 0)
        confidence_predicate = "AND (confidence IS NULL OR confidence >= :min_confidence)"

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
          {confidence_predicate}
        ORDER BY acq_time {order}
        {limit_sql}
        """
    )

    with get_engine().begin() as conn:
        result = conn.execute(stmt, params)
        rows = result.mappings().all()
    return [dict(r) for r in rows]

