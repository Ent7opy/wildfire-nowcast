"""Strategy pattern for batch fire detection scoring.

Each scoring dimension is implemented as a ``ScoringStrategy``-conforming class.
``run_scoring_stage()`` is the single driver that handles the fetch → compute →
batch-update loop.  Adding a new scoring dimension requires exactly one class and
one test file; nothing else.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Protocol, TYPE_CHECKING

from sqlalchemy import text

if TYPE_CHECKING:
    from sqlalchemy.engine import Connection

from api.db import get_engine
from api.fires.scoring import (
    compute_persistence_scores,
    compute_weather_plausibility_scores,
    mask_false_sources,
)

LOGGER = logging.getLogger(__name__)

# ── Neutral-fallback configuration ────────────────────────────────────────────
# Large batches (global backfills / bulk repairs) can make geospatial scoring
# prohibitively slow.  Scores are replaced with a neutral value to keep ingest
# fail-closed gates enforceable while preserving throughput.
# SCIENCE_DEBT SD-01: replace with chunked computation to reach science_grade.

_LARGE_BATCH_WEATHER_NEUTRAL_THRESHOLD = int(
    os.getenv("FIRE_SCORING_WEATHER_NEUTRAL_THRESHOLD", "5000")
)
_NEUTRAL_WEATHER_SCORE: float = 0.5
_WEATHER_TIME_TOLERANCE_HOURS = float(
    os.getenv("FIRE_SCORING_WEATHER_TIME_TOLERANCE_HOURS", "6")
)
_LARGE_BATCH_PERSISTENCE_NEUTRAL_THRESHOLD = int(
    os.getenv("FIRE_SCORING_PERSISTENCE_NEUTRAL_THRESHOLD", "20000")
)
_NEUTRAL_PERSISTENCE_SCORE: float = 0.3
_DISABLE_NEUTRAL_FALLBACK = (
    str(os.getenv("FIRE_SCORING_DISABLE_NEUTRAL_FALLBACK", "false")).strip().lower()
    in {"1", "true", "yes", "on"}
)


# ── Protocol ──────────────────────────────────────────────────────────────────

class ScoringStrategy(Protocol):
    """Structural interface for a single scoring dimension.

    Implementations are plain classes — no inheritance required.  Any object
    that provides these attributes and methods satisfies the Protocol, which
    makes it trivial to stub in tests without subclassing.
    """

    #: Short identifier used in log step names (e.g. ``"persistence"``).
    name: str
    #: Columns to SELECT from ``fire_detections`` for this stage.
    select_fields: tuple[str, ...]
    #: Column to UPDATE in ``fire_detections``.  Must be a compile-time constant
    #: — never derived from user input.
    update_column: str
    #: Named parameter used in the per-detection UPDATE (e.g. ``"masked"``).
    update_param: str
    #: Minimum batch size that triggers bulk neutral-value assignment (0 = off).
    neutral_threshold: int
    #: Value written when the neutral fallback fires.
    neutral_value: Any

    def compute(self, detections: list[dict]) -> dict[int, Any]:
        """Compute scores for *detections* and return a mapping of id → value."""
        ...

    def count_result(self, results: dict[int, Any]) -> int:
        """Return the meaningful count to report for a completed stage."""
        ...


# ── Driver ────────────────────────────────────────────────────────────────────

def _log_step(step: str, started_at: float, *, batch_id: int, rows: int | None = None) -> None:
    elapsed = time.perf_counter() - started_at
    suffix = f", rows={rows}" if rows is not None else ""
    LOGGER.info("batch=%s %s completed in %.3fs%s", batch_id, step, elapsed, suffix)


def run_scoring_stage(
    batch_id: int,
    strategy: ScoringStrategy,
    conn: Connection | None = None,
) -> int:
    """Execute one scoring stage for *batch_id* using *strategy*.

    The driver owns the entire fetch → compute → batch-update lifecycle.
    *strategy* supplies the SELECT fields, compute logic, UPDATE column, and
    optional large-batch neutral fallback.

    ``update_column``, ``update_param``, and ``select_fields`` are
    design-time constants on each strategy class — not user input — so
    interpolating them into SQL is safe.  All runtime values (IDs, scores)
    are bound as parameters.

    Args:
        batch_id: Ingest batch to process.
        strategy: ScoringStrategy instance driving this stage.
        conn: Existing connection to reuse (useful for multi-stage transactions).

    Returns:
        Count of detections meaningfully updated (strategy-defined).
    """
    def _execute(active_conn: Connection) -> int:
        fields = ", ".join(strategy.select_fields)
        select_stmt = text(
            f"SELECT {fields} FROM fire_detections WHERE ingest_batch_id = :batch_id"
        )

        started = time.perf_counter()
        rows = active_conn.execute(select_stmt, {"batch_id": batch_id}).mappings().all()
        _log_step(f"{strategy.name}.fetch_batch", started, batch_id=batch_id, rows=len(rows))

        detections = [dict(r) for r in rows]
        if not detections:
            return 0

        det_count = len(detections)
        neutral_threshold = strategy.neutral_threshold
        if (
            (not _DISABLE_NEUTRAL_FALLBACK)
            and neutral_threshold > 0
            and det_count >= neutral_threshold
        ):
            LOGGER.warning(
                "Batch %s has %s detections; assigning neutral %s=%s for bulk throughput.",
                batch_id,
                det_count,
                strategy.update_column,
                strategy.neutral_value,
            )
            neutral_stmt = text(
                f"UPDATE fire_detections"
                f" SET {strategy.update_column} = :value"
                f" WHERE ingest_batch_id = :batch_id"
            )
            started = time.perf_counter()
            active_conn.execute(neutral_stmt, {"batch_id": batch_id, "value": strategy.neutral_value})
            _log_step(
                f"{strategy.name}.update_neutral",
                started,
                batch_id=batch_id,
                rows=det_count,
            )
            return det_count

        started = time.perf_counter()
        results = strategy.compute(detections)
        _log_step(
            f"{strategy.name}.compute_scores", started, batch_id=batch_id, rows=len(results)
        )

        update_stmt = text(
            f"UPDATE fire_detections"
            f" SET {strategy.update_column} = :{strategy.update_param}"
            f" WHERE id = :detection_id"
        )
        params = [
            {"detection_id": det_id, strategy.update_param: value}
            for det_id, value in results.items()
        ]
        started = time.perf_counter()
        active_conn.execute(update_stmt, params)
        _log_step(
            f"{strategy.name}.update_batch", started, batch_id=batch_id, rows=len(params)
        )

        return strategy.count_result(results)

    if conn is not None:
        return _execute(conn)
    with get_engine().begin() as new_conn:
        return _execute(new_conn)


# ── Strategy implementations ──────────────────────────────────────────────────

class FalseSourceMaskingStrategy:
    """Marks detections near known industrial false-positive sources."""

    name = "false_source"
    select_fields = ("id", "lat", "lon")
    update_column = "false_source_masked"
    update_param = "masked"
    neutral_threshold = 0  # masking is cheap; no neutral fallback needed
    neutral_value = False

    def compute(self, detections: list[dict]) -> dict[int, Any]:
        return mask_false_sources(detections)

    def count_result(self, results: dict[int, Any]) -> int:
        return sum(1 for v in results.values() if v)


class PersistenceScoringStrategy:
    """Scores detections by spatial-temporal clustering with nearby fire history."""

    name = "persistence"
    select_fields = ("id", "lat", "lon", "acq_time", "sensor")
    update_column = "persistence_score"
    update_param = "score"
    neutral_threshold = _LARGE_BATCH_PERSISTENCE_NEUTRAL_THRESHOLD
    neutral_value = _NEUTRAL_PERSISTENCE_SCORE

    def compute(self, detections: list[dict]) -> dict[int, Any]:
        return compute_persistence_scores(detections)

    def count_result(self, results: dict[int, Any]) -> int:
        return len(results)


class LandcoverScoringStrategy:
    """Scores detections by land-cover plausibility (forest, water, urban, etc.)."""

    name = "landcover"
    select_fields = ("id", "lat", "lon")
    update_column = "landcover_score"
    update_param = "score"
    neutral_threshold = 0  # raster lookup is fast; no neutral fallback needed
    neutral_value = 0.5

    def compute(self, detections: list[dict]) -> dict[int, Any]:
        from api.fires.landcover import compute_landcover_scores
        return compute_landcover_scores(detections)

    def count_result(self, results: dict[int, Any]) -> int:
        return len(results)


class WeatherScoringStrategy:
    """Scores detections by meteorological plausibility (RH, precipitation, wind)."""

    name = "weather"
    select_fields = ("id", "lat", "lon", "acq_time")
    update_column = "weather_score"
    update_param = "score"
    neutral_threshold = _LARGE_BATCH_WEATHER_NEUTRAL_THRESHOLD
    neutral_value = _NEUTRAL_WEATHER_SCORE

    def compute(self, detections: list[dict]) -> dict[int, Any]:
        return compute_weather_plausibility_scores(
            detections,
            time_tolerance_hours=_WEATHER_TIME_TOLERANCE_HOURS,
        )

    def count_result(self, results: dict[int, Any]) -> int:
        return len(results)
