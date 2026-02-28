"""Recompute fire scoring fields for batches with incomplete derived columns."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

# Add project root to sys.path so local packages are importable when run as a script.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sqlalchemy import text

from api.db import get_engine
from api.fires.repo import update_all_scoring_for_batch
from ingest import repository as ingest_repository


def _parse_dt(raw: str | None, *, end: bool = False) -> datetime | None:
    if raw is None:
        return None
    text_raw = str(raw).strip()
    if not text_raw:
        return None
    if len(text_raw) == 10:
        dt = datetime.strptime(text_raw, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return dt + timedelta(days=1) if end else dt
    dt = datetime.fromisoformat(text_raw.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def _collect_incomplete_batches(
    *,
    limit: int | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    force_recompute: bool = False,
    neutral_persistence: float = 0.3,
    neutral_landcover: float = 0.5,
    neutral_weather: float = 0.5,
) -> list[int]:
    predicates = " OR ".join(f"{col} IS NULL" for col in ingest_repository.REQUIRED_SCORING_COLUMNS)
    selection_predicate = f"({predicates})"
    if force_recompute:
        selection_predicate = (
            f"({selection_predicate}) OR "
            "(persistence_score = :neutral_persistence) OR "
            "(landcover_score = :neutral_landcover) OR "
            "(weather_score = :neutral_weather)"
        )
    limit_sql = ""
    params: dict[str, object] = {}
    time_sql = ""
    if start_time is not None:
        params["start_time"] = start_time
        time_sql += " AND acq_time >= :start_time\n"
    if end_time is not None:
        params["end_time"] = end_time
        time_sql += " AND acq_time < :end_time\n"
    if limit is not None:
        limit_sql = "LIMIT :limit"
        params["limit"] = int(limit)
    if force_recompute:
        params["neutral_persistence"] = float(neutral_persistence)
        params["neutral_landcover"] = float(neutral_landcover)
        params["neutral_weather"] = float(neutral_weather)

    stmt = text(
        f"""
        SELECT ingest_batch_id
        FROM fire_detections
        WHERE ingest_batch_id IS NOT NULL
          AND (source IS NULL OR source NOT LIKE 'mvt_%')
          {time_sql}
          AND ({selection_predicate})
        GROUP BY ingest_batch_id
        ORDER BY ingest_batch_id DESC
        {limit_sql}
        """
    )
    with get_engine().begin() as conn:
        rows = conn.execute(stmt, params).scalars().all()
    return [int(row) for row in rows if row is not None]


def _recompute_batch(batch_id: int) -> dict[str, int]:
    with get_engine().begin() as conn:
        counts = update_all_scoring_for_batch(batch_id, conn=conn)
        remaining_incomplete = ingest_repository.count_rows_with_null_columns_for_batch(
            batch_id,
            columns=ingest_repository.REQUIRED_SCORING_COLUMNS,
            exclude_source_like="mvt_%",
            conn=conn,
        )
    if remaining_incomplete > 0:
        raise RuntimeError(
            f"Batch {batch_id} still has {remaining_incomplete} production rows with NULL scoring fields"
        )
    return counts


def _parse_batch_ids(values: Iterable[int] | None) -> list[int]:
    if not values:
        return []
    unique: list[int] = []
    seen: set[int] = set()
    for value in values:
        batch_id = int(value)
        if batch_id <= 0:
            raise ValueError(f"Invalid batch id: {batch_id}")
        if batch_id not in seen:
            seen.add(batch_id)
            unique.append(batch_id)
    return unique


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Recompute fire scoring for batches with missing derived fields."
    )
    parser.add_argument(
        "--batch-id",
        action="append",
        type=int,
        dest="batch_ids",
        help="Specific ingest_batch_id to recompute (repeatable). If omitted, scans all incomplete batches.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional legacy limit when auto-discovering incomplete batches.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Maximum number of batches to process.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of batch ids to process per loop iteration.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Optional lower bound (YYYY-MM-DD or ISO8601) on fire_detections.acq_time.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Optional upper bound (YYYY-MM-DD or ISO8601) on fire_detections.acq_time.",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Include batches containing neutral fallback values even if NULLs are absent.",
    )
    parser.add_argument(
        "--neutral-persistence",
        type=float,
        default=0.3,
        help="Neutral persistence value used to target fallback rows (default: 0.3).",
    )
    parser.add_argument(
        "--neutral-landcover",
        type=float,
        default=0.5,
        help="Neutral landcover value used to target fallback rows (default: 0.5).",
    )
    parser.add_argument(
        "--neutral-weather",
        type=float,
        default=0.5,
        help="Neutral weather value used to target fallback rows (default: 0.5).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List candidate batches without mutating data.",
    )
    args = parser.parse_args(argv)

    start_time = _parse_dt(args.start, end=False)
    end_time = _parse_dt(args.end, end=True)
    if start_time and end_time and start_time >= end_time:
        raise ValueError("--start must be earlier than --end")
    max_batches = args.max_batches if args.max_batches is not None else args.limit
    if max_batches is not None and int(max_batches) <= 0:
        raise ValueError("--max-batches must be positive")
    batch_size = max(1, int(args.batch_size))

    batch_ids = _parse_batch_ids(args.batch_ids)
    if not batch_ids:
        batch_ids = _collect_incomplete_batches(
            limit=max_batches,
            start_time=start_time,
            end_time=end_time,
            force_recompute=bool(args.force_recompute),
            neutral_persistence=float(args.neutral_persistence),
            neutral_landcover=float(args.neutral_landcover),
            neutral_weather=float(args.neutral_weather),
        )
    elif max_batches is not None:
        batch_ids = batch_ids[: int(max_batches)]

    summary: dict[str, object] = {
        "as_of_utc": datetime.now(timezone.utc).isoformat(),
        "batch_ids": batch_ids,
        "dry_run": bool(args.dry_run),
        "window": {
            "start_time": start_time.isoformat() if start_time else None,
            "end_time": end_time.isoformat() if end_time else None,
        },
        "force_recompute": bool(args.force_recompute),
        "batch_size": batch_size,
        "max_batches": int(max_batches) if max_batches is not None else None,
        "neutral_targets": {
            "persistence": float(args.neutral_persistence),
            "landcover": float(args.neutral_landcover),
            "weather": float(args.neutral_weather),
        },
        "updated_batches": [],
    }

    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return 0

    for chunk_start in range(0, len(batch_ids), batch_size):
        chunk = batch_ids[chunk_start : chunk_start + batch_size]
        for batch_id in chunk:
            counts = _recompute_batch(batch_id)
            summary["updated_batches"].append(
                {
                    "batch_id": batch_id,
                    **counts,
                }
            )

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
