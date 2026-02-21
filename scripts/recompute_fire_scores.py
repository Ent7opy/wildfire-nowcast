"""Recompute fire scoring fields for batches with incomplete derived columns."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
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


def _collect_incomplete_batches(limit: int | None = None) -> list[int]:
    predicates = " OR ".join(f"{col} IS NULL" for col in ingest_repository.REQUIRED_SCORING_COLUMNS)
    limit_sql = ""
    params: dict[str, object] = {}
    if limit is not None:
        limit_sql = "LIMIT :limit"
        params["limit"] = int(limit)

    stmt = text(
        f"""
        SELECT ingest_batch_id
        FROM fire_detections
        WHERE ingest_batch_id IS NOT NULL
          AND (source IS NULL OR source NOT LIKE 'mvt_%')
          AND ({predicates})
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
        help="Optional limit when auto-discovering incomplete batches.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List candidate batches without mutating data.",
    )
    args = parser.parse_args(argv)

    batch_ids = _parse_batch_ids(args.batch_ids)
    if not batch_ids:
        batch_ids = _collect_incomplete_batches(limit=args.limit)

    summary: dict[str, object] = {
        "as_of_utc": datetime.now(timezone.utc).isoformat(),
        "batch_ids": batch_ids,
        "dry_run": bool(args.dry_run),
        "updated_batches": [],
    }

    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return 0

    for batch_id in batch_ids:
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
