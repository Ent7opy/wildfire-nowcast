"""Repair utility for fire detection data integrity and batch metadata."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add project root to sys.path so local packages are importable when run as a script.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sqlalchemy import text

from api.db import get_engine
from ingest import repository as ingest_repository


def _purge_synthetic_rows(source_like: str, *, dry_run: bool) -> int:
    count_stmt = text(
        """
        SELECT COUNT(*)
        FROM fire_detections
        WHERE source LIKE :source_like
        """
    )
    delete_stmt = text(
        """
        DELETE FROM fire_detections
        WHERE source LIKE :source_like
        """
    )
    with get_engine().begin() as conn:
        count = int(conn.execute(count_stmt, {"source_like": source_like}).scalar_one() or 0)
        if not dry_run and count > 0:
            conn.execute(delete_stmt, {"source_like": source_like})
    return count


def _backfill_thermal_columns(*, dry_run: bool) -> int:
    count_stmt = text(
        """
        SELECT COUNT(*)
        FROM fire_detections
        WHERE brightness IS NULL OR bright_t31 IS NULL
        """
    )
    update_stmt = text(
        """
        UPDATE fire_detections
        SET
            brightness = COALESCE(
                brightness,
                CASE
                    WHEN NULLIF(raw_properties->>'brightness', '') ~ '^-?[0-9]+(\\.[0-9]+)?$'
                    THEN (raw_properties->>'brightness')::double precision
                    ELSE NULL
                END,
                CASE
                    WHEN NULLIF(raw_properties->>'bright_ti4', '') ~ '^-?[0-9]+(\\.[0-9]+)?$'
                    THEN (raw_properties->>'bright_ti4')::double precision
                    ELSE NULL
                END
            ),
            bright_t31 = COALESCE(
                bright_t31,
                CASE
                    WHEN NULLIF(raw_properties->>'bright_t31', '') ~ '^-?[0-9]+(\\.[0-9]+)?$'
                    THEN (raw_properties->>'bright_t31')::double precision
                    ELSE NULL
                END,
                CASE
                    WHEN NULLIF(raw_properties->>'bright_ti5', '') ~ '^-?[0-9]+(\\.[0-9]+)?$'
                    THEN (raw_properties->>'bright_ti5')::double precision
                    ELSE NULL
                END
            )
        WHERE brightness IS NULL OR bright_t31 IS NULL
        """
    )
    with get_engine().begin() as conn:
        count = int(conn.execute(count_stmt).scalar_one() or 0)
        if not dry_run and count > 0:
            conn.execute(update_stmt)
    return count


def _fail_stale_running_batches(
    *,
    stale_after_hours: float,
    reason: str,
    dry_run: bool,
) -> list[int]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=float(stale_after_hours))
    select_stmt = text(
        """
        SELECT id
        FROM ingest_batches
        WHERE status = 'running'
          AND started_at < :cutoff
        ORDER BY id
        """
    )
    update_stmt = text(
        """
        UPDATE ingest_batches
        SET
            status = 'failed',
            completed_at = COALESCE(completed_at, NOW()),
            "metadata" = COALESCE("metadata", '{}'::jsonb) || jsonb_build_object(
                'repair_running_to_failed_at', :repair_time,
                'repair_reason', :reason
            )
        WHERE id = :batch_id
        """
    )

    with get_engine().begin() as conn:
        batch_ids = [int(row) for row in conn.execute(select_stmt, {"cutoff": cutoff}).scalars().all()]
        if not dry_run:
            repair_time = datetime.now(timezone.utc).isoformat()
            for batch_id in batch_ids:
                conn.execute(
                    update_stmt,
                    {
                        "batch_id": batch_id,
                        "repair_time": repair_time,
                        "reason": reason,
                    },
                )
    return batch_ids


def _repair_batch_metadata(
    *,
    batch_id: int,
    require_denoiser: bool,
    dry_run: bool,
) -> dict[str, object]:
    get_status_stmt = text(
        """
        SELECT status
        FROM ingest_batches
        WHERE id = :batch_id
        """
    )
    count_stmt = text(
        """
        SELECT COUNT(*)
        FROM fire_detections
        WHERE ingest_batch_id = :batch_id
        """
    )
    update_stmt = text(
        """
        UPDATE ingest_batches
        SET
            status = :status,
            completed_at = COALESCE(completed_at, NOW()),
            record_count = :record_count,
            records_fetched = :records_fetched,
            records_inserted = :records_inserted,
            records_skipped_duplicates = :records_skipped_duplicates,
            "metadata" = COALESCE("metadata", '{}'::jsonb) || jsonb_build_object(
                'repair_audit_at', :repair_audit_at,
                'repair_previous_status', :repair_previous_status,
                'repair_reason', :repair_reason,
                'repair_require_denoiser', :repair_require_denoiser,
                'repair_scoring_incomplete_rows', :repair_scoring_incomplete_rows,
                'repair_denoiser_incomplete_rows', :repair_denoiser_incomplete_rows
            )
        WHERE id = :batch_id
        """
    )

    with get_engine().begin() as conn:
        original_status = conn.execute(get_status_stmt, {"batch_id": batch_id}).scalar_one_or_none()
        if original_status is None:
            raise RuntimeError(f"Batch {batch_id} does not exist")

        row_count = int(conn.execute(count_stmt, {"batch_id": batch_id}).scalar_one() or 0)
        scoring_incomplete = ingest_repository.count_rows_with_null_columns_for_batch(
            batch_id,
            columns=ingest_repository.REQUIRED_SCORING_COLUMNS,
            exclude_source_like="mvt_%",
            conn=conn,
        )
        denoiser_incomplete = ingest_repository.count_rows_with_null_columns_for_batch(
            batch_id,
            columns=ingest_repository.REQUIRED_DENOISER_COLUMNS,
            exclude_source_like="mvt_%",
            conn=conn,
        )
        quality_ok = scoring_incomplete == 0 and (denoiser_incomplete == 0 if require_denoiser else True)
        target_status = "succeeded" if quality_ok else "failed"

        if not dry_run:
            conn.execute(
                update_stmt,
                {
                    "batch_id": batch_id,
                    "status": target_status,
                    "record_count": row_count,
                    "records_fetched": row_count,
                    "records_inserted": row_count,
                    "records_skipped_duplicates": 0,
                    "repair_audit_at": datetime.now(timezone.utc).isoformat(),
                    "repair_previous_status": original_status,
                    "repair_reason": "batch metadata repair from detection table state",
                    "repair_require_denoiser": bool(require_denoiser),
                    "repair_scoring_incomplete_rows": int(scoring_incomplete),
                    "repair_denoiser_incomplete_rows": int(denoiser_incomplete),
                },
            )

    return {
        "batch_id": batch_id,
        "original_status": original_status,
        "target_status": target_status,
        "row_count": row_count,
        "scoring_incomplete_rows": int(scoring_incomplete),
        "denoiser_incomplete_rows": int(denoiser_incomplete),
        "quality_ok": bool(quality_ok),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Repair fire_detections data integrity issues.")
    parser.add_argument(
        "--batch-id",
        type=int,
        default=12,
        help="Batch id to repair metadata for (default: 12).",
    )
    parser.add_argument(
        "--synthetic-source-like",
        type=str,
        default="mvt_%",
        help="Pattern for synthetic detections to purge (default: mvt_%%).",
    )
    parser.add_argument(
        "--stale-running-hours",
        type=float,
        default=2.0,
        help="Mark running batches older than this many hours as failed (default: 2).",
    )
    parser.add_argument(
        "--require-denoiser",
        action="store_true",
        help="Require denoiser fields when deciding whether repaired batch can be marked succeeded.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report intended mutations without applying them.",
    )
    args = parser.parse_args(argv)

    stale_failed = _fail_stale_running_batches(
        stale_after_hours=float(args.stale_running_hours),
        reason="stale running batch normalized during repair",
        dry_run=bool(args.dry_run),
    )
    purged_synthetic = _purge_synthetic_rows(args.synthetic_source_like, dry_run=bool(args.dry_run))
    thermal_candidates = _backfill_thermal_columns(dry_run=bool(args.dry_run))
    batch_repair = _repair_batch_metadata(
        batch_id=int(args.batch_id),
        require_denoiser=bool(args.require_denoiser),
        dry_run=bool(args.dry_run),
    )

    summary = {
        "as_of_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": bool(args.dry_run),
        "stale_running_batches_failed": stale_failed,
        "purged_synthetic_rows": purged_synthetic,
        "thermal_backfill_candidates": thermal_candidates,
        "batch_repair": batch_repair,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
