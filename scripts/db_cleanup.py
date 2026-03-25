"""Database cleanup utility with configurable per-table retention policies."""

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import text

# Add project root to sys.path so 'api' can be imported when running as a script
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from api.db import get_engine, SessionLocal  # noqa: E402

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Default retention for most time-series tables (days).
DEFAULT_RETENTION_DAYS = 14

# Per-table cleanup config: (table_name, time_column, retention_days).
#
# retention_days=0 means "use now() as cutoff" — appropriate for the
# reverse_geocode_cache where expires_at already encodes expiry intent.
#
# Ordering is intentional to respect CASCADE foreign keys:
#   fire_detections deleted first → CASCADEs into fire_event_memberships
#   and denoiser_labels_v2 (for non-null fire_detection_id rows).
#   Subsequent entries for those tables clean up any orphaned rows
#   (e.g. denoiser_labels_v2 rows with null fire_detection_id).
TABLE_CONFIG: list[tuple[str, str, int]] = [
    # ---- fire detection pipeline (CASCADE ordering) -------------------------
    ("fire_detections",        "acq_time",                DEFAULT_RETENTION_DAYS),
    ("denoiser_labels_v2",     "labeled_at",              DEFAULT_RETENTION_DAYS),
    ("fire_event_memberships", "linked_at",               DEFAULT_RETENTION_DAYS),
    ("fire_events",            "start_time",              DEFAULT_RETENTION_DAYS),
    ("fire_fronts",            "created_at",              DEFAULT_RETENTION_DAYS),
    # ---- ML drift metrics --------------------------------------------------
    ("denoiser_drift_metrics", "created_at",              DEFAULT_RETENTION_DAYS),
    # ---- weather / forecast (existing) -------------------------------------
    ("weather_runs",           "run_time",                DEFAULT_RETENTION_DAYS),
    ("spread_forecast_runs",   "forecast_reference_time", DEFAULT_RETENTION_DAYS),
    # ---- operational tables with extended retention ------------------------
    ("ingest_batches",         "created_at",              30),
    ("export_jobs",            "created_at",              7),
    # ---- cache: 0 days → cutoff = now() (remove any row past expires_at) --
    # NOTE: migration uses 'expires_at'; sprint doc said 'cached_at' which
    # does not exist in the actual schema (20260312_add_reverse_geocode_cache).
    ("reverse_geocode_cache",  "expires_at",              0),
]

# Batch size for tables with potentially large row counts.
# Smaller batches reduce lock hold time and replication lag.
DELETE_BATCH_SIZE = 10_000

# Tables whose delete path uses batched DELETEs.
BATCHED_TABLES = {"fire_detections"}


def _count_eligible(session, table: str, time_col: str, cutoff: datetime) -> int:
    """Return the number of rows that would be deleted."""
    row = session.execute(
        text(f"SELECT COUNT(*) FROM {table} WHERE {time_col} < :cutoff"),
        {"cutoff": cutoff},
    ).scalar()
    return row or 0


def _delete_batch(session, table: str, time_col: str, cutoff: datetime) -> int:
    """Delete one batch of rows. Returns actual deleted count."""
    result = session.execute(
        text(
            f"DELETE FROM {table}"
            f" WHERE id IN ("
            f"   SELECT id FROM {table} WHERE {time_col} < :cutoff LIMIT :batch_size"
            f")"
        ),
        {"cutoff": cutoff, "batch_size": DELETE_BATCH_SIZE},
    )
    return result.rowcount


def _delete_batched(session, table: str, time_col: str, cutoff: datetime) -> int:
    """Delete all eligible rows in batches. Returns total deleted count."""
    total = 0
    while True:
        try:
            deleted = _delete_batch(session, table, time_col, cutoff)
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.error(f"Batch delete failed for {table} after {total} rows: {exc}")
            break
        total += deleted
        if deleted < DELETE_BATCH_SIZE:
            break
        logger.info(f"  {table}: batch done, {total} rows deleted so far …")
    return total


def _delete_all(session, table: str, time_col: str, cutoff: datetime) -> int:
    """Delete all eligible rows in a single statement. Returns deleted count."""
    result = session.execute(
        text(f"DELETE FROM {table} WHERE {time_col} < :cutoff"),
        {"cutoff": cutoff},
    )
    return result.rowcount


def _cutoff_label(retention_days: int, cutoff: datetime) -> str:
    if retention_days == 0:
        return "now() [expired entries]"
    return f"{cutoff.date()} ({retention_days}d)"


def cleanup(dry_run: bool = False) -> None:
    """Delete records beyond per-table retention windows, then VACUUM."""
    now = datetime.now(timezone.utc)
    engine = get_engine()
    prefix = "[DRY RUN] " if dry_run else ""
    logger.info(f"{prefix}Starting cleanup — wall clock {now.isoformat()}")

    vacuumed_tables: list[str] = []

    with SessionLocal() as session:
        for table, time_col, retention_days in TABLE_CONFIG:
            cutoff = now - timedelta(days=retention_days)
            label = _cutoff_label(retention_days, cutoff)

            if dry_run:
                planned = _count_eligible(session, table, time_col, cutoff)
                logger.info(
                    f"{prefix}{table}: {planned} rows would be deleted"
                    f" (col={time_col}, cutoff={label})"
                )
                continue

            # Live delete path
            if table in BATCHED_TABLES:
                deleted = _delete_batched(session, table, time_col, cutoff)
            else:
                try:
                    deleted = _delete_all(session, table, time_col, cutoff)
                    session.commit()
                except Exception as exc:
                    session.rollback()
                    logger.error(f"Delete failed for {table}: {exc}")
                    continue

            logger.info(
                f"{table}: deleted {deleted} rows"
                f" (col={time_col}, cutoff={label})"
            )
            if deleted > 0:
                vacuumed_tables.append(table)

    if dry_run:
        logger.info(f"{prefix}Dry-run complete — no rows were mutated.")
        return

    if not vacuumed_tables:
        logger.info("No rows deleted; skipping VACUUM.")
        return

    # VACUUM ANALYZE must run outside a transaction block (requires AUTOCOMMIT).
    logger.info("Reclaiming disk space with VACUUM ANALYZE …")
    try:
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            for table in vacuumed_tables:
                logger.info(f"Vacuuming {table} …")
                conn.execute(text(f"VACUUM ANALYZE {table}"))
        logger.info("VACUUM ANALYZE complete.")
    except Exception as exc:
        logger.error(f"Error during VACUUM ANALYZE: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove time-series rows beyond per-table retention windows."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print per-table row counts that would be deleted; do not mutate data.",
    )
    args = parser.parse_args()
    cleanup(dry_run=args.dry_run)
