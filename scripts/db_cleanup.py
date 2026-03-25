"""Database cleanup utility with configurable per-table retention policies."""

import argparse
import logging
import os
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
DEFAULT_RETENTION_DAYS = int(os.environ.get("RETENTION_DAYS", "14"))

# Archive-ingested data has a shorter TTL (default 3 days).
DEFAULT_ARCHIVE_RETENTION_DAYS = int(os.environ.get("ARCHIVE_RETENTION_DAYS", "3"))

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
    # NOTE: fire_detections is handled separately by _cleanup_fire_detections()
    # (two-tier retention: archive vs NRT).  It must run first to respect CASCADE.
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


def _count_eligible(
    session, table: str, time_col: str, cutoff: datetime, *, extra_where: str = ""
) -> int:
    """Return the number of rows that would be deleted."""
    row = session.execute(
        text(f"SELECT COUNT(*) FROM {table} WHERE {time_col} < :cutoff{extra_where}"),
        {"cutoff": cutoff},
    ).scalar()
    return row or 0


def _delete_batch(
    session, table: str, time_col: str, cutoff: datetime, *, extra_where: str = ""
) -> int:
    """Delete one batch of rows. Returns actual deleted count."""
    result = session.execute(
        text(
            f"DELETE FROM {table}"
            f" WHERE id IN ("
            f"   SELECT id FROM {table} WHERE {time_col} < :cutoff{extra_where} LIMIT :batch_size"
            f")"
        ),
        {"cutoff": cutoff, "batch_size": DELETE_BATCH_SIZE},
    )
    return result.rowcount


def _delete_batched(
    session, table: str, time_col: str, cutoff: datetime, *, extra_where: str = ""
) -> int:
    """Delete all eligible rows in batches. Returns total deleted count."""
    total = 0
    while True:
        try:
            deleted = _delete_batch(session, table, time_col, cutoff, extra_where=extra_where)
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


def _query_known_forecast_storage_paths() -> set[str]:
    """Return the set of storage_path values currently in spread_forecast_rasters."""
    stmt = text("SELECT storage_path FROM spread_forecast_rasters")
    with get_engine().connect() as conn:
        rows = conn.execute(stmt).fetchall()
    return {row[0] for row in rows}


def find_orphan_forecast_files(
    forecasts_dir: Path, repo_root: Path, known_paths: set[str]
) -> list[Path]:
    """Return .tif files under forecasts_dir whose repo-relative path is not in known_paths.

    Args:
        forecasts_dir: Absolute path to the forecasts directory (e.g. REPO_ROOT/data/forecasts).
        repo_root: Repository root used to compute repo-relative paths (must be an ancestor of
            forecasts_dir).
        known_paths: Set of repo-relative storage_path values from spread_forecast_rasters.

    Returns:
        List of absolute Paths for orphaned .tif files.
    """
    if not forecasts_dir.exists():
        return []

    orphans = []
    for tif in forecasts_dir.rglob("*.tif"):
        try:
            rel = str(tif.relative_to(repo_root))
        except ValueError:
            # Should not happen, but skip rather than crash
            logger.warning("Could not relativize path %s against %s", tif, repo_root)
            continue
        if rel not in known_paths:
            orphans.append(tif)
    return orphans


def purge_orphan_forecast_files(repo_root: Path, *, dry_run: bool) -> int:
    """Delete raster files under data/forecasts/ that have no DB row in spread_forecast_rasters.

    Also removes empty run directories left behind after file deletion.

    Returns:
        Number of orphaned files found (and deleted when not dry_run).
    """
    forecasts_dir = repo_root / "data" / "forecasts"
    if not forecasts_dir.exists():
        logger.info("data/forecasts/ does not exist — nothing to clean.")
        return 0

    known_paths = _query_known_forecast_storage_paths()
    orphans = find_orphan_forecast_files(forecasts_dir, repo_root, known_paths)

    if not orphans:
        logger.info("No orphaned forecast raster files found.")
        return 0

    action = "Would remove" if dry_run else "Removing"
    for path in orphans:
        logger.info("%s orphaned raster: %s", action, path)
        if not dry_run:
            path.unlink(missing_ok=True)

    if not dry_run:
        # Clean up empty run directories (run_{id}/ dirs left after file removal)
        for run_dir in forecasts_dir.rglob("run_*"):
            if run_dir.is_dir():
                try:
                    run_dir.rmdir()  # Only succeeds if directory is empty
                    logger.info("Removed empty run directory: %s", run_dir)
                except OSError:
                    pass  # Directory still has files — leave it

    logger.info("%s %d orphaned forecast raster file(s).", action, len(orphans))
    return len(orphans)


def _cleanup_fire_detections(
    session,
    now: datetime,
    *,
    dry_run: bool,
    prefix: str,
) -> int:
    """Delete fire_detections with two-tier retention: archive (short) then NRT (standard).

    Archive rows (is_archive=true) are deleted first with ARCHIVE_RETENTION_DAYS,
    then NRT rows (is_archive=false) with DEFAULT_RETENTION_DAYS.  Both use batched
    deletes.  Must run before other fire-pipeline tables to respect CASCADE ordering.

    Returns total number of deleted rows (0 when dry_run).
    """
    # extra_where fragments are hardcoded literals — never from external input.
    tiers: list[tuple[str, int, str]] = [
        ("archive", DEFAULT_ARCHIVE_RETENTION_DAYS, " AND is_archive = true"),
        ("NRT",     DEFAULT_RETENTION_DAYS,         " AND is_archive = false"),
    ]

    total_deleted = 0
    for tier_label, retention_days, extra_where in tiers:
        cutoff = now - timedelta(days=retention_days)
        label = _cutoff_label(retention_days, cutoff)

        if dry_run:
            planned = _count_eligible(
                session, "fire_detections", "acq_time", cutoff, extra_where=extra_where
            )
            logger.info(
                f"{prefix}fire_detections ({tier_label}): {planned} rows would be deleted"
                f" (cutoff={label})"
            )
            continue

        deleted = _delete_batched(
            session, "fire_detections", "acq_time", cutoff, extra_where=extra_where
        )
        logger.info(
            f"fire_detections ({tier_label}): deleted {deleted} rows (cutoff={label})"
        )
        total_deleted += deleted

    return total_deleted


def cleanup(dry_run: bool = False) -> None:
    """Delete records beyond per-table retention windows, purge orphaned raster files, then VACUUM."""
    now = datetime.now(timezone.utc)
    engine = get_engine()
    prefix = "[DRY RUN] " if dry_run else ""
    logger.info(f"{prefix}Starting cleanup — wall clock {now.isoformat()}")

    vacuumed_tables: list[str] = []

    with SessionLocal() as session:
        # fire_detections first (CASCADE ordering) with two-tier archive/NRT retention.
        fd_deleted = _cleanup_fire_detections(session, now, dry_run=dry_run, prefix=prefix)
        if fd_deleted > 0:
            vacuumed_tables.append("fire_detections")

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

    # Clean up orphaned raster files on disk (CASCADE already handled DB child rows).
    # Runs in both live and dry-run modes — purge_orphan_forecast_files respects dry_run.
    logger.info("%sPurging orphaned spread forecast raster files …", prefix)
    purge_orphan_forecast_files(REPO_ROOT, dry_run=dry_run)

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
