"""Database cleanup utility with 14-day retention policy."""

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

RETENTION_DAYS = 14

TABLES_TO_CLEAN = [
    ("fire_detections", "acq_time"),
    ("weather_runs", "run_time"),
    ("spread_forecast_runs", "forecast_reference_time"),
]


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


def cleanup(*, dry_run: bool = False) -> None:
    """Delete records older than RETENTION_DAYS, vacuum tables, and remove orphaned raster files."""
    cutoff_time = datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)
    mode = "[DRY RUN] " if dry_run else ""
    logger.info(
        "%sCleaning records older than %s (%d days retention)",
        mode,
        cutoff_time,
        RETENTION_DAYS,
    )

    with SessionLocal() as session:
        try:
            for table_name, time_col in TABLES_TO_CLEAN:
                count_stmt = text(
                    f"SELECT COUNT(*) FROM {table_name} WHERE {time_col} < :cutoff"
                )
                count = int(
                    session.execute(count_stmt, {"cutoff": cutoff_time}).scalar_one() or 0
                )
                if dry_run:
                    logger.info("[DRY RUN] Would delete %d rows from %s.", count, table_name)
                else:
                    logger.info("Deleting old records from %s...", table_name)
                    stmt = text(f"DELETE FROM {table_name} WHERE {time_col} < :cutoff")
                    result = session.execute(stmt, {"cutoff": cutoff_time})
                    logger.info("Deleted %d rows from %s.", result.rowcount, table_name)

            if not dry_run:
                session.commit()
                logger.info("Deletions committed successfully.")
        except Exception as e:
            session.rollback()
            logger.error("Error during deletion: %s", e)
            return

    # Clean up orphaned raster files on disk (CASCADE already handled DB child rows)
    logger.info("%sPurging orphaned spread forecast raster files...", mode)
    purge_orphan_forecast_files(REPO_ROOT, dry_run=dry_run)

    if not dry_run:
        # VACUUM ANALYZE must be run outside of a transaction block
        logger.info("Reclaiming disk space with VACUUM ANALYZE...")
        try:
            with get_engine().connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
                for table_name, _ in TABLES_TO_CLEAN:
                    logger.info("Vacuuming %s...", table_name)
                    conn.execute(text(f"VACUUM ANALYZE {table_name}"))
            logger.info("VACUUM ANALYZE complete.")
        except Exception as e:
            logger.error("Error during VACUUM ANALYZE: %s", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DB cleanup with raster file hygiene.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be deleted without making any changes.",
    )
    args = parser.parse_args()
    cleanup(dry_run=args.dry_run)
