"""Database cleanup utility with 14-day retention policy."""

import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import text

# Add project root to sys.path so 'api' can be imported when running as a script
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from api.db import get_engine, SessionLocal

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RETENTION_DAYS = 14

TABLES_TO_CLEAN = [
    ("fire_detections", "acq_time"),
    ("weather_runs", "run_time"),
    ("spread_forecast_runs", "forecast_reference_time"),
]

def cleanup():
    """Delete records older than RETENTION_DAYS and vacuum tables."""
    cutoff_time = datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)
    logger.info(f"Cleaning records older than {cutoff_time} ({RETENTION_DAYS} days retention)")

    engine = get_engine()
    
    with SessionLocal() as session:
        try:
            for table_name, time_col in TABLES_TO_CLEAN:
                logger.info(f"Deleting old records from {table_name}...")
                stmt = text(f"DELETE FROM {table_name} WHERE {time_col} < :cutoff")
                result = session.execute(stmt, {"cutoff": cutoff_time})
                logger.info(f"Deleted {result.rowcount} rows from {table_name}.")
            
            session.commit()
            logger.info("Deletions committed successfully.")
        except Exception as e:
            session.rollback()
            logger.error(f"Error during deletion: {e}")
            return

    # VACUUM ANALYZE must be run outside of a transaction block
    logger.info("Reclaiming disk space with VACUUM ANALYZE...")
    try:
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            for table_name, _ in TABLES_TO_CLEAN:
                logger.info(f"Vacuuming {table_name}...")
                conn.execute(text(f"VACUUM ANALYZE {table_name}"))
        logger.info("VACUUM ANALYZE complete.")
    except Exception as e:
        logger.error(f"Error during VACUUM ANALYZE: {e}")

if __name__ == "__main__":
    cleanup()
