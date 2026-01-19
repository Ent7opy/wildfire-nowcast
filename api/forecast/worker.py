"""RQ worker tasks for JIT forecast pipeline."""
import logging
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from redis import Redis
from rq import Queue

from api.forecast import repo

# Add ingest module to path for imports
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)

REDIS_URL = f"redis://{os.getenv('REDIS_HOST', 'localhost')}:{os.getenv('REDIS_PORT', '6379')}"
redis_conn = Redis.from_url(REDIS_URL)
queue = Queue(connection=redis_conn, default_timeout=120)


def handle_jit_pipeline_failure(job, connection, type, value, traceback):
    """RQ failure callback to update DB status when job fails or times out."""
    try:
        # Extract job_id from the first argument passed to run_jit_forecast_pipeline
        if job.args and len(job.args) > 0:
            job_id = job.args[0]
            error_msg = f"{type.__name__}: {str(value)}" if type else "Job failed"
            logger.error(f"RQ failure callback: job_id={job_id}, error={error_msg}")
            repo.update_jit_job_status(job_id, "failed", error=error_msg)
    except Exception as e:
        logger.error(f"Failed to update job status in failure callback: {e}")


def run_jit_forecast_pipeline(job_id: UUID, bbox: tuple[float, float, float, float], forecast_params: dict):
    """Execute JIT forecast pipeline: ingest terrain -> weather -> run forecast."""
    logger.info(f"JIT forecast pipeline started: job_id={job_id}, bbox={bbox}")
    
    try:
        from ingest.dem_preprocess import ingest_terrain_for_bbox
        from ingest.weather_ingest import ingest_weather_for_bbox
        
        repo.update_jit_job_status(job_id, "ingesting_terrain")
        logger.info(f"JIT job {job_id}: starting terrain ingestion")
        
        terrain_output_dir = REPO_ROOT / "data" / "terrain"
        terrain_output_dir.mkdir(parents=True, exist_ok=True)
        terrain_id = ingest_terrain_for_bbox(bbox, terrain_output_dir)
        logger.info(f"JIT job {job_id}: terrain ingestion completed, terrain_id={terrain_id}")
        
        repo.update_jit_job_status(job_id, "ingesting_weather")
        logger.info(f"JIT job {job_id}: starting weather ingestion")
        
        weather_output_dir = REPO_ROOT / "data" / "weather"
        weather_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Parse forecast_reference_time or use current time
        if forecast_params.get("forecast_reference_time"):
            forecast_time = datetime.fromisoformat(
                forecast_params["forecast_reference_time"].replace("Z", "+00:00")
            )
            if forecast_time.tzinfo is None:
                forecast_time = forecast_time.replace(tzinfo=timezone.utc)
        else:
            forecast_time = datetime.now(timezone.utc)
        
        horizons_hours = forecast_params.get("horizons_hours", [24, 48, 72])
        max_horizon = max(horizons_hours)
        
        weather_run_id = ingest_weather_for_bbox(
            bbox=bbox,
            forecast_time=forecast_time,
            output_dir=weather_output_dir,
            horizon_hours=max_horizon,
        )
        logger.info(f"JIT job {job_id}: weather ingestion completed, weather_run_id={weather_run_id}")
        
        repo.update_jit_job_status(job_id, "running_forecast")
        logger.info(f"JIT job {job_id}: starting forecast")
        
        # TODO: Call run_spread_forecast in T10
        forecast_result = None
        
        result = {
            "terrain_id": terrain_id,
            "weather_run_id": weather_run_id,
            "forecast_run_id": forecast_result
        }
        
        repo.update_jit_job_status(job_id, "completed", result=result)
        logger.info(f"JIT forecast pipeline completed: job_id={job_id}")
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(
            f"JIT forecast pipeline failed: job_id={job_id}, error={error_msg}\n{traceback.format_exc()}"
        )
        repo.update_jit_job_status(job_id, "failed", error=error_msg)
