"""RQ worker tasks for JIT forecast pipeline."""
import logging
import os
import traceback
from uuid import UUID

from redis import Redis
from rq import Queue

from api.forecast import repo

logger = logging.getLogger(__name__)

REDIS_URL = f"redis://{os.getenv('REDIS_HOST', 'localhost')}:{os.getenv('REDIS_PORT', '6379')}"
redis_conn = Redis.from_url(REDIS_URL)
queue = Queue(connection=redis_conn)


def run_jit_forecast_pipeline(job_id: UUID, bbox: tuple[float, float, float, float], forecast_params: dict):
    """Execute JIT forecast pipeline: ingest terrain -> weather -> run forecast."""
    logger.info(f"JIT forecast pipeline started: job_id={job_id}, bbox={bbox}")
    
    try:
        repo.update_jit_job_status(job_id, "ingesting_terrain")
        logger.info(f"JIT job {job_id}: starting terrain ingestion")
        
        # TODO: Call ingest_terrain_for_bbox in T05
        terrain_result = None
        
        repo.update_jit_job_status(job_id, "ingesting_weather")
        logger.info(f"JIT job {job_id}: starting weather ingestion")
        
        # TODO: Call ingest_weather_for_bbox in T05
        weather_result = None
        
        repo.update_jit_job_status(job_id, "running_forecast")
        logger.info(f"JIT job {job_id}: starting forecast")
        
        # TODO: Call run_spread_forecast in T10
        forecast_result = None
        
        result = {
            "terrain_id": terrain_result,
            "weather_run_id": weather_result,
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
