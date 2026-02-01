"""RQ worker tasks for JIT forecast pipeline."""
import logging
import os
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import quote_plus
from uuid import UUID

from redis import Redis
from redis.lock import Lock as RedisLock
from rq import Queue

from api.config import settings
from api.forecast import repo

# Add ingest module to path for imports
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)

REDIS_URL = f"redis://{os.getenv('REDIS_HOST', 'localhost')}:{os.getenv('REDIS_PORT', '6379')}"
redis_conn = Redis.from_url(REDIS_URL)
queue = Queue(connection=redis_conn, default_timeout=120)

# Lock timeout in seconds for cache operations
CACHE_LOCK_TIMEOUT = 300  # 5 minutes


def _acquire_cache_lock(lock_key: str, timeout: int = CACHE_LOCK_TIMEOUT) -> Optional[RedisLock]:
    """Acquire a distributed lock for cache operations.
    
    Args:
        lock_key: Unique key for the lock
        timeout: Lock timeout in seconds
        
    Returns:
        RedisLock if acquired, None otherwise
    """
    lock = RedisLock(redis_conn, lock_key, timeout=timeout, blocking_timeout=5)
    if lock.acquire():
        return lock
    return None


def _cleanup_run_artifacts(run_dir: Path) -> None:
    """Clean up intermediate artifacts on forecast failure.
    
    Args:
        run_dir: Directory containing forecast artifacts
    """
    if run_dir.exists() and run_dir.is_dir():
        try:
            shutil.rmtree(run_dir)
            logger.info(f"Cleaned up forecast artifacts at {run_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up forecast artifacts at {run_dir}: {e}")


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
    
    run_dir: Optional[Path] = None
    run_id: Optional[int] = None

    try:
        from ingest.dem_preprocess import ingest_terrain_for_bbox
        from ingest.weather_ingest import ingest_weather_for_bbox

        # Check for cached terrain with distributed lock to prevent race conditions
        terrain_lock_key = f"jit:terrain:lock:{bbox[0]}:{bbox[1]}:{bbox[2]}:{bbox[3]}"
        cached_terrain = repo.find_cached_terrain(bbox)
        
        if cached_terrain:
            terrain_id = cached_terrain["id"]
            logger.info(f"JIT job {job_id}: cache hit for terrain, terrain_id={terrain_id}")
        else:
            # Acquire lock to prevent duplicate terrain ingestion
            terrain_lock = _acquire_cache_lock(terrain_lock_key)
            if terrain_lock is None:
                logger.warning(f"JIT job {job_id}: could not acquire terrain lock, checking cache again")
                # Another worker may have ingested terrain, check cache again
                cached_terrain = repo.find_cached_terrain(bbox)
                if cached_terrain:
                    terrain_id = cached_terrain["id"]
                    logger.info(f"JIT job {job_id}: cache hit for terrain after lock retry, terrain_id={terrain_id}")
                else:
                    raise RuntimeError("Could not acquire terrain lock and no cached terrain found")
            else:
                try:
                    # Double-check cache after acquiring lock
                    cached_terrain = repo.find_cached_terrain(bbox)
                    if cached_terrain:
                        terrain_id = cached_terrain["id"]
                        logger.info(f"JIT job {job_id}: cache hit for terrain after lock, terrain_id={terrain_id}")
                    else:
                        repo.update_jit_job_status(job_id, "ingesting_terrain")
                        logger.info(f"JIT job {job_id}: starting terrain ingestion")

                        terrain_output_dir = REPO_ROOT / "data" / "terrain"
                        terrain_output_dir.mkdir(parents=True, exist_ok=True)
                        terrain_id = ingest_terrain_for_bbox(bbox, terrain_output_dir)
                        logger.info(f"JIT job {job_id}: terrain ingestion completed, terrain_id={terrain_id}")
                finally:
                    terrain_lock.release()

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

        # Check for cached weather (within 6 hour freshness window) with distributed lock
        weather_lock_key = f"jit:weather:lock:{bbox[0]}:{bbox[1]}:{bbox[2]}:{bbox[3]}:{forecast_time.isoformat()}"
        cached_weather = repo.find_cached_weather(bbox, freshness_hours=6, required_horizon_hours=max_horizon)
        
        if cached_weather:
            weather_run_id = cached_weather["id"]
            logger.info(f"JIT job {job_id}: cache hit for weather, weather_run_id={weather_run_id}")
        else:
            # Acquire lock to prevent duplicate weather ingestion
            weather_lock = _acquire_cache_lock(weather_lock_key)
            if weather_lock is None:
                logger.warning(f"JIT job {job_id}: could not acquire weather lock, checking cache again")
                # Another worker may have ingested weather, check cache again
                cached_weather = repo.find_cached_weather(bbox, freshness_hours=6, required_horizon_hours=max_horizon)
                if cached_weather:
                    weather_run_id = cached_weather["id"]
                    logger.info(f"JIT job {job_id}: cache hit for weather after lock retry, weather_run_id={weather_run_id}")
                else:
                    raise RuntimeError("Could not acquire weather lock and no cached weather found")
            else:
                try:
                    # Double-check cache after acquiring lock
                    cached_weather = repo.find_cached_weather(bbox, freshness_hours=6, required_horizon_hours=max_horizon)
                    if cached_weather:
                        weather_run_id = cached_weather["id"]
                        logger.info(f"JIT job {job_id}: cache hit for weather after lock, weather_run_id={weather_run_id}")
                    else:
                        repo.update_jit_job_status(job_id, "ingesting_weather")
                        logger.info(f"JIT job {job_id}: starting weather ingestion")

                        weather_output_dir = REPO_ROOT / "data" / "weather"
                        weather_output_dir.mkdir(parents=True, exist_ok=True)

                        weather_run_id = ingest_weather_for_bbox(
                            bbox=bbox,
                            forecast_time=forecast_time,
                            output_dir=weather_output_dir,
                            horizon_hours=max_horizon,
                        )
                        logger.info(f"JIT job {job_id}: weather ingestion completed, weather_run_id={weather_run_id}")
                finally:
                    weather_lock.release()

        repo.update_jit_job_status(job_id, "running_forecast")
        logger.info(f"JIT job {job_id}: starting forecast")

        # Run spread forecast and persist products
        from ml.spread.service import SpreadForecastRequest, run_spread_forecast
        from ingest.spread_forecast import save_forecast_rasters, build_contour_records
        from ingest.spread_repository import (
            create_spread_forecast_run,
            finalize_spread_forecast_run,
            insert_spread_forecast_contours,
            insert_spread_forecast_rasters,
        )
        from api.core.grid import GridSpec, get_grid_window_for_bbox

        # Create forecast run record
        run_id = create_spread_forecast_run(
            region_name=forecast_params.get("region_name", "location-based"),
            model_name="HeuristicSpreadModelV0",
            model_version="v0",
            forecast_reference_time=forecast_time,
            bbox=bbox,
        )
        logger.info(f"JIT job {job_id}: created forecast run_id={run_id}")

        try:
            # Build and execute forecast request
            request = SpreadForecastRequest(
                region_name=forecast_params.get("region_name"),
                bbox=bbox,
                forecast_reference_time=forecast_time,
                horizons_hours=horizons_hours,
            )
            forecast = run_spread_forecast(request)
            logger.info(f"JIT job {job_id}: forecast computation completed")

            # Capture operational metadata
            extra_meta = {}
            try:
                attrs = dict(getattr(forecast.probabilities, "attrs", {}) or {})
                for k in (
                    "weather_bias_corrected",
                    "weather_bias_corrector_path",
                    "calibration_applied",
                    "calibration_source",
                    "calibration_run_id",
                    "calibration_run_dir",
                ):
                    if k in attrs:
                        extra_meta[k] = attrs.get(k)
            except Exception:
                pass

            # Derive grid and window for persistence
            if forecast_params.get("region_name"):
                from api.fires.service import get_region_grid_spec
                grid = get_region_grid_spec(forecast_params["region_name"])
            else:
                grid = GridSpec.from_bbox(bbox)
            window = get_grid_window_for_bbox(grid, bbox, clip=True)

            # Save rasters
            region_dir_name = forecast_params.get("region_name", "location-based")
            run_dir = REPO_ROOT / "data" / "forecasts" / region_dir_name / f"run_{run_id}"
            raster_records = save_forecast_rasters(forecast, grid, window, run_dir, emit_cog=True)
            insert_spread_forecast_rasters(run_id, raster_records)
            logger.info(f"JIT job {job_id}: saved {len(raster_records)} rasters")

            # Generate and persist contours
            thresholds = forecast_params.get("thresholds", [0.3, 0.5, 0.7])
            contour_records = build_contour_records(
                forecast=forecast, grid=grid, window=window, thresholds=thresholds
            )
            insert_spread_forecast_contours(run_id, contour_records)
            logger.info(f"JIT job {job_id}: saved {len(contour_records)} contours")

            # Finalize forecast run
            finalize_spread_forecast_run(run_id, status="completed", extra_metadata=extra_meta)

            # Build result with TileJSON URLs for UI consumption
            tilejson_urls = []
            for r in raster_records:
                storage_path = str(r["storage_path"])
                titiler_path = storage_path.replace(
                    settings.data_dir_local_prefix, settings.data_dir_titiler_mount
                )
                encoded_path = quote_plus(titiler_path)
                tilejson_url = (
                    f"{settings.titiler_public_base_url}/cog/WebMercatorQuad/tilejson.json?url={encoded_path}"
                )
                tilejson_urls.append(tilejson_url)

            result = {
                "terrain_id": terrain_id,
                "weather_run_id": weather_run_id,
                "forecast_run_id": run_id,
                "tilejson_urls": tilejson_urls,
            }

            repo.update_jit_job_status(job_id, "completed", result=result)
            logger.info(f"JIT forecast pipeline completed: job_id={job_id}, run_id={run_id}")

        except Exception as forecast_error:
            # Mark forecast run as failed and clean up artifacts
            if run_id is not None:
                finalize_spread_forecast_run(run_id, status="failed", extra_metadata={"error": str(forecast_error)})
            # Clean up intermediate artifacts
            if run_dir is not None:
                _cleanup_run_artifacts(run_dir)
            raise

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(
            f"JIT forecast pipeline failed: job_id={job_id}, error={error_msg}\n{traceback.format_exc()}"
        )
        repo.update_jit_job_status(job_id, "failed", error=error_msg)
        # Clean up any artifacts if they were created
        if run_dir is not None:
            _cleanup_run_artifacts(run_dir)
