"""RQ worker tasks for JIT forecast pipeline."""
import logging
import json
import os
import re
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import quote_plus
from uuid import UUID

from redis.lock import Lock as RedisLock
from rq import Queue, Retry

from api.cache import get_redis
from api.config import settings
from api.constants import DEFAULT_HORIZONS_HOURS
from api.notifications import notify
from api.forecast import repo
from api.forecast.cache_lock import acquire_forecast_result_lock, release_forecast_result_lock
from api.forecast.model_catalog import resolve_request_model_selection
from ml.spread.factory import get_model_version_hint, get_spread_model
from ml.spread.region_key import bbox_region_name
from ml.spread.service import (
    SPREAD_SHADOW_ENABLED_ENV,
    STRICT_FORECAST_INPUTS_ENV,
)

# Add ingest module to path for imports
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)

queue = Queue(connection=get_redis(), default_timeout=120)
failed_forecast_queue = Queue("failed_forecast", connection=get_redis())

# Lock timeout in seconds for cache operations
CACHE_LOCK_TIMEOUT = 300  # 5 minutes

_DEFAULT_FORECAST_JOB_MAX_RETRIES = 3
_DEFAULT_FORECAST_JOB_RETRY_INTERVALS = [10, 30, 60]

_intervals_raw = os.getenv("FORECAST_JOB_RETRY_INTERVALS", "")
_FORECAST_RETRY = Retry(
    max=int(os.getenv("FORECAST_JOB_MAX_RETRIES", str(_DEFAULT_FORECAST_JOB_MAX_RETRIES))),
    interval=[int(x.strip()) for x in _intervals_raw.split(",") if x.strip()]
    if _intervals_raw.strip()
    else list(_DEFAULT_FORECAST_JOB_RETRY_INTERVALS),
)
del _intervals_raw


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _default_forecast_reference_time() -> datetime:
    return datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)


def _parse_iso8601_datetime(value: str) -> datetime:
    if not value or not value.strip():
        raise ValueError("Empty datetime string")

    value = value.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"

    tz_pattern = r"([+-])(\d{2})(\d{2})$"
    match = re.search(tz_pattern, value)
    if match and ":00" not in value[-6:]:
        sign, hours, minutes = match.groups()
        value = value[:match.start()] + f"{sign}{hours}:{minutes}"

    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as e:
        raise ValueError(f"Invalid ISO 8601 datetime format: {value}") from e

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return parsed


def _normalize_horizons(horizons_hours: list[int] | None) -> list[int]:
    if horizons_hours is None:
        return list(DEFAULT_HORIZONS_HOURS)
    if len(horizons_hours) == 0:
        raise ValueError("horizons_hours must not be empty.")
    normalized = [int(h) for h in horizons_hours]
    if any(h <= 0 for h in normalized):
        raise ValueError("horizons_hours must contain only positive integers.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("horizons_hours must not contain duplicates.")
    return normalized


def _acquire_cache_lock(lock_key: str, timeout: int = CACHE_LOCK_TIMEOUT) -> Optional[RedisLock]:
    """Acquire a distributed lock for cache operations.
    
    Args:
        lock_key: Unique key for the lock
        timeout: Lock timeout in seconds
        
    Returns:
        RedisLock if acquired, None otherwise
    """
    lock = RedisLock(get_redis(), lock_key, timeout=timeout, blocking_timeout=5)
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


def move_to_dead_letter(job, connection, type, value, traceback):
    """RQ on_failure callback: update DB status, park job in dead-letter queue, and alert ops."""
    handle_jit_pipeline_failure(job, connection, type, value, traceback)

    try:
        failed_forecast_queue.enqueue_job(job)
    except Exception as e:
        logger.error("Failed to move forecast job %s to dead-letter queue: %s", job.id, e)

    error_name = type.__name__ if type else "Unknown"
    error_msg = str(value) if value else ""
    notify(
        event_type="forecast_job_failed",
        title="Forecast job exhausted retries",
        body=f"Forecast job {job.id} failed after all retries: {error_name}: {error_msg}",
        severity="critical",
        job_id=str(job.id),
        error_type=error_name,
    )


def run_jit_forecast_pipeline(job_id: UUID, bbox: tuple[float, float, float, float], forecast_params: dict):
    """Execute JIT forecast pipeline: ingest terrain -> weather -> run forecast."""
    logger.info(f"JIT forecast pipeline started: job_id={job_id}, bbox={bbox}")
    
    run_dir: Optional[Path] = None
    run_id: Optional[int] = None
    result_lock: Optional[RedisLock] = None

    try:
        # Parse request-level settings first (used by cache key and strict behavior).
        if forecast_params.get("forecast_reference_time"):
            forecast_time = _parse_iso8601_datetime(forecast_params["forecast_reference_time"])
        else:
            forecast_time = _default_forecast_reference_time()

        horizons_hours = _normalize_horizons(forecast_params.get("horizons_hours"))
        thresholds = [float(t) for t in forecast_params.get("thresholds", [0.3, 0.5, 0.7])]
        region_name = forecast_params.get("region_name")
        effective_region_name = str(region_name) if region_name else bbox_region_name(bbox)
        strict_inputs = bool(
            forecast_params.get("strict_inputs")
            if forecast_params.get("strict_inputs") is not None
            else _env_bool(STRICT_FORECAST_INPUTS_ENV, default=False)
        )
        use_result_cache = bool(forecast_params.get("use_result_cache", True))
        model_name, model_params, selected_model_id = resolve_request_model_selection(
            model_id=forecast_params.get("model_id"),
            model_name=forecast_params.get("model_name"),
            model_params=forecast_params.get("model_params"),
        )
        shadow_enabled = _env_bool(SPREAD_SHADOW_ENABLED_ENV, default=False)
        shadow_model_name = None
        shadow_model_params = None
        shadow_model_id = os.getenv("SPREAD_SHADOW_MODEL_ID")
        if shadow_enabled and shadow_model_id:
            try:
                shadow_model_name, shadow_model_params, _ = resolve_request_model_selection(
                    model_id=shadow_model_id,
                    model_name=None,
                    model_params=None,
                )
            except Exception:
                logger.exception("Failed to resolve shadow model_id=%s. Continuing without shadow.", shadow_model_id)
                shadow_model_name = None
                shadow_model_params = None

        cache_key = repo.build_forecast_result_cache_key(
            bbox=bbox,
            forecast_reference_time=forecast_time,
            horizons_hours=horizons_hours,
            region_name=effective_region_name,
            model_id=selected_model_id,
            model_name=model_name,
            model_params=model_params,
            strict_inputs=strict_inputs,
            thresholds=thresholds,
        )

        if use_result_cache:
            result_lock = acquire_forecast_result_lock(cache_key)

        # Exact-result cache can bypass the full pipeline.
        if use_result_cache:
            cached_run = repo.find_cached_forecast_run(
                cache_key=cache_key,
                freshness_minutes=settings.forecast_result_cache_ttl_minutes,
            )
        else:
            cached_run = None

        if cached_run:
            cached_run_id = int(cached_run["id"])
            rasters = repo.list_rasters_for_run(cached_run_id)
            contours = repo.list_contours_for_run(cached_run_id)
            tilejson_urls = []
            for r in rasters:
                storage_path = str(r["storage_path"])
                titiler_path = storage_path.replace(
                    settings.data_dir_local_prefix, settings.data_dir_titiler_mount
                )
                encoded_path = quote_plus(titiler_path)
                tilejson_url = (
                    f"{settings.titiler_public_base_url}/cog/WebMercatorQuad/tilejson.json?url={encoded_path}"
                )
                tilejson_urls.append(tilejson_url)

            repo.update_jit_job_status(
                job_id,
                "completed",
                result={
                    "terrain_id": None,
                    "weather_run_id": None,
                    "forecast_run_id": cached_run_id,
                    "run_id": cached_run_id,
                    "tilejson_urls": tilejson_urls,
                    "contours": {
                        "type": "FeatureCollection",
                        "features": [
                            {
                                "type": "Feature",
                                "geometry": json.loads(c["geom_geojson"]),
                                "properties": {
                                    "horizon_hours": c["horizon_hours"],
                                    "threshold": c["threshold"],
                                },
                            }
                            for c in contours
                        ],
                    },
                    "cache_hit": True,
                    "cache_source": "forecast_result",
                },
            )
            logger.info(
                "JIT forecast pipeline cache hit: job_id=%s run_id=%s",
                job_id,
                cached_run_id,
            )
            return

        from ingest.dem_preprocess import ingest_terrain_for_bbox

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
                        terrain_id = ingest_terrain_for_bbox(
                            bbox,
                            terrain_output_dir,
                            region_name=effective_region_name,
                        )
                        logger.info(f"JIT job {job_id}: terrain ingestion completed, terrain_id={terrain_id}")
                finally:
                    terrain_lock.release()
        # Weather data is sourced from the background weather_point_cache
        # (populated by the orchestrator's weather job).  The spread model's
        # _load_weather_cube queries the cache directly — no JIT download needed.
        # If the cache has no data for this region the spread model falls back
        # to calm-conditions weather automatically.

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
        from api.core.grid import get_grid_window_for_bbox

        model = get_spread_model(model_name, model_params)
        model_version = get_model_version_hint(model_name)

        # Create forecast run record
        run_id = create_spread_forecast_run(
            region_name=effective_region_name,
            model_name=model_name,
            model_version=model_version,
            forecast_reference_time=forecast_time,
            bbox=bbox,
            metadata={
                "model_id": selected_model_id,
                "model_params": model_params,
                "region_name": region_name,
                "effective_region_name": effective_region_name,
                "strict_inputs": strict_inputs,
                "cache_key": cache_key,
                "use_result_cache": use_result_cache,
                "shadow_enabled": bool(shadow_model_name),
                "shadow_model_id": shadow_model_id if shadow_model_name else None,
            },
        )
        logger.info(f"JIT job {job_id}: created forecast run_id={run_id}")

        try:
            # Build and execute forecast request
            request = SpreadForecastRequest(
                region_name=effective_region_name,
                bbox=bbox,
                forecast_reference_time=forecast_time,
                horizons_hours=horizons_hours,
                strict_inputs=strict_inputs,
                model_name=model_name,
                model_params=model_params,
                shadow_model_name=shadow_model_name,
                shadow_model_params=shadow_model_params,
            )
            forecast = run_spread_forecast(request, model=model)
            logger.info(f"JIT job {job_id}: forecast computation completed")

            # Compute max spread probability across all horizons and grid cells.
            try:
                max_spread_prob: float | None = float(forecast.probabilities.max())
            except Exception:
                max_spread_prob = None

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
                    "weather_fallback_used",
                    "weather_fallback_reason",
                    "terrain_fallback_used",
                    "confidence_level",
                    "staleness_hours",
                    "fallback_used",
                    "sanity_fallback_used",
                    "sanity_fallback_reason",
                    "sanity_original_model_name",
                    "sanity_served_model_name",
                    "mvp_guardrail_triggered",
                    "mvp_guardrail_mode",
                    "mvp_guardrail_reason",
                    "mvp_guardrail_metrics",
                    "shadow_evaluated",
                    "shadow_metrics_summary",
                    "model_name",
                    "model_version",
                ):
                    if k in attrs:
                        extra_meta[k] = attrs.get(k)
            except Exception:
                pass

            # Derive grid and window for persistence
            from api.fires.service import get_region_grid_spec

            grid = get_region_grid_spec(effective_region_name)
            window = get_grid_window_for_bbox(grid, bbox, clip=True)

            # Save rasters
            run_dir = REPO_ROOT / "data" / "forecasts" / effective_region_name / f"run_{run_id}"
            raster_records = save_forecast_rasters(forecast, grid, window, run_dir, emit_cog=True)
            insert_spread_forecast_rasters(run_id, raster_records)
            logger.info(f"JIT job {job_id}: saved {len(raster_records)} rasters")

            # Generate and persist contours
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
                "weather_run_id": None,  # weather sourced from point cache, no single run_id
                "forecast_run_id": run_id,
                "run_id": run_id,
                "tilejson_urls": tilejson_urls,
                "contours": {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "geometry": json.loads(c["geom_geojson"]),
                            "properties": {
                                "horizon_hours": c["horizon_hours"],
                                "threshold": c["threshold"],
                            },
                        }
                        for c in contour_records
                    ],
                },
                "cache_hit": False,
                "cache_source": None,
                "confidence_level": extra_meta.get("confidence_level"),
                "staleness_hours": extra_meta.get("staleness_hours"),
                "fallback_used": bool(extra_meta.get("fallback_used", False)),
                "weather_bias_corrected": extra_meta.get("weather_bias_corrected"),
                "shadow_evaluated": bool(extra_meta.get("shadow_evaluated", False)),
                "shadow_metrics_summary": extra_meta.get("shadow_metrics_summary"),
                "max_spread_prob": max_spread_prob,
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
    finally:
        release_forecast_result_lock(result_lock)
