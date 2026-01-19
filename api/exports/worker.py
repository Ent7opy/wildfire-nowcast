"""RQ worker configuration and job definitions."""
import csv
import logging
import os
import traceback
from datetime import datetime, timezone
from redis import Redis
from rq import Queue

from api.config import settings

logger = logging.getLogger(__name__)

REDIS_URL = f"redis://{os.getenv('REDIS_HOST', 'localhost')}:{os.getenv('REDIS_PORT', '6379')}"
redis_conn = Redis.from_url(REDIS_URL)
queue = Queue(connection=redis_conn)

def export_task(job_id, kind, request):
    """Execute an export job."""
    # Import here to avoid loading DB/app stack at module level if not needed,
    # though worker process usually preloads.
    from api.exports import repo
    
    logger.info(f"Export job started: job_id={job_id}, kind={kind}")
    
    try:
        repo.update_job_status(job_id, "running")
        
        if kind == "fires_csv":
            result = _export_fires_csv(job_id, request)
        else:
            raise ValueError(f"Unsupported export kind: {kind}")
        
        repo.update_job_status(job_id, "succeeded", result=result)
        logger.info(f"Export job succeeded: job_id={job_id}, file_path={result.get('file_path')}, row_count={result.get('row_count')}")
    except Exception as e:
        logger.error(f"Export job failed: job_id={job_id}, kind={kind}, error={str(e)}\n{traceback.format_exc()}")
        repo.update_job_status(job_id, "failed", error=str(e))

def _export_fires_csv(job_id, request):
    """Export fire detections as CSV."""
    from api.fires import repo as fires_repo
    
    # Parse request
    min_lon = request.get("min_lon")
    min_lat = request.get("min_lat")
    max_lon = request.get("max_lon")
    max_lat = request.get("max_lat")
    start_time_str = request.get("start_time")
    end_time_str = request.get("end_time")
    limit = request.get("limit", 10000)
    
    # Validate required fields
    if None in (min_lon, min_lat, max_lon, max_lat, start_time_str, end_time_str):
        raise ValueError("Missing required fields: min_lon, min_lat, max_lon, max_lat, start_time, end_time")
    
    # Parse timestamps
    dt_start = datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))
    dt_end = datetime.fromisoformat(end_time_str.replace("Z", "+00:00"))
    if dt_start.tzinfo is None:
        dt_start = dt_start.replace(tzinfo=timezone.utc)
    if dt_end.tzinfo is None:
        dt_end = dt_end.replace(tzinfo=timezone.utc)
    
    # Validate bbox and time range
    if not (min_lon < max_lon and min_lat < max_lat):
        raise ValueError("Invalid bbox: max must be greater than min")
    if dt_start >= dt_end:
        raise ValueError("Invalid time range: end_time must be after start_time")
    
    # Fetch fire detections
    detections = fires_repo.list_fire_detections_bbox_time(
        bbox=(min_lon, min_lat, max_lon, max_lat),
        start_time=dt_start,
        end_time=dt_end,
        limit=limit,
        columns=["id", "lat", "lon", "acq_time", "confidence", "frp", "sensor", "source"]
    )
    
    # Write to file
    export_dir = settings.exports_dir / str(job_id)
    export_dir.mkdir(parents=True, exist_ok=True)
    file_path = export_dir / "fires.csv"
    
    fieldnames = ["id", "lat", "lon", "acq_time", "confidence", "frp", "sensor", "source"]
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if detections:
            writer.writerows(detections)
    
    return {
        "download_url": f"/exports/{job_id}/download",
        "file_path": str(file_path),
        "row_count": len(detections)
    }
