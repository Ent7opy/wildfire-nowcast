"""Authoritative LFMC ingestion scaffold for ECMWF ecLand.

This script is a production-oriented scaffold for ingesting daily LFMC
reanalysis (target variable: `lfmc`) and registering runs in
`fuel_moisture_runs` with provider `ecmwf_ecland_lfmc`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import sqlalchemy as sa
import xarray as xr

from api.db import get_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("lfmc_ecland_ingest")

LFMC_PROVIDER = "ecmwf_ecland_lfmc"
LFMC_API_URL_ENV = "LFMC_ECLAND_API_URL"
LFMC_API_TOKEN_ENV = "LFMC_ECLAND_API_TOKEN"


def _parse_run_time(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    cleaned = value.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    parsed = datetime.fromisoformat(cleaned)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _require_api_url() -> str:
    api_url = str(os.getenv(LFMC_API_URL_ENV, "")).strip()
    if not api_url:
        raise RuntimeError(
            "STOP: We are missing an authoritative source for ECMWF ecLand LFMC API URL. "
            f"Set {LFMC_API_URL_ENV} and rerun."
        )
    return api_url.rstrip("/")


def _auth_headers() -> dict[str, str]:
    token = str(os.getenv(LFMC_API_TOKEN_ENV, "")).strip()
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def _create_run_record(
    *,
    run_time: datetime,
    bbox: tuple[float, float, float, float],
    storage_path: str,
) -> int:
    stmt = sa.text(
        """
        INSERT INTO fuel_moisture_runs (
            run_time,
            bbox_min_lon,
            bbox_min_lat,
            bbox_max_lon,
            bbox_max_lat,
            status,
            storage_path,
            provider
        )
        VALUES (
            :run_time,
            :min_lon,
            :min_lat,
            :max_lon,
            :max_lat,
            'running',
            :storage_path,
            :provider
        )
        RETURNING id
        """
    )
    with get_engine().begin() as conn:
        row = conn.execute(
            stmt,
            {
                "run_time": run_time,
                "min_lon": float(bbox[0]),
                "min_lat": float(bbox[1]),
                "max_lon": float(bbox[2]),
                "max_lat": float(bbox[3]),
                "storage_path": storage_path,
                "provider": LFMC_PROVIDER,
            },
        ).mappings().first()
    if row is None or row.get("id") is None:
        raise RuntimeError("Failed to create fuel_moisture_runs row for LFMC ingestion.")
    return int(row["id"])


def _update_run_record_remote_job_id(
    *,
    run_id: int,
    remote_job_id: str,
) -> None:
    """Update the remote_job_id field in fuel_moisture_runs."""
    stmt = sa.text(
        """
        UPDATE fuel_moisture_runs
        SET remote_job_id = :remote_job_id
        WHERE id = :run_id
        """
    )
    with get_engine().begin() as conn:
        conn.execute(
            stmt,
            {
                "run_id": int(run_id),
                "remote_job_id": str(remote_job_id),
            },
        )


def _finalize_run_record(
    *,
    run_id: int,
    status: str,
    storage_path: str,
    coverage_fraction: float | None = None,
) -> None:
    stmt = sa.text(
        """
        UPDATE fuel_moisture_runs
        SET status = :status, storage_path = :storage_path, coverage_fraction = :coverage_fraction
        WHERE id = :run_id
        """
    )
    with get_engine().begin() as conn:
        conn.execute(
            stmt,
            {
                "run_id": int(run_id),
                "status": str(status),
                "storage_path": str(storage_path),
                "coverage_fraction": coverage_fraction,
            },
        )


def _cancel_job(
    *,
    client: httpx.Client,
    api_url: str,
    job_id: str,
) -> None:
    """Best-effort cancel of remote LFMC job. Logs but does not raise on failure."""
    try:
        # Try DELETE first
        response = client.delete(f"{api_url}/jobs/{job_id}", headers=_auth_headers())
        response.raise_for_status()
        LOGGER.info("Cancelled LFMC ecLand job %s via DELETE", job_id)
    except Exception as delete_err:
        LOGGER.debug("DELETE failed for job %s, trying POST cancel: %s", job_id, delete_err)
        try:
            # Fallback to POST cancel endpoint
            response = client.post(f"{api_url}/jobs/{job_id}/cancel", headers=_auth_headers())
            response.raise_for_status()
            LOGGER.info("Cancelled LFMC ecLand job %s via POST /cancel", job_id)
        except Exception as post_err:
            LOGGER.warning(
                "Failed to cancel LFMC ecLand job %s (DELETE: %s, POST /cancel: %s)",
                job_id,
                delete_err,
                post_err,
            )


def _submit_job(
    *,
    client: httpx.Client,
    api_url: str,
    run_time: datetime,
    bbox: tuple[float, float, float, float],
) -> str:
    payload = {
        "product": "lfmc",
        "provider": "ecmwf_ecland",
        "run_time": run_time.isoformat(),
        "bbox": {
            "min_lon": float(bbox[0]),
            "min_lat": float(bbox[1]),
            "max_lon": float(bbox[2]),
            "max_lat": float(bbox[3]),
        },
        "format": "netcdf",
    }
    response = client.post(f"{api_url}/jobs", json=payload, headers=_auth_headers())
    response.raise_for_status()
    body = response.json()
    job_id = body.get("job_id") or body.get("id")
    if not job_id:
        raise RuntimeError("LFMC ecLand API response missing job_id.")
    return str(job_id)


def _poll_job_until_ready(
    *,
    client: httpx.Client,
    api_url: str,
    job_id: str,
    poll_seconds: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    deadline = time.time() + int(timeout_seconds)
    while True:
        response = client.get(f"{api_url}/jobs/{job_id}", headers=_auth_headers())
        response.raise_for_status()
        body = response.json()
        status = str(body.get("status", "")).strip().lower()
        if status in {"completed", "succeeded", "success", "done"}:
            return body
        if status in {"failed", "error", "cancelled"}:
            raise RuntimeError(f"LFMC ecLand job failed (job_id={job_id}, status={status}): {body}")
        if time.time() >= deadline:
            raise TimeoutError(f"LFMC ecLand job timed out (job_id={job_id}, status={status})")
        LOGGER.info("Polling LFMC ecLand job %s status=%s", job_id, status or "unknown")
        time.sleep(max(1, int(poll_seconds)))


def _download_result(
    *,
    client: httpx.Client,
    body: dict[str, Any],
    target_path: Path,
) -> None:
    download_url = body.get("download_url") or body.get("result_url") or body.get("asset_url")
    if not download_url:
        raise RuntimeError("LFMC ecLand job response missing download URL.")
    with client.stream("GET", str(download_url), headers=_auth_headers()) as response:
        response.raise_for_status()
        with target_path.open("wb") as fh:
            for chunk in response.iter_bytes():
                fh.write(chunk)


def _check_orphaned_jobs(
    *,
    client: httpx.Client,
    api_url: str,
    timeout_seconds: int = 1800,
) -> None:
    """Query for running jobs older than timeout threshold and attempt cancellation.

    Reconciles orphaned remote jobs from previous timeout events.
    """
    stmt = sa.text(
        """
        SELECT id, remote_job_id
        FROM fuel_moisture_runs
        WHERE status = 'running'
          AND remote_job_id IS NOT NULL
          AND created_at < NOW() - INTERVAL '1 second' * :timeout_seconds
        """
    )
    with get_engine().begin() as conn:
        rows = conn.execute(
            stmt,
            {"timeout_seconds": int(timeout_seconds)},
        ).mappings().fetchall()

    for row in rows:
        run_id = int(row["id"])
        job_id = str(row["remote_job_id"])
        LOGGER.info("Found orphaned LFMC job %s (run_id=%s), attempting cancellation", job_id, run_id)
        try:
            _cancel_job(client=client, api_url=api_url, job_id=job_id)
            _finalize_run_record(
                run_id=run_id,
                status="failed",
                storage_path="",
                coverage_fraction=None,
            )
            LOGGER.info("Marked orphaned run %s as failed after remote cancellation", run_id)
        except Exception as e:
            LOGGER.warning("Error processing orphaned job %s: %s", job_id, e)


def ingest_lfmc_ecland_for_bbox(
    *,
    bbox: tuple[float, float, float, float],
    run_time: datetime | None = None,
    output_dir: Path | None = None,
    poll_seconds: int = 300,
    timeout_seconds: int = 1800,
) -> dict[str, Any]:
    resolved_time = (run_time or datetime.now(timezone.utc)).astimezone(timezone.utc)
    api_url = _require_api_url()
    out_root = output_dir or (Path("data") / "fuels" / "lfmc_ecland")
    out_root.mkdir(parents=True, exist_ok=True)

    out_name = f"lfmc_ecland_{resolved_time:%Y%m%dT%HZ}_bbox_{bbox[0]:.4f}_{bbox[1]:.4f}_{bbox[2]:.4f}_{bbox[3]:.4f}.nc"
    out_path = out_root / out_name
    run_id = _create_run_record(run_time=resolved_time, bbox=bbox, storage_path=f"pending://{out_name}")
    coverage_fraction: float | None = None
    job_id: str | None = None

    try:
        with httpx.Client(timeout=120.0) as client:
            # Check for orphaned jobs from previous timeout events before starting new work
            try:
                _check_orphaned_jobs(
                    client=client,
                    api_url=api_url,
                    timeout_seconds=int(timeout_seconds),
                )
            except Exception as e:
                LOGGER.warning("Error checking for orphaned jobs: %s", e)

            job_id = _submit_job(client=client, api_url=api_url, run_time=resolved_time, bbox=bbox)
            # Persist the remote job_id in case of timeout
            _update_run_record_remote_job_id(run_id=run_id, remote_job_id=job_id)

            body = _poll_job_until_ready(
                client=client,
                api_url=api_url,
                job_id=job_id,
                poll_seconds=int(poll_seconds),
                timeout_seconds=int(timeout_seconds),
            )
            _download_result(client=client, body=body, target_path=out_path)

        ds = xr.open_dataset(out_path)
        try:
            if "lfmc" not in ds.data_vars:
                raise RuntimeError("Downloaded LFMC file does not contain variable 'lfmc'.")
            arr = ds["lfmc"]
            total = int(arr.size)
            valid = int(arr.notnull().sum())
            coverage_fraction = float(valid) / float(total) if total > 0 else None
        finally:
            ds.close()
        _finalize_run_record(
            run_id=run_id,
            status="completed",
            storage_path=str(out_path),
            coverage_fraction=coverage_fraction,
        )
    except TimeoutError as timeout_err:
        # On timeout, attempt to cancel the remote job before marking as failed
        LOGGER.error("LFMC ingestion timed out for run_id=%s, attempting remote job cancellation", run_id)
        if job_id:
            try:
                with httpx.Client(timeout=30.0) as cancel_client:
                    _cancel_job(client=cancel_client, api_url=api_url, job_id=job_id)
            except Exception as cancel_err:
                LOGGER.warning("Error cancelling remote job %s: %s", job_id, cancel_err)
        _finalize_run_record(run_id=run_id, status="failed", storage_path=str(out_path), coverage_fraction=None)
        raise
    except Exception as e:
        LOGGER.error("LFMC ingestion failed for run_id=%s: %s", run_id, e)
        _finalize_run_record(run_id=run_id, status="failed", storage_path=str(out_path), coverage_fraction=None)
        raise

    result = {
        "run_id": int(run_id),
        "provider": LFMC_PROVIDER,
        "storage_path": str(out_path),
        "bbox": [float(v) for v in bbox],
        "run_time": resolved_time.isoformat(),
    }
    LOGGER.info("LFMC ecLand ingestion completed: %s", json.dumps(result))
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest LFMC from ECMWF ecLand.")
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        required=True,
        help="AOI bbox in WGS84.",
    )
    parser.add_argument("--run-time", type=str, default=None, help="ISO8601 reference time (default: current UTC hour).")
    parser.add_argument("--output-dir", type=Path, default=Path("data/fuels/lfmc_ecland"))
    parser.add_argument("--poll-seconds", type=int, default=300, help="Job polling interval in seconds.")
    parser.add_argument("--timeout-seconds", type=int, default=1800, help="Max end-to-end job wait time (seconds).")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = ingest_lfmc_ecland_for_bbox(
        bbox=tuple(float(v) for v in args.bbox),
        run_time=_parse_run_time(args.run_time),
        output_dir=args.output_dir,
        poll_seconds=int(args.poll_seconds),
        timeout_seconds=int(args.timeout_seconds),
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()

