"""Copernicus Global Drought Observatory (GDO) drought index ingest.

Downloads the Combined Drought Indicator (CDI) or Soil Moisture Anomaly (SMA)
raster from the Copernicus Climate Data Store (CDS), clips it to a configured
bbox, and writes a row to ``drought_index_runs`` so downstream feature
extraction can locate the latest valid raster.

Data source
-----------
Product: Copernicus GDO — Soil Moisture Anomaly (SMA) or CDI composite
Resolution: ~5 km (0.04°)
Cadence: Weekly (typically published Monday for the prior week)
API: Copernicus Climate Data Store (CDS) — same mechanism as lfmc_ecland_ingest
Docs: https://cds.climate.copernicus.eu/

Environment
-----------
CDSAPI_KEY        — ``<uid>:<api_key>`` token from the Copernicus CDS portal
                    (https://cds.climate.copernicus.eu/how-to-api).
                    If absent the job logs a WARNING and skips — the orchestrator
                    continues normally.
DROUGHT_DATASET   — CDS dataset name (default: derived-drought-index-obs)
DROUGHT_VARIABLE  — Variable within the dataset (default: soil_moisture_anomaly)
DROUGHT_OUTPUT_DIR — Local raster cache dir (default: data/drought/gdo)

Staleness
---------
A drought index older than 10 days emits a WARNING with a science_grade
mitigation note.  The ignition probability model can still run; the ML
model weights this signal accordingly.  A BLOCKER is never raised — drought
is a slow-moving background signal.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import sqlalchemy as sa

from api.db import get_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("drought_ingest")

DROUGHT_PROVIDER = "copernicus_gdo"
_CDSAPI_KEY_ENV = "CDSAPI_KEY"
_DATASET_ENV = "DROUGHT_DATASET"
_VARIABLE_ENV = "DROUGHT_VARIABLE"
_OUTPUT_DIR_ENV = "DROUGHT_OUTPUT_DIR"

_DEFAULT_DATASET = "derived-drought-index-obs"
_DEFAULT_VARIABLE = "soil_moisture_anomaly"
_DEFAULT_OUTPUT_DIR = Path("data/drought/gdo")

# A drought index older than this is considered stale (WARNING, not BLOCKER).
_STALE_WARNING_DAYS = 10


def _cdsapi_key() -> str | None:
    """Return the CDS API key, or None if not configured."""
    key = str(os.getenv(_CDSAPI_KEY_ENV, "")).strip()
    return key if key else None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _latest_completed_ingest(provider: str = DROUGHT_PROVIDER) -> dict[str, Any] | None:
    """Return the most recent completed drought ingest row, or None."""
    stmt = sa.text(
        """
        SELECT id, valid_time, storage_path, created_at
        FROM drought_index_runs
        WHERE provider = :provider
          AND status = 'completed'
        ORDER BY valid_time DESC
        LIMIT 1
        """
    )
    with get_engine().connect() as conn:
        row = conn.execute(stmt, {"provider": provider}).mappings().first()
    if row is None:
        return None
    return dict(row)


def _already_ingested(valid_time: datetime, provider: str = DROUGHT_PROVIDER) -> bool:
    """Return True if a completed run already exists for this valid_time."""
    stmt = sa.text(
        """
        SELECT 1 FROM drought_index_runs
        WHERE provider = :provider
          AND status = 'completed'
          AND valid_time = :valid_time
        LIMIT 1
        """
    )
    with get_engine().connect() as conn:
        row = conn.execute(stmt, {"provider": provider, "valid_time": valid_time}).first()
    return row is not None


def _create_run_record(
    *,
    valid_time: datetime,
    bbox: tuple[float, float, float, float],
    storage_path: str,
    variable: str,
) -> int:
    stmt = sa.text(
        """
        INSERT INTO drought_index_runs (
            valid_time,
            bbox_min_lon,
            bbox_min_lat,
            bbox_max_lon,
            bbox_max_lat,
            status,
            storage_path,
            provider,
            variable
        )
        VALUES (
            :valid_time,
            :min_lon,
            :min_lat,
            :max_lon,
            :max_lat,
            'running',
            :storage_path,
            :provider,
            :variable
        )
        RETURNING id
        """
    )
    with get_engine().begin() as conn:
        row = conn.execute(
            stmt,
            {
                "valid_time": valid_time,
                "min_lon": float(bbox[0]),
                "min_lat": float(bbox[1]),
                "max_lon": float(bbox[2]),
                "max_lat": float(bbox[3]),
                "storage_path": storage_path,
                "provider": DROUGHT_PROVIDER,
                "variable": variable,
            },
        ).mappings().first()
    if row is None or row.get("id") is None:
        raise RuntimeError("Failed to create drought_index_runs row.")
    return int(row["id"])


def _finalize_run_record(
    *,
    run_id: int,
    status: str,
    storage_path: str,
    coverage_fraction: float | None = None,
) -> None:
    stmt = sa.text(
        """
        UPDATE drought_index_runs
        SET status = :status,
            storage_path = :storage_path,
            coverage_fraction = :coverage_fraction
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


def _check_staleness(latest: dict[str, Any] | None) -> None:
    """Emit a WARNING if the most recent completed ingest is older than the threshold."""
    if latest is None:
        LOGGER.warning(
            "WARNING: No completed drought index ingest found in drought_index_runs. "
            "Ignition probability model will run without a drought signal. "
            "Mitigation: run drought ingest successfully at least once; "
            "target: science_grade auto-alerting pipeline."
        )
        return

    valid_time = latest.get("valid_time")
    if valid_time is None:
        return

    if not isinstance(valid_time, datetime):
        return

    age = _utc_now() - valid_time.astimezone(timezone.utc)
    if age > timedelta(days=_STALE_WARNING_DAYS):
        LOGGER.warning(
            "WARNING: Drought index is stale (valid_time=%s, age=%.1f days > threshold=%d days). "
            "Ignition probability model will use degraded drought signal. "
            "Mitigation: ensure CDSAPI_KEY is set and CDS API is reachable; "
            "target: science_grade freshness alerting.",
            valid_time.isoformat(),
            age.total_seconds() / 86400,
            _STALE_WARNING_DAYS,
        )


def _resolve_valid_time(dataset_response: dict[str, Any], fallback: datetime) -> datetime:
    """Extract valid_time from a CDS API dataset response, or fall back."""
    for key in ("valid_time", "data_time", "date", "time"):
        raw = dataset_response.get(key)
        if raw:
            try:
                if isinstance(raw, str):
                    return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc)
                if isinstance(raw, (int, float)):
                    return datetime.fromtimestamp(float(raw), tz=timezone.utc)
            except (ValueError, OSError):
                continue
    return fallback


def _compute_coverage(nc_path: Path, variable: str) -> float | None:
    """Return the fraction of non-null cells in the raster, or None on error."""
    try:
        import xarray as xr

        ds = xr.open_dataset(nc_path)
        try:
            candidates = [v for v in ds.data_vars if variable.lower() in v.lower()] or list(ds.data_vars)
            if not candidates:
                return None
            arr = ds[candidates[0]]
            total = int(arr.size)
            valid = int(arr.notnull().sum())
            return float(valid) / float(total) if total > 0 else None
        finally:
            ds.close()
    except Exception as exc:
        LOGGER.debug("Coverage computation skipped: %s", exc)
        return None


def _fetch_via_cdsapi(
    *,
    api_key: str,
    dataset: str,
    variable: str,
    bbox: tuple[float, float, float, float],
    output_path: Path,
) -> dict[str, Any]:
    """Submit a CDS API request and download the result to output_path.

    Returns the metadata dict from the CDS client result object so the caller
    can extract valid_time.  Uses the ``cdsapi`` Python package (must be
    installed in the environment).
    """
    import cdsapi  # type: ignore[import]

    # CDS area order: North / West / South / East
    area = [
        float(bbox[3]),  # North (max_lat)
        float(bbox[0]),  # West (min_lon)
        float(bbox[1]),  # South (min_lat)
        float(bbox[2]),  # East (max_lon)
    ]

    # Construct the URL from key for the CDS client.
    # The cdsapi.Client accepts a ``key`` kwarg directly.
    client = cdsapi.Client(
        url="https://cds.climate.copernicus.eu/api",
        key=api_key,
        quiet=True,
        wait_until_complete=True,
        delete=True,
    )

    result = client.retrieve(
        dataset,
        {
            "variable": variable,
            "format": "netcdf",
            "area": area,
        },
        str(output_path),
    )

    meta: dict[str, Any] = {}
    try:
        meta = dict(result.__dict__) if hasattr(result, "__dict__") else {}
    except Exception:
        pass
    return meta


def ingest_drought_index(
    *,
    bbox: tuple[float, float, float, float] | None = None,
    output_dir: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Fetch the latest Copernicus GDO drought index raster and store it.

    Parameters
    ----------
    bbox:
        WGS84 bounding box (min_lon, min_lat, max_lon, max_lat).
        Defaults to global coverage (-180, -90, 180, 90).
    output_dir:
        Local directory for raster files.
        Defaults to ``data/drought/gdo``.
    force:
        Re-ingest even if a completed record for the same valid_time exists.

    Returns a dict with run metadata.
    Raises RuntimeError on hard failures.
    Logs a WARNING and returns a skip dict if CDSAPI_KEY is absent.
    """
    api_key = _cdsapi_key()
    if api_key is None:
        LOGGER.warning(
            "WARNING: CDSAPI_KEY is not set. Drought index ingest skipped. "
            "Register at https://cds.climate.copernicus.eu/how-to-api and set CDSAPI_KEY. "
            "Mitigation: set the key and rerun; target: science_grade automated ingest."
        )
        return {"skipped": True, "reason": "CDSAPI_KEY not configured"}

    dataset = str(os.getenv(_DATASET_ENV, _DEFAULT_DATASET)).strip() or _DEFAULT_DATASET
    variable = str(os.getenv(_VARIABLE_ENV, _DEFAULT_VARIABLE)).strip() or _DEFAULT_VARIABLE
    resolved_bbox: tuple[float, float, float, float] = bbox or (-180.0, -90.0, 180.0, 90.0)
    out_root = output_dir or (Path(os.getenv(_OUTPUT_DIR_ENV, "")) or _DEFAULT_OUTPUT_DIR)
    out_root.mkdir(parents=True, exist_ok=True)

    latest = _latest_completed_ingest()
    _check_staleness(latest)

    run_ts = _utc_now()
    tmp_name = f"drought_{dataset}_{variable}_{run_ts:%Y%m%dT%HZ}.nc.tmp"
    tmp_path = out_root / tmp_name
    run_id: int | None = None
    coverage_fraction: float | None = None

    try:
        meta = _fetch_via_cdsapi(
            api_key=api_key,
            dataset=dataset,
            variable=variable,
            bbox=resolved_bbox,
            output_path=tmp_path,
        )

        valid_time = _resolve_valid_time(meta, fallback=run_ts)

        # Idempotency: skip if a completed record already exists for this valid_time.
        if not force and _already_ingested(valid_time):
            tmp_path.unlink(missing_ok=True)
            LOGGER.info(
                "Drought ingest skipped (already ingested valid_time=%s)", valid_time.isoformat()
            )
            return {
                "skipped": True,
                "reason": "already_ingested",
                "valid_time": valid_time.isoformat(),
            }

        final_name = (
            f"drought_{dataset}_{variable}_{valid_time:%Y%m%dT%HZ}_"
            f"bbox_{resolved_bbox[0]:.2f}_{resolved_bbox[1]:.2f}_"
            f"{resolved_bbox[2]:.2f}_{resolved_bbox[3]:.2f}.nc"
        )
        final_path = out_root / final_name
        tmp_path.rename(final_path)

        coverage_fraction = _compute_coverage(final_path, variable)

        run_id = _create_run_record(
            valid_time=valid_time,
            bbox=resolved_bbox,
            storage_path=str(final_path),
            variable=variable,
        )
        _finalize_run_record(
            run_id=run_id,
            status="completed",
            storage_path=str(final_path),
            coverage_fraction=coverage_fraction,
        )

    except Exception:
        if run_id is not None:
            _finalize_run_record(
                run_id=run_id,
                status="failed",
                storage_path=str(tmp_path),
                coverage_fraction=None,
            )
        tmp_path.unlink(missing_ok=True)
        raise

    result = {
        "run_id": int(run_id),
        "provider": DROUGHT_PROVIDER,
        "dataset": dataset,
        "variable": variable,
        "storage_path": str(final_path),
        "bbox": [float(v) for v in resolved_bbox],
        "valid_time": valid_time.isoformat(),
        "coverage_fraction": coverage_fraction,
    }
    LOGGER.info(
        "Drought index ingest completed: source=%s variable=%s valid_time=%s path=%s coverage=%.2f",
        dataset,
        variable,
        valid_time.isoformat(),
        final_path,
        coverage_fraction or 0.0,
    )
    return result


def run_drought_ingest() -> int:
    """Orchestrator-compatible entry point. Returns 0 on success, 1 on failure."""
    try:
        result = ingest_drought_index()
        if result.get("skipped"):
            # A missing key or duplicate is not a pipeline failure.
            return 0
        return 0
    except Exception as exc:
        LOGGER.error("Drought ingest failed: %s", exc)
        return 1
