"""CLI entrypoint for NASA FIRMS ingestion."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from sqlalchemy.engine import Connection

from ingest import repository
from ingest.config import FIRMSIngestSettings, settings as ingest_settings
from ingest.firms_client import (
    FirmsValidationSummary,
    build_firms_url,
    redact_firms_url,
    fetch_csv_rows,
    parse_detection_rows,
)
from ingest.logging_utils import log_event

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("firms_ingest")
# httpx logs include full request URLs; avoid leaking FIRMS API keys.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

MAX_FIRMS_DAY_RANGE = 10
NRT_RETENTION_DAYS_HINT = 7
_DENOISER_V2_REQUIRED_COLUMNS: tuple[str, ...] = (
    "front_id",
    "event_id",
    "event_score",
    "denoiser_decision",
    "review_required",
)


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _area_key_from_bbox(bbox: str) -> str:
    """Build deterministic area key from normalized FIRMS bbox string."""
    parts = [float(part.strip()) for part in bbox.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Invalid FIRMS bbox: {bbox!r}. Expected 'w,s,e,n'.")
    return ",".join(f"{value:.6f}" for value in parts)


def _filter_detections_by_watermark(
    detections: list,
    *,
    watermark_time_utc: datetime | None,
    grace_minutes: int,
) -> tuple[list, datetime | None]:
    """Filter detections to incremental window with late-arrival grace."""
    if not detections:
        return [], None

    max_acq_time = max((_as_utc(d.acq_time) for d in detections), default=None)
    if watermark_time_utc is None:
        return detections, max_acq_time

    threshold = _as_utc(watermark_time_utc) - timedelta(minutes=max(0, int(grace_minutes)))
    filtered = [d for d in detections if (_as_utc(d.acq_time) or datetime.min.replace(tzinfo=timezone.utc)) > threshold]
    max_filtered = max((_as_utc(d.acq_time) for d in filtered), default=None)
    return filtered, max_filtered


def _resolve_denoiser_model_run_dir(config: "FIRMSIngestSettings") -> str | None:
    """Resolve denoiser model path from promoted registry model with env fallback."""
    try:
        from api.model_registry import resolve_active_model

        active = resolve_active_model("denoiser")
        if active and active.get("artifact_uri"):
            return str(active["artifact_uri"])
    except Exception:
        LOGGER.warning("Failed to resolve active promoted denoiser model; using env fallback if provided.")

    return config.denoiser_model_run_dir


def _resolve_denoiser_pipeline_version(config: "FIRMSIngestSettings") -> str:
    return str(getattr(config, "denoiser_pipeline_version", "v1") or "v1").strip().lower()


def _resolve_denoiser_module_name(config: "FIRMSIngestSettings") -> str:
    if _resolve_denoiser_pipeline_version(config) == "v2":
        return "ml.denoiser_inference_v2"
    return "ml.denoiser_inference"


def run_firms_ingest(
    day_range: Optional[int],
    area: Optional[str],
    sources: Optional[str],
) -> int:
    """Run the FIRMS ingestion pipeline."""
    config = ingest_settings

    # Validate FIRMS API key is configured
    if not config.map_key or config.map_key.strip() == "":
        LOGGER.error(
            "FIRMS_MAP_KEY environment variable is required but not set.\n"
            "  1. Get a free API key at: https://firms.modaps.eosdis.nasa.gov/api/\n"
            "  2. Copy .env.example to .env and add your key: FIRMS_MAP_KEY=your_key_here\n"
            "  3. Or set it directly: export FIRMS_MAP_KEY=your_key_here"
        )
        return 2
    bbox = _resolve_area(area) if area else config.resolved_area
    area_key = _area_key_from_bbox(bbox)
    effective_day_range = day_range if day_range is not None else config.day_range
    source_list = _resolve_sources(sources) or config.sources

    if not 1 <= int(effective_day_range) <= MAX_FIRMS_DAY_RANGE:
        LOGGER.error(
            "Invalid day_range=%s. FIRMS area CSV API supports 1-%s days (NRT sources are typically ~%s days).",
            effective_day_range,
            MAX_FIRMS_DAY_RANGE,
            NRT_RETENTION_DAYS_HINT,
        )
        return 2

    if effective_day_range > NRT_RETENTION_DAYS_HINT and any(
        str(s).upper().endswith("_NRT") for s in source_list
    ):
        LOGGER.warning(
            "Requested day_range=%s with NRT sources. FIRMS NRT feeds typically retain ~%s days; "
            "older ranges may return 0 rows. For historical training data, use non-NRT archive sources "
            "or an offline export flow.",
            effective_day_range,
            NRT_RETENTION_DAYS_HINT,
        )

    LOGGER.info(
        "Starting FIRMS ingestion",
        extra={
            "day_range": effective_day_range,
            "area": bbox,
            "sources": source_list,
        },
    )

    for source in source_list:
        watermark = repository.get_ingest_watermark(source, area_key)
        watermark_time_utc = _as_utc((watermark or {}).get("last_acq_time_utc"))
        grace_minutes = int(config.firms_watermark_grace_minutes)

        source_uri = build_firms_url(config.map_key, source, bbox, effective_day_range)
        batch_id = repository.create_ingest_batch(
            source,
            redact_firms_url(source_uri, config.map_key),
            bbox,
            effective_day_range,
            metadata_extra={
                "area_key": area_key,
                "watermark_before": watermark_time_utc.isoformat() if watermark_time_utc else None,
                "watermark_grace_minutes": grace_minutes,
            },
        )
        LOGGER.info("Created ingest batch %s for %s", batch_id, source)

        fetched_count = 0
        inserted = 0
        skipped_duplicates = 0
        rows_after_watermark_filter = 0
        watermark_advanced_to: datetime | None = None
        try:
            csv_rows = fetch_csv_rows(
                map_key=config.map_key,
                source=source,
                bbox=bbox,
                day_range=effective_day_range,
                timeout_seconds=config.request_timeout_seconds,
            )
            fetched_count = len(csv_rows)
            detections, validation = parse_detection_rows(csv_rows, source, batch_id)
            parsed_count = len(detections)
            filtered_detections, watermark_advanced_to = _filter_detections_by_watermark(
                detections,
                watermark_time_utc=watermark_time_utc,
                grace_minutes=grace_minutes,
            )
            rows_after_watermark_filter = len(filtered_detections)
            _log_firms_validation(source, batch_id, validation)
            if filtered_detections:
                # Keep insert+scoring in one DB transaction so partially scored rows
                # cannot be committed if the process is interrupted mid-run.
                with repository.get_engine().begin() as conn:
                    inserted = repository.insert_detections(filtered_detections, conn=conn)
                    skipped_duplicates = rows_after_watermark_filter - inserted

                    if inserted > 0:
                        _update_all_scoring_atomic(batch_id, conn=conn)
                        _assert_batch_scoring_complete(batch_id, conn=conn)
            else:
                inserted = 0
                skipped_duplicates = 0

            should_run_denoiser = inserted > 0 and (config.denoiser_enabled or config.denoiser_required)
            denoiser_ran = False
            if should_run_denoiser:
                denoiser_model_run_dir = _resolve_denoiser_model_run_dir(config)
                if not denoiser_model_run_dir:
                    if config.denoiser_required:
                        raise RuntimeError(
                            "Denoiser is required but no promoted denoiser model or "
                            "DENOISER_MODEL_RUN_DIR fallback is configured."
                        )
                    LOGGER.warning("Denoiser is enabled but no model run directory is configured; skipping inference.")
                else:
                    _run_denoiser_inference(batch_id, config, model_run_dir=denoiser_model_run_dir)
                    denoiser_ran = True
            if denoiser_ran or config.denoiser_required:
                _assert_batch_denoiser_complete(batch_id, config=config)

            repository.finalize_ingest_batch(
                batch_id,
                status="succeeded",
                fetched=fetched_count,
                inserted=inserted,
                skipped=max(skipped_duplicates, 0),
            )
            if watermark_advanced_to is not None:
                repository.advance_ingest_watermark(
                    source=source,
                    area_key=area_key,
                    last_acq_time_utc=watermark_advanced_to,
                    last_batch_id=batch_id,
                )
            log_event(
                LOGGER,
                "firms.watermark",
                "Applied FIRMS incremental watermark filter",
                source=source,
                batch_id=batch_id,
                area_key=area_key,
                watermark_before=watermark_time_utc.isoformat() if watermark_time_utc else None,
                rows_after_watermark_filter=rows_after_watermark_filter,
                watermark_advanced_to=watermark_advanced_to.isoformat() if watermark_advanced_to else None,
            )
            LOGGER.info(
                "Ingested source=%s batch=%s fetched=%s parsed=%s post_watermark=%s inserted=%s duplicates=%s",
                source,
                batch_id,
                fetched_count,
                parsed_count,
                rows_after_watermark_filter,
                inserted,
                skipped_duplicates,
            )
        except Exception:  # pragma: no cover - defensive logging
            LOGGER.exception("Ingest failed for source=%s batch=%s", source, batch_id)
            persisted_after_cleanup = 0
            try:
                persisted_before_cleanup = repository.count_detections_for_batch(batch_id)
                if persisted_before_cleanup > 0:
                    deleted = repository.delete_detections_for_batch(batch_id)
                    LOGGER.warning(
                        "Removed %s persisted detections for failed batch %s",
                        deleted,
                        batch_id,
                    )
                persisted_after_cleanup = repository.count_detections_for_batch(batch_id)
            except Exception:
                LOGGER.exception("Failed to cleanup detections for failed batch %s", batch_id)
            repository.finalize_ingest_batch(
                batch_id,
                status="failed",
                fetched=fetched_count,
                inserted=persisted_after_cleanup,
                skipped=max(rows_after_watermark_filter - persisted_after_cleanup, 0),
            )
            return 1

    if config.firms_reconcile_unscored_batches:
        _reconcile_unscored_batches(max_batches=int(config.firms_reconcile_max_batches))

    return 0


def _resolve_area(value: str) -> str:
    cleaned = value.strip()
    if cleaned.lower() == "world":
        return "-180,-90,180,90"
    return cleaned


def _resolve_sources(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    return [segment.strip() for segment in value.split(",") if segment.strip()]


def _update_all_scoring_atomic(
    batch_id: int,
    *,
    conn: Connection | None = None,
) -> None:
    """Update all scoring columns for detections in the batch atomically.
    
    This function wraps all scoring updates (false source masking, persistence,
    landcover, weather, and fire likelihood) in a single database transaction.
    This ensures atomicity - either all scores are updated or none are - and
    prevents connection pool exhaustion during batch processing.
    
    Addresses: INGEST-004 (Scoring Updates Not Atomic), CRIT-003 (Connection Pool Exhaustion)
    
    Args:
        batch_id: The ingest batch ID to process
    """
    try:
        from api.fires.repo import update_all_scoring_for_batch

        LOGGER.info("Updating all scoring for batch %s (atomic transaction)", batch_id)
        counts = update_all_scoring_for_batch(batch_id, conn=conn)
        LOGGER.info(
            "Batch %s scoring complete: masked=%s, persistence=%s, landcover=%s, weather=%s, likelihood=%s",
            batch_id,
            counts["masked_count"],
            counts["persistence_count"],
            counts["landcover_count"],
            counts["weather_count"],
            counts["likelihood_count"],
        )
    except Exception:
        LOGGER.exception("Failed to update scoring for batch %s", batch_id)
        raise  # Re-raise to ensure batch failure is recorded


def _assert_batch_scoring_complete(
    batch_id: int,
    *,
    conn: Connection | None = None,
) -> None:
    """Fail the batch if any required scoring column is still NULL."""
    remaining_incomplete = repository.count_rows_with_null_columns_for_batch(
        batch_id,
        columns=repository.REQUIRED_SCORING_COLUMNS,
        exclude_source_like="mvt_%",
        conn=conn,
    )
    if remaining_incomplete > 0:
        raise RuntimeError(
            f"Batch {batch_id} still has {remaining_incomplete} production rows with NULL scoring fields"
        )


def _assert_batch_denoiser_complete(
    batch_id: int,
    *,
    config: "FIRMSIngestSettings",
) -> None:
    """Fail the batch if denoiser inference left production rows unscored."""
    pipeline_version = _resolve_denoiser_pipeline_version(config)
    shadow_mode = bool(getattr(config, "denoiser_shadow_mode", False))
    if pipeline_version == "v2":
        required_columns = _DENOISER_V2_REQUIRED_COLUMNS
        # In shadow mode, keep legacy fields untouched while ensuring v2 writes are complete.
        if not shadow_mode:
            required_columns = required_columns + repository.REQUIRED_DENOISER_COLUMNS
    else:
        required_columns = repository.REQUIRED_DENOISER_COLUMNS

    remaining_incomplete = repository.count_rows_with_null_columns_for_batch(
        batch_id,
        columns=required_columns,
        exclude_source_like="mvt_%",
    )
    if remaining_incomplete > 0:
        raise RuntimeError(
            f"Batch {batch_id} still has {remaining_incomplete} production rows with NULL denoiser fields"
        )


def _reconcile_unscored_batches(max_batches: int = 5) -> None:
    """Best-effort repair for historical rows that still have NULL fire_likelihood."""
    candidate_batch_ids = repository.list_batches_with_unscored_likelihood(limit=max(1, max_batches))
    if not candidate_batch_ids:
        return

    LOGGER.info(
        "Reconciling %s batch(es) with NULL fire_likelihood: %s",
        len(candidate_batch_ids),
        candidate_batch_ids,
    )
    for batch_id in candidate_batch_ids:
        try:
            _update_all_scoring_atomic(batch_id)
        except Exception:
            LOGGER.exception("Failed to reconcile unscored batch %s", batch_id)


def _run_denoiser_inference(
    batch_id: int,
    config: "FIRMSIngestSettings",
    *,
    model_run_dir: str | None = None,
    ) -> None:
    """Trigger denoiser inference via subprocess or direct module call."""
    model_run_dir = model_run_dir or config.denoiser_model_run_dir
    if not model_run_dir:
        LOGGER.warning(
            "Denoiser is enabled but DENOISER_MODEL_RUN_DIR is not set. Skipping inference."
        )
        return

    pipeline_version = _resolve_denoiser_pipeline_version(config)
    invoke_method = str(getattr(config, "denoiser_invoke_method", "uv") or "uv").strip().lower()
    module_name = _resolve_denoiser_module_name(config)

    LOGGER.info(
        "Starting denoiser inference for batch %s (pipeline=%s, method=%s)",
        batch_id,
        pipeline_version,
        invoke_method,
    )

    # Use direct module import if configured
    if invoke_method == "module":
        _run_denoiser_module_direct(batch_id, config, model_run_dir=model_run_dir)
        return

    # Build command based on invocation method
    if invoke_method == "python":
        # Use Python directly - works in containerized environments without uv
        cmd = [
            sys.executable,
            "-m",
            module_name,
        ]
    else:
        # Default: uv run (original behavior)
        cmd = [
            "uv",
            "run",
            "--project",
            "ml",
            "-m",
            module_name,
        ]

    cmd.extend(_build_denoiser_argv(batch_id=batch_id, model_run_dir=model_run_dir, config=config))

    try:
        # We capture output to get the JSON summary
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )

        # The module prints JSON to stdout as its last line
        output = result.stdout.strip()
        last_line = output.splitlines()[-1] if output else ""
        if last_line.startswith("{") and last_line.endswith("}"):
            stats = json.loads(last_line)
            log_event(
                LOGGER,
                "firms.denoiser_inference",
                "Denoiser inference complete",
                **stats,
            )
        else:
            LOGGER.warning("Denoiser inference finished but no JSON summary found in stdout.")

    except subprocess.CalledProcessError as e:
        LOGGER.error(
            "Denoiser inference failed for batch %s: %s\nStdout: %s\nStderr: %s",
            batch_id,
            e,
            e.stdout,
            e.stderr,
        )
        raise RuntimeError(f"Denoiser inference failed for batch {batch_id}") from e


def _run_denoiser_module_direct(
    batch_id: int,
    config: "FIRMSIngestSettings",
    *,
    model_run_dir: str,
) -> None:
    """Run denoiser inference by directly importing the module (no subprocess).
    
    This avoids subprocess overhead and works in environments where uv/python
    command-line invocation is problematic.
    """
    try:
        # Import the denoiser inference module dynamically for v1/v2.
        module_name = _resolve_denoiser_module_name(config)
        module = importlib.import_module(module_name)
        denoiser_main = getattr(module, "main")

        # Build arguments as if they came from command line.
        argv = _build_denoiser_argv(batch_id=batch_id, model_run_dir=model_run_dir, config=config)

        # Capture the result - the module should return stats or print JSON
        # We need to capture stdout to get the JSON output
        import io
        from contextlib import redirect_stdout
        
        f = io.StringIO()
        with redirect_stdout(f):
            denoiser_main(argv)
        
        output = f.getvalue().strip()
        last_line = output.splitlines()[-1] if output else ""
        if last_line.startswith("{") and last_line.endswith("}"):
            stats = json.loads(last_line)
            log_event(
                LOGGER,
                "firms.denoiser_inference",
                "Denoiser inference complete (direct module)",
                **stats,
            )
        else:
            LOGGER.warning("Denoiser inference finished but no JSON summary found in output.")

    except ImportError as e:
        LOGGER.error(
            "Failed to import denoiser inference module for direct invocation: %s",
            e
        )
        raise RuntimeError(
            f"Denoiser module not available for direct invocation: {e}"
        ) from e
    except Exception as e:
        LOGGER.error("Denoiser inference failed for batch %s: %s", batch_id, e)
        raise RuntimeError(f"Denoiser inference failed for batch {batch_id}") from e


def _build_denoiser_argv(
    *,
    batch_id: int,
    model_run_dir: str,
    config: "FIRMSIngestSettings",
) -> list[str]:
    argv = [
        "--batch-id",
        str(batch_id),
        "--model-run",
        model_run_dir,
    ]
    pipeline_version = _resolve_denoiser_pipeline_version(config)
    if pipeline_version == "v2":
        argv.extend(
            [
                "--strong-filter-threshold",
                str(getattr(config, "denoiser_strong_filter_threshold", 0.5)),
                "--downweight-threshold",
                str(getattr(config, "denoiser_downweight_threshold", 0.7)),
                "--uncertainty-band-low",
                str(getattr(config, "denoiser_uncertainty_band_low", 0.45)),
                "--uncertainty-band-high",
                str(getattr(config, "denoiser_uncertainty_band_high", 0.55)),
                "--event-front-radius-m",
                str(getattr(config, "denoiser_event_front_radius_m", 2500.0)),
                "--event-front-max-gap-minutes",
                str(getattr(config, "denoiser_event_front_max_gap_minutes", 45)),
                "--event-link-radius-m",
                str(getattr(config, "denoiser_event_link_radius_m", 10000.0)),
                "--event-link-max-gap-days",
                str(getattr(config, "denoiser_event_link_max_gap_days", 11)),
                "--event-static-persistence-threshold",
                str(getattr(config, "denoiser_event_static_persistence_threshold", 0.85)),
            ]
        )
        if bool(getattr(config, "denoiser_event_strict_static_split", True)):
            argv.append("--event-strict-static-split")
        if bool(getattr(config, "denoiser_shadow_mode", False)):
            argv.append("--shadow-mode")
        return argv

    argv.extend(
        [
            "--threshold",
            str(config.denoiser_threshold),
            "--batch-size",
            str(config.denoiser_batch_size),
        ]
    )
    if config.denoiser_region:
        argv.extend(["--region", config.denoiser_region])
    if bool(getattr(config, "denoiser_strict_features", False)):
        argv.append("--strict-features")
    return argv


def _log_firms_validation(
    source: str, batch_id: int, summary: FirmsValidationSummary
) -> None:
    """Emit a structured summary of FIRMS validation results."""
    log_event(
        LOGGER,
        "firms.validation_summary",
        "FIRMS validation summary",
        source=source,
        batch_id=batch_id,
        total_rows=summary.total_rows,
        parsed_rows=summary.parsed_rows,
        skipped_invalid_coord=summary.skipped_invalid_coord,
        skipped_invalid_time=summary.skipped_invalid_time,
        missing_confidence=summary.missing_confidence,
        confidence_out_of_range=summary.confidence_out_of_range,
        brightness_missing=summary.brightness_missing,
        brightness_out_of_range=summary.brightness_out_of_range,
        sensors=dict(summary.sensor_counts),
        confidence_buckets=dict(summary.confidence_buckets),
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NASA FIRMS ingestion pipeline.")
    parser.add_argument(
        "--day-range",
        type=int,
        default=None,
        help="Override FIRMS_DAY_RANGE (number of past days; FIRMS area API supports 1-10).",
    )
    parser.add_argument(
        "--area",
        type=str,
        default="world",
        help='Bounding box "w,s,e,n" or "world". Defaults to "world".',
    )
    parser.add_argument(
        "--sources",
        type=str,
        default=None,
        help="Comma-separated FIRMS sources (defaults to env config).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    exit_code = run_firms_ingest(args.day_range, args.area, args.sources)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main(sys.argv[1:])
