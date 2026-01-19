"""CLI entrypoint for NASA FIRMS ingestion."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from typing import List, Optional

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


def run_firms_ingest(
    day_range: Optional[int],
    area: Optional[str],
    sources: Optional[str],
) -> int:
    """Run the FIRMS ingestion pipeline."""
    config = ingest_settings
    bbox = _resolve_area(area) if area else config.resolved_area
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
        source_uri = build_firms_url(config.map_key, source, bbox, effective_day_range)
        batch_id = repository.create_ingest_batch(
            source,
            redact_firms_url(source_uri, config.map_key),
            bbox,
            effective_day_range,
        )
        LOGGER.info("Created ingest batch %s for %s", batch_id, source)

        fetched_count = 0
        inserted = 0
        skipped_duplicates = 0
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
            _log_firms_validation(source, batch_id, validation)
            inserted = repository.insert_detections(detections)
            skipped_duplicates = parsed_count - inserted

            if config.denoiser_enabled and inserted > 0:
                _run_denoiser_inference(batch_id, config)

            repository.finalize_ingest_batch(
                batch_id,
                status="succeeded",
                fetched=fetched_count,
                inserted=inserted,
                skipped=max(skipped_duplicates, 0),
            )
            LOGGER.info(
                "Ingested source=%s batch=%s fetched=%s parsed=%s inserted=%s duplicates=%s",
                source,
                batch_id,
                fetched_count,
                parsed_count,
                inserted,
                skipped_duplicates,
            )
        except Exception:  # pragma: no cover - defensive logging
            LOGGER.exception("Ingest failed for source=%s batch=%s", source, batch_id)
            repository.finalize_ingest_batch(
                batch_id,
                status="failed",
                fetched=fetched_count,
                inserted=inserted,
                skipped=max(skipped_duplicates, 0),
            )
            return 1

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


def _run_denoiser_inference(batch_id: int, config: "FIRMSIngestSettings") -> None:
    """Trigger denoiser inference via subprocess."""
    if not config.denoiser_model_run_dir:
        LOGGER.warning(
            "Denoiser is enabled but DENOISER_MODEL_RUN_DIR is not set. Skipping inference."
        )
        return

    LOGGER.info("Starting denoiser inference for batch %s", batch_id)

    cmd = [
        "uv",
        "run",
        "--project",
        "ml",
        "-m",
        "ml.denoiser_inference",
        "--batch-id",
        str(batch_id),
        "--model-run",
        config.denoiser_model_run_dir,
        "--threshold",
        str(config.denoiser_threshold),
        "--batch-size",
        str(config.denoiser_batch_size),
    ]

    if config.denoiser_region:
        cmd.extend(["--region", config.denoiser_region])

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


