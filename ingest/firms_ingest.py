"""CLI entrypoint for NASA FIRMS ingestion."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional

from ingest import repository
from ingest.config import settings as ingest_settings
from ingest.firms_client import (
    build_firms_url,
    fetch_csv_rows,
    parse_detection_rows,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("firms_ingest")


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
        batch_id = repository.create_ingest_batch(source, source_uri, bbox, effective_day_range)
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
            detections = parse_detection_rows(csv_rows, source, batch_id)
            parsed_count = len(detections)
            inserted = repository.insert_detections(detections)
            skipped_duplicates = parsed_count - inserted

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


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NASA FIRMS ingestion pipeline.")
    parser.add_argument(
        "--day-range",
        type=int,
        default=None,
        help="Override FIRMS_DAY_RANGE (number of past days).",
    )
    parser.add_argument(
        "--area",
        type=str,
        default=None,
        help='Bounding box "w,s,e,n" or "world".',
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


