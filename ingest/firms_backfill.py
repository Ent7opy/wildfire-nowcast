"""Historical backfill for NASA FIRMS detections using the area CSV API.

Why this exists:
- The standard FIRMS area CSV "day_range" API is limited to 1–10 days.
- NRT feeds often retain only ~7 days.
- For training, we want months of detections in `fire_detections`.

This backfill tool walks a date range backwards in <=10-day chunks by using the optional
`/YYYY-MM-DD` suffix supported by the area CSV endpoint.
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import date, datetime, timedelta
from typing import List, Optional

from ingest import repository
from ingest.config import settings as ingest_settings
from ingest.firms_client import build_firms_url, fetch_csv_rows, parse_detection_rows, redact_firms_url

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("firms_backfill")
# httpx logs include full request URLs; avoid leaking FIRMS API keys.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

MAX_FIRMS_DAY_RANGE = 10


def _parse_ymd(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _resolve_area(value: str) -> str:
    cleaned = value.strip()
    if cleaned.lower() == "world":
        return "-180,-90,180,90"
    return cleaned


def _resolve_sources(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    return [segment.strip() for segment in value.split(",") if segment.strip()]


def _update_false_source_masking(batch_id: int) -> None:
    """Update false_source_masked column for detections in the batch."""
    try:
        from api.fires.repo import update_false_source_masking

        LOGGER.info("Updating false-source masking for batch %s", batch_id)
        masked_count = update_false_source_masking(batch_id)
        LOGGER.info("Marked %s detections as false sources in batch %s", masked_count, batch_id)
    except Exception:
        LOGGER.exception("Failed to update false-source masking for batch %s", batch_id)


def run_backfill(
    *,
    start: date,
    end: date,
    chunk_days: int,
    area: str,
    sources: List[str],
    sleep_seconds: float,
    max_chunks: Optional[int],
    dry_run: bool,
) -> int:
    if chunk_days < 1 or chunk_days > MAX_FIRMS_DAY_RANGE:
        LOGGER.error("chunk_days must be 1-%s", MAX_FIRMS_DAY_RANGE)
        return 2
    if end < start:
        LOGGER.error("end date must be >= start date")
        return 2

    bbox = _resolve_area(area)
    config = ingest_settings

    current_end = end
    chunks_done = 0
    total_inserted = 0

    while current_end >= start:
        if max_chunks is not None and chunks_done >= max_chunks:
            LOGGER.warning("Reached max_chunks=%s; stopping early.", max_chunks)
            break

        current_start = max(start, current_end - timedelta(days=chunk_days - 1))
        effective_day_range = (current_end - current_start).days + 1
        date_str = current_end.strftime("%Y-%m-%d")

        LOGGER.info(
            "Backfill chunk %s: %s..%s (day_range=%s)",
            chunks_done + 1,
            current_start.isoformat(),
            current_end.isoformat(),
            effective_day_range,
        )

        for source in sources:
            source_uri = build_firms_url(config.map_key, source, bbox, effective_day_range, date=date_str)
            if dry_run:
                LOGGER.info("[dry-run] would fetch %s", source_uri)
                continue

            batch_id = repository.create_ingest_batch(
                source,
                redact_firms_url(source_uri, config.map_key),
                bbox,
                effective_day_range,
                metadata_extra={"as_of_date": date_str, "range_start": current_start.isoformat(), "range_end": current_end.isoformat()},
            )

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
                    date=date_str,
                )
                fetched_count = len(csv_rows)
                detections, _validation = parse_detection_rows(csv_rows, source, batch_id)
                parsed_count = len(detections)
                inserted = repository.insert_detections(detections)
                skipped_duplicates = parsed_count - inserted

                if inserted > 0:
                    _update_false_source_masking(batch_id)

                repository.finalize_ingest_batch(
                    batch_id,
                    status="succeeded",
                    fetched=fetched_count,
                    inserted=inserted,
                    skipped=max(skipped_duplicates, 0),
                )
                total_inserted += inserted
                LOGGER.info(
                    "Backfill source=%s as_of=%s fetched=%s parsed=%s inserted=%s duplicates=%s",
                    source,
                    date_str,
                    fetched_count,
                    parsed_count,
                    inserted,
                    skipped_duplicates,
                )
            except Exception:
                LOGGER.exception("Backfill failed for source=%s as_of=%s", source, date_str)
                repository.finalize_ingest_batch(
                    batch_id,
                    status="failed",
                    fetched=fetched_count,
                    inserted=inserted,
                    skipped=max(skipped_duplicates, 0),
                )
                return 1

            if sleep_seconds:
                time.sleep(sleep_seconds)

        chunks_done += 1
        current_end = current_start - timedelta(days=1)

    LOGGER.info("Backfill complete. chunks=%s inserted=%s", chunks_done, total_inserted)
    return 0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Historical backfill for NASA FIRMS area CSV API.")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD), inclusive.")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD), inclusive.")
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=10,
        help="Days per request window (1-10). Higher is fewer requests.",
    )
    parser.add_argument(
        "--area",
        type=str,
        default=None,
        help='Bounding box "w,s,e,n" or "world". Defaults to FIRMS_AREA env/config.',
    )
    parser.add_argument(
        "--sources",
        type=str,
        default=None,
        help="Comma-separated FIRMS sources. Defaults to FIRMS_SOURCES env/config.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Sleep between source requests to reduce rate-limit risk.",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Stop after N chunks (useful for QA).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned requests without fetching.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    config = ingest_settings
    area = _resolve_area(args.area) if args.area else config.resolved_area
    sources = _resolve_sources(args.sources) or config.sources
    exit_code = run_backfill(
        start=_parse_ymd(args.start),
        end=_parse_ymd(args.end),
        chunk_days=int(args.chunk_days),
        area=area,
        sources=sources,
        sleep_seconds=float(args.sleep_seconds),
        max_chunks=args.max_chunks,
        dry_run=bool(args.dry_run),
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()


