"""Scheduler/orchestrator for FIRMS, weather, terrain, and perimeter ingestion."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from ingest.dem_preprocess import DemIngestSettings, ingest_terrain_for_bbox
from ingest.firms_ingest import run_firms_ingest
from ingest.nifc_perimeters_ingest import fetch_nifc_perimeters, ingest_perimeters
from ingest.weather_ingest import run_weather_ingest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("ingest_orchestrator")

JOB_FIRMS = "firms"
JOB_WEATHER = "weather"
JOB_TERRAIN = "terrain"
JOB_PERIMETERS = "perimeters"
JOB_ORDER = (JOB_FIRMS, JOB_WEATHER, JOB_TERRAIN, JOB_PERIMETERS)


@dataclass
class ScheduledJob:
    """Single scheduled ingestion job."""

    name: str
    interval_seconds: float
    runner: Callable[[], int]
    next_run_at: float = 0.0


class ShutdownFlag:
    """Signal-safe process shutdown flag."""

    def __init__(self) -> None:
        self._stop = False

    def request_stop(self, _signum: int, _frame: object) -> None:
        self._stop = True

    def is_set(self) -> bool:
        return self._stop


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Orchestrate FIRMS, weather, terrain, and perimeter ingestion with one-shot "
            "or recurring schedule mode."
        )
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--loop",
        action="store_true",
        help="Run continuously using per-job intervals.",
    )
    mode_group.add_argument(
        "--once",
        action="store_true",
        help="Run one orchestration cycle and exit (default).",
    )

    parser.add_argument(
        "--jobs",
        type=str,
        default=",".join(JOB_ORDER),
        help=(
            "Comma-separated job list. Supported: firms,weather,terrain,perimeters. "
            "Execution order is fixed as firms->weather->terrain->perimeters."
        ),
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=30.0,
        help="Scheduler poll sleep cap in seconds (loop mode).",
    )
    parser.add_argument(
        "--run-on-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In loop mode, run all selected jobs immediately on startup.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop orchestration immediately if any job fails.",
    )

    parser.add_argument(
        "--firms-interval-minutes",
        type=float,
        default=30.0,
        help="FIRMS run interval in minutes (loop mode).",
    )
    parser.add_argument(
        "--weather-interval-minutes",
        type=float,
        default=180.0,
        help="Weather run interval in minutes (loop mode).",
    )
    parser.add_argument(
        "--terrain-interval-minutes",
        type=float,
        default=1440.0,
        help="Terrain run interval in minutes (loop mode).",
    )
    parser.add_argument(
        "--perimeters-interval-minutes",
        type=float,
        default=1440.0,
        help="Perimeters run interval in minutes (loop mode).",
    )

    parser.add_argument(
        "--firms-day-range",
        type=int,
        default=None,
        help="Override FIRMS day range.",
    )
    parser.add_argument(
        "--firms-area",
        type=str,
        default=None,
        help="FIRMS bbox as 'w,s,e,n' or 'world'.",
    )
    parser.add_argument(
        "--firms-sources",
        type=str,
        default=None,
        help="Comma-separated FIRMS source list.",
    )

    parser.add_argument(
        "--weather-run-time",
        type=str,
        default=None,
        help="ISO8601 model run time (UTC).",
    )
    parser.add_argument(
        "--weather-bbox",
        type=float,
        nargs=4,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        help="Override weather bbox.",
    )
    parser.add_argument(
        "--weather-horizon-hours",
        type=int,
        default=None,
        help="Override weather forecast horizon.",
    )
    parser.add_argument(
        "--weather-step-hours",
        type=int,
        default=None,
        help="Override weather forecast step.",
    )
    parser.add_argument(
        "--weather-include-precip",
        action="store_true",
        help="Enable precipitation in weather ingest.",
    )
    parser.add_argument(
        "--weather-patch-mode",
        action="store_true",
        help="Enable weather patch mode optimization.",
    )

    parser.add_argument(
        "--terrain-bbox",
        type=float,
        nargs=4,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        help="Override DEM/terrain bbox.",
    )
    parser.add_argument(
        "--terrain-region-name",
        type=str,
        default=None,
        help="DEM/terrain region name override.",
    )
    parser.add_argument(
        "--terrain-output-dir",
        type=str,
        default=None,
        help="Output directory for DEM/terrain artifacts.",
    )
    parser.add_argument(
        "--terrain-cog",
        action="store_true",
        help="Emit Cloud Optimized GeoTIFFs for terrain outputs.",
    )

    parser.add_argument(
        "--perimeters-year",
        type=int,
        action="append",
        default=None,
        help="NIFC fire year to ingest (repeatable). Defaults to current UTC year.",
    )
    parser.add_argument(
        "--perimeters-bbox",
        type=float,
        nargs=4,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        help="Optional NIFC perimeter spatial filter.",
    )
    parser.add_argument(
        "--perimeters-timeout-seconds",
        type=float,
        default=120.0,
        help="NIFC request timeout per page.",
    )

    args = parser.parse_args(argv)
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if args.poll_seconds <= 0:
        raise SystemExit("--poll-seconds must be > 0")

    selected = _parse_jobs(args.jobs)
    if not selected:
        raise SystemExit("--jobs must include at least one valid job")

    interval_flags = (
        ("--firms-interval-minutes", args.firms_interval_minutes),
        ("--weather-interval-minutes", args.weather_interval_minutes),
        ("--terrain-interval-minutes", args.terrain_interval_minutes),
        ("--perimeters-interval-minutes", args.perimeters_interval_minutes),
    )
    for flag, value in interval_flags:
        if value <= 0:
            raise SystemExit(f"{flag} must be > 0")


def _parse_jobs(raw_jobs: str) -> list[str]:
    selected = [part.strip().lower() for part in raw_jobs.split(",") if part.strip()]
    seen: set[str] = set()
    unique: list[str] = []
    for name in selected:
        if name not in JOB_ORDER:
            valid = ", ".join(JOB_ORDER)
            raise SystemExit(f"Unsupported job '{name}'. Supported jobs: {valid}")
        if name not in seen:
            unique.append(name)
            seen.add(name)

    ordered = [name for name in JOB_ORDER if name in unique]
    return ordered


def _build_weather_argv(args: argparse.Namespace) -> list[str]:
    argv: list[str] = []
    if args.weather_run_time:
        argv.extend(["--run-time", args.weather_run_time])
    if args.weather_bbox:
        argv.extend(["--bbox", *[str(v) for v in args.weather_bbox]])
    if args.weather_horizon_hours is not None:
        argv.extend(["--horizon-hours", str(args.weather_horizon_hours)])
    if args.weather_step_hours is not None:
        argv.extend(["--step-hours", str(args.weather_step_hours)])
    if args.weather_include_precip:
        argv.append("--include-precip")
    if args.weather_patch_mode:
        argv.append("--patch-mode")
    return argv


def _run_firms(args: argparse.Namespace) -> int:
    return int(run_firms_ingest(args.firms_day_range, args.firms_area, args.firms_sources))


def _run_weather(args: argparse.Namespace) -> int:
    return int(run_weather_ingest(_build_weather_argv(args)))


def _run_terrain(args: argparse.Namespace) -> int:
    settings = DemIngestSettings()
    bbox = tuple(args.terrain_bbox) if args.terrain_bbox else settings.bbox
    output_dir = Path(args.terrain_output_dir) if args.terrain_output_dir else settings.data_dir
    region_name = args.terrain_region_name or settings.region_name
    ingest_terrain_for_bbox(
        bbox=bbox,
        output_dir=output_dir,
        region_name=region_name,
        emit_cog=args.terrain_cog,
    )
    return 0


def _run_perimeters(args: argparse.Namespace) -> int:
    years = args.perimeters_year or [datetime.now(timezone.utc).year]
    bbox = tuple(args.perimeters_bbox) if args.perimeters_bbox else None

    total_inserted = 0
    for year in years:
        features = fetch_nifc_perimeters(
            year=year,
            bbox=bbox,
            timeout_seconds=args.perimeters_timeout_seconds,
        )
        total_inserted += ingest_perimeters(features)

    LOGGER.info(
        "Perimeters ingest complete (years=%s inserted=%s)",
        years,
        total_inserted,
    )
    return 0


def _run_with_logging(name: str, runner: Callable[[], int]) -> int:
    started = time.monotonic()
    LOGGER.info("Job started: %s", name)
    try:
        code = int(runner())
    except Exception:  # pragma: no cover - defensive wrapper
        LOGGER.exception("Job failed with unhandled exception: %s", name)
        return 1

    elapsed = time.monotonic() - started
    if code == 0:
        LOGGER.info("Job succeeded: %s (%.2fs)", name, elapsed)
    else:
        LOGGER.error("Job failed: %s exit_code=%s (%.2fs)", name, code, elapsed)
    return code


def build_jobs(args: argparse.Namespace) -> list[ScheduledJob]:
    selected = _parse_jobs(args.jobs)

    runners: dict[str, Callable[[], int]] = {
        JOB_FIRMS: lambda: _run_firms(args),
        JOB_WEATHER: lambda: _run_weather(args),
        JOB_TERRAIN: lambda: _run_terrain(args),
        JOB_PERIMETERS: lambda: _run_perimeters(args),
    }

    intervals_seconds = {
        JOB_FIRMS: args.firms_interval_minutes * 60.0,
        JOB_WEATHER: args.weather_interval_minutes * 60.0,
        JOB_TERRAIN: args.terrain_interval_minutes * 60.0,
        JOB_PERIMETERS: args.perimeters_interval_minutes * 60.0,
    }

    return [
        ScheduledJob(
            name=name,
            interval_seconds=float(intervals_seconds[name]),
            runner=runners[name],
        )
        for name in selected
    ]


def run_once(jobs: Sequence[ScheduledJob], *, stop_on_error: bool) -> int:
    failures = 0
    for job in jobs:
        code = _run_with_logging(job.name, job.runner)
        if code != 0:
            failures += 1
            if stop_on_error:
                return 1
    return 1 if failures else 0


def run_scheduler(
    jobs: list[ScheduledJob],
    *,
    poll_seconds: float,
    run_on_start: bool,
    stop_on_error: bool,
    stop_requested: Callable[[], bool],
    now_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> int:
    failures = 0
    now = now_fn()
    for job in jobs:
        job.next_run_at = now if run_on_start else now + job.interval_seconds

    while not stop_requested():
        now = now_fn()
        due = [job for job in jobs if job.next_run_at <= now]

        if due:
            for job in due:
                code = _run_with_logging(job.name, job.runner)
                if code != 0:
                    failures += 1
                    if stop_on_error:
                        return 1
                job.next_run_at = now_fn() + job.interval_seconds
            continue

        next_run_at = min(job.next_run_at for job in jobs)
        sleep_seconds = max(0.0, min(poll_seconds, next_run_at - now))
        sleep_fn(sleep_seconds)

    LOGGER.info("Shutdown requested; stopping scheduler loop")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    jobs = build_jobs(args)

    if not args.loop:
        exit_code = run_once(jobs, stop_on_error=args.stop_on_error)
        raise SystemExit(exit_code)

    shutdown = ShutdownFlag()
    signal.signal(signal.SIGINT, shutdown.request_stop)
    signal.signal(signal.SIGTERM, shutdown.request_stop)

    exit_code = run_scheduler(
        jobs,
        poll_seconds=args.poll_seconds,
        run_on_start=args.run_on_start,
        stop_on_error=args.stop_on_error,
        stop_requested=shutdown.is_set,
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main(sys.argv[1:])
