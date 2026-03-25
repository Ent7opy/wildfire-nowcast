"""Scheduler/orchestrator for FIRMS, weather, terrain, perimeter, industrial, drift-monitor, and cleanup ingestion."""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from ingest.config import REPO_ROOT
from ingest.dem_preprocess import DemIngestSettings, ingest_terrain_for_bbox
from ingest.firms_ingest import run_firms_ingest
from ingest.industrial_sources_ingest import run_industrial_ingest
from ingest.nifc_perimeters_ingest import fetch_nifc_perimeters, ingest_perimeters
from ingest.weather_ingest import run_weather_ingest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("ingest_orchestrator")

JOB_FIRMS = "firms"
JOB_WEATHER = "weather"
JOB_LFMC = "lfmc"
JOB_LULC = "lulc"
JOB_TERRAIN = "terrain"
JOB_PERIMETERS = "perimeters"
JOB_INDUSTRIAL = "industrial"
JOB_DENOISER_DRIFT = "denoiser_drift"
JOB_CLEANUP = "cleanup"
JOB_ORDER = (JOB_FIRMS, JOB_WEATHER, JOB_LFMC, JOB_LULC, JOB_TERRAIN, JOB_PERIMETERS, JOB_INDUSTRIAL, JOB_DENOISER_DRIFT, JOB_CLEANUP)
DEFAULT_DASHBOARD_PATH = REPO_ROOT / "data" / "ingest" / "orchestrator_dashboard.json"

# Watermarks whose source name ends with _NRT are near-real-time feeds.
# A stale NRT watermark (>30 days old) indicates the ingest pipeline has been
# halted long enough that the upstream NRT archive window has expired; resetting
# to NULL triggers safe bootstrap mode on the next ingest cycle.
_WATERMARK_FUTURE_TOLERANCE_SECONDS: float = 300.0  # 5 min clock-skew grace
_WATERMARK_NRT_MAX_AGE_DAYS: int = 30
_DEFAULT_CONUS_BBOX: tuple[float, float, float, float] = (-179.5, 18.0, -66.0, 72.0)


@dataclass
class ScheduledJob:
    """Single scheduled ingestion job."""

    name: str
    interval_seconds: float
    runner: Callable[[], int]
    next_run_at: float = 0.0


@dataclass
class JobMetrics:
    attempts: int = 0
    successes: int = 0
    failures: int = 0
    retries: int = 0
    skipped_fresh: int = 0
    last_exit_code: int | None = None
    last_outcome: str | None = None
    last_started_at: str | None = None
    last_finished_at: str | None = None


class ShutdownFlag:
    """Signal-safe process shutdown flag."""

    def __init__(self) -> None:
        self._stop = False

    def request_stop(self, _signum: int, _frame: object) -> None:
        self._stop = True

    def is_set(self) -> bool:
        return self._stop


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


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
            f"Comma-separated job list. Supported: {','.join(JOB_ORDER)}. "
            f"Execution order is fixed as {'->'.join(JOB_ORDER)}."
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
        "--enforce-freshness",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip jobs whose source data is already fresh according to /health/data-freshness policy.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries per failed job run.",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=20.0,
        help="Linear backoff base in seconds between retries.",
    )
    parser.add_argument(
        "--dashboard-path",
        type=str,
        default=str(DEFAULT_DASHBOARD_PATH),
        help="Write orchestrator freshness/retry/idempotency dashboard JSON to this path.",
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
        "--industrial-interval-minutes",
        type=float,
        default=1440.0,
        help="Industrial source ingest interval in minutes (loop mode).",
    )
    parser.add_argument(
        "--denoiser-drift-interval-minutes",
        type=float,
        default=1440.0,
        help="Denoiser drift monitor run interval in minutes (loop mode).",
    )
    parser.add_argument(
        "--lfmc-interval-minutes",
        type=float,
        default=360.0,
        help="LFMC ecLand ingest interval in minutes (loop mode). 4x/day matches API job delay.",
    )
    parser.add_argument(
        "--lulc-interval-minutes",
        type=float,
        default=10080.0,
        help="LULC WorldCover backfill interval in minutes (loop mode). Weekly; tiles are static.",
    )
    parser.add_argument(
        "--cleanup-interval-minutes",
        type=float,
        default=1440.0,
        help="Database cleanup run interval in minutes (loop mode).",
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

    parser.add_argument(
        "--industrial-source-profile",
        type=str,
        default=None,
        help="Industrial source profile key from configs/industrial_authority_profiles.yaml.",
    )
    parser.add_argument(
        "--industrial-config",
        type=str,
        default=None,
        help="Optional path to industrial authority profile config.",
    )
    parser.add_argument(
        "--industrial-start",
        type=str,
        default=None,
        help="Optional industrial ingest window start (ISO8601).",
    )
    parser.add_argument(
        "--industrial-end",
        type=str,
        default=None,
        help="Optional industrial ingest window end (ISO8601).",
    )
    parser.add_argument(
        "--industrial-run-id",
        type=str,
        default=None,
        help="Optional explicit run id for industrial ingest.",
    )
    parser.add_argument(
        "--industrial-curated-file",
        action="append",
        default=None,
        help="Repeatable curated file path for curated/hybrid industrial profiles.",
    )
    parser.add_argument(
        "--industrial-timeout-seconds",
        type=float,
        default=45.0,
        help="Timeout for industrial endpoint checks and HTTP downloads.",
    )
    parser.add_argument(
        "--industrial-dry-run",
        action="store_true",
        help="Run industrial ingest in dry-run mode.",
    )

    parser.add_argument(
        "--lfmc-bbox",
        type=float,
        nargs=4,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        help="Override LFMC ingest bbox. Defaults to CONUS (-179.5 18.0 -66.0 72.0).",
    )
    parser.add_argument(
        "--lfmc-timeout-seconds",
        type=int,
        default=1800,
        help=(
            "Max seconds to wait for an LFMC API job to complete before giving up. "
            "Capped well below the FIRMS/weather interval to avoid blocking the pipeline."
        ),
    )

    parser.add_argument(
        "--lulc-bbox",
        type=float,
        nargs=4,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        help="Override LULC WorldCover backfill bbox. Defaults to CONUS (-179.5 18.0 -66.0 72.0).",
    )
    parser.add_argument(
        "--lulc-lookback-days",
        type=int,
        default=7,
        help="Backfill fire_detections this many days back on each LULC run.",
    )
    parser.add_argument(
        "--lulc-version",
        type=str,
        default="v200",
        help="ESA WorldCover product version.",
    )
    parser.add_argument(
        "--lulc-year",
        type=int,
        default=2021,
        help="ESA WorldCover reference year.",
    )
    parser.add_argument(
        "--lulc-force",
        action="store_true",
        help="Recompute LULC classes even for detections that are already classified.",
    )
    parser.add_argument(
        "--lulc-cache-dir",
        type=str,
        default=None,
        help="Local cache dir for ESA WorldCover tiles. Defaults to data/lulc/worldcover/v200_2021_tiles.",
    )

    args = parser.parse_args(argv)
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if args.poll_seconds <= 0:
        raise SystemExit("--poll-seconds must be > 0")
    if args.max_retries < 0:
        raise SystemExit("--max-retries must be >= 0")
    if args.retry_backoff_seconds < 0:
        raise SystemExit("--retry-backoff-seconds must be >= 0")

    selected = _parse_jobs(args.jobs)
    if not selected:
        raise SystemExit("--jobs must include at least one valid job")

    interval_flags = (
        ("--firms-interval-minutes", args.firms_interval_minutes),
        ("--weather-interval-minutes", args.weather_interval_minutes),
        ("--lfmc-interval-minutes", args.lfmc_interval_minutes),
        ("--lulc-interval-minutes", args.lulc_interval_minutes),
        ("--terrain-interval-minutes", args.terrain_interval_minutes),
        ("--perimeters-interval-minutes", args.perimeters_interval_minutes),
        ("--industrial-interval-minutes", args.industrial_interval_minutes),
        ("--denoiser-drift-interval-minutes", args.denoiser_drift_interval_minutes),
        ("--cleanup-interval-minutes", args.cleanup_interval_minutes),
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


def _run_lfmc(args: argparse.Namespace) -> int:
    """Run LFMC ecLand ingest.

    Timeout is enforced via ``--lfmc-timeout-seconds`` (default 1800s) to prevent
    a stalled or slow API job from blocking terrain/perimeters/industrial downstream.
    FIRMS and weather have already completed before this job starts (JOB_ORDER).
    """
    # Lazy import: lfmc_ecland_ingest pulls httpx/xarray/sqlalchemy at module scope.
    from ingest.lfmc_ecland_ingest import ingest_lfmc_ecland_for_bbox

    bbox: tuple[float, float, float, float] = tuple(args.lfmc_bbox) if args.lfmc_bbox else _DEFAULT_CONUS_BBOX  # type: ignore[assignment]
    try:
        ingest_lfmc_ecland_for_bbox(bbox=bbox, timeout_seconds=args.lfmc_timeout_seconds)
    except Exception as exc:
        tracking_id = uuid.uuid4().hex[:12]
        LOGGER.warning(
            "[tracking_id=%s] LFMC ingest failed (timeout_seconds=%s): %s "
            "— skipping job, next cycle will retry. "
            "Fuel data may be stale until successful ingest. "
            "Mitigation: next-cycle retry (mvp_operational); "
            "target: science_grade auto-alerting.",
            tracking_id,
            args.lfmc_timeout_seconds,
            exc,
        )
        return 1
    return 0


def _run_lulc(args: argparse.Namespace) -> int:
    """Backfill fire_detections with ESA WorldCover landcover classes."""
    # Lazy import: lulc_worldcover_ingest pulls numpy/rasterio/sqlalchemy at module scope.
    from ingest.lulc_worldcover_ingest import backfill_worldcover

    bbox: tuple[float, float, float, float] = tuple(args.lulc_bbox) if args.lulc_bbox else _DEFAULT_CONUS_BBOX  # type: ignore[assignment]
    now = _utc_now()
    cache_dir = Path(args.lulc_cache_dir) if args.lulc_cache_dir else Path("data/lulc/worldcover/v200_2021_tiles")

    result = backfill_worldcover(
        start_time=now - timedelta(days=args.lulc_lookback_days),
        end_time=now,
        bbox=bbox,
        version=args.lulc_version,
        year=args.lulc_year,
        force=args.lulc_force,
        max_tiles=0,
        query_batch_size=50000,
        update_batch_size=5000,
        cache_dir=cache_dir,
    )
    LOGGER.info("LULC WorldCover backfill complete: %s", result)
    return 0


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


def _build_industrial_argv(args: argparse.Namespace) -> list[str]:
    argv: list[str] = []
    source_profile = args.industrial_source_profile or "global_wri_gppd_silver"
    argv.extend(["--source-profile", source_profile])
    if args.industrial_config:
        argv.extend(["--config", args.industrial_config])
    if args.industrial_start:
        argv.extend(["--start", args.industrial_start])
    if args.industrial_end:
        argv.extend(["--end", args.industrial_end])
    if args.industrial_run_id:
        argv.extend(["--run-id", args.industrial_run_id])
    if args.industrial_curated_file:
        for path in args.industrial_curated_file:
            argv.extend(["--curated-file", str(path)])
    if args.industrial_timeout_seconds:
        argv.extend(["--timeout-seconds", str(args.industrial_timeout_seconds)])
    if args.industrial_dry_run:
        argv.append("--dry-run")
    return argv


def _run_industrial(args: argparse.Namespace) -> int:
    return int(run_industrial_ingest(_build_industrial_argv(args)))


def _run_denoiser_drift(args: argparse.Namespace) -> int:
    """Run drift monitor (alert-only; never auto-rollbacks from orchestrator)."""
    # Lazy import: avoids pulling api.db into module scope at startup.
    from ingest.denoiser_drift_monitor import monitor_denoiser_drift

    psi_warn = float(os.getenv("DENOISER_DRIFT_PSI_WARN_THRESHOLD", "0.2"))
    psi_hard = float(os.getenv("DENOISER_DRIFT_PSI_HARD_THRESHOLD", "0.35"))
    mean_delta_warn = float(os.getenv("DENOISER_DRIFT_MEAN_DELTA_WARN_THRESHOLD", "0.10"))
    mean_delta_hard = float(os.getenv("DENOISER_DRIFT_MEAN_DELTA_HARD_THRESHOLD", "0.20"))
    window_hours = int(os.getenv("DENOISER_DRIFT_WINDOW_HOURS", "24"))
    baseline_days = int(os.getenv("DENOISER_DRIFT_BASELINE_DAYS", "7"))
    min_samples = int(os.getenv("DENOISER_DRIFT_MIN_SAMPLES", "500"))

    summary = monitor_denoiser_drift(
        window_hours=window_hours,
        baseline_days=baseline_days,
        min_samples=min_samples,
        psi_warn_threshold=psi_warn,
        psi_hard_threshold=psi_hard,
        mean_delta_warn_threshold=mean_delta_warn,
        mean_delta_hard_threshold=mean_delta_hard,
        allow_rollback=False,  # orchestrator is alert-only; rollback is a manual ops decision
    )

    metrics = summary.get("metrics", {})
    psi_entry = metrics.get("psi_score", {})
    mean_entry = metrics.get("score_mean_delta", {})
    psi_severity = psi_entry.get("severity", "ok")
    mean_severity = mean_entry.get("severity", "ok")
    hard_violation = psi_severity == "hard" or mean_severity == "hard"
    warn_violation = psi_severity == "warn" or mean_severity == "warn"
    psi_val = psi_entry.get("value") or 0.0
    mean_val = mean_entry.get("value") or 0.0

    if hard_violation:
        LOGGER.error(
            "BLOCKER: denoiser drift hard violation — "
            "psi=%.4f (hard_threshold=%.2f) score_mean_delta=%.4f (hard_threshold=%.2f) "
            "— manual review required; rollback via `make model-rollback FAMILY=denoiser`",
            psi_val,
            psi_hard,
            mean_val,
            mean_delta_hard,
        )
        return 1  # surfaces as job failure in dashboard; no rollback triggered
    if warn_violation:
        LOGGER.warning(
            "Denoiser drift WARNING: psi=%.4f score_mean_delta=%.4f "
            "— approaching threshold, monitor closely [target: science_grade]",
            psi_val,
            mean_val,
        )
    else:
        LOGGER.info(
            "Denoiser drift OK: psi=%.4f score_mean_delta=%.4f",
            psi_val,
            mean_val,
        )
    return 0


def _run_cleanup(args: argparse.Namespace) -> int:
    """Run database cleanup.

    Performs real cleanup by default. Set CLEANUP_DRY_RUN=true to count eligible
    rows without deleting (operator dry-run only; scheduled path always deletes).
    """
    # Lazy import: avoids pulling api.db into module scope at startup.
    from scripts.db_cleanup import cleanup  # noqa: PLC0415

    dry_run = os.getenv("CLEANUP_DRY_RUN", "false").lower() in ("1", "true", "yes")
    cleanup(dry_run=dry_run)
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
        JOB_LFMC: lambda: _run_lfmc(args),
        JOB_LULC: lambda: _run_lulc(args),
        JOB_TERRAIN: lambda: _run_terrain(args),
        JOB_PERIMETERS: lambda: _run_perimeters(args),
        JOB_INDUSTRIAL: lambda: _run_industrial(args),
        JOB_DENOISER_DRIFT: lambda: _run_denoiser_drift(args),
        JOB_CLEANUP: lambda: _run_cleanup(args),
    }

    intervals_seconds = {
        JOB_FIRMS: args.firms_interval_minutes * 60.0,
        JOB_WEATHER: args.weather_interval_minutes * 60.0,
        JOB_LFMC: args.lfmc_interval_minutes * 60.0,
        JOB_LULC: args.lulc_interval_minutes * 60.0,
        JOB_TERRAIN: args.terrain_interval_minutes * 60.0,
        JOB_PERIMETERS: args.perimeters_interval_minutes * 60.0,
        JOB_INDUSTRIAL: args.industrial_interval_minutes * 60.0,
        JOB_DENOISER_DRIFT: args.denoiser_drift_interval_minutes * 60.0,
        JOB_CLEANUP: args.cleanup_interval_minutes * 60.0,
    }

    return [
        ScheduledJob(
            name=name,
            interval_seconds=float(intervals_seconds[name]),
            runner=runners[name],
        )
        for name in selected
    ]


def _is_nrt_source(source: str) -> bool:
    """Return True for near-real-time sources (source name ends with _NRT)."""
    return source.upper().endswith("_NRT")


def validate_and_reset_watermarks(
    *,
    now_utc: datetime | None = None,
    list_fn: Callable[[], list[dict[str, Any]]] | None = None,
    reset_fn: Callable[[str, str], None] | None = None,
) -> int:
    """Validate all DB watermarks at startup; reset corrupt ones to NULL.

    A watermark is corrupt when:
    - ``last_acq_time_utc`` is in the future beyond clock-skew tolerance, OR
    - ``last_acq_time_utc`` is older than 30 days for NRT sources (upstream
      archive window has expired, so the watermark cannot safely anchor ingest).

    Corrupt watermarks are reset to NULL, which triggers bootstrap mode
    (6-hour initial lookback) on the next FIRMS ingest cycle.

    Returns the number of watermarks that were reset.
    """
    if now_utc is None:
        now_utc = _utc_now()
    if list_fn is None:
        from ingest.repository import list_all_watermarks

        list_fn = list_all_watermarks
    if reset_fn is None:
        from ingest.repository import reset_ingest_watermark

        reset_fn = lambda s, a: reset_ingest_watermark(source=s, area_key=a)  # noqa: E731

    try:
        watermarks = list_fn()
    except Exception as exc:
        LOGGER.warning(
            "Startup watermark validation skipped — could not load watermarks: %s", exc
        )
        return 0

    future_cutoff = now_utc + timedelta(seconds=_WATERMARK_FUTURE_TOLERANCE_SECONDS)
    stale_cutoff = now_utc - timedelta(days=_WATERMARK_NRT_MAX_AGE_DAYS)
    reset_count = 0

    for wm in watermarks:
        source: str = wm["source"]
        area_key: str = wm["area_key"]
        ts: datetime | None = wm.get("last_acq_time_utc")

        if ts is None:
            continue  # NULL is valid — already in bootstrap mode

        reason: str | None = None
        if ts > future_cutoff:
            reason = (
                f"timestamp is in the future "
                f"(last_acq_time_utc={ts.isoformat()}, "
                f"now={now_utc.isoformat()}, "
                f"tolerance={_WATERMARK_FUTURE_TOLERANCE_SECONDS:.0f}s)"
            )
        elif _is_nrt_source(source) and ts < stale_cutoff:
            reason = (
                f"NRT watermark exceeds {_WATERMARK_NRT_MAX_AGE_DAYS}-day staleness limit "
                f"(last_acq_time_utc={ts.isoformat()}, now={now_utc.isoformat()})"
            )

        if reason is None:
            continue

        LOGGER.warning(
            "Corrupt watermark detected — resetting to NULL: "
            "source=%s area_key=%s before=%s reason=[%s] reset_to=NULL",
            source,
            area_key,
            ts.isoformat(),
            reason,
        )
        try:
            reset_fn(source, area_key)
        except Exception as exc:
            LOGGER.warning(
                "Failed to reset watermark: source=%s area_key=%s error=%s",
                source,
                area_key,
                exc,
            )
            continue
        reset_count += 1

    if reset_count:
        LOGGER.warning(
            "Startup watermark validation complete: %d corrupt watermark(s) reset to NULL",
            reset_count,
        )
    else:
        LOGGER.info("Startup watermark validation complete: all %d watermark(s) sane", len(watermarks))

    return reset_count


def _safe_data_status_snapshot() -> dict[str, Any] | None:
    try:
        from api.data_status import build_data_status_snapshot

        return build_data_status_snapshot(include_internal=True)
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.warning("Unable to load data freshness snapshot: %s", exc)
        return None


def _init_metrics(jobs: Sequence[ScheduledJob]) -> dict[str, JobMetrics]:
    return {job.name: JobMetrics() for job in jobs}


def _write_dashboard(
    *,
    dashboard_path: Path | None,
    metrics: dict[str, JobMetrics],
    snapshot: dict[str, Any] | None,
) -> None:
    if dashboard_path is None:
        return

    payload = {
        "generated_at": _utc_now().isoformat(),
        "metrics": {name: asdict(value) for name, value in metrics.items()},
        "data_freshness": snapshot,
        "idempotency_dashboard": (snapshot or {}).get("idempotency_dashboard"),
    }

    dashboard_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dashboard_path.with_suffix(dashboard_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(dashboard_path)


def _is_job_fresh(job_name: str, snapshot: dict[str, Any] | None) -> bool:
    # FIRMS is a near-real-time feed and must run on its configured interval
    # even when data-freshness reports "fresh". This keeps incremental ingest
    # cadence aligned with availability latency in the upstream source.
    if job_name == JOB_FIRMS:
        return False
    if not snapshot:
        return False
    source = snapshot.get("sources", {}).get(job_name, {})
    return str(source.get("state", "")).lower() == "fresh"


def _run_job_with_retries(
    *,
    job: ScheduledJob,
    metric: JobMetrics,
    max_retries: int,
    retry_backoff_seconds: float,
    sleep_fn: Callable[[float], None],
) -> int:
    attempts = 0
    while True:
        attempts += 1
        metric.attempts += 1
        metric.last_started_at = _utc_now().isoformat()
        code = _run_with_logging(job.name, job.runner)
        metric.last_finished_at = _utc_now().isoformat()
        metric.last_exit_code = code

        if code == 0:
            metric.successes += 1
            metric.last_outcome = "success"
            return 0

        if attempts > max_retries:
            metric.failures += 1
            metric.last_outcome = "failed"
            return 1

        metric.retries += 1
        sleep_seconds = retry_backoff_seconds * attempts
        LOGGER.warning(
            "Retrying job=%s attempt=%s/%s in %.1fs",
            job.name,
            attempts,
            max_retries,
            sleep_seconds,
        )
        if sleep_seconds > 0:
            sleep_fn(sleep_seconds)


def _execute_job(
    *,
    job: ScheduledJob,
    metrics: dict[str, JobMetrics],
    max_retries: int,
    retry_backoff_seconds: float,
    enforce_freshness: bool,
    dashboard_path: Path | None,
    sleep_fn: Callable[[float], None],
    status_snapshot_fn: Callable[[], dict[str, Any] | None],
) -> int:
    metric = metrics[job.name]
    should_capture_snapshot = enforce_freshness or dashboard_path is not None

    snapshot = status_snapshot_fn() if enforce_freshness else None
    if enforce_freshness and _is_job_fresh(job.name, snapshot):
        metric.skipped_fresh += 1
        metric.last_outcome = "skipped_fresh"
        metric.last_finished_at = _utc_now().isoformat()
        LOGGER.info("Skipping fresh job=%s", job.name)
        _write_dashboard(dashboard_path=dashboard_path, metrics=metrics, snapshot=snapshot)
        return 0

    code = _run_job_with_retries(
        job=job,
        metric=metric,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        sleep_fn=sleep_fn,
    )

    fresh_snapshot = status_snapshot_fn() if should_capture_snapshot else None
    _write_dashboard(dashboard_path=dashboard_path, metrics=metrics, snapshot=fresh_snapshot)
    return code


def run_once(
    jobs: Sequence[ScheduledJob],
    *,
    stop_on_error: bool,
    max_retries: int = 0,
    retry_backoff_seconds: float = 0.0,
    enforce_freshness: bool = False,
    dashboard_path: Path | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    status_snapshot_fn: Callable[[], dict[str, Any] | None] = _safe_data_status_snapshot,
) -> int:
    failures = 0
    metrics = _init_metrics(jobs)

    for job in jobs:
        code = _execute_job(
            job=job,
            metrics=metrics,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            enforce_freshness=enforce_freshness,
            dashboard_path=dashboard_path,
            sleep_fn=sleep_fn,
            status_snapshot_fn=status_snapshot_fn,
        )
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
    max_retries: int = 0,
    retry_backoff_seconds: float = 0.0,
    enforce_freshness: bool = False,
    dashboard_path: Path | None = None,
    status_snapshot_fn: Callable[[], dict[str, Any] | None] = _safe_data_status_snapshot,
    now_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> int:
    failures = 0
    metrics = _init_metrics(jobs)
    last_heartbeat_at = now_fn()

    now = now_fn()
    for job in jobs:
        job.next_run_at = now if run_on_start else now + job.interval_seconds

    while not stop_requested():
        now = now_fn()
        due = [job for job in jobs if job.next_run_at <= now]

        if due:
            for job in due:
                code = _execute_job(
                    job=job,
                    metrics=metrics,
                    max_retries=max_retries,
                    retry_backoff_seconds=retry_backoff_seconds,
                    enforce_freshness=enforce_freshness,
                    dashboard_path=dashboard_path,
                    sleep_fn=sleep_fn,
                    status_snapshot_fn=status_snapshot_fn,
                )
                if code != 0:
                    failures += 1
                    if stop_on_error:
                        return 1
                job.next_run_at = now_fn() + job.interval_seconds
            continue

        next_run_at = min(job.next_run_at for job in jobs)
        if now - last_heartbeat_at >= 60.0:
            next_due_name = min(jobs, key=lambda j: j.next_run_at).name
            eta_seconds = max(0.0, next_run_at - now)
            LOGGER.info(
                "Scheduler heartbeat: next_due_job=%s in %.1fs",
                next_due_name,
                eta_seconds,
            )
            if dashboard_path is not None:
                snapshot = status_snapshot_fn()
                _write_dashboard(
                    dashboard_path=dashboard_path,
                    metrics=metrics,
                    snapshot=snapshot,
                )
            last_heartbeat_at = now

        sleep_seconds = max(0.0, min(poll_seconds, next_run_at - now))
        sleep_fn(sleep_seconds)

    LOGGER.info("Shutdown requested; stopping scheduler loop")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    jobs = build_jobs(args)
    dashboard_path = Path(args.dashboard_path)

    validate_and_reset_watermarks()

    if not args.loop:
        exit_code = run_once(
            jobs,
            stop_on_error=args.stop_on_error,
            max_retries=args.max_retries,
            retry_backoff_seconds=args.retry_backoff_seconds,
            enforce_freshness=args.enforce_freshness,
            dashboard_path=dashboard_path,
        )
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
        max_retries=args.max_retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
        enforce_freshness=args.enforce_freshness,
        dashboard_path=dashboard_path,
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main(sys.argv[1:])
