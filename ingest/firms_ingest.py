"""CLI entrypoint for NASA FIRMS ingestion."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional

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
_DENOISER_V2_RUNTIME_THRESHOLD_KEYS: tuple[str, ...] = (
    "strong_filter_threshold",
    "downweight_threshold",
    "uncertainty_band_low",
    "uncertainty_band_high",
    "event_front_radius_m",
    "event_front_max_gap_minutes",
    "event_link_radius_m",
    "event_link_max_gap_days",
    "event_static_persistence_threshold",
    "event_strict_static_split",
)


class DenoiserTimeoutError(RuntimeError):
    """Raised when the denoiser subprocess exceeds its configured timeout.

    Callers should treat this as a fail-closed signal: do not insert the
    batch without denoiser scores.
    """


@dataclass(frozen=True)
class DenoiserRuntimePolicy:
    model_run_dir: str
    model_id: str | None
    pipeline_version: str
    threshold_profile: str
    threshold_source: str
    thresholds: dict[str, Any]
    using_promoted_model: bool


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


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
    hard_window_start_utc: datetime | None = None,
) -> tuple[list, datetime | None]:
    """Filter detections to incremental window using watermark and optional hard start."""
    if not detections:
        return [], None

    max_acq_time = max((_as_utc(d.acq_time) for d in detections), default=None)
    threshold: datetime | None = None
    if watermark_time_utc is not None:
        threshold = _as_utc(watermark_time_utc) - timedelta(minutes=max(0, int(grace_minutes)))

    hard_window_start_utc = _as_utc(hard_window_start_utc)
    if hard_window_start_utc is not None:
        threshold = max(threshold, hard_window_start_utc) if threshold is not None else hard_window_start_utc

    if threshold is None:
        return detections, max_acq_time

    filtered = [d for d in detections if (_as_utc(d.acq_time) or datetime.min.replace(tzinfo=timezone.utc)) > threshold]
    max_filtered = max((_as_utc(d.acq_time) for d in filtered), default=None)
    return filtered, max_filtered


def _max_detection_time_utc(detections: list) -> datetime | None:
    """Return max acquisition timestamp from parsed detections."""
    return max((_as_utc(d.acq_time) for d in detections), default=None)


def _resolve_active_denoiser_model() -> dict[str, Any] | None:
    """Resolve promoted denoiser metadata from model registry."""
    try:
        from api.model_registry import resolve_active_model

        return resolve_active_model("denoiser")
    except Exception:
        LOGGER.warning("Failed to resolve active promoted denoiser model; using env fallback if provided.")
        return None


def _resolve_denoiser_pipeline_version(config: "FIRMSIngestSettings") -> str:
    return str(getattr(config, "denoiser_pipeline_version", "v2") or "v2").strip().lower()


def _resolve_denoiser_threshold_profile(config: "FIRMSIngestSettings") -> str:
    return str(getattr(config, "denoiser_threshold_profile", "strict_v1") or "strict_v1").strip().lower()


def _resolve_v2_thresholds_from_config(config: "FIRMSIngestSettings") -> dict[str, Any]:
    return {
        "strong_filter_threshold": float(getattr(config, "denoiser_strong_filter_threshold", 0.5)),
        "downweight_threshold": float(getattr(config, "denoiser_downweight_threshold", 0.7)),
        "uncertainty_band_low": float(getattr(config, "denoiser_uncertainty_band_low", 0.45)),
        "uncertainty_band_high": float(getattr(config, "denoiser_uncertainty_band_high", 0.55)),
        "event_front_radius_m": float(getattr(config, "denoiser_event_front_radius_m", 2500.0)),
        "event_front_max_gap_minutes": int(getattr(config, "denoiser_event_front_max_gap_minutes", 45)),
        "event_link_radius_m": float(getattr(config, "denoiser_event_link_radius_m", 10000.0)),
        "event_link_max_gap_days": int(getattr(config, "denoiser_event_link_max_gap_days", 11)),
        "event_static_persistence_threshold": float(
            getattr(config, "denoiser_event_static_persistence_threshold", 0.85)
        ),
        "event_strict_static_split": bool(getattr(config, "denoiser_event_strict_static_split", True)),
    }


def _parse_v2_thresholds_from_runtime_contract(runtime_contract: dict[str, Any]) -> dict[str, Any]:
    thresholds = runtime_contract.get("thresholds")
    if not isinstance(thresholds, dict):
        raise RuntimeError("runtime_contract.thresholds must be an object")

    missing = [key for key in _DENOISER_V2_RUNTIME_THRESHOLD_KEYS if key not in thresholds]
    if missing:
        raise RuntimeError(
            "runtime_contract.thresholds is missing required keys: " + ", ".join(sorted(missing))
        )

    return {
        "strong_filter_threshold": float(thresholds["strong_filter_threshold"]),
        "downweight_threshold": float(thresholds["downweight_threshold"]),
        "uncertainty_band_low": float(thresholds["uncertainty_band_low"]),
        "uncertainty_band_high": float(thresholds["uncertainty_band_high"]),
        "event_front_radius_m": float(thresholds["event_front_radius_m"]),
        "event_front_max_gap_minutes": int(thresholds["event_front_max_gap_minutes"]),
        "event_link_radius_m": float(thresholds["event_link_radius_m"]),
        "event_link_max_gap_days": int(thresholds["event_link_max_gap_days"]),
        "event_static_persistence_threshold": float(thresholds["event_static_persistence_threshold"]),
        "event_strict_static_split": bool(thresholds["event_strict_static_split"]),
    }


def _resolve_denoiser_runtime_policy(config: "FIRMSIngestSettings") -> DenoiserRuntimePolicy | None:
    """Resolve denoiser runtime policy from registry contract with controlled fallback."""
    pipeline_version = _resolve_denoiser_pipeline_version(config)
    threshold_profile = _resolve_denoiser_threshold_profile(config)
    allow_unsafe_override = bool(getattr(config, "denoiser_allow_unsafe_threshold_override", False))

    active = _resolve_active_denoiser_model()
    model_id = str(active.get("model_id")) if active and active.get("model_id") else None
    model_run_dir = (
        str(active.get("artifact_uri"))
        if active and active.get("artifact_uri")
        else config.denoiser_model_run_dir
    )
    if not model_run_dir:
        return None

    runtime_contract: dict[str, Any] | None = None
    if active:
        metrics_json = active.get("metrics_json") or {}
        if isinstance(metrics_json, dict):
            runtime_contract = metrics_json.get("runtime_contract")

    if isinstance(runtime_contract, dict):
        contract_pipeline = str(runtime_contract.get("pipeline_version") or "").strip().lower()
        if contract_pipeline and contract_pipeline != pipeline_version:
            if not allow_unsafe_override:
                raise RuntimeError(
                    "Promoted denoiser pipeline mismatch: "
                    f"registry={contract_pipeline}, runtime={pipeline_version}"
                )
            LOGGER.warning(
                "Unsafe denoiser override enabled: accepting pipeline mismatch "
                "(registry=%s, runtime=%s)",
                contract_pipeline,
                pipeline_version,
            )

    if threshold_profile == "strict_v1":
        if not isinstance(runtime_contract, dict):
            if not allow_unsafe_override:
                raise RuntimeError(
                    "DENOISER_THRESHOLD_PROFILE=strict_v1 requires metrics_json.runtime_contract "
                    "on the promoted denoiser model."
                )
            LOGGER.warning(
                "Unsafe denoiser override enabled: using environment thresholds because "
                "runtime_contract is missing."
            )
        else:
            contract_profile = str(runtime_contract.get("threshold_profile") or "").strip().lower()
            if contract_profile != "strict_v1":
                if not allow_unsafe_override:
                    raise RuntimeError(
                        "Promoted denoiser runtime_contract.threshold_profile must be strict_v1 "
                        f"(got: {contract_profile or 'missing'})"
                    )
                LOGGER.warning(
                    "Unsafe denoiser override enabled: accepting threshold profile mismatch "
                    "(registry=%s, runtime=%s)",
                    contract_profile or "missing",
                    threshold_profile,
                )
            else:
                thresholds = _parse_v2_thresholds_from_runtime_contract(runtime_contract)
                return DenoiserRuntimePolicy(
                    model_run_dir=model_run_dir,
                    model_id=model_id,
                    pipeline_version=pipeline_version,
                    threshold_profile=threshold_profile,
                    threshold_source="registry_contract",
                    thresholds=thresholds,
                    using_promoted_model=bool(active and active.get("artifact_uri")),
                )

    if threshold_profile == "strict_v1" and not allow_unsafe_override:
        raise RuntimeError(
            "Denoiser strict profile is enabled but runtime contract thresholds could not be resolved."
        )

    if threshold_profile == "strict_v1" and allow_unsafe_override:
        LOGGER.warning(
            "Unsafe denoiser override is active: using environment-configured thresholds instead "
            "of promoted runtime contract."
        )

    return DenoiserRuntimePolicy(
        model_run_dir=model_run_dir,
        model_id=model_id,
        pipeline_version=pipeline_version,
        threshold_profile=threshold_profile,
        threshold_source="env_config",
        thresholds=_resolve_v2_thresholds_from_config(config),
        using_promoted_model=bool(active and active.get("artifact_uri")),
    )


def _resolve_denoiser_model_run_dir(config: "FIRMSIngestSettings") -> str | None:
    """Backward-compatible helper for tests and fallback call sites."""
    policy = _resolve_denoiser_runtime_policy(config)
    return policy.model_run_dir if policy else None


def _resolve_denoiser_module_name(config: "FIRMSIngestSettings") -> str:
    if _resolve_denoiser_pipeline_version(config) == "v2":
        return "ml.denoiser_inference_v2"
    return "ml.denoiser_inference"


def run_firms_ingest(
    day_range: Optional[int],
    area: Optional[str],
    sources: Optional[str],
    archive_date: Optional[str] = None,
) -> int:
    """Run the FIRMS ingestion pipeline.

    Args:
        archive_date: When set (YYYY-MM-DD), fetches exactly that one calendar day
            using the FIRMS DATE parameter (day_range is forced to 1).  The watermark
            filter is bypassed so historical detections are not silently dropped, and
            the watermark is NOT advanced so the live-ingest state is unaffected.
    """
    config = ingest_settings
    is_archive_mode = archive_date is not None

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
    # Archive mode always fetches exactly 1 day (the specific date); ignore caller-supplied day_range.
    effective_day_range = 1 if is_archive_mode else (day_range if day_range is not None else config.day_range)
    source_list = _resolve_sources(sources) or config.sources
    initial_lookback_minutes = int(config.firms_initial_lookback_minutes)
    incremental_lookback_minutes = int(config.firms_incremental_lookback_minutes)

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
            "initial_lookback_minutes": initial_lookback_minutes,
            "incremental_lookback_minutes": incremental_lookback_minutes,
        },
    )

    denoiser_requested = bool(config.denoiser_enabled or config.denoiser_required)
    denoiser_policy: DenoiserRuntimePolicy | None = None
    if denoiser_requested:
        try:
            denoiser_policy = _resolve_denoiser_runtime_policy(config)
        except RuntimeError as exc:
            LOGGER.error("Denoiser runtime policy error: %s", exc)
            return 2

        if config.denoiser_required and (not denoiser_policy or not denoiser_policy.using_promoted_model):
            LOGGER.error(
                "Denoiser is required but no promoted denoiser model is active in model_registry."
            )
            return 2

        if denoiser_policy:
            LOGGER.info(
                "Resolved denoiser policy: model_id=%s pipeline=%s profile=%s source=%s thresholds=%s",
                denoiser_policy.model_id or "none",
                denoiser_policy.pipeline_version,
                denoiser_policy.threshold_profile,
                denoiser_policy.threshold_source,
                denoiser_policy.thresholds,
            )
        elif config.denoiser_enabled:
            LOGGER.warning("Denoiser is enabled but no model run directory is configured; inference will be skipped.")

    for source in source_list:
        watermark = repository.get_ingest_watermark(source, area_key)
        watermark_time_utc = _as_utc((watermark or {}).get("last_acq_time_utc"))
        grace_minutes = int(config.firms_watermark_grace_minutes)
        now_utc = _utc_now()
        is_bootstrap = watermark_time_utc is None
        watermark_age_minutes: float | None = None
        if watermark_time_utc is not None:
            watermark_age_minutes = max(0.0, (now_utc - watermark_time_utc).total_seconds() / 60.0)
        is_recovery = bool(
            watermark_time_utc is not None and watermark_age_minutes is not None and watermark_age_minutes > initial_lookback_minutes
        )
        active_lookback_minutes = (
            initial_lookback_minutes if (is_bootstrap or is_recovery) else incremental_lookback_minutes
        )
        if is_bootstrap:
            lookback_mode = "bootstrap"
        elif is_recovery:
            lookback_mode = "recovery"
        else:
            lookback_mode = "incremental"

        source_uri = build_firms_url(config.map_key, source, bbox, effective_day_range)
        batch_id = repository.create_ingest_batch(
            source,
            redact_firms_url(source_uri, config.map_key),
            bbox,
            effective_day_range,
            metadata_extra={
                "area_key": area_key,
                "watermark_before": watermark_time_utc.isoformat() if watermark_time_utc else None,
                "watermark_age_minutes": round(watermark_age_minutes, 2) if watermark_age_minutes is not None else None,
                "watermark_grace_minutes": grace_minutes,
                "lookback_mode": lookback_mode,
                "lookback_minutes": active_lookback_minutes,
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
                date=archive_date,
            )
            fetched_count = len(csv_rows)
            detections, validation = parse_detection_rows(csv_rows, source, batch_id)
            if is_archive_mode:
                for det in detections:
                    det.is_archive = True
            parsed_count = len(detections)
            max_detected_acq_utc = _max_detection_time_utc(detections)
            if is_bootstrap:
                hard_window_start_utc = _utc_now() - timedelta(minutes=active_lookback_minutes)
            else:
                # FIRMS observations are delayed versus wall-clock time; use the
                # freshest available acquisition timestamp to define the incremental
                # tail window instead of `now - 30m`.
                anchor = max_detected_acq_utc or _utc_now()
                hard_window_start_utc = anchor - timedelta(minutes=active_lookback_minutes)
            if is_archive_mode:
                # Bypass the watermark filter entirely: a live watermark at today's date
                # would silently drop all historical detections.  Duplicates are handled
                # by the DB unique constraint in insert_detections.
                filtered_detections = detections
                watermark_advanced_to = None
            else:
                filtered_detections, watermark_advanced_to = _filter_detections_by_watermark(
                    detections,
                    watermark_time_utc=watermark_time_utc,
                    grace_minutes=grace_minutes,
                    hard_window_start_utc=hard_window_start_utc,
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

            should_run_denoiser = inserted > 0 and denoiser_requested
            denoiser_ran = False
            if should_run_denoiser:
                denoiser_model_run_dir = denoiser_policy.model_run_dir if denoiser_policy else None
                if not denoiser_model_run_dir:
                    if config.denoiser_required:
                        raise RuntimeError(
                            "Denoiser is required but no model run directory could be resolved."
                        )
                    LOGGER.warning("Denoiser is enabled but no model run directory is configured; skipping inference.")
                else:
                    _run_denoiser_inference(
                        batch_id,
                        config,
                        model_run_dir=denoiser_model_run_dir,
                        runtime_policy=denoiser_policy,
                    )
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
            if watermark_advanced_to is not None and not is_archive_mode:
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
                lookback_mode=lookback_mode,
                max_detected_acq_utc=max_detected_acq_utc.isoformat() if max_detected_acq_utc else None,
                window_start_utc=hard_window_start_utc.isoformat(),
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
        except DenoiserTimeoutError:
            LOGGER.error(
                "Denoiser timed out for source=%s batch=%s — rolling back inserted detections",
                source,
                batch_id,
            )
            try:
                deleted = repository.delete_detections_for_batch(batch_id)
                LOGGER.warning(
                    "Rolled back %s detections for timed-out denoiser batch %s",
                    deleted,
                    batch_id,
                )
            except Exception:
                LOGGER.exception("Failed to rollback detections for timed-out batch %s", batch_id)
            repository.finalize_ingest_batch(
                batch_id,
                status="failed",
                fetched=fetched_count,
                inserted=0,
                skipped=0,
            )
            return 1
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

    if config.denoiser_enabled and not is_archive_mode:
        _reconcile_undenoised_batches(
            max_batches=int(config.firms_reconcile_max_batches),
            config=config,
            runtime_policy=denoiser_policy,
        )

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


def _reconcile_undenoised_batches(
    max_batches: int,
    config: "FIRMSIngestSettings",
    runtime_policy: "DenoiserRuntimePolicy | None" = None,
) -> None:
    """Best-effort denoiser backfill for batches ingested without denoiser scoring."""
    candidate_batch_ids = repository.list_batches_with_undenoised_detections(
        limit=max(1, max_batches)
    )
    if not candidate_batch_ids:
        return
    LOGGER.info(
        "Denoiser backfill: %s batch(es) with NULL denoiser_decision: %s",
        len(candidate_batch_ids),
        candidate_batch_ids,
    )
    for batch_id in candidate_batch_ids:
        try:
            _run_denoiser_inference(batch_id, config, runtime_policy=runtime_policy)
        except Exception:
            LOGGER.exception("Failed denoiser backfill for batch %s", batch_id)


def _effective_denoiser_thresholds(
    config: "FIRMSIngestSettings",
    runtime_policy: DenoiserRuntimePolicy | None,
) -> dict[str, Any]:
    pipeline_version = _resolve_denoiser_pipeline_version(config)
    if pipeline_version == "v2":
        if runtime_policy is not None:
            return dict(runtime_policy.thresholds)
        return _resolve_v2_thresholds_from_config(config)
    return {
        "threshold": float(getattr(config, "denoiser_threshold", 0.5)),
        "batch_size": int(getattr(config, "denoiser_batch_size", 500)),
    }


def _run_denoiser_inference(
    batch_id: int,
    config: "FIRMSIngestSettings",
    *,
    model_run_dir: str | None = None,
    runtime_policy: DenoiserRuntimePolicy | None = None,
) -> None:
    """Trigger denoiser inference via subprocess or direct module call."""
    model_run_dir = model_run_dir or (runtime_policy.model_run_dir if runtime_policy else None) or config.denoiser_model_run_dir
    if not model_run_dir:
        LOGGER.warning(
            "Denoiser is enabled but DENOISER_MODEL_RUN_DIR is not set. Skipping inference."
        )
        return

    pipeline_version = _resolve_denoiser_pipeline_version(config)
    if runtime_policy and runtime_policy.pipeline_version != pipeline_version:
        raise RuntimeError(
            "Denoiser runtime policy pipeline mismatch: "
            f"policy={runtime_policy.pipeline_version}, runtime={pipeline_version}"
        )
    invoke_method = str(getattr(config, "denoiser_invoke_method", "uv") or "uv").strip().lower()
    module_name = _resolve_denoiser_module_name(config)
    threshold_profile = runtime_policy.threshold_profile if runtime_policy else _resolve_denoiser_threshold_profile(config)
    threshold_source = runtime_policy.threshold_source if runtime_policy else "env_config"
    effective_thresholds = _effective_denoiser_thresholds(config, runtime_policy)

    LOGGER.info(
        "Starting denoiser inference for batch %s (model_id=%s, pipeline=%s, method=%s, profile=%s, threshold_source=%s, thresholds=%s)",
        batch_id,
        (runtime_policy.model_id if runtime_policy else None) or "none",
        pipeline_version,
        invoke_method,
        threshold_profile,
        threshold_source,
        effective_thresholds,
    )

    # Use direct module import if configured
    if invoke_method == "module":
        _run_denoiser_module_direct(
            batch_id,
            config,
            model_run_dir=model_run_dir,
            runtime_policy=runtime_policy,
        )
        return

    # Build command based on invocation method
    if invoke_method == "python":
        # Use Python directly - works in containerized environments without uv
        python_exec = getattr(config, "denoiser_python_executable", None) or sys.executable
        cmd = [
            python_exec,
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

    cmd.extend(
        _build_denoiser_argv(
            batch_id=batch_id,
            model_run_dir=model_run_dir,
            config=config,
            runtime_policy=runtime_policy,
        )
    )

    subprocess_timeout: int | None = int(config.denoiser_subprocess_timeout_seconds) or None

    try:
        # We capture output to get the JSON summary
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=subprocess_timeout,
        )

        # The module prints JSON to stdout as its last line
        output = result.stdout.strip()
        last_line = output.splitlines()[-1] if output else ""
        if last_line.startswith("{") and last_line.endswith("}"):
            stats = json.loads(last_line)
            stats["model_id"] = (runtime_policy.model_id if runtime_policy else None)
            stats["pipeline_version"] = pipeline_version
            stats["threshold_profile"] = threshold_profile
            stats["threshold_source"] = threshold_source
            stats["effective_thresholds"] = effective_thresholds
            log_event(
                LOGGER,
                "firms.denoiser_inference",
                "Denoiser inference complete",
                **stats,
            )
        else:
            LOGGER.warning("Denoiser inference finished but no JSON summary found in stdout.")

    except subprocess.TimeoutExpired as e:
        LOGGER.error(
            "Denoiser inference timed out after %ss for batch %s. "
            "Batch will NOT be inserted (fail-closed). "
            "Increase DENOISER_SUBPROCESS_TIMEOUT_SECONDS if the model is large.",
            subprocess_timeout,
            batch_id,
        )
        raise DenoiserTimeoutError(
            f"Denoiser inference timed out after {subprocess_timeout}s for batch {batch_id}"
        ) from e
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
    runtime_policy: DenoiserRuntimePolicy | None = None,
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
        argv = _build_denoiser_argv(
            batch_id=batch_id,
            model_run_dir=model_run_dir,
            config=config,
            runtime_policy=runtime_policy,
        )

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
            stats["model_id"] = (runtime_policy.model_id if runtime_policy else None)
            stats["pipeline_version"] = _resolve_denoiser_pipeline_version(config)
            stats["threshold_profile"] = (
                runtime_policy.threshold_profile if runtime_policy else _resolve_denoiser_threshold_profile(config)
            )
            stats["threshold_source"] = runtime_policy.threshold_source if runtime_policy else "env_config"
            stats["effective_thresholds"] = _effective_denoiser_thresholds(config, runtime_policy)
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
    runtime_policy: DenoiserRuntimePolicy | None = None,
) -> list[str]:
    argv = [
        "--batch-id",
        str(batch_id),
        "--model-run",
        model_run_dir,
    ]
    pipeline_version = _resolve_denoiser_pipeline_version(config)
    if runtime_policy and runtime_policy.pipeline_version != pipeline_version:
        raise RuntimeError(
            "Denoiser runtime policy pipeline mismatch while building argv: "
            f"policy={runtime_policy.pipeline_version}, runtime={pipeline_version}"
        )
    if pipeline_version == "v2":
        thresholds = runtime_policy.thresholds if runtime_policy else _resolve_v2_thresholds_from_config(config)
        argv.extend(
            [
                "--strong-filter-threshold",
                str(thresholds["strong_filter_threshold"]),
                "--downweight-threshold",
                str(thresholds["downweight_threshold"]),
                "--uncertainty-band-low",
                str(thresholds["uncertainty_band_low"]),
                "--uncertainty-band-high",
                str(thresholds["uncertainty_band_high"]),
                "--event-front-radius-m",
                str(thresholds["event_front_radius_m"]),
                "--event-front-max-gap-minutes",
                str(thresholds["event_front_max_gap_minutes"]),
                "--event-link-radius-m",
                str(thresholds["event_link_radius_m"]),
                "--event-link-max-gap-days",
                str(thresholds["event_link_max_gap_days"]),
                "--event-static-persistence-threshold",
                str(thresholds["event_static_persistence_threshold"]),
            ]
        )
        if bool(thresholds.get("event_strict_static_split", getattr(config, "denoiser_event_strict_static_split", True))):
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
