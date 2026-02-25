"""Event-level denoiser inference v2 with fail-closed safety rules."""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from api.db import get_engine
from ml.denoiser.eventize import EventizeParams, eventize_detections
from ingest.config import settings as ingest_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("denoiser_inference_v2")

_NEUTRAL_LANDCOVER = (0.5, -1.0)
_NEUTRAL_PERSISTENCE = (0.3, 0.5, -1.0)
_NEUTRAL_WEATHER = (0.5, -1.0)


def _normalize_static_score(series: pd.Series, neutral_values: tuple[float, ...]) -> tuple[pd.Series, pd.Series]:
    numeric = pd.to_numeric(series, errors="coerce")
    available = numeric.notna()
    for neutral in neutral_values:
        available &= ~np.isclose(numeric, neutral, atol=1e-12, rtol=0.0)
    cleaned = numeric.where(available, np.nan)
    return available.astype(bool), cleaned.astype(float)


def _load_bundle(model_run_dir: str) -> dict[str, Any]:
    bundle_path = os.path.join(model_run_dir, "model_bundle.pkl")
    if os.path.exists(bundle_path):
        return joblib.load(bundle_path)

    # Minimal compatibility fallback.
    model = joblib.load(os.path.join(model_run_dir, "model.pkl"))
    with open(os.path.join(model_run_dir, "feature_list.json"), "r", encoding="utf-8") as f:
        features = json.load(f)
    return {
        "model": model,
        "features": features,
        "slice_cols": ["sensor", "biome_slice"],
        "global_calibrator": {"type": "identity", "model": None},
        "slice_calibrators": {},
        "thresholds": {
            "decision": 0.5,
            "strong_filter": ingest_settings.denoiser_strong_filter_threshold,
            "downweight": ingest_settings.denoiser_downweight_threshold,
            "uncertainty_band_low": ingest_settings.denoiser_uncertainty_band_low,
            "uncertainty_band_high": ingest_settings.denoiser_uncertainty_band_high,
        },
    }


def _apply_calibrator(cal: dict[str, Any], scores: np.ndarray) -> np.ndarray:
    ctype = cal.get("type")
    model = cal.get("model")
    if ctype == "isotonic" and model is not None:
        return np.asarray(model.predict(scores), dtype=float)
    if ctype == "platt" and model is not None:
        return np.asarray(model.predict_proba(scores.reshape(-1, 1))[:, 1], dtype=float)
    return np.asarray(scores, dtype=float)


def _slice_key(row: pd.Series, slice_cols: list[str]) -> str:
    return "|".join(f"{col}={row.get(col, 'unknown')}" for col in slice_cols)


def _predict_raw(model: Any, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x)[:, 1], dtype=float)
    pred = np.asarray(model.predict(x), dtype=float)
    return np.clip(pred, 0.0, 1.0)


def _get_batch_detections(engine: Engine, batch_id: int) -> pd.DataFrame:
    query = text(
        """
        SELECT
            id,
            front_id,
            event_id,
            source,
            sensor,
            acq_time,
            lat,
            lon,
            confidence,
            frp,
            brightness,
            bright_t31,
            scan,
            track,
            landcover_score,
            persistence_score,
            weather_score,
            raw_properties,
            denoised_score,
            is_noise
        FROM fire_detections
        WHERE ingest_batch_id = :batch_id
        """
    )
    with engine.begin() as conn:
        return pd.read_sql(query, conn, params={"batch_id": int(batch_id)})


def _build_event_features(batch_df: pd.DataFrame) -> pd.DataFrame:
    df = batch_df.copy()
    if df.empty:
        return pd.DataFrame()
    df["acq_time"] = pd.to_datetime(df["acq_time"], utc=True)
    df["event_id"] = df["event_id"].fillna(df["id"].map(lambda x: f"det_{int(x)}"))
    df["is_day"] = df["raw_properties"].apply(
        lambda x: 1 if isinstance(x, dict) and x.get("daynight") == "D" else 0
    )
    df["landcover_is_available"], df["landcover_score_clean"] = _normalize_static_score(
        df["landcover_score"], _NEUTRAL_LANDCOVER
    )
    df["persistence_is_available"], df["persistence_score_clean"] = _normalize_static_score(
        df["persistence_score"], _NEUTRAL_PERSISTENCE
    )
    df["weather_is_available"], df["weather_score_clean"] = _normalize_static_score(
        df["weather_score"], _NEUTRAL_WEATHER
    )

    out = (
        df.groupby("event_id", dropna=False)
        .agg(
            source=("source", lambda s: str(s.dropna().mode().iloc[0]) if not s.dropna().empty else "unknown"),
            sensor=("sensor", lambda s: str(s.dropna().mode().iloc[0]) if not s.dropna().empty else "unknown"),
            detection_count=("id", "count"),
            start_time=("acq_time", "min"),
            end_time=("acq_time", "max"),
            confidence_mean=("confidence", "mean"),
            confidence_max=("confidence", "max"),
            frp_mean=("frp", "mean"),
            frp_max=("frp", "max"),
            brightness_mean=("brightness", "mean"),
            bright_t31_mean=("bright_t31", "mean"),
            scan_mean=("scan", "mean"),
            track_mean=("track", "mean"),
            landcover_mean=("landcover_score_clean", "mean"),
            persistence_mean=("persistence_score_clean", "mean"),
            weather_mean=("weather_score_clean", "mean"),
            landcover_is_available=("landcover_is_available", "max"),
            persistence_is_available=("persistence_is_available", "max"),
            weather_is_available=("weather_is_available", "max"),
            is_day_ratio=("is_day", "mean"),
        )
        .reset_index()
    )
    out["landcover_is_available"] = out["landcover_is_available"].astype(np.float32)
    out["persistence_is_available"] = out["persistence_is_available"].astype(np.float32)
    out["weather_is_available"] = out["weather_is_available"].astype(np.float32)
    out["duration_hours"] = (out["end_time"] - out["start_time"]).dt.total_seconds() / 3600.0
    doy = out["start_time"].dt.dayofyear.fillna(1)
    out["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    out["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    out["biome_slice"] = pd.cut(
        out["landcover_mean"].fillna(0.5),
        bins=[-np.inf, 0.25, 0.6, np.inf],
        labels=["low_fuel", "mixed_fuel", "high_fuel"],
    ).astype(str)
    return out


def _decide_event(
    row: pd.Series,
    *,
    strong_filter_threshold: float,
    downweight_threshold: float,
    uncertainty_band_low: float,
    uncertainty_band_high: float,
    fail_closed_frp_mw: float,
    fail_closed_confidence: float,
    high_risk_landcover_min: float,
) -> tuple[str, bool]:
    score = float(row["event_score"])
    frp_max = float(row.get("frp_max") or 0.0)
    conf_max = float(row.get("confidence_max") or 0.0)
    landcover = float(row.get("landcover_mean") or 0.0)

    fail_closed = frp_max > fail_closed_frp_mw or (
        conf_max >= fail_closed_confidence and landcover >= high_risk_landcover_min
    )
    if fail_closed:
        return "review", True
    if uncertainty_band_low <= score <= uncertainty_band_high:
        return "review", True
    if score < strong_filter_threshold:
        return "drop", False
    if score < downweight_threshold:
        return "downweight", False
    return "pass", False


def run_inference_v2(
    *,
    batch_id: int,
    model_run_dir: str,
    shadow_mode: bool,
    strong_filter_threshold: float,
    downweight_threshold: float,
    uncertainty_band_low: float,
    uncertainty_band_high: float,
    event_front_radius_m: float,
    event_front_max_gap_minutes: int,
    event_link_radius_m: float,
    event_link_max_gap_days: int,
    event_static_persistence_threshold: float,
    event_strict_static_split: bool,
    fail_closed_frp_mw: float = 500.0,
    fail_closed_confidence: float = 80.0,
    high_risk_landcover_min: float = 0.6,
) -> dict[str, Any]:
    engine = get_engine()

    # Ensure front/event IDs exist for this batch.
    eventize_stats = eventize_detections(
        engine,
        batch_id=batch_id,
        params=EventizeParams(
            front_link_radius_m=float(event_front_radius_m),
            front_max_gap_minutes=int(event_front_max_gap_minutes),
            event_link_radius_m=float(event_link_radius_m),
            event_max_gap_days=int(event_link_max_gap_days),
            static_persistence_threshold=float(event_static_persistence_threshold),
            strict_static_split=bool(event_strict_static_split),
        ),
    )
    LOGGER.info("Eventize completed for batch %s: %s", batch_id, eventize_stats)

    detections = _get_batch_detections(engine, batch_id)
    if detections.empty:
        return {"batch_id": batch_id, "count": 0, "events": 0, "shadow_mode": shadow_mode}

    bundle = _load_bundle(model_run_dir)
    model = bundle["model"]
    features = list(bundle["features"])
    slice_cols = list(bundle.get("slice_cols", ["sensor", "biome_slice"]))
    global_cal = bundle.get("global_calibrator", {"type": "identity", "model": None})
    slice_cals = dict(bundle.get("slice_calibrators", {}))

    events_df = _build_event_features(detections)
    for col in features:
        if col not in events_df.columns:
            events_df[col] = np.nan

    raw = _predict_raw(model, events_df[features])
    calibrated = np.zeros(len(events_df), dtype=float)
    for idx, row in events_df.iterrows():
        key = _slice_key(row, slice_cols)
        cal = slice_cals.get(key, global_cal)
        calibrated[idx] = float(_apply_calibrator(cal, np.asarray([raw[idx]]))[0])

    events_df["event_score"] = calibrated

    decisions: list[str] = []
    review_flags: list[bool] = []
    for _, row in events_df.iterrows():
        decision, review_required = _decide_event(
            row,
            strong_filter_threshold=strong_filter_threshold,
            downweight_threshold=downweight_threshold,
            uncertainty_band_low=uncertainty_band_low,
            uncertainty_band_high=uncertainty_band_high,
            fail_closed_frp_mw=fail_closed_frp_mw,
            fail_closed_confidence=fail_closed_confidence,
            high_risk_landcover_min=high_risk_landcover_min,
        )
        decisions.append(decision)
        review_flags.append(review_required)

    events_df["denoiser_decision"] = decisions
    events_df["review_required"] = review_flags

    # Persist event-level decisions.
    update_event_stmt = text(
        """
        UPDATE fire_events
        SET
            event_score = :event_score,
            denoiser_decision = :denoiser_decision,
            review_required = :review_required,
            updated_at = NOW()
        WHERE event_id = :event_id
        """
    )

    review_upsert_stmt = text(
        """
        INSERT INTO denoiser_review_queue (
            event_id,
            reason,
            severity,
            status,
            payload_json,
            created_at,
            updated_at
        )
        VALUES (
            :event_id,
            :reason,
            :severity,
            'open',
            :payload_json,
            NOW(),
            NOW()
        )
        """
    )

    update_detection_stmt = text(
        """
        UPDATE fire_detections d
        SET
            event_id = s.event_id,
            front_id = COALESCE(d.front_id, s.front_id),
            event_score = s.event_score,
            denoiser_decision = s.denoiser_decision,
            review_required = s.review_required,
            denoiser_model_id = :model_id,
            denoiser_scored_at = NOW(),
            denoised_score = CASE WHEN :shadow_mode THEN d.denoised_score ELSE s.event_score END,
            is_noise = CASE WHEN :shadow_mode THEN d.is_noise ELSE (s.denoiser_decision = 'drop') END
        FROM (
            SELECT
                d0.id,
                d0.event_id,
                d0.front_id,
                e.event_score,
                e.denoiser_decision,
                e.review_required
            FROM fire_detections d0
            JOIN (
                SELECT
                    event_id,
                    event_score,
                    denoiser_decision,
                    review_required
                FROM fire_events
                WHERE event_id = ANY(:event_ids)
            ) e ON e.event_id = d0.event_id
            WHERE d0.ingest_batch_id = :batch_id
        ) s
        WHERE d.id = s.id
        """
    )

    with engine.begin() as conn:
        for row in events_df.itertuples(index=False):
            conn.execute(
                update_event_stmt,
                {
                    "event_id": row.event_id,
                    "event_score": float(row.event_score),
                    "denoiser_decision": row.denoiser_decision,
                    "review_required": bool(row.review_required),
                },
            )
            if bool(row.review_required):
                conn.execute(
                    review_upsert_stmt,
                    {
                        "event_id": row.event_id,
                        "reason": "fail_closed_or_uncertainty",
                        "severity": "high",
                        "payload_json": json.dumps(
                            {
                                "event_score": float(row.event_score),
                                "frp_max": float(row.frp_max or 0),
                                "confidence_max": float(row.confidence_max or 0),
                            }
                        ),
                    },
                )

        conn.execute(
            update_detection_stmt,
            {
                "batch_id": int(batch_id),
                "event_ids": list(events_df["event_id"].astype(str).unique()),
                "shadow_mode": bool(shadow_mode),
                "model_id": os.path.basename(model_run_dir.rstrip(os.sep)),
            },
        )

        if shadow_mode:
            # Persist summary for live-vs-shadow monitoring.
            summary_row = {
                "run_id": f"shadow_batch_{batch_id}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
                "model_id": os.path.basename(model_run_dir.rstrip(os.sep)),
                "family": "denoiser",
                "status": "shadow",
                "metrics_json": json.dumps({
                    "batch_id": int(batch_id),
                    "event_count": int(len(events_df)),
                    "decision_counts": events_df["denoiser_decision"].value_counts().to_dict(),
                }),
                "gate_report_json": None,
                "slice_metrics_json": None,
                "artifact_uri": model_run_dir,
            }
            conn.execute(
                text(
                    """
                    INSERT INTO denoiser_eval_runs (
                        run_id,
                        model_id,
                        family,
                        status,
                        metrics_json,
                        gate_report_json,
                        slice_metrics_json,
                        artifact_uri,
                        evaluated_at,
                        created_at
                    ) VALUES (
                        :run_id,
                        :model_id,
                        :family,
                        :status,
                        CAST(:metrics_json AS jsonb),
                        CAST(:gate_report_json AS jsonb),
                        CAST(:slice_metrics_json AS jsonb),
                        :artifact_uri,
                        NOW(),
                        NOW()
                    )
                    """
                ),
                summary_row,
            )

    decision_counts = {k: int(v) for k, v in events_df["denoiser_decision"].value_counts().to_dict().items()}
    summary = {
        "batch_id": int(batch_id),
        "count": int(len(detections)),
        "events": int(len(events_df)),
        "mean_event_score": float(events_df["event_score"].mean()),
        "decision_counts": decision_counts,
        "review_count": int(events_df["review_required"].sum()),
        "shadow_mode": bool(shadow_mode),
        "strong_filter_threshold": float(strong_filter_threshold),
        "downweight_threshold": float(downweight_threshold),
        "uncertainty_band_low": float(uncertainty_band_low),
        "uncertainty_band_high": float(uncertainty_band_high),
    }
    return summary


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run denoiser v2 inference for a batch.")
    parser.add_argument("--batch-id", type=int, required=True)
    parser.add_argument("--model-run", type=str, default=ingest_settings.denoiser_model_run_dir)
    parser.add_argument("--shadow-mode", action="store_true")
    parser.add_argument(
        "--strong-filter-threshold",
        type=float,
        default=ingest_settings.denoiser_strong_filter_threshold,
    )
    parser.add_argument(
        "--downweight-threshold",
        type=float,
        default=ingest_settings.denoiser_downweight_threshold,
    )
    parser.add_argument(
        "--uncertainty-band-low",
        type=float,
        default=ingest_settings.denoiser_uncertainty_band_low,
    )
    parser.add_argument(
        "--uncertainty-band-high",
        type=float,
        default=ingest_settings.denoiser_uncertainty_band_high,
    )
    parser.add_argument(
        "--event-front-radius-m",
        type=float,
        default=ingest_settings.denoiser_event_front_radius_m,
    )
    parser.add_argument(
        "--event-front-max-gap-minutes",
        type=int,
        default=ingest_settings.denoiser_event_front_max_gap_minutes,
    )
    parser.add_argument(
        "--event-link-radius-m",
        type=float,
        default=ingest_settings.denoiser_event_link_radius_m,
    )
    parser.add_argument(
        "--event-link-max-gap-days",
        type=int,
        default=ingest_settings.denoiser_event_link_max_gap_days,
    )
    parser.add_argument(
        "--event-static-persistence-threshold",
        type=float,
        default=ingest_settings.denoiser_event_static_persistence_threshold,
    )
    parser.add_argument("--event-strict-static-split", action="store_true")
    parser.add_argument("--no-event-strict-static-split", action="store_true")
    args = parser.parse_args(argv)

    if not args.model_run:
        raise SystemExit("Missing --model-run / DENOISER_MODEL_RUN_DIR for v2 inference")

    event_strict_static_split = bool(ingest_settings.denoiser_event_strict_static_split)
    if args.no_event_strict_static_split:
        event_strict_static_split = False
    elif args.event_strict_static_split:
        event_strict_static_split = True

    summary = run_inference_v2(
        batch_id=int(args.batch_id),
        model_run_dir=str(args.model_run),
        shadow_mode=bool(args.shadow_mode),
        strong_filter_threshold=float(args.strong_filter_threshold),
        downweight_threshold=float(args.downweight_threshold),
        uncertainty_band_low=float(args.uncertainty_band_low),
        uncertainty_band_high=float(args.uncertainty_band_high),
        event_front_radius_m=float(args.event_front_radius_m),
        event_front_max_gap_minutes=int(args.event_front_max_gap_minutes),
        event_link_radius_m=float(args.event_link_radius_m),
        event_link_max_gap_days=int(args.event_link_max_gap_days),
        event_static_persistence_threshold=float(args.event_static_persistence_threshold),
        event_strict_static_split=event_strict_static_split,
    )
    LOGGER.info("Denoiser v2 inference summary: %s", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
