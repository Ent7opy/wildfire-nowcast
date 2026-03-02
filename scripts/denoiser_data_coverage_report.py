#!/usr/bin/env python3
"""Export denoiser feature coverage/neutral diagnostics for covered known labels."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import text

# Add project root to sys.path so local packages are importable when run as a script.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api.db import get_engine

_LANDCOVER_NEUTRAL = (0.5, -1.0)
_PERSISTENCE_NEUTRAL = (0.3, 0.5, -1.0)
_WEATHER_NEUTRAL = (0.5, -1.0)

_TARGETS = {
    "landcover_available_min": 0.70,
    "persistence_available_min": 0.70,
    "weather_available_min": 0.60,
    "neutral_rate_max": 0.35,
}


def _active_industrial_policy(policy_version: str | None) -> dict[str, Any] | None:
    stmt = text(
        """
        SELECT
            policy_version,
            strict_no_go,
            gold_buffer_m,
            silver_buffer_min_m,
            silver_buffer_max_m
        FROM industrial_mask_policies
        WHERE (
                :policy_version IS NOT NULL
                AND policy_version = :policy_version
              )
           OR (
                :policy_version IS NULL
                AND (active_to IS NULL OR active_to > NOW())
              )
        ORDER BY active_from DESC, policy_version DESC
        LIMIT 1
        """
    )
    with get_engine().begin() as conn:
        row = conn.execute(stmt, {"policy_version": policy_version}).mappings().first()
    return dict(row) if row else None


def _parse_dt(value: str, *, end: bool = False) -> datetime:
    raw = str(value).strip()
    if not raw:
        raise ValueError("datetime value is empty")
    if len(raw) == 10:
        dt = datetime.strptime(raw, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return dt + timedelta(days=1) if end else dt
    dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def _coverage_filter_sql() -> str:
    return """
        EXISTS (
            SELECT 1
            FROM perimeter_coverage_masks pcm
            WHERE pcm.is_active
              AND pcm.authority_profile = :authority_profile
              AND pcm.geom && d.geom
              AND ST_Intersects(pcm.geom, d.geom)
              AND (pcm.valid_from IS NULL OR d.acq_time >= pcm.valid_from)
              AND (pcm.valid_to IS NULL OR d.acq_time <= pcm.valid_to)
        )
    """


def _rate_expr(column: str, neutral_values: tuple[float, ...], *, available: bool) -> str:
    neutral_sql = ", ".join(str(v) for v in neutral_values)
    if available:
        return (
            f"AVG(CASE WHEN {column} IS NOT NULL AND {column} NOT IN ({neutral_sql}) "
            f"THEN 1.0 ELSE 0.0 END)"
        )
    return f"AVG(CASE WHEN {column} IS NULL OR {column} IN ({neutral_sql}) THEN 1.0 ELSE 0.0 END)"


def _rows_to_dict(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        clean: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, datetime):
                clean[key] = value.isoformat()
            elif isinstance(value, Decimal):
                clean[key] = float(value)
            elif isinstance(value, float):
                clean[key] = float(value)
            elif isinstance(value, int):
                clean[key] = int(value)
            else:
                clean[key] = value
        out.append(clean)
    return out


def build_data_coverage_report(
    *,
    start_time: datetime,
    end_time: datetime,
    rule_version: str,
    authority_profile: str,
    industrial_policy_version: str | None = None,
) -> dict[str, Any]:
    params = {
        "start_time": start_time,
        "end_time": end_time,
        "rule_version": rule_version,
        "authority_profile": authority_profile,
    }

    base = f"""
        FROM denoiser_labels_v2 l
        JOIN fire_detections d ON d.id = l.fire_detection_id
        WHERE l.rule_version = :rule_version
          AND l.label IN ('POSITIVE', 'NEGATIVE')
          AND d.acq_time >= :start_time
          AND d.acq_time < :end_time
          AND {_coverage_filter_sql()}
    """

    overall_sql = text(
        f"""
        SELECT
            COUNT(*) AS n_rows,
            {_rate_expr("d.landcover_score", _LANDCOVER_NEUTRAL, available=True)} AS landcover_available_rate,
            {_rate_expr("d.landcover_score", _LANDCOVER_NEUTRAL, available=False)} AS landcover_neutral_rate,
            {_rate_expr("d.persistence_score", _PERSISTENCE_NEUTRAL, available=True)} AS persistence_available_rate,
            {_rate_expr("d.persistence_score", _PERSISTENCE_NEUTRAL, available=False)} AS persistence_neutral_rate,
            {_rate_expr("d.weather_score", _WEATHER_NEUTRAL, available=True)} AS weather_available_rate,
            {_rate_expr("d.weather_score", _WEATHER_NEUTRAL, available=False)} AS weather_neutral_rate
        {base}
        """
    )

    by_label_sql = text(
        f"""
        SELECT
            l.label,
            COUNT(*) AS n_rows,
            {_rate_expr("d.landcover_score", _LANDCOVER_NEUTRAL, available=True)} AS landcover_available_rate,
            {_rate_expr("d.landcover_score", _LANDCOVER_NEUTRAL, available=False)} AS landcover_neutral_rate,
            {_rate_expr("d.persistence_score", _PERSISTENCE_NEUTRAL, available=True)} AS persistence_available_rate,
            {_rate_expr("d.persistence_score", _PERSISTENCE_NEUTRAL, available=False)} AS persistence_neutral_rate,
            {_rate_expr("d.weather_score", _WEATHER_NEUTRAL, available=True)} AS weather_available_rate,
            {_rate_expr("d.weather_score", _WEATHER_NEUTRAL, available=False)} AS weather_neutral_rate
        {base}
        GROUP BY l.label
        ORDER BY l.label
        """
    )

    by_month_sql = text(
        f"""
        SELECT
            date_trunc('month', d.acq_time) AS month_utc,
            l.label,
            COUNT(*) AS n_rows,
            {_rate_expr("d.landcover_score", _LANDCOVER_NEUTRAL, available=True)} AS landcover_available_rate,
            {_rate_expr("d.persistence_score", _PERSISTENCE_NEUTRAL, available=True)} AS persistence_available_rate,
            {_rate_expr("d.weather_score", _WEATHER_NEUTRAL, available=True)} AS weather_available_rate,
            {_rate_expr("d.landcover_score", _LANDCOVER_NEUTRAL, available=False)} AS landcover_neutral_rate,
            {_rate_expr("d.persistence_score", _PERSISTENCE_NEUTRAL, available=False)} AS persistence_neutral_rate,
            {_rate_expr("d.weather_score", _WEATHER_NEUTRAL, available=False)} AS weather_neutral_rate
        {base}
        GROUP BY 1, 2
        ORDER BY 1, 2
        """
    )

    by_sensor_sql = text(
        f"""
        SELECT
            COALESCE(d.sensor, 'unknown') AS sensor,
            l.label,
            COUNT(*) AS n_rows,
            {_rate_expr("d.landcover_score", _LANDCOVER_NEUTRAL, available=True)} AS landcover_available_rate,
            {_rate_expr("d.persistence_score", _PERSISTENCE_NEUTRAL, available=True)} AS persistence_available_rate,
            {_rate_expr("d.weather_score", _WEATHER_NEUTRAL, available=True)} AS weather_available_rate
        {base}
        GROUP BY 1, 2
        ORDER BY 1, 2
        """
    )

    industrial_policy = _active_industrial_policy(policy_version=industrial_policy_version)
    industrial_metrics: dict[str, Any] | None = None
    if industrial_policy is not None:
        industrial_sql = text(
            """
            WITH base AS (
                SELECT d.id, d.geom
                FROM denoiser_labels_v2 l
                JOIN fire_detections d ON d.id = l.fire_detection_id
                WHERE l.rule_version = :rule_version
                  AND l.label IN ('POSITIVE', 'NEGATIVE')
                  AND d.acq_time >= :start_time
                  AND d.acq_time < :end_time
                  AND EXISTS (
                    SELECT 1
                    FROM perimeter_coverage_masks pcm
                    WHERE pcm.is_active
                      AND pcm.authority_profile = :authority_profile
                      AND pcm.geom && d.geom
                      AND ST_Intersects(pcm.geom, d.geom)
                      AND (pcm.valid_from IS NULL OR d.acq_time >= pcm.valid_from)
                      AND (pcm.valid_to IS NULL OR d.acq_time <= pcm.valid_to)
                  )
            ),
            no_go AS (
                SELECT DISTINCT b.id
                FROM base b
                JOIN industrial_no_go_zones z
                  ON z.is_active
                 AND z.policy_version = :industrial_policy_version
                 AND z.geom && b.geom
                 AND ST_Intersects(z.geom, b.geom)
            ),
            gold_match AS (
                SELECT DISTINCT b.id
                FROM base b
                JOIN industrial_sources i
                  ON COALESCE(i.is_active, TRUE)
                 AND i.authority_tier = 'gold'
                 AND ST_DWithin(b.geom::geography, i.geom::geography, :gold_buffer_m)
            ),
            silver_match AS (
                SELECT DISTINCT b.id
                FROM base b
                JOIN industrial_sources i
                  ON COALESCE(i.is_active, TRUE)
                 AND i.authority_tier = 'silver'
                 AND ST_DWithin(
                    b.geom::geography,
                    i.geom::geography,
                    LEAST(
                        :silver_buffer_max_m,
                        GREATEST(
                            :silver_buffer_min_m,
                            COALESCE(i.coordinate_precision_m::double precision, :silver_buffer_min_m)
                        )
                    )
                 )
            )
            SELECT
                COUNT(*) AS n_rows,
                COUNT(*) FILTER (
                    WHERE (
                        g.id IS NOT NULL
                        OR (s.id IS NOT NULL AND g.id IS NULL)
                    ) AND ng.id IS NULL
                ) AS mask_eligible_rows,
                COUNT(*) FILTER (WHERE g.id IS NOT NULL AND ng.id IS NULL) AS masked_gold_rows,
                COUNT(*) FILTER (WHERE s.id IS NOT NULL AND g.id IS NULL AND ng.id IS NULL) AS masked_silver_rows,
                COUNT(*) FILTER (WHERE ng.id IS NOT NULL) AS no_go_rows
            FROM base b
            LEFT JOIN gold_match g ON g.id = b.id
            LEFT JOIN silver_match s ON s.id = b.id
            LEFT JOIN no_go ng ON ng.id = b.id
            """
        )

    with get_engine().begin() as conn:
        overall = conn.execute(overall_sql, params).mappings().first()
        by_label = conn.execute(by_label_sql, params).mappings().all()
        by_month = conn.execute(by_month_sql, params).mappings().all()
        by_sensor = conn.execute(by_sensor_sql, params).mappings().all()
        if industrial_policy is not None:
            industrial_row = conn.execute(
                industrial_sql,
                {
                    **params,
                    "industrial_policy_version": str(industrial_policy["policy_version"]),
                    "gold_buffer_m": float(industrial_policy["gold_buffer_m"]),
                    "silver_buffer_min_m": float(industrial_policy["silver_buffer_min_m"]),
                    "silver_buffer_max_m": float(industrial_policy["silver_buffer_max_m"]),
                },
            ).mappings().first()
            if industrial_row is not None:
                total = int(industrial_row["n_rows"] or 0)
                industrial_metrics = {
                    "policy_version": str(industrial_policy["policy_version"]),
                    "strict_no_go": bool(industrial_policy["strict_no_go"]),
                    "gold_buffer_m": float(industrial_policy["gold_buffer_m"]),
                    "silver_buffer_min_m": float(industrial_policy["silver_buffer_min_m"]),
                    "silver_buffer_max_m": float(industrial_policy["silver_buffer_max_m"]),
                    "mask_eligible_rows": int(industrial_row["mask_eligible_rows"] or 0),
                    "masked_gold_rows": int(industrial_row["masked_gold_rows"] or 0),
                    "masked_silver_rows": int(industrial_row["masked_silver_rows"] or 0),
                    "no_go_rows": int(industrial_row["no_go_rows"] or 0),
                    "mask_eligible_rate": float((industrial_row["mask_eligible_rows"] or 0) / total) if total else 0.0,
                    "masked_gold_rate": float((industrial_row["masked_gold_rows"] or 0) / total) if total else 0.0,
                    "masked_silver_rate": float((industrial_row["masked_silver_rows"] or 0) / total) if total else 0.0,
                    "no_go_rate": float((industrial_row["no_go_rows"] or 0) / total) if total else 0.0,
                }

    if overall is None:
        raise SystemExit("No covered known labels found in requested window.")

    overall_dict = dict(overall)
    gates = {
        "landcover_available_pass": float(overall_dict["landcover_available_rate"] or 0.0)
        >= _TARGETS["landcover_available_min"],
        "persistence_available_pass": float(overall_dict["persistence_available_rate"] or 0.0)
        >= _TARGETS["persistence_available_min"],
        "weather_available_pass": float(overall_dict["weather_available_rate"] or 0.0)
        >= _TARGETS["weather_available_min"],
        "landcover_neutral_pass": float(overall_dict["landcover_neutral_rate"] or 1.0)
        <= _TARGETS["neutral_rate_max"],
        "persistence_neutral_pass": float(overall_dict["persistence_neutral_rate"] or 1.0)
        <= _TARGETS["neutral_rate_max"],
        "weather_neutral_pass": float(overall_dict["weather_neutral_rate"] or 1.0)
        <= _TARGETS["neutral_rate_max"],
    }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window": {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
        },
        "rule_version": rule_version,
        "authority_profile": authority_profile,
        "targets": _TARGETS,
        "overall": _rows_to_dict([overall_dict])[0],
        "meets_targets": all(bool(v) for v in gates.values()),
        "target_results": gates,
        "by_label": _rows_to_dict([dict(r) for r in by_label]),
        "by_month_label": _rows_to_dict([dict(r) for r in by_month]),
        "by_sensor_label": _rows_to_dict([dict(r) for r in by_sensor]),
        "industrial_metrics": industrial_metrics,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export denoiser feature coverage and neutral-rate diagnostics."
    )
    parser.add_argument("--start", required=True, help="Start datetime (YYYY-MM-DD or ISO8601)")
    parser.add_argument("--end", required=True, help="End datetime (YYYY-MM-DD or ISO8601)")
    parser.add_argument("--rule-version", default="v2_default")
    parser.add_argument("--authority-profile", default="wfigs_us")
    parser.add_argument("--industrial-policy-version", default=None)
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON path (default: reports/denoiser_v2/data_coverage_<UTC timestamp>.json)",
    )
    args = parser.parse_args(argv)

    start_time = _parse_dt(args.start, end=False)
    end_time = _parse_dt(args.end, end=True)
    if start_time >= end_time:
        raise ValueError("--start must be earlier than --end")

    report = build_data_coverage_report(
        start_time=start_time,
        end_time=end_time,
        rule_version=str(args.rule_version),
        authority_profile=str(args.authority_profile),
        industrial_policy_version=args.industrial_policy_version,
    )

    if args.out:
        out_path = Path(args.out)
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = Path("reports") / "denoiser_v2" / f"data_coverage_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
