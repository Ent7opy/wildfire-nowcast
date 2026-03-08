"""Ground-truth + weak-supervision labeling for denoiser v2."""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.sql.elements import TextClause

from api.core.grid import DEFAULT_CELL_SIZE_DEG
from api.db import get_engine
from ml.denoiser.coverage_authority import require_coverage_freshness

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("denoiser_label_v2")


DEFAULT_PARAMS = {
    "positive_buffer_m": 2315.0,
    "positive_time_pad_hours": 48,
    "positive_confidence_floor": 30.0,
    "positive_low_confidence_max": 30.0,
    "negative_industrial_radius_m": 1000.0,
    "negative_far_dist_m": 10000.0,
    "negative_time_pad_days": 30,
    "negative_frp_floor_mw": 5.0,
    "chronic_static_days_threshold": 200,
    "chronic_static_window_days": 365,
    "biophysical_landcover_max": 0.1,
    "agricultural_landcover_score": 0.7,
    "agricultural_landcover_score_tolerance": 0.05,
    "agricultural_lulc_classes": [40],
    "probable_positive_frp_mw": 100.0,
    "probable_positive_confidence": 70.0,
    "probable_positive_landcover_min": 0.5,
    "negative_event_static_ratio_min": 0.7,
    "negative_event_persistence_min": 0.85,
    "negative_event_min_days": 3,
}

_DEFAULT_GLOBAL_BBOX: tuple[float, float, float, float] = (-180.0, -90.0, 180.0, 90.0)
_ALLOWED_PERIMETER_SOURCES = {"authoritative_perimeters", "fire_perimeters"}
_ALLOWED_AUTHORITATIVE_TIERS = {"silver", "gold", "both"}
_INDUSTRIAL_GOLD_BUFFER_M = 375.0
_INDUSTRIAL_SILVER_BUFFER_M = 750.0

_GOVERNANCE_SQL = """
      AND ap.is_authoritative
      AND ap.poly_featurestatus IN ('Approved', 'Certified')
      AND ap.poly_featureaccess = 'Public'
      AND ap.poly_isvisible = 'Yes'
      AND COALESCE(ap.attr_isvalid, 1) = 1
      AND COALESCE(ap.attr_isquarantined, 0) = 0
"""


def _authoritative_tier_sql(alias: str = "ap") -> str:
    return f"""
      AND (
            (:authoritative_tier = 'both' AND {alias}.tier IN ('silver', 'gold'))
         OR (:authoritative_tier = 'silver' AND {alias}.tier = 'silver')
         OR (:authoritative_tier = 'gold' AND {alias}.tier = 'gold')
      )
    """


def _build_perimeter_sql(*, perimeter_source: str) -> tuple[TextClause, TextClause]:
    if perimeter_source == "authoritative_perimeters":
        positive_sql = text(
            f"""
            CREATE TEMP TABLE tmp_label_positive_ids ON COMMIT DROP AS
            SELECT DISTINCT c.id
            FROM tmp_label_candidates c
            JOIN authoritative_perimeters ap
              ON ap.geom && ST_Expand(c.geom, :positive_buffer_deg)
             AND ST_DWithin(c.geom::geography, ap.geom::geography, :positive_buffer_m)
            WHERE c.in_coverage
              {_GOVERNANCE_SQL}
              {_authoritative_tier_sql("ap")}
              AND c.acq_time >= COALESCE(
                    ap.attr_firediscoverydatetime,
                    ap.poly_polygondatetime,
                    c.acq_time
                  ) - make_interval(hours => :positive_time_pad_hours)
              AND c.acq_time <= COALESCE(
                    ap.attr_containmentdatetime,
                    ap.attr_controldatetime,
                    ap.poly_polygondatetime,
                    c.acq_time
                  ) + make_interval(hours => :positive_time_pad_hours)
              AND COALESCE(c.frp, 0) >= :negative_frp_floor_mw
              AND COALESCE(c.confidence, 0) >= :positive_confidence_floor
              AND NOT c.is_low_confidence
              AND NOT c.is_agri_lulc
            """
        )

        far_low_sql = text(
            f"""
            CREATE TEMP TABLE tmp_label_far_low_ids ON COMMIT DROP AS
            SELECT c.id
            FROM tmp_label_candidates c
            WHERE c.in_coverage
              AND COALESCE(c.frp, 0) < :negative_frp_floor_mw
              AND NOT EXISTS (
                SELECT 1
                FROM authoritative_perimeters ap
                WHERE ap.geom && ST_Expand(c.geom, :negative_far_dist_deg)
                  AND ST_DWithin(c.geom::geography, ap.geom::geography, :negative_far_dist_m)
                  {_GOVERNANCE_SQL}
                  {_authoritative_tier_sql("ap")}
                  AND c.acq_time >= COALESCE(
                        ap.attr_firediscoverydatetime,
                        ap.poly_polygondatetime,
                        c.acq_time
                      ) - make_interval(days => :negative_time_pad_days)
                  AND c.acq_time <= COALESCE(
                        ap.attr_containmentdatetime,
                        ap.attr_controldatetime,
                        ap.poly_polygondatetime,
                        c.acq_time
                      ) + make_interval(days => :negative_time_pad_days)
              )
            """
        )
        return positive_sql, far_low_sql

    if perimeter_source == "fire_perimeters":
        positive_sql = text(
            """
            CREATE TEMP TABLE tmp_label_positive_ids ON COMMIT DROP AS
            SELECT DISTINCT c.id
            FROM tmp_label_candidates c
            JOIN fire_perimeters fp
              ON fp.geom && ST_Expand(c.geom, :positive_buffer_deg)
             AND ST_DWithin(c.geom::geography, fp.geom::geography, :positive_buffer_m)
            WHERE c.in_coverage
              AND c.acq_time >= fp.fire_start - make_interval(hours => :positive_time_pad_hours)
              AND (
                fp.fire_end IS NULL
                OR c.acq_time <= fp.fire_end + make_interval(hours => :positive_time_pad_hours)
              )
              AND COALESCE(c.frp, 0) >= :negative_frp_floor_mw
              AND COALESCE(c.confidence, 0) >= :positive_confidence_floor
              AND NOT c.is_low_confidence
              AND NOT c.is_agri_lulc
            """
        )

        far_low_sql = text(
            """
            CREATE TEMP TABLE tmp_label_far_low_ids ON COMMIT DROP AS
            SELECT c.id
            FROM tmp_label_candidates c
            WHERE c.in_coverage
              AND COALESCE(c.frp, 0) < :negative_frp_floor_mw
              AND NOT EXISTS (
                SELECT 1
                FROM fire_perimeters fp
                WHERE fp.geom && ST_Expand(c.geom, :negative_far_dist_deg)
                  AND ST_DWithin(c.geom::geography, fp.geom::geography, :negative_far_dist_m)
                  AND c.acq_time >= fp.fire_start - make_interval(days => :negative_time_pad_days)
                  AND (
                    fp.fire_end IS NULL
                    OR c.acq_time <= fp.fire_end + make_interval(days => :negative_time_pad_days)
                  )
              )
            """
        )
        return positive_sql, far_low_sql

    raise ValueError(f"Unsupported perimeter_source={perimeter_source!r}")


def _parse_authority_profiles(raw: str) -> list[str]:
    profiles = [token.strip() for token in str(raw).split(",") if token.strip()]
    if not profiles:
        raise ValueError("authority_profile cannot be empty")
    return sorted(set(profiles))


def _check_perimeter_coverage(
    engine: Engine,
    aoi_bbox: Tuple[float, float, float, float],
    start_time: datetime,
    end_time: datetime,
    *,
    authority_profiles: list[str],
) -> int:
    min_lon, min_lat, max_lon, max_lat = aoi_bbox
    stmt = text(
        """
        SELECT COUNT(*) AS n
        FROM perimeter_coverage_masks
        WHERE is_active
          AND authority_profile = ANY(:authority_profiles)
          AND geom && ST_MakeEnvelope(:min_lon, :min_lat, :max_lon, :max_lat, 4326)
          AND (valid_from IS NULL OR valid_from <= :end_time)
          AND (valid_to IS NULL OR valid_to >= :start_time)
        """
    )
    with engine.begin() as conn:
        row = conn.execute(
            stmt,
            {
                "min_lon": min_lon,
                "min_lat": min_lat,
                "max_lon": max_lon,
                "max_lat": max_lat,
                "start_time": start_time,
                "end_time": end_time,
                "authority_profiles": authority_profiles,
            },
        ).mappings().first()
    return int(row["n"]) if row else 0


def _active_coverage_mask_ids(
    engine: Engine,
    start_time: datetime,
    end_time: datetime,
    *,
    authority_profiles: list[str],
) -> list[str]:
    stmt = text(
        """
        SELECT mask_id
        FROM perimeter_coverage_masks
        WHERE is_active
          AND authority_profile = ANY(:authority_profiles)
          AND (valid_from IS NULL OR valid_from <= :end_time)
          AND (valid_to IS NULL OR valid_to >= :start_time)
        ORDER BY mask_id
        """
    )
    with engine.begin() as conn:
        rows = conn.execute(
            stmt,
            {
                "start_time": start_time,
                "end_time": end_time,
                "authority_profiles": authority_profiles,
            },
        ).mappings().all()
    return [str(row["mask_id"]) for row in rows]


def _active_industrial_policy(
    engine: Engine,
    *,
    policy_version: str | None = None,
) -> dict | None:
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
    with engine.begin() as conn:
        row = conn.execute(stmt, {"policy_version": policy_version}).mappings().first()
    return dict(row) if row else None


def _log_step(step: str, started_at: float, *, rows: int | None = None) -> None:
    elapsed = time.perf_counter() - started_at
    suffix = ""
    if rows is not None:
        suffix = f", rows={rows}"
    LOGGER.info("%s completed in %.3fs%s", step, elapsed, suffix)


def _label_single_window(
    engine: Engine,
    *,
    aoi_bbox: Tuple[float, float, float, float],
    start_time: datetime,
    end_time: datetime,
    rule_version: str,
    params: Dict,
    authority_profile: str,
    authority_profiles: list[str],
    perimeter_source: str,
    authoritative_tier: str,
    rule_params_payload: Dict,
    industrial_policy: dict | None,
    industrial_negatives_global: bool,
) -> dict[str, int]:
    min_lon, min_lat, max_lon, max_lat = aoi_bbox

    p = params
    meters_to_deg = 1.0 / 111000.0
    query_params = {
        "start_time": start_time,
        "end_time": end_time,
        "min_lon": min_lon,
        "min_lat": min_lat,
        "max_lon": max_lon,
        "max_lat": max_lat,
        "positive_buffer_m": float(p["positive_buffer_m"]),
        "positive_buffer_deg": float(p["positive_buffer_m"]) * meters_to_deg,
        "positive_time_pad_hours": int(p["positive_time_pad_hours"]),
        "positive_confidence_floor": float(p["positive_confidence_floor"]),
        "positive_low_confidence_max": float(p["positive_low_confidence_max"]),
        "negative_industrial_radius_m": float(p["negative_industrial_radius_m"]),
        "negative_industrial_radius_deg": float(p["negative_industrial_radius_m"]) * meters_to_deg,
        "negative_far_dist_m": float(p["negative_far_dist_m"]),
        "negative_far_dist_deg": float(p["negative_far_dist_m"]) * meters_to_deg,
        "negative_time_pad_days": int(p["negative_time_pad_days"]),
        "negative_frp_floor_mw": float(p["negative_frp_floor_mw"]),
        "chronic_static_days_threshold": int(p["chronic_static_days_threshold"]),
        "chronic_static_window_days": int(p["chronic_static_window_days"]),
        "grid_size": float(DEFAULT_CELL_SIZE_DEG),
        "biophysical_landcover_max": float(p["biophysical_landcover_max"]),
        "agricultural_landcover_score": float(p["agricultural_landcover_score"]),
        "agricultural_landcover_score_tolerance": float(p["agricultural_landcover_score_tolerance"]),
        "agricultural_lulc_classes": [int(code) for code in p["agricultural_lulc_classes"]],
        "probable_positive_frp_mw": float(p["probable_positive_frp_mw"]),
        "probable_positive_confidence": float(p["probable_positive_confidence"]),
        "probable_positive_landcover_min": float(p["probable_positive_landcover_min"]),
        "negative_event_static_ratio_min": float(p["negative_event_static_ratio_min"]),
        "negative_event_persistence_min": float(p["negative_event_persistence_min"]),
        "negative_event_min_days": int(p["negative_event_min_days"]),
        "rule_version": rule_version,
        "source": "ground_truth_v2_authoritative"
        if perimeter_source == "authoritative_perimeters"
        else "ground_truth_v2_legacy",
        "rule_params": json.dumps(rule_params_payload, default=str),
        "labeled_at": datetime.utcnow(),
        "authority_profile": authority_profile,
        "authority_profiles": authority_profiles,
        "authoritative_tier": authoritative_tier,
        "industrial_policy_version": (
            str(industrial_policy.get("policy_version")) if industrial_policy is not None else None
        ),
        "industrial_strict_no_go": bool((industrial_policy or {}).get("strict_no_go", False)),
        "industrial_gold_buffer_m": float(
            _INDUSTRIAL_GOLD_BUFFER_M
        ),
        "industrial_gold_buffer_deg": float(
            _INDUSTRIAL_GOLD_BUFFER_M
        )
        * meters_to_deg,
        "industrial_silver_buffer_m": float(_INDUSTRIAL_SILVER_BUFFER_M),
        "industrial_silver_buffer_deg": float(_INDUSTRIAL_SILVER_BUFFER_M)
        * meters_to_deg,
        "industrial_negatives_global": bool(industrial_negatives_global),
    }

    create_candidates_sql = text(
        """
        CREATE TEMP TABLE tmp_label_candidates ON COMMIT DROP AS
        SELECT
            d.id,
            d.event_id,
            d.lat,
            d.lon,
            d.acq_time,
            d.confidence,
            d.frp,
            d.landcover_score,
            d.landcover_class,
            d.landcover_label,
            (
                COALESCE(d.confidence, 50.0) < :positive_low_confidence_max
                OR lower(trim(COALESCE(d.raw_properties->>'confidence', ''))) IN ('l', 'low')
                OR lower(trim(COALESCE(d.raw_properties->>'confidence_text', ''))) IN ('l', 'low')
                OR lower(trim(COALESCE(d.raw_properties->>'confidence_label', ''))) IN ('l', 'low')
                OR lower(trim(COALESCE(d.raw_properties->>'firms_confidence', ''))) IN ('l', 'low')
            ) AS is_low_confidence,
            (COALESCE(d.landcover_score, 0.5) <= :biophysical_landcover_max) AS is_zero_fuel_lulc,
            (
                (
                    d.landcover_class = ANY(:agricultural_lulc_classes)
                )
                OR (
                    COALESCE(d.raw_properties->>'landcover_class', '') ~ '^[0-9]+$'
                    AND (d.raw_properties->>'landcover_class')::integer = ANY(:agricultural_lulc_classes)
                )
                OR (
                    COALESCE(d.raw_properties->>'landcover_code', '') ~ '^[0-9]+$'
                    AND (d.raw_properties->>'landcover_code')::integer = ANY(:agricultural_lulc_classes)
                )
                OR lower(trim(COALESCE(d.landcover_label, ''))) LIKE ANY (
                    ARRAY['%crop%', '%cropland%', '%agri%', '%agriculture%', '%farmland%']
                )
                OR lower(trim(COALESCE(d.raw_properties->>'landcover_label', ''))) LIKE ANY (
                    ARRAY['%crop%', '%cropland%', '%agri%', '%agriculture%', '%farmland%']
                )
                OR lower(trim(COALESCE(d.raw_properties->>'landcover_type', ''))) LIKE ANY (
                    ARRAY['%crop%', '%cropland%', '%agri%', '%agriculture%', '%farmland%']
                )
                OR lower(trim(COALESCE(d.raw_properties->>'lulc_label', ''))) LIKE ANY (
                    ARRAY['%crop%', '%cropland%', '%agri%', '%agriculture%', '%farmland%']
                )
                OR abs(
                    COALESCE(d.landcover_score, -999.0) - :agricultural_landcover_score
                ) <= :agricultural_landcover_score_tolerance
            ) AS is_agri_lulc,
            COALESCE(d.false_source_masked, FALSE) AS false_source_masked,
            COALESCE(d.persistence_score, 0.0) AS persistence_score,
            d.geom,
            EXISTS (
                SELECT 1
                FROM perimeter_coverage_masks pcm
                WHERE pcm.is_active
                  AND pcm.authority_profile = ANY(:authority_profiles)
                  AND pcm.geom && d.geom
                  AND ST_Intersects(pcm.geom, d.geom)
                  AND (pcm.valid_from IS NULL OR d.acq_time >= pcm.valid_from)
                  AND (pcm.valid_to IS NULL OR d.acq_time <= pcm.valid_to)
            ) AS in_coverage
        FROM fire_detections d
        WHERE d.acq_time BETWEEN :start_time AND :end_time
          AND d.geom && ST_MakeEnvelope(:min_lon, :min_lat, :max_lon, :max_lat, 4326)
        """
    )

    create_candidates_indexes = [
        text("CREATE UNIQUE INDEX tmp_label_candidates_id_idx ON tmp_label_candidates (id)"),
        text("CREATE INDEX tmp_label_candidates_cov_idx ON tmp_label_candidates (in_coverage)"),
    ]

    create_chronic_cells_sql = text(
        """
        CREATE TEMP TABLE tmp_label_chronic_cells ON COMMIT DROP AS
        SELECT
            floor(lat / :grid_size) AS i_lat,
            floor(lon / :grid_size) AS j_lon
        FROM fire_detections
        WHERE acq_time BETWEEN (
            :start_time - make_interval(days => :chronic_static_window_days)
        ) AND :end_time
        GROUP BY 1, 2
        HAVING COUNT(DISTINCT date(acq_time)) >= :chronic_static_days_threshold
        """
    )

    create_chronic_idx_sql = text(
        "CREATE INDEX tmp_label_chronic_cells_idx ON tmp_label_chronic_cells (i_lat, j_lon)"
    )

    create_positive_sql, create_far_low_sql = _build_perimeter_sql(perimeter_source=perimeter_source)

    if industrial_policy is not None:
        create_industrial_sources_sql = text(
            """
            CREATE TEMP TABLE tmp_label_industrial_sources ON COMMIT DROP AS
            SELECT
                i.id,
                i.authority_tier,
                i.valid_from,
                i.valid_to,
                ST_Buffer(
                    i.geom::geography,
                    CASE
                        WHEN i.authority_tier = 'gold' THEN :industrial_gold_buffer_m
                        ELSE :industrial_silver_buffer_m
                    END
                )::geometry AS eff_geom
            FROM industrial_sources i
            WHERE COALESCE(i.is_active, TRUE)
              AND i.authority_tier IN ('gold', 'silver')
            """
        )
        create_industrial_sources_indexes = [
            text("CREATE INDEX tmp_label_industrial_sources_eff_gix ON tmp_label_industrial_sources USING GIST (eff_geom)"),
            text(
                "CREATE INDEX tmp_label_industrial_sources_tier_time_idx "
                "ON tmp_label_industrial_sources (authority_tier, valid_from, valid_to)"
            ),
        ]
        create_industrial_sql = text(
            """
            CREATE TEMP TABLE tmp_label_industrial_ids ON COMMIT DROP AS
            WITH base AS (
                SELECT
                    c.id,
                    c.geom,
                    c.acq_time
                FROM tmp_label_candidates c
            ),
            no_go AS (
                SELECT DISTINCT c.id
                FROM base c
                JOIN industrial_no_go_zones z
                  ON z.is_active
                 AND z.policy_version = :industrial_policy_version
                 AND z.geom && c.geom
                 AND ST_Intersects(z.geom, c.geom)
            ),
            gold_ids AS (
                SELECT b.id
                FROM base b
                WHERE EXISTS (
                    SELECT 1
                    FROM tmp_label_industrial_sources i
                    WHERE i.authority_tier = 'gold'
                      AND (i.valid_from IS NULL OR i.valid_from <= b.acq_time)
                      AND (i.valid_to IS NULL OR i.valid_to >= b.acq_time)
                      AND i.eff_geom && b.geom
                      AND ST_Intersects(i.eff_geom, b.geom)
                )
            ),
            silver_ids AS (
                SELECT b.id
                FROM base b
                WHERE EXISTS (
                    SELECT 1
                    FROM tmp_label_industrial_sources i
                    WHERE i.authority_tier = 'silver'
                      AND (i.valid_from IS NULL OR i.valid_from <= b.acq_time)
                      AND (i.valid_to IS NULL OR i.valid_to >= b.acq_time)
                      AND i.eff_geom && b.geom
                      AND ST_Intersects(i.eff_geom, b.geom)
                )
                AND NOT EXISTS (
                    SELECT 1
                    FROM tmp_label_industrial_sources i
                    WHERE i.authority_tier = 'gold'
                      AND (i.valid_from IS NULL OR i.valid_from <= b.acq_time)
                      AND (i.valid_to IS NULL OR i.valid_to >= b.acq_time)
                      AND i.eff_geom && b.geom
                      AND ST_Intersects(i.eff_geom, b.geom)
                )
            )
            SELECT id
            FROM (
                SELECT id FROM gold_ids
                UNION
                SELECT id FROM silver_ids
            ) matched
            WHERE NOT (
                :industrial_strict_no_go
                AND EXISTS (SELECT 1 FROM no_go ng WHERE ng.id = matched.id)
            )
            """
        )
    else:
        create_industrial_sources_sql = None
        create_industrial_sources_indexes: list[TextClause] = []
        create_industrial_sql = text(
            """
            CREATE TEMP TABLE tmp_label_industrial_ids ON COMMIT DROP AS
            SELECT DISTINCT c.id
            FROM tmp_label_candidates c
            JOIN industrial_sources i
              ON i.geom && ST_Expand(c.geom, :negative_industrial_radius_deg)
             AND ST_DWithin(c.geom::geography, i.geom::geography, :negative_industrial_radius_m)
            """
        )

    create_chronic_ids_sql = text(
        """
        CREATE TEMP TABLE tmp_label_chronic_ids ON COMMIT DROP AS
        SELECT c.id
        FROM tmp_label_candidates c
        JOIN tmp_label_chronic_cells cc
          ON floor(c.lat / :grid_size) = cc.i_lat
         AND floor(c.lon / :grid_size) = cc.j_lon
        WHERE c.in_coverage
        """
    )

    create_negative_sql = text(
        """
        CREATE TEMP TABLE tmp_label_negative_ids ON COMMIT DROP AS
        SELECT id FROM tmp_label_industrial_ids
        UNION
        SELECT id FROM tmp_label_far_low_ids
        UNION
        SELECT id FROM tmp_label_chronic_ids
        UNION
        SELECT id FROM tmp_label_event_static_ids
        UNION
        SELECT c.id
        FROM tmp_label_candidates c
        WHERE c.in_coverage
          AND c.is_zero_fuel_lulc
        UNION
        SELECT c.id
        FROM tmp_label_candidates c
        WHERE c.in_coverage
          AND c.is_agri_lulc
        """
    )

    create_probable_sql = text(
        """
        CREATE TEMP TABLE tmp_label_probable_positive_ids ON COMMIT DROP AS
        SELECT c.id
        FROM tmp_label_candidates c
        WHERE c.in_coverage
          AND COALESCE(c.frp, 0) >= :probable_positive_frp_mw
          AND COALESCE(c.confidence, 0) >= :probable_positive_confidence
          AND COALESCE(c.landcover_score, 0.5) >= :probable_positive_landcover_min
          AND NOT c.is_agri_lulc
        """
    )

    create_event_static_ids_sql = text(
        """
        CREATE TEMP TABLE tmp_label_event_static_ids ON COMMIT DROP AS
        WITH event_static AS (
            SELECT
                event_id
            FROM tmp_label_candidates
            WHERE in_coverage
              AND event_id IS NOT NULL
            GROUP BY event_id
            HAVING AVG(
                CASE
                    WHEN false_source_masked
                         OR COALESCE(persistence_score, 0.0) >= :negative_event_persistence_min
                    THEN 1.0
                    ELSE 0.0
                END
            ) >= :negative_event_static_ratio_min
               AND (
                    EXTRACT(EPOCH FROM (MAX(acq_time) - MIN(acq_time))) / 86400.0
               ) >= :negative_event_min_days
        )
        SELECT c.id
        FROM tmp_label_candidates c
        JOIN event_static e ON e.event_id = c.event_id
        WHERE c.in_coverage
        """
    )

    create_final_sql = text(
        """
        CREATE TEMP TABLE tmp_label_final ON COMMIT DROP AS
        SELECT
            c.id AS fire_detection_id,
            c.event_id,
            CASE
                WHEN p.id IS NOT NULL THEN 'POSITIVE'
                WHEN n.id IS NOT NULL THEN 'NEGATIVE'
                WHEN pp.id IS NOT NULL THEN 'PROBABLE_POSITIVE'
                ELSE 'UNKNOWN'
            END AS label,
            CASE
                WHEN p.id IS NULL AND n.id IS NULL AND pp.id IS NOT NULL THEN TRUE
                ELSE FALSE
            END AS weak_supervision
        FROM tmp_label_candidates c
        LEFT JOIN tmp_label_positive_ids p ON p.id = c.id
        LEFT JOIN tmp_label_negative_ids n ON n.id = c.id
        LEFT JOIN tmp_label_probable_positive_ids pp ON pp.id = c.id
        """
    )

    upsert_sql = text(
        """
        INSERT INTO denoiser_labels_v2 (
            fire_detection_id,
            event_id,
            label,
            rule_version,
            source,
            rule_params,
            weak_supervision,
            labeled_at
        )
        SELECT
            fire_detection_id,
            event_id,
            label,
            :rule_version,
            :source,
            CAST(:rule_params AS jsonb),
            weak_supervision,
            :labeled_at
        FROM tmp_label_final
        ON CONFLICT (fire_detection_id, rule_version) DO UPDATE SET
            event_id = EXCLUDED.event_id,
            label = EXCLUDED.label,
            source = EXCLUDED.source,
            rule_params = EXCLUDED.rule_params,
            weak_supervision = EXCLUDED.weak_supervision,
            labeled_at = EXCLUDED.labeled_at
        """
    )

    counts_sql = text(
        """
        SELECT label, COUNT(*) AS n
        FROM tmp_label_final
        GROUP BY label
        """
    )

    with engine.begin() as conn:
        started = time.perf_counter()
        conn.execute(create_candidates_sql, query_params)
        _log_step("label_v2.create_candidates", started)

        started = time.perf_counter()
        for stmt in create_candidates_indexes:
            conn.execute(stmt)
        _log_step("label_v2.candidate_indexes", started)

        total_rows = int(conn.execute(text("SELECT COUNT(*) FROM tmp_label_candidates")).scalar_one() or 0)
        if total_rows == 0:
            return {}

        started = time.perf_counter()
        conn.execute(create_chronic_cells_sql, query_params)
        conn.execute(create_chronic_idx_sql)
        _log_step("label_v2.chronic_cells", started)

        started = time.perf_counter()
        conn.execute(create_positive_sql, query_params)
        _log_step("label_v2.positive", started)

        started = time.perf_counter()
        if create_industrial_sources_sql is not None:
            conn.execute(create_industrial_sources_sql, query_params)
            for stmt in create_industrial_sources_indexes:
                conn.execute(stmt)
            _log_step("label_v2.industrial_sources_temp", started)
            started = time.perf_counter()
        conn.execute(create_industrial_sql, query_params)
        _log_step("label_v2.negative_industrial", started)

        started = time.perf_counter()
        conn.execute(create_far_low_sql, query_params)
        _log_step("label_v2.negative_far_low", started)

        started = time.perf_counter()
        conn.execute(create_chronic_ids_sql, query_params)
        _log_step("label_v2.negative_chronic", started)

        started = time.perf_counter()
        conn.execute(create_event_static_ids_sql, query_params)
        static_rows = int(conn.execute(text("SELECT COUNT(*) FROM tmp_label_event_static_ids")).scalar_one() or 0)
        _log_step("label_v2.negative_event_static", started, rows=static_rows)

        started = time.perf_counter()
        conn.execute(create_negative_sql, query_params)
        _log_step("label_v2.negative_union", started)

        started = time.perf_counter()
        conn.execute(create_probable_sql, query_params)
        _log_step("label_v2.probable_positive", started)

        started = time.perf_counter()
        conn.execute(create_final_sql)
        _log_step("label_v2.finalize_labels", started)

        started = time.perf_counter()
        upsert_result = conn.execute(upsert_sql, query_params)
        _log_step("label_v2.upsert", started, rows=int(upsert_result.rowcount or 0))

        rows = conn.execute(counts_sql).mappings().all()

    counts = {str(row["label"]): int(row["n"]) for row in rows}
    LOGGER.info("Label v2 counts: %s", counts)
    return counts


def label_detections_v2(
    engine: Engine,
    aoi_bbox: Tuple[float, float, float, float],
    start_time: datetime,
    end_time: datetime,
    *,
    rule_version: str = "v2_default",
    params: Optional[Dict] = None,
    chunk_days: int = 0,
    authority_profile: str = "wfigs_us",
    coverage_max_age_hours: float = 72.0,
    perimeter_source: str = "authoritative_perimeters",
    authoritative_tier: str = "both",
    industrial_policy_version: str | None = "global_authoritative_industrial_v1",
    industrial_negatives_global: bool = False,
) -> dict[str, int]:
    perimeter_source = str(perimeter_source).strip().lower()
    if perimeter_source not in _ALLOWED_PERIMETER_SOURCES:
        raise ValueError(
            f"perimeter_source must be one of {sorted(_ALLOWED_PERIMETER_SOURCES)}"
        )
    authoritative_tier = str(authoritative_tier).strip().lower()
    if authoritative_tier not in _ALLOWED_AUTHORITATIVE_TIERS:
        raise ValueError(
            f"authoritative_tier must be one of {sorted(_ALLOWED_AUTHORITATIVE_TIERS)}"
        )

    authority_profiles = _parse_authority_profiles(authority_profile)

    p = {**DEFAULT_PARAMS, **(params or {})}

    freshness_rows: list[dict] = []
    for profile in authority_profiles:
        freshness_rows.append(
            require_coverage_freshness(
                authority_profile=profile,
                max_age_hours=float(coverage_max_age_hours),
                engine=engine,
            )
        )
    freshness = freshness_rows[0]
    LOGGER.info(
        "Label v2 authority profiles=%s freshest_runs=%s",
        authority_profiles,
        [
            {
                "authority_profile": row.get("authority_profile"),
                "run_id": row.get("run_id"),
                "age_hours": row.get("age_hours"),
            }
            for row in freshness_rows
        ],
    )

    coverage_count = _check_perimeter_coverage(
        engine,
        aoi_bbox,
        start_time,
        end_time,
        authority_profiles=authority_profiles,
    )
    if coverage_count == 0:
        raise SystemExit(
            "No active perimeter coverage masks found for selected window and authority profile. "
            "Build perimeter_coverage_masks from authoritative source before labeling v2."
        )

    mask_ids = _active_coverage_mask_ids(
        engine,
        start_time,
        end_time,
        authority_profiles=authority_profiles,
    )
    LOGGER.info(
        "Label v2 coverage mask count=%s authority_profiles=%s",
        len(mask_ids),
        authority_profiles,
    )
    industrial_policy = _active_industrial_policy(
        engine,
        policy_version=industrial_policy_version,
    )
    if industrial_policy is None:
        raise SystemExit(
            "Missing active industrial mask policy for labeling v2. "
            "Run `make industrial-build-policy` before labeling."
        )
    LOGGER.info(
        "Label v2 industrial policy=%s strict_no_go=%s",
        industrial_policy.get("policy_version"),
        industrial_policy.get("strict_no_go"),
    )

    rule_params_payload: Dict[str, object] = {
        "label_params": p,
        "perimeter_source": perimeter_source,
        "authoritative_tier": authoritative_tier,
        "authority_profile": authority_profile,
        "authority_profiles": authority_profiles,
        "coverage_run_id": freshness.get("run_id"),
        "coverage_runs": {
            row.get("authority_profile"): row.get("run_id") for row in freshness_rows
        },
        "coverage_mask_ids": mask_ids,
        "industrial_policy_version": industrial_policy.get("policy_version"),
        "industrial_strict_no_go": bool(industrial_policy.get("strict_no_go")),
        "industrial_gold_buffer_m": float(_INDUSTRIAL_GOLD_BUFFER_M),
        "industrial_silver_buffer_m": float(_INDUSTRIAL_SILVER_BUFFER_M),
        "industrial_negatives_global": bool(industrial_negatives_global),
        "governance_filters": {
            "poly_featurestatus": ["Approved", "Certified"],
            "poly_featureaccess": "Public",
            "poly_isvisible": "Yes",
            "attr_isvalid": 1,
            "attr_isquarantined": 0,
        },
    }

    if int(chunk_days) <= 0:
        counts = _label_single_window(
            engine,
            aoi_bbox=aoi_bbox,
            start_time=start_time,
            end_time=end_time,
            rule_version=rule_version,
            params=p,
            authority_profile=authority_profile,
            authority_profiles=authority_profiles,
            perimeter_source=perimeter_source,
            authoritative_tier=authoritative_tier,
            rule_params_payload=rule_params_payload,
            industrial_policy=industrial_policy,
            industrial_negatives_global=industrial_negatives_global,
        )
        if not counts:
            raise SystemExit("No detections found in selected window for labeling v2.")
        return counts

    if start_time > end_time:
        raise ValueError("--start must be <= --end")

    totals: dict[str, int] = {}
    chunk_delta = timedelta(days=int(chunk_days))
    cursor = start_time
    chunks = 0

    while cursor <= end_time:
        chunk_end = min(end_time, cursor + chunk_delta)
        LOGGER.info("Label v2 chunk window: %s -> %s", cursor.isoformat(), chunk_end.isoformat())

        counts = _label_single_window(
            engine,
            aoi_bbox=aoi_bbox,
            start_time=cursor,
            end_time=chunk_end,
            rule_version=rule_version,
            params=p,
            authority_profile=authority_profile,
            authority_profiles=authority_profiles,
            perimeter_source=perimeter_source,
            authoritative_tier=authoritative_tier,
            rule_params_payload=rule_params_payload,
            industrial_policy=industrial_policy,
            industrial_negatives_global=industrial_negatives_global,
        )
        if counts:
            for key, value in counts.items():
                totals[key] = totals.get(key, 0) + int(value)

        chunks += 1
        if chunk_end >= end_time:
            break
        cursor = chunk_end + timedelta(microseconds=1)

    if not totals:
        raise SystemExit("No detections found in selected window for labeling v2.")

    totals["_chunks"] = chunks
    LOGGER.info("Label v2 chunked totals: %s", totals)
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(description="Label detections for denoiser v2.")
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        default=list(_DEFAULT_GLOBAL_BBOX),
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        help="Optional AOI bounding box. Defaults to global extent.",
    )
    parser.add_argument("--start", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--version", type=str, default="v2_default")
    parser.add_argument("--chunk-days", type=int, default=0, help="Optional chunking window in days")
    parser.add_argument("--authority-profile", type=str, default="wfigs_us")
    parser.add_argument("--coverage-max-age-hours", type=float, default=72.0)
    parser.add_argument(
        "--perimeter-source",
        type=str,
        default="authoritative_perimeters",
        choices=sorted(_ALLOWED_PERIMETER_SOURCES),
    )
    parser.add_argument(
        "--authoritative-tier",
        type=str,
        default="both",
        choices=sorted(_ALLOWED_AUTHORITATIVE_TIERS),
    )
    parser.add_argument(
        "--industrial-policy-version",
        type=str,
        default="global_authoritative_industrial_v1",
        help="Industrial masking policy version used for policy-aligned negatives.",
    )
    parser.add_argument(
        "--industrial-negatives-global",
        action="store_true",
        help=(
            "Apply authoritative industrial-negative labeling outside perimeter-covered scope. "
            "Perimeter-positive logic remains covered-first."
        ),
    )
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d") + timedelta(days=1)

    counts = label_detections_v2(
        get_engine(),
        tuple(args.bbox),
        start,
        end,
        rule_version=args.version,
        chunk_days=args.chunk_days,
        authority_profile=args.authority_profile,
        coverage_max_age_hours=args.coverage_max_age_hours,
        perimeter_source=args.perimeter_source,
        authoritative_tier=args.authoritative_tier,
        industrial_policy_version=args.industrial_policy_version,
        industrial_negatives_global=args.industrial_negatives_global,
    )
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
