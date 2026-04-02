"""Weekly operator accuracy computation for denoiser review queue labels.

Evaluates per-operator (by resolved_by value) label quality by checking how
many of their 'confirmed_fire' / 'marked_noise' decisions are later corroborated
by authoritative perimeters ingested after the resolution was made.

Accuracy definitions:
    fire_accuracy  = confirmed_fires later covered by an authoritative perimeter
                     / total confirmed_fire labels made by this operator
    noise_accuracy = marked_noise items NOT covered by any perimeter within the
                     lookback window / total marked_noise labels

Derived label_weight = min(1.0, max(0.5, mean(fire_accuracy, noise_accuracy)))
    - Perfect operator → weight 1.0
    - Randomly labelling operator → weight ~0.5 (floor)

Currently the UI sends resolved_by='operator' for all resolutions, so this
produces a single aggregate row.  Individual weighting is available once auth
is added (out of scope).

NOTE: accuracy is retrospective — perimeters can arrive days to weeks after an
operator decision.  The job should be run weekly; early runs will underestimate
accuracy for recently resolved items.  Accuracy improves over time as more
perimeters are ingested.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine

LOGGER = logging.getLogger("denoiser_operator_accuracy")

# Spatial buffer for "covered by perimeter" check — matches label_v2 default.
_POSITIVE_BUFFER_M = 2315.0
_POSITIVE_BUFFER_DEG = _POSITIVE_BUFFER_M / 111_320.0  # rough degree conversion for bbox index

# Require the perimeter to have been ingested after the operator label, but allow
# a grace window so very recent labels aren't marked inaccurate prematurely.
_PERIMETER_GRACE_DAYS = 14  # ignore labels from the last N days when computing accuracy


def compute_operator_accuracy(engine: Engine) -> dict[str, Any]:
    """Compute per-operator accuracy and upsert into operator_label_quality.

    Returns a summary dict with per-operator stats for logging.
    """
    now = datetime.now(timezone.utc)

    # -- Fire accuracy: confirmed_fire labels where detection later fell inside a perimeter
    fire_accuracy_stmt = text(
        f"""
        WITH operator_fire AS (
            SELECT
                dl.rule_params->>'resolved_by'   AS resolved_by,
                dl.rule_params->>'review_queue_id' AS queue_id,
                dl.fire_detection_id,
                dl.labeled_at
            FROM denoiser_labels_v2 dl
            WHERE dl.source = 'review_queue'
              AND dl.label = 'POSITIVE'
              AND dl.labeled_at < NOW() - INTERVAL '{_PERIMETER_GRACE_DAYS} days'
        ),
        covered AS (
            SELECT DISTINCT of.resolved_by, of.fire_detection_id
            FROM operator_fire of
            JOIN fire_detections fd ON fd.id = of.fire_detection_id
            JOIN authoritative_perimeters ap
              ON ap.geom && ST_Expand(fd.geom, :buffer_deg)
             AND ST_DWithin(fd.geom::geography, ap.geom::geography, :buffer_m)
             AND ap.created_at > of.labeled_at
        )
        SELECT
            of.resolved_by,
            COUNT(DISTINCT of.fire_detection_id)   AS fire_label_count,
            COUNT(DISTINCT c.fire_detection_id)     AS fire_correct_count
        FROM operator_fire of
        LEFT JOIN covered c
          ON c.resolved_by = of.resolved_by
         AND c.fire_detection_id = of.fire_detection_id
        GROUP BY of.resolved_by
        """
    )

    noise_accuracy_stmt = text(
        f"""
        WITH operator_noise AS (
            SELECT
                dl.rule_params->>'resolved_by'   AS resolved_by,
                dl.fire_detection_id,
                dl.labeled_at
            FROM denoiser_labels_v2 dl
            WHERE dl.source = 'review_queue'
              AND dl.label = 'NEGATIVE'
              AND dl.labeled_at < NOW() - INTERVAL '{_PERIMETER_GRACE_DAYS} days'
        ),
        covered AS (
            SELECT DISTINCT of.resolved_by, of.fire_detection_id
            FROM operator_noise of
            JOIN fire_detections fd ON fd.id = of.fire_detection_id
            JOIN authoritative_perimeters ap
              ON ap.geom && ST_Expand(fd.geom, :buffer_deg)
             AND ST_DWithin(fd.geom::geography, ap.geom::geography, :buffer_m)
             AND ap.created_at > of.labeled_at
        )
        SELECT
            of.resolved_by,
            COUNT(DISTINCT of.fire_detection_id)   AS noise_label_count,
            -- Correct noise = NOT covered (operator correctly said it's noise)
            COUNT(DISTINCT of.fire_detection_id)
              FILTER (WHERE c.fire_detection_id IS NULL) AS noise_correct_count
        FROM operator_noise of
        LEFT JOIN covered c
          ON c.resolved_by = of.resolved_by
         AND c.fire_detection_id = of.fire_detection_id
        GROUP BY of.resolved_by
        """
    )

    upsert_stmt = text(
        """
        INSERT INTO operator_label_quality
            (resolved_by, fire_label_count, fire_correct_count,
             noise_label_count, noise_correct_count,
             fire_accuracy, noise_accuracy, label_weight, computed_at)
        VALUES
            (:resolved_by, :fire_label_count, :fire_correct_count,
             :noise_label_count, :noise_correct_count,
             :fire_accuracy, :noise_accuracy, :label_weight, :computed_at)
        ON CONFLICT (resolved_by) DO UPDATE SET
            fire_label_count    = EXCLUDED.fire_label_count,
            fire_correct_count  = EXCLUDED.fire_correct_count,
            noise_label_count   = EXCLUDED.noise_label_count,
            noise_correct_count = EXCLUDED.noise_correct_count,
            fire_accuracy       = EXCLUDED.fire_accuracy,
            noise_accuracy      = EXCLUDED.noise_accuracy,
            label_weight        = EXCLUDED.label_weight,
            computed_at         = EXCLUDED.computed_at
        """
    )

    params = {
        "buffer_m": _POSITIVE_BUFFER_M,
        "buffer_deg": _POSITIVE_BUFFER_DEG,
    }

    results: dict[str, Any] = {"operators": [], "computed_at": now.isoformat()}

    with engine.begin() as conn:
        fire_rows = {
            row["resolved_by"]: dict(row._mapping)
            for row in conn.execute(fire_accuracy_stmt, params).fetchall()
        }
        noise_rows = {
            row["resolved_by"]: dict(row._mapping)
            for row in conn.execute(noise_accuracy_stmt, params).fetchall()
        }

        all_operators = set(fire_rows) | set(noise_rows)

        for operator in all_operators:
            fire = fire_rows.get(operator, {})
            noise = noise_rows.get(operator, {})

            fire_count = int(fire.get("fire_label_count", 0))
            fire_correct = int(fire.get("fire_correct_count", 0))
            noise_count = int(noise.get("noise_label_count", 0))
            noise_correct = int(noise.get("noise_correct_count", 0))

            fire_accuracy = fire_correct / fire_count if fire_count > 0 else None
            noise_accuracy = noise_correct / noise_count if noise_count > 0 else None

            # Derive label weight: average of available accuracy values, clamped to [0.5, 1.0]
            available = [a for a in (fire_accuracy, noise_accuracy) if a is not None]
            if available:
                avg_accuracy = sum(available) / len(available)
                label_weight = min(1.0, max(0.5, avg_accuracy))
            else:
                label_weight = 1.0  # no data yet → assume full weight

            conn.execute(
                upsert_stmt,
                {
                    "resolved_by": operator,
                    "fire_label_count": fire_count,
                    "fire_correct_count": fire_correct,
                    "noise_label_count": noise_count,
                    "noise_correct_count": noise_correct,
                    "fire_accuracy": fire_accuracy,
                    "noise_accuracy": noise_accuracy,
                    "label_weight": label_weight,
                    "computed_at": now,
                },
            )

            LOGGER.info(
                "Operator accuracy: resolved_by=%r fire_acc=%s noise_acc=%s "
                "label_weight=%.3f (fire=%d/%d noise=%d/%d)",
                operator,
                f"{fire_accuracy:.3f}" if fire_accuracy is not None else "n/a",
                f"{noise_accuracy:.3f}" if noise_accuracy is not None else "n/a",
                label_weight,
                fire_correct,
                fire_count,
                noise_correct,
                noise_count,
            )

            results["operators"].append(
                {
                    "resolved_by": operator,
                    "fire_label_count": fire_count,
                    "fire_correct_count": fire_correct,
                    "noise_label_count": noise_count,
                    "noise_correct_count": noise_correct,
                    "fire_accuracy": fire_accuracy,
                    "noise_accuracy": noise_accuracy,
                    "label_weight": label_weight,
                }
            )

    return results
