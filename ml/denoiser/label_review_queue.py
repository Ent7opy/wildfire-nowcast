"""Ingest resolved operator review queue items as denoiser_labels_v2 training rows.

The review queue is event-centric — each queue row has an event_id but no
fire_detection_id.  To produce per-detection training labels we join through
fire_event_memberships, emitting one label row per detection that is a member
of the resolved event.

Label mapping:
    resolved_notes = 'confirmed_fire'  →  label = 'POSITIVE'
    resolved_notes = 'marked_noise'    →  label = 'NEGATIVE'

Auto-resolved items (resolved_by LIKE 'auto:%') are excluded — they are already
covered by the authoritative perimeter labeling pass.

Conflict rule: if a detection already carries a non-weak-supervision perimeter
label that disagrees with the operator label, the perimeter wins.  The conflict
is logged to label_conflicts for QA, and the operator label is NOT inserted.

The rule_version for review-queue labels is a fixed constant
('review_queue_v1'), separate from perimeter-based rule versions so both can
coexist in the table for the same detection.

Implementation note: conflict detection and label insertion are done in bulk
SQL rather than row-by-row Python loops to avoid O(N) round-trips.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

from sqlalchemy import text
from sqlalchemy.engine import Engine

LOGGER = logging.getLogger("denoiser_label_review_queue")

REVIEW_QUEUE_RULE_VERSION = "review_queue_v1"
_DEFAULT_LABEL_WEIGHT = 0.8
_VALID_RESOLVED_NOTES = frozenset({"confirmed_fire", "marked_noise"})
_NOTES_TO_LABEL = {"confirmed_fire": "POSITIVE", "marked_noise": "NEGATIVE"}


def _get_label_weight() -> float:
    try:
        return float(os.environ.get("REVIEW_QUEUE_LABEL_WEIGHT", str(_DEFAULT_LABEL_WEIGHT)))
    except ValueError:
        LOGGER.warning(
            "Invalid REVIEW_QUEUE_LABEL_WEIGHT env value; using default %.1f",
            _DEFAULT_LABEL_WEIGHT,
        )
        return _DEFAULT_LABEL_WEIGHT


def warn_if_unprocessed_labels_exist(engine: Engine) -> int:
    """Return the count of resolved operator queue items not yet in denoiser_labels_v2.

    Logs a WARNING if any exist and REVIEW_QUEUE_FEEDBACK_ENABLED is not set.
    Used in label_v2.py main() to surface the gap during ops review.
    """
    stmt = text(
        """
        SELECT COUNT(DISTINCT rq.id) AS n
        FROM denoiser_review_queue rq
        JOIN fire_event_memberships fem ON fem.event_id = rq.event_id
        WHERE rq.status = 'resolved'
          AND rq.resolved_notes IN ('confirmed_fire', 'marked_noise')
          AND rq.resolved_by NOT LIKE 'auto:%'
          AND NOT EXISTS (
              SELECT 1
              FROM denoiser_labels_v2 dl
              WHERE dl.fire_detection_id = fem.fire_detection_id
                AND dl.source = 'review_queue'
          )
        """
    )
    with engine.connect() as conn:
        row = conn.execute(stmt).fetchone()
    count = int(row["n"]) if row else 0
    if count > 0:
        LOGGER.warning(
            "WARNING [target: science_grade]: %d resolved review queue event(s) have not been "
            "ingested as training labels. Set REVIEW_QUEUE_FEEDBACK_ENABLED=true to close the "
            "learning loop.",
            count,
        )
    return count


def ingest_review_queue_labels(
    engine: Engine,
    *,
    label_weight: float | None = None,
) -> dict[str, int]:
    """Ingest resolved operator review queue items as denoiser_labels_v2 rows.

    Returns counts: {"inserted": n, "conflicts": n, "skipped_invalid_notes": n}.

    Uses bulk SQL to avoid per-row round-trips:
      1. One query fetches all candidates joined with any conflicting perimeter label.
      2. One batch INSERT logs conflicts to label_conflicts.
      3. One batch INSERT/ON CONFLICT upserts clean labels into denoiser_labels_v2.
    """
    if label_weight is None:
        label_weight = _get_label_weight()

    labeled_at = datetime.now(timezone.utc)

    # -- Step 1: fetch all eligible candidates in one query.
    #
    # Joins fire_event_memberships to expand event → detections.
    # LEFT JOINs denoiser_labels_v2 to detect conflicts in the same pass —
    # a conflict exists when an authoritative (non-weak-supervision) perimeter
    # label disagrees with the operator's resolved_notes.
    #
    # Columns returned:
    #   queue_id, event_id, resolved_by, resolved_notes, fire_detection_id,
    #   existing_perimeter_label (NULL if no conflict source exists)
    fetch_stmt = text(
        """
        SELECT
            rq.id                AS queue_id,
            rq.event_id,
            rq.resolved_by,
            rq.resolved_notes,
            fem.fire_detection_id,
            existing.label       AS existing_perimeter_label
        FROM denoiser_review_queue rq
        JOIN fire_event_memberships fem ON fem.event_id = rq.event_id
        LEFT JOIN LATERAL (
            SELECT label
            FROM denoiser_labels_v2
            WHERE fire_detection_id = fem.fire_detection_id
              AND source LIKE 'ground_truth_v2%'
              AND weak_supervision = false
            LIMIT 1
        ) existing ON true
        WHERE rq.status = 'resolved'
          AND rq.resolved_notes IN ('confirmed_fire', 'marked_noise')
          AND rq.resolved_by NOT LIKE 'auto:%'
          AND NOT EXISTS (
              SELECT 1
              FROM denoiser_labels_v2 dl
              WHERE dl.fire_detection_id = fem.fire_detection_id
                AND dl.source = 'review_queue'
          )
        """
    )

    # -- Step 2: batch insert conflicts into label_conflicts.
    conflict_insert_stmt = text(
        """
        INSERT INTO label_conflicts
            (event_id, fire_detection_id, perimeter_label, operator_label, resolved_by, created_at)
        VALUES
            (:event_id, :fire_detection_id, :perimeter_label, :operator_label, :resolved_by, now())
        """
    )

    # -- Step 3: batch upsert clean labels into denoiser_labels_v2.
    label_insert_stmt = text(
        """
        INSERT INTO denoiser_labels_v2
            (fire_detection_id, event_id, label, rule_version, source,
             rule_params, weak_supervision, labeled_at, label_weight)
        VALUES
            (:fire_detection_id, :event_id, :label, :rule_version, 'review_queue',
             CAST(:rule_params AS jsonb), false, :labeled_at, :label_weight)
        ON CONFLICT (fire_detection_id, rule_version) DO UPDATE SET
            event_id       = EXCLUDED.event_id,
            label          = EXCLUDED.label,
            source         = EXCLUDED.source,
            rule_params    = EXCLUDED.rule_params,
            weak_supervision = EXCLUDED.weak_supervision,
            labeled_at     = EXCLUDED.labeled_at,
            label_weight   = EXCLUDED.label_weight
        """
    )

    conflict_params: list[dict] = []
    label_params: list[dict] = []
    skipped_invalid = 0

    with engine.begin() as conn:
        candidates = conn.execute(fetch_stmt).fetchall()

        for row in candidates:
            notes = (row["resolved_notes"] or "").strip().lower()
            if notes not in _VALID_RESOLVED_NOTES:
                LOGGER.debug(
                    "Skipping queue_id=%s: unrecognised resolved_notes=%r",
                    row["queue_id"],
                    row["resolved_notes"],
                )
                skipped_invalid += 1
                continue

            operator_label = _NOTES_TO_LABEL[notes]
            existing_label = row["existing_perimeter_label"]

            if existing_label is not None and existing_label != operator_label:
                LOGGER.warning(
                    "Label conflict: fire_detection_id=%s event_id=%s "
                    "perimeter_label=%r operator_label=%r resolved_by=%r — perimeter wins",
                    row["fire_detection_id"],
                    row["event_id"],
                    existing_label,
                    operator_label,
                    row["resolved_by"],
                )
                conflict_params.append(
                    {
                        "event_id": row["event_id"],
                        "fire_detection_id": row["fire_detection_id"],
                        "perimeter_label": existing_label,
                        "operator_label": operator_label,
                        "resolved_by": row["resolved_by"],
                    }
                )
            else:
                label_params.append(
                    {
                        "fire_detection_id": row["fire_detection_id"],
                        "event_id": row["event_id"],
                        "label": operator_label,
                        "rule_version": REVIEW_QUEUE_RULE_VERSION,
                        "rule_params": json.dumps(
                            {
                                "resolved_by": row["resolved_by"],
                                "resolved_notes": row["resolved_notes"],
                                "review_queue_id": row["queue_id"],
                            }
                        ),
                        "labeled_at": labeled_at,
                        "label_weight": label_weight,
                    }
                )

        if conflict_params:
            conn.execute(conflict_insert_stmt, conflict_params)

        if label_params:
            conn.execute(label_insert_stmt, label_params)

    counts = {
        "inserted": len(label_params),
        "conflicts": len(conflict_params),
        "skipped_invalid_notes": skipped_invalid,
    }
    LOGGER.info(
        "Review queue label ingestion complete: inserted=%d conflicts=%d skipped_invalid=%d",
        counts["inserted"],
        counts["conflicts"],
        counts["skipped_invalid_notes"],
    )
    return counts
