"""Perimeter authority tier ranking and conflict resolution.

Defines the canonical authority tier hierarchy for fire perimeter sources
and provides helpers for conflict-aware upsert logic.  A lower numeric rank
means higher authority -- a source may only overwrite a record whose existing
authority rank is >= its own.

Tier semantics (matching existing ``authoritative_perimeters.tier``):

    gold   (1) -- NIFC official / WFIGS FFP-FODR certified / CWFIS NBAC
    silver (2) -- WFIGS approved+visible / Copernicus EMS certified
    bronze (3) -- WFIGS non-certified / other provisional sources
    blocked(4) -- quarantined records (never authoritative)
"""

from __future__ import annotations

import logging
from typing import Any

LOGGER = logging.getLogger("perimeter_authority")

# Canonical tier-to-rank mapping.  Lower rank = higher authority.
TIER_RANK: dict[str, int] = {
    "gold": 1,
    "silver": 2,
    "bronze": 3,
    "blocked": 4,
}

# Default rank for records that have no tier set (treated as lowest authority).
DEFAULT_RANK = 99

# Source-to-tier mapping for the legacy ``fire_perimeters`` table, which does
# not have an inherent tier column.
FIRE_PERIMETERS_SOURCE_TIER: dict[str, str] = {
    "NIFC": "gold",
}


def tier_rank(tier: str | None) -> int:
    """Return the numeric rank for a tier label (lower = higher authority)."""
    if tier is None:
        return DEFAULT_RANK
    return TIER_RANK.get(tier.strip().lower(), DEFAULT_RANK)


def should_overwrite(incoming_tier: str | None, existing_tier: str | None) -> bool:
    """Return True if the incoming tier has equal or higher authority.

    Equal authority is allowed so that the same source can update its own
    records (e.g. NIFC re-publishing a corrected perimeter).
    """
    return tier_rank(incoming_tier) <= tier_rank(existing_tier)


def log_authority_conflict(
    *,
    source: str,
    source_id: str,
    incoming_tier: str,
    existing_tier: str,
    extra: dict[str, Any] | None = None,
) -> None:
    """Emit a WARNING when a lower-authority source attempts to overwrite."""
    LOGGER.warning(
        "Authority conflict: source=%s source_id=%s attempted tier=%s "
        "but existing record has higher-authority tier=%s -- skipping overwrite",
        source,
        source_id,
        incoming_tier,
        existing_tier,
    )


def record_authority_conflict(
    conn: Any,
    *,
    table_name: str,
    source: str,
    source_id: str,
    incoming_tier: str,
    existing_tier: str,
    outcome: str,
    run_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    """Insert a row into the ``perimeter_authority_conflicts`` audit table."""
    from sqlalchemy import text as _text

    stmt = _text("""
        INSERT INTO perimeter_authority_conflicts
            (table_name, source, source_id, incoming_tier, existing_tier,
             outcome, run_id, details)
        VALUES
            (:table_name, :source, :source_id, :incoming_tier, :existing_tier,
             :outcome, :run_id, CAST(:details AS json))
    """)
    import json as _json

    conn.execute(stmt, {
        "table_name": table_name,
        "source": source,
        "source_id": source_id,
        "incoming_tier": incoming_tier,
        "existing_tier": existing_tier,
        "outcome": outcome,
        "run_id": run_id,
        "details": _json.dumps(details) if details else None,
    })


def authority_aware_upsert(
    conn: Any,
    *,
    insert_stmt: Any,
    rows: list[dict[str, Any]],
    table_name: str = "authoritative_perimeters",
    source_label: str = "unknown",
    logger: logging.Logger | None = None,
) -> tuple[int, int]:
    """Filter *rows* by authority tier then execute *insert_stmt* for accepted rows.

    For each row the function checks the existing tier in *table_name*.  If the
    incoming tier does not have equal or higher authority the row is rejected and
    an audit record is written to ``perimeter_authority_conflicts``.

    Note: SELECT-then-INSERT has no row-level lock.  Safe because ingest jobs
    are serialized by the orchestrator.  If ingest is ever parallelized, add
    ``SELECT ... FOR UPDATE``.

    Parameters
    ----------
    conn:
        An active SQLAlchemy connection (inside a transaction).
    insert_stmt:
        A ``sqlalchemy.text`` INSERT ... ON CONFLICT statement.
    rows:
        Dicts that must contain ``source_profile``, ``source_layer``,
        ``source_object_id``, ``tier``, and optionally ``run_id``.
    table_name:
        The target table name used for the tier lookup and audit trail.
    source_label:
        Human-readable label used in log messages (e.g. "WFIGS", "CWFIS").
    logger:
        Logger instance; falls back to the module-level ``LOGGER``.

    Returns
    -------
    (upserted, authority_rejected)
    """
    from sqlalchemy import text as _text

    _log = logger or LOGGER

    existing_tier_stmt = _text(f"""
        SELECT tier FROM {table_name}
        WHERE source_profile = :source_profile
          AND source_layer = :source_layer
          AND source_object_id = :source_object_id
    """)

    accepted: list[dict[str, Any]] = []
    authority_rejected = 0

    for row in rows:
        result = conn.execute(
            existing_tier_stmt,
            {
                "source_profile": row["source_profile"],
                "source_layer": row["source_layer"],
                "source_object_id": row["source_object_id"],
            },
        ).fetchone()
        existing_tier = result[0] if result else None

        if existing_tier is not None and not should_overwrite(
            row["tier"], existing_tier
        ):
            authority_rejected += 1
            log_authority_conflict(
                source=row["source_profile"],
                source_id=row["source_object_id"],
                incoming_tier=row["tier"],
                existing_tier=existing_tier,
            )
            record_authority_conflict(
                conn,
                table_name=table_name,
                source=row["source_profile"],
                source_id=row["source_object_id"],
                incoming_tier=row["tier"],
                existing_tier=existing_tier,
                outcome="rejected",
                run_id=row.get("run_id"),
            )
            continue

        if existing_tier is not None:
            record_authority_conflict(
                conn,
                table_name=table_name,
                source=row["source_profile"],
                source_id=row["source_object_id"],
                incoming_tier=row["tier"],
                existing_tier=existing_tier,
                outcome="accepted",
                run_id=row.get("run_id"),
            )
        accepted.append(row)

    upserted = 0
    if accepted:
        result = conn.execute(insert_stmt, accepted)
        upserted = int(result.rowcount or 0)

    if authority_rejected:
        _log.warning(
            "%s authority conflicts: %d records rejected (lower authority).",
            source_label,
            authority_rejected,
        )

    return upserted, authority_rejected
