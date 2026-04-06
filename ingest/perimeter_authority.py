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
