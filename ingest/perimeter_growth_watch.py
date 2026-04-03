"""Perimeter growth delta watch.

Detects rapid size increases in official fire perimeters by comparing the two
most recent perimeter records for each fire.

Thresholds use size-tiered dual gates: alert fires on EITHER the absolute OR
the percentage threshold, whichever is lower for the fire's current size.
Critical takes precedence when both levels are exceeded.

Tier table (max_current_acres, warn_pct, crit_pct, warn_abs, crit_abs):
  - Small  (<  1,000 acres): 10%/25%, 50/200 acres absolute
  - Medium ( 10,000 acres): 7%/15%, 200/750 acres absolute
  - Large  (>  10,000 acres): 5%/10%, 500/2,000 acres absolute
"""

from __future__ import annotations

import logging
import re
from typing import Any

from api.notifications import notify

LOGGER = logging.getLogger(__name__)

# Thresholds: (max_current_acres, warn_pct, crit_pct, warn_abs, crit_abs)
# Checked in order; first matching tier applies.
_GROWTH_THRESHOLDS: list[tuple[float, float, float, float, float]] = [
    # small fires  (< 1,000 acres)
    (1_000.0,      10.0, 25.0,   50.0,   200.0),
    # medium fires (1,000 – 10,000 acres)
    (10_000.0,      7.0, 15.0,  200.0,   750.0),
    # large fires  (> 10,000 acres)
    (float("inf"),  5.0, 10.0,  500.0, 2_000.0),
]

_ALL_SOURCES = ["nifc_wfigs", "cwfis", "copernicus_ems"]


def _slugify(name: str) -> str:
    """Return a notification-safe slug from a fire name."""
    slug = name.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    return slug.strip("_")


def _determine_severity(
    current_acres: float,
    abs_growth: float,
    growth_pct: float,
) -> str | None:
    """Return "critical", "warning", or None for the given growth metrics.

    Looks up the size tier by *current_acres*, then fires on EITHER the
    absolute OR the percentage gate (whichever is lower for this fire size).
    Critical takes precedence over warning.

    Args:
        current_acres: Most-recent perimeter size in acres.
        abs_growth:    Absolute acreage increase (current − previous).
        growth_pct:    Percentage increase relative to previous size.

    Returns:
        "critical", "warning", or None.
    """
    warn_pct = crit_pct = warn_abs = crit_abs = None

    for max_acres, wp, cp, wa, ca in _GROWTH_THRESHOLDS:
        if current_acres <= max_acres:
            warn_pct, crit_pct, warn_abs, crit_abs = wp, cp, wa, ca
            break

    if warn_pct is None:
        # Fallback: use largest-fire tier (should never happen given float("inf"))
        _, warn_pct, crit_pct, warn_abs, crit_abs = _GROWTH_THRESHOLDS[-1]

    is_crit = growth_pct >= crit_pct or abs_growth >= crit_abs
    is_warn = growth_pct >= warn_pct or abs_growth >= warn_abs

    if is_crit:
        return "critical"
    if is_warn:
        return "warning"
    return None


def check_perimeter_growth(source: str, session: Any) -> list[dict[str, Any]]:
    """Check for significant perimeter growth for all fires from *source*.

    For each fire with at least two perimeter records (same fire_name, ordered
    by created_at DESC), computes growth from the second-most-recent to the
    most-recent and fires a notification if thresholds are met.

    Args:
        source:  Perimeter source string, e.g. "nifc_wfigs".
        session: SQLAlchemy session.

    Returns:
        List of growth result dicts for fires that triggered an alert.
    """
    results: list[dict[str, Any]] = []

    try:
        from sqlalchemy import text  # local import to match ingest style

        # Retrieve the two most-recent perimeters per fire_name for this source.
        # We use a window function to rank rows within each fire group.
        sql = """
            WITH ranked AS (
                SELECT
                    fire_name,
                    source_id,
                    acres,
                    created_at,
                    ROW_NUMBER() OVER (
                        PARTITION BY fire_name
                        ORDER BY created_at DESC
                    ) AS rn
                FROM fire_perimeters
                WHERE source = :source
                  AND fire_end IS NULL
                  AND acres IS NOT NULL
            )
            SELECT
                cur.fire_name,
                cur.source_id       AS current_source_id,
                cur.acres           AS current_acres,
                cur.created_at      AS current_created_at,
                prev.acres          AS prev_acres,
                prev.created_at     AS prev_created_at
            FROM ranked cur
            JOIN ranked prev
              ON cur.fire_name = prev.fire_name
             AND cur.rn = 1
             AND prev.rn = 2
        """
        rows = session.execute(text(sql), {"source": source}).fetchall()

        if not rows:
            LOGGER.info(
                "perimeter_growth_watch: no fire pairs found for source=%s", source
            )
            return results

        LOGGER.info(
            "perimeter_growth_watch: checking %d fire pair(s) for source=%s",
            len(rows), source,
        )

        for row in rows:
            fire_name: str = row.fire_name
            prev_acres: float = float(row.prev_acres)
            current_acres: float = float(row.current_acres)

            if prev_acres <= 0:
                LOGGER.info(
                    "perimeter_growth_watch: skipping %s — prev_acres=%.1f",
                    fire_name, prev_acres,
                )
                continue

            growth_pct = (current_acres - prev_acres) / prev_acres * 100.0
            abs_growth = current_acres - prev_acres

            severity = _determine_severity(current_acres, abs_growth, growth_pct)

            if severity is None:
                LOGGER.info(
                    "perimeter_growth_watch: %s growth=%.1f%% abs=%.0f acres — below threshold",
                    fire_name, growth_pct, abs_growth,
                )
                continue

            fire_name_slug = _slugify(fire_name)
            event_type = f"perimeter_growth:{source}:{fire_name_slug}"

            LOGGER.warning(
                "perimeter_growth_watch: %s grew %.1f%% (%.0f → %.0f acres) severity=%s",
                fire_name, growth_pct, prev_acres, current_acres, severity,
            )

            notify(
                event_type,
                title=f"{fire_name} perimeter grew {growth_pct:.0f}%",
                body=(
                    f"Official perimeter grew from {prev_acres:.0f} to "
                    f"{current_acres:.0f} acres "
                    f"(+{abs_growth:,.0f} acres, {growth_pct:+.0f}%) since last update."
                ),
                severity=severity,
                aoi_id=None,
                fire_name=fire_name,
                prev_acres=prev_acres,
                current_acres=current_acres,
                growth_pct=growth_pct,
                abs_growth=abs_growth,
            )

            results.append(
                {
                    "fire_name": fire_name,
                    "source": source,
                    "prev_acres": prev_acres,
                    "current_acres": current_acres,
                    "growth_pct": growth_pct,
                    "abs_growth": abs_growth,
                    "severity": severity,
                }
            )

    except Exception:
        LOGGER.exception(
            "perimeter_growth_watch: error checking source=%s", source
        )

    return results


def run_perimeter_growth_checks(session: Any) -> list[dict[str, Any]]:
    """Run perimeter growth checks across all known sources.

    Sources checked: nifc_wfigs, cwfis, copernicus_ems.

    Returns:
        Flat list of growth result dicts from all sources combined.
    """
    all_results: list[dict[str, Any]] = []
    for source in _ALL_SOURCES:
        results = check_perimeter_growth(source, session)
        all_results.extend(results)

    LOGGER.info(
        "perimeter_growth_watch: %d growth alert(s) across %d source(s)",
        len(all_results), len(_ALL_SOURCES),
    )
    return all_results
