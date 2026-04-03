"""NIFC National Preparedness Level (PL) fetcher.

Provides the current national resource Preparedness Level (1-5) for inclusion
in critical-severity notifications. Uses a two-tier env-var strategy since
NIFC does not publish a stable machine-readable PL API.

# STAGE-GAP WARNING (mvp_operational → science_grade): get_preparedness_level()
# currently relies on operator-configured env vars (NIFC_PL_URL, NIFC_PL_OVERRIDE)
# rather than a live authoritative feed. Mitigation: operators should set
# NIFC_PL_OVERRIDE during high-PL events. Target stage: integrate with NIFC
# official machine-readable PL feed when published.

Environment variables:
    NIFC_PL_URL              Optional URL that returns JSON with a
                             ``preparedness_level`` or ``pl`` integer field.
                             If set and reachable, this is the primary source.
    NIFC_PL_OVERRIDE         Integer 1-5 set manually by operators during
                             high-PL incidents when no API is available.
    NIFC_PL_CACHE_TTL_SECONDS  Cache lifetime in seconds (default: 1800 / 30 min).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

LOGGER = logging.getLogger(__name__)

_CACHE_TTL_SECONDS: int = int(os.getenv("NIFC_PL_CACHE_TTL_SECONDS", "1800"))

# In-memory cache: (fetched_at, preparedness_level)
_cache: tuple[datetime, int | None] | None = None

_VALID_PL_RANGE = range(1, 6)  # 1-5 inclusive


def _fetch_from_url(http_client: httpx.Client) -> int | None:
    """Attempt to fetch PL from NIFC_PL_URL. Returns None on any failure."""
    url = os.getenv("NIFC_PL_URL", "").strip()
    if not url:
        return None
    try:
        resp = http_client.get(url, timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
        raw = data.get("preparedness_level") or data.get("pl")
        if raw is None:
            LOGGER.warning(
                "nifc_pl_watch: NIFC_PL_URL response missing 'preparedness_level'/'pl' field"
            )
            return None
        pl = int(raw)
        if pl not in _VALID_PL_RANGE:
            LOGGER.warning(
                "nifc_pl_watch: NIFC_PL_URL returned out-of-range PL=%d (expected 1-5)", pl
            )
            return None
        return pl
    except httpx.HTTPError as exc:
        LOGGER.warning("nifc_pl_watch: NIFC_PL_URL fetch failed (%s) — falling back", exc)
        return None
    except Exception as exc:
        LOGGER.warning("nifc_pl_watch: NIFC_PL_URL parse error (%s) — falling back", exc)
        return None


def _read_override() -> int | None:
    """Read NIFC_PL_OVERRIDE env var. Returns None if unset or invalid."""
    raw = os.getenv("NIFC_PL_OVERRIDE", "").strip()
    if not raw:
        return None
    try:
        pl = int(raw)
    except ValueError:
        LOGGER.warning(
            "nifc_pl_watch: NIFC_PL_OVERRIDE=%r is not a valid integer — ignoring", raw
        )
        return None
    if pl not in _VALID_PL_RANGE:
        LOGGER.warning(
            "nifc_pl_watch: NIFC_PL_OVERRIDE=%d is out of valid range 1-5 — ignoring", pl
        )
        return None
    return pl


def get_preparedness_level(http_client: httpx.Client | None = None) -> int | None:
    """Fetch current NIFC National Preparedness Level (1-5), or None if unavailable.

    Uses a 30-minute in-memory cache. Returns None on any network or parse failure
    rather than raising — callers should treat None as "unknown, do not display."

    PL meanings:
        1 = Low activity
        2 = Normal
        3 = Above normal
        4 = High — mutual aid limited
        5 = Critical — all national resources committed

    Resolution order:
        Tier 1: NIFC_PL_URL (operator-configured JSON endpoint)
        Tier 2: NIFC_PL_OVERRIDE (manually set integer, 1-5)
        Tier 3: None (no source configured)
    """
    global _cache

    now = datetime.now(timezone.utc)
    ttl = timedelta(seconds=_CACHE_TTL_SECONDS)

    if _cache is not None:
        fetched_at, cached_pl = _cache
        if now - fetched_at < ttl:
            LOGGER.debug("nifc_pl_watch: returning cached PL=%s", cached_pl)
            return cached_pl

    pl: int | None = None

    # Tier 1: URL source.
    url = os.getenv("NIFC_PL_URL", "").strip()
    if url:
        if http_client is not None:
            pl = _fetch_from_url(http_client)
        else:
            with httpx.Client() as client:
                pl = _fetch_from_url(client)

    # Tier 2: manual override.
    if pl is None:
        pl = _read_override()

    # Tier 3: no source.
    if pl is None and not url and not os.getenv("NIFC_PL_OVERRIDE", "").strip():
        LOGGER.debug(
            "nifc_pl_watch: no PL source configured — returning None. "
            "Set NIFC_PL_URL or NIFC_PL_OVERRIDE."
        )

    _cache = (now, pl)
    return pl


def get_preparedness_level_context() -> dict[str, Any]:
    """Return a dict suitable for spreading into a notify() call's **context.

    Returns {} if PL is unavailable (callers can spread this safely).
    Returns {"nifc_preparedness_level": N, "nifc_pl_resource_warning": str} if PL >= 4.
    Returns {"nifc_preparedness_level": N} if PL < 4.
    """
    pl = get_preparedness_level()
    if pl is None:
        return {}
    ctx: dict[str, Any] = {"nifc_preparedness_level": pl}
    if pl >= 4:
        ctx["nifc_pl_resource_warning"] = (
            f"National Preparedness Level {pl} — "
            + (
                "mutual aid severely limited."
                if pl == 4
                else "all national resources committed."
            )
        )
    return ctx


def clear_cache() -> None:
    """Clear the in-memory cache. Used in tests."""
    global _cache
    _cache = None
