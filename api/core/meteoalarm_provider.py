"""MeteoAlarm weather warning provider.

Fetches active weather warnings for Europe from the MeteoAlarm CAP/ATOM
feeds.  One feed exists per country; this provider iterates over the
supported country list and aggregates results.

Feed format
-----------
MeteoAlarm publishes per-country ATOM feeds at:

    https://feeds.meteoalarm.org/feeds/meteoalarm-legacy-atom-{country}

where ``{country}`` is a lowercase ISO 3166-1 alpha-2 code (e.g. ``gr``,
``es``, ``it``).  Each entry is a CAP (Common Alerting Protocol) XML block
embedded in the ``<content>`` element.

Geographic scope
----------------
Feeds cover European countries that publish to MeteoAlarm.  Queries for
points outside Europe return an empty list; the caller should never see an
error, only an empty result.

Reference
---------
MeteoAlarm ATOM feed documentation:
  https://feeds.meteoalarm.org/feeds/meteoalarm-legacy-atom-de  (example)

CAP standard: ITU-T X.1303 / OASIS CAP v1.2
"""

from __future__ import annotations

import functools
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any

import httpx

from api.core.weather_warnings import WeatherWarning, WeatherWarningProvider

LOGGER = logging.getLogger(__name__)

# MeteoAlarm feed URL template
_FEED_URL = "https://feeds.meteoalarm.org/feeds/meteoalarm-legacy-atom-{country}"

# Countries that publish to MeteoAlarm (ISO 3166-1 alpha-2, lowercase)
_METEOALARM_COUNTRIES = [
    "al", "at", "ba", "be", "bg", "by", "ch", "cy", "cz", "de",
    "dk", "ee", "es", "fi", "fr", "gr", "hr", "hu", "ie", "il",
    "is", "it", "lt", "lu", "lv", "me", "mk", "mt", "nl", "no",
    "pl", "pt", "ro", "rs", "se", "si", "sk", "ua", "uk",
]

# CAP namespace
_CAP_NS = "urn:oasis:names:tc:emergency:cap:1.2"
_ATOM_NS = "http://www.w3.org/2005/Atom"

# Mapping from MeteoAlarm event name keywords → canonical warning_type
_EVENT_TYPE_MAP: list[tuple[tuple[str, ...], str]] = [
    (("wind", "gale", "föhn", "bora"), "wind"),
    (("heat", "high temperature", "extreme temperature"), "heat"),
    (("drought", "forest fire", "wildfire", "fire danger"), "drought"),
    (("thunderstorm", "thunder", "lightning"), "thunderstorm"),
    (("rain", "rainfall", "flood", "flash flood", "coastal event"), "rain"),
    (("snow", "blizzard", "ice", "avalanche", "freezing"), "snow"),
    (("fog",), "fog"),
]

# Mapping from MeteoAlarm severity → canonical
_SEVERITY_MAP: dict[str, str] = {
    "extreme": "red",
    "severe": "red",
    "moderate": "orange",
    "minor": "yellow",
    "unknown": "yellow",
    "": "yellow",
}

# Request timeout for individual country feeds
_FEED_TIMEOUT_S = 8.0


@functools.lru_cache(maxsize=256)
def _canonical_event_type(event_name: str) -> str:
    lower = event_name.lower()
    for keywords, canonical in _EVENT_TYPE_MAP:
        if any(kw in lower for kw in keywords):
            return canonical
    return "other"


def _canonical_severity(meteoalarm_severity: str) -> str:
    return _SEVERITY_MAP.get(meteoalarm_severity.lower(), "yellow")


def _parse_cap_polygon(poly_str: str) -> dict[str, Any] | None:
    """Parse a CAP polygon string ('lat lon lat lon …') into a GeoJSON Polygon."""
    try:
        coords_raw = poly_str.strip().split()
        if len(coords_raw) < 6 or len(coords_raw) % 2 != 0:
            return None
        coords: list[list[float]] = []
        for i in range(0, len(coords_raw), 2):
            lat, lon = float(coords_raw[i]), float(coords_raw[i + 1])
            coords.append([lon, lat])   # GeoJSON is [lon, lat]
        # Close the ring if necessary
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        return {"type": "Polygon", "coordinates": [coords]}
    except (ValueError, IndexError):
        return None


def _parse_iso_datetime(dt_str: str) -> datetime | None:
    """Parse CAP datetime strings (ISO 8601) to UTC-aware datetime."""
    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc)
    except (ValueError, AttributeError):
        return None


def _parse_atom_feed(xml_bytes: bytes, country_code: str) -> list[WeatherWarning]:
    """Parse a MeteoAlarm ATOM feed XML into a list of WeatherWarning objects."""
    warnings: list[WeatherWarning] = []
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        LOGGER.debug("Failed to parse ATOM feed for %s: %s", country_code, exc)
        return warnings

    for entry in root.findall(f"{{{_ATOM_NS}}}entry"):
        # Each entry carries a <content> block with embedded CAP XML
        content_el = entry.find(f"{{{_ATOM_NS}}}content")
        if content_el is None:
            continue

        cap_text = content_el.text or ""
        if not cap_text.strip():
            continue

        try:
            cap = ET.fromstring(cap_text)
        except ET.ParseError:
            continue

        cap_prefix = f"{{{_CAP_NS}}}"

        # Extract info block (take the first English one, or just the first)
        info_blocks = cap.findall(f"{cap_prefix}info")
        if not info_blocks:
            continue
        info = next(
            (b for b in info_blocks if (b.findtext(f"{cap_prefix}language") or "").lower().startswith("en")),
            info_blocks[0],
        )

        event_name = info.findtext(f"{cap_prefix}event") or ""
        severity_raw = info.findtext(f"{cap_prefix}severity") or ""
        headline = info.findtext(f"{cap_prefix}headline") or event_name
        description = info.findtext(f"{cap_prefix}description") or ""
        onset_str = info.findtext(f"{cap_prefix}onset") or ""
        expires_str = info.findtext(f"{cap_prefix}expires") or ""

        onset = _parse_iso_datetime(onset_str)
        expires = _parse_iso_datetime(expires_str)
        if onset is None or expires is None:
            continue

        # Polygon from <area>/<polygon>
        geometry: dict[str, Any] | None = None
        area_el = info.find(f"{cap_prefix}area")
        if area_el is not None:
            poly_str = area_el.findtext(f"{cap_prefix}polygon") or ""
            if poly_str:
                geometry = _parse_cap_polygon(poly_str)

        # Fall back to a country-level bounding box if no polygon
        if geometry is None:
            LOGGER.debug(
                "No polygon for warning in %s (%s) — skipping", country_code, event_name
            )
            continue

        # Unique ID from the CAP identifier or entry id
        cap_id = cap.findtext(f"{cap_prefix}identifier") or ""
        entry_id = entry.findtext(f"{{{_ATOM_NS}}}id") or ""
        warning_id = f"{country_code}:{cap_id or entry_id}"

        warnings.append(
            WeatherWarning(
                id=warning_id,
                source="meteoalarm",
                warning_type=_canonical_event_type(event_name),
                severity=_canonical_severity(severity_raw),
                headline=headline,
                description=description,
                onset=onset,
                expires=expires,
                geometry=geometry,
                country_code=country_code.upper(),
                metadata={
                    "cap_event": event_name,
                    "cap_severity": severity_raw,
                    "cap_id": cap_id,
                },
            )
        )

    return warnings


class MeteoAlarmProvider(WeatherWarningProvider):
    """Fetch active weather warnings from MeteoAlarm per-country ATOM feeds.

    The provider fetches all configured country feeds concurrently via
    ``httpx.AsyncClient`` and aggregates results.  Errors for individual
    countries are logged at DEBUG level and suppressed so a single
    unavailable country feed does not block the full response.
    """

    def __init__(
        self,
        countries: list[str] | None = None,
        timeout: float = _FEED_TIMEOUT_S,
    ) -> None:
        self._countries = countries or _METEOALARM_COUNTRIES
        self._timeout = timeout

    async def get_warnings_for_region(self) -> list[WeatherWarning]:
        """Fetch all currently active warnings across European MeteoAlarm countries."""
        import asyncio

        async def _fetch_country(client: httpx.AsyncClient, country: str) -> list[WeatherWarning]:
            url = _FEED_URL.format(country=country)
            try:
                resp = await client.get(url, timeout=self._timeout)
                resp.raise_for_status()
                return _parse_atom_feed(resp.content, country)
            except Exception as exc:
                LOGGER.debug("MeteoAlarm feed unavailable for %s: %s", country, exc)
                return []

        async with httpx.AsyncClient(follow_redirects=True) as client:
            results = await asyncio.gather(
                *(_fetch_country(client, c) for c in self._countries),
                return_exceptions=False,
            )

        all_warnings: list[WeatherWarning] = []
        for country_warnings in results:
            all_warnings.extend(country_warnings)

        return all_warnings
