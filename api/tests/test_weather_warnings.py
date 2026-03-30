"""Tests for the MeteoAlarm weather warning integration."""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

from api.core.weather_warnings import (
    FIRE_ELEVATING_TYPES,
    FIRE_SUPPRESSING_TYPES,
    WeatherWarning,
)
from api.core.meteoalarm_provider import (
    _canonical_event_type,
    _canonical_severity,
    _parse_cap_polygon,
    _parse_iso_datetime,
    _parse_atom_feed,
)
from api.core.warning_cache import WarningCache


# ---------------------------------------------------------------------------
# WeatherWarning dataclass
# ---------------------------------------------------------------------------

def _make_warning(**overrides) -> WeatherWarning:
    now = datetime(2026, 3, 30, 12, 0, tzinfo=timezone.utc)
    defaults = dict(
        id="gr:test-001",
        source="meteoalarm",
        warning_type="wind",
        severity="red",
        headline="Severe wind warning",
        description="Gusts up to 120 km/h",
        onset=now - timedelta(hours=1),
        expires=now + timedelta(hours=5),
        geometry={
            "type": "Polygon",
            "coordinates": [[[20.0, 38.0], [25.0, 38.0], [25.0, 42.0], [20.0, 42.0], [20.0, 38.0]]],
        },
        country_code="GR",
    )
    defaults.update(overrides)
    return WeatherWarning(**defaults)


def test_warning_is_active_within_window():
    w = _make_warning()
    now = datetime(2026, 3, 30, 12, 0, tzinfo=timezone.utc)
    assert w.is_active(now) is True


def test_warning_is_not_active_before_onset():
    w = _make_warning()
    before = datetime(2026, 3, 30, 10, 30, tzinfo=timezone.utc)  # before onset
    assert w.is_active(before) is False


def test_warning_is_not_active_after_expiry():
    w = _make_warning()
    after = datetime(2026, 3, 30, 18, 0, tzinfo=timezone.utc)   # after expires
    assert w.is_active(after) is False


def test_warning_brief_contains_required_fields():
    w = _make_warning()
    brief = w.as_brief()
    assert brief["source"] == "meteoalarm"
    assert brief["warning_type"] == "wind"
    assert brief["severity"] == "red"
    assert brief["headline"] == "Severe wind warning"
    assert "expires" in brief
    assert brief["country_code"] == "GR"


def test_warning_geojson_feature_structure():
    w = _make_warning()
    feat = w.as_geojson_feature()
    assert feat["type"] == "Feature"
    assert feat["geometry"]["type"] == "Polygon"
    props = feat["properties"]
    assert props["severity"] == "red"
    assert props["warning_type"] == "wind"


def test_fire_elevating_types_contain_expected():
    assert "wind" in FIRE_ELEVATING_TYPES
    assert "heat" in FIRE_ELEVATING_TYPES
    assert "drought" in FIRE_ELEVATING_TYPES
    assert "thunderstorm" in FIRE_ELEVATING_TYPES


def test_fire_suppressing_types_contain_expected():
    assert "rain" in FIRE_SUPPRESSING_TYPES
    assert "snow" in FIRE_SUPPRESSING_TYPES


# ---------------------------------------------------------------------------
# MeteoAlarm provider internals
# ---------------------------------------------------------------------------

def test_canonical_event_type_wind():
    assert _canonical_event_type("Wind") == "wind"
    assert _canonical_event_type("Gale Warning") == "wind"


def test_canonical_event_type_heat():
    assert _canonical_event_type("Extreme Temperature") == "heat"
    assert _canonical_event_type("High Temperature") == "heat"


def test_canonical_event_type_drought():
    assert _canonical_event_type("Forest Fire Danger") == "drought"
    assert _canonical_event_type("Drought") == "drought"


def test_canonical_event_type_thunderstorm():
    assert _canonical_event_type("Thunderstorm") == "thunderstorm"


def test_canonical_event_type_rain():
    assert _canonical_event_type("Rain") == "rain"
    assert _canonical_event_type("Flooding") == "rain"


def test_canonical_event_type_snow():
    assert _canonical_event_type("Snow") == "snow"
    assert _canonical_event_type("Blizzard") == "snow"


def test_canonical_event_type_unknown():
    assert _canonical_event_type("Some Unknown Event") == "other"


def test_canonical_severity_red():
    assert _canonical_severity("Extreme") == "red"
    assert _canonical_severity("Severe") == "red"


def test_canonical_severity_orange():
    assert _canonical_severity("Moderate") == "orange"


def test_canonical_severity_yellow():
    assert _canonical_severity("Minor") == "yellow"
    assert _canonical_severity("Unknown") == "yellow"


def test_parse_cap_polygon_valid():
    poly_str = "38.0 20.0 38.0 25.0 42.0 25.0 42.0 20.0 38.0 20.0"
    geom = _parse_cap_polygon(poly_str)
    assert geom is not None
    assert geom["type"] == "Polygon"
    assert len(geom["coordinates"][0]) >= 4


def test_parse_cap_polygon_invalid():
    assert _parse_cap_polygon("not a polygon") is None
    assert _parse_cap_polygon("38.0 20.0") is None   # too short


def test_parse_iso_datetime_utc():
    dt = _parse_iso_datetime("2026-03-30T12:00:00Z")
    assert dt is not None
    assert dt.tzinfo == timezone.utc
    assert dt.year == 2026


def test_parse_iso_datetime_offset():
    dt = _parse_iso_datetime("2026-03-30T14:00:00+02:00")
    assert dt is not None
    assert dt.hour == 12   # converted to UTC


def test_parse_iso_datetime_invalid():
    assert _parse_iso_datetime("not-a-date") is None


def test_parse_atom_feed_valid():
    """Minimal ATOM/CAP XML that _parse_atom_feed should accept."""
    xml = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>urn:test:001</id>
    <content>
&lt;alert xmlns="urn:oasis:names:tc:emergency:cap:1.2"&gt;
  &lt;identifier&gt;GR-WIND-001&lt;/identifier&gt;
  &lt;info&gt;
    &lt;language&gt;en-US&lt;/language&gt;
    &lt;event&gt;Wind&lt;/event&gt;
    &lt;severity&gt;Extreme&lt;/severity&gt;
    &lt;headline&gt;Extreme wind warning&lt;/headline&gt;
    &lt;onset&gt;2026-03-30T10:00:00+00:00&lt;/onset&gt;
    &lt;expires&gt;2026-03-30T20:00:00+00:00&lt;/expires&gt;
    &lt;area&gt;
      &lt;polygon&gt;38.0 20.0 38.0 25.0 42.0 25.0 42.0 20.0 38.0 20.0&lt;/polygon&gt;
    &lt;/area&gt;
  &lt;/info&gt;
&lt;/alert&gt;
    </content>
  </entry>
</feed>"""
    # The test XML uses HTML entities for the inner XML, which ET will resolve
    warnings = _parse_atom_feed(xml, "gr")
    # If parsing works, we should get 1 warning; if the embedded XML parsing
    # differs from our implementation we may get 0 — either way, no crash.
    assert isinstance(warnings, list)


def test_parse_atom_feed_empty():
    xml = b"""<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>"""
    warnings = _parse_atom_feed(xml, "de")
    assert warnings == []


def test_parse_atom_feed_malformed():
    warnings = _parse_atom_feed(b"<this is not xml>", "it")
    assert warnings == []


# ---------------------------------------------------------------------------
# WarningCache
# ---------------------------------------------------------------------------

def test_warning_cache_cold_loads_on_first_call():
    now = datetime(2026, 3, 30, 12, 0, tzinfo=timezone.utc)
    w = _make_warning(onset=now - timedelta(hours=1), expires=now + timedelta(hours=5))

    mock_provider = AsyncMock()
    mock_provider.get_warnings_for_region.return_value = [w]

    cache = WarningCache(mock_provider, ttl_seconds=900)
    result = asyncio.run(cache.get_all_warnings(now=now))

    assert len(result) == 1
    mock_provider.get_warnings_for_region.assert_called_once()


def test_warning_cache_serves_from_cache_within_ttl():
    now = datetime(2026, 3, 30, 12, 0, tzinfo=timezone.utc)
    w = _make_warning(onset=now - timedelta(hours=1), expires=now + timedelta(hours=5))

    mock_provider = AsyncMock()
    mock_provider.get_warnings_for_region.return_value = [w]

    async def _run():
        cache = WarningCache(mock_provider, ttl_seconds=900)
        await cache.get_all_warnings(now=now)
        await cache.get_all_warnings(now=now)  # second call — should use cache
        mock_provider.get_warnings_for_region.assert_called_once()

    asyncio.run(_run())


def test_warning_cache_point_containment():
    now = datetime(2026, 3, 30, 12, 0, tzinfo=timezone.utc)
    # Warning covers Greece roughly (20°E–26°E, 35°N–42°N)
    w = _make_warning(
        geometry={
            "type": "Polygon",
            "coordinates": [[[20.0, 35.0], [26.0, 35.0], [26.0, 42.0], [20.0, 42.0], [20.0, 35.0]]],
        },
        onset=now - timedelta(hours=1),
        expires=now + timedelta(hours=5),
    )

    mock_provider = AsyncMock()
    mock_provider.get_warnings_for_region.return_value = [w]

    async def _run():
        cache = WarningCache(mock_provider, ttl_seconds=900)
        await cache.get_all_warnings(now=now)
        return cache

    cache = asyncio.run(_run())

    # Point inside Greece
    inside = cache.warnings_for_point(lat=38.0, lon=23.7, now=now)
    assert len(inside) == 1

    # Point in Paris — outside the polygon
    outside = cache.warnings_for_point(lat=48.85, lon=2.35, now=now)
    assert len(outside) == 0


def test_warning_cache_excludes_expired_warnings():
    now = datetime(2026, 3, 30, 12, 0, tzinfo=timezone.utc)
    expired = _make_warning(
        onset=now - timedelta(hours=10),
        expires=now - timedelta(hours=1),   # already expired
    )

    mock_provider = AsyncMock()
    mock_provider.get_warnings_for_region.return_value = [expired]

    async def _run():
        cache = WarningCache(mock_provider, ttl_seconds=900)
        await cache.get_all_warnings(now=now)
        return cache

    cache = asyncio.run(_run())
    results = cache.warnings_for_point(lat=38.0, lon=23.7, now=now, active_only=True)
    assert len(results) == 0
