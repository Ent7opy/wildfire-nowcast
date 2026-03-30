"""Source-agnostic weather warning data contract.

This module defines the shared ``WeatherWarning`` dataclass and the
``WeatherWarningProvider`` abstract base class.  Concrete implementations
(``MeteoAlarmProvider``, and in future NOAA and BoM providers) live in
separate modules and are registered through this interface.

Design goals
------------
- Source-agnostic: ``WeatherWarning`` carries a ``source`` field so consumers
  can present a unified list regardless of origin.
- Graceful degradation: Outside Europe (or any other geographic scope) the
  provider returns an empty list rather than an error.
- Cacheable: providers are expected to be wrapped by
  ``api.core.warning_cache.WarningCache``.

Fire-relevant warning types
---------------------------
Four MeteoAlarm (and analogous NOAA/BoM) types are directly relevant to fire:
- ``"wind"``       — accelerates spread, changes direction unpredictably
- ``"heat"``       — elevates ignition risk and fire intensity
- ``"drought"``    — direct fire danger signal (includes forest-fire warnings)
- ``"thunderstorm"`` — potential ignition via lightning
- ``"rain"``       — suppresses risk
- ``"snow"``       — suppresses risk
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# Canonical severity levels (highest → lowest)
SEVERITY_ORDER = ("red", "orange", "yellow", "green")

# Warning types that elevate fire risk
FIRE_ELEVATING_TYPES = frozenset({"wind", "heat", "drought", "thunderstorm"})

# Warning types that suppress fire risk
FIRE_SUPPRESSING_TYPES = frozenset({"rain", "snow"})


@dataclass(frozen=True)
class WeatherWarning:
    """A single active weather warning from any provider.

    Attributes:
        id:           Provider-specific identifier (stable within a run).
        source:       Data origin: ``"meteoalarm"``, ``"noaa"``, ``"bom"``.
        warning_type: Canonical type from ``FIRE_ELEVATING_TYPES`` /
                      ``FIRE_SUPPRESSING_TYPES`` or ``"other"``.
        severity:     Canonical severity: ``"red"``, ``"orange"``,
                      ``"yellow"``, or ``"green"``.
        headline:     Short human-readable headline.
        description:  Longer description (may be empty string).
        onset:        Warning start time (UTC).
        expires:      Warning expiry time (UTC).
        geometry:     GeoJSON geometry dict (Polygon or MultiPolygon).
        country_code: ISO 3166-1 alpha-2 code or ``None`` if unknown.
        metadata:     Provider-specific extras (preserved for transparency).
    """
    id: str
    source: str
    warning_type: str
    severity: str
    headline: str
    description: str
    onset: datetime
    expires: datetime
    geometry: dict[str, Any]
    country_code: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_active(self, at: datetime) -> bool:
        """Return True when *at* falls within [onset, expires)."""
        return self.onset <= at < self.expires

    def as_brief(self) -> dict[str, Any]:
        """Compact dict for embedding in fire detail API responses."""
        return {
            "source": self.source,
            "warning_type": self.warning_type,
            "severity": self.severity,
            "headline": self.headline,
            "expires": self.expires.isoformat(),
            "country_code": self.country_code,
        }

    def as_geojson_feature(self) -> dict[str, Any]:
        """GeoJSON Feature for the map warning layer."""
        return {
            "type": "Feature",
            "geometry": self.geometry,
            "properties": {
                "id": self.id,
                "source": self.source,
                "warning_type": self.warning_type,
                "severity": self.severity,
                "headline": self.headline,
                "onset": self.onset.isoformat(),
                "expires": self.expires.isoformat(),
                "country_code": self.country_code,
            },
        }


class WeatherWarningProvider(ABC):
    """Abstract base for weather warning data sources."""

    @abstractmethod
    async def get_warnings_for_region(self) -> list[WeatherWarning]:
        """Fetch all currently active warnings for the provider's region.

        Implementations should return an empty list (not raise) when the
        region has no active warnings or the provider is out of scope.
        """
        ...
