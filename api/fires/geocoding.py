"""Reverse geocoding helpers for event centroid naming."""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import JSONB

from api.config import settings
from api.db import get_engine

LOGGER = logging.getLogger(__name__)

_RATE_LOCK = threading.Lock()
_LAST_PROVIDER_CALL_MONOTONIC = 0.0


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned if cleaned else None


def _quantize_coord(value: float) -> float:
    return round(float(value), settings.geocoding_cache_precision)


def _validate_point(lat: float, lon: float) -> tuple[float, float]:
    lat_f = float(lat)
    lon_f = float(lon)
    if not (-90.0 <= lat_f <= 90.0):
        raise ValueError("lat must be between -90 and 90")
    if not (-180.0 <= lon_f <= 180.0):
        raise ValueError("lon must be between -180 and 180")
    return lat_f, lon_f


def _wait_for_provider_window() -> None:
    global _LAST_PROVIDER_CALL_MONOTONIC

    min_interval = float(settings.geocoding_min_interval_seconds)
    if min_interval <= 0:
        return

    with _RATE_LOCK:
        now = time.monotonic()
        elapsed = now - _LAST_PROVIDER_CALL_MONOTONIC
        wait_seconds = min_interval - elapsed
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        _LAST_PROVIDER_CALL_MONOTONIC = time.monotonic()


def _build_headers() -> dict[str, str]:
    user_agent = settings.geocoding_user_agent.strip()
    if not user_agent:
        raise ValueError("GEOCODING_USER_AGENT must be configured for reverse geocoding")
    return {"User-Agent": user_agent}


def _fetch_nominatim(lat: float, lon: float) -> dict[str, Any]:
    _wait_for_provider_window()

    base = settings.geocoding_nominatim_base_url.rstrip("/")
    timeout = httpx.Timeout(settings.geocoding_timeout_seconds)
    params: dict[str, Any] = {
        "format": "jsonv2",
        "lat": f"{lat:.6f}",
        "lon": f"{lon:.6f}",
        "zoom": int(settings.geocoding_zoom),
        "addressdetails": 1,
        "accept-language": settings.geocoding_accept_language,
    }
    email = _clean_text(settings.geocoding_email)
    if email:
        params["email"] = email

    with httpx.Client(timeout=timeout, headers=_build_headers()) as client:
        response = client.get(f"{base}/reverse", params=params)
        response.raise_for_status()
        payload = response.json()

    if not isinstance(payload, dict):
        return {}
    return payload


def _select_admin1(address: dict[str, Any]) -> str | None:
    for key in ("state", "region", "province", "state_district"):
        value = _clean_text(address.get(key))
        if value:
            return value
    return None


def _select_admin2(address: dict[str, Any]) -> str | None:
    for key in ("county", "municipality", "city", "town", "village", "suburb", "hamlet"):
        value = _clean_text(address.get(key))
        if value:
            return value
    return None


def _select_feature_name(payload: dict[str, Any], address: dict[str, Any]) -> str | None:
    payload_name = _clean_text(payload.get("name"))
    if payload_name:
        return payload_name

    for key in ("forest", "natural", "reserve", "park", "island", "archipelago"):
        value = _clean_text(address.get(key))
        if value:
            return value
    return None


def _compose_location_label(
    *,
    feature_name: str | None,
    admin1_name: str | None,
    admin2_name: str | None,
    country_name: str | None,
    display_name: str | None,
) -> str | None:
    primary = feature_name or admin1_name or admin2_name
    if primary and country_name and primary.lower() != country_name.lower():
        return f"{primary}, {country_name}"
    if primary:
        return primary
    if country_name:
        return country_name
    return display_name


def _parse_nominatim(payload: dict[str, Any]) -> dict[str, Any]:
    address = payload.get("address")
    address_obj = address if isinstance(address, dict) else {}

    country_name = _clean_text(address_obj.get("country"))
    admin1_name = _select_admin1(address_obj)
    admin2_name = _select_admin2(address_obj)
    feature_name = _select_feature_name(payload, address_obj)
    display_name = _clean_text(payload.get("display_name"))

    location_name = _compose_location_label(
        feature_name=feature_name,
        admin1_name=admin1_name,
        admin2_name=admin2_name,
        country_name=country_name,
        display_name=display_name,
    )

    return {
        "location_name": location_name,
        "country_name": country_name,
        "admin1_name": admin1_name,
        "admin2_name": admin2_name,
        "display_name": display_name,
    }


def _read_cached_result(
    *,
    provider: str,
    cached_lat: float,
    cached_lon: float,
) -> dict[str, Any] | None:
    stmt = text(
        """
        SELECT
            status,
            location_name,
            country_name,
            admin1_name,
            admin2_name,
            display_name,
            updated_at,
            expires_at
        FROM reverse_geocode_cache
        WHERE provider = :provider
          AND cached_lat = :cached_lat
          AND cached_lon = :cached_lon
          AND expires_at > :now_utc
        LIMIT 1
        """
    )

    with get_engine().begin() as conn:
        row = conn.execute(
            stmt,
            {
                "provider": provider,
                "cached_lat": cached_lat,
                "cached_lon": cached_lon,
                "now_utc": _utc_now(),
            },
        ).mappings().first()

    if row is None:
        return None
    return dict(row)


def _write_cache(
    *,
    provider: str,
    cached_lat: float,
    cached_lon: float,
    status: str,
    location_name: str | None,
    country_name: str | None,
    admin1_name: str | None,
    admin2_name: str | None,
    display_name: str | None,
    raw_payload: dict[str, Any] | None,
    ttl: timedelta,
) -> None:
    now_utc = _utc_now()
    expires_at = now_utc + ttl

    stmt = text(
        """
        INSERT INTO reverse_geocode_cache (
            provider,
            cached_lat,
            cached_lon,
            status,
            location_name,
            country_name,
            admin1_name,
            admin2_name,
            display_name,
            raw_payload,
            updated_at,
            expires_at
        )
        VALUES (
            :provider,
            :cached_lat,
            :cached_lon,
            :status,
            :location_name,
            :country_name,
            :admin1_name,
            :admin2_name,
            :display_name,
            :raw_payload,
            :updated_at,
            :expires_at
        )
        ON CONFLICT (provider, cached_lat, cached_lon)
        DO UPDATE SET
            status = EXCLUDED.status,
            location_name = EXCLUDED.location_name,
            country_name = EXCLUDED.country_name,
            admin1_name = EXCLUDED.admin1_name,
            admin2_name = EXCLUDED.admin2_name,
            display_name = EXCLUDED.display_name,
            raw_payload = EXCLUDED.raw_payload,
            updated_at = EXCLUDED.updated_at,
            expires_at = EXCLUDED.expires_at
        """
    ).bindparams(bindparam("raw_payload", type_=JSONB))

    with get_engine().begin() as conn:
        conn.execute(
            stmt,
            {
                "provider": provider,
                "cached_lat": cached_lat,
                "cached_lon": cached_lon,
                "status": status,
                "location_name": location_name,
                "country_name": country_name,
                "admin1_name": admin1_name,
                "admin2_name": admin2_name,
                "display_name": display_name,
                "raw_payload": raw_payload,
                "updated_at": now_utc,
                "expires_at": expires_at,
            },
        )


def _ttl_for_status(status: str) -> timedelta:
    if status == "error":
        return timedelta(hours=1)
    return timedelta(hours=int(settings.geocoding_cache_ttl_hours))


def reverse_geocode_point(lat: float, lon: float) -> dict[str, Any]:
    """Resolve a place label for a coordinate using configured open geocoder."""
    lat_f, lon_f = _validate_point(lat, lon)

    provider = settings.geocoding_provider.strip().lower()
    if provider != "nominatim":
        raise ValueError(f"Unsupported geocoding provider: {provider}")

    cached_lat = _quantize_coord(lat_f)
    cached_lon = _quantize_coord(lon_f)

    cached = _read_cached_result(
        provider=provider,
        cached_lat=cached_lat,
        cached_lon=cached_lon,
    )
    if cached:
        return {
            "lat": lat_f,
            "lon": lon_f,
            "cached_lat": cached_lat,
            "cached_lon": cached_lon,
            "provider": provider,
            "cache_hit": True,
            "status": cached.get("status"),
            "location_name": cached.get("location_name"),
            "country": cached.get("country_name"),
            "admin1_name": cached.get("admin1_name"),
            "admin2_name": cached.get("admin2_name"),
            "display_name": cached.get("display_name"),
            "updated_at": cached.get("updated_at").isoformat() if cached.get("updated_at") else None,
            "expires_at": cached.get("expires_at").isoformat() if cached.get("expires_at") else None,
        }

    if not settings.geocoding_enabled:
        return {
            "lat": lat_f,
            "lon": lon_f,
            "cached_lat": cached_lat,
            "cached_lon": cached_lon,
            "provider": provider,
            "cache_hit": False,
            "status": "disabled",
            "location_name": None,
            "country": None,
            "admin1_name": None,
            "admin2_name": None,
            "display_name": None,
            "updated_at": None,
            "expires_at": None,
        }

    payload: dict[str, Any] | None = None
    parsed = {
        "location_name": None,
        "country_name": None,
        "admin1_name": None,
        "admin2_name": None,
        "display_name": None,
    }
    status = "unresolved"

    try:
        payload = _fetch_nominatim(lat_f, lon_f)
        parsed = _parse_nominatim(payload)
        if parsed["location_name"]:
            status = "resolved"
    except Exception as exc:  # noqa: BLE001
        status = "error"
        LOGGER.warning(
            "reverse geocode request failed: provider=%s lat=%.6f lon=%.6f error=%s",
            provider,
            lat_f,
            lon_f,
            exc,
        )

    _write_cache(
        provider=provider,
        cached_lat=cached_lat,
        cached_lon=cached_lon,
        status=status,
        location_name=parsed["location_name"],
        country_name=parsed["country_name"],
        admin1_name=parsed["admin1_name"],
        admin2_name=parsed["admin2_name"],
        display_name=parsed["display_name"],
        raw_payload=payload,
        ttl=_ttl_for_status(status),
    )

    return {
        "lat": lat_f,
        "lon": lon_f,
        "cached_lat": cached_lat,
        "cached_lon": cached_lon,
        "provider": provider,
        "cache_hit": False,
        "status": status,
        "location_name": parsed["location_name"],
        "country": parsed["country_name"],
        "admin1_name": parsed["admin1_name"],
        "admin2_name": parsed["admin2_name"],
        "display_name": parsed["display_name"],
        "updated_at": _utc_now().isoformat(),
        "expires_at": (_utc_now() + _ttl_for_status(status)).isoformat(),
    }
