#!/usr/bin/env python3
"""Load perimeter coverage masks into perimeter_coverage_masks table."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import text

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api.db import get_engine  # noqa: E402


def _parse_dt(value: str | None) -> datetime | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def _to_multipolygon_geojson(geometry: dict[str, Any]) -> dict[str, Any]:
    gtype = str(geometry.get("type", "")).strip()
    coords = geometry.get("coordinates")
    if gtype == "MultiPolygon":
        return {"type": "MultiPolygon", "coordinates": coords}
    if gtype == "Polygon":
        return {"type": "MultiPolygon", "coordinates": [coords]}
    raise ValueError(f"Unsupported geometry type={gtype!r}; expected Polygon or MultiPolygon")


def _load_features(input_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and payload.get("type") == "FeatureCollection":
        features = list(payload.get("features") or [])
    elif isinstance(payload, dict) and payload.get("type") == "Feature":
        features = [payload]
    elif isinstance(payload, list):
        features = payload
    else:
        raise ValueError("Input must be a GeoJSON FeatureCollection, Feature, or list of Features")
    if not features:
        raise ValueError("No features found in input")
    return features


def load_masks(
    *,
    input_path: Path,
    default_provider: str,
    default_reliability_tier: str,
    default_valid_from: datetime | None,
    default_valid_to: datetime | None,
    default_is_active: bool,
) -> dict[str, int]:
    features = _load_features(input_path)
    params: list[dict[str, Any]] = []
    for i, feat in enumerate(features):
        if not isinstance(feat, dict):
            continue
        properties = feat.get("properties") or {}
        geometry = feat.get("geometry")
        if not geometry:
            continue
        mp_geojson = _to_multipolygon_geojson(geometry)
        mask_id = str(properties.get("mask_id") or f"{default_provider}_{i + 1:05d}")
        params.append(
            {
                "mask_id": mask_id,
                "provider": str(properties.get("provider") or default_provider),
                "reliability_tier": str(properties.get("reliability_tier") or default_reliability_tier),
                "valid_from": _parse_dt(properties.get("valid_from")) or default_valid_from,
                "valid_to": _parse_dt(properties.get("valid_to")) or default_valid_to,
                "geom_json": json.dumps(mp_geojson),
                "is_active": bool(properties.get("is_active", default_is_active)),
            }
        )
    if not params:
        raise ValueError("No valid features with geometry found in input")

    upsert = text(
        """
        INSERT INTO perimeter_coverage_masks (
            mask_id,
            provider,
            reliability_tier,
            valid_from,
            valid_to,
            geom,
            is_active,
            created_at,
            updated_at
        ) VALUES (
            :mask_id,
            :provider,
            :reliability_tier,
            :valid_from,
            :valid_to,
            ST_SetSRID(ST_GeomFromGeoJSON(:geom_json), 4326),
            :is_active,
            NOW(),
            NOW()
        )
        ON CONFLICT (mask_id) DO UPDATE SET
            provider = EXCLUDED.provider,
            reliability_tier = EXCLUDED.reliability_tier,
            valid_from = EXCLUDED.valid_from,
            valid_to = EXCLUDED.valid_to,
            geom = EXCLUDED.geom,
            is_active = EXCLUDED.is_active,
            updated_at = NOW()
        """
    )

    with get_engine().begin() as conn:
        conn.execute(upsert, params)

    return {"input_features": len(features), "loaded_rows": len(params)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load denoiser perimeter coverage masks from GeoJSON")
    parser.add_argument("--input", required=True, help="Path to GeoJSON file")
    parser.add_argument("--provider", default="unknown_provider")
    parser.add_argument("--reliability-tier", default="gold")
    parser.add_argument("--valid-from", default=None, help="Default ISO datetime for all rows")
    parser.add_argument("--valid-to", default=None, help="Default ISO datetime for all rows")
    parser.add_argument("--inactive", action="store_true", help="Load rows with is_active=false by default")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = load_masks(
        input_path=Path(args.input),
        default_provider=str(args.provider),
        default_reliability_tier=str(args.reliability_tier),
        default_valid_from=_parse_dt(args.valid_from),
        default_valid_to=_parse_dt(args.valid_to),
        default_is_active=not bool(args.inactive),
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main(sys.argv[1:])
