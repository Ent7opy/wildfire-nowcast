"""Load industrial no-go zones from policy config or GeoJSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import text

from ingest.repository import get_engine

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_CONFIG = REPO_ROOT / "configs" / "industrial_policy_global_v1.yaml"


def _bbox_geojson(bbox: list[float]) -> str:
    if len(bbox) != 4:
        raise ValueError("bbox must contain exactly 4 values")
    min_lon, min_lat, max_lon, max_lat = [float(v) for v in bbox]
    poly = {
        "type": "MultiPolygon",
        "coordinates": [
            [[
                [min_lon, min_lat],
                [max_lon, min_lat],
                [max_lon, max_lat],
                [min_lon, max_lat],
                [min_lon, min_lat],
            ]]
        ],
    }
    return json.dumps(poly)


def _load_config_zones(path: Path) -> tuple[str, list[dict[str, Any]]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    policy_version = str(payload.get("policy_version") or "").strip()
    zones = payload.get("no_go_zones") or []
    if not policy_version:
        raise ValueError("policy config missing policy_version")
    if not isinstance(zones, list):
        raise ValueError("policy config no_go_zones must be a list")
    return policy_version, zones


def _load_geojson(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("type") != "FeatureCollection":
        raise ValueError("GeoJSON input must be a FeatureCollection")
    features = payload.get("features") or []
    zones: list[dict[str, Any]] = []
    for idx, feature in enumerate(features):
        props = feature.get("properties") or {}
        geom = feature.get("geometry")
        if not geom:
            continue
        zone_id = str(props.get("zone_id") or f"no_go_{idx+1:04d}")
        zones.append(
            {
                "zone_id": zone_id,
                "zone_name": str(props.get("zone_name") or zone_id),
                "reason": str(props.get("reason") or "policy_no_go"),
                "region_code": str(props.get("region_code") or "unknown"),
                "is_active": bool(props.get("is_active", True)),
                "geom_json": json.dumps(geom),
            }
        )
    return zones


def _normalize_zone_rows(
    *,
    zones: list[dict[str, Any]],
    policy_version: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for zone in zones:
        zone_id = str(zone.get("zone_id") or "").strip()
        if not zone_id:
            continue
        if "geom_json" in zone:
            geom_json = str(zone["geom_json"])
        else:
            bbox = zone.get("bbox")
            if not isinstance(bbox, list):
                continue
            geom_json = _bbox_geojson(bbox)
        out.append(
            {
                "zone_id": zone_id,
                "zone_name": str(zone.get("zone_name") or zone_id),
                "reason": str(zone.get("reason") or "policy_no_go"),
                "region_code": str(zone.get("region_code") or "unknown"),
                "policy_version": policy_version,
                "is_active": bool(zone.get("is_active", True)),
                "geom_json": geom_json,
            }
        )
    return out


def upsert_no_go_rows(rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    stmt = text(
        """
        INSERT INTO industrial_no_go_zones (
            zone_id,
            zone_name,
            reason,
            region_code,
            geom,
            is_active,
            policy_version,
            created_at,
            updated_at
        ) VALUES (
            :zone_id,
            :zone_name,
            :reason,
            :region_code,
            ST_SetSRID(ST_GeomFromGeoJSON(:geom_json), 4326),
            :is_active,
            :policy_version,
            NOW(),
            NOW()
        )
        ON CONFLICT (zone_id) DO UPDATE SET
            zone_name = EXCLUDED.zone_name,
            reason = EXCLUDED.reason,
            region_code = EXCLUDED.region_code,
            geom = EXCLUDED.geom,
            is_active = EXCLUDED.is_active,
            policy_version = EXCLUDED.policy_version,
            updated_at = NOW()
        """
    )
    with get_engine().begin() as conn:
        result = conn.execute(stmt, rows)
    return int(result.rowcount or 0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load industrial no-go zones")
    parser.add_argument("--config", default=str(DEFAULT_POLICY_CONFIG))
    parser.add_argument("--input", default=None, help="Optional GeoJSON FeatureCollection path")
    parser.add_argument("--policy-version", default=None)
    parser.add_argument("--deactivate-existing", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.input:
        zones = _load_geojson(Path(args.input).expanduser().resolve())
        policy_version = str(args.policy_version or "").strip()
        if not policy_version:
            raise SystemExit("--policy-version is required when --input is used")
    else:
        cfg_policy_version, cfg_zones = _load_config_zones(Path(args.config).expanduser().resolve())
        policy_version = str(args.policy_version or cfg_policy_version).strip()
        zones = cfg_zones

    rows = _normalize_zone_rows(zones=zones, policy_version=policy_version)

    if args.deactivate_existing:
        with get_engine().begin() as conn:
            conn.execute(
                text(
                    """
                    UPDATE industrial_no_go_zones
                    SET is_active = FALSE,
                        updated_at = NOW()
                    WHERE policy_version = :policy_version
                    """
                ),
                {"policy_version": policy_version},
            )

    upserted = upsert_no_go_rows(rows)
    summary = {
        "policy_version": policy_version,
        "rows_input": len(rows),
        "rows_upserted": upserted,
    }
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
