"""Build perimeter_coverage_masks from authoritative geometry with provenance."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import text

from ingest.repository import get_engine

LOGGER = logging.getLogger("coverage_mask_builder")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


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


def _latest_successful_run_id(source_profile: str | None = None) -> str | None:
    if source_profile:
        stmt = text(
            """
            SELECT run_id
            FROM authoritative_perimeter_ingest_runs
            WHERE source_profile = :profile
              AND status = 'succeeded'
            ORDER BY finished_at DESC NULLS LAST, started_at DESC
            LIMIT 1
            """
        )
        params = {"profile": source_profile}
    else:
        stmt = text(
            """
            SELECT run_id
            FROM authoritative_perimeter_ingest_runs
            WHERE status = 'succeeded'
            ORDER BY finished_at DESC NULLS LAST, started_at DESC
            LIMIT 1
            """
        )
        params = {}
    with get_engine().begin() as conn:
        row = conn.execute(stmt, params).mappings().first()
    if row is None:
        return None
    return str(row["run_id"])


def build_coverage_masks(
    *,
    input_path: Path,
    authority_profile: str,
    tier_policy: str,
    source_uri: str,
    source_version: str,
    run_id: str,
    default_valid_from: datetime | None,
    default_valid_to: datetime | None,
    default_is_active: bool,
) -> dict[str, Any]:
    features = _load_features(input_path)
    params: list[dict[str, Any]] = []
    for i, feat in enumerate(features):
        if not isinstance(feat, dict):
            continue
        geometry = feat.get("geometry")
        if not isinstance(geometry, dict):
            continue

        props = feat.get("properties") or {}
        mp_geojson = _to_multipolygon_geojson(geometry)
        mask_id = str(props.get("mask_id") or f"{authority_profile}_{i + 1:05d}")
        valid_from = _parse_dt(props.get("valid_from")) or default_valid_from
        valid_to = _parse_dt(props.get("valid_to")) or default_valid_to
        is_active = bool(props.get("is_active", default_is_active))

        provenance = {
            "authority_profile": authority_profile,
            "tier_policy": tier_policy,
            "source_uri": source_uri,
            "source_version": source_version,
            "run_id": run_id,
            "feature_index": i,
            "properties": props,
        }
        params.append(
            {
                "mask_id": mask_id,
                "provider": str(props.get("provider") or authority_profile),
                "reliability_tier": str(props.get("reliability_tier") or "gold"),
                "valid_from": valid_from,
                "valid_to": valid_to,
                "coverage_start": valid_from,
                "coverage_end": valid_to,
                "geom_json": json.dumps(mp_geojson),
                "is_active": is_active,
                "authority_profile": authority_profile,
                "tier_policy": tier_policy,
                "run_id": run_id,
                "source_uri": source_uri,
                "source_version": source_version,
                "provenance_json": json.dumps(provenance),
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
            authority_profile,
            tier_policy,
            run_id,
            source_uri,
            source_version,
            coverage_start,
            coverage_end,
            provenance_json,
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
            :authority_profile,
            :tier_policy,
            :run_id,
            :source_uri,
            :source_version,
            :coverage_start,
            :coverage_end,
            CAST(:provenance_json AS json),
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
            authority_profile = EXCLUDED.authority_profile,
            tier_policy = EXCLUDED.tier_policy,
            run_id = EXCLUDED.run_id,
            source_uri = EXCLUDED.source_uri,
            source_version = EXCLUDED.source_version,
            coverage_start = EXCLUDED.coverage_start,
            coverage_end = EXCLUDED.coverage_end,
            provenance_json = EXCLUDED.provenance_json,
            updated_at = NOW()
        """
    )

    with get_engine().begin() as conn:
        result = conn.execute(upsert, params)
    return {
        "input_features": len(features),
        "loaded_rows": len(params),
        "rowcount": int(result.rowcount or 0),
        "authority_profile": authority_profile,
        "run_id": run_id,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build perimeter_coverage_masks from authoritative geometry source."
    )
    parser.add_argument("--input", required=True, help="Path to authoritative GeoJSON geometry")
    parser.add_argument("--authority-profile", required=True, help="Coverage authority profile id")
    parser.add_argument(
        "--tier-policy",
        default="silver_gold",
        choices=["gold_only", "silver_only", "silver_gold"],
    )
    parser.add_argument("--source-uri", required=True, help="Authoritative source URI")
    parser.add_argument("--source-version", required=True, help="Dataset/API version string")
    parser.add_argument(
        "--run-source-profile",
        default=None,
        help=(
            "Source profile to resolve latest successful ingest run id "
            "(e.g., wfigs_interagency_perimeters_full_history)."
        ),
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Optional authoritative ingest run id. "
            "If omitted, resolves from --run-source-profile, else falls back to latest successful run."
        ),
    )
    parser.add_argument("--valid-from", default=None)
    parser.add_argument("--valid-to", default=None)
    parser.add_argument("--inactive", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_id = str(args.run_id).strip() if args.run_id else None
    if not run_id:
        source_profile = str(args.run_source_profile).strip() if args.run_source_profile else None
        run_id = _latest_successful_run_id(source_profile)
    if not run_id:
        raise SystemExit(
            "No successful authoritative_perimeter_ingest_runs found. "
            "Run ingest.wfigs_authority_ingest first or pass --run-id/--run-source-profile."
        )

    summary = build_coverage_masks(
        input_path=Path(args.input),
        authority_profile=str(args.authority_profile),
        tier_policy=str(args.tier_policy),
        source_uri=str(args.source_uri),
        source_version=str(args.source_version),
        run_id=run_id,
        default_valid_from=_parse_dt(args.valid_from),
        default_valid_to=_parse_dt(args.valid_to),
        default_is_active=not bool(args.inactive),
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
