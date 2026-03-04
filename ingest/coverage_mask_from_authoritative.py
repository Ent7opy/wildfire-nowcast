"""Build jurisdictional perimeter_coverage_masks from authoritative monitoring extents.

Semantic model:
- Coverage masks represent where/when an authority can produce perimeter truth.
- They are not unions of observed fire footprints.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text

from ingest.repository import get_engine


@dataclass(frozen=True)
class JurisdictionMaskSpec:
    authority_profile: str
    source_profile: str
    provider: str
    reliability_tier: str
    tier_policy: str
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float
    valid_from: datetime
    valid_to: datetime
    source_uri: str


def _dt_utc(raw: str) -> datetime:
    return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc)


_JURISDICTION_SPECS: tuple[JurisdictionMaskSpec, ...] = (
    # WFIGS (US): valid for Jan-Aug 2025 only.
    JurisdictionMaskSpec(
        authority_profile="wfigs_us",
        source_profile="wfigs_2025_ytd_perimeters",
        provider="WFIGS",
        reliability_tier="gold",
        tier_policy="jurisdiction_extent",
        min_lon=-179.5,
        min_lat=18.0,
        max_lon=-66.0,
        max_lat=72.0,
        valid_from=_dt_utc("2025-01-01T00:00:00Z"),
        valid_to=_dt_utc("2025-08-31T23:59:59Z"),
        source_uri="authoritative://wfigs_us/jurisdiction_extent",
    ),
    # CWFIS (Canada): valid only through 2024-12-04 due to completeness lag.
    JurisdictionMaskSpec(
        authority_profile="cwfis_ca",
        source_profile="cwfis_nbac_historical",
        provider="CWFIS",
        reliability_tier="silver",
        tier_policy="jurisdiction_extent",
        min_lon=-141.0,
        min_lat=41.5,
        max_lon=-52.0,
        max_lat=84.5,
        valid_from=_dt_utc("2000-01-01T00:00:00Z"),
        valid_to=_dt_utc("2024-12-04T23:59:59Z"),
        source_uri="authoritative://cwfis_ca/jurisdiction_extent",
    ),
    # Copernicus (EU): narrow high-confidence validity window only.
    JurisdictionMaskSpec(
        authority_profile="copernicus_eu",
        source_profile="copernicus_ems_wildfire_activations",
        provider="Copernicus",
        reliability_tier="silver",
        tier_policy="jurisdiction_extent",
        min_lon=-12.0,
        min_lat=34.0,
        max_lon=32.0,
        max_lat=72.0,
        valid_from=_dt_utc("2025-08-08T00:00:00Z"),
        valid_to=_dt_utc("2025-08-15T23:59:59Z"),
        source_uri="authoritative://copernicus_eu/jurisdiction_extent",
    ),
)


def _latest_successful_runs(*, source_profiles: list[str]) -> dict[str, dict[str, Any]]:
    stmt = text(
        """
        SELECT DISTINCT ON (source_profile)
            source_profile,
            run_id,
            finished_at,
            source_uri
        FROM authoritative_perimeter_ingest_runs
        WHERE status = 'succeeded'
          AND source_profile = ANY(:source_profiles)
        ORDER BY source_profile, finished_at DESC NULLS LAST, started_at DESC
        """
    )
    with get_engine().begin() as conn:
        rows = conn.execute(stmt, {"source_profiles": source_profiles}).mappings().all()
    return {str(row["source_profile"]): dict(row) for row in rows}


def rebuild_jurisdictional_masks(*, replace_existing: bool = True) -> dict[str, Any]:
    source_profiles = [spec.source_profile for spec in _JURISDICTION_SPECS]
    run_map = _latest_successful_runs(source_profiles=source_profiles)
    missing = [profile for profile in source_profiles if profile not in run_map]
    if missing:
        raise SystemExit(
            "STOP: We are missing authoritative ingest run metadata for source_profile(s): "
            f"{', '.join(sorted(missing))}. Cannot build jurisdictional coverage masks safely."
        )

    upsert_stmt = text(
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
        )
        VALUES (
            :mask_id,
            :provider,
            :reliability_tier,
            :valid_from,
            :valid_to,
            ST_Multi(ST_MakeEnvelope(:min_lon, :min_lat, :max_lon, :max_lat, 4326)),
            TRUE,
            :authority_profile,
            :tier_policy,
            :run_id,
            :source_uri,
            :source_version,
            :valid_from,
            :valid_to,
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
    delete_stmt = text("DELETE FROM perimeter_coverage_masks")

    params: list[dict[str, Any]] = []
    for spec in _JURISDICTION_SPECS:
        latest = run_map[spec.source_profile]
        run_id = str(latest["run_id"])
        source_version = run_id
        provenance = {
            "mode": "jurisdiction_extent_v1",
            "authority_profile": spec.authority_profile,
            "source_profile": spec.source_profile,
            "source_run_finished_at": (
                latest["finished_at"].isoformat() if latest.get("finished_at") is not None else None
            ),
            "semantic_shift": "monitoring_jurisdiction_not_fire_footprint",
            "bbox": [spec.min_lon, spec.min_lat, spec.max_lon, spec.max_lat],
            "valid_from": spec.valid_from.isoformat(),
            "valid_to": spec.valid_to.isoformat(),
        }
        params.append(
            {
                "mask_id": f"{spec.authority_profile}_jurisdiction_extent_v1",
                "provider": spec.provider,
                "reliability_tier": spec.reliability_tier,
                "valid_from": spec.valid_from,
                "valid_to": spec.valid_to,
                "min_lon": spec.min_lon,
                "min_lat": spec.min_lat,
                "max_lon": spec.max_lon,
                "max_lat": spec.max_lat,
                "authority_profile": spec.authority_profile,
                "tier_policy": spec.tier_policy,
                "run_id": run_id,
                "source_uri": spec.source_uri or str(latest.get("source_uri") or ""),
                "source_version": source_version,
                "provenance_json": json.dumps(provenance),
            }
        )

    with get_engine().begin() as conn:
        if replace_existing:
            conn.execute(delete_stmt)
        result = conn.execute(upsert_stmt, params)

    return {
        "replaced": bool(replace_existing),
        "rows_written": int(result.rowcount or 0),
        "masks": [
            {
                "authority_profile": p["authority_profile"],
                "mask_id": p["mask_id"],
                "run_id": p["run_id"],
                "valid_from": p["valid_from"].isoformat(),
                "valid_to": p["valid_to"].isoformat(),
                "bbox": [p["min_lon"], p["min_lat"], p["max_lon"], p["max_lat"]],
            }
            for p in params
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild perimeter_coverage_masks with jurisdictional monitoring extents."
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-replace",
        action="store_true",
        help="Do not delete existing rows for core authority profiles before upsert.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "masks": [
                        {
                            "authority_profile": spec.authority_profile,
                            "source_profile": spec.source_profile,
                            "bbox": [spec.min_lon, spec.min_lat, spec.max_lon, spec.max_lat],
                            "valid_from": spec.valid_from.isoformat(),
                            "valid_to": spec.valid_to.isoformat(),
                        }
                        for spec in _JURISDICTION_SPECS
                    ],
                }
            )
        )
        return

    summary = rebuild_jurisdictional_masks(replace_existing=not bool(args.no_replace))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
