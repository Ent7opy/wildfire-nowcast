"""Authoritative industrial source ingestion for denoiser masking."""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import yaml
from sqlalchemy import JSON, bindparam, text

from ingest.industrial_taxonomy import as_iso3, infer_thermal_potential_class, normalize_taxonomy
from ingest.repository import get_engine

LOGGER = logging.getLogger("industrial_ingest")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE_CONFIG = REPO_ROOT / "configs" / "industrial_authority_profiles.yaml"


@dataclass
class IngestStats:
    records_fetched: int = 0
    records_upserted: int = 0
    records_skipped: int = 0


def _parse_dt(value: str | None) -> datetime | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    if len(raw) == 10:
        return datetime.strptime(raw, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _coerce_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "y", "on"}:
        return True
    if token in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _ensure_profile_required_fields(profile_name: str, profile: dict[str, Any]) -> None:
    missing = [
        key
        for key in (
            "authority_name",
            "source_uri",
            "source_version",
            "source_profile",
            "authority_tier",
            "verification_mode",
            "sector_taxonomy",
            "adapters",
        )
        if not profile.get(key)
    ]
    if missing:
        raise ValueError(f"Profile {profile_name!r} missing required fields: {', '.join(missing)}")


def _load_profiles(config_path: Path) -> dict[str, dict[str, Any]]:
    if not config_path.exists():
        raise FileNotFoundError(f"Profile config not found: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("industrial_authority_profiles.yaml must contain a non-empty 'profiles' mapping")
    for name, profile in profiles.items():
        if not isinstance(profile, dict):
            raise ValueError(f"Profile {name!r} must be a mapping")
        _ensure_profile_required_fields(name, profile)
    return profiles


def _load_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".json"}:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, dict):
            if "features" in payload and isinstance(payload["features"], list):
                rows = []
                for feat in payload["features"]:
                    props = feat.get("properties") or {}
                    rows.append(props)
                return pd.DataFrame(rows)
            return pd.DataFrame([payload])
    raise ValueError(f"Unsupported curated file format: {path}")


def _require_columns(df: pd.DataFrame, required: list[str], profile_name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Profile {profile_name!r} missing required columns in source data: {', '.join(missing)}"
        )


def _check_endpoint(profile: dict[str, Any], timeout_seconds: float) -> None:
    if not _coerce_bool(profile.get("endpoint_required"), default=False):
        return
    url = str(profile.get("endpoint_check_url") or profile.get("source_uri") or "").strip()
    if not url:
        raise ValueError("Hybrid/endpoint profile requires endpoint_check_url or source_uri")
    try:
        with httpx.Client(timeout=timeout_seconds, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
    except Exception as exc:
        raise RuntimeError(f"Endpoint verification failed for profile={profile.get('source_profile')}: {exc}")


def _load_source_frame(
    *,
    profile_name: str,
    profile: dict[str, Any],
    curated_files: list[Path],
    timeout_seconds: float,
) -> pd.DataFrame:
    adapter = profile.get("adapters") or {}
    adapter_type = str(adapter.get("type") or "").strip().lower()

    if adapter_type == "http_csv":
        source_uri = str(profile["source_uri"])
        LOGGER.info("Downloading source profile=%s uri=%s", profile_name, source_uri)
        return pd.read_csv(source_uri)

    if adapter_type in {"curated_csv", "hybrid_curated_csv"}:
        if not curated_files:
            raise ValueError(
                f"Profile {profile_name!r} requires --curated-file (repeatable) for adapter {adapter_type}"
            )
        frames = [_load_frame(path) for path in curated_files]
        return pd.concat(frames, ignore_index=True)

    raise ValueError(f"Unsupported adapter type for profile {profile_name!r}: {adapter_type}")


def _maybe_parse_time_column(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def _filter_window(
    frame: pd.DataFrame,
    *,
    start_time: datetime | None,
    end_time: datetime | None,
    column_candidates: list[str],
) -> pd.DataFrame:
    if start_time is None and end_time is None:
        return frame

    col = None
    for candidate in column_candidates:
        if candidate in frame.columns:
            col = candidate
            break
    if col is None:
        return frame

    ts = _maybe_parse_time_column(frame[col])
    mask = pd.Series(True, index=frame.index)
    if start_time is not None:
        mask &= ts >= start_time
    if end_time is not None:
        mask &= ts <= end_time
    return frame.loc[mask].copy()


def _normalize_rows(
    *,
    profile_name: str,
    profile: dict[str, Any],
    frame: pd.DataFrame,
    run_id: str,
) -> tuple[list[dict[str, Any]], int]:
    adapter = profile["adapters"]
    column_map = dict(adapter.get("column_map") or {})
    required_keys = ["source_id", "lat", "lon"]
    required_columns = [column_map[k] for k in required_keys if k in column_map]
    _require_columns(frame, required_columns, profile_name)

    sector_taxonomy = normalize_taxonomy(str(profile.get("sector_taxonomy")))
    tier = str(profile.get("authority_tier")).strip().lower()
    verification_mode = str(profile.get("verification_mode")).strip().lower()
    source_profile = str(profile.get("source_profile"))
    source_uri = str(profile.get("source_uri"))
    source_version = str(profile.get("source_version"))
    authority_name = str(profile.get("authority_name"))
    country_default = as_iso3(profile.get("country_iso3_default"))
    coordinate_precision_type = str(profile.get("coordinate_precision_type") or "reported").strip().lower()
    coordinate_precision_default = float(profile.get("coordinate_precision_m") or 1000.0)
    allowed_sector_values = set(
        str(v).strip().lower()
        for v in (adapter.get("allowed_primary_fuel") or [])
        if str(v).strip()
    )

    active_values = set(str(v).strip().lower() for v in adapter.get("active_values", ["active", "yes", "1", "true"]))
    active_col = column_map.get("is_active")

    rows: list[dict[str, Any]] = []
    skipped = 0

    for _, item in frame.iterrows():
        source_id = str(item.get(column_map.get("source_id", ""), "")).strip()
        if not source_id:
            skipped += 1
            continue

        lat_raw = item.get(column_map.get("lat", ""))
        lon_raw = item.get(column_map.get("lon", ""))
        try:
            lat = float(lat_raw)
            lon = float(lon_raw)
        except (TypeError, ValueError):
            skipped += 1
            continue
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            skipped += 1
            continue

        country_iso3 = as_iso3(item.get(column_map.get("country_iso3", ""))) or country_default
        if country_iso3 is None:
            skipped += 1
            continue

        sector_code = str(item.get(column_map.get("sector_code", ""), "")).strip() or None
        facility_type = str(item.get(column_map.get("type", ""), "")).strip() or None
        if allowed_sector_values:
            probe = (sector_code or facility_type or "").strip().lower()
            if probe not in allowed_sector_values:
                skipped += 1
                continue
        tpc = infer_thermal_potential_class(
            sector_code=sector_code,
            sector_taxonomy=sector_taxonomy,
            facility_type=facility_type,
        )

        coord_precision_raw = item.get(column_map.get("coordinate_precision_m", ""))
        try:
            coordinate_precision_m = float(coord_precision_raw)
        except (TypeError, ValueError):
            coordinate_precision_m = coordinate_precision_default

        valid_from = _parse_dt(str(item.get(column_map.get("valid_from", ""), "")).strip() or None)
        valid_to = _parse_dt(str(item.get(column_map.get("valid_to", ""), "")).strip() or None)
        last_verified_raw = str(item.get(column_map.get("last_verified_at", ""), "")).strip() or None
        last_verified_at = _parse_dt(last_verified_raw)
        if verification_mode == "hybrid" and last_verified_at is None:
            skipped += 1
            continue
        if last_verified_at is None:
            last_verified_at = _now_utc()

        if active_col:
            is_active = str(item.get(active_col, "")).strip().lower() in active_values
        else:
            is_active = _coerce_bool(item.get("is_active"), default=True)

        name = str(item.get(column_map.get("name", ""), "")).strip() or None
        jurisdiction_code = str(item.get(column_map.get("jurisdiction_code", ""), "")).strip() or None

        row_meta = {
            "profile": profile_name,
            "authority_name": authority_name,
            "source_uri": source_uri,
            "source_version": source_version,
            "ingested_at": _now_utc().isoformat(),
            "raw_record": {
                key: (None if pd.isna(value) else value)
                for key, value in item.to_dict().items()
            },
        }

        rows.append(
            {
                "name": name,
                "type": facility_type,
                "source": source_profile,
                "source_version": source_version,
                "source_profile": source_profile,
                "authority_name": authority_name,
                "authority_tier": tier,
                "country_iso3": country_iso3,
                "jurisdiction_code": jurisdiction_code,
                "source_id": source_id,
                "sector_code": sector_code,
                "sector_taxonomy": sector_taxonomy,
                "thermal_potential_class": float(tpc),
                "coordinate_precision_type": coordinate_precision_type,
                "coordinate_precision_m": float(coordinate_precision_m),
                "verification_mode": verification_mode,
                "valid_from": valid_from,
                "valid_to": valid_to,
                "last_verified_at": last_verified_at,
                "is_active": bool(is_active),
                "run_id": run_id,
                "lat": float(lat),
                "lon": float(lon),
                "meta": row_meta,
            }
        )

    return rows, skipped


def _create_run(
    *,
    run_id: str,
    source_profile: str,
    source_uri: str,
    source_version: str,
) -> None:
    stmt = text(
        """
        INSERT INTO authoritative_industrial_ingest_runs (
            run_id,
            source_profile,
            status,
            started_at,
            source_uri,
            source_version,
            created_at,
            updated_at
        )
        VALUES (
            :run_id,
            :source_profile,
            'running',
            NOW(),
            :source_uri,
            :source_version,
            NOW(),
            NOW()
        )
        """
    )
    with get_engine().begin() as conn:
        conn.execute(
            stmt,
            {
                "run_id": run_id,
                "source_profile": source_profile,
                "source_uri": source_uri,
                "source_version": source_version,
            },
        )


def _finish_run(
    *,
    run_id: str,
    status: str,
    stats: IngestStats,
    error_text: str | None = None,
    metrics: dict[str, Any] | None = None,
) -> None:
    stmt = text(
        """
        UPDATE authoritative_industrial_ingest_runs
        SET
            status = :status,
            records_fetched = :records_fetched,
            records_upserted = :records_upserted,
            records_skipped = :records_skipped,
            error_text = :error_text,
            metrics_json = CAST(:metrics_json AS json),
            finished_at = NOW(),
            updated_at = NOW()
        WHERE run_id = :run_id
        """
    )
    with get_engine().begin() as conn:
        conn.execute(
            stmt,
            {
                "run_id": run_id,
                "status": status,
                "records_fetched": int(stats.records_fetched),
                "records_upserted": int(stats.records_upserted),
                "records_skipped": int(stats.records_skipped),
                "error_text": error_text,
                "metrics_json": json.dumps(metrics or {}),
            },
        )


def _upsert_sources(rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0

    stmt = text(
        """
        INSERT INTO industrial_sources (
            name,
            type,
            source,
            source_version,
            source_profile,
            authority_name,
            authority_tier,
            country_iso3,
            jurisdiction_code,
            source_id,
            sector_code,
            sector_taxonomy,
            thermal_potential_class,
            coordinate_precision_type,
            coordinate_precision_m,
            verification_mode,
            valid_from,
            valid_to,
            last_verified_at,
            is_active,
            run_id,
            geom,
            meta,
            created_at
        ) VALUES (
            :name,
            :type,
            :source,
            :source_version,
            :source_profile,
            :authority_name,
            :authority_tier,
            :country_iso3,
            :jurisdiction_code,
            :source_id,
            :sector_code,
            :sector_taxonomy,
            :thermal_potential_class,
            :coordinate_precision_type,
            :coordinate_precision_m,
            :verification_mode,
            :valid_from,
            :valid_to,
            :last_verified_at,
            :is_active,
            :run_id,
            ST_SetSRID(ST_MakePoint(:lon, :lat), 4326),
            :meta,
            NOW()
        )
        ON CONFLICT (source_profile, source_id) DO UPDATE SET
            name = EXCLUDED.name,
            type = EXCLUDED.type,
            source = EXCLUDED.source,
            source_version = EXCLUDED.source_version,
            authority_name = EXCLUDED.authority_name,
            authority_tier = EXCLUDED.authority_tier,
            country_iso3 = EXCLUDED.country_iso3,
            jurisdiction_code = EXCLUDED.jurisdiction_code,
            sector_code = EXCLUDED.sector_code,
            sector_taxonomy = EXCLUDED.sector_taxonomy,
            thermal_potential_class = EXCLUDED.thermal_potential_class,
            coordinate_precision_type = EXCLUDED.coordinate_precision_type,
            coordinate_precision_m = EXCLUDED.coordinate_precision_m,
            verification_mode = EXCLUDED.verification_mode,
            valid_from = EXCLUDED.valid_from,
            valid_to = EXCLUDED.valid_to,
            last_verified_at = EXCLUDED.last_verified_at,
            is_active = EXCLUDED.is_active,
            run_id = EXCLUDED.run_id,
            geom = EXCLUDED.geom,
            meta = EXCLUDED.meta
        """
    ).bindparams(bindparam("meta", type_=JSON))

    with get_engine().begin() as conn:
        result = conn.execute(stmt, rows)
    return int(result.rowcount or 0)


def _resolve_run_id(source_profile: str, explicit_run_id: str | None) -> str:
    if explicit_run_id:
        return explicit_run_id
    return f"{source_profile}_{_now_utc().strftime('%Y%m%d%H%M%S')}"


def ingest_sources(
    *,
    source_profile: str,
    config_path: Path,
    start_time: datetime | None,
    end_time: datetime | None,
    curated_files: list[Path],
    run_id: str | None,
    timeout_seconds: float,
    dry_run: bool,
) -> dict[str, Any]:
    profiles = _load_profiles(config_path)
    if source_profile not in profiles:
        raise ValueError(f"Unknown source profile: {source_profile}")

    profile = profiles[source_profile]
    resolved_run_id = _resolve_run_id(source_profile, run_id)

    _check_endpoint(profile, timeout_seconds=timeout_seconds)

    frame = _load_source_frame(
        profile_name=source_profile,
        profile=profile,
        curated_files=curated_files,
        timeout_seconds=timeout_seconds,
    )

    frame = _filter_window(
        frame,
        start_time=start_time,
        end_time=end_time,
        column_candidates=[
            (profile.get("adapters") or {}).get("column_map", {}).get("last_verified_at", ""),
            (profile.get("adapters") or {}).get("column_map", {}).get("valid_from", ""),
            "updated_at",
        ],
    )

    normalized_rows, skipped = _normalize_rows(
        profile_name=source_profile,
        profile=profile,
        frame=frame,
        run_id=resolved_run_id,
    )

    stats = IngestStats(
        records_fetched=int(len(frame)),
        records_upserted=0,
        records_skipped=int(skipped),
    )

    if dry_run:
        return {
            "run_id": resolved_run_id,
            "source_profile": source_profile,
            "source_uri": profile.get("source_uri"),
            "source_version": profile.get("source_version"),
            "records_fetched": stats.records_fetched,
            "records_upserted": 0,
            "records_skipped": stats.records_skipped,
            "dry_run": True,
        }

    _create_run(
        run_id=resolved_run_id,
        source_profile=source_profile,
        source_uri=str(profile.get("source_uri")),
        source_version=str(profile.get("source_version")),
    )

    try:
        stats.records_upserted = _upsert_sources(normalized_rows)
        _finish_run(
            run_id=resolved_run_id,
            status="succeeded",
            stats=stats,
            metrics={
                "verification_mode": profile.get("verification_mode"),
                "authority_tier": profile.get("authority_tier"),
            },
        )
    except Exception as exc:
        _finish_run(
            run_id=resolved_run_id,
            status="failed",
            stats=stats,
            error_text=str(exc),
        )
        raise

    return {
        "run_id": resolved_run_id,
        "source_profile": source_profile,
        "source_uri": profile.get("source_uri"),
        "source_version": profile.get("source_version"),
        "records_fetched": stats.records_fetched,
        "records_upserted": stats.records_upserted,
        "records_skipped": stats.records_skipped,
        "dry_run": False,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest authoritative industrial source data.")
    parser.add_argument("--source-profile", required=False, help="Profile key from industrial_authority_profiles.yaml")
    parser.add_argument("--config", default=str(DEFAULT_PROFILE_CONFIG))
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--curated-file",
        action="append",
        default=[],
        help="Path to curated CSV/Parquet/JSON (repeatable)",
    )
    parser.add_argument("--timeout-seconds", type=float, default=45.0)
    parser.add_argument("--dry-run", action="store_true")

    # Backward compatible shortcuts.
    parser.add_argument("--wri", action="store_true", help="Shortcut for --source-profile global_wri_gppd_silver")
    return parser.parse_args(argv)


def run_industrial_ingest(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    source_profile = args.source_profile
    if args.wri and not source_profile:
        source_profile = "global_wri_gppd_silver"
    if not source_profile:
        raise SystemExit("--source-profile is required (or use --wri shortcut)")

    start_time = _parse_dt(args.start)
    end_time = _parse_dt(args.end)
    if start_time and end_time and end_time < start_time:
        raise SystemExit("--end must be >= --start")

    curated_files = [Path(path).expanduser().resolve() for path in args.curated_file]
    missing = [str(path) for path in curated_files if not path.exists()]
    if missing:
        raise SystemExit(f"Curated files not found: {', '.join(missing)}")

    summary = ingest_sources(
        source_profile=str(source_profile),
        config_path=Path(args.config).expanduser().resolve(),
        start_time=start_time,
        end_time=end_time,
        curated_files=curated_files,
        run_id=args.run_id,
        timeout_seconds=float(args.timeout_seconds),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(summary))
    return 0


def main() -> None:
    raise SystemExit(run_industrial_ingest())


if __name__ == "__main__":
    main()
