#!/usr/bin/env python3
"""Validate curated industrial authority file against profile mapping and hard requirements."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROFILE_CONFIG = REPO_ROOT / "configs" / "industrial_authority_profiles.yaml"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ingest.industrial_taxonomy import as_iso3


def _parse_dt_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def _load_profiles(config_path: Path) -> dict[str, dict[str, Any]]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise SystemExit("Profile config must include a non-empty 'profiles' map")
    return profiles


def _load_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, low_memory=False)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, dict) and isinstance(payload.get("features"), list):
            rows = [feature.get("properties") or {} for feature in payload["features"]]
            return pd.DataFrame(rows)
        if isinstance(payload, dict):
            return pd.DataFrame([payload])
    raise SystemExit(f"Unsupported file format: {path}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate curated industrial input")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", default=str(DEFAULT_PROFILE_CONFIG))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    profile_name = str(args.profile).strip()
    input_path = Path(args.input).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    profiles = _load_profiles(config_path)
    if profile_name not in profiles:
        raise SystemExit(f"Unknown profile: {profile_name}")

    profile = profiles[profile_name]
    column_map = dict((profile.get("adapters") or {}).get("column_map") or {})
    df = _load_frame(input_path)

    required_columns = sorted({str(col) for col in column_map.values() if str(col).strip()})
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns for profile={profile_name}: {', '.join(missing)}")

    source_col = column_map.get("source_id")
    lat_col = column_map.get("lat")
    lon_col = column_map.get("lon")
    iso_col = column_map.get("country_iso3")
    last_verified_col = column_map.get("last_verified_at")
    verification_mode = str(profile.get("verification_mode") or "").strip().lower()

    if not source_col or not lat_col or not lon_col:
        raise SystemExit("Profile column_map must include source_id, lat, lon")

    source_series = df[source_col].astype(str).str.strip()
    source_missing = int((source_series == "").sum())

    lat_numeric = pd.to_numeric(df[lat_col], errors="coerce")
    lon_numeric = pd.to_numeric(df[lon_col], errors="coerce")
    invalid_lat = int((lat_numeric.isna() | (lat_numeric < -90) | (lat_numeric > 90)).sum())
    invalid_lon = int((lon_numeric.isna() | (lon_numeric < -180) | (lon_numeric > 180)).sum())

    iso_invalid = 0
    if iso_col and iso_col in df.columns:
        for raw in df[iso_col].tolist():
            token = str(raw).strip()
            if not token:
                iso_invalid += 1
                continue
            if as_iso3(token) is None:
                iso_invalid += 1

    invalid_last_verified = 0
    if last_verified_col and last_verified_col in df.columns:
        parsed = _parse_dt_series(df[last_verified_col])
        invalid_last_verified = int(parsed.isna().sum())
        if verification_mode == "hybrid" and invalid_last_verified > 0:
            raise SystemExit(
                f"Hybrid profile requires parseable {last_verified_col}; invalid rows={invalid_last_verified}"
            )

    failures: list[str] = []
    if source_missing > 0:
        failures.append(f"empty source_id rows={source_missing}")
    if invalid_lat > 0:
        failures.append(f"invalid latitude rows={invalid_lat}")
    if invalid_lon > 0:
        failures.append(f"invalid longitude rows={invalid_lon}")
    if iso_invalid > 0:
        failures.append(f"invalid country ISO3 rows={iso_invalid}")

    summary = {
        "profile": profile_name,
        "input": str(input_path),
        "rows": int(len(df)),
        "source_missing": source_missing,
        "invalid_lat": invalid_lat,
        "invalid_lon": invalid_lon,
        "invalid_iso3": iso_invalid,
        "invalid_last_verified": invalid_last_verified,
        "status": "ok" if not failures else "failed",
    }

    print(json.dumps(summary))
    if failures:
        raise SystemExit("Validation failed: " + "; ".join(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
