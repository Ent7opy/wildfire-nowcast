#!/usr/bin/env python3
"""Build BR hybrid curated file from CTF identity + IBGE geocoded coordinate base."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IBGE_DIR = REPO_ROOT / "data" / "authority" / "industrial" / "br" / "raw_ibge_csvs"
CATALOG_URL = "https://dados.gov.br/api/publico/conjuntos-dados/buscar"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm_col(name: str) -> str:
    token = str(name).strip().lower()
    token = (
        token.replace("á", "a")
        .replace("à", "a")
        .replace("â", "a")
        .replace("ã", "a")
        .replace("é", "e")
        .replace("ê", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ô", "o")
        .replace("õ", "o")
        .replace("ú", "u")
        .replace("ç", "c")
    )
    return re.sub(r"[^a-z0-9]", "", token)


def _find_column(columns: list[str], aliases: list[str]) -> str | None:
    normalized = {_norm_col(col): col for col in columns}
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    return None


def _parse_dt(value: Any) -> str:
    token = str(value or "").strip()
    if not token or token.lower() in {"nan", "none", "null"}:
        return ""
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(token, fmt).date().isoformat()
        except ValueError:
            continue
    try:
        parsed = pd.to_datetime(token, errors="coerce", utc=True)
    except Exception:
        parsed = pd.NaT
    if pd.isna(parsed):
        return ""
    return parsed.date().isoformat()


def _digits(value: Any) -> str:
    return re.sub(r"\D", "", str(value or ""))


def _norm_mun(value: Any) -> str:
    digits = _digits(value)
    if len(digits) >= 7:
        return digits[:7]
    return ""


def _norm_cnpj(value: Any) -> str:
    digits = _digits(value)
    if len(digits) < 14:
        return ""
    return digits[:14]


def _extract_date_from_header(path: Path) -> str:
    pattern = re.compile(
        r"data\s+de\s+extrac[aã]o[^0-9]*(\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2})",
        re.IGNORECASE,
    )
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for _ in range(40):
                line = handle.readline()
                if not line:
                    break
                match = pattern.search(line)
                if match:
                    return _parse_dt(match.group(1))
    except Exception:
        return ""
    return ""


def _list_ctf_inputs(input_paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in input_paths:
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            raise SystemExit(f"CTF input not found: {path}")
        if path.is_dir():
            files.extend(sorted(path.glob("*.csv")))
        else:
            files.append(path)
    if not files:
        raise SystemExit("No CTF CSV files found")
    return files


def _catalog_ctf_sources() -> dict[str, Any]:
    try:
        response = httpx.get(CATALOG_URL, params={"nome": "ctf"}, timeout=60)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        return {"error": str(exc), "records": []}

    out: list[dict[str, Any]] = []
    for rec in payload.get("registros") or []:
        org = str(rec.get("organizationName") or "")
        if "ibama" not in org.lower():
            continue
        resources = rec.get("resourcesAcessoRapido") or []
        out.append(
            {
                "id": rec.get("id"),
                "name": rec.get("name"),
                "title": rec.get("title"),
                "organizationName": org,
                "resources": [
                    {
                        "format": r.get("format"),
                        "name": r.get("name"),
                        "url": r.get("url"),
                    }
                    for r in resources
                ],
            }
        )
    return {"records": out}


def _read_csv_with_sep(path: Path) -> pd.DataFrame:
    for sep in (";", ","):
        try:
            df = pd.read_csv(path, sep=sep, encoding="utf-8", low_memory=False)
            if len(df.columns) > 1:
                return df
        except Exception:
            continue
    return pd.read_csv(path, sep=None, engine="python", encoding="utf-8", low_memory=False)


def _resolve_ctf_columns(columns: list[str]) -> dict[str, str | None]:
    return {
        "facility_id": _find_column(
            columns,
            [
                "cnpj",
                "cnpjcpf",
                "cnpjcpfresponsavel",
                "codigounidade",
                "codigounidadectf",
                "idunidade",
            ],
        ),
        "facility_name": _find_column(
            columns,
            [
                "razaosocial",
                "nomerazaosocial",
                "nomefantasia",
                "nome",
            ],
        ),
        "activity": _find_column(
            columns,
            [
                "descricaoatividade",
                "descricaoatividadeprincipal",
                "atividade",
                "descricao",
            ],
        ),
        "category": _find_column(
            columns,
            [
                "codigocategoria",
                "categoria",
                "codcategoria",
                "codigocategoriactf",
                "codigoatividade",
            ],
        ),
        "situation": _find_column(columns, ["situacao", "status", "situacaocadastral"]),
        "mun_code": _find_column(
            columns,
            [
                "codmunicipio",
                "codigomunicipio",
                "codigomunicipioibge",
                "codmun",
                "municipioibge",
                "codigoibgemunicipio",
            ],
        ),
        "state": _find_column(columns, ["uf", "siglauf", "siglauf".replace("uf", "siglauf")]),
        "lat": _find_column(columns, ["latitude", "lat"]),
        "lon": _find_column(columns, ["longitude", "lon", "long"]),
        "valid_from": _find_column(
            columns,
            [
                "datainicioatividade",
                "datainicio",
                "datalicencainicio",
                "inicioatividade",
            ],
        ),
        "valid_to": _find_column(
            columns,
            [
                "datafimatividade",
                "datafim",
                "datalicencafim",
                "fimatividade",
            ],
        ),
    }


def _to_category_code(value: Any) -> str:
    token = str(value or "").strip()
    if not token or token.lower() in {"nan", "none", "null"}:
        return ""
    token = token.replace(",", ".")
    try:
        f = float(token)
        if f.is_integer():
            return str(int(f))
    except ValueError:
        pass
    m = re.search(r"\d+", token)
    return m.group(0) if m else token


def _is_active(situation_value: Any) -> bool:
    token = str(situation_value or "").strip().lower()
    if not token:
        return True
    if "inativ" in token or "cancel" in token or "baix" in token or "susp" in token:
        return False
    if "ativ" in token:
        return True
    return True


def _stable_index(key: str, size: int) -> int:
    if size <= 1:
        return 0
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return value % size


def _build_ibge_coordinate_index(
    *,
    ibge_dir: Path,
    target_municipalities: set[str],
    species_codes: set[str],
    chunksize: int,
    sample_cap: int,
) -> tuple[dict[str, list[tuple[float, float]]], dict[str, tuple[float, float, int]]]:
    samples: dict[str, list[tuple[float, float]]] = defaultdict(list)
    stats: dict[str, tuple[float, float, int]] = {}
    sums_lat: dict[str, float] = defaultdict(float)
    sums_lon: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)

    files = sorted(ibge_dir.glob("*.csv"))
    if not files:
        raise SystemExit(f"No IBGE CSV files found in {ibge_dir}")

    usecols = ["COD_MUN", "COD_ESPECIE", "LATITUDE", "LONGITUDE"]
    for file_path in files:
        for chunk in pd.read_csv(
            file_path,
            sep=";",
            usecols=usecols,
            chunksize=chunksize,
            encoding="utf-8",
            low_memory=False,
        ):
            chunk["COD_MUN"] = chunk["COD_MUN"].apply(_norm_mun)
            chunk["COD_ESPECIE"] = chunk["COD_ESPECIE"].astype(str).str.extract(r"(\d+)", expand=False).fillna("")
            if target_municipalities:
                chunk = chunk[chunk["COD_MUN"].isin(target_municipalities)]
            if species_codes:
                chunk = chunk[chunk["COD_ESPECIE"].isin(species_codes)]
            if chunk.empty:
                continue

            lat = pd.to_numeric(chunk["LATITUDE"], errors="coerce")
            lon = pd.to_numeric(chunk["LONGITUDE"], errors="coerce")
            valid = chunk[(lat.between(-90, 90)) & (lon.between(-180, 180))].copy()
            if valid.empty:
                continue

            valid["LATITUDE"] = pd.to_numeric(valid["LATITUDE"], errors="coerce")
            valid["LONGITUDE"] = pd.to_numeric(valid["LONGITUDE"], errors="coerce")

            for row in valid.itertuples(index=False):
                mun = row.COD_MUN
                lat_v = float(row.LATITUDE)
                lon_v = float(row.LONGITUDE)
                counts[mun] += 1
                sums_lat[mun] += lat_v
                sums_lon[mun] += lon_v
                if len(samples[mun]) < sample_cap:
                    samples[mun].append((lat_v, lon_v))

    for mun, cnt in counts.items():
        if cnt > 0:
            stats[mun] = (sums_lat[mun], sums_lon[mun], cnt)

    return samples, stats


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BR hybrid curated industrial file")
    parser.add_argument("--ctf-input", action="append", required=True, help="CTF CSV file or directory (repeatable)")
    parser.add_argument("--ibge-dir", default=str(DEFAULT_IBGE_DIR))
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--categories", default="1,2,4,5", help="CTF category codes to keep")
    parser.add_argument(
        "--species-codes",
        default="",
        help="Comma-separated COD_ESPECIE values considered industrial for IBGE fallback",
    )
    parser.add_argument("--allow-municipality-fallback", action="store_true")
    parser.add_argument("--fallback-sample-cap", type=int, default=2000)
    parser.add_argument("--chunksize", type=int, default=300000)
    parser.add_argument("--extracted-at", default=None, help="Override LastVerifiedAt date (YYYY-MM-DD)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    ctf_files = _list_ctf_inputs(args.ctf_input)
    ibge_dir = Path(args.ibge_dir).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    allowed_categories = {tok.strip() for tok in str(args.categories).split(",") if tok.strip()}
    species_codes = {tok.strip() for tok in str(args.species_codes).split(",") if tok.strip()}

    catalog_snapshot = _catalog_ctf_sources()

    normalized_rows: list[dict[str, Any]] = []
    ctf_files_info: list[dict[str, Any]] = []

    extraction_date_override = _parse_dt(args.extracted_at) if args.extracted_at else ""
    if args.extracted_at and not extraction_date_override:
        raise SystemExit("--extracted-at must be parseable date")

    for ctf_file in ctf_files:
        df = _read_csv_with_sep(ctf_file)
        columns = list(df.columns)
        colmap = _resolve_ctf_columns(columns)

        required_missing = [
            key
            for key in ("facility_id", "facility_name", "activity", "category")
            if not colmap.get(key)
        ]
        if required_missing:
            raise SystemExit(
                f"STOP: CTF file {ctf_file} missing required identity columns: {', '.join(required_missing)}"
            )

        extraction_date = extraction_date_override or _extract_date_from_header(ctf_file)
        if not extraction_date:
            raise SystemExit(
                "STOP: Missing authoritative extraction timestamp for CTF file. "
                "Provide --extracted-at or include 'Data de Extração' in header metadata."
            )

        muni_col = colmap.get("mun_code")
        if not muni_col:
            raise SystemExit(f"STOP: CTF file {ctf_file} missing municipality code column for join")

        file_rows_before = len(df)
        kept = 0
        for item in df.itertuples(index=False):
            row = item._asdict()
            facility_id_raw = row.get(colmap["facility_id"])  # type: ignore[index]
            cnpj = _norm_cnpj(facility_id_raw)
            facility_id = cnpj or str(facility_id_raw or "").strip()
            if not facility_id:
                continue

            category_code = _to_category_code(row.get(colmap["category"]))  # type: ignore[index]
            if allowed_categories and category_code not in allowed_categories:
                continue

            mun_code = _norm_mun(row.get(muni_col))
            if not mun_code:
                continue

            lat = None
            lon = None
            if colmap.get("lat") and colmap.get("lon"):
                try:
                    lat = float(row.get(colmap["lat"]))  # type: ignore[index]
                    lon = float(row.get(colmap["lon"]))  # type: ignore[index]
                    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                        lat = None
                        lon = None
                except Exception:
                    lat = None
                    lon = None

            valid_from = _parse_dt(row.get(colmap["valid_from"])) if colmap.get("valid_from") else ""
            valid_to = _parse_dt(row.get(colmap["valid_to"])) if colmap.get("valid_to") else ""

            normalized_rows.append(
                {
                    "facility_id": facility_id,
                    "facility_name": str(row.get(colmap["facility_name"]) or "").strip(),  # type: ignore[index]
                    "activity": str(row.get(colmap["activity"]) or "").strip(),  # type: ignore[index]
                    "category_code": category_code,
                    "mun_code": mun_code,
                    "state_code": str(row.get(colmap["state"]) or "").strip() if colmap.get("state") else "",
                    "is_active": _is_active(row.get(colmap["situation"])) if colmap.get("situation") else True,
                    "last_verified_at": extraction_date,
                    "valid_from": valid_from,
                    "valid_to": valid_to,
                    "lat": lat,
                    "lon": lon,
                }
            )
            kept += 1

        ctf_files_info.append(
            {
                "path": str(ctf_file),
                "rows_before": int(file_rows_before),
                "rows_after_category_filter": int(kept),
                "columns": columns,
                "column_mapping": colmap,
                "extraction_date": extraction_date,
            }
        )

    if not normalized_rows:
        raise SystemExit("STOP: No rows remained after CTF category/identity filters")

    needs_coord = [row for row in normalized_rows if row["lat"] is None or row["lon"] is None]
    fallback_summary: dict[str, Any] = {
        "enabled": bool(args.allow_municipality_fallback),
        "species_codes": sorted(species_codes),
        "rows_needing_coordinates": len(needs_coord),
        "rows_filled_from_ibge": 0,
        "rows_unmatched": 0,
    }

    if needs_coord:
        if not args.allow_municipality_fallback:
            raise SystemExit(
                "STOP: CTF rows without coordinates require municipality fallback, but "
                "--allow-municipality-fallback is disabled."
            )
        if not species_codes:
            raise SystemExit(
                "STOP: --species-codes is required for municipality fallback to avoid unbounded coordinate matching."
            )

        target_municipalities = {row["mun_code"] for row in needs_coord if row["mun_code"]}
        samples, muni_stats = _build_ibge_coordinate_index(
            ibge_dir=ibge_dir,
            target_municipalities=target_municipalities,
            species_codes=species_codes,
            chunksize=int(args.chunksize),
            sample_cap=int(args.fallback_sample_cap),
        )

        for row in needs_coord:
            mun = row["mun_code"]
            sample = samples.get(mun) or []
            if sample:
                idx_key = row["facility_id"] or f"{mun}:{row['facility_name']}"
                idx = _stable_index(str(idx_key), len(sample))
                lat, lon = sample[idx]
                row["lat"] = lat
                row["lon"] = lon
                fallback_summary["rows_filled_from_ibge"] += 1
                continue

            stat = muni_stats.get(mun)
            if stat is not None:
                sum_lat, sum_lon, cnt = stat
                row["lat"] = sum_lat / cnt
                row["lon"] = sum_lon / cnt
                fallback_summary["rows_filled_from_ibge"] += 1
            else:
                fallback_summary["rows_unmatched"] += 1

    output_rows: list[dict[str, Any]] = []
    dropped_invalid_coords = 0
    for row in normalized_rows:
        lat = row["lat"]
        lon = row["lon"]
        if lat is None or lon is None:
            dropped_invalid_coords += 1
            continue
        if not (-90.0 <= float(lat) <= 90.0 and -180.0 <= float(lon) <= 180.0):
            dropped_invalid_coords += 1
            continue

        output_rows.append(
            {
                "Facility_ID": str(row["facility_id"]),
                "FacilityName": str(row["facility_name"]),
                "ActivityType": str(row["activity"]),
                "Latitude": f"{float(lat):.6f}",
                "Longitude": f"{float(lon):.6f}",
                "SectorCode": str(row["category_code"]),
                "CountryISO3": "BRA",
                "StateCode": str(row["state_code"]),
                "LicenseStart": str(row["valid_from"]),
                "LicenseEnd": str(row["valid_to"]),
                "LastVerifiedAt": str(row["last_verified_at"]),
                "is_active": bool(row["is_active"]),
                "source_join_method": "ctf_direct_coord" if row["lat"] is not None and row["lon"] is not None else "ibge_municipality",
            }
        )

    if not output_rows:
        raise SystemExit(
            "STOP: No BR curated rows could be produced without violating authoritative constraints."
        )

    fieldnames = [
        "Facility_ID",
        "FacilityName",
        "ActivityType",
        "Latitude",
        "Longitude",
        "SectorCode",
        "CountryISO3",
        "StateCode",
        "LicenseStart",
        "LicenseEnd",
        "LastVerifiedAt",
        "is_active",
        "source_join_method",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    manifest = {
        "generated_at": _now_iso(),
        "source_profile": "br_ibama_sigel_hybrid",
        "source_version": "ibama_sigel_2025",
        "catalog_snapshot": catalog_snapshot,
        "ctf_files": ctf_files_info,
        "ibge_dir": str(ibge_dir),
        "categories_kept": sorted(allowed_categories),
        "fallback_summary": fallback_summary,
        "rows_input": len(normalized_rows),
        "rows_output": len(output_rows),
        "rows_dropped_invalid_coords": dropped_invalid_coords,
        "output_csv": str(out_path),
        "warning": (
            "WARNING: Municipality fallback assigns coordinates from IBGE municipality-level industrial points "
            "and must be reviewed for promotion-grade provenance."
            if args.allow_municipality_fallback
            else None
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({"output": str(out_path), "manifest": str(manifest_path), "rows_output": len(output_rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
