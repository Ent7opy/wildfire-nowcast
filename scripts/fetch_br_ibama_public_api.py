#!/usr/bin/env python3
"""Probe public IBAMA endpoints and emit curated BR industrial CSV when machine access works."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

CANDIDATE_ENDPOINTS = [
    "https://siscom.ibama.gov.br/geoserver/wfs?service=WFS&version=1.1.0&request=GetCapabilities",
    "https://siscom.ibama.gov.br/geoserver/ows?service=WFS&version=1.1.0&request=GetCapabilities",
    "https://pamgia.ibama.gov.br/server/rest/services?f=pjson",
]
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    ),
    "Accept": "application/json,text/xml,text/html;q=0.9,*/*;q=0.8",
    "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
}
WFS_TYPENAME_RE = re.compile(r"<Name>([^<]+)</Name>", re.IGNORECASE)


@dataclass
class ProbeResult:
    url: str
    status_code: int | None
    content_type: str | None
    body_sha256: str | None
    sample: str
    error: str | None
    fetched_at: str


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _request_with_retry(
    client: httpx.Client,
    *,
    url: str,
    timeout_seconds: float,
    retries: int,
    params: dict[str, Any] | None = None,
) -> httpx.Response:
    attempt = 0
    while True:
        attempt += 1
        try:
            response = client.get(url, params=params, timeout=timeout_seconds)
            return response
        except Exception:
            if attempt >= retries:
                raise
            time.sleep(min(2 * attempt, 6))


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _looks_like_block_page(text: str) -> bool:
    probe = text.lower()
    markers = [
        "cloudflare",
        "attention required",
        "acesso negado",
        "access denied",
        "captcha",
        "forbidden",
    ]
    return any(marker in probe for marker in markers)


def _probe_endpoints(client: httpx.Client, timeout_seconds: float, retries: int) -> list[ProbeResult]:
    results: list[ProbeResult] = []
    for url in CANDIDATE_ENDPOINTS:
        try:
            response = _request_with_retry(
                client,
                url=url,
                timeout_seconds=timeout_seconds,
                retries=retries,
            )
            text = response.text or ""
            results.append(
                ProbeResult(
                    url=url,
                    status_code=int(response.status_code),
                    content_type=response.headers.get("content-type"),
                    body_sha256=_hash_text(text),
                    sample=text[:500],
                    error=None,
                    fetched_at=_iso_now(),
                )
            )
        except Exception as exc:
            results.append(
                ProbeResult(
                    url=url,
                    status_code=None,
                    content_type=None,
                    body_sha256=None,
                    sample="",
                    error=str(exc),
                    fetched_at=_iso_now(),
                )
            )
    return results


def _is_machine_accessible_wfs(probe: ProbeResult) -> bool:
    if probe.status_code != 200:
        return False
    if _looks_like_block_page(probe.sample):
        return False
    return "wfs_capabilities" in probe.sample.lower() or "<wfs:" in probe.sample.lower()


def _is_machine_accessible_arcgis(probe: ProbeResult) -> bool:
    if probe.status_code != 200:
        return False
    if _looks_like_block_page(probe.sample):
        return False
    sample = probe.sample.lower()
    return "services" in sample and ("folders" in sample or "currentversion" in sample)


def _parse_wfs_typenames(xml_text: str) -> list[str]:
    names = [name.strip() for name in WFS_TYPENAME_RE.findall(xml_text)]
    out: list[str] = []
    seen: set[str] = set()
    for name in names:
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _is_industrial_name(name: str) -> bool:
    probe = name.lower()
    keywords = [
        "ind",
        "industr",
        "petro",
        "oleo",
        "gas",
        "min",
        "miner",
        "energia",
        "sider",
        "metal",
        "cimento",
        "term",
        "planta",
        "facility",
    ]
    return any(k in probe for k in keywords)


def _geometry_to_latlon(geometry: dict[str, Any] | None) -> tuple[float, float] | tuple[None, None]:
    if not isinstance(geometry, dict):
        return (None, None)
    if "x" in geometry and "y" in geometry:
        try:
            return (float(geometry["y"]), float(geometry["x"]))
        except (TypeError, ValueError):
            return (None, None)
    coords = geometry.get("coordinates")
    if isinstance(coords, list) and len(coords) >= 2 and isinstance(coords[0], (int, float)):
        try:
            return (float(coords[1]), float(coords[0]))
        except (TypeError, ValueError):
            return (None, None)
    rings = geometry.get("rings")
    if isinstance(rings, list) and rings and isinstance(rings[0], list):
        pts = [pt for ring in rings for pt in ring if isinstance(pt, list) and len(pt) >= 2]
        if pts:
            xs = [float(pt[0]) for pt in pts]
            ys = [float(pt[1]) for pt in pts]
            return (sum(ys) / len(ys), sum(xs) / len(xs))
    return (None, None)


def _first_non_empty(attributes: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        for candidate in (key, key.upper(), key.lower()):
            if candidate in attributes:
                value = str(attributes[candidate]).strip()
                if value and value.lower() not in {"nan", "none", "null"}:
                    return value
    return ""


def _normalize_feature(attributes: dict[str, Any], lat: float, lon: float, fetched_at: str) -> dict[str, str] | None:
    if lat is None or lon is None:
        return None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None

    facility_id = _first_non_empty(
        attributes,
        [
            "facility_id",
            "id",
            "objectid",
            "gid",
            "codigo",
            "cod",
            "empreendimento_id",
        ],
    )
    if not facility_id:
        return None

    facility_name = _first_non_empty(
        attributes,
        ["facility_name", "name", "nome", "empreendimento", "razao_social", "fantasia"],
    )
    activity = _first_non_empty(attributes, ["activity", "atividade", "tipo_atividade", "classe"])
    sector_code = _first_non_empty(attributes, ["sector", "setor", "cnae", "codigo_setor"])
    state_code = _first_non_empty(attributes, ["uf", "estado", "sigla_uf"])
    license_start = _first_non_empty(attributes, ["license_start", "data_inicio", "dt_inicio"])
    license_end = _first_non_empty(attributes, ["license_end", "data_fim", "dt_fim"])
    last_verified = _first_non_empty(
        attributes,
        ["last_verified_at", "updated_at", "data_atualizacao", "dt_atualizacao", "timestamp"],
    )

    return {
        "Facility_ID": facility_id,
        "FacilityName": facility_name,
        "ActivityType": activity,
        "Latitude": f"{lat:.6f}",
        "Longitude": f"{lon:.6f}",
        "SectorCode": sector_code,
        "CountryISO3": "BRA",
        "StateCode": state_code,
        "LicenseStart": license_start,
        "LicenseEnd": license_end,
        "LastVerifiedAt": last_verified or fetched_at,
    }


def _extract_from_wfs(
    client: httpx.Client,
    *,
    capabilities_url: str,
    timeout_seconds: float,
    retries: int,
    max_layers: int,
    max_features_per_layer: int,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    stats: dict[str, Any] = {
        "kind": "wfs",
        "capabilities_url": capabilities_url,
        "layers_scanned": 0,
        "layers_selected": 0,
        "features_seen": 0,
    }
    rows: list[dict[str, str]] = []

    capabilities_resp = _request_with_retry(
        client,
        url=capabilities_url,
        timeout_seconds=timeout_seconds,
        retries=retries,
    )
    capabilities_text = capabilities_resp.text
    type_names = _parse_wfs_typenames(capabilities_text)
    stats["layers_scanned"] = len(type_names)

    selected = [name for name in type_names if _is_industrial_name(name)][:max_layers]
    stats["layers_selected"] = len(selected)

    for type_name in selected:
        feature_resp = _request_with_retry(
            client,
            url="https://siscom.ibama.gov.br/geoserver/wfs",
            params={
                "service": "WFS",
                "version": "1.1.0",
                "request": "GetFeature",
                "typeName": type_name,
                "outputFormat": "application/json",
                "maxFeatures": str(max_features_per_layer),
            },
            timeout_seconds=timeout_seconds,
            retries=retries,
        )
        if feature_resp.status_code != 200:
            continue
        try:
            payload = feature_resp.json()
        except Exception:
            continue
        features = payload.get("features") or []
        stats["features_seen"] += len(features)

        for feature in features:
            attrs = feature.get("properties") or {}
            geom = feature.get("geometry") or {}
            lat, lon = _geometry_to_latlon(geom)
            row = _normalize_feature(attrs, lat, lon, fetched_at=_iso_now())
            if row is not None:
                rows.append(row)

    return rows, stats


def _extract_from_arcgis(
    client: httpx.Client,
    *,
    services_url: str,
    timeout_seconds: float,
    retries: int,
    max_layers: int,
    max_features_per_layer: int,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    stats: dict[str, Any] = {
        "kind": "arcgis",
        "services_url": services_url,
        "services_seen": 0,
        "layers_scanned": 0,
        "layers_selected": 0,
        "features_seen": 0,
    }
    rows: list[dict[str, str]] = []

    services_resp = _request_with_retry(
        client,
        url=services_url,
        timeout_seconds=timeout_seconds,
        retries=retries,
    )
    if services_resp.status_code != 200:
        return rows, stats
    try:
        services_payload = services_resp.json()
    except Exception:
        return rows, stats

    services = services_payload.get("services") or []
    stats["services_seen"] = len(services)

    selected_layers: list[str] = []
    for service in services:
        name = str(service.get("name") or "").strip()
        srv_type = str(service.get("type") or "").strip()
        if not name or not srv_type:
            continue
        if not _is_industrial_name(name):
            continue
        selected_layers.append(f"{name}/{srv_type}")

    for service_ref in selected_layers[:max_layers]:
        service_url = f"https://pamgia.ibama.gov.br/server/rest/services/{service_ref}"
        meta_resp = _request_with_retry(
            client,
            url=service_url,
            params={"f": "pjson"},
            timeout_seconds=timeout_seconds,
            retries=retries,
        )
        if meta_resp.status_code != 200:
            continue
        try:
            meta_payload = meta_resp.json()
        except Exception:
            continue

        layers = meta_payload.get("layers") or []
        stats["layers_scanned"] += len(layers)
        for layer in layers:
            layer_id = layer.get("id")
            layer_name = str(layer.get("name") or "")
            if layer_id is None or not _is_industrial_name(layer_name):
                continue
            stats["layers_selected"] += 1
            query_url = f"{service_url}/{layer_id}/query"
            query_resp = _request_with_retry(
                client,
                url=query_url,
                params={
                    "f": "json",
                    "where": "1=1",
                    "outFields": "*",
                    "returnGeometry": "true",
                    "resultRecordCount": str(max_features_per_layer),
                    "outSR": "4326",
                },
                timeout_seconds=timeout_seconds,
                retries=retries,
            )
            if query_resp.status_code != 200:
                continue
            try:
                query_payload = query_resp.json()
            except Exception:
                continue
            features = query_payload.get("features") or []
            stats["features_seen"] += len(features)
            for feature in features:
                attrs = feature.get("attributes") or {}
                geom = feature.get("geometry") or {}
                lat, lon = _geometry_to_latlon(geom)
                row = _normalize_feature(attrs, lat, lon, fetched_at=_iso_now())
                if row is not None:
                    rows.append(row)

    return rows, stats


def _required_columns() -> list[str]:
    return [
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
    ]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_required_columns())
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe/fetch BR public IBAMA industrial data")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--probe-out", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=45.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-layers", type=int, default=25)
    parser.add_argument("--max-features-per-layer", type=int, default=3000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    out_path = Path(args.out).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    probe_path = Path(args.probe_out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    probe_path.parent.mkdir(parents=True, exist_ok=True)

    extraction_rows: list[dict[str, str]] = []
    extraction_stats: list[dict[str, Any]] = []

    with httpx.Client(headers=DEFAULT_HEADERS, follow_redirects=True) as client:
        probes = _probe_endpoints(
            client,
            timeout_seconds=float(args.timeout_seconds),
            retries=max(1, int(args.retries)),
        )

        probe_payload = {
            "generated_at": _iso_now(),
            "candidate_endpoints": CANDIDATE_ENDPOINTS,
            "results": [probe.__dict__ for probe in probes],
        }
        probe_path.write_text(json.dumps(probe_payload, indent=2, ensure_ascii=False), encoding="utf-8")

        wfs_probe = next((p for p in probes if "geoserver" in p.url and _is_machine_accessible_wfs(p)), None)
        arc_probe = next((p for p in probes if "rest/services" in p.url and _is_machine_accessible_arcgis(p)), None)

        if wfs_probe is not None:
            rows, stats = _extract_from_wfs(
                client,
                capabilities_url=wfs_probe.url,
                timeout_seconds=float(args.timeout_seconds),
                retries=max(1, int(args.retries)),
                max_layers=int(args.max_layers),
                max_features_per_layer=int(args.max_features_per_layer),
            )
            extraction_rows.extend(rows)
            extraction_stats.append(stats)

        if arc_probe is not None:
            rows, stats = _extract_from_arcgis(
                client,
                services_url=arc_probe.url,
                timeout_seconds=float(args.timeout_seconds),
                retries=max(1, int(args.retries)),
                max_layers=int(args.max_layers),
                max_features_per_layer=int(args.max_features_per_layer),
            )
            extraction_rows.extend(rows)
            extraction_stats.append(stats)

    # De-duplicate by facility id + coordinates.
    deduped: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in extraction_rows:
        key = (row["Facility_ID"], row["Latitude"], row["Longitude"])
        deduped[key] = row
    rows_out = list(deduped.values())

    manifest = {
        "generated_at": _iso_now(),
        "source_profile": "br_ibama_sigel_hybrid",
        "source_uri": "https://www.gov.br/ibama/pt-br",
        "source_version": "ibama_sigel_2025",
        "window": {
            "start_date": args.start_date,
            "end_date": args.end_date,
        },
        "probe_output": str(probe_path),
        "output_csv": str(out_path),
        "rows_emitted": len(rows_out),
        "extraction_stats": extraction_stats,
        "blocker": None,
    }

    if not rows_out:
        manifest["blocker"] = (
            "STOP: All Brazil machine-accessible authoritative endpoints are blocked or returned "
            "no extractable industrial features. Keeping BRA strict no-go."
        )
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"manifest": str(manifest_path), "probe": str(probe_path), "rows_emitted": 0}))
        return 2

    _write_csv(out_path, rows_out)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(out_path),
                "manifest": str(manifest_path),
                "probe": str(probe_path),
                "rows_emitted": len(rows_out),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
