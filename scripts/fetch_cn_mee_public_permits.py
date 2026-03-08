#!/usr/bin/env python3
"""Fetch public MEE permit disclosure rows and emit a curated CN industrial CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from html import unescape
from pathlib import Path
from typing import Any

import httpx

LIST_URL = "https://permit.mee.gov.cn/perxxgkinfo/syssb/xkgg/xkgg!licenseInformation.action"
DETAIL_URL = (
    "https://permit.mee.gov.cn/perxxgkinfo/xkgkAction!xkgk.action"
    "?xkgk=getxxgkContent&dataid={dataid}"
)
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7",
}

TEMP_REPORT_KEY_RE = re.compile(r'name="tempReportKey"\s+value="([a-f0-9]{32})"', re.IGNORECASE)
TOTAL_PAGES_RE = re.compile(r"var\s+totalPages\s*=\s*(\d+)\s*;")
ROW_RE = re.compile(
    r"<tr>\s*(?P<body>.*?)xkgkAction!xkgk\.action\?xkgk=getxxgkContent&dataid="
    r"(?P<dataid>[a-f0-9]{32}).*?</tr>",
    re.IGNORECASE | re.DOTALL,
)
TITLE_RE = re.compile(r'<td[^>]*title="([^"]*)"', re.IGNORECASE)
LON_RE = re.compile(r'id="longitude"\s+value="([^"]+)"', re.IGNORECASE)
LAT_RE = re.compile(r'id="latitude"\s+value="([^"]+)"', re.IGNORECASE)


@dataclass
class LicenseRow:
    dataid: str
    province: str
    city: str
    permit_id: str
    facility_name: str
    industry_name: str
    valid_range: str
    issue_date: str


@dataclass
class DetailCoords:
    longitude: float
    latitude: float


def parse_temp_report_key(html: str) -> str | None:
    match = TEMP_REPORT_KEY_RE.search(html)
    return match.group(1) if match else None


def parse_total_pages(html: str) -> int | None:
    match = TOTAL_PAGES_RE.search(html)
    if not match:
        return None
    return int(match.group(1))


def parse_license_rows(html: str) -> list[LicenseRow]:
    rows: list[LicenseRow] = []
    for match in ROW_RE.finditer(html):
        body = match.group("body")
        dataid = match.group("dataid")
        titles = [unescape(v).strip() for v in TITLE_RE.findall(body)]
        if len(titles) < 7:
            continue
        rows.append(
            LicenseRow(
                dataid=dataid,
                province=titles[0],
                city=titles[1],
                permit_id=titles[2],
                facility_name=titles[3],
                industry_name=titles[4],
                valid_range=titles[5],
                issue_date=titles[6],
            )
        )
    return rows


def parse_detail_coords(html: str) -> DetailCoords | None:
    lon_match = LON_RE.search(html)
    lat_match = LAT_RE.search(html)
    if not lon_match or not lat_match:
        return None
    try:
        lon = float(lon_match.group(1))
        lat = float(lat_match.group(1))
    except ValueError:
        return None
    if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
        return None
    return DetailCoords(longitude=lon, latitude=lat)


def parse_iso_date(raw: str | None) -> date | None:
    token = str(raw or "").strip()
    if not token:
        return None
    try:
        return datetime.strptime(token, "%Y-%m-%d").date()
    except ValueError:
        return None


def parse_valid_range(raw: str | None) -> tuple[str | None, str | None]:
    token = str(raw or "").strip()
    if not token:
        return (None, None)
    if "至" in token:
        left, right = token.split("至", 1)
        left_date = parse_iso_date(left)
        right_date = parse_iso_date(right)
        return (
            left_date.isoformat() if left_date else None,
            right_date.isoformat() if right_date else None,
        )
    one = parse_iso_date(token)
    if one:
        return (one.isoformat(), None)
    return (None, None)


def _request_with_retry(
    client: httpx.Client,
    *,
    method: str,
    url: str,
    data: dict[str, Any] | None,
    timeout_seconds: float,
    retries: int,
) -> httpx.Response:
    attempt = 0
    while True:
        attempt += 1
        try:
            response = client.request(
                method,
                url,
                data=data,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            return response
        except Exception:
            if attempt >= retries:
                raise
            time.sleep(min(2 * attempt, 6))


def _save_html(path: Path, payload: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch CN public MEE permit disclosures")
    parser.add_argument("--start-date", required=True, help="Inclusive lower bound YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="Inclusive upper bound YYYY-MM-DD")
    parser.add_argument("--max-pages", type=int, default=200)
    parser.add_argument("--pause-ms", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=45.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--out", required=True, help="Output curated CSV path")
    parser.add_argument("--manifest", required=True, help="Output manifest JSON path")
    return parser.parse_args(argv)


def _required_columns() -> list[str]:
    return [
        "Permit_ID",
        "FacilityName",
        "IndustryName",
        "Latitude",
        "Longitude",
        "GBT_Code",
        "CountryISO3",
        "ProvinceCode",
        "PermitValidFrom",
        "PermitValidTo",
        "LastVerifiedAt",
    ]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    start_date = parse_iso_date(args.start_date)
    end_date = parse_iso_date(args.end_date)
    if start_date is None or end_date is None:
        raise SystemExit("--start-date and --end-date must be YYYY-MM-DD")
    if end_date < start_date:
        raise SystemExit("--end-date must be >= --start-date")

    out_path = Path(args.out).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    _ensure_parent(out_path)
    _ensure_parent(manifest_path)

    cn_root = out_path.parent
    license_raw_dir = cn_root / "raw" / "license_pages"
    detail_raw_dir = cn_root / "raw" / "detail_pages"
    license_raw_dir.mkdir(parents=True, exist_ok=True)
    detail_raw_dir.mkdir(parents=True, exist_ok=True)

    emitted_rows: list[dict[str, str]] = []
    endpoint_evidence: list[dict[str, Any]] = []
    total_pages_reported: int | None = None
    temp_report_key: str | None = None
    rows_seen = 0
    rows_in_window = 0

    with httpx.Client(headers=DEFAULT_HEADERS, follow_redirects=True) as client:
        for page_no in range(1, int(args.max_pages) + 1):
            method = "GET" if page_no == 1 else "POST"
            payload = None
            if page_no > 1:
                if not temp_report_key:
                    break
                payload = {
                    "page.pageNo": str(page_no),
                    "tempReportKey": temp_report_key,
                }

            response = _request_with_retry(
                client,
                method=method,
                url=LIST_URL,
                data=payload,
                timeout_seconds=float(args.timeout_seconds),
                retries=max(1, int(args.retries)),
            )
            html = response.text
            digest = _save_html(license_raw_dir / f"page_{page_no:04d}.html", html)
            endpoint_evidence.append(
                {
                    "kind": "license_page",
                    "page": page_no,
                    "url": str(response.request.url),
                    "method": method,
                    "status_code": int(response.status_code),
                    "body_sha256": digest,
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                }
            )

            parsed_key = parse_temp_report_key(html)
            if parsed_key:
                temp_report_key = parsed_key
            if total_pages_reported is None:
                total_pages_reported = parse_total_pages(html)

            page_rows = parse_license_rows(html)
            if not page_rows:
                break

            page_issue_dates: list[date] = []
            for row in page_rows:
                rows_seen += 1
                issue = parse_iso_date(row.issue_date)
                if issue is not None:
                    page_issue_dates.append(issue)
                if issue is None or issue < start_date or issue > end_date:
                    continue

                detail_url = DETAIL_URL.format(dataid=row.dataid)
                detail_response = _request_with_retry(
                    client,
                    method="GET",
                    url=detail_url,
                    data=None,
                    timeout_seconds=float(args.timeout_seconds),
                    retries=max(1, int(args.retries)),
                )
                detail_html = detail_response.text
                detail_digest = _save_html(detail_raw_dir / f"{row.dataid}.html", detail_html)
                endpoint_evidence.append(
                    {
                        "kind": "detail_page",
                        "dataid": row.dataid,
                        "url": str(detail_response.request.url),
                        "method": "GET",
                        "status_code": int(detail_response.status_code),
                        "body_sha256": detail_digest,
                        "fetched_at": datetime.now(timezone.utc).isoformat(),
                    }
                )

                coords = parse_detail_coords(detail_html)
                if coords is None:
                    continue

                valid_from, valid_to = parse_valid_range(row.valid_range)
                rows_in_window += 1
                emitted_rows.append(
                    {
                        "Permit_ID": row.permit_id,
                        "FacilityName": row.facility_name,
                        "IndustryName": row.industry_name,
                        "Latitude": f"{coords.latitude:.6f}",
                        "Longitude": f"{coords.longitude:.6f}",
                        "GBT_Code": "",
                        "CountryISO3": "CHN",
                        "ProvinceCode": row.province,
                        "PermitValidFrom": valid_from or "",
                        "PermitValidTo": valid_to or "",
                        "LastVerifiedAt": issue.isoformat(),
                    }
                )

            if args.pause_ms > 0:
                time.sleep(float(args.pause_ms) / 1000.0)

            # The listing is reverse chronological; stop once pages are clearly older than the window.
            if page_issue_dates and max(page_issue_dates) < start_date:
                break
            if total_pages_reported is not None and page_no >= total_pages_reported:
                break

    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_required_columns())
        writer.writeheader()
        for row in emitted_rows:
            writer.writerow(row)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_profile": "cn_mee_cied_hybrid",
        "source_uri": LIST_URL,
        "source_version": "mee_cied_2025",
        "window": {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        },
        "max_pages": int(args.max_pages),
        "total_pages_reported": total_pages_reported,
        "rows_seen": rows_seen,
        "rows_in_window": rows_in_window,
        "rows_emitted": len(emitted_rows),
        "output_csv": str(out_path),
        "raw_paths": {
            "license_pages": str(license_raw_dir),
            "detail_pages": str(detail_raw_dir),
        },
        "endpoint_evidence": endpoint_evidence,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        json.dumps(
            {
                "output": str(out_path),
                "manifest": str(manifest_path),
                "rows_emitted": len(emitted_rows),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
