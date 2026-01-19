#!/usr/bin/env python3
"""
Proof-of-concept for GFS partial downloads using HTTP Range requests.

Demonstrates two approaches:
1. NOMADS filter API (bbox-limited download, already in use)
2. Direct GRIB2 + .idx sidecar with HTTP Range requests (message-level extraction)

Usage:
    uv run --project ingest python scripts/gfs_partial_download_poc.py [--method filter|idx|both]
"""

import argparse
import logging
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlencode

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("gfs_poc")


# Test configuration: 10km x 10km patch in Greece (sample wildfire region)
TEST_BBOX = (23.7, 37.9, 23.8, 38.0)  # lon_min, lat_min, lon_max, lat_max
TEST_RUN_TIME = "20260117/00"  # YYYYMMDD/HH format (use current_date - 1 to 2 days for reliable GFS availability)
TEST_FORECAST_HOUR = 0  # f000 (analysis)

# Variables to fetch (wind + temp only for benchmark)
TEST_VARS = ["UGRD", "VGRD", "TMP"]
TEST_LEVELS = ["lev_10_m_above_ground", "lev_10_m_above_ground", "lev_2_m_above_ground"]


def benchmark_nomads_filter(
    bbox: tuple[float, float, float, float],
    run_time: str,
    forecast_hour: int,
    variables: list[str],
    levels: list[str],
) -> dict:
    """
    Benchmark NOMADS filter API with bbox parameters.

    This is the current approach used in ingest/weather_ingest.py.
    The filter API pre-subsets data on the server side.
    """
    base_url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
    dir_value = f"/gfs.{run_time.replace('/', '')[:8]}/{run_time.split('/')[1]}/atmos"
    file_value = f"gfs.t{run_time.split('/')[1]}z.pgrb2.0p25.f{forecast_hour:03d}"

    params = {
        "dir": dir_value,
        "file": file_value,
        "leftlon": bbox[0],
        "rightlon": bbox[2],
        "toplat": bbox[3],
        "bottomlat": bbox[1],
    }
    for var in variables:
        params[f"var_{var}"] = "on"
    for level in levels:
        params[level] = "on"

    url = f"{base_url}?{urlencode(params)}"

    LOGGER.info(f"Testing NOMADS filter API: {url[:120]}...")
    start = time.perf_counter()

    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp:
        try:
            with httpx.Client(timeout=30.0) as client:
                with client.stream("GET", url) as response:
                    response.raise_for_status()
                    bytes_downloaded = 0
                    for chunk in response.iter_bytes():
                        tmp.write(chunk)
                        bytes_downloaded += len(chunk)
            elapsed = time.perf_counter() - start

            LOGGER.info(f"NOMADS filter: {bytes_downloaded / 1024:.1f} KB in {elapsed:.2f}s")
            return {
                "method": "nomads_filter",
                "success": True,
                "elapsed_seconds": elapsed,
                "bytes_downloaded": bytes_downloaded,
                "url": url,
            }
        except Exception as e:
            elapsed = time.perf_counter() - start
            LOGGER.error(f"NOMADS filter failed: {e}")
            return {
                "method": "nomads_filter",
                "success": False,
                "elapsed_seconds": elapsed,
                "error": str(e),
            }
        finally:
            Path(tmp.name).unlink(missing_ok=True)


def parse_grib_idx(idx_content: str, variables: list[str]) -> list[tuple[str, int, int]]:
    """
    Parse GRIB2 .idx file to find byte ranges for specific variables.

    .idx format (example):
    1:0:d=2026011900:UGRD:10 m above ground:anl:
    2:123456:d=2026011900:VGRD:10 m above ground:anl:
    3:234567:d=2026011900:TMP:2 m above ground:anl:

    Returns list of (variable_description, start_byte, end_byte).
    """
    lines = idx_content.strip().split("\n")
    matches = []

    for i, line in enumerate(lines):
        parts = line.split(":")
        if len(parts) < 4:
            continue

        record_num = parts[0]
        start_byte = int(parts[1])
        variable = parts[3] if len(parts) > 3 else ""
        level = parts[4] if len(parts) > 4 else ""

        # Check if this variable matches any we're looking for
        for target_var in variables:
            if target_var in variable:
                # Calculate end byte (start of next record, or EOF)
                if i + 1 < len(lines):
                    next_parts = lines[i + 1].split(":")
                    end_byte = int(next_parts[1]) - 1 if len(next_parts) > 1 else None
                else:
                    end_byte = None  # Last record, fetch to EOF

                matches.append((f"{variable}:{level}", start_byte, end_byte))
                break

    return matches


def benchmark_idx_range_requests(
    bbox: tuple[float, float, float, float],
    run_time: str,
    forecast_hour: int,
    variables: list[str],
) -> dict:
    """
    Benchmark direct GRIB2 + .idx approach with HTTP Range requests.

    This approach:
    1. Downloads .idx sidecar (~10-50 KB)
    2. Parses to find byte offsets for target variables
    3. Uses HTTP Range headers to fetch only those GRIB messages

    Limitation: No server-side spatial subsetting. Full global grid per variable.
    """
    # Use AWS Open Data mirror (supports Range requests)
    base_url = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
    grib_path = f"gfs.{run_time.replace('/', '')[:8]}/{run_time.split('/')[1]}/atmos/gfs.t{run_time.split('/')[1]}z.pgrb2.0p25.f{forecast_hour:03d}"
    grib_url = f"{base_url}/{grib_path}"
    idx_url = f"{grib_url}.idx"

    LOGGER.info(f"Testing .idx + Range requests: {grib_url}")
    start = time.perf_counter()

    try:
        with httpx.Client(timeout=30.0) as client:
            # Step 1: Download .idx file
            idx_response = client.get(idx_url)
            idx_response.raise_for_status()
            idx_elapsed = time.perf_counter() - start
            LOGGER.info(f".idx file: {len(idx_response.content)} bytes in {idx_elapsed:.2f}s")

            # Step 2: Parse .idx to find variable byte ranges
            matches = parse_grib_idx(idx_response.text, variables)
            if not matches:
                raise ValueError(f"No matches found for {variables} in .idx")

            LOGGER.info(f"Found {len(matches)} GRIB messages to fetch")

            # Step 3: Fetch each message via Range request
            total_bytes = 0
            for var_desc, start_byte, end_byte in matches:
                if end_byte is None:
                    # Last message: fetch from start_byte to EOF
                    range_header = f"bytes={start_byte}-"
                else:
                    range_header = f"bytes={start_byte}-{end_byte}"

                headers = {"Range": range_header}
                msg_response = client.get(grib_url, headers=headers)
                msg_response.raise_for_status()
                msg_bytes = len(msg_response.content)
                total_bytes += msg_bytes
                LOGGER.info(f"  {var_desc}: {msg_bytes / 1024:.1f} KB")

            elapsed = time.perf_counter() - start
            LOGGER.info(f".idx + Range: {total_bytes / 1024:.1f} KB total in {elapsed:.2f}s")

            return {
                "method": "idx_range_requests",
                "success": True,
                "elapsed_seconds": elapsed,
                "bytes_downloaded": total_bytes,
                "idx_size_bytes": len(idx_response.content),
                "num_messages": len(matches),
                "note": "No spatial subsetting - full global grid per variable",
            }
    except Exception as e:
        elapsed = time.perf_counter() - start
        LOGGER.error(f".idx + Range failed: {e}")
        return {
            "method": "idx_range_requests",
            "success": False,
            "elapsed_seconds": elapsed,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="GFS partial download POC")
    parser.add_argument(
        "--method",
        choices=["filter", "idx", "both"],
        default="both",
        help="Which method to benchmark (default: both)",
    )
    args = parser.parse_args()

    LOGGER.info("=" * 60)
    LOGGER.info("GFS Partial Download POC")
    LOGGER.info(f"Test bbox: {TEST_BBOX} (10km x 10km patch)")
    LOGGER.info(f"Run time: {TEST_RUN_TIME}, forecast hour: f{TEST_FORECAST_HOUR:03d}")
    LOGGER.info(f"Variables: {TEST_VARS}")
    LOGGER.info("=" * 60)

    results = []

    if args.method in ("filter", "both"):
        LOGGER.info("\n[1/2] Benchmarking NOMADS filter API...")
        result = benchmark_nomads_filter(
            TEST_BBOX, TEST_RUN_TIME, TEST_FORECAST_HOUR, TEST_VARS, TEST_LEVELS
        )
        results.append(result)

    if args.method in ("idx", "both"):
        LOGGER.info("\n[2/2] Benchmarking .idx + Range requests...")
        result = benchmark_idx_range_requests(
            TEST_BBOX, TEST_RUN_TIME, TEST_FORECAST_HOUR, TEST_VARS
        )
        results.append(result)

    # Summary
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("RESULTS SUMMARY")
    LOGGER.info("=" * 60)
    for result in results:
        if result["success"]:
            method = result["method"]
            elapsed = result["elapsed_seconds"]
            kb = result["bytes_downloaded"] / 1024
            LOGGER.info(f"{method:20s}: {elapsed:5.2f}s, {kb:8.1f} KB")
        else:
            LOGGER.info(f"{result['method']:20s}: FAILED - {result.get('error', 'unknown')}")

    LOGGER.info("=" * 60)
    LOGGER.info("\nKEY FINDINGS:")
    LOGGER.info("- NOMADS filter: spatial subsetting on server, smaller downloads for small AOIs")
    LOGGER.info("- .idx + Range: variable subsetting only, fetches full global grid per var")
    LOGGER.info("- For JIT 10km patches, NOMADS filter is optimal (already in use)")
    LOGGER.info("- For full-resolution global coverage, .idx + Range may be useful")
    LOGGER.info("=" * 60)

    return 0 if all(r["success"] for r in results if r) else 1


if __name__ == "__main__":
    sys.exit(main())
