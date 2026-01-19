#!/usr/bin/env python3
"""Benchmark patch_mode vs full mode for small AOI weather ingestion.

This script measures the time difference between normal and patch_mode ingestion
for a small bounding box to validate the T14 acceptance criterion.

Expected improvement based on T13 findings:
- 24h horizon vs 72h: reduces from ~25 GRIB files to ~5 files (~80% reduction)
- 6h steps vs 3h: halves temporal resolution (5 timesteps vs 9 for 24h)
- No precipitation: skips APCP variable processing
- Combined: ~40-60% time reduction for small AOIs

Usage:
    python scripts/benchmark_weather_patch_mode.py --bbox 20.0 40.0 20.1 40.1 --run-time 2025-01-19T00:00Z
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

# Add ingest to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingest.weather_ingest import ingest_weather_for_bbox


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark patch_mode ingestion")
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        default=[20.0, 40.0, 20.1, 40.1],
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        help="Bounding box (default: small 0.1° test box)",
    )
    parser.add_argument(
        "--run-time",
        type=str,
        default="2025-01-19T00:00Z",
        help="Forecast run time (ISO 8601, default: 2025-01-19T00:00Z)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/weather_benchmark",
        help="Output directory for test runs (default: data/weather_benchmark)",
    )
    return parser.parse_args()


def benchmark_mode(mode_name: str, bbox: tuple, forecast_time: datetime, output_dir: Path, patch_mode: bool) -> float:
    """Run ingestion with specified mode and return elapsed time."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {mode_name}")
    print(f"{'='*60}")
    
    mode_output = output_dir / mode_name
    mode_output.mkdir(parents=True, exist_ok=True)
    
    start = time.time()
    try:
        weather_run_id = ingest_weather_for_bbox(
            bbox=bbox,
            forecast_time=forecast_time,
            output_dir=mode_output,
            patch_mode=patch_mode,
        )
        elapsed = time.time() - start
        print(f"✓ Completed in {elapsed:.2f}s (weather_run_id={weather_run_id})")
        return elapsed
    except Exception as e:
        elapsed = time.time() - start
        print(f"✗ Failed after {elapsed:.2f}s: {e}")
        raise


def main():
    args = parse_args()
    bbox = tuple(args.bbox)
    forecast_time = datetime.fromisoformat(args.run_time.replace("Z", "+00:00"))
    output_dir = Path(args.output_dir)
    
    print(f"Benchmark Configuration:")
    print(f"  Bbox: {bbox}")
    print(f"  Forecast time: {forecast_time}")
    print(f"  Output: {output_dir}")
    
    # Run full mode
    full_time = benchmark_mode("full_mode", bbox, forecast_time, output_dir, patch_mode=False)
    
    # Run patch mode
    patch_time = benchmark_mode("patch_mode", bbox, forecast_time, output_dir, patch_mode=True)
    
    # Report results
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"Full mode:  {full_time:.2f}s")
    print(f"Patch mode: {patch_time:.2f}s")
    print(f"Speedup:    {full_time / patch_time:.2f}x")
    print(f"Time saved: {full_time - patch_time:.2f}s ({(1 - patch_time/full_time)*100:.1f}%)")
    print(f"\nTarget: <10s for 10km x 10km AOI")
    print(f"Status: {'✓ PASS' if patch_time < 10 else '✗ FAIL'}")


if __name__ == "__main__":
    main()
