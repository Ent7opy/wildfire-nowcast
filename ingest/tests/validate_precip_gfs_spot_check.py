"""GFS APCP spot-check validation script.

Validates that ``_derive_per_step_precip`` produces physically correct output
against a *real* GFS 0.25° GRIB2 file for a fixed cycle.

Usage (requires FIRMS_MAP_KEY and network access to NOMADS):
    cd <repo-root>
    uv run python -m ingest.tests.validate_precip_gfs_spot_check

The script:
  1. Downloads two consecutive 3-hourly GFS GRIB files (f003, f006) for a
     known historical cycle using the NOMADS filter CGI.
  2. Loads raw APCP from each file via cfgrib.
  3. Applies ``_derive_per_step_precip`` and confirms the derivation is
     physically consistent:
       - No negative values.
       - f003 period == f003 raw  (first step of 0-6h bucket, no diff).
       - f006 period == f006 raw − f003 raw  (bucket-end step, diff applied).
       - Total (f003 + f006 period) == f006 raw  (conservation check).
  4. Prints a summary table of global stats (min/max/mean) so the human
     reviewer can cross-reference against a GRIB viewer or wgrib2.

This script does NOT use pytest fixtures or mock data.  It is intended to be
run manually as part of Task B1 acceptance validation.

Reference GFS cycle used: 2024-05-01 00Z (publicly available on NOMADS).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Allow running as a script from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import xarray as xr

from ingest.weather_ingest import _derive_per_step_precip

# ---------------------------------------------------------------------------
# Fixed validation cycle (historical, always available on NOMADS)
# ---------------------------------------------------------------------------
_VALIDATION_CYCLE = "20240501"
_VALIDATION_HH = "00"
_ATMOS_DIR = f"/gfs.{_VALIDATION_CYCLE}/{_VALIDATION_HH}/atmos"
_NOMADS_FILTER = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"

# Small bbox to minimise download size (~100 km × 100 km over the Pacific)
_BBOX = (-125.0, 35.0, -120.0, 40.0)


def _build_url(forecast_hour: int) -> str:
    from urllib.parse import urlencode

    params = {
        "dir": _ATMOS_DIR,
        "file": f"gfs.t{_VALIDATION_HH}z.pgrb2.0p25.f{forecast_hour:03d}",
        "leftlon": _BBOX[0],
        "rightlon": _BBOX[2],
        "toplat": _BBOX[3],
        "bottomlat": _BBOX[1],
        "var_APCP": "on",
        "lev_surface": "on",
    }
    return f"{_NOMADS_FILTER}?{urlencode(params)}"


def _download(url: str, dest: Path) -> None:
    import httpx

    print(f"  Downloading {url}")
    with httpx.Client(timeout=120) as client:
        with client.stream("GET", url) as resp:
            resp.raise_for_status()
            with dest.open("wb") as fh:
                for chunk in resp.iter_bytes():
                    fh.write(chunk)
    print(f"  Saved {dest.stat().st_size:,} bytes → {dest.name}")


def _load_apcp(path: Path) -> xr.DataArray:
    """Open APCP from a single GRIB2 file via cfgrib."""
    ds = xr.open_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {"shortName": "tp"},
            "indexpath": "",
        },
    )
    ds.load()
    # cfgrib may name the variable "tp" or "unknown"; take the first data var.
    var_name = next(iter(ds.data_vars))
    da = ds[var_name].copy()
    ds.close()
    return da


def _stats(da: xr.DataArray) -> dict:
    v = da.values.astype(float)
    return {
        "min": float(np.nanmin(v)),
        "max": float(np.nanmax(v)),
        "mean": float(np.nanmean(v)),
    }


def main() -> None:
    print("=" * 70)
    print("GFS APCP spot-check validation")
    print(f"  Cycle : {_VALIDATION_CYCLE} {_VALIDATION_HH}Z")
    print(f"  Bbox  : {_BBOX}")
    print("  Steps : f003 (0-3h period) + f006 (0-6h running total)")
    print("=" * 70)

    with tempfile.TemporaryDirectory(prefix="gfs_spot_") as tmpdir:
        f003_path = Path(tmpdir) / "f003.grib2"
        f006_path = Path(tmpdir) / "f006.grib2"

        print("\n[1] Downloading GRIB files …")
        _download(_build_url(3), f003_path)
        _download(_build_url(6), f006_path)

        print("\n[2] Loading raw APCP …")
        raw_f003 = _load_apcp(f003_path)
        raw_f006 = _load_apcp(f006_path)

        print(f"  f003 raw  stats: {_stats(raw_f003)}")
        print(f"  f006 raw  stats: {_stats(raw_f006)}")

        print("\n[3] Building combined DataArray with lead_time_hours …")
        # Stack into (time=2, lat, lon) with lead_time_hours coordinate.
        combined = xr.concat(
            [
                raw_f003.expand_dims("time").assign_coords(time=[0]),
                raw_f006.expand_dims("time").assign_coords(time=[1]),
            ],
            dim="time",
        ).assign_coords(lead_time_hours=("time", [3, 6]))

        print("\n[4] Applying _derive_per_step_precip …")
        derived = _derive_per_step_precip(combined)

        period_f003 = derived.isel(time=0).values.astype(float)
        period_f006 = derived.isel(time=1).values.astype(float)
        raw_f003_vals = raw_f003.values.astype(float)
        raw_f006_vals = raw_f006.values.astype(float)

        print(f"  f003 period stats (should == f003 raw): {_stats(derived.isel(time=0))}")
        print(f"  f006 period stats (should == f006_raw - f003_raw): {_stats(derived.isel(time=1))}")

        # ---------------------------------------------------------------
        # Assertion 1: No negative values
        # ---------------------------------------------------------------
        assert (period_f003 >= 0).all(), "FAIL: negative values in f003 period"
        assert (period_f006 >= 0).all(), "FAIL: negative values in f006 period"
        print("\n  [PASS] No negative precipitation values")

        # ---------------------------------------------------------------
        # Assertion 2: f003 period == f003 raw (first step, no diff)
        # ---------------------------------------------------------------
        # lt=3 → lt%6 != 0 → no differencing → period must equal raw.
        np.testing.assert_allclose(
            period_f003, raw_f003_vals,
            rtol=1e-6,
            err_msg="FAIL: f003 period should equal raw APCP (no differencing)",
        )
        print("  [PASS] f003 period == f003 raw (no differencing at non-boundary step)")

        # ---------------------------------------------------------------
        # Assertion 3: f006 period == f006_raw − f003_raw  (bucket-end diff)
        # ---------------------------------------------------------------
        expected_f006_period = np.clip(raw_f006_vals - raw_f003_vals, 0.0, None)
        np.testing.assert_allclose(
            period_f006, expected_f006_period,
            rtol=1e-6,
            err_msg="FAIL: f006 period should equal f006_raw - f003_raw",
        )
        print("  [PASS] f006 period == f006_raw − f003_raw (bucket-end differencing)")

        # ---------------------------------------------------------------
        # Assertion 4: Conservation — sum of periods == f006 raw total
        # ---------------------------------------------------------------
        total_derived = period_f003 + period_f006
        np.testing.assert_allclose(
            total_derived, raw_f006_vals,
            rtol=1e-6,
            err_msg="FAIL: f003_period + f006_period should equal f006_raw (conservation)",
        )
        print("  [PASS] f003_period + f006_period == f006_raw (mass conservation)")

        # ---------------------------------------------------------------
        # Assertion 5: Metadata attributes
        # ---------------------------------------------------------------
        assert derived.attrs.get("step_type") == "per_step", "FAIL: step_type attr"
        assert derived.attrs.get("source_step_type") == "accum", "FAIL: source_step_type attr"
        assert derived.attrs.get("units") == "kg m**-2", "FAIL: units attr"
        print("  [PASS] Metadata attributes correct")

    print("\n" + "=" * 70)
    print("All spot-check assertions PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
