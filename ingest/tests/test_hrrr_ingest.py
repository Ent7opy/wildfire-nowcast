"""Unit tests for ingest/hrrr_ingest.py (no network calls)."""

from __future__ import annotations

from datetime import datetime, timezone

from ingest.hrrr_ingest import (
    build_hrrr_urls,
    is_conus_bbox,
    parse_hrrr_idx,
    snap_to_hrrr_cycle,
    HRRR_VARIABLE_IDX_KEYS,
    HRRR_MODEL_NAME,
)


# ---------------------------------------------------------------------------
# snap_to_hrrr_cycle
# ---------------------------------------------------------------------------

def test_snap_to_hrrr_cycle_already_snapped():
    dt = datetime(2024, 7, 15, 12, 0, 0, tzinfo=timezone.utc)
    assert snap_to_hrrr_cycle(dt) == dt


def test_snap_to_hrrr_cycle_rounds_down():
    dt = datetime(2024, 7, 15, 12, 45, 30, tzinfo=timezone.utc)
    snapped = snap_to_hrrr_cycle(dt)
    assert snapped == datetime(2024, 7, 15, 12, 0, 0, tzinfo=timezone.utc)


def test_snap_to_hrrr_cycle_naive_input():
    """Naive datetimes are treated as UTC."""
    dt = datetime(2024, 7, 15, 18, 30, 0)
    snapped = snap_to_hrrr_cycle(dt)
    assert snapped.hour == 18
    assert snapped.minute == 0


# ---------------------------------------------------------------------------
# is_conus_bbox
# ---------------------------------------------------------------------------

def test_is_conus_bbox_inside():
    assert is_conus_bbox((-120.0, 35.0, -100.0, 45.0)) is True


def test_is_conus_bbox_outside_lon():
    assert is_conus_bbox((-130.0, 35.0, -60.0, 45.0)) is False  # exactly on boundary = False


def test_is_conus_bbox_outside_lat():
    assert is_conus_bbox((-115.0, 10.0, -100.0, 45.0)) is False


def test_is_conus_bbox_europe():
    assert is_conus_bbox((10.0, 40.0, 30.0, 55.0)) is False


# ---------------------------------------------------------------------------
# build_hrrr_urls
# ---------------------------------------------------------------------------

def test_build_hrrr_urls_structure():
    run_time = datetime(2024, 8, 1, 12, 0, 0, tzinfo=timezone.utc)
    grib_url, idx_url = build_hrrr_urls(run_time, forecast_hour=6)

    assert "noaa-hrrr-bdp-pds.s3.amazonaws.com" in grib_url
    assert "hrrr.20240801" in grib_url
    assert "t12z" in grib_url
    assert "wrfsubhf06.grib2" in grib_url
    assert idx_url == grib_url + ".idx"


def test_build_hrrr_urls_forecast_hour_zero():
    run_time = datetime(2024, 8, 1, 0, 0, 0, tzinfo=timezone.utc)
    grib_url, _ = build_hrrr_urls(run_time, forecast_hour=0)
    assert "wrfsubhf00.grib2" in grib_url


def test_build_hrrr_urls_max_forecast_hour():
    run_time = datetime(2024, 8, 1, 6, 0, 0, tzinfo=timezone.utc)
    grib_url, _ = build_hrrr_urls(run_time, forecast_hour=18)
    assert "wrfsubhf18.grib2" in grib_url


# ---------------------------------------------------------------------------
# parse_hrrr_idx
# ---------------------------------------------------------------------------

# Realistic subset of a wrfsubhf .idx file
SAMPLE_IDX = """\
1:0:d=2024080100:REFC:entire atmosphere:anl:
2:150000:d=2024080100:UGRD:10 m above ground:anl:
3:280000:d=2024080100:VGRD:10 m above ground:anl:
4:410000:d=2024080100:TMP:2 m above ground:anl:
5:540000:d=2024080100:RH:2 m above ground:anl:
6:670000:d=2024080100:APCP:surface:0-1 hour acc fcst:
7:800000:d=2024080100:WIND:10 m above ground:anl:
"""


def test_parse_hrrr_idx_finds_all_core_variables():
    result = parse_hrrr_idx(SAMPLE_IDX, HRRR_VARIABLE_IDX_KEYS)
    found = {r["canonical"] for r in result}
    assert found == {"u10", "v10", "t2m", "rh2m"}


def test_parse_hrrr_idx_byte_ranges():
    result = parse_hrrr_idx(SAMPLE_IDX, HRRR_VARIABLE_IDX_KEYS)
    by_name = {r["canonical"]: r for r in result}

    assert by_name["u10"]["start_byte"] == 150000
    assert by_name["u10"]["end_byte"] == 279999   # next message starts at 280000
    assert by_name["v10"]["start_byte"] == 280000
    assert by_name["rh2m"]["end_byte"] == 669999  # next is APCP at 670000


def test_parse_hrrr_idx_eof_marker():
    """Last variable in file should have end_byte = -1 (EOF)."""
    idx_text = "1:0:d=2024080100:UGRD:10 m above ground:anl:\n"
    result = parse_hrrr_idx(idx_text, {"u10": "UGRD:10 m above ground"})
    assert result[0]["end_byte"] == -1


def test_parse_hrrr_idx_missing_variable():
    idx_text = "1:0:d=2024080100:REFC:entire atmosphere:anl:\n"
    result = parse_hrrr_idx(idx_text, {"u10": "UGRD:10 m above ground"})
    assert result == []


def test_parse_hrrr_idx_empty_input():
    result = parse_hrrr_idx("", HRRR_VARIABLE_IDX_KEYS)
    assert result == []


def test_parse_hrrr_idx_precip():
    from ingest.hrrr_ingest import HRRR_PRECIP_IDX_KEY, HRRR_VARIABLE_IDX_PRECIP
    result = parse_hrrr_idx(
        SAMPLE_IDX,
        {HRRR_VARIABLE_IDX_PRECIP: HRRR_PRECIP_IDX_KEY},
    )
    assert len(result) == 1
    assert result[0]["canonical"] == HRRR_VARIABLE_IDX_PRECIP


# ---------------------------------------------------------------------------
# Constants / model name
# ---------------------------------------------------------------------------

def test_hrrr_model_name():
    assert HRRR_MODEL_NAME == "hrrr_3km"


def test_hrrr_variable_keys_coverage():
    assert "u10" in HRRR_VARIABLE_IDX_KEYS
    assert "v10" in HRRR_VARIABLE_IDX_KEYS
    assert "t2m" in HRRR_VARIABLE_IDX_KEYS
    assert "rh2m" in HRRR_VARIABLE_IDX_KEYS
