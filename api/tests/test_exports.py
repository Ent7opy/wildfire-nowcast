"""Tests for GeoPackage / CSV export formats and the sync over-limit guard."""

from __future__ import annotations

import os
import sqlite3
import struct
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from unittest.mock import MagicMock
from uuid import uuid4

from fastapi.testclient import TestClient

import api.routes.exports as exports_routes
from api.main import app
from api.routes.exports import MAX_SYNC_FEATURES

client = TestClient(app)

_BBOX_PARAMS = {
    "min_lon": 0,
    "min_lat": 0,
    "max_lon": 30,
    "max_lat": 30,
    "start_time": "2026-01-01T00:00:00Z",
    "end_time": "2026-01-02T00:00:00Z",
}

_SAMPLE_DETECTIONS = [
    {
        "id": 1,
        "lat": 10.0,
        "lon": 20.0,
        "acq_time": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "confidence": 100,
        "frp": 10.5,
        "sensor": "VIIRS",
        "source": "NRT",
    },
    {
        "id": 2,
        "lat": 11.5,
        "lon": 21.5,
        "acq_time": datetime(2026, 1, 1, 12, tzinfo=timezone.utc),
        "confidence": 80,
        "frp": 5.0,
        "sensor": "MODIS",
        "source": "NRT",
    },
]


def _fires_mock(detections=_SAMPLE_DETECTIONS, has_more=False):
    return MagicMock(
        return_value={
            "data": detections,
            "has_more": has_more,
            "next_cursor": detections[-1]["id"] if has_more and detections else None,
            "limit": MAX_SYNC_FEATURES,
        }
    )


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


def test_export_fires_csv_has_lat_lon_columns(monkeypatch):
    monkeypatch.setattr(exports_routes.fires_repo, "list_fire_detections_bbox_time", _fires_mock())
    resp = client.get("/fires/export", params={**_BBOX_PARAMS, "format": "csv"})
    assert resp.status_code == 200
    assert "text/csv" in resp.headers["content-type"]
    lines = resp.text.splitlines()
    header = lines[0].split(",")
    assert "lat" in header
    assert "lon" in header
    # Data rows should contain the expected coordinates
    assert any("10.0" in line and "20.0" in line for line in lines[1:])


def test_export_fires_csv_content_disposition(monkeypatch):
    monkeypatch.setattr(exports_routes.fires_repo, "list_fire_detections_bbox_time", _fires_mock())
    resp = client.get("/fires/export", params={**_BBOX_PARAMS, "format": "csv"})
    assert "attachment" in resp.headers["content-disposition"]
    assert resp.headers["content-disposition"].endswith(".csv")


# ---------------------------------------------------------------------------
# GeoPackage export – fires
# ---------------------------------------------------------------------------


def _is_valid_gpkg(data: bytes) -> bool:
    """Basic GeoPackage validation: check SQLite magic and GPKG application_id."""
    if len(data) < 100:
        return False
    # SQLite magic header
    if not data.startswith(b"SQLite format 3\x00"):
        return False
    # application_id at offset 68 (big-endian int32) must be 0x47504b47
    app_id = struct.unpack(">I", data[68:72])[0]
    return app_id == 0x47504B47


@contextmanager
def _gpkg_db(data: bytes):
    """Write GeoPackage bytes to a temp file and yield an open sqlite3 connection."""
    tmp = tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False)
    tmp.write(data)
    tmp.close()
    try:
        con = sqlite3.connect(tmp.name)
        try:
            yield con
        finally:
            con.close()
    finally:
        os.unlink(tmp.name)


def test_export_fires_gpkg_returns_valid_geopackage(monkeypatch):
    monkeypatch.setattr(exports_routes.fires_repo, "list_fire_detections_bbox_time", _fires_mock())
    resp = client.get("/fires/export", params={**_BBOX_PARAMS, "format": "gpkg"})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/geopackage+sqlite3"
    assert "attachment" in resp.headers["content-disposition"]
    assert resp.headers["content-disposition"].endswith(".gpkg")
    assert _is_valid_gpkg(resp.content)


def test_export_fires_gpkg_layer_has_expected_rows(monkeypatch):
    monkeypatch.setattr(exports_routes.fires_repo, "list_fire_detections_bbox_time", _fires_mock())
    resp = client.get("/fires/export", params={**_BBOX_PARAMS, "format": "gpkg"})
    assert resp.status_code == 200

    with _gpkg_db(resp.content) as con:
        rows = con.execute("SELECT * FROM fire_detections").fetchall()
    assert len(rows) == 2


def test_export_fires_gpkg_has_gpkg_contents_row(monkeypatch):
    monkeypatch.setattr(exports_routes.fires_repo, "list_fire_detections_bbox_time", _fires_mock())
    resp = client.get("/fires/export", params={**_BBOX_PARAMS, "format": "gpkg"})
    assert resp.status_code == 200

    with _gpkg_db(resp.content) as con:
        rows = con.execute("SELECT table_name, data_type FROM gpkg_contents").fetchall()
    assert any(r[0] == "fire_detections" and r[1] == "features" for r in rows)


# ---------------------------------------------------------------------------
# GeoPackage export – forecast contours
# ---------------------------------------------------------------------------


def test_export_forecast_contours_gpkg(monkeypatch):
    mock_contours = [
        {
            "horizon_hours": 24,
            "threshold": 0.5,
            "geom_geojson": '{"type":"MultiPolygon","coordinates":[]}',
        }
    ]
    monkeypatch.setattr(
        exports_routes.forecast_repo, "list_contours_for_run", MagicMock(return_value=mock_contours)
    )
    resp = client.get("/forecast/1/contours/export?format=gpkg")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/geopackage+sqlite3"
    assert _is_valid_gpkg(resp.content)


# ---------------------------------------------------------------------------
# GeoPackage export – AOI
# ---------------------------------------------------------------------------


def test_export_aoi_gpkg(monkeypatch):
    aoi_id = uuid4()
    mock_aoi = {
        "id": aoi_id,
        "name": "Test AOI",
        "description": "desc",
        "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 1], [1, 0], [0, 0]]]},
        "area_km2": 5.0,
        "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
    }
    monkeypatch.setattr(exports_routes.aois_repo, "get_aoi", MagicMock(return_value=mock_aoi))
    resp = client.get(f"/aois/{aoi_id}/export?format=gpkg")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/geopackage+sqlite3"
    assert _is_valid_gpkg(resp.content)


# ---------------------------------------------------------------------------
# Over-limit guard
# ---------------------------------------------------------------------------


def _error_msg(resp) -> str:
    """Extract error message from API error response (supports both detail and message keys)."""
    body = resp.json()
    return body.get("detail") or body.get("message") or str(body)


def test_export_fires_over_limit_returns_413(monkeypatch):
    """When the repo signals has_more=True, the endpoint must return 413."""
    monkeypatch.setattr(
        exports_routes.fires_repo,
        "list_fire_detections_bbox_time",
        _fires_mock(has_more=True),
    )
    resp = client.get("/fires/export", params={**_BBOX_PARAMS, "format": "csv", "limit": 10})
    assert resp.status_code == 413
    msg = _error_msg(resp)
    assert "10" in msg  # limit reflected in message
    assert MAX_SYNC_FEATURES == 10000
    assert str(MAX_SYNC_FEATURES) in msg
    assert "/exports" in msg  # hint to async endpoint


def test_export_fires_over_limit_message_contains_hint(monkeypatch):
    monkeypatch.setattr(
        exports_routes.fires_repo,
        "list_fire_detections_bbox_time",
        _fires_mock(has_more=True),
    )
    resp = client.get("/fires/export", params={**_BBOX_PARAMS, "format": "geojson", "limit": 5000})
    assert resp.status_code == 413
    msg = _error_msg(resp)
    assert "bounding box" in msg.lower() or "narrow" in msg.lower()


# ---------------------------------------------------------------------------
# Existing GeoJSON export unchanged
# ---------------------------------------------------------------------------


def test_export_fires_geojson_unchanged(monkeypatch):
    monkeypatch.setattr(exports_routes.fires_repo, "list_fire_detections_bbox_time", _fires_mock())
    resp = client.get("/fires/export", params={**_BBOX_PARAMS, "format": "geojson"})
    assert resp.status_code == 200
    fc = resp.json()
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) == 2
    feat = fc["features"][0]
    assert feat["type"] == "Feature"
    assert feat["geometry"]["type"] == "Point"
    # GeoJSON point: [lon, lat]
    assert feat["geometry"]["coordinates"] == [20.0, 10.0]


def test_export_fires_geojson_invalid_format_rejected(monkeypatch):
    resp = client.get("/fires/export", params={**_BBOX_PARAMS, "format": "shapefile"})
    assert resp.status_code == 422


def test_export_fires_csv_empty_result(monkeypatch):
    monkeypatch.setattr(
        exports_routes.fires_repo,
        "list_fire_detections_bbox_time",
        _fires_mock(detections=[]),
    )
    resp = client.get("/fires/export", params={**_BBOX_PARAMS, "format": "csv"})
    assert resp.status_code == 200
    assert "text/csv" in resp.headers["content-type"]
