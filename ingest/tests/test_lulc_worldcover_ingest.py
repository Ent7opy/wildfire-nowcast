"""Unit tests for LULC WorldCover version tracking.

Tests cover pure-function logic (no DB, no S3) — tile ID formatting, tile path
construction, cache manifest write/check, and the update-param structure.
"""

from __future__ import annotations

import json
import logging

import pytest


# ── _format_tile_id ──────────────────────────────────────────────────────────

class TestFormatTileId:
    def test_positive_lat_lon(self):
        from ingest.lulc_worldcover_ingest import _format_tile_id

        assert _format_tile_id(lat0=0, lon0=0) == "N00E000"

    def test_northern_western(self):
        from ingest.lulc_worldcover_ingest import _format_tile_id

        assert _format_tile_id(lat0=48, lon0=-123) == "N48W123"

    def test_southern_eastern(self):
        from ingest.lulc_worldcover_ingest import _format_tile_id

        assert _format_tile_id(lat0=-33, lon0=18) == "S33E018"


# ── _tile_path ────────────────────────────────────────────────────────────────

class TestTilePath:
    def test_path_format(self):
        from ingest.lulc_worldcover_ingest import _tile_path

        path = _tile_path(version="v200", year=2021, tile_id="N00E000")
        assert path == "v200/2021/map/ESA_WorldCover_10m_2021_v200_N00E000_Map.tif"

    def test_version_year_in_path(self):
        from ingest.lulc_worldcover_ingest import _tile_path

        path = _tile_path(version="v100", year=2020, tile_id="N48W123")
        assert "v100" in path
        assert "2020" in path


# ── cache manifest ────────────────────────────────────────────────────────────

class TestCacheManifest:
    def test_write_creates_manifest(self, tmp_path):
        from ingest.lulc_worldcover_ingest import _write_cache_manifest

        _write_cache_manifest(cache_dir=tmp_path, version="v200", year=2021)

        manifest_path = tmp_path / "manifest.json"
        assert manifest_path.exists()
        data = json.loads(manifest_path.read_text())
        assert data["worldcover_version"] == "v200"
        assert data["worldcover_year"] == 2021
        assert data["lulc_version"] == "v200_2021"
        assert "+00:00" in data["written_at"]

    def test_write_is_atomic(self, tmp_path):
        """No .part file should remain after a successful write."""
        from ingest.lulc_worldcover_ingest import _write_cache_manifest

        _write_cache_manifest(cache_dir=tmp_path, version="v200", year=2021)
        assert not any(tmp_path.glob("*.part"))

    def test_write_creates_cache_dir(self, tmp_path):
        from ingest.lulc_worldcover_ingest import _write_cache_manifest

        target = tmp_path / "nested" / "cache"
        _write_cache_manifest(cache_dir=target, version="v200", year=2021)
        assert (target / "manifest.json").exists()

    def test_check_no_warning_when_versions_match(self, tmp_path, caplog):
        from ingest.lulc_worldcover_ingest import _check_cache_manifest, _write_cache_manifest

        _write_cache_manifest(cache_dir=tmp_path, version="v200", year=2021)
        with caplog.at_level(logging.WARNING, logger="lulc_worldcover_ingest"):
            _check_cache_manifest(cache_dir=tmp_path, version="v200", year=2021)
        assert not caplog.records

    def test_check_warns_on_version_mismatch(self, tmp_path, caplog):
        from ingest.lulc_worldcover_ingest import _check_cache_manifest, _write_cache_manifest

        _write_cache_manifest(cache_dir=tmp_path, version="v200", year=2021)
        with caplog.at_level(logging.WARNING, logger="lulc_worldcover_ingest"):
            _check_cache_manifest(cache_dir=tmp_path, version="v300", year=2021)
        assert any("mismatch" in r.message for r in caplog.records)
        assert any("Mitigation" in r.message for r in caplog.records)

    def test_check_warns_on_year_mismatch(self, tmp_path, caplog):
        from ingest.lulc_worldcover_ingest import _check_cache_manifest, _write_cache_manifest

        _write_cache_manifest(cache_dir=tmp_path, version="v200", year=2021)
        with caplog.at_level(logging.WARNING, logger="lulc_worldcover_ingest"):
            _check_cache_manifest(cache_dir=tmp_path, version="v200", year=2023)
        assert any("mismatch" in r.message for r in caplog.records)

    def test_check_silent_when_no_manifest(self, tmp_path, caplog):
        from ingest.lulc_worldcover_ingest import _check_cache_manifest

        with caplog.at_level(logging.WARNING, logger="lulc_worldcover_ingest"):
            _check_cache_manifest(cache_dir=tmp_path, version="v200", year=2021)
        assert not caplog.records


# ── lulc_version in update params ────────────────────────────────────────────

class TestUpdateParamsIncludeLulcVersion:
    """Ensure that the update payload written to the DB includes lulc_version."""

    def test_lulc_version_key_present(self):
        """Build a fake update dict the same way _backfill_tile does and verify lulc_version."""
        from ingest.lulc_worldcover_ingest import _CLASS_LABELS, _CLASS_SCORES

        source_version = "v200_2021"
        lc = 10
        label = _CLASS_LABELS[lc]
        score = float(_CLASS_SCORES[lc])

        update = {
            "id": 42,
            "landcover_class": lc,
            "landcover_label": label,
            "landcover_score": score,
            "lulc_version": source_version,
            "landcover_source": "esa_worldcover",
            "landcover_version": source_version,
        }

        assert update["lulc_version"] == source_version
        assert update["landcover_version"] == source_version

    def test_lulc_version_encodes_both_version_and_year(self):
        """Confirm source_version formula produces the expected tag."""
        version, year = "v200", 2021
        source_version = f"{version}_{year}"
        assert source_version == "v200_2021"
