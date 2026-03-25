"""Unit tests for spread forecast orphan file cleanup (scripts/db_cleanup.py).

These tests exercise find_orphan_forecast_files() — the pure filesystem logic —
without requiring a database connection.
"""

import sys
from pathlib import Path

# Make the scripts/ directory importable
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.db_cleanup import find_orphan_forecast_files


def _make_forecast_file(repo_root: Path, region: str, run_id: int, horizon: int) -> Path:
    """Create a dummy .tif file under data/forecasts/ and return its absolute path."""
    run_dir = repo_root / "data" / "forecasts" / region / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    tif = run_dir / f"spread_h{horizon:03d}_cog.tif"
    tif.write_bytes(b"")  # empty placeholder
    return tif


def _storage_path(repo_root: Path, tif: Path) -> str:
    """Return the repo-relative path as stored in spread_forecast_rasters.storage_path."""
    return str(tif.relative_to(repo_root))


class TestFindOrphanForecastFiles:
    def test_empty_forecasts_dir_returns_nothing(self, tmp_path):
        forecasts_dir = tmp_path / "data" / "forecasts"
        forecasts_dir.mkdir(parents=True)
        orphans = find_orphan_forecast_files(forecasts_dir, tmp_path, known_paths=set())
        assert orphans == []

    def test_nonexistent_forecasts_dir_returns_nothing(self, tmp_path):
        forecasts_dir = tmp_path / "data" / "forecasts"
        # Don't create the directory
        orphans = find_orphan_forecast_files(forecasts_dir, tmp_path, known_paths=set())
        assert orphans == []

    def test_all_files_registered_returns_no_orphans(self, tmp_path):
        tif1 = _make_forecast_file(tmp_path, "balkans", run_id=1, horizon=24)
        tif2 = _make_forecast_file(tmp_path, "balkans", run_id=1, horizon=48)

        known = {_storage_path(tmp_path, tif1), _storage_path(tmp_path, tif2)}
        orphans = find_orphan_forecast_files(tmp_path / "data" / "forecasts", tmp_path, known)
        assert orphans == []

    def test_unregistered_file_detected_as_orphan(self, tmp_path):
        tif_active = _make_forecast_file(tmp_path, "balkans", run_id=1, horizon=24)
        tif_orphan = _make_forecast_file(tmp_path, "balkans", run_id=2, horizon=24)

        known = {_storage_path(tmp_path, tif_active)}
        orphans = find_orphan_forecast_files(tmp_path / "data" / "forecasts", tmp_path, known)
        assert orphans == [tif_orphan]

    def test_multiple_orphans_across_regions(self, tmp_path):
        tif_active = _make_forecast_file(tmp_path, "balkans", run_id=10, horizon=24)
        tif_orphan1 = _make_forecast_file(tmp_path, "balkans", run_id=9, horizon=24)
        tif_orphan2 = _make_forecast_file(tmp_path, "location-based", run_id=5, horizon=48)

        known = {_storage_path(tmp_path, tif_active)}
        orphans = find_orphan_forecast_files(tmp_path / "data" / "forecasts", tmp_path, known)
        assert set(orphans) == {tif_orphan1, tif_orphan2}

    def test_idempotent_after_cleanup(self, tmp_path):
        """Re-running after cleanup finds no orphans (files were deleted)."""
        tif_active = _make_forecast_file(tmp_path, "balkans", run_id=1, horizon=24)
        tif_orphan = _make_forecast_file(tmp_path, "balkans", run_id=2, horizon=24)

        known = {_storage_path(tmp_path, tif_active)}
        forecasts_dir = tmp_path / "data" / "forecasts"

        # First pass — detects the orphan
        orphans = find_orphan_forecast_files(forecasts_dir, tmp_path, known)
        assert len(orphans) == 1

        # Simulate deletion
        tif_orphan.unlink()

        # Second pass — no orphans remain
        orphans = find_orphan_forecast_files(forecasts_dir, tmp_path, known)
        assert orphans == []

    def test_active_file_not_deleted_by_mistake(self, tmp_path):
        """Only files absent from known_paths are returned as orphans."""
        tif1 = _make_forecast_file(tmp_path, "balkans", run_id=1, horizon=24)
        tif2 = _make_forecast_file(tmp_path, "balkans", run_id=1, horizon=48)
        tif3 = _make_forecast_file(tmp_path, "balkans", run_id=1, horizon=72)

        # All three are known
        known = {
            _storage_path(tmp_path, tif1),
            _storage_path(tmp_path, tif2),
            _storage_path(tmp_path, tif3),
        }
        orphans = find_orphan_forecast_files(tmp_path / "data" / "forecasts", tmp_path, known)
        assert orphans == []

    def test_non_tif_files_ignored(self, tmp_path):
        """Non-.tif files (e.g. .json metadata) are not treated as orphans."""
        run_dir = tmp_path / "data" / "forecasts" / "balkans" / "run_1"
        run_dir.mkdir(parents=True)
        (run_dir / "metadata.json").write_text("{}")
        (run_dir / "spread_h024_cog.tif").write_bytes(b"")

        tif = run_dir / "spread_h024_cog.tif"
        known = {_storage_path(tmp_path, tif)}
        orphans = find_orphan_forecast_files(tmp_path / "data" / "forecasts", tmp_path, known)
        assert orphans == []
