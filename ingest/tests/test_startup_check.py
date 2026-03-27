"""Tests for ingest.startup_check — startup configuration validation."""
import logging

import pytest

from ingest.startup_check import (
    StartupError,
    run_ingest_startup_checks,
    validate_denoiser_model_path,
    validate_firms_map_key,
)


# ---------------------------------------------------------------------------
# validate_firms_map_key
# ---------------------------------------------------------------------------


class TestValidateFirmsMapKey:
    def test_key_present_passes(self, monkeypatch):
        monkeypatch.setenv("FIRMS_MAP_KEY", "my-api-key")
        validate_firms_map_key()

    def test_key_missing_raises(self, monkeypatch):
        monkeypatch.delenv("FIRMS_MAP_KEY", raising=False)
        with pytest.raises(StartupError, match="FIRMS_MAP_KEY"):
            validate_firms_map_key()

    def test_empty_key_raises(self, monkeypatch):
        monkeypatch.setenv("FIRMS_MAP_KEY", "")
        with pytest.raises(StartupError, match="FIRMS_MAP_KEY"):
            validate_firms_map_key()

    def test_whitespace_only_key_raises(self, monkeypatch):
        monkeypatch.setenv("FIRMS_MAP_KEY", "   ")
        with pytest.raises(StartupError, match="FIRMS_MAP_KEY"):
            validate_firms_map_key()

    def test_error_message_mentions_nasa_url(self, monkeypatch):
        monkeypatch.delenv("FIRMS_MAP_KEY", raising=False)
        with pytest.raises(StartupError, match="firms.modaps.eosdis.nasa.gov"):
            validate_firms_map_key()


# ---------------------------------------------------------------------------
# validate_denoiser_model_path
# ---------------------------------------------------------------------------


class TestValidateDenoiserModelPath:
    def test_denoiser_disabled_skips_all_checks(self, monkeypatch):
        monkeypatch.setenv("DENOISER_ENABLED", "false")
        monkeypatch.delenv("DENOISER_MODEL_RUN_DIR", raising=False)
        validate_denoiser_model_path()  # must not raise

    def test_denoiser_disabled_false_variant(self, monkeypatch):
        monkeypatch.setenv("DENOISER_ENABLED", "0")
        validate_denoiser_model_path()

    def test_denoiser_enabled_required_no_dir_raises(self, monkeypatch):
        monkeypatch.setenv("DENOISER_ENABLED", "true")
        monkeypatch.setenv("DENOISER_REQUIRED", "true")
        monkeypatch.setenv("DENOISER_MODEL_RUN_DIR", "")
        with pytest.raises(StartupError, match="DENOISER_MODEL_RUN_DIR"):
            validate_denoiser_model_path()

    def test_denoiser_enabled_required_dir_unset_raises(self, monkeypatch):
        monkeypatch.setenv("DENOISER_ENABLED", "true")
        monkeypatch.setenv("DENOISER_REQUIRED", "true")
        monkeypatch.delenv("DENOISER_MODEL_RUN_DIR", raising=False)
        with pytest.raises(StartupError, match="DENOISER_MODEL_RUN_DIR"):
            validate_denoiser_model_path()

    def test_denoiser_enabled_required_existing_path_passes(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DENOISER_ENABLED", "true")
        monkeypatch.setenv("DENOISER_REQUIRED", "true")
        monkeypatch.setenv("DENOISER_MODEL_RUN_DIR", str(tmp_path))
        validate_denoiser_model_path()

    def test_denoiser_enabled_required_missing_path_raises(self, monkeypatch):
        monkeypatch.setenv("DENOISER_ENABLED", "true")
        monkeypatch.setenv("DENOISER_REQUIRED", "true")
        monkeypatch.setenv("DENOISER_MODEL_RUN_DIR", "/nonexistent/run_001")
        with pytest.raises(StartupError, match="does not exist"):
            validate_denoiser_model_path()

    def test_denoiser_enabled_not_required_missing_dir_warns(self, monkeypatch, caplog):
        monkeypatch.setenv("DENOISER_ENABLED", "true")
        monkeypatch.setenv("DENOISER_REQUIRED", "false")
        monkeypatch.setenv("DENOISER_MODEL_RUN_DIR", "")
        with caplog.at_level(logging.WARNING, logger="ingest_orchestrator"):
            validate_denoiser_model_path()  # must not raise
        assert "DENOISER_MODEL_RUN_DIR" in caplog.text

    def test_denoiser_enabled_not_required_missing_path_warns(self, monkeypatch, caplog):
        monkeypatch.setenv("DENOISER_ENABLED", "true")
        monkeypatch.setenv("DENOISER_REQUIRED", "false")
        monkeypatch.setenv("DENOISER_MODEL_RUN_DIR", "/nonexistent/run_001")
        with caplog.at_level(logging.WARNING, logger="ingest_orchestrator"):
            validate_denoiser_model_path()  # must not raise
        assert "does not exist" in caplog.text

    def test_denoiser_enabled_true_variants(self, monkeypatch, tmp_path):
        """'1', 'yes', 'true' all count as enabled."""
        for val in ("1", "yes", "YES", "True"):
            monkeypatch.setenv("DENOISER_ENABLED", val)
            monkeypatch.setenv("DENOISER_REQUIRED", "true")
            monkeypatch.setenv("DENOISER_MODEL_RUN_DIR", str(tmp_path))
            validate_denoiser_model_path()


# ---------------------------------------------------------------------------
# run_ingest_startup_checks (integration of all checks)
# ---------------------------------------------------------------------------


class TestRunIngestStartupChecks:
    def test_valid_config_passes(self, monkeypatch):
        monkeypatch.setenv("FIRMS_MAP_KEY", "test-key-abc")
        monkeypatch.setenv("DENOISER_ENABLED", "false")
        run_ingest_startup_checks()

    def test_missing_firms_key_raises(self, monkeypatch):
        monkeypatch.setenv("FIRMS_MAP_KEY", "")
        monkeypatch.setenv("DENOISER_ENABLED", "false")
        with pytest.raises(StartupError, match="FIRMS_MAP_KEY"):
            run_ingest_startup_checks()

    def test_denoiser_required_missing_path_raises(self, monkeypatch):
        monkeypatch.setenv("FIRMS_MAP_KEY", "test-key-abc")
        monkeypatch.setenv("DENOISER_ENABLED", "true")
        monkeypatch.setenv("DENOISER_REQUIRED", "true")
        monkeypatch.setenv("DENOISER_MODEL_RUN_DIR", "/no/such/path")
        with pytest.raises(StartupError, match="does not exist"):
            run_ingest_startup_checks()

    def test_denoiser_not_required_missing_path_passes(self, monkeypatch, caplog):
        monkeypatch.setenv("FIRMS_MAP_KEY", "test-key-abc")
        monkeypatch.setenv("DENOISER_ENABLED", "true")
        monkeypatch.setenv("DENOISER_REQUIRED", "false")
        monkeypatch.setenv("DENOISER_MODEL_RUN_DIR", "/no/such/path")
        with caplog.at_level(logging.WARNING, logger="ingest_orchestrator"):
            run_ingest_startup_checks()  # must not raise
        assert "does not exist" in caplog.text
