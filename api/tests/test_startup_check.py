"""Tests for api.startup_check — startup configuration validation."""
import json
import logging
import types

import pytest

from api.startup_check import (
    StartupError,
    run_api_startup_checks,
    validate_database_url,
    validate_spread_model_artifact_paths,
    warn_optional_config,
)


# ---------------------------------------------------------------------------
# validate_database_url
# ---------------------------------------------------------------------------


class TestValidateDatabaseUrl:
    def test_valid_postgresql_url(self):
        validate_database_url("postgresql://user:pass@localhost:5432/mydb")

    def test_valid_postgresql_asyncpg(self):
        validate_database_url("postgresql+asyncpg://user:pass@localhost:5432/mydb")

    def test_valid_postgresql_psycopg2(self):
        validate_database_url("postgresql+psycopg2://user:pass@localhost:5432/mydb")

    def test_empty_string_raises(self):
        with pytest.raises(StartupError, match="DATABASE_URL is not set"):
            validate_database_url("")

    def test_whitespace_only_raises(self):
        with pytest.raises(StartupError, match="DATABASE_URL is not set"):
            validate_database_url("   ")

    def test_sqlite_scheme_raises(self):
        with pytest.raises(StartupError, match="unsupported scheme 'sqlite'"):
            validate_database_url("sqlite:///test.db")

    def test_mysql_scheme_raises(self):
        with pytest.raises(StartupError, match="unsupported scheme 'mysql'"):
            validate_database_url("mysql://user:pass@localhost:3306/db")

    def test_bare_string_raises(self):
        with pytest.raises(StartupError, match="unsupported scheme"):
            validate_database_url("not-a-url")

    def test_missing_hostname_raises(self):
        with pytest.raises(StartupError, match="missing a hostname"):
            validate_database_url("postgresql:///mydb")

    def test_missing_dbname_raises(self):
        with pytest.raises(StartupError, match="missing a database name"):
            validate_database_url("postgresql://user:pass@localhost/")

    def test_default_built_url_passes(self):
        """The URL built from POSTGRES_* defaults is always structurally valid."""
        validate_database_url("postgresql://wildfire:wildfire@localhost:5432/wildfire")


# ---------------------------------------------------------------------------
# validate_spread_model_artifact_paths
# ---------------------------------------------------------------------------


class TestValidateSpreadModelArtifactPaths:
    def test_no_env_var_passes(self, monkeypatch):
        monkeypatch.delenv("SPREAD_MODEL_CATALOG_JSON", raising=False)
        validate_spread_model_artifact_paths()  # must not raise

    def test_empty_env_var_passes(self, monkeypatch):
        monkeypatch.setenv("SPREAD_MODEL_CATALOG_JSON", "")
        validate_spread_model_artifact_paths()

    def test_invalid_json_passes_silently(self, monkeypatch):
        monkeypatch.setenv("SPREAD_MODEL_CATALOG_JSON", "not-valid-json{{")
        validate_spread_model_artifact_paths()  # deferred to catalog load

    def test_catalog_without_artifact_paths_passes(self, monkeypatch):
        catalog = {"v0_default": {"model_name": "HeuristicSpreadModelV0", "model_params": {}}}
        monkeypatch.setenv("SPREAD_MODEL_CATALOG_JSON", json.dumps(catalog))
        validate_spread_model_artifact_paths()

    def test_existing_model_run_dir_passes(self, monkeypatch, tmp_path):
        model_dir = tmp_path / "run_001"
        model_dir.mkdir()
        catalog = {
            "v1": {
                "model_name": "LearnedSpreadModelV1",
                "model_params": {"model_run_dir": str(model_dir)},
            }
        }
        monkeypatch.setenv("SPREAD_MODEL_CATALOG_JSON", json.dumps(catalog))
        validate_spread_model_artifact_paths()

    def test_missing_model_run_dir_raises(self, monkeypatch):
        catalog = {
            "v1": {
                "model_name": "LearnedSpreadModelV1",
                "model_params": {"model_run_dir": "/nonexistent/path/run_001"},
            }
        }
        monkeypatch.setenv("SPREAD_MODEL_CATALOG_JSON", json.dumps(catalog))
        with pytest.raises(StartupError, match="does not exist"):
            validate_spread_model_artifact_paths()

    def test_missing_calibrator_run_dir_raises(self, monkeypatch, tmp_path):
        model_dir = tmp_path / "run_001"
        model_dir.mkdir()
        catalog = {
            "v1": {
                "model_name": "LearnedSpreadModelV1",
                "model_params": {
                    "model_run_dir": str(model_dir),
                    "calibrator_run_dir": "/nonexistent/calibrator",
                },
            }
        }
        monkeypatch.setenv("SPREAD_MODEL_CATALOG_JSON", json.dumps(catalog))
        with pytest.raises(StartupError, match="does not exist"):
            validate_spread_model_artifact_paths()

    def test_error_message_includes_model_id(self, monkeypatch):
        catalog = {
            "my_special_model": {
                "model_name": "LearnedSpreadModelV1",
                "model_params": {"model_run_dir": "/does/not/exist"},
            }
        }
        monkeypatch.setenv("SPREAD_MODEL_CATALOG_JSON", json.dumps(catalog))
        with pytest.raises(StartupError, match="my_special_model"):
            validate_spread_model_artifact_paths()


# ---------------------------------------------------------------------------
# warn_optional_config
# ---------------------------------------------------------------------------


class TestWarnOptionalConfig:
    def test_key_set_no_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="api.startup_check"):
            warn_optional_config("my-real-gemini-key")
        assert "GEMINI_API_KEY" not in caplog.text

    def test_empty_key_logs_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="api.startup_check"):
            warn_optional_config("")
        assert "GEMINI_API_KEY" in caplog.text

    def test_whitespace_key_logs_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="api.startup_check"):
            warn_optional_config("   ")
        assert "GEMINI_API_KEY" in caplog.text


# ---------------------------------------------------------------------------
# run_api_startup_checks (integration of all checks)
# ---------------------------------------------------------------------------


def _make_settings(database_url: str, gemini_api_key: str = "key") -> object:
    return types.SimpleNamespace(database_url=database_url, gemini_api_key=gemini_api_key)


class TestRunApiStartupChecks:
    def test_valid_config_passes(self, monkeypatch):
        monkeypatch.delenv("SPREAD_MODEL_CATALOG_JSON", raising=False)
        run_api_startup_checks(_make_settings("postgresql://u:p@localhost:5432/db"))

    def test_bad_database_url_raises(self, monkeypatch):
        monkeypatch.delenv("SPREAD_MODEL_CATALOG_JSON", raising=False)
        with pytest.raises(StartupError, match="unsupported scheme"):
            run_api_startup_checks(_make_settings("sqlite:///test.db"))

    def test_missing_gemini_key_warns_but_does_not_raise(self, monkeypatch, caplog):
        monkeypatch.delenv("SPREAD_MODEL_CATALOG_JSON", raising=False)
        with caplog.at_level(logging.WARNING, logger="api.startup_check"):
            run_api_startup_checks(_make_settings("postgresql://u:p@localhost:5432/db", gemini_api_key=""))
        assert "GEMINI_API_KEY" in caplog.text

    def test_missing_spread_artifact_path_raises(self, monkeypatch):
        catalog = {
            "v1": {
                "model_name": "LearnedSpreadModelV1",
                "model_params": {"model_run_dir": "/no/such/path"},
            }
        }
        monkeypatch.setenv("SPREAD_MODEL_CATALOG_JSON", json.dumps(catalog))
        with pytest.raises(StartupError, match="does not exist"):
            run_api_startup_checks(_make_settings("postgresql://u:p@localhost:5432/db"))
