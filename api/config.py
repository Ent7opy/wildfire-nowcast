from __future__ import annotations

import os
import subprocess
from importlib import metadata
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[1]


def _get_project_version() -> str:
    try:
        return metadata.version("wildfire-nowcast-api")
    except metadata.PackageNotFoundError:
        return "0.1.0"


def _get_git_commit() -> str:
    if (value := os.getenv("GIT_COMMIT")):
        return value

    repo_root = Path(__file__).resolve().parent.parent
    if (repo_root / ".git").exists():
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            pass

    return "unknown"


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(case_sensitive=False)

    app_name: str = "Wildfire Nowcast API"
    version: str = Field(default_factory=_get_project_version)
    environment: str = Field(default="dev", validation_alias="APP_ENV")
    git_commit: str = Field(default_factory=_get_git_commit, validation_alias="GIT_COMMIT")

    # Database settings. Accept POSTGRES_* (local/docker) or PG* / DATABASE_URL (e.g. Railway PostGIS).
    database_url_override: str | None = Field(
        default=None,
        validation_alias=AliasChoices("DATABASE_URL", "DATABASE_PRIVATE_URL"),
    )
    postgres_host: str = Field(
        default="localhost",
        validation_alias=AliasChoices("POSTGRES_HOST", "PGHOST"),
    )
    postgres_port: int = Field(
        default=5432,
        validation_alias=AliasChoices("POSTGRES_PORT", "PGPORT"),
    )
    postgres_user: str = Field(
        default="wildfire",
        validation_alias=AliasChoices("POSTGRES_USER", "PGUSER"),
    )
    postgres_password: str = Field(
        default="wildfire",
        validation_alias=AliasChoices("POSTGRES_PASSWORD", "PGPASSWORD"),
    )
    postgres_db: str = Field(
        default="wildfire",
        validation_alias=AliasChoices("POSTGRES_DB", "PGDATABASE"),
    )

    @property
    def database_url(self) -> str:
        """Connection URL: DATABASE_URL/DATABASE_PRIVATE_URL if set, else built from POSTGRES_* / PG*.

        Railway's PostGIS plugin emits postgres:// but SQLAlchemy 1.4+ requires postgresql://.
        """
        if self.database_url_override and self.database_url_override.strip():
            url = self.database_url_override.strip()
            # SQLAlchemy 1.4+ dropped the legacy 'postgres' dialect alias.
            if url.startswith("postgres://"):
                url = url.replace("postgres://", "postgresql://", 1)
            return url
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # TiTiler settings
    titiler_public_base_url: str = Field(
        default="http://localhost:8080", validation_alias="TITILER_PUBLIC_BASE_URL"
    )
    # Vector Tile Server settings (MVP default: same port mapping logic as TiTiler, but different port)
    vector_tiles_public_base_url: str = Field(
        default="http://localhost:7800", validation_alias="VECTOR_TILES_PUBLIC_BASE_URL"
    )
    vector_tiles_internal_base_url: str = Field(
        default="http://tiles:7800", validation_alias="VECTOR_TILES_INTERNAL_BASE_URL"
    )

    # Mapping for DB paths -> TiTiler container paths.
    # e.g., "data/forecasts/run_1/spread_h024_cog.tif" -> "/data/forecasts/run_1/spread_h024_cog.tif"
    # TiTiler then accesses it via filesystem.
    data_dir_local_prefix: str = Field(default="data/", validation_alias="DATA_DIR_LOCAL_PREFIX")
    data_dir_titiler_mount: str = Field(default="/data/", validation_alias="DATA_DIR_TITILER_MOUNT")

    # CORS settings (comma-separated list of allowed origins)
    cors_allow_origins: str = Field(
        default=(
            "http://localhost:8501,"
            "http://127.0.0.1:8501,"
            "http://localhost:5173,"
            "http://127.0.0.1:5173,"
            "http://localhost:3000,"
            "http://127.0.0.1:3000"
        ),
        validation_alias="CORS_ALLOW_ORIGINS",
    )

    # Export settings
    exports_dir: Path = Field(
        default=REPO_ROOT / "data" / "exports",
        validation_alias="EXPORTS_DIR",
    )

    forecast_result_cache_ttl_minutes: int = Field(
        default=60, validation_alias="FORECAST_RESULT_CACHE_TTL_MINUTES"
    )
    forecast_fail_closed_on_stale: bool = Field(
        default=True, validation_alias="FORECAST_FAIL_CLOSED_ON_STALE"
    )

    # Data freshness/staleness policy (minutes)
    data_stale_firms_minutes: int = Field(
        default=180, validation_alias="DATA_STALE_FIRMS_MINUTES"
    )
    data_stale_weather_minutes: int = Field(
        default=360, validation_alias="DATA_STALE_WEATHER_MINUTES"
    )
    data_stale_terrain_minutes: int = Field(
        default=10080, validation_alias="DATA_STALE_TERRAIN_MINUTES"
    )
    data_stale_perimeters_minutes: int = Field(
        default=4320, validation_alias="DATA_STALE_PERIMETERS_MINUTES"
    )
    data_stale_lfmc_minutes: int = Field(
        default=480, validation_alias="DATA_STALE_LFMC_MINUTES"
    )
    data_stale_lulc_minutes: int = Field(
        default=10080, validation_alias="DATA_STALE_LULC_MINUTES"
    )
    data_status_critical_sources: str = Field(
        default="firms,weather",
        validation_alias="DATA_STATUS_CRITICAL_SOURCES",
    )

    # Reverse geocoding settings (open Nominatim by default)
    geocoding_enabled: bool = Field(
        default=True, validation_alias="GEOCODING_ENABLED"
    )
    geocoding_provider: str = Field(
        default="nominatim", validation_alias="GEOCODING_PROVIDER"
    )
    geocoding_nominatim_base_url: str = Field(
        default="https://nominatim.openstreetmap.org",
        validation_alias="GEOCODING_NOMINATIM_BASE_URL",
    )
    geocoding_user_agent: str = Field(
        default="wildfire-nowcast/0.1",
        validation_alias="GEOCODING_USER_AGENT",
    )
    geocoding_email: str | None = Field(
        default=None, validation_alias="GEOCODING_EMAIL"
    )
    geocoding_timeout_seconds: float = Field(
        default=5.0, ge=0.5, le=30.0, validation_alias="GEOCODING_TIMEOUT_SECONDS"
    )
    geocoding_min_interval_seconds: float = Field(
        default=1.0, ge=0.0, le=60.0, validation_alias="GEOCODING_MIN_INTERVAL_SECONDS"
    )
    geocoding_cache_ttl_hours: int = Field(
        default=24 * 14, ge=1, le=24 * 365, validation_alias="GEOCODING_CACHE_TTL_HOURS"
    )
    geocoding_cache_precision: int = Field(
        default=2, ge=0, le=6, validation_alias="GEOCODING_CACHE_PRECISION"
    )
    geocoding_zoom: int = Field(
        default=6, ge=3, le=18, validation_alias="GEOCODING_ZOOM"
    )
    geocoding_accept_language: str = Field(
        default="en", validation_alias="GEOCODING_ACCEPT_LANGUAGE"
    )

    # Gemini assistant (server-side; never exposed to the browser)
    gemini_api_key: str = Field(default="", validation_alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.5-flash", validation_alias="GEMINI_MODEL")
    gemini_api_base_url: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta",
        validation_alias="GEMINI_API_BASE_URL",
    )

    @property
    def data_status_critical_sources_set(self) -> set[str]:
        return {
            source.strip().lower()
            for source in self.data_status_critical_sources.split(",")
            if source.strip()
        }


settings = AppSettings()
