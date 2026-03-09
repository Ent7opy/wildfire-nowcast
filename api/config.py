from __future__ import annotations

import os
import subprocess
from importlib import metadata
from pathlib import Path

from pydantic import Field
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

    # Database settings
    postgres_host: str = Field(default="localhost", validation_alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, validation_alias="POSTGRES_PORT")
    postgres_user: str = Field(default="wildfire", validation_alias="POSTGRES_USER")
    postgres_password: str = Field(default="wildfire", validation_alias="POSTGRES_PASSWORD")
    postgres_db: str = Field(default="wildfire", validation_alias="POSTGRES_DB")

    @property
    def database_url(self) -> str:
        """Construct database URL from individual components."""
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
    data_status_critical_sources: str = Field(
        default="firms,weather",
        validation_alias="DATA_STATUS_CRITICAL_SOURCES",
    )

    @property
    def data_status_critical_sources_set(self) -> set[str]:
        return {
            source.strip().lower()
            for source in self.data_status_critical_sources.split(",")
            if source.strip()
        }


settings = AppSettings()
