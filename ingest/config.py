"""Configuration helpers for ingestion pipelines."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env", override=False)


class FIRMSIngestSettings(BaseSettings):
    """Environment-driven configuration for the FIRMS ingestion pipeline."""

    model_config = SettingsConfigDict(
        env_file=None,
        case_sensitive=False,
        extra="ignore",
    )

    map_key: str = Field(default="", validation_alias="FIRMS_MAP_KEY")
    sources: List[str] = Field(
        default_factory=lambda: ["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT"],
        validation_alias="FIRMS_SOURCES",
    )
    area: str = Field(default="world", validation_alias="FIRMS_AREA")
    day_range: int = Field(default=1, validation_alias="FIRMS_DAY_RANGE")
    request_timeout_seconds: float = Field(
        default=30.0,
        validation_alias="FIRMS_REQUEST_TIMEOUT_SECONDS",
    )

    # Denoiser settings
    denoiser_enabled: bool = Field(default=False, validation_alias="DENOISER_ENABLED")
    denoiser_model_run_dir: Optional[str] = Field(
        default=None, validation_alias="DENOISER_MODEL_RUN_DIR"
    )
    denoiser_threshold: float = Field(default=0.5, validation_alias="DENOISER_THRESHOLD")
    denoiser_batch_size: int = Field(default=500, validation_alias="DENOISER_BATCH_SIZE")
    denoiser_region: Optional[str] = Field(default=None, validation_alias="DENOISER_REGION")

    @field_validator("sources", mode="before")
    @classmethod
    def _split_sources(cls, value: object) -> List[str]:
        if value is None:
            return ["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT"]
        if isinstance(value, str):
            return [segment.strip() for segment in value.split(",") if segment.strip()]
        if isinstance(value, list):
            return value
        raise ValueError("FIRMS_SOURCES must be a comma-separated string or list.")

    @field_validator("area", mode="before")
    @classmethod
    def _normalize_area(cls, value: object) -> str:
        if value is None:
            return "world"
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned.lower() == "world":
                return "world"
            parts = [p.strip() for p in cleaned.split(",")]
            if len(parts) != 4:
                raise ValueError("FIRMS_AREA must be 'world' or 'west,south,east,north'")
            float_parts: Tuple[float, float, float, float] = tuple(float(p) for p in parts)
            return ",".join(str(p) for p in float_parts)
        raise ValueError("FIRMS_AREA must be a string")

    @field_validator("day_range", mode="before")
    @classmethod
    def _validate_day_range(cls, value: object) -> int:
        val = int(value)  # raises if not numeric
        if not 1 <= val <= 10:
            raise ValueError("FIRMS_DAY_RANGE must be between 1 and 10")
        return val

    @property
    def resolved_area(self) -> str:
        """Convert the configured area label into the FIRMS bbox string."""
        if self.area.lower() == "world":
            return "-180,-90,180,90"
        return self.area


settings = FIRMSIngestSettings()


class WeatherIngestSettings(BaseSettings):
    """Environment-driven configuration for weather ingestion."""

    model_config = SettingsConfigDict(
        env_file=None,
        case_sensitive=False,
        extra="ignore",
    )

    model_name: str = Field(default="gfs_0p25", validation_alias="WEATHER_MODEL")
    gfs_base_url_primary: str = Field(
        default="https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl",
        validation_alias="WEATHER_GFS_BASE_URL",
    )
    gfs_base_url_fallback: Optional[str] = Field(
        default=None,
        validation_alias="WEATHER_GFS_FALLBACK_URL",
    )
    base_dir: Path = Field(
        default=REPO_ROOT / "data" / "weather",
        validation_alias="WEATHER_BASE_DIR",
    )
    bbox_min_lon: float = Field(default=5.0, validation_alias="WEATHER_BBOX_MIN_LON")
    bbox_max_lon: float = Field(default=20.0, validation_alias="WEATHER_BBOX_MAX_LON")
    bbox_min_lat: float = Field(default=35.0, validation_alias="WEATHER_BBOX_MIN_LAT")
    bbox_max_lat: float = Field(default=47.0, validation_alias="WEATHER_BBOX_MAX_LAT")
    horizon_hours: int = Field(default=24, validation_alias="WEATHER_HORIZON_HOURS")
    step_hours: int = Field(default=6, validation_alias="WEATHER_STEP_HOURS")
    run_time: Optional[datetime] = Field(default=None, validation_alias="WEATHER_RUN_TIME")
    request_timeout_seconds: int = Field(
        default=60,
        validation_alias="WEATHER_REQUEST_TIMEOUT_SECONDS",
    )
    include_precipitation: bool = Field(
        default=False,
        validation_alias="WEATHER_INCLUDE_PRECIP",
    )

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        return (
            float(self.bbox_min_lon),
            float(self.bbox_min_lat),
            float(self.bbox_max_lon),
            float(self.bbox_max_lat),
        )


weather_settings = WeatherIngestSettings()


