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
    firms_watermark_grace_minutes: int = Field(
        default=90,
        validation_alias="FIRMS_WATERMARK_GRACE_MINUTES",
    )
    firms_initial_lookback_minutes: int = Field(
        default=360,
        validation_alias="FIRMS_INITIAL_LOOKBACK_MINUTES",
    )
    firms_incremental_lookback_minutes: int = Field(
        default=30,
        validation_alias="FIRMS_INCREMENTAL_LOOKBACK_MINUTES",
    )
    firms_reconcile_unscored_batches: bool = Field(
        default=True,
        validation_alias="FIRMS_RECONCILE_UNSCORED_BATCHES",
    )
    firms_reconcile_max_batches: int = Field(
        default=5,
        validation_alias="FIRMS_RECONCILE_MAX_BATCHES",
    )

    # Denoiser settings
    denoiser_enabled: bool = Field(default=False, validation_alias="DENOISER_ENABLED")
    denoiser_required: bool = Field(default=True, validation_alias="DENOISER_REQUIRED")
    denoiser_model_run_dir: Optional[str] = Field(
        default=None, validation_alias="DENOISER_MODEL_RUN_DIR"
    )
    denoiser_threshold: float = Field(default=0.5, validation_alias="DENOISER_THRESHOLD")
    denoiser_batch_size: int = Field(default=500, validation_alias="DENOISER_BATCH_SIZE")
    denoiser_region: Optional[str] = Field(default=None, validation_alias="DENOISER_REGION")
    denoiser_pipeline_version: str = Field(
        default="v2",
        validation_alias="DENOISER_PIPELINE_VERSION",
    )
    denoiser_threshold_profile: str = Field(
        default="strict_v1",
        validation_alias="DENOISER_THRESHOLD_PROFILE",
    )
    denoiser_allow_unsafe_threshold_override: bool = Field(
        default=False,
        validation_alias="DENOISER_ALLOW_UNSAFE_THRESHOLD_OVERRIDE",
    )
    denoiser_shadow_mode: bool = Field(
        default=False,
        validation_alias="DENOISER_SHADOW_MODE",
    )
    denoiser_strong_filter_threshold: float = Field(
        default=0.5,
        validation_alias="DENOISER_STRONG_FILTER_THRESHOLD",
    )
    denoiser_downweight_threshold: float = Field(
        default=0.7,
        validation_alias="DENOISER_DOWNWEIGHT_THRESHOLD",
    )
    denoiser_uncertainty_band_low: float = Field(
        default=0.45,
        validation_alias="DENOISER_UNCERTAINTY_BAND_LOW",
    )
    denoiser_uncertainty_band_high: float = Field(
        default=0.55,
        validation_alias="DENOISER_UNCERTAINTY_BAND_HIGH",
    )
    denoiser_event_front_radius_m: float = Field(
        default=2500.0,
        validation_alias="DENOISER_EVENT_FRONT_RADIUS_M",
    )
    denoiser_event_front_max_gap_minutes: int = Field(
        default=45,
        validation_alias="DENOISER_EVENT_FRONT_MAX_GAP_MINUTES",
    )
    denoiser_event_link_radius_m: float = Field(
        default=10000.0,
        validation_alias="DENOISER_EVENT_LINK_RADIUS_M",
    )
    denoiser_event_link_max_gap_days: int = Field(
        default=11,
        validation_alias="DENOISER_EVENT_LINK_MAX_GAP_DAYS",
    )
    denoiser_event_static_persistence_threshold: float = Field(
        default=0.85,
        validation_alias="DENOISER_EVENT_STATIC_PERSISTENCE_THRESHOLD",
    )
    denoiser_event_strict_static_split: bool = Field(
        default=True,
        validation_alias="DENOISER_EVENT_STRICT_STATIC_SPLIT",
    )
    denoiser_strict_features: bool = Field(
        default=False,
        validation_alias="DENOISER_STRICT_FEATURES",
    )
    denoiser_invoke_method: str = Field(
        default="uv", validation_alias="DENOISER_INVOKE_METHOD"
    )
    """How to invoke the denoiser: 'uv' (default), 'python', or 'module'.

    - 'uv': Uses 'uv run --project ml -m ml.denoiser_inference' (requires uv in PATH)
    - 'python': Uses 'python -m ml.denoiser_inference' (uses sys.executable or DENOISER_PYTHON_EXECUTABLE)
    - 'module': Directly imports and calls the module (no subprocess)
    """
    denoiser_python_executable: Optional[str] = Field(
        default=None, validation_alias="DENOISER_PYTHON_EXECUTABLE"
    )
    """Override the Python executable used when DENOISER_INVOKE_METHOD=python.
    Defaults to sys.executable if not set. Useful in Docker when the ML venv
    has a different Python than the ingest venv (e.g., /app/.venv/bin/python3).
    """
    denoiser_subprocess_timeout_seconds: int = Field(
        default=600,
        validation_alias="DENOISER_SUBPROCESS_TIMEOUT_SECONDS",
    )
    """Hard timeout (seconds) for the denoiser subprocess. Prevents a stalled
    denoiser from blocking the ingest job indefinitely. Defaults to 180s (3 min).
    Set to 0 to disable (not recommended for production).
    """

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

    @field_validator("firms_watermark_grace_minutes", mode="before")
    @classmethod
    def _validate_watermark_grace_minutes(cls, value: object) -> int:
        val = int(value)
        if val < 0:
            raise ValueError("FIRMS_WATERMARK_GRACE_MINUTES must be >= 0")
        return val

    @field_validator(
        "firms_initial_lookback_minutes",
        "firms_incremental_lookback_minutes",
        mode="before",
    )
    @classmethod
    def _validate_positive_lookback_minutes(cls, value: object) -> int:
        val = int(value)
        if val <= 0:
            raise ValueError("FIRMS lookback windows must be > 0 minutes")
        return val

    @field_validator("denoiser_pipeline_version")
    @classmethod
    def _validate_denoiser_pipeline_version(cls, value: str) -> str:
        normalized = str(value or "").strip().lower()
        if normalized not in {"v1", "v2"}:
            raise ValueError("DENOISER_PIPELINE_VERSION must be one of: v1, v2")
        return normalized

    @field_validator("denoiser_threshold_profile")
    @classmethod
    def _validate_denoiser_threshold_profile(cls, value: str) -> str:
        normalized = str(value or "").strip().lower()
        if normalized not in {"strict_v1", "env"}:
            raise ValueError("DENOISER_THRESHOLD_PROFILE must be one of: strict_v1, env")
        return normalized

    @field_validator(
        "denoiser_strong_filter_threshold",
        "denoiser_downweight_threshold",
        "denoiser_uncertainty_band_low",
        "denoiser_uncertainty_band_high",
        mode="after",
    )
    @classmethod
    def _validate_probability_threshold(cls, value: float) -> float:
        numeric = float(value)
        if not 0.0 <= numeric <= 1.0:
            raise ValueError("Denoiser thresholds must be between 0.0 and 1.0")
        return numeric

    @field_validator("denoiser_event_static_persistence_threshold", mode="after")
    @classmethod
    def _validate_event_static_threshold(cls, value: float) -> float:
        numeric = float(value)
        if not 0.0 <= numeric <= 1.0:
            raise ValueError("DENOISER_EVENT_STATIC_PERSISTENCE_THRESHOLD must be between 0.0 and 1.0")
        return numeric

    @field_validator(
        "denoiser_event_front_radius_m",
        "denoiser_event_link_radius_m",
        mode="after",
    )
    @classmethod
    def _validate_positive_radius(cls, value: float) -> float:
        numeric = float(value)
        if numeric <= 0.0:
            raise ValueError("Event association radii must be > 0")
        return numeric

    @field_validator(
        "denoiser_event_front_max_gap_minutes",
        "denoiser_event_link_max_gap_days",
        mode="after",
    )
    @classmethod
    def _validate_positive_gap(cls, value: int) -> int:
        numeric = int(value)
        if numeric <= 0:
            raise ValueError("Event association gap windows must be > 0")
        return numeric

    @field_validator("denoiser_uncertainty_band_high", mode="after")
    @classmethod
    def _validate_uncertainty_band_order(cls, value: float, info) -> float:
        low = info.data.get("denoiser_uncertainty_band_low")
        if low is not None and float(value) < float(low):
            raise ValueError(
                "DENOISER_UNCERTAINTY_BAND_HIGH must be greater than or equal to "
                "DENOISER_UNCERTAINTY_BAND_LOW"
            )
        return float(value)

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
    # Default to global coverage for worldwide wildfire monitoring
    bbox_min_lon: float = Field(default=-180.0, validation_alias="WEATHER_BBOX_MIN_LON")
    bbox_max_lon: float = Field(default=180.0, validation_alias="WEATHER_BBOX_MAX_LON")
    bbox_min_lat: float = Field(default=-90.0, validation_alias="WEATHER_BBOX_MIN_LAT")
    bbox_max_lat: float = Field(default=90.0, validation_alias="WEATHER_BBOX_MAX_LAT")
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


class FuelIngestSettings(BaseSettings):
    """Environment-driven configuration for fuel/moisture feature ingestion."""

    model_config = SettingsConfigDict(
        env_file=None,
        case_sensitive=False,
        extra="ignore",
    )

    provider_url: str = Field(
        default="https://forest-fire.emergency.copernicus.eu/apps/effis_current_situation/",
        validation_alias="FUEL_PROVIDER_URL",
    )
    cache_root: Path = Field(
        default=REPO_ROOT / "data" / "fuels",
        validation_alias="FUEL_CACHE_ROOT",
    )
    freshness_ttl_hours: int = Field(
        default=24,
        validation_alias="FUEL_FRESHNESS_TTL_HOURS",
    )
    request_timeout_seconds: float = Field(
        default=30.0,
        validation_alias="FUEL_REQUEST_TIMEOUT_SECONDS",
    )
    max_retries: int = Field(
        default=3,
        validation_alias="FUEL_MAX_RETRIES",
    )
    retry_backoff_seconds: float = Field(
        default=5.0,
        validation_alias="FUEL_RETRY_BACKOFF_SECONDS",
    )
    enable_network_fetch: bool = Field(
        default=False,
        validation_alias="FUEL_ENABLE_NETWORK_FETCH",
    )


fuel_settings = FuelIngestSettings()
