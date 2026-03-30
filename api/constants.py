# api/constants.py
# Fixed-by-design values shared across routes and workers.
# Values that differ between deployments belong in api/config.py (Pydantic settings).

# Forecast horizons
DEFAULT_HORIZONS_HOURS: list[int] = [24, 48, 72]
MAX_HORIZON_HOURS: int = 72

# Fire detection columns
FIRE_DETECTION_BASE_COLUMNS: list[str] = [
    "id",
    "lat",
    "lon",
    "acq_time",
    "confidence",
    "brightness",
    "bright_t31",
    "frp",
    "sensor",
    "source",
    "confidence_score",
    "persistence_score",
    "landcover_score",
    "weather_score",
    "false_source_masked",
    "fire_likelihood",
]

FIRE_DETECTION_DENOISER_COLUMNS: list[str] = [
    "denoised_score",
    "is_noise",
    "event_id",
    "event_score",
    "denoiser_decision",
    "review_required",
]

# Bounding box
MAX_BBOX_AREA_DEG2: float = 25.0  # 5° × 5° hard cap

# Archive — default only; runtime value is read from MAX_ARCHIVE_RANGE_DAYS env var
MAX_ARCHIVE_RANGE_DAYS_DEFAULT: int = 7
