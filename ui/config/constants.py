"""Application constants and configuration."""

from typing import List

from config.theme import MapConfig

# Map defaults (imported from theme for centralized design system)
DEFAULT_MAP_CENTER = MapConfig.DEFAULT_CENTER
DEFAULT_ZOOM_LEVEL = MapConfig.DEFAULT_ZOOM
MAP_HEIGHT = MapConfig.HEIGHT

# Time windows (used for fires filtering)
TIME_WINDOW_OPTIONS = ["Last 6 hours", "Last 12 hours", "Last 24 hours", "Last 48 hours"]

# Placeholder data
PLACEHOLDER_FIRE_LOCATIONS: List[List[float]] = [
    [42.6977, 23.3219],  # Sofia area
    [43.2141, 27.9147],  # Varna area
    [42.5048, 27.4626],  # Burgas area
    [43.8486, 25.9542],  # Ruse area
    [42.1354, 24.7453],  # Plovdiv area
]

PLACEHOLDER_FORECAST_POLYGON: List[List[float]] = [
    [42.7, 25.0],
    [42.8, 25.0],
    [42.8, 25.2],
    [42.7, 25.2],
    [42.7, 25.0],
]

PLACEHOLDER_RISK_POLYGON: List[List[float]] = [
    [42.0, 24.0],
    [43.5, 24.0],
    [43.5, 27.0],
    [42.0, 27.0],
    [42.0, 24.0],
]
