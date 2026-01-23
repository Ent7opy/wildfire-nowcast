"""Centralized design system for Wildfire Nowcast UI.

This module provides a single source of truth for all design tokens including
colors, thresholds, sizing, and configuration values used throughout the UI.

Usage Examples:
    From PyDeck layers:
        get_fill_color=f"properties.fire_likelihood >= {FireThresholds.HIGH} ? {FireColors.HIGH_FILL} : ..."

    From Python code:
        tooltip_style = {"backgroundColor": UIColors.TOOLTIP_BG}
"""


class FireColors:
    """Colors for fire severity visualization based on likelihood thresholds.

    Fire points are colored based on their likelihood score:
    - High likelihood (>=0.66): Red
    - Medium likelihood (0.33-0.66): Orange
    - Low likelihood (<0.33): Yellow

    Each color has a FILL variant with opacity for map rendering.
    """
    # Base RGB colors (no alpha channel)
    HIGH = [255, 0, 0]        # Red - high likelihood
    MEDIUM = [255, 165, 0]    # Orange - medium likelihood
    LOW = [255, 255, 0]       # Yellow - low likelihood

    # Map layer variants with opacity (RGBA format for PyDeck)
    HIGH_FILL = [255, 0, 0, 230]        # High opacity red
    MEDIUM_FILL = [255, 165, 0, 210]    # High opacity orange
    LOW_FILL = [255, 255, 0, 190]       # High opacity yellow

    # Outline for fire points (white with transparency)
    OUTLINE = [255, 255, 255, 180]


class FireThresholds:
    """Likelihood thresholds for fire severity classification.

    These values determine when a fire point transitions between
    low/medium/high severity visualization:
    - fire_likelihood >= 0.66: High (red)
    - fire_likelihood >= 0.33: Medium (orange)
    - fire_likelihood < 0.33: Low (yellow)
    """
    HIGH = 0.66      # Threshold for high likelihood classification
    MEDIUM = 0.33    # Threshold for medium likelihood classification


class RiskColors:
    """Colors for fire risk index visualization.

    Risk zones are colored based on their risk score (0.0-1.0):
    - Low risk (<0.3): Green
    - Medium risk (0.3-0.6): Gold/Yellow
    - High risk (>=0.6): Crimson/Red

    Each level has both FILL (semi-transparent) and STROKE (more opaque) variants
    for polygon rendering on the map.
    """
    # Base RGB colors (no alpha channel)
    LOW = [34, 139, 34]       # Forest green
    MEDIUM = [255, 215, 0]    # Gold
    HIGH = [220, 20, 60]      # Crimson

    # Fill variants with reduced opacity for polygon interiors
    LOW_FILL = [34, 139, 34, 80]
    MEDIUM_FILL = [255, 215, 0, 100]
    HIGH_FILL = [220, 20, 60, 120]

    # Stroke variants with higher opacity for polygon borders
    LOW_STROKE = [34, 139, 34, 180]
    MEDIUM_STROKE = [255, 215, 0, 180]
    HIGH_STROKE = [220, 20, 60, 180]


class RiskThresholds:
    """Risk score thresholds for classification.

    These values determine risk zone coloring:
    - risk_score < 0.3: Low risk (green)
    - 0.3 <= risk_score < 0.6: Medium risk (yellow)
    - risk_score >= 0.6: High risk (red)
    """
    MEDIUM = 0.3   # Threshold between low and medium risk
    HIGH = 0.6     # Threshold between medium and high risk


class ForecastColors:
    """Colors for forecast overlay visualization.

    Forecast contours are displayed as semi-transparent orange overlays
    on the map to show predicted fire spread areas.
    """
    FILL = [255, 165, 0, 40]    # Semi-transparent orange for filled areas
    STROKE = [255, 165, 0, 200]  # Opaque orange for contour boundaries


class PointSizing:
    """Size thresholds and values for fire point rendering.

    Fire point size is determined by FRP (Fire Radiative Power):
    - FRP > 100: 12px (large, intense fires)
    - FRP > 50: 8px (medium intensity)
    - FRP > 20: 5px (small fires)
    - FRP <= 20: 3px (minimal detection)

    Pixel constraints ensure points remain visible at all zoom levels.
    """
    # FRP (Fire Radiative Power) thresholds
    LARGE_FRP = 100   # FRP threshold for large points
    MEDIUM_FRP = 50   # FRP threshold for medium points
    SMALL_FRP = 20    # FRP threshold for small points

    # Pixel sizes corresponding to FRP thresholds
    LARGE_SIZE = 12   # Pixels for large fires
    MEDIUM_SIZE = 8   # Pixels for medium fires
    SMALL_SIZE = 5    # Pixels for small fires
    MIN_SIZE = 3      # Pixels for minimal fires

    # Pixel constraints (PyDeck zoom-independent sizing)
    MIN_PIXELS = 3    # Minimum point size at any zoom level
    MAX_PIXELS = 12   # Maximum point size at any zoom level


class MapConfig:
    """Map display configuration and default view settings.

    These values control the initial map state and rendering parameters.
    """
    HEIGHT = 600                    # Map height in pixels
    DEFAULT_CENTER = [20.0, 0.0]   # [lat, lon] - Center of world map
    DEFAULT_ZOOM = 2               # Initial zoom level (world view)


class UIColors:
    """General UI colors for non-map interface elements.

    These colors are used for tooltips, backgrounds, and other UI chrome.
    """
    TOOLTIP_BG = "#333"       # Dark gray background for map tooltips
    TOOLTIP_TEXT = "white"    # White text for map tooltips
