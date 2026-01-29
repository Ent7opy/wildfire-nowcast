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

    Fire points use a 5-tier graduated color scheme from yellow to deep red:
    - Very high (>=0.8): Deep red
    - High (>=0.6): Red
    - Medium (>=0.4): Ember orange
    - Low (>=0.2): Amber
    - Very low (<0.2): Yellow

    Each tier has a FILL variant with opacity for map rendering.
    """
    # 5-tier gradient colors (RGBA for PyDeck)
    VERY_HIGH_FILL = [220, 38, 38, 240]    # Deep red (>=0.8)
    HIGH_FILL = [239, 68, 68, 230]         # Red (>=0.6)
    MEDIUM_FILL = [255, 107, 53, 220]      # Ember orange (>=0.4)
    LOW_FILL = [251, 191, 36, 200]         # Amber (>=0.2)
    VERY_LOW_FILL = [253, 224, 71, 180]    # Yellow (<0.2)

    # Base RGB colors (no alpha) for legend swatches
    VERY_HIGH = [220, 38, 38]
    HIGH = [239, 68, 68]
    MEDIUM = [255, 107, 53]
    LOW = [251, 191, 36]
    VERY_LOW = [253, 224, 71]

    # Unscored / NULL fire_likelihood (gray — visually distinct from yellow)
    UNSCORED_FILL = [128, 128, 128, 150]
    UNSCORED = [128, 128, 128]

    # Outline colors (conditional by confidence)
    OUTLINE_HIGH = [255, 107, 53, 200]     # Ember orange glow for high-confidence
    OUTLINE_DEFAULT = [255, 255, 255, 100] # Subtle white for lower confidence


class FireThresholds:
    """Likelihood thresholds for 5-tier fire severity classification.

    These values determine graduated color transitions:
    - fire_likelihood >= 0.8: Very high (deep red)
    - fire_likelihood >= 0.6: High (red)
    - fire_likelihood >= 0.4: Medium (ember orange)
    - fire_likelihood >= 0.2: Low (amber)
    - fire_likelihood < 0.2: Very low (yellow)
    """
    VERY_HIGH = 0.8  # Deep red threshold
    HIGH = 0.6       # Red threshold
    MEDIUM = 0.4     # Ember orange threshold
    LOW = 0.2        # Amber threshold


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
    BASEMAP_DARK = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
    BASEMAP_LIGHT = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"


class UIColors:
    """General UI colors for non-map interface elements.

    These colors are used for tooltips, backgrounds, and other UI chrome.
    """
    TOOLTIP_BG = "#252930"    # Card background for map tooltips
    TOOLTIP_TEXT = "#e0e0e0"  # Light text for map tooltips


class DarkTheme:
    """Dark theme color tokens for UI backgrounds, text, and accents."""
    BG_PRIMARY = "#0a1628"       # Main background (navy)
    BG_SECONDARY = "#1a1d29"     # Sidebar / secondary (charcoal)
    BG_CARD = "#252930"          # Card backgrounds
    ACCENT_EMBER = "#ff6b35"     # Primary accent (ember orange)
    ACCENT_WARNING = "#e63946"   # Warning red
    ACCENT_AMBER = "#fbbf24"     # Medium-priority amber
    TEXT_PRIMARY = "#e0e0e0"     # Primary text
    TEXT_SECONDARY = "rgba(255,255,255,0.7)"  # Muted text
    BORDER_SUBTLE = "rgba(255,255,255,0.08)"  # Subtle borders


class Typography:
    """Font and sizing tokens."""
    FONT_STACK = "'Inter', -apple-system, 'Segoe UI', 'Roboto', sans-serif"
    HEADER_SIZE = "18px"
    HEADER_WEIGHT = "600"
    BODY_SIZE = "14px"
    BODY_LINE_HEIGHT = "16px"
    CAPTION_SIZE = "12px"


class FilterPresets:
    """Predefined filter combinations for quick access.

    Each preset is a tuple of (name, hours_start, hours_end, min_likelihood, apply_denoiser).
    - hours_start: How many hours ago the time range starts (e.g., 24 = 24 hours ago)
    - hours_end: How many hours ago the time range ends (0 = now)
    These provide quick one-click access to common filter configurations.
    """

    # Preset format: (name, hours_start, hours_end, min_likelihood, apply_denoiser)
    # NOTE: "High" presets align with FireThresholds.HIGH (0.6) so all passing
    # fires are guaranteed to render as red or deep red on the map.
    LAST_HOUR_HIGH = ("Last Hour High", 1, 0, 0.6, True)        # Last 1h, high confidence
    LAST_6H_MEDIUM = ("Last 6h Medium+", 6, 0, 0.33, True)      # Last 6h, medium+ confidence
    LAST_24H_HIGH = ("Last 24h High", 24, 0, 0.6, True)         # Last 24h, high confidence
    LAST_24H_ALL = ("Last 24h All", 24, 0, 0.0, False)          # Last 24h, all fires
    CUSTOM = ("Custom", None, None, None, None)                  # User-defined filters

    @classmethod
    def all_presets(cls):
        """Return list of all preset tuples (excludes Custom)."""
        return [
            cls.LAST_HOUR_HIGH,
            cls.LAST_6H_MEDIUM,
            cls.LAST_24H_HIGH,
            cls.LAST_24H_ALL,
        ]

    @classmethod
    def all_presets_with_custom(cls):
        """Return list of all preset tuples including Custom."""
        return [
            cls.LAST_HOUR_HIGH,
            cls.LAST_6H_MEDIUM,
            cls.LAST_24H_HIGH,
            cls.LAST_24H_ALL,
            cls.CUSTOM,
        ]
