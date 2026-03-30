"""Shared terrain label utility — maps ESA WorldCover class codes to human-readable labels.

Kept in sync with ingest/lulc_worldcover_ingest.py _CLASS_LABELS.
Used by the review queue API and the fire detail view.
"""

from __future__ import annotations

# Maps ESA WorldCover class code → human-readable label.
TERRAIN_LABELS: dict[int, str] = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}


def terrain_label_from_class(class_code: int | None) -> str | None:
    """Return human-readable terrain label for an ESA WorldCover class code.

    Returns None if class_code is None or not a known ESA WorldCover class.
    """
    if class_code is None:
        return None
    return TERRAIN_LABELS.get(int(class_code))
