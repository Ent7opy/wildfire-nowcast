"""Model-facing helpers for fire detections on the analysis grid."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Optional

import numpy as np

from api.core.grid import GridSpec, GridWindow, get_grid_window_for_bbox
from api.fires.grid_mapping import aggregate_indices_to_grid, fires_to_indices
from api.fires.repo import BBox, list_fire_detections_bbox_time
from api.terrain import features_repo, repo as terrain_repo
from api.terrain.dem_loader import grid_spec_from_metadata

Mode = Literal["count", "presence", "sum", "max"]


def normalize_firms_confidence(
    confidence: Optional[float],
    sensor: Optional[str],
) -> float:
    """Normalize FIRMS confidence to a 0-1 prior for fire likelihood scoring.

    FIRMS confidence semantics differ by sensor:

    - MODIS (Terra/Aqua):
      Confidence is categorical and mapped to numeric values:
        - Low (l): 10 → Detections with low confidence, often false positives
        - Nominal (n): 50 → Standard confidence level for most fire detections
        - High (h): 90 → High confidence detections, typically large/intense fires
      Scale: 0-100 (after categorical-to-numeric mapping)

    - VIIRS (S-NPP/NOAA-20):
      Confidence is directly numeric, representing detection quality:
        - 0-30: Low confidence, high false positive rate
        - 30-70: Nominal confidence, typical fire detections
        - 70-100: High confidence, well-validated detections
      Scale: 0-100

    Both sensors use 0-100 scale, but interpretation differs slightly:
    - MODIS confidence is more categorical and coarse (3 levels)
    - VIIRS confidence is continuous and more granular

    Normalization strategy:
    - Treat confidence as a weak prior (not a hard gate)
    - Normalize to 0-1 scale where:
      - 0 = no confidence signal (missing or 0)
      - 0.1 = low confidence (MODIS low, VIIRS 0-30)
      - 0.5 = nominal confidence (MODIS nominal, VIIRS 30-70)
      - 1.0 = high confidence (MODIS high, VIIRS 70-100)
    - This prior will contribute at most 20% weight to composite Fire Likelihood Score

    Args:
        confidence: Raw FIRMS confidence value (0-100) or None
        sensor: Sensor identifier (e.g., "VIIRS", "Terra", "Aqua") or None

    Returns:
        Normalized confidence prior in range [0, 1]
        Returns 0.5 (neutral prior) if confidence is missing
    """
    # Handle missing confidence: return neutral prior (0.5)
    if confidence is None:
        return 0.5

    # Ensure confidence is in valid range [0, 100]
    confidence_clamped = max(0.0, min(100.0, confidence))

    # Simple linear normalization: 0-100 → 0-1
    # This works for both MODIS and VIIRS since both use 0-100 scale
    normalized = confidence_clamped / 100.0

    return normalized


def compute_confidence_prior(
    confidence: Optional[float],
    sensor: Optional[str],
    max_weight: float = 0.2,
) -> float:
    """Compute confidence contribution to Fire Likelihood Score.

    This applies the normalized confidence with a maximum weight constraint,
    ensuring confidence alone cannot dominate the fire likelihood assessment.

    Args:
        confidence: Raw FIRMS confidence value (0-100) or None
        sensor: Sensor identifier or None
        max_weight: Maximum weight for confidence contribution (default 0.2 = 20%)

    Returns:
        Weighted confidence prior in range [0, max_weight]
    """
    normalized = normalize_firms_confidence(confidence, sensor)
    return normalized * max_weight


def get_region_grid_spec(region_name: str) -> GridSpec:
    """Load the canonical region GridSpec used for stable indexing."""

    features_md = features_repo.get_latest_terrain_features_metadata_for_region(region_name)
    if features_md is not None:
        return GridSpec(
            crs=f"EPSG:{features_md.crs_epsg}",
            cell_size_deg=float(features_md.cell_size_deg),
            origin_lat=float(features_md.origin_lat),
            origin_lon=float(features_md.origin_lon),
            n_lat=int(features_md.grid_n_lat),
            n_lon=int(features_md.grid_n_lon),
        )

    dem_md = terrain_repo.get_latest_dem_metadata_for_region(region_name)
    if dem_md is not None:
        return grid_spec_from_metadata(dem_md)

    raise ValueError(f"No terrain metadata found for region '{region_name}'.")


@dataclass(frozen=True, slots=True)
class FireHeatmapWindow:
    grid: GridSpec
    window: GridWindow
    heatmap: np.ndarray  # (lat, lon) in analysis order, window-shaped
    points: list[dict[str, Any]] | None = None


def get_fire_cells_heatmap(
    region_name: str | None,
    bbox: BBox,
    start_time: datetime,
    end_time: datetime,
    *,
    grid: GridSpec | None = None,
    mode: Mode = "count",
    value_col: str | None = None,
    clip: bool = True,
    include_points: bool = False,
    limit: int | None = None,
    include_noise: bool = False,
    weight_by_denoised_score: bool = False,
) -> FireHeatmapWindow:
    """Return a bbox-windowed fire heatmap on the region analysis grid.

    Mapping is always performed on the **region** GridSpec (stable origin/extent). The
    output heatmap is then computed on the **AOI window** to keep arrays small.

    If region_name is None, a grid must be provided.

    Weighting:
    - If `weight_by_denoised_score` is True, `mode` defaults to "sum" and `value_col`
      defaults to "denoised_score" (unless explicitly provided).
    - Unscored detections may have NULL `denoised_score`; when weighting by denoised
      score, those are treated as full-weight (1.0) to avoid NaN poisoning.
    """

    if grid is None:
        if region_name is None:
            raise ValueError("Either region_name or grid must be provided.")
        grid = get_region_grid_spec(region_name)

    win = get_grid_window_for_bbox(grid, bbox, clip=clip)

    # Handle weighting shortcut.
    if weight_by_denoised_score:
        if mode == "count":
            mode = "sum"
        if value_col is None:
            value_col = "denoised_score"

    # Derive dimensions from the coordinate arrays to guarantee consistency even if
    # window indices are degenerate (e.g. i0 == i1) or otherwise inconsistent.
    height = int(win.lat.size)
    width = int(win.lon.size)

    # Query only what aggregation needs.
    cols: list[str] = ["lat", "lon"]
    if mode in ("sum", "max"):
        if not value_col:
            raise ValueError(f"mode='{mode}' requires value_col.")
        cols.append(value_col)

    # If the AOI window is empty in either dimension, return a correctly-shaped empty
    # heatmap that matches (len(window.lat), len(window.lon)) and avoid a DB query.
    if height == 0 or width == 0:
        heatmap = aggregate_indices_to_grid(
            i=np.asarray([], dtype=int),
            j=np.asarray([], dtype=int),
            shape=(height, width),
            mode=mode,
        )
        return FireHeatmapWindow(grid=grid, window=win, heatmap=heatmap, points=([] if include_points else None))

    detections = list_fire_detections_bbox_time(
        bbox=bbox,
        start_time=start_time,
        end_time=end_time,
        columns=cols,
        limit=limit,
        order="asc",
        include_noise=include_noise,
    )
    mapped = fires_to_indices(detections, grid, drop_outside=True)

    if not mapped:
        heatmap = aggregate_indices_to_grid(
            i=np.asarray([], dtype=int),
            j=np.asarray([], dtype=int),
            shape=(height, width),
            mode=mode,
        )
        return FireHeatmapWindow(grid=grid, window=win, heatmap=heatmap, points=([] if include_points else None))

    i = np.asarray([r["i"] for r in mapped], dtype=int)
    j = np.asarray([r["j"] for r in mapped], dtype=int)
    in_win = (i >= win.i0) & (i < win.i1) & (j >= win.j0) & (j < win.j1)

    i_loc = i[in_win] - win.i0
    j_loc = j[in_win] - win.j0

    vals = None
    if mode in ("sum", "max"):
        # NOTE: SQL NULL → Python None → numpy NaN when dtype=float. If we pass NaNs into
        # np.add.at (used by aggregation), a single NaN can poison an entire cell and
        # silently corrupt downstream calculations.
        #
        # For denoiser weighting, treat unscored detections (NULL denoised_score) as
        # full-weight (1.0) rather than producing NaNs.
        if weight_by_denoised_score and value_col == "denoised_score":
            vals_all = np.asarray([(r[value_col] if r[value_col] is not None else 1.0) for r in mapped], dtype=float)  # type: ignore[index]
            vals_all = np.nan_to_num(vals_all, nan=1.0, posinf=1.0, neginf=1.0)
        else:
            vals_all = np.asarray([r[value_col] for r in mapped], dtype=float)  # type: ignore[index]
        vals = vals_all[in_win]

    heatmap = aggregate_indices_to_grid(
        i=i_loc,
        j=j_loc,
        shape=(height, width),
        mode=mode,
        values=vals,
    )

    points = None
    if include_points:
        points = []
        for r, keep in zip(mapped, in_win, strict=False):
            if not keep:
                continue
            rr = dict(r)
            rr["i_local"] = int(rr["i"] - win.i0)
            rr["j_local"] = int(rr["j"] - win.j0)
            points.append(rr)

    return FireHeatmapWindow(grid=grid, window=win, heatmap=heatmap, points=points)

