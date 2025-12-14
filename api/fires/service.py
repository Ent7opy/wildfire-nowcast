"""Model-facing helpers for fire detections on the analysis grid."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

import numpy as np

from api.core.grid import GridSpec, GridWindow, get_grid_window_for_bbox
from api.fires.grid_mapping import aggregate_indices_to_grid, fires_to_indices
from api.fires.repo import BBox, list_fire_detections_bbox_time
from api.terrain import features_repo, repo as terrain_repo
from api.terrain.dem_loader import grid_spec_from_metadata

Mode = Literal["count", "presence", "sum", "max"]


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
    region_name: str,
    bbox: BBox,
    start_time: datetime,
    end_time: datetime,
    *,
    mode: Mode = "count",
    value_col: str | None = None,
    clip: bool = True,
    include_points: bool = False,
    limit: int | None = None,
) -> FireHeatmapWindow:
    """Return a bbox-windowed fire heatmap on the region analysis grid.

    Mapping is always performed on the **region** GridSpec (stable origin/extent). The
    output heatmap is then computed on the **AOI window** to keep arrays small.
    """

    grid = get_region_grid_spec(region_name)
    win = get_grid_window_for_bbox(grid, bbox, clip=clip)
    height = win.i1 - win.i0
    width = win.j1 - win.j0

    # Query only what aggregation needs.
    cols: list[str] = ["lat", "lon"]
    if mode in ("sum", "max"):
        if not value_col:
            raise ValueError(f"mode='{mode}' requires value_col.")
        cols.append(value_col)

    detections = list_fire_detections_bbox_time(
        bbox=bbox,
        start_time=start_time,
        end_time=end_time,
        columns=cols,
        limit=limit,
        order="asc",
    )
    mapped = fires_to_indices(detections, grid, drop_outside=True)

    if height <= 0 or width <= 0:
        empty = np.zeros((0, 0), dtype=(np.int32 if mode == "count" else np.float32))
        return FireHeatmapWindow(grid=grid, window=win, heatmap=empty, points=([] if include_points else None))

    if not mapped:
        heatmap = np.zeros((height, width), dtype=(np.int32 if mode == "count" else np.float32))
        return FireHeatmapWindow(grid=grid, window=win, heatmap=heatmap, points=([] if include_points else None))

    i = np.asarray([r["i"] for r in mapped], dtype=int)
    j = np.asarray([r["j"] for r in mapped], dtype=int)
    in_win = (i >= win.i0) & (i < win.i1) & (j >= win.j0) & (j < win.j1)

    i_loc = i[in_win] - win.i0
    j_loc = j[in_win] - win.j0

    vals = None
    if mode in ("sum", "max"):
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

