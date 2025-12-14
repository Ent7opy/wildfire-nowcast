"""Terrain/grid validation helpers (fail fast on misalignment).

These checks are intentionally strict and low-level: they validate that rasters are
on the exact same CRS/resolution/extent as the `GridSpec` used for indexing.

Why
- The grid stack is easy to silently misconfigure (CRS mismatch, off-by-one extent,
  wrong origin edge, subtle resampling).
- These validators catch that early, before incorrect (i, j) indexing leaks into ML/UI.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio
from rasterio.crs import CRS

from api.core.grid import GridSpec

LOGGER = logging.getLogger(__name__)


def _maybe_raise(messages: Iterable[str], *, strict: bool) -> None:
    msgs = [m for m in messages if m]
    if not msgs:
        return
    joined = "; ".join(msgs)
    if strict:
        raise ValueError(joined)
    LOGGER.warning("%s", joined)


def validate_raster_matches_grid(
    raster_path: str | Path,
    grid: GridSpec,
    *,
    strict: bool = True,
) -> None:
    """Validate a single north-up GeoTIFF matches the provided `GridSpec`.

    Checks
    - CRS equals `grid.crs`
    - Pixel size equals `grid.cell_size_deg` (within a tiny tolerance)
    - Width/height equal `grid.n_lon` / `grid.n_lat`
    - Bounds/extents match the grid origin edges and size
    - Transform is north-up (no rotation/shear)
    """

    path = Path(raster_path)
    if not path.exists():
        _maybe_raise([f"Raster not found: {path}"], strict=strict)
        return

    atol = 1e-9
    atol_rot = 1e-12
    with rasterio.open(path) as src:
        expected_crs = CRS.from_string(grid.crs)
        src_crs = src.crs
        msgs: list[str] = []

        if src_crs is None:
            msgs.append(f"{path.name}: missing CRS (expected {grid.crs})")
        elif src_crs != expected_crs:
            msgs.append(f"{path.name}: CRS mismatch (got {src_crs}, expected {expected_crs})")

        # North-up (no rotation/shear).
        t = src.transform
        if not (abs(t.b) < atol_rot and abs(t.d) < atol_rot):
            msgs.append(f"{path.name}: transform has rotation/shear (transform={t})")

        cell_x = float(abs(t.a))
        cell_y = float(abs(t.e))
        if not np.isclose(cell_x, cell_y, rtol=0.0, atol=atol_rot):
            msgs.append(f"{path.name}: non-square pixels (x={cell_x}, y={cell_y})")
        if not np.isclose(cell_x, float(grid.cell_size_deg), rtol=0.0, atol=atol):
            msgs.append(
                f"{path.name}: cell_size_deg mismatch (got {cell_x}, expected {grid.cell_size_deg})"
            )

        if int(src.width) != int(grid.n_lon):
            msgs.append(f"{path.name}: width mismatch (got {src.width}, expected {grid.n_lon})")
        if int(src.height) != int(grid.n_lat):
            msgs.append(f"{path.name}: height mismatch (got {src.height}, expected {grid.n_lat})")

        # Bounds must match origin edges + extent (half-open grid extent).
        left, bottom, right, top = map(float, src.bounds)
        exp_left = float(grid.origin_lon)
        exp_bottom = float(grid.origin_lat)
        exp_right = float(grid.origin_lon + grid.n_lon * grid.cell_size_deg)
        exp_top = float(grid.origin_lat + grid.n_lat * grid.cell_size_deg)

        if not np.isclose(left, exp_left, rtol=0.0, atol=atol):
            msgs.append(f"{path.name}: left bound mismatch (got {left}, expected {exp_left})")
        if not np.isclose(bottom, exp_bottom, rtol=0.0, atol=atol):
            msgs.append(f"{path.name}: bottom bound mismatch (got {bottom}, expected {exp_bottom})")
        if not np.isclose(right, exp_right, rtol=0.0, atol=atol):
            msgs.append(f"{path.name}: right bound mismatch (got {right}, expected {exp_right})")
        if not np.isclose(top, exp_top, rtol=0.0, atol=atol):
            msgs.append(f"{path.name}: top bound mismatch (got {top}, expected {exp_top})")

        # Transform edges should also match (redundant but produces clearer errors).
        if not np.isclose(float(t.c), exp_left, rtol=0.0, atol=atol):
            msgs.append(f"{path.name}: transform.c (west) mismatch (got {t.c}, expected {exp_left})")
        if not np.isclose(float(t.f), exp_top, rtol=0.0, atol=atol):
            msgs.append(f"{path.name}: transform.f (north) mismatch (got {t.f}, expected {exp_top})")

        _maybe_raise(msgs, strict=strict)


def validate_terrain_stack(
    dem_path: str | Path | None,
    slope_path: str | Path,
    aspect_path: str | Path,
    grid: GridSpec,
    *,
    strict: bool = True,
) -> None:
    """Validate DEM/slope/aspect rasters match each other and the `GridSpec`.

    - Always validates slope + aspect.
    - Validates DEM too when `dem_path` is provided.
    """

    paths: list[Path] = [Path(slope_path), Path(aspect_path)]
    if dem_path is not None:
        paths.append(Path(dem_path))

    # First: each matches the grid contract.
    for p in paths:
        validate_raster_matches_grid(p, grid, strict=strict)

    # Second: pairwise exact alignment (CRS/transform/shape/bounds).
    # This is mostly redundant if grid validation passes, but gives clearer messages
    # when a caller passes the wrong grid.
    def _sig(p: Path) -> tuple[str | None, tuple[float, float, float, float, float, float], int, int, tuple[float, float, float, float]]:
        with rasterio.open(p) as src:
            crs_str = src.crs.to_string() if src.crs is not None else None
            t = src.transform
            transform_6 = (float(t.a), float(t.b), float(t.c), float(t.d), float(t.e), float(t.f))
            b = src.bounds
            bounds_4 = (float(b.left), float(b.bottom), float(b.right), float(b.top))
            return crs_str, transform_6, int(src.width), int(src.height), bounds_4

    msgs: list[str] = []
    existing = [p for p in paths if p.exists()]
    if len(existing) >= 2:
        base = existing[0]
        base_sig = _sig(base)
        for other in existing[1:]:
            other_sig = _sig(other)
            if other_sig != base_sig:
                msgs.append(
                    f"{other.name}: raster does not exactly match {base.name} "
                    f"(got={other_sig}, expected={base_sig})"
                )
    _maybe_raise(msgs, strict=strict)

