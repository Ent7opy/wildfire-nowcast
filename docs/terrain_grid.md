## Terrain + Grid Contract (Contributor Guide)

Audience: anyone touching **terrain rasters**, **grid indexing**, **ML features**, **API windowing**, or the **UI map**.

This doc describes the **contract** that makes `(lat, lon) ↔ (i, j)` stable across ingest → API → ML → UI, and explains how terrain rasters (DEM / slope / aspect) must align to that grid.

### Canonical CRS + resolution

Source of truth: `docs/grid_choice.md`.

- **CRS**: `EPSG:4326` (WGS84 lon/lat)
- **Cell size**: `0.01°` (default)

### Grid definition (what `GridSpec` means)

`GridSpec` lives in `api/core/grid.py`.

- **Indices**: `(i, j) = (lat_index, lon_index)` (0-based)
- **Direction**:
  - `i` increases **south → north** (latitude increases)
  - `j` increases **west → east** (longitude increases)
- **Origin**:
  - `origin_lat` and `origin_lon` are the **southern / western cell edges**
  - They are **not** cell centers
- **Cell centers**:
  - `lat[i] = origin_lat + (i + 0.5) * cell_size_deg`
  - `lon[j] = origin_lon + (j + 0.5) * cell_size_deg`
- **Extent is half-open**:
  - Bounds are `(min_lon, min_lat, max_lon, max_lat) = (origin_lon, origin_lat, origin_lon + n_lon*cell, origin_lat + n_lat*cell)`
  - Points on the **max edge** are **out of bounds** (they map to index `n_lat` / `n_lon`)

### Windowing (half-open slices)

Most APIs use half-open windows `(i0:i1, j0:j1)`:

- `i0`, `j0` are inclusive
- `i1`, `j1` are exclusive
- `len(lat) == i1 - i0`, `len(lon) == j1 - j0`

Use:
- `api.core.grid.get_grid_window_for_bbox(...)` (bbox → indices + coords)
- `api.core.grid.window_coords(...)` (indices → cell-center coords)

### Terrain alignment contract (DEM / slope / aspect)

All terrain rasters for a region must be on the **exact same** grid:

- **CRS equals** the grid CRS (`EPSG:4326`)
- **Pixel size equals** `grid.cell_size_deg` (square pixels)
- **Width/height equal** `grid.n_lon` / `grid.n_lat`
- **Bounds match** the grid edges:
  - left == `grid.origin_lon`
  - bottom == `grid.origin_lat`
  - right == `grid.origin_lon + grid.n_lon * cell`
  - top == `grid.origin_lat + grid.n_lat * cell`

Runtime enforcement (fail fast):
- `api/terrain/validate.py`:
  - `validate_raster_matches_grid(...)`
  - `validate_terrain_stack(...)`

### GeoTIFF row order vs analysis convention (important!)

GeoTIFFs are usually **north-up**:
- raster row `0` is the **northmost** pixels
- row index increases **southward**

But analysis arrays in this repo are **lat-ascending**:
- `lat` is monotonic increasing (south → north)
- array index `i` increases (south → north)

Therefore:
- When reading via rasterio windows (see `api/terrain/window.py`), we convert
  analysis indices to raster row offsets and **flip rows once** (`np.flipud`) so
  consumers always get `(lat, lon)` arrays with **lat increasing**.
- When writing synthetic test rasters, you typically need to **flip on write** to
  store as north-up.

### File locations and naming conventions

On-disk paths (typical):

- **DEM**: `data/dem/<region>/dem_<region>_epsg4326_0p01deg.tif`
- **Slope**: `data/terrain/<region>/slope_<region>_epsg4326_0p01deg.tif`
- **Aspect**: `data/terrain/<region>/aspect_<region>_epsg4326_0p01deg.tif`

Database metadata:

- `terrain_metadata` stores DEM path + grid fields (origin, size, cell size, bbox)
- `terrain_features_metadata` stores slope/aspect paths + grid fields and nodata/conventions

### Usage recipes

#### 1) Get a grid window for a bbox / AOI

```python
from api.core.grid import GridSpec, get_grid_window_for_bbox

grid = GridSpec.from_bbox(lat_min=35.0, lat_max=36.0, lon_min=5.0, lon_max=6.0)  # snapped
bbox = (5.12, 35.45, 5.99, 36.00)  # (min_lon, min_lat, max_lon, max_lat)
win = get_grid_window_for_bbox(grid, bbox, clip=True)
# win.i0, win.i1, win.j0, win.j1 are half-open indices
```

#### 2) Load slope/aspect (and optional DEM) for that window

```python
from api.terrain.window import load_terrain_window

tw = load_terrain_window("smoke_grid", bbox, include_dem=True)
# tw.slope, tw.aspect, tw.elevation are numpy arrays shaped (n_lat, n_lon) in (lat, lon) order
```

#### 3) Map fire detections → (i, j) and build a heatmap

```python
import numpy as np
from api.core.grid import GridSpec, latlon_to_index

grid = GridSpec.from_bbox(lat_min=35.0, lat_max=36.0, lon_min=5.0, lon_max=6.0)
heat = np.zeros((grid.n_lat, grid.n_lon), dtype=np.float32)

# detections: arrays of lat/lon (EPSG:4326)
i, j = latlon_to_index(grid, lat=detections_lat, lon=detections_lon)
in_bounds = (0 <= i) & (i < grid.n_lat) & (0 <= j) & (j < grid.n_lon)
np.add.at(heat, (i[in_bounds], j[in_bounds]), 1.0)
```

### Common pitfalls (things that silently break alignment)

- **North-up vs analysis order**: reading a GeoTIFF directly as an array without flipping/coord sorting can invert latitude.
- **Lon normalization**: standardize longitudes to `[-180, 180]` before indexing if your source uses `[0, 360)`.
- **Boundary semantics**: `max_lat` / `max_lon` edges are out-of-bounds by design (half-open extent).
- **“Looks aligned” but isn’t**: rasters can have the same CRS/resolution but differ by a half-cell origin shift. Use `validate_terrain_stack(...)`.

