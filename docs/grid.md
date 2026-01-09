# Analysis Grid Contract

**Audience**: Anyone touching terrain rasters, grid indexing, ML features, API windowing, or the UI map.

This doc describes the **contract** that makes `(lat, lon) ↔ (i, j)` stable across ingest → API → ML → UI, and explains how all data (fires, terrain, weather) must align to that grid.

---

## 1. Grid Choice & Rationale

### Canonical Grid Specification

- **CRS**: `EPSG:4326` (WGS84 geographic)
  - Matches FIRMS, GFS, and Copernicus GLO-30
  - Avoids extra reprojections in the ingest path
  - Distortion from non-equal-area is acceptable for the ~1 km MVP
  - UI can reproject to `EPSG:3857` separately

- **Resolution**: Fixed `0.01°` in both lat/lon
  - ≈1.11 km at the equator, ≈0.9–1.0 km in mid-latitudes
  - DEM is downsampled from 30 m
  - GFS is interpolated from 0.25°
  - FIRMS points land on this grid without reprojection

- **Size example**: A 10° × 10° region becomes a 1000 × 1000 grid (~1M cells)

### GridSpec Parameterization

Canonical dataclass (in `api/core/grid.py`):
- `crs`: always `EPSG:4326` for the MVP
- `cell_size_deg`: `0.01`
- `origin_lat`, `origin_lon`: lower/left cell edges (southern/western edges), snapped down to the cell grid
- `n_lat`, `n_lon`: cell counts north–south / east–west

**Construction from bbox** `(lat_min, lat_max, lon_min, lon_max)`:
- `origin_lat = floor(lat_min / cell) * cell`
- `origin_lon = floor(lon_min / cell) * cell`
- `n_lat = ceil((lat_max - origin_lat) / cell)`
- `n_lon = ceil((lon_max - origin_lon) / cell)`

**Derived helpers**:
- Cell centers: `lat[i] = origin_lat + (i + 0.5) * cell`; `lon[j] = origin_lon + (j + 0.5) * cell`
- Index lookup: `i = floor((lat - origin_lat) / cell)`; `j = floor((lon - origin_lon) / cell)`
- Bounds: `(min_lon, min_lat, max_lon, max_lat) = (origin_lon, origin_lat, origin_lon + n_lon * cell, origin_lat + n_lat * cell)`

---

## 2. Grid Definition & Indexing

### Indexing Convention (Analysis Order)

- Indices are `(i, j) = (lat_index, lon_index)` (0-based)
- **Direction**:
  - `i` increases **south → north** (latitude increasing)
  - `j` increases **west → east** (longitude increasing)

**Index formula**:
- \(i = \lfloor (lat - origin\_lat) / cell\_size \rfloor\)
- \(j = \lfloor (lon - origin\_lon) / cell\_size \rfloor\)

This matches `api.core.grid.latlon_to_index` and is used by `api.fires.grid_mapping.fires_to_indices`.

### Origin & Cell Centers

- `origin_lat` and `origin_lon` are the **southern / western cell edges** (not cell centers)
- **Cell centers**:
  - `lat[i] = origin_lat + (i + 0.5) * cell_size_deg`
  - `lon[j] = origin_lon + (j + 0.5) * cell_size_deg`

### Bounds & Boundary Rules

The grid extent is **half-open** in both dimensions:
- Valid indices: `0 <= i < n_lat` and `0 <= j < n_lon`
- Points **outside** the region grid are **dropped by default**
- Points exactly on the **upper** boundary are out-of-bounds:
  - `lat == origin_lat + n_lat * cell_size` → out-of-bounds
  - `lon == origin_lon + n_lon * cell_size` → out-of-bounds

### Longitude Normalization

Before indexing, longitudes are normalized to `[-180, 180]`:
- `normalize_lon(lon)` wraps values like `190 → -170`

**Note**: The DB schema constrains `fire_detections.lon` to `[-180, 180]`, so normalization is usually a no-op, but we keep it for robustness.

---

## 3. Windowing (Half-Open Slices)

Most APIs use half-open windows `(i0:i1, j0:j1)`:
- `i0`, `j0` are inclusive
- `i1`, `j1` are exclusive
- `len(lat) == i1 - i0`, `len(lon) == j1 - j0`

**Use**:
- `api.core.grid.get_grid_window_for_bbox(...)` (bbox → indices + coords)
- `api.core.grid.window_coords(...)` (indices → cell-center coords)

---

## 4. Fire Detection Mapping

### Fire Detections → Grid Indices

Fire detections (lon/lat points) are mapped onto the grid for consistent 2D heatmaps:

```python
from api.core.grid import GridSpec, latlon_to_index
import numpy as np

grid = GridSpec.from_bbox(lat_min=35.0, lat_max=36.0, lon_min=5.0, lon_max=6.0)
heat = np.zeros((grid.n_lat, grid.n_lon), dtype=np.float32)

# detections: arrays of lat/lon (EPSG:4326)
i, j = latlon_to_index(grid, lat=detections_lat, lon=detections_lon)
in_bounds = (0 <= i) & (i < grid.n_lat) & (0 <= j) & (j < grid.n_lon)
np.add.at(heat, (i[in_bounds], j[in_bounds]), 1.0)
```

### Heatmap Array Convention

Aggregations produce dense arrays with shape:
- `(n_lat, n_lon)` for full-grid heatmaps
- `(window_height, window_width)` for AOI-window heatmaps

In both cases arrays are in **analysis order** (lat increasing south→north).

---

## 5. Terrain Alignment Contract

### Requirements

All terrain rasters for a region must be on the **exact same** grid:
- **CRS equals** the grid CRS (`EPSG:4326`)
- **Pixel size equals** `grid.cell_size_deg` (square pixels)
- **Width/height equal** `grid.n_lon` / `grid.n_lat`
- **Bounds match** the grid edges:
  - left == `grid.origin_lon`
  - bottom == `grid.origin_lat`
  - right == `grid.origin_lon + grid.n_lon * cell`
  - top == `grid.origin_lat + grid.n_lat * cell`

### Runtime Enforcement

`api/terrain/validate.py`:
- `validate_raster_matches_grid(...)`
- `validate_terrain_stack(...)`

### GeoTIFF Row Order vs Analysis Convention (Important!)

**GeoTIFFs are usually north-up**:
- Raster row `0` is the **northmost** pixels
- Row index increases **southward**

**But analysis arrays in this repo are lat-ascending**:
- `lat` is monotonic increasing (south → north)
- Array index `i` increases (south → north)

**Therefore**:
- When reading via rasterio windows (see `api/terrain/window.py`), we convert analysis indices to raster row offsets and **flip rows once** (`np.flipud`) so consumers always get `(lat, lon)` arrays with **lat increasing**
- When writing synthetic test rasters, you typically need to **flip on write** to store as north-up

### File Locations & Naming

**On-disk paths** (typical):
- **DEM**: `data/dem/<region>/dem_<region>_epsg4326_0p01deg.tif`
- **Slope**: `data/terrain/<region>/slope_<region>_epsg4326_0p01deg.tif`
- **Aspect**: `data/terrain/<region>/aspect_<region>_epsg4326_0p01deg.tif`

**Database metadata**:
- `terrain_metadata` stores DEM path + grid fields (origin, size, cell size, bbox)
- `terrain_features_metadata` stores slope/aspect paths + grid fields and nodata/conventions

### Usage Recipe: Load Terrain Window

```python
from api.terrain.window import load_terrain_window

tw = load_terrain_window("smoke_grid", bbox, include_dem=True)
# tw.slope, tw.aspect, tw.elevation are numpy arrays shaped (n_lat, n_lon) in (lat, lon) order
```

---

## 6. Pipeline Alignment

All data sources must align to the canonical grid:

### DEM (`ingest/dem_preprocess.py`)
- Mosaics Copernicus GLO-30
- Resamples onto the `GridSpec` (EPSG:4326, 0.01°)
- Writes `dem_{region}_epsg4326_0p01deg.tif` (+ optional COG)
- `terrain_metadata` stores grid fields so the grid can be reconstructed

### Weather (`ingest/weather_ingest.py`)
- Downloads GFS 0.25° GRIB
- Normalizes lon/lat, crops to the snapped grid bbox
- Interpolates to the same `GridSpec` via `xarray.interp`
- Saves NetCDF on the analysis grid with grid attributes (`crs`, `cell_size_deg`, `origin_lat`, `origin_lon`, `n_lat`, `n_lon`)

### Spread/Risk Outputs
- Must assume the canonical analysis grid (EPSG:4326, 0.01°)
- Any new rasters should either originate on this grid or be resampled into it using `GridSpec` helpers

### Raster Export Note (GeoTIFF)

GeoTIFF rasters are commonly stored "north-up", where row 0 is the **northernmost** pixels. Do not silently flip ML arrays.

- Keep modeling arrays in **analysis order**
- Only flip / adjust transforms when explicitly exporting rasters for visualization

---

## 7. Common Pitfalls

Things that silently break alignment:

- **North-up vs analysis order**: Reading a GeoTIFF directly as an array without flipping/coord sorting can invert latitude
- **Lon normalization**: Standardize longitudes to `[-180, 180]` before indexing if your source uses `[0, 360)`
- **Boundary semantics**: `max_lat` / `max_lon` edges are out-of-bounds by design (half-open extent)
- **"Looks aligned" but isn't**: Rasters can have the same CRS/resolution but differ by a half-cell origin shift. Use `validate_terrain_stack(...)`

---

## 8. Usage Recipes

### Get a Grid Window for a Bbox / AOI

```python
from api.core.grid import GridSpec, get_grid_window_for_bbox

grid = GridSpec.from_bbox(lat_min=35.0, lat_max=36.0, lon_min=5.0, lon_max=6.0)  # snapped
bbox = (5.12, 35.45, 5.99, 36.00)  # (min_lon, min_lat, max_lon, max_lat)
win = get_grid_window_for_bbox(grid, bbox, clip=True)
# win.i0, win.i1, win.j0, win.j1 are half-open indices
```

### Map Fire Detections → (i, j) and Build a Heatmap

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
