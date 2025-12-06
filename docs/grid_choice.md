# Analysis Grid Choice

- **CRS**: `EPSG:4326` (WGS84 geographic). Matches FIRMS, GFS, and Copernicus GLO-30; avoids extra reprojections in the ingest path. Distortion from non–equal-area is acceptable for the ~1 km MVP; UI can reproject to `EPSG:3857` separately.
- **Resolution**: fixed `0.01°` in both lat/lon (≈1.11 km at the equator, ≈0.9–1.0 km in mid-latitudes). Inputs map as: DEM is downsampled from 30 m; GFS is interpolated from 0.25°; FIRMS points land on this grid without reprojection.
- **Size example**: a 10° × 10° region becomes a 1000 × 1000 grid (~1M cells).

## GridSpec parameterization

Canonical dataclass (in `api/core/grid.py`):
- `crs`: always `EPSG:4326` for the MVP.
- `cell_size_deg`: `0.01`.
- `origin_lat`, `origin_lon`: lower/left cell edges (southern/western edges), snapped down to the cell grid.
- `n_lat`, `n_lon`: cell counts north–south / east–west.

Construction from bbox `(lat_min, lat_max, lon_min, lon_max)`:
- `origin_lat = floor(lat_min / cell) * cell`
- `origin_lon = floor(lon_min / cell) * cell`
- `n_lat = ceil((lat_max - origin_lat) / cell)`
- `n_lon = ceil((lon_max - origin_lon) / cell)`

Derived helpers:
- Cell centers: `lat[i] = origin_lat + (i + 0.5) * cell`; `lon[j] = origin_lon + (j + 0.5) * cell`.
- Index lookup: `i = floor((lat - origin_lat) / cell)`; `j = floor((lon - origin_lon) / cell)`.
- Bounds: `(min_lon, min_lat, max_lon, max_lat) = (origin_lon, origin_lat, origin_lon + n_lon * cell, origin_lat + n_lat * cell)`.

## Pipeline alignment

- **DEM** (`ingest/dem_preprocess.py`): mosaics Copernicus GLO-30, resamples onto the `GridSpec` (EPSG:4326, 0.01°), and writes `dem_{region}_epsg4326_0p01deg.tif` (+ optional COG). `terrain_metadata` stores `crs_epsg`, `cell_size_deg`, `origin_lat`, `origin_lon`, `grid_n_lat`, `grid_n_lon`, and `bbox` so the grid can be reconstructed.
- **Weather** (`ingest/weather_ingest.py`): downloads GFS 0.25° GRIB, normalizes lon/lat, crops to the snapped grid bbox, interpolates to the same `GridSpec` via `xarray.interp`, saves NetCDF on the analysis grid, and writes grid attributes (`crs`, `cell_size_deg`, `origin_lat`, `origin_lon`, `n_lat`, `n_lon`).
- **Spread/risk outputs**: must assume the canonical analysis grid (EPSG:4326, 0.01°). Any new rasters should either originate on this grid or be resampled into it using `GridSpec` helpers.

