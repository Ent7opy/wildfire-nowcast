## DEM ingest (Copernicus GLO-30)

### What it does
- Stitches Copernicus GLO-30 tiles for a configured bbox.
- Reprojects/resamples to the canonical analysis grid (EPSG:4326, 0.01° lat/lon).
- Writes GeoTIFF (and optional Cloud Optimized GeoTIFF).
- Stores a `terrain_metadata` row with bbox, CRS, grid definition (cell size, origin, grid shape), and raster path.

### Configuration
- Environment variables (env-var overrides):
  - `DEM_DATA_DIR` (default: `data/dem`)
  - `DEM_REGION_NAME` (default: `test_region`)
  - `DEM_SOURCE` (default: `copernicus_glo30`)
  - `DEM_BBOX` (`min_lon,min_lat,max_lon,max_lat`) or individual `DEM_BBOX_*`
- CLI overrides (take precedence when provided):
  - `--bbox MIN_LON MIN_LAT MAX_LON MAX_LAT`
  - `--region-name NAME`
  - `--cog` (also write COG)

### How to run
- Via make (recommended):
  ```bash
  make ingest-dem ARGS="--cog"
  ```
- Direct:
  ```bash
  uv run --project ingest -m ingest.dem_preprocess --cog
  ```

### Outputs
- File: `data/dem/{region}/dem_{region}_epsg4326_0p01deg.tif` (and `_cog.tif` if selected)
- Metadata: row in `terrain_metadata` with region, source, bbox, CRS, grid origin, grid shape, cell size (deg), resolution_m (approx km), and raster path.

### Demo / sampling
- Example script: `examples/demo_dem_query.py`
  - Loads the latest DEM for the region
  - Clips to a sample bbox and prints elevations for sample points

