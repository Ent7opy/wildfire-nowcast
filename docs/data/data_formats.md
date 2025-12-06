# Data Formats (Fire, Weather, DEM)

Canonical shapes for core datasets so API, ML, and UI code can rely on the same field names and layouts.

## Fire detections
- **Tables**: `fire_detections` + `ingest_batches` (see `data_schema_fires.md` for full detail).
- **Key columns**: `id`, `geom` (Point EPSG:4326), `lat`, `lon`, `acq_time` (UTC), `sensor`, `source`, `confidence`, `brightness`, `bright_t31`, `frp`, `scan`, `track`, `raw_properties` (JSONB), `dedupe_hash`, `denoised_score`, `is_noise`, `ingest_batch_id`, `created_at`.
- **Deduplication**: `(source, dedupe_hash)` where `dedupe_hash` is a rounded lat/lon + timestamp hash (`ingest.models.compute_dedupe_hash`).
- **Standard Python shape (API/ML)**: `ingest.models.FireDetection` (Pydantic) and `ingest.models.DetectionRecord` for inserts. Both expose the same field names as the DB schema.

Example: lifting an ingest record into the shared API/ML shape
```python
from ingest.models import DetectionRecord, FireDetection

record = DetectionRecord(... )  # produced by parse_detection_rows
dto = FireDetection.from_record(record)
payload = dto.model_dump()
```

## Weather forecast grids
- **Source**: NOAA GFS 0.25° via `ingest.weather_ingest`.
- **Canonical variables**: `u10`, `v10`, `t2m`, `rh2m`; optional `tp` (precipitation) when `WEATHER_INCLUDE_PRECIP=true` or `--include-precip`.
- **Dimensions/coords**: `time`, `lat`, `lon`; plus coordinates `forecast_reference_time` (scalar) and `lead_time_hours` (aligned with `time`). Dataset is transposed to `("time", "lat", "lon")`, lat is sorted ascending, lon is wrapped to `[-180, 180]` when needed.
- **Storage layout**: `data/weather/{model}/{YYYY}/{MM}/{DD}/{HH}/{model}_{YYYYMMDDTHHZ}_0-{horizon}h_<region>.nc` where `<region>` is `global` or `bbox_<min_lon>_<min_lat>_<max_lon>_<max_lat>`.

Example: opening a run and selecting a slice
```python
import xarray as xr

ds = xr.open_dataset("data/weather/gfs_0p25/2025/12/06/06/gfs_0p25_20251206T06Z_0-24h_bbox_5.0_35.0_20.0_47.0.nc")
ds["u10"].sel(time=ds.time[0], lat=40.0, lon=10.0, method="nearest")
```

## DEM (terrain)
- **Source**: Copernicus GLO-30 stitched via `ingest.dem_preprocess`.
- **Target CRS/resolution**: default EPSG:4326 and 1000 m unless overridden; stored as GeoTIFF (optionally COG).
- **File pattern**: `data/dem/{region}/dem_{region}_epsg{crs}_{resolution_m}m.tif` (or `_cog.tif` when `--cog` is used).
- **Metadata table**: `terrain_metadata` columns `id, region_name, dem_source, crs_epsg, resolution_m, bbox (POLYGON EPSG:4326), raster_path, created_at`.
- **Consumer dims**: raster is opened with `x`/`y` coordinates (renamed from `lon`/`lat` when needed) in `api.terrain.dem_loader`.

Example: clip the latest DEM for a region
```python
from api.terrain.dem_loader import load_dem_for_bbox

dem = load_dem_for_bbox("test_region", bbox=(5.0, 35.0, 20.0, 47.0))
elev = dem.sel(x=10.0, y=40.0, method="nearest").item()
```

## Ingestion alignment checklist
- FIRMS: `ingest.firms_client.parse_detection_rows` → `DetectionRecord` → `insert_detections` writes directly to the standardized columns (including `dedupe_hash`). See `../ingest/ingest_firms.md` for configuration, validation, and run commands.
- Weather: `ingest.weather_ingest.build_weather_dataset` enforces variable names (`u10`, `v10`, `t2m`, `rh2m`, `tp`), coordinate names (`time`, `lat`, `lon`, `forecast_reference_time`, `lead_time_hours`), and NetCDF layout under `data/weather/`.
- DEM: `ingest.dem_preprocess` outputs GeoTIFF/COG using the filename pattern above and stores bbox/CRS/resolution in `terrain_metadata`; `api.terrain.dem_loader` exposes clipped `x`/`y` rasters for downstream use.


