# Weather Ingestion Optimization for JIT Forecasts

This document describes approaches to optimize GFS weather data ingestion for small area-of-interest (AOI) Just-In-Time (JIT) forecasts, specifically targeting <5s retrieval for 10km x 10km patches.

## Context

The wildfire nowcast JIT pipeline needs to ingest weather data on-demand when a user clicks a fire anywhere in the world. Pre-downloading full regional GFS runs (typically 100+ MB for a region) is too slow for interactive response. The goal is to fetch only the necessary data for a small AOI.

## GFS Data Sources

NOAA GFS 0.25° forecasts are available via:

1. **NOMADS filter API** (https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl)
   - Server-side spatial subsetting (bbox parameters)
   - Server-side variable subsetting
   - Returns filtered GRIB2 file
   - Already in use by `ingest/weather_ingest.py`

2. **Direct GRIB2 files** (AWS Open Data: https://noaa-gfs-bdp-pds.s3.amazonaws.com)
   - Full global GRIB2 files (~500 MB per forecast hour)
   - `.idx` sidecar files list byte offsets for each GRIB message
   - Supports HTTP Range requests for message-level extraction

## Approach 1: NOMADS Filter API (RECOMMENDED)

### How it works

The NOMADS filter API accepts CGI parameters for spatial and variable subsetting:

```python
base_url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
params = {
    "dir": "/gfs.20260119/00/atmos",
    "file": "gfs.t00z.pgrb2.0p25.f000",
    "leftlon": 23.7,
    "rightlon": 23.8,
    "toplat": 38.0,
    "bottomlat": 37.9,
    "var_UGRD": "on",
    "var_VGRD": "on",
    "var_TMP": "on",
    "lev_10_m_above_ground": "on",
    "lev_2_m_above_ground": "on",
}
url = f"{base_url}?{urlencode(params)}"
response = httpx.get(url)
# Returns spatially-subsetted GRIB2 (~10-50 KB for 10km x 10km)
```

### Advantages

- **Spatial subsetting on server side**: Only data within bbox is returned
- **Small downloads**: 10-50 KB for 10km x 10km patches (vs 500+ MB full file)
- **Fast**: Typically 1-3 seconds for small AOIs
- **Already implemented**: Used in `ingest/weather_ingest.py` since initial implementation

### Limitations

- **NOMADS availability**: Filter API has rate limits and occasional downtime
- **Fallback required**: Need AWS Open Data or NCEI as backup sources
- **Resolution**: Limited to published GFS resolution (0.25°)

### Benchmark

See `scripts/gfs_partial_download_poc.py` for benchmark script.

Run with:
```bash
uv run --project ingest python scripts/gfs_partial_download_poc.py
```

Expected performance for 10km x 10km patch (u10, v10, t2m):
- Download size: ~15-30 KB
- Latency: 1-3 seconds (typical)
- Variables: All requested variables in single file

## Approach 2: HTTP Range Requests with .idx Sidecar

### How it works

1. Download `.idx` file (~10-50 KB) that lists byte offsets for each GRIB message
2. Parse `.idx` to find target variables (e.g., UGRD:10 m above ground)
3. Use HTTP `Range` header to fetch only those GRIB messages

```python
# Fetch .idx file
idx_url = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20260119/00/atmos/gfs.t00z.pgrb2.0p25.f000.idx"
idx_content = httpx.get(idx_url).text

# Parse to find UGRD byte range (example: bytes 123456-234567)
# Use Range header to fetch only that message
headers = {"Range": "bytes=123456-234567"}
grib_url = idx_url.replace(".idx", "")
message = httpx.get(grib_url, headers=headers)
```

### Advantages

- **Variable subsetting**: Fetch only needed variables (skip precipitation, etc.)
- **AWS availability**: AWS Open Data is highly reliable (S3)
- **Deterministic**: No server-side processing variability

### Limitations

- **No spatial subsetting**: Each message contains the full global grid (~50-100 KB per variable)
- **Larger downloads**: For 10km x 10km, downloading full global grid is wasteful
- **Post-processing required**: Must spatially subset after download (adds latency)

### Benchmark

Expected performance for 10km x 10km patch (u10, v10, t2m):
- Download size: ~150-300 KB (3 full global messages)
- Latency: 2-5 seconds
- Post-subset overhead: +0.5-1s for spatial trimming

**Conclusion**: Not optimal for small AOIs due to lack of spatial subsetting.

## Approach 3: Aggressive Post-Download Subsetting

If NOMADS filter is unavailable (downtime, rate limit), fall back to:

1. Download full GRIB2 via AWS Open Data (with .idx Range requests for variable filtering)
2. Load with `cfgrib` (xarray)
3. Immediately subset to bbox + margin (e.g., bbox + 0.5° padding)
4. Discard unused data before interpolation

### Implementation

This is already implemented in `ingest/weather_ingest.py` via fallback URL logic:

```python
base_urls = [
    "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl",  # Primary (filter)
    "https://noaa-gfs-bdp-pds.s3.amazonaws.com",                 # Fallback (direct)
]
```

The fallback uses direct S3 URLs (no filter), resulting in larger downloads but guaranteed availability.

## Caching Strategy

For repeated requests in the same region:

1. **Terrain**: Cache indefinitely (terrain doesn't change)
   - Check `terrain_features` table for overlapping bbox before ingesting
   - Reuse existing terrain_features_id

2. **Weather**: Cache for 6 hours (GFS runs every 6h)
   - Check `weather_runs` table for overlapping bbox + recent run_time
   - Reuse existing weather_run_id if within freshness window

3. **Spatial Index**: Use PostGIS `ST_Intersects` on bbox geometry column
   - Already indexed via GIST in `weather_runs` and `terrain_features` tables

See task T18 in plan for caching implementation details.

## Recommendations for JIT Pipeline (T14)

When implementing "patch mode" in T14:

1. **Keep NOMADS filter as default**: Already optimal for small AOIs
2. **Add aggressive spatial subsetting in fallback path**: After loading GRIB, subset to bbox + margin before regridding
3. **Skip unnecessary variables**: If model doesn't need precipitation, don't request it
4. **Target <10s total ingestion time**: Download (1-3s) + regrid (2-4s) + persist (1-2s)

### Patch Mode Flag

```python
def ingest_weather_for_bbox(
    bbox: tuple[float, float, float, float],
    forecast_time: datetime,
    output_dir: Path,
    patch_mode: bool = False,
) -> int:
    """
    Ingest weather for arbitrary bbox.

    Args:
        bbox: (lon_min, lat_min, lon_max, lat_max)
        forecast_time: GFS run time (UTC)
        output_dir: Where to save NetCDF
        patch_mode: If True, optimize for small AOI (<50km x 50km):
            - Skip precipitation variable
            - Use minimal forecast horizon (0-24h instead of 0-72h)
            - Aggressive spatial subsetting in fallback path

    Returns:
        weather_run_id from database
    """
```

## Performance Targets

| Scenario | Download Method | AOI Size | Expected Time |
|----------|----------------|----------|---------------|
| Small JIT patch | NOMADS filter | 10km x 10km | 1-3s |
| Medium JIT patch | NOMADS filter | 50km x 50km | 2-5s |
| Large region | NOMADS filter | 500km x 500km | 10-20s |
| Fallback (AWS) | Direct + subset | 10km x 10km | 5-10s |

## References

- NOMADS filter documentation: https://nomads.ncep.noaa.gov/
- AWS Open Data GFS: https://registry.opendata.aws/noaa-gfs-bdp-pds/
- GRIB2 .idx format: https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/
- POC script: `scripts/gfs_partial_download_poc.py`

## Related Tasks

- T13: Research GFS partial downloads (this document)
- T14: Implement patch mode in weather_ingest.py
- T18: Add caching to prevent redundant ingestion
