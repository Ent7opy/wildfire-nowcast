## Land-cover data for fire plausibility scoring

### What it does
- Provides land-cover classification data to score fire detection plausibility.
- Used by `api.fires.landcover.compute_landcover_scores()` to penalize detections in urban areas, water bodies, and deserts.
- Supports ESA WorldCover 10m or similar land-cover classification rasters.

### Data source
**Recommended: ESA WorldCover 10m (2021)**
- Global land-cover map at 10m resolution based on Sentinel-1 and Sentinel-2 data.
- Download tiles from: https://esa-worldcover.org/en/data-access
- Classification scheme:
  - 10: Tree cover / forest
  - 20: Shrubland
  - 30: Grassland
  - 40: Cropland
  - 50: Built-up / urban
  - 60: Bare / sparse vegetation / desert
  - 70: Snow and ice
  - 80: Permanent water bodies
  - 90: Herbaceous wetland
  - 95: Mangroves
  - 100: Moss and lichen

**Alternative: MODIS Land Cover (MCD12Q1)**
- Global land-cover at 500m resolution (coarser, but smaller file size).
- Download from: https://lpdaac.usgs.gov/products/mcd12q1v061/
- Requires remapping MODIS classes to the scoring scheme in `api/fires/landcover.py`.

### Setup instructions

#### 1. Download land-cover data for your region
For ESA WorldCover:
- Navigate to https://esa-worldcover.org/en/data-access
- Select tiles covering your area of interest (AOI)
- Download GeoTIFF tiles

#### 2. Merge and reproject (if needed)
If you downloaded multiple tiles or need to reproject:
```bash
# Merge multiple tiles
gdal_merge.py -o landcover_merged.tif tile1.tif tile2.tif tile3.tif

# Reproject to EPSG:4326 (if not already)
gdalwarp -t_srs EPSG:4326 -tr 0.0001 0.0001 -r near \
  landcover_merged.tif landcover_epsg4326.tif
```

#### 3. Place file in the expected location
```bash
cp landcover_epsg4326.tif data/landcover.tif
```

Expected file location: `data/landcover.tif`

### File format requirements
- **CRS:** EPSG:4326 (WGS84 lat/lon)
- **Pixel values:** Integer land-cover class codes (10, 20, 30, etc.)
- **Data type:** Byte or UInt8
- **Format:** GeoTIFF

### Scoring rules
See `api/fires/landcover.py` for the current mapping:
- **High fire plausibility (1.0):** Forest, shrubland, grassland
- **Moderate plausibility (0.7):** Cropland
- **Low plausibility (0.1):** Urban, water, desert, ice

### Fallback behavior
If `data/landcover.tif` is not present:
- `compute_landcover_scores()` returns a neutral score of **0.5** for all detections.
- This allows the fire detection pipeline to run without land-cover data, but with reduced filtering accuracy.

### Testing
After placing the land-cover file:
```bash
cd api
uv run pytest tests/test_fire_landcover_scoring.py -v
```

### Example usage
```python
from api.fires.landcover import compute_landcover_scores

detections = [
    {"id": 1, "lat": 42.5, "lon": 23.0},   # Forest area (Balkans)
    {"id": 2, "lat": 40.7, "lon": -74.0},  # Urban area (NYC)
]

scores = compute_landcover_scores(detections)
# Expected: scores[1] ≈ 1.0 (forest), scores[2] ≈ 0.1 (urban)
```
