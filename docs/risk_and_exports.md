# Risk Index & Export Features

## Overview

The Wildfire Nowcast & Forecast platform provides a fire risk index heatmap and comprehensive export capabilities for analysts and reports.

## Fire Risk Index

### What is the Risk Index?

The fire risk index is a composite score (0–1) that estimates the likelihood of new fire ignitions in a given area based on static and dynamic factors.

### How is it Computed?

The risk score combines:

1. **Static factors (60% weight)**: Land cover-based flammability
   - Forest, shrubland, grassland: high risk (score ≈ 1.0)
   - Cropland: moderate risk (score ≈ 0.7)
   - Urban, water, desert, ice: low risk (score ≈ 0.1)

2. **Dynamic factors (40% weight)**: Weather conditions
   - **Increases risk**: Low humidity (<40%), moderate-to-high wind (>3 m/s)
   - **Decreases risk**: High humidity (>70%), recent precipitation (>10mm in 72h)

### Risk Levels

Risk scores are classified into three levels:

- **Low (0.0–0.3)**: Minimal fire risk — shown in green
- **Medium (0.3–0.6)**: Moderate fire risk — shown in yellow
- **High (0.6–1.0)**: Elevated fire risk — shown in red

### API Endpoint

**GET /risk**

Returns a GeoJSON FeatureCollection with grid cells covering the requested bounding box.

**Parameters:**
- `min_lon`, `min_lat`, `max_lon`, `max_lat` (required): Bounding box coordinates
- `cell_size_km` (optional, default 10.0): Grid cell size in kilometers (range: 1–100)
- `time` (optional): Reference time for weather data (ISO 8601 format, defaults to now)
- `include_weather` (optional, default true): Whether to include dynamic weather factors

**Example:**
```bash
curl "http://localhost:8000/risk?min_lon=20.0&min_lat=40.0&max_lon=22.0&max_lat=42.0&cell_size_km=10"
```

**Response:**
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[20.0, 40.0], [20.1, 40.0], [20.1, 40.1], [20.0, 40.1], [20.0, 40.0]]]
      },
      "properties": {
        "risk_score": 0.65,
        "risk_level": "high",
        "components": {
          "static": 0.8,
          "dynamic": 0.4
        }
      }
    }
  ]
}
```

### Visualization in UI

The risk layer can be toggled in the sidebar. It renders as a heatmap with color-coded cells:
- Green cells: low risk
- Yellow cells: medium risk
- Red cells: high risk

Hover over a cell to see the detailed risk score and components.

## Export Features

### CSV/GeoJSON Exports

#### Fire Detections Export

**GET /fires/export**

Export fire detections for a bounding box and time range.

**Parameters:**
- `min_lon`, `min_lat`, `max_lon`, `max_lat` (required): Bounding box
- `start_time`, `end_time` (required): Time range (ISO 8601)
- `format` (required): `csv` or `geojson`
- `limit` (optional, default 1000, max 10000): Maximum number of fires

**Example:**
```bash
# CSV export
curl "http://localhost:8000/fires/export?min_lon=20&min_lat=40&max_lon=22&max_lat=42&start_time=2026-01-01T00:00:00Z&end_time=2026-01-02T00:00:00Z&format=csv" -o fires.csv

# GeoJSON export
curl "http://localhost:8000/fires/export?min_lon=20&min_lat=40&max_lon=22&max_lat=42&start_time=2026-01-01T00:00:00Z&end_time=2026-01-02T00:00:00Z&format=geojson" -o fires.geojson
```

#### Risk Grid Export

**GET /risk/export**

Export the fire risk grid for a bounding box.

**Parameters:**
- `min_lon`, `min_lat`, `max_lon`, `max_lat` (required): Bounding box
- `format` (required): `csv` or `geojson`
- `cell_size_km` (optional, default 10.0): Grid cell size
- `time` (optional): Reference time for weather
- `include_weather` (optional, default true): Include weather factors

**Example:**
```bash
# CSV export (contains center_lon, center_lat, risk_score, risk_level, component scores)
curl "http://localhost:8000/risk/export?min_lon=20&min_lat=40&max_lon=22&max_lat=42&format=csv" -o risk.csv

# GeoJSON export
curl "http://localhost:8000/risk/export?min_lon=20&min_lat=40&max_lon=22&max_lat=42&format=geojson" -o risk.geojson
```

#### Forecast Contours Export

**GET /forecast/{run_id}/contours/export**

Export forecast contours for a specific forecast run.

**Parameters:**
- `run_id` (required, in path): Forecast run ID
- `format` (optional, default `geojson`): Export format

**Example:**
```bash
curl "http://localhost:8000/forecast/123/contours/export?format=geojson" -o forecast_contours.geojson
```

### PNG Map Export

**GET /exports/map.png**

Export a static map image (PNG) with selected layers.

**Parameters:**
- `min_lon`, `min_lat`, `max_lon`, `max_lat` (required): Bounding box
- `start_time`, `end_time` (optional): Time range for fires
- `run_id` (optional): Forecast run ID
- `include_fires` (optional, default true): Include fire detections
- `include_risk` (optional, default true): Include risk heatmap
- `include_forecast` (optional, default false): Include forecast contours
- `width` (optional, default 1600, range 400–4000): Image width in pixels
- `height` (optional, default 900, range 300–3000): Image height in pixels

**Example:**
```bash
curl "http://localhost:8000/exports/map.png?min_lon=20&min_lat=40&max_lon=22&max_lat=42&start_time=2026-01-01T00:00:00Z&end_time=2026-01-02T00:00:00Z&include_risk=true&width=1600&height=900" -o map.png
```

**Features:**
- Rendered map with graticule (lat/lon grid lines)
- Color-coded risk heatmap (if enabled)
- Fire points (if enabled)
- Forecast contours (if enabled)
- Legend showing active layers
- Timestamp of generation

**Requirements:**
- Server must have Pillow (PIL) installed: `pip install pillow`

### UI Export Controls

Both CSV and PNG exports are available in the UI sidebar:

1. **Export fires (CSV)**: Downloads fire detections for the current viewport and time window as CSV
2. **Export map (PNG)**: Downloads a static map image with currently visible layers

The PNG export automatically includes:
- Current viewport bounding box
- Current time window for fires
- Active layers (fires, risk, forecast)
- Forecast run ID (if forecast layer is active)

## Use Cases

### For Analysts

- **CSV exports**: Import fire data into spreadsheets or GIS tools for custom analysis
- **GeoJSON exports**: Load into QGIS, ArcGIS, or other geospatial software
- **Risk grid CSV**: Analyze risk patterns across regions, correlate with other datasets

### For Reports & Sharing

- **PNG exports**: Include map snapshots in reports, presentations, or documentation
- **Shareable artifacts**: Generate static images for stakeholders without API access
- **Timestamped outputs**: All exports include generation time for provenance

### For Integration

- **API-first design**: All exports accessible via REST API for automation
- **Programmatic access**: Use exports in scripts, notebooks, or downstream pipelines
- **Format flexibility**: Choose CSV for tabular analysis or GeoJSON for geospatial workflows

## Limitations & Notes

### Risk Index

- **Weather data availability**: Dynamic risk factors require weather data ingestion. If unavailable, risk is computed using static (land cover) factors only.
- **Grid resolution**: Large bboxes with small cell sizes may be capped at 500 cells to prevent oversized responses.
- **Static land cover**: Based on ESA WorldCover or similar. If land cover data is not available, a neutral score (0.5) is used.

### PNG Export

- **Simplified rendering**: MVP uses basic 2D rendering. Forecast contours are not fully rendered in the current implementation.
- **Performance**: Large images (high resolution) or complex data may take longer to generate.
- **Dependency**: Requires Pillow library on the server.

### CSV/GeoJSON Exports

- **Sync export limits**: Fires export is capped at 10,000 features for sync endpoints. For larger exports, use async export jobs (future enhancement).
- **Memory**: Very large exports may consume significant memory. Consider filtering by time/bbox to reduce size.

## Future Enhancements

- **Async export jobs**: For large datasets (>10k features), queue exports and download when ready
- **MBTiles export**: For offline map viewing
- **Shapefile export**: For legacy GIS compatibility
- **Enhanced PNG rendering**: Full forecast contour rendering, basemap integration
- **Risk raster tiles**: Serve risk as raster tiles (COG + TileJSON) for performance

---

For more information, see:
- [API Documentation](../api/) (OpenAPI/Swagger UI at `/docs`)
- [Getting Started Guide](GETTING_STARTED.md)
- [Architecture Overview](architecture.md)
