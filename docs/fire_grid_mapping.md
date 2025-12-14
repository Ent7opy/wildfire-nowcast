## Fire detections → analysis grid mapping

This project maps fire detections (lon/lat points) onto a stable **region analysis grid** so downstream ML and UI can build consistent 2D heatmaps.

### Canonical grid

- **Grid**: region `GridSpec` (stable origin/extent)
- **CRS**: EPSG:4326
- **Cell size**: typically `0.01°` (but always use the region’s stored `GridSpec.cell_size_deg`)

`GridSpec` fields:
- **origin_lat / origin_lon**: the **southern/western cell edges**
- **n_lat / n_lon**: grid shape

### Indexing convention (analysis order)

- Indices are `(i, j) = (lat_index, lon_index)`
- **i increases south → north** (latitude increasing)
- **j increases west → east** (longitude increasing)

Index formula:

- \(i = \lfloor (lat - origin\_lat) / cell\_size \rfloor\)
- \(j = \lfloor (lon - origin\_lon) / cell\_size \rfloor\)

This matches `api.core.grid.latlon_to_index` and is used by `api.fires.grid_mapping.fires_to_indices`.

### Longitude normalization

Before indexing, longitudes are normalized to `[-180, 180]`:

- `normalize_lon(lon)` wraps values like `190 → -170`

Note: the DB schema constrains `fire_detections.lon` to `[-180, 180]`, so normalization is usually a no-op, but we keep it for robustness.

### Bounds + boundary rule

The grid extent is **half-open** in both dimensions:

- Valid indices: `0 <= i < n_lat` and `0 <= j < n_lon`
- Points **outside** the region grid are **dropped by default**
- Points exactly on the **upper** boundary are out-of-bounds and dropped:
  - `lat == origin_lat + n_lat * cell_size` → out-of-bounds
  - `lon == origin_lon + n_lon * cell_size` → out-of-bounds

### Heatmap array convention

Aggregations produce dense arrays with shape:

- `(n_lat, n_lon)` for full-grid heatmaps
- `(window_height, window_width)` for AOI-window heatmaps

In both cases arrays are in **analysis order** (lat increasing south→north).

### Raster export note (GeoTIFF)

GeoTIFF rasters are commonly stored “north-up”, where row 0 is the **northernmost** pixels. Do not silently flip ML arrays.

- Keep modeling arrays in **analysis order**.
- Only flip / adjust transforms when explicitly exporting rasters for visualization.
