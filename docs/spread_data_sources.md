# Spread Model Data Sources

This document is the authoritative declaration of primary data sources for each spread model
input. All production inference and training must trace to one of the sources declared here.

Per the [Spread Maturity Policy](spread_maturity_policy.md), a **missing or undeclared source
is a hard stop** — it blocks gate report pass and model promotion.

---

## 1. Fire Detections

| Attribute | Value |
|-----------|-------|
| **Primary source** | NASA FIRMS — VIIRS SNPP NRT + VIIRS NOAA-20 NRT |
| **Provider** | NASA / EOSDIS |
| **API endpoint** | `https://firms.modaps.eosdis.nasa.gov/api/area/csv` |
| **Source IDs** | `VIIRS_SNPP_NRT`, `VIIRS_NOAA20_NRT` |
| **Native resolution** | ~375 m (VIIRS I-band) |
| **Analysis grid** | Resampled to 0.01° (~1 km) canonical grid |
| **Temporal latency** | Near-real-time, ~3 h after overpass (NRT stream) |
| **Ingestion** | `ingest/firms_ingest.py` — watermark-based incremental |
| **Denoiser gate** | Denoiser v2 scores applied inline; threshold profile controlled by `DENOISER_THRESHOLD_PROFILE` |

### Quality acceptance criteria

| Metric | Threshold |
|--------|-----------|
| Denoiser event recall | ≥ 0.92 |
| Denoiser event precision | ≥ 0.75 |
| Denoiser global F1 | ≥ 0.85 |
| Denoiser ROC-AUC | ≥ 0.95 |
| Sensor bias (SNPP vs NOAA-20) | ≤ 5 % |
| Min positive events in eval window | ≥ 50 |

### Stage-gap warnings

- **WARN-FIRE-001** (`mvp_operational → science_grade`): Cross-sensor inter-calibration between
  VIIRS SNPP and NOAA-20 is not explicitly validated in the MVP gate. Target: `science_grade`.
  Exit criteria: sensor-parity evaluation on a held-out season showing bias < 5 %.

---

## 2. Weather

| Attribute | Value |
|-----------|-------|
| **Primary source (global)** | NOAA Global Forecast System (GFS) 0.25° |
| **Primary source (CONUS)** | NOAA High-Resolution Rapid Refresh (HRRR) 3 km |
| **Provider** | NOAA / NWS |
| **GFS distribution** | NOAA NOMADS filter endpoint; AWS S3 `noaa-gfs-bdp-pds` |
| **HRRR distribution** | AWS S3 `noaa-hrrr-bdp-pds` |
| **Format** | GRIB2, decoded via cfgrib |
| **Native resolution** | 0.25° (GFS); 3 km (HRRR) |
| **Analysis grid** | Bilinearly interpolated to 0.01° canonical grid |
| **Temporal resolution** | 6-hourly cycles (GFS: 00/06/12/18 UTC); hourly (HRRR) |
| **Ingestion** | `ingest/weather_ingest.py` |
| **Bias correction** | Optional: `WeatherBiasCorrector` JSON artifact; path via `WEATHER_BIAS_CORRECTOR_PATH` |

### Variables ingested

| Model variable | Channel name | Units | Description |
|----------------|-------------|-------|-------------|
| `UGRD` 10 m | `u10` | m/s | Eastward wind component |
| `VGRD` 10 m | `v10` | m/s | Northward wind component |
| `TMP` 2 m | `t2m` | K | 2 m air temperature |
| `RH` 2 m | `rh2m` | % | 2 m relative humidity |
| `APCP` | `precip_24h` | mm | 24 h accumulated precipitation |
| Derived | `dfmc` | fraction | Dead fuel moisture (Nelson 1984 / NFDRS EMC; see §5) |

### Quality acceptance criteria

| Metric | Threshold |
|--------|-----------|
| Required variables present | `u10`, `v10` (hard) |
| GRIB file integrity | cfgrib parse must succeed |
| Run age at inference time | < 12 h (warn), < 24 h (fallback trigger) |
| Bias corrector applied | Recommended; absence logged as `WARNING` |

### Fallback behaviour

If no completed weather run is found for the AOI and reference time, the pipeline substitutes a
**calm-wind fallback cube** (zero wind, NaN temperature/humidity/DFMC) and sets
`weather_fallback_used=True` in forecast lineage attrs. Strict-inputs mode (`STRICT_FORECAST_INPUTS=1`)
converts this fallback to a hard error.

### Stage-gap warnings

- **WARN-WX-001** (`mvp_operational → science_grade`): Bias correction is optional in MVP. Target:
  `science_grade`. Exit criteria: bias corrector artifact validated and enforced for all promoted
  models.
- **WARN-WX-002** (`mvp_operational → science_grade`): HRRR preference is CONUS-only; no equivalent
  high-resolution source is configured for international domains. Target: `science_grade`.

---

## 3. Terrain

| Attribute | Value |
|-----------|-------|
| **Primary source** | DEM-derived rasters stored in `terrain_features_metadata` table |
| **Upstream DEM** | SRTM 30 m / NASADEM (to be declared per region at `science_grade`) |
| **Analysis grid** | Snapped to 0.01° canonical grid (EPSG:4326) |
| **Ingestion** | `ingest/terrain_features.py` → `api/terrain/window.py` |

### Derived channels

| Channel | Description | Computation |
|---------|-------------|-------------|
| `slope_deg` | Surface slope in degrees | Gradient magnitude of elevation via scipy |
| `aspect_sin` | sin(aspect angle) | Aspect direction (N=0°, E=90°) |
| `aspect_cos` | cos(aspect angle) | Same as above |
| `elevation_m` | Elevation above sea level | DEM cell value |
| `ruggedness` | Local terrain roughness | Gradient magnitude of elevation (proxy) |
| `tpi` | Topographic Position Index | elevation − grid mean elevation |

### Quality acceptance criteria

| Metric | Threshold |
|--------|-----------|
| Spatial coverage of AOI | 100 % (no masked cells in valid area) |
| Coordinate alignment with analysis grid | Tolerance ≤ 1e-12° |
| Slope range | [0°, 90°] |
| Elevation range | [-500 m, 9000 m] |

### Fallback behaviour

If terrain data is unavailable for a region, the pipeline substitutes **zero-filled terrain**
(slope=0, aspect=0, elevation=0) and sets `terrain_fallback_used=True`. This disables
terrain-based spread factors and is logged as `WARNING`.

### Stage-gap warnings

- **WARN-TERR-001** (`mvp_operational → science_grade`): The upstream DEM source is not explicitly
  declared per region in the current pipeline. Target: `science_grade`. Exit criteria: DEM provenance
  recorded per-region in `terrain_features_metadata` and validated against SRTM/NASADEM checksums.

---

## 4. Fuels

Fuel state enters the spread model through three distinct channels with different sources.

### 4a. Land Cover / NDVI proxy

| Attribute | Value |
|-----------|-------|
| **Primary source** | ESA WorldCover 10 m (v100, 2020; v200, 2021) |
| **Provider** | ESA / Copernicus Land Service |
| **Distribution** | AWS S3 `s3://esa-worldcover/v100/…` / `v200/…` |
| **Format** | GeoTIFF tiles (~3°×3° each) |
| **Native resolution** | 10 m |
| **Analysis grid** | Majority-resampled to 0.01° canonical grid |
| **Ingestion** | `ingest/lulc_worldcover_ingest.py` |
| **Channel** | `ndvi` (vegetation presence proxy; 1.0 for vegetated, 0.1 for non-vegetated) |

| Metric | Threshold |
|--------|-----------|
| Tile coverage of AOI | 100 % (no missing tiles) |
| Class distribution sanity | ≥ 1 vegetated class present in any fire-active AOI |

### 4b. Live Fuel Moisture Content (LFMC)

| Attribute | Value |
|-----------|-------|
| **Primary source** | ECMWF ecLand reanalysis LFMC |
| **Provider** | ECMWF |
| **API** | `LFMC_ECLAND_API_URL` + `LFMC_ECLAND_API_TOKEN` |
| **Temporal resolution** | Daily reanalysis |
| **Ingestion** | `ingest/lfmc_ecland_ingest.py` |
| **Channel** | `lfmc` (live fuel moisture fraction) |
| **Maturity note** | Production scaffold at `mvp_operational`; full validation deferred |

| Metric | Threshold |
|--------|-----------|
| Data freshness | ≤ 48 h old at inference time (warn if exceeded) |
| Value range | [0.0, 4.0] (fraction; >1 indicates full saturation) |

### 4c. Dead Fuel Moisture Content (DFMC)

| Attribute | Value |
|-----------|-------|
| **Primary source** | Derived from weather inputs — not an external dataset |
| **Formula** | Nelson (1984) NFDRS piecewise Equilibrium Moisture Content |
| **Inputs** | `t2m` (K), `rh2m` (%) from weather cube |
| **Implementation** | `ml/spread_features.py:_compute_dfmc` |
| **Channel** | `dfmc` (dead fuel moisture fraction; clamped [0.0, 0.40]) |

Nelson (1984) EMC piecewise formula (NFDRS standard):
- RH < 10 %: `EMC = 0.03229 + 0.281073·h − 0.000578·h·T_F`
- 10 ≤ RH < 50 %: `EMC = 2.22749 + 0.160107·h − 0.014784·T_F`
- RH ≥ 50 %: `EMC = 21.0606 + 0.005565·h² − 0.00035·h·T_F − 0.483199·h`

where `h` = relative humidity (%), `T_F` = temperature (°F). Output in %, then divided by 100.

**When weather is from the calm fallback cube**, DFMC is set to NaN and flagged in lineage attrs.

---

## 5. Source Declaration in Gate Config

Every spread champion–challenger evaluation config must include a `data_sources` block that
explicitly declares the source for each input category. The gate enforces this with
**STOP-SRC-001** when any required key is absent.

### Required keys

| Key | Required value (example) |
|-----|--------------------------|
| `data_sources.fires` | e.g. `"nasa_firms_viirs_nrt"` |
| `data_sources.weather` | e.g. `"noaa_gfs_025deg"` or `"noaa_hrrr_3km"` |
| `data_sources.terrain` | e.g. `"srtm30_dem_derived"` |
| `data_sources.fuels` | e.g. `"esa_worldcover_10m_ndvi+ecmwf_ecland_lfmc+nfdrs_dfmc"` |

### Example

```yaml
data_sources:
  fires: "nasa_firms_viirs_nrt"
  weather: "noaa_gfs_025deg"
  terrain: "srtm30_dem_derived"
  fuels: "esa_worldcover_10m_ndvi+ecmwf_ecland_lfmc+nfdrs_dfmc"
```

Absence of this block or any required key triggers **STOP-SRC-001** and sets `gate_report.pass`
to `false`.

---

## 6. Inference Lineage Attributes

Every `SpreadForecast.probabilities` xarray DataArray carries the following lineage attributes
at inference time (set by `ml/spread/service.py`):

| Attribute | Type | Description |
|-----------|------|-------------|
| `lineage_fires_source` | str | Always `"nasa_firms_viirs_nrt"` |
| `lineage_weather_source` | str | Model name from weather run (`"gfs"`, `"hrrr"`, or `"fallback_zeros"`) |
| `lineage_weather_run_id` | str \| None | DB row ID of the weather run used |
| `lineage_terrain_source` | str | `"dem_derived"` or `"fallback_zeros"` |
| `lineage_fuels_ndvi_source` | str | Always `"esa_worldcover_10m"` |
| `lineage_fuels_lfmc_source` | str | Always `"ecmwf_ecland_lfmc"` |
| `lineage_fuels_dfmc_source` | str | Always `"nfdrs_nelson1984"` |
| `lineage_data_sources_declared` | bool | `True` when `docs/spread_data_sources.md` is present |
