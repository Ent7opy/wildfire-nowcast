# Spec: Weather Context in Fire Detail Panel

**Status:** Draft
**Author:** Vanyo Ivanov
**Date:** 2026-03-28
**Origin:** Proposal — `docs/feature_meteoalarm_weather_warnings.md`
**Priority:** P1
**Maturity target:** `mvp_operational`

---

## Problem Statement

When a user selects a fire detection, the detail panel shows satellite-derived attributes (confidence, FRP, acquisition time) but no ambient weather conditions. Wind speed, humidity, and temperature at the time of detection are the most operationally relevant context for understanding fire behaviour — but the user has to go elsewhere to find them. The data to answer this is already in the system; it just isn't surfaced.

---

## Goals

1. A user selecting any fire detection sees current weather conditions for that location in the detail panel — without leaving the tool or opening a second data source.
2. The feature works globally, not just in Europe.
3. No new infrastructure is introduced: the implementation uses weather data the system already collects.

---

## Non-Goals

**MeteoAlarm / official warning integration (v1).** Showing expert-issued categorical warnings (RED wind warning, etc.) alongside raw weather values is a genuine enhancement — but it is a separate feature and should not block this one. See the original proposal for that scope.

**Microclimate precision.** GFS data is 0.25° resolution (~25km grid cells). This is synoptic-scale weather, not point-accurate. The UI must reflect this clearly. High-resolution local weather is a future data source consideration.

**Weather forecast horizon.** This spec shows current conditions at acquisition time. Forward-looking weather (6h, 24h) for a specific fire location is a separate feature.

**User-defined weather alerts or notifications.** Passive contextual display only. No thresholds, no push alerts.

---

## How It Works

### Existing infrastructure (no changes needed)

The ingest orchestrator already runs `weather_ingest.py` on a 3-hour cycle, storing GFS 0.25° forecast data as netCDF files with metadata in the `weather_runs` table. A 14-day rolling retention window means fresh data is always available. This runs independently of spread forecasts.

`api/fires/scoring.py` already contains `_get_weather_data_for_point(lat, lon, ref_time, ...)`, which:
1. Queries `weather_runs` for the most recent completed run whose bbox contains the point
2. Loads the netCDF from disk
3. Nearest-neighbour interpolates to the fire coordinates and acquisition time
4. Returns wind speed (m/s), relative humidity (%), temperature (°C), and precipitation (mm)

This function is currently used only by the risk grid. It is the right primitive for the detail panel.

### What needs to be built

**1. API: expose weather at a fire location**

Add a `weather` field to the existing fire detail response (or a dedicated sub-endpoint — implementation decision). The response shape:

```json
{
  "weather": {
    "wind_speed_ms": 12.4,
    "wind_direction_deg": 230,
    "relative_humidity_pct": 18,
    "temperature_c": 36.2,
    "precip_mm_24h": 0.0,
    "source_run_time": "2026-03-28T06:00:00Z",
    "data_age_hours": 2.1,
    "resolution_note": "GFS 0.25° — nearest grid point (~25 km)"
  }
}
```

If no weather run covers the fire location or the most recent run is older than `time_tolerance_hours`, `weather` must be `null` with a reason field — not absent, not an error response.

**2. UI: render weather in the fire detail panel**

Display the weather block beneath existing fire attributes (confidence, FRP, land cover). Show wind speed with direction, humidity, temperature, and precipitation. Display `data_age_hours` so the user knows how fresh the conditions are. Include the resolution note as a tooltip or subtext — don't hide it.

---

## User Stories

- As a fire analyst clicking a detection in Portugal, I want to see the wind speed and humidity at that location so that I can assess spread potential without switching tools.

- As a fire analyst in a region with no active spread forecast, I want weather context to be available regardless — not only for fires where a spread model has been triggered.

- As a user clicking a fire in the US or Australia, I want weather to appear exactly as it does for European fires — the feature must not be geographically gated.

- As an analyst sharing a screenshot of a fire detail panel, I want the data source and resolution clearly indicated so that colleagues don't over-interpret the precision of the values shown.

---

## Requirements

### Must-Have (P0)

**Weather data in fire detail response**

The fires detail API must return a `weather` block for any fire detection where a recent GFS run exists covering that location. The block must include wind speed (m/s), wind direction (degrees), relative humidity (%), temperature (°C), 24h precipitation (mm), the GFS run timestamp, and data age in hours.

`_get_weather_data_for_point` already computes wind speed from u10/v10 components. Wind direction must be derived from the same components (`atan2(-u, -v)` in meteorological convention — wind direction is where the wind comes from).

*Acceptance criteria:*
- [ ] Fire detail response includes `weather` block when GFS data covers the fire location
- [ ] `weather` is `null` (not absent) when no data is available, with a `reason` string
- [ ] Wind direction is returned in degrees (0–360, meteorological convention)
- [ ] `data_age_hours` reflects time elapsed since the GFS run used, not since ingestion
- [ ] Response time for the fire detail endpoint does not regress by more than 200ms (netCDF read is local disk, should be fast)

**UI weather block in fire detail panel**

Wind, humidity, temperature, and precipitation are displayed in the detail panel when `weather` is non-null. When `weather` is null, a short inline message explains why ("Weather data not available for this location"). The GFS resolution note is visible — not hidden.

*Acceptance criteria:*
- [ ] Weather block renders correctly for a fire with data
- [ ] Null state renders without errors and shows the reason string
- [ ] Resolution note is visible in the UI (tooltip or subtext, not omitted)
- [ ] Wind direction renders as a compass bearing or arrow, not a raw degree value

---

### Nice-to-Have (P1)

**Wind direction arrow in the panel.** A simple compass rose or directional arrow makes wind direction immediately readable for non-meteorologists. Degree values alone require mental conversion.

**Humidity fire-risk colour coding.** Relative humidity below ~25% is a meaningful threshold for fire behaviour. A subtle colour indicator (e.g. amber below 25%, red below 15%) adds interpretive value without requiring the analyst to apply the threshold themselves.

**Weather data in the export endpoint.** If a user exports fire data, the weather block should be included in the export payload. Currently the exports endpoint likely omits it since it doesn't exist yet.

---

### Future Considerations (P2)

**MeteoAlarm warnings as an overlay.** The original proposal remains valid as a follow-on: once raw weather is surfaced, adding the "official dangerous conditions" layer from MeteoAlarm gives non-expert users an interpretive signal on top of the numbers. The detail panel already has the right location to add this — a `warnings` block alongside `weather`.

**Higher-resolution weather.** GFS at 0.25° is a ~25km grid. For fires near complex terrain or coastlines, HRRR (3km, US-only) or ECMWF HRES would be materially more accurate. The `_get_weather_data_for_point` abstraction can support multiple model backends — the API response already includes `source_run_time` which is model-agnostic.

**Weather at acquisition time vs. now.** Currently the spec returns conditions nearest to the fire's acquisition time. Showing current conditions (right now, not when the satellite passed) and near-term forecast (next 6h) would be valuable for ongoing incidents. The `weather_runs` table stores forecast horizons, so this is achievable without new ingest.

---

## Success Metrics

**Leading (2 weeks post-launch)**

- Fire detail panel open rate that results in a weather block being displayed. Target: >80% of detail panel opens show weather data (i.e., GFS coverage is as good as expected globally).
- Zero null-state errors in production (null is valid, errors are not).

**Lagging (6 weeks)**

- Session depth: do users who see weather context in the detail panel spend longer in the tool or open more fire details per session? Hypothesis: yes, because they're getting more value from each click.
- Support/feedback signal: reduction in "where do I find weather for this area?" type questions, if any exist.

---

## Open Questions

| Question | Owner | Blocking? |
|---|---|---|
| Should `weather` be folded into the existing fire detail response or served as a separate sub-endpoint (`/fires/{id}/weather`)? Inline is simpler for the UI; separate avoids slowing down every detail load for users who don't scroll to the weather block. | Engineering | No — default to inline unless profiling shows it matters |
| The `weather_runs` bbox index was added in migration `20260119_add_weather_bbox_index.py`. Is it a functional index on `ST_MakeEnvelope(...)` or a plain column index? If it's plain columns, the spatial containment query may not use it efficiently for point lookups. | Engineering | No — worth verifying before launch |
| What is the acceptable staleness threshold for weather data? `_get_weather_data_for_point` has a `time_tolerance_hours` parameter. The current default may be tuned for spread forecasting, not fire detail display. | Engineering | No — can tune post-launch based on null rate |

---

## Implementation Notes

`_get_weather_data_for_point` currently lives in `api/fires/scoring.py`. If it is called from both the risk grid and the new fire detail endpoint, consider moving it to a shared module (e.g. `api/core/weather.py`) to avoid a circular import. This is a refactor, not a rewrite.

Wind direction derivation from u10/v10:

```python
import math
wind_dir_deg = (math.degrees(math.atan2(-u10, -v10)) + 360) % 360
```

This follows meteorological convention (direction the wind is coming *from*). Confirm this matches the convention used elsewhere in the UI before shipping.
