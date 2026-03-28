# Feature Brief: MeteoAlarm Weather Warning Integration

**Status:** Proposal
**Origin:** Earth Tools strategy — consolidating MeteoWatch into the wildfire intelligence layer
**Priority:** P1 (high value, not time-critical)

---

## Problem

The wildfire tool's risk scoring currently uses raw numerical weather data (humidity, precipitation, wind speed) to estimate fire conditions. These are unvalidated signals — the system doesn't know whether conditions are dangerous enough to have triggered a human expert's judgment.

Separately, MeteoWatch (our MeteoAlarm wrapper) surfaces official weather warnings for Europe but has no connection to fire detections or risk scoring. The two tools live side by side without informing each other.

A user monitoring a fire cluster in Greece has to manually cross-reference the wildfire map and MeteoWatch to understand that there's also an active RED wind warning in that region. We're forcing them to connect the dots.

---

## Opportunity

MeteoAlarm warnings represent meteorologist-issued judgments that conditions are dangerous. For fire specifically, four warning types are directly relevant:

- **Wind** — accelerates spread, changes fire direction unpredictably
- **Extreme heat** — elevates ignition risk and fire intensity
- **Drought / Forest fire** — direct fire danger signal
- **Thunderstorm** — potential ignition via lightning

When any of these are active at RED or ORANGE severity in an area where we already have fire detections or elevated risk scores, that's a compounded signal no single data source provides alone. Surfacing that combination is something MeteoAlarm, Copernicus, and NASA FIRMS each individually cannot do.

---

## Proposed Solution

Integrate MeteoAlarm warning data as a first-class layer within the wildfire tool, in three places:

### 1. Risk Score Enhancement
Add official weather warnings as a third component in the risk scoring model, alongside the existing static (land cover) and dynamic (raw weather) factors. A grid cell inside a RED wind or heat warning should score higher than raw weather data alone would suggest — because a meteorologist has already determined those conditions are dangerous.

Rain and snow warnings should have the inverse effect, modestly suppressing risk scores.

### 2. Map Warning Layer
A new toggleable overlay showing MeteoAlarm warning polygons on the fire map, colour-coded by severity (red / orange / yellow). Users should be able to visually see the intersection of fire hotspots and active weather warnings without switching tools.

### 3. Fire Details Context
When a user selects a fire detection or cluster, the detail panel should list any active MeteoAlarm warnings overlapping that location — type, severity level, and time remaining. A fire in a RED wind warning expiring in 6 hours tells a very different story from the same fire with no active warnings.

### 4. Priority Feed Boosting
Fire detections that spatially intersect with RED-level warnings in fire-relevant categories should receive a priority boost in the alert feed. The combination of satellite-confirmed detection + official dangerous weather warning is a stronger signal than either alone.

---

## Scope & Constraints

- **Geographic coverage:** MeteoAlarm covers Europe only. Outside Europe, the integration gracefully degrades — no warning layer, risk scoring falls back to the existing two-component formula. This is acceptable for now and sets up the pattern for extending to other warning systems (NOAA for North America, BoM for Australia) later.
- **Data freshness:** MeteoAlarm refreshes warnings approximately every 15 minutes. The integration should cache warnings server-side and serve from cache rather than fetching per-request.
- **No new external dependency for users:** This is fully backend-fetched; users don't interact with MeteoAlarm directly.

---

## Success Criteria

- A user looking at an active fire detection in Europe can see, without leaving the wildfire tool, whether there are active dangerous-weather warnings in that area.
- The risk grid visibly scores higher in regions with active RED/ORANGE wind, heat, or drought warnings compared to equivalent regions without warnings.
- The priority feed promotes fire-plus-warning combinations above fire-only detections of equivalent satellite confidence.
- MeteoWatch can be retired or repurposed as a standalone page — its core value is now embedded in the wildfire tool.

---

## Why Now (Eventually)

This is not urgent, but it closes the strategic gap identified in the Earth Tools review: MeteoWatch currently duplicates MeteoAlarm with no differentiation. Integrating its data pipeline into the wildfire tool gives that work a purpose, eliminates redundancy, and makes the wildfire tool meaningfully richer. It's the right direction for Earth Tools as a platform — one place that correlates environmental signals rather than a collection of separate single-purpose pages.
