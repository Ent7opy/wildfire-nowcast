# Task: Review Queue — Expanded decision panel

**Location:** `ui/src/components/ReviewQueuePanel.tsx`, `api/routes/internal.py`, `api/routes/fires.py`
**Impact:** High — transforms the queue from a coin-flip interface into a genuine decision-support tool
**Maturity target:** `mvp_operational`
**Depends on:** Task 12 (location context enrichment)

## Problem

Even after Task 12 adds location context to queue items, an operator still cannot make a high-quality decision from the list view alone. Confirming or dismissing a fire requires situational awareness: what does the area look like, what's the wind doing, are there other confirmed fires nearby, has this location been flagged before?

Without this context, operator decisions are low-quality guesses. Low-quality labels are worse than no labels — they corrupt any future retraining signal.

## Proposed Solution

When an operator clicks a queue item, expand it into a decision panel (inline or side-panel, not a modal — the operator needs to see context without losing their place in the list).

### Decision panel contents

**1. Plain-language reason summary**

Replace the raw reason chip with a full sentence explaining why this specific item was flagged. Generated server-side using the payload values:

- HARD BYPASS example: *"Flagged automatically: 21 MW fire radiative power in dense conifer forest. High-confidence FIRMS detection. Treated as confirmed fire until reviewed."*
- UNCERTAINTY example: *"Model score was 0.50 — right at the decision boundary (threshold: 0.45–0.55). Low FRP suggests possible industrial or agricultural burn."*

**2. Map thumbnail**

A small embedded map (300×200 px) centred on the event centroid, using the existing MapLibre GL instance. Show:
- Satellite basemap
- Detection point marker
- Wind direction arrow (from the latest weather cube for this location — already available via `weather_ingest`)
- 10 km radius circle for spatial context

Re-use the existing map component; do not load a second MapLibre instance. Use a locked viewport (no pan/zoom on the thumbnail). Provide a "View on map" button that closes the panel and flies the main map to the event.

**3. Nearby confirmed fires**

Query `fire_events` for confirmed (non-review) events within 100 km, active in the last 48 hours. Display as a compact list: *"2 active fires within 100 km — largest: 340 MW, 67 km SW."*

If none: *"No confirmed fires within 100 km in the last 48 hours."*

This query can reuse the existing fires repo spatial query pattern.

**4. Detection history for this location**

Query `denoiser_review_queue` for prior review items within a 5 km radius of the current event centroid, in the last 30 days. Show outcome summary:
- *"This location: flagged 3× in past 30 days — 2 confirmed fires, 1 marked noise."*
- *"First time this location has been flagged."*

This is a strong signal: a location with a history of confirmed fires should lean toward confirmation.

**5. Current fire weather (brief)**

From the weather cube: wind speed + direction, relative humidity, temperature. One line:
- *"Wind: 18 km/h NE · RH: 22% · Temp: 34°C"*

High wind + low RH is a meaningful danger signal that an operator will recognise.

### API changes

Add a `GET /internal/denoiser/review-queue/{event_id}/detail` endpoint that returns all decision panel data in one request:

```python
{
    "reason_summary": str,           # Plain-language explanation
    "centroid_lat": float,
    "centroid_lon": float,
    "wind_speed_kmh": float | None,
    "wind_direction_deg": float | None,
    "relative_humidity_pct": float | None,
    "temperature_c": float | None,
    "nearby_fires_count": int,
    "nearby_fires_max_frp_mw": float | None,
    "nearby_fires_nearest_km": float | None,
    "location_history_flagged": int,
    "location_history_confirmed": int,
    "location_history_noise": int,
}
```

The UI fetches this lazily (on item expand), not on initial queue load.

## Acceptance Criteria

- [ ] Clicking a queue item expands an inline decision panel without losing list position
- [ ] Panel shows a plain-language reason summary using actual payload values
- [ ] Panel shows a map thumbnail centred on the event with wind direction overlay
- [ ] "View on map" button flies the main map to the event and closes the panel
- [ ] Panel shows nearby confirmed fire count, max FRP, and distance to nearest
- [ ] Panel shows location history (flagged / confirmed / noise counts for 5 km / 30 days)
- [ ] Panel shows current wind, RH, and temperature
- [ ] Detail data is fetched lazily on expand — not included in the main queue list response
- [ ] Panel degrades gracefully when weather or history data is unavailable

## Notes

- Lazy fetch on expand is important — do not add decision panel data to the bulk queue endpoint; it would make the 200-item load unacceptably slow
- The weather data should come from the existing weather cube, not a new external call
- Do not build a new map component for the thumbnail — use a locked viewport on the existing MapLibre instance or a static tile image if a second instance is too expensive
- "Reason summary" text generation should live server-side so it is consistent and testable, not client-side string interpolation
