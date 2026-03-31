# Task: Ignition Probability — API layer

**Location:** `api/routes/ignition.py` (new), `api/main.py`, `api/ignition/` (new)
**Impact:** High — exposes the ignition probability surface to the UI and external consumers
**Maturity target:** `mvp_operational`
**Depends on:** Task 17 (ignition probability model must be registered and promoted)

## Problem

The ignition probability model (Task 17) runs offline. There is no endpoint that runs inference on demand or on a cadence and returns the probability surface to the UI. The existing risk endpoint (`GET /risk`) is the closest analogue but serves a rule-based heuristic; the ignition endpoint must serve ML-derived probabilities across three forecast horizons.

## Proposed Solution

Add a new `GET /ignition` endpoint following the pattern established by `api/routes/risk.py` and `api/routes/forecast.py`.

### Endpoint

```
GET /ignition
```

**Query parameters:**
- `min_lon`, `min_lat`, `max_lon`, `max_lat` — bounding box (required)
- `cell_size_km` — grid resolution (default 10.0, range 1–50 km; cap grid at 500 cells)
- `horizon` — forecast horizon: `now` | `+24h` | `+48h` (default `now`)

**Response shape:**

```json
{
  "horizon": "now",
  "valid_time": "2026-03-31T12:00:00Z",
  "model_id": "<registered model id>",
  "cells": [
    {
      "cell_id": "...",
      "lat": 37.5,
      "lon": -120.2,
      "probability": 0.68,
      "level": "high",
      "signals": {
        "drought_index": 0.72,
        "thunderstorm_active": false,
        "days_since_last_burn": 1240,
        "relative_humidity": 18.0,
        "wind_speed_kmh": 42.0
      }
    }
  ],
  "coverage_warnings": ["drought_index_stale: last updated 2026-03-24"]
}
```

`level` maps to the categorical thresholds from the model runtime contract (`low` / `elevated` / `high` / `critical`).

`signals` returns the top contributing feature values per cell for UI transparency — not the full feature vector. Pick the 5 most influential features from the model's feature importances at registration time and store them in the runtime contract so the API knows which to surface.

`coverage_warnings` is an array of advisory strings emitted when a signal is stale or missing (drought index >10 days, no thunderstorm data for region, etc.). These warnings must be visible to the UI to show appropriate caveats. They are never silent.

### Forecast horizons

- `now` — use the most recent ingested weather values (same as the risk grid)
- `+24h` — use the GFS +24h forecast from the weather cube (already in system via `weather_ingest`)
- `+48h` — use GFS +48h forecast; attach a fixed `low_confidence: true` flag in the response root to prompt the UI to display the appropriate caveat

### Caching

Cache the response for 21600s (6 hours) aligned to GFS cycle times. Use the existing `cache_*` dependency pattern from `api/deps.py`. The cache key must include `horizon` so the three horizons are cached independently.

### Model availability

If no ignition model is promoted and `IGNITION_REQUIRED=true`, return `503 Service Unavailable` with a structured error body:
```json
{"error": "ignition_model_unavailable", "detail": "No promoted ignition model found."}
```
Do not return zeros or a degraded response — a silent failure here would be worse than an honest 503 for operators relying on the layer.

### Router registration

Add the router in `api/main.py`:
```python
from api.routes.ignition import ignition_router
app.include_router(ignition_router, prefix="/ignition")
```

Create `api/ignition/` as a subpackage if the grid computation logic is non-trivial. The endpoint handler should be thin — delegate to `api/ignition/grid.py` (analogous to `api/risk/grid.py`).

## Acceptance Criteria

- [ ] `GET /ignition` returns a valid grid response for all three horizons (`now`, `+24h`, `+48h`)
- [ ] `+48h` response includes `"low_confidence": true` at the response root
- [ ] `coverage_warnings` is populated when drought index is stale (>10 days) or missing
- [ ] Grid is capped at 500 cells; requests that would exceed this are clipped, not rejected
- [ ] Response is cached for 6h per horizon per bbox (cache keys are horizon-scoped)
- [ ] Returns `503` (not zeros) when no ignition model is promoted and `IGNITION_REQUIRED=true`
- [ ] `model_id` in the response matches the promoted model in the registry
- [ ] Unit tests cover: correct horizon routing to weather cube, stale signal warning emission, 503 when model absent
- [ ] Integration test: end-to-end call with a real (or seeded) database returns a well-formed response

## Notes

- The risk grid (`api/risk/grid.py`) is the closest structural reference — read it before building the ignition grid module. The cell generation and bbox-to-grid logic can likely be extracted or reused directly.
- `signals` in the response is for UI transparency, not a debugging dump. Only surface the top 5 features by model importance, not every feature in the vector. Store this list in the model's runtime contract at training time (Task 17) so the API doesn't hardcode feature names.
- Do not add the `horizon` parameter to the risk endpoint. They are separate layers with separate semantics.
- The `valid_time` in the response is the time the weather inputs are valid for, not the time of the API call. For `now` this is the latest weather ingestion timestamp; for `+24h` it is now + 24h.
- If the weather cube does not have +48h forecast data (e.g. because GFS has not refreshed yet), return the most recent available forecast step with a `coverage_warnings` entry rather than failing.
