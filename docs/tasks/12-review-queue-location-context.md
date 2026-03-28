# Task: Review Queue — Location context enrichment

**Location:** `api/routes/internal.py`, `api/fires/repo.py`, `ui/src/components/ReviewQueuePanel.tsx`, `ui/src/api/review.ts`
**Impact:** High — operators currently have zero geographic context to make decisions
**Maturity target:** `mvp_operational`

## Problem

Every item in the Review Queue is spatially anonymous. There is no location name, no country, no distance to nearest settlement. An operator cannot confirm or dismiss a fire detection without knowing where in the world it is.

The `fire_events` table has geometry (centroid lat/lon). The review queue already links to `event_id`. The data exists — it just isn't surfaced.

## Proposed Solution

### API changes

Extend `GET /internal/denoiser/review-queue` to JOIN against `fire_events` and return location fields:

```python
# Additional fields to include in the review queue response
{
    "centroid_lat": float,
    "centroid_lon": float,
    "country_code": str | None,      # ISO 3166-1 alpha-2, from reverse geocode or spatial join
    "region_name": str | None,       # State/province level if available
    "nearest_place": str | None,     # Nearest named settlement + distance, e.g. "34 km NE of Redding, CA"
    "terrain_label": str | None,     # Human-readable landcover, e.g. "Dense conifer forest"
}
```

**Reverse geocoding approach:** Use the existing PostGIS installation. A spatial join against a Natural Earth admin boundaries table (already likely available or easy to load) gives country and region without an external API call. For nearest place name, use a pre-loaded populated places table (Natural Earth `ne_10m_populated_places`) with a `ST_Distance` query capped at 200 km.

**Terrain label mapping:** `landcover_mean` in the event payload maps to ESA WorldCover classes. Add a lookup table or in-code mapping:

| landcover_mean approx | Label |
|---|---|
| Tree cover (class 10) | Dense forest |
| Shrubland (class 20) | Shrubland |
| Grassland (class 30) | Grassland |
| Cropland (class 40) | Agricultural land |
| Built-up (class 50) | Urban / built-up |
| Bare / sparse (class 60) | Bare ground |

If `landcover_mean` is a continuous score rather than a discrete class, use the dominant class from the event's detection cluster if available, otherwise omit.

### UI changes

Display in each queue item row (compact, below the existing FRP/confidence line):
- Country flag emoji + region name, e.g. 🇺🇸 Northern California
- Nearest place, e.g. *34 km NE of Redding*
- Terrain label, e.g. *Dense forest*

On click / expand: show a small static map thumbnail (see Task 13 for the full decision panel).

## Acceptance Criteria

- [ ] Review queue API response includes `centroid_lat`, `centroid_lon`, `country_code`, `region_name`, `nearest_place`, `terrain_label`
- [ ] `nearest_place` uses a spatial query against a populated places dataset — no external geocoding API
- [ ] `terrain_label` is a human-readable string, not a raw class number or float
- [ ] Each queue item in the UI displays country/region and nearest place
- [ ] Items without location data (null event geometry) degrade gracefully — show "Location unavailable"
- [ ] API response time does not increase by more than 200 ms for a full 200-item queue fetch

## Notes

- Prefer PostGIS spatial joins over external geocoding APIs — no rate limits, no latency, no cost
- Natural Earth datasets are public domain and small enough to load into PostGIS as a one-time migration
- If Natural Earth populated places aren't already loaded, add a data migration or seed script — do not hardcode or approximate
- `terrain_label` logic should live in a shared utility so it can be reused in the fire detail view (Task 03)
- This task is a prerequisite for Task 13 (decision panel)
