# Task: Ignition Probability — UI layer

**Location:** `ui/src/components/`, `ui/src/state/`, `ui/src/`
**Impact:** High — makes the ignition probability feature visible and actionable for users
**Maturity target:** `mvp_operational`
**Depends on:** Task 18 (API endpoint must be available)

## Problem

The ignition probability surface exists in the API but is invisible to the user. The feature brief's core promise — "where are conditions primed for ignition, before any fire starts?" — cannot be delivered without a map layer and priority feed integration. This task closes that gap.

There are two distinct user-facing surfaces to build:
1. A **map layer** showing the ignition probability heatmap
2. **Priority feed integration** — proactive surfacing of critical ignition-risk zones, and contextual ignition risk in the fire details panel

## Proposed Solution

### Map layer

Add an `IgnitionProbabilityLayer` using Deck.GL's `HeatmapLayer` or `GridCellLayer`, toggled independently from the fire detection layer and the existing risk grid layer.

**Visual design constraints from the feature brief:**
- The colour scale must be distinct from the existing risk grid. The risk grid uses the existing risk colour ramp (which users associate with "confirmed fire risk"). The ignition layer must use a different hue family — consider amber/orange for `elevated`, orange-red for `high`, deep red/magenta for `critical`. The goal is that a user who has both layers active can tell them apart without a legend.
- `low` probability cells should be invisible (opacity 0) or very faint to avoid visual noise.
- Include a layer legend clearly labelled **"Ignition Risk"** (not "Fire Risk" — see the feature brief: this is about conditions, not confirmed events).

**Horizon selector:**
- Add a horizon control to the layer panel: `Now` / `+24h` / `+48h`
- `+48h` should display a visible caveat: *"Lower confidence — 48h forecast"*
- The horizon selector is only shown when the ignition layer is active

**Data fetching:**
- Use React Query (`@tanstack/react-query`) to call `GET /ignition` with the current map bbox and selected horizon
- Refetch when the bbox changes significantly (≥ 10 km shift or ≥ 1 zoom level change) — not on every map move
- Cache for 6 hours client-side to match server cache cadence
- If the API returns `coverage_warnings`, show them as a small advisory banner on the layer panel: *"Drought data: last updated 7 days ago"*
- If the API returns `503`, show a layer-level error state: *"Ignition model unavailable"* — do not crash the map

### Priority feed integration

The priority feed (wherever active-fire priority items are surfaced in the UI) must be extended for two new ignition-risk scenarios:

**Scenario 1: Critical conditions, no active fires**
When the ignition layer is active and a `critical`-level cell exists in the current viewport with no confirmed fire detections within 50 km, surface a priority item:
> *"No active fires detected — but conditions in [region name / coordinates] are critical for ignition right now."*

Region name should use the existing reverse-geocoding / location context enrichment pattern (Task 12 in the review queue work). Fall back to coordinates if unavailable.

**Scenario 2: Active fire with high ignition risk nearby**
When the fire details panel is open for a confirmed fire and any cell within 50 km of the fire centroid has `high` or `critical` ignition probability, add a context block to the details panel:
> *"Conditions in this area are primed for new ignitions. [N] high-risk cells within 50 km."*

This is not a separate notification — it is additional context appended to the existing fire details panel. Do not redesign the panel; add a contextual block below the existing weather summary.

### State management

Add ignition-related state to the existing Zustand map store:
- `ignitionLayerActive: boolean` — layer toggle
- `ignitionHorizon: 'now' | '+24h' | '+48h'` — current horizon
- `ignitionData: IgnitionGridResponse | null` — cached API response

Do not create a separate store for this. Extend the existing map store.

### Layer toggle

Add the ignition layer to the existing layer control panel alongside the fire detection and risk layers. The toggle label should read **"Ignition Risk"**. It should be off by default (the layer is additive and unfamiliar to first-time users).

## Acceptance Criteria

- [ ] `IgnitionProbabilityLayer` renders ignition probability cells as a heatmap with a colour scale distinct from the risk grid
- [ ] `low` cells are invisible; `elevated`, `high`, `critical` are visually distinguishable
- [ ] Layer is toggled independently from fire detection and risk layers; off by default
- [ ] Horizon selector (`Now` / `+24h` / `+48h`) is shown when the layer is active; `+48h` shows a confidence caveat
- [ ] `coverage_warnings` from the API surface as an advisory banner on the layer panel
- [ ] API `503` shows a graceful error state on the layer — map remains functional
- [ ] Priority feed surfaces a warning when `critical` cells exist in viewport with no fires within 50 km
- [ ] Fire details panel shows an ignition context block when `high` / `critical` cells are within 50 km of the fire
- [ ] Ignition layer state (`active`, `horizon`, `data`) lives in the existing Zustand map store
- [ ] React Query refetch triggers on meaningful bbox changes, not every map move
- [ ] Unit tests cover: priority feed warning logic (critical cells, no nearby fires), fire panel context block rendering

## Notes

- The colour scale is a UX-critical decision. Before finalising, confirm the chosen palette is distinguishable from the existing risk grid at a glance — the feature brief specifically calls out that visual confusion between "risk of ignition" and "confirmed fire risk" must be avoided.
- The `HeatmapLayer` in Deck.GL interpolates across point values and may blur cell boundaries in ways that mislead users about resolution. Consider `GridCellLayer` or `ScatterplotLayer` with opacity-mapped fill if sharp cell boundaries better communicate the discrete grid nature of the forecast. Either is acceptable — choose based on what looks clearer in practice.
- The priority feed items should not fire when the ignition layer is toggled off. If the user has turned the layer off, they have opted out of seeing ignition signals.
- Do not add ignition data to the main fires list query. The ignition endpoint is a separate spatial query with its own cadence.
- For the fire details panel integration, fetch ignition data lazily when the panel opens, using the fire's centroid + 50 km radius bbox. Do not include ignition context in the bulk fire list response.
- The `+48h` confidence caveat should be a fixed UI label, not dynamically driven by per-cell uncertainty scores. The model does not output per-cell confidence intervals at `mvp_operational` maturity.
