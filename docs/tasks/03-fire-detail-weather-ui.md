# Task: Render weather conditions in the fire detail panel

## Goal

When a user clicks a fire detection, the detail panel should show the weather block returned by the API. The standard for success is operational legibility — a fire analyst should be able to glance at the weather block and immediately understand whether conditions are dangerous, without needing meteorological training or a second data source.

## Context

Read `docs/spec_meteoalarm_weather_warnings.md` for the API response shape and null-state copy. Read the existing fire detail panel component to understand where the weather block fits in the layout.

Three things matter most for this to actually be useful:

**Wind direction** must be human-readable. Raw degrees are not enough.

**Relative humidity** must communicate fire risk, not just display a number. The API response includes a computed risk level field — use it to visually distinguish dangerous conditions. The label should be readable without hover.

**Data provenance must be visible.** Show data age and resolution inline. If the API response indicates bias correction was applied, say so — this is a differentiator and the user should see it. Don't hide provenance in tooltips.

## Constraints

- When weather data is unavailable, show a clear inline explanation — no blank sections
- All display logic for RH thresholds should come from the API response, not be hardcoded in the component

## Done when

- Weather block renders correctly for a fire with data, including RH risk label and human-readable wind direction
- Data age, resolution, and bias correction status are visible inline
- Null state renders without errors with the reason string from the API
- Component tests cover the data, null, and elevated/critical RH states
