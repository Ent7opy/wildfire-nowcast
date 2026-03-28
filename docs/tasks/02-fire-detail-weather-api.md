# Task: Add weather conditions to the fire detail API response

## Goal

When a user fetches a fire detection's details, the response should include current weather conditions at that location — wind speed, wind direction, relative humidity, temperature, and 24h precipitation.

## Context

Read `docs/spec_meteoalarm_weather_warnings.md` for the full response contract, null-state requirements, and acceptance criteria. That spec is the source of truth for this task.

Two things the spec requires that need particular attention:

**Bias correction labelling.** The GFS ingest pipeline applies bias correction. The response must make this visible — users evaluating this tool against raw FIRMS-based platforms need to know the data has been processed. Read the ingest pipeline to understand what's being corrected and how to represent it accurately.

**RH fire-risk level.** Relative humidity below 25% and below 15% are standard fire-weather thresholds with operational meaning. The response must include a computed risk level for the RH value — not just the raw number. The UI will use this to colour-code the display; it should come from the API, not be hardcoded in the frontend.

## Constraints

- Use weather data already in the system — no new external calls at request time
- If no weather data covers the fire location, return a structured null with a reason (not an absent field, not an error)
- Task 01 must be complete before this one — the shared weather lookup function is the right primitive to build on

## Done when

- Fire detail response includes a weather block when GFS data covers the location
- The null case is handled correctly per the spec
- Bias correction is represented in the response
- RH fire-risk level is computed server-side
- Response time regression is within the threshold specified in the spec
- Tests cover the happy path, the null case, and the RH threshold boundaries
