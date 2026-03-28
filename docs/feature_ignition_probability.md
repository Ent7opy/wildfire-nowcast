# Feature Brief: Ignition Probability Layer

**Status:** Proposal
**Origin:** Earth Tools strategy — extending wildfire-nowcast from reactive detection to proactive risk
**Priority:** P1 (high strategic value, foundational to long-term differentiation)

---

## Problem

Every capability in wildfire-nowcast today is reactive. The satellite detects a fire, we score it, we project its spread. The tool is genuinely good at "fire is here" — but it is blind to the hours or days before ignition when conditions are silently building toward a fire that hasn't started yet.

This is a meaningful gap. The most actionable moment in wildfire management isn't after a fire is detected — it's before. Pre-positioning resources, issuing early warnings, and raising public awareness are all far more effective when there's lead time. By the time a satellite confirms ignition, that lead time is gone.

---

## Opportunity

The data required to estimate ignition probability already exists and is largely already flowing through the system:

- **Land cover flammability** — already used in the static risk score
- **Weather conditions** — humidity, wind, temperature already used in dynamic risk
- **Drought index** — available from Copernicus Global Drought Observatory; not yet integrated
- **Lightning forecasts and strike data** — NOAA operates lightning detection instruments (LIS/OTD on satellite); thunderstorm warnings from MeteoAlarm flag imminent ignition risk from natural sources
- **Recent fire history** — areas that have burned recently are temporarily suppressed; areas that haven't burned in many years have accumulated fuel load

When these signals converge — dry vegetation, low humidity, strong winds, high drought index, incoming thunderstorm activity — the conditions for ignition are primed. A map that shows where those conditions are stacking up, before any fire starts, is something no current public tool offers at this level of accessibility.

This is the proactive counterpart to the existing spread model. Spread asks: *given a fire, where does it go?* Ignition probability asks: *given conditions, where does a fire start?*

---

## Proposed Solution

A new **Ignition Probability** map layer, displayed alongside the existing fire detection and risk layers, showing the estimated likelihood of a new ignition event occurring in a given area within the next 24–48 hours.

### Input signals (proposed)

- Fuel moisture / drought index (Copernicus GDO or equivalent)
- Land cover flammability (already in system)
- Current and forecast weather — humidity, temperature, wind speed
- Recent precipitation history (already in system)
- Lightning strike probability / thunderstorm forecast (new data source)
- Time-since-last-burn for the area (fuel accumulation proxy)

### Output

A continuous probability surface rendered as a heatmap layer, with a categorical classification (low / elevated / high / critical) per grid cell. Updated on a regular cadence as weather forecasts refresh — ideally every 6 hours to align with GFS model cycles.

### Forecast horizons

- **Now** — current ignition conditions based on live weather
- **+24h** — based on weather forecast
- **+48h** — based on weather forecast (lower confidence, shown with appropriate caveats)

### User-facing presentation

- Toggled on/off independently from the fire detection and spread layers
- Severity-coded colour scale distinct from the existing risk grid to avoid visual confusion between "risk of ignition" and "confirmed fire risk"
- Where ignition probability is HIGH and there are no current detections, the priority feed should surface a warning: "No active fires detected, but conditions in this region are critical for ignition"
- Where ignition probability is HIGH *and* there is an active fire nearby, this context should be visible in the fire details panel — conditions are primed for new ignition events in addition to spread from the known fire

---

## Why This Is Strategically Important

The spread model makes wildfire-nowcast useful after a fire starts. The ignition probability layer makes it useful before. Together, they cover the full arc of a fire event — from precondition through detection through spread — in a way that no individual data service currently does.

This is the feature that most clearly answers the question "what does wildfire-nowcast give you that you can't get elsewhere?" A meteorologist looking at a RED fire danger warning can tell you conditions are bad. The ignition probability layer tells you *specifically where* those conditions are most dangerous, at grid-cell resolution, updated every 6 hours.

It also opens Earth Tools toward a broader vision: a platform that detects environmental threat preconditions, not just confirmed events. That's the direction worth building toward.

---

## Constraints & Open Questions

- **Lightning data source:** NOAA's lightning data has varying coverage and latency depending on the instrument. Worth evaluating what's freely available and at what resolution before committing to this as a primary signal. Thunderstorm forecast warnings (e.g. from MeteoAlarm in Europe) may be a simpler proxy to start with.
- **Drought index integration:** Copernicus GDO publishes at weekly cadence, which limits real-time responsiveness. This is better suited as a slow-moving background signal (fuel moisture proxy) than a live trigger.
- **Calibration:** Ignition probability is harder to validate than spread (we can measure spread against satellite observations; ignitions that don't happen are harder to evaluate). Confidence levels and uncertainty should be clearly communicated to users from the start.
- **Scope:** Like the MeteoAlarm integration, full global coverage may be limited by data source availability. Europe and North America have the best public data coverage; other regions may have degraded signal quality initially.

---

## Success Criteria

- Users can see, before any fire is detected, which areas are in high ignition-risk conditions right now and over the next 48 hours.
- The priority feed proactively surfaces critical ignition-risk zones, not just active fires.
- Post-launch, at least some ignition events in high-probability zones can be traced back to the map having shown elevated risk in the hours before detection — validating the model directionally.
- The layer is understandable to a non-expert user without explanation: the visual and labelling makes clear this is "conditions that could start a fire" rather than "confirmed fire activity."
