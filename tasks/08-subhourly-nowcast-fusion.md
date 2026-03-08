# Task 08: Subhourly Nowcast Fusion

## Objective
Add an auxiliary high-frequency nowcast ingest path (geostationary hotspot feed) and fuse it with existing polar-orbiting detections.

## Primary Targets
- `/Users/vanyoivanov/Projects/wildfire-nowcast/ingest/orchestrator.py`
- `/Users/vanyoivanov/Projects/wildfire-nowcast/ingest/` (new geostationary ingest module)
- `/Users/vanyoivanov/Projects/wildfire-nowcast/api/fires/repo.py`
- `/Users/vanyoivanov/Projects/wildfire-nowcast/api/routes/fires.py`

## Scope
- Introduce feature-flagged ingest for subhourly hotspot telemetry.
- Normalize source confidence and dedupe/fuse with FIRMS detections for nowcast views.
- Expose source provenance in fires/events API responses.

## Out Of Scope
- Denoiser model retraining.
- Spread promotion gate changes.

## Independence Boundary
- Can run as an optional source path; existing FIRMS pipeline remains unchanged when disabled.

## Deliverables
- New ingest job and scheduler wiring for subhourly source.
- Fusion logic with deterministic dedupe/source-priority rules.
- Tests for source merge behavior and provenance fields.

## Exit Criteria
- When enabled, nowcast updates include fused subhourly detections.
- Provenance clearly indicates contributing source per detection/event.
- Disabling the feature flag restores current behavior exactly.

