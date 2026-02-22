# Task 11: Source Freshness Coverage Expansion

## Objective
Expand system freshness/health diagnostics to include fuel-moisture and denoiser-health signals, not only FIRMS/weather/terrain/perimeters.

## Primary Targets
- `/Users/vanyoivanov/Projects/wildfire-nowcast/api/data_status.py`
- `/Users/vanyoivanov/Projects/wildfire-nowcast/api/config.py`
- `/Users/vanyoivanov/Projects/wildfire-nowcast/ingest/orchestrator.py`

## Scope
- Add freshness status for fuel feature runs and denoiser scoring/drift signals.
- Extend health payloads and dashboard outputs with new source states and thresholds.
- Add machine-readable recovery hints per new failure mode.

## Out Of Scope
- Uncertainty quantification algorithm changes.
- Fuel ingestion provider implementation details.

## Independence Boundary
- Uses existing run tables/metrics and can be delivered without changing model behavior.

## Deliverables
- Freshness snapshot schema updates with fuels + denoiser health sections.
- Configurable stale thresholds for added sources.
- Tests for freshness-state transitions and health endpoint output.

## Exit Criteria
- Health endpoints reflect all operationally critical data/model sources.
- Dashboard indicates stale/missing fuel or denoiser-health degradations.
- Health payload includes deterministic remediation hints for each new alert type.
