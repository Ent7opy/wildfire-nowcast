# Task 07: Fire-Cluster Forecast Support

## Objective
Enable forecasting by fire cluster/event identifier so analysts can request spread directly from tracked fire objects.

## Primary Targets
- `/Users/vanyoivanov/Projects/wildfire-nowcast/ml/spread/service.py`
- `/Users/vanyoivanov/Projects/wildfire-nowcast/api/routes/forecast.py`
- `/Users/vanyoivanov/Projects/wildfire-nowcast/api/fires/repo.py`
- `/Users/vanyoivanov/Projects/wildfire-nowcast/ml/tests/test_spread_service.py`

## Scope
- Implement `fire_cluster_id` (or event id) resolution to forecast bbox/window.
- Add request validation and user-facing errors for missing/invalid cluster references.
- Persist cluster-linked metadata in forecast run records.

## Out Of Scope
- Event/front association algorithm changes.
- Spread model architecture changes.

## Independence Boundary
- Uses existing event tables and can be delivered without ingest changes.

## Deliverables
- Service support replacing current `NotImplementedError` path for cluster-based requests.
- API contract update for cluster-triggered forecasts.
- Tests for valid cluster resolution, stale/ended clusters, and not-found behavior.

## Exit Criteria
- Cluster-based forecast requests execute end-to-end.
- Forecast records include source cluster/event identifiers.
- No regressions for existing bbox-based forecast requests.

