# Task 04: Operational Uncertainty Policy

## Objective
Implement a model-output uncertainty layer with abstention thresholds and clear confidence tiers for served forecasts.

## Primary Targets
- `/Users/vanyoivanov/Projects/wildfire-nowcast/ml/calibration.py`
- `/Users/vanyoivanov/Projects/wildfire-nowcast/ml/spread/service.py`
- `/Users/vanyoivanov/Projects/wildfire-nowcast/api/routes/forecast.py`

## Scope
- Add uncertainty quantification for spread outputs (calibrated confidence + prediction-region metadata).
- Define abstention triggers based on uncertainty width/confidence degradation.
- Surface uncertainty decisions in forecast metadata and service logs.

## Out Of Scope
- Data freshness source policies.
- Champion/challenger promotion criteria.

## Independence Boundary
- Can be built on current forecast inputs and models without ingest pipeline changes.

## Deliverables
- Configurable uncertainty and abstention policy implementation.
- Forecast response metadata for confidence tier and abstention reason.
- Tests for normal-confidence, low-confidence, and abstained forecast paths.

## Exit Criteria
- Every served forecast has explicit uncertainty metadata.
- High-uncertainty cases produce deterministic abstention/degradation behavior.
- API and service tests cover policy branches.
