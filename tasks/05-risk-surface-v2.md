# Task 05: Risk Surface V2

## Objective
Upgrade risk scoring from heuristic blends to a structured hazard-exposure-vulnerability (HEV) framework.

## Primary Targets
- `/Users/vanyoivanov/Projects/wildfire-nowcast/api/risk/grid.py`
- `/Users/vanyoivanov/Projects/wildfire-nowcast/api/routes/risk.py`
- `/Users/vanyoivanov/Projects/wildfire-nowcast/api/tests/test_risk_endpoint.py`

## Scope
- Define HEV component contracts and compute each component explicitly.
- Calibrate risk bins and category labels against observed outcomes/proxies.
- Preserve endpoint compatibility while improving explainability fields.

## Out Of Scope
- Spread model changes.
- Denoiser/eventization changes.

## Independence Boundary
- Can be implemented as a risk-layer refactor without changing ingest orchestration.

## Deliverables
- Risk computation module with HEV decomposition and documented formulas.
- API response fields for component contributions and risk category confidence.
- Validation report for calibration/discrimination by region and season.

## Exit Criteria
- Endpoint remains backward compatible for existing consumers.
- Risk categories are calibrated with documented thresholds.
- Risk endpoint tests cover HEV component integrity and schema.
