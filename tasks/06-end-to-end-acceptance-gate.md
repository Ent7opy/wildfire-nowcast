# Task 06: End-to-End Acceptance Gate

## Objective
Create an independent acceptance harness that measures current system reliability, latency, and safety behavior across ingest -> forecast -> API.

## Primary Targets
- `/Users/vanyoivanov/Projects/wildfire-nowcast/api/tests/`
- `/Users/vanyoivanov/Projects/wildfire-nowcast/ingest/tests/`
- `/Users/vanyoivanov/Projects/wildfire-nowcast/ml/tests/`
- `/Users/vanyoivanov/Projects/wildfire-nowcast/tasks/`

## Scope
- Add a baseline acceptance matrix and reproducible run commands.
- Validate degraded-data behavior and rollback drills in integration scenarios.
- Produce a single evidence artifact for release/promotion review.

## Out Of Scope
- Feature implementation changes in ingest/model/API business logic.

## Independence Boundary
- This task validates existing behavior and can run before any other feature task.

## Deliverables
- Acceptance matrix with explicit pass/fail thresholds.
- CI-equivalent local command set for repeatable checks.
- Operator checklist for incident rollback readiness.

## Exit Criteria
- Acceptance harness runs end-to-end from one command set.
- Outputs include machine-readable results and human-readable summary.
- Acceptance output includes release-ready evidence requirements.
