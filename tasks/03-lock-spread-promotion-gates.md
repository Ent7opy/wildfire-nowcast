# Task 03: Lock Spread Promotion Gates

## Objective
Formalize champion/challenger promotion rules using reproducible, slice-aware, data-backed thresholds.

## Primary Targets
- `/Users/vanyoivanov/Projects/wildfire-nowcast/ml/eval_spread_champion_challenger.py`
- `/Users/vanyoivanov/Projects/wildfire-nowcast/configs/spread_champion_challenger.yaml`
- `/Users/vanyoivanov/Projects/wildfire-nowcast/ml/eval_spread_calibration.py`

## Scope
- Standardize gate metrics and required slices (horizon, region bucket, season/time).
- Derive thresholds from historical backtests and evaluation distributions.
- Emit auditable pass/fail reasons and rollback triggers per run.

## Out Of Scope
- New model architecture training work.
- Input feature ingestion changes.

## Independence Boundary
- Operates on existing model outputs and evaluation artifacts only.

## Deliverables
- Gate policy spec with metric definitions and threshold derivation method.
- Updated eval outputs with decision rationale payloads.
- Tests for gate behavior on threshold edges and missing-metric scenarios.

## Exit Criteria
- Promotion recommendation is deterministic for identical evaluation inputs.
- Gate outputs explain exactly why a challenger passes or fails.
- Thresholds are tied to observed project history, not hardcoded defaults.
