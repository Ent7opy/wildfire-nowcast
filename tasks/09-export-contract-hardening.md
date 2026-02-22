# Task 09: Export Contract Hardening

## Objective
Harden export endpoints and background export jobs to produce consistent, reproducible decision artifacts at operational scale.

## Primary Targets
- `/Users/vanyoivanov/Projects/wildfire-nowcast/api/routes/exports.py`
- `/Users/vanyoivanov/Projects/wildfire-nowcast/api/exports/worker.py`
- `/Users/vanyoivanov/Projects/wildfire-nowcast/api/tests/test_exports_endpoint.py`

## Scope
- Normalize export data contracts between sync endpoints and async worker jobs.
- Add large-window pagination/chunking and deterministic ordering for CSV/GeoJSON exports.
- Include artifact metadata (query window, thresholds, model/version timestamps) in export outputs.

## Out Of Scope
- Risk scoring algorithm changes.
- Forecast model behavior changes.

## Independence Boundary
- Works with current API/storage models and does not require ingest/model changes.

## Deliverables
- Unified export payload schema and row accounting.
- Fixed worker/sync behavior parity for fire and risk exports.
- Tests for contract consistency, large exports, and metadata completeness.

## Exit Criteria
- Export endpoints and worker outputs agree on row counts and schema.
- Large export requests remain bounded and stable.
- Each exported artifact contains enough metadata for reproducibility.

