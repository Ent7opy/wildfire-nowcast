# Task 01: Stabilize Data Realism

## Objective
Replace synthetic fuel/moisture proxy generation with provider-backed ingestion for NDVI/LFMC/DFMC/precip features.

## Primary Targets
- `/Users/vanyoivanov/Projects/wildfire-nowcast/ingest/fuels_ingest.py`
- `/Users/vanyoivanov/Projects/wildfire-nowcast/ingest/config.py`
- `/Users/vanyoivanov/Projects/wildfire-nowcast/ml/spread_features.py`

## Scope
- Implement network/provider-backed fetch + transform path.
- Persist normalized feature cubes with reproducible metadata (provider, run time, bbox, resolution).
- Load these features in spread inputs without synthetic fallback in the default production path.

## Out Of Scope
- Forecast gating policy changes.
- Promotion/evaluation threshold changes.

## Independence Boundary
- Can be completed without changing API contracts or model training code.

## Deliverables
- Provider-backed ingest path with retries, timeout handling, and deterministic file naming.
- Fuel feature metadata contract and loader validation.
- Unit/integration tests for ingest success, partial provider failure, and cache reuse.

## Exit Criteria
- Default runtime path no longer emits synthetic NDVI/LFMC/DFMC/precip values.
- Fuel feature metadata is persisted and queryable for each run.
- Targeted ingest/feature tests pass.
