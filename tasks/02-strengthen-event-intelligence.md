# Task 02: Strengthen Event Intelligence

## Objective
Upgrade detection-to-front/event association to better represent true fire continuity and persistence.

## Primary Targets
- `/Users/vanyoivanov/Projects/wildfire-nowcast/ml/denoiser/eventize.py`
- `/Users/vanyoivanov/Projects/wildfire-nowcast/ml/denoiser/label_v2.py`
- `/Users/vanyoivanov/Projects/wildfire-nowcast/ingest/firms_ingest.py`

## Scope
- Replace coarse hash-bucket linkage with explicit spatial-temporal association logic.
- Improve handling of persistence-heavy non-wildfire sources during association.
- Quantify effect on label quality and event continuity metrics.

## Out Of Scope
- Denoiser model architecture changes.
- Spread model changes.

## Independence Boundary
- Can be delivered using existing ingestion and inference pipelines.

## Deliverables
- Configurable association implementation with deterministic IDs.
- Comparison report (baseline vs updated association quality).
- Tests for idempotency, event stability, and static-source separation.

## Exit Criteria
- Event/front mapping is deterministic across reruns on identical input windows.
- Event continuity metrics improve without raising industrial-source merge errors.
- Targeted eventization + labeling tests pass.
