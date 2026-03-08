# Task 10: Docs And Contract Integrity

## Objective
Align repository documentation with actual runtime contracts so operators and contributors can trust docs as executable guidance.

## Primary Targets
- `/Users/vanyoivanov/Projects/wildfire-nowcast/docs/README.md`
- `/Users/vanyoivanov/Projects/wildfire-nowcast/docs/OPS_RUNBOOK.md`
- `/Users/vanyoivanov/Projects/wildfire-nowcast/README.md`
- `/Users/vanyoivanov/Projects/wildfire-nowcast/ml/spread/service.py` (doc references)

## Scope
- Fix broken/missing internal doc references and stale command guidance.
- Add explicit docs for spread uncertainty metadata, denoiser v2 decisions, and risk output semantics.
- Document stable contracts for ingest -> interpret -> deliver artifacts.

## Out Of Scope
- Behavioral code changes in core model/ingest APIs.
- New model or ingestion features.

## Independence Boundary
- Documentation-only task; no runtime logic changes required.

## Deliverables
- Corrected doc links and contract sections.
- Updated runbook decision trees for forecast/risk/denoiser outputs.
- Quick-start sections that map to current Makefile/runtime behavior.

## Exit Criteria
- No broken internal documentation references remain.
- Runbook steps map directly to existing commands and endpoints.
- Core output contracts are documented in one discoverable location.

