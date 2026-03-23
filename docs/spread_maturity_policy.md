# Spread Maturity Policy

This project uses two spread-model maturity stages:

- `mvp_operational`: working end-to-end operational baseline.
- `science_grade`: science-quality promotion target.

For the full specification — hard stop definitions, stage warning format, science debt register
schema, metric thresholds, and the `science_grade` promotion checklist — see
**[docs/spread_gate_requirements.md](spread_gate_requirements.md)**.

## Non-Negotiable Hard Stops

`STOP`/`BLOCKER` always apply, regardless of stage:

- authoritative source missing for required inputs (`STOP-SOURCE-001`)
- train/infer feature-contract mismatch (`STOP-CONTRACT-001`)
- invalid geospatial alignment (`STOP-GEO-001`)
- fake/fabricated data in production paths

## Stage-Gap Warnings

`WARNING` is stage-aware and must include:

- mitigation action
- tracking ID
- target stage (`science_grade` unless specified otherwise)

Warnings track science debt; they never override hard stops.

## Gate Report Contract

Spread gate reports must include:

- `maturity_stage`
- `hard_stops`
- `stage_warnings`
- `promotion_decision`
- `science_debt_register`
