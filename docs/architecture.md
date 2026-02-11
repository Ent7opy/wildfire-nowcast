# Architecture

## System Model

The product follows a stable four-stage loop:

1. **Ingest**: collect and normalize external geospatial signals
2. **Interpret**: derive nowcast, spread outlook, and risk estimates
3. **Deliver**: expose map layers, queries, and exports
4. **Learn**: evaluate outcomes and improve future runs

## Core Artifacts

- Time-stamped observations
- Derived context features
- Forecast/risk products
- User-facing map and export outputs
- Evaluation records for continuous improvement

## Architecture Principles

- Separate data collection from model interpretation.
- Keep serving interfaces stable while internals evolve.
- Preserve lineage from raw signal to final product.
- Design for partial-data operation without hiding degradation.

## Evolution Strategy

Implementation details may change, but the contracts above should remain stable:

- Inputs stay observable and versioned.
- Outputs stay interpretable and time-bounded.
- Evaluation remains first-class, not optional.
