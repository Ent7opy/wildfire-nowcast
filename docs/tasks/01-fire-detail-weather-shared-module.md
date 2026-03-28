# Task: Extract weather point lookup into a shared module

## Goal

The weather point lookup function currently lives in the fire scoring module. Move it to a shared location so it can be called from both the risk grid and the new fire detail endpoint.

## Context

Read `api/fires/scoring.py` to find the function. Identify all existing callers. Move it to wherever makes sense architecturally given the existing module layout — avoid circular imports. Existing callers must continue working after the move.

This is a refactor only. No behaviour changes.

## Done when

- The function is importable from its new location
- All existing callers have been updated
- Tests pass
