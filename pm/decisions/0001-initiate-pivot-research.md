# 0001 — Initiate pivot research via agent swarm

**Date:** 2026-04-21
**Status:** Accepted

## Context

Wildfire Nowcast was built for the Accedia IDC jury, not for users. Scope (detection + spread + weather + chatbot + archive) is too broad for solo maintenance. Q2 2026 re-scope goal requires a narrower, AI-native product that fits within the Earth Tools portfolio at `earth-tools.org`.

An interview-based research plan exists (`docs/wildfire-nowcast-research-plan.md`) but was never executed and is not feasible for a solo operator in the remaining Q2 window.

A competitive brief exists (`docs/competitive-brief.md`, March 2026) but was written pre-pivot and still assumes the broad "detection + spread + weather" scope. Its positioning recommendation ("ground truth for active fires") is not binding on the pivot.

## Decision

Run a Phase 1 research swarm of 6 parallel agents:

1. Reddit ethnographer (Playwright, 12-month lookback, wildfire-adjacent subs)
2. X / Twitter ethnographer (Playwright + WebSearch fallback, fire professionals + researchers)
3. GitHub adjacency scout (open-source wildfire tools and their issue trackers)
4. Non-North-America geographic gap scout (Mediterranean, Australia, Amazon, SE Asia)
5. AI-native leverage scout (where LLM / agent infra changes wildfire tooling economics)
6. Internal repo archaeologist (what's load-bearing vs. over-built in the current codebase)

Each agent produces a condensed log in `research-log/` and raw evidence in `signals/`. PM_CLAUDE synthesises into `backlog.md` and surfaces to Vanyo for problem selection before any build work begins.

## Consequences

- `docs/competitive-brief.md` remains as a reference; will be marked `superseded` if evidence contradicts it.
- `docs/wildfire-nowcast-research-plan.md` is not cancelled — interviews may be used after narrowing.
- PM workspace at `pm/` is committed to the repo (not gitignored).
- Phase 2 (problem selection) and Phase 3 (design/spec) are gated on this swarm completing.
