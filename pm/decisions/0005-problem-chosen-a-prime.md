# 0005 — Problem chosen: A' (Fire Stewardship Agent)

**Date:** 2026-04-21
**Status:** Accepted
**Stakeholder sign-off:** Vanyo — "go"

## Decision

Wildfire Nowcast will pivot to **A' — Fire Stewardship Agent**:

> *A free, open, AI-native fire intelligence agent for people whose relationship to land is stewardship — conservation trusts, protected-area managers, Indigenous fire crews, Firewise communities, LTER field scientists, environmental journalists — that watches their specific polygons and explains, in context, what is happening to their place.*

## Companion layering

- **D (FIRMS client library)** → 5-hour byproduct publish after A' v1 ships. No roadmap.
- **E (MCP / agent-consumable surface)** → thin side artifact over A''s backend, ships 2–3 weeks after A' v1.
- **C (Greek / Mediterranean)** → distribution accelerator via Natura 2000 archetype, not separate product.
- **F (AI-disinformation triage)** → parked until a named partner approaches.
- **B (insurance)** → rejected (ADR 0003).

## Positioning line (canonical)

> *"Free, open, AI-native fire intelligence for stewardship — depth over speed."*

This sentence is shared by: the Accedia end-of-May 2026 talk, the repo README, earth-tools.org/wildfire, the Land Trust Alliance Wildfire Resilience Network launch post.

## Load-bearing commitments

- Non-profit (donations only, no paid tiers)
- Free-tier infrastructure target: $0–10/month at 50 users / 100 AOIs scale
- AI-native leverage = **L2-style multi-source situation briefs** anchored in each site's history, NOT L1 threshold alerts
- First archetype well-served at v1: **conservation land trusts** via Land Trust Alliance Wildfire Resilience Network distribution
- Q2 2026 (end June 2026) deadline: v1 shipping

## Phase 3 dispatch

Three agents dispatched in parallel to produce build-ready artifacts:

1. **Brief 11 — A' v1 spec writer** (`docs/SPEC-A-prime-v1.md`)
2. **Brief 12 — Architecture-to-code plan** (`docs/pivot-architecture.md`, cut/migrate/keep list, migration sequence)
3. **Brief 13 — LTA Wildfire Resilience Network launch post drafter** (draft held until v1 ships; drafted now as a design-shaping tool)

## Deliverables into the main repo (not just PM workspace)

This is the first ADR where artifacts leave `pm/` and land in `docs/` + code. A' is the committed direction; PM workspace continues to track evolution but the spec, architecture plan, and code belong in the project proper.
