# Brief 11 — A' v1 spec writer

## Why this exists

A' is the chosen direction (ADR 0005). Before writing code, we need a tight, honest spec for the minimum shippable v1. The spec is a design-shaping tool, not a compliance document.

**Read in order:**
1. `pm/PM_CLAUDE.md`
2. `pm/north-star.md`
3. `pm/decisions/0004-phase-2-synthesis-and-revised-thesis.md`
4. `pm/decisions/0005-problem-chosen-a-prime.md`
5. `pm/research-log/2026-04-21-free-tier-architecture.md` (agent 09's architecture — binding)
6. `pm/research-log/2026-04-21-user-archetypes.md` (agent 10 — first archetype is conservation land trusts via LTA WRN)

## Goal

Produce `docs/SPEC-A-prime-v1.md` — the minimum shippable v1 of Fire Stewardship Agent, targeted at conservation land trusts as the first archetype, that ships by end of Q2 2026 on free-tier infra.

## What v1 must do (non-negotiable)

- User signs in (Clerk / Supabase Auth / similar free tier). No account gating beyond that.
- User creates one or more AOIs (upload GeoJSON / draw on map / paste coords). Attach a human-readable name ("Spring Creek Preserve") and optional description ("~2,000 ac, mixed conifer, recently treated 2024").
- User configures per-AOI rules: alert-distance threshold (e.g. 25 km), quiet hours, notification channels (email via Resend / webhook).
- Background: GitHub Actions cron (every 15 min) polls FIRMS per bucket and matches to AOIs.
- On a matching detection that passes the gate, generate an L2-style situation brief via Gemini 2.5 Flash-Lite through AI Gateway, with structured output: summary, distance, direction, context (weather, authority perimeters if available, the site's prior events on file), recommended watch items, uncertainty notes.
- Deliver the brief via the user's chosen channel. Link back to a per-AOI page with map + brief history.
- All source code MIT / Apache-2 or equivalent, published to GitHub.

## What v1 is explicitly NOT

- Not a spread forecast (cut per ADR 0004 / agent 09 cut list)
- Not a denoiser UI / review queue (cut)
- Not multi-org / multi-tenancy beyond one-user-many-AOIs
- Not a mobile app (responsive web only)
- Not a replacement for Watch Duty alerts — it's a stewardship tool, not a public-safety tool

## Deliverable structure

Write `docs/SPEC-A-prime-v1.md`:

- `## One-page overview` — thesis, first archetype, positioning line
- `## User stories (JTBD-framed)` — 6–10 concrete stories, each: as a [archetype], when [trigger], I want [capability], so that [outcome]. Each story has acceptance criteria.
- `## Core flows` — sign-up, create AOI, configure rules, receive brief, review history. One numbered flow per.
- `## Data model` — tables, columns, PostGIS geometry columns. Aim for <10 tables. Match agent 09's target.
- `## API surface` — `/api/aoi/*`, `/api/brief/*`, `/api/mcp/*` (v2 hook). REST. Each endpoint: method, path, request schema, response schema.
- `## LLM brief format` — the exact structured-output schema the briefs follow, with a worked example grounded in real-sounding (but clearly illustrative) detection + polygon + weather context. This is the most important section — the product IS the brief.
- `## Scope boundaries` — list of "v1 does NOT do X" items, each with 1-line rationale.
- `## Acceptance for v1 launch` — a numbered checklist that defines "v1 is done": one archetype well-served end-to-end, infra running at claimed cost, canonical positioning sentence on the landing page, at least one land-trust user has used it with real AOIs.
- `## Open questions` — anything the spec can't resolve without Vanyo input.

## Constraints

- ≤ 3,500 words total.
- Every acceptance criterion testable.
- Don't invent API endpoints you haven't designed the data model for.
- If a section depends on a Phase 1 / 2 finding, cite the research-log file.
- `pm/**` is permitted for Write; `docs/` is also permitted.

## Time budget

~45 min.
