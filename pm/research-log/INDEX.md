# Research log index

This directory contains scout / pm research notes. They are NOT decisions or contracts; they're substrate for future synthesis. For decisions, see `pm/decisions/` (ADRs). Raw evidence cited from these notes lives in `pm/signals/`.

## Pivot-era research (2026-04-21 / 04-22)

External scouting and Stage 0–2 dev logs that informed the A' pivot.

| File | Date | Summary |
|---|---|---|
| `2026-04-21-ai-leverage.md` | 2026-04-21 | Catalogs candidate AI-native leverage points that survive the "strip the AI layer" usefulness test. |
| `2026-04-21-critique-a.md` | 2026-04-21 | Adversarial critique that argued for killing Candidate A (non-profit AOI agent) in its ADR-0003 form. |
| `2026-04-21-critique-d.md` | 2026-04-21 | Adversarial critique that argued for folding Candidate D ("FIRMS done right" substrate) into A as a small byproduct library. |
| `2026-04-21-free-tier-architecture.md` | 2026-04-21 | Free-tier architecture sketch for A+D+E with cost math at three scale points; calls out Vercel Hobby commercial-clause risk. |
| `2026-04-21-github.md` | 2026-04-21 | GitHub adjacency scan of ~40 wildfire / FIRMS / VIIRS / fire-spread repos worth watching. |
| `2026-04-21-non-na-geography.md` | 2026-04-21 | Non-North-America regional gap scan (Mediterranean Europe, etc.) for stewardship-user demand. |
| `2026-04-21-reddit.md` | 2026-04-21 | Reddit ethnography across local subs surfacing "where is this smoke coming from?" pain pattern. |
| `2026-04-21-repo.md` | 2026-04-21 | Pre-pivot repo archaeology inventorying ~17 subsystems and their survivability through a narrow pivot. |
| `2026-04-21-twitter.md` | 2026-04-21 | X / Twitter ethnography on professional wildfire voice patterns via Playwright + Google site-search. |
| `2026-04-21-user-archetypes.md` | 2026-04-21 | Validation of non-profit user archetypes for Candidate A with public URLs for every named entity. |
| `2026-04-21-stage0-scaffold.md` | 2026-04-21 | Stage 0 dev log: Next.js 16 scaffold on `pivot/a-prime` with locked framework versions. |
| `2026-04-21-stage1-aoi-crud.md` | 2026-04-21 | Stage 1 dev log: AOI CRUD + Neon schema with PGlite tests. |
| `2026-04-22-stage2-firms-cron.md` | 2026-04-22 | Stage 2 dev log: FIRMS poll, AOI matcher, GH Actions cron, with PostGIS testcontainer integration tests. |

## Post-launch audits (2026-05-07)

Targeted audits of shipped surfaces for type safety, auth, perf, a11y, and security.

| File | Date | Summary |
|---|---|---|
| `2026-05-07-app-api-type-audit.md` | 2026-05-07 | Type and input-validation audit across all 14 `app/api/` route handlers and the shared `_lib/handle.ts`. |
| `2026-05-07-cron-auth-audit.md` | 2026-05-07 | Audit of the shared-bearer auth path on `/api/aoi/poll` — the only Clerk-bypassing route. |
| `2026-05-07-dashboard-a11y-audit.md` | 2026-05-07 | Accessibility audit of every authed dashboard page plus the public share view via JSX read + mental keyboard simulation. |
| `2026-05-07-poll-route-perf-audit.md` | 2026-05-07 | Static performance audit of `/api/aoi/poll` (526 LOC) and its downstream matcher / dispatch entry points. |
| `2026-05-07-token-endpoint-security-audit.md` | 2026-05-07 | Security audit of the five public token-bearing endpoints landed in Stages 6–7 (share + notify control links). |

## Brainstorms and post-mortems (2026-05-07)

Reflections, framings, and rolled-back attempts that did not become decisions.

| File | Date | Summary |
|---|---|---|
| `2026-05-07-catalog-exhaustion.md` | 2026-05-07 | Self-reflection on what the loop should do when the obvious chore catalog runs out. |
| `2026-05-07-incident-classes.md` | 2026-05-07 | Brainstorm of ten incident classes with a symptom → diagnostic → mitigation → rollback → recovery template. |
| `2026-05-07-launch-readiness-path.md` | 2026-05-07 | Path-forward note splitting launch-readiness items into LOOP-flippable vs Vanyo-blocked. |
| `2026-05-07-observability-gap.md` | 2026-05-07 | Brainstorm reframing the gap exposed by PR #439 as observability rather than runbook coverage. |
| `2026-05-07-poll-route-simplify-attempt.md` | 2026-05-07 | Post-mortem on a rolled-back simplification of `/api/aoi/poll` — best safe reduction was -55 LOC (10.5%), below the >15% threshold. |
| `2026-05-07-v1-stop-line.md` | 2026-05-07 | Brainstorm framing (not making) the v1 vs v1.1 stop-line decision. |
| `2026-05-07-weather-context-options.md` | 2026-05-07 | Survey of source options for the brief `weather_note` field that has been hardcoded null since Stage 3. |
