# Wildfire Nowcast — Pivot Architecture & Code Plan

> **HISTORICAL — DO NOT TREAT AS CURRENT STATE.**
>
> This document was the build-ready execution plan for the A' pivot (Stages 0–7). Stage 7 (cutover) merged on `master` and the legacy Python / Docker / Railway stack has been deleted from the repo. References below to "the current Railway/Docker stack", `apps/nextjs/`, `pivot/a-prime`, "old Railway services", `Dockerfile.*`, `docker-compose.yml`, `railway.toml`, `nixpacks.toml`, etc. describe the FROM side of the now-completed migration. They are preserved for historical context only.
>
> For current architecture see `CLAUDE.md` (stack section), `README.md`, and `docs/SPEC-A-prime-v1.md`. The Next.js app lives at the repo root, not under `apps/nextjs/`.

**Date:** 2026-04-21
**Status:** Historical — plan executed; superseded by current root-level Next.js app
**Authority:** ADR 0005, PM_CLAUDE, research-log `2026-04-21-free-tier-architecture.md` (cut list is BINDING), research-log `2026-04-21-repo.md`

> **Amendment 2026-04-21 — Vercel project reuse + root layout.**
>
> Original plan proposed a separate Vercel project with `apps/nextjs/` subdir. Revised after Vanyo confirmed the existing `wildfire-nowcast.vercel.app` project (UI-only; Railway backend currently offline):
>
> 1. **Reuse the existing `wildfire-nowcast` Vercel project.** Don't create a second one.
> 2. **All pivot work happens on a `pivot/a-prime` branch.** Vercel's git integration auto-generates preview URLs per branch; master stays on the current service-offline UI until cutover.
> 3. **Next.js 16 scaffold lands at the repo root**, not at `apps/nextjs/`. Old Vite `ui/` and Python `api/` / `ml/` / `ingest/` subdirs stay in place until their respective cut stages; the new Next.js app does not reference them.
> 4. **Cutover (Stage 7)** becomes: merge `pivot/a-prime` → `master`, Vercel production URL flips, rollback via Vercel's one-click "Promote previous deployment."
> 5. Earth Tools iframe env var already points at `wildfire-nowcast.vercel.app` — no change needed on the Earth Tools side at cutover.
>
> All Stage numbering and reversibility claims below remain valid; only the "separate Vercel project" language is superseded.

---
**Scope:** Ordered, reversible migration from the current Railway/Docker stack to a Vercel + Neon free-tier stack, landable by end of Q2 2026 by a solo developer.

This document is the execution plan. The *what* (the A' product) is in `docs/SPEC-A-prime-v1.md`. The *why* is in `pm/decisions/0005-problem-chosen-a-prime.md`. Here we say *how* and *in what order*.

---

## 1. End-state architecture

Recapping agent 09's reference architecture. No re-derivation.

```
                    ┌────────────────────────────────────────┐
                    │ GitHub Actions cron (every 15 min)     │
                    │   - reads active AOIs from Neon        │
                    │   - buckets AOIs by 5°x5° tile         │
                    │   - invokes Vercel function per bucket │
                    └──────────────────┬─────────────────────┘
                                       ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ Vercel Functions (Next.js 16 App Router, Fluid Compute)     │
 │                                                             │
 │  /api/aoi/poll         — per-bucket FIRMS fetch + match     │
 │  /api/aoi/brief        — LLM-reasoned brief (gated)         │
 │  /api/aoi/crud         — user-facing AOI CRUD               │
 │  /api/mcp/*            — MCP / REST surface (candidate E)   │
 │  /api/firms/*          — thin library wrapper (candidate D) │
 └──────────┬──────────────────┬──────────────────┬────────────┘
            ▼                  ▼                  ▼
    ┌──────────────┐   ┌────────────────┐   ┌──────────────┐
    │ Neon Postgres│   │ Vercel AI GW   │   │ FIRMS API    │
    │ (+PostGIS)   │   │ → Gemini 2.5   │   │ NASA MAP_KEY │
    │ scale-to-0   │   │  Flash-Lite    │   │              │
    └──────────────┘   └────────────────┘   └──────────────┘

Notifications: Resend (free 3k/mo) OR Slack/Discord webhook (free)
UI: Vercel-hosted Next.js 16 (Cache Components; mostly RSC + static)
Optional: Cloudflare R2 for cached FIRMS snapshots
```

**Runtime contracts:**
- Every scheduled function invocation is idempotent under AOI_id + cron_tick_hash.
- Each LLM call is schema-validated (Zod) and produces a row in `aoi_briefs` before any notification is sent.
- No background workers, no queues, no long-running connections, no tile servers. If a requirement re-introduces one, escalate to PM.

**Target cost:** $0/mo at 50 users / 100 AOIs (AI Gateway $5 free credit absorbs LLM spend). Ceiling monitoring via Vercel usage alerts at 60% of each budget line.

---

## 2. Starting state (honest)

Current repo inventory: `pm/research-log/2026-04-21-repo.md`. Condensed load-bearing facts:

- **Docker Compose**: 8 services (api, ui, db, redis, migrate, titiler, worker, ingest_scheduler, tiles). Plus Railway runtime targeting api + ingest_scheduler with `Dockerfile.api` and `Dockerfile.ingest`.
- **Alembic**: 64 revisions under `api/migrations/versions/`. Most predate any production usage.
- **Code volume**: ~40k LOC Python + ~2k LOC React slated for CUT; ~3k LOC MIGRATED; ~2k LOC KEPT.
- **Open issues**: 10 from the 2026-04-04 audit batch, all tagged "never ran in prod". Zero user-reported issues.
- **Moat narrative risk**: denoiser's latest gate_report is `"pass": false` (precision 0.124, F1 0.22). Prod bypasses registry. We cannot lean on it.
- **What is genuinely precious**: `ingest/firms_client.py`, `ingest/firms_ingest.py`, `ingest/firms_backfill.py`; `api/aois/` + `api/routes/aois.py`; `ui/src/map/*`; `api/notifications.py` (simplified); `api/data_status.py`; `api/core/meteoalarm_provider.py`.

Cutover principle: the old stack stays 100% functional on `master` until Stage 7. The new stack lives at the repo root on the `pivot/a-prime` branch and deploys to the existing `wildfire-nowcast` Vercel project as branch preview URLs. Master continues to deploy the current service-offline UI until cutover. No big-bang rewrite. (See amendment at top of this document.)

---

## 3. The collapsed data model

Target: ≤10 tables. Neon Postgres 16 + PostGIS 3.5 (Neon supports PostGIS). New Alembic baseline — we do NOT carry the 64-revision history forward.

```sql
-- 3.1 users (Clerk-issued id is the PK)
CREATE TABLE users (
  id              TEXT PRIMARY KEY,              -- Clerk user_id
  email           TEXT NOT NULL,
  display_name    TEXT,
  gemini_api_key_enc BYTEA,                      -- optional BYOK, encrypted at rest
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  deleted_at      TIMESTAMPTZ
);

-- 3.2 aois — the stewardship polygon
CREATE TABLE aois (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id         TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  name            TEXT NOT NULL,
  polygon         GEOMETRY(MultiPolygon, 4326) NOT NULL,
  bbox            GEOMETRY(Polygon, 4326) NOT NULL,     -- denormalized envelope
  centroid        GEOMETRY(Point, 4326) NOT NULL,       -- for MCP "nearest" lookups
  region_bucket   TEXT NOT NULL,                        -- e.g. "5x5:W15_N45" (hashing key for cron)
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  archived_at     TIMESTAMPTZ
);
CREATE INDEX aois_polygon_gix   ON aois USING GIST (polygon);
CREATE INDEX aois_bbox_gix      ON aois USING GIST (bbox);
CREATE INDEX aois_region_bucket ON aois (region_bucket) WHERE archived_at IS NULL;

-- 3.3 aoi_rules — per-AOI monitoring contract
CREATE TABLE aoi_rules (
  aoi_id              UUID PRIMARY KEY REFERENCES aois(id) ON DELETE CASCADE,
  min_frp_mw          REAL NOT NULL DEFAULT 5.0,
  min_confidence      TEXT NOT NULL DEFAULT 'nominal',  -- low|nominal|high
  distance_buffer_km  REAL NOT NULL DEFAULT 5.0,        -- "near" AOI = within buffer
  quiet_hours         JSONB,                            -- {tz, start_hour, end_hour}
  paused_until        TIMESTAMPTZ,
  notify_channels     JSONB NOT NULL DEFAULT '[]'::jsonb, -- [{type:"email"|"webhook", target:...}]
  updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 3.4 firms_detections — short-TTL detection cache (NOT a historical archive)
-- Rows older than 14 days are pruned by the cron.
CREATE TABLE firms_detections (
  id              BIGSERIAL PRIMARY KEY,
  source          TEXT NOT NULL,                 -- VIIRS_NOAA20_NRT | MODIS_NRT etc
  detected_at     TIMESTAMPTZ NOT NULL,
  geom            GEOMETRY(Point, 4326) NOT NULL,
  frp_mw          REAL,
  confidence      TEXT,
  bright_ti4      REAL,
  bright_ti5      REAL,
  scan_track      REAL,
  content_hash    TEXT NOT NULL,                 -- dedupe across overlapping bucket fetches
  ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX firms_detections_hash ON firms_detections (content_hash);
CREATE INDEX firms_detections_gix        ON firms_detections USING GIST (geom);
CREATE INDEX firms_detections_detected   ON firms_detections (detected_at);

-- 3.5 aoi_events — matches between detections and AOIs
CREATE TABLE aoi_events (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  aoi_id          UUID NOT NULL REFERENCES aois(id) ON DELETE CASCADE,
  detection_id    BIGINT NOT NULL REFERENCES firms_detections(id) ON DELETE CASCADE,
  distance_m      REAL NOT NULL,                 -- min distance from detection to polygon
  inside_polygon  BOOLEAN NOT NULL,
  detected_at     TIMESTAMPTZ NOT NULL,
  dedupe_key      TEXT NOT NULL,                 -- aoi_id + spatial_cell + day, last 24h
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX aoi_events_dedupe ON aoi_events (dedupe_key);
CREATE INDEX aoi_events_aoi           ON aoi_events (aoi_id, detected_at DESC);

-- 3.6 aoi_briefs — LLM-generated situation briefs (the product deliverable)
CREATE TABLE aoi_briefs (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  aoi_id          UUID NOT NULL REFERENCES aois(id) ON DELETE CASCADE,
  event_id        UUID REFERENCES aoi_events(id) ON DELETE SET NULL,
  model           TEXT NOT NULL,                 -- e.g. "gemini-2.5-flash-lite"
  prompt_version  TEXT NOT NULL,                 -- pinned, checked into repo
  payload         JSONB NOT NULL,                -- structured output (summary, distance, direction, action)
  markdown        TEXT NOT NULL,                 -- rendered brief for notification / UI
  cost_usd_est    NUMERIC(10,6),
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX aoi_briefs_aoi ON aoi_briefs (aoi_id, created_at DESC);

-- 3.7 notifications_log — delivery receipts + rate-limit state
CREATE TABLE notifications_log (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  aoi_id          UUID NOT NULL REFERENCES aois(id) ON DELETE CASCADE,
  brief_id        UUID REFERENCES aoi_briefs(id) ON DELETE SET NULL,
  channel         TEXT NOT NULL,                 -- email|webhook
  target_hash     TEXT NOT NULL,                 -- sha256(target) for rate-limit key
  status          TEXT NOT NULL,                 -- sent|failed|throttled
  error           TEXT,
  sent_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX notif_log_rate ON notifications_log (aoi_id, target_hash, sent_at DESC);

-- 3.8 industrial_mask_static — no-go polygons (shipped as static, not ingested)
-- Loaded once at migrate time from a GeoJSON in the repo.
CREATE TABLE industrial_mask_static (
  id              SERIAL PRIMARY KEY,
  category        TEXT NOT NULL,                 -- flare|refinery|landfill|etc
  geom            GEOMETRY(Polygon, 4326) NOT NULL
);
CREATE INDEX industrial_mask_gix ON industrial_mask_static USING GIST (geom);

-- 3.9 job_runs — cron/function observability
CREATE TABLE job_runs (
  id              BIGSERIAL PRIMARY KEY,
  job_name        TEXT NOT NULL,                 -- poll_bucket|brief|notify|prune
  bucket          TEXT,                          -- region_bucket, if relevant
  started_at      TIMESTAMPTZ NOT NULL,
  finished_at     TIMESTAMPTZ,
  status          TEXT NOT NULL,                 -- ok|error|partial
  error           TEXT,
  stats           JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX job_runs_recent ON job_runs (job_name, started_at DESC);

-- 3.10 (reserved) aoi_history_snapshot — OPTIONAL, Stage 6+. Not built in v1.
```

**Mapping to A' flow:**
- `users`, `aois`, `aoi_rules` — who/what we're watching (Stages 1).
- `firms_detections` + `aoi_events` — the detection → AOI match loop (Stage 2).
- `aoi_briefs` — the LLM-leverage deliverable (Stage 3).
- `notifications_log` — delivery + rate-limit (Stage 4).
- `industrial_mask_static` — static no-go filter at write time (Stage 2, seeded in Stage 0).
- `job_runs` — solo-operator observability (Stage 2 onward).

Storage envelope at target scale (500u / 1000 AOIs / 14-day TTL): ~45 MB. Inside Neon Free 0.5 GB.

---

## 4. Migration sequence

Eight stages. Each stage ends with a green CI and a reversible commit. No stage takes more than 5 solo-days. Old stack stays functional through Stage 6.

### Stage 0 — Scaffold the new app in-repo

- **Goal:** `apps/nextjs/` Next.js 16 App Router skeleton deploys to a separate Vercel project with preview green. Zero existing code touched.
- **Files touched (add):**
  - `apps/nextjs/package.json`, `apps/nextjs/tsconfig.json`, `apps/nextjs/next.config.ts`
  - `apps/nextjs/app/layout.tsx`, `apps/nextjs/app/page.tsx` (marketing landing placeholder)
  - `apps/nextjs/app/api/health/route.ts` (`GET` returns `{ ok: true, build: … }`)
  - `apps/nextjs/.env.example` (NEON_DATABASE_URL, CLERK_*, FIRMS_MAP_KEY, AI_GATEWAY_KEY, RESEND_API_KEY)
  - `.github/workflows/nextjs-ci.yml` (typecheck + lint + build)
  - `docs/pivot-architecture.md` (this file)
  - `vercel.json` at repo root scoped to `apps/nextjs/` via `rootDirectory`
- **Files touched (delete):** none.
- **Reversibility:** yes — revert in 1 commit.
- **Verification:** GH Actions green; Vercel preview `/api/health` returns 200; `cd apps/nextjs && npx tsc --noEmit && npm run lint && npm run build` clean.

### Stage 1 — AOI CRUD on Neon

- **Goal:** A logged-in user can create, list, edit, archive an AOI with a polygon. Data persists to Neon. PostGIS queries work.
- **Files touched (add):**
  - `apps/nextjs/db/schema.sql` (the ≤10-table schema from §3, split into one Alembic-style migration `001_init.sql`)
  - `apps/nextjs/db/migrate.ts` (runs on Vercel build via `postinstall` or a one-shot GH Action; uses `node-postgres`)
  - `apps/nextjs/lib/db.ts` (Neon HTTP driver `@neondatabase/serverless`, pooled for functions)
  - `apps/nextjs/lib/auth.ts` (Clerk)
  - `apps/nextjs/app/api/aois/route.ts` — `GET list`, `POST create`
  - `apps/nextjs/app/api/aois/[id]/route.ts` — `GET one`, `PATCH update`, `DELETE archive`
  - `apps/nextjs/app/(app)/aois/page.tsx` (RSC list)
  - `apps/nextjs/app/(app)/aois/new/page.tsx` (polygon draw via MapLibre `@mapbox/mapbox-gl-draw`)
  - `apps/nextjs/lib/firms/region-bucket.ts` (deterministic 5°×5° tile hashing — lat/lon floor)
  - `apps/nextjs/tests/aois.e2e.test.ts` (Playwright) + `apps/nextjs/tests/region-bucket.test.ts` (unit)
- **Files touched (delete):** none.
- **Reversibility:** yes — revert in 1 commit; Neon DB drop is manual but cheap.
- **Verification:** Playwright test creates AOI, reads it back, archives it. `region_bucket` computed correctly for known lat/lon. Alembic-equivalent `db/migrate.ts` idempotent.

### Stage 2 — Port FIRMS ingest as Vercel functions + GH Actions cron

- **Goal:** Every 15 minutes, GH Actions cron fetches FIRMS for all unique `region_bucket`s that have active AOIs, matches detections against polygons, writes `aoi_events` with dedupe, prunes detections older than 14 days. Static industrial mask applied at write time.
- **Files touched (add):**
  - `apps/nextjs/lib/firms/client.ts` (TS port of `ingest/firms_client.py` — just the HTTP + CSV parse, no watermark DB; stateless)
  - `apps/nextjs/lib/firms/match.ts` (PostGIS `ST_Intersects` + `ST_DWithin` in one SQL using `aoi_rules.distance_buffer_km`)
  - `apps/nextjs/lib/firms/industrial-mask.ts` (loaded once per cold start from `industrial_mask_static`)
  - `apps/nextjs/app/api/cron/poll/route.ts` (`POST` receives `{ bucket, aoi_ids }`; verified by `CRON_SECRET`)
  - `apps/nextjs/app/api/cron/dispatch/route.ts` (`POST` from GH Action; reads active AOIs, groups by bucket, fan-outs to `poll`)
  - `.github/workflows/firms-cron.yml` (`schedule: '*/15 * * * *'`; `curl` dispatch endpoint)
  - `apps/nextjs/scripts/seed-industrial-mask.ts` (one-shot: imports GeoJSON from `apps/nextjs/data/industrial_mask.geojson`)
  - `apps/nextjs/data/industrial_mask.geojson` (extracted from current `industrial_sources` table — one-time SQL dump; regenerated offline)
- **Files touched (delete):** none yet. Old ingest keeps running in parallel on Railway.
- **Reversibility:** yes — disable the cron workflow (one edit) or revert the three new files.
- **Verification:**
  - Unit: FIRMS client parses a fixture CSV correctly.
  - Integration: Playwright seeds an AOI over a known fire in a recent FIRMS response, triggers `/api/cron/dispatch`, asserts `aoi_events` row appears.
  - Observability: `job_runs` rows with `status=ok` and non-empty `stats`.
  - Budget check: 96 cron runs × ~20 bucket fan-outs × ~1.5s ≈ 48 GB-min/day → well under Vercel budget. Log invocation count for first 72h.

### Stage 3 — LLM brief generation (gated)

- **Goal:** When `aoi_events` has a new event passing the gate, generate a structured brief via Vercel AI Gateway → Gemini 2.5 Flash-Lite, validated by Zod, persisted to `aoi_briefs`.
- **Files touched (add):**
  - `apps/nextjs/lib/llm/schema.ts` (Zod: `{ summary, distance_desc, bearing, frp_band, action, confidence }`)
  - `apps/nextjs/lib/llm/prompt.ts` (pinned template `v1`; AOI history context from last 30d events)
  - `apps/nextjs/lib/llm/gate.ts` (rules: ≥2 pixels in 24h OR FRP>5MW OR first-ever detection near AOI; content_hash dedupe over 24h)
  - `apps/nextjs/lib/llm/client.ts` (`@ai-sdk/google` through AI Gateway; `generateObject` for structured output)
  - `apps/nextjs/app/api/aoi/brief/route.ts` (`POST { aoi_id, event_id }`; returns 202 if gate fails with reason)
  - `apps/nextjs/tests/gate.test.ts` (unit: every gate branch)
  - `apps/nextjs/tests/brief.e2e.test.ts` (fixture event → asserts schema-valid brief row)
- **Files touched (delete):** none.
- **Reversibility:** yes — removing the brief call from `poll` is one line.
- **Verification:**
  - Every schema field populated, no hallucinated numbers (bearing bucketed, distance bucketed).
  - Gate rejects ≥90% of synthetic noise events; passes ≥95% of synthetic real events.
  - Cost telemetry in `aoi_briefs.cost_usd_est`; weekly rollup query.

### Stage 4 — Notifications (Resend + webhook)

- **Goal:** A brief delivered via the AOI's configured channels (email via Resend OR Slack/Discord webhook). Rate-limited per `(aoi_id, target_hash)` with a 15-minute dedupe window (matching current `NOTIFICATION_RATE_LIMIT_SECONDS` semantics). Quiet-hours respected.
- **Files touched (add):**
  - `apps/nextjs/lib/notify/resend.ts`
  - `apps/nextjs/lib/notify/webhook.ts` (Slack-compat + Discord-compat payload shape)
  - `apps/nextjs/lib/notify/dispatch.ts` (reads `aoi_rules.notify_channels`; writes `notifications_log`; enforces `paused_until`/`quiet_hours`)
  - `apps/nextjs/app/api/aoi/notify/route.ts` (`POST { brief_id }`)
  - `apps/nextjs/tests/notify.test.ts`
- **Files touched (delete):** none.
- **Reversibility:** yes.
- **Verification:** Playwright with Resend test keys + webhook.site: brief generated → email + webhook received within 60s; second identical brief throttled; `paused_until` blocks delivery.

### Stage 5 — UI (map, AOI list, brief history)

- **Goal:** Product UI meeting `docs/SPEC-A-prime-v1.md` acceptance criteria. Next.js 16 Cache Components (PPR) for the marketing shell; RSC for authed pages; Deck.GL + MapLibre for the map.
- **Files touched (add):**
  - `apps/nextjs/app/(marketing)/page.tsx` (Cache Components `use cache`)
  - `apps/nextjs/app/(app)/layout.tsx` (Clerk-gated)
  - `apps/nextjs/app/(app)/aois/[id]/page.tsx` (AOI detail + brief history)
  - `apps/nextjs/components/Map.tsx` (MapLibre GL 5 + Deck.GL 9 — GeoJSON layer fed by an API route that returns the last 24h of events, no tile server)
  - `apps/nextjs/components/AoiEditor.tsx` (create/edit polygon)
  - `apps/nextjs/components/BriefCard.tsx`
  - `apps/nextjs/components/RulesForm.tsx` (min_frp, distance, quiet hours, channels)
- **Files touched (delete):** none.
- **Reversibility:** yes — feature-flagged under `/v2` path until cutover if needed.
- **Verification:** Playwright end-to-end: sign-up → draw AOI → seed fixture event → observe brief card + notification. Lighthouse performance ≥ 90 on marketing page.

### Stage 6 — The Big Cut

**Goal:** Delete the CUT subsystems from agent 09's list on `master`, in dependency order derived in §5. After each sub-stage the repo still builds and tests pass; the old Railway stack may no longer boot, but `apps/nextjs/` does.

Ordered sub-stages (each is one PR, each reversible via `git revert` in 1 commit):

- **6a — Spread stack & its consumers.** Delete `ml/spread/**`, `api/routes/forecast.py`, `api/forecast/**`, `ui/src/**/forecast*`, forecast SSE code. Unblocks weather/terrain/fuels cuts. (**~22k LOC**)
- **6b — Weather / terrain / fuels / LULC / LFMC / drought / lightning / HRRR / GFS ingests.** Delete `ingest/weather_ingest.py`, `ingest/terrain_features.py`, `ingest/fuels_ingest.py`, `ingest/lulc_worldcover_ingest.py`, `ingest/lfmc_ecland_ingest.py`, and related. Remove related configs under `configs/`.
- **6c — Perimeter authority ingests.** Delete `ingest/nifc_perimeters_ingest.py`, `ingest/cwfis_authority_ingest.py`, `ingest/wfigs_authority_ingest.py`, `ingest/copernicus_ems_authority_ingest.py`.
- **6d — Archive scrubber.** Delete `api/routes/archive.py`, `ui/src/components/ArchiveRangeScrubber*`, `api/firms/archive*`, `ingest/firms_backfill.py` (remaining after library extract in 7a). Remove RQ rate-limit wiring for archive.
- **6e — Denoiser + review queue + model registry + ignition.** Delete `ml/denoiser/**`, `api/model_registry.py`, `api/routes/ignition.py`, `api/routes/review.py`, `ui/src/components/ReviewQueuePanel*`, `models/denoiser_v2/**`, `models/spread_v3/**`. (Archive artifacts to R2 if preserving; git-lfs is *not* used in this repo today, so delete in place.)
- **6f — AI chat assistant.** Delete `api/routes/assistant.py`, `api/assistant/**`, `ui/src/components/AIChatAssistant*`, Gemini-chat env vars from `.env.example`.
- **6g — Industrial sources taxonomy & pipeline (keep static mask in new app).** Delete `ingest/industrial_sources_ingest.py`, `api/industrial_coverage.py`, industrial taxonomy builder scripts. The static GeoJSON seeded in Stage 2 is the replacement.
- **6h — Exports, risk stub, misc dormant routes.** Delete `api/routes/exports.py`, `api/exports/**`, `api/routes/risk.py`, export worker code.
- **6i — Old UI app.** Delete `ui/` (old Vite app). From this commit, `apps/nextjs/` is the only UI.
- **6j — Orchestrator + RQ worker + async queue + ingest_scheduler service.** Delete `ingest/orchestrator.py`, `ingest/scheduler*`, `api/worker.py`, RQ wiring, `Dockerfile.ingest`, `railway.ingest.toml`.
- **6k — Old API surface reduction.** Delete `api/routes/aois.py` (replaced by Next.js route), `api/notifications.py` (replaced), `api/data_status.py` (to be re-exposed via MCP later). Keep `api/core/meteoalarm_provider.py` only if MCP plan E will reuse it, else delete.

After each sub-stage: `make lint && make test` must be green on whatever remains of the old stack, OR — if a sub-stage breaks the old stack irretrievably (6j does) — CI passes only on the new `apps/nextjs/` pipeline.

- **Files touched:** see the appendix inventory table (§8).
- **Reversibility:** each sub-stage is a single PR; `git revert <sha>` restores it. After 6j the old Railway deployment cannot be booted; before 6j it can.
- **Verification:** CI green after each sub-stage; manual smoke on the old Railway preview after 6a–6i (still bootable), then switch to `apps/nextjs/` smoke for 6j onward.

### Stage 7 — Retire Railway + cutover DNS

- **Goal:** Production traffic moves to Vercel. Railway services stopped. DNS switched.
- **Steps:**
  1. Take a final Neon dump of whatever old data is worth preserving (AOIs from `api/aois/` that real users registered, if any). Import into new Neon DB.
  2. Promote `apps/nextjs/` Vercel project to production domain (swap A/CNAME in the DNS provider to Vercel; TTL lowered to 60s 24h prior).
  3. Stop Railway services via `railway down` (consistent with existing `a324697 Add Railway up/down controls` workflow).
  4. Delete `railway.toml`, `railway.ingest.toml`, `Dockerfile.api`, `Dockerfile.ingest`, `docker-compose.yml`, `docker-compose.*.yml`, `Makefile` targets that reference them (keep only the new `apps/nextjs/` make targets or drop Makefile entirely).
- **Files touched (delete):** see §8 appendix.
- **Reversibility:** **with-caveats**. For 24h, DNS can be flipped back to Railway if we preserve the Railway project rather than destroying it. After 24h (once Neon has diverged), rollback means re-seeding from the dump. Rollback window target: 24h.
- **Verification:** `/api/health` on new domain 200; synthetic AOI monitor alive for 2 full cron cycles (30 min); no 5xx rate spike in Vercel analytics over 6h.

### Stage 8 (companion) — Library D + MCP E

Out of critical path; drafted after Stage 7 per ADR 0005.

- **8a — `@earthtools/firms` library:** extract Stage-2 `lib/firms/*` into `packages/firms/` + Python mirror `packages/firms-py/`; publish to npm + PyPI. ~1 week.
- **8b — MCP server:** `apps/nextjs/app/api/mcp/*` with `@modelcontextprotocol/sdk` tools (`list_active_fires`, `get_aoi_history`, `subscribe_aoi`). ~4 days.

Shipping these is a copy of Stage-2/Stage-5 assets, not new compute — no architecture risk.

---

## 5. Dependency cut graph

Which cuts unblock which. Ordering in Stage 6 is derived from this.

```
spread stack (v1/v2/v3 + calibration + champion-challenger)
   ├── blocks → weather ingest             [cut 6a then 6b]
   ├── blocks → terrain ingest             [6b]
   ├── blocks → fuels/LULC/LFMC/drought    [6b]
   ├── blocks → lightning/HRRR/GFS         [6b]
   └── blocks → forecast route + SSE       [6a]

archive scrubber
   └── blocks → firms_backfill tail + archive rate-limit [6d]

denoiser v2
   ├── blocks → review queue (HITL)        [6e]
   ├── blocks → model registry             [6e]
   ├── blocks → ignition route             [6e]
   └── blocks → industrial taxonomy pipe   [6g]

orchestrator + RQ worker
   ├── depends on → every ingest job above being cut first
   └── blocks → redis container + Dockerfile.ingest + railway.ingest.toml [6j → 7]

old UI (ui/)
   └── blocks → titiler + pg_tileserv      [6i → 7]

Railway api + docker-compose
   └── depends on → 6j & 6k complete       [→ 7]
```

Read top-down: spread is the root cascade; cutting it is the highest-LOC single move. Orchestrator and the old UI are the sinks; they must go last inside Stage 6.

---

## 6. Risk register

Top five with detection, mitigation, rollback.

| # | Risk | Detection | Mitigation | Rollback |
|---|---|---|---|---|
| R1 | **Vercel Hobby "non-commercial" clause blocks launch.** Donation-funded public tool could be flagged. | Vercel email / account flag. | (a) Get written Vercel confirmation pre-launch. (b) Keep Pro $20/mo upgrade path verified. (c) Structure donations through Land Trust Alliance partner, not a direct SKU. | Upgrade to Pro in one click ($20/mo still within spirit); or move public API to Cloudflare Workers (the library/MCP surface is platform-neutral by design). |
| R2 | **GH Actions cron drift during GH incidents** → stewardship AOIs go silent when it matters. | `job_runs` gap-detection query (no `status=ok` in last 30 min) pings a webhook. | Redundant Vercel Cron Hobby (1 free cron job) as a hot-backup `/api/cron/dispatch` every 30 min. | Vercel Cron takes over automatically (both are idempotent). |
| R3 | **FIRMS rate limit (5k tx / 10 min) breached during a major fire event** — many overlapping buckets. | HTTP 429 from FIRMS tracked in `job_runs.stats`. | Deterministic 5°×5° bucket hashing already coalesces; add jitter + exponential backoff; escalate to FIRMS quota request if sustained. | Per-bucket circuit-breaker skips 1 cycle and retries next tick — users see one stale 15-min window, not an outage. |
| R4 | **LLM cost blow-up** — gate too permissive, or a real fire fan-outs parallel briefs for nearby AOIs. | AI Gateway daily spend alert + per-AOI rate limit on `aoi_briefs` (max 4/day/AOI). | Gate is load-bearing: extensive unit tests per §Stage 3. Per-AOI cap is hard-coded. BYOK flow (`users.gemini_api_key_enc`) lets power users self-pay. | Kill-switch env `LLM_ENABLED=false` degrades to threshold-only notifications; `aoi_briefs` rows skipped; `notifications_log` annotates `markdown=fallback`. |
| R5 | **Neon Free storage exceeded** before Stage 7 cutover when seeding historical AOI data. | Neon dashboard / `SELECT pg_database_size`. | 14-day TTL enforced by prune cron; reject any import > 250 MB; v1 explicitly not an archive. | Upgrade to Neon Launch $19/mo (still within spirit; flagged as known 10× scale cost). |

Lower-tier risks tracked but not top-5: Clerk 10k MAU ceiling (trivial to migrate to Supabase Auth), Resend 3k/mo (switch to webhooks exclusively), Cloudflare R2 unused in v1.

---

## 7. Rollback playbook

If Q2 2026 deadline is threatened, the **minimum-regret pause state** is:

1. **Through end of Stage 5**: master keeps the old Docker stack. `apps/nextjs/` is a separate Vercel project on a preview/subdomain. A Vanyo-sized pause means: stop merging to `master`, keep both stacks compilable, ship nothing. Zero user-facing change.
2. **During Stage 6**: stop at the last green sub-stage (6a–6i are all safely reversible via `git revert`; the repo after any one of them still boots the old Railway services until 6j). If forced to pause mid-Stage 6, revert to the last green sub-stage SHA and redeploy Railway from there.
3. **Between 6j and Stage 7**: the old Railway stack cannot boot (orchestrator is gone). Rollback from this point means `git revert` of 6j's PR specifically, which restores orchestrator + worker + redis + Dockerfile.ingest. Railway redeploy from the restored commit takes ~15 min. *This is the one-way-ish gate — target not to enter 6j until Stages 0–5 are stakeholder-signed-off.*
4. **Stage 7 and beyond**: Railway retired. Rollback = re-provision Railway from the pre-6j commit + Neon dump replay. 24h window with DNS TTL lowered.

Pause posture summary (per-Stage, until cutover):
- Stages 0–5: **fully reversible** (revert one branch; old stack untouched).
- Stage 6a–6i: **reversible per sub-stage** (revert one PR; old stack still bootable).
- Stage 6j–6k: **reversible with caveat** (revert 6j specifically to restore worker stack).
- Stage 7: **reversible 24h** (DNS + Neon dump; after that, divergence makes it regret-heavy).

---

## 8. Appendix — file inventory (execution checklist)

Mirrors agent 09's cut list with current LOC, target LOC, and the Stage 6 ticket that owns it. LOC numbers are from research-log `2026-04-21-repo.md`; exact counts may drift.

| Path / subsystem | Current LOC | Target LOC | Action | Stage 6 sub-stage / Stage | Ticket title |
|---|---|---|---|---|---|
| `ingest/orchestrator.py` + scheduler entrypoints | 1,396 | 0 | CUT | 6j | Remove ingest_scheduler + orchestrator |
| `ml/spread/**` (v1+v2+v3+calibration+champion-challenger) | ~22,000 | 0 | CUT | 6a | Remove spread forecasting stack |
| `ml/denoiser/**` + `api/model_registry.py` + ignition | ~9,400 | 0 | CUT | 6e | Remove denoiser + registry + ignition |
| `api/routes/archive.py` + `ArchiveRangeScrubber.tsx` + archive rate-limit | ~800 | 0 | CUT | 6d | Remove archive scrubber |
| `ingest/*_authority_ingest.py` (NIFC/WFIGS/CWFIS/CopernicusEMS) | ~2,000 | 0 | CUT | 6c | Remove perimeter authority ingests |
| Industrial sources pipeline + taxonomy + coverage route | ~1,200 | 0 (seed static GeoJSON) | CUT pipeline / KEEP mask | 6g | Replace industrial pipeline with static mask |
| Fuels / LULC / LFMC / drought / lightning / HRRR / GFS / terrain ingests | ~6,000 | 0 | CUT | 6b | Remove spread-aux ingests |
| `api/routes/forecast.py` + `api/forecast/**` + SSE | ~1,500 | 0 | CUT | 6a | Remove forecast route & SSE |
| `api/routes/risk.py` stub | 61 | 0 | CUT | 6h | Remove risk stub |
| `api/routes/exports.py` + `api/exports/**` + export worker | ~700 | 0 | CUT | 6h | Remove exports |
| `api/routes/assistant.py` + `AIChatAssistant.tsx` | ~670 | 0 | CUT | 6f | Remove chat assistant |
| `ReviewQueuePanel.tsx` + review endpoints + tables | ~800 | 0 | CUT | 6e | Remove review queue |
| `docker-compose.yml` (worker, ingest_scheduler, titiler, tiles, redis) | 265 | 0 | CUT | 7 | Retire docker-compose |
| `railway.toml`, `railway.ingest.toml`, `Dockerfile.ingest` | ~40 | 0 | CUT | 7 | Retire Railway config |
| `ingest/firms_client.py`, `ingest/firms_ingest.py`, `ingest/firms_backfill.py` | ~1,400 | ~600 (TS port in `apps/nextjs/lib/firms/`) | MIGRATE | Stage 2 (library form in Stage 8a) | Port FIRMS client to Next.js |
| `api/aois/**`, `api/routes/aois.py`, `ingest/aoi_watch.py` | ~600 | ~400 (Next.js routes) | MIGRATE | Stage 1 | Port AOI CRUD |
| `api/notifications.py` | 397 | ~200 (Resend + webhook, drop SMTP + HMAC) | MIGRATE | Stage 4 | Simplify notifications |
| `api/data_status.py` + `api/core/meteoalarm_provider.py` | ~300 | ~300 | KEEP (expose via MCP E) | Stage 8b | MCP server |
| `ui/src/map/**`, `FireMap.tsx`, `WatchlistDashboard.tsx` | ~2,000 | ~1,500 (Next.js components) | MIGRATE | Stage 5 | Port map UI |
| `api/migrations/versions/*` (64 revs) | ~5,000 | ~300 (one init migration in `apps/nextjs/db/`) | REWRITE | Stage 1 | Collapse schema |
| `models/denoiser_v2/**`, `models/spread_v3/**` | large binaries | 0 (archive to R2 if preserving) | CUT | 6e | Purge unused model artifacts |
| `ui/` (old Vite app) | ~15,000 | 0 | CUT | 6i | Retire old UI |

**LOC net:** deletes ~60k; migrates ~3k; keeps ~1k; adds ~4k new Next.js code. The pivot is net-deletion.

---

**End of plan.**
