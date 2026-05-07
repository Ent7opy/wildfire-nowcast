# Wildfire Nowcast — Architecture

> **Current as of 2026-05-07.** Reflects `master` @ `b4aa923` with Stages 0–8 merged. The legacy Python / Docker / Railway stack has been deleted; what's described below is what's running on Vercel + Neon today.
>
> Strategic context lives in `pm/north-star.md` and ADRs `pm/decisions/0001`–`0007`. Product spec: `docs/SPEC-A-prime-v1.md`. The earlier pivot-execution version of this document (Stages 0–7 plan) is preserved in git history if needed (`git log -- docs/pivot-architecture.md`).

---

## 1. System overview

Wildfire Nowcast is a free, open fire-stewardship agent for users who care about specific polygons — conservation trusts, Natura 2000 site managers, Firewise communities, Indigenous fire crews, LTER scientists, environmental journalists. They draw an Area of Interest (AOI), the system polls NASA FIRMS every 15 minutes for thermal anomalies, matches detections against their polygons, and — when a gate passes — generates a contextual situation brief via an LLM and emails it to them.

The pipeline:

```
GitHub Actions cron (*/15 min)
    └─→ POST /api/aoi/poll  (CRON_SECRET bearer)
          ├─→ enumerate active 5°×5° buckets from `aois.region_bucket`
          ├─→ per bucket:
          │     1. fetchAreaCsv()   — NASA FIRMS area-CSV API
          │     2. matchDetectionsToAois()  — PostGIS ST_DWithin + dedupe
          │     3. for each new event: generateBriefForEvent()
          │           ├─→ evaluateGate()              — pure-TS pre-LLM filter
          │           ├─→ pre-fetch authority perimeter (NIFC/CWFIS)
          │           ├─→ generateBriefViaGateway()   — Vercel AI Gateway → Gemini
          │           └─→ INSERT aoi_briefs, UPDATE aoi_events.last_brief_at
          │     4. dispatchBrief() → Resend email + signed action tokens
          └─→ pruneOldDetections() — 14-day retention sweep
```

Everything else — dashboard, AOI editor, brief view, share links, exports, sign-in — is Next.js App Router pages and API routes on the same Vercel deployment. Per-user authentication via Clerk; AOIs are scoped by `users.id` (Clerk user id is the PK).

Free-tier infra ceiling: Vercel Hobby + Neon Free + Resend Free + Vercel AI Gateway $5 starter credit. Cost target ≈ $0/mo at v1 scale (50 users, 100 AOIs).

---

## 2. Stack

- **Next.js 16** (App Router), TypeScript 5, hosted on Vercel.
- **Drizzle ORM** + **PostgreSQL 16 / PostGIS 3.5** on **Neon** (autoscale-to-zero).
- **PGlite** for unit tests; **`@testcontainers/postgresql`** with `postgis/postgis:16-3.5` for spatial integration tests.
- **GitHub Actions cron** (`.github/workflows/firms-poll.yml`) — `*/15 * * * *`, POSTs `/api/aoi/poll` with `Authorization: Bearer ${CRON_SECRET}`.
- **Vercel AI Gateway** (`@ai-sdk/google` via `AI_GATEWAY_API_KEY`) — `gemini-2.5-flash-lite` by default; `generateObject` with Zod-validated structured output.
- **Resend** (`RESEND_API_KEY`) for notification email; webhook channel reserved but not implemented (always logged as `skipped, channel_not_implemented`).
- **Clerk** (`CLERK_SECRET_KEY`, `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY`, `CLERK_WEBHOOK_SIGNING_SECRET`) for auth + Svix-verified webhook that syncs `users` on `user.created`/`updated`/`deleted`.
- **Vitest** (unit + integration), **eslint 9**, **pnpm 10**.
- **`vercel.json`** pins `framework: nextjs`, `outputDirectory: .next` — no monorepo subdir, the Next app is at the repo root.

---

## 3. Components

### 3.1 Ingest — `lib/firms/`

- `client.ts` — `fetchAreaCsv()`: pure HTTP + CSV parse against `https://firms.modaps.eosdis.nasa.gov/api/area/csv/...`. No DB writes. Lazy `FIRMS_MAP_KEY` read; missing key returns `{ ok: false, code: "config_missing" }` rather than throwing.
- `buckets.ts` — `getActiveBuckets()`, `bucketToBbox()`. The 5°×5° tile key (e.g. `5x5:W125_N35`) is derived from each AOI's centroid at create time and stored as `aois.region_bucket` so the cron can coalesce FIRMS calls.
- `matcher.ts` — inserts detections into `firms_detections` with the static-industrial-mask flag computed inline (PostGIS `ST_Intersects` or turf-style point-in-polygon under PGlite), then UPSERTs `aoi_events` keyed by a `dedupe_hash` over `(aoi_id, bucket, rounded centroid, source, 24h-window)`. Re-poll within the window extends `last_seen_at` and bumps `detection_count`.
- `dedupe.ts`, `industrial-seed.ts`, `prune.ts`, `freshness.ts` — supporting modules. `prune.ts` implements the 14-day retention sweep called at the end of every poll.

### 3.2 Spatial — PostGIS + turf fallback

The two-backend repository pattern: every DB module works on both Neon+PostGIS (production) and PGlite (unit tests). Spatial code falls back to turf.js when `lib/db/client.ts` reports the connection is non-PostGIS. The pattern is implemented in `lib/firms/matcher.ts` and `lib/geo/polygon.ts`. PGlite has no GIST indexes; tests use bounding-box pre-filters and accept the cost.

### 3.3 AI — `lib/ai/`

- `gate.ts` — pure-TypeScript pre-LLM filter implementing SPEC §Flow 6. Rejects briefs that don't meet FRP / proximity / "first-ever" / 24h-rebrief-suppression conditions; returns a `GateReason` enum so we can audit pass-rate.
- `prompt.ts` — pinned template `v1` (versioned in `aoi_briefs.prompt_version`).
- `authority/` — Stage 8 authority-perimeter pre-fetch. `sources.ts` lists the confirmed key-free GeoJSON endpoints (NIFC for US, CWFIS for Canada); `fetch.ts` pulls the most recent feature near the detection; the orchestrator folds it into the prompt context. ICNF (Portugal / Mediterranean) is deferred — see §6 R6 and `pm/blockers.md`.
- `gateway.ts` — `generateBriefViaGateway()`: thin wrapper over `@ai-sdk/google` with model + cost telemetry. `AI_GATEWAY_API_KEY` missing returns `config_missing`.
- `schema.ts` — Zod schema for the brief payload; structured output is validated twice (once by `generateObject`, once on read for defence-in-depth).
- `render.ts` — deterministic Markdown renderer that turns the validated payload into the email/web body.
- `generate.ts` — `generateBriefForEvent(db, eventId)`: loads AOI/event/rules, runs the gate, generates, validates, renders, and persists transactionally to `aoi_briefs` + bumps `aoi_events.last_brief_at`.

### 3.4 Dispatch — `lib/notify/`

- `dispatch.ts` — `dispatchBrief(db, briefId)` is the single entry point called by the poll route after a brief lands. Loads brief + AOI + rules + user email, resolves channels (`aoi_rules.notify_channels`, fallback to `users.email`), and per-channel runs idempotency check → pause/quiet-hours gate → send → persist `notifications_log` row.
- `resend.ts` — Resend SDK wrapper.
- `markdown.ts`, `footer.ts` — email body + per-recipient footer with action links.
- `action-tokens.ts`, `actions.ts` — Stage 7 signed bearer tokens for snooze / pause / unsubscribe / feedback links. The token IS the auth — recipient clicks link in their email, the route redeems the row in `notify_action_tokens`. A forwarded email does not grant the recipient permission to mutate the AOI (token is bound to `(brief, channel, target, action)`).
- Webhook channel rows are persisted as `skipped, channel_not_implemented` — slot reserved for future Slack/Discord delivery.

### 3.5 Auth — `lib/auth/`, `middleware.ts`, `app/api/webhooks/clerk/`

- `middleware.ts` — Clerk middleware over `/api/aoi/*`, `/api/brief/*`, `/api/export/*`, `/api/me`, `/dashboard/*`. Public: `/`, `/sign-in/*`, `/sign-up/*`, `/api/aoi/poll` (CRON_SECRET), `/api/webhooks/clerk` (Svix-signed), `/brief/share/[token]` (capability URL). When `CLERK_SECRET_KEY` is unset the middleware no-ops so the app boots; route handlers' `requireUserId()` returns 503 `service_unavailable` instead.
- `lib/auth/context.ts` — `requireUserId()` is the single seam every authed route hits via `withDb`. JIT user-row provisioning runs on first authed request; the Clerk webhook keeps it in sync on email change / deletion.

### 3.6 UI — `app/`

- **Marketing** — `app/page.tsx`, `app/layout.tsx`. Public landing.
- **Sign-in / sign-up** — `app/sign-in/[[...rest]]/page.tsx`, `app/sign-up/[[...rest]]/page.tsx` (Clerk catch-all routes).
- **Dashboard** — `app/dashboard/page.tsx` (AOI list), `app/dashboard/aoi/new/page.tsx` (polygon draw via MapLibre), `app/dashboard/aoi/[id]/page.tsx` (AOI detail + rules form + brief history), `app/dashboard/brief/[id]/page.tsx` (brief view). Components in `app/dashboard/_components/` (aoi-list, aoi-map, freshness-banner, rules-form, share-toggle).
- **Sharing** — `app/brief/share/[token]/page.tsx` is a public read-only brief view gated by a 32-byte share token (`lib/share/token.ts`). Toggleable per-brief on the dashboard.
- **API endpoints**:
  - `POST /api/aoi/poll` — cron entry (CRON_SECRET).
  - `GET/POST /api/aoi`, `GET/PATCH/DELETE /api/aoi/[id]`, `PATCH /api/aoi/[id]/rules`, `GET /api/aoi/[id]/export` — AOI CRUD + per-AOI rules + per-AOI export.
  - `POST/DELETE /api/brief/[id]/share` — share-token mint / revoke.
  - `GET /api/export/aois.geojson`, `GET /api/export/briefs.csv` — Stage 6 user-data exports.
  - `GET/POST/PATCH /api/notify/{snooze,pause,unsubscribe,feedback}/[token]` — Stage 7 email action redemptions.
  - `GET /api/me` — current Clerk user echo.
  - `POST /api/webhooks/clerk` — Svix-verified user sync.

---

## 4. Data flow (single FIRMS poll tick)

```
GH Actions cron
    │  POST /api/aoi/poll  Authorization: Bearer $CRON_SECRET
    ▼
app/api/aoi/poll/route.ts
    │
    ├── getActiveBuckets(db)                    → ["5x5:W125_N35", ...]
    │
    ├── for each bucket:
    │     │  job_runs INSERT (parent + per-bucket child rows)
    │     │
    │     ├── fetchAreaCsv({source, bbox})      → FIRMS rows (or freshness outcome)
    │     ├── matchDetectionsToAois(...)        → INSERT firms_detections
    │     │                                       UPSERT aoi_events (dedupe_hash)
    │     │
    │     ├── for each new aoi_event (status='new'):
    │     │     ├── generateBriefForEvent(db, event.id)
    │     │     │     ├── evaluateGate()        → pass | reject(reason)
    │     │     │     ├── fetchAuthorityPerimeter()  (Stage 8, NIFC/CWFIS only)
    │     │     │     ├── generateBriefViaGateway(prompt) → Zod-validated payload
    │     │     │     ├── render() → markdown
    │     │     │     └── INSERT aoi_briefs, UPDATE aoi_events.last_brief_at
    │     │     │
    │     │     └── dispatchBrief(db, brief.id)
    │     │           ├── resolve channels from aoi_rules.notify_channels
    │     │           ├── check pause / quiet_hours / target idempotency
    │     │           ├── mintActionToken() per (brief, channel, target, action)
    │     │           ├── sendEmail() via Resend
    │     │           └── INSERT notifications_log
    │     │
    │     └── job_runs UPDATE (status, outcome, retry_pending, counters)
    │
    └── pruneOldDetections(db, retentionDays=14)
```

Everything is idempotent at every persistence boundary: `firms_detections` deduped on `(source, acq_date, acq_time, lat, lon)`; `aoi_events` UPSERT on `dedupe_hash`; `notifications_log` partial unique index on `(brief_id, channel, target_hash) WHERE status IN ('sent','skipped')`; action tokens bound to `(brief, channel, target, action)`. A retried cron tick produces no double-sends.

---

## 5. Schema — `db/schema/index.ts`

| Table | Stage | Purpose |
|---|---|---|
| `users` | 1 | Clerk user id is the PK; populated by Clerk webhook + JIT path. |
| `aois` | 1 | Polygon, bbox, centroid, `region_bucket` (5°×5° tile key), `area_ha`. GIST indexes on `polygon` + `bbox`. |
| `aoi_rules` | 1 | Per-AOI `distance_buffer_km`, `min_confidence`, `min_frp_mw`, `quiet_hours`, `paused_until`, `notify_channels`. |
| `firms_detections` | 2 | 14-day cache of FIRMS pixels. Deduped on `(source, acq_date, acq_time, lat, lon)`. `is_industrial_static` set at insert. |
| `aoi_events` | 2 | Per-AOI matched events. `dedupe_hash` UPSERT key; `status ∈ {new, open, closed}`; `last_brief_at` for the gate. |
| `industrial_mask_static` | 2 | Static catalog of industrial / volcanic heat sources (gas flares, refineries) seeded from `db/seeds/`. |
| `job_runs` | 2 / 7 / 8 | One row per `/api/aoi/poll` invocation + per-bucket children. `outcome` (Stage 8: success / rate_limited / network_error / timeout / partial), `retry_pending`, `detections_pruned` (Stage 7). |
| `aoi_briefs` | 3 | Generated briefs. `payload` (Zod-validated JSON), `rendered_markdown`, `gate_reason`, `model`, `prompt_version`, `cost_usd_est`, `share_token`, `last_notified_at`. |
| `notifications_log` | 4 | One row per send attempt. `status ∈ {sent, failed, skipped, config_missing}`. |
| `notify_action_tokens` | 7 | Bearer tokens for email snooze / pause / unsubscribe / feedback. |
| `brief_feedback` | 7 | Per-(brief, recipient) "Was this helpful?" capture. |

Migrations are hand-authored SQL in `db/migrations/0000_init.sql` … `0006_stage8.sql`; PostGIS specifics (geometry types, GIST indexes, partial-WHERE indexes) live in the SQL, the Drizzle schema mirrors them. Each migration ships with a `*.test.sql` PGlite-compatible variant.

---

## 6. Risks (current state)

| # | Risk | Status | Mitigation |
|---|---|---|---|
| R1 | **Vercel Hobby "non-commercial" clause** could flag a donation-funded NGO tool. | Unresolved — `pm/blockers.md` 2026-05-07. | Pre-launch written confirmation pending; Pro $20/mo upgrade path verified as fallback. |
| R2 | **GitHub Actions cron drift** during a GH incident → AOIs go silent. | Detected by `job_runs` gap query (Stage 8 freshness). | Per-bucket `outcome` + `retry_pending` surfaced in the dashboard freshness banner so users see staleness rather than silently degraded results. |
| R3 | **FIRMS rate limit** (5,000 tx / 10 min) breached during a major event with many overlapping buckets. | Bucket coalescing already in place; `job_runs.outcome='rate_limited'` recorded. | Per-bucket circuit-breaker skips one cycle and retries next tick; users see one stale 15-min window, not an outage. |
| R4 | **LLM cost blow-up** if the gate is too permissive. | Gate enforced in `lib/ai/gate.ts` with branch tests; `aoi_briefs.cost_usd_est` telemetered. | Kill-switch via missing `AI_GATEWAY_API_KEY` (briefs degrade to threshold-only notifications via `lib/notify/` fallback path). Per-AOI rebrief suppression in the gate caps fan-out. |
| R5 | **Neon Free 0.5 GB exceeded** as `firms_detections` grows. | 14-day prune sweep runs at the end of every poll (`pruneOldDetections`); `job_runs.detections_pruned` recorded. | Storage envelope at 500 users / 1,000 AOIs ≈ 45 MB — well inside Free. Neon Launch $19/mo is the documented escalation. |
| R6 | **Mediterranean briefs lack authority context.** Stage 8 ships NIFC + CWFIS only; ICNF / fogos.pt has no public polygon feed. | Unresolved — `pm/blockers.md` 2026-05-07. | Vanyo to ask ICNF directly, or accept ICNF deferral to v1.1. Briefs still ship without authority context for non-US/CA AOIs; the renderer omits the section cleanly. |

---

## 7. Deferred to v1.1

Drawn from `pm/backlog.md` Stage 8 candidates and the 2026-05-07 product review. None of these block v1 launch.

- **ICNF / fogos.pt authority source for Mediterranean AOIs** — see R6.
- **Path B authority pre-fetch via LLM tool-call** — Stage 8 shipped Path A (deterministic pre-fetch). Tool-calling lets the model decide which sources to consult per detection; deferred until Path A's coverage gaps are quantified.
- **Weather / fuels context** in briefs — explicitly cut from v1 (ADR 0005). May return as a single concise weather-context paragraph if a free, no-key-required source proves usable.
- **Webhook delivery (Slack / Discord)** — schema supports `notify_channels: [{type:"webhook",...}]`; dispatcher records `skipped, channel_not_implemented`. Implementation is a single new file in `lib/notify/`.
- **Richer AOI editor** — current MapLibre draw covers the v1 spec (≤ 100,000 ha simple polygons). Multi-polygon edit, import-from-GeoJSON UI, and snap-to-features are post-launch.
- **MCP API surface** — `@earthtools/firms` library extraction + MCP server (`/api/mcp/*`) was Stage 8b in ADR 0005 and remains out-of-critical-path. Planned as a copy of existing assets, not new compute.
- **Second-archetype outreach** (beyond conservation trusts) — pm-only work tracked in `pm/backlog.md`.

---

**End of document.**
