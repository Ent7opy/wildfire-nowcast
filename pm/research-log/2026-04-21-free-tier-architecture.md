# Free-tier architecture for A + D + E — 2026-04-21

**Agent:** 09 — Free-tier architect + solo-footprint estimator
**Method:** WebFetch of current pricing pages (2026-04-21), read of current repo (`CLAUDE.md`, `docker-compose.yml`, `railway.toml`), cost math at 3 scale points. Raw citations in `signals/2026-04-21-free-tier-raw.md`.
**Verdict:** **Achievable at target** (~$0/mo for 50 users / 100 AOIs). **Marginal at 10× scale** (~$20–25/mo). Biggest single risk: Vercel Hobby's non-commercial clause interpretation for donation-funded tools.

## Reference architecture (A + D + E)

**Principle:** no always-on compute, no continuous global ingest. All fire data is fetched on demand from FIRMS per AOI bbox, on a 15-minute cron per bucket of AOIs. State is tiny (AOIs + dedupe + short-TTL detection cache). The LLM reasoning step is **gated** so 90%+ of polls produce no LLM call.

```
                    ┌────────────────────────────────────────┐
                    │ GitHub Actions cron (every 15 min)     │
                    │   - reads active AOIs from Neon        │
                    │   - buckets AOIs by tile/region        │
                    │   - invokes Vercel function per bucket │
                    └──────────────────┬─────────────────────┘
                                       ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ Vercel Functions (Node/Python runtime, fluid compute)       │
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
    │ (+PostGIS)   │   │ → Gemini Flash │   │ NASA MAP_KEY │
    │ scale-to-0   │   │    Lite default│   │              │
    └──────────────┘   └────────────────┘   └──────────────┘

Notifications: Resend (free 3k/mo) OR Slack/Discord webhook (free)
UI: Vercel-hosted Next.js (Cache Components; mostly static + RSC)
Optional: Cloudflare R2 for cached FIRMS snapshots
```

**Data flow (AOI poll):**
1. GH Action cron fires every 15 min; POSTs to `/api/aoi/poll` with a bucket of AOIs sharing a region.
2. Function calls FIRMS `area/csv/<KEY>/VIIRS_NOAA20_NRT/<bbox>/1`. One call per ~5°×5° bucket covers dozens of user AOIs.
3. Parsed detections intersected with each AOI polygon in PostGIS (`ST_Intersects`).
4. For AOIs with new detections that pass distance/confidence thresholds AND the per-AOI "don't wake me for noise" rules, append to `aoi_event`.
5. **Gate:** if event is genuinely new (dedupe via content hash over last 24h) and LLM-worthy (≥2 pixels OR VIIRS FRP > 5 MW OR first detection near AOI), invoke `/api/aoi/brief`.
6. Brief function calls Gemini 2.5 Flash-Lite via AI Gateway with structured-output schema (summary, distance, direction, confidence, recommended action). Notifies via Resend / webhook.

**State (Neon):** `users`, `aois` (PostGIS polygon), `aoi_rules`, `aoi_events`, `firms_cache`. Expected <50 MB at 500 users/1000 AOIs — inside Neon Free 0.5 GB.

**Auth:** Clerk free tier (10k MAU) or Supabase Auth. BYO-Gemini-key option lets heavy users avoid Gateway spend entirely.

**MCP / E:** same Vercel functions expose `list_active_fires`, `get_aoi_history`, `subscribe_aoi` as MCP tools (`@modelcontextprotocol/sdk`) + REST. No extra infra.

**D:** `@earthtools/firms` — TS/Python library bundling the watermark + dedup + industrial-masking logic from `ingest/firms_client.py` + `ingest/firms_ingest.py`. Published npm/PyPI. Zero runtime cost.

## Cost model

Assumptions (target — 50 users / 100 AOIs):
- Average polygon 500 km²; buckets collapse to ~20 FIRMS calls per cron
- Cron every 15 min → 96/day → ~2,880 cron invocations/mo
- 20 FIRMS calls × 96/day = 1,920 transactions/day ≈ 80k/mo (vs 5k/10-min limit → headroom)
- Gate passes ~5% of polls → ~400 LLM briefs/mo
- LLM prompt: 2k in + 400 out. Gemini 2.5 Flash-Lite: 400 × (2000 × $0.10/1M + 400 × $0.40/1M) = **~$0.14/mo**
- Vercel function invocations: ~30k/mo = 3% of Hobby budget
- Vercel GB-hours: 30k × 1.5s × 1 GB = 12.5 GB-hr = 12.5% of 100 GB-hr budget
- Neon CU-hours: ~6.5 CU-hr/mo with scale-to-zero = 7% of 100 CU-hr budget
- Neon egress: ~200 MB/mo = 4% of 5 GB
- GitHub Actions on public repo: unlimited cron minutes

| Scale | Users / AOIs | Vercel | Neon | LLM | FIRMS | Total |
|---|---|---|---|---|---|---|
| (a) small | 10 / 20 | $0 | $0 | ~$0.03 | $0 | **$0** |
| (b) target | 50 / 100 | $0 | $0 | ~$0.14 | $0 | **~$0/mo** (AI GW $5 credit absorbs) |
| (c) scale | 500 / 1000 | ~$0 (borderline GB-hr) | ~$19 (outgrows 0.5 GB → Launch plan) | ~$1.50 | still bucketable | **~$20–25/mo** |

**Comparison:** current Railway stack (api + ingest_scheduler + Postgres + Redis + worker + titiler + pg_tileserv) ≈ **$30–50/mo steady**, more under bursts. Savings ~$25/mo + eliminates always-on failure modes.

**LLM sensitivity:** Gemini Flash (not Lite): ~$0.70/mo target. If gating removed entirely: ~$3/mo target → ~$30/mo at scale. **Gate layer is load-bearing.**

## Free-tier risk table

| Service | Free-tier ceiling | First to blow | Migration |
|---|---|---|---|
| Vercel Hobby | 100 GB-hrs, 1M invocations, **non-commercial use** | Commercial-use clause if donations flow | Pro $20/mo; or public API → Cloudflare Workers |
| Neon Free | 0.5 GB, 100 CU-hrs | Storage as `aoi_events` grows (~2.5M events) | Neon Launch $19/mo (10 GB) |
| Cloudflare Workers | 100k req/day | Tile / MCP traffic at scale | Workers Paid $5/mo |
| R2 | 10 GB, egress free | Cached FIRMS unbounded | $0.015/GB-month |
| AI Gateway | $5/mo credit | Non-gated or verbose briefs | BYOK, or buy credits |
| GitHub Actions | ≥5-min cron, best-effort | Cron drift during GH incidents | Vercel Cron Hobby 1 job fallback |
| FIRMS | 5k tx / 10 min | Un-bucketed small AOIs | Request quota; bucket coalescing (designed in) |
| Gemini API | RPM limits | Burst after wildfire event → parallel briefs | AI Gateway paid pass-through |

## Cut list (from current repo)

| Path / subsystem | Action | Rationale | Confidence |
|---|---|---|---|
| `ingest/orchestrator.py` 1,396 LOC + `ingest_scheduler` service | **CUT** | Continuous global ingest incompatible with free tier; replaced by per-AOI cron | high |
| `ml/spread/` full stack (v1+v2+v3+calibration+champion-challenger) | **CUT** | ~22k LOC solo-unmaintainable, duplicates NOAA/Technosylva, never reliably ran, v3 has no gate | high |
| `ml/denoiser/` 8.9k LOC + drift monitor + `api/model_registry.py` + `api/routes/ignition.py` | **CUT or REWRITE small** | Latest gate fails (F1 0.22); prod miscalibrated (95% drop). Replace with simple confidence/FRP/industrial-mask filter in the library. | high |
| `api/routes/archive.py` + `ArchiveRangeScrubber.tsx` | **CUT** | Demo surface; 5/10 recent commits are archive bugfixes | high |
| `ingest/*_authority_ingest.py` (NIFC/WFIGS/CWFIS/CopernicusEMS, ~2k LOC) | **CUT** | Duplicates gov't feeds, never run in prod (#325) | high |
| Industrial sources full taxonomy (ingest + builder + taxonomy + coverage) | **CUT pipeline; KEEP static no-go GeoJSON in library** | Value is in the static mask, not the ingestion | high |
| Fuels / LULC / LFMC / DFMC / lightning / drought / HRRR / terrain / weather ingests | **CUT** | Auxiliary for spread; spread cut → these cut | high |
| `api/routes/forecast.py` + `api/forecast/` + SSE | **CUT** | Spread endpoint | high |
| `api/routes/risk.py` stub | **CUT** | Never implemented | high |
| `api/routes/exports.py` + `api/exports/` + PDF/PNG worker | **CUT** | Dormant | high |
| `api/routes/assistant.py` + `AIChatAssistant.tsx` | **CUT** | Chatbot bolt-on, AI-decorative | high |
| `ReviewQueuePanel.tsx` + review queue endpoints + tables | **CUT** | Feeds denoiser HITL; denoiser cut | high |
| docker-compose `worker`, `ingest_scheduler`, `titiler`, `tiles` | **CUT** | Not free-tier compatible; UI renders FIRMS directly as MapLibre GeoJSON | high |
| docker-compose `redis` + RQ | **CUT** | No async queue needed; GH Actions + Vercel cron replace it | high |
| `railway.toml`, `railway.ingest.toml`, `Dockerfile.ingest` | **CUT** | Target is Vercel + Neon, not Railway | high |
| `ingest/firms_client.py`, `ingest/firms_ingest.py`, `ingest/firms_backfill.py` | **MIGRATE → `@earthtools/firms` library (D)** | Genuinely good code; the core artifact | high |
| `api/aois/`, `api/routes/aois.py` 354 LOC, `ingest/aoi_watch.py` | **KEEP + REWRITE as Vercel functions** | Cleanest primitive; foundation of A | high |
| `api/notifications.py` 397 LOC | **MIGRATE — simplify** | Keep webhook + rate-limit; drop SMTP + HMAC for v1 (use Resend) | med |
| `api/data_status.py`, `api/core/meteoalarm_provider.py` | **KEEP small; expose via MCP (E)** | Freshness + MeteoAlarm are useful MCP tools | med |
| `ui/src/map/*`, `FireMap.tsx`, `WatchlistDashboard.tsx` | **MIGRATE — Next.js + Cache Components** | Keep MapLibre + Deck.GL; drop titiler/pg_tileserv deps | high |
| `api/migrations/` (64 revs) | **REWRITE** | Collapse to <10 tables | high |
| `models/denoiser_v2/` (~100 runs), `models/spread_v3/` | **CUT (archive to R2 or git-lfs if preserving)** | Large binaries, none promoted | high |

**LOC impact:** ~40k LOC Python + ~2k LOC React CUT; ~3k LOC MIGRATED; ~2k LOC KEPT. **The pivot deletes more code than it adds.**

## Things we'd need to build new

1. **AOI polygon CRUD + rules UI** (Next.js RSC) — ~2 weeks. Pause / snooze / quiet hours / distance thresholds.
2. **LLM brief generator with structured output + gate rules** (AI SDK + Zod) — ~3 days. Gate is load-bearing.
3. **GitHub Actions cron workflow** hitting `/api/aoi/poll?bucket=...` — ~1 day. Bucket hashing is the key decision.
4. **`@earthtools/firms` library** (TS + Python) — extract existing client + tests; publish CI. ~1 week.
5. **MCP server** wrapping library + Neon state — `@modelcontextprotocol/sdk` + Vercel functions. ~4 days.
6. **Notification adapters** (Resend, Slack/Discord, ntfy.sh) — ~2 days.
7. **BYO-Gemini-key flow** (encrypted column per user) — ~2 days. Insulates against LLM cost blow-up.

## Verdict

**Achievable.** Target scale (50u/100 AOIs, 1 notification/AOI/week) lands at **~$0/mo** (AI Gateway $5 free credit absorbs ~$0.14 Flash-Lite spend). All tiers at 5–15% of ceilings.

**Marginal at (c) 500u/1000 AOIs:** ~$20–25/mo, driven by Neon Launch upgrade. Outside $0–10 strict but inside spirit if occasional upgrade steps are accepted.

**Single biggest risk: Vercel Hobby non-commercial clause.** Donation-funded public tool could be challenged. Mitigations ranked: (1) Vercel Pro $20/mo once donations exist, (2) public API → Cloudflare Workers, (3) get written Vercel confirmation.

**Single biggest cut: the spread-forecasting stack cascade** (spread + auxiliary ingests + forecast route + denoiser chain feeding it) — removes ~30k LOC and retires the ingest_scheduler, worker, redis, titiler, pg_tileserv containers in one move. This is *the* decision that takes the project from Railway-$30–50/mo to Vercel+Neon Free.
