# SPEC — Fire Stewardship Agent (A') v1

**Status:** Draft, tied to ADR 0005 (Accepted 2026-04-21).
**Target ship:** End of Q2 2026 (June 2026).
**Target infra cost:** ~$0/mo at 50 users / 100 AOIs (see `pm/research-log/2026-04-21-free-tier-architecture.md`).
**First archetype:** Conservation land trusts via Land Trust Alliance Wildfire Resilience Network (see `pm/research-log/2026-04-21-user-archetypes.md`, archetype 1).
**Positioning line:** *"Free, open, AI-native fire intelligence for stewardship — depth over speed."*

---

## One-page overview

**Thesis.** A' is a free, open, AI-native fire intelligence agent for stewardship-motivated users — conservation trusts, protected-area managers, Indigenous fire crews, Firewise communities, LTER field scientists, environmental journalists. It watches each user's polygons and, when a detection matters, produces an L2-style *situation brief* that explains what is happening to that specific place, in context.

**Why v1 targets land trusts.** LTA's Wildfire Resilience Network is a ready-made, warm distribution channel (agent 10). Land trusts already have named polygons in their GIS, mission-aligned donation behaviour, and a peer network that meets about wildfire. Shipping for one archetype well is a stronger v1 than shipping for all eight poorly.

**What the user buys, in one sentence.** When fire touches their place, they get a brief written by an agent that knows their polygon, not a generic app notification.

**What v1 is not.** Not a Watch Duty replacement. Not a spread simulator. Not a chatbot. Not a mobile app. Not multi-org. See "Scope boundaries" below.

---

## User stories (JTBD-framed)

Eight stories. Each one is a job a land trust steward needs done. Acceptance criteria are testable.

### US-1. First AOI in under 5 minutes
**As a** land trust stewardship director, **when** I first land on the site after a peer mentions it in the LTA WRN newsletter, **I want to** sign in and have my first preserve watched within 5 minutes, **so that** I can evaluate the tool without booking a demo.

Acceptance:
- Cold visit → receiving first "watch confirmed" email in ≤ 5 min, measured on a clean browser.
- Sign-in via Clerk (Google / email OTP). No credit card.
- AOI created by drag-and-drop GeoJSON upload OR by drawing a polygon on map OR by pasting a bbox. At least two of the three work in v1.
- Polygon area supported up to 100,000 ha (covers the largest land trust holdings; larger splits into sub-AOIs).

### US-2. Per-AOI alert rules that respect quiet hours
**As a** stewardship director, **when** I configure a preserve, **I want to** set a distance threshold, quiet hours in my local timezone, and a min-confidence gate, **so that** I am not woken at 02:00 for a nominal detection 90 km away.

Acceptance:
- Each AOI has: `alert_distance_km` (default 25), `min_confidence` (low/nominal/high), `quiet_hours_local` (HH:MM–HH:MM + IANA tz), `channels` (email and/or webhook).
- Quiet-hours detections are held and delivered in a morning digest at the top of the quiet-hours window, not dropped.
- Changing a rule takes effect on the next cron tick (≤ 15 min).

### US-3. Receive an L2 situation brief, not a templated alert
**As a** stewardship director, **when** a detection matches my preserve, **I want to** receive a brief that names the preserve, explains distance and direction, integrates weather and any authority perimeter, and references my site's prior fires, **so that** I can make a decision in one read.

Acceptance:
- Brief contains all fields from the LLM schema below (summary, key_facts, context, recommended_watch_items, uncertainty, next_brief_hint).
- `context.prior_events` is non-empty if the site has a prior event on file in the last 5 years; otherwise reads "no prior events on file."
- Brief is reproducible: the same input context produces a brief whose structured fields are identical modulo prose phrasing (determinism assert on the schema, not the summary string).
- P95 end-to-end latency from FIRMS detection to brief delivery ≤ 18 min (15 min cron bucket + 3 min LLM + send).

### US-4. Per-AOI page with map and brief history
**As a** stewardship director, **when** a colleague asks "what's happened at Spring Creek this season," **I want to** open a per-AOI page and see every brief, every detection, and the current live map, **so that** I can answer without digging through email.

Acceptance:
- `/aoi/{id}` renders: polygon on MapLibre, last 90 days of detections as points, all briefs in reverse-chron, freshness timestamp of last poll.
- Briefs permalink is shareable with a non-authenticated reader for 30 days (signed token).
- Page is mobile-responsive at 375 px width.

### US-5. One-click unsubscribe / pause / snooze
**As a** stewardship director, **when** a large fire season produces many overlapping detections, **I want to** snooze an AOI for 24h or pause it indefinitely from a link in any brief email, **so that** I retain control without logging in.

Acceptance:
- Every email has snooze-24h / pause / unsubscribe links (signed token, no login).
- Pausing suppresses briefs but continues polling (history stays complete).
- Unsubscribing removes email channel; AOI continues if webhook channel remains.

### US-6. Export my AOIs and briefs
**As a** stewardship director, **when** I want to cite the tool in a grant report, **I want to** export my AOIs as GeoJSON and my briefs as JSON or Markdown, **so that** I can include evidence without screenshots.

Acceptance:
- `GET /api/aoi/{id}/export?format=geojson` and `?format=markdown` return within 5 s for AOIs with ≤ 500 briefs.
- Markdown export includes positioning-line footer and a link back to the AOI page.

### US-7. BYO Gemini key for heavy users
**As a** technically-inclined user with a heavy AOI footprint, **when** I see my briefs are being rate-limited by the shared AI Gateway budget, **I want to** paste my own Gemini API key in settings, **so that** my usage does not affect other users and I am not forced onto a paid tier I do not yet have.

Acceptance:
- User settings accept a Gemini key stored encrypted (pgcrypto column or Vercel KV sealed). Never returned to client in plaintext.
- When present, user's briefs are generated with their key; metered in a separate counter.
- Removing the key restores the shared gateway.

### US-8. MCP hook for v2
**As a** power user (journalist or researcher), **when** v2 ships MCP in a few weeks, **I want to** call `list_active_fires`, `get_aoi_history`, `subscribe_aoi` from Claude / ChatGPT / my own agent, **so that** the tool composes into my workflow without me building a scraper.

Acceptance (v1-side hook only):
- v1 ships the REST endpoints behind these tool names; the MCP wrapper is a v2 side-artifact (ADR 0005).
- v1 auth pattern supports bearer tokens (user-minted in settings) so the MCP wrapper has a clean path.

---

## Core flows

### Flow 1 — Sign-up (cold → watching)
1. User lands on `/`, sees positioning line and "Start watching your place" CTA.
2. Sign-in via Clerk (Google / email OTP).
3. Onboarding screen: "Add your first AOI" — upload GeoJSON / draw / bbox.
4. Defaults applied: 25 km distance, nominal confidence, no quiet hours, email channel = account email.
5. Confirmation email "Now watching {AOI name}. First poll at {UTC time}."
6. First cron tick within 15 min runs a backfill poll over last 24 h so a user arriving mid-fire gets immediate context.

### Flow 2 — Create AOI
1. `/aoi/new` → three tabs: Upload, Draw, Paste.
2. On submit: server validates geometry (`ST_IsValid`, area ≤ 100,000 ha, SRID 4326, simplified to ≤ 500 vertices via `ST_SimplifyPreserveTopology`).
3. Save → compute and store the bounding-box-bucket key (see Flow 5).
4. Redirect to `/aoi/{id}` with empty state and "First poll scheduled at {ts}."

### Flow 3 — Configure rules
1. `/aoi/{id}/rules` → form (distance, confidence gate, quiet hours + tz dropdown, channels).
2. Save → `UPDATE aoi_rules` and return to `/aoi/{id}`.
3. Next cron tick picks up new rules (no eager recompute).

### Flow 4 — Receive a brief
1. Cron poll finds detection; gate passes (see Flow 6).
2. `POST /api/brief/generate` invoked server-side with structured context.
3. Gemini 2.5 Flash-Lite returns JSON brief.
4. Brief persisted to `aoi_briefs`; dispatcher sends to each active channel.
5. Email body = rendered Markdown of the brief + snooze/pause/unsubscribe links.
6. Webhook body = the raw brief JSON.

### Flow 5 — Cron poll (every 15 min, bucketed)
1. GitHub Actions cron triggers `POST /api/aoi/poll/tick`.
2. Tick loads all non-paused AOIs, groups by bucket key (5°×5° geohash-ish over AOI bbox center).
3. For each bucket: one FIRMS call (`VIIRS_NOAA20_NRT` + `VIIRS_SNPP_NRT` union, 1-day window, bucket bbox).
4. For each AOI in bucket: `ST_Intersects(detection, buffer(polygon, alert_distance_km))`.
5. Matching detections dedup'd (content hash: sat + frp-rounded + lat/lon-rounded-3dp + acq_time) vs. last 24 h.
6. New detections written to `aoi_events`; LLM gate evaluated (Flow 6).

### Flow 6 — LLM gate
Gate passes (brief is generated) when **any** of:
- First detection for this AOI in the last 72 h (prior-absence signal), OR
- ≥ 2 pixels inside the alert buffer in the current tick, OR
- Any detection with FRP > 5 MW inside the buffer, OR
- Any detection within 0.5 × `alert_distance_km` (close-in proximity).

Gate fails (event is logged silently, no brief) otherwise. ~5% pass rate assumed per agent 09.

### Flow 7 — Review history
1. `/aoi/{id}` shows briefs reverse-chron.
2. Click → `/aoi/{id}/brief/{brief_id}` renders full brief with the detection(s) that triggered it on the map.
3. Share link copy → signed 30-day URL.

---

## Data model

Target: ≤ 10 tables, Neon Postgres + PostGIS. All timestamps `timestamptz`. SRID 4326.

| Table | Key columns |
|---|---|
| `users` | `id uuid pk`, `clerk_id text unique`, `email text`, `gemini_key_ciphertext bytea null`, `created_at` |
| `aois` | `id uuid pk`, `user_id fk`, `name text`, `description text null`, `geom geometry(Polygon,4326)`, `bbox geometry(Polygon,4326)`, `bucket_key text`, `area_ha numeric`, `created_at`, `paused_until timestamptz null`, `deleted_at null` |
| `aoi_rules` | `aoi_id fk pk`, `alert_distance_km int default 25`, `min_confidence text default 'nominal'`, `quiet_hours_start time null`, `quiet_hours_end time null`, `quiet_hours_tz text null`, `channels jsonb` (e.g. `[{"type":"email","addr":"..."},{"type":"webhook","url":"..."}]`) |
| `aoi_events` | `id uuid pk`, `aoi_id fk`, `detection_ts timestamptz`, `sat text`, `lat double precision`, `lon double precision`, `frp double precision null`, `confidence text`, `distance_km double precision`, `bearing_deg double precision`, `content_hash text unique`, `gate_passed bool`, `brief_id uuid null`, `created_at` |
| `aoi_briefs` | `id uuid pk`, `aoi_id fk`, `triggering_event_ids uuid[]`, `model text`, `schema_version int`, `payload jsonb`, `rendered_markdown text`, `share_token text unique null`, `share_expires_at null`, `created_at` |
| `notifications` | `id uuid pk`, `brief_id fk`, `channel_type text`, `channel_addr text`, `status text` (queued/sent/failed), `status_detail text null`, `rate_limit_key text`, `sent_at null` |
| `firms_cache` | `bucket_key text`, `window_start timestamptz`, `window_end timestamptz`, `payload_hash text`, `payload jsonb`, `fetched_at`, pk `(bucket_key, window_start)` |
| `poll_runs` | `id uuid pk`, `started_at`, `finished_at null`, `bucket_count int`, `firms_calls int`, `new_events int`, `briefs_generated int`, `errors jsonb` |

Eight tables. PostGIS indexes on `aois.geom` and `aois.bbox`; btree on `aoi_events(aoi_id, detection_ts desc)`; unique on `aoi_events.content_hash`.

---

## API surface

All endpoints return JSON. Auth: Clerk session cookie on the web, bearer token (user-minted) for programmatic access.

| Method | Path | Purpose | Request | Response |
|---|---|---|---|---|
| GET | `/api/me` | Current user | — | `{id, email, has_byo_key}` |
| POST | `/api/aoi` | Create AOI | `{name, description?, geometry (GeoJSON)}` | `{id, name, bucket_key, first_poll_at}` |
| GET | `/api/aoi` | List my AOIs | — | `[{id, name, paused_until, last_event_at}]` |
| GET | `/api/aoi/{id}` | AOI detail | — | `{aoi, rules, recent_events[50], recent_briefs[20]}` |
| PATCH | `/api/aoi/{id}` | Rename / re-geom / pause | partial `{name?, geometry?, paused_until?}` | updated AOI |
| DELETE | `/api/aoi/{id}` | Soft-delete | — | `{ok:true}` |
| PUT | `/api/aoi/{id}/rules` | Replace rules | rules payload | saved rules |
| GET | `/api/aoi/{id}/export?format=geojson\|markdown` | Export | — | file |
| GET | `/api/brief/{id}` | Brief detail | — | `{brief, events, aoi}` |
| GET | `/api/brief/{id}/share/{token}` | Public shared brief (30-day) | — | `{brief, events}` |
| POST | `/api/aoi/poll/tick` | Cron entry (GH Actions only, HMAC-signed) | `{bucket?: string, force?: bool}` | `{run_id, firms_calls, new_events, briefs_generated}` |
| POST | `/api/brief/generate` | Internal — called from poll | `{aoi_id, event_ids}` | `{brief_id}` |
| POST | `/api/notifications/webhook/{token}` | Inbound for snooze/pause/unsub signed links | — | `{ok:true}` |
| GET | `/api/mcp/list_active_fires` | v2-hook: bbox query | `?bbox=` | `{detections[]}` |
| GET | `/api/mcp/get_aoi_history` | v2-hook | `?aoi_id=&since=` | `{events[], briefs[]}` |
| POST | `/api/mcp/subscribe_aoi` | v2-hook | `{geometry, rules}` | `{aoi_id}` |

Rate limits: `/api/aoi/poll/tick` locked to GH Actions via HMAC header + IP allowlist of GH's runner ranges. User-facing endpoints: 60 req/min/user.

---

## LLM brief format

This section is the product. A brief is a JSON object validated against the schema below (Zod on server + structured-output mode to Gemini 2.5 Flash-Lite). The renderer turns it into Markdown for email and HTML for web.

### Schema (v1)

```json
{
  "schema_version": 1,
  "aoi": {
    "id": "uuid",
    "name": "string",
    "area_ha": "number"
  },
  "summary": "string, 1–2 sentences, reads like a staffer's radio report",
  "key_facts": {
    "nearest_detection_km": "number",
    "bearing_from_aoi_deg": "number (0=N, 90=E)",
    "wind_dir_deg": "number | null",
    "wind_speed_kmh": "number | null",
    "wind_toward_aoi": "boolean | null",
    "detection_count_in_window": "integer",
    "max_frp_mw": "number | null",
    "satellites": "string[] (e.g. ['VIIRS_NOAA20'])",
    "window_hours": "integer"
  },
  "context": {
    "weather_note": "string | null — one-line narrative (e.g. 'RH 18%, gusting 40 km/h from NW')",
    "authority_perimeter": {
      "source": "string | null (e.g. 'PT-ICNF', 'NIFC', null if none)",
      "posted_ts": "timestamptz | null",
      "contains_detection": "boolean | null"
    },
    "prior_events": [
      {"date": "YYYY-MM-DD", "description": "string, one line", "outcome": "string | null"}
    ]
  },
  "recommended_watch_items": [
    "string — a concrete thing for the steward to watch, not an imperative. e.g. 'Re-check at 06:00 local; wind shift forecast to push the head fire SE.'"
  ],
  "uncertainty": "string — explicit about what is NOT known. e.g. 'No authority perimeter published yet; FRP low may indicate smouldering or a partial burn.'",
  "next_brief_hint": {
    "when": "string — 'in 3 hours' | 'on polygon breach' | 'daily at 06:00 local'",
    "trigger": "string"
  }
}
```

### Worked example (illustrative)

Inputs: AOI "Spring Creek Preserve" (2,040 ha, Sonoma County CA). Two VIIRS_NOAA20 detections 14 km N at 04:17 UTC, max FRP 11 MW. Wind 240° @ 28 km/h (from WSW, toward ENE — away from the AOI). No authority perimeter yet. Prior event on file: 2020-08-20 (LNU Lightning Complex edge, perimeter reached within 3 km).

```json
{
  "schema_version": 1,
  "aoi": {"id": "…", "name": "Spring Creek Preserve", "area_ha": 2040},
  "summary": "Two VIIRS detections 14 km N of Spring Creek Preserve at 04:17 UTC, max FRP 11 MW. Wind is blowing the head away from the preserve for now, but the 2020 LNU edge reached within 3 km of this same boundary — worth a morning re-check.",
  "key_facts": {
    "nearest_detection_km": 14.0,
    "bearing_from_aoi_deg": 357,
    "wind_dir_deg": 240,
    "wind_speed_kmh": 28,
    "wind_toward_aoi": false,
    "detection_count_in_window": 2,
    "max_frp_mw": 11.0,
    "satellites": ["VIIRS_NOAA20"],
    "window_hours": 1
  },
  "context": {
    "weather_note": "RH ~22%, winds 240° @ 28 km/h pushing activity ENE away from the preserve.",
    "authority_perimeter": {"source": null, "posted_ts": null, "contains_detection": null},
    "prior_events": [
      {"date": "2020-08-20", "description": "LNU Lightning Complex eastern edge.", "outcome": "Perimeter reached within 3 km of the preserve's north boundary; no incursion."}
    ]
  },
  "recommended_watch_items": [
    "Re-check at 06:00 local — overnight inversion breakup can flip local winds.",
    "Watch CAL FIRE SoCo incident page for a posted perimeter; none yet.",
    "If a third detection lands within 10 km, treat as a separate event not continuation."
  ],
  "uncertainty": "No authority perimeter published yet. 2 pixels is the floor for us to brief; could be a small slash burn rather than a running fire.",
  "next_brief_hint": {
    "when": "on polygon breach, else 06:00 local digest",
    "trigger": "new detection < 10 km OR authority perimeter published"
  }
}
```

The Markdown renderer reads that JSON and produces a 200–300 word email that leads with the summary, shows key_facts as a compact table, prose-renders context, bullets the watch items, italicizes uncertainty, and ends with the next_brief_hint and snooze/pause/unsubscribe links.

### Why this schema, concretely
- `summary` is the only free-prose field that hits the reader first — optimized for a mobile email preview pane.
- `key_facts` is structured so the UI can render compact cards without re-parsing prose.
- `uncertainty` is mandatory — non-negotiable per AGENTS.md ("warn, don't bypass").
- `prior_events` is what separates an L2 brief from a template: without the site's own history, the brief is generic.

---

## Scope boundaries (v1 does NOT)

- **Not spread forecasting.** Deleted per ADR 0004. The brief may mention wind direction, but it is not a forecast. Rationale: solo-unmaintainable and duplicates NOAA/Technosylva.
- **Not a denoiser UI, HITL review queue, or confidence-drift dashboard.** Latest denoiser gate failed (F1 0.22). Replaced by a simple confidence + FRP + industrial-mask filter from `@earthtools/firms`.
- **Not multi-org / shared workspaces.** One user, many AOIs. Team sharing deferred.
- **Not a native mobile app.** Responsive web only. Push notifications are email + webhook.
- **Not a general fire map.** No homepage map of "all fires on Earth right now." v1 is AOI-first; a user with no AOIs sees an empty dashboard and a CTA to create one.
- **Not a chatbot.** `AIChatAssistant` is cut. The AI is the brief, not a dialog partner.
- **Not multilingual in v1.** English only. i18n is an archetype-3 (Natura 2000) unlock for v2.
- **Not a public API without a user.** MCP endpoints require a bearer token minted by a signed-in user.
- **Not a replacement for Watch Duty.** Land trusts may still use WD for their personal addresses; A' is for their preserves.
- **Not an archive scrubber / historical replay.** Cut (agent 09).

---

## Acceptance for v1 launch

A numbered, testable checklist. v1 ships when all pass.

1. **Land trust archetype served end-to-end.** At least one LTA-member land trust has created ≥ 1 AOI with a real preserve polygon and received ≥ 1 brief from a real FIRMS detection (not a synthetic test). Evidence: their AOI id + brief id logged in a launch note.
2. **Cold start to first watch ≤ 5 min.** Measured from `/` load to "watch confirmed" email on a clean browser.
3. **Infra cost claim holds.** 7 consecutive days of operation at ≥ 10 AOIs cost ≤ $1 total across Vercel + Neon + AI Gateway (agent 09 target is $0 at 50u/100 AOIs; v1 launch sample will be below that).
4. **Brief schema conformance = 100%.** Every brief persisted validates against the v1 Zod schema. A schema-failing LLM response is logged as a gate miss, not delivered.
5. **P95 end-to-end latency ≤ 18 min** from FIRMS detection timestamp to brief send, measured over the launch week.
6. **Gate passes ≤ 8% of ticks.** If the gate is passing > 8% at the target archetype's usage, the LLM cost model is wrong and we fix gate thresholds before inviting more users.
7. **Landing page carries the canonical positioning line** verbatim: *"Free, open, AI-native fire intelligence for stewardship — depth over speed."*
8. **Repo public + MIT / Apache-2 licensed.** Link from the landing page.
9. **LTA WRN newsletter post drafted** (brief 13) and cleared by Vanyo. Held until items 1–8 pass; posted in the next LTA WRN cycle.
10. **Rollback plan documented.** If Vercel Hobby's non-commercial clause becomes a blocker (the biggest agent-09 risk), the migration to Cloudflare Workers + Vercel Pro is documented in `docs/pivot-architecture.md` (brief 12) so a solo operator can execute in one afternoon.

---

## Open questions (for Vanyo)

1. **Auth vendor.** Clerk free tier (10k MAU) or Supabase Auth or Auth.js? Clerk is the fastest path; Supabase Auth couples more cleanly if we ever want a unified Neon + auth story. Default: **Clerk** unless Vanyo prefers otherwise.
2. **BYO key in v1 or v2?** US-7 is written for v1, but it's ~2 days and the gate plus AI Gateway $5 credit comfortably covers the target archetype. Acceptable to push to v2 if timeline squeezed.
3. **Authority-perimeter fetch in v1.** The brief schema has `authority_perimeter` but the cut list removed the authority ingests. Acceptable v1 answer: **leave the field always `null` in v1, wire it live in v1.1** as an LLM tool-call to NIFC/ICNF/CWFIS public GeoJSON per-brief (no ingest pipeline). Need a ruling.
4. **Share-link TTL.** Default 30 days. Land trusts may want "forever for grant reporting" — possibly make it per-brief `expires_at` user-selectable up to 1 year.
5. **Digest cadence in quiet hours.** Default: release held briefs at the top of the quiet window. Alternative: send a single digest brief that merges the held ones. Digest merging is an LLM call — cheap, but it changes the "one detection = one brief" invariant.
6. **First interview timing.** Now that there is a spec, does Vanyo want to do 1–2 targeted interviews with LTA WRN stewards before coding (per PM_CLAUDE "targeted interviews may come after narrowing") or ship the spec and let the newsletter post be the interview?
