# 2026-05-07 — `/api/aoi/poll` performance audit (research, no behavioural change)

**Branch:** `chore/research-poll-route-perf`
**Files audited:** `app/api/aoi/poll/route.ts` (526 LOC), `lib/firms/matcher.ts`, `lib/firms/buckets.ts`, `lib/firms/prune.ts`, `lib/ai/generate.ts` (entry only), `lib/notify/dispatch.ts` (entry only).
**User count today:** 0. Audit is pre-launch; numbers are static estimates.

## 1. Method

Static read of the cron entry point and its three fan-out helpers. No runtime profiling — there is no production traffic. Per-call cost estimates pulled from observable shape (one SQL `execute` per loop iteration, one external API per event, etc.). The Vercel function-duration constant in the file (`maxDuration = 60`) is the operative wall-clock budget; the brief's "300s" assumption refers to Pro-tier defaults the project does not have.

## 2. Findings (with line references in `route.ts`)

### F1 — Buckets are polled strictly serially (lines 234–243)
`for (const bucket of buckets) { await runOneBucket(...) }`. Each bucket waits on the prior. With one user / one bucket: free. With six active buckets each taking ~6 s (FIRMS fetch + matcher + zero events): ~36 s — under the 60 s ceiling but already close. With many users spanning more buckets it dies before completing.
- 0 users: not a problem.
- 100 users (heavy clustering, ~10–15 buckets): tight; one slow FIRMS call per bucket pushes past 60 s.
- 1000 users: certainly dies. Hard ceiling of `60 / per_bucket_seconds` buckets per tick.

### F2 — Per-event brief generation is serial (lines 333–354)
`for (const eventId of matchOutcome.createdEventIds) { await briefGen(...) }`. Each call hits the AI Gateway (~5–15 s p50 for Gemini structured output). With 5 events in one bucket, that's 25–75 s on its own — already over budget.
- 0 users: irrelevant (no events).
- 100 users: realistic during a fire weekend in California; multiple AOIs in one 5°×5° tile generate concurrent events. Easily blows the 60 s cap.
- 1000 users: catastrophic. This is the dominant failure mode.

### F3 — Per-brief dispatch is serial (lines 360–376)
Same shape as F2. Resend is fast (~300–800 ms) so the cost is bounded, but it still serialises N round-trips when they could be one `Promise.all`. Combined with F2, total event-handling per bucket is `O(N events × (LLM_ms + email_ms))`.

### F4 — Per-detection insert in matcher (`matcher.ts:128–177`)
`for (const d of args.detections) { await db.execute(INSERT ...) }`. A 5°×5° FIRMS tile during fire season can return 1000+ detections. That is 1000+ sequential round-trips through the Drizzle/Neon pooler. Even at 5 ms per round-trip (optimistic for serverless), that's 5 s per bucket of pure DB latency. Industrial-mask check is an inline `EXISTS` subquery per row — fine in PostGIS, but multiplies the per-row cost.
- 0 users: still costs CPU and DB seconds because polling happens regardless of users (poll-all enumerates buckets from `aois`, so zero-user state polls zero buckets — actually free today).
- 100 users: 5–20 s of pure insert latency per heavy bucket.
- 1000 users: dominates wall clock.

### F5 — Per-AOI-match upsert in matcher (`matcher.ts:104–112`)
`for (const match of matches) { await upsertEvent(...) }` — two SQL statements per AOI hit (SELECT existing, then UPDATE or INSERT). N AOIs in a hot bucket → 2N round-trips. Could be one `INSERT ... ON CONFLICT (aoi_id, dedupe_hash) DO UPDATE`.

### F6 — `pruneOldDetections` runs every tick (lines 199–207)
Hits every cron tick (every 15 min = 96/day). The DELETE scans by `detected_at`; assuming an index on that column the scan is bounded, but it still acquires a write lock on a hot table every tick. Running once per day (or letting Postgres autovacuum + a daily cron) would cut 95 writes/day. Today the table is empty, so cost ~0; at scale it's a few hundred ms per tick of pointless work.

### F7 — `closeJobRun` writes are not batched (lines 480–511)
Per bucket: one `openJobRun` (INSERT) + one `closeJobRun` (UPDATE). Plus the parent. This is bounded (`O(buckets)` not `O(events)`) and acceptable.

### F8 — In-memory detection accumulation (matcher line 84+)
The matcher accepts `args.detections` already in memory (parsed by the FIRMS client). A 5° tile during a peak fire season can be hundreds of KB to a few MB of CSV — fine for a Node function, not a leak vector. Not a concern.

### F9 — No transactional scope around matcher work
`insertDetections` then `findAoiMatches` then `upsertEvent` run as separate auto-committed statements. If the function dies mid-bucket the partial detections persist but their events do not. Today this is recovered on the next tick (re-poll inserts zero new rows via `ON CONFLICT DO NOTHING`, and the matcher's `pollStart` filter means orphan detections are never re-considered for events). **Worth flagging:** orphan detections from a crashed tick will never produce events. Acceptable, but should be documented somewhere; currently it is not.

### F10 — `maxDuration = 60` is a Hobby ceiling
Vercel Hobby caps at 60 s per function invocation. The brief mentioned 300 s but that is Pro. F1+F2 combined will push us over 60 s at well under 100 active users. The single biggest leverage is parallelisation (F1, F2) before raising the ceiling — even Pro's 300 s is only buying ~5× headroom against an `O(buckets × events × LLM_latency)` shape.

## 3. Highest-priority recommendation

**Parallelise per-bucket execution (F1) using `Promise.all` with a small concurrency cap (e.g. 3).** Buckets are independent — separate FIRMS bbox, separate AOI subset, separate `job_run` child row. `Promise.all` over `runOneBucket` gives near-linear wall-clock savings. A concurrency cap (e.g. via a tiny in-process semaphore) protects the FIRMS rate limit (already 6 req/min in `lib/firms/client.ts`'s token bucket) and avoids opening too many Neon pool slots at once.

This is the single change that buys the most headroom against the 60 s wall. F2 (parallel brief gen) is a close second but introduces AI Gateway concurrency questions (cost spikes, model rate limits) that warrant a separate decision.

**Estimated leverage:** at 6 buckets, wall clock ~`max(per_bucket)` instead of `sum(per_bucket)` — typically 3–4× faster end-to-end at the 100-user mark.

**Why not shipped in this PR:** changes the failure isolation contract (one slow bucket no longer delays others, but a Promise.all rejection needs `Promise.allSettled` + per-bucket try/catch — already present, so this is straightforward, but it's still a behavioural change worth reviewer eyes).

## 4. Lower-priority recommendations (ordered)

1. **F2 — parallelise brief generation per bucket** with `Promise.allSettled(map)`. Gate on AI Gateway rate-limit posture; consider a concurrency cap of 2–3.
2. **F4 — batch detection inserts** into a single multi-row `INSERT ... VALUES (...), (...), ... ON CONFLICT DO NOTHING RETURNING is_industrial_static`. Cuts per-bucket DB time from `O(N × RTT)` to `O(1 × RTT)`. The industrial check moves to a single `EXISTS` join or a CTE; PGlite path needs a parallel implementation.
3. **F5 — collapse `upsertEvent` to one `INSERT ... ON CONFLICT (aoi_id, dedupe_hash) DO UPDATE`** with `excluded.*` references. Halves round-trips; matches the pattern used for detections.
4. **F3 — parallelise dispatch** via `Promise.allSettled`. Low risk; Resend is per-recipient anyway.
5. **F6 — gate `pruneOldDetections` on a "once per UTC day" check** (e.g. `SELECT MAX(finished_at) FROM job_runs WHERE detections_pruned > 0`). Saves 95 writes/day at scale.
6. **F9 — document the orphan-detection-on-crash invariant** in a comment block at the top of `matcher.ts`. No code change.

## 5. Anti-pattern report (looks bad, actually fine)

- **Two-statement SELECT-then-UPDATE/INSERT in `upsertEvent`.** Looks like a race window. In practice, only the cron writes events, only one cron runs at a time (`concurrency: { group: firms-poll, cancel-in-progress: false }` in the workflow), and the unique index on `(aoi_id, dedupe_hash)` is the canonical guard. Worth collapsing for performance (recommendation 3) but not for correctness.
- **`dbNow()` round-trip at the start of each match call** (matcher line 92). One extra SELECT per bucket. Required for clock-skew safety between Vercel and Neon — the comment explains why. Don't optimise this away.
- **Per-row `is_industrial_static` subquery.** Looks like an N+1 against `industrial_mask_static`. PostGIS GIST index on the mask makes each lookup O(log N) and the mask is small (~70 rows in the seed). Real cost is in the per-row round-trip (F4), not the subquery.
- **`testFirmsFetch` / `testBriefGen` / `testNotifyDispatch` module-level mutables.** Looks fragile. They exist because the team chose not to take a DI framework dependency; tests reset them in `afterEach`. Fine.

## 6. What we can't measure without real users

- **Actual per-bucket FIRMS latency distribution.** Have only NASA's published "<5 s p95" claim. Need real cron logs to validate.
- **AI Gateway p50/p95 for our prompt size.** Token count varies with detection count and event history. Need observability on `aoi_briefs.created_at - aoi_events.created_at`.
- **Resend per-call latency from Vercel's edge.** Bounded but unmeasured.
- **Neon cold-start cost for the autoscale-to-zero plan.** First poll after idle pays a wake-up tax; subsequent ticks within the keep-warm window do not. Need a few days of cron logs to characterise.
- **Tail-latency correlation across buckets** (does FIRMS being slow for one bucket predict slowness for others same tick?). Drives whether the F1 concurrency cap should be 3, 6, or unbounded.

## Decision queued for Vanyo (not blocking)

When real cron data exists (post first 10 user signups), revisit the F1/F2 parallelisation. The fix is small (~30 LOC delta) and the leverage is large; deferring purely because we lack production telemetry, not because of any design uncertainty.
