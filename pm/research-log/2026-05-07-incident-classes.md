# Incident classes — post-launch brainstorm

Date: 2026-05-07
Author: scout
Status: brainstorm + framework. Not a polished runbook. Goal: enumerate the shape of likely incidents so a future scout (or operator) can fill in detailed steps per class.

The cold-start runbook (PR #425) covers measuring the happy path. Launch-readiness item #10 (rollback runbook) is still failing. This note sketches ten incident classes the operator may face and stubs the response template: **symptom → first diagnostic → mitigation → rollback path → recovery validation**.

The shared substrate for diagnostics:
- Vercel deployment + function logs: `vercel logs wildfire-nowcast` (or the project dashboard → Deployments → Functions).
- GitHub Actions run history for the FIRMS poll: `.github/workflows/firms-poll.yml` runs (workflow page or `gh run list -w "FIRMS poll"`).
- DB inspection: `psql $DATABASE_URL` against Neon, then read `job_runs`, `notifications_log`, `aoi_briefs`, `users`.
- Resend dashboard for delivery status; AI Gateway dashboard for token / quota usage; Clerk dashboard for webhook delivery.

## 1. FIRMS API down or rate-limited

**Symptom**: `job_runs.outcome` accumulates `'rate_limited'` or `'network_error'` rows; the dashboard "polling delayed" banner (Stage 8) shows up. Cron continues silently retrying.
**First diagnostic**: `select outcome, count(*) from job_runs where started_at > now() - interval '2 hours' group by 1;`. Cross-check FIRMS status on `https://firms.modaps.eosdis.nasa.gov/`.
**Mitigation**: nothing if FIRMS itself is down — the system is designed to degrade. If we are being rate-limited, reduce poll frequency by editing the `*/15 * * * *` cron to `*/30` and re-deploy the workflow file.
**Rollback path**: revert the cron edit when FIRMS recovers.
**Recovery validation**: one full `outcome='ok'` cycle across all buckets in `job_runs`; banner clears.
**Incident vs. transient**: <1 hour of failures = transient; >2 hours or >50% of buckets = real incident, post status note.

## 2. AI Gateway 500s or quota exhausted

**Symptom**: `aoi_briefs` insertions stall while `job_runs` shows `ok`; users get watch-confirmed but no follow-up briefs. Vercel function logs show 500s from `@ai-sdk/google` calls.
**First diagnostic**: Vercel AI Gateway dashboard → token usage + error rate. `select max(created_at) from aoi_briefs;` to confirm freshness.
**Mitigation**: feature-flag brief generation off (env var) so the rest of the pipeline keeps working; users see "brief unavailable" rather than nothing.
**Rollback path**: if a recent prompt change correlates, revert the prompt snapshot PR (#419 introduced snapshots — check `lib/ai/prompts/`).
**Recovery validation**: fresh `aoi_briefs` row with non-null `summary` for a known active AOI.

## 3. Resend bouncing or rate-limited

**Symptom**: `notifications_log.status='failed'` accumulating; users miss alerts.
**First diagnostic**: `select status, count(*) from notifications_log where created_at > now() - interval '1 day' group by 1;`. Resend dashboard → activity feed for bounce reason codes.
**Mitigation**: pause notification dispatch (env flag) to stop hammering Resend during a rate-limit window; keep generating briefs so digest can replay.
**Rollback path**: if a sender domain change broke DKIM, revert the DNS / Resend domain config.
**Recovery validation**: send a test alert to a known-good address; confirm `status='delivered'`.

## 4. Clerk webhook delivery broken

**Symptom**: new sign-ups don't sync to `users` table; the JIT path covers first authenticated request but `email` stays as `<userId>@pending.invalid`.
**First diagnostic**: Clerk dashboard → Webhooks → recent delivery attempts. `select count(*) from users where email like '%@pending.invalid';`.
**Mitigation**: rotate the webhook signing secret if the failure is auth-related; manually backfill affected users via a one-shot script.
**Rollback path**: revert any recent changes to `app/api/webhooks/clerk/` or middleware.
**Recovery validation**: create a test sign-up, confirm `users.email` populates within ~30 s.

## 5. Neon free-tier ceiling hit

**Symptom**: DB writes fail with quota / size errors. The 14-day prune (Stage 7) normally keeps us under the ceiling — if it isn't running, the table that grows fastest is `firms_detections`.
**First diagnostic**: Neon dashboard → storage. `select pg_size_pretty(pg_database_size(current_database()));` and per-table sizes.
**Mitigation**: run the prune manually (`pnpm tsx scripts/prune.ts` or equivalent invocation — confirm the actual entry point); if storage is genuinely full, truncate `firms_detections` rows older than 7 days as a panic-button.
**Rollback path**: not really a deploy issue; if the prune cron stopped, restore the workflow file.
**Recovery validation**: storage drops below ceiling; new writes succeed.

## 6. Vercel function timeout (60s `maxDuration`)

**Symptom**: `app/api/aoi/poll/route.ts` (and `app/api/aoi/route.ts`) cap at `maxDuration = 60`. If a tick exceeds that, Vercel kills the function mid-bucket; some buckets unprocessed.
**First diagnostic**: Vercel function logs filtered by 504 / "Function execution timed out". Cross-check `job_runs` for missing buckets in the affected window.
**Mitigation**: the FIRMS poll is bucket-scoped — invoke the workflow with a specific `bucket` input (`workflow_dispatch` already supports it) to drain the missed bucket out-of-band.
**Rollback path**: if a recent code change made the route slower, revert it.
**Recovery validation**: the next scheduled tick completes within budget; no missing buckets in `job_runs`.

## 7. Bad code deployed

**Symptom**: error rate spike right after a deploy, visible on Vercel deployment dashboard.
**First diagnostic**: Vercel → Deployments → compare current vs. previous; check the diff.
**Mitigation/rollback**: Vercel "Promote to production" on the prior good deployment — instant. This is the canonical rollback for code regressions.
**Recovery validation**: error rate returns to baseline; spot-check a brief generation end-to-end.

## 8. PostGIS-vs-PGlite drift

**Symptom**: tests green, production behaves differently — typically `lib/firms/matcher.ts` returning different AOI hits than expected.
**First diagnostic**: pull the production query and run it against a Neon read replica with the same inputs; compare to PGlite's turf.js fallback path.
**Mitigation**: revert the offending change. There is no clean hotfix because the tests don't cover the divergence.
**Rollback path**: same as #7 — promote prior deployment.
**Recovery validation**: add an integration test (`@testcontainers/postgresql` with `postgis/postgis:16-3.5`) that reproduces the divergence before re-shipping.

## 9. Token leak (audit + revoke)

**Symptom**: a share token (`/brief/share/<token>`) or notify-action token (`/api/notify/<action>/<token>`) is published somewhere it shouldn't be — Slack, GitHub gist, screenshot.
**First diagnostic**: identify the token and the user/AOI it maps to via DB lookup.
**Mitigation**: rotate the token (issue a new one; invalidate the old). If the leak is broad, rotate the signing secret so all outstanding tokens of that class become invalid.
**Rollback path**: not applicable.
**Recovery validation**: requests with the leaked token return 404/410.

## 10. Bot scrapes / abuse

**Symptom**: traffic spike on unauthenticated GETs (`/brief/share/<token>`, `/api/notify/<action>/<token>`). Stage 6+ has no rate limits.
**First diagnostic**: Vercel analytics → top routes + IPs.
**Mitigation**: temporarily add a Vercel Edge Middleware rule to rate-limit by IP; or block obvious offending UAs.
**Rollback path**: remove the middleware rule once the wave passes.
**Recovery validation**: traffic shape returns to baseline; legitimate share-link clicks still work.

## Recommendation: which to runbook first

Three classes are worth turning into full runbooks before the others:

1. **#7 Bad code deployed** — highest frequency, sharpest tool (Vercel "promote prior deployment"). A 5-line runbook here pays for itself the first time it's used.
2. **#5 Neon free-tier ceiling** — lowest reversibility once writes start failing, and the prune-cron failure mode is silent. Document the manual prune invocation and the panic-truncate threshold.
3. **#1 FIRMS rate-limit / outage** — highest expected frequency given external-API dependency, and the "transient vs. real incident" judgment is non-obvious. A clear threshold ("2 hours, 50% of buckets") removes that judgment from the operator's plate.

The remaining seven can stay as this brainstorm until they actually fire once.
