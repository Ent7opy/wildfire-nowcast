# Observability gap, not runbook gap

Date: 2026-05-07
Author: scout (brainstorm)
Trigger: PR #439 product-reviewer pass — "all 10 incident classes assume a human operator already noticed the symptom; the codebase has no alerting wired."

## The gap, named

Three layers are needed for an incident response process to function. Right now we have one and a half.

1. **Signal generation** — the running code emits structured evidence of what happened (success counts, error reasons, latencies, costs). We have most of this at rest in tables.
2. **Aggregation** — those signals land somewhere queryable on a useful time axis. We have the rows but no roll-up.
3. **Alerting** — a threshold crossing causes a human to be paged or emailed without that human first opening a dashboard. We have **none** of this.

Layer 3 is the actual missing piece. Every runbook in `pm/runbooks/` (the PR #439 set) starts with "operator notices X" — but for most failure modes there is no path by which the operator would notice without manually opening Vercel logs or running a SQL query they have not been told to run on a schedule. The runbooks are downstream of a problem we have not solved.

## What signals already exist

Walking the code on `master`:

- `job_runs` (`db/schema/index.ts:387`) — already records per-bucket and parent run rows with `outcome` text, `started_at`, `finished_at`, `error`. Stage 2 onward writes to this on every poll tick. Read by nothing automated.
- `notifications_log.status` — Stage 4 records send outcomes (sent, suppressed, error) per email attempt. Read by nothing automated.
- `aoi_briefs` — Stage 3 populates `gate_reason`, `latency_ms`, `cost_usd_est` (`db/schema/index.ts:280-286`) on every brief generation. Read by nothing automated.
- `console.warn` / `console.info` in `app/api/aoi/poll/route.ts` (notably the prune-failure path at lines 206-208). Visible only if a human opens Vercel function logs in the right time window.

The pattern is consistent. Stage briefs were diligent about *recording* outcome — there is genuinely a lot of evidence in the database. None of it is *read*.

## What signals are missing

Even with everything above, we cannot currently answer:

- Per-tick error rate over the last N hours (would need a roll-up over `job_runs.outcome`).
- FIRMS quota burn — we know we got rate-limited via `job_runs.outcome` strings, but we do not compute a daily call count against the documented FIRMS transaction ceiling.
- "How many users got a brief in the last hour" — requires joining `notifications_log` to `aoi_briefs` over a time window.
- Repeated prune failure — the warn-only branch swallows failures. Three consecutive failures should be loud; today they are silent.

## Cheap-first recommendations

In ascending cost order. v1 should pick exactly one. Defer the rest.

### Option A — weekly status digest cron (cheapest)

Add a second GitHub Actions cron (or Vercel cron) that runs once a week, hits a new admin endpoint, and emails Vanyo a digest computed from `job_runs` and `notifications_log` aggregates. ~30-60 LOC, zero new infrastructure, uses the Resend integration that already exists for Stage 4. Rough shape: count of poll ticks, count of error outcomes, count of briefs generated, count of notifications sent, top three `error` strings. Catches "the system stopped doing anything" within 7 days, which is the single failure mode currently most likely to silently persist.

### Option B — Vercel Log Drains to an external aggregator (medium)

Vercel supports Log Drains as a first-class integration. The lowest-cost destinations with documented Vercel integrations include Datadog, Logtail (now Better Stack), and Axiom. Pick one with a free tier; route function logs to it; configure an alert on `[poll]` log lines containing `failed` or `error`. Costs an external account and ~$0/mo at our volume but locks us into a third-party product.

### Option C — `/api/admin/health` + external uptime monitor (heavier)

Add a Clerk-protected (or shared-secret) endpoint that returns last-successful-poll timestamp, last-successful-brief timestamp, FIRMS daily call count, last-prune outcome. Pair with an external uptime monitor (UptimeRobot or BetterStack Uptime, both have documented free tiers) hitting the endpoint every 5 minutes. Catches "the cron stopped firing" within minutes. ~80 LOC plus a small piece of state to track FIRMS calls per day.

## What stays runbook-shaped vs observability-shaped

Of the 10 PR #439 classes:

- **Runbook-shaped** (a human will notice via existing surfaces): "Bad code deployed" (Vercel deploys page surfaces failing builds), "Clerk outage" (login page is the surface), "Resend domain unverified" (visible at first send attempt during setup).
- **Observability-shaped** (no human will notice without instrumentation): "Neon ceiling hit", "FIRMS rate-limited" (already in `job_runs.outcome` but unread), "AI Gateway quota exhausted" (visible in `aoi_briefs.gate_reason` but unread), "Prune sweep failing", "Notification dispatch silently dropping".

Five of the ten classes are observability-shaped. Writing runbooks for them before wiring observability is writing playbooks for symptoms no one will see.

## Recommendation for next pm / product-reviewer pass

Pick **Option A** (weekly digest) for v1. Reasoning:

- It is the only option whose entire surface area sits inside infrastructure we already operate (Resend + GitHub Actions or Vercel cron + the existing DB).
- It does not require a new account, secret, or third-party dependency — so it does not add a new entry to `pm/blockers.md`.
- It is bounded at well under the 200-LOC scout cap and could plausibly be a single chore PR rather than a stage.
- It catches the failure mode (silent total stop) that none of the other layers currently catch.

Defer Options B and C until at least one weekly digest has actually been received and read. Defer remaining incident-runbook work for the five observability-shaped classes until the digest exists; the digest itself will tell us which thresholds are worth alerting on, replacing guesswork with a week of real numbers.

## Single most concrete next step

Land a chore-sized PR adding `app/api/admin/digest/route.ts` (Clerk-gated or shared-secret) that returns the four aggregates above as JSON, plus a weekly GitHub Actions workflow that calls it and pipes the output through Resend to Vanyo. Everything else waits on that.
