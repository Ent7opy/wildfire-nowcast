# Poll route simplify attempt — rolled back below 15% threshold

**Date**: 2026-05-07
**Branch**: `chore/simplify-poll-route` (not pushed; reverted in tree)
**Target**: `app/api/aoi/poll/route.ts` (526 LOC starting)
**Outcome**: rolled back. Best honest reduction without behavioral risk: -55 LOC (10.5%). Below the brief's >15% threshold.

## What I tried

1. **Trimmed header / inline comments** that explained *what* (e.g. "Test-only injection point: lets the integration suite bypass the live FIRMS call without introducing a DI framework. Production leaves it null." × 3 nearly-identical blocks). Kept the *why* comments (build-without-blocking gates, COALESCE retry guard, Stage 7 retention sweep ordering rationale).
   - Saving: ~22 LOC of comments.
2. **Collapsed `runOneBucket` error-path emission** — the route had two ~18-LOC `return { …all-zeros, status: "error", error }` literals (one for FIRMS fetch error, one for the catch-all). Extracted `emptyOutcome(bucket, source, start, error?)`.
   - Saving: ~24 LOC.
3. **Extracted `closeChildAsError(db, id, msg, mapped)`** to share the 9-line `closeJobRun({ status:"error", firmsRequestCount:1, outcome, retryPending, finishedAt })` shape between the two error paths.
   - Saving: ~9 LOC.
4. **Simplified parent close status ternary** `runs.length === 0 ? "ok" : partial ? "partial" : "ok"` → `partial ? "partial" : "ok"` (when length is 0, `partial` is necessarily false).
   - Saving: 4 LOC, behavior identical.
5. **Inlined `if (body.bucket) { … } else { … }`** into a single conditional expression (4 → 3 lines).
6. **Removed `totalFirmsCalls` accumulator** in favor of `runs.length` (one per bucket; behaviorally equivalent and the variable was only used at the close site).
   - Saving: ~3 LOC.

Cumulative net: **-55 LOC, 469 → 526, 10.5% reduction**.

## Why I stopped

Brief gate: ">15% reduction or write a brainstorm note". Pushing for >15% required one of:

- **Extracting the briefs-phase loop and the notifications-phase loop into helpers** (`generateBriefsForEvents`, `dispatchAllBriefs`). These are ~22 + 20 LOC each, but they have multiple inputs/outputs (briefsGenerated, briefSkipReason, generatedBriefIds, briefError; notificationsSent/Failed/Skipped, notificationConfigMissing, briefError mutation). Extracting them means returning a 4–6-field result object and threading `briefError` accumulation in/out. Net LOC neutral or worse; semantic clarity arguably worse (you trade one local control flow for two helper signatures and a returned-state-object). **This is the "premature abstraction" pattern the brief warns against.**
- **Squashing the success-path return literal** (17 fields, mostly direct copies from `matchOutcome` + locals). Could spread `...matchOutcome` partially, but field names don't fully line up (`detectionsSkippedIndustrial` vs `eventsUpdated` are matchOutcome; the rest are local). Not worth the readability hit.
- **Compressing the SQL UPDATE in `closeJobRun`** by templating column names. Behavioral risk (SQL identifier interpolation), and Drizzle's tagged-template safety would be lost.

## Genuine duplication that *did* simplify cleanly

Just for scout-loop continuity if a future pass picks this up: the *real* simplification opportunities were

- The empty-outcome literal (now identified as `emptyOutcome` helper).
- The error-path `closeJobRun` shape (now identified as `closeChildAsError`).
- A few comment blocks that explained TypeScript rather than intent.

That's it. The rest of the route is genuinely linear orchestration — auth → body parse → env gates → DB → prune → enumerate → fan-out → close-parent → respond — and each step is doing distinct work. There isn't a 27%-reduction-shaped simplification hiding here the way `dispatch.ts` (PR #413) had with its reconcile/template/render duplications.

## Recommendation

If a perf parallelization pass (deferred from PR #432) lands later, that pass will naturally restructure `runOneBucket` into a `Promise.all` over a per-bucket promise, at which point the briefs/notify phases will likely move into their own functions for genuine reasons (independent concurrency boundaries). At that point the simplifications listed above (`emptyOutcome`, `closeChildAsError`, the comment trim, the ternary collapse) become free riders on the perf PR. Holding them back avoids a behavior-neutral churn-PR that would have to be re-touched.

No further simplify pass on this file is recommended without a behavioral driver.
