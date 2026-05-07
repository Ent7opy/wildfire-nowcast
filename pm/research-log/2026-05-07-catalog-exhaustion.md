# Catalog exhaustion: what should the loop do when the obvious work runs out?

**Date:** 2026-05-07
**Author:** scout (brainstorm)
**Status:** self-reflection; recommendation at end

## 1. State of the catalog as of 2026-05-07

Approximate inventory of what has been worked across ~36 hours of autonomous loop activity (PRs through #401):

- **Test coverage** — broad fill across `lib/firms/`, `lib/ai/`, `lib/notify/`, `lib/auth/`, `lib/share/`, `lib/db/aoi-repository.ts`. Edge cases (timezone boundaries, malformed FIRMS rows, dispatcher behavioral fixes) covered in named PRs.
- **Schema docstrings** landed across Drizzle tables.
- **Security audits** — three (token endpoints, cron auth, app/api type audit). No critical findings outstanding.
- **A11y** — two passes through rules UI and AOI screens.
- **Brainstorms** — observability gap, incident classes, launch readiness, each with adversarial critique attached.
- **Drift** — README and `north-star.md` references converged on current paths.
- **Dead code** — removed across `lib/firms/` exports.
- **Refactors** — `decodeRows` consolidated across 4 files; cron route simplify pass attempted, rolled back below threshold.

Stages 0–7 are all merged. There is no open dev brief.

## 2. The diminishing-returns question

Each scout PR adds tests/docs at +X LOC and accretes review burden on whoever next walks the catalog. The marginal value of "one more `lib/firms/` test" trends toward zero, while the marginal cost (longer test suite, more files for a future reader to skim, more chances of a flake) is monotone non-negative.

A measurable threshold is plausible but not yet codified. Candidates:

- **3-PR drought rule.** If scout's last 3 PRs each closed with "no concrete fix; research only," pause.
- **Diff-size collapse.** If trailing 5 PR median net-added LOC < 30, that signals the surface is mined out.
- **Adversarial-finding rate.** If reviewer's last 3 reviews each said "no adversarial findings," the loop is rubber-stamping.

None of these are currently tracked as metrics — they would have to be summarized by the orchestrator from `pm/loop-log/`. I do not have hard numbers to assert any threshold has been crossed, but a qualitative read of recent PRs (#397–#401: brief landing, dispatcher fixes, harness add, status snapshot) suggests we are closer to the threshold than 36 hours ago.

## 3. The trap of self-justifying iteration

"The loop can iterate productively" and "the loop should iterate" are different claims, and the second does not follow from the first.

Two patterns suggest the loop is approaching navel-gazing:

- **Brainstorm recursion.** The observability chain produced three brainstorms in sequence, each finding a flaw in the previous. The third was useful (it correctly identified that "untested runbooks don't satisfy launch readiness"), but a fourth would almost certainly recurse on the same theme rather than discover new ground. This very note is at risk of being the fourth.
- **Pattern propagation past saturation.** `decodeRows` consolidation across 4 files was useful; a 5th file would be churn. The scout doesn't currently know where saturation is — it pattern-matches and proposes.

The honest read: the loop has demonstrated competence and is now at risk of confusing motion with progress.

## 4. Signals that the obvious catalog is exhausted

Concrete, checkable signals (not all currently instrumented):

- Last 3 scout PRs research-only, no code change merged.
- Last 3 reviewer LGTMs cite "no adversarial findings."
- Trailing 5 PR diffs all under 100 net LOC.
- Brainstorms recurse on themes already covered (observability, launch readiness, catalog exhaustion itself).
- No new entries in `pm/blockers.md` resolved by Vanyo in the last 24 hours (the loop is not waiting on external input but also not producing anything Vanyo needs to act on).

If 3+ of those fire simultaneously, the catalog is exhausted in the sense that matters: further scout activity has lower expected value than the review/accretion cost.

## 5. What should the loop do when those signals fire?

**Option A — Pause until external state changes.** Stop ticking. Wait for Vanyo to merge a harness PR, sign off on a launch decision, drop a new brief, or move a blocker. Honest, low-risk, and explicitly acknowledges the loop's dependency on human-driven direction.

**Option B — Shift to user-validation prep.** The loop can't run a cold-start test with a real LTA-WRN site manager, but it can: draft outreach scripts, prepare a one-page product explainer aimed at conservation-trust audiences, draft observation-session protocols Vanyo could run, prepare interview question banks. This is genuinely new surface area, not catalog filler.

**Option C — Continue at lower frequency.** Fewer ticks per UTC day, smaller scope. Defensible but mostly hides the diminishing-returns problem rather than addressing it.

## 6. Recommendation

**Option B, with Option A as the fallback if B produces nothing of substance within ~3 ticks.**

Reasoning:

- A is correct in spirit but maximally passive. It assumes the loop has nothing useful to contribute until external state changes; that is probably true for *code*, but probably false for *user-facing artifacts* the product still lacks.
- B addresses the actual constraint on the project right now: stages 0–7 are merged, but the product has no validated users. Outreach drafts, explainer copy, and observation protocols are work the loop can plausibly do well (text-shaped, evidence-cited, adversarially reviewable) and that Vanyo would otherwise have to do himself.
- C is the worst option because it preserves the appearance of progress while the marginal value of each PR continues to fall. It also makes the diminishing-returns problem harder to detect.

If the loop attempts B and the first 3 outputs are weak (generic, fabricated user personas, copy that doesn't reflect the north-star thesis), fall back to A and surface a blocker asking Vanyo for direction. Do not silently revert to defensive catalog churn.

The adversarial framing is worth saying directly: yes, the loop is approaching navel-gazing on the defensive-work axis. The honest move is to admit the catalog is mined out and either pivot the loop's output type (B) or pause (A), not to find ever-smaller fixes to justify the next tick.
