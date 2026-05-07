# Launch-readiness path-forward — what the LOOP can flip vs what waits for Vanyo

Date: 2026-05-07
Author: scout
Source-of-truth: `pm/launch-readiness.md` (rebuilt 2026-05-07 after Stage 9 brief
landed). Cites SPEC `docs/SPEC-A-prime-v1.md` §"Acceptance for v1 launch".

## 1. Current state

From `pm/launch-readiness.md` §Summary as of this commit:

- **Passing (2/10):** #4 brief schema conformance, #7 positioning line.
- **Partial (3/10):** #2 cold-start ≤ 5 min, #8 license + repo public, #10
  rollback plan.
- **Not started (5/10):** #1 LTA archetype end-to-end, #3 infra cost, #5
  P95 latency ≤ 18 min, #6 gate pass-rate ≤ 8%, #9 LTA WRN newsletter post.

Note: #8 was flipped to passing earlier today by PR #407 (LICENSE landed at
repo root). The launch-readiness table currently still shows #8 as `partial`
because PM_CLAUDE has not yet rebuilt it post-#407; this brainstorm treats #8
as effectively `passing` and focuses on the user's four target items: #1, #2,
#9, #10.

## 2. Items the LOOP can move forward

### #2 — cold-start ≤ 5 min (partial → measurable, then `passing` only with a human)

What landed:
- Stage 7 (PR #406): draw-on-map, paste, upload polygon-input UX.
- Stage 9 (PR #422): watch-confirmed email + first-AOI backfill — closes Flow 1
  step 5 + step 6, the terminal events of the SLA.
- Cold-start runbook (PR #425) + area-cap warning (PR #426): documents the
  procedure end-to-end.

What the loop can still chore:
- A scout PR adding a small "cold-start dry-run" script under
  `scripts/cold-start-check.ts` that walks the path *as the server sees it*
  (create-AOI POST → watch-confirmed mailer invoked → first-poll backfill
  scheduled) and prints elapsed wall-clock for each hop. This is not the SPEC
  measurement (the SPEC requires a real browser) but it lets the loop assert
  the *server-side* ceiling is well under 5 min, which is the only failure
  mode the loop can realistically be blamed for. Outcome: #2 stays `partial`
  in the table but gains a sub-row "server path measured ≤ N s" so the
  remaining gap is unambiguously the human-in-browser portion.

Cannot flip to `passing` without Vanyo: the SPEC sentence is "measured on a
clean browser." That requires a human with a fresh Chrome profile and a
stopwatch. The loop will not fabricate that observation.

### #10 — rollback plan documented (partial → `passing` if Vercel-Hobby clears, else write the runbook)

What landed:
- `docs/pivot-architecture.md` §6 R1 high-level mitigation note.
- Incident-classes brainstorm (PR #439): catalogues the *kinds* of incidents
  a runbook would need to cover, not the runbook itself.

What the loop can chore (independent of Vanyo's Vercel-Hobby resolution):
- A scout PR writing 1–2 concrete runbooks under
  `docs/runbooks/`, each ≤ 150 LOC, for the highest-likelihood incident
  classes the brainstorm flagged. Suggested first two:
  1. **Vercel-Hobby revoked / commercial flag tripped** → migrate
     `app/api/*` routes to Cloudflare Workers (which the architecture doc
     already names as the rollback target); concrete steps for DNS cutover,
     env-var copy, cron-trigger re-wire to GitHub Actions only.
  2. **Neon free-tier exhausted / DB unavailable** → flip the FIRMS poll to
     "queue locally, defer dispatch" mode using a small SQLite checkpoint;
     concrete because Drizzle already abstracts the dialect (the PGlite/Neon
     two-backend pattern in `lib/firms/matcher.ts` is the precedent).

After 1 runbook lands, #10 stays `partial` but with much stronger evidence;
after 2 land, the SPEC sentence "documented" is satisfiable on the loop's
own evidence and the row can flip to `passing`. Net-added LOC: ~250 across
two PRs, well under the ≤ 200 per-PR cap.

## 3. Items that genuinely require Vanyo / external action

### #1 — LTA archetype served end-to-end
- **Action:** Vanyo identifies one named LTA-member land trust contact,
  walks them through cold-start in a real browser, observes a real FIRMS
  detection trigger a brief on their preserve polygon. Surfaced by
  `pm/product-reviews/2026-05-07.md` §6 and AGENTS.md no-fabricated-users.
- **Loop visibility:** PM_CLAUDE can verify by inspecting `aoi_briefs` for a
  row whose `aoi_id` belongs to a non-Vanyo Clerk `user_id`.

### #9 — LTA WRN newsletter post
- **Action:** Vanyo drafts and clears the post per brief 13 referenced in
  the SPEC; held until #1–#8 pass.
- **Loop visibility:** A markdown file in `pm/launch/newsletter-draft.md`
  with a "cleared by Vanyo" line and a target cycle date. Loop will not
  auto-draft this because the voice is Vanyo's and the LTA WRN editorial
  policy is external.

### Two upstream blockers (gate other items)
- **Vercel-Hobby non-commercial confirmation** (`pm/blockers.md` 2026-05-07)
  — gates #10 directly and #3 indirectly (cost claim assumes the tier).
- **Clerk webhook signing secret** (`pm/blockers.md` 2026-05-06) — does not
  gate any of #1/#2/#9/#10 directly but is a launch-day correctness item.
- **ICNF perimeter source** (`pm/blockers.md` 2026-05-07) — affects #1 for
  the Mediterranean archetype slice but not the LTA US slice the SPEC
  acceptance row names.

## 4. Recommendation: cheapest items to flip this week

The loop can fully drive **#10 partial → `passing`** by writing two
incident-class runbooks (Vercel-Hobby fallback + Neon-exhaustion fallback)
into `docs/runbooks/`. Two scout PRs, each ≤ 150 LOC, each defensible
without external observation. This is the single most concrete next step.

A weaker secondary candidate is the cold-start dry-run script for #2; it
does not flip the row but tightens the partial evidence.

## 5. Honest assessment: is 2/10 misleading?

Yes, somewhat. Of the ten SPEC acceptance items:

- **Pre-launch verifiable (5):** #2, #4, #7, #8, #10. Of these, #4, #7, #8
  pass and #2, #10 are partial-with-paths.
- **Post-launch observation (4):** #1, #3, #5, #6. By construction these
  cannot pass before a real user exists; the SPEC acceptance metric is "≥
  10 AOIs for 7 days" or "P95 across real polls."
- **Gated on the above (1):** #9 is held until 1–8 pass.

Honest score: **3/5 of pre-launchable items are passing** (#4, #7, #8) and
the remaining two have concrete paths (Stage 9 + browser-measurement for
#2; runbooks for #10). The 2/10 figure understates readiness because it
counts post-launch metrics in the denominator.

But the inverse is also true: the post-launch observation items are not
free passes — they will produce real numbers that may falsify the
architecture's cost / latency claims, and #1 in particular is the
load-bearing claim of the entire pivot. So "3/5 pre-launch" is fair
arithmetic; "ready to ship" is a stronger claim that depends on Vanyo's
LTA-contact action (#1) plus the Vercel-Hobby confirmation (#10's
remaining branch).

## 6. Cited

- `pm/launch-readiness.md` (2026-05-07 rebuild)
- `docs/SPEC-A-prime-v1.md` §"Acceptance for v1 launch" #1–#10
- `pm/blockers.md` 2026-05-07 (Vercel-Hobby, ICNF) and 2026-05-06 (Clerk webhook)
- `pm/product-reviews/2026-05-07.md` §4, §6
- PRs #406 (Stage 7), #407 (LICENSE), #411 (Stage 8), #422 (Stage 9), #425
  (cold-start runbook), #426 (area-cap warning), #439 (incident-classes
  brainstorm)
- `docs/pivot-architecture.md` §6 R1
