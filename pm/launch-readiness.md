# Launch readiness — A' v1

Last updated: 2026-05-07 (Stage 9 brief landed)
Maintained by: pm (per `pm/product-reviews/2026-05-07.md` §5 #10)

This file is the single source of truth for whether v1 is shippable.
"Stage merged" is not "acceptance passing." Each row maps a numbered
acceptance item from `docs/SPEC-A-prime-v1.md` §"Acceptance for v1
launch" to the current state of the artifact, the on-disk evidence,
and the stage or blocker that gates it.

Status legend:
- `passing` — code is in `master`, tests cover the criterion, and the
  criterion is observable in production-equivalent.
- `partial` — code exists but a stated sub-condition is missing (e.g.
  the page exists but a required element is not yet rendered).
- `not started` — no code in `master` addresses this.

| # | Acceptance item (from SPEC) | Status | Evidence | Gating |
|---|---|---|---|---|
| 1 | Land trust archetype served end-to-end. ≥1 LTA-member land trust has created ≥1 AOI with a real preserve polygon and received ≥1 brief from a real FIRMS detection. | not started | No outreach has happened (per `pm/PM_CLAUDE.md` no-fabricated-users constraint and review §4). The full backend supports it. | Stage 7 (so a non-Vanyo user can complete the loop), then launch acceptance #9 (newsletter post) — and a named LTA contact, currently absent. |
| 2 | Cold start to first watch ≤ 5 min, measured on a clean browser. | partial | Stage 7 (PR #406) shipped draw-on-map + paste/upload, so the polygon-input UX is no longer the gap. Sign-in (Clerk) → `/dashboard/aoi/new` → `POST /api/aoi` → AOI persisted is in place. **The terminal event of the SLA — a "watch confirmed" email (Flow 1 step 5) — still does not fire**, and there is no first-AOI backfill (Flow 1 step 6). Until both ship, the 5-min SLA is structurally unmeasurable. | Stage 9 (`pm/briefs/23-stage9-watch-confirmed-and-first-poll-backfill.md`) is the gating stage. Once Stage 9 ships, run a measured cold-start in a clean browser; if the email arrives within 5 min, this row flips to `passing`. |
| 3 | Infra cost claim holds. 7 consecutive days at ≥ 10 AOIs cost ≤ $1 across Vercel + Neon + AI Gateway. | not started | Cannot be measured until a real user load exists. Architecture target ($0 at 50u/100 AOIs) is documented in `docs/pivot-architecture.md`. | Post-launch observation. Also gated on the Vercel-Hobby blocker being resolved (the legal status of the hosting tier affects whether this metric is even measurable on the chosen infra). |
| 4 | Brief schema conformance = 100%. Every persisted brief validates against the v1 Zod schema; failures logged as gate misses. | passing | Stage 3 plumbing in `lib/ai/generate.ts` calls Gemini in structured-output mode with the Zod schema in `lib/ai/schema.ts`. Validation failure paths are tested. Persistence only happens after schema-validation success. | None — passing. |
| 5 | P95 end-to-end latency ≤ 18 min from FIRMS detection to brief send. | not started | Latency telemetry exists per-brief (`aoi_briefs.latency_ms` is the AI gateway call, not end-to-end). End-to-end measurement requires real FIRMS-poll → brief-send timestamps from production runs. | Post-launch observation. Stage 8 may add a `data_freshness` surface that lights this up for the user as well. |
| 6 | Gate passes ≤ 8% of ticks. | not started | Gate logic is in `lib/ai/gate.ts` (four conditions per SPEC §Flow 6). `aoi_briefs.gate_reason` records which condition fired. Pass-rate cannot be computed without poll runs against a real-user AOI set. | Post-launch observation. Possible chore: add a derived count to `job_runs` (`gates_evaluated` vs `briefs_generated`) to make the rate trivially queryable. |
| 7 | Landing page carries the canonical positioning line verbatim: "Free, open, AI-native fire intelligence for stewardship — depth over speed." | passing | `app/page.tsx:16-18` renders the line; the second clause uses an accent span but the text is verbatim. The footer also re-renders `POSITIONING_LINE` from `lib/export/positioning.ts`. | None — passing. |
| 8 | Repo public + MIT / Apache-2 licensed. Link from the landing page. | partial | Landing page has a `View source` link (`app/page.tsx:75`) pointing at `REPO_URL`. **No top-level `LICENSE` file exists in the repo** (verified by glob; matches were only in `node_modules`). The repo's GitHub-side license metadata may or may not be set; the on-disk artifact is missing. | One-line chore: add `LICENSE` (MIT, per spec preference) at repo root, confirm the GitHub repo is public. Vanyo-actionable for the public-flag if the repo is currently private. |
| 9 | LTA WRN newsletter post drafted (brief 13) and cleared by Vanyo. Held until items 1–8 pass; posted in the next LTA WRN cycle. | not started | Brief 13 referenced in spec, status unverified. Post is correctly held; the holding condition (1–8 passing) is not yet met. | Items 1, 2, 5, 6, 8 above. |
| 10 | Rollback plan documented. If Vercel Hobby's non-commercial clause becomes a blocker, the migration to Cloudflare Workers + Vercel Pro is documented in `docs/pivot-architecture.md`. | partial | `docs/pivot-architecture.md` §6 R1 mentions the migration as a mitigation; the explicit step-by-step "solo operator can execute in one afternoon" runbook is the standard the spec sets, and only the high-level mitigation note exists today. | Pair with the Vercel-Hobby blocker (`pm/blockers.md` 2026-05-07): if confirmation lands negative, write the runbook; if positive, file the confirmation and mark this row passing on the strength of the existing high-level note. |

## Summary

- **Passing now:** 2 of 10 (#4 schema conformance, #7 positioning line).
- **Partial:** 3 of 10 (#2 cold-start, #8 license, #10 rollback).
- **Not started / not measurable:** 5 of 10 (#1 real LTA user, #3
  infra cost observed, #5 latency observed, #6 gate-rate observed,
  #9 newsletter).

Three of the not-started rows (#3, #5, #6) are post-launch
observations that cannot pass before launch by definition — they are
launch-week metrics. The actually-blocking gaps are #1, #2, #8, #9, #10.

## What needs to happen, in order

1. **Stage 9 (`pm/briefs/23-stage9-watch-confirmed-and-first-poll-backfill.md`)**
   flips #2 from `partial` (unmeasurable) to measurable by shipping
   the watch-confirmed email (Flow 1 step 5) and first-AOI backfill
   (Flow 1 step 6). Stage 7 (PR #406) already shipped the draw-on-map
   path; Stage 9 closes the remaining cold-start gap. Once Stage 9
   merges and a clean-browser cold-start is timed, #2 flips to
   `passing`.
2. **License chore** — one-line PR adding a top-level `LICENSE` file
   flips #8 from `partial` to `passing` (assuming the repo is already
   public on GitHub).
3. **Vercel-Hobby blocker resolution** (Vanyo-actionable, in
   `pm/blockers.md`) flips #10 from `partial` to either `passing`
   (positive confirmation) or to a small docs PR writing the rollback
   runbook (negative confirmation).
4. **Stage 8** (perimeter fetch + data-freshness honesty + outreach
   plan) ✅ merged (PR #411). NIFC + CWFIS perimeters shipped; ICNF
   deferred to v1.1 (active blocker). The thesis-adherence improvement
   for the brief depth that #1's eventual real user will judge the
   tool on is now in `master` for North American AOIs.
5. **Real LTA contact** (Vanyo-actionable — review §6 calls this
   out): walk one named user through cold-start → first brief.
   Without this step #1 cannot pass and #9 should not post.

## Update protocol

PM_CLAUDE updates this file when:
- A stage merges that flips a row's status (note the PR # in the
  Evidence column).
- A blocker that gates a row is resolved.
- A product review surfaces a new gap that maps to one of the rows.

This file is intentionally short and rebuilt per pass; do not let it
accrete history. History lives in PRs and ADRs.
