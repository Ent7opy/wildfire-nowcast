# Blockers — what Vanyo can unblock

Single source of truth for the small handful of things PM_CLAUDE cannot do directly: account creation, secret-setup on platforms without exposed APIs, local CLI installs.

Update mode: PM_CLAUDE adds entries; Vanyo marks `[x]` when done; PM_CLAUDE removes resolved entries during ADR / synthesis turns.

---

## Active

- [ ] **2026-05-07 — ICNF (Portugal) authority-perimeter source missing.** Stage 8 (PR #411) ships authority-perimeter pre-fetch for NIFC (US) and CWFIS (Canada) only. ICNF / ANEPC SGIF publishes only map tiles; `fogos.pt` returns incident **points**, not polygons (wrong shape for `BriefAuthorityPerimeterSchema.contains_detection`). Per AGENTS.md no-fabrication rule, no guess endpoint was added — Mediterranean AOIs (Natura 2000 archetype, target launch slice) currently get briefs without authority context. **What Vanyo does:** (a) ask ICNF whether they expose a public GeoJSON perimeter feed, OR (b) confirm with a Portuguese stewardship contact whether `fogos.pt` polygons (not points) are available via a non-public path, OR (c) accept ICNF deferral to v1.1 and explicitly choose CWFIS/NIFC-only as the v1 surface. Block-class: thesis-adherence — Mediterranean briefs are noticeably thinner without it. Surfaced by Stage 8 dev investigation 2026-05-07.

- [ ] **2026-05-07 — Confirm Vercel Hobby non-commercial clause for stewardship-NGO use.** R1 in `docs/pivot-architecture.md`. Pre-launch confirmation was a mitigation that was never executed. Surfaced by product review 2026-05-07 §4. **What Vanyo does:** send Vercel sales (sales@vercel.com) and/or post in the Vercel community a short message asking whether free-tier Hobby hosting is acceptable for a non-monetized open-source tool used by conservation NGOs (no revenue, no ads, no paid features; donations possible to Earth Tools as the non-profit operator). Until written confirmation lands, the launch infra plan has an unresolved legal risk that gates v1 launch acceptance #10 (rollback plan documentation is the mitigation, but only if confirmation does not arrive). Block-class: launch-blocking. PM_CLAUDE verifies by linking the reply email/thread URL into a follow-up ADR or a `pm/signals/` note, then moves this to Resolved.

- [ ] **2026-05-06 — Stage 5 (Clerk webhook signing secret).** Stage 5 brief 19 needs `CLERK_WEBHOOK_SIGNING_SECRET` for the Svix-verified `/api/webhooks/clerk` route that syncs the `users` table on `user.created` / `user.updated` / `user.deleted`. **What Vanyo does:** in the Clerk dashboard → Webhooks → "Add Endpoint" → URL `https://wildfire-nowcast.vercel.app/api/webhooks/clerk` (and the Preview URL pattern), subscribe to `user.created`, `user.updated`, `user.deleted`, copy the "Signing Secret" (starts with `whsec_`), add to Vercel `wildfire-nowcast` (Preview + Production) as `CLERK_WEBHOOK_SIGNING_SECRET`. Without this, Clerk users sign in but the local `users` table never populates — JIT provisioning will paper over it for the first request, but re-syncs (email change, deletion) will not propagate. PM_CLAUDE verifies via `vercel env ls` after Vanyo checks the box.

---

## Resolved (for the record)

- [x] **2026-05-06** — Stage 5 (Clerk) keys provisioned. Created Clerk free-tier project; added `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` and `CLERK_SECRET_KEY` to Vercel `wildfire-nowcast` (Preview + Production). Verified via `vercel env ls`.
- [x] **2026-05-06** — Stage 4 (Resend) key provisioned. `RESEND_API_KEY` on Vercel (Preview + Production). Sender-domain verification deferred until Stage 4 dev work — `onboarding@resend.dev` is fine for testing.
- [x] **2026-05-06** — Stage 3 (Vercel AI Gateway) key provisioned. `AI_GATEWAY_API_KEY` on Vercel (Preview + Production).
- [x] **2026-05-06** — Stage 2 (`CRON_SECRET`) provisioned on both Vercel (Preview + Production) and as a GitHub Actions repository secret. Cron `schedule:` un-commented in `.github/workflows/firms-poll.yml` in the same chore PR that recorded this resolution.
- [x] **2026-05-06** — Stage 2 (Docker) confirmed running locally for spatial-integration tests. Docker Engine 29.4.1.
- [x] **2026-05-06** — Stage 1 (Neon Postgres) provisioned. `DATABASE_URL` on Vercel `wildfire-nowcast` (Preview + Production), pooled connection string. Verified via `vercel env ls`.
- [x] **2026-05-06** — Vercel CLI installed locally and project linked (`vanyoivanov98-2068s-projects/wildfire-nowcast`). PM_CLAUDE can now run `vercel env ls`, `vercel logs`, `vercel env pull` to verify deploys directly instead of trusting the git integration blindly.
- [x] **2026-05-06** — Stage 7 (cutover) PR #386 merged. Legacy stack removed; `loop.md` + `.claude/agents/` harness landed on `master`. Pivot now operates as the trunk.
- [x] **2026-04-22** — `FIRMS_MAP_KEY` moved from Railway → Vercel env vars. Confirmed by Vanyo.
- [x] **2026-04-21** — `.claude/pm/` → `pm/` move. Vanyo's global gitignore excluded `.claude/`; resolved by moving PM workspace to a project-native path.
