# Stage 0 — Next.js 16 scaffold on `pivot/a-prime` (2026-04-21)

**Agent:** 14 — Stage 0 dev
**Branch:** `pivot/a-prime` pushed to origin
**Commits:** `9988d24` (scaffold) + `02f87d4` (home page copy), on top of `0f91a19` from `origin/master`
**Build status:** `pnpm install` clean; `pnpm exec tsc --noEmit` clean; `pnpm exec eslint .` clean; `pnpm build` clean. `/` and `/_not-found` prerendered static via Turbopack.

## Framework versions (locked in)

| Package | Version |
|---|---|
| next | 16.2.4 |
| react | 19.2.4 |
| typescript | 5.9.3 |
| tailwindcss | 4.2.4 |
| eslint | 9.39.4 (flat config) |
| pnpm | 10.30.0 |

## Expected Vercel preview URL pattern

`https://wildfire-nowcast-git-pivot-a-prime-<team-slug>.vercel.app`

Exact URL not verified (no `.vercel/project.json` checked in; Vercel CLI not installed). Relying on existing `wildfire-nowcast` Vercel project's git integration to pick up the branch push automatically.

## Deviations from brief

1. **Research-log write denied** in the sub-agent sandbox — same intermittent `pm/**` permission issue as Phase 1/2. PM persisted this log on its behalf.
2. **README left untouched.** Brief said "if needed"; the existing README accurately describes `master` (the current service-offline state). Stage 0 adds no user-visible surface yet.
3. **`git stash -u` failed** at session start due to pre-existing uncommitted Phase 3 artifacts (`docs/SPEC-A-prime-v1.md`, `docs/launch-draft-lta-wrn.md`, `docs/pivot-architecture.md`, research JSONs, `.playwright-mcp/`). Worked around by branching directly from `origin/master` and staging only scaffold-owned paths by name — never `git add -A`. Pre-existing untracked files remain untracked on both branches. PM commits them separately on `master` as a follow-up (this turn).

## Scope discipline

Zero changes to: `api/`, `ml/`, `ingest/`, `ui/`, `models/`, `configs/`, `Makefile`, `docker-compose.yml`, `railway.toml`, `railway.ingest.toml`, `Dockerfile*`. Purely additive per Stage 0 brief.
