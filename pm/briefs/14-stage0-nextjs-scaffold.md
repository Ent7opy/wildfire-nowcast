# Brief 14 — Stage 0: Next.js 16 scaffold on `pivot/a-prime` branch

## Why this exists

First execution stage of the A' pivot (ADR 0005). Intentionally small and purely additive. Goal: get a Next.js 16 skeleton deploying green to a Vercel preview URL on a `pivot/a-prime` branch, without touching existing code.

**Read in order:**
1. `/Users/vanyoivanov/Projects/wildfire-nowcast/pm/PM_CLAUDE.md`
2. `/Users/vanyoivanov/Projects/wildfire-nowcast/pm/north-star.md`
3. `/Users/vanyoivanov/Projects/wildfire-nowcast/pm/decisions/0005-problem-chosen-a-prime.md`
4. `/Users/vanyoivanov/Projects/wildfire-nowcast/docs/pivot-architecture.md` — **read the amendment at top first**; it supersedes "separate Vercel project" language
5. `/Users/vanyoivanov/Projects/wildfire-nowcast/docs/SPEC-A-prime-v1.md` — for voice / positioning, not for feature implementation (Stage 0 doesn't implement features)

## Goal

Create branch `pivot/a-prime`. Add a minimal Next.js 16 App Router scaffold at the repo root (not a subdir). Ship a single home page showing the canonical positioning line. Vercel preview for the branch must build green. Zero changes to existing `api/`, `ml/`, `ingest/`, `ui/`, `models/`, `configs/` directories.

## Scope (strict)

**Do:**
- Create `pivot/a-prime` branch from `master` (don't push if branch exists; pull latest)
- Initialize Next.js 16 at repo root via `pnpm dlx create-next-app@latest .` OR manual scaffold if the CLI errors on a non-empty dir
- Use: Next.js 16 App Router, React 19, TypeScript strict, Tailwind CSS v4, pnpm (matches Earth Tools' package manager)
- Add a single page at `app/page.tsx` with: the canonical positioning line ("Free, open, AI-native fire intelligence for stewardship — depth over speed."), a short 2–3 paragraph "what this is / what's coming" explainer in stewardship voice (read `docs/launch-draft-lta-wrn.md` for tone), and a placeholder "not ready yet" note
- Add `app/layout.tsx` with sensible metadata (title "Wildfire Nowcast", description matching the thesis)
- Commit in small logical chunks: (1) scaffold, (2) home page content, (3) README update if needed
- Push branch to origin; confirm Vercel picks it up (look at `.vercel/project.json` if present, or just trust the existing Vercel git integration)

**Do NOT:**
- Touch any files in `api/`, `ml/`, `ingest/`, `ui/`, `models/`, `configs/`, `Makefile`, `docker-compose.yml`, `railway.toml`, `railway.ingest.toml`, `Dockerfile*`
- Add database, auth, AI SDK, or any backend plumbing yet (those are Stage 1+)
- Delete anything (all deletion is Stage 6)
- Merge to master
- Rename the Vercel project

## Framework guidance (binding per Vercel plugin context)

- **Next.js 16 App Router.** Server Components by default; add `'use client'` only where interactivity requires it.
- **Use `proxy.ts` instead of `middleware.ts`** — Next.js 16 renamed it. Stage 0 probably doesn't need one; don't add it speculatively.
- **Tailwind CSS v4** with semantic design tokens. Keep styling minimal for Stage 0 — this is a placeholder page, not a design pass.
- **Do NOT use `@vercel/postgres` or `@vercel/kv`** — both are sunset. Stage 1 will wire Neon directly.
- **Package manager: pnpm.** Matches Earth Tools.
- Check framework-version-specific gotchas against https://nextjs.org/docs (the plugin's knowledge-update chunk flags training data as unreliable for Next.js 16).

## Verification

Before reporting done:
1. `pnpm install` completes clean
2. `pnpm build` completes clean with no TypeScript / ESLint errors
3. Branch is pushed to origin
4. Check that Vercel has picked up the branch (branch appears in `vercel` CLI output or Vercel dashboard URL reachable — don't actually log in to Vercel; trust the git-integration + reporting success if build passes locally)

## Output

1. **Branch on origin:** `pivot/a-prime` with 2–3 commits
2. **Write `pm/research-log/2026-04-21-stage0-scaffold.md`** (≤500 words): what you scaffolded, exact command run, framework versions locked in, any deviations from the brief, known issues, the preview URL pattern Vercel will generate (e.g. `wildfire-nowcast-git-pivot-a-prime-<user>.vercel.app` — format depends on Vercel team config; just note the pattern)

## When done

Respond with a single-paragraph completion status: the branch name + last commit SHA, Next.js + React versions, `pnpm build` status, the exact Vercel preview URL you expect (or note if uncertain).

## Time budget

~40 min. If you hit a 15-minute block on any single error, stop and report back.
