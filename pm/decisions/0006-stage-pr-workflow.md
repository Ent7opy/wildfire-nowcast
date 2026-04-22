# 0006 — Stage PR workflow

**Date:** 2026-04-21
**Status:** Accepted
**Stakeholder sign-off:** Vanyo

## Context

The A' pivot is structured as 8 sequential stages (per `docs/pivot-architecture.md`). Landing all 8 directly on `pivot/a-prime` would produce an unreviewable mega-diff at cutover. Vanyo flagged this; we agreed on per-stage PRs.

## Decision

**`pivot/a-prime` is the integration branch.** It always represents the cumulative "approved Stage N state" of the pivot.

### Per-stage workflow

1. **Stage branch off `pivot/a-prime`.** Naming: `stage-<N>-<short-name>` (e.g. `stage-1-aoi-crud`). Sub-stages of Stage 6 use `stage-6a-cut-spread`, `stage-6b-cut-denoiser`, etc.
2. **Dev agent works on the stage branch.** Brief lives at `pm/briefs/<NN>-stage-<N>-<name>.md`. Agent runs local build / lint / typecheck before pushing.
3. **Agent pushes branch, drafts a PR description in markdown** at the end of its work and reports it back to PM_CLAUDE.
4. **PM_CLAUDE opens the PR via `gh pr create`** with `pivot/a-prime` as the base, the agent's draft polished, the relevant ADR / brief / spec sections linked. PRs are opened as **draft by default** — Vanyo gets notified when CI is green, not when the agent first pushes.
5. **CI runs.** If failing, PM_CLAUDE respawns a fix agent against the same branch (referencing the failing CI logs via `gh run view`). Loop until green.
6. **PR moves from draft → ready for review** once CI is green and PM_CLAUDE has self-reviewed the diff.
7. **Vanyo reviews on GitHub, requests changes or merges.**
8. **PM_CLAUDE only dispatches the next stage agent after merge** so the next stage branches off the updated `pivot/a-prime`.

### Cutover (Stage 7)

Final PR is `pivot/a-prime → master`. Each commit in that diff has already been individually reviewed via its stage PR. Cutover review is "look at the integration narrative + verify all stage merges are present + check the deletion stages didn't drop anything we wanted to keep."

## Pre-flight discipline

Before dispatching any stage agent, PM_CLAUDE verifies:
- `pivot/a-prime` is at expected SHA (no surprise commits since last stage merge)
- Last stage PR is actually merged (not just approved)
- `pm/blockers.md` for that stage is unblocked (or the stage has a "build-without-blocking" pattern — see below)

## Build-without-blocking pattern

External services (Neon, Clerk, Resend, AI Gateway, FIRMS key) take time to set up and depend on Vanyo. Where possible, dev agents produce code that builds and tests green *without* the live service:

- Stage 1 (Neon): Drizzle migrations + CRUD + unit tests against PGlite in-memory; preview activates when `DATABASE_URL` is set on Vercel
- Stage 3 (LLM): structured-output schema + prompt + a stub provider for tests; preview activates when `AI_GATEWAY_API_KEY` is set
- Stage 4 (Resend): notification dispatcher with a stub channel; activates on `RESEND_API_KEY`

This decouples dev velocity from Vanyo's calendar.

## CI failure handling

PM_CLAUDE owns CI. On a red check:
1. `gh run view <run-id> --log-failed` to identify the failure
2. If trivial (typo, lint nit) — open Edit + push fix
3. If non-trivial — spawn a focused fix agent with the failing log + the suspected file paths
4. Push fix to the same stage branch, CI re-runs, loop
5. Vanyo only sees a notification when the PR is green and ready

## Out of scope

- Branch protection rules on master (defer; solo workflow doesn't need them yet)
- Required-review settings (Vanyo is the only reviewer; configuring won't change behavior)
- Conventional Commits enforcement (commit messages stay clear-prose; no tooling overhead)
