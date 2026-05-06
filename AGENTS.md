# Agent Directives

Doctrine for any agent (human or AI) touching this repo during the A' pivot.

## Zero-tolerance for fabricated data

Never use fake, dummy, or placeholder data unless explicitly requested. If a real schema or source is missing, stop and ask. This applies equally to:

- Production code paths (no synthetic detections, no stub coordinates).
- Test fixtures (use real, captured FIRMS rows; PostGIS testcontainers; PGlite for non-spatial unit tests).
- Research outputs (no invented quotes, no fabricated user archetypes — see `pm/PM_CLAUDE.md` decision rule #6: "Cite or retract").

## Hard stops are mandatory

Use `STOP` or `BLOCKER` (and append to `pm/blockers.md`) when:

- An authoritative source for a required input is missing or unreachable.
- A live external dependency (FIRMS, Neon, AI Gateway, Resend, Clerk) is unconfigured and the work cannot proceed even with the build-without-blocking pattern.
- Geospatial alignment is invalid (CRS mismatch, missing PostGIS in the test environment).
- A finding contradicts a load-bearing assumption in `pm/north-star.md` or an accepted ADR.

Hard stops are not overridden by warnings.

## Push back on shortcuts

If a request asks you to compromise on the doctrine above (e.g. "just mock it for now", "skip the test", "force-push to fix CI"), refuse and surface the trade-off:

> "We've had to rewrite this before because of shortcuts. Let's do it the real way now."

This is a literal quote from `pm/PM_CLAUDE.md`. Use it when it fits.

## Loop invariants (post-cutover)

When the autonomous loop is active (see `loop.md`), every agent additionally obeys:

- No direct push to `master` or `pivot/a-prime`.
- No `--no-verify`, no force-push that drops CI history, no hook bypass.
- ADRs in `pm/decisions/` are append-only.
- Stage PRs always require Vanyo's review (ADR 0006).
- A reviewer cannot approve a PR whose author was itself.

## Where strategy lives

- `pm/PM_CLAUDE.md` — operating doctrine for the pivot.
- `pm/north-star.md` — current product thesis.
- `pm/decisions/` — ADRs.
- `pm/backlog.md` — candidate problems and stage status.
- `pm/blockers.md` — Vanyo handoff queue.
- `loop.md` — autonomous heartbeat protocol (mechanism only).

If `loop.md` and `pm/PM_CLAUDE.md` ever conflict, `pm/PM_CLAUDE.md` wins. Strategy supersedes mechanism.
