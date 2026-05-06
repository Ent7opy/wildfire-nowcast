# 0007 — Loosen the auto-merge gate to CI + reviewer LGTM

**Date:** 2026-05-06
**Status:** Accepted
**Stakeholder sign-off:** Vanyo (chat instruction 2026-05-06: _"Let's have auto merges when CI passes and reviewers approve the PR"_)
**Supersedes:** ADR 0006 (in part — see below)

---

## Context

ADR 0006 (2026-04-21) defined the stage-PR workflow and made every stage PR require Vanyo's manual review. The auto-merge gate in `loop.md` further restricted autonomous merges to dev- or scout-authored PRs only, excluded a long list of files (schema, migrations, deps, workflows), and capped diff size at 600 net additions.

In practice during the first day of loop operation (2026-05-06), the gate fired on nearly every productive PR:

- PR #388 (Stage 3 dev work, reviewer LGTM'd twice) — gated on `stage-pr:3` label.
- PR #389 (pm chore landing the Stage 3 brief) — gated on `agent: pm` author identity.
- PR #390 (scout dead-export cleanup) — passed the gate, but only after a separate fix unblocked Vercel.

Vanyo's observation: every productive piece of work funnels back through him. The loop is not autonomous; it is a slightly faster pipeline for Vanyo-mediated work.

## Decision

The auto-merge gate is reduced to:

1. All required CI checks green on the head SHA.
2. A `reviewer` agent review exists, body starts with `LGTM`, contains zero `BLOCKER:` lines.
3. The reviewer was not the same agent as the PR's author (preserves loop.md hard rule #5).
4. Diff touches none of the harness-self-modifying files (`pm/PM_CLAUDE.md`, `pm/north-star.md`, `pm/decisions/`, `loop.md`, `.claude/agents/`, `.claude/settings*.json`, `CLAUDE.md`, `AGENTS.md`).
5. No `needs-human` label.

Removed from the gate:

- Author identity check (`dev` or `scout` only).
- `stage-pr:*` label exclusion.
- `db/migrations/`, `db/schema/` exclusion (CI failure on a bad migration is the safety net).
- `.github/workflows/` exclusion (workflow changes go through the same review).
- `package.json` / `pnpm-lock.yaml` exclusion (deps no longer require a human eye).
- 600-line net-additions cap.

## Consequences

**Positive:**

- The loop is genuinely autonomous between idle ticks. Productive work flows from `dev` / `scout` / `pm` / orchestrator through `reviewer` to merge without Vanyo touching it.
- Vanyo's attention is reserved for: harness-self-modifying changes, escalations from `pm/blockers.md`, and explicit ADR-class decisions per `pm/PM_CLAUDE.md`.
- pm-authored chores (briefs, blocker reconciliations) merge autonomously when the reviewer approves — they no longer pile up.

**Negative / risk:**

- A bad migration that passes both unit and integration tests but breaks production-only invariants (e.g. a postgres-extension version mismatch) merges without Vanyo seeing it. Mitigation: the testcontainer integration tests run against the same `postgis/postgis:16-3.5` image production uses; deviations should be rare. If they happen, a follow-up ADR can re-add `db/migrations/` to the exclusion list.
- A bad dependency upgrade (e.g. a minor bump that introduces a breaking change at runtime) merges without Vanyo. Mitigation: the scout chore "dependency upgrade" cap is one package per UTC day — blast radius is small.
- The reviewer agent becomes the only adversarial check. If the reviewer is too lenient, low-quality code lands. Mitigation: the reviewer's contract (`.claude/agents/reviewer`) is explicitly adversarial; weakening it would be a noticed change because it lives in the harness-modifying-files exclusion list.

## What this does NOT change

- Loop.md hard rules #1-5 and #7 are unchanged.
- The reviewer's contract is unchanged.
- The orchestrator's responsibility to label `needs-human` and stop on gate failure is unchanged.
- `pm/blockers.md` escalation rules are unchanged.
- Branch protection on `master` is governed by GitHub repo settings, not this ADR. Vanyo removed the manual-merge requirement on 2026-05-06.

## Why ADR 0006 is superseded "in part"

ADR 0006 also defined the brief / branch-naming / label conventions for stage PRs. Those conventions are still good — keep them. Only the "Vanyo always reviews" clause is overturned by this ADR.
