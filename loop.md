# `loop.md` — Wildfire Nowcast Autonomous Heartbeat

**Authoritative protocol for the agent harness driving the A' pivot.**

This file is read at the start of every loop tick. Strategy lives in `pm/PM_CLAUDE.md`; this file describes only **mechanism**. If they ever conflict, `pm/PM_CLAUDE.md` wins.

Do not edit `loop.md` from inside the loop. Changes to this file are PR-reviewed by Vanyo and never auto-merged.

---

## Heartbeat — what to do each tick

Each tick, the orchestrator picks the **first** matching action and executes it. If a step requires more than ~15 minutes of autonomous work, delegate to the relevant subagent in a worktree (see Roles).

1. **Open PRs awaiting review (`needs-review` label and no `needs-human` label).**
   Spawn `reviewer` against the PR. On `LGTM` and CI green and auto-merge gate satisfied → merge with `--squash --delete-branch`. On `BLOCKER:` lines → label `needs-human`, summarize blockers in the PR body, stop.

2. **Open PRs with reviewer-blocking comments and no `needs-human` label.**
   Spawn `dev` on the PR's worktree to address the comments. Push fix, re-request review (`needs-review`).

3. **CI red on a PR opened by `dev` or `scout`.**
   Spawn `dev` against the failing branch with the failing logs (`gh run view <id> --log-failed`). Push fix.

4. **Resolved items (`[x]`) in `pm/blockers.md`.**
   Spawn `pm` to verify the resolution (e.g. env var actually set, secret reachable in CI), reconcile, remove the entry, append to the resolved-for-record block.

5. **Stage in progress per `pm/backlog.md` with no open PR for it.**
   Spawn `pm` to confirm the brief at `pm/briefs/NN-stage-N-*.md` is current. Then spawn `dev` with the brief. Open draft PR per ADR 0006. Apply label `stage-pr:N`.

6. **Stage ready to start** (prior stage merged, all stage-N entries in `pm/blockers.md` checked, no in-progress stage).
   Spawn `pm` to write the next brief in `pm/briefs/` if missing, update `pm/backlog.md` status, then go to step 5.

7. **Standing improvement work.** Spawn `scout` for one piece of work from the catalog. **No daily cap.** Multiple scout PRs per day are fine as long as each is concretely defensible (one chore per PR, ≤200 net-added lines). Catalog, in rough priority order — but the orchestrator picks based on signal, not strict ordering:

   - **Dead-code / unused-export audit** on a subtree not recently audited.
   - **Dependency upgrade** — one package, patch/minor.
   - **Doc drift** — references to moved paths, removed tech (e.g. legacy stack mentions), out-of-date examples in `docs/`, `README.md`, `CLAUDE.md`, `AGENTS.md`.
   - **Test coverage gaps** — find a module with branches uncovered by the existing suite, add the missing tests. Use `pnpm vitest --coverage` if available, else read code + grep for un-asserted branches.
   - **Refactor for simplicity** — apply the `simplify` skill to a single file or module that's grown bloated (>400 LOC for what it does). Push back on hand-defensive code, premature abstraction, dead branches.
   - **Code-quality pass** — tighten types (replace `any` / loose unions), remove `// TODO` items that are actually small, fix lint warnings (not errors — warnings).
   - **Performance audit** — pick a hot path (cron route, dispatcher, brief-generator) and look for synchronous loops over awaited calls, N+1 queries, redundant re-reads. Document findings in `pm/research-log/` even if no fix lands this PR.
   - **Accessibility audit** on a UI route — keyboard navigation, focus order, aria-labels, color-contrast.
   - **Security pass** — input validation, SQL-injection surface, CSRF surface, XSS in user-rendered content.
   - **Schema documentation** — add table / column comments to `db/schema/index.ts` for tables that have grown organically.
   - **API documentation** — generate or update OpenAPI for `app/api/*`, document handler contracts.
   - **Brainstorming notes to `pm/research-log/`** — speculative ideas, half-formed hypotheses, "I noticed X while doing Y" notes. Adversarial framing: write the doc you would want a successor agent to find.
   - **Legal / hygiene** — `LICENSE` file, security policy, `.editorconfig`, etc., if missing.
   - **Typo / broken-link sweep** in `pm/`, `docs/`, top-level `*.md`.

   The orchestrator's job is to keep the catalog open. **If `scout` returns "nothing to do" twice in a row, the catalog is wrong, not the repo.** The fix is to expand the catalog (which is what brainstorming notes are for), not to log idle.

8. **Product review when due.** Spawn `product-reviewer` if any of:
   - Three or more stages have merged since the last product review.
   - The most recent file in `pm/product-reviews/` is older than 7 UTC days (or none exists).
   - A candidate-direction-change ADR is being drafted.

   The reviewer reads strategy + spec + recent merges + the actual code surface and writes one structured review to `pm/product-reviews/YYYY-MM-DD.md`. The orchestrator opens a chore PR for the review file (label `needs-review`); it is gate-eligible under ADR 0007. Cap: **one product review per UTC week.** If this week's slot is used, fall through to step 7 — there is always more standing improvement work.

**There is no step 9.** "Idle" is not a steady state. If the orchestrator believes there is no productive work, the orchestrator is wrong — it should consult the catalog (step 7) or expand the catalog (write a brainstorm note). The only legitimate exit from a tick is either (a) a productive action was taken, or (b) the orchestrator is genuinely blocked on something in `pm/blockers.md` AND the catalog has been examined, in which case write a single line to `pm/loop-log/YYYY-MM-DD.md` naming the blocker and exit. **Pure idle ticks ("no state change") are a setup failure.**

---

## Roles (subagent contracts)

Defined in `.claude/agents/`. Each agent runs in its own git worktree where applicable.

| Role | Reads | Writes | Forbidden |
|---|---|---|---|
| **`pm`** | everything | `pm/briefs/`, `pm/research-log/`, `pm/signals/`, `pm/blockers.md`, `pm/backlog.md`, `pm/decisions/` (append-only) | code outside `pm/`. Writing a new ADR or amending `pm/PM_CLAUDE.md` / `pm/north-star.md` requires Vanyo sign-off via `pm/blockers.md`. |
| **`dev`** | everything | code, tests, migrations, workflows on a stage or chore branch | pushing to `master` or `pivot/a-prime` directly. `--no-verify`, force-push, hook bypass. Editing `pm/` (except adding research notes via `pm` only). |
| **`reviewer`** | the PR diff, brief acceptance criteria, CI logs | PR comments only | approving a PR whose author was itself. Approving anything in the auto-merge exclusion list (see gate). |
| **`scout`** | everything | a single-purpose improvement branch (chore, refactor, test-coverage gap, doc fix, brainstorm note, audit finding, etc.) | scope creep beyond ~200 net-added lines per PR. Touching schema, ingest, or auth on the same branch as another concern. (Schema/ingest/auth changes are fine — just dedicated PRs, since they need careful review.) |
| **`product-reviewer`** | strategy (`pm/north-star.md`, `pm/PM_CLAUDE.md`, `docs/SPEC-A-prime-v1.md`, `docs/pivot-architecture.md`), backlog, ADRs, recent merges, the actual code surface | one review file per dispatch in `pm/product-reviews/YYYY-MM-DD.md`; one-line entry to `pm/blockers.md` if findings are ADR-class | code, briefs, ADRs, PR diffs (that's `reviewer`). Recommending a contradiction of an existing ADR without explicit reason. Praise. Padding. |
| **`cutover`** | everything | one-shot only — Phase 0 deletion + harness landing | running more than once. |

All agents inherit the AGENTS.md hard rules (no fabricated data, no science-grade claims without evidence, push back on shortcuts).

---

## Auto-merge gate

A PR auto-merges iff **every** condition holds:

- All required CI checks green on the head SHA at merge time.
- A `reviewer` review exists with body starting `LGTM` and **zero** lines starting with `BLOCKER:`.
- The reviewer was not the same agent as the PR's author (loop.md hard rule #5).
- Diff touches **none** of:
  - `pm/PM_CLAUDE.md`, `pm/north-star.md`, `pm/decisions/`
  - `loop.md`, `.claude/agents/`, `.claude/settings*.json`
  - `CLAUDE.md`, `AGENTS.md`
- No `needs-human` label.

Anything failing → label `needs-human`, mention `@Ent7opy` in a single PR comment with the failing condition, stop.

**Why this gate is short:** per Vanyo directive 2026-05-06 + ADR 0007, the loop's autonomy is gated on CI + adversarial reviewer LGTM, not on author identity, file paths, or PR size. The remaining exclusion list is the small set of files that govern the harness itself — letting the loop edit those without Vanyo's eye creates a recursion risk (a bad agent change to `loop.md` cannot self-correct). Schema, migrations, deps, and stage labels are no longer gates; CI failure on a bad migration is the safety net.

---

## Escalation — what pages Vanyo

Append to `pm/blockers.md` (the canonical handoff queue) when any of:

- A required external service / secret is missing (Neon, AI Gateway, Resend, Clerk, CRON_SECRET, FIRMS key — pattern in the existing `pm/blockers.md`).
- `reviewer` rejects the same PR ≥2× with the same root cause.
- An ADR-class decision arises per `pm/PM_CLAUDE.md` § "Escalation to Vanyo": candidate direction change, >1 week build effort, load-bearing assumption contradicted, new tool / API key required.
- CI infrastructure is broken (workflow YAML invalid, secrets missing on the runner, runner unavailable — distinct from a failing test).
- The orchestrator has examined the catalog and genuinely cannot find productive work for three consecutive ticks (signal that the catalog is impoverished and the harness needs a Vanyo-side refresh, not that the repo is done).
- An auto-merge would have fired but the gate said `needs-human` for a reason that recurs across ≥3 PRs (signal that the gate needs tuning, not that each PR needs review).

Vanyo unblocks by checking `[x]` items in `pm/blockers.md`. Step 4 of the heartbeat reconciles on the next tick.

---

## State files

| Path | Purpose | Writer |
|---|---|---|
| `loop.md` | This protocol. | Vanyo only. |
| `pm/PM_CLAUDE.md` | Strategic doctrine. | Vanyo only. |
| `pm/north-star.md` | Working product thesis. | Vanyo only (or `pm` agent on Vanyo's instruction via blocker resolution). |
| `pm/decisions/` | ADRs. Append-only. | `pm` agent (with Vanyo sign-off, recorded in the ADR). |
| `pm/backlog.md` | Stage list, status. | `pm` agent. |
| `pm/blockers.md` | Vanyo handoff queue. | Any agent (append); Vanyo checks `[x]`; `pm` reconciles. |
| `pm/briefs/NN-*.md` | Versioned dev-agent prompts. | `pm` agent. |
| `pm/research-log/` | Condensed agent outputs (≤800 words). | `scout`, `pm`. |
| `pm/signals/` | Raw evidence (quotes, links, screenshots). | `scout`, `pm`. |
| `pm/loop-log/YYYY-MM-DD.md` | Per-tick decision log. | Orchestrator. |
| `pm/product-reviews/YYYY-MM-DD.md` | High-level product critique. | `product-reviewer` agent. |

---

## Hard rules (invariants)

1. **No fabricated data, ever.** From `AGENTS.md` and `pm/PM_CLAUDE.md`. Applies to research outputs, test fixtures, ADR citations, brief acceptance criteria.
2. **No direct push to `master` or `pivot/a-prime`.** Every change is a PR.
3. **`pm/decisions/` is append-only.** ADRs are never edited in place.
4. **Never bypass CI.** No `--no-verify`, no force-push that drops CI history, no skip-hooks.
5. **A reviewer cannot approve a PR whose author agent was itself.** Agent identity is recorded in the PR body's `agent: <role>` line.
6. **Stage PRs follow the auto-merge gate like any other PR.** Per ADR 0007 (supersedes ADR 0006), CI green + reviewer LGTM is sufficient. Vanyo's manual review is not required for stage PRs.
7. **If `loop.md` and `pm/PM_CLAUDE.md` conflict, `pm/PM_CLAUDE.md` wins.** Strategy supersedes mechanism.

---

## Loop-log entry format

Each tick writes one line to `pm/loop-log/YYYY-MM-DD.md`:

```
HH:MM:SSZ <action> <target> <result>
```

Examples:
```
09:00:14Z review PR#42 LGTM-merged
09:30:02Z dev stage-3-brief-gen pushed-needs-review
10:00:01Z idle no-eligible-work
```

When idle three ticks in a row, the next idle tick adds an `escalate` line and appends a blocker:

```
10:30:05Z escalate idle-3x appended-pm/blockers.md
```

---

## How the loop is hosted

Two viable harnesses:

- **`/loop` skill in dynamic mode** — single Claude Code session, self-paces via `ScheduleWakeup`. Best while `loop.md` is being tuned (you can interrupt mid-stream and edit). Default for week 1.
- **CronCreate scheduled task** — fires `claude /loop` every 30 min from the user's machine. Cleanest, survives restarts. Switch to this once the auto-merge gate has demonstrated 7 days without a false positive.

The loop never schedules itself. The harness is started by Vanyo and stopped by Vanyo.
