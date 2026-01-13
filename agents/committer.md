You are now staging and committing verified changes.

Goal:
- Small, semantic commits
- No artifacts
- No unrelated changes

---

## Commit standard

Use **Conventional Commits**:

Format:
`type(scope): summary`

Types: `fix`, `feat`, `refactor`, `test`, `docs`, `chore`  
Summary: imperative, ≤72 chars  
Scope: repo-area/module

---

## Commit safety gates (required)

### 1) Artifact / junk file gate
- Ran/Would run:
  - `git status --porcelain=v1`
  - `git ls-files -o --exclude-standard`
- Do NOT commit generated files (`*.nc`, `*.tif`, datasets, outputs, build dirs)
- Add `.gitignore` rules only if they prevent recurrence

### 2) Scope gate
- Ran/Would run:
  - `git diff --name-only`
  - `git diff`
- Only stage hunks relevant to the approved fixes
- Prefer `git add -p`
- Never use `git add .` without explicit confirmation

### 3) Quality gate
- Ran/Would run:
  - `git diff --check`
  - Repo-specific tests/lint from verification plan

---

## Commit execution procedure

A) Decide commit boundaries
- 1 commit for small fixes
- 2–6 commits for larger changes
- Map files → commits

B) Stage per commit
- Ran/Would run:
  - `git add -p`
  - `git status`

C) Commit
- Ran/Would run:
  - `git commit -m "type(scope): message"`

D) Repeat until done

E) Final cleanliness check
- Ran/Would run:
  - `git status --porcelain=v1`

---

## REQUIRED FINAL OUTPUT

1) Commit log:
   - Ran/Would run: `git --no-pager log --oneline -n <N>`

2) Diff summary:
   - Changed files + one-line reason each

3) Acceptance criteria → evidence:
   - Criterion → proof (test/log/manual) + location

4) Verification recap:
   - Commands run / would run + outcomes

5) Known issues / follow-ups:
   - Explicitly listed

---

## STOP CONDITION
If hooks/tests fail:
- Do NOT bypass
- Fix or report blocker + minimal next step
