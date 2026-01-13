You are now implementing **approved fixes** from the review.

Scope:
- Implement **only** the agreed C1 items (and explicitly approved C2 items)
- Follow repo patterns exactly
- Keep changes minimal and targeted

---

## Non-negotiable rules

1) **No new scope**
- Do not fix anything not listed as approved
- If you discover a new issue, stop and report

2) **Repo consistency**
- Match existing layering, naming, config, logging, error handling
- Avoid refactors unless required for correctness

3) **Execution honesty**
- Use **Ran:** or **Would run:** for all commands

4) **Data & correctness > elegance**
- Do not optimize, restructure, or rename unless it fixes a real issue

---

# FIX MODE OUTPUT

## A) Implementation plan (confirmed)
- Restate which review items are being implemented
- List files to be touched

## B) Implementation steps
For each step:
- File(s)
- Change made
- Why here (repo pattern + path)
- Verification (**Ran / Would run**) + expected result

## C) Verification results
- Unit tests (paths + commands)
- Integration/E2E/manual checks (if any)
- Observed outputs/logs

---

## STOP CONDITION
Do NOT commit yet.
Handoff to Semantic Commits Prompt after verification.
