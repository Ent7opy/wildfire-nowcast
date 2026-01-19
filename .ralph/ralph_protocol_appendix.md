---

# RALPH LOOP ORCHESTRATOR INTEGRATION (REQUIRED)

## Input (from orchestrator)
- You must treat `.ralph/inbox/<ROLE>.json` as the source of truth for this run.

## Output (machine-readable)
- You must write a **single valid JSON file** to `.ralph/out/<ROLE>.json` (path is provided by the orchestrator in the prompt).
- Do not rely on stdout for the final result; the orchestrator reads the JSON file.

## General rules
- Keep scope tight: only work on the requested task / requested changes.
- Use **Ran:** / **Would run:** honesty for verification commands.
- If blocked, output status="blocked" and a short reason + the minimal next step.

