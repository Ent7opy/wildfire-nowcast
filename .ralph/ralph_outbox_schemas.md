# Ralph outbox schemas (copy into your role prompts)

## Worker outbox (.ralph/out/worker.json)
{
  "selected_task_id": "T01",
  "status": "implemented | blocked",
  "summary": "short, concrete summary",
  "files_changed": ["path1", "path2"],
  "verification": ["Ran: ...", "Would run: ..."],
  "blockers": ["if blocked..."]
}

## Reviewer outbox (.ralph/out/review.json)
{
  "task_id": "T01",
  "verdict": "approve | changes_requested",
  "requested_changes": [
    {"severity":"must","title":"...","evidence":"path:line or command output","suggested_fix":"..."}
  ],
  "notes": ["optional"]
}

## Bug fixer outbox (.ralph/out/fix.json)
{
  "task_id": "T01",
  "status": "fixed | blocked",
  "applied_changes": ["..."],
  "files_changed": ["..."],
  "verification": ["Ran: ...", "Would run: ..."],
  "blockers": ["if blocked..."]
}

## Committer outbox (.ralph/out/commit.json)
{
  "task_id": "T01",
  "commits": [
    {"sha":"abc1234","message":"feat(scope): ..."}
  ],
  "verification": ["Ran: ...", "Would run: ..."],
  "notes": ["optional"]
}

