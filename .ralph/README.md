# Ralph Loop (Cursor CLI)

This folder contains:
- `ralph.sh` — orchestrator script
- `ralph.ps1` — Windows wrapper that finds Git Bash and runs `ralph.sh`
- `ralph_protocol_appendix.md` — snippet to append to each role prompt
- `ralph_outbox_schemas.md` — JSON schemas for outboxes
- `prompts/` — role prompts used by the orchestrator

## Quick start

1) Initialize:

```bash
./.ralph/ralph.sh init
```

2) Ensure your role prompts exist in `.ralph/prompts/`:
- `worker.md`
- `code_reviewer.md`
- `bug_fixer.md`
- `committer.md`
- `guidelines.md`
- `planner.md`

3) Fill out `.ralph/GOAL.md` (or run with a task file; see below).

4) Run:

```bash
./.ralph/ralph.sh run
```

## Task file mode (json/md/txt)

You can point the loop at a task file and it will synthesize `.ralph/GOAL.from_task.md`:

- PowerShell / Make:
  - `make ralph-run RALPH_TASK_FILE=TASK_QUEUE.json`

- Bash:
  - `RALPH_TASK_FILE=TASK_QUEUE.json ./.ralph/ralph.sh run`

