#!/usr/bin/env bash
set -euo pipefail

# Ralph Loop Orchestrator for Cursor CLI (multi-role)
#
# Requirements:
#   - cursor-agent installed + authenticated
#   - jq installed
#   - git repo (recommended)
#
# Usage:
#   ./.ralph/ralph.sh init
#   ./.ralph/ralph.sh plan
#   ./.ralph/ralph.sh run
#   ./.ralph/ralph.sh status
#
# Environment:
#   CURSOR_API_KEY     (or login via cursor-agent login)
#   RALPH_MODEL        (optional) e.g. "auto", "gpt-5", ...
#   RALPH_MAX_ITERS    (optional) default 50
#   RALPH_REVIEW_MAX   (optional) default 5
#   RALPH_AGENT_BIN    (optional) default "cursor-agent"
#   RALPH_PROMPTS_DIR  (optional) default ".ralph/prompts"
#
# Notes:
# - In CI / some headless environments cursor-agent may hang after producing output.
#   This script uses an outbox JSON file contract and will terminate the agent once the
#   outbox contains valid JSON.

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
RALPH_DIR="${RALPH_DIR:-$ROOT/.ralph}"
PROMPTS_DIR="${RALPH_PROMPTS_DIR:-$RALPH_DIR/prompts}"
INBOX_DIR="$RALPH_DIR/inbox"
OUT_DIR="$RALPH_DIR/out"
LOG_DIR="$RALPH_DIR/logs"
export PATH="$RALPH_DIR/bin:$PATH"

DEFAULT_GOAL_FILE="$RALPH_DIR/GOAL.md"
GOAL_FILE="${RALPH_GOAL_FILE:-$DEFAULT_GOAL_FILE}"
TASK_FILE="${RALPH_TASK_FILE:-${2:-}}"
if [[ -n "$TASK_FILE" && -z "${RALPH_GOAL_FILE:-}" ]]; then
  GOAL_FILE="$RALPH_DIR/GOAL.from_task.md"
fi
PLAN_FILE="$RALPH_DIR/plan.json"
STATE_FILE="$RALPH_DIR/state.json"

AGENT_BIN="${RALPH_AGENT_BIN:-cursor-agent}"
MODEL="${RALPH_MODEL:-auto}"
MAX_ITERS="${RALPH_MAX_ITERS:-50}"
MAX_REVIEW_CYCLES="${RALPH_REVIEW_MAX:-5}"

# Prompt files (you can override by placing your own files in .ralph/prompts/)
PLANNER_PROMPT="$PROMPTS_DIR/planner.md"
WORKER_PROMPT="$PROMPTS_DIR/worker.md"
REVIEWER_PROMPT="$PROMPTS_DIR/code_reviewer.md"
FIXER_PROMPT="$PROMPTS_DIR/bug_fixer.md"
COMMITTER_PROMPT="$PROMPTS_DIR/committer.md"
GUIDELINES_PROMPT="$PROMPTS_DIR/guidelines.md"

mkdir -p "$RALPH_DIR" "$PROMPTS_DIR" "$INBOX_DIR" "$OUT_DIR" "$LOG_DIR"

die() { echo "❌ $*" >&2; exit 1; }
info() { echo "ℹ️  $*" >&2; }
ok() { echo "✅ $*" >&2; }

need_cmd() { 
  command -v "$1" >/dev/null 2>&1 || die "Missing dependency: $1"; 
}

timestamp() { date +"%Y%m%d_%H%M%S"; }

json_ok() { jq -e . "$1" >/dev/null 2>&1; }

# If RALPH_TASK_FILE is set, synthesize a GOAL markdown file from it.
prepare_goal_from_task_file() {
  [[ -n "$TASK_FILE" ]] || return 0
  [[ -f "$TASK_FILE" ]] || die "Missing task file: $TASK_FILE"

  local ext
  ext="${TASK_FILE##*.}"
  case "$ext" in
    json)
      cat >"$GOAL_FILE" <<EOF
# Goal

Execute the tasks described in \`$TASK_FILE\`.

## Task file (verbatim)

\`\`\`json
$(cat "$TASK_FILE")
\`\`\`
EOF
      ;;
    md)
      cat >"$GOAL_FILE" <<EOF
# Goal

Execute the tasks described in \`$TASK_FILE\`.

## Task file (verbatim)

$(cat "$TASK_FILE")
EOF
      ;;
    *)
      cat >"$GOAL_FILE" <<EOF
# Goal

Execute the tasks described in \`$TASK_FILE\`.

## Task file (verbatim)

\`\`\`text
$(cat "$TASK_FILE")
\`\`\`
EOF
      ;;
  esac
}

# Wait until a JSON file exists and is valid JSON (or timeout).
wait_for_json() {
  local path="$1"
  local seconds="${2:-600}"
  local log="${3:-}"
  local pid="${4:-}"
  local i=0
  while (( i < seconds )); do
    if [[ -f "$path" ]] && json_ok "$path"; then
      return 0
    fi

    # If the agent process already exited, don't wait the full timeout.
    if [[ -n "${pid:-}" ]] && ! kill -0 "$pid" >/dev/null 2>&1; then
      info "Agent process exited before producing outbox JSON."
      if [[ -n "${log:-}" ]]; then
        info "See log: $log"
        if [[ -f "$log" ]]; then
          info "Last agent output (tail):"
          tail -n 40 "$log" >&2 || true
        fi
      fi
      return 2
    fi

    # Periodic heartbeat so it doesn't look stuck.
    if (( i % 10 == 0 )); then
      if [[ -n "${log:-}" && -f "$log" ]]; then
        local bytes=""
        bytes="$(wc -c <"$log" 2>/dev/null || true)"
        info "Waiting for outbox JSON: $path (elapsed ${i}s). Log bytes: ${bytes:-?}"
      else
        info "Waiting for outbox JSON: $path (elapsed ${i}s)."
      fi
    fi

    sleep 1
    ((i++))
  done
  return 1
}

# Run cursor-agent with a role prompt wrapper. Expects the agent to write valid JSON to OUTBOX.
run_agent() {
  local role="$1"
  local role_prompt_file="$2"
  local inbox_file="$3"
  local outbox_file="$4"

  [[ -f "$role_prompt_file" ]] || die "Missing prompt file: $role_prompt_file"
  [[ -f "$inbox_file" ]] || die "Missing inbox file: $inbox_file"
  need_cmd "$AGENT_BIN"

  rm -f "$outbox_file"
  mkdir -p "$(dirname "$outbox_file")"

  local log="$LOG_DIR/$(timestamp)_${role}.jsonl"

  # Wrapper prompt: keep short; agent should read inbox + repo files for context.
  local wrapper
  wrapper="$(cat "$GUIDELINES_PROMPT" 2>/dev/null || true)"
  wrapper+=$'\n\n'
  wrapper+="## ROLE PROMPT (${role})"$'\n'
  wrapper+="(Read carefully, then follow the Orchestrator Protocol at the bottom.)"$'\n\n'
  wrapper+="$(cat "$role_prompt_file")"$'\n\n'
  wrapper+="---"$'\n'
  wrapper+="# Orchestrator Protocol (REQUIRED)"$'\n'
  wrapper+="- You are running under a Ralph loop orchestrator."$'\n'
  wrapper+="- Your **ONLY** authoritative input is JSON at: $inbox_file"$'\n'
  wrapper+="- You must produce your **ONLY** machine output by writing valid JSON to: $outbox_file"$'\n'
  wrapper+="- Do not write any other outbox files unless explicitly instructed in the inbox."$'\n'
  wrapper+="- If blocked, still write outbox JSON with status=\"blocked\" and a short reason + next step."$'\n'
  wrapper+="- IMPORTANT: Only the COMMITTER role may run git commit."$'\n'
  wrapper+="- When done, stop."$'\n'

  # Build CLI args
  local -a args
  args+=("$AGENT_BIN" "-p" "--force" "--output-format" "stream-json")
  # MCP approvals in headless mode often require --approve-mcps (safe to ignore if unsupported)
  args+=("--approve-mcps")
  if [[ -n "$MODEL" ]]; then
    args+=("--model" "$MODEL")
  fi
  args+=("$wrapper")

  info "Running role: $role"
  set +e
  "${args[@]}" >"$log" 2>&1 &
  local pid=$!
  set -e

  wait_for_json "$outbox_file" 1800 "$log" "$pid"
  local wait_rc=$?
  if [[ "$wait_rc" -eq 0 ]]; then
    # Kill agent if it lingers (known in some headless environments).
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1 || true
    fi
    wait "$pid" >/dev/null 2>&1 || true
    ok "$role produced outbox: $outbox_file"
    return 0
  elif [[ "$wait_rc" -eq 2 ]]; then
    # Agent exited early; log already printed by wait_for_json
    return 1
  fi

  info "Outbox not produced in time. See log: $log"
  # Try to stop agent
  if kill -0 "$pid" >/dev/null 2>&1; then
    kill "$pid" >/dev/null 2>&1 || true
  fi
  wait "$pid" >/dev/null 2>&1 || true
  return 1
}

ensure_templates() {
  # Only create missing templates; never overwrite user edits.
  if [[ ! -f "$GOAL_FILE" ]]; then
    cat >"$GOAL_FILE" <<'EOF'
# Goal

Describe the overall objective, scope, constraints, and any acceptance criteria at the project level.

## Context
- Repo: (what is this repo for?)
- Constraints: (no new deps, performance, etc.)
- Commands: (how to run tests/lint/build)

## Definition of Done
- (bullets)
EOF
    ok "Created $GOAL_FILE"
  fi

  if [[ ! -f "$PLANNER_PROMPT" ]]; then
    cat >"$PLANNER_PROMPT" <<'EOF'
# Planner Agent (Ralph Loop)

You create a task plan for the goal and write it to the outbox JSON file specified by the orchestrator.

## Inputs
Read:
- .ralph/GOAL.md (primary requirements)
- repository structure + existing patterns
- any existing .ralph/state.json (if present) to avoid duplicating completed work

## Output requirements (STRICT)
Write valid JSON to the outbox path in the inbox. The JSON must match:

{
  "version": 1,
  "goal_summary": "1-3 sentences",
  "tasks": [
    {
      "id": "T01",
      "title": "short imperative title",
      "priority": "P0|P1|P2",
      "depends_on": ["T.."],
      "description": "short but concrete",
      "files_likely": ["path/like/this.ts"],
      "acceptance": ["observable criteria..."],
      "verification": ["commands or checks; use 'Would run:' if not executed"]
    }
  ]
}

Rules:
- 5–20 tasks. Prefer smaller tasks that can be implemented + reviewed in one loop.
- Include dependencies only when truly needed.
- Keep tasks repo-grounded: refer to actual paths/patterns you can find.
- No implementation.

EOF
    ok "Created $PLANNER_PROMPT"
  fi

  # The remaining role prompts are expected to exist (you'll copy your md files here).
}

init_state_from_plan() {
  [[ -f "$PLAN_FILE" ]] || die "Missing plan file: $PLAN_FILE"
  jq '
    . as $plan |
    {
      version: 1,
      created_at: (now | todate),
      goal_summary: ($plan.goal_summary // ""),
      tasks: ($plan.tasks | map(. + {status:"todo", commits:[], notes:[], review_cycles:0}))
    }
  ' "$PLAN_FILE" >"$STATE_FILE"
  ok "Initialized state: $STATE_FILE"
}

status() {
  [[ -f "$STATE_FILE" ]] || die "No state.json yet. Run: ./.ralph/ralph.sh plan"
  echo
  echo "== Ralph Status =="
  jq -r '
    .tasks as $t |
    "Tasks: \($t|length) | Done: \($t|map(select(.status=="done"))|length) | Todo: \($t|map(select(.status=="todo"))|length) | Blocked: \($t|map(select(.status=="blocked"))|length)"' "$STATE_FILE"
  echo
  jq -r '
    .tasks
    | sort_by(.priority)
    | map("\(.id) [\(.priority)] \(.status) - \(.title)")
    | .[]
  ' "$STATE_FILE"
  echo
}

# Return runnable tasks as a compact list: [{id,title,priority,depends_on}]
runnable_tasks_json() {
  jq '
    .tasks as $tasks |
    [ .tasks[]
      | select(.status=="todo")
      | select(
          (.depends_on // []) | map(. as $d | ($tasks | .[] | select(.id == $d)).status == "done") | all
        )
      | {id,title,priority,depends_on}
    ]
  ' "$STATE_FILE"
}

choose_task() {
  local runnable
  runnable="$(runnable_tasks_json)"
  local count
  count="$(echo "$runnable" | jq 'length')"
  if [[ "$count" -eq 0 ]]; then
    return 1
  fi

  # Prepare inbox for worker to choose and implement one task
  local inbox="$INBOX_DIR/worker.json"
  echo "$runnable" | jq '{
    goal_file: ".ralph/GOAL.md",
    plan_file: ".ralph/plan.json",
    state_file: ".ralph/state.json",
    runnable_tasks: .,
    instruction: "Pick ONE runnable task (most important), implement it fully, then write outbox with selected_task_id and status. Do not commit.",
    outbox_schema: {
      selected_task_id: "T01",
      status: "implemented|blocked",
      summary: "short, concrete summary",
      files_changed: ["path1","path2"],
      verification: ["Ran: ...","Would run: ..."],
      blockers: ["if blocked..."]
    }
  }' >"$inbox"

  run_agent "worker" "$WORKER_PROMPT" "$inbox" "$OUT_DIR/worker.json" || { info "Worker failed to produce outbox."; return 1; }

  local task_id
  task_id="$(jq -r '.selected_task_id // empty' "$OUT_DIR/worker.json")"
  [[ -n "$task_id" ]] || die "Worker outbox missing selected_task_id"

  echo "$task_id"
}

update_task_status() {
  local task_id="$1"
  local new_status="$2"
  local note="${3:-}"
  tmp="$(mktemp)"
  jq --arg id "$task_id" --arg st "$new_status" --arg note "$note" '
    .tasks = (.tasks | map(if .id==$id then .status=$st | (if $note!="" then .notes += [$note] else . end) else . end))
  ' "$STATE_FILE" >"$tmp"
  mv "$tmp" "$STATE_FILE"
}

bump_review_cycles() {
  local task_id="$1"
  tmp="$(mktemp)"
  jq --arg id "$task_id" '
    .tasks = (.tasks | map(if .id==$id then .review_cycles = ((.review_cycles // 0)+1) else . end))
  ' "$STATE_FILE" >"$tmp"
  mv "$tmp" "$STATE_FILE"
}

get_task_json() {
  local task_id="$1"
  jq --arg id "$task_id" '.tasks | map(select(.id==$id)) | .[0]' "$STATE_FILE"
}

review_loop() {
  local task_id="$1"

  local cycles=0
  while (( cycles < MAX_REVIEW_CYCLES )); do
    cycles=$((cycles+1))

    # Reviewer inbox
    local inbox="$INBOX_DIR/reviewer.json"
    jq --arg id "$task_id" '
      {
        goal_file: ".ralph/GOAL.md",
        plan_file: ".ralph/plan.json",
        state_file: ".ralph/state.json",
        task: (.tasks | map(select(.id==$id)) | .[0]),
        instruction: "Review the repo changes for this task. Do NOT implement. You may run: git status, git diff, tests. Output verdict approve|changes_requested plus change list.",
        outbox_schema: {
          task_id: "T01",
          verdict: "approve|changes_requested",
          requested_changes: [
            {severity:"must", title:"...", evidence:"path:line or command output", suggested_fix:"..."}
          ],
          notes: ["optional"]
        }
      }
    ' "$STATE_FILE" >"$inbox"

    run_agent "reviewer" "$REVIEWER_PROMPT" "$inbox" "$OUT_DIR/review.json" || { info "Reviewer failed."; return 1; }

    local verdict
    verdict="$(jq -r '.verdict' "$OUT_DIR/review.json")"

    if [[ "$verdict" == "approve" ]]; then
      ok "Review approved"
      return 0
    fi

    if [[ "$verdict" != "changes_requested" ]]; then
      die "Reviewer verdict must be approve or changes_requested. Got: $verdict"
    fi

    bump_review_cycles "$task_id"

    # Bug fixer inbox contains requested changes
    local fix_inbox="$INBOX_DIR/bug_fixer.json"
    local task_json
    task_json="$(get_task_json "$task_id")"
    jq --argjson task "$task_json" '
      {
        goal_file: ".ralph/GOAL.md",
        plan_file: ".ralph/plan.json",
        state_file: ".ralph/state.json",
        task: $task,
        requested_changes: (.requested_changes // []),
        instruction: "Implement ONLY the requested changes. Do not expand scope. Do not commit.",
        outbox_schema: {
          task_id: "T01",
          status: "fixed|blocked",
          applied_changes: ["..."],
          files_changed: ["..."],
          verification: ["Ran: ...","Would run: ..."],
          blockers: ["if blocked..."]
        }
      }
    ' "$OUT_DIR/review.json" >"$fix_inbox"

    run_agent "bug_fixer" "$FIXER_PROMPT" "$fix_inbox" "$OUT_DIR/fix.json" || { info "Bug fixer failed."; return 1; }
    info "Bug fixer applied changes; re-reviewing..."
  done

  die "Exceeded max review cycles ($MAX_REVIEW_CYCLES) for $task_id"
}

commit_task() {
  local task_id="$1"

  local inbox="$INBOX_DIR/committer.json"
  jq --arg id "$task_id" '
    {
      goal_file: ".ralph/GOAL.md",
      plan_file: ".ralph/plan.json",
      state_file: ".ralph/state.json",
      task: (.tasks | map(select(.id==$id)) | .[0]),
      instruction: "Create semantic commits for the approved changes. Output commit list with shas + messages.",
      outbox_schema: {
        task_id: "T01",
        status: "committed|blocked",
        commits: [{sha:"abc1234", message:"feat(scope): ..."}],
        verification: ["Ran: ...","Would run: ..."],
        notes: ["optional"]
      }
    }
  ' "$STATE_FILE" >"$inbox"

  run_agent "committer" "$COMMITTER_PROMPT" "$inbox" "$OUT_DIR/commit.json" || { info "Committer failed."; return 1; }

  local c_status
  c_status="$(jq -r '.status // "committed"' "$OUT_DIR/commit.json")"
  if [[ "$c_status" == "blocked" ]]; then
    local reason
    reason="$(jq -r '.notes | join("; ")' "$OUT_DIR/commit.json")"
    update_task_status "$task_id" "blocked" "committer blocked: $reason"
    return 1
  fi

  # safer update: read commits directly from outbox
  tmp="$(mktemp)"
  jq --arg id "$task_id" --slurpfile c "$OUT_DIR/commit.json" '
    .tasks = (.tasks | map(if .id==$id then .commits = ($c[0].commits // []) else . end))
  ' "$STATE_FILE" >"$tmp"
  mv "$tmp" "$STATE_FILE"
  return 0
}

plan() {
  need_cmd jq

  info "Cleaning up previous Ralph state and outputs..."
  rm -f "$STATE_FILE" "$PLAN_FILE"
  rm -rf "$INBOX_DIR" "$OUT_DIR"
  mkdir -p "$INBOX_DIR" "$OUT_DIR"

  prepare_goal_from_task_file
  [[ -f "$GOAL_FILE" ]] || die "Missing goal file: $GOAL_FILE (run ./.ralph/ralph.sh init first)"

  # Planner inbox
  local inbox="$INBOX_DIR/planner.json"
  cat >"$inbox" <<EOF
{"goal_file":"$GOAL_FILE","existing_state_file":"$STATE_FILE","instruction":"Create a concrete task plan and write it as JSON to the outbox path."}
EOF

  run_agent "planner" "$PLANNER_PROMPT" "$inbox" "$PLAN_FILE" || { info "Planner failed."; return 1; }
  init_state_from_plan
}

run_loop() {
  need_cmd jq
  need_cmd git
  [[ -f "$STATE_FILE" ]] || plan

  local iter=0
  while (( iter < MAX_ITERS )); do
    iter=$((iter+1))
    info "=== Iteration $iter/$MAX_ITERS ==="

    # Are we done?
    local remaining
    remaining="$(jq '[.tasks[] | select(.status!="done")] | length' "$STATE_FILE")"
    if [[ "$remaining" -eq 0 ]]; then
      ok "All tasks done."
      return 0
    fi

    # Ensure clean working tree before starting a new task (optional but recommended)
    if [[ -n "$(git status --porcelain=v1)" ]]; then
      info "Working tree is dirty before starting a new task."
      info "This likely means a previous run ended mid-task. You can:"
      info "  - commit/stash/reset manually, then re-run"
      info "  - or let the committer handle it if it's already approved"
    fi

    local task_id
    # Check for resumable tasks first (implemented but not done)
    task_id="$(jq -r '.tasks[] | select(.status=="implemented") | .id' "$STATE_FILE" | head -n 1)"

    if [[ -z "$task_id" ]]; then
      if ! task_id="$(choose_task)"; then
        info "No runnable tasks (deps may be blocked)."
        status
        die "Cannot continue."
      fi

      # Mark as implemented (or blocked) based on worker output
      local w_status
      w_status="$(jq -r '.status' "$OUT_DIR/worker.json")"
      local w_note
      w_note="$(jq -r '.summary // ""' "$OUT_DIR/worker.json")"

      if [[ "$w_status" == "blocked" ]]; then
        update_task_status "$task_id" "blocked" "worker: $w_note"
        info "Worker blocked on $task_id. Trying next iteration."
        continue
      fi

      if [[ "$w_status" != "implemented" ]]; then
        update_task_status "$task_id" "blocked" "worker returned unexpected status: $w_status"
        info "Worker returned unexpected status ($w_status); marking blocked."
        continue
      fi

      update_task_status "$task_id" "implemented" "worker: $w_note"
    else
      info "Resuming implemented task: $task_id"
    fi

    # Review / fix loop
    if ! review_loop "$task_id"; then
      info "Review/Fix loop failed for $task_id. Marking blocked."
      update_task_status "$task_id" "blocked" "Review/Fix loop failed."
      continue
    fi

    # Commit
    if commit_task "$task_id"; then
      update_task_status "$task_id" "done" "committer: done"
      ok "Task $task_id done."
    else
      info "Commit failed for $task_id. Task marked as blocked."
    fi
  done

  die "Reached MAX_ITERS=$MAX_ITERS without finishing."
}

init() {
  need_cmd jq
  ensure_templates
  ok "Initialized .ralph/ structure."
  info "Next:"
  info "  1) Copy your role prompts into: $PROMPTS_DIR"
  info "     - worker.md, code_reviewer.md, bug_fixer.md, committer.md, guidelines.md"
  info "  2) Fill out: $GOAL_FILE"
  info "  3) Run: ./.ralph/ralph.sh run"
}

case "${1:-}" in
  init) init ;;
  plan) plan ;;
  run) run_loop ;;
  status) status ;;
  *)
    cat <<EOF
Ralph Loop Orchestrator

Commands:
  init    Create .ralph/ structure and starter templates
  plan    Generate plan.json + state.json
  run     Run the full loop (plan -> worker -> reviewer -> fixer -> reviewer -> committer)
  status  Show current task status

EOF
    ;;
esac

