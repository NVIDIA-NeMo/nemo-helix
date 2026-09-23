#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

###############################################################################
# run_all_cli_evals.sh
#
# Runs all CLI Harbor evals (standard + easy variants) and collects results
# into a single batch directory for later comparison.
#
# Usage:
#   ./tests/agentic-use/scripts/run_all_cli_evals.sh [OPTIONS]
#
# Options:
#   -j, --jobs N        Max parallel harbor runs (default: 2)
#   -o, --output DIR    Output directory name under jobs/ (default: batch-<timestamp>)
#   -s, --skip-build    Skip the docker build step
#   -f, --filter GLOB   Only run evals matching this glob (e.g. "files-*")
#   --easy-only         Only run easy variants
#   --standard-only     Only run standard (non-easy) variants
#   --dry-run           Print what would be run without executing
#   -h, --help          Show this help
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Defaults
MAX_JOBS=2
OUTPUT_DIR=""
SKIP_BUILD=false
FILTER=""
EASY_ONLY=false
STANDARD_ONLY=false
DRY_RUN=false

# All 19 eval base names (each has a standard and -easy variant)
EVAL_BASES=(
    auditor-config-crud-cli
    auditor-default-job-cli
    auditor-target-crud-cli
    auth-authorization-cli
    data-designer-config-cli
    entities-basic-cli
    evaluator-academic-benchmark-cli
    evaluator-llm-judge-cli
    evaluator-simple-job-cli
    files-crud-cli
    files-upload-dataset-cli
    guardrails-content-safety-cli
    guardrails-custom-config-cli
    inference-chat-completions-cli
    inference-igw-provider-cli
    inference-mockllm-cli
    inference-provider-reg-cli
    secrets-crud-cli
    workspace-basic-cli
)

usage() {
    sed -n '/^# Usage:/,/^###/p' "$0" | head -n -1 | sed 's/^# //'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -j|--jobs)       MAX_JOBS="$2"; shift 2 ;;
        -o|--output)     OUTPUT_DIR="$2"; shift 2 ;;
        -s|--skip-build) SKIP_BUILD=true; shift ;;
        -f|--filter)     FILTER="$2"; shift 2 ;;
        --easy-only)     EASY_ONLY=true; shift ;;
        --standard-only) STANDARD_ONLY=true; shift ;;
        --dry-run)       DRY_RUN=true; shift ;;
        -h|--help)       usage ;;
        *)               echo "Unknown option: $1"; usage ;;
    esac
done

# Generate output directory name
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="batch-$(date +%Y-%m-%d__%H-%M-%S)"
fi
BATCH_DIR="$PROJECT_ROOT/jobs/$OUTPUT_DIR"

# Check required env vars
if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "ERROR: ANTHROPIC_API_KEY must be set"
    echo "  export ANTHROPIC_API_KEY='your-key-here'"
    exit 1
fi
if [[ -z "${ANTHROPIC_BASE_URL:-}" ]]; then
    echo "WARNING: ANTHROPIC_BASE_URL not set, defaulting to https://inference-api.nvidia.com"
    export ANTHROPIC_BASE_URL="https://inference-api.nvidia.com"
fi

# Build eval list
EVALS=()
for base in "${EVAL_BASES[@]}"; do
    # Apply filter if set
    if [[ -n "$FILTER" ]]; then
        # shellcheck disable=SC2053
        if [[ "$base" != $FILTER ]]; then
            continue
        fi
    fi

    if [[ "$EASY_ONLY" == false ]]; then
        EVALS+=("$base")
    fi
    if [[ "$STANDARD_ONLY" == false ]]; then
        EVALS+=("${base}-easy")
    fi
done

TOTAL=${#EVALS[@]}
echo "============================================"
echo "  Harbor CLI Eval Batch Run"
echo "============================================"
echo "  Evals to run:  $TOTAL"
echo "  Max parallel:  $MAX_JOBS"
echo "  Output dir:    $BATCH_DIR"
echo "  Dry run:       $DRY_RUN"
echo "============================================"
echo ""
echo "Eval list:"
for eval_name in "${EVALS[@]}"; do
    echo "  - $eval_name"
done
echo ""

if [[ "$DRY_RUN" == true ]]; then
    echo "[DRY RUN] Would run ${TOTAL} evals. Exiting."
    exit 0
fi

# Build docker image (once)
if [[ "$SKIP_BUILD" == false ]]; then
    echo "Building Docker image..."
    docker build -f "$PROJECT_ROOT/Dockerfile.agentic-base" -t nhx-agentic-base:latest "$PROJECT_ROOT"
    echo "Docker image built successfully."
    echo ""
fi

# Create batch directory and metadata
mkdir -p "$BATCH_DIR"
cat > "$BATCH_DIR/batch_config.json" <<EOF
{
    "started_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "max_jobs": $MAX_JOBS,
    "total_evals": $TOTAL,
    "evals": $(printf '%s\n' "${EVALS[@]}" | jq -R . | jq -s .),
    "model": "aws/anthropic/bedrock-claude-sonnet-4-5-v1",
    "agent": "claude-code"
}
EOF

# Track running jobs using parallel arrays (bash 3.x compatible)
PIDS=()          # background PIDs
PID_NAMES=()     # eval name for each PID
PID_STARTS=()    # start timestamp for each PID
COMPLETED=0
FAILED=0

# Get summary (reward + tokens) from a completed eval's result.json
# Output format: "PASS 1.2M tokens" or "FAIL 500k tokens" or "???"
get_eval_summary() {
    local eval_name="$1"
    local job_name="${OUTPUT_DIR}__${eval_name}"
    # Find trial result.json (contains task_name field, unlike the job-level one)
    local trial_result
    trial_result=$(find "$BATCH_DIR/$job_name" -name "result.json" -path "*/result.json" 2>/dev/null | while read -r f; do
        if python3 -c "import json,sys; d=json.load(open('$f')); sys.exit(0 if 'task_name' in d else 1)" 2>/dev/null; then
            echo "$f"
            break
        fi
    done)
    if [[ -n "$trial_result" ]]; then
        python3 -c "
import json
from pathlib import Path
d = json.load(open('$trial_result'))
r = d.get('verifier_result', {}).get('rewards', {}).get('reward')
if r is not None and r >= 1.0:
    status = 'PASS'
elif r is not None:
    status = 'FAIL'
else:
    status = 'ERROR'
ar = d.get('agent_result', {})
inp = ar.get('n_input_tokens', 0)
out = ar.get('n_output_tokens', 0)
# Count tool calls from trajectory.json or JSONL session logs
trial_dir = Path('$trial_result').parent
traj = trial_dir / 'agent' / 'trajectory.json'
tc = 0
if traj.exists():
    try:
        tj = json.loads(traj.read_text())
        tc = sum(len(s.get('tool_calls', [])) for s in tj.get('steps', []))
    except Exception:
        pass
else:
    sessions = trial_dir / 'agent' / 'sessions' / 'projects'
    if sessions.exists():
        for jf in sessions.rglob('*.jsonl'):
            try:
                for line in jf.read_text().splitlines():
                    m = json.loads(line)
                    if m.get('type') == 'assistant':
                        for b in m.get('message', {}).get('content', []):
                            if b.get('type') == 'tool_use':
                                tc += 1
            except Exception:
                pass
def fmt(n):
    if n >= 1_000_000: return f'{n/1_000_000:.1f}M'
    if n >= 1_000: return f'{n/1_000:.0f}k'
    return str(n)
print(f'{status} ({fmt(inp)} in, {fmt(out)} out, {tc} tools)')
" 2>/dev/null || echo "???"
    else
        echo "???"
    fi
}

# Run a single eval
run_eval() {
    local eval_name="$1"
    local job_name="${OUTPUT_DIR}__${eval_name}"
    local log_file="$BATCH_DIR/${eval_name}.log"

    echo "[$(date +%H:%M:%S)] START: $eval_name"

    harbor run \
        -p "tests/agentic-use/${eval_name}" \
        --agent claude-code \
        --model aws/anthropic/bedrock-claude-sonnet-4-5-v1 \
        --n-tasks 1 \
        --job-name "$job_name" \
        --jobs-dir "$BATCH_DIR" \
        > "$log_file" 2>&1

    local exit_code=$?
    local summary
    summary=$(get_eval_summary "$eval_name")
    echo "[$(date +%H:%M:%S)] DONE:  $eval_name (exit=$exit_code) => $summary"
    return $exit_code
}

# Reap any finished jobs from the PIDS array, returns once at least one slot is free
reap_finished() {
    while [[ ${#PIDS[@]} -ge $MAX_JOBS ]]; do
        local new_pids=()
        local new_names=()
        local new_starts=()
        local reaped=false

        for i in "${!PIDS[@]}"; do
            local pid="${PIDS[$i]}"
            local name="${PID_NAMES[$i]}"
            local start="${PID_STARTS[$i]}"

            if ! kill -0 "$pid" 2>/dev/null; then
                wait "$pid" 2>/dev/null
                local exit_code=$?
                local end_time
                end_time=$(date +%s)
                local duration=$(( end_time - start ))
                local summary
                summary=$(get_eval_summary "$name")

                if [[ $exit_code -eq 0 ]]; then
                    COMPLETED=$((COMPLETED + 1))
                    echo "  [${COMPLETED}+${FAILED}/${TOTAL}] $name completed (${duration}s) => $summary"
                else
                    FAILED=$((FAILED + 1))
                    echo "  [${COMPLETED}+${FAILED}/${TOTAL}] $name FAILED (${duration}s, exit=$exit_code) => $summary"
                fi
                reaped=true
            else
                new_pids+=("$pid")
                new_names+=("$name")
                new_starts+=("$start")
            fi
        done

        PIDS=("${new_pids[@]+"${new_pids[@]}"}")
        PID_NAMES=("${new_names[@]+"${new_names[@]}"}")
        PID_STARTS=("${new_starts[@]+"${new_starts[@]}"}")

        if [[ "$reaped" == true ]]; then
            return
        fi
        sleep 2
    done
}

# Main execution loop
echo ""
echo "Starting eval runs..."
echo "----------------------------------------------"

cd "$PROJECT_ROOT"

for eval_name in "${EVALS[@]}"; do
    reap_finished

    run_eval "$eval_name" &
    PIDS+=($!)
    PID_NAMES+=("$eval_name")
    PID_STARTS+=($(date +%s))
done

# Wait for remaining jobs
echo ""
echo "Waiting for remaining jobs to finish..."
for i in "${!PIDS[@]}"; do
    local_pid="${PIDS[$i]}"
    eval_name="${PID_NAMES[$i]}"
    start_time="${PID_STARTS[$i]}"

    wait "$local_pid" 2>/dev/null
    exit_code=$?
    end_time=$(date +%s)
    duration=$(( end_time - start_time ))
    summary=$(get_eval_summary "$eval_name")

    if [[ $exit_code -eq 0 ]]; then
        COMPLETED=$((COMPLETED + 1))
        echo "  [${COMPLETED}+${FAILED}/${TOTAL}] $eval_name completed (${duration}s) => $summary"
    else
        FAILED=$((FAILED + 1))
        echo "  [${COMPLETED}+${FAILED}/${TOTAL}] $eval_name FAILED (${duration}s, exit=$exit_code) => $summary"
    fi
done

# Write summary metadata
cat > "$BATCH_DIR/batch_summary.json" <<EOF
{
    "finished_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "total": $TOTAL,
    "completed": $COMPLETED,
    "failed": $FAILED
}
EOF

echo ""
echo "============================================"
echo "  Batch Run Complete"
echo "============================================"
echo "  Total:     $TOTAL"
echo "  Completed: $COMPLETED"
echo "  Failed:    $FAILED"
echo "  Results:   $BATCH_DIR"
echo ""
echo "Generate report with:"
echo "  python tests/agentic-use/scripts/generate_eval_report.py $BATCH_DIR"
echo "============================================"
