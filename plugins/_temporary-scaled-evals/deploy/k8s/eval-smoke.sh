#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# End-to-end evaluation smoke: build the hello-world task through the control
# plane, then actually run it on the sandbox_k8s runtime and assert the reward.
#
# smoke.sh stops at "the image reached GAR". This one goes further and is the
# only check that exercises Harbor, the agent-sandbox CRDs, and sandbox RBAC.
#
# The task runs the `oracle` agent, which applies the task's own reference
# solution, so a healthy run scores exactly 1.0 and needs no model credentials.
#
#   ./apply.sh && ./eval-smoke.sh
set -euo pipefail

NS=nemo-helix-scaled-evals
PORT="${PORT:-18081}"
BASE="http://127.0.0.1:$PORT/apis/scaled-evals"
TASK_SRC="$(cd "$(dirname "$0")/../../examples/tasks/hello-world" && pwd)"
WORK="$(mktemp -d)"

step() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
fail() { printf '\033[31mFAIL: %s\033[0m\n' "$1" >&2; exit 1; }
json() {
  python3 -c '
import json, sys
value = json.load(open(sys.argv[1]))
for key in sys.argv[2].split("."):
    if not isinstance(value, dict):
        value = None
        break
    value = value.get(key)
print("" if value is None else value)
' "$1" "$2"
}

step "port-forwarding to the API"
kubectl port-forward -n "$NS" deploy/scaled-evals-api "$PORT:8080" >"$WORK/pf.log" 2>&1 &
PF_PID=$!
trap 'kill $PF_PID 2>/dev/null || true; rm -rf "$WORK"' EXIT
for i in $(seq 1 30); do
  curl -sf "$BASE/healthz" -o /dev/null 2>/dev/null && break
  [ "$i" = 30 ] && { cat "$WORK/pf.log"; fail "port-forward never came up"; }
  sleep 2
done

step "building the hello-world pack"
# The pack serves two masters: the Cloud Build context (root Dockerfile) and the
# per-eval task tree dispatch stages (the dir holding task.toml). Ship both --
# the task's own environment/Dockerfile is copied to the root for the build.
mkdir -p "$WORK/pack"
cp -R "$TASK_SRC/task" "$WORK/pack/task"
cp "$TASK_SRC/task/environment/Dockerfile" "$WORK/pack/Dockerfile"
tar -czf "$WORK/pack.tar.gz" -C "$WORK/pack" .

step "creating the task"
NAME="hello-world-$(date +%Y%m%d%H%M%S)-$$"
curl -sf -X POST "$BASE/v1/tasks" -H 'content-type: application/json' \
  -d "{\"name\":\"$NAME\",\"description\":\"hello-world oracle eval smoke\"}" \
  -o "$WORK/task.json" || fail "task create"
TASK_ID="$(json "$WORK/task.json" "id")"
echo "task_id: $TASK_ID"

curl -sf -X POST "$BASE/v1/tasks/$TASK_ID/revisions" -o "$WORK/rev.json" || fail "revision create"
REVISION="$(json "$WORK/rev.json" "revision")"
curl -sf -X PUT --upload-file "$WORK/pack.tar.gz" \
  -H 'Content-Type: application/gzip' "$(json "$WORK/rev.json" "upload.url")" \
  -o /dev/null || fail "pack upload"
curl -sf -X POST "$BASE/v1/tasks/$TASK_ID/finalize" -o /dev/null || fail "finalize"

step "waiting for Cloud Build"
for i in $(seq 1 120); do
  curl -sf "$BASE/v1/tasks/$TASK_ID" -o "$WORK/task_now.json" || fail "task get"
  STATUS="$(json "$WORK/task_now.json" "status")"
  printf '  [%03d] %s\n' "$i" "$STATUS"
  case "$STATUS" in
    ready) break ;;
    failed) fail "build failed: $(json "$WORK/task_now.json" "build_error")" ;;
  esac
  sleep 10
done
[ "$STATUS" = ready ] || fail "revision never reached ready (last: $STATUS)"
echo "image_ref: $(json "$WORK/task_now.json" "image_ref")"

step "creating the evaluation"
curl -sf -X POST "$BASE/v1/evaluations" -H 'content-type: application/json' \
  -d "{\"name\":\"$NAME-oracle\",\"task_id\":\"$TASK_ID\",\"task_revision\":$REVISION,\"runtime\":\"sandbox_k8s\"}" \
  -o "$WORK/ev.json" || fail "evaluation create"
EV_ID="$(json "$WORK/ev.json" "id")"
echo "evaluation_id: $EV_ID"

step "waiting for the sandbox to run"
# Pod scheduling plus an image pull plus the verifier; the task's own timeouts
# are 900s each, so give the whole thing room.
for i in $(seq 1 150); do
  curl -sf "$BASE/v1/evaluations/$EV_ID" -o "$WORK/ev_now.json" || fail "evaluation get"
  STATUS="$(json "$WORK/ev_now.json" "status")"
  printf '  [%03d] %-10s %s\n' "$i" "$STATUS" "$(json "$WORK/ev_now.json" "status_detail" | cut -c1-90)"
  case "$STATUS" in
    succeeded|failed|cancelled) break ;;
  esac
  sleep 10
done

REWARD="$(json "$WORK/ev_now.json" "reward")"
echo
echo "status: $STATUS   reward: $REWARD"
[ "$STATUS" = succeeded ] || fail "evaluation did not succeed: $(json "$WORK/ev_now.json" "status_detail")"
# The oracle applies the reference solution, so anything below 1.0 means the
# harness ran but the task did not actually pass.
python3 -c "import sys; sys.exit(0 if float('${REWARD:-0}') == 1.0 else 1)" \
  || fail "oracle reward was $REWARD, expected 1.0"

printf '\n\033[32mPASS\033[0m — %s ran on sandbox_k8s and scored %s.\n' "$EV_ID" "$REWARD"
