#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# End-to-end evaluation smoke: build a task through the control plane, then
# actually run it on the sandbox_k8s runtime and assert what came back.
#
# smoke.sh stops at "the image reached GAR". This one goes further and is the
# only check that exercises Harbor, the agent-sandbox CRDs, and sandbox RBAC.
#
# The task runs the `oracle` agent, which applies the task's own reference
# solution, so a healthy run scores exactly 1.0 and needs no model credentials.
#
#   ./apply.sh && ./eval-smoke.sh            # happy path, the default
#   ./eval-smoke.sh artifacts                # downloads after a healthy run
#   ./eval-smoke.sh cancel                   # cancel queued, then cancel running
#   ./eval-smoke.sh retry                    # fail an execution, then retry it
#   ./eval-smoke.sh restart                  # restart the controller mid-run
#   ./eval-smoke.sh fanout                   # benchmark fan-out: ramp and caps
#   ./eval-smoke.sh all                      # every scenario, one task build
#
# The fault-injection scenarios need a run long enough to interrupt, so they
# build a slow variant of hello-world: the same task with a sleep in front of
# the reference solution. It is synthesised here rather than committed as a
# second fixture so it cannot drift from hello-world.
#
# Fan-out needs many *tasks*, not many runs of one task, and building an image
# per member would measure Cloud Build instead of the controller. So it builds
# once and registers the remaining members against that same signed image.
set -euo pipefail

NS=nemo-helix-scaled-evals
PORT="${PORT:-18081}"
BASE="http://127.0.0.1:$PORT/apis/scaled-evals"
TASK_SRC="$(cd "$(dirname "$0")/../../examples/tasks/hello-world" && pwd)"
WORK="$(mktemp -d)"
# Long enough to observe a run and intervene, well inside the task's own 900s
# agent timeout.
SLOW_SECONDS="${SLOW_SECONDS:-240}"
# Enough members to see a ramp and to exceed the member cap below, small enough
# that one run fits in a coffee break.
MEMBERS="${MEMBERS:-10}"
# The per-run cap sent with the request. Deliberately below MEMBERS so it binds.
MEMBER_CAP="${MEMBER_CAP:-3}"
# What in-flight count the run is asserted against. Separate from MEMBER_CAP so
# the settings-level caps can be isolated: raise MEMBER_CAP out of the way and
# set this to the limit under test, or the run cannot say which cap bound.
ASSERT_CAP="${ASSERT_CAP:-$MEMBER_CAP}"
SCENARIO="${1:-happy}"

step() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
note() { printf '    %s\n' "$1"; }
fail() { printf '\033[31mFAIL: %s\033[0m\n' "$1" >&2; exit 1; }
pass() { printf '\033[32m  ok\033[0m — %s\n' "$1"; }
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

# ---------------------------------------------------------------- api helpers

port_forward() {
  step "port-forwarding to the API"
  kubectl port-forward -n "$NS" deploy/scaled-evals-api "$PORT:8080" >"$WORK/pf.log" 2>&1 &
  PF_PID=$!
  trap 'kill $PF_PID 2>/dev/null || true; rm -rf "$WORK"' EXIT
  for i in $(seq 1 30); do
    curl -sf "$BASE/healthz" -o /dev/null 2>/dev/null && return 0
    [ "$i" = 30 ] && { cat "$WORK/pf.log"; fail "port-forward never came up"; }
    sleep 2
  done
}

# build_pack <sleep_seconds>
#
# The pack serves two masters: the Cloud Build context (root Dockerfile) and the
# per-eval task tree dispatch stages (the dir holding task.toml). Ship both --
# the task's own environment/Dockerfile is copied to the root for the build.
build_pack() {
  local sleep_seconds="$1"
  rm -rf "$WORK/pack"
  mkdir -p "$WORK/pack"
  cp -R "$TASK_SRC/task" "$WORK/pack/task"
  cp "$TASK_SRC/task/environment/Dockerfile" "$WORK/pack/Dockerfile"
  if [ "$sleep_seconds" -gt 0 ]; then
    # The oracle agent runs solve.sh, so sleeping here makes the *agent* phase
    # long. Sleeping in the verifier instead would leave nothing to cancel.
    printf '\nsleep %s\n' "$sleep_seconds" >> "$WORK/pack/task/solution/solve.sh"
  fi
  tar -czf "$WORK/pack.tar.gz" -C "$WORK/pack" .
}

# build_task <label> <sleep_seconds>  ->  sets TASK_ID and REVISION
build_task() {
  local label="$1" sleep_seconds="$2"
  step "building the $label pack"
  build_pack "$sleep_seconds"

  NAME="$label-$(date +%Y%m%d%H%M%S)-$$"
  curl -sf -X POST "$BASE/v1/tasks" -H 'content-type: application/json' \
    -d "{\"name\":\"$NAME\",\"description\":\"$label oracle eval smoke\"}" \
    -o "$WORK/task.json" || fail "task create"
  TASK_ID="$(json "$WORK/task.json" "id")"
  note "task_id: $TASK_ID"

  curl -sf -X POST "$BASE/v1/tasks/$TASK_ID/revisions" -o "$WORK/rev.json" || fail "revision create"
  REVISION="$(json "$WORK/rev.json" "revision")"
  curl -sf -X PUT --upload-file "$WORK/pack.tar.gz" \
    -H 'Content-Type: application/gzip' "$(json "$WORK/rev.json" "upload.url")" \
    -o /dev/null || fail "pack upload"
  curl -sf -X POST "$BASE/v1/tasks/$TASK_ID/finalize" -o /dev/null || fail "finalize"

  step "waiting for Cloud Build"
  local status
  for i in $(seq 1 120); do
    curl -sf "$BASE/v1/tasks/$TASK_ID" -o "$WORK/task_now.json" || fail "task get"
    status="$(json "$WORK/task_now.json" "status")"
    printf '  [%03d] %s\n' "$i" "$status"
    case "$status" in
      ready) break ;;
      failed) fail "build failed: $(json "$WORK/task_now.json" "build_error")" ;;
    esac
    sleep 10
  done
  [ "$status" = ready ] || fail "revision never reached ready (last: $status)"
  note "image_ref: $(json "$WORK/task_now.json" "image_ref")"
}

# clone_task <label>  ->  echoes a task id finalized against $IMAGE_REF
#
# Finalize accepts an already-signed image instead of building the uploaded
# pack. The pack still has to be uploaded -- finalize rejects a revision with
# no object behind it -- but nothing rebuilds it.
#
# Reuse is still asynchronous: finalize returns `building` and the revision
# reaches `ready` once the control plane has resolved the image's digest. So
# this only submits; wait_tasks_ready waits for the whole set at once rather
# than paying that latency ten times over.
clone_task() {
  local label="$1" tid out="$WORK/clone-$1.json"
  curl -sf -X POST "$BASE/v1/tasks" -H 'content-type: application/json' \
    -d "{\"name\":\"$NAME-$label\",\"description\":\"fan-out member $label\"}" \
    -o "$out" || fail "clone $label: task create"
  tid="$(json "$out" id)"
  curl -sf -X PUT --upload-file "$WORK/pack.tar.gz" \
    -H 'Content-Type: application/gzip' "$(json "$out" upload.url)" \
    -o /dev/null || fail "clone $label: pack upload"
  curl -sf -X POST "$BASE/v1/tasks/$tid/finalize" -H 'content-type: application/json' \
    -d "{\"image_ref\":\"$IMAGE_REF\",\"image_digest\":\"$IMAGE_DIGEST\"}" \
    -o /dev/null \
    || fail "clone $label: finalize did not accept the reused image"
  printf '%s' "$tid"
}

# wait_tasks_ready <task_id>...
wait_tasks_ready() {
  local ids=("$@") pending status tid i
  for i in $(seq 1 120); do
    pending=()
    for tid in "${ids[@]}"; do
      status="$(curl -sf "$BASE/v1/tasks/$tid" -o "$WORK/t_now.json" && json "$WORK/t_now.json" status)"
      case "$status" in
        ready) ;;
        failed) fail "task $tid failed: $(json "$WORK/t_now.json" build_error)" ;;
        *) pending+=("$tid") ;;
      esac
    done
    printf '  [%03d] %d/%d ready\n' "$i" "$((${#ids[@]} - ${#pending[@]}))" "${#ids[@]}"
    [ "${#pending[@]}" -eq 0 ] && return 0
    ids=("${pending[@]}")
    sleep 5
  done
  fail "${#ids[@]} member tasks never reached ready"
}

# create_eval <suffix>  ->  sets EV_ID
create_eval() {
  curl -sf -X POST "$BASE/v1/evaluations" -H 'content-type: application/json' \
    -d "{\"name\":\"$NAME-$1\",\"task_id\":\"$TASK_ID\",\"task_revision\":$REVISION,\"runtime\":\"sandbox_k8s\"}" \
    -o "$WORK/ev.json" || fail "evaluation create"
  EV_ID="$(json "$WORK/ev.json" "id")"
  note "evaluation_id: $EV_ID"
}

# field <evaluation_id> <dotted.key>
field() {
  curl -sf "$BASE/v1/evaluations/$1" -o "$WORK/ev_now.json" || fail "evaluation get"
  json "$WORK/ev_now.json" "$2"
}

# wait_until <evaluation_id> <extended-regex of statuses> <max polls> [sleep]
#
# Sets STATUS. Returns non-zero if the pattern was never reached, so callers
# decide whether that is a failure.
wait_until() {
  local id="$1" pattern="$2" polls="$3" nap="${4:-5}"
  for i in $(seq 1 "$polls"); do
    curl -sf "$BASE/v1/evaluations/$id" -o "$WORK/ev_now.json" || fail "evaluation get"
    STATUS="$(json "$WORK/ev_now.json" "status")"
    printf '  [%03d] %-12s %s\n' "$i" "$STATUS" "$(json "$WORK/ev_now.json" "status_detail" | cut -c1-80)"
    if printf '%s' "$STATUS" | grep -Eq "^($pattern)$"; then return 0; fi
    sleep "$nap"
  done
  return 1
}

# ------------------------------------------------------------ cluster helpers

# The Kubernetes Job is named after the platform step, not the evaluation, so
# the only link back is the job_id label. That label holds the Platform Job
# name, which embeds a truncated evaluation id. Matching on the label rather
# than the object name is what keeps these assertions from being vacuous.
JOB_ID_LABEL='nhx\.nvidia\.com/job_id'

job_ids_for() {
  kubectl get jobs -n "$NS" -l nhx.nvidia.com/managed_by=jobs-controller \
    -o jsonpath="{range .items[*]}{.metadata.labels.$JOB_ID_LABEL}{\"\n\"}{end}" 2>/dev/null |
    grep "evaluation-${1:0:14}" || true
}

job_count_for() { job_ids_for "$1" | grep -c . || true; }

k8s_job_names_for() {
  kubectl get jobs -n "$NS" -l nhx.nvidia.com/managed_by=jobs-controller \
    -o jsonpath="{range .items[*]}{.metadata.labels.$JOB_ID_LABEL}{\" \"}{.metadata.name}{\"\n\"}{end}" 2>/dev/null |
    awk -v want="evaluation-${1:0:14}" 'index($1, want) { print $2 }'
}

sandbox_count() { kubectl get sandboxes.agents.x-k8s.io -n "$NS" -o name 2>/dev/null | wc -l | tr -d ' '; }

# Wait for a cluster-side count to reach a value, since teardown is not
# instantaneous once the API reports a terminal status.
wait_for_count() {
  local what="$1" want="$2" polls="$3" got
  for _ in $(seq 1 "$polls"); do
    got="$(eval "$what")"
    [ "$got" = "$want" ] && return 0
    sleep 5
  done
  note "last observed: $got, wanted: $want"
  return 1
}

# ------------------------------------------------------------------ scenarios

scenario_happy() {
  build_task hello-world 0
  step "creating the evaluation"
  create_eval oracle

  step "waiting for the sandbox to run"
  # Pod scheduling plus an image pull plus the verifier; the task's own timeouts
  # are 900s each, so give the whole thing room.
  wait_until "$EV_ID" 'succeeded|failed|cancelled' 150 10 || true

  REWARD="$(json "$WORK/ev_now.json" "reward")"
  printf '\nstatus: %s   reward: %s\n' "$STATUS" "$REWARD"
  [ "$STATUS" = succeeded ] || fail "evaluation did not succeed: $(json "$WORK/ev_now.json" "status_detail")"
  # The oracle applies the reference solution, so anything below 1.0 means the
  # harness ran but the task did not actually pass.
  python3 -c "import sys; sys.exit(0 if float('${REWARD:-0}') == 1.0 else 1)" \
    || fail "oracle reward was $REWARD, expected 1.0"
  pass "$EV_ID ran on sandbox_k8s and scored $REWARD"
  HAPPY_EV_ID="$EV_ID"
}

# Downloads are the half of the contract the happy path never touches: a run can
# succeed while its artifacts are unreachable.
scenario_artifacts() {
  local id="${HAPPY_EV_ID:-}"
  [ -n "$id" ] || { scenario_happy; id="$HAPPY_EV_ID"; }
  step "downloading artifacts, provenance, SBOM, and the archive"
  local base="$BASE/v1/evaluations/$id"

  local count
  count="$(curl -sf "$base/artifacts" |
    python3 -c 'import json,sys; print(len(json.load(sys.stdin).get("data",[])))')"
  [ "${count:-0}" -gt 0 ] || fail "no artifacts listed"
  pass "$count artifacts listed"

  for name in scaled-evals-provenance.json scaled-evals-sbom.cdx.json; do
    curl -sfL "$base/artifacts/$name" -o "$WORK/$name" || fail "$name did not download"
    python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$WORK/$name" \
      || fail "$name is not valid JSON"
    pass "$name downloaded and parses ($(wc -c <"$WORK/$name" | tr -d ' ') bytes)"
  done

  # The archive is built asynchronously, so it may not be ready the instant the
  # evaluation is.
  local state
  for _ in $(seq 1 24); do
    state="$(curl -sf "$base/archive" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("status",""))')"
    [ "$state" = ready ] && break
    sleep 5
  done
  [ "$state" = ready ] || fail "archive never became ready (last: $state)"
  curl -sfL "$base/archive/download" -o "$WORK/archive.tgz" || fail "archive download"
  tar -tzf "$WORK/archive.tgz" >/dev/null 2>&1 || fail "archive is not a readable tarball"
  pass "archive downloaded and unpacks ($(wc -c <"$WORK/archive.tgz" | tr -d ' ') bytes)"
}

# Two branches meet here: cancelling before a sandbox exists cancels the
# Platform Job outright, while cancelling a live run has to tear the sandbox
# down first and then settle. Only the second has ever run outside unit tests.
scenario_cancel() {
  step "cancel while the evaluation is still queued"
  create_eval cancel-queued
  curl -sf -X POST "$BASE/v1/evaluations/$EV_ID/cancel" -o /dev/null || fail "cancel (queued)"
  wait_until "$EV_ID" 'cancelled' 36 5 || fail "queued evaluation never reached cancelled (last: $STATUS)"
  wait_for_count "job_count_for $EV_ID" 0 12 || fail "Platform Job survived a queued cancel"
  pass "queued cancel settled and left no Platform Job"

  step "cancel while the sandbox is live"
  create_eval cancel-running
  wait_until "$EV_ID" 'running' 60 5 || fail "evaluation never started running (last: $STATUS)"
  # `running` is set when dispatch launches; the sandbox has to actually exist
  # before cancelling, or this silently retests the queued branch instead of
  # the teardown branch it is here for.
  wait_for_count "sandbox_count" 1 24 || fail "no sandbox appeared, so the teardown branch was not covered"
  curl -sf -X POST "$BASE/v1/evaluations/$EV_ID/cancel" -o /dev/null || fail "cancel (running)"
  wait_until "$EV_ID" 'cancelled' 60 5 || fail "running evaluation never reached cancelled (last: $STATUS)"
  wait_for_count "sandbox_count" 0 24 || fail "sandbox survived the cancel"
  wait_for_count "job_count_for $EV_ID" 0 12 || fail "Platform Job survived the cancel"
  pass "live cancel tore down the sandbox and the Platform Job"
}

# A failing verifier is not enough: that yields `succeeded` with reward 0. A
# retryable failure needs an infrastructure fault, so delete the Platform Job
# out from under a live run and let the controller notice.
scenario_retry() {
  step "failing an execution by deleting its Platform Job"
  create_eval retry
  wait_until "$EV_ID" 'running' 60 5 || fail "evaluation never started running (last: $STATUS)"
  local before
  before="$(field "$EV_ID" current_execution)"
  # Scoped to this evaluation's own job; the namespace holds other work.
  local victims
  victims="$(k8s_job_names_for "$EV_ID")"
  [ -n "$victims" ] || fail "no Platform Job found to delete for $EV_ID"
  printf '%s\n' "$victims" |
    xargs -r kubectl delete job -n "$NS" --wait=false >/dev/null ||
    fail "could not delete the Platform Job"
  note "execution before the fault: $before (deleted $(printf '%s\n' "$victims" | grep -c .) job)"

  wait_until "$EV_ID" 'failed|succeeded|cancelled' 60 5 || fail "evaluation never settled after the fault"
  [ "$STATUS" = failed ] || fail "expected the deleted job to fail the evaluation, got $STATUS"
  pass "the controller failed the evaluation after its job vanished"

  step "retrying it"
  curl -sf -X POST "$BASE/v1/evaluations/$EV_ID/retry" -o /dev/null || fail "retry"
  wait_until "$EV_ID" 'queued|provisioning|running' 24 5 || fail "retry did not restart the evaluation"
  local after
  after="$(field "$EV_ID" current_execution)"
  [ "$after" -gt "$before" ] || fail "retry did not advance current_execution ($before -> $after)"
  # Exactly one: a revived old job or a duplicate submission would both show up
  # as a second job carrying this evaluation's id.
  wait_for_count "job_count_for $EV_ID" 1 24 || fail "retry did not leave exactly one Platform Job"
  pass "retry advanced execution $before -> $after with exactly one Platform Job"
}

# Deterministic job naming should make resubmission idempotent, but the window
# between creating a job and recording its name has only been closed in tests.
scenario_restart() {
  step "restarting the controller mid-run"
  create_eval restart
  wait_until "$EV_ID" 'running' 60 5 || fail "evaluation never started running (last: $STATUS)"
  # The controller runs inside the API process, so this restarts both.
  kubectl rollout restart -n "$NS" deploy/scaled-evals-api >/dev/null || fail "rollout restart"
  kubectl rollout status -n "$NS" deploy/scaled-evals-api --timeout=300s >/dev/null || fail "rollout never completed"
  # The port-forward died with the old pod.
  kill "$PF_PID" 2>/dev/null || true
  port_forward

  wait_for_count "job_count_for $EV_ID" 1 12 || fail "restart left a duplicate or no Platform Job"
  pass "exactly one Platform Job survived the restart"

  wait_until "$EV_ID" 'succeeded|failed|cancelled' 150 10 || fail "evaluation never settled after the restart"
  [ "$STATUS" = succeeded ] || fail "evaluation did not survive the restart: $(json "$WORK/ev_now.json" status_detail)"
  pass "$EV_ID completed across a controller restart"
}

# Every scenario above drives one evaluation, so none of them can show what
# happens when a benchmark fans out: how fast the controller submits, whether
# the concurrency cap binds, and whether a status ever moves backward.
scenario_fanout() {
  build_task hello-world 0
  IMAGE_REF="$(json "$WORK/task_now.json" image_ref)"
  IMAGE_DIGEST="$(json "$WORK/task_now.json" image_digest)"
  [ -n "$IMAGE_REF" ] || fail "the built task exposed no image_ref to reuse"
  [ -n "$IMAGE_DIGEST" ] || fail "the built task exposed no image_digest to reuse"

  step "registering $MEMBERS member tasks against one built image"
  local members="$TASK_ID" i
  for i in $(seq 2 "$MEMBERS"); do
    members="$members $(clone_task "m$i")"
    printf '  [%03d/%03d] submitted\n' "$i" "$MEMBERS"
  done
  # shellcheck disable=SC2086 -- $members is a deliberate word-split list.
  wait_tasks_ready $members
  pass "$MEMBERS member tasks ready without $((MEMBERS - 1)) extra builds"

  step "creating the benchmark and running it"
  # shellcheck disable=SC2086 -- $members is a deliberate word-split list.
  python3 -c '
import json, sys
print(json.dumps({
    "name": sys.argv[1],
    "description": "fan-out smoke",
    "tasks": [{"task_id": t} for t in sys.argv[2:]],
}))' "$NAME-bm" $members > "$WORK/bm_req.json"
  curl -sf -X POST "$BASE/v1/benchmarks" -H 'content-type: application/json' \
    --data-binary "@$WORK/bm_req.json" -o "$WORK/bm.json" || fail "benchmark create"
  local bm_id
  bm_id="$(json "$WORK/bm.json" id)"
  note "benchmark_id: $bm_id"

  curl -sf -X POST "$BASE/v1/benchmark-runs" -H 'content-type: application/json' \
    -d "{\"name\":\"$NAME-run\",\"benchmark_id\":\"$bm_id\",\"runtime\":\"sandbox_k8s\",\"max_concurrent_members\":$MEMBER_CAP}" \
    -o "$WORK/run.json" || fail "benchmark run create"
  local run_id
  run_id="$(json "$WORK/run.json" id)"
  note "benchmark_run_id: $run_id   run cap: $MEMBER_CAP   asserting: $ASSERT_CAP"

  step "watching the fan-out"
  python3 "$(dirname "$0")/fanout-probe.py" "$BASE" "$run_id" \
    --expect-members "$MEMBERS" --member-cap "$ASSERT_CAP" \
    | tee "$WORK/fanout.json" || fail "fan-out probe reported a violation"

  local per_min settled
  per_min="$(json "$WORK/fanout.json" submissions_per_minute)"
  settled="$(python3 -c '
import json, sys
counts = json.load(open(sys.argv[1]))["final_status_counts"]
print(counts.get("succeeded", 0))' "$WORK/fanout.json")"
  [ "$settled" = "$MEMBERS" ] || fail "only $settled/$MEMBERS members succeeded"
  pass "all $MEMBERS members succeeded, cap held at $ASSERT_CAP, no status regressed"

  # The caps gate the *claim*, and submission follows the claim, so a binding
  # cap paces submissions by completions. Measuring the controller's drain rate
  # under one measures the cap instead: run `MEMBER_CAP=$MEMBERS` for that.
  if [ "$ASSERT_CAP" -lt "$MEMBERS" ]; then
    note "submit rate not asserted: the cap of $ASSERT_CAP paced submissions (${per_min}/min)"
    return 0
  fi
  # One submission per reconcile pass at the 10s default is ~6/min. Anything at
  # or below that means the drain loop is not draining.
  python3 -c "import sys; sys.exit(0 if float('${per_min:-0}') > 6.0 else 1)" \
    || fail "submitted ${per_min}/min, at or below the one-per-pass ceiling of 6/min"
  pass "controller submitted ${per_min} evaluations/minute"
}

# ----------------------------------------------------------------------- main

port_forward

case "$SCENARIO" in
  happy)     scenario_happy ;;
  artifacts) scenario_artifacts ;;
  cancel)    build_task slow-world "$SLOW_SECONDS"; scenario_cancel ;;
  retry)     build_task slow-world "$SLOW_SECONDS"; scenario_retry ;;
  restart)   build_task slow-world "$SLOW_SECONDS"; scenario_restart ;;
  fanout)    scenario_fanout ;;
  all)
    scenario_happy
    scenario_artifacts
    build_task slow-world "$SLOW_SECONDS"
    scenario_cancel
    scenario_retry
    scenario_restart
    scenario_fanout
    ;;
  *) fail "unknown scenario '$SCENARIO' (happy|artifacts|cancel|retry|restart|fanout|all)" ;;
esac

printf '\n\033[32mPASS\033[0m — scenario %s.\n' "$SCENARIO"
