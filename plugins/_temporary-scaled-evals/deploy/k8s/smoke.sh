#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Same end-to-end check as the compose stack's smoke, run against the cluster
# through a port-forward: create a task, upload a pack, finalize, and confirm
# the image reached GAR. The build itself goes through Cloud Build here, not
# BuildKit, so this exercises a genuinely different finalize path.
#
#   ./apply.sh && ./smoke.sh
set -euo pipefail

NS=nemo-platform-scaled-evals
PORT="${PORT:-18080}"
BASE="http://127.0.0.1:$PORT/apis/scaled-evals"
WORK="$(mktemp -d)"

step() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
fail() { printf '\033[31mFAIL: %s\033[0m\n' "$1" >&2; exit 1; }
# json <file> <dotted.path> -> value, or empty when absent
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
# To the deployment, not the Service: readyz gates endpoints, and this has to
# work even while readiness is still settling.
kubectl port-forward -n "$NS" deploy/scaled-evals-api "$PORT:8080" >"$WORK/pf.log" 2>&1 &
PF_PID=$!
trap 'kill $PF_PID 2>/dev/null || true; rm -rf "$WORK"' EXIT
for i in $(seq 1 30); do
  curl -sf "$BASE/healthz" -o /dev/null 2>/dev/null && break
  [ "$i" = 30 ] && { cat "$WORK/pf.log"; fail "port-forward never came up"; }
  sleep 2
done

step "waiting for /v1/readyz"
for i in $(seq 1 60); do
  curl -sf "$BASE/v1/readyz" -o "$WORK/readyz.json" 2>/dev/null && break
  [ "$i" = 60 ] && fail "readyz never returned 200; try: kubectl logs -n $NS deploy/scaled-evals-api"
  sleep 5
done
python3 -m json.tool "$WORK/readyz.json"
for check in postgres schema object_store platform_jobs_controller; do
  [ "$(json "$WORK/readyz.json" "checks.$check")" = ok ] || fail "readyz check '$check' is not ok"
done

step "building a task pack"
cat > "$WORK/Dockerfile" <<'EOF'
FROM python:3.12-slim-bookworm
WORKDIR /app
RUN echo "scaled-evals gke smoke" > /app/marker.txt
EOF
tar -czf "$WORK/pack.tar.gz" -C "$WORK" Dockerfile

step "POST /v1/tasks"
NAME="gke-smoke-$(date +%Y%m%d%H%M%S)-$$"
curl -sf -X POST "$BASE/v1/tasks" -H 'content-type: application/json' \
  -d "{\"name\":\"$NAME\",\"description\":\"scaled-evals gke smoke\"}" \
  -o "$WORK/task.json" || fail "task create"
TASK_ID="$(json "$WORK/task.json" "id")"
echo "task_id: $TASK_ID"

step "POST /v1/tasks/$TASK_ID/revisions"
curl -sf -X POST "$BASE/v1/tasks/$TASK_ID/revisions" -o "$WORK/rev.json" || fail "revision create"
UPLOAD_URL="$(json "$WORK/rev.json" "upload.url")"
REVISION="$(json "$WORK/rev.json" "revision")"
echo "revision: $REVISION"

step "PUT the pack to GCS"
# A V4 signed URL straight to storage.googleapis.com — no cluster ingress, no
# platform auth. This is the path that needs Workload Identity signBlob.
curl -sf -X PUT --upload-file "$WORK/pack.tar.gz" \
  -H 'Content-Type: application/gzip' "$UPLOAD_URL" -o /dev/null || fail "pack upload"

step "POST /v1/tasks/$TASK_ID/finalize"
curl -sf -X POST "$BASE/v1/tasks/$TASK_ID/finalize" -o "$WORK/finalize.json" || fail "finalize"
python3 -m json.tool "$WORK/finalize.json"

step "waiting for Cloud Build"
STATUS=""
# Cloud Build pulls its own builder image and pushes to GAR, so allow longer
# than the compose stack's local BuildKit.
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

IMAGE_REF="$(json "$WORK/task_now.json" "image_ref")"
IMAGE_DIGEST="$(json "$WORK/task_now.json" "image_digest")"
echo "image_ref:    $IMAGE_REF"
echo "image_digest: $IMAGE_DIGEST"

step "confirming GAR actually serves that digest"
# `images describe` also queries Container Analysis, which is billed to the
# active gcloud project rather than the one in the ref, so it fails with
# USER_PROJECT_DENIED whenever the operator's default project is elsewhere.
# `images list` stays inside Artifact Registry.
GAR_DIGEST="$(gcloud artifacts docker images list "${IMAGE_REF%:*}" --include-tags \
  --filter="tags:${IMAGE_REF##*:}" --format='value(version)' 2>/dev/null | head -1)"
[ -n "$GAR_DIGEST" ] || fail "GAR has no image at $IMAGE_REF"
[ "$GAR_DIGEST" = "$IMAGE_DIGEST" ] || fail "GAR digest $GAR_DIGEST != recorded $IMAGE_DIGEST"

printf '\n\033[32mPASS\033[0m — %s built by Cloud Build and pushed; GAR digest matches.\n' "$IMAGE_REF"
