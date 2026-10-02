#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# End-to-end check for the compose stack: create a task, upload a task pack,
# finalize it, and confirm the build worker built the image and pushed it to the
# registry. Fails loudly on the first broken step.
#
#   docker compose up -d && ./smoke.sh
#
# The task pack is generated here rather than vendored: the build path only
# needs a Dockerfile at the root of the tarball, so a real benchmark corpus
# would prove nothing extra and Phase 1 vendors no fixtures.
set -euo pipefail

BASE="${BASE:-http://127.0.0.1:8080/apis/scaled-evals}"
REGISTRY="${REGISTRY:-http://127.0.0.1:5000}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

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

step "waiting for /v1/readyz"
for i in $(seq 1 60); do
  if curl -sf "$BASE/v1/readyz" -o "$WORK/readyz.json" 2>/dev/null; then break; fi
  [ "$i" = 60 ] && fail "readyz never returned 200; try: docker compose logs api"
  sleep 5
done
python3 -m json.tool "$WORK/readyz.json"
for check in postgres schema object_store buildkit registry platform_jobs_controller; do
  [ "$(json "$WORK/readyz.json" "checks.$check")" = ok ] || fail "readyz check '$check' is not ok"
done

step "building a task pack"
cat > "$WORK/Dockerfile" <<'EOF'
FROM python:3.12-slim-bookworm
WORKDIR /app
# Stand-in for a Harbor task environment: the pack root only has to carry a
# Dockerfile, which BuildKit uses as both context and dockerfile.
RUN echo "scaled-evals compose smoke" > /app/marker.txt
EOF
tar -czf "$WORK/pack.tar.gz" -C "$WORK" Dockerfile

step "POST /v1/tasks"
# Task names are slugged and unique, so a fixed name makes the second run fail
# with a 409 against a stack that is working perfectly.
NAME="compose-smoke-$(date +%Y%m%d%H%M%S)-$$"
curl -sf -X POST "$BASE/v1/tasks" -H 'content-type: application/json' \
  -d "{\"name\":\"$NAME\",\"description\":\"scaled-evals compose smoke\"}" \
  -o "$WORK/task.json" || fail "task create"
TASK_ID="$(json "$WORK/task.json" "id")"
echo "task_id: $TASK_ID"

step "POST /v1/tasks/$TASK_ID/revisions"
curl -sf -X POST "$BASE/v1/tasks/$TASK_ID/revisions" -o "$WORK/rev.json" || fail "revision create"
UPLOAD_URL="$(json "$WORK/rev.json" "upload.url")"
REVISION="$(json "$WORK/rev.json" "revision")"
echo "revision: $REVISION"

step "PUT the pack to the object store"
# Straight to the store on a presigned URL — this never traverses the platform.
curl -sf -X PUT --upload-file "$WORK/pack.tar.gz" \
  -H 'Content-Type: application/gzip' "$UPLOAD_URL" -o /dev/null || fail "pack upload"

step "POST /v1/tasks/$TASK_ID/finalize"
curl -sf -X POST "$BASE/v1/tasks/$TASK_ID/finalize" -o "$WORK/finalize.json" || fail "finalize"
python3 -m json.tool "$WORK/finalize.json"

step "waiting for the build worker"
STATUS=""
for i in $(seq 1 90); do
  curl -sf "$BASE/v1/tasks/$TASK_ID" -o "$WORK/task_now.json" || fail "task get"
  STATUS="$(json "$WORK/task_now.json" "status")"
  printf '  [%02d] %s\n' "$i" "$STATUS"
  case "$STATUS" in
    ready) break ;;
    failed) fail "build failed: $(json "$WORK/task_now.json" "build_error")" ;;
  esac
  sleep 5
done
[ "$STATUS" = ready ] || fail "revision never reached ready (last: $STATUS)"

IMAGE_REF="$(json "$WORK/task_now.json" "image_ref")"
IMAGE_DIGEST="$(json "$WORK/task_now.json" "image_digest")"
echo "image_ref:    $IMAGE_REF"
echo "image_digest: $IMAGE_DIGEST"

step "confirming the registry actually serves that digest"
# The database saying "ready" is not evidence the push landed.
SERVED="$(curl -sf -o /dev/null -D- \
  -H 'Accept: application/vnd.oci.image.index.v1+json,application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.list.v2+json,application/vnd.docker.distribution.manifest.v2+json' \
  "$REGISTRY/v2/$TASK_ID/manifests/rev$REVISION" | tr -d '\r' | awk -F': ' 'tolower($1)=="docker-content-digest"{print $2}')"
[ -n "$SERVED" ] || fail "registry has no manifest for rev$REVISION"
[ "$SERVED" = "$IMAGE_DIGEST" ] || fail "registry digest $SERVED != recorded $IMAGE_DIGEST"

printf '\n\033[32mPASS\033[0m — %s built and pushed; registry digest matches.\n' "$IMAGE_REF"
