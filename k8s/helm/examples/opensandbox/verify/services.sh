#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Verify the NeMo Compose services extension (README, "Compose services") on an installed server.
#
# Checks: the extension's health route; a sandbox with a run-once service and a Redis sidecar
# comes up with both, sees the run-once output on a shared volume, and reaches Redis by name;
# the exec route runs in a service; a crashing service is reported, with its log, by the
# services route.
#
# Usage:
#   ./services.sh                 # shared-kernel server
#   ./services.sh kata-qemu       # Kata server
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "${DIR}/lib.sh"

PROFILE="${1:-shared-kernel}"
SERVICE_IMAGE="${SERVICE_IMAGE:-docker.io/library/redis:7-alpine}"
FAIL_TIMEOUT_S="${FAIL_TIMEOUT_S:-180}"

api() {
  # Usage: api <method> <path> [json-body]; prints the response body and fails on an HTTP error.
  curl -fsS --max-time 30 -X "$1" -H "OPEN-SANDBOX-API-KEY: ${API_KEY}" -H "Content-Type: application/json" \
    ${3:+-d "$3"} "${BASE_URL}$2"
}

create() {
  # Usage: create <services-spec-json>; creates a busybox sandbox carrying the spec and sets SANDBOX_ID.
  local body
  body="$(api POST /v1/sandboxes "{
    \"image\": {\"uri\": \"${SANDBOX_IMAGE}\"},
    \"entrypoint\": [\"/bin/sh\", \"-c\", \"sleep infinity\"],
    \"timeout\": ${SANDBOX_TIMEOUT_S},
    \"resourceLimits\": {\"cpu\": \"250m\", \"memory\": \"256Mi\"},
    \"metadata\": {\"purpose\": \"services-verify\"},
    \"extensions\": {\"opensandbox.extensions.nemo-compose-services\": $(json_field "$1" 'json.dumps(json.dumps(o))')}
  }")" || die "create failed"
  SANDBOX_ID="$(json_field "${body}" 'o["id"]')"
  ok "created sandbox id=${SANDBOX_ID}"
}

require_profile "${PROFILE}"
preflight_control_plane
load_api_key
start_port_forward

health="$(api GET /v1/nemo-ext/health)" || die "extension not active; see README, Compose services"
ok "extension health: ${health}"

# A run-once service writes to a shared volume, and a Redis sidecar must pass its readiness
# check, before the sandbox container starts.
info "creating sandbox with services seed (run once) and cache (${SERVICE_IMAGE})"
create "$(cat <<JSON
{
  "volumes": ["shared"],
  "services": [
    {"name": "seed", "image": "${SANDBOX_IMAGE}", "role": "run_once",
     "volume_mounts": [{"name": "shared", "mount_path": "/shared"}],
     "command": ["sh", "-c", "echo seeded > /shared/seed"]},
    {"name": "cache", "image": "${SERVICE_IMAGE}", "ports": [6379],
     "readiness": {"exec": ["redis-cli", "ping"]}}
  ],
  "main": {"volume_mounts": [{"name": "shared", "mount_path": "/shared"}]}
}
JSON
)"
wait_sandbox_running
# Create returns once the pod has an IP; the pod is Ready once every container, sidecars included, is.
kubectl wait pod -n "${WORKLOAD_NS}" -l "opensandbox.io/id=${SANDBOX_ID}" --for=condition=Ready \
  --timeout="${READY_TIMEOUT_S}s" >/dev/null || die "sandbox pod never became ready"
POD="$(find_sandbox_pod)"

status="$(api GET "/v1/nemo-ext/sandboxes/${SANDBOX_ID}/services")"
[[ "$(json_field "${status}" 'o["failure"] is None and not o["not_ready"]')" == "True" ]] \
  || die "services route should report all ready: ${status}"
ok "services route: ${status}"

# The sandbox can only read this if seed ran to completion before it started.
seed="$(kubectl exec -n "${WORKLOAD_NS}" "${POD}" -c sandbox -- cat /shared/seed)"
[[ "${seed}" == "seeded" ]] || die "sandbox does not see the run-once output (got '${seed}')"
ok "shared volume carries the run-once output"

kubectl exec -n "${WORKLOAD_NS}" "${POD}" -c sandbox -- nc -z -w 5 cache 6379 \
  || die "sandbox cannot reach cache:6379 by name"
ok "sandbox reaches cache:6379 by its Compose name"

pong="$(api POST "/v1/nemo-ext/sandboxes/${SANDBOX_ID}/services/cache/exec" '{"command":["redis-cli","ping"]}')"
[[ "$(json_field "${pong}" 'o["stdout"].strip()')" == "PONG" ]] || die "exec in cache failed: ${pong}"
ok "exec route runs in the service container"

api DELETE "/v1/sandboxes/${SANDBOX_ID}" >/dev/null
SANDBOX_ID=""

# A service that exits at once: the services route must report it, with its log, well before
# the sandbox would time out.
info "creating sandbox with a crashing service (expect a reported failure within ${FAIL_TIMEOUT_S}s)"
create "$(cat <<JSON
{"services": [{"name": "broken", "image": "${SANDBOX_IMAGE}",
  "command": ["sh", "-c", "echo bad config >&2; exit 3"], "readiness": {"exec": ["true"]}}]}
JSON
)"
start=${SECONDS}
failure=""
until [[ -n "${failure}" ]] || (( SECONDS - start >= FAIL_TIMEOUT_S )); do
  sleep 3
  # The route 404s until the pod exists; keep polling through that.
  status="$(api GET "/v1/nemo-ext/sandboxes/${SANDBOX_ID}/services" 2>/dev/null)" || continue
  failure="$(json_field "${status}" 'o["failure"] or ""')"
done
[[ "${failure}" == *"broken"* && "${failure}" == *"bad config"* ]] \
  || die "expected a failure for service 'broken' with its log, got: ${status:-no response}"
ok "crashing service reported after $((SECONDS - start))s: ${failure%%$'\n'*}"

ok "PASS services verification (${PROFILE})"
