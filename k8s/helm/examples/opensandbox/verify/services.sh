#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Verify the NeMo services extension (nemo-opensandbox-ext.yaml) on an installed server.
#
# Checks: extension health route; a sandbox with a run-once service and a Redis
# service starts only once both are done/ready; the sandbox reaches Redis by name
# and sees the run-once output on a shared volume; the exec route runs in the
# service and refuses non-service containers; a crashing service fails the
# create with NEMO::SERVICE_FAILED and the sandbox is removed.
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
  # Usage: api <method> <path> [json-body]; prints "<http-code> <body>".
  local method="$1" path="$2" body="${3:-}" out code
  out="$(mktemp)"
  code="$(curl -sS --max-time "${READY_TIMEOUT_S}" -o "${out}" -w '%{http_code}' -X "${method}" \
    -H "OPEN-SANDBOX-API-KEY: ${API_KEY}" -H "Content-Type: application/json" \
    ${body:+-d "${body}"} "${BASE_URL}${path}")" || code="000"
  echo "${code} $(cat "${out}")"
  rm -f "${out}"
}

create_body() {
  # Usage: create_body <services-spec-json>; prints a sandbox create request carrying the spec.
  python3 - "$1" "${SANDBOX_IMAGE}" "${SANDBOX_TIMEOUT_S}" <<'PY'
import json, sys
spec, image, timeout = sys.argv[1], sys.argv[2], int(sys.argv[3])
print(json.dumps({
    "image": {"uri": image},
    "entrypoint": ["/bin/sh", "-c", "sleep infinity"],
    "timeout": timeout,
    "resourceLimits": {"cpu": "250m", "memory": "256Mi"},
    "metadata": {"purpose": "services-verify"},
    "extensions": {"nemo.nvidia.com/services": spec},
}))
PY
}

delete_sandbox() {
  # Deletes the sandbox in SANDBOX_ID, if any, and clears it.
  [[ -n "${SANDBOX_ID}" ]] || return 0
  api DELETE "/v1/sandboxes/${SANDBOX_ID}" >/dev/null
  SANDBOX_ID=""
}

require_profile "${PROFILE}"
preflight_control_plane
load_api_key
start_port_forward

read -r code body <<<"$(api GET /v1/nemo-ext/health)"
[[ "${code}" == "200" ]] || die "extension not active (GET /v1/nemo-ext/health -> ${code} ${body}); apply nemo-opensandbox-ext.yaml and restart ${SERVER_DEPLOY}"
ok "extension health: ${body}"

SPEC="$(python3 - "${SERVICE_IMAGE}" "${SANDBOX_IMAGE}" <<'PY'
import json, sys
redis, busybox = sys.argv[1], sys.argv[2]
shared = [{"name": "shared", "mount_path": "/shared"}]
print(json.dumps({
    "version": 1,
    "volumes": ["shared"],
    "services": [
        {"name": "seed", "image": busybox, "run_once": True, "volume_mounts": shared,
         "command": ["sh", "-c", "echo seeded > /shared/seed"]},
        {"name": "cache", "image": redis, "ports": [6379],
         "readiness": {"exec": ["redis-cli", "ping"]}},
    ],
    "sandbox": {"volume_mounts": shared},
}))
PY
)"

info "creating sandbox with services seed (run once) and cache (${SERVICE_IMAGE})"
read -r code body <<<"$(api POST /v1/sandboxes "$(create_body "${SPEC}")")"
[[ "${code}" == "200" || "${code}" == "201" || "${code}" == "202" ]] || die "create failed: ${code} ${body}"
SANDBOX_ID="$(json_field "${body}" 'o["id"]')"
ok "created sandbox id=${SANDBOX_ID}"
wait_sandbox_running
POD="$(find_sandbox_pod)"

order="$(kubectl get pod -n "${WORKLOAD_NS}" "${POD}" -o jsonpath='{range .spec.initContainers[*]}{.name}{" "}{end}')"
[[ "${order}" == *"seed cache "* ]] || die "unexpected init container order: ${order}"
ok "init containers: ${order}"

seed="$(kubectl exec -n "${WORKLOAD_NS}" "${POD}" -c sandbox -- cat /shared/seed)"
[[ "${seed}" == "seeded" ]] || die "sandbox does not see the run-once output (got '${seed}')"
ok "shared volume carries the run-once output"

kubectl exec -n "${WORKLOAD_NS}" "${POD}" -c sandbox -- nc -z -w 5 cache 6379 \
  || die "sandbox cannot reach cache:6379 by name"
ok "sandbox reaches cache:6379 by its Compose name"

read -r code body <<<"$(api POST "/v1/sandboxes/${SANDBOX_ID}/containers/cache/exec" '{"command":["redis-cli","ping"]}')"
[[ "${code}" == "200" && "$(json_field "${body}" 'o["stdout"].strip()')" == "PONG" ]] \
  || die "exec in cache failed: ${code} ${body}"
ok "exec route runs in the service container"

read -r code body <<<"$(api POST "/v1/sandboxes/${SANDBOX_ID}/containers/sandbox/exec" '{"command":["id"]}')"
[[ "${code}" == "404" ]] || die "exec route must refuse the sandbox container, got ${code} ${body}"
ok "exec route refuses non-service containers"
delete_sandbox

FAIL_SPEC="$(python3 - "${SANDBOX_IMAGE}" <<'PY'
import json, sys
print(json.dumps({"version": 1, "services": [
    {"name": "broken", "image": sys.argv[1], "command": ["sh", "-c", "echo bad config >&2; exit 3"],
     "readiness": {"exec": ["true"]}},
]}))
PY
)"
info "creating sandbox with a crashing service (expect NEMO::SERVICE_FAILED within ${FAIL_TIMEOUT_S}s)"
start=${SECONDS}
read -r code body <<<"$(READY_TIMEOUT_S="${FAIL_TIMEOUT_S}" api POST /v1/sandboxes "$(create_body "${FAIL_SPEC}")")"
[[ "${code}" == "422" && "${body}" == *"NEMO::SERVICE_FAILED"* && "${body}" == *"bad config"* ]] \
  || die "expected 422 NEMO::SERVICE_FAILED with the service log, got ${code} ${body}"
ok "crashing service failed the create after $((SECONDS - start))s"

sleep 5
left="$(batchsandbox_names_for_request services-verify | awk 'NF')"
[[ -z "${left}" ]] || warn "BatchSandboxes still present after failed create: ${left}"

ok "PASS services verification (${PROFILE})"
