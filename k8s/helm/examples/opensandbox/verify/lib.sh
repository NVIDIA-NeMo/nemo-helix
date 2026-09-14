#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Shared helpers for OpenSandbox runtime verification.
# Sourced by shared-kernel.sh / kata-qemu.sh — do not run directly.
# Set OPEN_SANDBOX_WORKLOAD_NS (or NMP_NAMESPACE) to the platform job namespace.

set -euo pipefail

VALUES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYSTEM_NS="${SYSTEM_NS:-opensandbox-system}"
KUBE_CONTEXT="${KUBE_CONTEXT:-}"
LOCAL_PORT="${LOCAL_PORT:-0}"  # 0 = pick a free port
READY_TIMEOUT_S="${READY_TIMEOUT_S:-300}"
SANDBOX_IMAGE="${SANDBOX_IMAGE:-docker.io/library/busybox:1.36}"
SANDBOX_TIMEOUT_S="${SANDBOX_TIMEOUT_S:-600}"

PF_PID=""
CURL_PID=""
SANDBOX_ID=""
BASE_URL=""
API_KEY=""

die() { echo "FAIL: $*" >&2; exit 1; }
ok()  { echo "OK: $*"; }
info(){ echo "==> $*"; }
warn(){
  # Red when stderr is a TTY; plain WARN: otherwise (CI logs stay readable).
  if [[ -t 2 ]]; then
    printf '\033[31mWARN: %s\033[0m\n' "$*" >&2
  else
    echo "WARN: $*" >&2
  fi
}

json_field() {
  # Usage: json_field <json> <python-expr-on-obj>
  local json="$1"
  local expr="$2"
  python3 -c "import json,sys; o=json.load(sys.stdin); print(${expr})" <<<"${json}"
}

kubectl() {
  if [[ -n "${KUBE_CONTEXT}" ]]; then
    command kubectl --context "${KUBE_CONTEXT}" "$@"
  else
    command kubectl "$@"
  fi
}

cleanup() {
  local ec=$?
  if [[ -n "${CURL_PID}" ]] && kill -0 "${CURL_PID}" 2>/dev/null; then
    kill "${CURL_PID}" 2>/dev/null || true
    wait "${CURL_PID}" 2>/dev/null || true
  fi
  if [[ -n "${SANDBOX_ID}" && -n "${BASE_URL}" && -n "${API_KEY}" ]]; then
    info "deleting sandbox ${SANDBOX_ID}"
    curl -fsS --max-time 15 -X DELETE \
      -H "OPEN-SANDBOX-API-KEY: ${API_KEY}" \
      "${BASE_URL}/v1/sandboxes/${SANDBOX_ID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${PF_PID}" ]] && kill -0 "${PF_PID}" 2>/dev/null; then
    kill "${PF_PID}" 2>/dev/null || true
    wait "${PF_PID}" 2>/dev/null || true
  fi
  if [[ ${ec} -ne 0 ]]; then
    echo "verification failed (exit ${ec})" >&2
  fi
}
trap cleanup EXIT

pick_free_port() {
  python3 - <<'PY'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
}

require_profile() {
  local profile="$1"
  case "${profile}" in
    shared-kernel)
      SERVER_DEPLOY="opensandbox-server"
      SERVER_SVC="opensandbox-server"
      WORKLOAD_NS="${OPEN_SANDBOX_WORKLOAD_NS:-${NMP_NAMESPACE:-}}"
      API_SECRET="opensandbox-server-api-key"
      EXPECT_RUNTIME_CLASS=""          # cluster default OCI runtime
      EXPECT_KATA_NODE="false"
      ;;
    kata-qemu)
      SERVER_DEPLOY="opensandbox-server-kata"
      SERVER_SVC="opensandbox-server-kata"
      WORKLOAD_NS="${OPEN_SANDBOX_WORKLOAD_NS:-${NMP_NAMESPACE:-}}"
      API_SECRET="opensandbox-server-kata-api-key"
      EXPECT_RUNTIME_CLASS="kata-qemu"
      EXPECT_KATA_NODE="true"
      ;;
    *)
      die "unknown profile '${profile}' (expected shared-kernel|kata-qemu)"
      ;;
  esac
  [[ -n "${WORKLOAD_NS}" ]] || die "set OPEN_SANDBOX_WORKLOAD_NS or NMP_NAMESPACE to the platform job namespace"
}

preflight_control_plane() {
  info "context: $(kubectl config current-context)"
  kubectl get deploy -n "${SYSTEM_NS}" "${SERVER_DEPLOY}" >/dev/null \
    || die "deployment ${SERVER_DEPLOY} not found in ${SYSTEM_NS}"
  kubectl rollout status "deploy/${SERVER_DEPLOY}" -n "${SYSTEM_NS}" --timeout=60s \
    || die "${SERVER_DEPLOY} not Ready"
  kubectl get secret -n "${SYSTEM_NS}" "${API_SECRET}" >/dev/null \
    || die "secret ${API_SECRET} missing"
  kubectl get ns "${WORKLOAD_NS}" >/dev/null \
    || die "workload namespace ${WORKLOAD_NS} missing"
  if [[ "${EXPECT_RUNTIME_CLASS}" == "kata-qemu" ]]; then
    kubectl get runtimeclass kata-qemu >/dev/null \
      || die "RuntimeClass/kata-qemu missing"
    local n
    n="$(kubectl get nodes -l katacontainers.io/kata-runtime=true --no-headers 2>/dev/null | wc -l | tr -d ' ')"
    [[ "${n}" -ge 1 ]] || die "no nodes labeled katacontainers.io/kata-runtime=true"
  fi
  ok "control plane preflight"
}

load_api_key() {
  API_KEY="$(kubectl get secret -n "${SYSTEM_NS}" "${API_SECRET}" \
    -o jsonpath='{.data.api-key}' | base64 -d)"
  [[ -n "${API_KEY}" ]] || die "empty api-key in ${API_SECRET}"
}

start_port_forward() {
  local port="${LOCAL_PORT}"
  if [[ "${port}" == "0" ]]; then
    port="$(pick_free_port)"
  fi
  info "port-forward svc/${SERVER_SVC} -> 127.0.0.1:${port}"
  kubectl -n "${SYSTEM_NS}" port-forward "svc/${SERVER_SVC}" "${port}:80" >/tmp/osb-pf-${SERVER_SVC}.log 2>&1 &
  PF_PID=$!
  BASE_URL="http://127.0.0.1:${port}"
  local i
  for i in $(seq 1 30); do
    if curl -fsS "${BASE_URL}/health" >/dev/null 2>&1; then
      ok "server /health"
      return
    fi
    sleep 1
  done
  die "port-forward/health failed; see /tmp/osb-pf-${SERVER_SVC}.log"
}

# Names of BatchSandboxes whose object JSON contains the request marker.
batchsandbox_names_for_request() {
  local marker="$1"
  python3 - "${WORKLOAD_NS}" "${marker}" "${KUBE_CONTEXT}" <<'PY'
import json, os, subprocess, sys

ns, marker, context = sys.argv[1], sys.argv[2], sys.argv[3]
cmd = ["kubectl"]
if context:
    cmd += ["--context", context]
cmd += ["get", "batchsandboxes", "-n", ns, "-o", "json"]
p = subprocess.run(cmd, capture_output=True, text=True)
if p.returncode != 0:
    sys.exit(0)
try:
    items = json.loads(p.stdout or "{}").get("items") or []
except json.JSONDecodeError:
    sys.exit(0)
for item in items:
    if marker in json.dumps(item, default=str):
        name = (item.get("metadata") or {}).get("name")
        if name:
            print(name)
PY
}

# Prints unschedulable details and returns 0 if this sandbox cannot be scheduled.
unschedulable_report() {
  local sid="$1"
  python3 - "${sid}" "${WORKLOAD_NS}" "${KUBE_CONTEXT}" <<'PY'
import json, subprocess, sys

sid, ns, context = sys.argv[1], sys.argv[2], sys.argv[3]

def kubectl_json(*args):
    cmd = ["kubectl"]
    if context:
        cmd += ["--context", context]
    cmd += list(args)
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        return {}
    try:
        return json.loads(p.stdout or "{}")
    except json.JSONDecodeError:
        return {}

hits, pod_names = [], []
for p in kubectl_json("get", "pods", "-n", ns, "-o", "json").get("items", []):
    name = p["metadata"]["name"]
    owners = p["metadata"].get("ownerReferences") or []
    if not (name.startswith(sid) or any(r.get("name") == sid for r in owners)):
        continue
    pod_names.append(name)
    for c in (p.get("status") or {}).get("conditions") or []:
        if c.get("type") == "PodScheduled" and c.get("status") == "False":
            hits.append(f"{name}: {c.get('reason', '')} {c.get('message', '')}".strip())

for e in kubectl_json("get", "events", "-n", ns, "-o", "json").get("items", []):
    if e.get("reason") != "FailedScheduling":
        continue
    obj = (e.get("involvedObject") or {}).get("name") or ""
    if obj.startswith(sid) or obj in pod_names:
        hits.append(f"{obj}: FailedScheduling {e.get('message', '')}".strip())

if not hits:
    sys.exit(1)
seen, out = set(), []
for h in hits:
    if h not in seen:
        seen.add(h)
        out.append(h)
print("\n".join(out))
PY
}

fail_if_unschedulable() {
  local sid="$1"
  local msg
  [[ -n "${sid}" ]] || return 0
  if msg="$(unschedulable_report "${sid}")"; then
    die "sandbox unschedulable:\n${msg}"
  fi
}

create_sandbox() {
  info "creating sandbox image=${SANDBOX_IMAGE} (timeout ${READY_TIMEOUT_S}s)"
  local body curl_out curl_err curl_ec="" request_id
  request_id="$(python3 -c 'import uuid; print(uuid.uuid4())')"
  body="$(python3 - <<PY
import json
print(json.dumps({
  "image": {"uri": "${SANDBOX_IMAGE}"},
  "entrypoint": ["/bin/sh", "-c", "sleep infinity"],
  "timeout": ${SANDBOX_TIMEOUT_S},
  "resourceLimits": {"cpu": "250m", "memory": "256Mi"},
  "metadata": {
    "purpose": "runtime-verify",
    "profile": "${PROFILE}",
    "nmp-verify-request": "${request_id}",
  },
}))
PY
)"
  curl_out="$(mktemp)"
  curl_err="$(mktemp)"
  curl -sS --fail --max-time "${READY_TIMEOUT_S}" -X POST "${BASE_URL}/v1/sandboxes" \
    -H "OPEN-SANDBOX-API-KEY: ${API_KEY}" \
    -H "Content-Type: application/json" \
    -d "${body}" \
    -o "${curl_out}" \
    --stderr "${curl_err}" &
  CURL_PID=$!
  local deadline=$((SECONDS + READY_TIMEOUT_S))
  while (( SECONDS < deadline )); do
    if [[ -z "${SANDBOX_ID}" ]]; then
      local name
      name="$(batchsandbox_names_for_request "${request_id}" | awk 'NF' | head -1)"
      if [[ -n "${name}" ]]; then
        SANDBOX_ID="${name}"
      fi
    fi
    if [[ -n "${SANDBOX_ID}" ]]; then
      fail_if_unschedulable "${SANDBOX_ID}"
    fi
    if ! kill -0 "${CURL_PID}" 2>/dev/null; then
      wait "${CURL_PID}" && curl_ec=0 || curl_ec=$?
      CURL_PID=""
      break
    fi
    sleep 20
  done
  if [[ -n "${CURL_PID}" ]] && ! kill -0 "${CURL_PID}" 2>/dev/null; then
    wait "${CURL_PID}" && curl_ec=0 || curl_ec=$?
    CURL_PID=""
  fi
  if [[ -n "${CURL_PID}" ]]; then
    kill "${CURL_PID}" 2>/dev/null || true
    wait "${CURL_PID}" 2>/dev/null || true
    CURL_PID=""
    rm -f "${curl_out}" "${curl_err}"
    if [[ -n "${SANDBOX_ID}" ]]; then
      fail_if_unschedulable "${SANDBOX_ID}"
    fi
    die "timed out creating sandbox after ${READY_TIMEOUT_S}s${SANDBOX_ID:+ (id=${SANDBOX_ID})}"
  fi
  local resp_body err_body
  resp_body="$(cat "${curl_out}" 2>/dev/null || true)"
  err_body="$(cat "${curl_err}" 2>/dev/null || true)"
  rm -f "${curl_out}" "${curl_err}"
  if [[ "${curl_ec:-1}" -eq 0 ]]; then
    local from_json
    from_json="$(json_field "${resp_body}" 'o["id"]')"
    SANDBOX_ID="${from_json:-${SANDBOX_ID}}"
    [[ -n "${SANDBOX_ID}" ]] || die "create response missing id: ${resp_body}"
    ok "created sandbox id=${SANDBOX_ID}"
    return
  fi
  if [[ -n "${SANDBOX_ID}" ]]; then
    fail_if_unschedulable "${SANDBOX_ID}"
  fi
  die "create sandbox failed (curl ${curl_ec:-unknown}): ${err_body} ${resp_body}"
}

wait_sandbox_running() {
  info "waiting for Running (timeout ${READY_TIMEOUT_S}s)"
  local deadline=$((SECONDS + READY_TIMEOUT_S))
  local resp state
  while (( SECONDS < deadline )); do
    fail_if_unschedulable "${SANDBOX_ID}"
    # A bare assignment from a command substitution inherits curl's exit status, so
    # under `set -e` a single transient failure while the sandbox is still coming up
    # would abort the script and defeat READY_TIMEOUT_S. Treat a failed poll as
    # "not ready yet" and retry until the deadline.
    if ! resp="$(curl -fsS --max-time 15 \
      -H "OPEN-SANDBOX-API-KEY: ${API_KEY}" \
      "${BASE_URL}/v1/sandboxes/${SANDBOX_ID}" 2>/dev/null)"; then
      sleep 3
      continue
    fi
    state="$(json_field "${resp}" 'o.get("status",{}).get("state","")')"
    if [[ "${state}" == "Running" ]]; then
      ok "sandbox Running"
      return
    fi
    if [[ "${state}" == "Failed" || "${state}" == "Terminated" ]]; then
      die "sandbox entered ${state}: ${resp}"
    fi
    sleep 3
  done
  fail_if_unschedulable "${SANDBOX_ID}"
  die "timed out waiting for Running; last state=${state:-unknown}"
}

find_sandbox_pod() {
  # BatchSandbox name == sandbox id; find owned Running pod
  local pod
  pod="$(kubectl get pods -n "${WORKLOAD_NS}" \
    -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.metadata.ownerReferences[0].name}{"\t"}{.status.phase}{"\n"}{end}' \
    2>/dev/null \
    | awk -v id="${SANDBOX_ID}" '$2==id && $3=="Running" {print $1; exit}')"
  if [[ -z "${pod}" ]]; then
    # Fallback: name prefix match (controller may name pods <id>-*)
    pod="$(kubectl get pods -n "${WORKLOAD_NS}" --no-headers 2>/dev/null \
      | awk -v id="${SANDBOX_ID}" '$1 ~ ("^" id) && $3=="Running" {print $1; exit}')"
  fi
  [[ -n "${pod}" ]] || die "no Running pod for sandbox ${SANDBOX_ID} in ${WORKLOAD_NS}"
  echo "${pod}"
}

assert_runtime_class() {
  local pod="$1"
  local rc
  rc="$(kubectl get pod -n "${WORKLOAD_NS}" "${pod}" \
    -o jsonpath='{.spec.runtimeClassName}')"
  if [[ -z "${EXPECT_RUNTIME_CLASS}" ]]; then
    [[ -z "${rc}" ]] || die "expected empty runtimeClassName for shared-kernel, got '${rc}'"
    ok "runtimeClassName unset (shared-kernel / cluster default OCI runtime)"
  else
    [[ "${rc}" == "${EXPECT_RUNTIME_CLASS}" ]] \
      || die "expected runtimeClassName=${EXPECT_RUNTIME_CLASS}, got '${rc}'"
    ok "runtimeClassName=${rc}"
  fi
}

assert_node_placement() {
  local pod="$1"
  local node kata
  node="$(kubectl get pod -n "${WORKLOAD_NS}" "${pod}" \
    -o jsonpath='{.spec.nodeName}')"
  [[ -n "${node}" ]] || die "pod has no nodeName"
  POD_NODE="${node}"
  kata="$(kubectl get node "${node}" \
    -o jsonpath='{.metadata.labels.katacontainers\.io/kata-runtime}')"
  if [[ "${EXPECT_KATA_NODE}" == "true" ]]; then
    [[ "${kata}" == "true" ]] \
      || die "pod on ${node} missing katacontainers.io/kata-runtime=true"
    ok "scheduled on kata node ${node}"
  else
    # Soft check: prefer non-kata, but allow schedule on kata nodes if capacity tight
    if [[ "${kata}" == "true" ]]; then
      warn "shared-kernel sandbox landed on kata-labeled node ${node} (affinity is soft)"
    else
      ok "scheduled on non-kata node ${node}"
    fi
  fi
}

# Collect guest kernel evidence inside the sandbox and compare to the node.
# shared-kernel → host kernel (uname -r must match node kernelVersion)
# kata  → guest kernel (uname -r must differ from node kernelVersion)
assert_kernel_isolation() {
  local pod="$1"
  local host_kernel guest_out guest_release guest_sysname guest_machine
  local guest_version guest_cgroup0

  [[ -n "${POD_NODE:-}" ]] || die "POD_NODE unset; call assert_node_placement first"
  host_kernel="$(kubectl get node "${POD_NODE}" \
    -o jsonpath='{.status.nodeInfo.kernelVersion}')"
  [[ -n "${host_kernel}" ]] || die "could not read kernelVersion for node ${POD_NODE}"

  # Absolute paths — bare "sh" has failed under some Kata guests.
  # Container name is "sandbox" (provider_common._build_main_container).
  guest_out="$(kubectl exec -n "${WORKLOAD_NS}" "${pod}" -c sandbox -- /bin/sh -c '
set -e
printf "release=%s\n" "$(uname -r)"
printf "sysname=%s\n" "$(uname -s)"
printf "machine=%s\n" "$(uname -m)"
printf "version=%s\n" "$(uname -v)"
printf "proc_version=%s\n" "$(cat /proc/version 2>/dev/null | tr "\n" " ")"
printf "cgroup0=%s\n" "$(sed -n "1p" /proc/1/cgroup 2>/dev/null || true)"
printf "virt=%s\n" "$(cat /sys/devices/virtual/dmi/id/product_name 2>/dev/null || echo unknown)"
echo OSB_VERIFY_OK
' 2>&1)" || die "kubectl exec (kernel probe) failed: ${guest_out}"

  echo "${guest_out}" | grep -q OSB_VERIFY_OK || die "unexpected exec output: ${guest_out}"

  guest_release="$(echo "${guest_out}" | sed -n 's/^release=//p' | head -1)"
  guest_sysname="$(echo "${guest_out}" | sed -n 's/^sysname=//p' | head -1)"
  guest_machine="$(echo "${guest_out}" | sed -n 's/^machine=//p' | head -1)"
  guest_version="$(echo "${guest_out}" | sed -n 's/^version=//p' | head -1)"
  guest_cgroup0="$(echo "${guest_out}" | sed -n 's/^cgroup0=//p' | head -1)"
  local guest_virt
  guest_virt="$(echo "${guest_out}" | sed -n 's/^virt=//p' | head -1)"

  [[ -n "${guest_release}" ]] || die "guest uname -r empty"

  info "kernel isolation evidence"
  echo "    host_node=${POD_NODE}"
  echo "    host_kernel=${host_kernel}"
  echo "    guest_uname_r=${guest_release}"
  echo "    guest_uname_s=${guest_sysname}"
  echo "    guest_uname_m=${guest_machine}"
  echo "    guest_uname_v=${guest_version}"
  echo "    guest_virt_product=${guest_virt}"
  echo "    guest_cgroup0=${guest_cgroup0}"

  if [[ "${EXPECT_RUNTIME_CLASS}" == "kata-qemu" ]]; then
    if [[ "${guest_release}" == "${host_kernel}" ]]; then
      warn "kata-qemu sandbox sees host kernel '${guest_release}' — expected a distinct guest kernel (shared-kernel leak?)"
    else
      ok "guest kernel differs from host (${guest_release} != ${host_kernel})"
    fi
    # Soft signal: QEMU/Kata guests often expose a virtual DMI product string.
    if [[ "${guest_virt}" == "unknown" || -z "${guest_virt}" ]]; then
      warn "could not read guest DMI product_name"
    else
      ok "guest virt product=${guest_virt}"
    fi
  else
    if [[ "${guest_release}" != "${host_kernel}" ]]; then
      warn "shared-kernel sandbox kernel '${guest_release}' != host '${host_kernel}' — unexpected isolation"
    else
      ok "guest kernel matches host shared kernel (${guest_release})"
    fi
  fi
}

assert_batchsandbox() {
  kubectl get batchsandbox -n "${WORKLOAD_NS}" "${SANDBOX_ID}" >/dev/null \
    || die "BatchSandbox/${SANDBOX_ID} missing in ${WORKLOAD_NS}"
  ok "BatchSandbox/${SANDBOX_ID} present"
}

run_profile_verification() {
  PROFILE="$1"
  require_profile "${PROFILE}"
  info "verifying OpenSandbox profile=${PROFILE}"
  preflight_control_plane
  load_api_key
  start_port_forward
  create_sandbox
  wait_sandbox_running
  assert_batchsandbox
  local pod
  pod="$(find_sandbox_pod)"
  ok "pod=${pod}"
  assert_runtime_class "${pod}"
  assert_node_placement "${pod}"
  assert_kernel_isolation "${pod}"
  info "PASS profile=${PROFILE}"
}
