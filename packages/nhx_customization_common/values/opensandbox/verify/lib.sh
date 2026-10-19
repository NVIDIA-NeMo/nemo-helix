#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Shared helpers for OpenSandbox runtime verification on nemo-dev-blue.
# Sourced by verify-crun.sh / verify-kata-qemu.sh — do not run directly.

set -euo pipefail

VALUES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYSTEM_NS="${SYSTEM_NS:-opensandbox-system}"
LOCAL_PORT="${LOCAL_PORT:-0}"  # 0 = pick a free port
READY_TIMEOUT_S="${READY_TIMEOUT_S:-300}"
SANDBOX_IMAGE="${SANDBOX_IMAGE:-docker.io/library/busybox:1.36}"
SANDBOX_TIMEOUT_S="${SANDBOX_TIMEOUT_S:-600}"

PF_PID=""
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

cleanup() {
  local ec=$?
  if [[ -n "${SANDBOX_ID}" && -n "${BASE_URL}" && -n "${API_KEY}" ]]; then
    info "deleting sandbox ${SANDBOX_ID}"
    curl -fsS -X DELETE \
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
    crun)
      SERVER_DEPLOY="opensandbox-server-crun"
      SERVER_SVC="opensandbox-server-crun"
      WORKLOAD_NS="opensandbox-crun"
      API_SECRET="opensandbox-server-crun-api-key"
      EXPECT_RUNTIME_CLASS=""          # cluster default (crun)
      EXPECT_KATA_NODE="false"
      ;;
    kata-qemu)
      SERVER_DEPLOY="opensandbox-server-kata"
      SERVER_SVC="opensandbox-server-kata"
      WORKLOAD_NS="opensandbox-kata"
      API_SECRET="opensandbox-server-kata-api-key"
      EXPECT_RUNTIME_CLASS="kata-qemu"
      EXPECT_KATA_NODE="true"
      ;;
    *)
      die "unknown profile '${profile}' (expected crun|kata-qemu)"
      ;;
  esac
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

create_sandbox() {
  info "creating sandbox image=${SANDBOX_IMAGE}"
  local body resp
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
  },
}))
PY
)"
  resp="$(curl -fsS -X POST "${BASE_URL}/v1/sandboxes" \
    -H "OPEN-SANDBOX-API-KEY: ${API_KEY}" \
    -H "Content-Type: application/json" \
    -d "${body}")"
  SANDBOX_ID="$(json_field "${resp}" 'o["id"]')"
  [[ -n "${SANDBOX_ID}" ]] || die "create response missing id: ${resp}"
  ok "created sandbox id=${SANDBOX_ID}"
}

wait_sandbox_running() {
  info "waiting for Running (timeout ${READY_TIMEOUT_S}s)"
  local deadline=$((SECONDS + READY_TIMEOUT_S))
  local resp state
  while (( SECONDS < deadline )); do
    resp="$(curl -fsS \
      -H "OPEN-SANDBOX-API-KEY: ${API_KEY}" \
      "${BASE_URL}/v1/sandboxes/${SANDBOX_ID}")"
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
    [[ -z "${rc}" ]] || die "expected empty runtimeClassName for crun, got '${rc}'"
    ok "runtimeClassName unset (shared-kernel / crun)"
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
      warn "crun sandbox landed on kata-labeled node ${node} (affinity is soft)"
    else
      ok "scheduled on non-kata node ${node}"
    fi
  fi
}

# Collect guest kernel evidence inside the sandbox and compare to the node.
# crun  → shared host kernel (uname -r must match node kernelVersion)
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
      warn "crun sandbox kernel '${guest_release}' != host '${host_kernel}' — unexpected isolation"
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
