#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Shared settings for setup_authentik_taskset_runs.sh and authentik_user_token.sh. Sourced, not run.
#
#   KIND_CLUSTER_NAME   the kind cluster (default nhx-authentik-reuse, run.sh's reusable cluster)
#   NHX_E2E_STATE_DIR   secrets that must outlive one step: registry password, signing key, app passwords, tokens
#                       (default ~/.cache/nhx-e2e/<cluster>, 0700; never printed)

# shellcheck disable=SC2034  # used by the scripts that source this
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CLUSTER="${KIND_CLUSTER_NAME:-nhx-authentik-reuse}"
NAMESPACE="${NHX_AUTHENTIK_K8S_NAMESPACE:-nemo-authentik}"
RELEASE="${NHX_AUTHENTIK_K8S_HELM_RELEASE:-authentik-demo}"
GATEWAY_PORT="${NHX_AUTHENTIK_K8S_GATEWAY_PORT:-18082}"
GATEWAY_URL="https://127.0.0.1:${GATEWAY_PORT}"
AUTHENTIK_API_PORT="${AUTHENTIK_API_PORT:-19000}"
STATE_DIR="${NHX_E2E_STATE_DIR:-${HOME}/.cache/nhx-e2e/${CLUSTER}}"
GENERATED_DIR="${REPO_ROOT}/contrib/auth/authentik/.generated"
CLUSTER_DIR="${GENERATED_DIR}/k8s/${CLUSTER}"
# run.sh writes an isolated kubeconfig; every kubectl and helm call here and after uses it.
export KUBECONFIG="${CLUSTER_DIR}/kubeconfig.yaml"
export KUBE_CONTEXT="kind-${CLUSTER}"
# auth.admin_email in contrib/auth/authentik/helm/values.yaml; no demo user has it until the setup gives it.
PLATFORM_ADMIN_EMAIL="admin@example.com"

log() { printf '[authentik-e2e] %s\n' "$*" >&2; }
die() { printf '[authentik-e2e] error: %s\n' "$*" >&2; exit 1; }

# authentik_admin <command> <args...>: authentik_admin.py through a short-lived port-forward to the Authentik
# server. The bootstrap token comes from run.sh's generated authentik.env and never reaches the log.
authentik_admin() {
    kubectl --context "${KUBE_CONTEXT}" -n "${NAMESPACE}" port-forward svc/authentik-server "${AUTHENTIK_API_PORT}:80" \
        >/dev/null 2>&1 &
    local forward=$!
    local status=0
    for _ in $(seq 1 30); do curl -s -o /dev/null "http://127.0.0.1:${AUTHENTIK_API_PORT}/" && break; sleep 1; done
    (
        set -a
        # shellcheck disable=SC1091
        source "${GENERATED_DIR}/authentik.env"
        set +a
        AUTHENTIK_API="http://127.0.0.1:${AUTHENTIK_API_PORT}" python3 "${REPO_ROOT}/e2e/k8s/scripts/authentik_admin.py" "$@"
    ) || status=$?
    kill "${forward}" 2>/dev/null || true
    return "${status}"
}

# The gateway on GATEWAY_URL, restarted whenever kubectl port-forward drops (it does, under load).
gateway_keepalive() {
    local pid_file="${STATE_DIR}/port-forward.pid"
    [[ -f "${pid_file}" ]] && kill "$(cat "${pid_file}")" 2>/dev/null || true
    # run.sh's own port-forward holds the port.
    [[ -f "${CLUSTER_DIR}/port-forward.pid" ]] && kill "$(cat "${CLUSTER_DIR}/port-forward.pid")" 2>/dev/null || true
    rm -f "${CLUSTER_DIR}/port-forward.pid"
    sleep 1
    nohup bash -c "while true; do
        kubectl --kubeconfig '${KUBECONFIG}' --context '${KUBE_CONTEXT}' -n '${NAMESPACE}' \
            port-forward svc/nemo-helix-envoy '${GATEWAY_PORT}:8080'
        sleep 1
    done" >"${STATE_DIR}/port-forward.log" 2>&1 &
    echo $! >"${pid_file}"
    for _ in $(seq 1 60); do
        curl -sf --cacert "${CLUSTER_DIR}/ca.crt" "${GATEWAY_URL}/health/ready" >/dev/null && return
        sleep 2
    done
    die "gateway not ready on ${GATEWAY_URL}; see ${STATE_DIR}/port-forward.log"
}
