#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Smoke-test Helm NetworkPolicy enforcement with Chainsaw.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd -P)"
source "${SCRIPT_DIR}/lib.sh"

usage() {
    cat <<EOF
Usage: ${0##*/}

Runs the Helm NetworkPolicy Chainsaw smoke test against the current Kubernetes context.

Environment:
  NAMESPACE | KUBE_NAMESPACE    Target namespace. Defaults to the current kube context namespace or default.
  CHAINSAW_IMAGE                Docker image used when local chainsaw is unavailable.
  CHAINSAW_REPORT_FORMAT        Report format. Defaults to nil. CI sets XML.
  CHAINSAW_REPORT_NAME          Report file base name. Defaults to report-network-policy-smoke.
  CHAINSAW_REPORT_PATH          Report output directory. Defaults to the repo root.
EOF
}

case "${1:-}" in
    "")
        ;;
    -h | --help)
        usage
        exit 0
        ;;
    *)
        log_error "Unexpected argument: $1"
        usage
        exit 2
        ;;
esac

KUBE_CONTEXT_NAMESPACE="$(kubectl config view --minify -o 'jsonpath={..namespace}' 2>/dev/null || true)"
NAMESPACE="${NAMESPACE:-${KUBE_NAMESPACE:-${KUBE_CONTEXT_NAMESPACE:-default}}}"
HELM_RELEASE_NAME="${HELM_RELEASE_NAME:-nemo-helix}"

CHAINSAW_IMAGE="${CHAINSAW_IMAGE:-ghcr.io/kyverno/chainsaw:v0.2.3}"
CHAINSAW_TEST_DIR="${CHAINSAW_TEST_DIR:-${REPO_ROOT}/e2e/k8s/chainsaw/network-policies}"
CHAINSAW_REPORT_FORMAT="${CHAINSAW_REPORT_FORMAT:-nil}"
CHAINSAW_REPORT_NAME="${CHAINSAW_REPORT_NAME:-report-network-policy-smoke}"
CHAINSAW_REPORT_PATH="${CHAINSAW_REPORT_PATH:-${REPO_ROOT}}"

if [ "${HELM_RELEASE_NAME}" != "nemo-helix" ]; then
    log_error "NetworkPolicy Chainsaw smoke test expects HELM_RELEASE_NAME=nemo-helix"
    exit 1
fi

if [ ! -d "${CHAINSAW_TEST_DIR}" ]; then
    log_error "Chainsaw test directory not found: ${CHAINSAW_TEST_DIR}"
    exit 1
fi

mkdir -p "${CHAINSAW_REPORT_PATH}"
CHAINSAW_REPORT_PATH="$(cd "${CHAINSAW_REPORT_PATH}" && pwd -P)"

run_chainsaw() {
    local test_dir="$1"
    local report_path="$2"

    chainsaw test "${test_dir}" \
        --namespace "${NAMESPACE}" \
        --report-format "${CHAINSAW_REPORT_FORMAT}" \
        --report-name "${CHAINSAW_REPORT_NAME}" \
        --report-path "${report_path}" \
        --fail-fast \
        --no-color
}

flattened_kubeconfig=""

cleanup() {
    if [ -n "${flattened_kubeconfig}" ]; then
        rm -f "${flattened_kubeconfig}"
    fi
}
trap cleanup EXIT

if command -v chainsaw >/dev/null 2>&1; then
    log_info "Running NetworkPolicy smoke test with local chainsaw"
    run_chainsaw "${CHAINSAW_TEST_DIR}" "${CHAINSAW_REPORT_PATH}"
    log_info "NetworkPolicy Chainsaw smoke test passed"
    exit 0
fi

if ! command -v docker >/dev/null 2>&1; then
    log_error "chainsaw is not installed and docker is unavailable"
    exit 1
fi

flattened_kubeconfig="$(mktemp)"
if ! kubectl config view --flatten --raw > "${flattened_kubeconfig}"; then
    log_error "Failed to create flattened kubeconfig for Docker-mounted Chainsaw"
    exit 1
fi

if [ ! -s "${flattened_kubeconfig}" ]; then
    log_error "Flattened kubeconfig is empty for Docker-mounted Chainsaw"
    exit 1
fi

case "${CHAINSAW_TEST_DIR}" in
    "${REPO_ROOT}"/*)
        chainsaw_test_dir_container="/repo/${CHAINSAW_TEST_DIR#"${REPO_ROOT}/"}"
        ;;
    *)
        log_error "Docker-mounted Chainsaw requires CHAINSAW_TEST_DIR to be under ${REPO_ROOT}"
        exit 1
        ;;
esac

log_info "Running NetworkPolicy smoke test with ${CHAINSAW_IMAGE}"
docker run --rm \
    --user "$(id -u):$(id -g)" \
    --network="${CHAINSAW_DOCKER_NETWORK:-host}" \
    -v "${REPO_ROOT}:/repo:ro" \
    -v "${CHAINSAW_REPORT_PATH}:/chainsaw-report" \
    -v "${flattened_kubeconfig}:/kubeconfig:ro" \
    -e KUBECONFIG=/kubeconfig \
    "${CHAINSAW_IMAGE}" test "${chainsaw_test_dir_container}" \
        --namespace "${NAMESPACE}" \
        --report-format "${CHAINSAW_REPORT_FORMAT}" \
        --report-name "${CHAINSAW_REPORT_NAME}" \
        --report-path /chainsaw-report \
        --fail-fast \
        --no-color

log_info "NetworkPolicy Chainsaw smoke test passed"
