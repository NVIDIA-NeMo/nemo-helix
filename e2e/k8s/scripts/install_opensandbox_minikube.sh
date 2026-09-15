#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Install the cluster-level, shared-kernel OpenSandbox control plane on a running
# minikube cluster and copy the API-key Secret into the platform job namespace.
# This uses opensandbox-server.yaml with secure_runtime unset; it does not apply
# the separate opensandbox-server-kata-qemu.yaml configuration.
#
# The platform Helm chart does not install OpenSandbox. Run this after
# setup_local_minikube_gpu.sh (or setup_local_minikube_cpu.sh), then helm-upgrade
# the platform with sandboxClusterCapable=true.
#
# Charts come from published GitHub Release tarballs by default (no checkout).
# OpenSandbox does not publish a Helm repo index; tags look like
# helm/opensandbox-controller/0.2.0. There is no helm/opensandbox-server
# release, so the server chart is taken from the published opensandbox umbrella
# tarball. Overlay k8s/helm/examples/opensandbox/*.yaml so images stay on
# controller v0.2.0 / server v0.2.1 and [secure_runtime] stays unset.
#
# Usage:
#   ./e2e/k8s/scripts/install_opensandbox_minikube.sh
#
# Environment:
#   OPENSANDBOX_DIR              Optional checkout with kubernetes/charts/{opensandbox-controller,opensandbox-server}
#   OPENSANDBOX_CONTROLLER_CHART Optional chart path or .tgz URL (overrides download / checkout controller)
#   OPENSANDBOX_SERVER_CHART     Optional chart path or .tgz URL (overrides download / checkout server)
#   OPENSANDBOX_CONTROLLER_VERSION  GitHub Release chart version (default: 0.2.0)
#   OPENSANDBOX_UMBRELLA_VERSION    Umbrella tarball that contains the server chart (default: 0.2.2)
#   KUBE_NAMESPACE               Job / Helm-release namespace (default: default). Alias: NMP_NAMESPACE
#   MINIKUBE_PROFILE             kubectl/helm --context (default: minikube). Does not change the current kubeconfig context.
#   SKIP_VERIFY=1                Skip k8s/helm/examples/opensandbox/verify/shared-kernel.sh
#   HELM_TIMEOUT                 Helm --wait timeout (default: 10m)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "${SCRIPT_DIR}/lib.sh"

REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
EXAMPLES="${REPO_ROOT}/k8s/helm/examples/opensandbox"
CONTROL_PLANE_NS="opensandbox-system"
KUBE_NAMESPACE="${KUBE_NAMESPACE:-${NMP_NAMESPACE:-default}}"
NMP_NAMESPACE="${KUBE_NAMESPACE}"
MINIKUBE_PROFILE="${MINIKUBE_PROFILE:-minikube}"
export KUBE_CONTEXT="${KUBE_CONTEXT:-${MINIKUBE_PROFILE}}"
HELM_TIMEOUT="${HELM_TIMEOUT:-10m}"
API_SECRET="opensandbox-server-api-key"
OPENSANDBOX_CONTROLLER_VERSION="${OPENSANDBOX_CONTROLLER_VERSION:-0.2.0}"
OPENSANDBOX_UMBRELLA_VERSION="${OPENSANDBOX_UMBRELLA_VERSION:-0.2.2}"
OPENSANDBOX_RELEASES="${OPENSANDBOX_RELEASES:-https://github.com/opensandbox-group/OpenSandbox/releases/download}"

CHART_CACHE=""
SERVER_VALUES=""
cleanup() {
    rm -f "${SERVER_VALUES:-}"
    if [ -n "${CHART_CACHE:-}" ]; then
        rm -rf "${CHART_CACHE}"
    fi
}
trap cleanup EXIT

kubectl() {
    command kubectl --context "${KUBE_CONTEXT}" "$@"
}

helm() {
    command helm --kube-context "${KUBE_CONTEXT}" "$@"
}

for tool in kubectl helm openssl python3 curl tar; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
        log_error "${tool} is not installed. Please install it first."
        exit 1
    fi
done

if ! kubectl get node >/dev/null 2>&1; then
    log_error "kubectl cannot reach a cluster. Start minikube first:"
    log_error "  ./e2e/k8s/scripts/setup_local_minikube_gpu.sh"
    exit 1
fi

download_tarball() {
    local url="$1"
    local dest="$2"
    log_info "Downloading ${url}"
    curl -fsSL -L -o "${dest}" "${url}"
}

resolve_charts() {
    if [ -n "${OPENSANDBOX_CONTROLLER_CHART:-}" ] && [ -n "${OPENSANDBOX_SERVER_CHART:-}" ]; then
        CONTROLLER_CHART="${OPENSANDBOX_CONTROLLER_CHART}"
        SERVER_CHART="${OPENSANDBOX_SERVER_CHART}"
        log_info "Using explicit chart paths/URLs"
        return
    fi
    if [ -n "${OPENSANDBOX_CONTROLLER_CHART:-}" ] || [ -n "${OPENSANDBOX_SERVER_CHART:-}" ]; then
        log_error "Set both OPENSANDBOX_CONTROLLER_CHART and OPENSANDBOX_SERVER_CHART, or neither."
        exit 1
    fi

    if [ -n "${OPENSANDBOX_DIR:-}" ]; then
        CONTROLLER_CHART="${OPENSANDBOX_DIR}/kubernetes/charts/opensandbox-controller"
        SERVER_CHART="${OPENSANDBOX_DIR}/kubernetes/charts/opensandbox-server"
        if [ ! -d "${CONTROLLER_CHART}" ] || [ ! -d "${SERVER_CHART}" ]; then
            log_error "OpenSandbox charts not found under ${OPENSANDBOX_DIR}/kubernetes/charts/"
            exit 1
        fi
        log_info "Using charts from ${OPENSANDBOX_DIR}"
        return
    fi

    CHART_CACHE="$(mktemp -d)"
    local controller_url="${OPENSANDBOX_RELEASES}/helm/opensandbox-controller/${OPENSANDBOX_CONTROLLER_VERSION}/opensandbox-controller-${OPENSANDBOX_CONTROLLER_VERSION}.tgz"
    local umbrella_url="${OPENSANDBOX_RELEASES}/helm/opensandbox/${OPENSANDBOX_UMBRELLA_VERSION}/opensandbox-${OPENSANDBOX_UMBRELLA_VERSION}.tgz"
    local controller_tgz="${CHART_CACHE}/opensandbox-controller.tgz"
    local umbrella_tgz="${CHART_CACHE}/opensandbox.tgz"

    download_tarball "${controller_url}" "${controller_tgz}"
    download_tarball "${umbrella_url}" "${umbrella_tgz}"
    tar -xzf "${umbrella_tgz}" -C "${CHART_CACHE}"

    CONTROLLER_CHART="${controller_tgz}"
    SERVER_CHART="${CHART_CACHE}/opensandbox/charts/opensandbox-server"
    if [ ! -f "${CONTROLLER_CHART}" ] || [ ! -d "${SERVER_CHART}" ]; then
        log_error "Published tarballs did not contain the expected charts."
        log_error "Controller: ${CONTROLLER_CHART}"
        log_error "Server: ${SERVER_CHART}"
        log_error "OpenSandbox does not publish a helm/opensandbox-server tarball; the server chart is nested in the opensandbox umbrella chart."
        exit 1
    fi
    log_info "Using published tarballs (controller ${OPENSANDBOX_CONTROLLER_VERSION}, server from umbrella ${OPENSANDBOX_UMBRELLA_VERSION})"
}

resolve_charts

log_info "Installing shared-kernel OpenSandbox (control plane: ${CONTROL_PLANE_NS}, jobs: ${KUBE_NAMESPACE}, context: ${KUBE_CONTEXT})"

kubectl create namespace "${CONTROL_PLANE_NS}" --dry-run=client -o yaml | kubectl apply -f -
kubectl create namespace "${KUBE_NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -

log_info "Applying BatchSandbox template ConfigMap..."
kubectl apply -f "${REPO_ROOT}/e2e/k8s/values/batchsandbox-minikube-template.yaml"

if ! kubectl get secret nvcrimagepullsecret -n "${KUBE_NAMESPACE}" >/dev/null 2>&1; then
    log_error "Secret nvcrimagepullsecret is missing in ${KUBE_NAMESPACE}."
    log_error "The BatchSandbox template hard-codes that name. Run setup_local_minikube_gpu.sh first,"
    log_error "or create the pull Secret, or edit imagePullSecrets in ${REPO_ROOT}/e2e/k8s/values/batchsandbox-minikube-template.yaml."
    exit 1
fi

if kubectl get secret "${API_SECRET}" -n "${CONTROL_PLANE_NS}" >/dev/null 2>&1; then
    log_info "Reusing existing ${API_SECRET} in ${CONTROL_PLANE_NS}"
else
    log_info "Creating ${API_SECRET} in ${CONTROL_PLANE_NS}"
    kubectl create secret generic "${API_SECRET}" \
      -n "${CONTROL_PLANE_NS}" \
      --from-literal=api-key="$(openssl rand -hex 32)"
fi

log_info "Copying ${API_SECRET} into job namespace ${KUBE_NAMESPACE}..."
kubectl get secret "${API_SECRET}" -n "${CONTROL_PLANE_NS}" -o json | python3 -c '
import json, sys
secret = json.load(sys.stdin)
for key in ("uid", "resourceVersion", "creationTimestamp", "namespace", "managedFields"):
    secret.get("metadata", {}).pop(key, None)
json.dump(secret, sys.stdout)
' | kubectl apply -n "${KUBE_NAMESPACE}" -f -

SERVER_VALUES="$(mktemp)"
sed "s/REPLACE_WITH_RELEASE_NAMESPACE/${KUBE_NAMESPACE}/g" \
  "${EXAMPLES}/opensandbox-server.yaml" > "${SERVER_VALUES}"

log_info "Helm upgrade opensandbox-controller..."
helm upgrade --install opensandbox-controller "${CONTROLLER_CHART}" \
  --namespace "${CONTROL_PLANE_NS}" \
  --wait --timeout "${HELM_TIMEOUT}" \
  -f "${EXAMPLES}/opensandbox-controller.yaml"

log_info "Helm upgrade opensandbox-server..."
helm upgrade --install opensandbox-server "${SERVER_CHART}" \
  --namespace "${CONTROL_PLANE_NS}" \
  --wait --timeout "${HELM_TIMEOUT}" \
  -f "${SERVER_VALUES}"

log_info "Restarting opensandbox-server to pick up the BatchSandbox template..."
kubectl rollout restart deployment/opensandbox-server -n "${CONTROL_PLANE_NS}"
kubectl rollout status deployment/opensandbox-server -n "${CONTROL_PLANE_NS}" --timeout "${HELM_TIMEOUT}"

if [ "${SKIP_VERIFY:-}" != "1" ]; then
    log_info "Verifying shared-kernel OpenSandbox..."
    OPEN_SANDBOX_WORKLOAD_NS="${KUBE_NAMESPACE}" \
      NMP_NAMESPACE="${KUBE_NAMESPACE}" \
      "${EXAMPLES}/verify/shared-kernel.sh"
fi

log_info "=========================================="
log_info "OpenSandbox shared-kernel install complete"
log_info "=========================================="
log_info "Control plane: ${CONTROL_PLANE_NS}"
log_info "Job namespace: ${KUBE_NAMESPACE}"
log_info "Service DNS:   opensandbox-server.${CONTROL_PLANE_NS}.svc.cluster.local"
log_info ""
log_info "Point the platform Helm release at the server (then helm upgrade):"
log_info "  sandboxClusterCapable: true"
log_info "  opensandbox:"
log_info "    domain: opensandbox-server.${CONTROL_PLANE_NS}.svc.cluster.local"
log_info "    protocol: http"
log_info "    apiKeySecret: ${API_SECRET}"
log_info "    apiKeySecretKey: api-key"
log_info ""
log_info "If you previously set sandboxed_gym_default=false to skip OpenSandbox, restore it to true."
