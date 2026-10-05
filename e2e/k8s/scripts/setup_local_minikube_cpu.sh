#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Script: setup_local_minikube_cpu.sh
# Description: Sets up a local minikube cluster for CPU-only auth Helm/E2E validation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"

MINIKUBE_PROFILE="${MINIKUBE_PROFILE:-minikube-auth}"
KUBE_NAMESPACE="${KUBE_NAMESPACE:-default}"
INGRESS_NODEPORT="${INGRESS_NODEPORT:-30080}"
INGRESS_HOST_PORT="${INGRESS_HOST_PORT:-30080}"
MINIKUBE_CPUS="${MINIKUBE_CPUS:-4}"
MINIKUBE_MEMORY_MB="${MINIKUBE_MEMORY_MB:-6144}"

for tool in minikube docker kubectl helm; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        log_error "$tool is not installed. Please install it first."
        exit 1
    fi
done

start_minikube() {
    minikube start \
      --driver docker \
      --container-runtime docker \
      --cpus "${MINIKUBE_CPUS}" \
      --memory "${MINIKUBE_MEMORY_MB}" \
      --wait-timeout=15m \
      --force \
      --profile "${MINIKUBE_PROFILE}" \
      --docker-opt max-concurrent-downloads=16 \
      --ports="${INGRESS_HOST_PORT}:${INGRESS_NODEPORT}" \
      --extra-config=kube-proxy.proxy-mode=ipvs
}

if ! minikube status -p "${MINIKUBE_PROFILE}" >/dev/null 2>&1; then
    log_info "Starting CPU-only minikube profile ${MINIKUBE_PROFILE}..."
    start_minikube
else
    log_info "Minikube profile ${MINIKUBE_PROFILE} is already running"
fi

log_info "Enabling ingress addon..."
minikube addons enable ingress -p "${MINIKUBE_PROFILE}"

log_info "Waiting for ingress controller..."
kubectl -n ingress-nginx wait --for=condition=Available deployment/ingress-nginx-controller --timeout=5m

log_info "Patching ingress controller NodePort to ${INGRESS_NODEPORT}..."
kubectl patch svc ingress-nginx-controller -n ingress-nginx --type='strategic' \
  -p="{\"spec\":{\"ports\":[{\"port\":80,\"nodePort\":${INGRESS_NODEPORT}}]}}" >/dev/null

log_info "Configuring ingress controller..."
kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: ingress-nginx-controller
  namespace: ingress-nginx
data:
  worker-processes: "4"
EOF

# Validation webhook can be flaky in local minikube docker-driver setups.
kubectl delete -A ValidatingWebhookConfiguration ingress-nginx-admission >/dev/null 2>&1 || true

log_info "Waiting for CoreDNS..."
kubectl -n kube-system wait --for=condition=Ready pod -l k8s-app=kube-dns --timeout=5m

if [ "${KUBE_NAMESPACE}" != "default" ]; then
    log_info "Creating namespace ${KUBE_NAMESPACE}..."
    kubectl create namespace "${KUBE_NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -
fi

KUBECTL_NS=(kubectl -n "${KUBE_NAMESPACE}")

log_info "Creating platform secrets in namespace ${KUBE_NAMESPACE}..."
create_platform_secrets "${KUBE_NAMESPACE}"

MINIKUBE_IP="$(minikube ip -p "${MINIKUBE_PROFILE}")"

log_info "=========================================="
log_info "Minikube auth harness base is ready"
log_info "=========================================="
log_info "Profile: ${MINIKUBE_PROFILE}"
log_info "Namespace: ${KUBE_NAMESPACE}"
log_info "Minikube IP: ${MINIKUBE_IP}"
log_info "Ingress URL: http://localhost:${INGRESS_HOST_PORT}"
log_info ""
log_info "Next steps:"
log_info "  1. eval \"\$(minikube -p ${MINIKUBE_PROFILE} docker-env)\""
log_info "  2. cd <nemo-helix-checkout> && CI_COMMIT_SHA=\$(git rev-parse HEAD) BAKE_TAG=local IMAGE_REGISTRY=docker.io/my-registry docker buildx bake docker-cpu"
log_info "  3. MINIKUBE_PROFILE=${MINIKUBE_PROFILE} KUBE_NAMESPACE=${KUBE_NAMESPACE} ./e2e/k8s/scripts/install_nhx_auth_e2e.sh"
log_info "  4. MINIKUBE_PROFILE=${MINIKUBE_PROFILE} ./e2e/k8s/scripts/run_auth_e2e.sh"
