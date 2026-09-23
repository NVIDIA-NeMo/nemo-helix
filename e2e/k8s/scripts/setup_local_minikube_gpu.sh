#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Script: setup_local_minikube_gpu.sh
# Description: Sets up a local minikube cluster with GPU support for e2e testing
# Usage: ./setup_local_minikube_gpu.sh
#
# Environment Variables:
#   NGC_API_KEY (required) - NVIDIA NGC API key for pulling GPU images
#   HF_TOKEN (optional) - HuggingFace token for model downloads
#   MINIKUBE_PROFILE (optional) - Minikube profile name (defaults to minikube)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"

MINIKUBE_PROFILE="${MINIKUBE_PROFILE:-minikube}"

log_info "Validating environment..."

if [ -z "${NGC_API_KEY:-}" ]; then
    log_error "NGC_API_KEY environment variable is required"
    log_error "Export your NGC API key: export NGC_API_KEY='your-key'"
    exit 1
fi

for tool in minikube docker kubectl; do
    if ! command -v $tool &> /dev/null; then
        log_error "$tool is not installed. Please install it first."
        exit 1
    fi
done

log_info "Setting up minikube cluster with (optional) GPU support..."


if nvidia-smi > /dev/null 2>&1; then
    log_info "GPUs detected, using --gpus all"
    GPUS="all"
else
    log_info "No GPUs detected, using --gpus ''"
    GPUS=""
fi

# Extra minikube config (e.g. kube-proxy in ipvs mode)
MINIKUBE_EXTRA_ARGS=(--extra-config=kube-proxy.proxy-mode=ipvs)

# Allows direct access via localhost:30080 on macOS/Windows where minikube IP isn't routable
INGRESS_NODEPORT=30080
INGRESS_HOST_PORT=30080

# Start minikube with shared driver, runtime, and extra-config.
start_minikube() {
    minikube start \
      --driver docker \
      --gpus "${GPUS}" \
      --container-runtime docker \
      --cpus max \
      --memory no-limit \
      --wait-timeout=15m \
      --force \
      --profile "${MINIKUBE_PROFILE}" \
      --docker-opt max-concurrent-downloads=16 \
      --ports=${INGRESS_HOST_PORT}:${INGRESS_NODEPORT} \
      "${MINIKUBE_EXTRA_ARGS[@]}"
}

# Start minikube with GPU support if not already running
if ! minikube status -p "${MINIKUBE_PROFILE}" &> /dev/null; then
    log_info "Starting minikube with GPU support..."
    start_minikube

    log_info "Minikube cluster started successfully"
else
    log_info "Minikube cluster already running"
fi

log_info "Setting up ingress controller..."

minikube addons enable ingress -p "${MINIKUBE_PROFILE}"

log_info "Waiting for ingress controller to be ready..."
kubectl -n ingress-nginx wait --for=condition=Available deployment/ingress-nginx-controller --timeout=5m

log_info "Setting ingress controller NodePort to ${INGRESS_NODEPORT} (accessible via localhost:${INGRESS_HOST_PORT})..."
kubectl patch svc ingress-nginx-controller -n ingress-nginx --type='strategic' \
  -p="{\"spec\":{\"ports\":[{\"port\":80,\"nodePort\":${INGRESS_NODEPORT}}]}}"

log_info "Configuring ingress controller..."
kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: ingress-nginx-controller
  namespace: ingress-nginx
data:
  worker-processes: "10"
EOF

# Validation webhook is flaky in minikube
kubectl delete -A ValidatingWebhookConfiguration ingress-nginx-admission 2>/dev/null || true

log_info "Waiting for CoreDNS to be ready..."
kubectl -n kube-system wait --for=condition=Ready pod -l k8s-app=kube-dns --timeout=5m

# Label node so k8s-nim-operator / NIM pods can schedule (they use nodeSelector
# feature.node.kubernetes.io/pci-10de.present: "true"). The NVIDIA GPU operator
# normally adds this via NFD; in minikube we add it manually.
log_info "Labeling node for GPU scheduling (pci-10de.present)..."
kubectl label nodes --all feature.node.kubernetes.io/pci-10de.present=true --overwrite

KUBE_NAMESPACE="${KUBE_NAMESPACE:-default}"
KUBECTL_NS="kubectl -n ${KUBE_NAMESPACE}"

if [ "${KUBE_NAMESPACE}" != "default" ]; then
    log_info "Creating namespace ${KUBE_NAMESPACE}..."
    kubectl create namespace "${KUBE_NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -
fi

log_info "Creating Kubernetes secrets in namespace '${KUBE_NAMESPACE}'..."
create_platform_secrets "${KUBE_NAMESPACE}"

MINIKUBE_IP=$(minikube ip -p "${MINIKUBE_PROFILE}")
CLUSTER_URL="http://${MINIKUBE_IP}"

log_info "=========================================="
log_info "Minikube GPU Setup Complete!"
log_info "=========================================="
log_info ""
log_info "Cluster Information:"
log_info "  Profile: ${MINIKUBE_PROFILE}"
log_info "  IP Address: ${MINIKUBE_IP}"
log_info "  Cluster URL: ${CLUSTER_URL}"
log_info ""
log_info "To install NeMo Helix via Helm (e2e values):"
log_info "  make install-helm-e2e"
log_info "  or: ./e2e/k8s/scripts/install_helm_e2e.sh"
log_info ""
log_info "To run e2e GPU tests (after Helm install):"
log_info "  NHX_E2E_INTERNAL_HOST=nemo-helix-api:8080 uv run --project platform --frozen pytest e2e --kubernetes --feature gpu --cluster-url=\"${CLUSTER_URL}\" -v"
log_info ""
log_info "To check cluster status:"
log_info "  minikube status -p ${MINIKUBE_PROFILE}"
log_info "  kubectl get pods"
log_info ""
log_info "To cleanup:"
log_info "  minikube delete -p ${MINIKUBE_PROFILE}"
log_info "=========================================="
