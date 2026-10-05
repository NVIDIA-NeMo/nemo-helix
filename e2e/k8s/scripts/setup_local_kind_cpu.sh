#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Script: setup_local_kind_cpu.sh
# Description: Sets up a local kind cluster for CPU-only Kubernetes E2E tests.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"

KIND_CLUSTER_NAME="${KIND_CLUSTER_NAME:-nhx-e2e}"
KIND_NODE_IMAGE="${KIND_NODE_IMAGE:-kindest/node:v1.33.7@sha256:d26ef333bdb2cbe9862a0f7c3803ecc7b4303d8cea8e814b481b09949d353040}"
KUBE_NAMESPACE="${KUBE_NAMESPACE:-default}"
KUBE_GATEWAY_NAME="${KUBE_GATEWAY_NAME:-nhx-e2e-gateway}"
CLOUD_PROVIDER_KIND_VERSION="${CLOUD_PROVIDER_KIND_VERSION:-v0.10.0}"
GATEWAY_API_VERSION="${GATEWAY_API_VERSION:-v1.4.1}"
GATEWAY_API_STANDARD_CRD_BASE_URL="${GATEWAY_API_STANDARD_CRD_BASE_URL:-https://raw.githubusercontent.com/kubernetes-sigs/gateway-api/${GATEWAY_API_VERSION}/config/crd/standard}"
CLOUD_PROVIDER_KIND_CONTAINER="cloud-provider-kind-${KIND_CLUSTER_NAME}"
CLOUD_PROVIDER_KIND_IMAGE="registry.k8s.io/cloud-provider-kind/cloud-controller-manager:${CLOUD_PROVIDER_KIND_VERSION}"
KIND_ENABLE_GATEWAY="${KIND_ENABLE_GATEWAY:-true}"
KIND_ENABLE_NETWORK_POLICIES="${KIND_ENABLE_NETWORK_POLICIES:-false}"
CALICO_VERSION="${CALICO_VERSION:-v3.32.1}"
CALICO_MANIFEST_URL="${CALICO_MANIFEST_URL:-https://raw.githubusercontent.com/projectcalico/calico/${CALICO_VERSION}/manifests/calico.yaml}"
CALICO_IMAGE_REGISTRY="${CALICO_IMAGE_REGISTRY:-docker.io/calico}"
CALICO_PRELOAD_IMAGES="${CALICO_PRELOAD_IMAGES:-false}"
KIND_NETWORK_POLICY_POD_SUBNET="${KIND_NETWORK_POLICY_POD_SUBNET:-192.168.0.0/16}"

for tool in kind docker kubectl helm; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        log_error "$tool is not installed. Please install it first."
        exit 1
    fi
done

if [ -z "${NGC_API_KEY:-}" ]; then
    log_error "NGC_API_KEY environment variable is required"
    exit 1
fi

retry_command() {
    local max_attempts=5
    local delay_seconds=5
    local attempt

    for attempt in $(seq 1 "${max_attempts}"); do
        if "$@"; then
            return 0
        fi
        if [ "${attempt}" -eq "${max_attempts}" ]; then
            return 1
        fi
        log_warn "Command failed; retrying in ${delay_seconds}s (${attempt}/${max_attempts}): $*"
        sleep "${delay_seconds}"
    done
}

install_network_policy_provider() {
    local calico_manifest
    local calico_image
    local rewritten_calico_manifest

    log_info "Installing Calico ${CALICO_VERSION} for NetworkPolicy enforcement..."

    if [ "${CALICO_PRELOAD_IMAGES}" = "true" ]; then
        for image in cni node kube-controllers; do
            calico_image="${CALICO_IMAGE_REGISTRY}/${image}:${CALICO_VERSION}"
            retry_command docker pull "${calico_image}"
            kind load docker-image --name "${KIND_CLUSTER_NAME}" "${calico_image}"
        done
    fi

    calico_manifest="$(mktemp)"
    rewritten_calico_manifest="$(mktemp)"
    retry_command curl -fsSLo "${calico_manifest}" "${CALICO_MANIFEST_URL}"
    sed "s#quay.io/calico#${CALICO_IMAGE_REGISTRY}#g" "${calico_manifest}" > "${rewritten_calico_manifest}"
    retry_command kubectl apply -f "${rewritten_calico_manifest}"
    rm "${calico_manifest}" "${rewritten_calico_manifest}"

    kubectl -n kube-system rollout status daemonset/calico-node --timeout=5m
    kubectl -n kube-system rollout status deployment/calico-kube-controllers --timeout=5m
    kubectl -n kube-system wait --for=condition=ready pod -l k8s-app=kube-dns --timeout=5m
}

if ! kind get clusters 2>/dev/null | grep -Fxq "${KIND_CLUSTER_NAME}"; then
    log_info "Creating kind cluster ${KIND_CLUSTER_NAME}..."
    if [ "${KIND_ENABLE_NETWORK_POLICIES}" = "true" ]; then
        kind_config="$(mktemp)"
        trap 'rm -f "${kind_config}"' EXIT
        cat > "${kind_config}" <<EOF
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
networking:
  disableDefaultCNI: true
  podSubnet: ${KIND_NETWORK_POLICY_POD_SUBNET}
nodes:
  - role: control-plane
    image: ${KIND_NODE_IMAGE}
EOF
        kind create cluster --name "${KIND_CLUSTER_NAME}" --config "${kind_config}"
        install_network_policy_provider
    else
        kind create cluster --name "${KIND_CLUSTER_NAME}" --image "${KIND_NODE_IMAGE}"
    fi
else
    log_info "kind cluster ${KIND_CLUSTER_NAME} already exists"
    kubectl config use-context "kind-${KIND_CLUSTER_NAME}" >/dev/null
    if [ "${KIND_ENABLE_NETWORK_POLICIES}" = "true" ] && ! kubectl -n kube-system get daemonset calico-node >/dev/null 2>&1; then
        if kubectl -n kube-system get daemonset kindnet >/dev/null 2>&1; then
            log_error "KIND_ENABLE_NETWORK_POLICIES=true requires a cluster created without Kind's default CNI. Delete ${KIND_CLUSTER_NAME} or use a new KIND_CLUSTER_NAME."
            exit 1
        fi
        log_error "KIND_ENABLE_NETWORK_POLICIES=true requires a cluster created with Calico. Delete ${KIND_CLUSTER_NAME} or use a new KIND_CLUSTER_NAME."
        exit 1
    fi
    if [ "${KIND_ENABLE_NETWORK_POLICIES}" = "true" ]; then
        kubectl -n kube-system rollout status daemonset/calico-node --timeout=5m
        kubectl -n kube-system rollout status deployment/calico-kube-controllers --timeout=5m
        kubectl -n kube-system wait --for=condition=ready pod -l k8s-app=kube-dns --timeout=5m
    fi
fi

if [ "${KIND_ENABLE_GATEWAY}" = "true" ]; then
    log_info "Allowing LoadBalancer traffic to control-plane nodes..."
    kubectl label nodes --all node.kubernetes.io/exclude-from-external-load-balancers- >/dev/null 2>&1 || true

    if ! kubectl api-resources --api-group=networking.k8s.io | awk '{print $1}' | grep -Fxq "servicecidrs"; then
        log_error "Kubernetes ServiceCIDR API is not available. cloud-provider-kind ${CLOUD_PROVIDER_KIND_VERSION} requires a kind node image with Kubernetes 1.33 or newer."
        log_error "Current node image setting: ${KIND_NODE_IMAGE}"
        exit 1
    fi

    log_info "Installing Gateway API CRDs (${GATEWAY_API_VERSION})..."
    for crd in gatewayclasses gateways httproutes grpcroutes referencegrants; do
        retry_command kubectl apply --server-side -f "${GATEWAY_API_STANDARD_CRD_BASE_URL}/gateway.networking.k8s.io_${crd}.yaml"
    done

    log_info "Starting cloud-provider-kind (${CLOUD_PROVIDER_KIND_VERSION})..."
    docker rm -f "${CLOUD_PROVIDER_KIND_CONTAINER}" >/dev/null 2>&1 || true
    retry_command docker pull "${CLOUD_PROVIDER_KIND_IMAGE}"
    docker run -d --name "${CLOUD_PROVIDER_KIND_CONTAINER}" --rm \
        --network host \
        -v /var/run/docker.sock:/var/run/docker.sock \
        "${CLOUD_PROVIDER_KIND_IMAGE}" \
        --gateway-channel standard >/dev/null

    log_info "Waiting for Gateway API CRDs and GatewayClass..."
    kubectl wait --for=condition=Established crd/gateways.gateway.networking.k8s.io --timeout=2m
    kubectl wait --for=condition=Established crd/httproutes.gateway.networking.k8s.io --timeout=2m

    for attempt in $(seq 1 60); do
        if kubectl get gatewayclass cloud-provider-kind >/dev/null 2>&1; then
            break
        fi
        if [ "${attempt}" -eq 60 ]; then
            log_error "Timed out waiting for GatewayClass cloud-provider-kind"
            docker logs "${CLOUD_PROVIDER_KIND_CONTAINER}" || true
            exit 1
        fi
        sleep 2
    done
fi

if [ "${KUBE_NAMESPACE}" != "default" ]; then
    log_info "Creating namespace ${KUBE_NAMESPACE}..."
    kubectl create namespace "${KUBE_NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -
fi

KUBECTL_NS=(kubectl -n "${KUBE_NAMESPACE}")

log_info "Creating Kubernetes secrets in namespace ${KUBE_NAMESPACE}..."
create_platform_secrets "${KUBE_NAMESPACE}"

if [ "${KIND_ENABLE_GATEWAY}" = "true" ]; then
    log_info "Creating Gateway ${KUBE_NAMESPACE}/${KUBE_GATEWAY_NAME}..."
    kubectl apply -f - <<EOF
apiVersion: gateway.networking.k8s.io/v1
kind: Gateway
metadata:
  name: ${KUBE_GATEWAY_NAME}
  namespace: ${KUBE_NAMESPACE}
spec:
  gatewayClassName: cloud-provider-kind
  listeners:
    - name: http
      protocol: HTTP
      port: 80
      allowedRoutes:
        namespaces:
          from: Same
EOF

    log_info "Gateway created; it will be programmed after the chart creates its HTTPRoute."
fi

if [ -n "${GITHUB_ENV:-}" ]; then
    {
        echo "KIND_CLUSTER_NAME=${KIND_CLUSTER_NAME}"
        if [ "${KIND_ENABLE_GATEWAY}" = "true" ]; then
            echo "KUBE_GATEWAY_NAME=${KUBE_GATEWAY_NAME}"
            echo "NHX_E2E_CLUSTER_URL="
        fi
    } >> "${GITHUB_ENV}"
fi

log_info "=========================================="
log_info "kind E2E cluster is ready"
log_info "=========================================="
log_info "Cluster: ${KIND_CLUSTER_NAME}"
log_info "Namespace: ${KUBE_NAMESPACE}"
log_info "Gateway setup: ${KIND_ENABLE_GATEWAY}"
if [ "${KIND_ENABLE_GATEWAY}" = "true" ]; then
    log_info "Gateway: ${KUBE_GATEWAY_NAME}"
    log_info "Cluster URL: assigned after Helm install programs the Gateway"
    log_info "cloud-provider-kind container: ${CLOUD_PROVIDER_KIND_CONTAINER}"
fi
log_info "NetworkPolicy enforcement: ${KIND_ENABLE_NETWORK_POLICIES}"
log_info "=========================================="
