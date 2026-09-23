#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Install NeMo Helix through Helm for local and CI Kubernetes E2E runs.
#
# This script intentionally handles both kind and minikube installs. Cluster
# setup remains in the setup_local_* scripts; this script owns Helm values,
# optional RustFS setup, chart install, and release readiness.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"

NAMESPACE="${NAMESPACE:-${KUBE_NAMESPACE:-default}}"
HELM_RELEASE_NAME="${HELM_RELEASE_NAME:-nemo-helix}"
HELM_CHART="${HELM_CHART:-${REPO_ROOT}/k8s/helm}"
HELM_VALUES="${HELM_VALUES:-${HELM_VALUES_FILE:-${REPO_ROOT}/e2e/k8s/values/default.yaml}}"
HELM_EXTRA_ARGS="${HELM_EXTRA_ARGS:-}"
NHX_E2E_REGISTRY="${NHX_E2E_REGISTRY:-}"
NHX_E2E_TAG="${NHX_E2E_TAG:-}"
NHX_E2E_PULL_POLICY="${NHX_E2E_PULL_POLICY:-}"
REQUIRE_NHX_E2E_IMAGES="${REQUIRE_NHX_E2E_IMAGES:-false}"
POSTGRES_IMAGE="${POSTGRES_IMAGE:-docker.io/library/postgres}"
BUSYBOX_IMAGE="${BUSYBOX_IMAGE:-docker.io/library/busybox:stable}"
RELEASE_READY_SCRIPT="${RELEASE_READY_SCRIPT:-${SCRIPT_DIR}/wait_for_release_ready.sh}"
INSTALL_RUSTFS="${INSTALL_RUSTFS:-false}"
RUSTFS_STORAGECLASS="${RUSTFS_STORAGECLASS:-standard}"
RUSTFS_BUCKET="${RUSTFS_BUCKET:-e2e-k8s-test}"
RUSTFS_ACCESS_KEY="${RUSTFS_ACCESS_KEY:-rustfsadmin}"
RUSTFS_SECRET_KEY="${RUSTFS_SECRET_KEY:-rustfsadmin}"
MINIKUBE_PROFILE="${MINIKUBE_PROFILE:-minikube}"
EXTRA_HELM_ARGS=()


require_non_empty() {
    local name="$1"
    if [ -z "${!name:-}" ]; then
        log_error "${name} is required for ${HELM_RELEASE_NAME} Helm install"
        exit 1
    fi
}

validate_file_inputs() {
    require_non_empty NAMESPACE
    require_non_empty HELM_RELEASE_NAME
    require_non_empty HELM_CHART
    require_non_empty HELM_VALUES
    require_non_empty POSTGRES_IMAGE
    require_non_empty BUSYBOX_IMAGE
    require_non_empty RELEASE_READY_SCRIPT

    if [ ! -d "${HELM_CHART}" ]; then
        log_error "HELM_CHART does not exist or is not a directory: ${HELM_CHART}"
        exit 1
    fi

    if [ ! -f "${HELM_VALUES}" ]; then
        log_error "HELM_VALUES does not exist or is not a file: ${HELM_VALUES}"
        exit 1
    fi

    if [ ! -x "${RELEASE_READY_SCRIPT}" ]; then
        log_error "RELEASE_READY_SCRIPT does not exist or is not executable: ${RELEASE_READY_SCRIPT}"
        exit 1
    fi
}

validate_image_inputs() {
    if [ "${REQUIRE_NHX_E2E_IMAGES}" = "true" ]; then
        require_non_empty NHX_E2E_REGISTRY
        require_non_empty NHX_E2E_TAG
    fi

    if [ -n "${NHX_E2E_REGISTRY}" ] && [ -z "${NHX_E2E_TAG}" ]; then
        log_error "NHX_E2E_TAG is required when NHX_E2E_REGISTRY is set"
        exit 1
    fi

    if [ -z "${NHX_E2E_REGISTRY}" ] && [ -n "${NHX_E2E_TAG}" ]; then
        log_error "NHX_E2E_REGISTRY is required when NHX_E2E_TAG is set"
        exit 1
    fi
}

install_rustfs() {
    log_info "Installing RustFS for S3-compatible E2E storage in namespace ${NAMESPACE}"

    if ! helm repo list 2>/dev/null | awk '{print $1}' | grep -Fxq "rustfs"; then
        helm repo add rustfs https://charts.rustfs.com
    fi
    helm repo update rustfs

    helm upgrade -i rustfs rustfs/rustfs \
        -n "${NAMESPACE}" \
        --create-namespace \
        --version 0.0.85 \
        --set mode.standalone.enabled=true \
        --set mode.distributed.enabled=false \
        --set storageclass.name="${RUSTFS_STORAGECLASS}" \
        --timeout 5m

    log_info "Waiting for RustFS pod to become ready"
    if ! kubectl -n "${NAMESPACE}" wait --for=condition=ready pod -l app.kubernetes.io/name=rustfs --timeout=300s; then
        log_error "RustFS pod failed to become ready"
        kubectl -n "${NAMESPACE}" describe pods -l app.kubernetes.io/name=rustfs || true
        kubectl -n "${NAMESPACE}" logs -l app.kubernetes.io/name=rustfs --tail=50 || true
        return 1
    fi

    log_info "Creating RustFS bucket ${RUSTFS_BUCKET}"
    if ! kubectl -n "${NAMESPACE}" run aws-cli --rm -i --restart=Never \
        --image=amazon/aws-cli:2.22.35 \
        --pod-running-timeout=2m \
        --env="AWS_ACCESS_KEY_ID=${RUSTFS_ACCESS_KEY}" \
        --env="AWS_SECRET_ACCESS_KEY=${RUSTFS_SECRET_KEY}" \
        -- --endpoint-url http://rustfs-svc:9000 s3 mb "s3://${RUSTFS_BUCKET}"; then
        log_error "Failed to create RustFS bucket ${RUSTFS_BUCKET}"
        return 1
    fi
}

add_nvidia_helm_repo() {
    if helm repo list 2>/dev/null | awk '{print $1}' | grep -Fxq "nvidia"; then
        return 0
    fi

    log_info "Adding NVIDIA Helm repo for chart dependencies"
    if [ -n "${NGC_API_KEY:-}" ]; then
        helm repo add nvidia https://helm.ngc.nvidia.com/nvidia --username='$oauthtoken' --password="${NGC_API_KEY}"
    else
        helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
    fi
}

collect_install_diagnostics() {
    echo "--- helm list -A ---"
    helm list -A || true
    echo "--- helm status ${NAMESPACE}/${HELM_RELEASE_NAME} ---"
    helm status -n "${NAMESPACE}" "${HELM_RELEASE_NAME}" || true
    echo "--- kubectl get pods -A ---"
    kubectl get pods -A || true
    echo "--- kubectl describe pods -n ${NAMESPACE} ---"
    kubectl describe pods -n "${NAMESPACE}" || true
}

maybe_export_minikube_cluster_url() {
    if [ -n "${NHX_E2E_CLUSTER_URL:-}" ]; then
        return 0
    fi

    if ! command -v minikube >/dev/null 2>&1; then
        return 0
    fi

    if ! minikube status -p "${MINIKUBE_PROFILE}" >/dev/null 2>&1; then
        return 0
    fi

    local minikube_ip
    minikube_ip="$(minikube ip -p "${MINIKUBE_PROFILE}")"
    NHX_E2E_CLUSTER_URL="http://${minikube_ip}"
    export NHX_E2E_CLUSTER_URL

    if [ -n "${GITHUB_ENV:-}" ]; then
        echo "NHX_E2E_CLUSTER_URL=${NHX_E2E_CLUSTER_URL}" >> "${GITHUB_ENV}"
    fi
}

log_info "Validating Helm install environment"
for tool in kubectl helm; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
        log_error "${tool} is not installed. Please install it first."
        exit 1
    fi
done

validate_file_inputs
validate_image_inputs

if [ -n "${HELM_EXTRA_ARGS}" ]; then
    read -r -a EXTRA_HELM_ARGS <<< "${HELM_EXTRA_ARGS}"
    if echo "${HELM_EXTRA_ARGS}" | grep -q "s3-rustfs"; then
        INSTALL_RUSTFS=true
    fi
fi

if [ "${INSTALL_RUSTFS}" = "true" ]; then
    install_rustfs
fi

HELM_ARGS=(
    "${HELM_RELEASE_NAME}"
    "${HELM_CHART}"
    -n "${NAMESPACE}"
    -f "${HELM_VALUES}"
    "${EXTRA_HELM_ARGS[@]}"
    --set postgresql.image.repository="${POSTGRES_IMAGE}"
    --set core.storage.volumePermissionsImage="${BUSYBOX_IMAGE}"
    --create-namespace
    --timeout 15m
)

if [ -n "${NHX_E2E_REGISTRY}" ]; then
    HELM_ARGS+=(
        --set api.image.repository="${NHX_E2E_REGISTRY}/nhx-api"
        --set core.image.repository="${NHX_E2E_REGISTRY}/nhx-api"
        --set-string platformConfig.platform.image_registry="${NHX_E2E_REGISTRY}"
    )
fi

if [ -n "${NHX_E2E_TAG}" ]; then
    HELM_ARGS+=(
        --set api.image.tag="${NHX_E2E_TAG}"
        --set core.image.tag="${NHX_E2E_TAG}"
        --set-string platformConfig.platform.image_tag="${NHX_E2E_TAG}"
    )
fi

if [ -n "${NHX_E2E_PULL_POLICY}" ]; then
    HELM_ARGS+=(
        --set api.image.pullPolicy="${NHX_E2E_PULL_POLICY}"
        --set core.image.pullPolicy="${NHX_E2E_PULL_POLICY}"
    )
fi

IMAGE_PULL_SECRET_INDEX=0

if [ -n "${NGC_API_KEY:-}" ]; then
    HELM_ARGS+=(--set "imagePullSecrets[${IMAGE_PULL_SECRET_INDEX}].name=nvcrimagepullsecret")
    IMAGE_PULL_SECRET_INDEX=$((IMAGE_PULL_SECRET_INDEX + 1))
fi

if [ -n "${GITHUB_TOKEN:-}" ]; then
    HELM_ARGS+=(--set "imagePullSecrets[${IMAGE_PULL_SECRET_INDEX}].name=ghcr-pull")
    IMAGE_PULL_SECRET_INDEX=$((IMAGE_PULL_SECRET_INDEX + 1))
fi

log_info "Helm install inputs:"
printf '  release: %s\n' "${HELM_RELEASE_NAME}"
printf '  namespace: %s\n' "${NAMESPACE}"
printf '  chart: %s\n' "${HELM_CHART}"
printf '  values: %s\n' "${HELM_VALUES}"
if [ -n "${NHX_E2E_REGISTRY}" ]; then
    printf '  api image: %s/nhx-api:%s\n' "${NHX_E2E_REGISTRY}" "${NHX_E2E_TAG}"
    printf '  core image: %s/nhx-api:%s\n' "${NHX_E2E_REGISTRY}" "${NHX_E2E_TAG}"
    printf '  platform image registry: %s\n' "${NHX_E2E_REGISTRY}"
    printf '  platform image tag: %s\n' "${NHX_E2E_TAG}"
else
    printf '  image overrides: chart defaults\n'
fi
printf '  postgres image: %s\n' "${POSTGRES_IMAGE}"
printf '  core storage volume permissions image: %s\n' "${BUSYBOX_IMAGE}"
if [ -n "${HELM_EXTRA_ARGS}" ]; then
    printf '  extra Helm args: %s\n' "${HELM_EXTRA_ARGS}"
fi

add_nvidia_helm_repo
helm repo update nvidia 2>/dev/null || true
helm dependency build "${HELM_CHART}"

if ! helm upgrade -i "${HELM_ARGS[@]}"; then
    log_error "Helm install/upgrade failed"
    collect_install_diagnostics
    exit 1
fi

if ! "${RELEASE_READY_SCRIPT}"; then
    log_error "Release readiness check failed"
    collect_install_diagnostics
    exit 1
fi

maybe_export_minikube_cluster_url

log_info "Helm values from chart (${HELM_RELEASE_NAME}):"
helm get values -n "${NAMESPACE}" "${HELM_RELEASE_NAME}" || true

log_info "NeMo Helix Helm install complete"
if [ -n "${NHX_E2E_CLUSTER_URL:-}" ]; then
    log_info "Cluster URL: ${NHX_E2E_CLUSTER_URL}"
fi
