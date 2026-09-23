#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Wait for a Helm release to become ready, failing fast on image pull errors.

set -euo pipefail

NAMESPACE="${NAMESPACE:-default}"
HELM_RELEASE_NAME="${HELM_RELEASE_NAME:-nemo-helix}"
TIMEOUT_SECONDS="${RELEASE_READY_TIMEOUT_SECONDS:-900}"
LOG_INTERVAL_SECONDS="${RELEASE_READY_LOG_INTERVAL_SECONDS:-30}"
GATEWAY_NAME="${KUBE_GATEWAY_NAME:-}"
HTTPROUTE_NAME="${KUBE_HTTPROUTE_NAME:-nemo-helix}"
CLUSTER_INFO_URL="${NHX_E2E_CLUSTER_URL:-}"
SELECTOR="app.kubernetes.io/instance=${HELM_RELEASE_NAME}"
CORE_STORAGE_PVC_NAME="${CORE_STORAGE_PVC_NAME:-${HELM_RELEASE_NAME}-core-storage}"
CORE_STORAGE_BINDER_POD_NAME="${CORE_STORAGE_BINDER_POD_NAME:-${HELM_RELEASE_NAME}-core-storage-binder}"
CORE_STORAGE_BINDER_IMAGE="${CORE_STORAGE_BINDER_IMAGE:-registry.k8s.io/pause:3.10}"
CORE_STORAGE_BINDER_ENABLED="${CORE_STORAGE_BINDER_ENABLED:-true}"
CLUSTER_INFO_URL_EXPORTED=false

if ! command -v jq >/dev/null 2>&1; then
    echo "jq is required for release readiness checks" >&2
    exit 1
fi

deadline=$((SECONDS + TIMEOUT_SECONDS))
next_status_log=$((SECONDS + LOG_INTERVAL_SECONDS))

log() {
    echo "[release-ready] $*"
}

collect_pod_diagnostics() {
    local pod="$1"

    echo "--- pod ${pod} ---"
    kubectl -n "${NAMESPACE}" get pod "${pod}" -o wide || true
    echo "--- describe pod ${pod} ---"
    kubectl -n "${NAMESPACE}" describe pod "${pod}" || true
    echo "--- events for pod ${pod} ---"
    kubectl -n "${NAMESPACE}" get events \
        --field-selector "involvedObject.kind=Pod,involvedObject.name=${pod}" \
        --sort-by='.lastTimestamp' || true
}

check_image_pull_failures() {
    local pods_json
    local failures

    pods_json="$(kubectl -n "${NAMESPACE}" get pods -l "${SELECTOR}" -o json 2>/dev/null || true)"
    if [ -z "${pods_json}" ]; then
        return 0
    fi

    failures="$(jq -r '
      .items[]
      | .metadata.name as $pod
      | ((.status.initContainerStatuses // []) + (.status.containerStatuses // []))[]
      | select(.state.waiting.reason == "ErrImagePull"
          or .state.waiting.reason == "ImagePullBackOff"
          or .state.waiting.reason == "InvalidImageName")
      | [$pod, .name, .image, .state.waiting.reason, (.state.waiting.message // "")]
      | @tsv
    ' <<< "${pods_json}")"

    if [ -z "${failures}" ]; then
        return 0
    fi

    echo "Detected image pull failure in Helm release ${HELM_RELEASE_NAME}:" >&2
    echo "${failures}" | while IFS=$'\t' read -r pod container image reason message; do
        echo "pod=${pod} container=${container} image=${image} reason=${reason}" >&2
        if [ -n "${message}" ]; then
            echo "${message}" >&2
        fi
        collect_pod_diagnostics "${pod}" >&2
    done
    return 1
}

json_count() {
    jq -r '.items | length'
}

deployments_ready() {
    local json
    json="$(kubectl -n "${NAMESPACE}" get deployments -l "${SELECTOR}" -o json 2>/dev/null || true)"
    [ -n "${json}" ] || return 1
    [ "$(json_count <<< "${json}")" -gt 0 ] || return 1
    jq -e '
      all(.items[];
        ((.status.observedGeneration // 0) >= (.metadata.generation // 0))
        and ((.status.readyReplicas // 0) >= (.spec.replicas // 1))
        and ((.status.updatedReplicas // 0) >= (.spec.replicas // 1)))
    ' <<< "${json}" >/dev/null
}

statefulsets_ready() {
    local json
    json="$(kubectl -n "${NAMESPACE}" get statefulsets -l "${SELECTOR}" -o json 2>/dev/null || true)"
    [ -n "${json}" ] || return 1
    jq -e '
      all(.items[];
        ((.status.observedGeneration // 0) >= (.metadata.generation // 0))
        and ((.status.readyReplicas // 0) >= (.spec.replicas // 1)))
    ' <<< "${json}" >/dev/null
}

jobs_ready() {
    local json
    json="$(kubectl -n "${NAMESPACE}" get jobs -l "${SELECTOR}" -o json 2>/dev/null || true)"
    [ -n "${json}" ] || return 1
    jq -e '
      all(.items[];
        any(.status.conditions[]?; .type == "Complete" and .status == "True"))
    ' <<< "${json}" >/dev/null
}

jobs_failed() {
    local json
    json="$(kubectl -n "${NAMESPACE}" get jobs -l "${SELECTOR}" -o json 2>/dev/null || true)"
    [ -n "${json}" ] || return 1
    jq -e '
      any(.items[];
        any(.status.conditions[]?; .type == "Failed" and .status == "True"))
    ' <<< "${json}" >/dev/null
}

core_storage_ready() {
    local pvc_json

    pvc_json="$(kubectl -n "${NAMESPACE}" get pvc "${CORE_STORAGE_PVC_NAME}" -o json 2>/dev/null || true)"
    [ -n "${pvc_json}" ] || return 1
    jq -e '.status.phase == "Bound"' <<< "${pvc_json}" >/dev/null
}

ensure_core_storage_consumer() {
    local pvc_json
    local pvc_phase

    if [ "${CORE_STORAGE_BINDER_ENABLED}" != "true" ]; then
        return 0
    fi

    pvc_json="$(kubectl -n "${NAMESPACE}" get pvc "${CORE_STORAGE_PVC_NAME}" -o json 2>/dev/null || true)"
    if [ -z "${pvc_json}" ]; then
        return 0
    fi

    pvc_phase="$(jq -r '.status.phase // ""' <<< "${pvc_json}")"
    if [ "${pvc_phase}" = "Bound" ]; then
        if kubectl -n "${NAMESPACE}" get pod "${CORE_STORAGE_BINDER_POD_NAME}" >/dev/null 2>&1; then
            log "Core storage PVC is bound; deleting binder pod ${CORE_STORAGE_BINDER_POD_NAME}"
            kubectl -n "${NAMESPACE}" delete pod "${CORE_STORAGE_BINDER_POD_NAME}" --ignore-not-found --wait=false >/dev/null
        fi
        return 0
    fi

    if kubectl -n "${NAMESPACE}" get pod "${CORE_STORAGE_BINDER_POD_NAME}" >/dev/null 2>&1; then
        return 0
    fi

    log "Creating binder pod ${CORE_STORAGE_BINDER_POD_NAME} for PVC ${CORE_STORAGE_PVC_NAME}"
    kubectl -n "${NAMESPACE}" apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: ${CORE_STORAGE_BINDER_POD_NAME}
  labels:
    app.kubernetes.io/instance: ${HELM_RELEASE_NAME}
    app.kubernetes.io/name: nemo-helix
    app.kubernetes.io/component: core-storage-binder
spec:
  restartPolicy: Never
  containers:
    - name: binder
      image: ${CORE_STORAGE_BINDER_IMAGE}
      imagePullPolicy: IfNotPresent
      volumeMounts:
        - name: core-storage
          mountPath: /mnt/core-storage
  volumes:
    - name: core-storage
      persistentVolumeClaim:
        claimName: ${CORE_STORAGE_PVC_NAME}
EOF
}

gateway_address() {
    if [ -z "${GATEWAY_NAME}" ]; then
        return 1
    fi
    kubectl -n "${NAMESPACE}" get gateway "${GATEWAY_NAME}" \
        -o jsonpath='{.status.addresses[0].value}' 2>/dev/null
}

resolve_cluster_info_url() {
    local address

    if [ -n "${CLUSTER_INFO_URL}" ]; then
        return 0
    fi

    address="$(gateway_address || true)"
    if [ -z "${address}" ]; then
        return 1
    fi

    CLUSTER_INFO_URL="http://${address}"
    if [ -n "${GITHUB_ENV:-}" ] && [ "${CLUSTER_INFO_URL_EXPORTED}" = "false" ]; then
        echo "NHX_E2E_CLUSTER_URL=${CLUSTER_INFO_URL}" >> "${GITHUB_ENV}"
        CLUSTER_INFO_URL_EXPORTED=true
    fi
    log "Resolved Gateway URL: ${CLUSTER_INFO_URL}"
}

gateway_ready() {
    if [ -z "${GATEWAY_NAME}" ]; then
        return 0
    fi
    kubectl -n "${NAMESPACE}" get gateway "${GATEWAY_NAME}" -o json 2>/dev/null \
      | jq -e 'any(.status.conditions[]?; .type == "Programmed" and .status == "True")' >/dev/null
}

httproute_ready() {
    if [ -z "${GATEWAY_NAME}" ]; then
        return 0
    fi
    kubectl -n "${NAMESPACE}" get httproute "${HTTPROUTE_NAME}" -o json 2>/dev/null \
      | jq -e '
          any(.status.parents[]?;
            any(.conditions[]?; .type == "Accepted" and .status == "True"))
        ' >/dev/null
}

gateway_url_ready() {
    if [ -z "${GATEWAY_NAME}" ]; then
        return 0
    fi
    resolve_cluster_info_url
}

log_wait_status() {
    local missing=()

    deployments_ready || missing+=("deployments")
    statefulsets_ready || missing+=("statefulsets")
    jobs_ready || missing+=("jobs")
    core_storage_ready || missing+=("core-storage-pvc")
    gateway_ready || missing+=("gateway")
    httproute_ready || missing+=("httproute")
    gateway_url_ready || missing+=("gateway-address")

    if [ "${#missing[@]}" -eq 0 ]; then
        log "All readiness gates are satisfied; waiting for final loop check"
    else
        log "Still waiting for: ${missing[*]}"
    fi

    kubectl -n "${NAMESPACE}" get deployments,statefulsets,jobs,pods,pvc -l "${SELECTOR}" -o wide || true
    if [ -n "${GATEWAY_NAME}" ]; then
        kubectl -n "${NAMESPACE}" get gateway "${GATEWAY_NAME}" -o wide || true
        kubectl -n "${NAMESPACE}" get httproute "${HTTPROUTE_NAME}" -o wide || true
    fi
}

log "Waiting up to ${TIMEOUT_SECONDS}s for release ${NAMESPACE}/${HELM_RELEASE_NAME}"

while [ "${SECONDS}" -lt "${deadline}" ]; do
    check_image_pull_failures
    ensure_core_storage_consumer

    if jobs_failed; then
        log "A release Job failed"
        kubectl -n "${NAMESPACE}" get jobs -l "${SELECTOR}" -o wide || true
        kubectl -n "${NAMESPACE}" describe jobs -l "${SELECTOR}" || true
        exit 1
    fi

    if deployments_ready && statefulsets_ready && jobs_ready && core_storage_ready && gateway_ready && httproute_ready && gateway_url_ready; then
        log "Release ${NAMESPACE}/${HELM_RELEASE_NAME} is ready"
        exit 0
    fi

    if [ "${SECONDS}" -ge "${next_status_log}" ]; then
        log_wait_status
        next_status_log=$((SECONDS + LOG_INTERVAL_SECONDS))
    fi

    sleep 5
done

log "Timed out waiting for release ${NAMESPACE}/${HELM_RELEASE_NAME}"
kubectl -n "${NAMESPACE}" get deployments,statefulsets,jobs,pods,pvc -l "${SELECTOR}" -o wide || true
if [ -n "${GATEWAY_NAME}" ]; then
    kubectl get gatewayclass cloud-provider-kind -o yaml || true
    kubectl -n "${NAMESPACE}" describe gateway "${GATEWAY_NAME}" || true
    kubectl -n "${NAMESPACE}" describe httproute "${HTTPROUTE_NAME}" || true
    if [ -n "${KIND_CLUSTER_NAME:-}" ]; then
        docker logs "cloud-provider-kind-${KIND_CLUSTER_NAME}" || true
    fi
fi
kubectl -n "${NAMESPACE}" get events --sort-by='.lastTimestamp' || true
exit 1
