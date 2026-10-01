#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


set -e

if [ "${BASH_VERSINFO[0]}" -lt 5 ] || { [ "${BASH_VERSINFO[0]}" -eq 5 ] && [ "${BASH_VERSINFO[1]}" -lt 1 ]; }; then
    echo "install_nhx_e2e.sh requires Bash 5.1 or newer for wait -n -p support" >&2
    exit 1
fi

REPO_ROOT=$(git rev-parse --show-toplevel)

NAMESPACE="${NAMESPACE:-default}"
HELM_RELEASE_NAME="${HELM_RELEASE_NAME:-nemo-helix}"
NHX_E2E_REGISTRY="${NHX_E2E_REGISTRY:-}"
NHX_E2E_TAG="${NHX_E2E_TAG:-}"
HELM_EXTRA_ARGS="${HELM_EXTRA_ARGS:-}"
HELM_CHART="${HELM_CHART:-${REPO_ROOT}/platform/k8s/helm}"
HELM_VALUES="${HELM_VALUES:-${HELM_VALUES_FILE:-${REPO_ROOT}/e2e/k8s/values/default.yaml}}"
POSTGRES_IMAGE="${POSTGRES_IMAGE:-harbor.aire.nvidia.com/docker-proxy/postgres}"
BUSYBOX_IMAGE="${BUSYBOX_IMAGE:-harbor.aire.nvidia.com/docker-proxy/busybox}"
RELEASE_READY_SCRIPT="${RELEASE_READY_SCRIPT:-${REPO_ROOT}/e2e/k8s/scripts/wait_for_release_ready.sh}"
EXTRA_HELM_ARGS=()

if [ -n "${HELM_EXTRA_ARGS}" ]; then
    read -r -a EXTRA_HELM_ARGS <<< "${HELM_EXTRA_ARGS}"
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
    --wait
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

run_helm_with_release_monitor() {
    local helm_pid
    local monitor_pid
    local helm_status
    local monitor_status
    local helm_done=false
    local monitor_done=false
    local completed_pid
    local completed_status
    local wait_pids

    "${RELEASE_READY_SCRIPT}" &
    monitor_pid="$!"

    helm upgrade -i "${HELM_ARGS[@]}" &
    helm_pid="$!"

    while true; do
        wait_pids=()
        if [ "${helm_done}" = "false" ]; then
            wait_pids+=("${helm_pid}")
        fi
        if [ "${monitor_done}" = "false" ]; then
            wait_pids+=("${monitor_pid}")
        fi

        if [ "${#wait_pids[@]}" -eq 0 ]; then
            break
        fi

        completed_pid=""
        set +e
        wait -n -p completed_pid "${wait_pids[@]}"
        completed_status="$?"
        set -e

        if [ -z "${completed_pid}" ]; then
            echo "wait -n returned no completed child (status ${completed_status}); stopping Helm install" >&2
            kill "${helm_pid}" "${monitor_pid}" 2>/dev/null || true
            return 1
        fi

        case "${completed_pid}" in
            "${helm_pid}")
                helm_done=true
                helm_status="${completed_status}"
                ;;
            "${monitor_pid}")
                monitor_done=true
                monitor_status="${completed_status}"
                ;;
        esac

        if [ "${monitor_done}" = "true" ] && [ "${monitor_status}" -ne 0 ]; then
            if [ "${helm_done}" = "false" ]; then
                echo "Release readiness monitor failed; stopping Helm install" >&2
                kill "${helm_pid}" 2>/dev/null || true
                wait "${helm_pid}" 2>/dev/null || true
            fi
            return "${monitor_status}"
        fi

        if [ "${helm_done}" = "true" ] && [ "${helm_status}" -ne 0 ]; then
            if [ "${monitor_done}" = "false" ]; then
                echo "Helm install failed; stopping release readiness monitor" >&2
                kill "${monitor_pid}" 2>/dev/null || true
                wait "${monitor_pid}" 2>/dev/null || true
            fi
            return "${helm_status}"
        fi
    done

    return 0
}

# Install NHX platform
if ! run_helm_with_release_monitor; then
    echo "--- kubectl get pods -A ---"
    kubectl get pods -A
    echo "--- kubectl describe pods -n ${NAMESPACE} ---"
    kubectl describe pods -n "${NAMESPACE}"
    exit 1
fi
