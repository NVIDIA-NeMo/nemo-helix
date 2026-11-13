#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Install shared-kernel OpenSandbox on a kind cluster made by setup_local_kind_cpu.sh, for the
# sandboxed Gym e2e (the gym_sandbox_e2e tests in e2e/test_evaluator_plugin.py).
#
# A wrapper around install_opensandbox_minikube.sh, which is cluster-agnostic apart from its
# defaults. This one names the kind context and the kind BatchSandbox template, pulls OpenSandbox's
# own images onto the node first, and only warns about a missing pull Secret: on kind the sandbox
# images are loaded onto the node, so the template's ghcr-pull Secret is a fallback.
#
# Run it before installing the platform with e2e/k8s/values/gym-sandbox.yaml, so the API-key Secret
# is in the release namespace when the first job starts.
#
# Usage:
#   KIND_CLUSTER_NAME=nhx-e2e KUBE_NAMESPACE=nemo-helix ./e2e/k8s/scripts/install_opensandbox_kind.sh
#
# Environment:
#   KIND_CLUSTER_NAME   kind cluster name (default: nhx-e2e). The kubectl/helm context is kind-<name>.
#   KUBE_NAMESPACE      Job / Helm-release namespace (default: default)
#   SKIP_PREPULL=1      Do not pull OpenSandbox's images onto the node first
#   Everything else install_opensandbox_minikube.sh reads, such as SKIP_VERIFY and HELM_TIMEOUT.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "${SCRIPT_DIR}/lib.sh"

REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
EXAMPLES="${REPO_ROOT}/k8s/helm/examples/opensandbox"

export KIND_CLUSTER_NAME="${KIND_CLUSTER_NAME:-nhx-e2e}"
export KUBE_CONTEXT="${KUBE_CONTEXT:-kind-${KIND_CLUSTER_NAME}}"
export BATCHSANDBOX_TEMPLATE="${BATCHSANDBOX_TEMPLATE:-${REPO_ROOT}/e2e/k8s/values/batchsandbox-kind-template.yaml}"
export PULL_SECRET_CHECK="${PULL_SECRET_CHECK:-warn}"

# The controller, server, execd and egress images, read from the values this install applies so a
# version bump there needs no second edit here.
opensandbox_images() {
    awk '
        /^[[:space:]]*repository:/ { repository = $2 }
        /^[[:space:]]*tag:/ && repository != "" {
            tag = $2
            gsub(/"/, "", tag)
            print repository ":" tag
            repository = ""
        }
    ' "${EXAMPLES}/opensandbox-controller.yaml" "${EXAMPLES}/opensandbox-server.yaml"
    grep -ohE 'docker\.io/opensandbox/(execd|egress):[A-Za-z0-9._-]+' "${EXAMPLES}/opensandbox-server.yaml"
}

# These come from Docker Hub without credentials. Pulled here, with retries, a rate-limited pull fails
# this step by name. Left to the kubelet, it would fail the Helm wait, or for the egress sidecar land
# inside the first sandbox's create wait and fail a test instead.
if [ "${SKIP_PREPULL:-}" != "1" ]; then
    IMAGES=()
    while IFS= read -r image; do
        IMAGES+=("${image}")
    done < <(opensandbox_images | sort -u)
    if [ "${#IMAGES[@]}" -eq 0 ]; then
        log_error "Found no OpenSandbox images in ${EXAMPLES}/opensandbox-{controller,server}.yaml"
        exit 1
    fi

    log_info "Pulling OpenSandbox images onto the kind node: ${IMAGES[*]}"
    attempt=1
    until "${SCRIPT_DIR}/prepull_kind_images.sh" "${IMAGES[@]}"; do
        if [ "${attempt}" -ge 3 ]; then
            log_error "Could not pull the OpenSandbox images after ${attempt} attempts"
            exit 1
        fi
        log_warn "Pulling the OpenSandbox images failed; retrying in 15s (${attempt}/3)"
        attempt=$((attempt + 1))
        sleep 15
    done
fi

exec "${SCRIPT_DIR}/install_opensandbox_minikube.sh"
