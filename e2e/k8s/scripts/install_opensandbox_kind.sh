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
#   KIND_CLUSTER_NAME   kind cluster name (default: nhx-e2e). The kubectl/helm context is always
#                       kind-<name>, so the images are pulled onto the cluster that is installed on.
#   KUBE_NAMESPACE      Job / Helm-release namespace (default: default)
#   SKIP_PREPULL=1      Do not pull the images onto the node first. Also what happens, with a warning,
#                       under bash older than 4.4, which prepull_kind_images.sh cannot run on.
#   Everything else install_opensandbox_minikube.sh reads, such as SKIP_VERIFY and HELM_TIMEOUT.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "${SCRIPT_DIR}/lib.sh"

REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
EXAMPLES="${REPO_ROOT}/k8s/helm/examples/opensandbox"

export KIND_CLUSTER_NAME="${KIND_CLUSTER_NAME:-nhx-e2e}"
# Not overridable: the pre-pull below finds the node by cluster name, and the install must land on
# that same cluster.
export KUBE_CONTEXT="kind-${KIND_CLUSTER_NAME}"
export BATCHSANDBOX_TEMPLATE="${BATCHSANDBOX_TEMPLATE:-${REPO_ROOT}/e2e/k8s/values/batchsandbox-kind-template.yaml}"
export PULL_SECRET_CHECK="${PULL_SECRET_CHECK:-warn}"

# The image the install's closing verification runs a sandbox from, read from the verification's
# own default and exported, so what is pulled here is what it then uses.
if [ "${SKIP_VERIFY:-}" != "1" ] && [ -z "${SANDBOX_IMAGE:-}" ]; then
    SANDBOX_IMAGE="$(sed -n 's/^SANDBOX_IMAGE=".*:-\(.*\)}"$/\1/p' "${EXAMPLES}/verify/lib.sh")"
    if [ -z "${SANDBOX_IMAGE}" ]; then
        log_error "Could not read the verification image default from ${EXAMPLES}/verify/lib.sh"
        exit 1
    fi
    export SANDBOX_IMAGE
fi

# The controller, server, execd and egress images, read from the values this install applies so a
# version bump there needs no second edit here, and the verification sandbox's image.
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
    if [ "${SKIP_VERIFY:-}" != "1" ]; then
        echo "${SANDBOX_IMAGE}"
    fi
}

# prepull_kind_images.sh uses mapfile and expands an empty array under `set -u`, which bash before
# 4.4 (stock macOS has 3.2) cannot do. There the kubelet pulls the images instead, as it did before.
if [ "${SKIP_PREPULL:-}" != "1" ] &&
    { [ "${BASH_VERSINFO[0]}" -lt 4 ] || { [ "${BASH_VERSINFO[0]}" -eq 4 ] && [ "${BASH_VERSINFO[1]}" -lt 4 ]; }; }; then
    log_warn "Not pulling the images onto the node first: that needs bash 4.4 or later, and this is ${BASH_VERSION}."
    SKIP_PREPULL=1
fi

# These come from Docker Hub without credentials. Pulled here, with retries, a rate-limited pull fails
# this step by name. Left to the kubelet, it would fail the Helm wait, or land inside a sandbox's
# create wait: the verification's below, or for the egress sidecar the first Gym host's, in a test.
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
