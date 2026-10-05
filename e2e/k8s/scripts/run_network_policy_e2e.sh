#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Local end-to-end path for Helm NetworkPolicy enforcement.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
source "${SCRIPT_DIR}/lib.sh"

export KIND_ENABLE_NETWORK_POLICIES="${KIND_ENABLE_NETWORK_POLICIES:-true}"
export KIND_ENABLE_GATEWAY="${KIND_ENABLE_GATEWAY:-false}"
export NAMESPACE="${NAMESPACE:-${KUBE_NAMESPACE:-default}}"
export HELM_VALUES="${HELM_VALUES:-${REPO_ROOT}/e2e/k8s/values/kind.yaml}"

if [ -z "${NHX_E2E_REGISTRY:-}" ] || [ -z "${NHX_E2E_TAG:-}" ]; then
    log_warn "NHX_E2E_REGISTRY/NHX_E2E_TAG not set; Helm will use chart default images. For source checkouts, set them to a branch-built nhx-api image."
fi

network_policy_values="${REPO_ROOT}/e2e/k8s/values/network-policies.yaml"
if [ -n "${HELM_EXTRA_ARGS:-}" ]; then
    export HELM_EXTRA_ARGS="${HELM_EXTRA_ARGS} -f ${network_policy_values}"
else
    export HELM_EXTRA_ARGS="-f ${network_policy_values}"
fi

log_info "Setting up kind with Calico NetworkPolicy enforcement"
"${SCRIPT_DIR}/setup_local_kind_cpu.sh"

log_info "Installing Helm chart with NetworkPolicy values"
"${SCRIPT_DIR}/install_helm_e2e.sh"

log_info "Running NetworkPolicy smoke test"
"${SCRIPT_DIR}/test_network_policies.sh"
