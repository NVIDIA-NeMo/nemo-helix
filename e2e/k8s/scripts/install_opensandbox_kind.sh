#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Install shared-kernel OpenSandbox on a kind cluster made by setup_local_kind_cpu.sh, for the
# sandboxed Gym e2e (the gym_sandbox_e2e tests in e2e/test_evaluator_plugin.py).
#
# A wrapper around install_opensandbox_minikube.sh, which is cluster-agnostic apart from its
# defaults. This one names the kind context and the kind BatchSandbox template, and skips the pull
# Secret check: the template's ghcr-pull Secret is a fallback for images already on the node.
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
#   Everything else install_opensandbox_minikube.sh reads, such as SKIP_VERIFY and HELM_TIMEOUT.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

KIND_CLUSTER_NAME="${KIND_CLUSTER_NAME:-nhx-e2e}"
export KUBE_CONTEXT="${KUBE_CONTEXT:-kind-${KIND_CLUSTER_NAME}}"
export BATCHSANDBOX_TEMPLATE="${BATCHSANDBOX_TEMPLATE:-${REPO_ROOT}/e2e/k8s/values/batchsandbox-kind-template.yaml}"
export SANDBOX_PULL_SECRET="${SANDBOX_PULL_SECRET-}"

exec "${SCRIPT_DIR}/install_opensandbox_minikube.sh"
