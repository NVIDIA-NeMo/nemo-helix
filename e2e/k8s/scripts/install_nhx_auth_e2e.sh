#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Install the auth-enabled local E2E harness on minikube.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"

export NAMESPACE="${NAMESPACE:-${KUBE_NAMESPACE:-default}}"
export HELM_RELEASE_NAME="${HELM_RELEASE_NAME:-nemo-helix}"
export HELM_VALUES="${HELM_VALUES:-${REPO_ROOT}/e2e/k8s/values/minikube-auth.yaml}"
export NHX_E2E_REGISTRY="${NHX_E2E_REGISTRY:-my-registry}"
export NHX_E2E_TAG="${NHX_E2E_TAG:-local}"
export POSTGRES_IMAGE="${POSTGRES_IMAGE:-docker.io/library/postgres}"
export BUSYBOX_IMAGE="${BUSYBOX_IMAGE:-docker.io/library/busybox:stable}"

export REQUIRE_NHX_E2E_IMAGES="${REQUIRE_NHX_E2E_IMAGES:-true}"

exec "${SCRIPT_DIR}/install_helm_e2e.sh"
