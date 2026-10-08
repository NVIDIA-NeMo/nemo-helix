#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Verify the shared-kernel OpenSandbox profile (cluster default OCI runtime).
#
# Checks: server Ready + Secret, /health, create sandbox, Running,
# empty runtimeClassName, shared-kernel match (uname -r == node kernel),
# Harbor mounts, cleanup.
#
# Usage:
#   ./shared-kernel.sh
#   READY_TIMEOUT_S=600 SANDBOX_IMAGE=docker.io/library/busybox:1.36 ./shared-kernel.sh
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "${DIR}/lib.sh"
run_profile_verification shared-kernel
