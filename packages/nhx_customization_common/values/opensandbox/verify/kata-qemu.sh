#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Verify the kata-qemu OpenSandbox profile on nemo-dev-blue.
#
# Checks: RuntimeClass + kata nodes, server Ready + Secret, /health,
# create sandbox, Running, runtimeClassName=kata-qemu, kata node placement,
# guest kernel differs from host (uname -r), cleanup.
#
# Usage:
#   ./kata-qemu.sh
#   READY_TIMEOUT_S=600 ./kata-qemu.sh
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "${DIR}/lib.sh"
run_profile_verification kata-qemu
