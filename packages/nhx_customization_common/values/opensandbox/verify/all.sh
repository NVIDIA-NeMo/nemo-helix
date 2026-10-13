#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run both OpenSandbox runtime verifications (crun then kata-qemu).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${DIR}/crun.sh"
"${DIR}/kata-qemu.sh"
echo "==> PASS all OpenSandbox runtime verifications"
