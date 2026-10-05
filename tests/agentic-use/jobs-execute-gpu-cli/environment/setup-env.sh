#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

echo '=== Pre-pulling container images ==='
if command -v docker &> /dev/null && [ -S /var/run/docker.sock ]; then
    docker pull nvidia/cuda:12.8.0-base-ubuntu22.04 2>/dev/null && echo 'nvidia/cuda:12.8.0-base-ubuntu22.04 pulled' || echo 'WARNING: Failed to pull nvidia/cuda image'
    docker pull busybox:latest 2>/dev/null && echo 'busybox:latest pulled' || echo 'WARNING: Failed to pull busybox'
else
    echo 'WARNING: Docker not available for pre-pull'
fi

echo '=== Creating workspace ==='
/app/.venv/bin/nhx workspaces create --name gpu-job-workspace || echo 'Workspace may already exist'

echo '=== Environment setup complete ==='
