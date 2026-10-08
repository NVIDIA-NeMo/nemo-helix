#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Pre-configure the environment before the agent starts.
# This script runs after the API is healthy and CLI auth is configured.
set -e

echo '=== Setting up inference provider ==='
/app/.venv/bin/nemo secrets create --name nvidia-api-key --value "$ANTHROPIC_API_KEY" --description 'NVIDIA inference API key'
/app/.venv/bin/nemo inference providers create --name nvidia-inference --host-url https://inference-api.nvidia.com/v1 --api-key-secret-name nvidia-api-key --description 'NVIDIA inference provider for audit target'

echo '=== Setup complete ==='
