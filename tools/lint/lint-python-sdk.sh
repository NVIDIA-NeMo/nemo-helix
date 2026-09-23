#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
# Verify Python SDK is up to date with OpenAPI spec.
OUTPUT_DIR="${CI_PROJECT_DIR:-$(pwd)}/python-sdk-lint"
uv run --frozen nemo-helix-sdk-tools is-up-to-date --output-dir "${OUTPUT_DIR}"
