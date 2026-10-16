#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
# Verify the chart's OpenSandbox extension ConfigMap matches the extension's source files.
uv run --frozen python script/generate_opensandbox_ext_configmap.py --check
