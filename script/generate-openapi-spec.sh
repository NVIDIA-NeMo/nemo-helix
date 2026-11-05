#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

shim="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/uv"

exec "${shim}" run --frozen python -m script.generate_openapi_spec "$@"
