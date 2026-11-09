#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

uv_bin="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/uv"

exec "${uv_bin}" run --frozen python -m script.generate_openapi_spec "$@"
