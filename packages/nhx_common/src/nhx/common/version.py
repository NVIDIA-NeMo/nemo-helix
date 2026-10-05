# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAPI metadata versions.

Package release versions are derived from Git tags at build time. OpenAPI still
requires an ``info.version`` value, so keep that field as a neutral placeholder.
"""

OPENAPI_SPEC_VERSION = "0.0.0"

# Backward-compatible name used by existing service code.
platform_api_version = OPENAPI_SPEC_VERSION
