# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared MCP utilities for NeMo Helix.

This module provides common utilities used across all MCP servers in the platform,
including error handling and shared patterns.
"""

from nhx.common.mcp.error_handling import format_error_response

__all__ = [
    "format_error_response",
]
