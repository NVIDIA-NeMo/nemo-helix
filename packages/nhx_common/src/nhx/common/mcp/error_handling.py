# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Error handling utilities for MCP tools."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_ERROR_HINT = "Check the MCP server logs for details, then retry after fixing the request or platform state."


def format_error_response(error: Exception) -> dict[str, Any]:
    """
    Format an exception into a standard error response for MCP tools.

    Provides consistent error structure across all MCP tools with logging.

    Args:
        error: The exception to format

    Returns:
        Dictionary with success=False and a structured error object

    Example:
        >>> try:
        ...     result = await some_operation()
        ... except Exception as e:
        ...     return format_error_response(e)
    """
    logger.error(f"Error in MCP tool: {error}", exc_info=True)
    message = str(error) or type(error).__name__
    return {
        "success": False,
        "error": {
            "code": type(error).__name__,
            "message": message,
            "hint": _DEFAULT_ERROR_HINT,
            "retryable": isinstance(error, (ConnectionError, TimeoutError)),
        },
    }
