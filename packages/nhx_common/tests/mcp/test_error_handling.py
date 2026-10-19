# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared MCP error response formatting."""

from __future__ import annotations

from nhx.common.mcp import format_error_response


def test_format_error_response_returns_structured_error() -> None:
    response = format_error_response(ValueError("bad input"))

    assert response["success"] is False
    assert response["error"] == {
        "code": "ValueError",
        "message": "bad input",
        "hint": "Check the MCP server logs for details, then retry after fixing the request or platform state.",
        "retryable": False,
    }
    assert "error_type" not in response


def test_format_error_response_uses_exception_type_when_message_empty() -> None:
    class EmptyMessageError(Exception):
        pass

    response = format_error_response(EmptyMessageError())

    assert response["error"]["code"] == "EmptyMessageError"
    assert response["error"]["message"] == "EmptyMessageError"


def test_format_error_response_marks_connection_failures_retryable() -> None:
    response = format_error_response(ConnectionError("connection reset"))

    assert response["error"]["retryable"] is True


def test_format_error_response_marks_timeouts_retryable() -> None:
    response = format_error_response(TimeoutError("request timed out"))

    assert response["error"]["retryable"] is True
