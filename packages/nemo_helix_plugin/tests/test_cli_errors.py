# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from io import StringIO

import httpx
from nemo_helix_plugin.cli_errors import (
    format_http_request_error,
    format_http_status_error,
    print_http_request_error,
    print_http_status_error,
)
from rich.console import Console


def test_format_http_status_error_includes_request_target_and_404_hint() -> None:
    request = httpx.Request("GET", "http://test/apis/agents/v2/workspaces/default/agents")
    response = httpx.Response(404, request=request, json={"detail": "Not Found"})
    error = httpx.HTTPStatusError("not found", request=request, response=response)

    message = format_http_status_error(error, action="GET agent API")

    assert "Error: GET agent API failed: HTTP 404 Not Found" in message
    assert "Response:" not in message
    assert "Request: GET http://test/apis/agents/v2/workspaces/default/agents" in message
    assert "Target: agents API route /apis/agents/v2/workspaces/default/agents" in message
    assert "route may not be deployed" in message
    assert "nemo config view" in message


def test_format_http_status_error_prefers_specific_detail_without_response_dump() -> None:
    request = httpx.Request("GET", "http://test/apis/agents/v2/workspaces/default/agents/missing")
    response = httpx.Response(
        404,
        request=request,
        json={
            "detail": "Agent 'missing' not found in workspace 'default'.",
            "path": "/apis/agents/v2/workspaces/default/agents/missing",
            "method": "GET",
        },
    )
    error = httpx.HTTPStatusError("not found", request=request, response=response)

    message = format_http_status_error(error, action="GET agent API")

    assert (
        "Error: GET agent API failed: Agent 'missing' not found in workspace 'default'. (HTTP 404 Not Found)" in message
    )
    assert "Response:" not in message
    assert "Request: GET http://test/apis/agents/v2/workspaces/default/agents/missing" in message
    assert "Target:" not in message
    assert "Check the resource name and workspace" in message


def test_print_http_status_error_uses_rich_error_colors() -> None:
    request = httpx.Request("GET", "http://test/apis/agents/v2/workspaces/default/agents")
    response = httpx.Response(404, request=request, json={"detail": "Not Found"})
    error = httpx.HTTPStatusError("not found", request=request, response=response)
    stream = StringIO()
    console = Console(file=stream, force_terminal=True, color_system="standard", width=200)

    print_http_status_error(error, action="GET agent API", console=console)

    output = stream.getvalue()
    assert "\x1b[" in output
    assert "Error:" in output
    assert "Request:" in output
    assert "Hint:" in output


def test_format_http_request_error_includes_request_target_and_hint() -> None:
    request = httpx.Request("GET", "http://test/apis/agents/v2/workspaces/default/agents")
    error = httpx.ConnectError("connection refused", request=request)

    message = format_http_request_error(error, action="GET agent API")

    assert "Error: GET agent API failed: connection refused" in message
    assert "Request: GET http://test/apis/agents/v2/workspaces/default/agents" in message
    assert "Target: agents API route /apis/agents/v2/workspaces/default/agents" in message
    assert "nemo config view" in message


def test_print_http_request_error_uses_rich_error_colors() -> None:
    request = httpx.Request("GET", "http://test/apis/agents/v2/workspaces/default/agents")
    error = httpx.ConnectError("connection refused", request=request)
    stream = StringIO()
    console = Console(file=stream, force_terminal=True, color_system="standard", width=200)

    print_http_request_error(error, action="GET agent API", console=console)

    output = stream.getvalue()
    assert "\x1b[" in output
    assert "Error:" in output
    assert "Request:" in output
    assert "Hint:" in output
