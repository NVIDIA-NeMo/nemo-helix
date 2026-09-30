# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integration smoke test for NeMo MCP server.

Creates and tests an MCP server instance that is connected to a running NeMo Helix instance.
"""

from __future__ import annotations

import os
from typing import Any, Generator

import pytest
from fastmcp import FastMCP
from mcp.types import TextContent
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nhx.core.mcp.server import create_server


@pytest.fixture(scope="module")
def nhx_base_url() -> str:
    """Get NeMo Helix base URL from environment or use default."""
    return os.environ.get("NHX_BASE_URL", "http://localhost:8080")


@pytest.fixture(scope="module")
def workspaces_client(nhx_base_url: str) -> Generator[WorkspacesClient, None, None]:
    """Typed workspaces client for direct API validation."""
    with WorkspacesClient(base_url=nhx_base_url) as client:
        yield client


@pytest.fixture(scope="module")
def mcp_server(nhx_base_url: str) -> Generator[FastMCP, None, None]:
    """Create MCP server instance."""
    server = create_server(nhx_base_url)
    yield server


def _text_content(tool_result: Any) -> str:
    first_content = tool_result.content[0]
    assert isinstance(first_content, TextContent)
    return first_content.text


class TestMCPServerSmoke:
    """Smoke tests for MCP server basic functionality."""

    def test_nhx_connection(self, workspaces_client: WorkspacesClient) -> None:
        """Verify we can connect to NeMo Helix instance."""
        # This will raise if NeMo Helix is not accessible
        response = workspaces_client.list_workspaces()
        assert response is not None
        assert response.page().items is not None

    def test_mcp_server_created(self, mcp_server: FastMCP) -> None:
        """Verify MCP server instance is created."""
        assert mcp_server is not None

    @pytest.mark.asyncio
    async def test_list_workspaces_tool_registered(self, mcp_server: FastMCP) -> None:
        """Verify list_workspaces tool is registered."""
        tools = await mcp_server.list_tools()
        tool_names = [t.name for t in tools]
        assert "list_workspaces" in tool_names

    @pytest.mark.asyncio
    async def test_list_workspaces_matches_api(self, mcp_server: FastMCP, workspaces_client: WorkspacesClient) -> None:
        """
        Verify MCP tool returns consistent data with the API.

        This ensures the MCP server is properly connected to NeMo Helix
        and returning real data.
        """
        import json

        # Get workspaces via SDK
        api_response = workspaces_client.list_workspaces()
        api_workspace_ids = {ws.id for ws in api_response.items()}

        # Get workspaces via MCP tool
        tool_result = await mcp_server.call_tool("list_workspaces", {})

        mcp_result = json.loads(_text_content(tool_result))
        assert isinstance(mcp_result, dict)  # Type narrowing for ty

        # Verify MCP returns same workspace IDs
        mcp_workspace_ids = {ws["id"] for ws in mcp_result["workspaces"]}

        assert api_workspace_ids == mcp_workspace_ids, (
            f"MCP workspaces {mcp_workspace_ids} should match API workspaces {api_workspace_ids}"
        )

        # Verify counts match
        assert mcp_result["total"] == api_response.page().metadata["total_results"], (
            f"MCP total {mcp_result['total']} should match API count {api_response.page().metadata['total_results']}"
        )

    @pytest.mark.asyncio
    async def test_list_workspaces_error_handling(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """
        Verify tool handles connection errors gracefully.

        Mounts a controlled failing tool to test error handling.
        """
        import json

        import nhx.core.mcp.server as mcp_server_module
        from nhx.common.mcp import format_error_response

        def create_failing_entities_mcp(_base_url: str | None = None) -> FastMCP:
            server = FastMCP("Failing Entities Service")

            @server.tool(description="List workspaces in the NeMo Helix")
            async def list_workspaces() -> dict[str, object]:
                try:
                    raise RuntimeError("platform unavailable")
                except Exception as e:
                    return format_error_response(e)

            return server

        monkeypatch.setattr(mcp_server_module, "create_entities_mcp", create_failing_entities_mcp)
        bad_server = mcp_server_module.create_server("http://unused.example.com")

        # Execute tool - should return error, not raise
        tool_result = await bad_server.call_tool("list_workspaces", {})

        result = json.loads(_text_content(tool_result))

        assert result == {
            "success": False,
            "error": {
                "code": "RuntimeError",
                "message": "platform unavailable",
                "hint": "Check the MCP server logs for details, then retry after fixing the request or platform state.",
                "retryable": False,
            },
        }
