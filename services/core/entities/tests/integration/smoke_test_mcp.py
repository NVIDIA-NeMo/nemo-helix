# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integration smoke test for the entities service MCP server.

Creates and tests an MCP server instance that is connected to a running NeMo Helix instance.
"""

from __future__ import annotations

import os
from typing import Any, Generator

import pytest
from fastmcp import FastMCP
from mcp.types import TextContent
from nemo_helix import NeMoHelix
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nhx.common.sdk_factory import get_platform_sdk
from nhx.core.entities.mcp.server import create_server


@pytest.fixture(scope="module")
def nhx_base_url() -> str:
    """Get NeMo Helix base URL from environment or use default."""
    return os.environ.get("NHX_BASE_URL", "http://localhost:8080")


@pytest.fixture(scope="module")
def nemo_sdk(nhx_base_url: str) -> Generator[NeMoHelix, None, None]:
    """Create NeMo SDK client for direct API validation."""
    os.environ.setdefault("NHX_BASE_URL", nhx_base_url)
    client = get_platform_sdk()
    yield client


@pytest.fixture(scope="module")
def mcp_server(nhx_base_url: str) -> Generator[FastMCP, None, None]:
    """Create entities MCP server instance."""
    server = create_server(nhx_base_url)
    yield server


def _text_content(tool_result: Any) -> str:
    first_content = tool_result.content[0]
    assert isinstance(first_content, TextContent)
    return first_content.text


class TestEntitiesMCPServerSmoke:
    """Smoke tests for entities MCP server basic functionality."""

    def test_nhx_connection(self, nemo_sdk: NeMoHelix) -> None:
        """Verify we can connect to NeMo Helix instance."""
        response = client_from_platform(nemo_sdk, WorkspacesClient).list_workspaces()
        assert response is not None
        assert response.page().items is not None

    def test_mcp_server_created(self, mcp_server: FastMCP) -> None:
        """Verify MCP server instance is created."""
        assert mcp_server is not None

    @pytest.mark.asyncio
    async def test_tools_registered(self, mcp_server: FastMCP) -> None:
        """Verify all expected tools are registered."""
        tools = await mcp_server.list_tools()
        tool_names = [t.name for t in tools]
        assert "list_workspaces" in tool_names
        assert "create_workspace" in tool_names
        assert "delete_workspace" in tool_names

    @pytest.mark.asyncio
    async def test_list_workspaces_matches_sdk(self, mcp_server: FastMCP, nemo_sdk: NeMoHelix) -> None:
        """
        Verify list_workspaces MCP tool returns consistent data with SDK.

        Ensures the MCP server is properly connected to NeMo Helix and returning real data.
        """
        import json

        # Get workspaces via SDK
        sdk_response = client_from_platform(nemo_sdk, WorkspacesClient).list_workspaces()
        sdk_workspace_ids = {ws.id for ws in sdk_response.items()}

        # Get workspaces via MCP tool
        tool_result = await mcp_server.call_tool("list_workspaces", {})

        mcp_result = json.loads(_text_content(tool_result))
        assert isinstance(mcp_result, dict)  # Type narrowing for ty
        assert mcp_result["success"] is True

        # Verify MCP returns same workspace IDs
        mcp_workspace_ids = {ws["id"] for ws in mcp_result["workspaces"]}
        assert sdk_workspace_ids == mcp_workspace_ids, (
            f"MCP workspaces {mcp_workspace_ids} should match SDK workspaces {sdk_workspace_ids}"
        )

        # Verify counts match
        assert mcp_result["total"] == sdk_response.page().metadata["total_results"], (
            f"MCP total {mcp_result['total']} should match SDK count {sdk_response.page().metadata['total_results']}"
        )

    @pytest.mark.asyncio
    async def test_list_workspaces_error_handling(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """
        Verify list_workspaces handles connection errors gracefully.

        Creates a server with a controlled failing SDK client to test error handling.
        """
        import json

        class FailingWorkspacesClient:
            def list_workspaces(self, *args: object, **kwargs: object) -> object:
                raise RuntimeError("platform unavailable")

        monkeypatch.setattr(
            "nhx.core.entities.mcp.server.client_from_platform",
            lambda sdk, client_cls: FailingWorkspacesClient(),
        )
        bad_server = create_server("http://unused.example.com")
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
