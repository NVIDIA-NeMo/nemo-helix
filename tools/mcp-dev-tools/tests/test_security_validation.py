#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test security validation for MCP dev tools.

Tests that path validation properly blocks attempts to escape the repository
or access unauthorized files/directories.
"""

import json
from pathlib import Path

import pytest

from nhx_dev_mcp import create_server


@pytest.fixture
def server():
    """Create MCP server instance for testing."""
    # Use repo root as working directory
    repo_root = Path(__file__).parent.parent.parent.parent.resolve()
    return create_server(str(repo_root))


class TestSharedPathValidation:
    """Test the shared validate_path function."""

    def test_blocks_absolute_paths(self, server):
        """Should block absolute paths."""
        # Access the validate_path function from within the server's create_server scope
        # Since it's a closure, we'll test via the actual tools that use it
        pass  # Tested via tool tests below

    def test_blocks_parent_directory_traversal(self, server):
        """Should block parent directory traversal."""
        pass  # Tested via tool tests below


class TestListDirectorySecurity:
    """Test list_directory security validation."""

    @pytest.mark.asyncio
    async def test_blocks_parent_directory_traversal(self, server):
        """Should block attempts to traverse to parent directories."""
        result = json.loads(
            (await server.call_tool("list_directory", {"path": "../../../"}))
            .content[0]
            .text
        )

        assert result["success"] is False
        assert "Parent directory references (..) not allowed" in result["error"]

    @pytest.mark.asyncio
    async def test_blocks_absolute_paths(self, server):
        """Should block absolute paths."""
        result = json.loads(
            (await server.call_tool("list_directory", {"path": "/etc"})).content[0].text
        )

        assert result["success"] is False
        assert "Absolute paths not allowed" in result["error"]

    @pytest.mark.asyncio
    async def test_allows_valid_relative_paths(self, server):
        """Should allow valid relative paths within repo."""
        result = json.loads(
            (await server.call_tool("list_directory", {"path": "services"}))
            .content[0]
            .text
        )

        assert result["success"] is True


class TestFindFilesSecurity:
    """Test find_files security validation."""

    @pytest.mark.asyncio
    async def test_blocks_parent_directory_in_path(self, server):
        """Should block parent directory references in path."""
        result = json.loads(
            (
                await server.call_tool(
                    "find_files", {"pattern": "*.py", "path": "../../etc"}
                )
            )
            .content[0]
            .text
        )

        assert result["success"] is False
        assert "Parent directory references (..) not allowed" in result["error"]

    @pytest.mark.asyncio
    async def test_blocks_absolute_path(self, server):
        """Should block absolute paths."""
        result = json.loads(
            (await server.call_tool("find_files", {"pattern": "*.py", "path": "/etc"}))
            .content[0]
            .text
        )

        assert result["success"] is False
        assert "Absolute paths not allowed" in result["error"]

    @pytest.mark.asyncio
    async def test_allows_valid_find_operations(self, server):
        """Should allow valid find operations."""
        result = json.loads(
            (
                await server.call_tool(
                    "find_files",
                    {
                        "pattern": "nhx_dev_mcp.py",
                        "path": "tools/mcp-dev-tools",
                        "file_type": "f",
                    },
                )
            )
            .content[0]
            .text
        )

        assert result["success"] is True
        assert "nhx_dev_mcp.py" in result["stdout"]


class TestRunPytestSecurity:
    """Test run_pytest security validation."""

    @pytest.mark.asyncio
    async def test_blocks_parent_directory_traversal(self, server):
        """Should block attempts to traverse to parent directories."""
        result = json.loads(
            (await server.call_tool("run_pytest", {"path": "../../../"}))
            .content[0]
            .text
        )

        assert result["success"] is False
        assert "Parent directory references (..) not allowed" in result["error"]

    @pytest.mark.asyncio
    async def test_blocks_absolute_paths(self, server):
        """Should block absolute paths."""
        result = json.loads(
            (await server.call_tool("run_pytest", {"path": "/tmp/tests"}))
            .content[0]
            .text
        )

        assert result["success"] is False
        assert "Absolute paths not allowed" in result["error"]
