# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""List the tools an agent's declared MCP servers expose, from inside its runtime.

Discovery runs here rather than on the platform because stdio servers only exist where the
agent runs. Every server is asked concurrently, each under its own timeout, so one slow or
broken server never hides the others. The agent config is fixed for a runtime's lifetime, so
successful listings are cached until shutdown; failures are cached briefly so a broken server
is not respawned on every request. Concurrent callers share one discovery pass.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Mapping
from contextlib import AsyncExitStack
from pathlib import Path

import anyio
import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import PaginatedRequestParams, Tool
from nemo_agents_plugin.agent_config import AgentConfig, McpServerConfig
from nemo_agents_plugin.schema import AgentMcpServerTools, AgentTool

logger = logging.getLogger(__name__)

DEFAULT_DISCOVERY_TIMEOUT_SECONDS = 20.0
DEFAULT_ERROR_RETRY_SECONDS = 30.0
_MAX_ERROR_LENGTH = 500


class McpToolCatalog:
    """Cached, single-flight ``tools/list`` across every MCP server an agent declares."""

    def __init__(
        self,
        config: AgentConfig,
        *,
        cwd: Path,
        timeout_seconds: float = DEFAULT_DISCOVERY_TIMEOUT_SECONDS,
        error_retry_seconds: float = DEFAULT_ERROR_RETRY_SECONDS,
    ) -> None:
        self._servers = dict(config.mcp.servers) if config.mcp is not None else {}
        self._blocked = frozenset(config.tools.blocked) if config.tools is not None else frozenset()
        self._cwd = cwd
        self._timeout_seconds = timeout_seconds
        self._error_retry_seconds = error_retry_seconds
        # Server name -> (listing, monotonic expiry or None for no expiry).
        self._cache: dict[str, tuple[AgentMcpServerTools, float | None]] = {}
        self._lock = asyncio.Lock()

    async def list_servers(self, *, refresh: bool = False) -> list[AgentMcpServerTools]:
        """Return every declared server's tools in declaration order, discovering what is stale."""
        async with self._lock:
            now = time.monotonic()
            stale = [name for name in self._servers if refresh or not self._is_fresh(name, now)]
            results = await asyncio.gather(*(self._discover(name) for name in stale))
            for result in results:
                expiry = None if result.status == "discovered" else time.monotonic() + self._error_retry_seconds
                self._cache[result.name] = (result, expiry)
            return [self._cache[name][0] for name in self._servers]

    def _is_fresh(self, name: str, now: float) -> bool:
        cached = self._cache.get(name)
        return cached is not None and (cached[1] is None or cached[1] > now)

    async def _discover(self, name: str) -> AgentMcpServerTools:
        server = self._servers[name]
        try:
            with anyio.fail_after(self._timeout_seconds):
                tools = await list_mcp_server_tools(server, cwd=self._cwd)
        except Exception as error:  # a broken server must not hide the others
            logger.warning("Failed to list tools for MCP server %r.", name, exc_info=True)
            return AgentMcpServerTools(
                name=name,
                transport=server.transport,
                status="error",
                error=_describe_error(error, timeout_seconds=self._timeout_seconds),
            )
        return AgentMcpServerTools(
            name=name,
            transport=server.transport,
            status="discovered",
            tools=[
                AgentTool(
                    name=tool.name,
                    title=tool.title,
                    description=tool.description,
                    input_schema=tool.inputSchema,
                    enabled=self._is_enabled(server, tool.name),
                )
                for tool in tools
            ],
        )

    def _is_enabled(self, server: McpServerConfig, tool_name: str) -> bool:
        if server.allowed_tools is not None and tool_name not in server.allowed_tools:
            return False
        return tool_name not in server.blocked_tools and tool_name not in self._blocked


async def list_mcp_server_tools(server: McpServerConfig, *, cwd: Path) -> list[Tool]:
    """Connect to one MCP server the way the harness adapters do and page through ``tools/list``."""
    transport = server.transport.strip().lower().replace("_", "-")
    target = os.path.expandvars(server.url).strip()
    async with AsyncExitStack() as stack:
        if transport == "stdio":
            params = StdioServerParameters(command=target, args=server.args, env=server.env or None, cwd=cwd)
            read, write = await stack.enter_async_context(stdio_client(params))
        elif transport == "sse":
            read, write = await stack.enter_async_context(
                sse_client(target, headers=_expand_headers(server.custom_headers))
            )
        elif transport in ("streamable-http", "http"):
            http_client = await stack.enter_async_context(
                httpx.AsyncClient(headers=_expand_headers(server.custom_headers), follow_redirects=True)
            )
            read, write, _ = await stack.enter_async_context(streamable_http_client(target, http_client=http_client))
        else:
            raise ValueError(f"Unsupported MCP transport {server.transport!r}.")

        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        tools: list[Tool] = []
        cursor: str | None = None
        while True:
            page = await session.list_tools(params=PaginatedRequestParams(cursor=cursor) if cursor else None)
            tools.extend(page.tools)
            cursor = page.nextCursor
            if not cursor:
                return tools


def _expand_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {name: os.path.expandvars(value) for name, value in headers.items()}


def _describe_error(error: BaseException, *, timeout_seconds: float) -> str:
    # anyio task groups wrap the real failure; surface the first leaf instead of the group.
    while isinstance(error, BaseExceptionGroup) and error.exceptions:
        error = error.exceptions[0]
    if isinstance(error, TimeoutError):
        return f"Timed out after {timeout_seconds:g}s listing tools."
    message = str(error) or type(error).__name__
    return message[:_MAX_ERROR_LENGTH]
