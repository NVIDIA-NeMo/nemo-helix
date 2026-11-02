# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""On-demand status probe for the MCP servers an agent config declares.

The Fabric runtime spawns MCP servers lazily on a chat turn, and a server that fails to
launch only degrades the agent. This probe connects to every declared server from the
serving process, so stdio commands resolve against the same runtime ``PATH`` the agent
inherits (host venv for subprocess deployments, the image for docker and k8s).
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import tempfile
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any, Literal, TextIO

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import PaginatedRequestParams, Tool
from nemo_agents_plugin.agent_config import AgentConfig, McpServerConfig
from pydantic import BaseModel, Field

PROBE_TIMEOUT_SECONDS = 15.0
_STDERR_TAIL_CHARS = 2000
_BRACED_VAR = re.compile(r"\$\{([^}]+)\}")
_STDIO_TRANSPORTS = {"stdio", "command", "process"}
_STREAMABLE_HTTP_TRANSPORTS = {"", "http", "streamable_http", "streamablehttp"}

McpServerState = Literal["running", "unresolved", "failed", "unsupported"]


class McpToolStatus(BaseModel):
    name: str
    description: str | None = None
    input_schema: dict[str, Any] = Field(default_factory=dict, description="JSON Schema for the tool's arguments.")
    exposed: bool = Field(description="False when the agent's tool policy hides this tool from the model.")


class McpServerStatus(BaseModel):
    name: str
    transport: str
    target: str = Field(description="Command for stdio servers, URL otherwise.")
    args: list[str] = Field(default_factory=list)
    state: McpServerState
    tools: list[McpToolStatus] = Field(default_factory=list)
    error: str | None = None
    stderr: str | None = None


class McpStatusResponse(BaseModel):
    servers: list[McpServerStatus]
    checked_at: datetime


def _normalize_transport(transport: str) -> str:
    normalized = transport.strip().lower().replace("-", "_")
    if normalized in _STDIO_TRANSPORTS:
        return "stdio"
    if normalized in _STREAMABLE_HTTP_TRANSPORTS:
        return "streamable_http"
    return normalized


def _describe_error(error: BaseException) -> str:
    while isinstance(error, BaseExceptionGroup) and error.exceptions:
        error = error.exceptions[0]
    return str(error) or type(error).__name__


def _expand_braced(value: str) -> str:
    return _BRACED_VAR.sub(lambda match: os.environ.get(match.group(1), match.group(0)), value)


def _read_tail(errlog: TextIO) -> str | None:
    errlog.seek(0)
    tail = errlog.read()[-_STDERR_TAIL_CHARS:].strip()
    return tail or None


def _is_exposed(tool: str, spec: McpServerConfig, config: AgentConfig) -> bool:
    globally_blocked = config.tools.blocked if config.tools is not None else []
    if tool in spec.blocked_tools or tool in globally_blocked:
        return False
    return spec.allowed_tools is None or tool in spec.allowed_tools


@asynccontextmanager
async def _open_streams(
    transport: str, target: str, spec: McpServerConfig, env: dict[str, str], errlog: TextIO
) -> AsyncGenerator[tuple[Any, Any]]:
    if transport == "stdio":
        params = StdioServerParameters(command=target, args=spec.args, env=env or None)
        async with stdio_client(params, errlog=errlog) as streams:
            yield streams
        return
    headers = {name: os.path.expandvars(value) for name, value in spec.custom_headers.items()}
    if transport == "sse":
        async with sse_client(target, headers=headers) as streams:
            yield streams
        return
    async with streamablehttp_client(target, headers=headers) as (read, write, _):
        yield read, write


async def _list_all_tools(session: ClientSession) -> list[Tool]:
    tools: list[Tool] = []
    cursor: str | None = None
    while True:
        page = await session.list_tools(params=PaginatedRequestParams(cursor=cursor))
        if page.nextCursor and page.nextCursor == cursor:
            raise RuntimeError("MCP server repeated its tool-list pagination cursor.")
        tools.extend(page.tools)
        cursor = page.nextCursor
        if not cursor:
            return tools


async def probe_mcp_server(
    name: str,
    spec: McpServerConfig,
    config: AgentConfig,
    *,
    timeout_seconds: float = PROBE_TIMEOUT_SECONDS,
) -> McpServerStatus:
    """Connect to one declared server and list its tools."""
    transport = _normalize_transport(spec.transport)
    raw_target = spec.url.strip()
    target = os.path.expandvars(raw_target)
    # Report the unexpanded value so credentials substituted from the environment never leave the process.
    status = McpServerStatus(name=name, transport=transport, target=raw_target, args=spec.args, state="failed")
    raw_env = {**config.environment.env, **spec.env}
    # Match the runtime harnesses, which expand `${VAR}` env references from the process env.
    env = {key: _expand_braced(value) for key, value in raw_env.items()}
    pairs = [
        (raw_target, target),
        *((raw_env[key], env[key]) for key in env),
        *((header, os.path.expandvars(header)) for header in spec.custom_headers.values()),
    ]
    # Map every environment-substituted value back to its reference for redaction.
    expanded = {new: old for old, new in pairs if new and new != old}

    def redact(text: str | None) -> str | None:
        for value in sorted(expanded, key=len, reverse=True):
            text = text.replace(value, expanded[value]) if text else text
        return text

    if transport not in {"stdio", "sse", "streamable_http"}:
        status.state = "unsupported"
        status.error = f"Status probing does not support the '{transport}' transport."
        return status

    if transport == "stdio":
        # Resolve against the PATH the server inherits, but keep host paths out of the response.
        search_path = env.get("PATH", os.environ.get("PATH", ""))
        if not target or shutil.which(target, path=search_path) is None:
            status.state = "unresolved"
            status.error = f"Command '{target}' was not found on the runtime PATH."
            return status

    with tempfile.TemporaryFile(mode="w+") as errlog:
        try:
            with anyio.fail_after(timeout_seconds):
                async with _open_streams(transport, target, spec, env, errlog) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        tools = await _list_all_tools(session)
        except TimeoutError:
            status.error = f"No response from the MCP server within {timeout_seconds:g}s."
            status.stderr = redact(_read_tail(errlog))
            return status
        except Exception as error:
            status.error = redact(_describe_error(error))
            status.stderr = redact(_read_tail(errlog))
            return status

    status.state = "running"
    status.tools = [
        McpToolStatus(
            name=tool.name,
            description=tool.description,
            input_schema=tool.inputSchema,
            exposed=_is_exposed(tool.name, spec, config),
        )
        for tool in tools
    ]
    return status


async def probe_mcp_servers(config: AgentConfig) -> McpStatusResponse:
    """Probe every MCP server the config declares, concurrently."""
    servers = config.mcp.servers if config.mcp is not None else {}
    statuses = await asyncio.gather(*(probe_mcp_server(name, spec, config) for name, spec in servers.items()))
    return McpStatusResponse(servers=list(statuses), checked_at=datetime.now(UTC))


class McpStatusCache:
    """Single-flight probe with a short TTL so overlapping requests share one set of probe processes."""

    def __init__(self, ttl_seconds: float = 3.0) -> None:
        self._ttl_seconds = ttl_seconds
        self._lock = asyncio.Lock()
        self._cached: tuple[float, McpStatusResponse] | None = None

    async def get(self, config: AgentConfig) -> McpStatusResponse:
        async with self._lock:
            if self._cached is not None and time.monotonic() - self._cached[0] < self._ttl_seconds:
                return self._cached[1]
            response = await probe_mcp_servers(config)
            self._cached = (time.monotonic(), response)
            return response
