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
import shutil
import tempfile
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any, Literal, TextIO

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from nemo_agents_plugin.agent_config import AgentConfig, McpServerConfig
from pydantic import BaseModel, Field

PROBE_TIMEOUT_SECONDS = 15.0
_STDERR_TAIL_CHARS = 2000
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
    transport: str, target: str, spec: McpServerConfig, errlog: TextIO
) -> AsyncGenerator[tuple[Any, Any]]:
    if transport == "stdio":
        params = StdioServerParameters(command=target, args=spec.args, env=spec.env or None)
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


async def probe_mcp_server(
    name: str,
    spec: McpServerConfig,
    config: AgentConfig,
    *,
    timeout_seconds: float = PROBE_TIMEOUT_SECONDS,
) -> McpServerStatus:
    """Connect to one declared server and list its tools."""
    transport = _normalize_transport(spec.transport)
    target = os.path.expandvars(spec.url).strip()
    status = McpServerStatus(name=name, transport=transport, target=target, args=spec.args, state="failed")

    if transport not in {"stdio", "sse", "streamable_http"}:
        status.state = "unsupported"
        status.error = f"Status probing does not support the '{transport}' transport."
        return status

    if transport == "stdio":
        # Resolve against the PATH the server inherits, but keep host paths out of the response.
        search_path = spec.env.get("PATH", os.environ.get("PATH", ""))
        if not target or shutil.which(target, path=search_path) is None:
            status.state = "unresolved"
            status.error = f"Command '{target}' was not found on the runtime PATH."
            return status

    with tempfile.TemporaryFile(mode="w+") as errlog:
        try:
            with anyio.fail_after(timeout_seconds):
                async with _open_streams(transport, target, spec, errlog) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        tools = (await session.list_tools()).tools
        except TimeoutError:
            status.error = f"No response from the MCP server within {timeout_seconds:g}s."
            status.stderr = _read_tail(errlog)
            return status
        except Exception as error:
            status.error = _describe_error(error)
            status.stderr = _read_tail(errlog)
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
