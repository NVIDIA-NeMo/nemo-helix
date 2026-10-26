# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime status and tools of the MCP servers an agent declares.

An agent that declares a stdio MCP server degrades silently when the server
never spawns: the harness simply gets fewer tools. The models here describe,
per declared server, whether the runtime can actually reach it and which tools
it exposes, and :func:`probe_mcp_servers` produces that answer *from inside the
runtime process* so the ``PATH`` a stdio command is resolved against is the one
the subprocess would really inherit.

Nothing here reports host filesystem layout. Whether a stdio command resolved
is a boolean, never a path; messages are one short sentence and never quote the
``PATH`` that was searched; and any absolute path that still reaches a message
is reduced to its final component on the way out. A spawned server's stderr is
never included in the response — the declared env it runs with can hold
secrets that a crashing process might echo back — so a launch failure is
reported as a short generic sentence, with the captured stderr logged
server-side only, for operators to read from logs rather than the API.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
import re
import shutil
import tempfile
import time
from collections.abc import AsyncGenerator, Callable, Collection
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import IO, Any, Literal

from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp.types import Implementation, Tool
from nemo_agents_plugin.agent_config import AgentConfig, McpServerConfig
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Transport spellings the Fabric adapters normalize to a stdio subprocess.
_STDIO_TRANSPORTS = frozenset({"stdio", "command", "process"})
# Transport spellings that address a server over the network. An empty
# transport means streamable HTTP, matching adapter normalization.
_REMOTE_TRANSPORTS = frozenset({"", "http", "streamable_http", "streamablehttp", "sse", "websocket"})

DEFAULT_PROBE_TIMEOUT_SECONDS = 10.0
# How long a status check reuses a server's last probe, and how long any caller reuses a failed one,
# before launching the server again.
MCP_STATUS_CACHE_SECONDS = 30.0

# Bytes of captured stderr kept for a failed launch. Enough for a traceback's
# last frames without turning the status response into a log dump.
_STDERR_SNIPPET_BYTES = 2000

McpServerState = Literal[
    "running",
    "unresolved",
    "spawn_failed",
    "unreachable",
    "not_started",
    "unknown",
]


class McpTool(BaseModel):
    """One tool an MCP server advertised through ``tools/list``."""

    name: str
    title: str | None = Field(default=None, description="Human-readable display name, when the server sets one.")
    description: str | None = None
    input_schema: dict[str, Any] = Field(default_factory=dict, description="JSON Schema for the tool's arguments.")
    enabled: bool = Field(
        default=True,
        description="False when the agent's allowed/blocked tool policy hides this tool from the model.",
    )


class McpServerInfo(BaseModel):
    """The name and version a server reported when its MCP session initialized."""

    name: str
    version: str


class McpServerStatus(BaseModel):
    """Runtime state of one MCP server an agent declares."""

    name: str = Field(description="Server name as declared under mcp.servers in the agent config.")
    transport: str = Field(description="Transport as declared (e.g. 'stdio', 'streamable_http').")
    target: str = Field(description="Command to spawn for stdio servers, or the endpoint URL for remote ones.")
    args: list[str] = Field(default_factory=list, description="Arguments passed to a stdio command.")
    exposure: str = Field(default="", description="'harness_native' or 'fabric_managed'.")
    state: McpServerState = Field(
        description=(
            "running: a probe from the runtime process started (stdio) or connected to (remote) the server, "
            "initialized an MCP session, and listed its tools. The probe is its own launch or connection, "
            "not the harness's live one, so this says the server can come up, not that the harness has it. "
            "unresolved: stdio command not found on the runtime PATH. "
            "spawn_failed: command resolved but the server died or never initialized. "
            "unreachable: remote server did not answer. not_started: nothing is running to check against. "
            "unknown: transport not understood, so no check was made."
        )
    )
    command_resolved: bool = Field(
        default=False,
        description=(
            "Whether a stdio command was found on the runtime PATH. Always false for remote "
            "transports, which have no command to resolve. The path it resolved to is "
            "deliberately not reported."
        ),
    )
    detail: str = Field(default="", description="Human-readable explanation of the state.")
    error: str = Field(default="", description="Short, generic error message when the server failed to start.")
    server_info: McpServerInfo | None = Field(
        default=None, description="Name and version the server reported, when it started."
    )
    latency_ms: int | None = Field(
        default=None,
        description=(
            "Milliseconds the probe took to launch or connect, initialize, and list tools; null when no "
            "connection was attempted."
        ),
    )
    checked_at: datetime | None = Field(
        default=None, description="When this server's probe finished; null when it was never probed."
    )
    tools: list[McpTool] = Field(default_factory=list, description="Tools the server advertised, when it started.")


class McpStatusResponse(BaseModel):
    """A runtime's answer to "are your MCP servers up, and what do they expose?"."""

    runtime_instance_id: str = Field(default="", description="Id of the runtime process that ran the check.")
    checked_at: datetime = Field(
        description=(
            "When the oldest server result in this response was checked, so every server is at least this "
            "fresh. A runtime reuses each server's probe (see the endpoint), so this can predate the request."
        )
    )
    servers: list[McpServerStatus] = Field(default_factory=list, description="One entry per declared MCP server.")


def declared_mcp_servers(config: AgentConfig) -> list[McpServerStatus]:
    """Return every declared server as a ``not_started`` status, in name order."""
    servers = config.mcp.servers if config.mcp is not None else {}
    return [
        _base_status(
            name,
            server,
            state="not_started",
            detail="Agent is not deployed.",
        )
        for name, server in sorted(servers.items())
    ]


async def probe_mcp_servers(
    config: AgentConfig,
    *,
    timeout: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
) -> list[McpServerStatus]:
    """Check every declared MCP server against the current process environment."""
    servers = config.mcp.servers if config.mcp is not None else {}
    blocked = _blocked_tools(config)
    ordered = sorted(servers.items())
    results = await asyncio.gather(*(probe_mcp_server(name, server, blocked, timeout) for name, server in ordered))
    return list(results)


class McpStatusCache:
    """Reuse recent probes so polling a runtime does not respawn every stdio server.

    Each probe launches a fresh copy of a stdio server or opens a session with a remote
    one. Results are kept per server: a caller says how old a running server's result it
    will accept, while a failed result is retried after ``failure_ttl`` whatever the caller
    asks. Concurrent callers wait on the one probe pass in flight.
    """

    def __init__(
        self,
        config: AgentConfig,
        *,
        failure_ttl: float = MCP_STATUS_CACHE_SECONDS,
        timeout: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._servers = sorted(config.mcp.servers.items()) if config.mcp is not None else []
        self._blocked = _blocked_tools(config)
        self._failure_ttl = failure_ttl
        self._timeout = timeout
        self._clock = clock
        self._lock = asyncio.Lock()
        # Server name -> (status, clock reading when its probe finished).
        self._entries: dict[str, tuple[McpServerStatus, float]] = {}

    async def get(self, *, max_age: float | None) -> tuple[list[McpServerStatus], datetime]:
        """Return every declared server's status, probing those whose result is older than *max_age*.

        ``max_age=None`` accepts a running server's result of any age; ``0`` probes everything.
        The returned time is when the oldest result was checked.
        """
        arrived = self._clock()
        async with self._lock:
            now = self._clock()
            stale = [
                (name, server) for name, server in self._servers if not self._is_fresh(name, arrived, now, max_age)
            ]
            results = await asyncio.gather(
                *(probe_mcp_server(name, server, self._blocked, self._timeout) for name, server in stale)
            )
            finished = self._clock()
            for status in results:
                self._entries[status.name] = (status, finished)
            statuses = [self._entries[name][0] for name, _ in self._servers]
        checked = [status.checked_at for status in statuses if status.checked_at is not None]
        return statuses, min(checked, default=datetime.now(UTC))

    def _is_fresh(self, name: str, arrived: float, now: float, max_age: float | None) -> bool:
        entry = self._entries.get(name)
        if entry is None:
            return False
        status, finished = entry
        # A probe that finished after this caller arrived ran while it waited on the lock, so share it.
        if finished > arrived:
            return True
        if status.state == "running":
            return max_age is None or now - finished < max_age
        limit = self._failure_ttl if max_age is None else min(self._failure_ttl, max_age)
        return now - finished < limit


async def probe_mcp_server(
    name: str,
    server: McpServerConfig,
    blocked: Collection[str] = (),
    timeout: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
) -> McpServerStatus:
    """Probe one declared server, marking tools hidden by its policy or *blocked* as disabled."""
    transport = _normalize_transport(server.transport)
    if transport in _STDIO_TRANSPORTS:
        status = await _probe_stdio_server(name, server, timeout)
    elif transport in _REMOTE_TRANSPORTS:
        status = await _probe_remote_server(name, server, timeout)
    else:
        status = _base_status(
            name,
            server,
            state="unknown",
            detail=f"Unsupported transport '{server.transport}'.",
        )
    tools = [tool.model_copy(update={"enabled": _is_enabled(server, blocked, tool.name)}) for tool in status.tools]
    return status.model_copy(update={"tools": tools, "checked_at": datetime.now(UTC)})


async def _probe_stdio_server(name: str, server: McpServerConfig, timeout: float) -> McpServerStatus:
    command = os.path.expandvars(server.url).strip()
    env = {**_default_environment(), **server.env}
    if not command:
        return _base_status(
            name,
            server,
            state="unresolved",
            detail="No command configured.",
        )

    if shutil.which(command, path=env.get("PATH")) is None:
        return _base_status(
            name,
            server,
            state="unresolved",
            detail="Command not found.",
        )

    started = time.monotonic()
    session, error = await _start_stdio_server(command, server.args, env, timeout)
    latency_ms = _elapsed_ms(started)
    if session is None:
        return _base_status(
            name,
            server,
            state="spawn_failed",
            command_resolved=True,
            detail="Server did not start.",
            error=error,
            latency_ms=latency_ms,
        )
    server_info, tools = session
    return _base_status(
        name,
        server,
        state="running",
        command_resolved=True,
        detail=f"Started with {len(tools)} tool(s).",
        server_info=server_info,
        latency_ms=latency_ms,
        tools=tools,
    )


async def _start_stdio_server(
    command: str,
    args: list[str],
    env: dict[str, str],
    timeout: float,
) -> tuple[tuple[McpServerInfo, list[McpTool]] | None, str]:
    """Spawn the server, initialize a session, and list its tools."""
    from mcp import StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command=command, args=list(args), env=env)
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as errlog:
        try:
            async with asyncio.timeout(timeout):
                async with stdio_client(params, errlog=errlog) as (read, write):
                    return await _list_tools(read, write), ""
        except TimeoutError:
            logger.debug("MCP stdio probe of '%s' timed out: %s", command, _stderr_tail(errlog))
            return None, f"Server did not initialize within {timeout:g}s."
        except Exception:  # noqa: BLE001 — any launch failure is reportable status, not a 500.
            logger.debug("MCP stdio probe of '%s' failed: %s", command, _stderr_tail(errlog), exc_info=True)
            return None, "Server exited before initializing."


async def _probe_remote_server(name: str, server: McpServerConfig, timeout: float) -> McpServerStatus:
    """Initialize an MCP session with a remote server and list its tools.

    A plain HTTP request is not a usable health check: a healthy streamable HTTP
    server rejects a bare GET (406, or 400 without a session), an SSE endpoint
    never finishes its response, and httpx cannot dial a websocket at all. So the
    probe speaks MCP over the declared transport, as the harness would.
    """
    import httpx

    url = os.path.expandvars(server.url).strip()
    if not url:
        return _base_status(
            name,
            server,
            state="unreachable",
            detail="No URL configured.",
        )
    transport = _normalize_transport(server.transport)
    if transport == "websocket" and importlib.util.find_spec("websockets") is None:
        return _base_status(
            name,
            server,
            state="unknown",
            detail="Websocket transport is not installed in this runtime.",
        )
    # Header values may reference env vars, which the Fabric adapters expand before connecting.
    headers = {key: os.path.expandvars(value) for key, value in server.custom_headers.items()}
    started = time.monotonic()
    try:
        async with asyncio.timeout(timeout):
            async with _remote_streams(transport, url, headers, timeout) as (read, write):
                server_info, tools = await _list_tools(read, write)
    except TimeoutError:
        logger.debug("MCP remote probe of '%s' timed out", server.url, exc_info=True)
        return _base_status(
            name,
            server,
            state="unreachable",
            detail="Server did not respond.",
            error=f"Server did not initialize within {timeout:g}s.",
            latency_ms=_elapsed_ms(started),
        )
    except Exception as exc:  # noqa: BLE001 — connection failures are reportable status.
        logger.debug("MCP remote probe of '%s' failed", server.url, exc_info=True)
        cause = _root_cause(exc)
        if isinstance(cause, httpx.HTTPStatusError):
            return _base_status(
                name,
                server,
                state="unreachable",
                detail=f"Responded with HTTP {cause.response.status_code}.",
                latency_ms=_elapsed_ms(started),
            )
        return _base_status(
            name,
            server,
            state="unreachable",
            detail="Server did not respond.",
            error=f"{type(cause).__name__}.",
            latency_ms=_elapsed_ms(started),
        )
    return _base_status(
        name,
        server,
        state="running",
        detail=f"Connected with {len(tools)} tool(s).",
        server_info=server_info,
        latency_ms=_elapsed_ms(started),
        tools=tools,
    )


@asynccontextmanager
async def _remote_streams(
    transport: str,
    url: str,
    headers: dict[str, str],
    timeout: float,
) -> AsyncGenerator[tuple[MemoryObjectReceiveStream, MemoryObjectSendStream]]:
    """Open the MCP client streams for a remote transport, yielding ``(read, write)``."""
    if transport == "sse":
        from mcp.client.sse import sse_client

        async with sse_client(url, headers=headers or None, timeout=timeout) as (read, write):
            yield read, write
    elif transport == "websocket":
        # The websocket client takes no headers; the Fabric adapters do not send them either.
        from mcp.client.websocket import websocket_client

        async with websocket_client(url) as (read, write):
            yield read, write
    else:
        import httpx
        from mcp.client.streamable_http import streamable_http_client

        # Redirects followed, as the MCP SDK's own client factory does.
        async with httpx.AsyncClient(headers=headers, timeout=timeout, follow_redirects=True) as http_client:
            async with streamable_http_client(url, http_client=http_client) as (read, write, _):
                yield read, write


async def _list_tools(
    read: MemoryObjectReceiveStream, write: MemoryObjectSendStream
) -> tuple[McpServerInfo, list[McpTool]]:
    """Initialize an MCP session over open streams and return the server's identity and every tool it lists."""
    from mcp import ClientSession
    from mcp.types import PaginatedRequestParams

    async with ClientSession(read, write) as session:
        initialized = await session.initialize()
        tools: list[Tool] = []
        cursor: str | None = None
        while True:
            page = await session.list_tools(params=PaginatedRequestParams(cursor=cursor) if cursor else None)
            tools.extend(page.tools)
            cursor = page.nextCursor
            if not cursor:
                return _server_info(initialized.serverInfo), [_tool(tool) for tool in tools]


def _server_info(info: Implementation) -> McpServerInfo:
    return McpServerInfo(name=info.name, version=info.version)


def _tool(tool: Tool) -> McpTool:
    return McpTool(name=tool.name, title=tool.title, description=tool.description, input_schema=tool.inputSchema)


def _is_enabled(server: McpServerConfig, blocked: Collection[str], tool_name: str) -> bool:
    if server.allowed_tools is not None and tool_name not in server.allowed_tools:
        return False
    return tool_name not in server.blocked_tools and tool_name not in blocked


def _blocked_tools(config: AgentConfig) -> frozenset[str]:
    return frozenset(config.tools.blocked) if config.tools is not None else frozenset()


def _elapsed_ms(started: float) -> int:
    return round((time.monotonic() - started) * 1000)


def _root_cause(exc: BaseException) -> BaseException:
    """Unwrap the exception groups the MCP clients' task groups raise to the first leaf error."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return exc


def _base_status(
    name: str,
    server: McpServerConfig,
    *,
    state: McpServerState,
    command_resolved: bool = False,
    detail: str = "",
    error: str = "",
    server_info: McpServerInfo | None = None,
    latency_ms: int | None = None,
    tools: list[McpTool] | None = None,
) -> McpServerStatus:
    return McpServerStatus(
        name=name,
        transport=server.transport,
        target=_redact_paths(server.url),
        args=list(server.args),
        exposure=server.exposure,
        state=state,
        command_resolved=command_resolved,
        detail=_redact_paths(detail),
        error=_redact_paths(error),
        server_info=server_info,
        latency_ms=latency_ms,
        tools=tools or [],
    )


# An absolute POSIX path, not preceded by a word character, ':' or '/' so that
# URL paths ("http://host/mcp") and relative fragments are left alone.
_ABSOLUTE_PATH_RE = re.compile(r"(?<![\w.:/])/(?:[^\s'\"()\\,;]+/)+([^\s'\"()\\,;]*)")


def _redact_paths(text: str) -> str:
    """Reduce absolute host paths in *text* to their final component."""
    return _ABSOLUTE_PATH_RE.sub(lambda match: f".../{match.group(1)}" if match.group(1) else ".../", text)


def _normalize_transport(transport: str) -> str:
    return transport.strip().lower().replace("-", "_")


def _default_environment() -> dict[str, str]:
    """The env an MCP stdio subprocess inherits, per the mcp SDK's allowlist."""
    from mcp.client.stdio import get_default_environment

    return get_default_environment()


def _stderr_tail(errlog: IO[str]) -> str:
    """Read the tail of the server's captured stderr, for server-side logs only."""
    try:
        errlog.flush()
        errlog.seek(0)
        return errlog.read()[-_STDERR_SNIPPET_BYTES:].strip()
    except (OSError, ValueError):
        return ""


def config_from_dict(config: dict) -> AgentConfig | None:
    """Load an agent config dict, returning None when it is not a v1 agent spec."""
    try:
        return AgentConfig.model_validate(config)
    except Exception:  # noqa: BLE001 — a NAT-format or malformed config simply declares no MCP servers here.
        return None
