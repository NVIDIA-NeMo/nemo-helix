# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the MCP server status probe.

The stdio cases spawn real subprocesses — a working MCP server, a command that
exits immediately, and a command that is not on PATH — because the failure this
probe exists to catch is exactly a subprocess that does not come up.
"""

from __future__ import annotations

import asyncio
import os
import socket
import sys
import textwrap
import threading
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from nemo_agents_plugin import mcp_status
from nemo_agents_plugin.agent_config import AgentConfig, McpServerConfig
from nemo_agents_plugin.mcp_status import McpServerStatus, McpStatusCache, declared_mcp_servers, probe_mcp_servers

# A minimal stdio MCP server that advertises one tool.
_WORKING_SERVER = textwrap.dedent(
    """
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("probe-test")

    @server.tool()
    def extract_iocs(text: str) -> str:
        return text

    server.run()
    """
)

# A stdio MCP server whose tools exercise allowed/blocked policy and tool metadata.
_POLICY_SERVER = textwrap.dedent(
    '''
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("policy-test")

    @server.tool(title="Add numbers")
    def add(left: float, right: float) -> float:
        """Add two numbers."""
        return left + right

    @server.tool()
    def subtract(left: float, right: float) -> float:
        """Subtract right from left."""
        return left - right

    @server.tool()
    def wipe() -> str:
        """Pretend to delete everything."""
        return "gone"

    server.run()
    '''
)

_CRASHING_SERVER = "import sys; sys.stderr.write('ModuleNotFoundError: no such package\\n'); sys.exit(1)"


def _config(servers: dict[str, dict[str, Any]], *, blocked: list[str] | None = None) -> AgentConfig:
    return AgentConfig.model_validate(
        {
            "config_format": "nemo-agents-spec-v1",
            "name": "probe-agent",
            "default_harness": "hermes",
            "harnesses": {"hermes": {"kind": "hermes"}},
            "mcp": {"servers": servers},
            "tools": {"blocked": blocked or []},
        }
    )


class TestDeclaredMcpServers:
    def test_lists_every_declared_server_as_not_started(self) -> None:
        config = _config(
            {
                "iocs": {"transport": "stdio", "url": "email-security-triage-iocs"},
                "docs": {"transport": "streamable_http", "url": "https://example.invalid/mcp"},
            }
        )

        statuses = declared_mcp_servers(config)

        assert [s.name for s in statuses] == ["docs", "iocs"]
        assert {s.state for s in statuses} == {"not_started"}
        assert statuses[1].target == "email-security-triage-iocs"

    def test_no_mcp_section_declares_nothing(self) -> None:
        config = AgentConfig.model_validate(
            {
                "config_format": "nemo-agents-spec-v1",
                "name": "probe-agent",
                "default_harness": "hermes",
                "harnesses": {"hermes": {"kind": "hermes"}},
            }
        )

        assert declared_mcp_servers(config) == []


class TestProbeStdioServers:
    async def test_command_not_on_path_is_unresolved(self) -> None:
        config = _config({"iocs": {"transport": "stdio", "url": "nemo-mcp-server-that-is-not-installed"}})

        [status] = await probe_mcp_servers(config, timeout=5)

        assert status.state == "unresolved"
        assert status.command_resolved is False
        assert status.detail == "Command not found."
        # The PATH the probe searched is host layout, not status.
        assert os.environ["PATH"] not in status.detail

    async def test_resolved_command_that_dies_is_spawn_failed_with_generic_error(self) -> None:
        config = _config(
            {
                "iocs": {
                    "transport": "stdio",
                    "url": sys.executable,
                    "args": ["-c", _CRASHING_SERVER],
                }
            }
        )

        [status] = await probe_mcp_servers(config, timeout=15)

        assert status.state == "spawn_failed"
        assert status.command_resolved is True
        assert status.error == "Server exited before initializing."
        assert "ModuleNotFoundError" not in status.error

    async def test_working_server_is_running_and_lists_its_tools(self, tmp_path: Path) -> None:
        script = tmp_path / "server.py"
        script.write_text(_WORKING_SERVER, encoding="utf-8")
        config = _config(
            {
                "iocs": {
                    "transport": "stdio",
                    "url": sys.executable,
                    "args": [str(script)],
                }
            }
        )

        [status] = await probe_mcp_servers(config, timeout=30)

        assert status.state == "running", status.error
        assert status.command_resolved is True
        assert [tool.name for tool in status.tools] == ["extract_iocs"]

    async def test_running_server_reports_tool_details_policy_and_identity(self, tmp_path: Path) -> None:
        script = tmp_path / "server.py"
        script.write_text(_POLICY_SERVER, encoding="utf-8")
        server = {"transport": "stdio", "url": sys.executable, "args": [str(script)]}
        config = _config(
            {
                "allowlisted": {**server, "allowed_tools": ["add", "wipe"]},
                "blocklisted": {**server, "blocked_tools": ["subtract"]},
            },
            blocked=["wipe"],
        )

        allowlisted, blocklisted = await probe_mcp_servers(config, timeout=30)

        assert {tool.name: tool.enabled for tool in allowlisted.tools} == {
            "add": True,
            "subtract": False,
            "wipe": False,
        }
        assert {tool.name: tool.enabled for tool in blocklisted.tools} == {
            "add": True,
            "subtract": False,
            "wipe": False,
        }
        add = allowlisted.tools[0]
        assert (add.title, add.description) == ("Add numbers", "Add two numbers.")
        assert set(add.input_schema["properties"]) == {"left", "right"}
        assert allowlisted.server_info is not None
        assert allowlisted.server_info.name == "policy-test"
        assert allowlisted.latency_ms is not None
        assert allowlisted.checked_at is not None

    async def test_empty_command_is_unresolved(self) -> None:
        config = _config({"iocs": {"transport": "stdio", "url": "   "}})

        [status] = await probe_mcp_servers(config, timeout=5)

        assert status.state == "unresolved"


class TestStderrNotExposed:
    """A crashed server's stderr — which can hold whatever its declared env held — never reaches the API."""

    async def test_stderr_secret_env_value_is_not_in_the_response(self) -> None:
        secret = "sk-super-secret-token-value"  # noqa: S105 — test fixture, not a real credential.
        server = "import os, sys; sys.stderr.write(os.environ['API_KEY']); sys.exit(1)"
        config = _config(
            {
                "iocs": {
                    "transport": "stdio",
                    "url": sys.executable,
                    "args": ["-c", server],
                    "env": {"API_KEY": secret},
                }
            }
        )

        [status] = await probe_mcp_servers(config, timeout=15)

        assert status.state == "spawn_failed"
        assert secret not in status.error
        assert status.error == "Server exited before initializing."


class TestProbeRemoteServers:
    """Remote servers are probed with a real MCP handshake against a live server.

    A bare HTTP request is what these tests guard against: a healthy streamable HTTP
    server rejects it and an SSE endpoint never finishes answering it.
    """

    async def test_unreachable_remote_server_is_reported(self) -> None:
        config = _config({"docs": {"transport": "streamable_http", "url": "http://127.0.0.1:1/mcp"}})

        [status] = await probe_mcp_servers(config, timeout=5)

        assert status.state == "unreachable"
        assert status.error

    async def test_connection_error_does_not_echo_expanded_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import httpx
        import respx

        secret = "probe-test-token-123"
        monkeypatch.setenv("PROBE_TEST_TOKEN", secret)
        config = _config(
            {"docs": {"transport": "streamable_http", "url": "https://example.invalid/mcp?token=$PROBE_TEST_TOKEN"}}
        )

        with respx.mock:
            respx.route(host="example.invalid").mock(side_effect=httpx.ConnectError("Could not connect"))
            [status] = await probe_mcp_servers(config, timeout=5)

        assert status.state == "unreachable"
        assert status.error == "ConnectError."
        assert secret not in status.model_dump_json()

    async def test_healthy_streamable_http_server_is_running_and_lists_its_tools(
        self, remote_mcp_server: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PROBE_REMOTE_TOKEN", _REMOTE_TOKEN)
        config = _config(
            {
                "docs": {
                    "transport": "streamable_http",
                    "url": f"{remote_mcp_server}/mcp",
                    "custom_headers": {"Authorization": "Bearer ${PROBE_REMOTE_TOKEN}"},
                }
            }
        )

        [status] = await probe_mcp_servers(config, timeout=15)

        assert status.state == "running", status.error
        assert [tool.name for tool in status.tools] == ["lookup"]
        assert status.detail == "Connected with 1 tool(s)."

    async def test_healthy_sse_server_is_running(self, remote_mcp_server: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PROBE_REMOTE_TOKEN", _REMOTE_TOKEN)
        config = _config(
            {
                "docs": {
                    "transport": "sse",
                    "url": f"{remote_mcp_server}/sse",
                    "custom_headers": {"Authorization": "Bearer ${PROBE_REMOTE_TOKEN}"},
                }
            }
        )

        [status] = await probe_mcp_servers(config, timeout=15)

        assert status.state == "running", status.error
        assert [tool.name for tool in status.tools] == ["lookup"]

    async def test_rejected_credentials_report_the_http_status(self, remote_mcp_server: str) -> None:
        config = _config(
            {
                "docs": {
                    "transport": "streamable_http",
                    "url": f"{remote_mcp_server}/mcp",
                    "custom_headers": {"Authorization": "Bearer wrong-token"},
                }
            }
        )

        [status] = await probe_mcp_servers(config, timeout=15)

        assert status.state == "unreachable"
        assert status.detail == "Responded with HTTP 401."


class TestMcpStatusCache:
    """Polling a runtime must not relaunch every stdio server on each request."""

    @pytest.fixture
    def probe_calls(self, monkeypatch: pytest.MonkeyPatch) -> list[str]:
        """Probe without spawning anything: servers whose url starts with 'down' fail, the rest run."""
        calls: list[str] = []

        async def fake_probe(
            name: str, server: McpServerConfig, blocked: Any = (), timeout: float = 0
        ) -> McpServerStatus:
            calls.append(name)
            await asyncio.sleep(0)
            [status] = declared_mcp_servers(_config({name: server.model_dump()}))
            state = "unreachable" if server.url.startswith("down") else "running"
            return status.model_copy(update={"state": state, "checked_at": datetime.now(UTC)})

        monkeypatch.setattr(mcp_status, "probe_mcp_server", fake_probe)
        return calls

    @staticmethod
    def _cache(now: list[float]) -> McpStatusCache:
        config = _config({"up": {"transport": "stdio", "url": "up"}, "down": {"transport": "stdio", "url": "down"}})
        return McpStatusCache(config, failure_ttl=30, clock=lambda: now[0])

    async def test_running_result_is_reused_within_max_age(self, probe_calls: list[str]) -> None:
        now = [100.0]
        cache = self._cache(now)

        first, first_checked_at = await cache.get(max_age=30)
        now[0] += 29
        second, second_checked_at = await cache.get(max_age=30)

        assert probe_calls == ["down", "up"]
        assert second == first
        assert second_checked_at == first_checked_at

    async def test_expired_results_are_probed_again(self, probe_calls: list[str]) -> None:
        now = [100.0]
        cache = self._cache(now)

        await cache.get(max_age=30)
        now[0] += 30
        await cache.get(max_age=30)

        assert probe_calls == ["down", "up", "down", "up"]

    async def test_tool_listing_keeps_running_results_but_retries_failures(self, probe_calls: list[str]) -> None:
        now = [100.0]
        cache = self._cache(now)

        await cache.get(max_age=None)
        now[0] += 10
        await cache.get(max_age=None)
        now[0] += 3600
        await cache.get(max_age=None)

        assert probe_calls == ["down", "up", "down"]

    async def test_zero_max_age_reprobes_everything(self, probe_calls: list[str]) -> None:
        now = [100.0]
        cache = self._cache(now)

        await cache.get(max_age=None)
        await cache.get(max_age=0)

        assert probe_calls == ["down", "up", "down", "up"]

    async def test_concurrent_callers_share_one_probe(self, probe_calls: list[str]) -> None:
        cache = McpStatusCache(_config({"iocs": {"transport": "stdio", "url": "iocs"}}))

        await asyncio.gather(*(cache.get(max_age=30) for _ in range(5)))

        assert probe_calls == ["iocs"]

    async def test_checked_at_is_the_oldest_server_result(self, probe_calls: list[str]) -> None:
        now = [100.0]
        cache = self._cache(now)

        first, _ = await cache.get(max_age=None)
        now[0] += 31
        servers, checked_at = await cache.get(max_age=None)

        down, up = servers
        assert up == first[1]
        assert down.checked_at is not None and up.checked_at is not None
        assert down.checked_at >= up.checked_at
        assert checked_at == up.checked_at


class TestProbeUnknownTransport:
    async def test_unknown_transport_is_not_checked(self) -> None:
        config = _config({"weird": {"transport": "carrier-pigeon", "url": "somewhere"}})

        [status] = await probe_mcp_servers(config, timeout=5)

        assert status.state == "unknown"
        assert status.detail == "Unsupported transport 'carrier-pigeon'."


_REMOTE_TOKEN = "probe-remote-token"  # noqa: S105 — test fixture, not a real credential.


@pytest.fixture
def remote_mcp_server() -> Iterator[str]:
    """A live FastMCP server on a loopback port, serving streamable HTTP at /mcp and SSE at /sse.

    Every request must carry ``Authorization: Bearer <_REMOTE_TOKEN>``, so a passing
    probe proves the declared headers were sent with their env vars expanded.
    """
    import uvicorn
    from mcp.server.fastmcp import FastMCP
    from starlette.applications import Starlette

    server = FastMCP("probe-remote")

    @server.tool()
    def lookup(text: str) -> str:
        return text

    # One app serving both transports: /mcp (streamable HTTP) and /sse + /messages/ (SSE).
    http_app = server.streamable_http_app()
    app = Starlette(
        routes=[*http_app.routes, *server.sse_app().routes],
        lifespan=lambda _: server.session_manager.run(),
    )

    async def require_token(scope: Any, receive: Any, send: Any) -> None:
        expected = f"Bearer {_REMOTE_TOKEN}".encode()
        if scope["type"] == "http" and dict(scope["headers"]).get(b"authorization") != expected:
            await send({"type": "http.response.start", "status": 401, "headers": []})
            await send({"type": "http.response.body", "body": b""})
            return
        await app(scope, receive, send)

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    uvicorn_server = uvicorn.Server(uvicorn.Config(require_token, log_level="warning"))
    thread = threading.Thread(target=uvicorn_server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not uvicorn_server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("MCP test server did not start")
        time.sleep(0.01)
    try:
        yield f"http://127.0.0.1:{sock.getsockname()[1]}"
    finally:
        uvicorn_server.should_exit = True
        thread.join(timeout=10)
        sock.close()
