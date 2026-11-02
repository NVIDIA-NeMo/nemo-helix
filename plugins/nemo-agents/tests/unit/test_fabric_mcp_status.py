# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient
from mcp.types import ListToolsResult, PaginatedRequestParams, Tool
from nemo_agents_plugin.agent_config import AgentConfig, McpServerConfig
from nemo_agents_plugin.fabric import mcp_status, server
from nemo_agents_plugin.fabric.mcp_status import (
    McpServerStatus,
    McpStatusCache,
    _list_all_tools,
    probe_mcp_server,
)
from nemo_agents_plugin.fabric.server import create_fabric_serving_app

_SERVER_SCRIPT = """
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("probe-test")

@mcp.tool()
def add(a: int, b: int) -> int:
    \"\"\"Add two numbers.\"\"\"
    return a + b

@mcp.tool()
def wipe() -> str:
    return "gone"

mcp.run()
"""


_BASE_CONFIG: dict[str, Any] = {
    "config_format": "nemo-agents-spec-v1",
    "name": "test-agent",
    "default_harness": "hermes",
    "harnesses": {"hermes": {"kind": "hermes", "model": {"provider": "nvidia", "model": "nvidia/test-model"}}},
}


def _config(servers: dict[str, dict[str, Any]], **extra: Any) -> AgentConfig:
    return AgentConfig.model_validate({**_BASE_CONFIG, "mcp": {"servers": servers}, **extra})


def _stdio(*args: str, **extra: Any) -> dict[str, Any]:
    return {"transport": "stdio", "url": sys.executable, "args": list(args), **extra}


@pytest.fixture()
def server_script(tmp_path: Path) -> str:
    path = tmp_path / "server.py"
    path.write_text(_SERVER_SCRIPT, encoding="utf-8")
    return str(path)


async def test_unresolved_stdio_command() -> None:
    spec = McpServerConfig(transport="stdio", url="missing-cmd", env={"PATH": "/nowhere"})

    status = await probe_mcp_server("iocs", spec, _config({}))

    assert status.state == "unresolved"
    assert "nowhere" not in status.model_dump_json()
    assert "missing-cmd" in (status.error or "")


async def test_running_stdio_server_lists_tools_with_policy(server_script: str) -> None:
    config = _config({"calc": _stdio(server_script, blocked_tools=["wipe"])})

    [status] = (await McpStatusCache().get(config)).servers
    assert status.state == "running"
    assert {tool.name: tool.exposed for tool in status.tools} == {"add": True, "wipe": False}
    assert status.tools[0].description == "Add two numbers."
    assert status.tools[0].input_schema["required"] == ["a", "b"]


async def test_global_tool_block_and_allowlist_hide_tools(server_script: str) -> None:
    config = _config({"calc": _stdio(server_script, allowed_tools=["add"])}, tools={"blocked": ["add"]})

    [status] = (await McpStatusCache().get(config)).servers

    assert [tool.exposed for tool in status.tools] == [False, False]


async def test_crashing_stdio_server_reports_stderr() -> None:
    spec = McpServerConfig.model_validate(_stdio("-c", "import sys; sys.stderr.write('boom'); sys.exit(3)"))

    status = await probe_mcp_server("broken", spec, _config({}), timeout_seconds=2)

    assert status.state == "failed"
    assert status.error
    assert status.stderr == "boom"


async def test_unsupported_transport() -> None:
    spec = McpServerConfig(transport="websocket", url="ws://localhost:1")

    status = await probe_mcp_server("ws", spec, _config({}))

    assert status.state == "unsupported"


async def test_stdio_probe_inherits_environment_env() -> None:
    spec = McpServerConfig.model_validate(
        _stdio("-c", "import os, sys; sys.stderr.write(os.environ['PROBE_VAR']); sys.exit(1)")
    )
    config = _config({}, environment={"env": {"PROBE_VAR": "from-environment"}})

    status = await probe_mcp_server("env", spec, config, timeout_seconds=5)

    assert status.stderr == "from-environment"


async def test_stdio_probe_expands_env_references_and_redacts_them(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROBE_TOKEN", "s3cret")
    spec = McpServerConfig.model_validate(
        _stdio(
            "-c",
            "import os, sys; t = os.environ['TOKEN']; sys.stderr.write(f'{t == os.environ[\"EXPECTED\"]} {t}'); sys.exit(1)",
            env={"TOKEN": "${PROBE_TOKEN}", "EXPECTED": "s3cret"},
        )
    )

    status = await probe_mcp_server("env", spec, _config({}), timeout_seconds=5)

    assert status.stderr == "True ${PROBE_TOKEN}"
    assert "s3cret" not in status.model_dump_json()


async def test_status_never_reports_expanded_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROBE_SECRET", "s3cret")
    spec = McpServerConfig(transport="http", url="http://127.0.0.1:1/${PROBE_SECRET}")

    status = await probe_mcp_server("remote", spec, _config({}), timeout_seconds=5)

    assert status.target == "http://127.0.0.1:1/${PROBE_SECRET}"
    assert "s3cret" not in status.model_dump_json()


async def test_list_all_tools_follows_cursors() -> None:
    pages = {
        None: ListToolsResult(tools=[Tool(name="a", inputSchema={})], nextCursor="p2"),
        "p2": ListToolsResult(tools=[Tool(name="b", inputSchema={})]),
    }

    class Session:
        async def list_tools(self, *, params: PaginatedRequestParams) -> ListToolsResult:
            return pages[params.cursor]

    assert [tool.name for tool in await _list_all_tools(Session())] == ["a", "b"]  # ty: ignore[invalid-argument-type]


async def test_list_all_tools_rejects_repeated_cursor() -> None:
    class Session:
        async def list_tools(self, *, params: PaginatedRequestParams) -> ListToolsResult:
            return ListToolsResult(tools=[Tool(name="a", inputSchema={})], nextCursor="p2")

    with pytest.raises(RuntimeError, match="repeated its tool-list pagination cursor"):
        await _list_all_tools(Session())  # ty: ignore[invalid-argument-type]


def _counting_probe(monkeypatch: pytest.MonkeyPatch, delay: float = 0) -> list[str]:
    calls: list[str] = []

    async def probe(name: str, spec: McpServerConfig, config: AgentConfig) -> McpServerStatus:
        calls.append(name)
        await asyncio.sleep(delay)
        return McpServerStatus(name=name, transport="stdio", target=spec.url, state="running")

    monkeypatch.setattr(mcp_status, "probe_mcp_server", probe)
    return calls


async def test_status_cache_shares_probes_across_overlapping_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _counting_probe(monkeypatch, delay=0.05)
    cache = McpStatusCache(ttl_seconds=60)
    config = _config({"a": _stdio(), "b": _stdio()})

    responses, single = await asyncio.gather(
        asyncio.gather(*(cache.get(config) for _ in range(5))), cache.get_server(config, "a")
    )

    assert sorted(calls) == ["a", "b"]
    assert all(response.servers[0] is single for response in responses)


async def test_status_cache_reprobes_after_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _counting_probe(monkeypatch)
    cache = McpStatusCache(ttl_seconds=0)
    config = _config({"a": _stdio()})

    await cache.get_server(config, "a")
    await cache.get_server(config, "a")

    assert calls == ["a", "a"]


async def test_status_cache_ignores_undeclared_server(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _counting_probe(monkeypatch)

    assert await McpStatusCache().get_server(_config({"a": _stdio()}), "nope") is None
    assert calls == []


def test_mcp_server_endpoints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def validate(config: AgentConfig, *, base_dir: Path) -> object:
        return object()

    monkeypatch.setattr(server, "_validate_agent_config", validate)
    config_path = tmp_path / "agent.yaml"
    config_path.write_text(
        yaml.safe_dump({**_BASE_CONFIG, "mcp": {"servers": {"iocs": {"transport": "stdio", "url": "missing-cmd"}}}}),
        encoding="utf-8",
    )

    with TestClient(create_fabric_serving_app(config_path)) as client:
        listed = client.get("/v1/mcp/servers")
        single = client.get("/v1/mcp/servers/iocs")
        missing = client.get("/v1/mcp/servers/nope")

    assert listed.status_code == 200
    body = listed.json()
    assert [(s["name"], s["state"]) for s in body["servers"]] == [("iocs", "unresolved")]
    assert body["checked_at"]
    assert single.status_code == 200
    assert single.json() == body["servers"][0]
    assert missing.status_code == 404
