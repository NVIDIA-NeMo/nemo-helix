# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient
from nemo_agents_plugin.agent_config import AgentConfig, McpServerConfig
from nemo_agents_plugin.fabric import server
from nemo_agents_plugin.fabric.mcp_status import probe_mcp_server, probe_mcp_servers
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

    response = await probe_mcp_servers(config)

    [status] = response.servers
    assert status.state == "running"
    assert {tool.name: tool.exposed for tool in status.tools} == {"add": True, "wipe": False}
    assert status.tools[0].description == "Add two numbers."
    assert status.tools[0].input_schema["required"] == ["a", "b"]


async def test_global_tool_block_and_allowlist_hide_tools(server_script: str) -> None:
    config = _config({"calc": _stdio(server_script, allowed_tools=["add"])}, tools={"blocked": ["add"]})

    [status] = (await probe_mcp_servers(config)).servers

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


def test_mcp_status_endpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def validate(config: AgentConfig, *, base_dir: Path) -> object:
        return object()

    monkeypatch.setattr(server, "_validate_agent_config", validate)
    config_path = tmp_path / "agent.yaml"
    config_path.write_text(
        yaml.safe_dump({**_BASE_CONFIG, "mcp": {"servers": {"iocs": {"transport": "stdio", "url": "missing-cmd"}}}}),
        encoding="utf-8",
    )

    with TestClient(create_fabric_serving_app(config_path)) as client:
        response = client.get("/mcp/status")

    assert response.status_code == 200
    body = response.json()
    assert [(s["name"], s["state"]) for s in body["servers"]] == [("iocs", "unresolved")]
    assert body["checked_at"]
