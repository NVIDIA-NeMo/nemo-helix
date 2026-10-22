# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
from mcp.types import Tool
from nemo_agents_plugin.agent_config import AgentConfig, McpServerConfig
from nemo_agents_plugin.fabric import tool_discovery
from nemo_agents_plugin.fabric.tool_discovery import McpToolCatalog

_SERVER_SCRIPT = textwrap.dedent(
    '''
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("calculator")


    @mcp.tool(title="Add numbers")
    def add(left: float, right: float) -> float:
        """Add two numbers."""
        return left + right


    @mcp.tool()
    def subtract(left: float, right: float) -> float:
        """Subtract right from left."""
        return left - right


    @mcp.tool()
    def wipe() -> str:
        """Pretend to delete everything."""
        return "gone"


    mcp.run()
    '''
)


def _config(servers: dict[str, dict[str, Any]], *, blocked: list[str] | None = None) -> AgentConfig:
    return AgentConfig.model_validate(
        {
            "config_format": "nemo-agents-spec-v1",
            "name": "calc",
            "default_harness": "deepagents",
            "harnesses": {"deepagents": {"kind": "deepagents"}},
            "mcp": {"servers": servers},
            "tools": {"blocked": blocked or []},
        }
    )


@pytest.fixture()
def stdio_server(tmp_path: Path) -> dict[str, Any]:
    script = tmp_path / "calculator_server.py"
    script.write_text(_SERVER_SCRIPT, encoding="utf-8")
    return {"transport": "stdio", "url": sys.executable, "args": [str(script)]}


async def test_lists_stdio_server_tools_and_applies_tool_policy(tmp_path: Path, stdio_server: dict[str, Any]) -> None:
    config = _config(
        {
            "allowlisted": {**stdio_server, "allowed_tools": ["add", "wipe"]},
            "blocklisted": {**stdio_server, "blocked_tools": ["subtract"]},
        },
        blocked=["wipe"],
    )

    servers = await McpToolCatalog(config, cwd=tmp_path).list_servers()

    assert [(server.name, server.status, server.error) for server in servers] == [
        ("allowlisted", "discovered", None),
        ("blocklisted", "discovered", None),
    ]
    assert {tool.name: tool.enabled for tool in servers[0].tools} == {"add": True, "subtract": False, "wipe": False}
    assert {tool.name: tool.enabled for tool in servers[1].tools} == {"add": True, "subtract": False, "wipe": False}
    add = servers[0].tools[0]
    assert (add.title, add.description) == ("Add numbers", "Add two numbers.")
    assert set(add.input_schema["properties"]) == {"left", "right"}


async def test_failing_server_does_not_hide_the_others(tmp_path: Path, stdio_server: dict[str, Any]) -> None:
    config = _config(
        {
            "broken": {"transport": "stdio", "url": str(tmp_path / "missing-server")},
            "calculator": stdio_server,
        }
    )

    broken, calculator = await McpToolCatalog(config, cwd=tmp_path).list_servers()

    assert (broken.status, broken.tools) == ("error", [])
    assert broken.error
    assert calculator.status == "discovered"
    assert [tool.name for tool in calculator.tools] == ["add", "subtract", "wipe"]


class _FakeLister:
    """Stands in for ``list_mcp_server_tools``, counting calls and failing on demand."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.failing: set[str] = set()
        self.release = asyncio.Event()
        self.release.set()

    async def __call__(self, server: McpServerConfig, *, cwd: Path) -> list[Tool]:
        self.calls.append(server.url)
        await self.release.wait()
        if server.url in self.failing:
            raise RuntimeError(f"{server.url} is down")
        return [Tool(name=f"{server.url}-tool", inputSchema={"type": "object"})]


@pytest.fixture()
def fake_lister(monkeypatch: pytest.MonkeyPatch) -> _FakeLister:
    lister = _FakeLister()
    monkeypatch.setattr(tool_discovery, "list_mcp_server_tools", lister)
    return lister


def _two_servers() -> AgentConfig:
    return _config({"a": {"transport": "stdio", "url": "a"}, "b": {"transport": "sse", "url": "b"}})


async def test_successful_listings_are_cached_until_refresh(tmp_path: Path, fake_lister: _FakeLister) -> None:
    catalog = McpToolCatalog(_two_servers(), cwd=tmp_path)

    first = await catalog.list_servers()
    second = await catalog.list_servers()
    assert fake_lister.calls == ["a", "b"]
    assert first == second

    await catalog.list_servers(refresh=True)
    assert fake_lister.calls == ["a", "b", "a", "b"]


async def test_concurrent_callers_share_one_discovery_pass(tmp_path: Path, fake_lister: _FakeLister) -> None:
    catalog = McpToolCatalog(_two_servers(), cwd=tmp_path)
    fake_lister.release.clear()

    pending = [asyncio.create_task(catalog.list_servers()) for _ in range(5)]
    await asyncio.sleep(0)
    fake_lister.release.set()
    results = await asyncio.gather(*pending)

    assert fake_lister.calls == ["a", "b"]
    assert all(result == results[0] for result in results)


async def test_failures_are_retried_only_after_the_retry_window(tmp_path: Path, fake_lister: _FakeLister) -> None:
    fake_lister.failing.add("b")
    cached = McpToolCatalog(_two_servers(), cwd=tmp_path, error_retry_seconds=60)
    await cached.list_servers()
    _, b = await cached.list_servers()
    assert (b.status, b.error) == ("error", "b is down")
    assert fake_lister.calls == ["a", "b"]

    fake_lister.calls.clear()
    retried = McpToolCatalog(_two_servers(), cwd=tmp_path, error_retry_seconds=0)
    await retried.list_servers()
    fake_lister.failing.clear()
    _, b = await retried.list_servers()
    assert b.status == "discovered"
    assert fake_lister.calls == ["a", "b", "b"]


async def test_slow_server_times_out(tmp_path: Path, fake_lister: _FakeLister) -> None:
    fake_lister.release.clear()

    (server,) = await McpToolCatalog(
        _config({"slow": {"transport": "stdio", "url": "slow"}}), cwd=tmp_path, timeout_seconds=0.01
    ).list_servers()

    assert (server.status, server.error) == ("error", "Timed out after 0.01s listing tools.")


async def test_agent_without_mcp_lists_nothing(tmp_path: Path, fake_lister: _FakeLister) -> None:
    config = AgentConfig.model_validate(
        {
            "config_format": "nemo-agents-spec-v1",
            "name": "calc",
            "default_harness": "deepagents",
            "harnesses": {"deepagents": {"kind": "deepagents"}},
        }
    )

    assert await McpToolCatalog(config, cwd=tmp_path).list_servers() == []
    assert fake_lister.calls == []
