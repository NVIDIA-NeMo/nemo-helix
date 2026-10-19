# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Fabric calculator example."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from nemo_agents_plugin.agent_config import load_agent_config
from nemo_agents_plugin.fabric.translator import translate_agent_config
from nemo_fabric import Fabric

EXAMPLE_DIR = Path(__file__).parents[2] / "examples/nemo-agent-config/calculator-agent"
BASE_CONFIG_PATH = EXAMPLE_DIR / "agent.yaml"
MCP_CONFIG_PATH = EXAMPLE_DIR / "agent-with-mcp.yaml"


def test_base_example_has_no_mcp_servers() -> None:
    config = load_agent_config(BASE_CONFIG_PATH)
    fabric_config = translate_agent_config(config)

    assert config.name == "calculator-agent"
    assert fabric_config.harness is not None
    assert fabric_config.harness.adapter_id == "nvidia.fabric.langchain.deepagents"
    assert fabric_config.mcp is not None
    assert fabric_config.mcp.servers == {}
    assert fabric_config.relay is not None
    assert fabric_config.environment is not None
    assert fabric_config.environment.workspace == "./workspace"
    assert fabric_config.environment.artifacts == "./artifacts"
    assert fabric_config.relay.output_dir == "./artifacts/relay"

    plan = Fabric().plan(fabric_config, base_dir=EXAMPLE_DIR)
    assert plan.adapter.adapter_id == "nvidia.fabric.langchain.deepagents"
    assert "mcp_servers" not in plan.capability_plan


def test_mcp_example_adds_harness_native_stdio_server() -> None:
    config = load_agent_config(MCP_CONFIG_PATH)
    fabric_config = translate_agent_config(config)

    assert config.name == "calculator-agent-with-mcp"
    assert fabric_config.mcp is not None
    calculator = fabric_config.mcp.servers["calculator"]
    assert calculator.transport == "stdio"
    assert calculator.url == "calculator-server"
    assert calculator.exposure == "harness_native"
    assert fabric_config.relay is not None
    assert fabric_config.relay.observability is not None
    assert fabric_config.relay.observability.atif is None
    assert fabric_config.relay.observability.atof is not None

    plan = Fabric().plan(fabric_config, base_dir=EXAMPLE_DIR)
    assert plan.adapter.adapter_id == "nvidia.fabric.langchain.deepagents"
    assert list(plan.capability_plan["native"]["mcp_servers"]) == ["calculator"]


@pytest.mark.asyncio
async def test_calculator_server_over_stdio() -> None:
    server = StdioServerParameters(
        command=sys.executable,
        args=["-c", "from mcps.calculator import main; main()"],
        cwd=str(EXAMPLE_DIR),
    )

    async with stdio_client(server) as streams:
        async with ClientSession(*streams) as session:
            await session.initialize()
            tools = await session.list_tools()
            result = await session.call_tool("multiply", {"numbers": [12, 8]})

    assert {tool.name for tool in tools.tools} == {"add", "subtract", "multiply", "divide", "compare"}
    assert result.isError is False
    assert result.structuredContent == {"result": 96.0}
