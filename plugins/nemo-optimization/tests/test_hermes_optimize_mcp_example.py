# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The hermes-optimize MCP example is self-contained: its analyzer, config and scoring all resolve.

No model is called. The example's value is that a Hermes run against it needs nothing outside this
repository, so these tests pin the three pieces that would otherwise only be checked by a live run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from nemo_optimization.backends.optuna.fabric_trial import _build_metrics

_EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "hermes-optimize"
_CONFIG = yaml.safe_load((_EXAMPLE / "optimize-mcp.yaml").read_text(encoding="utf-8"))
_DATASET = json.loads((_EXAMPLE / "dataset-mcp.json").read_text(encoding="utf-8"))


def _email(row: dict[str, str]) -> str:
    """The instruction the optimizer builds from a row (``_row_instruction``): ``subject\\n\\nbody``."""
    return f"{row['subject']}\n\n{row['body']}"


def test_the_mock_analyzer_replays_each_row_s_canned_analysis() -> None:
    """The judge scores the coordinator against the row label, so the replayed analysis must agree with it."""
    from phishing_analyzer_mcp.server import analyze, load_responses

    responses = load_responses(_EXAMPLE / "dataset-mcp.json")
    for row in _DATASET:
        verbatim = analyze(_email(row), responses)
        assert verbatim["label"] == row["label"] == row["analysis"]["label"], row["id"]
        assert verbatim["matched"] == "exact"
        # Framing around the email is tolerated as long as the body is intact, whether the agent
        # wrapped the body alone or the full subject + body (which matches both of the row's keys)...
        for wrapped in (row["body"], _email(row)):
            framed = analyze(f"Please analyze this:\n{wrapped}\nThanks", responses)
            assert (framed["label"], framed["matched"]) == (row["label"], "body"), row["id"]
    # ...but an edited body is not analyzed: the fixture is the input-binding check.
    edited = analyze(_DATASET[0]["body"].replace("iPhone", "laptop"), responses)
    assert (edited["label"], edited["matched"]) == ("unknown", "none")


async def test_the_analyzer_serves_its_tool_over_mcp_stdio() -> None:
    """Hermes reaches the analyzer as a stdio MCP server, so exercise that transport, not the function."""
    params = StdioServerParameters(command=sys.executable, args=["-m", "phishing_analyzer_mcp.server"])
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = await session.list_tools()
        assert [tool.name for tool in tools.tools] == ["email_phishing_analyzer"]
        result = await session.call_tool("email_phishing_analyzer", {"text": _email(_DATASET[0])})
    assert result.structuredContent is not None
    assert result.structuredContent["label"] == "phishing"


def test_the_example_config_declares_the_bundled_server_and_the_exactly_once_evaluator() -> None:
    server = _CONFIG["mcp"]["servers"]["email-phishing-analyzer"]
    assert server == {"transport": "stdio", "url": "phishing-analyzer-mcp", "exposure": "harness_native"}
    assert "run_hook" not in _CONFIG["eval"]
    assert _CONFIG["eval"]["fabric"]["capture_trajectory"] is True  # the evaluator reads the trajectory

    metrics = _build_metrics(_CONFIG, _CONFIG["eval"])
    by_type = {type(metric).__name__: metric for metric in metrics}
    exactly_once = by_type["ToolCallCountMetric"]
    assert (exactly_once.tool_name, exactly_once.expected_calls) == ("email_phishing_analyzer", 1)
    # Every study objective must be an output some evaluator actually emits.
    emitted = {spec.name for metric in metrics for spec in metric.output_spec()}
    for objective in _CONFIG["optimizer"]["eval_metrics"].values():
        assert objective["evaluator_name"] in emitted, objective


def test_fabric_accepts_the_example_agent_config() -> None:
    """The MCP block must survive Fabric's own validation and planning, not just our YAML reading."""
    pytest.importorskip("nemo_fabric")
    from nemo_fabric import Fabric, FabricConfig

    agent = {key: value for key, value in _CONFIG.items() if key not in {"optimizer", "eval"}}
    plan = Fabric().plan(FabricConfig.from_mapping(agent))
    servers = plan.capability_plan["mcp_servers"]
    assert servers["email-phishing-analyzer"]["url"] == "phishing-analyzer-mcp"
