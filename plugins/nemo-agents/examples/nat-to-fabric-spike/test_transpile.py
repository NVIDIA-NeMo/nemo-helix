# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the NAT -> Fabric transpile spike.

Run: pytest test_transpile.py
These assert the behaviors the spike claims (topology preserved as Deep Agents
subagents, MCP servers carried one-to-one, credentials flagged not guessed, NAT
builtins surfaced) and the unhappy paths a Stage 1 spike will hit off its fixture
(cycles, missing/dangling refs, stdio edge cases, non-MCP groups).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import transpile as T

HERE = Path(__file__).parent
FIXTURE = HERE / "nat_agent" / "config.yml"


@pytest.fixture()
def fixture_result():
    config = yaml.safe_load(FIXTURE.read_text())
    report = T.Report()
    fabric = T.transpile(config, report)
    return fabric, report


def _config(**over) -> dict:
    base = {
        "llms": {"m": {"_type": "nim", "model_name": "x", "temperature": 0.0}},
        "functions": {},
        "function_groups": {},
        "workflow": {"_type": "react_agent", "tool_names": [], "llm_name": "m"},
    }
    base.update(over)
    return base


# --- topology -------------------------------------------------------------

def test_reasoning_wrapper_maps_wrapped_agent_as_main(fixture_result):
    fabric, _ = fixture_result
    # The wrapped react_agent, not the reasoning_agent, drives metadata + subagents.
    assert fabric["metadata"]["name"] == "research_orchestrator"
    assert "wrapping react_agent 'research_orchestrator'" in fabric["metadata"]["description"]


def test_subagents_preserved_with_names_and_tools(fixture_result):
    fabric, _ = fixture_result
    subs = {s["name"]: s for s in fabric["harness"]["settings"]["deepagents"]["subagents"]}
    assert set(subs) == {"math_agent", "time_agent", "jira_agent"}
    assert subs["math_agent"]["tools"] == ["mcp_math"]
    assert subs["time_agent"]["tools"] == ["mcp_time", "current_timezone"]
    assert "arithmetic" in subs["math_agent"]["description"].lower()


def test_nested_agents_keep_identity_not_flattened():
    # orchestrator -> team_lead(agent) -> worker(agent) -> mcp_srv
    config = _config(
        functions={
            "worker": {"_type": "tool_calling_agent", "tool_names": ["srv"], "llm_name": "m", "description": "W"},
            "team_lead": {"_type": "react_agent", "tool_names": ["worker"], "llm_name": "m", "description": "TL"},
            "root": {"_type": "react_agent", "tool_names": ["team_lead"], "llm_name": "m"},
        },
        function_groups={"srv": {"_type": "mcp_client", "server": {"transport": "streamable-http", "url": "${U}"}}},
        workflow={"_type": "react_agent", "tool_names": ["team_lead"], "llm_name": "m"},
    )
    report = T.Report()
    fabric = T.transpile(config, report)
    subs = {s["name"]: s for s in fabric["harness"]["settings"]["deepagents"]["subagents"]}
    # every nested agent survives as its own subagent, not hoisted into its parent
    assert set(subs) == {"team_lead", "worker"}
    assert subs["team_lead"]["delegates_to"] == ["worker"]
    assert subs["worker"]["tools"] == ["srv"]


def test_cycle_does_not_recurse_forever():
    config = _config(
        functions={
            "a": {"_type": "react_agent", "tool_names": ["b"], "llm_name": "m", "description": "A"},
            "b": {"_type": "react_agent", "tool_names": ["a"], "llm_name": "m", "description": "B"},
        },
        workflow={"_type": "react_agent", "tool_names": ["a"], "llm_name": "m"},
    )
    report = T.Report()
    fabric = T.transpile(config, report)  # must not raise RecursionError
    names = {s["name"] for s in fabric["harness"]["settings"]["deepagents"]["subagents"]}
    assert names == {"a", "b"}  # each emitted once


def test_router_agent_branches_become_subagents_and_metadata_derived():
    config = _config(
        functions={
            "a": {"_type": "react_agent", "tool_names": [], "llm_name": "m", "description": "A"},
            "b": {"_type": "react_agent", "tool_names": [], "llm_name": "m", "description": "B"},
        },
        workflow={"_type": "router_agent", "branches": ["a", "b"], "llm_name": "m"},
    )
    report = T.Report()
    fabric = T.transpile(config, report)
    names = {s["name"] for s in fabric["harness"]["settings"]["deepagents"]["subagents"]}
    assert names == {"a", "b"}
    # metadata is derived from the input, not hardcoded to the fixture agent.
    assert fabric["metadata"]["name"] == "router-agent"
    assert "router_agent" in fabric["metadata"]["description"]


# --- mcp carryover --------------------------------------------------------

def test_all_mcp_servers_carried(fixture_result):
    fabric, _ = fixture_result
    servers = fabric["mcp"]["servers"]
    assert set(servers) == {"mcp_math", "mcp_time", "mcp_jira"}
    for spec in servers.values():
        assert {"transport", "url", "exposure"} <= set(spec)
        assert spec["exposure"] == "harness_native"


def test_stdio_server_becomes_command_url(fixture_result):
    fabric, _ = fixture_result
    time_server = fabric["mcp"]["servers"]["mcp_time"]
    assert time_server["transport"] == "stdio"
    assert time_server["url"] == "python -m mcp_server_time --local-timezone=America/Los_Angeles"


def test_stdio_missing_command_is_error_not_garbage():
    config = _config(
        function_groups={"srv": {"_type": "mcp_client", "server": {"transport": "stdio"}}},
        workflow={"_type": "react_agent", "tool_names": ["srv"], "llm_name": "m"},
    )
    report = T.Report()
    fabric = T.transpile(config, report)
    assert fabric["mcp"]["servers"]["srv"]["url"] != "''"
    assert any("stdio server has no command" in e for e in report.errors)


def test_env_vars_detected_including_auth_block(fixture_result):
    _, report = fixture_result
    # the Jira redirect_uri lives in the authentication block; it must still be caught.
    assert report.env_vars == {"MATH_MCP_URL", "CORPORATE_MCP_JIRA_URL", "NAT_REDIRECT_URI"}


def test_env_default_syntax_captured():
    config = _config(
        function_groups={
            "srv": {"_type": "mcp_client", "server": {"transport": "streamable-http", "url": "${HOST:-localhost}/mcp"}},
        },
        workflow={"_type": "react_agent", "tool_names": ["srv"], "llm_name": "m"},
    )
    report = T.Report()
    T.transpile(config, report)
    assert "HOST" in report.env_vars


# --- credentials: flag, do not guess --------------------------------------

def test_oauth2_server_flagged_as_gap(fixture_result):
    _, report = fixture_result
    assert len(report.auth_gaps) == 1
    assert "mcp_jira" in report.auth_gaps[0]
    assert "mcp_oauth2" in report.auth_gaps[0]


def test_custom_headers_flagged_as_gap():
    config = _config(
        function_groups={
            "srv": {
                "_type": "mcp_client",
                "server": {"transport": "streamable-http", "url": "${U}", "custom_headers": {"X-Token": "${TOK}"}},
            },
        },
        workflow={"_type": "react_agent", "tool_names": ["srv"], "llm_name": "m"},
    )
    report = T.Report()
    T.transpile(config, report)
    assert any("custom_headers" in g for g in report.auth_gaps)


# --- builtins & error surfacing -------------------------------------------

def test_builtins_flagged_for_mcp_equivalent(fixture_result):
    _, report = fixture_result
    assert report.builtins == {"current_timezone", "code_generation"}


def test_dangling_ref_is_error_not_builtin():
    config = _config(workflow={"_type": "react_agent", "tool_names": ["typo_tool"], "llm_name": "m"})
    report = T.Report()
    T.transpile(config, report)
    assert "typo_tool" not in report.builtins
    assert any("typo_tool" in e and "not defined" in e for e in report.errors)


def test_non_mcp_function_group_is_error():
    config = _config(
        function_groups={"calc": {"_type": "calculator"}},
        workflow={"_type": "react_agent", "tool_names": ["calc"], "llm_name": "m"},
    )
    report = T.Report()
    T.transpile(config, report)
    assert any("calc" in e and "not MCP" in e for e in report.errors)


# --- models ---------------------------------------------------------------

def test_models_default_is_main_agent_model(fixture_result):
    fabric, _ = fixture_result
    models = fabric["models"]
    assert models["default"]["model"] == "nvidia/llama-3.3-nemotron-super-49b-v1"
    assert models["default"]["provider"] == "nvidia"
    assert models["default"]["api_key_env"] == "NVIDIA_API_KEY"
    assert models["default"]["settings"]["max_tokens"] == 2000
    assert "worker_llm" in models


def test_missing_llm_name_is_graceful():
    config = _config(workflow={"_type": "sequential_executor", "tool_list": [], "llm_name": None})
    report = T.Report()
    fabric = T.transpile(config, report)  # must not KeyError
    assert fabric["models"]["default"]["provider"] == T.RESOLVE
    assert any("no llm_name" in e for e in report.errors)


def test_non_nim_llm_not_mislabeled_nvidia():
    config = _config(llms={"m": {"_type": "openai", "model_name": "gpt-4o", "temperature": 0.0}})
    report = T.Report()
    fabric = T.transpile(config, report)
    assert fabric["models"]["default"]["provider"] == "openai"
    assert "api_key_env" not in fabric["models"]["default"]


# --- contract validation --------------------------------------------------

def test_output_passes_structural_check(fixture_result):
    fabric, _ = fixture_result
    T.structural_check(fabric)


def test_structural_check_rejects_bad_mcp_server(fixture_result):
    fabric, _ = fixture_result
    del fabric["mcp"]["servers"]["mcp_math"]["exposure"]
    with pytest.raises(ValueError, match="exposure"):
        T.structural_check(fabric)


def test_fixture_has_no_errors(fixture_result):
    _, report = fixture_result
    assert report.errors == []


def test_middleware_flagged_as_feature_needing_a_home():
    # NASSE is NAT middleware; it maps to Relay, not carried by the transpiler.
    config = _config(middleware={"nasse": {"_type": "nasse_guard"}})
    report = T.Report()
    T.transpile(config, report)
    assert any(f.startswith("middleware:") and "Relay" in f for f in report.features)


def test_default_name_from_filename_when_no_wrapper():
    # A plain react_agent (no reasoning wrapper) should take the input filename, not "react-agent".
    config = _config(workflow={"_type": "react_agent", "tool_names": [], "llm_name": "m"})
    report = T.Report()
    fabric = T.transpile(config, report, default_name="scout")
    assert fabric["metadata"]["name"] == "scout"


def test_analyze_renders_composition(fixture_result):
    fabric, report = fixture_result
    analysis = T.render_analysis(fabric, report)
    assert "# NAT agent analysis: research_orchestrator" in analysis
    assert "math_agent" in analysis and "tool_calling_agent" in analysis
    assert "mcp_math" in analysis
    assert "4 agent(s)" in analysis


def test_custom_top_level_agent_type_is_reported_not_crash():
    # Real blueprints (e.g. AI-Q) use custom registered _types, not stock archetypes.
    config = _config(workflow={"_type": "chat_deepresearcher_agent", "enable_clarifier": True})
    report = T.Report()
    fabric = T.transpile(config, report)  # must not KeyError on the unknown _type
    assert any("custom NAT type" in e for e in report.errors)
    T.structural_check(fabric)  # still emits a valid skeleton
