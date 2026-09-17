# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for compose_defense (build a chosen defense subset) + the compose-defense endpoint."""

from __future__ import annotations

import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_agent_hardener_plugin.api.v2 import runs as runs_module
from nemo_agent_hardener_plugin.jobs.defenses import compose_defense

PREFIX = "/apis/agent-hardener/v2/workspaces/{workspace}"


def _hardened_workflow() -> str:
    return yaml.safe_dump(
        {
            "llms": {"llm": {"_type": "openai"}, "safety_llm": {"_type": "openai"}},
            "functions": {
                "send_email": {"_type": "email", "middleware": ["custom_guardrail_1"]},
                "read_file": {"_type": "fs", "middleware": ["custom_guardrail_2"]},
            },
            "middleware": {
                "custom_guardrail_1": {"_type": "pre_tool_verifier", "target_function_or_group": "send_email"},
                "custom_guardrail_2": {"_type": "pre_tool_verifier", "target_function_or_group": "read_file"},
            },
            # The workflow entry is a middleware-bearing component too, not just a marker.
            "workflow": {"_type": "react_agent", "middleware": ["custom_guardrail_1", "custom_guardrail_2"]},
        },
        sort_keys=False,
    )


def _mitigations() -> dict:
    return {
        "workflow": {"before": "workflow: {}\n", "after": _hardened_workflow()},
        "policy": {"before": "version: 1\n", "after": "version: 1\nhardened: true\n"},
    }


def test_compose_keeps_only_selected_guardrail() -> None:
    workflow_yaml, policy_yaml = compose_defense(_mitigations(), ["custom_guardrail_1", "openshell_policy"])
    assert workflow_yaml is not None and policy_yaml is not None
    config = yaml.safe_load(workflow_yaml)

    # Only the selected guardrail survives, in the global middleware and on its tool.
    assert set(config["middleware"]) == {"custom_guardrail_1"}
    assert config["functions"]["send_email"]["middleware"] == ["custom_guardrail_1"]
    assert config["functions"]["read_file"]["middleware"] == []  # custom_guardrail_2 reference dropped
    # The workflow component's refs are pruned too — a name left pointing at a deleted middleware makes
    # the victim fail config validation ("middleware type not found") and never serve.
    assert config["workflow"]["middleware"] == ["custom_guardrail_1"]
    # safety_llm kept while a guardrail remains; hardened policy selected.
    assert "safety_llm" in config["llms"]
    assert "hardened: true" in policy_yaml


def test_compose_leaves_no_dangling_middleware_reference() -> None:
    """Every surviving reference must name a middleware that still exists."""
    for selection in ([], ["custom_guardrail_1"], ["custom_guardrail_2"], ["custom_guardrail_1", "custom_guardrail_2"]):
        workflow_yaml, _ = compose_defense(_mitigations(), selection)
        assert workflow_yaml is not None
        config = yaml.safe_load(workflow_yaml)
        defined = set(config.get("middleware") or {})
        referenced = set(config["workflow"].get("middleware") or [])
        for tool in config.get("functions", {}).values():
            referenced |= set(tool.get("middleware") or [])
        assert referenced <= defined, f"dangling refs {referenced - defined} for selection {selection}"


def test_compose_drops_all_guardrails_and_safety_llm() -> None:
    workflow_yaml, policy_yaml = compose_defense(_mitigations(), [])
    assert workflow_yaml is not None
    config = yaml.safe_load(workflow_yaml)

    assert config["middleware"] == {}
    assert "safety_llm" not in config["llms"]  # no guardrails left → safety_llm removed
    assert config["functions"]["send_email"]["middleware"] == []
    # openshell_policy not selected → baseline policy.
    assert policy_yaml == "version: 1\n"


def test_compose_handles_missing_sections() -> None:
    workflow_yaml, policy_yaml = compose_defense({}, ["custom_guardrail_1"])
    assert workflow_yaml is None
    assert policy_yaml is None


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(runs_module.router, prefix=PREFIX)
    return TestClient(app, raise_server_exceptions=False)


def test_compose_defense_endpoint_composes_selection() -> None:
    resp = _client().post(
        "/apis/agent-hardener/v2/workspaces/default/runs/run-1/compose-defense",
        json={"mitigations": _mitigations(), "selected_defense_ids": ["custom_guardrail_2"]},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    config = yaml.safe_load(body["workflow_yaml"])
    assert set(config["middleware"]) == {"custom_guardrail_2"}
    assert body["policy_yaml"] == "version: 1\n"  # policy not selected → baseline
