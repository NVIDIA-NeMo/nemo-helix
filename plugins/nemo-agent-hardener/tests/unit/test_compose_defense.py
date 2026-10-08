# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for compose_defense (build a chosen defense subset) + the compose-defense endpoint."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_agent_hardener_plugin.api.v2 import runs as runs_module
from nemo_agent_hardener_plugin.jobs.defenses import compose_defense

PREFIX = "/apis/agent-hardener/v2/workspaces/{workspace}"


def _hardened_guardrails() -> str:
    return (
        "version = 1\n"
        "[[components]]\n"
        'kind = "agent_hardener.pre_tool_verifier"\n'
        "[components.config.model]\n"
        'model = "m"\n'
        "[[components.config.guardrails]]\n"
        'name = "custom_guardrail_1"\n'
        'target_tool = "send_email"\n'
        'system_instructions = "Refuse exfiltration."\n'
        "[[components.config.guardrails]]\n"
        'name = "custom_guardrail_2"\n'
        'target_tool = "read_file"\n'
        'system_instructions = "Refuse credential paths."\n'
    )


def _mitigations() -> dict:
    return {
        "guardrails": {"before": "version = 1\n", "after": _hardened_guardrails()},
        "policy": {"before": "version: 1\n", "after": "version: 1\nhardened: true\n"},
    }


def _rails(toml_text: str) -> list[dict]:
    import tomllib  # noqa: PLC0415

    document = tomllib.loads(toml_text)
    return [rail for entry in document["components"] for rail in entry["config"]["guardrails"]]


def test_compose_keeps_only_selected_guardrail() -> None:
    guardrails_toml, policy_yaml = compose_defense(_mitigations(), ["custom_guardrail_1", "openshell_policy"])
    assert guardrails_toml is not None and policy_yaml is not None

    rails = _rails(guardrails_toml)
    assert [rail["name"] for rail in rails] == ["custom_guardrail_1"]
    assert rails[0]["target_tool"] == "send_email"
    assert "hardened: true" in policy_yaml


def test_pruning_needs_no_reference_cleanup() -> None:
    """Each guardrail is one self-contained table, so removing it cannot dangle a reference.

    The NAT version had to strip the name from every middleware-bearing component as well; missing
    one left the victim failing config validation and never serving.
    """
    guardrails_toml, _ = compose_defense(_mitigations(), ["custom_guardrail_2"])
    assert guardrails_toml is not None
    assert [rail["name"] for rail in _rails(guardrails_toml)] == ["custom_guardrail_2"]
    assert "custom_guardrail_1" not in guardrails_toml


def test_compose_drops_every_guardrail_when_none_selected() -> None:
    guardrails_toml, _ = compose_defense(_mitigations(), [])
    assert guardrails_toml is not None
    assert _rails(guardrails_toml) == []


def test_compose_tolerates_a_malformed_guardrail_file() -> None:
    """A display/selection path must not fail the run on a surprise."""
    bad = {"guardrails": {"before": "", "after": "::: not toml"}}
    assert compose_defense(bad, ["custom_guardrail_1"])[0] == "::: not toml"


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
    assert [rail["name"] for rail in _rails(body["guardrails_toml"])] == ["custom_guardrail_2"]
    assert body["policy_yaml"] == "version: 1\n"  # policy not selected → baseline
