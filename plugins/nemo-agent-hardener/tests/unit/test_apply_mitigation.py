# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for POST /runs/{name}/apply-mitigation and the strip_gateway_url helper it relies on."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_agent_hardener_plugin.agent_resolver import strip_gateway_url
from nemo_agent_hardener_plugin.api.v2 import runs as runs_module
from nemo_agent_hardener_plugin.entities import AgentHardenerManifest, AgentHardenerRun
from nemo_agents_plugin.entities import Agent
from nemo_platform_plugin.entity_client import NemoEntityNotFoundError, get_entity_client

PREFIX = "/apis/agent-hardener/v2/workspaces/{workspace}"
GATEWAY = "http://localhost:8080/apis/inference-gateway/v2/workspaces/default/openai/-/v1"

#: What the run hardened: the guardrail set the victim actually ran, as plugins.toml.
HARDENED_GUARDRAILS = (
    "version = 1\n"
    "[[components]]\n"
    'kind = "agent_hardener.pre_tool_verifier"\n'
    "[components.config.model]\n"
    'model = "m"\n'
    "[[components.config.guardrails]]\n"
    'name = "custom_guardrail_1"\n'
    'target_tool = "Clock"\n'
    'system_instructions = "Refuse clock tampering."\n'
)

#: The stored agent config the guardrail is adopted onto.
GATEWAY_BOUND_AGENT: dict[str, Any] = {
    "config_format": "nemo-agents-spec-v1",
    "models": {"llm": {"provider": "nvidia", "model": "m", "base_url": GATEWAY, "api_key": "not-used"}},
}


def test_strip_gateway_url_removes_only_injected_values() -> None:
    config = {**GATEWAY_BOUND_AGENT, "models": dict(GATEWAY_BOUND_AGENT["models"])}
    config["models"]["author"] = {"base_url": "https://api.example.com/v1", "api_key": "sk-real"}

    stripped = strip_gateway_url(config)

    # The injected gateway base_url + placeholder key are gone, so the stored agent stays
    # deployment-neutral and its next deploy re-injects whatever gateway that environment has...
    assert "base_url" not in stripped["models"]["llm"]
    assert "api_key" not in stripped["models"]["llm"]
    # ...but anything the author set is untouched.
    assert stripped["models"]["author"]["base_url"] == "https://api.example.com/v1"
    assert stripped["models"]["author"]["api_key"] == "sk-real"
    # Input is not mutated.
    assert config["models"]["llm"]["base_url"] == GATEWAY


@pytest.fixture
def mock_entity_client() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def client(mock_entity_client: AsyncMock) -> TestClient:
    app = FastAPI()
    app.include_router(runs_module.router, prefix=PREFIX)
    app.dependency_overrides[get_entity_client] = lambda: mock_entity_client
    return TestClient(app, raise_server_exceptions=False)


def _run(agent: str = "clockbot") -> AgentHardenerRun:
    return AgentHardenerRun(name="run-1", workspace="default", agent=agent)


def test_apply_mitigation_updates_agent_config(client: TestClient, mock_entity_client: AsyncMock) -> None:
    agent = Agent(name="clockbot", workspace="default", config=dict(GATEWAY_BOUND_AGENT))
    saved: list[Agent] = []
    mock_entity_client.get = AsyncMock(side_effect=[_run(), agent])
    mock_entity_client.update = AsyncMock(side_effect=lambda entity: saved.append(entity) or entity)

    resp = client.post(
        "/apis/agent-hardener/v2/workspaces/default/runs/run-1/apply-mitigation",
        json={"guardrails_toml": HARDENED_GUARDRAILS},
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["applied"] is True
    assert resp.json()["agent"] == "clockbot"
    # The guardrail is re-homed onto the entity as a Relay component — the one place
    # relay.components[] is produced — with the gateway binding stripped.
    stored = saved[0].config
    assert "base_url" not in stored["models"]["llm"]
    (component,) = stored["telemetry"]["relay_components"]
    assert component["kind"] == "agent_hardener.pre_tool_verifier"
    assert [rail["name"] for rail in component["config"]["guardrails"]] == ["custom_guardrail_1"]


def test_apply_mitigation_rejects_a_malformed_guardrail_file(client: TestClient, mock_entity_client: AsyncMock) -> None:
    resp = client.post(
        "/apis/agent-hardener/v2/workspaces/default/runs/run-1/apply-mitigation",
        json={"guardrails_toml": "::: not toml"},
    )
    assert resp.status_code == 422, resp.text
    mock_entity_client.update.assert_not_called()


def test_apply_mitigation_missing_run_is_404(client: TestClient, mock_entity_client: AsyncMock) -> None:
    mock_entity_client.get = AsyncMock(side_effect=NemoEntityNotFoundError("nope"))
    resp = client.post(
        "/apis/agent-hardener/v2/workspaces/default/runs/ghost/apply-mitigation",
        json={"guardrails_toml": HARDENED_GUARDRAILS},
    )
    assert resp.status_code == 404, resp.text


def test_apply_mitigation_run_without_agent_is_409(client: TestClient, mock_entity_client: AsyncMock) -> None:
    mock_entity_client.get = AsyncMock(return_value=_run(agent=""))
    resp = client.post(
        "/apis/agent-hardener/v2/workspaces/default/runs/run-1/apply-mitigation",
        json={"guardrails_toml": HARDENED_GUARDRAILS},
    )
    assert resp.status_code == 409, resp.text


def test_apply_mitigation_byo_manifest_is_409(client: TestClient, mock_entity_client: AsyncMock) -> None:
    # A project run stores its *manifest* name in `agent`, so the empty-agent guard passes and the
    # lookup would otherwise adopt onto an unrelated agent that happens to share the name.
    run = AgentHardenerRun(name="run-1", workspace="default", agent="lc1", manifest_id="lc1")
    manifest = AgentHardenerManifest(name="lc1", workspace="default", source_type="project")
    mock_entity_client.get = AsyncMock(side_effect=[run, manifest])

    resp = client.post(
        "/apis/agent-hardener/v2/workspaces/default/runs/run-1/apply-mitigation",
        json={"guardrails_toml": HARDENED_GUARDRAILS},
    )

    assert resp.status_code == 409, resp.text
    mock_entity_client.update.assert_not_called()
