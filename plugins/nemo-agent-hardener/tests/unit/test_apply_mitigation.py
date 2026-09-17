# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for POST /runs/{name}/apply-mitigation and the strip_gateway_url helper it relies on."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_agent_hardener_plugin.agent_resolver import strip_gateway_url
from nemo_agent_hardener_plugin.api.v2 import runs as runs_module
from nemo_agent_hardener_plugin.entities import AgentHardenerRun
from nemo_agents_plugin.entities import Agent
from nemo_platform_plugin.entity_client import NemoEntityNotFoundError, get_entity_client

PREFIX = "/apis/agent-hardener/v2/workspaces/{workspace}"
GATEWAY = "http://localhost:8080/apis/inference-gateway/v2/workspaces/default/openai/-/v1"

HARDENED_WORKFLOW = yaml.safe_dump(
    {
        "llms": {"llm": {"_type": "openai", "model_name": "m", "base_url": GATEWAY, "api_key": "not-used"}},
        "middleware": {"custom_guardrail_1": {"_type": "pre_tool_verifier", "target_function_or_group": "Clock"}},
        "workflow": {"_type": "react_agent"},
    },
    sort_keys=False,
)


def test_strip_gateway_url_removes_only_injected_values() -> None:
    config = yaml.safe_load(HARDENED_WORKFLOW)
    config["llms"]["author"] = {"_type": "openai", "base_url": "https://api.example.com/v1", "api_key": "sk-real"}

    stripped = strip_gateway_url(config)

    # The injected gateway base_url + placeholder key are gone...
    assert "base_url" not in stripped["llms"]["llm"]
    assert "api_key" not in stripped["llms"]["llm"]
    # ...but author-set values and the hardening (middleware) are preserved.
    assert stripped["llms"]["author"]["base_url"] == "https://api.example.com/v1"
    assert stripped["llms"]["author"]["api_key"] == "sk-real"
    assert "custom_guardrail_1" in stripped["middleware"]
    # Input is not mutated.
    assert config["llms"]["llm"]["base_url"] == GATEWAY


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
    agent = Agent(name="clockbot", workspace="default", config={"llms": {}})
    saved: list[Agent] = []
    mock_entity_client.get = AsyncMock(side_effect=[_run(), agent])
    mock_entity_client.update = AsyncMock(side_effect=lambda entity: saved.append(entity) or entity)

    resp = client.post(
        "/apis/agent-hardener/v2/workspaces/default/runs/run-1/apply-mitigation",
        json={"workflow_yaml": HARDENED_WORKFLOW},
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["applied"] is True
    assert resp.json()["agent"] == "clockbot"
    # The stored config is the hardened workflow with the gateway binding stripped.
    assert "base_url" not in saved[0].config["llms"]["llm"]
    assert "custom_guardrail_1" in saved[0].config["middleware"]


def test_apply_mitigation_rejects_non_yaml(client: TestClient, mock_entity_client: AsyncMock) -> None:
    resp = client.post(
        "/apis/agent-hardener/v2/workspaces/default/runs/run-1/apply-mitigation",
        json={"workflow_yaml": "not: valid: yaml: ::"},
    )
    assert resp.status_code == 422, resp.text
    mock_entity_client.update.assert_not_called()


def test_apply_mitigation_missing_run_is_404(client: TestClient, mock_entity_client: AsyncMock) -> None:
    mock_entity_client.get = AsyncMock(side_effect=NemoEntityNotFoundError("nope"))
    resp = client.post(
        "/apis/agent-hardener/v2/workspaces/default/runs/ghost/apply-mitigation",
        json={"workflow_yaml": HARDENED_WORKFLOW},
    )
    assert resp.status_code == 404, resp.text


def test_apply_mitigation_run_without_agent_is_409(client: TestClient, mock_entity_client: AsyncMock) -> None:
    mock_entity_client.get = AsyncMock(return_value=_run(agent=""))
    resp = client.post(
        "/apis/agent-hardener/v2/workspaces/default/runs/run-1/apply-mitigation",
        json={"workflow_yaml": HARDENED_WORKFLOW},
    )
    assert resp.status_code == 409, resp.text
