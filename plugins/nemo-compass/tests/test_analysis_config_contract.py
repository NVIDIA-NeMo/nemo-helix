# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for carrying scheduled-analysis models into Platform state."""

import json
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_compass_plugin.entities import AnalysisConfig, AnalysisConfigStatus
from nemo_compass_plugin.sdk_resources.analysis_configs import (
    _build_enable_body,
    _build_status_update_body,
    _build_update_body,
)
from nemo_compass_plugin.service import CompassService
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.dependencies import get_nemo_client
from nemo_helix_plugin.entity_client import NemoEntityNotFoundError, get_entity_client
from nhx.testing import mock_async_nemo_client


class _FakeAgents:
    """The Agents API as the caller sees it: lists the workspace's Compass agent and records creates."""

    def __init__(self, *, existing: str | None = None, register_status: int = 200) -> None:
        self.existing = existing
        self.register_status = register_status
        self.registered: list[dict[str, object]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if not request.url.path.endswith("/agents"):
            return httpx.Response(404, json={"detail": f"unexpected {request.method} {request.url.path}"})
        if request.method == "GET":
            assert json.loads(request.url.params["filter"]) == {"plugin_managed_agent.type": "compass.analyst"}
            items = [self._agent(self.existing)] if self.existing else []
            pagination = {
                "page": 1,
                "page_size": 1,
                "current_page_size": len(items),
                "total_pages": 1,
                "total_results": len(items),
            }
            return httpx.Response(200, json={"data": items, "pagination": pagination})
        body = json.loads(request.content)
        self.registered.append(body)
        if self.register_status != 200:
            return httpx.Response(self.register_status, json={"detail": "nope"})
        return httpx.Response(200, json=self._agent(body["name"]))

    @staticmethod
    def _agent(name: str) -> dict[str, object]:
        return {
            "name": name,
            "workspace": "default",
            "config": {},
            "config_format": "nemo-agents-spec-v1",
            "plugin_managed_agent": {"type": "compass.analyst"},
        }


def _app(entity_client: AsyncMock, agents: _FakeAgents | None = None) -> FastAPI:
    app = FastAPI()
    for spec in CompassService().get_routers():
        app.include_router(spec.router, prefix=spec.prefix)
    app.dependency_overrides[get_entity_client] = lambda: entity_client
    fake_agents = agents or _FakeAgents(existing="compass-analyst")
    app.dependency_overrides[get_nemo_client] = lambda: mock_async_nemo_client(fake_agents, AsyncNemoClient)
    return app


def _missing_config_store() -> AsyncMock:
    entity_client = AsyncMock()
    entity_client.get.side_effect = NemoEntityNotFoundError("missing")
    entity_client.create.side_effect = lambda config: config
    return entity_client


_ENABLE = "/v2/workspaces/default/analysis-configs/calculator-agent/enable"
_MODELS = {"default_model": "default/gpt-5", "fast_model": "default/gpt-5-mini"}


def test_enabling_analysis_creates_the_workspaces_compass_agent() -> None:
    agents = _FakeAgents()

    response = TestClient(_app(_missing_config_store(), agents)).post(_ENABLE, json=_MODELS)

    assert response.status_code == 200
    assert agents.registered == [{"name": "compass-analyst", "plugin_managed_agent": {"type": "compass.analyst"}}]


def test_enabling_analysis_keeps_an_existing_compass_agent() -> None:
    agents = _FakeAgents(existing="our-compass")

    response = TestClient(_app(_missing_config_store(), agents)).post(_ENABLE, json=_MODELS)

    assert response.status_code == 200
    assert agents.registered == []


@pytest.mark.parametrize("status", [403, 409])
def test_the_config_is_saved_even_when_the_agent_cannot_be_created(status: int) -> None:
    """A caller who may configure analysis but not create agents still saves the config; the scheduler reports it."""
    entity_client = _missing_config_store()
    agents = _FakeAgents(register_status=status)

    response = TestClient(_app(entity_client, agents)).post(_ENABLE, json=_MODELS)

    assert response.status_code == 200
    assert len(agents.registered) == 1
    entity_client.create.assert_awaited_once()


def test_updating_a_config_creates_the_compass_agent_too() -> None:
    entity_client = AsyncMock()
    entity_client.get.return_value = AnalysisConfig(
        name="calculator-agent", workspace="default", agent="calculator-agent", default_model="default/old"
    )
    entity_client.update.side_effect = lambda config: config
    agents = _FakeAgents()

    response = TestClient(_app(entity_client, agents)).patch(
        "/v2/workspaces/default/analysis-configs/calculator-agent", json={"enabled": True}
    )

    assert response.status_code == 200
    assert [body["name"] for body in agents.registered] == ["compass-analyst"]


def test_enable_request_requires_and_persists_model_pair() -> None:
    entity_client = AsyncMock()
    entity_client.get.side_effect = NemoEntityNotFoundError("missing")
    entity_client.create.side_effect = lambda config: config

    response = TestClient(_app(entity_client)).post(
        "/v2/workspaces/default/analysis-configs/calculator-agent/enable",
        json={
            "default_model": "default/gpt-5",
            "fast_model": "default/gpt-5-mini",
        },
    )

    assert response.status_code == 200
    created = entity_client.create.await_args.args[0]
    assert created.default_model == "default/gpt-5"
    assert created.fast_model == "default/gpt-5-mini"
    assert response.json()["default_model"] == "default/gpt-5"
    assert response.json()["fast_model"] == "default/gpt-5-mini"


def test_reenable_refreshes_persisted_model_pair() -> None:
    entity_client = AsyncMock()
    existing = AnalysisConfig(
        name="calculator-agent",
        workspace="default",
        agent="calculator-agent",
        enabled=False,
        default_model="default/old-model",
        fast_model="default/old-model",
    )
    entity_client.get.return_value = existing
    entity_client.update.side_effect = lambda config: config

    response = TestClient(_app(entity_client)).post(
        "/v2/workspaces/default/analysis-configs/calculator-agent/enable",
        json={
            "default_model": "default/gpt-5",
            "fast_model": "default/gpt-5-mini",
        },
    )

    assert response.status_code == 200
    updated = entity_client.update.await_args.args[0]
    assert updated.enabled is True
    assert updated.default_model == "default/gpt-5"
    assert updated.fast_model == "default/gpt-5-mini"


def test_enable_sdk_body_contains_both_model_refs() -> None:
    assert _build_enable_body(
        default_model="default/gpt-5",
        fast_model="default/gpt-5-mini",
    ).model_dump(mode="json") == {
        "default_model": "default/gpt-5",
        "fast_model": "default/gpt-5-mini",
    }


def test_update_config_sdk_body_omits_none_enabled() -> None:
    assert _build_update_body(enabled=None).model_dump(mode="json", exclude_unset=True) == {}
    assert _build_update_body(enabled=False).model_dump(mode="json", exclude_unset=True) == {"enabled": False}


def test_update_run_status_sdk_body_omits_unset_fields_and_keeps_explicit_null_timestamps() -> None:
    empty = _build_status_update_body()
    partial = _build_status_update_body(
        status=AnalysisConfigStatus.RUNNING,
        last_submitted_job="job-123",
        last_error="",
    )
    clear_timestamps = _build_status_update_body(
        last_successful_run_at=None,
        last_attempted_at=None,
        last_completed_at=None,
    )

    assert empty.model_dump(mode="json", exclude_unset=True) == {}
    assert partial.model_dump(mode="json", exclude_unset=True) == {
        "status": "running",
        "last_submitted_job": "job-123",
        "last_error": "",
    }
    assert clear_timestamps.model_dump(mode="json", exclude_unset=True) == {
        "last_successful_run_at": None,
        "last_attempted_at": None,
        "last_completed_at": None,
    }
