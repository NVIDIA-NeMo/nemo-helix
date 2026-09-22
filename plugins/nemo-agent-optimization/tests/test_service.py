# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The plugin's own API segment: the job collection, plus the strategies listing."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from nemo_agent_optimization_plugin import service as service_module
from nemo_agent_optimization_plugin.jobs.run_strategy import RunStrategyJob
from nemo_agent_optimization_plugin.schemas.strategies import STRATEGIES_PATH, OptimizationStrategy
from nemo_agent_optimization_plugin.service import (
    STRATEGIES_LIST_PERMISSION,
    AgentOptimizationService,
)
from nemo_helix_plugin.authz import get_path_scope
from nemo_helix_plugin.job import job_collection_path_for


def _routers() -> list[Any]:
    return AgentOptimizationService().get_routers()


def _paths(method: str) -> set[str]:
    paths: set[str] = set()
    for spec in _routers():
        for route in spec.router.routes:
            if not isinstance(route, APIRoute) or method not in (route.methods or set()):
                continue
            paths.add(f"/apis/{AgentOptimizationService.name}{spec.prefix}{route.path}")
    return paths


def _strategies_router() -> APIRouter:
    for spec in _routers():
        if spec.prefix == "/v2":
            return spec.router
    raise AssertionError("no /v2 router mounted")


def _client(monkeypatch: pytest.MonkeyPatch, strategies: list[OptimizationStrategy]) -> TestClient:
    monkeypatch.setattr(service_module, "discover_strategies", lambda: strategies)
    app = FastAPI()
    app.include_router(_strategies_router(), prefix="/apis/agent-optimization/v2")
    return TestClient(app)


def test_the_service_name_is_the_api_segment() -> None:
    assert AgentOptimizationService.name == "agent-optimization"


def test_the_run_strategy_submit_route_is_mounted() -> None:
    collection = f"/apis/agent-optimization/v2/workspaces/{{workspace}}{job_collection_path_for(RunStrategyJob)}"
    assert collection in _paths("POST")


def test_the_strategies_route_is_not_workspace_scoped() -> None:
    """Which strategies exist is an install property, identical in every workspace."""
    assert "/apis/agent-optimization/v2/strategies" in _paths("GET")


def test_the_path_constant_matches_the_mounted_route() -> None:
    """The CLI builds its URL from STRATEGIES_PATH, so the two must not drift."""
    assert STRATEGIES_PATH in _paths("GET")


def test_strategies_lists_what_the_platform_has_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each entry is the declaration its own plugin made, forwarded unchanged."""
    installed = [
        OptimizationStrategy(name="acme", description="Something else."),
        OptimizationStrategy(name="nat", description="Numeric HPO over a Fabric agent workflow."),
    ]

    response = _client(monkeypatch, installed).get("/apis/agent-optimization/v2/strategies")

    assert response.status_code == 200
    assert response.json() == {
        "data": [
            {"name": "acme", "description": "Something else."},
            {"name": "nat", "description": "Numeric HPO over a Fabric agent workflow."},
        ]
    }


def test_strategies_is_empty_rather_than_absent_when_none_are_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The route belongs to the router, not to any strategy, so it answers either way."""
    response = _client(monkeypatch, []).get("/apis/agent-optimization/v2/strategies")

    assert response.status_code == 200
    assert response.json() == {"data": []}


def test_the_strategies_route_is_guarded() -> None:
    """A read scope and a list permission, like every other route on this service."""
    assert STRATEGIES_LIST_PERMISSION.id == "agent-optimization.strategies.list"

    route = next(r for r in _strategies_router().routes if isinstance(r, APIRoute) and r.path == "/strategies")
    assert get_path_scope(route.endpoint) == ["agent-optimization:read", "platform:read"]
