# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Service-surface tests: the war-game job route is mounted and its authz derives cleanly.

These guard the regression where the service exposed runs/manifests but never mounted the job
collection, so ``POST /jobs`` (the Studio "Run war-game" path) 404'd, and the derivation gate that
every route carries a ``@path_rule`` (the OPA bundle fails closed otherwise).
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_agent_hardener_plugin.service import AgentHardenerPluginService


def test_service_declares_jobs_dependency() -> None:
    assert "jobs" in AgentHardenerPluginService().dependencies


def test_service_routes_include_war_game_jobs_path() -> None:
    service = AgentHardenerPluginService()
    app = FastAPI()
    for spec in service.get_routers():
        app.include_router(spec.router, prefix=spec.prefix)

    spec = TestClient(app).get("/openapi.json").json()

    assert "/v2/workspaces/{workspace}/jobs" in spec["paths"]
    assert "post" in spec["paths"]["/v2/workspaces/{workspace}/jobs"]
    assert "WarGameJobRequest" in spec["components"]["schemas"]


def test_service_authz_derives_from_routes() -> None:
    """Authz is derived from the ``@path_rule``/``@scope`` stamps on every route (there is no
    ``get_authz_contribution``). Doubles as the derivation gate: the service must derive with no
    problems (every route ruled) and no fail-closed DENY bindings.
    """
    from nemo_platform_plugin.authz_discovery import _derive_service_contribution

    contribution, problems, _warnings = _derive_service_contribution(AgentHardenerPluginService())

    assert problems == []
    assert not any(spec.deny for methods in contribution.endpoints.values() for spec in methods.values())

    # Job collection (scope.child("jobs") → agent-hardener.jobs.*) plus route permissions.
    for perm_id in (
        "agent-hardener.jobs.create",
        "agent-hardener.jobs.list",
        "agent-hardener.runs.list",
        "agent-hardener.runs.events.write",
        "agent-hardener.manifests.write",
        "agent-hardener.manifests.inspect",
    ):
        assert perm_id in contribution.permissions

    base = "/apis/agent-hardener/v2/workspaces/{workspace}"
    assert contribution.endpoints[f"{base}/jobs"]["post"].permissions == ["agent-hardener.jobs.create"]
    assert contribution.endpoints[f"{base}/runs"]["get"].permissions == ["agent-hardener.runs.list"]
    assert contribution.endpoints[f"{base}/runs"]["get"].scopes == ["agent-hardener:read", "platform:read"]
    assert contribution.endpoints[f"{base}/runs/{{name}}/events"]["post"].permissions == [
        "agent-hardener.runs.events.write"
    ]
