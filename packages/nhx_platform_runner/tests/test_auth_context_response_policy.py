# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Auth-context response policy survives service mounting and real auth middleware."""

from types import SimpleNamespace

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.routing import iter_route_contexts
from fastapi.testclient import TestClient
from nemo_helix_plugin.api.response_policy import PolicyRoute, PolicyRoutes
from nemo_helix_plugin.auth.access import AuthContextResponsePolicy
from nhx.common.auth.middleware import AuthorizationMiddleware
from nhx.common.config import AuthConfig
from nhx.common.service import RouterConfig, Service
from nhx.platform_runner import server
from pydantic import BaseModel


class SnapshotItem(BaseModel):
    name: str
    auth_context: dict[str, str] | None = None


class SnapshotService(Service):
    def __init__(self) -> None:
        super().__init__(name="snapshot-test", module_name="test.snapshot")

    def get_routers(self) -> list[RouterConfig]:
        router = APIRouter()
        routes = PolicyRoutes(router, AuthContextResponsePolicy(dto_type=SnapshotItem, field="auth_context"))

        @routes.get("/item")
        async def get_item() -> SnapshotItem:
            return SnapshotItem(name="item", auth_context={"principal_id": "creator"})

        return [RouterConfig(router, tag="Snapshots", description="Snapshot tests", prefix="/v2")]


def _registered_count(app: FastAPI) -> int:
    return sum(isinstance(context.original_route, PolicyRoute) for context in iter_route_contexts(app.router.routes))


@pytest.mark.parametrize(
    ("auth_enabled", "headers", "visible"),
    [
        (True, {"X-NHX-Principal-Id": "service:models"}, True),
        (
            True,
            {"X-NHX-Principal-Id": "service:models", "X-NHX-Principal-On-Behalf-Of": "user@example.com"},
            False,
        ),
        (True, {"X-NHX-Principal-Id": "user@example.com"}, False),
        (False, {}, True),
    ],
)
def test_platform_projects_registered_response(monkeypatch, auth_enabled, headers, visible) -> None:
    auth_config = AuthConfig(enabled=auth_enabled)
    platform_config = SimpleNamespace(seed_on_startup=False, redirect_root_to_studio=False, services="")
    monkeypatch.setattr(server, "get_platform_config", lambda: platform_config)
    monkeypatch.setattr(server, "get_auth_config", lambda: auth_config)
    monkeypatch.setattr("nhx.common.auth.middleware.get_auth_config", lambda: auth_config)

    async def allow_request(self, auth_client, request, scopes, *, anonymous_denial_is_401):
        return None

    monkeypatch.setattr(AuthorizationMiddleware, "_call_pdp", allow_request)
    service = SnapshotService()
    assert _registered_count(service.app) == 1
    app = server.create_app(services=[service])
    assert _registered_count(app) == 1

    response = TestClient(app).get("/apis/snapshot-test/v2/item", headers=headers)

    assert response.status_code == 200, response.text
    assert ("auth_context" in response.json()) is visible
