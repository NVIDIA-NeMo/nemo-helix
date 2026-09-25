# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the saved customization job template routes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi import APIRouter, FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from nemo_customizer.api.v2 import templates
from nemo_customizer.api.v2.schemas import CustomizationJobTemplatePage
from nemo_customizer.entities import CustomizationJobTemplate
from nemo_customizer.router import CustomizationRouterService
from nemo_platform_plugin.authz_discovery import _derive_service_contribution
from nemo_platform_plugin.authz_merge import merge_authz_contributions
from nemo_platform_plugin.entity_client import (
    NemoEntityConflictError,
    NemoEntityNotFoundError,
    get_entity_client,
)
from nemo_platform_plugin.service import RouterSpec

WORKSPACE = "default"
BASE = f"/v2/workspaces/{WORKSPACE}/job-templates"
ENDPOINT = "/apis/customization/v2/workspaces/{workspace}/job-templates"

STATIC_AUTHZ = Path("services/core/auth/src/nmp/core/auth/assets/static-authz.yaml")


class _FakeContributor:
    """Minimal backend, so the router builds without depending on what is installed."""

    name = "fake"
    dependencies: list[str] = []

    def get_routers(self) -> list[RouterSpec]:
        return [RouterSpec(router=APIRouter(), prefix="/v2/workspaces/{workspace}/fake", tag="Fake")]


class FakeEntityClient:
    def __init__(self) -> None:
        self.entities: dict[str, CustomizationJobTemplate] = {}
        self.deleted: list[str] = []
        self.update_error: Exception | None = None
        self.delete_error: Exception | None = None

    async def create(self, entity: CustomizationJobTemplate) -> CustomizationJobTemplate:
        if entity.name in self.entities:
            raise NemoEntityConflictError(entity.name)
        self.entities[entity.name] = entity
        return entity

    async def get(self, _model: Any, name: str, workspace: str) -> CustomizationJobTemplate:
        if name not in self.entities:
            raise NemoEntityNotFoundError(f"{workspace}/{name}")
        return self.entities[name]

    async def update(self, entity: CustomizationJobTemplate) -> CustomizationJobTemplate:
        if self.update_error is not None:
            raise self.update_error
        self.entities[entity.name] = entity
        return entity

    async def delete(self, _model: Any, name: str, workspace: str) -> None:
        if self.delete_error is not None:
            raise self.delete_error
        self.deleted.append(f"{workspace}/{name}")
        self.entities.pop(name, None)

    async def list(self, _model: Any, **_kwargs: Any) -> Any:
        from types import SimpleNamespace

        return SimpleNamespace(data=list(self.entities.values()), pagination=None)


@pytest.fixture
def entities() -> FakeEntityClient:
    return FakeEntityClient()


@pytest.fixture
def client(entities: FakeEntityClient) -> TestClient:
    app = FastAPI()
    app.include_router(templates.router, prefix="/v2/workspaces/{workspace}")
    app.dependency_overrides[get_entity_client] = lambda: entities
    return TestClient(app)


def body(**overrides: Any) -> dict[str, Any]:
    return {
        "name": overrides.pop("name", "t1"),
        "backend": overrides.pop("backend", "stub"),
        "config": overrides.pop("config", {"model": "meta/llama"}),
        **overrides,
    }


class TestCrud:
    def test_creates_and_reads_back(self, client: TestClient) -> None:
        assert client.post(BASE, json=body()).status_code == 201

        got = client.get(f"{BASE}/t1").json()
        assert got["name"] == "t1"
        assert got["config"] == {"model": "meta/llama"}

    def test_rejects_a_duplicate_name(self, client: TestClient) -> None:
        client.post(BASE, json=body())

        assert client.post(BASE, json=body()).status_code == 409

    def test_missing_template_is_404(self, client: TestClient) -> None:
        for call in (client.get, client.delete):
            assert call(f"{BASE}/nope").status_code == 404

    def test_patch_leaves_unsent_fields_alone(self, client: TestClient) -> None:
        client.post(BASE, json=body(description="first"))

        got = client.patch(f"{BASE}/t1", json={"description": "second"}).json()

        assert got["description"] == "second"
        assert got["config"] == {"model": "meta/llama"}

    def test_stores_the_config_as_given(self, client: TestClient) -> None:
        """The router does not interpret the config, so an unknown shape round-trips."""
        odd = {"nested": {"deep": [{"x": None}]}, "future_key": "value"}
        client.post(BASE, json=body(config=odd))

        assert client.get(f"{BASE}/t1").json()["config"] == odd

    def test_an_explicit_null_is_ignored_not_stored(self, client: TestClient, entities: FakeEntityClient) -> None:
        """`description` is not nullable on the entity, so a null must not reach it."""
        client.post(BASE, json=body(description="first"))

        res = client.patch(f"{BASE}/t1", json={"description": None})

        assert res.status_code == 200
        assert entities.entities["t1"].description == "first"

    def test_update_conflict_is_409(self, client: TestClient, entities: FakeEntityClient) -> None:
        client.post(BASE, json=body())
        entities.update_error = NemoEntityConflictError("changed")

        res = client.patch(f"{BASE}/t1", json={"description": "second"})

        assert res.status_code == 409

    def test_deletes(self, client: TestClient, entities: FakeEntityClient) -> None:
        client.post(BASE, json=body())

        assert client.delete(f"{BASE}/t1").status_code == 204
        assert entities.deleted == [f"{WORKSPACE}/t1"]

    def test_delete_raced_missing_is_404(self, client: TestClient, entities: FakeEntityClient) -> None:
        client.post(BASE, json=body())
        entities.delete_error = NemoEntityNotFoundError("gone")

        res = client.delete(f"{BASE}/t1")

        assert res.status_code == 404

    def test_list_route_declares_page_response_model(self) -> None:
        route = next(
            route
            for route in templates.router.routes
            if isinstance(route, APIRoute) and route.path == "/job-templates" and "GET" in route.methods
        )

        assert route.response_model is CustomizationJobTemplatePage


@pytest.fixture
def router_service(monkeypatch: pytest.MonkeyPatch) -> CustomizationRouterService:
    """The router with deterministic discovery, so these assertions are about templates."""
    monkeypatch.setattr(
        "nemo_customizer.router.discover_customization_contributors",
        lambda: {"fake": _FakeContributor()},
    )
    return CustomizationRouterService()


class TestAuthz:
    """Permissions are derived from the routes; nothing is hand-registered."""

    def test_each_route_declares_its_own_permission(self, router_service: CustomizationRouterService) -> None:
        contrib, problems, _ = _derive_service_contribution(router_service)

        assert problems == []
        assert contrib.endpoints[ENDPOINT]["get"].permissions == ["customization.job-templates.list"]
        assert contrib.endpoints[ENDPOINT]["post"].permissions == ["customization.job-templates.create"]
        item = f"{ENDPOINT}/{{name}}"
        assert contrib.endpoints[item]["get"].permissions == ["customization.job-templates.read"]
        assert contrib.endpoints[item]["patch"].permissions == ["customization.job-templates.update"]
        assert contrib.endpoints[item]["delete"].permissions == ["customization.job-templates.delete"]

    @pytest.mark.skipif(not STATIC_AUTHZ.exists(), reason="static authz assets not available")
    def test_roles_are_granted_without_editing_static_authz(self, router_service: CustomizationRouterService) -> None:
        """Studio's 403 on the entity route is only fixed if real roles end up with these."""
        base = yaml.safe_load(STATIC_AUTHZ.read_text())
        contrib, _, _ = _derive_service_contribution(router_service)

        merged = merge_authz_contributions(base, [contrib.to_dict()])

        def granted(role: str) -> set[str]:
            perms = merged["authz"]["roles"][role].get("permissions") or []
            return {p.rsplit(".", 1)[-1] for p in perms if "job-templates" in p}

        assert granted("Viewer") == {"list", "read"}
        assert granted("Editor") == {"list", "read", "create", "update", "delete"}
