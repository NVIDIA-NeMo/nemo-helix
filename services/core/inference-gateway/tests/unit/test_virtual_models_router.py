# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for VirtualModel CRUD endpoints.

Uses ``create_test_client`` with a real in-process SQLite entity store so that
CRUD semantics (conflict detection, optimistic locking, pagination, etc.) are
exercised against the actual entity client implementation rather than a mock.

The inference gateway's model-cache background refresh is disabled
(``refresh_model_cache_interval_sec=0``) so tests don't need a running
models service.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient
from nemo_platform_plugin.client.errors import ConflictError, NotFoundError
from nemo_platform_plugin.inference_middleware import InferenceMiddlewareError, NemoInferenceMiddleware
from nemo_platform_plugin.virtual_models.client import AsyncVirtualModelsClient
from nemo_platform_plugin.virtual_models.types import CreateVirtualModelRequest, UpdateVirtualModelRequest
from nmp.core.inference_gateway.api.dependencies import global_middleware_registry
from nmp.core.inference_gateway.api.middleware_registry import MiddlewareRegistry
from nmp.core.inference_gateway.config import InferenceGatewayConfig
from nmp.core.inference_gateway.service import InferenceGatewayService
from nmp.testing import create_test_client

# Base URL prefix for the inference-gateway service
BASE = "/apis/inference-gateway/v2/workspaces/default/virtual-models"


@pytest.fixture
def client() -> TestClient:
    """TestClient backed by a real in-process entity store.

    Model-cache refresh is disabled so no connection to the models service
    is required during tests.
    """
    with create_test_client(
        InferenceGatewayService,
        client_type=TestClient,
        service_configs={
            InferenceGatewayService: InferenceGatewayConfig(
                refresh_model_cache_interval_sec=0,
                mock_provider_prefix="igw-mock-",
            )
        },
    ) as tc:
        yield tc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create(client: TestClient, name: str, **kwargs) -> dict:
    """POST a VirtualModel and assert 201."""
    payload = {"name": name, **kwargs}
    resp = client.post(BASE, json=payload)
    assert resp.status_code == 201, resp.text
    return resp.json()


def _make_plugin() -> NemoInferenceMiddleware:
    plugin = MagicMock(spec=NemoInferenceMiddleware)
    plugin.get_middleware_config = AsyncMock(return_value={"stored": True})
    plugin.validate_middleware_config = AsyncMock(side_effect=lambda _config_type, config: config)
    return plugin


def _install_registry(client: TestClient, plugins: dict[str, NemoInferenceMiddleware]) -> MiddlewareRegistry:
    registry = MiddlewareRegistry(plugins=plugins)
    client.app.dependency_overrides[global_middleware_registry] = lambda: registry
    return registry


@asynccontextmanager
async def _typed_client(client: TestClient) -> AsyncIterator[AsyncVirtualModelsClient]:
    """Public typed client wired to the mounted IGW app over an in-process ASGI transport."""
    transport = httpx.ASGITransport(app=client.app)
    async with httpx.AsyncClient(transport=transport) as http_client:
        yield AsyncVirtualModelsClient(
            base_url="http://testserver",
            workspace="default",
            http_client=http_client,
        )


# ---------------------------------------------------------------------------
# Typed client contract
# ---------------------------------------------------------------------------


class TestTypedClientContract:
    """The public typed client and the mounted IGW router share one wire contract.

    These run the real client against the real router and entity store, so they
    catch drift that mocked-transport client tests cannot: status codes, the
    pagination envelope, ``response_model_exclude_none`` on list responses, and
    the filter expression grammar.
    """

    @pytest.mark.asyncio
    async def test_create_and_get_roundtrip_carries_entity_metadata(self, client: TestClient) -> None:
        async with _typed_client(client) as typed:
            created = (
                await typed.create_virtual_model(
                    body=CreateVirtualModelRequest(
                        name="vm-typed-client",
                        default_model_entity="default/model-a",
                    )
                )
            ).data()
            fetched = (await typed.get_virtual_model(name="vm-typed-client")).data()

        assert created.name == "vm-typed-client"
        assert created.id is not None
        assert created.created_at is not None
        assert created.created_by is not None
        assert fetched.id == created.id
        assert fetched.updated_at is not None
        assert fetched.default_model_entity == "default/model-a"

    @pytest.mark.asyncio
    async def test_list_follows_pagination_and_keeps_metadata(self, client: TestClient) -> None:
        """Server pages are followed to exhaustion; list responses omit None fields."""
        async with _typed_client(client) as typed:
            for name in ("vm-a", "vm-b", "vm-c"):
                await typed.create_virtual_model(body=CreateVirtualModelRequest(name=name))

            listed = [model async for model in (await typed.list_virtual_models(query_params={"page_size": 1})).items()]

        assert sorted(model.name for model in listed) == ["vm-a", "vm-b", "vm-c"]
        assert all(model.id is not None for model in listed)
        assert all(model.created_at is not None for model in listed)

    @pytest.mark.asyncio
    async def test_list_query_params_reach_the_router(self, client: TestClient) -> None:
        """``exclude_autoprovisioned`` and the filter expression are server-parseable as sent."""
        async with _typed_client(client) as typed:
            await typed.create_virtual_model(body=CreateVirtualModelRequest(name="vm-keep"))
            await typed.create_virtual_model(body=CreateVirtualModelRequest(name="vm-auto", autoprovisioned=True))

            unprovisioned = await typed.list_virtual_models(query_params={"exclude_autoprovisioned": True})
            assert [model.name async for model in unprovisioned.items()] == ["vm-keep"]

            filtered = await typed.list_virtual_models(query_params={"filter": 'name:"vm-keep"'})
            assert [model.name async for model in filtered.items()] == ["vm-keep"]

    @pytest.mark.asyncio
    async def test_update_then_delete_then_get_raises_not_found(self, client: TestClient) -> None:
        async with _typed_client(client) as typed:
            await typed.create_virtual_model(
                body=CreateVirtualModelRequest(name="vm-up", default_model_entity="default/model-a")
            )

            updated = (
                await typed.update_virtual_model(
                    name="vm-up", body=UpdateVirtualModelRequest(default_model_entity=None)
                )
            ).data()
            assert updated.default_model_entity is None
            assert updated.id is not None

            deleted = await typed.delete_virtual_model(name="vm-up")
            assert deleted.http_response.status_code == 204
            assert deleted.data() is None

            with pytest.raises(NotFoundError) as not_found:
                await typed.get_virtual_model(name="vm-up")
            assert not_found.value.status_code == 404

    @pytest.mark.asyncio
    async def test_duplicate_create_raises_conflict(self, client: TestClient) -> None:
        async with _typed_client(client) as typed:
            await typed.create_virtual_model(body=CreateVirtualModelRequest(name="vm-dup"))

            with pytest.raises(ConflictError) as conflict:
                await typed.create_virtual_model(body=CreateVirtualModelRequest(name="vm-dup"))
        assert conflict.value.status_code == 409


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------


class TestCreateVirtualModel:
    def test_create_minimal(self, client: TestClient):
        """Only name is required; all pipeline lists default to empty."""
        data = _create(client, "vm-minimal")
        assert data["name"] == "vm-minimal"
        assert data["workspace"] == "default"
        assert data["request_middleware"] == []
        assert data["response_middleware"] == []
        assert data["post_response_middleware"] == []
        assert data["models"] == []
        assert data["default_model_entity"] is None
        assert data["autoprovisioned"] is False
        assert data["override_proxy"] is None
        assert "id" in data
        assert "created_at" in data
        assert "updated_at" in data

    def test_create_with_all_fields(self, client: TestClient):
        """All optional fields are persisted correctly."""
        _install_registry(
            client,
            {
                "nemo-switchyard": _make_plugin(),
                "nemo-guardrails": _make_plugin(),
                "nemo-logger": _make_plugin(),
            },
        )
        data = _create(
            client,
            "vm-full",
            default_model_entity="default/llama-70b",
            request_middleware=[{"name": "nemo-switchyard", "config_type": "routellm_config", "config": {}}],
            response_middleware=[
                {"name": "nemo-guardrails", "config_type": "guardrail_config", "config_id": "default/safe"}
            ],
            post_response_middleware=[{"name": "nemo-logger", "config_type": "log_config", "config": {}}],
            models=[{"model": "default/claude-sonnet", "backend_format": "ANTHROPIC_MESSAGES"}],
            override_proxy="nemo-switchyard.http-proxy",
        )
        assert data["name"] == "vm-full"
        assert data["default_model_entity"] == "default/llama-70b"
        assert data["models"] == [{"model": "default/claude-sonnet", "backend_format": "ANTHROPIC_MESSAGES"}]
        assert len(data["request_middleware"]) == 1
        assert data["request_middleware"][0]["name"] == "nemo-switchyard"
        assert len(data["response_middleware"]) == 1
        assert len(data["post_response_middleware"]) == 1
        assert data["override_proxy"] == "nemo-switchyard.http-proxy"

    def test_create_validates_all_middleware_phases(self, client: TestClient):
        """POST validates every middleware call before persisting the VM."""
        plugin = _make_plugin()
        _install_registry(client, {"my-plugin": plugin})

        _create(
            client,
            "vm-validates",
            request_middleware=[{"name": "my-plugin", "config_type": "req", "config": {"phase": "request"}}],
            response_middleware=[{"name": "my-plugin", "config_type": "resp", "config": {"phase": "response"}}],
            post_response_middleware=[{"name": "my-plugin", "config_type": "post", "config": {"phase": "post"}}],
        )

        assert plugin.validate_middleware_config.await_count == 3
        plugin.validate_middleware_config.assert_any_await("req", {"phase": "request"})
        plugin.validate_middleware_config.assert_any_await("resp", {"phase": "response"})
        plugin.validate_middleware_config.assert_any_await("post", {"phase": "post"})

    def test_create_resolves_config_id_before_validation(self, client: TestClient):
        """POST resolves config_id through the plugin and validates the returned config."""
        plugin = _make_plugin()
        plugin.get_middleware_config = AsyncMock(return_value={"from": "store"})
        _install_registry(client, {"my-plugin": plugin})

        _create(
            client,
            "vm-config-id",
            request_middleware=[{"name": "my-plugin", "config_type": "stored_config", "config_id": "default/cfg"}],
        )

        plugin.get_middleware_config.assert_awaited_once_with("stored_config", "default/cfg")
        plugin.validate_middleware_config.assert_awaited_once_with("stored_config", {"from": "store"})

    def test_create_unknown_middleware_plugin_returns_422(self, client: TestClient):
        """POST rejects middleware references to plugins IGW did not load."""
        _install_registry(client, {})

        resp = client.post(
            BASE,
            json={
                "name": "vm-missing-plugin",
                "request_middleware": [{"name": "missing-plugin", "config_type": "cfg", "config": {}}],
            },
        )

        assert resp.status_code == 422
        assert "missing-plugin" in resp.json()["detail"]
        assert client.get(f"{BASE}/vm-missing-plugin").status_code == 404

    def test_create_invalid_middleware_config_returns_422(self, client: TestClient):
        """POST rejects plugin validation failures before persistence."""
        plugin = _make_plugin()
        plugin.validate_middleware_config = AsyncMock(side_effect=ValueError("bad config"))
        _install_registry(client, {"my-plugin": plugin})

        resp = client.post(
            BASE,
            json={
                "name": "vm-invalid-config",
                "request_middleware": [{"name": "my-plugin", "config_type": "cfg", "config": {"invalid": True}}],
            },
        )

        assert resp.status_code == 422
        assert "bad config" in resp.json()["detail"]
        assert client.get(f"{BASE}/vm-invalid-config").status_code == 404

    def test_create_middleware_config_error_returns_422(self, client: TestClient):
        """Handled plugin validation errors are treated as invalid VM config."""
        plugin = _make_plugin()
        plugin.validate_middleware_config = AsyncMock(
            side_effect=InferenceMiddlewareError("not allowed", status_code=409)
        )
        _install_registry(client, {"my-plugin": plugin})

        resp = client.post(
            BASE,
            json={
                "name": "vm-plugin-error",
                "request_middleware": [{"name": "my-plugin", "config_type": "cfg", "config": {}}],
            },
        )

        assert resp.status_code == 422
        assert "not allowed" in resp.json()["detail"]

    def test_create_middleware_config_server_error_returns_status_code(self, client: TestClient):
        """Plugin validation 5xx errors remain server failures."""
        plugin = _make_plugin()
        plugin.validate_middleware_config = AsyncMock(
            side_effect=InferenceMiddlewareError("dependency unavailable", status_code=503)
        )
        _install_registry(client, {"my-plugin": plugin})

        resp = client.post(
            BASE,
            json={
                "name": "vm-plugin-server-error",
                "request_middleware": [{"name": "my-plugin", "config_type": "cfg", "config": {}}],
            },
        )

        assert resp.status_code == 503
        assert "dependency unavailable" in resp.json()["detail"]

    def test_create_config_id_without_plugin_support_returns_422(self, client: TestClient):
        """POST rejects config_id when the referenced plugin does not implement config lookup."""
        plugin = _make_plugin()
        plugin.get_middleware_config = AsyncMock(side_effect=NotImplementedError("inline only"))
        _install_registry(client, {"my-plugin": plugin})

        resp = client.post(
            BASE,
            json={
                "name": "vm-no-config-id",
                "request_middleware": [{"name": "my-plugin", "config_type": "cfg", "config_id": "default/cfg"}],
            },
        )

        assert resp.status_code == 422
        assert "config_id" in resp.json()["detail"]

    def test_create_bad_config_id_returns_422(self, client: TestClient):
        """POST treats plugin config_id lookup ValueErrors as validation failures."""
        plugin = _make_plugin()
        plugin.get_middleware_config = AsyncMock(side_effect=ValueError("config not found"))
        _install_registry(client, {"my-plugin": plugin})

        resp = client.post(
            BASE,
            json={
                "name": "vm-bad-config-id",
                "request_middleware": [{"name": "my-plugin", "config_type": "cfg", "config_id": "default/missing"}],
            },
        )

        assert resp.status_code == 422
        assert "config not found" in resp.json()["detail"]

    def test_create_missing_config_id_returns_404(self, client: TestClient):
        """A definitive 404 from the plugin's store surfaces as a 404 to the caller.

        ``MiddlewareConfigNotFoundError`` is the typed signal plugins raise when
        the referenced entity does not exist. Mapping it to 404 lets the caller
        distinguish "I referenced something that doesn't exist" from generic
        validation failures (422). It also lines up with the same exception
        being the eviction trigger for IGW's resolved-middleware cache.
        """
        from nemo_platform_plugin.inference_middleware import MiddlewareConfigNotFoundError

        plugin = _make_plugin()
        plugin.get_middleware_config = AsyncMock(side_effect=MiddlewareConfigNotFoundError("default/missing"))
        _install_registry(client, {"my-plugin": plugin})

        resp = client.post(
            BASE,
            json={
                "name": "vm-deleted-config-id",
                "request_middleware": [{"name": "my-plugin", "config_type": "cfg", "config_id": "default/missing"}],
            },
        )

        assert resp.status_code == 404
        assert "default/missing" in resp.json()["detail"]

    def test_create_config_id_middleware_server_error_returns_status_code(self, client: TestClient):
        """Plugin config_id lookup 5xx errors remain server failures."""
        plugin = _make_plugin()
        plugin.get_middleware_config = AsyncMock(side_effect=InferenceMiddlewareError("lookup failed", status_code=503))
        _install_registry(client, {"my-plugin": plugin})

        resp = client.post(
            BASE,
            json={
                "name": "vm-config-id-plugin-error",
                "request_middleware": [{"name": "my-plugin", "config_type": "cfg", "config_id": "default/cfg"}],
            },
        )

        assert resp.status_code == 503
        assert "lookup failed" in resp.json()["detail"]

    def test_create_autoprovisioned(self, client: TestClient):
        """autoprovisioned can be set by callers that opt into controller-managed cleanup."""
        resp = client.post(
            BASE,
            json={"name": "vm-auto", "autoprovisioned": True, "default_model_entity": "default/model-a"},
        )
        assert resp.status_code == 201, resp.text
        data = resp.json()
        assert data["autoprovisioned"] is True
        assert data["default_model_entity"] == "default/model-a"

    def test_create_duplicate_returns_409(self, client: TestClient):
        """Creating two VirtualModels with the same name in the same workspace → 409."""
        _create(client, "vm-dup")
        resp = client.post(BASE, json={"name": "vm-dup"})
        assert resp.status_code == 409
        assert "already exists" in resp.json()["detail"]

    def test_create_missing_name_returns_422(self, client: TestClient):
        """Name is required; omitting it produces a validation error."""
        resp = client.post(BASE, json={})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Get
# ---------------------------------------------------------------------------


class TestGetVirtualModel:
    def test_get_existing(self, client: TestClient):
        """GET returns the entity that was just created."""
        _create(client, "vm-get")
        resp = client.get(f"{BASE}/vm-get")
        assert resp.status_code == 200
        assert resp.json()["name"] == "vm-get"

    def test_get_not_found_returns_404(self, client: TestClient):
        """GET for a non-existent name → 404."""
        resp = client.get(f"{BASE}/does-not-exist")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------


class TestListVirtualModels:
    def test_list_empty(self, client: TestClient):
        """List returns expected pagination envelope even when there are no entities."""
        resp = client.get(BASE)
        assert resp.status_code == 200
        body = resp.json()
        assert "data" in body
        assert "pagination" in body
        assert "sort" in body

    def test_list_returns_created_entities(self, client: TestClient):
        """Entities created in this workspace appear in the list."""
        _create(client, "vm-list-a")
        _create(client, "vm-list-b")
        resp = client.get(BASE)
        assert resp.status_code == 200
        names = {item["name"] for item in resp.json()["data"]}
        assert "vm-list-a" in names
        assert "vm-list-b" in names

    def test_list_pagination(self, client: TestClient):
        """page_size is respected and pagination metadata is accurate."""
        for i in range(5):
            _create(client, f"vm-pg-{i}")
        resp = client.get(f"{BASE}?page=1&page_size=2")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["data"]) <= 2
        assert body["pagination"]["page_size"] == 2

    def test_list_sort_reflected_in_response(self, client: TestClient):
        """The sort query param value is echoed in the response."""
        resp = client.get(f"{BASE}?sort=created_at")
        assert resp.status_code == 200
        assert resp.json()["sort"] == "created_at"

    def test_list_excludes_autoprovisioned(self, client: TestClient):
        """exclude_autoprovisioned=true excludes controller-managed VirtualModels."""
        _create(client, "vm-manual", default_model_entity="default/model-a")
        _create(client, "vm-auto", autoprovisioned=True, default_model_entity="default/model-b")

        resp = client.get(f"{BASE}?exclude_autoprovisioned=true")
        assert resp.status_code == 200, resp.text
        names = {item["name"] for item in resp.json()["data"]}
        assert "vm-manual" in names
        assert "vm-auto" not in names

    def test_list_filters_by_name_like(self, client: TestClient):
        """filter[name][$like] matches VirtualModels by substring."""
        _create(client, "vm-alpha")
        _create(client, "vm-beta")

        resp = client.get(f"{BASE}?filter[name][$like]=alph")
        assert resp.status_code == 200, resp.text
        names = {item["name"] for item in resp.json()["data"]}
        assert names == {"vm-alpha"}

    def _guardrail_call(self, config_id: str) -> dict:
        """A middleware call applying the stored guardrail config ``config_id``."""
        return {"name": "nemo-guardrails", "config_type": "guardrail_config", "config_id": config_id}

    def test_list_filters_by_guardrail_config(self, client: TestClient):
        """filter[guardrail_config] selects VirtualModels that apply that stored config."""
        _install_registry(client, {"nemo-guardrails": _make_plugin()})
        _create(client, "vm-guarded", request_middleware=[self._guardrail_call("default/content-safety")])
        _create(client, "vm-plain", default_model_entity="default/model-a")

        resp = client.get(f"{BASE}?filter[guardrail_config]=default/content-safety")
        assert resp.status_code == 200, resp.text
        assert {item["name"] for item in resp.json()["data"]} == {"vm-guarded"}

    def test_list_filters_by_guardrail_config_across_all_phases(self, client: TestClient):
        """A config attached to any pipeline phase matches; output rails are not request rails."""
        _install_registry(client, {"nemo-guardrails": _make_plugin()})
        _create(client, "vm-req", request_middleware=[self._guardrail_call("default/cs")])
        _create(client, "vm-resp", response_middleware=[self._guardrail_call("default/cs")])
        _create(client, "vm-post", post_response_middleware=[self._guardrail_call("default/cs")])

        resp = client.get(f"{BASE}?filter[guardrail_config]=default/cs")
        assert resp.status_code == 200, resp.text
        assert {item["name"] for item in resp.json()["data"]} == {"vm-req", "vm-resp", "vm-post"}

    def test_list_guardrail_config_filter_does_not_match_other_configs(self, client: TestClient):
        """A different config reference, and a same-named config of another type, both miss."""
        _install_registry(client, {"nemo-guardrails": _make_plugin(), "nemo-switchyard": _make_plugin()})
        _create(client, "vm-other-config", request_middleware=[self._guardrail_call("default/other")])
        _create(
            client,
            "vm-other-type",
            request_middleware=[
                {"name": "nemo-switchyard", "config_type": "routellm_config", "config_id": "default/cs"}
            ],
        )

        resp = client.get(f"{BASE}?filter[guardrail_config]=default/cs")
        assert resp.status_code == 200, resp.text
        # vm-other-type carries the same reference under a different config_type. Both rows have a
        # denormalized list, so the raw-text legacy arms are gated off and matching is exact.
        assert {item["name"] for item in resp.json()["data"]} == set()

    def test_guardrail_config_ids_derived_from_pipelines(self, client: TestClient):
        """The system-managed reference list is derived on create and deduped across phases."""
        _install_registry(client, {"nemo-guardrails": _make_plugin()})
        data = _create(
            client,
            "vm-derived",
            request_middleware=[self._guardrail_call("default/cs")],
            response_middleware=[self._guardrail_call("default/cs")],
        )
        assert data["guardrail_config_ids"] == ["default/cs"]

    def test_guardrail_config_ids_refreshed_on_patch(self, client: TestClient):
        """PATCH re-derives the list even though model_copy bypasses validation."""
        _install_registry(client, {"nemo-guardrails": _make_plugin()})
        _create(client, "vm-patched", request_middleware=[self._guardrail_call("default/cs")])

        resp = client.patch(
            f"{BASE}/vm-patched",
            json={"request_middleware": [self._guardrail_call("default/stricter")]},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["guardrail_config_ids"] == ["default/stricter"]

        assert {i["name"] for i in client.get(f"{BASE}?filter[guardrail_config]=default/stricter").json()["data"]} == {
            "vm-patched"
        }

    def test_guardrail_config_ids_cleared_when_rails_detached(self, client: TestClient):
        """Clearing the pipeline clears the derived list, so the config stops matching."""
        _install_registry(client, {"nemo-guardrails": _make_plugin()})
        _create(client, "vm-detached", request_middleware=[self._guardrail_call("default/cs")])

        resp = client.patch(f"{BASE}/vm-detached", json={"request_middleware": []})
        assert resp.status_code == 200, resp.text
        assert resp.json()["guardrail_config_ids"] == []

        names = {i["name"] for i in client.get(f"{BASE}?filter[guardrail_config]=default/cs").json()["data"]}
        assert "vm-detached" not in names

    def test_list_includes_autoprovisioned_by_default(self, client: TestClient):
        """Autoprovisioned VirtualModels are returned when the flag is not set."""
        _create(client, "vm-manual-2", default_model_entity="default/model-a")
        _create(client, "vm-auto-2", autoprovisioned=True, default_model_entity="default/model-b")

        names = {item["name"] for item in client.get(BASE).json()["data"]}
        assert {"vm-manual-2", "vm-auto-2"} <= names


# ---------------------------------------------------------------------------
# Update (PATCH)
# ---------------------------------------------------------------------------


class TestUpdateVirtualModel:
    def test_update_default_model_entity(self, client: TestClient):
        """PATCH can change default_model_entity."""
        _create(client, "vm-upd", default_model_entity="default/model-a")
        resp = client.patch(f"{BASE}/vm-upd", json={"default_model_entity": "default/model-b"})
        assert resp.status_code == 200
        assert resp.json()["default_model_entity"] == "default/model-b"

    def test_update_middleware_lists(self, client: TestClient):
        """PATCH replaces middleware lists when provided."""
        _create(client, "vm-mw")
        _install_registry(client, {"nemo-guardrails": _make_plugin()})
        mw = [{"name": "nemo-guardrails", "config_type": "guardrail_config", "config": {}}]
        resp = client.patch(f"{BASE}/vm-mw", json={"request_middleware": mw})
        assert resp.status_code == 200
        assert resp.json()["request_middleware"][0]["name"] == "nemo-guardrails"

    def test_update_invalid_middleware_config_returns_422_and_preserves_existing(self, client: TestClient):
        """PATCH rejects invalid middleware updates without mutating the stored VM."""
        _create(client, "vm-update-invalid", default_model_entity="default/original")
        plugin = _make_plugin()
        plugin.validate_middleware_config = AsyncMock(side_effect=ValueError("bad patch"))
        _install_registry(client, {"my-plugin": plugin})

        resp = client.patch(
            f"{BASE}/vm-update-invalid",
            json={
                "default_model_entity": "default/updated",
                "request_middleware": [{"name": "my-plugin", "config_type": "cfg", "config": {"bad": True}}],
            },
        )

        assert resp.status_code == 422
        assert "bad patch" in resp.json()["detail"]

        get_resp = client.get(f"{BASE}/vm-update-invalid")
        assert get_resp.status_code == 200
        data = get_resp.json()
        assert data["default_model_entity"] == "default/original"
        assert data["request_middleware"] == []

    def test_update_unknown_middleware_plugin_returns_422(self, client: TestClient):
        """PATCH rejects middleware references to plugins IGW did not load."""
        _create(client, "vm-update-missing-plugin")
        _install_registry(client, {})

        resp = client.patch(
            f"{BASE}/vm-update-missing-plugin",
            json={"request_middleware": [{"name": "missing-plugin", "config_type": "cfg", "config": {}}]},
        )

        assert resp.status_code == 422
        assert "missing-plugin" in resp.json()["detail"]

    def test_patch_omitted_fields_unchanged(self, client: TestClient):
        """Fields absent from the PATCH body retain their previous values."""
        _create(client, "vm-partial", default_model_entity="default/original", override_proxy="plug.proxy")
        resp = client.patch(f"{BASE}/vm-partial", json={"request_middleware": []})
        assert resp.status_code == 200
        data = resp.json()
        # Fields not in the PATCH body are unchanged
        assert data["default_model_entity"] == "default/original"
        assert data["override_proxy"] == "plug.proxy"

    def test_update_clear_default_model_entity(self, client: TestClient):
        """PATCH with null explicitly clears default_model_entity."""
        _create(client, "vm-clear", default_model_entity="default/model-x")
        resp = client.patch(f"{BASE}/vm-clear", json={"default_model_entity": None})
        assert resp.status_code == 200
        assert resp.json()["default_model_entity"] is None

    def test_update_can_clear_autoprovisioned(self, client: TestClient):
        """PATCH can clear autoprovisioned so a user can adopt a controller-created VM."""
        resp = client.post(
            BASE,
            json={"name": "vm-auto-update", "autoprovisioned": True},
        )
        assert resp.status_code == 201, resp.text

        patch_resp = client.patch(f"{BASE}/vm-auto-update", json={"autoprovisioned": False})
        assert patch_resp.status_code == 200
        assert patch_resp.json()["autoprovisioned"] is False

    def test_update_not_found_returns_404(self, client: TestClient):
        """PATCH for a non-existent name → 404."""
        resp = client.patch(f"{BASE}/ghost", json={"override_proxy": "plug.proxy"})
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


class TestDeleteVirtualModel:
    def test_delete_existing(self, client: TestClient):
        """DELETE returns 204 and the entity can no longer be fetched."""
        _create(client, "vm-del")
        resp = client.delete(f"{BASE}/vm-del")
        assert resp.status_code == 204

        get_resp = client.get(f"{BASE}/vm-del")
        assert get_resp.status_code == 404

    def test_delete_not_found_returns_404(self, client: TestClient):
        """DELETE for a non-existent name → 404."""
        resp = client.delete(f"{BASE}/no-such-vm")
        assert resp.status_code == 404

    def test_delete_with_stale_expected_db_version_returns_409(self, client: TestClient):
        """DELETE with a stale entity version is rejected and leaves the VM intact."""
        created = _create(client, "vm-stale-delete")
        patch_resp = client.patch(f"{BASE}/vm-stale-delete", json={"autoprovisioned": True})
        assert patch_resp.status_code == 200
        assert patch_resp.json()["db_version"] != created["db_version"]

        resp = client.delete(
            f"{BASE}/vm-stale-delete",
            params={"expected_db_version": created["db_version"]},
        )

        assert resp.status_code == 409

        get_resp = client.get(f"{BASE}/vm-stale-delete")
        assert get_resp.status_code == 200


@pytest.fixture
def team_client() -> Iterator[TestClient]:
    """Like ``client``, with a second workspace to address shared VirtualModels through."""
    with create_test_client(
        InferenceGatewayService,
        client_type=TestClient,
        workspaces=["default", "team-a"],
        service_configs={
            InferenceGatewayService: InferenceGatewayConfig(
                refresh_model_cache_interval_sec=0,
                mock_provider_prefix="igw-mock-",
            )
        },
    ) as tc:
        yield tc


class TestSharedVirtualModelWrites:
    """A VirtualModel shared from ``default`` is readable through another workspace, not writable."""

    TEAM = "/apis/inference-gateway/v2/workspaces/team-a/virtual-models"

    def test_update_through_another_workspace_does_not_modify_the_shared_vm(self, team_client: TestClient):
        _create(team_client, "vm-shared", default_model_entity="default/model-a")

        resp = team_client.patch(f"{self.TEAM}/vm-shared", json={"default_model_entity": "default/hijacked"})

        assert resp.status_code == 404, resp.text
        assert team_client.get(f"{BASE}/vm-shared").json()["default_model_entity"] == "default/model-a"

    def test_shared_vm_still_resolves_through_another_workspace(self, team_client: TestClient):
        _create(team_client, "vm-readable", default_model_entity="default/model-a")

        resp = team_client.get(f"{self.TEAM}/vm-readable")

        assert resp.status_code == 200
        assert resp.json()["workspace"] == "default"
