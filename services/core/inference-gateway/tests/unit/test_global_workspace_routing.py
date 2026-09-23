# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Routing against the global workspace (ASTD-526)."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from fastapi import HTTPException, Request
from nemo_platform.types.inference import ModelProvider, ServedModelMapping
from nemo_platform.types.inference.virtual_model import VirtualModel
from nemo_platform_plugin.inference_middleware import (
    ImmediateResponse,
    InferenceMiddlewareContext,
    InferenceRequest,
    NemoInferenceMiddleware,
)
from nmp.common.auth.dependencies import auth_client_context
from nmp.common.auth.models import Principal
from nmp.common.entities.global_workspace import workspace_lookup_order
from nmp.core.inference_gateway.api.authz import OPENAI_EXEC_PERMISSION
from nmp.core.inference_gateway.api.middleware_registry import MiddlewareRegistry, ResolvedMiddlewareCall
from nmp.core.inference_gateway.api.model_cache import (
    ModelCache,
    ModelEntityInfo,
    ModelProviderInfo,
    refresh_model_cache,
)
from nmp.core.inference_gateway.api.proxy import virtual_model_proxy
from nmp.core.inference_gateway.api.v2.openai import resolve_vm_for_model, resolve_vm_for_request
from nmp.core.inference_gateway.api.virtual_model_cache import VirtualModelCache


def _make_vm(workspace: str, name: str) -> VirtualModel:
    return VirtualModel(
        id=f"{workspace}/{name}",
        entity_id=f"{workspace}/{name}",
        name=name,
        workspace=workspace,
        parent=workspace,
        db_version=1,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        default_model_entity=f"{workspace}/{name}",
    )


def test_lookup_order_is_local_then_global() -> None:
    assert workspace_lookup_order("team-a") == ("team-a", "default")


def test_lookup_order_does_not_repeat_global() -> None:
    assert workspace_lookup_order("default") == ("default",)


class TestVirtualModelCache:
    def test_global_vm_resolves_from_another_workspace(self) -> None:
        cache = VirtualModelCache()
        cache.rebuild([_make_vm("default", "shared-llm")])
        assert cache.resolve("team-a", "shared-llm") is not None

    def test_get_does_not_fall_back_to_the_global_workspace(self) -> None:
        cache = VirtualModelCache()
        cache.rebuild([_make_vm("default", "shared-llm")])
        assert cache.get("team-a", "shared-llm") is None

    def test_local_vm_shadows_global(self) -> None:
        cache = VirtualModelCache()
        cache.rebuild([_make_vm("default", "llm"), _make_vm("team-a", "llm")])
        resolved = cache.resolve("team-a", "llm")
        assert resolved is not None
        assert resolved.workspace == "team-a"

    def test_workspace_scoped_vm_does_not_leak_to_other_workspaces(self) -> None:
        cache = VirtualModelCache()
        cache.rebuild([_make_vm("team-a", "private-llm")])
        assert cache.resolve("team-b", "private-llm") is None

    def test_unknown_name_still_misses(self) -> None:
        cache = VirtualModelCache()
        cache.rebuild([_make_vm("default", "shared-llm")])
        assert cache.resolve("team-a", "nope") is None


class TestResolveVMForModel:
    def test_global_vm_resolves_for_plain_name(self) -> None:
        cache = VirtualModelCache()
        cache.rebuild([_make_vm("default", "shared-llm")])
        assert resolve_vm_for_model(cache, ModelCache(), "team-a", "shared-llm") is not None

    def test_lora_composite_resolves_through_global_base(self) -> None:
        """An adapter fine-tuned in a workspace routes via the shared base model's VM."""
        cache = VirtualModelCache()
        cache.rebuild([_make_vm("default", "base-llm")])
        resolved = resolve_vm_for_model(cache, ModelCache(), "team-a", "base-llm&adapters/team-a/my-adapter")
        assert resolved is not None
        assert resolved.workspace == "default"

    def test_served_local_model_without_a_vm_does_not_fall_back_to_the_global_vm(self) -> None:
        cache = VirtualModelCache()
        cache.rebuild([_make_vm("default", "base-llm")])
        model_cache = ModelCache()
        model_cache.model_entity_info_map[("team-a", "base-llm")] = ModelEntityInfo(workspace="team-a", name="base-llm")
        assert resolve_vm_for_model(cache, model_cache, "team-a", "base-llm") is None
        assert resolve_vm_for_model(cache, model_cache, "team-a", "base-llm&adapters/team-a/my-adapter") is None


class TestModelCache:
    def _provider_info(self, workspace: str, name: str) -> ModelProviderInfo:
        return ModelProviderInfo(
            model_provider=ModelProvider(
                workspace=workspace,
                name=name,
                host_url="http://localhost:8080",
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
        )

    def test_global_provider_resolves_from_another_workspace(self) -> None:
        cache = ModelCache()
        cache.update_model_info(self._provider_info("default", "shared-provider"))
        assert cache.resolve_provider("team-a", "shared-provider") is not None

    def test_get_from_provider_does_not_fall_back_to_the_global_workspace(self) -> None:
        cache = ModelCache()
        cache.update_model_info(self._provider_info("default", "shared-provider"))
        assert cache.get_from_provider("team-a", "shared-provider") is None

    def test_local_provider_shadows_global(self) -> None:
        cache = ModelCache()
        cache.update_model_info(self._provider_info("default", "prov"))
        cache.update_model_info(self._provider_info("team-a", "prov"))
        resolved = cache.resolve_provider("team-a", "prov")
        assert resolved is not None
        assert resolved.model_provider.workspace == "team-a"

    def test_private_provider_does_not_leak(self) -> None:
        cache = ModelCache()
        cache.update_model_info(self._provider_info("team-a", "prov"))
        assert cache.resolve_provider("team-b", "prov") is None

    def test_qualified_model_entity_is_not_redirected_to_the_global_workspace(self) -> None:
        cache = ModelCache()
        cache.model_entity_info_map[("default", "shared-llm")] = ModelEntityInfo(workspace="default", name="shared-llm")
        assert cache.get_from_model_entity("team-a", "shared-llm") is None


class TestResolveVMForRequest:
    """A shared VirtualModel resolves only for a caller entitled in the global workspace."""

    @pytest.fixture(autouse=True)
    def _user(self):
        client = MagicMock()
        client.auth_enabled = True
        client.principal = Principal(id="user:bob", email="bob@example.com")
        client.has_permissions = AsyncMock(return_value=False)
        token = auth_client_context.set(client)
        yield client
        auth_client_context.reset(token)

    async def test_shared_vm_is_hidden_from_a_caller_without_access_to_the_global_workspace(self, _user) -> None:
        cache = VirtualModelCache()
        cache.rebuild([_make_vm("default", "shared-llm")])

        assert await resolve_vm_for_request(cache, ModelCache(), "team-a", "shared-llm", OPENAI_EXEC_PERMISSION) is None
        _user.has_permissions.assert_awaited_once_with("default", [OPENAI_EXEC_PERMISSION])

    async def test_shared_vm_resolves_for_an_entitled_caller(self, _user) -> None:
        _user.has_permissions.return_value = True
        cache = VirtualModelCache()
        cache.rebuild([_make_vm("default", "shared-llm")])

        resolved = await resolve_vm_for_request(cache, ModelCache(), "team-a", "shared-llm", OPENAI_EXEC_PERMISSION)

        assert resolved is not None
        assert resolved.workspace == "default"

    async def test_local_vm_needs_no_access_to_the_global_workspace(self, _user) -> None:
        cache = VirtualModelCache()
        cache.rebuild([_make_vm("team-a", "llm")])

        assert await resolve_vm_for_request(cache, ModelCache(), "team-a", "llm", OPENAI_EXEC_PERMISSION) is not None
        _user.has_permissions.assert_not_awaited()


def _provider(workspace: str, name: str, host_url: str, served_models: list[ServedModelMapping] | None = None):
    return ModelProvider(
        workspace=workspace,
        name=name,
        host_url=host_url,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        served_models=served_models or [],
    )


class TestRefreshModelCache:
    async def test_same_named_provider_does_not_inherit_the_global_providers_cache_entry(self) -> None:
        cache = ModelCache()
        global_info = ModelProviderInfo(model_provider=_provider("default", "nim", "http://global.nim"))
        global_info.secret_value = "global-key"
        cache.update_model_info(global_info)

        providers = [_provider("default", "nim", "http://global.nim"), _provider("team-a", "nim", "http://team-a.nim")]
        await refresh_model_cache(cache, AsyncMock(return_value=providers), secrets_sdk=MagicMock())

        global_after = cache.workspace_name_provider_map[("default", "nim")]
        local_after = cache.workspace_name_provider_map[("team-a", "nim")]
        assert global_after is not local_after
        assert global_after.model_provider.host_url == "http://global.nim"
        assert local_after.secret_value != "global-key"


class _RecordingPlugin(NemoInferenceMiddleware):
    def __init__(self) -> None:
        self.calls = 0

    async def process_request(
        self,
        ctx: InferenceMiddlewareContext,
        request: InferenceRequest,
        middleware_config: object,
    ) -> ImmediateResponse:
        self.calls += 1
        return ImmediateResponse(data={"ok": True})


def _request() -> Mock:
    request = Mock(spec=Request)
    request.method = "POST"
    request.headers = {"content-type": "application/json"}
    request.query_params = {}
    return request


async def _proxy(workspace: str, vm: VirtualModel, body_model: str, registry: MiddlewareRegistry, model_cache=None):
    assert vm.name is not None
    return await virtual_model_proxy(
        request=_request(),
        workspace=workspace,
        vm_name=vm.name,
        virtual_model=vm,
        trailing_uri="v1/chat/completions",
        json_body={"model": body_model, "messages": [{"role": "user", "content": "hi"}]},
        http_client=MagicMock(),
        model_cache=model_cache or ModelCache(),
        registry=registry,
    )


class TestSharedVirtualModelMiddleware:
    async def test_shared_vm_runs_its_request_middleware_for_another_workspace(self) -> None:
        plugin = _RecordingPlugin()
        registry = MiddlewareRegistry(plugins={"guard": plugin})
        registry.request_middleware_calls[("default", "llm")] = [
            ResolvedMiddlewareCall(plugin_name="guard", config_type="t", resolved_config={})
        ]

        await _proxy("team-a", _make_vm("default", "llm"), "llm", registry)

        assert plugin.calls == 1

    async def test_broken_shared_vm_fails_closed_for_another_workspace(self) -> None:
        registry = MiddlewareRegistry(plugins={})
        registry.broken_vms.add(("default", "llm"))

        with pytest.raises(HTTPException) as exc_info:
            await _proxy("team-a", _make_vm("default", "llm"), "llm", registry)

        assert exc_info.value.status_code == 503


class TestProxyModelEntityAccess:
    @pytest.fixture(autouse=True)
    def _user(self):
        client = MagicMock()
        client.auth_enabled = True
        client.principal = Principal(id="user:bob", email="bob@example.com")
        client.has_permissions = AsyncMock(return_value=False)
        token = auth_client_context.set(client)
        yield client
        auth_client_context.reset(token)

    async def test_body_naming_the_global_workspace_is_checked_against_the_route_workspace(self, _user) -> None:
        model_cache = ModelCache()
        model_cache.update_model_info(
            ModelProviderInfo(
                model_provider=_provider(
                    "default",
                    "nim",
                    "http://nim.local",
                    [ServedModelMapping(model_entity_id="default/foo", served_model_name="foo")],
                )
            )
        )
        model_cache.rebuild_model_entity_map()
        vm = _make_vm("team-a", "passthrough").model_copy(update={"default_model_entity": None})

        with pytest.raises(HTTPException) as exc_info:
            await _proxy("team-a", vm, "default/foo", MiddlewareRegistry(plugins={}), model_cache)

        assert exc_info.value.status_code == 404
        _user.has_permissions.assert_awaited_once_with("default", [OPENAI_EXEC_PERMISSION])
