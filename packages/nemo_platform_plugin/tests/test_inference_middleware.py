# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for nemo_platform_plugin.inference_middleware."""

from __future__ import annotations

import dataclasses
from typing import Any, get_args
from unittest.mock import MagicMock, patch

import anthropic.types as anthropic_types
import httpx
import openai.types.chat as openai_chat_types
import pytest
from nemo_platform_plugin.client.client import AsyncNemoClient
from nemo_platform_plugin.discovery import (
    _ALL_SURFACE_GROUPS,
    discover,
    discover_entry_points,
    discover_inference_middleware,
    discover_manifests,
)
from nemo_platform_plugin.inference_middleware import (
    BackendFormat,
    ImmediateResponse,
    InferenceMiddlewareCacheAccessor,
    InferenceMiddlewareContext,
    InferenceMiddlewareError,
    InferenceMiddlewareUnavailableError,
    InferenceRequest,
    InferenceResponse,
    ModelProviderInferenceTarget,
    NemoInferenceMiddleware,
    OpenAICompatibleInferenceTarget,
)
from nemo_platform_plugin.inference_middleware_models import MiddlewareCall, VirtualModel, VirtualModelInferenceConfig
from nemo_platform_plugin.inference_middleware_types import TypedResponse
from pydantic import TypeAdapter, ValidationError

# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_discovery_cache():
    discover_entry_points.cache_clear()
    discover.cache_clear()
    discover_manifests.cache_clear()
    yield
    discover_entry_points.cache_clear()
    discover.cache_clear()
    discover_manifests.cache_clear()


# ---------------------------------------------------------------------------
# Helpers and fixtures
# ---------------------------------------------------------------------------


class _MinimalMiddleware(NemoInferenceMiddleware):
    """Bare subclass — no overrides; used to verify defaults and inject cache."""


def _inject_mock_cache(plugin: NemoInferenceMiddleware) -> MagicMock:
    """Inject a mock InferenceMiddlewareCacheAccessor and return it."""
    cache = MagicMock(spec=InferenceMiddlewareCacheAccessor)
    plugin._inject_cache(cache)
    return cache


def _make_ep(name: str, value: object) -> MagicMock:
    ep = MagicMock()
    ep.name = name
    ep.value = f"nemo_test.{name}:Cls"
    ep.load.return_value = value
    return ep


@pytest.fixture
def plugin() -> _MinimalMiddleware:
    """Bare NemoInferenceMiddleware subclass with no cache injected."""
    return _MinimalMiddleware()


@pytest.fixture
def plugin_with_cache(plugin: _MinimalMiddleware) -> tuple[_MinimalMiddleware, MagicMock]:
    """Bare plugin with a mock InferenceMiddlewareCacheAccessor already injected."""
    cache = _inject_mock_cache(plugin)
    return plugin, cache


# ---------------------------------------------------------------------------
# TestNemoInferenceMiddlewareContract
# ---------------------------------------------------------------------------


class TestNemoInferenceMiddlewareContract:
    def test_can_instantiate_bare_subclass(self, plugin):
        assert isinstance(plugin, NemoInferenceMiddleware)

    async def test_process_request_default_returns_request_unchanged(self, plugin):
        body = {"model": "ws/llama", "messages": []}
        request = InferenceRequest(body=body, headers={}, path="v1/chat/completions")
        ctx = InferenceMiddlewareContext(
            request_id="test-id",
            virtual_model_name="test-vm",
            workspace="ws",
            original_request=InferenceRequest(body=dict(body), headers={}, path="v1/chat/completions"),
        )
        result = await plugin.process_request(ctx, request, middleware_config=None)
        assert result is request

    async def test_process_response_default_is_passthrough(self, plugin):
        ctx = InferenceMiddlewareContext(
            request_id="test-id",
            virtual_model_name="test-vm",
            workspace="ws",
            original_request=InferenceRequest(body={}, headers={}, path=""),
        )
        response = InferenceResponse(result={"choices": []}, headers={})
        result = await plugin.process_response(ctx, response, middleware_config=None)
        assert result is response

    async def test_lifecycle_hooks_default_to_noop(self, plugin):
        vm = VirtualModel(name="test", workspace="ws")
        await plugin.on_startup()
        await plugin.on_shutdown()
        await plugin.on_virtual_model_upserted(vm)
        await plugin.on_virtual_model_destroyed(vm)

    def test_immediate_response_is_dataclass(self):
        response = ImmediateResponse(data={"choices": []})

        assert dataclasses.is_dataclass(ImmediateResponse)
        assert response.response_body_annotations == {}

    def test_inference_response_is_dataclass(self):
        response = InferenceResponse(result={"choices": []}, headers={"x-test": "1"})

        assert dataclasses.is_dataclass(InferenceResponse)
        assert response.result == {"choices": []}
        assert response.headers == {"x-test": "1"}
        assert response.typed_body is None
        assert response.response_body_annotations == {}

    def test_inference_middleware_context_includes_backend_format(self):
        context = InferenceMiddlewareContext(
            request_id="test-id",
            virtual_model_name="test-vm",
            workspace="ws",
            original_request=InferenceRequest(body={"model": "ws/llama"}, headers={}, path="v1/chat/completions"),
            backend_format=BackendFormat.OPENAI_CHAT,
        )

        assert context.backend_format is BackendFormat.OPENAI_CHAT
        assert context.response_body_annotations == {}

    def test_typed_response_type_alias_uses_sdk_classes(self):
        assert openai_chat_types.ChatCompletion in get_args(TypedResponse)
        assert anthropic_types.Message in get_args(TypedResponse)

    def test_immediate_response_is_distinct_from_dict(self):
        wrapped = ImmediateResponse(data={"choices": []})
        assert isinstance(wrapped, ImmediateResponse)
        assert not isinstance(wrapped, dict)
        assert isinstance(wrapped.data, dict)

    async def test_process_request_can_return_immediate_response(self):
        class _ShortCircuitMiddleware(NemoInferenceMiddleware):
            async def process_request(self, ctx, request, middleware_config) -> ImmediateResponse:
                return ImmediateResponse(data={"id": "test", "choices": []})

        ctx = InferenceMiddlewareContext(
            request_id="x",
            virtual_model_name="vm",
            workspace="ws",
            original_request=InferenceRequest(body={}, headers={}, path=""),
        )
        request = InferenceRequest(body={}, headers={}, path="")
        result = await _ShortCircuitMiddleware().process_request(ctx, request, middleware_config=None)
        assert isinstance(result, ImmediateResponse)
        assert result.data == {"id": "test", "choices": []}

    def test_model_provider_inference_target_is_dataclass(self):
        assert dataclasses.is_dataclass(ModelProviderInferenceTarget)

    def test_model_provider_inference_target_fields(self):
        target = ModelProviderInferenceTarget(
            model_provider_gateway_url="http://igw/provider/ws/nim/-/v1",
            served_model_name="meta/llama-3.1-70b-instruct",
        )
        assert target.model_provider_gateway_url == "http://igw/provider/ws/nim/-/v1"
        assert target.served_model_name == "meta/llama-3.1-70b-instruct"

    def test_openai_compatible_inference_target_is_dataclass(self):
        assert dataclasses.is_dataclass(OpenAICompatibleInferenceTarget)

    def test_openai_compatible_inference_target_fields(self):
        target = OpenAICompatibleInferenceTarget(
            openai_base_url="http://igw/apis/inference-gateway/v2/workspaces/ws/openai/-/v1",
            model="ws/llama-3.1-70b-instruct",
        )
        assert target.openai_base_url == "http://igw/apis/inference-gateway/v2/workspaces/ws/openai/-/v1"
        assert target.model == "ws/llama-3.1-70b-instruct"


# ---------------------------------------------------------------------------
# TestCacheAccessorsBeforeInjection
# ---------------------------------------------------------------------------


class TestCacheAccessorsBeforeInjection:
    @pytest.mark.parametrize(
        "method,args",
        [
            ("get_model_providers_for_model", ("ws/llama",)),
            ("get_model_entity", ("ws/llama",)),
            ("list_model_entities_for_workspace", ()),
            ("get_virtual_model", ("ws/my-model",)),
            ("list_virtual_models_for_workspace", ("ws",)),
            ("get_inference_url_and_model", ("ws/llama",)),
            ("get_backend_format", ("ws/my-model", "ws/llama")),
            ("get_openai_compatible_inference_url_and_model", ("ws/llama",)),
        ],
    )
    def test_raises_runtime_error_before_injection(self, plugin, method, args):
        with pytest.raises(RuntimeError, match=method):
            getattr(plugin, method)(*args)


# ---------------------------------------------------------------------------
# TestCacheAccessorsAfterInjection
# ---------------------------------------------------------------------------


class TestCacheAccessorsAfterInjection:
    @pytest.mark.parametrize(
        "method,call_args,expected_cache_args,expected_cache_kwargs",
        [
            ("get_model_providers_for_model", ("ws/llama",), ("ws/llama",), {}),
            ("get_model_entity", ("ws/llama",), ("ws/llama",), {}),
            ("list_model_entities_for_workspace", ("ws",), ("ws",), {}),
            ("get_virtual_model", ("ws/smart-llama",), ("ws/smart-llama",), {}),
            ("list_virtual_models_for_workspace", ("ws",), ("ws",), {}),
            # append_v1_suffix defaults to True, so the cache is called with both args
            ("get_inference_url_and_model", ("ws/llama-70b",), ("ws/llama-70b", True), {}),
            (
                "get_backend_format",
                ("ws/smart-router", "ws/llama-70b"),
                ("ws/smart-router", "ws/llama-70b"),
                {},
            ),
            ("get_openai_compatible_inference_url_and_model", ("ws/llama-70b",), ("ws/llama-70b",), {}),
        ],
    )
    def test_delegates_to_cache(
        self,
        plugin_with_cache,
        method,
        call_args,
        expected_cache_args,
        expected_cache_kwargs,
    ):
        plugin, cache = plugin_with_cache
        sentinel = MagicMock()
        getattr(cache, method).return_value = sentinel
        result = getattr(plugin, method)(*call_args)
        getattr(cache, method).assert_called_once_with(*expected_cache_args, **expected_cache_kwargs)
        assert result is sentinel

    @pytest.mark.parametrize(
        "method,call_args",
        [
            ("get_model_entity", ("ws/nonexistent",)),
            ("get_virtual_model", ("ws/nonexistent",)),
        ],
    )
    def test_returns_none_when_not_found(self, plugin_with_cache, method, call_args):
        plugin, cache = plugin_with_cache
        getattr(cache, method).return_value = None
        assert getattr(plugin, method)(*call_args) is None


# ---------------------------------------------------------------------------
# TestPluginImplementedConfigMethods
# ---------------------------------------------------------------------------


class TestPluginImplementedConfigMethods:
    async def test_get_middleware_config_raises_not_implemented_by_default(self, plugin):
        with pytest.raises(NotImplementedError, match="get_middleware_config"):
            await plugin.get_middleware_config("routellm_config", "ws/cfg")

    async def test_validate_middleware_config_default_passthrough(self, plugin):
        cfg = {"type": "routellm", "threshold": 0.6}
        result = await plugin.validate_middleware_config("routellm_config", cfg)
        assert result == cfg

    async def test_get_middleware_config_can_be_overridden(self):
        class _PluginWithConfig(NemoInferenceMiddleware):
            async def get_middleware_config(self, config_type: str, config_id: str) -> Any:
                return {"fetched": True, "config_type": config_type, "id": config_id}

        result = await _PluginWithConfig().get_middleware_config("my_config", "ws/my-cfg")
        assert result == {"fetched": True, "config_type": "my_config", "id": "ws/my-cfg"}

    async def test_validate_middleware_config_can_raise_value_error(self):
        class _StrictPlugin(NemoInferenceMiddleware):
            async def validate_middleware_config(self, config_type: str, config: Any) -> Any:
                if config_type != "known_type":
                    raise ValueError(f"Unknown config_type: {config_type!r}")
                return config

        with pytest.raises(ValueError, match="Unknown config_type"):
            await _StrictPlugin().validate_middleware_config("bad_type", {})

    def test_get_platform_client_requires_injection(self, plugin: _MinimalMiddleware):
        with pytest.raises(RuntimeError, match="platform client"):
            plugin._get_platform_client("on_startup")

    async def test_get_platform_client_returns_injected_client(self, plugin: _MinimalMiddleware):
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, request=request))
        ) as http_client:
            client = AsyncNemoClient(base_url="http://test", http_client=http_client)
            plugin._inject_platform_client(client)

            assert plugin._get_platform_client("on_startup") is client


# ---------------------------------------------------------------------------
# TestInferenceMiddlewareExceptions
# ---------------------------------------------------------------------------


class TestInferenceMiddlewareExceptions:
    def test_base_error_defaults_to_500(self):
        err = InferenceMiddlewareError("something went wrong")
        assert err.status_code == 500
        assert err.detail == "something went wrong"
        assert str(err) == "something went wrong"

    def test_base_error_custom_status_code(self):
        err = InferenceMiddlewareError("rate limited", status_code=429)
        assert err.status_code == 429

    def test_unavailable_error_defaults_to_503(self):
        err = InferenceMiddlewareUnavailableError()
        assert err.status_code == 503

    def test_unavailable_error_custom_detail(self):
        err = InferenceMiddlewareUnavailableError("classifier is down")
        assert err.detail == "classifier is down"
        assert err.status_code == 503

    def test_unavailable_error_is_subclass_of_base(self):
        assert issubclass(InferenceMiddlewareUnavailableError, InferenceMiddlewareError)

    async def test_plugin_can_raise_middleware_error(self):
        class _BrokenMiddleware(NemoInferenceMiddleware):
            async def process_request(self, ctx, request, middleware_config):
                raise InferenceMiddlewareUnavailableError("classifier service is down")

        ctx = InferenceMiddlewareContext(
            request_id="x",
            virtual_model_name="vm",
            workspace="ws",
            original_request=InferenceRequest(body={}, headers={}, path=""),
        )
        request = InferenceRequest(body={}, headers={}, path="")
        with pytest.raises(InferenceMiddlewareUnavailableError, match="classifier service is down"):
            await _BrokenMiddleware().process_request(ctx, request, middleware_config=None)


# ---------------------------------------------------------------------------
# TestMiddlewareCall
# ---------------------------------------------------------------------------


class TestMiddlewareCall:
    def test_requires_name_and_config_type(self):
        call = MiddlewareCall(name="nemo-switchyard", config_type="routellm_config", config={})
        assert call.name == "nemo-switchyard"
        assert call.config_type == "routellm_config"
        assert call.config == {}
        assert call.config_id is None

    def test_inline_config(self):
        call = MiddlewareCall(
            name="nemo-switchyard",
            config_type="routellm_config",
            config={"router_type": "bert", "threshold": 0.6},
        )
        assert call.config == {"router_type": "bert", "threshold": 0.6}
        assert call.config_id is None

    def test_config_id_reference(self):
        call = MiddlewareCall(
            name="nemo-guardrails",
            config_type="guardrail_config",
            config_id="my-workspace/content-safety",
        )
        assert call.config_id == "my-workspace/content-safety"
        assert call.config is None

    def test_config_type_required(self):
        with pytest.raises(ValidationError):
            MiddlewareCall(name="nemo-switchyard")  # ty: ignore[missing-argument]

    @pytest.mark.parametrize(
        "payload",
        [
            {"name": "nemo-switchyard", "config_type": "routellm_config"},
            {
                "name": "nemo-switchyard",
                "config_type": "routellm_config",
                "config": {},
                "config_id": "default/router",
            },
            {
                "name": "nemo-switchyard",
                "config_type": "routellm_config",
                "config_id": "default/router",
                "config": {},
            },
        ],
    )
    def test_requires_exactly_one_config_source(self, payload: dict[str, object]):
        with pytest.raises(ValidationError, match="Exactly one of config or config_id"):
            MiddlewareCall.model_validate(payload)


# ---------------------------------------------------------------------------
# TestVirtualModel
# ---------------------------------------------------------------------------


class TestVirtualModel:
    def test_entity_type(self):
        assert VirtualModel.__entity_type__ == "virtual_model"

    def test_defaults(self):
        vm = VirtualModel(name="test", workspace="ws")
        assert vm.default_model_entity is None
        assert vm.models == []
        assert vm.request_middleware == []
        assert vm.response_middleware == []
        assert vm.post_response_middleware == []
        assert vm.override_proxy is None

    def test_with_middleware_calls(self):
        call = MiddlewareCall(
            name="nemo-switchyard",
            config_type="routellm_config",
            config={"threshold": 0.6},
        )
        vm = VirtualModel(
            name="smart-llama",
            workspace="my-workspace",
            default_model_entity="my-workspace/llama-3-1-8b",
            models=[
                VirtualModelInferenceConfig(
                    model="my-workspace/llama-3-1-8b",
                    backend_format=BackendFormat.OPENAI_CHAT,
                )
            ],
            request_middleware=[call],
        )
        assert vm.models[0].backend_format is BackendFormat.OPENAI_CHAT
        assert len(vm.request_middleware) == 1
        assert vm.request_middleware[0].name == "nemo-switchyard"
        assert vm.request_middleware[0].config_type == "routellm_config"

    def test_passthrough_virtual_model(self):
        vm = VirtualModel(
            name="llama-3-1-8b",
            workspace="my-workspace",
            default_model_entity="my-workspace/llama-3-1-8b",
        )
        assert vm.request_middleware == []
        assert vm.response_middleware == []
        assert vm.post_response_middleware == []

    def test_validation_retains_wire_metadata(self):
        """Entity metadata carried by an API response survives validation."""
        vm = TypeAdapter(VirtualModel).validate_python(
            {
                "id": "virtual-model-abc123",
                "name": "router",
                "workspace": "default",
                "created_at": "2026-01-01T00:00:00Z",
                "created_by": "creator",
                "updated_at": "2026-01-02T00:00:00Z",
                "updated_by": "updater",
                "parent": "parent-id",
            }
        )

        assert vm.id == "virtual-model-abc123"
        assert vm.created_at is not None
        assert vm.created_by == "creator"
        assert vm.updated_at is not None
        assert vm.updated_by == "updater"
        assert vm.parent == "parent-id"

    def test_validation_without_wire_metadata_leaves_entity_defaults(self):
        vm = TypeAdapter(VirtualModel).validate_python({"name": "router", "workspace": "default"})

        assert vm.id == ""
        assert vm.created_at is None
        assert vm.parent is None


# ---------------------------------------------------------------------------
# TestDiscoverInferenceMiddleware
# ---------------------------------------------------------------------------


class TestDiscoverInferenceMiddleware:
    def test_nemo_inference_middleware_in_all_surface_groups(self):
        assert "nemo.inference_middleware" in _ALL_SURFACE_GROUPS

    def test_uses_nemo_inference_middleware_group(self):
        with patch("nemo_platform_plugin.discovery.entry_points", return_value=[]) as mock_eps:
            discover_inference_middleware()
        mock_eps.assert_called_once_with(group="nemo.inference_middleware")

    def test_loads_middleware_class_keyed_by_entry_point_key(self):
        ep = _make_ep("nemo-switchyard", _MinimalMiddleware)
        with patch("nemo_platform_plugin.discovery.entry_points", return_value=[ep]):
            result = discover_inference_middleware()
        assert result["nemo-switchyard"] is _MinimalMiddleware

    def test_returned_class_is_instantiable(self):
        ep = _make_ep("nemo-switchyard", _MinimalMiddleware)
        with patch("nemo_platform_plugin.discovery.entry_points", return_value=[ep]):
            result = discover_inference_middleware()
        assert isinstance(result["nemo-switchyard"](), NemoInferenceMiddleware)

    def test_failing_plugin_is_skipped(self):
        bad = _make_ep("broken", None)
        bad.load.side_effect = ImportError("missing dep")
        good = _make_ep("nemo-switchyard", _MinimalMiddleware)
        with patch("nemo_platform_plugin.discovery.entry_points", return_value=[bad, good]):
            result = discover_inference_middleware()
        assert "broken" not in result
        assert "nemo-switchyard" in result

    def test_discover_manifests_includes_inference_middleware_plugins(self):
        ep = MagicMock()
        ep.name = "nemo-switchyard"
        ep.value = "nemo_switchyard:Middleware"
        dist = MagicMock()
        dist.metadata.get = lambda k, d="": {"Version": "1.0.0", "Summary": "Switchyard"}.get(k, d)
        ep.dist = dist

        with patch(
            "nemo_platform_plugin.discovery.entry_points",
            side_effect=lambda group: [ep] if group == "nemo.inference_middleware" else [],
        ):
            manifests = discover_manifests()

        assert "nemo-switchyard" in manifests
        assert manifests["nemo-switchyard"].version == "1.0.0"
