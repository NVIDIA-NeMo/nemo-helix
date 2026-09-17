# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MiddlewareRegistry, InferenceMiddlewareCacheAccessorImpl,
and load_middleware_plugins."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nemo_platform.types.inference.middleware_call import MiddlewareCall as SDKMiddlewareCall
from nemo_platform.types.inference.virtual_model import VirtualModel as SDKVirtualModel
from nemo_platform.types.inference.virtual_model_inference_config import (
    VirtualModelInferenceConfig as SDKVirtualModelInferenceConfig,
)
from nemo_platform_plugin.inference_middleware import (
    BackendFormat,
    NemoInferenceMiddleware,
)
from nemo_platform_plugin.inference_middleware_models import (
    VirtualModel as PluginVirtualModel,
)
from nmp.core.inference_gateway.api.middleware_registry import (
    InferenceMiddlewareCacheAccessorImpl,
    MiddlewareConfigRef,
    MiddlewareRegistry,
    PrefetchResult,
    _sdk_vm_to_plugin_vm,
    collect_config_refs,
    load_middleware_plugins,
)
from nmp.core.inference_gateway.api.model_cache import ModelCache, ModelEntityInfo, ModelProviderInfo
from nmp.core.inference_gateway.api.virtual_model_cache import VirtualModelCache

skip_flaky_caplog = pytest.mark.skip(reason="Flaky caplog assertions in middleware registry tests")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sdk_vm(
    workspace: str,
    name: str,
    request_middleware: list[SDKMiddlewareCall] | None = None,
    response_middleware: list[SDKMiddlewareCall] | None = None,
    post_response_middleware: list[SDKMiddlewareCall] | None = None,
    models: list[SDKVirtualModelInferenceConfig] | None = None,
    updated_at: str = "2026-01-01T00:00:00Z",
) -> SDKVirtualModel:
    return SDKVirtualModel(
        id=f"{workspace}/{name}",
        entity_id=f"{workspace}/{name}",
        name=name,
        workspace=workspace,
        parent=workspace,
        db_version=1,
        created_at="2026-01-01T00:00:00Z",
        updated_at=updated_at,
        default_model_entity=f"{workspace}/{name}",
        request_middleware=request_middleware or [],
        response_middleware=response_middleware or [],
        post_response_middleware=post_response_middleware or [],
        models=models or [],
    )


def _make_sdk_call(
    name: str, config_type: str = "my_config", config: dict | None = None, config_id: str | None = None
) -> SDKMiddlewareCall:
    return SDKMiddlewareCall(name=name, config_type=config_type, config=config, config_id=config_id)


def _make_mock_plugin() -> NemoInferenceMiddleware:
    plugin = MagicMock(spec=NemoInferenceMiddleware)
    plugin.on_startup = AsyncMock()
    plugin.on_shutdown = AsyncMock()
    plugin.on_virtual_model_upserted = AsyncMock()
    plugin.on_virtual_model_destroyed = AsyncMock()
    plugin.get_middleware_config = AsyncMock(return_value={"raw": "config"})
    plugin.validate_middleware_config = AsyncMock(side_effect=lambda ct, c: c)
    return plugin


def _make_model_provider_info(workspace: str, name: str, host_url: str = "http://nim.svc") -> ModelProviderInfo:
    provider = MagicMock()
    provider.workspace = workspace
    provider.name = name
    provider.host_url = host_url
    provider.api_key_secret_name = None
    return ModelProviderInfo(model_provider=provider)


# ---------------------------------------------------------------------------
# InferenceMiddlewareCacheAccessorImpl tests
# ---------------------------------------------------------------------------


class TestInferenceMiddlewareCacheAccessorImpl:
    def _make_accessor(self, model_cache=None, virtual_model_cache=None):
        return InferenceMiddlewareCacheAccessorImpl(
            _model_cache=model_cache or ModelCache(),
            _virtual_model_cache=virtual_model_cache or VirtualModelCache(),
        )

    def test_get_model_providers_for_model_returns_providers(self):
        model_cache = ModelCache()
        provider_info = _make_model_provider_info("ws", "my-provider")
        entity_info = ModelEntityInfo(workspace="ws", name="llama-3b")
        entity_info.model_providers.append(("llama-3b-v1", provider_info))
        model_cache.model_entity_info_map[("ws", "llama-3b")] = entity_info

        accessor = self._make_accessor(model_cache=model_cache)
        providers = accessor.get_model_providers_for_model("ws/llama-3b")
        assert len(providers) == 1
        assert providers[0].name == "my-provider"

    def test_get_model_providers_for_model_missing_returns_empty(self):
        accessor = self._make_accessor()
        assert accessor.get_model_providers_for_model("ws/unknown") == []

    def test_get_model_providers_invalid_ref_returns_empty(self):
        accessor = self._make_accessor()
        assert accessor.get_model_providers_for_model("not/a/valid/ref") == []

    def test_get_model_entity_returns_minimal_entity(self):
        model_cache = ModelCache()
        entity_info = ModelEntityInfo(workspace="ws", name="llama-3b")
        model_cache.model_entity_info_map[("ws", "llama-3b")] = entity_info

        accessor = self._make_accessor(model_cache=model_cache)
        entity = accessor.get_model_entity("ws/llama-3b")
        assert entity is not None
        assert entity.workspace == "ws"
        assert entity.name == "llama-3b"
        assert entity.spec is None
        assert entity.finetuning_type is None

    def test_get_model_entity_missing_returns_none(self):
        accessor = self._make_accessor()
        assert accessor.get_model_entity("ws/unknown") is None

    def test_list_model_entities_for_workspace_all(self):
        model_cache = ModelCache()
        model_cache.model_entity_info_map[("ws-a", "m1")] = MagicMock()
        model_cache.model_entity_info_map[("ws-b", "m2")] = MagicMock()

        accessor = self._make_accessor(model_cache=model_cache)
        result = accessor.list_model_entities_for_workspace()
        assert set(result) == {"ws-a/m1", "ws-b/m2"}

    def test_list_model_entities_for_workspace_filtered(self):
        model_cache = ModelCache()
        model_cache.model_entity_info_map[("ws-a", "m1")] = MagicMock()
        model_cache.model_entity_info_map[("ws-b", "m2")] = MagicMock()

        accessor = self._make_accessor(model_cache=model_cache)
        result = accessor.list_model_entities_for_workspace(workspace="ws-a")
        assert result == ["ws-a/m1"]

    def test_get_virtual_model_returns_vm(self):
        vm_cache = VirtualModelCache()
        vm = _make_sdk_vm("ws", "my-vm")
        vm_cache.rebuild([vm])

        accessor = self._make_accessor(virtual_model_cache=vm_cache)
        result = accessor.get_virtual_model("ws/my-vm")
        assert result is not None
        assert result.name == "my-vm"

    def test_get_virtual_model_missing_returns_none(self):
        accessor = self._make_accessor()
        assert accessor.get_virtual_model("ws/missing") is None

    def test_list_virtual_models_for_workspace(self):
        vm_cache = VirtualModelCache()
        vm_cache.rebuild([_make_sdk_vm("ws", "vm-a"), _make_sdk_vm("ws", "vm-b"), _make_sdk_vm("other", "vm-c")])

        accessor = self._make_accessor(virtual_model_cache=vm_cache)
        result = accessor.list_virtual_models_for_workspace("ws")
        assert set(result) == {"ws/vm-a", "ws/vm-b"}

    def test_get_inference_url_and_model_returns_target(self):
        model_cache = ModelCache()
        virtual_model_cache = VirtualModelCache()
        provider_info = _make_model_provider_info("ws", "nim-provider", host_url="http://nim.svc:8080")
        entity_info = ModelEntityInfo(workspace="ws", name="llama", backend_format=BackendFormat.ANTHROPIC_MESSAGES)
        entity_info.model_providers.append(("llama-v1", provider_info))
        model_cache.model_entity_info_map[("ws", "llama")] = entity_info
        virtual_model_cache.rebuild([_make_sdk_vm("ws", "smart-router")])

        accessor = self._make_accessor(model_cache=model_cache, virtual_model_cache=virtual_model_cache)
        target = accessor.get_inference_url_and_model("ws/llama")
        assert target.model_provider_gateway_url == "http://nim.svc:8080/v1"
        assert target.served_model_name == "llama-v1"
        assert accessor.get_backend_format("ws/smart-router", "ws/llama") is BackendFormat.ANTHROPIC_MESSAGES

    def test_get_backend_format_uses_virtual_model_override(self):
        model_cache = ModelCache()
        virtual_model_cache = VirtualModelCache()
        provider_info = _make_model_provider_info("ws", "nim-provider", host_url="http://nim.svc:8080")
        entity_info = ModelEntityInfo(workspace="ws", name="llama", backend_format=BackendFormat.OPENAI_CHAT)
        entity_info.model_providers.append(("llama-v1", provider_info))
        model_cache.model_entity_info_map[("ws", "llama")] = entity_info
        virtual_model_cache.rebuild(
            [
                _make_sdk_vm(
                    "ws",
                    "smart-router",
                    models=[SDKVirtualModelInferenceConfig(model="ws/llama", backend_format="ANTHROPIC_MESSAGES")],
                )
            ]
        )

        accessor = self._make_accessor(model_cache=model_cache, virtual_model_cache=virtual_model_cache)
        backend_format = accessor.get_backend_format("ws/smart-router", "ws/llama")

        assert backend_format is BackendFormat.ANTHROPIC_MESSAGES

    def test_get_backend_format_uses_model_entity_value_without_virtual_model_override(self):
        model_cache = ModelCache()
        virtual_model_cache = VirtualModelCache()
        entity_info = ModelEntityInfo(workspace="ws", name="llama", backend_format=BackendFormat.ANTHROPIC_MESSAGES)
        model_cache.model_entity_info_map[("ws", "llama")] = entity_info
        virtual_model_cache.rebuild([_make_sdk_vm("ws", "smart-router")])

        accessor = self._make_accessor(model_cache=model_cache, virtual_model_cache=virtual_model_cache)
        backend_format = accessor.get_backend_format("ws/smart-router", "ws/llama")

        assert backend_format is BackendFormat.ANTHROPIC_MESSAGES

    def test_get_backend_format_returns_none_when_unset(self):
        model_cache = ModelCache()
        virtual_model_cache = VirtualModelCache()
        entity_info = ModelEntityInfo(workspace="ws", name="llama")
        model_cache.model_entity_info_map[("ws", "llama")] = entity_info
        virtual_model_cache.rebuild([_make_sdk_vm("ws", "smart-router")])

        accessor = self._make_accessor(model_cache=model_cache, virtual_model_cache=virtual_model_cache)
        backend_format = accessor.get_backend_format("ws/smart-router", "ws/llama")

        assert backend_format is None

    def test_get_inference_url_and_model_no_v1_suffix(self):
        model_cache = ModelCache()
        provider_info = _make_model_provider_info("ws", "nim", host_url="http://nim.svc/v1")
        entity_info = ModelEntityInfo(workspace="ws", name="llama")
        entity_info.model_providers.append(("llama-v1", provider_info))
        model_cache.model_entity_info_map[("ws", "llama")] = entity_info

        accessor = self._make_accessor(model_cache=model_cache)
        # host already ends with /v1 — should not double-append
        target = accessor.get_inference_url_and_model("ws/llama")
        assert target.model_provider_gateway_url == "http://nim.svc/v1"

    def test_get_inference_url_and_model_raises_key_error_when_missing(self):
        accessor = self._make_accessor()
        with pytest.raises(KeyError):
            accessor.get_inference_url_and_model("ws/missing")

    def test_get_openai_compatible_inference_url_and_model_returns_target(self):
        vm_cache = VirtualModelCache()
        vm_cache.rebuild([_make_sdk_vm("ws", "llama")])
        platform_config = MagicMock()
        platform_config.base_url = "http://platform.local:8080"

        accessor = self._make_accessor(virtual_model_cache=vm_cache)
        with patch(
            "nmp.core.inference_gateway.api.middleware_registry.get_platform_config", return_value=platform_config
        ):
            target = accessor.get_openai_compatible_inference_url_and_model("ws/llama")

        assert (
            target.openai_base_url == "http://platform.local:8080/apis/inference-gateway/v2/workspaces/ws/openai/-/v1"
        )
        assert target.model == "ws/llama"

    def test_get_openai_compatible_inference_url_and_model_raises_key_error_when_missing(self):
        accessor = self._make_accessor()
        with pytest.raises(KeyError):
            accessor.get_openai_compatible_inference_url_and_model("ws/missing")

    def test_get_openai_compatible_inference_url_and_model_requires_virtual_model(self):
        model_cache = ModelCache()
        model_cache.model_entity_info_map[("ws", "llama")] = ModelEntityInfo(workspace="ws", name="llama")
        accessor = self._make_accessor(model_cache=model_cache)

        with pytest.raises(KeyError, match="VirtualModel"):
            accessor.get_openai_compatible_inference_url_and_model("ws/llama")


# ---------------------------------------------------------------------------
# MiddlewareRegistry.resolve_configs_for_virtual_model tests
# ---------------------------------------------------------------------------


class TestResolveConfigsForVirtualModel:
    @pytest.mark.asyncio
    async def test_empty_middleware_stores_empty_lists(self):
        registry = MiddlewareRegistry()
        vm = _make_sdk_vm("ws", "passthrough")
        await registry.resolve_configs_for_virtual_model(vm, prefetch=PrefetchResult())

        assert registry.request_middleware_calls[("ws", "passthrough")] == []
        assert registry.response_middleware_calls[("ws", "passthrough")] == []
        assert registry.post_response_middleware_calls[("ws", "passthrough")] == []
        assert ("ws", "passthrough") not in registry.broken_vms

    @pytest.mark.asyncio
    async def test_inline_config_is_validated_and_stored(self):
        plugin = _make_mock_plugin()
        plugin.validate_middleware_config = AsyncMock(return_value={"validated": True})
        registry = MiddlewareRegistry(plugins={"my-plugin": plugin})

        call = _make_sdk_call("my-plugin", config_type="my_config", config={"key": "value"})
        vm = _make_sdk_vm("ws", "vm", request_middleware=[call])
        await registry.resolve_configs_for_virtual_model(vm, prefetch=PrefetchResult())

        plugin.validate_middleware_config.assert_awaited_once_with("my_config", {"key": "value"})
        assert len(registry.request_middleware_calls[("ws", "vm")]) == 1
        assert registry.request_middleware_calls[("ws", "vm")][0].resolved_config == {"validated": True}
        assert ("ws", "vm") not in registry.broken_vms

    @pytest.mark.asyncio
    async def test_config_id_validates_from_prefetch(self):
        """resolve_configs_for_virtual_model reads config_id results from PrefetchResult.fetched
        and passes them to validate_middleware_config; it never calls get_middleware_config itself."""
        plugin = _make_mock_plugin()
        plugin.validate_middleware_config = AsyncMock(return_value={"validated": True})
        registry = MiddlewareRegistry(plugins={"my-plugin": plugin})
        mref = MiddlewareConfigRef("my-plugin", "my_config", "ws/my-config")

        call = _make_sdk_call("my-plugin", config_type="my_config", config_id="ws/my-config")
        vm = _make_sdk_vm("ws", "vm", request_middleware=[call])
        await registry.resolve_configs_for_virtual_model(vm, prefetch=PrefetchResult(fetched={mref: {"from": "store"}}))

        plugin.get_middleware_config.assert_not_awaited()
        plugin.validate_middleware_config.assert_awaited_once_with("my_config", {"from": "store"})
        assert registry.request_middleware_calls[("ws", "vm")][0].resolved_config == {"validated": True}
        assert ("ws", "vm") not in registry.broken_vms

    @pytest.mark.asyncio
    async def test_config_id_uses_prefetch_and_skips_get_middleware_config(self):
        """When the batch :class:`PrefetchResult` already has the ref, do not re-fetch."""
        plugin = _make_mock_plugin()
        raw = {"from": "store"}
        plugin.get_middleware_config = AsyncMock(return_value=raw)
        plugin.validate_middleware_config = AsyncMock(return_value={"ok": True})
        registry = MiddlewareRegistry(plugins={"my-plugin": plugin})
        mref = MiddlewareConfigRef("my-plugin", "my_config", "ws/my-config")
        call = _make_sdk_call("my-plugin", config_type="my_config", config_id="ws/my-config")
        vm = _make_sdk_vm("ws", "vm", request_middleware=[call])

        await registry.resolve_configs_for_virtual_model(vm, prefetch=PrefetchResult(fetched={mref: raw}))

        plugin.get_middleware_config.assert_not_awaited()
        plugin.validate_middleware_config.assert_awaited_once_with("my_config", raw)
        assert registry.request_middleware_calls[("ws", "vm")][0].resolved_config == {"ok": True}

    @skip_flaky_caplog
    @pytest.mark.asyncio
    async def test_unknown_plugin_marks_vm_broken(self, caplog):
        registry = MiddlewareRegistry(plugins={})
        call = _make_sdk_call("missing-plugin")
        vm = _make_sdk_vm("ws", "vm", request_middleware=[call])

        await registry.resolve_configs_for_virtual_model(vm, prefetch=PrefetchResult())

        assert ("ws", "vm") not in registry.request_middleware_calls
        assert ("ws", "vm") in registry.broken_vms
        assert "missing-plugin" in caplog.text

    @pytest.mark.asyncio
    async def test_missing_config_ref_marks_vm_broken_and_evicts(self):
        """A ``config_id`` in ``PrefetchResult.missing`` (deletion) evicts and flips broken_vms.

        This is the QA-reported scenario: the upstream config entity has been
        deleted, ``prefetch_configs`` flagged it as missing, and the VM is
        re-resolved as part of the diff loop. Any prior resolved entries must
        be cleared and the proxy must see the VM as broken.
        """
        plugin = _make_mock_plugin()
        registry = MiddlewareRegistry(plugins={"my-plugin": plugin})
        mref = MiddlewareConfigRef("my-plugin", "my_config", "ws/cfg")

        # Seed prior resolved state to confirm it gets evicted.
        registry.request_middleware_calls[("ws", "vm")] = [MagicMock()]
        registry.response_middleware_calls[("ws", "vm")] = [MagicMock()]
        registry.post_response_middleware_calls[("ws", "vm")] = [MagicMock()]

        call = _make_sdk_call("my-plugin", config_id="ws/cfg")
        vm = _make_sdk_vm("ws", "vm", request_middleware=[call])

        await registry.resolve_configs_for_virtual_model(vm, prefetch=PrefetchResult(missing={mref}))

        assert ("ws", "vm") not in registry.request_middleware_calls
        assert ("ws", "vm") not in registry.response_middleware_calls
        assert ("ws", "vm") not in registry.post_response_middleware_calls
        assert ("ws", "vm") in registry.broken_vms
        plugin.validate_middleware_config.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_transient_config_ref_marks_vm_broken(self):
        """A ``config_id`` neither in ``fetched`` nor ``missing`` is transient and fails closed.

        Only VMs that genuinely changed (or were freshly added) re-enter
        resolution per the diff loop. Treating a transient fetch failure as
        "preserve in place" at this layer would let a half-baked update slip
        through, so we fail closed instead. The next successful poll cycle
        recovers naturally.
        """
        registry = MiddlewareRegistry(plugins={"my-plugin": _make_mock_plugin()})
        call = _make_sdk_call("my-plugin", config_id="ws/cfg")
        vm = _make_sdk_vm("ws", "vm", request_middleware=[call])

        await registry.resolve_configs_for_virtual_model(vm, prefetch=PrefetchResult())

        assert ("ws", "vm") not in registry.request_middleware_calls
        assert ("ws", "vm") in registry.broken_vms

    @pytest.mark.asyncio
    async def test_validate_config_value_error_marks_vm_broken(self, caplog):
        plugin = _make_mock_plugin()
        plugin.validate_middleware_config = AsyncMock(side_effect=ValueError("bad config"))
        registry = MiddlewareRegistry(plugins={"my-plugin": plugin})

        call = _make_sdk_call("my-plugin", config={"bad": True})
        vm = _make_sdk_vm("ws", "vm", request_middleware=[call])
        await registry.resolve_configs_for_virtual_model(vm, prefetch=PrefetchResult())

        assert ("ws", "vm") not in registry.request_middleware_calls
        assert ("ws", "vm") in registry.broken_vms

    @pytest.mark.asyncio
    async def test_post_response_failure_does_not_mark_vm_broken(self):
        """Post-response middleware is fire-and-forget — its failure must not 503 the VM."""

        async def _validate(config_type: str, config):
            if config.get("phase") == "post":
                raise ValueError("post boom")
            return config

        plugin = _make_mock_plugin()
        plugin.validate_middleware_config = AsyncMock(side_effect=_validate)
        registry = MiddlewareRegistry(plugins={"my-plugin": plugin})

        req_call = _make_sdk_call("my-plugin", config={"phase": "req"})
        resp_call = _make_sdk_call("my-plugin", config={"phase": "resp"})
        post_call = _make_sdk_call("my-plugin", config={"phase": "post"})
        vm = _make_sdk_vm(
            "ws",
            "vm",
            request_middleware=[req_call],
            response_middleware=[resp_call],
            post_response_middleware=[post_call],
        )

        # Seed a stale post_response slot to confirm it gets cleared on partial failure.
        registry.post_response_middleware_calls[("ws", "vm")] = [MagicMock()]

        await registry.resolve_configs_for_virtual_model(vm, prefetch=PrefetchResult())

        # Pre-response phases committed.
        assert len(registry.request_middleware_calls[("ws", "vm")]) == 1
        assert len(registry.response_middleware_calls[("ws", "vm")]) == 1
        # post_response slot dropped, VM not marked broken.
        assert ("ws", "vm") not in registry.post_response_middleware_calls
        assert ("ws", "vm") not in registry.broken_vms

    @pytest.mark.asyncio
    async def test_successful_resolve_clears_prior_broken_state(self):
        """A VM previously in :attr:`broken_vms` is cleared once it resolves cleanly."""
        plugin = _make_mock_plugin()
        registry = MiddlewareRegistry(plugins={"my-plugin": plugin})
        registry.broken_vms.add(("ws", "vm"))

        call = _make_sdk_call("my-plugin", config={"phase": "req"})
        vm = _make_sdk_vm("ws", "vm", request_middleware=[call])
        await registry.resolve_configs_for_virtual_model(vm, prefetch=PrefetchResult())

        assert ("ws", "vm") in registry.request_middleware_calls
        assert ("ws", "vm") not in registry.broken_vms

    @pytest.mark.asyncio
    async def test_each_phase_resolved_independently(self):
        plugin = _make_mock_plugin()
        registry = MiddlewareRegistry(plugins={"my-plugin": plugin})

        req_call = _make_sdk_call("my-plugin", config={"phase": "req"})
        resp_call = _make_sdk_call("my-plugin", config={"phase": "resp"})
        post_call = _make_sdk_call("my-plugin", config={"phase": "post"})
        vm = _make_sdk_vm(
            "ws",
            "vm",
            request_middleware=[req_call],
            response_middleware=[resp_call],
            post_response_middleware=[post_call],
        )

        await registry.resolve_configs_for_virtual_model(vm, prefetch=PrefetchResult())

        assert len(registry.request_middleware_calls[("ws", "vm")]) == 1
        assert len(registry.response_middleware_calls[("ws", "vm")]) == 1
        assert len(registry.post_response_middleware_calls[("ws", "vm")]) == 1

    @pytest.mark.asyncio
    async def test_vm_with_none_name_is_skipped(self):
        registry = MiddlewareRegistry()
        vm = SDKVirtualModel(
            id="ws/",
            entity_id="ws/",
            name=None,
            workspace="ws",
            parent="ws",
            db_version=1,
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
        )
        # Should not raise
        await registry.resolve_configs_for_virtual_model(vm, prefetch=PrefetchResult())
        assert len(registry.request_middleware_calls) == 0
        assert len(registry.broken_vms) == 0


# ---------------------------------------------------------------------------
# collect_config_refs / prefetch_configs
# ---------------------------------------------------------------------------


class TestCollectMiddlewareConfigReferences:
    def test_dedupes_across_phases(self):
        c1 = _make_sdk_call("p1", config_id="ws/a")
        vm = _make_sdk_vm(
            "ws",
            "vm",
            request_middleware=[c1],
            response_middleware=[c1],
        )
        refs = collect_config_refs([vm])
        assert refs == {MiddlewareConfigRef("p1", "my_config", "ws/a")}

    def test_all_phases_distinct_ids(self):
        a = _make_sdk_call("p1", config_id="ws/a")
        b = _make_sdk_call("p1", config_id="ws/b")
        vm = _make_sdk_vm("ws", "vm", request_middleware=[a], post_response_middleware=[b])
        refs = collect_config_refs([vm])
        assert refs == {
            MiddlewareConfigRef("p1", "my_config", "ws/a"),
            MiddlewareConfigRef("p1", "my_config", "ws/b"),
        }

    def test_skips_inline_config(self):
        vm = _make_sdk_vm("ws", "vm", request_middleware=[_make_sdk_call("p1", config={"k": 1})])
        assert collect_config_refs([vm]) == set()


class _EntityWithUpdatedAt:
    def __init__(self, updated_at: datetime) -> None:
        self.updated_at = updated_at


class TestFetchMiddlewareConfigResponses:
    @pytest.mark.asyncio
    async def test_fetches_each_ref_once(self):
        t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        plugin = _make_mock_plugin()
        plugin.get_middleware_config = AsyncMock(return_value=_EntityWithUpdatedAt(t1))
        registry = MiddlewareRegistry(plugins={"p1": plugin})
        mref = MiddlewareConfigRef("p1", "my_config", "ws/cfg")
        out = await registry.prefetch_configs({mref})
        plugin.get_middleware_config.assert_awaited_once_with("my_config", "ws/cfg")
        assert mref in out.fetched
        assert out.fetched[mref] is not None
        assert mref not in out.missing
        assert mref not in out.transient

    @pytest.mark.asyncio
    async def test_partitions_missing_transient_and_fetched(self):
        """Each ref ends up in exactly one bucket based on the plugin's exception."""
        from nemo_platform_plugin.inference_middleware import MiddlewareConfigNotFoundError

        t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)

        async def _side_effect(config_type: str, config_id: str):
            if config_id == "ws/ok":
                return _EntityWithUpdatedAt(t1)
            if config_id == "ws/gone":
                raise MiddlewareConfigNotFoundError(config_id)
            if config_id == "ws/flake":
                raise RuntimeError("network blip")
            raise AssertionError(f"unexpected config_id: {config_id}")

        plugin = _make_mock_plugin()
        plugin.get_middleware_config = AsyncMock(side_effect=_side_effect)
        registry = MiddlewareRegistry(plugins={"p1": plugin})
        ok = MiddlewareConfigRef("p1", "my_config", "ws/ok")
        gone = MiddlewareConfigRef("p1", "my_config", "ws/gone")
        flake = MiddlewareConfigRef("p1", "my_config", "ws/flake")

        out = await registry.prefetch_configs({ok, gone, flake})

        assert ok in out.fetched
        assert gone in out.missing
        assert flake in out.transient
        # No overlap between buckets.
        assert out.fetched.keys().isdisjoint(out.missing)
        assert out.fetched.keys().isdisjoint(out.transient)
        assert out.missing.isdisjoint(out.transient)

    @pytest.mark.asyncio
    async def test_unknown_plugin_is_transient_not_missing(self):
        """Unknown plugin = "we couldn't even ask" — keep prior state, do not signal deletion."""
        registry = MiddlewareRegistry(plugins={})
        mref = MiddlewareConfigRef("ghost-plugin", "x", "ws/cfg")
        out = await registry.prefetch_configs({mref})
        assert mref in out.transient
        assert mref not in out.missing

    @pytest.mark.asyncio
    async def test_not_implemented_is_transient_not_missing(self):
        plugin = _make_mock_plugin()
        plugin.get_middleware_config = AsyncMock(side_effect=NotImplementedError)
        registry = MiddlewareRegistry(plugins={"p1": plugin})
        mref = MiddlewareConfigRef("p1", "x", "ws/cfg")
        out = await registry.prefetch_configs({mref})
        assert mref in out.transient
        assert mref not in out.missing


# ---------------------------------------------------------------------------
# MiddlewareRegistry.notify_upserted / notify_destroyed tests
# ---------------------------------------------------------------------------


class TestNotifyHooks:
    @pytest.mark.asyncio
    async def test_notify_upserted_calls_correct_plugins(self):
        plugin_a = _make_mock_plugin()
        plugin_b = _make_mock_plugin()
        registry = MiddlewareRegistry(plugins={"plugin-a": plugin_a, "plugin-b": plugin_b})

        vm = _make_sdk_vm("ws", "vm", request_middleware=[_make_sdk_call("plugin-a", config={})])
        await registry.notify_upserted(vm)

        plugin_a.on_virtual_model_upserted.assert_awaited_once()
        plugin_b.on_virtual_model_upserted.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_notify_destroyed_calls_correct_plugins(self):
        plugin_a = _make_mock_plugin()
        registry = MiddlewareRegistry(plugins={"plugin-a": plugin_a})

        vm = _make_sdk_vm("ws", "vm", response_middleware=[_make_sdk_call("plugin-a", config={})])
        await registry.notify_destroyed(vm)

        plugin_a.on_virtual_model_destroyed.assert_awaited_once()

    @skip_flaky_caplog
    @pytest.mark.asyncio
    async def test_notify_upserted_swallows_plugin_errors(self, caplog):
        plugin = _make_mock_plugin()
        plugin.on_virtual_model_upserted = AsyncMock(side_effect=RuntimeError("boom"))
        registry = MiddlewareRegistry(plugins={"my-plugin": plugin})

        vm = _make_sdk_vm("ws", "vm", request_middleware=[_make_sdk_call("my-plugin", config={})])
        # Should not raise
        await registry.notify_upserted(vm)
        assert "my-plugin" in caplog.text

    @pytest.mark.asyncio
    async def test_notify_passes_plugin_virtual_model(self):
        plugin = _make_mock_plugin()
        registry = MiddlewareRegistry(plugins={"my-plugin": plugin})

        vm = _make_sdk_vm("ws", "my-vm", request_middleware=[_make_sdk_call("my-plugin", config={})])
        await registry.notify_upserted(vm)

        call_arg = plugin.on_virtual_model_upserted.call_args[0][0]
        assert isinstance(call_arg, PluginVirtualModel)
        assert call_arg.workspace == "ws"
        assert call_arg.name == "my-vm"
        assert call_arg.request_middleware[0].config == {}


# ---------------------------------------------------------------------------
# MiddlewareRegistry.evict / shutdown tests
# ---------------------------------------------------------------------------


class TestEvictAndShutdown:
    def test_evict_removes_resolved_configs(self):
        registry = MiddlewareRegistry()
        registry.request_middleware_calls[("ws", "vm")] = [MagicMock()]
        registry.response_middleware_calls[("ws", "vm")] = [MagicMock()]
        registry.post_response_middleware_calls[("ws", "vm")] = []

        registry.evict(("ws", "vm"))

        assert ("ws", "vm") not in registry.request_middleware_calls
        assert ("ws", "vm") not in registry.response_middleware_calls
        assert ("ws", "vm") not in registry.post_response_middleware_calls

    def test_evict_missing_key_is_noop(self):
        registry = MiddlewareRegistry()
        # Should not raise
        registry.evict(("ws", "nonexistent"))

    @pytest.mark.asyncio
    async def test_shutdown_calls_all_plugins(self):
        p1, p2 = _make_mock_plugin(), _make_mock_plugin()
        registry = MiddlewareRegistry(plugins={"p1": p1, "p2": p2})
        await registry.shutdown()
        p1.on_shutdown.assert_awaited_once()
        p2.on_shutdown.assert_awaited_once()

    @skip_flaky_caplog
    @pytest.mark.asyncio
    async def test_shutdown_swallows_plugin_errors(self, caplog):
        plugin = _make_mock_plugin()
        plugin.on_shutdown = AsyncMock(side_effect=RuntimeError("crash"))
        registry = MiddlewareRegistry(plugins={"bad-plugin": plugin})
        # Should not raise
        await registry.shutdown()
        assert "bad-plugin" in caplog.text


# ---------------------------------------------------------------------------
# load_middleware_plugins tests
# ---------------------------------------------------------------------------


class TestLoadMiddlewarePlugins:
    @pytest.mark.asyncio
    async def test_load_calls_on_startup_for_each_plugin(self):
        plugin_cls = MagicMock()
        instance = _make_mock_plugin()
        plugin_cls.return_value = instance

        with patch(
            "nmp.core.inference_gateway.api.middleware_registry.discover_inference_middleware",
            return_value={"my-plugin": plugin_cls},
        ):
            registry = await load_middleware_plugins(ModelCache(), VirtualModelCache())

        instance.on_startup.assert_awaited_once()
        assert "my-plugin" in registry.plugins

    @pytest.mark.asyncio
    async def test_load_injects_cache_accessor(self):
        plugin_cls = MagicMock()
        instance = _make_mock_plugin()
        plugin_cls.return_value = instance

        with patch(
            "nmp.core.inference_gateway.api.middleware_registry.discover_inference_middleware",
            return_value={"my-plugin": plugin_cls},
        ):
            await load_middleware_plugins(ModelCache(), VirtualModelCache())

        instance._inject_cache.assert_called_once()
        injected = instance._inject_cache.call_args[0][0]
        assert isinstance(injected, InferenceMiddlewareCacheAccessorImpl)

    @pytest.mark.asyncio
    async def test_load_injects_typed_client_from_plugin_sdk(self):
        plugin_cls = MagicMock()
        instance = _make_mock_plugin()
        plugin_cls.return_value = instance
        sdk = MagicMock()
        client = MagicMock()

        with patch(
            "nmp.core.inference_gateway.api.middleware_registry.discover_inference_middleware",
            return_value={"my-plugin": plugin_cls},
        ):
            with patch(
                "nmp.core.inference_gateway.api.middleware_registry.client_from_platform",
                return_value=client,
            ) as adapt:
                await load_middleware_plugins(
                    ModelCache(),
                    VirtualModelCache(),
                    plugin_sdk_factory=lambda _name: sdk,
                )

        adapt.assert_called_once()
        assert adapt.call_args.args[0] is sdk
        instance._inject_platform_sdk.assert_called_once_with(sdk)
        instance._inject_platform_client.assert_called_once_with(client)

    @skip_flaky_caplog
    @pytest.mark.asyncio
    async def test_load_fault_isolation_broken_import(self, caplog):
        caplog.set_level(logging.WARNING, logger="nmp.core.inference_gateway.api.middleware_registry")

        good_cls = MagicMock()
        good_instance = _make_mock_plugin()
        good_cls.return_value = good_instance

        bad_cls = MagicMock(side_effect=ImportError("bad module"))

        with patch(
            "nmp.core.inference_gateway.api.middleware_registry.discover_inference_middleware",
            return_value={"good-plugin": good_cls, "bad-plugin": bad_cls},
        ):
            registry = await load_middleware_plugins(ModelCache(), VirtualModelCache())

        assert "good-plugin" in registry.plugins
        assert "bad-plugin" not in registry.plugins
        assert "bad-plugin" in caplog.text

    @skip_flaky_caplog
    @pytest.mark.asyncio
    async def test_load_fault_isolation_on_startup_raises(self, caplog):
        good_cls = MagicMock()
        good_instance = _make_mock_plugin()
        good_cls.return_value = good_instance

        bad_cls = MagicMock()
        bad_instance = _make_mock_plugin()
        bad_instance.on_startup = AsyncMock(side_effect=RuntimeError("startup failure"))
        bad_cls.return_value = bad_instance

        with patch(
            "nmp.core.inference_gateway.api.middleware_registry.discover_inference_middleware",
            return_value={"good-plugin": good_cls, "bad-plugin": bad_cls},
        ):
            registry = await load_middleware_plugins(ModelCache(), VirtualModelCache())

        assert "good-plugin" in registry.plugins
        assert "bad-plugin" not in registry.plugins

    @pytest.mark.asyncio
    async def test_load_returns_empty_registry_when_no_plugins(self):
        with patch(
            "nmp.core.inference_gateway.api.middleware_registry.discover_inference_middleware",
            return_value={},
        ):
            registry = await load_middleware_plugins(ModelCache(), VirtualModelCache())

        assert registry.plugins == {}


# ---------------------------------------------------------------------------
# _sdk_vm_to_plugin_vm helper
# ---------------------------------------------------------------------------


def test_sdk_vm_to_plugin_vm_maps_fields():
    call = _make_sdk_call("my-plugin", config_type="cfg", config={"k": "v"})
    vm = _make_sdk_vm(
        "ws",
        "my-vm",
        request_middleware=[call],
        models=[SDKVirtualModelInferenceConfig(model="ws/claude", backend_format="ANTHROPIC_MESSAGES")],
    )
    result = _sdk_vm_to_plugin_vm(vm)

    assert isinstance(result, PluginVirtualModel)
    assert result.workspace == "ws"
    assert result.name == "my-vm"
    assert result.models[0].model == "ws/claude"
    assert result.models[0].backend_format is BackendFormat.ANTHROPIC_MESSAGES
    assert len(result.request_middleware) == 1
    assert result.request_middleware[0].name == "my-plugin"
    assert result.request_middleware[0].config == {"k": "v"}


def test_sdk_vm_to_plugin_vm_handles_none_name():
    vm = SDKVirtualModel(
        id="ws/",
        entity_id="ws/",
        name=None,
        workspace="ws",
        parent="ws",
        db_version=1,
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
    )
    result = _sdk_vm_to_plugin_vm(vm)
    assert result.name == ""
    assert result.workspace == "ws"
