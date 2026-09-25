# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle identity regressions for SwitchyardMiddleware (NVBug 6563245).

The IGW lifecycle dispatcher (middleware_registry._sdk_vm_to_plugin_vm) builds
plugin VirtualModels WITHOUT mapping the entity id, so every hook receives
``id == ""``. The per-VM config bookkeeping therefore must key on the
workspace/name identity, never on ``.id``. These tests replicate the production
shape (no id) and the exact create/destroy interleaving from the QA skill-eval
suite that produced "Factory not found for config hash ..." 500s on
catalog-visible VMs.
"""

from __future__ import annotations

from typing import Any

import pytest
from nemo_platform_plugin.inference_middleware import (
    BackendFormat,
    InferenceMiddlewareContext,
    InferenceRequest,
    MiddlewareCall,
    VirtualModel,
    VirtualModelInferenceConfig,
)
from nemo_switchyard import _state
from nemo_switchyard.middleware import SwitchyardMiddleware


@pytest.fixture
async def middleware() -> SwitchyardMiddleware:
    mw = SwitchyardMiddleware()
    await mw.on_startup()
    yield mw
    await mw.on_shutdown()


def _vm_without_id(name: str, *, strong_probability: float = 1.0) -> VirtualModel:
    """A VM exactly as the IGW dispatcher delivers it: entity id NOT mapped."""
    return VirtualModel(
        workspace="ws",
        name=name,
        models=[
            VirtualModelInferenceConfig(model="ws/strong", backend_format=BackendFormat.OPENAI_CHAT),
            VirtualModelInferenceConfig(model="ws/weak", backend_format=BackendFormat.OPENAI_CHAT),
        ],
        request_middleware=[
            MiddlewareCall(
                name="nemo-switchyard",
                config_type="random_routing",
                config={
                    "strong": {"model": "ws/strong"},
                    "weak": {"model": "ws/weak"},
                    "strong_probability": strong_probability,
                    "rng_seed": 1,
                    "enable_stats": False,
                },
            )
        ],
        response_middleware=[],
        post_response_middleware=[],
    )


def _request_ctx(vm_name: str, request: InferenceRequest) -> InferenceMiddlewareContext:
    return InferenceMiddlewareContext(
        request_id="test-req",
        workspace="ws",
        virtual_model_name=vm_name,
        original_request=InferenceRequest(
            body=dict(request.body),
            headers=dict(request.headers),
            path=request.path,
        ),
    )


def _openai_request() -> InferenceRequest:
    body: dict[str, Any] = {
        "model": "ws/vm",
        "messages": [{"role": "user", "content": "hello"}],
    }
    return InferenceRequest(body=body, headers={}, path="v1/chat/completions", typed_body=body)


class TestLifecycleIdentityWithoutEntityId:
    """Reproduces NVBug 6563245 with the production (id-less) VM shape."""

    async def test_destroying_one_vm_keeps_other_vms_factory(self, middleware: SwitchyardMiddleware) -> None:
        """skill-eval pattern: old VM destroyed right after a new VM registers.

        With id-keyed bookkeeping both VMs collapsed into one mapping slot, so
        destroy(vm_old) unregistered the factory vm_new still referenced and its
        next request failed with "Factory not found for config hash" (500).
        """
        vm_old = _vm_without_id("skill-eval-old", strong_probability=1.0)
        vm_new = _vm_without_id("skill-eval-new", strong_probability=0.25)

        await middleware.on_virtual_model_upserted(vm_old)
        await middleware.on_virtual_model_upserted(vm_new)
        await middleware.on_virtual_model_destroyed(vm_old)

        # vm_new must still be fully wired: mapping present, factory resolvable,
        # and a live request must route instead of raising.
        assert ("ws/skill-eval-new", "random_routing", "request") in _state.VM_NAME_TO_CONFIG_HASH
        cfg_hash = _state.VM_NAME_TO_CONFIG_HASH[("ws/skill-eval-new", "random_routing", "request")]
        assert cfg_hash in _state.FACTORIES_BY_CONFIG_HASH, (
            "destroying vm_old unregistered vm_new's factory (NVBug 6563245 regression)"
        )
        request = _openai_request()
        result = await middleware.process_request(
            _request_ctx("skill-eval-new", request),
            request,
            {"config_type": "random_routing"},
        )
        assert result is not None

    async def test_identical_config_vms_share_factory_until_last_destroy(
        self, middleware: SwitchyardMiddleware
    ) -> None:
        """Two id-less VMs with IDENTICAL configs share one factory; the factory
        must survive until the LAST referencing VM is destroyed."""
        vm_a = _vm_without_id("twin-a", strong_probability=1.0)
        vm_b = _vm_without_id("twin-b", strong_probability=1.0)

        await middleware.on_virtual_model_upserted(vm_a)
        await middleware.on_virtual_model_upserted(vm_b)
        cfg_hash = _state.VM_NAME_TO_CONFIG_HASH[("ws/twin-a", "random_routing", "request")]
        assert cfg_hash == _state.VM_NAME_TO_CONFIG_HASH[("ws/twin-b", "random_routing", "request")]

        await middleware.on_virtual_model_destroyed(vm_a)
        assert cfg_hash in _state.FACTORIES_BY_CONFIG_HASH, "twin-b still references the shared factory"

        await middleware.on_virtual_model_destroyed(vm_b)
        assert cfg_hash not in _state.FACTORIES_BY_CONFIG_HASH, "last destroy must release the factory"

    async def test_reupsert_same_vm_keeps_single_mapping_entry(self, middleware: SwitchyardMiddleware) -> None:
        """Upserting the same VM twice (config change) must not leak mapping entries
        or leave the VM pointing at an unregistered factory."""
        await middleware.on_virtual_model_upserted(_vm_without_id("solo", strong_probability=1.0))
        await middleware.on_virtual_model_upserted(_vm_without_id("solo", strong_probability=0.5))

        assert list(_state.VM_CONFIG_MAPPING) == ["ws/solo"]
        cfg_hash = _state.VM_NAME_TO_CONFIG_HASH[("ws/solo", "random_routing", "request")]
        assert cfg_hash in _state.FACTORIES_BY_CONFIG_HASH

        await middleware.on_virtual_model_destroyed(_vm_without_id("solo", strong_probability=0.5))
        assert "ws/solo" not in _state.VM_CONFIG_MAPPING
