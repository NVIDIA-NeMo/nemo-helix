# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-VM bookkeeping is keyed by ``workspace/name``, never by the entity id.

Every VirtualModel here has an empty ``id``, so any bookkeeping keyed on ``id``
collapses all of them into one slot and these tests fail.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from nemo_helix_plugin.inference_middleware import (
    BackendFormat,
    InferenceMiddlewareContext,
    InferenceRequest,
)
from nemo_helix_plugin.inference_middleware_models import (
    MiddlewareCall,
    VirtualModel,
    VirtualModelInferenceConfig,
)
from nemo_switchyard import _state
from nemo_switchyard.middleware import SwitchyardMiddleware


@pytest.fixture
async def middleware() -> AsyncIterator[SwitchyardMiddleware]:
    mw = SwitchyardMiddleware()
    await mw.on_startup()
    yield mw
    await mw.on_shutdown()


def _vm(name: str, *, strong_probability: float = 1.0) -> VirtualModel:
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
    )


def _request_for(vm_name: str) -> tuple[InferenceMiddlewareContext, InferenceRequest]:
    body: dict[str, Any] = {"model": f"ws/{vm_name}", "messages": [{"role": "user", "content": "hello"}]}
    request = InferenceRequest(body=body, headers={}, path="v1/chat/completions", typed_body=body)
    ctx = InferenceMiddlewareContext(
        request_id="test-req",
        workspace="ws",
        virtual_model_name=vm_name,
        original_request=InferenceRequest(body=dict(body), headers={}, path=request.path),
    )
    return ctx, request


async def test_destroying_one_vm_keeps_another_vms_routing(middleware: SwitchyardMiddleware) -> None:
    """Create A, create B, destroy A: B must still route rather than fail with "Factory not found"."""
    vm_a = _vm("vm-a", strong_probability=1.0)
    vm_b = _vm("vm-b", strong_probability=0.25)
    assert vm_a.id == vm_b.id == ""

    await middleware.on_virtual_model_upserted(vm_a)
    await middleware.on_virtual_model_upserted(vm_b)
    await middleware.on_virtual_model_destroyed(vm_a)

    cfg_hash = _state.VM_NAME_TO_CONFIG_HASH[("ws/vm-b", "random_routing", "request")]
    assert cfg_hash in _state.FACTORIES_BY_CONFIG_HASH
    assert list(_state.VM_CONFIG_MAPPING) == ["ws/vm-b"]

    ctx, request = _request_for("vm-b")
    routed = await middleware.process_request(ctx, request, {"config_type": "random_routing"})
    assert routed.body["model"] in {"ws/strong", "ws/weak"}


async def test_identical_configs_share_a_factory_until_the_last_vm_is_destroyed(
    middleware: SwitchyardMiddleware,
) -> None:
    vm_a = _vm("twin-a")
    vm_b = _vm("twin-b")

    await middleware.on_virtual_model_upserted(vm_a)
    await middleware.on_virtual_model_upserted(vm_b)
    cfg_hash = _state.VM_NAME_TO_CONFIG_HASH[("ws/twin-a", "random_routing", "request")]
    assert cfg_hash == _state.VM_NAME_TO_CONFIG_HASH[("ws/twin-b", "random_routing", "request")]

    await middleware.on_virtual_model_destroyed(vm_a)
    assert cfg_hash in _state.FACTORIES_BY_CONFIG_HASH

    await middleware.on_virtual_model_destroyed(vm_b)
    assert cfg_hash not in _state.FACTORIES_BY_CONFIG_HASH


async def test_reupserting_a_vm_replaces_its_entry_and_destroy_releases_it(
    middleware: SwitchyardMiddleware,
) -> None:
    await middleware.on_virtual_model_upserted(_vm("solo", strong_probability=1.0))
    await middleware.on_virtual_model_upserted(_vm("solo", strong_probability=0.5))

    cfg_hash = _state.VM_NAME_TO_CONFIG_HASH[("ws/solo", "random_routing", "request")]
    assert _state.VM_CONFIG_MAPPING == {"ws/solo": [cfg_hash]}

    await middleware.on_virtual_model_destroyed(_vm("solo", strong_probability=0.5))
    assert _state.VM_CONFIG_MAPPING == {}
    assert cfg_hash not in _state.FACTORIES_BY_CONFIG_HASH
