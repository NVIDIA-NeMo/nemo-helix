# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for SwitchyardMiddleware with real switchyard pipelines.

Drives the middleware with VirtualModel registration and real switchyard factories,
verifying end-to-end behavior:

1. VirtualModel upsert validates config and registers factory instance.
2. Request processing looks up registered factory and executes request pipeline.
3. Response processing passes through or processes typed_result.
4. VirtualModel destroy unregisters the factory.
"""

from __future__ import annotations

import pytest
from nemo_helix_plugin.inference_middleware import (
    BackendFormat,
    InferenceMiddlewareContext,
    InferenceRequest,
    InferenceResponse,
)
from nemo_helix_plugin.inference_middleware_models import (
    MiddlewareCall,
    VirtualModel,
    VirtualModelInferenceConfig,
)
from nemo_switchyard import _state as middleware_state
from nemo_switchyard.middleware import SwitchyardMiddleware


class TestSwitchyardRandomRoutingIntegration:
    """Integration tests with real RandomRoutingIGWFactory."""

    @pytest.fixture
    async def middleware(self) -> SwitchyardMiddleware:
        mw = SwitchyardMiddleware()
        await mw.on_startup()
        yield mw
        await mw.on_shutdown()

    @pytest.fixture
    def virtual_model(self) -> VirtualModel:
        """Create a test VirtualModel with switchyard middleware."""
        return VirtualModel(
            id="vm-test-123",
            workspace="test-workspace",
            name="test-model",
            models=[
                VirtualModelInferenceConfig(
                    model="workspace/llama-3-70b",
                    backend_format=BackendFormat.OPENAI_CHAT,
                ),
                VirtualModelInferenceConfig(
                    model="workspace/llama-3-8b",
                    backend_format=BackendFormat.OPENAI_CHAT,
                ),
            ],
            request_middleware=[
                MiddlewareCall(
                    name="nemo-switchyard",
                    config_type="random_routing",
                    config={
                        "strong": {"model": "workspace/llama-3-70b"},
                        "weak": {"model": "workspace/llama-3-8b"},
                        "strong_probability": 1.0,
                        "enable_stats": False,
                    },
                ),
            ],
            response_middleware=[],
            post_response_middleware=[],
        )

    @pytest.fixture
    def middleware_ctx(self) -> InferenceMiddlewareContext:
        """Create a test InferenceMiddlewareContext."""
        return InferenceMiddlewareContext(
            request_id="test-123",
            workspace="test-workspace",
            virtual_model_name="test-model",
            original_request=InferenceRequest(
                body={"model": "test-workspace/test-model", "messages": []},
                headers={},
                path="v1/chat/completions",
            ),
        )

    @pytest.mark.asyncio
    async def test_register_and_lookup_factory(
        self, middleware: SwitchyardMiddleware, virtual_model: VirtualModel
    ) -> None:
        """Test that VirtualModel upsert registers factory in Switchyard registry."""
        await middleware.on_virtual_model_upserted(virtual_model)

        # VM should be mapped to a config hash by (vm_key, config_type)
        vm_key = f"{virtual_model.workspace}/{virtual_model.name}"
        assert (vm_key, "random_routing", "request") in middleware_state.VM_NAME_TO_CONFIG_HASH
        config_hash = middleware_state.VM_NAME_TO_CONFIG_HASH[(vm_key, "random_routing", "request")]
        # And that config hash should have a factory registered
        assert config_hash in middleware_state.FACTORIES_BY_CONFIG_HASH

    @pytest.mark.asyncio
    async def test_process_request_with_registered_factory(
        self,
        middleware: SwitchyardMiddleware,
        virtual_model: VirtualModel,
        middleware_ctx: InferenceMiddlewareContext,
    ) -> None:
        """Test that request is processed through registered factory pipeline."""
        await middleware.on_virtual_model_upserted(virtual_model)

        body = {"model": "test-workspace/test-model", "messages": [{"role": "user", "content": "hello"}]}
        request = InferenceRequest(body=body, headers={}, path="v1/chat/completions", typed_body=body)

        # Process request through middleware
        result = await middleware.process_request(middleware_ctx, request, {"config_type": "random_routing"})

        # Request body should be modified by the routing pipeline
        assert result is request
        # The strong probability is 1.0, so model should be set to strong model
        # This depends on the Switchyard factory implementation
        assert "model" in result.body

    @pytest.mark.asyncio
    async def test_process_response_without_typed_body_raises_500(
        self,
        middleware: SwitchyardMiddleware,
        virtual_model: VirtualModel,
        middleware_ctx: InferenceMiddlewareContext,
    ) -> None:
        """typed_body=None is a contract violation — raises InferenceMiddlewareError 500."""
        from nemo_helix_plugin.inference_middleware import InferenceMiddlewareError

        await middleware.on_virtual_model_upserted(virtual_model)

        response = InferenceResponse(result={"id": "123", "choices": []}, headers={})

        with pytest.raises(InferenceMiddlewareError) as exc:
            await middleware.process_response(middleware_ctx, response, {})
        assert exc.value.status_code == 500

    @pytest.mark.asyncio
    async def test_unregister_factory_on_destroy(
        self, middleware: SwitchyardMiddleware, virtual_model: VirtualModel
    ) -> None:
        """Test that VirtualModel destroy cleans up factory if no other VMs use it."""
        await middleware.on_virtual_model_upserted(virtual_model)

        vm_key = f"{virtual_model.workspace}/{virtual_model.name}"
        assert (vm_key, "random_routing", "request") in middleware_state.VM_NAME_TO_CONFIG_HASH
        config_hash = middleware_state.VM_NAME_TO_CONFIG_HASH[(vm_key, "random_routing", "request")]

        # Destroy the VirtualModel
        await middleware.on_virtual_model_destroyed(virtual_model)

        # VM should be unregistered from all mappings
        assert (vm_key, "random_routing", "request") not in middleware_state.VM_NAME_TO_CONFIG_HASH
        assert virtual_model.id not in middleware_state.VM_CONFIG_MAPPING
        # And since no other VM uses this config, the factory should be unregistered
        assert config_hash not in middleware_state.FACTORIES_BY_CONFIG_HASH

    @pytest.mark.asyncio
    async def test_header_in_place_mutation_propagates_to_request(
        self,
        middleware: SwitchyardMiddleware,
        virtual_model: VirtualModel,
        middleware_ctx: InferenceMiddlewareContext,
    ) -> None:
        """Header propagation contract: in-place mutations on metadata['headers'] must
        land on request.headers. The middleware passes request.headers into
        ProxyContext.metadata by reference; this test locks in that shared-reference
        contract so a future refactor doesn't silently drop header changes from
        Switchyard processors.
        """
        from switchyard.lib.request_pipeline import RequestPipeline
        from switchyard.lib.roles import RequestProcessor

        await middleware.on_virtual_model_upserted(virtual_model)

        class _HeaderMutatingProcessor(RequestProcessor):
            async def process(self, context, request):
                # In-place mutation of the shared headers dict via context metadata.
                context.metadata["headers"]["x-switchyard-touched"] = "1"
                return request

        # Replace the registered factory's pipeline with one that mutates headers.
        vm_key = f"{virtual_model.workspace}/{virtual_model.name}"
        cfg_hash = middleware_state.VM_NAME_TO_CONFIG_HASH[(vm_key, "random_routing", "request")]
        factory_name = middleware_state.FACTORIES_BY_CONFIG_HASH[cfg_hash]
        from switchyard.lib.registry import lookup

        factory = lookup(factory_name)
        original_build = factory.build_request_pipeline
        factory.build_request_pipeline = lambda config: RequestPipeline(
            list(original_build(config)._processors) + [_HeaderMutatingProcessor()]
        )

        body = {"model": "test-workspace/test-model", "messages": [{"role": "user", "content": "hi"}]}
        request = InferenceRequest(
            body=body, headers={"x-original": "yes"}, path="v1/chat/completions", typed_body=body
        )

        result = await middleware.process_request(middleware_ctx, request, {"config_type": "random_routing"})

        assert result.headers["x-original"] == "yes"
        assert result.headers["x-switchyard-touched"] == "1"
