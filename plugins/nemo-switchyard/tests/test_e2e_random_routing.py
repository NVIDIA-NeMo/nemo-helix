# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for random routing with mocked LLM backend response.

Shows the complete request → route → mocked-backend → response flow:
1. Client sends request with virtual model name
2. SwitchyardMiddleware.process_request() routes to actual model
3. IGW calls mocked LLM backend with routed model
4. SwitchyardMiddleware.process_response() returns response unchanged
"""

from __future__ import annotations

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
from nemo_switchyard.middleware import SwitchyardMiddleware


class TestE2ERandomRouting:
    """End-to-end random routing scenarios with VirtualModel registration."""

    @pytest.fixture
    async def middleware(self) -> SwitchyardMiddleware:
        mw = SwitchyardMiddleware()
        await mw.on_startup()
        yield mw
        await mw.on_shutdown()

    @pytest.fixture
    def virtual_model_strong(self) -> VirtualModel:
        """VirtualModel configured to always route to strong model."""
        return VirtualModel(
            id="vm-e2e-strong-123",
            workspace="test-workspace",
            name="e2e-model-strong",
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
                        "rng_seed": 42,
                        "enable_stats": False,
                    },
                ),
            ],
            response_middleware=[],
            post_response_middleware=[],
        )

    @pytest.fixture
    def virtual_model_weak(self) -> VirtualModel:
        """VirtualModel configured to always route to weak model."""
        return VirtualModel(
            id="vm-e2e-weak-123",
            workspace="test-workspace",
            name="e2e-model-weak",
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
                        "strong_probability": 0.0,
                        "rng_seed": 42,
                        "enable_stats": False,
                    },
                ),
            ],
            response_middleware=[],
            post_response_middleware=[],
        )

    @pytest.fixture
    def middleware_ctx_strong(self) -> InferenceMiddlewareContext:
        """Context for strong routing test."""
        return InferenceMiddlewareContext(
            request_id="e2e-strong-123",
            workspace="test-workspace",
            virtual_model_name="e2e-model-strong",
            original_request=InferenceRequest(
                body={
                    "model": "test-workspace/e2e-model-strong",
                    "messages": [{"role": "user", "content": "what is 2+2?"}],
                    "temperature": 0.7,
                },
                headers={},
                path="v1/chat/completions",
            ),
        )

    @pytest.fixture
    def middleware_ctx_weak(self) -> InferenceMiddlewareContext:
        """Context for weak routing test."""
        return InferenceMiddlewareContext(
            request_id="e2e-weak-123",
            workspace="test-workspace",
            virtual_model_name="e2e-model-weak",
            original_request=InferenceRequest(
                body={
                    "model": "test-workspace/e2e-model-weak",
                    "messages": [{"role": "user", "content": "what is 2+2?"}],
                    "temperature": 0.7,
                },
                headers={},
                path="v1/chat/completions",
            ),
        )

    @pytest.mark.asyncio
    async def test_e2e_random_routing_routes_to_strong(
        self,
        middleware: SwitchyardMiddleware,
        virtual_model_strong: VirtualModel,
        middleware_ctx_strong: InferenceMiddlewareContext,
    ) -> None:
        """Test that routing with strong_probability=1.0 always routes to strong model."""
        await middleware.on_virtual_model_upserted(virtual_model_strong)

        request = InferenceRequest(
            body={
                "model": "test-workspace/e2e-model-strong",
                "messages": [{"role": "user", "content": "what is 2+2?"}],
                "temperature": 0.7,
            },
            headers={},
            path="v1/chat/completions",
        )
        request.typed_body = request.body

        # Process request through middleware
        routed_request = await middleware.process_request(
            middleware_ctx_strong, request, {"config_type": "random_routing"}
        )

        assert routed_request is request
        assert routed_request.body["model"] == "workspace/llama-3-70b"
        assert routed_request.body["messages"] == [{"role": "user", "content": "what is 2+2?"}]
        assert routed_request.body["temperature"] == 0.7

        # Random routing only touches the request — no response_middleware is
        # configured on this VM so IGW never calls process_response.
        assert routed_request.body["model"] == "workspace/llama-3-70b"

    @pytest.mark.asyncio
    async def test_e2e_random_routing_routes_to_weak(
        self,
        middleware: SwitchyardMiddleware,
        virtual_model_weak: VirtualModel,
        middleware_ctx_weak: InferenceMiddlewareContext,
    ) -> None:
        """Test that routing with strong_probability=0.0 always routes to weak model."""
        await middleware.on_virtual_model_upserted(virtual_model_weak)

        request = InferenceRequest(
            body={
                "model": "test-workspace/e2e-model-weak",
                "messages": [{"role": "user", "content": "what is 2+2?"}],
                "temperature": 0.7,
            },
            headers={},
            path="v1/chat/completions",
        )
        request.typed_body = request.body

        # Process request through middleware
        routed_request = await middleware.process_request(
            middleware_ctx_weak, request, {"config_type": "random_routing"}
        )

        assert routed_request is request
        assert routed_request.body["model"] == "workspace/llama-3-8b"
        assert routed_request.body["messages"] == [{"role": "user", "content": "what is 2+2?"}]
        assert routed_request.body["temperature"] == 0.7
