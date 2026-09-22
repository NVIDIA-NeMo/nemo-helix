# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for streaming response handling in SwitchyardMiddleware.

Verifies:
1. Streaming responses (AsyncIterator) pass through unchanged without typed_result.
2. Streaming requests flow through the registered factory pipeline.
3. Proper error handling for missing factories.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

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
from nemo_switchyard.middleware import SwitchyardMiddleware


async def _mock_streaming_chunks() -> AsyncIterator[dict]:
    """Yield a sequence of OpenAI-style chunk dicts simulating a streamed response."""
    chunks = [
        {
            "id": "chatcmpl-stream-1",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "llama-3-70b",
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-stream-1",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "llama-3-70b",
            "choices": [{"index": 0, "delta": {"content": "hello"}, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-stream-1",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "llama-3-70b",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
    ]
    for chunk in chunks:
        yield chunk


class TestStreamingBehavior:
    """Streaming responses through the middleware."""

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
            id="vm-stream-123",
            workspace="test-workspace",
            name="stream-model",
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
            request_id="stream-123",
            workspace="test-workspace",
            virtual_model_name="stream-model",
            original_request=InferenceRequest(
                body={
                    "model": "test-workspace/stream-model",
                    "messages": [{"role": "user", "content": "stream me a poem"}],
                    "stream": True,
                },
                headers={},
                path="v1/chat/completions",
            ),
        )

    @pytest.mark.asyncio
    async def test_streaming_response_without_typed_body_passes_through(
        self,
        middleware: SwitchyardMiddleware,
        virtual_model: VirtualModel,
        middleware_ctx: InferenceMiddlewareContext,
    ) -> None:
        """typed_body=None with an async-iterator result passes through unchanged.
        This is the chained-streaming case: a prior switchyard pass translated the
        stream (clearing typed_body); a second pass should not 500."""
        await middleware.on_virtual_model_upserted(virtual_model)

        async def stream_response():
            async for chunk in _mock_streaming_chunks():
                yield chunk

        stream = stream_response()
        response = InferenceResponse(result=stream, headers={})

        out = await middleware.process_response(middleware_ctx, response, {})
        assert out is response
        assert out.result is stream

    @pytest.mark.asyncio
    async def test_process_streaming_request(
        self,
        middleware: SwitchyardMiddleware,
        virtual_model: VirtualModel,
        middleware_ctx: InferenceMiddlewareContext,
    ) -> None:
        """Test that streaming request flows through registered factory pipeline."""
        await middleware.on_virtual_model_upserted(virtual_model)

        body = {
            "model": "test-workspace/stream-model",
            "messages": [{"role": "user", "content": "stream me a poem"}],
            "stream": True,
        }
        request = InferenceRequest(body=body, headers={}, path="v1/chat/completions", typed_body=body)

        # Process request through middleware
        result = await middleware.process_request(middleware_ctx, request, {"config_type": "random_routing"})

        # Request should be modified by the routing pipeline
        assert result is request
        assert "model" in result.body
        # Stream flag should be preserved
        assert result.body.get("stream") is True

    @pytest.mark.asyncio
    async def test_response_without_typed_body_and_stream_result_passes_through(
        self,
        middleware: SwitchyardMiddleware,
        middleware_ctx: InferenceMiddlewareContext,
    ) -> None:
        """typed_body=None + async-iterator result passes through (chained streaming case).
        Only raises 500 when typed_body is None AND result is not an async iterator
        (i.e. IGW never populated typed_body at all)."""

        async def stream_response():
            async for chunk in _mock_streaming_chunks():
                yield chunk

        stream = stream_response()
        response = InferenceResponse(result=stream, headers={})

        out = await middleware.process_response(middleware_ctx, response, {})
        assert out is response
        assert out.result is stream
