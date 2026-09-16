# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the middleware pipeline execution functions.

These tests operate on the functions directly without any FastAPI or HTTP
involvement, making them fast and focused on pipeline semantics.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import openai.types.chat as openai_chat_types
import pytest
from nemo_platform_plugin.inference_middleware import (
    BackendFormat,
    ImmediateResponse,
    InferenceMiddlewareContext,
    InferenceMiddlewareError,
    InferenceRequest,
    InferenceResponse,
    NemoInferenceMiddleware,
)
from nmp.core.inference_gateway.api.middleware_registry import (
    ResolvedMiddlewareCall,
    build_inference_response,
    execute_post_response_middleware,
    execute_request_middleware,
    execute_response_middleware,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _call(plugin_name: str, resolved_config: object = None) -> ResolvedMiddlewareCall:
    return ResolvedMiddlewareCall(
        plugin_name=plugin_name,
        config_type="test_config",
        resolved_config=resolved_config or {},
    )


_HEADERS: dict[str, str] = {"content-type": "application/json"}
_BODY: dict = {"model": "ws/llama", "messages": []}
_REQUEST = InferenceRequest(body=_BODY, headers=_HEADERS, path="v1/chat/completions")
_RESPONSE = InferenceResponse(result={"id": "abc", "choices": []}, headers={})


def _make_ctx() -> InferenceMiddlewareContext:
    return InferenceMiddlewareContext(
        request_id="test-request-id",
        virtual_model_name="test-vm",
        workspace="test-ws",
        original_request=InferenceRequest(body=dict(_BODY), headers=dict(_HEADERS), path="v1/chat/completions"),
    )


def _plugin(
    process_request_return=None,
    process_response_return=None,
    process_request_side_effect=None,
    process_response_side_effect=None,
) -> NemoInferenceMiddleware:
    plugin = MagicMock(spec=NemoInferenceMiddleware)
    plugin.process_request = AsyncMock(
        return_value=process_request_return if process_request_return is not None else _REQUEST,
        side_effect=process_request_side_effect,
    )
    plugin.process_response = AsyncMock(
        return_value=process_response_return if process_response_return is not None else _RESPONSE,
        side_effect=process_response_side_effect,
    )
    return plugin


def _request_mock(plugin: NemoInferenceMiddleware) -> AsyncMock:
    return cast(AsyncMock, plugin.process_request)


def _response_mock(plugin: NemoInferenceMiddleware) -> AsyncMock:
    return cast(AsyncMock, plugin.process_response)


# ---------------------------------------------------------------------------
# execute_request_middleware
# ---------------------------------------------------------------------------


class TestExecuteRequestMiddleware:
    @pytest.mark.asyncio
    async def test_empty_chain_returns_request_unchanged(self):
        ctx = _make_ctx()
        result = await execute_request_middleware([], {}, ctx, _REQUEST)
        assert result is _REQUEST

    @pytest.mark.asyncio
    async def test_plugin_mutates_model_field(self):
        ctx = _make_ctx()
        mutated_body = {**_BODY, "model": "ws/other-model"}
        mutated_request = InferenceRequest(body=mutated_body, headers=_HEADERS, path="v1/chat/completions")
        plugin = _plugin(process_request_return=mutated_request)

        result = await execute_request_middleware([_call("p1")], {"p1": plugin}, ctx, _REQUEST)

        assert isinstance(result, InferenceRequest)
        assert result.body["model"] == "ws/other-model"
        _request_mock(plugin).assert_awaited_once_with(ctx, _REQUEST, {})

    @pytest.mark.asyncio
    async def test_chains_two_plugins_in_order(self):
        """Second plugin receives the first plugin's return value."""
        ctx = _make_ctx()
        step1 = InferenceRequest(body={**_BODY, "model": "ws/step1"}, headers=_HEADERS, path="v1/chat/completions")
        step2 = InferenceRequest(body={**_BODY, "model": "ws/step2"}, headers=_HEADERS, path="v1/chat/completions")
        p1 = _plugin(process_request_return=step1)
        p2 = _plugin(process_request_return=step2)

        result = await execute_request_middleware([_call("p1"), _call("p2")], {"p1": p1, "p2": p2}, ctx, _REQUEST)

        assert isinstance(result, InferenceRequest)
        assert result.body["model"] == "ws/step2"
        _request_mock(p1).assert_awaited_once_with(ctx, _REQUEST, {})
        _request_mock(p2).assert_awaited_once_with(ctx, step1, {})

    @pytest.mark.asyncio
    async def test_immediate_response_short_circuits(self):
        """Second plugin is never called when first returns ImmediateResponse."""
        ctx = _make_ctx()
        immediate = ImmediateResponse(data={"answer": 42})
        p1 = _plugin(process_request_return=immediate)
        p2 = _plugin()

        result = await execute_request_middleware([_call("p1"), _call("p2")], {"p1": p1, "p2": p2}, ctx, _REQUEST)

        assert isinstance(result, ImmediateResponse)
        assert result.data == {"answer": 42}
        _request_mock(p2).assert_not_awaited()

    @pytest.mark.asyncio
    async def test_missing_plugin_raises_503(self):
        from fastapi import HTTPException

        ctx = _make_ctx()
        with pytest.raises(HTTPException) as exc_info:
            await execute_request_middleware([_call("missing")], {}, ctx, _REQUEST)
        assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_inference_middleware_error_propagates_with_status(self):
        ctx = _make_ctx()
        plugin = _plugin(process_request_side_effect=InferenceMiddlewareError("quota exceeded", status_code=429))

        with pytest.raises(InferenceMiddlewareError) as exc_info:
            await execute_request_middleware([_call("p1")], {"p1": plugin}, ctx, _REQUEST)
        assert exc_info.value.status_code == 429
        assert exc_info.value.detail == "quota exceeded"

    @pytest.mark.asyncio
    async def test_generic_error_wrapped_as_500(self):
        ctx = _make_ctx()
        plugin = _plugin(process_request_side_effect=RuntimeError("boom"))

        with pytest.raises(InferenceMiddlewareError) as exc_info:
            await execute_request_middleware([_call("p1")], {"p1": plugin}, ctx, _REQUEST)
        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_resolved_config_passed_to_plugin(self):
        ctx = _make_ctx()
        plugin = _plugin(process_request_return=_REQUEST)
        config = {"threshold": 0.9}

        await execute_request_middleware([_call("p1", resolved_config=config)], {"p1": plugin}, ctx, _REQUEST)
        _request_mock(plugin).assert_awaited_once_with(ctx, _REQUEST, config)


# ---------------------------------------------------------------------------
# execute_response_middleware
# ---------------------------------------------------------------------------


class TestExecuteResponseMiddleware:
    @pytest.mark.asyncio
    async def test_empty_chain_returns_response_unchanged(self):
        ctx = _make_ctx()
        response = InferenceResponse(result={"id": "abc", "choices": []}, headers={})
        result = await execute_response_middleware([], {}, ctx, response)
        assert result is response

    @pytest.mark.asyncio
    async def test_chains_two_plugins(self):
        ctx = _make_ctx()
        step1 = InferenceResponse(result={"id": "step1"}, headers={})
        step2 = InferenceResponse(result={"id": "step2"}, headers={})
        p1 = _plugin(process_response_return=step1)
        p2 = _plugin(process_response_return=step2)
        initial = InferenceResponse(result={"id": "original"}, headers={})

        result = await execute_response_middleware(
            [_call("p1"), _call("p2")],
            {"p1": p1, "p2": p2},
            ctx,
            initial,
        )
        assert result is step2
        _response_mock(p1).assert_awaited_once()
        _response_mock(p2).assert_awaited_once()
        # Second plugin receives first plugin's output
        call_args = _response_mock(p2).call_args[0]
        assert call_args[1] is step1  # second positional arg is the InferenceResponse

    @pytest.mark.asyncio
    async def test_missing_plugin_passes_through(self, caplog):
        """Missing plugin logs a warning and passes the response through unchanged."""
        import logging

        ctx = _make_ctx()
        response = InferenceResponse(result={"id": "original"}, headers={})
        with caplog.at_level(logging.WARNING):
            result = await execute_response_middleware([_call("missing")], {}, ctx, response)
        assert result is response
        assert "missing" in caplog.text

    @pytest.mark.asyncio
    async def test_inference_error_propagates(self):
        ctx = _make_ctx()
        plugin = _plugin(process_response_side_effect=InferenceMiddlewareError("guardrail triggered", status_code=400))
        response = InferenceResponse(result={}, headers={})

        with pytest.raises(InferenceMiddlewareError) as exc_info:
            await execute_response_middleware([_call("p1")], {"p1": plugin}, ctx, response)
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_generic_error_wrapped_as_500(self):
        ctx = _make_ctx()
        plugin = _plugin(process_response_side_effect=RuntimeError("crash"))
        response = InferenceResponse(result={}, headers={})

        with pytest.raises(InferenceMiddlewareError) as exc_info:
            await execute_response_middleware([_call("p1")], {"p1": plugin}, ctx, response)
        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_typed_response_hook_receives_populated_envelope(self):
        seen_context: InferenceMiddlewareContext | None = None

        class _TypedPlugin(NemoInferenceMiddleware):
            async def process_response(
                self,
                ctx: InferenceMiddlewareContext,
                response: InferenceResponse,
                middleware_config: object,
            ) -> InferenceResponse:
                nonlocal seen_context
                seen_context = ctx
                assert isinstance(response.typed_body, openai_chat_types.ChatCompletion)
                response.typed_body = openai_chat_types.ChatCompletion.model_validate(
                    {
                        "id": "typed",
                        "object": "chat.completion",
                        "created": 1,
                        "model": "llama",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": "typed"},
                                "finish_reason": "stop",
                            }
                        ],
                    }
                )
                return response

        ctx = _make_ctx()
        ctx.backend_format = BackendFormat.OPENAI_CHAT
        response = build_inference_response(
            {
                "id": "raw",
                "object": "chat.completion",
                "created": 1,
                "model": "llama",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "raw"}, "finish_reason": "stop"}],
            },
            {},
            ctx.backend_format,
        )
        result = await execute_response_middleware(
            [_call("p1")],
            {"p1": _TypedPlugin()},
            ctx,
            response,
        )

        assert isinstance(result, InferenceResponse)
        assert isinstance(result.typed_body, openai_chat_types.ChatCompletion)
        assert result.typed_body.id == "typed"
        assert seen_context is not None
        assert seen_context.backend_format is BackendFormat.OPENAI_CHAT

    def test_build_inference_response_preserves_annotations_when_populating_typed_body(self):
        annotations = {"guardrails_data": {"config_ids": ["default/safety-config"]}}

        response = build_inference_response(
            {
                "id": "raw",
                "object": "chat.completion",
                "created": 1,
                "model": "llama",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "raw"}, "finish_reason": "stop"}],
            },
            {},
            BackendFormat.OPENAI_CHAT,
            response_body_annotations=annotations,
        )

        assert isinstance(response.typed_body, openai_chat_types.ChatCompletion)
        assert response.response_body_annotations == annotations
        assert response.response_body_annotations is not annotations

    @pytest.mark.asyncio
    async def test_typed_response_hook_can_mutate_typed_body_in_place(self):
        class _TypedPlugin(NemoInferenceMiddleware):
            async def process_response(
                self,
                ctx: InferenceMiddlewareContext,
                response: InferenceResponse,
                middleware_config: object,
            ) -> InferenceResponse:
                assert isinstance(response.typed_body, openai_chat_types.ChatCompletion)
                response.typed_body.id = "typed-in-place"
                return response

        ctx = _make_ctx()
        ctx.backend_format = BackendFormat.OPENAI_CHAT
        response = build_inference_response(
            {
                "id": "raw",
                "object": "chat.completion",
                "created": 1,
                "model": "llama",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "raw"}, "finish_reason": "stop"}],
            },
            {},
            ctx.backend_format,
        )
        result = await execute_response_middleware(
            [_call("p1")],
            {"p1": _TypedPlugin()},
            ctx,
            response,
        )

        assert isinstance(result.typed_body, openai_chat_types.ChatCompletion)
        assert result.typed_body.id == "typed-in-place"

    @pytest.mark.asyncio
    async def test_raw_mutation_without_clearing_typed_body_leaves_typed_canonical(self):
        class _RawMutatingPlugin(NemoInferenceMiddleware):
            async def process_response(
                self,
                ctx: InferenceMiddlewareContext,
                response: InferenceResponse,
                middleware_config: object,
            ) -> InferenceResponse:
                assert isinstance(response.result, dict)
                raw_result = cast(dict[str, object], response.result)
                raw_result["legacy"] = True
                return response

        ctx = _make_ctx()
        ctx.backend_format = BackendFormat.OPENAI_CHAT
        response = build_inference_response(
            {
                "id": "raw",
                "object": "chat.completion",
                "created": 1,
                "model": "llama",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "raw"}, "finish_reason": "stop"}],
            },
            {},
            ctx.backend_format,
        )
        result = await execute_response_middleware(
            [_call("p1")],
            {"p1": _RawMutatingPlugin()},
            ctx,
            response,
        )

        assert isinstance(result, InferenceResponse)
        assert isinstance(result.result, dict)
        raw_result = cast(dict[str, object], result.result)
        assert raw_result["legacy"] is True
        assert isinstance(result.typed_body, openai_chat_types.ChatCompletion)

    @pytest.mark.asyncio
    async def test_response_hook_can_make_raw_result_canonical_by_clearing_typed_body(self):
        class _RawPlugin(NemoInferenceMiddleware):
            async def process_response(
                self,
                ctx: InferenceMiddlewareContext,
                response: InferenceResponse,
                middleware_config: object,
            ) -> InferenceResponse:
                assert isinstance(response.result, dict)
                raw_result = cast(dict[str, object], response.result)
                raw_result["raw"] = "canonical"
                response.typed_body = None
                return response

        ctx = _make_ctx()
        ctx.backend_format = BackendFormat.OPENAI_CHAT
        response = build_inference_response(
            {
                "id": "raw",
                "object": "chat.completion",
                "created": 1,
                "model": "llama",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "raw"}, "finish_reason": "stop"}],
            },
            {},
            ctx.backend_format,
        )
        result = await execute_response_middleware(
            [_call("p1")],
            {"p1": _RawPlugin()},
            ctx,
            response,
        )

        assert isinstance(result.result, dict)
        raw_result = cast(dict[str, object], result.result)
        assert raw_result["raw"] == "canonical"
        assert result.typed_body is None

    @pytest.mark.asyncio
    async def test_default_response_hook_preserves_typed_body(self):
        class _NoopPlugin(NemoInferenceMiddleware):
            pass

        ctx = _make_ctx()
        ctx.backend_format = BackendFormat.OPENAI_CHAT
        response = build_inference_response(
            {
                "id": "raw",
                "object": "chat.completion",
                "created": 1,
                "model": "llama",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "raw"}, "finish_reason": "stop"}],
            },
            {},
            ctx.backend_format,
        )
        result = await execute_response_middleware(
            [_call("p1")],
            {"p1": _NoopPlugin()},
            ctx,
            response,
        )

        assert isinstance(result, InferenceResponse)
        assert isinstance(result.typed_body, openai_chat_types.ChatCompletion)

    @pytest.mark.asyncio
    async def test_spec_mock_preserves_typed_response_envelope(self):
        plugin = MagicMock(spec=NemoInferenceMiddleware)
        plugin.process_response = AsyncMock(side_effect=lambda _ctx, response, *_args: response)

        ctx = _make_ctx()
        ctx.backend_format = BackendFormat.OPENAI_CHAT
        response = build_inference_response(
            {
                "id": "raw",
                "object": "chat.completion",
                "created": 1,
                "model": "llama",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "raw"}, "finish_reason": "stop"}],
            },
            {},
            ctx.backend_format,
        )
        result = await execute_response_middleware(
            [_call("p1")],
            {"p1": plugin},
            ctx,
            response,
        )

        assert isinstance(result, InferenceResponse)
        _response_mock(plugin).assert_awaited_once()
        assert isinstance(result.typed_body, openai_chat_types.ChatCompletion)

    @pytest.mark.asyncio
    async def test_streaming_envelope_skips_unrecognised_chunks_keeps_typed_body(self):
        """Unrecognised chunks are skipped from the typed view; ``response.typed_body``
        is not torn down. The typed iterator yields only
        validated chunks; ``raw_chunks()`` still delivers everything for
        wire-level serialization."""

        async def raw_stream():
            yield {"id": "not-enough-fields"}  # malformed — skipped from typed view
            yield {  # valid — comes through
                "id": "chunk-1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "llama",
                "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": "stop"}],
            }

        response = build_inference_response(raw_stream(), {}, BackendFormat.OPENAI_CHAT)
        typed_body_before = response.typed_body
        assert typed_body_before is not None

        chunks = [chunk async for chunk in cast(AsyncIterator[object], response.typed_body)]

        # Malformed chunk skipped, valid chunk yielded.
        assert len(chunks) == 1
        # typed_body is NOT nulled — the typed view stays useful.
        assert response.typed_body is typed_body_before


# ---------------------------------------------------------------------------
# execute_post_response_middleware
# ---------------------------------------------------------------------------


class TestExecutePostResponseMiddleware:
    @pytest.mark.asyncio
    async def test_calls_each_plugin(self):
        ctx = _make_ctx()
        p1, p2 = _plugin(), _plugin()
        response = InferenceResponse(result={"id": "resp"}, headers={})
        await execute_post_response_middleware(
            [_call("p1"), _call("p2")],
            {"p1": p1, "p2": p2},
            ctx,
            response,
        )
        _response_mock(p1).assert_awaited_once()
        _response_mock(p2).assert_awaited_once()

    @pytest.mark.asyncio
    async def test_swallows_inference_error(self):
        ctx = _make_ctx()
        plugin = _plugin(process_response_side_effect=InferenceMiddlewareError("fail", status_code=500))
        response = InferenceResponse(result={}, headers={})
        # Must not raise
        await execute_post_response_middleware([_call("p1")], {"p1": plugin}, ctx, response)

    @pytest.mark.asyncio
    async def test_swallows_generic_error(self, caplog):
        """Unexpected exceptions are logged and swallowed."""
        import logging

        ctx = _make_ctx()
        plugin = _plugin(process_response_side_effect=RuntimeError("explode"))
        response = InferenceResponse(result={}, headers={})

        with caplog.at_level(logging.WARNING):
            await execute_post_response_middleware([_call("p1")], {"p1": plugin}, ctx, response)
        assert "p1" in caplog.text

    @pytest.mark.asyncio
    async def test_missing_plugin_is_silently_skipped(self):
        ctx = _make_ctx()
        response = InferenceResponse(result={}, headers={})
        # No plugins dict, no calls → no error
        await execute_post_response_middleware([_call("absent")], {}, ctx, response)

    @pytest.mark.asyncio
    async def test_continues_after_error_in_one_plugin(self):
        """If one plugin raises, subsequent plugins still run."""
        ctx = _make_ctx()
        p1 = _plugin(process_response_side_effect=RuntimeError("oops"))
        p2 = _plugin()
        response = InferenceResponse(result={}, headers={})

        await execute_post_response_middleware(
            [_call("p1"), _call("p2")],
            {"p1": p1, "p2": p2},
            ctx,
            response,
        )
        _response_mock(p2).assert_awaited_once()
