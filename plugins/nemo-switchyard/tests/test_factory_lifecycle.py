# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minute lifecycle invariants per factory at the SwitchyardMiddleware interface.

For each factory (random_routing, translate same-format, translate
cross-format), verify that process_request/process_response only mutates
what they're supposed to and leaves everything else untouched. The goal is
to catch silent in-place mutations, header drops, path drift, and
typed_body going out of sync with body.
"""

from __future__ import annotations

import copy
from typing import Any

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
from openai.types.chat import ChatCompletion


@pytest.fixture
async def middleware() -> SwitchyardMiddleware:
    mw = SwitchyardMiddleware()
    await mw.on_startup()
    yield mw
    await mw.on_shutdown()


def _make_openai_request(model: str, path: str = "v1/chat/completions") -> InferenceRequest:
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0.42,
        "max_tokens": 128,
        "top_p": 0.9,
    }
    return InferenceRequest(
        body=body,
        headers={"x-original-id": "abc-123", "authorization": "Bearer test"},
        path=path,
        typed_body=body,
    )


def _make_anthropic_request(model: str, path: str = "v1/messages") -> InferenceRequest:
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 128,
    }
    return InferenceRequest(
        body=body,
        headers={"x-original-id": "abc-123"},
        path=path,
        typed_body=body,
    )


def _make_chat_completion(model: str = "gpt-4") -> ChatCompletion:
    """Minimal valid ChatCompletion for response-side tests that need a typed body."""
    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl-test",
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "logprobs": None,
                }
            ],
            "created": 0,
            "model": model,
            "object": "chat.completion",
        }
    )


def _make_ctx(workspace: str, vm_name: str, request: InferenceRequest) -> InferenceMiddlewareContext:
    """Build a context whose original_request is a deep copy — IGW gives plugins a
    distinct working request, and we want to detect any plugin reaching back into
    the snapshot.
    """
    return InferenceMiddlewareContext(
        request_id="test-req",
        workspace=workspace,
        virtual_model_name=vm_name,
        original_request=InferenceRequest(
            body=copy.deepcopy(request.body),
            headers=dict(request.headers),
            path=request.path,
        ),
    )


class TestRandomRoutingLifecycle:
    """Random routing must rewrite ONLY body['model']; everything else preserved."""

    @pytest.fixture
    def vm(self) -> VirtualModel:
        return VirtualModel(
            id="vm-rr",
            workspace="ws",
            name="rr",
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
                        "strong_probability": 1.0,
                        "rng_seed": 1,
                        "enable_stats": False,
                    },
                )
            ],
            response_middleware=[],
            post_response_middleware=[],
        )

    @pytest.mark.asyncio
    async def test_only_model_field_mutates(self, middleware: SwitchyardMiddleware, vm: VirtualModel) -> None:
        await middleware.on_virtual_model_upserted(vm)
        request = _make_openai_request(model="ws/rr")
        non_model_before = {k: copy.deepcopy(v) for k, v in request.body.items() if k != "model"}
        headers_before = dict(request.headers)
        path_before = request.path
        ctx = _make_ctx("ws", "rr", request)
        ctx_original_before = copy.deepcopy(ctx.original_request.body)

        result = await middleware.process_request(ctx, request, {"config_type": "random_routing"})

        # strong_probability=1.0 + seed=1 → strong model.
        assert result is request
        assert result.body["model"] == "ws/strong"
        # Every other body field is untouched, byte-for-byte.
        for k, v in non_model_before.items():
            assert result.body[k] == v, f"field {k!r} mutated"
        # No new keys snuck into the body.
        assert set(result.body.keys()) == set(non_model_before) | {"model"}
        # Headers and path preserved.
        assert result.headers == headers_before
        assert result.path == path_before
        # typed_body is same object as body — in sync, no drift.
        assert result.typed_body is result.body
        assert result.body["model"] == "ws/strong"
        # Original request snapshot still has the pre-routing model.
        assert ctx.original_request.body == ctx_original_before
        assert ctx.original_request.body["model"] == "ws/rr"

    @pytest.mark.asyncio
    async def test_response_with_no_typed_body_raises_500(
        self, middleware: SwitchyardMiddleware, vm: VirtualModel
    ) -> None:
        """typed_body=None is a contract violation — raises InferenceMiddlewareError 500."""
        from nemo_helix_plugin.inference_middleware import InferenceMiddlewareError

        await middleware.on_virtual_model_upserted(vm)
        ctx = _make_ctx("ws", "rr", _make_openai_request(model="ws/rr"))
        response = InferenceResponse(
            result={"id": "x", "choices": [{"message": {"content": "hi"}}]},
            headers={"x-resp": "1"},
        )

        with pytest.raises(InferenceMiddlewareError) as exc:
            await middleware.process_response(ctx, response, {"config_type": "random_routing"})
        assert exc.value.status_code == 500


class TestTranslateSameFormatLifecycle:
    """Same-format translate (OpenAI in, OpenAI-backed model): body shape preserved
    field-for-field; path stays on the OpenAI route."""

    @pytest.fixture
    def vm(self) -> VirtualModel:
        return VirtualModel(
            id="vm-ts",
            workspace="ws",
            name="ts",
            models=[VirtualModelInferenceConfig(model="ws/m-openai", backend_format=BackendFormat.OPENAI_CHAT)],
            request_middleware=[
                MiddlewareCall(
                    name="nemo-switchyard",
                    config_type="translate",
                    config={"target_format": "openai_chat", "enable_stats": False},
                )
            ],
            response_middleware=[],
            post_response_middleware=[],
        )

    @pytest.mark.asyncio
    async def test_body_path_typed_body_preserved(self, middleware: SwitchyardMiddleware, vm: VirtualModel) -> None:
        await middleware.on_virtual_model_upserted(vm)
        request = _make_openai_request(model="ws/m-openai")
        body_before = copy.deepcopy(request.body)
        headers_before = dict(request.headers)
        ctx = _make_ctx("ws", "ts", request)

        result = await middleware.process_request(ctx, request, {"config_type": "translate"})

        assert result is request
        assert result.body == body_before  # in→OpenAI, out→OpenAI: identical body.
        assert result.headers == headers_before
        # Path stays on the OpenAI route.
        assert result.path == "v1/chat/completions"
        # typed_body is same object as body — always in sync.
        assert result.typed_body is result.body


class TestTranslateCrossFormatLifecycle:
    """Cross-format translate: body shape changes (OpenAI ↔ Anthropic), path follows
    target format, typed_body is still same object as body. Headers preserved."""

    @pytest.fixture
    def vm_to_anthropic(self) -> VirtualModel:
        # OpenAI request in → Anthropic-backed model out.
        return VirtualModel(
            id="vm-tc",
            workspace="ws",
            name="tc",
            models=[VirtualModelInferenceConfig(model="ws/m-claude", backend_format=BackendFormat.ANTHROPIC_MESSAGES)],
            request_middleware=[
                MiddlewareCall(
                    name="nemo-switchyard",
                    config_type="translate",
                    config={"target_format": "anthropic", "enable_stats": False},
                )
            ],
            response_middleware=[],
            post_response_middleware=[],
        )

    @pytest.mark.asyncio
    async def test_path_rewritten_typed_body_in_sync_headers_preserved(
        self, middleware: SwitchyardMiddleware, vm_to_anthropic: VirtualModel
    ) -> None:
        await middleware.on_virtual_model_upserted(vm_to_anthropic)
        # The body model must match an entry in the translate models list so
        # ModelFormatLookupProcessor stamps CTX_TARGET_FORMAT.
        request = _make_openai_request(model="ws/m-claude")
        headers_before = dict(request.headers)
        ctx = _make_ctx("ws", "tc", request)
        ctx_original_before = copy.deepcopy(ctx.original_request.body)

        result = await middleware.process_request(ctx, request, {"config_type": "translate"})

        assert result is request
        # Path follows the anthropic target.
        assert result.path == "v1/messages"
        # Body was translated to Anthropic shape; messages field must still be present.
        assert "messages" in result.body
        # typed_body is same object as body — valid for the new (Anthropic) format.
        assert result.typed_body is result.body
        # Headers untouched.
        assert result.headers == headers_before
        # Original snapshot retains pre-translate body shape (still has top_p,
        # temperature in OpenAI shape; no anthropic-specific transformations).
        assert ctx.original_request.body == ctx_original_before


class TestTypedBodyClearedAcrossFactories:
    """Across every factory, after process_request returns, request.typed_body is request.body.
    write_back_request always sets body and typed_body to the same object. Downstream middleware
    gets both views pointing at the same post-pipeline dict."""

    @pytest.fixture(params=["random_routing", "translate"])
    def vm_and_config(self, request) -> tuple[VirtualModel, str]:
        config_type = request.param
        if config_type == "random_routing":
            cfg = {
                "strong": {"model": "ws/m1"},
                "weak": {"model": "ws/m1"},
                "strong_probability": 1.0,
                "enable_stats": False,
            }
        elif config_type == "translate":
            cfg = {"target_format": "openai_chat", "enable_stats": False}
        else:
            cfg = {"enable_stats": False}

        vm = VirtualModel(
            id=f"vm-id-{config_type}",
            workspace="ws",
            name=f"vm-{config_type}",
            models=[VirtualModelInferenceConfig(model="ws/m1", backend_format=BackendFormat.OPENAI_CHAT)],
            request_middleware=[MiddlewareCall(name="nemo-switchyard", config_type=config_type, config=cfg)],
            response_middleware=[],
            post_response_middleware=[],
        )
        return vm, config_type

    @pytest.mark.asyncio
    async def test_typed_body_in_sync_after_pipeline(
        self,
        middleware: SwitchyardMiddleware,
        vm_and_config: tuple[VirtualModel, str],
    ) -> None:
        vm, config_type = vm_and_config
        await middleware.on_virtual_model_upserted(vm)
        request = _make_openai_request(model="ws/m1")
        ctx = _make_ctx("ws", vm.name, request)

        result = await middleware.process_request(ctx, request, {"config_type": config_type})

        # write_back_request sets body = typed_body = processed.body — always in sync.
        assert result.typed_body is result.body, f"{config_type}: typed_body was not cleared by write_back_request."


# Response lifecycle ---------------------------------------------------------
#
# The plugin's process_response contract:
#   - typed_body is None  → return response unchanged (no factory lookup).
#   - typed_body not None → run factory's response pipeline on it, write
#                              back to typed_body. response.result is NOT
#                              touched by the middleware here.
#
# With enable_stats=False, random_routing has an EMPTY
# response pipeline, so typed_body must come back identity-equal. Translate
# has FormatTranslateResponseProcessor; for streaming it deliberately
# passes through unchanged (cross-format streaming is a follow-up); for the
# same-format case CTX_ORIGINAL_FORMAT is absent on a fresh sy_context so it
# also no-ops. These are the only behaviors we lock down here.


def _vm_for(config_type: str, phases: tuple[str, ...] = ("request",)) -> VirtualModel:
    """Build a VM with the given switchyard config listed under each requested phase."""
    if config_type == "random_routing":
        cfg: dict[str, Any] = {
            "strong": {"model": "ws/m1"},
            "weak": {"model": "ws/m1"},
            "strong_probability": 1.0,
            "enable_stats": False,
        }
    elif config_type == "translate":
        cfg = {"target_format": "openai_chat", "enable_stats": False}
    else:
        cfg = {"enable_stats": False}
    call = MiddlewareCall(name="nemo-switchyard", config_type=config_type, config=cfg)
    return VirtualModel(
        id=f"vm-resp-{config_type}-{'-'.join(phases)}",
        workspace="ws",
        name=f"resp-{config_type}",
        models=[VirtualModelInferenceConfig(model="ws/m1", backend_format=BackendFormat.OPENAI_CHAT)],
        request_middleware=[call] if "request" in phases else [],
        response_middleware=[call] if "response" in phases else [],
        post_response_middleware=[],
    )


async def _streaming_chunks() -> Any:
    """Mock OpenAI-style streaming chunks (no Switchyard typing — plugin should
    pass through without trying to parse chunks)."""
    chunks = [
        {"choices": [{"index": 0, "delta": {"role": "assistant"}}]},
        {"choices": [{"index": 0, "delta": {"content": "hi"}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    for c in chunks:
        yield c


class TestResponseLifecyclePerFactory:
    """For each factory, verify response handling is correct in both
    non-streaming (typed_body=None and typed_body populated) and streaming
    (typed_body is AsyncIterator) shapes."""

    @pytest.fixture(params=["random_routing", "translate"])
    def vm_and_config(self, request) -> tuple[VirtualModel, str]:
        return _vm_for(request.param, phases=("request", "response")), request.param

    @pytest.mark.asyncio
    async def test_response_typed_none_raises_500(
        self,
        middleware: SwitchyardMiddleware,
        vm_and_config: tuple[VirtualModel, str],
    ) -> None:
        """typed_body=None is a contract violation — raises InferenceMiddlewareError 500.
        IGW must always populate typed_body for recognised backends."""
        from nemo_helix_plugin.inference_middleware import InferenceMiddlewareError

        vm, config_type = vm_and_config
        await middleware.on_virtual_model_upserted(vm)
        ctx = _make_ctx("ws", vm.name, _make_openai_request(model="ws/m1"))

        response = InferenceResponse(
            result={"id": "x"},
            headers={"x-resp": "1"},
        )

        with pytest.raises(InferenceMiddlewareError) as exc:
            await middleware.process_response(ctx, response, {"config_type": config_type})
        assert exc.value.status_code == 500

    @pytest.mark.asyncio
    async def test_streaming_response_typed_body_unknown_format_passes_through(
        self,
        middleware: SwitchyardMiddleware,
        vm_and_config: tuple[VirtualModel, str],
    ) -> None:
        """When typed_body is a TypedResponseStream but ctx.backend_format is None,
        _wrap_streaming returns None and the response passes through unchanged.
        This covers the case where IGW couldn't determine the backend format."""
        from nhx.core.inference_gateway.api.typed_response import TypedResponseStream

        vm, config_type = vm_and_config
        await middleware.on_virtual_model_upserted(vm)
        ctx = _make_ctx("ws", vm.name, _make_openai_request(model="ws/m1"))
        ctx.backend_format = None  # _wrap_streaming returns None → pass through

        async def _raw_chunks():
            for c in [{"delta": "a"}, {"delta": "b"}]:
                yield c

        typed_stream = TypedResponseStream(BackendFormat.OPENAI_CHAT, _raw_chunks())
        response = InferenceResponse(result={"raw": True}, headers={"x-resp": "1"}, typed_body=typed_stream)

        out = await middleware.process_response(ctx, response, {"config_type": config_type})

        assert out is response
        assert out.typed_body is typed_stream  # unchanged — pass-through

    @pytest.mark.asyncio
    async def test_response_headers_not_mutated(
        self,
        middleware: SwitchyardMiddleware,
        vm_and_config: tuple[VirtualModel, str],
    ) -> None:
        """Headers passed into process_response must not be mutated by any
        factory pipeline (no header processor exists in any of these chains)."""
        vm, config_type = vm_and_config
        await middleware.on_virtual_model_upserted(vm)
        ctx = _make_ctx("ws", vm.name, _make_openai_request(model="ws/m1"))

        response = InferenceResponse(
            result={"id": "x"},
            headers={"x-resp-id": "r1", "x-trace": "t1"},
            typed_body=_make_chat_completion(),
        )
        headers_id_before = id(response.headers)
        headers_before = dict(response.headers)

        await middleware.process_response(ctx, response, {"config_type": config_type})

        # Same dict instance — no replacement, no copy-back.
        assert id(response.headers) == headers_id_before
        assert response.headers == headers_before


class TestEmptyPipelineResponseIdentity:
    """Passthrough and random_routing have empty response pipelines (with
    enable_stats=False). When IGW provides a typed_body and the user opted into
    response-side processing, the empty pipeline returns typed_body unchanged —
    write_back_response sets typed_body to processed.body — the same pydantic object."""

    @pytest.fixture(params=["random_routing"])
    def vm_and_config(self, request) -> tuple[VirtualModel, str]:
        return _vm_for(request.param, phases=("request", "response")), request.param

    @pytest.mark.asyncio
    async def test_typed_body_returned_identity_equal(
        self,
        middleware: SwitchyardMiddleware,
        vm_and_config: tuple[VirtualModel, str],
    ) -> None:
        vm, config_type = vm_and_config
        await middleware.on_virtual_model_upserted(vm)
        ctx = _make_ctx("ws", vm.name, _make_openai_request(model="ws/m1"))

        # Use a real ChatCompletion — _wrap_non_streaming requires ChatCompletion or Message.
        typed = _make_chat_completion()
        response = InferenceResponse(result={"id": "x"}, headers={}, typed_body=typed)

        out = await middleware.process_response(ctx, response, {"config_type": config_type})

        assert out is response
        # Empty pipeline returns the pydantic model unchanged; write_back_response preserves identity.
        assert out.typed_body is typed


class TestStreamingTypedResult:
    """When IGW gives us a streaming typed_body (a TypedResponseStream wrapping
    the backend stream), the middleware wraps it into a Switchyard ChatResponse,
    runs the pipeline (routing: empty), then write_back_response sets
    response.result = processed.stream and clears typed_body.
    The stream must not be consumed during this process."""

    @pytest.fixture(params=["random_routing"])
    def vm_and_config(self, request) -> tuple[VirtualModel, str]:
        return _vm_for(request.param, phases=("request", "response")), request.param

    @pytest.mark.asyncio
    async def test_streaming_typed_body_processed_result_set_typed_body_cleared(
        self,
        middleware: SwitchyardMiddleware,
        vm_and_config: tuple[VirtualModel, str],
    ) -> None:
        """TypedResponseStream typed_body is wrapped into a StreamingChatResponse,
        the pipeline runs (no-op for routing), and write_back_response sets
        response.result to processed.stream with typed_body cleared to None.
        The underlying chunks remain unconsumed."""
        from nhx.core.inference_gateway.api.typed_response import TypedResponseStream

        vm, config_type = vm_and_config
        await middleware.on_virtual_model_upserted(vm)
        ctx = _make_ctx("ws", vm.name, _make_openai_request(model="ws/m1"))
        ctx.backend_format = BackendFormat.OPENAI_CHAT

        chunks_received: list[Any] = []

        async def _source():
            for chunk_dict in [{"delta": "a"}, {"delta": "b"}]:
                chunks_received.append(chunk_dict)
                yield chunk_dict

        typed_stream = TypedResponseStream(BackendFormat.OPENAI_CHAT, _source())
        response = InferenceResponse(result={"raw": True}, headers={"x-resp": "1"}, typed_body=typed_stream)

        out = await middleware.process_response(ctx, response, {"config_type": config_type})

        assert out is response
        # write_back_response: typed_body cleared, result replaced with processed.stream
        assert out.typed_body is None
        assert out.result is not None
        # Chunks have NOT been consumed yet — pipeline didn't iterate the stream.
        assert chunks_received == []

    @pytest.mark.asyncio
    async def test_streaming_unknown_backend_format_passes_through(
        self,
        middleware: SwitchyardMiddleware,
        vm_and_config: tuple[VirtualModel, str],
    ) -> None:
        """When ctx.backend_format is None, _wrap_streaming returns None and the
        response is passed through unchanged."""
        from nhx.core.inference_gateway.api.typed_response import TypedResponseStream

        vm, config_type = vm_and_config
        await middleware.on_virtual_model_upserted(vm)
        ctx = _make_ctx("ws", vm.name, _make_openai_request(model="ws/m1"))
        # backend_format is None by default — _wrap_streaming returns None

        async def _src():
            return
            yield  # noqa: unreachable

        typed_stream = TypedResponseStream(BackendFormat.OPENAI_CHAT, _src())
        response = InferenceResponse(result={"raw": True}, headers={}, typed_body=typed_stream)

        out = await middleware.process_response(ctx, response, {"config_type": config_type})

        assert out is response
        assert out.typed_body is typed_stream  # unchanged — pass-through


class TestStreamingResponseTranslation:
    """End-to-end: translate factory on response_middleware with a TypedResponseStream.

    Verifies the full streaming response translation path:
    TypedResponseStream → _wrap_streaming → StreamingChatResponse → translate
    pipeline (FormatTranslateResponseProcessor) → AnthropicStreamingChatResponse
    → write_back_response → response.result = translated_stream, typed_body = None.
    """

    @pytest.mark.asyncio
    async def test_translate_response_openai_stream_to_anthropic(
        self,
        middleware: SwitchyardMiddleware,
    ) -> None:
        """OpenAI TypedResponseStream through translate response pipeline produces
        Anthropic-format events in response.result; typed_body cleared to None."""
        from nhx.core.inference_gateway.api.typed_response import TypedResponseStream
        from switchyard.lib.chat_request.base import ChatRequestType
        from switchyard.lib.proxy_context import CTX_ORIGINAL_FORMAT

        # VM: OpenAI request, Anthropic backend — translate response back to OpenAI.
        vm = VirtualModel(
            id="vm-resp-translate",
            workspace="ws",
            name="resp-translate",
            models=[VirtualModelInferenceConfig(model="ws/claude", backend_format=BackendFormat.ANTHROPIC_MESSAGES)],
            request_middleware=[
                MiddlewareCall(
                    name="nemo-switchyard",
                    config_type="translate",
                    config={"target_format": "anthropic", "enable_stats": False},
                )
            ],
            response_middleware=[
                MiddlewareCall(
                    name="nemo-switchyard",
                    config_type="translate",
                    config={"target_format": "anthropic", "enable_stats": False},
                )
            ],
            post_response_middleware=[],
        )
        await middleware.on_virtual_model_upserted(vm)

        # Build a ctx that simulates what IGW would stamp after routing:
        # - backend_format: ANTHROPIC_MESSAGES (the backend that was called)
        # - sy_metadata carrying CTX_ORIGINAL_FORMAT: OPENAI_CHAT (client's original format)
        request = _make_openai_request(model="ws/claude")
        ctx = _make_ctx("ws", vm.name, request)
        ctx.backend_format = BackendFormat.ANTHROPIC_MESSAGES

        # Simulate: request pipeline ran and stamped CTX_ORIGINAL_FORMAT.
        from nemo_switchyard.middleware import _PLUGIN_STATE_NAMESPACE, _SY_METADATA_STATE_KEY

        ctx.state(_PLUGIN_STATE_NAMESPACE).set(
            _SY_METADATA_STATE_KEY,
            {CTX_ORIGINAL_FORMAT: ChatRequestType.OPENAI_CHAT},
        )

        # The backend returned Anthropic-format streaming chunks — IGW wrapped them
        # in a TypedResponseStream(ANTHROPIC_MESSAGES, ...).
        async def _anthropic_stream():
            yield {
                "type": "message_start",
                "message": {
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": "claude-3",
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 5, "output_tokens": 0},
                },
            }
            yield {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}
            yield {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hello"}}
            yield {"type": "content_block_stop", "index": 0}
            yield {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 3},
            }
            yield {"type": "message_stop"}

        typed_stream = TypedResponseStream(BackendFormat.ANTHROPIC_MESSAGES, _anthropic_stream())
        response = InferenceResponse(result={"raw": True}, headers={}, typed_body=typed_stream)

        out = await middleware.process_response(ctx, response, {"config_type": "translate"})

        assert out is response
        # write_back_response: result set to translated stream, typed_body cleared.
        assert out.typed_body is None
        assert out.result is not None
        assert out.result is not typed_stream  # pipeline produced a new iterator

        # Consume the translated stream and verify it contains OpenAI-shaped chunks.

        translated_stream = out.result
        chunks = [chunk async for chunk in translated_stream]  # type: ignore[union-attr]
        assert len(chunks) > 0
        # OpenAI chat chunk shape: has 'choices' key.
        for chunk in chunks:
            if isinstance(chunk, dict):
                assert "choices" in chunk or "object" in chunk, f"Expected OpenAI chat chunk shape, got: {chunk}"
            else:
                # Pydantic model — IGW's _sse_gen will model_dump it.
                assert hasattr(chunk, "choices"), f"Expected ChatCompletionChunk, got {type(chunk)}"


class TestStreamingRequestLifecycle:
    """Streaming requests flow through the full pipeline — no translate guard.
    Other config_types (random_routing) also work unchanged."""

    @pytest.mark.asyncio
    async def test_stream_with_translate_succeeds(self, middleware: SwitchyardMiddleware) -> None:
        """Translate on a streaming request must succeed end-to-end now that
        the response bridge is in place."""
        vm = _vm_for("translate", phases=("request",))
        await middleware.on_virtual_model_upserted(vm)

        body = {"model": "ws/m1", "messages": [{"role": "user", "content": "hi"}], "stream": True}
        request = InferenceRequest(body=body, headers={}, path="v1/chat/completions", typed_body=body)
        ctx = _make_ctx("ws", vm.name, request)

        # Should not raise — translate guard has been removed.
        result = await middleware.process_request(ctx, request, {"config_type": "translate"})
        assert result is request

    @pytest.mark.asyncio
    async def test_stream_with_random_routing_routes_and_preserves_invariants(
        self, middleware: SwitchyardMiddleware
    ) -> None:
        """Random routing on a streaming request must:
        - Rewrite body['model'] to the chosen tier (proving routing actually ran).
        - Preserve every other body field byte-for-byte (incl. stream=True).
        - Not introduce any new body keys.
        - Preserve headers and path.
        - Keep typed_body identical to result.body (no drift).

        Regression guard against streaming silently bypassing the request
        pipeline or routing leaving typed_body/body out of sync."""
        vm = VirtualModel(
            id="vm-stream-rr",
            workspace="ws",
            name="stream-rr",
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
                        "strong_probability": 1.0,  # deterministic: always strong
                        "rng_seed": 1,
                        "enable_stats": False,
                    },
                )
            ],
            response_middleware=[],
            post_response_middleware=[],
        )
        await middleware.on_virtual_model_upserted(vm)

        body = {
            "model": "ws/stream-rr",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "temperature": 0.42,
            "max_tokens": 128,
            "top_p": 0.9,
        }
        non_model_before = {k: copy.deepcopy(v) for k, v in body.items() if k != "model"}
        request = InferenceRequest(body=body, headers={"x-trace": "t1"}, path="v1/chat/completions", typed_body=body)
        ctx = _make_ctx("ws", vm.name, request)

        result = await middleware.process_request(ctx, request, {"config_type": "random_routing"})

        assert result is request
        # Routing actually ran on the streaming request.
        assert result.body["model"] == "ws/strong"
        # Every other body field byte-for-byte unchanged, INCLUDING stream=True.
        for k, v in non_model_before.items():
            assert result.body[k] == v, f"streaming-routing mutated unrelated field {k!r}"
        # No keys snuck in.
        assert set(result.body.keys()) == set(non_model_before) | {"model"}
        # Headers and path preserved.
        assert result.headers == {"x-trace": "t1"}
        assert result.path == "v1/chat/completions"
        # typed_body is same object as body — no drift.
        assert result.typed_body is result.body
        assert result.body["model"] == "ws/strong"
        assert result.body["stream"] is True


class TestSwitchyardMetadataBridge:
    """Bridge SY ProxyContext.metadata across IGW hooks via ctx.state.

    In standalone Switchyard the chain runner threads one ProxyContext through
    both pipelines, so anything stamped in the request pipeline is visible to
    the response pipeline. IGW invokes our hooks separately with their own
    scopes, so we explicitly carry the SY metadata through
    InferenceMiddlewareContext._state. These tests lock in the bridge contract
    so an upstream Switchyard fix that stamps CTX_ORIGINAL_REQUEST etc. flows
    through to process_response without further plumbing."""

    @pytest.mark.asyncio
    async def test_request_pipeline_stamps_propagate_to_response_hook(self, middleware: SwitchyardMiddleware) -> None:
        """End-to-end: stamp something on the SY ctx during process_request,
        confirm it shows up on the SY ctx in process_response."""
        from switchyard.lib.proxy_context import ProxyContext as _ProxyContext
        from switchyard.lib.registry import lookup as _lookup
        from switchyard.lib.request_pipeline import RequestPipeline
        from switchyard.lib.response_pipeline import ResponsePipeline
        from switchyard.lib.roles import RequestProcessor, ResponseProcessor

        vm = _vm_for("random_routing", phases=("request", "response"))
        await middleware.on_virtual_model_upserted(vm)

        # Patch the registered factory to stamp a sentinel during request and
        # capture what it sees on the response side.
        from nemo_switchyard import _state

        vm_key = f"{vm.workspace}/{vm.name}"
        cfg_hash = _state.VM_NAME_TO_CONFIG_HASH[(vm_key, "random_routing", "request")]
        factory = _lookup(_state.FACTORIES_BY_CONFIG_HASH[cfg_hash])

        captured: dict[str, Any] = {}

        class _StampingRequestProcessor(RequestProcessor):
            async def process(self, ctx: _ProxyContext, request: Any) -> Any:
                ctx.metadata["test_carry_key"] = "test_carry_value"
                return request

        class _CapturingResponseProcessor(ResponseProcessor):
            async def process(self, ctx: _ProxyContext, response: Any) -> Any:
                captured["seen"] = ctx.metadata.get("test_carry_key")
                return response

        factory.build_request_pipeline = lambda config: RequestPipeline([_StampingRequestProcessor()])
        factory.build_response_pipeline = lambda config: ResponsePipeline([_CapturingResponseProcessor()])

        request = _make_openai_request(model="ws/m1")
        ctx = _make_ctx("ws", vm.name, request)

        await middleware.process_request(ctx, request, {"config_type": "random_routing"})

        # Non-streaming ChatCompletion goes through the pipeline — simpler than a stream.
        response = InferenceResponse(result={"id": "x"}, headers={}, typed_body=_make_chat_completion())
        await middleware.process_response(ctx, response, {"config_type": "random_routing"})

        assert captured.get("seen") == "test_carry_value", (
            "metadata stamped during request pipeline did not reach response pipeline"
        )

    @pytest.mark.asyncio
    async def test_bridge_does_not_leak_headers(self, middleware: SwitchyardMiddleware) -> None:
        """The 'headers' key is per-hook (request.headers vs response.headers).
        Carrying request.headers across to response time would shadow the
        response headers — verify it doesn't."""
        vm = _vm_for("random_routing", phases=("request", "response"))
        await middleware.on_virtual_model_upserted(vm)

        request = _make_openai_request(model="ws/m1")
        request.headers = {"x-from-request": "yes"}
        ctx = _make_ctx("ws", vm.name, request)

        await middleware.process_request(ctx, request, {"config_type": "random_routing"})

        # Inspect the bridged metadata: should NOT include "headers".
        from nemo_switchyard.middleware import (
            _PLUGIN_STATE_NAMESPACE,
            _SY_METADATA_STATE_KEY,
        )

        bridged = ctx.state(_PLUGIN_STATE_NAMESPACE).get(_SY_METADATA_STATE_KEY, {})
        assert "headers" not in bridged

    @pytest.mark.asyncio
    async def test_bridge_per_request_scoped_no_cross_request_leak(self, middleware: SwitchyardMiddleware) -> None:
        """Two concurrent requests must not see each other's bridged metadata —
        ctx.state lives on InferenceMiddlewareContext._state, which IGW
        constructs per-request."""
        vm = _vm_for("random_routing", phases=("request", "response"))
        await middleware.on_virtual_model_upserted(vm)

        from nemo_switchyard.middleware import (
            _PLUGIN_STATE_NAMESPACE,
            _SY_METADATA_STATE_KEY,
        )

        req_a = _make_openai_request(model="ws/m1")
        ctx_a = _make_ctx("ws", vm.name, req_a)
        ctx_a.request_id = "req-A"
        ctx_a.state(_PLUGIN_STATE_NAMESPACE).set(_SY_METADATA_STATE_KEY, {"marker": "A"})

        req_b = _make_openai_request(model="ws/m1")
        ctx_b = _make_ctx("ws", vm.name, req_b)
        ctx_b.request_id = "req-B"

        # ctx_b's bridged metadata is independent of ctx_a's.
        assert ctx_b.state(_PLUGIN_STATE_NAMESPACE).get(_SY_METADATA_STATE_KEY) is None
        assert ctx_a.state(_PLUGIN_STATE_NAMESPACE).get(_SY_METADATA_STATE_KEY) == {"marker": "A"}


class TestPhaseAuthority:
    """Each list (request_middleware, response_middleware) is authoritative for
    its phase. Listing switchyard under one list does NOT implicitly enable the
    other phase — users must list it explicitly under both if they want both."""

    @pytest.mark.asyncio
    async def test_response_phase_rejected_when_only_in_request_middleware(
        self, middleware: SwitchyardMiddleware
    ) -> None:
        """User listed switchyard only under request_middleware. process_response
        must refuse with 400 (and a hint pointing at response_middleware)."""
        from nemo_helix_plugin.inference_middleware import InferenceMiddlewareError

        vm = _vm_for("random_routing", phases=("request",))
        await middleware.on_virtual_model_upserted(vm)
        ctx = _make_ctx("ws", vm.name, _make_openai_request(model="ws/m1"))
        # Non-streaming ChatCompletion now triggers factory lookup — no TypedResponseStream needed.
        response = InferenceResponse(result={"id": "x"}, headers={}, typed_body=_make_chat_completion())

        with pytest.raises(InferenceMiddlewareError) as exc:
            await middleware.process_response(ctx, response, {"config_type": "random_routing"})
        assert exc.value.status_code == 400
        assert "response_middleware" in str(exc.value).lower() or "response side" in str(exc.value).lower()

    @pytest.mark.asyncio
    async def test_request_phase_rejected_when_only_in_response_middleware(
        self, middleware: SwitchyardMiddleware
    ) -> None:
        """Reverse: switchyard only under response_middleware. process_request
        must refuse — running an arbitrary factory's request pipeline when the
        user didn't ask for it would be silently wrong."""
        from nemo_helix_plugin.inference_middleware import InferenceMiddlewareError

        vm = _vm_for("random_routing", phases=("response",))
        await middleware.on_virtual_model_upserted(vm)
        request = _make_openai_request(model="ws/m1")
        ctx = _make_ctx("ws", vm.name, request)

        with pytest.raises(InferenceMiddlewareError) as exc:
            await middleware.process_request(ctx, request, {"config_type": "random_routing"})
        assert exc.value.status_code == 400

    @pytest.mark.asyncio
    async def test_both_phases_run_when_listed_under_both_lists(self, middleware: SwitchyardMiddleware) -> None:
        """Listing switchyard under both lists registers both phases pointing at
        the same factory hash; both process_request and process_response succeed."""
        vm = _vm_for("random_routing", phases=("request", "response"))
        await middleware.on_virtual_model_upserted(vm)

        vm_key = f"{vm.workspace}/{vm.name}"
        from nemo_switchyard import _state

        req_hash = _state.VM_NAME_TO_CONFIG_HASH.get((vm_key, "random_routing", "request"))
        resp_hash = _state.VM_NAME_TO_CONFIG_HASH.get((vm_key, "random_routing", "response"))
        assert req_hash is not None and resp_hash is not None
        assert req_hash == resp_hash, "identical configs should share the same factory hash"

        # Both phases work end-to-end.
        request = _make_openai_request(model="ws/m1")
        ctx = _make_ctx("ws", vm.name, request)
        await middleware.process_request(ctx, request, {"config_type": "random_routing"})

        # Use a real ChatCompletion so _wrap_non_streaming succeeds.
        typed = _make_chat_completion()
        response = InferenceResponse(result={"id": "x"}, headers={}, typed_body=typed)
        out = await middleware.process_response(ctx, response, {"config_type": "random_routing"})
        # Empty pipeline: write_back_response preserves typed_body identity.
        assert out.typed_body is typed

    @pytest.mark.asyncio
    async def test_translate_allowed_in_response_middleware(self, middleware: SwitchyardMiddleware) -> None:
        """Translate on the response side is now fully wired up.
        VM-upsert must succeed and process_response must run the pipeline."""
        vm = _vm_for("translate", phases=("response",))
        # Should not raise — the guard has been removed.
        await middleware.on_virtual_model_upserted(vm)

        vm_key = f"{vm.workspace}/{vm.name}"
        from nemo_switchyard import _state

        assert (vm_key, "translate", "response") in _state.VM_NAME_TO_CONFIG_HASH

    @pytest.mark.asyncio
    async def test_translate_in_request_middleware_still_works(self, middleware: SwitchyardMiddleware) -> None:
        """Sanity check: the request-side path is not affected by the rejection."""
        vm = _vm_for("translate", phases=("request",))
        await middleware.on_virtual_model_upserted(vm)
        vm_key = f"{vm.workspace}/{vm.name}"
        from nemo_switchyard import _state

        assert (vm_key, "translate", "request") in _state.VM_NAME_TO_CONFIG_HASH

    @pytest.mark.asyncio
    async def test_destroy_cleans_up_both_phase_entries(self, middleware: SwitchyardMiddleware) -> None:
        """on_virtual_model_destroyed must remove all (vm_key, *, *) entries,
        regardless of phase — otherwise a re-upsert with a different config
        could see stale phase mappings."""
        vm = _vm_for("random_routing", phases=("request", "response"))
        await middleware.on_virtual_model_upserted(vm)
        vm_key = f"{vm.workspace}/{vm.name}"
        from nemo_switchyard import _state

        assert any(k[0] == vm_key for k in _state.VM_NAME_TO_CONFIG_HASH)

        await middleware.on_virtual_model_destroyed(vm)

        assert not any(k[0] == vm_key for k in _state.VM_NAME_TO_CONFIG_HASH)
