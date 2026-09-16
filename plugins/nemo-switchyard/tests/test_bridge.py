# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for _bridge: _wrap_streaming, _wrap_non_streaming, write_back_response, write_back_request."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import anthropic.types as anthropic_types
import openai.types.chat as openai_chat_types
import pytest
from nemo_helix_plugin.inference_middleware import BackendFormat, InferenceRequest, InferenceResponse
from nemo_switchyard._bridge import (
    _wrap_non_streaming,
    _wrap_streaming,
    write_back_request,
    write_back_response,
)
from switchyard.lib.chat_response.anthropic import AnthropicChatResponse, AnthropicStreamingChatResponse
from switchyard.lib.chat_response.openai_chat import CompletionChatResponse, StreamingChatResponse
from switchyard.lib.chat_response.openai_responses import ResponsesApiStreamingChatResponse


def _make_chat_completion(model: str = "gpt-4") -> openai_chat_types.ChatCompletion:
    return openai_chat_types.ChatCompletion.model_validate(
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


def _make_anthropic_message(model: str = "claude-3") -> anthropic_types.Message:
    return anthropic_types.Message.model_validate(
        {
            "id": "msg_test",
            "content": [{"type": "text", "text": "hi"}],
            "model": model,
            "role": "assistant",
            "stop_reason": "end_turn",
            "type": "message",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )


def _make_typed_stream(backend_format: BackendFormat):
    from nhx.core.inference_gateway.api.typed_response import TypedResponseStream

    async def _gen():
        return
        yield  # noqa: unreachable

    return TypedResponseStream(backend_format, _gen())


# ---------------------------------------------------------------------------
# _wrap_streaming
# ---------------------------------------------------------------------------


def test_wrap_streaming_openai():
    result = _wrap_streaming(_make_typed_stream(BackendFormat.OPENAI_CHAT), BackendFormat.OPENAI_CHAT)
    assert isinstance(result, StreamingChatResponse)


def test_wrap_streaming_anthropic():
    result = _wrap_streaming(_make_typed_stream(BackendFormat.ANTHROPIC_MESSAGES), BackendFormat.ANTHROPIC_MESSAGES)
    assert isinstance(result, AnthropicStreamingChatResponse)


def test_wrap_streaming_unknown_format_returns_none():
    stream = _make_typed_stream(BackendFormat.OPENAI_CHAT)
    unknown = MagicMock(spec=BackendFormat)
    unknown.__eq__ = lambda self, other: False
    assert _wrap_streaming(stream, unknown) is None  # type: ignore[arg-type]


def test_wrap_streaming_dispatches_on_explicit_format_not_stream():
    """backend_format comes from the caller (ctx), not from the stream."""
    stream = _make_typed_stream(BackendFormat.OPENAI_CHAT)
    result = _wrap_streaming(stream, BackendFormat.ANTHROPIC_MESSAGES)
    assert isinstance(result, AnthropicStreamingChatResponse)


# ---------------------------------------------------------------------------
# _wrap_non_streaming
# ---------------------------------------------------------------------------


def test_wrap_non_streaming_chat_completion():
    typed = _make_chat_completion()
    result = _wrap_non_streaming(typed)
    assert isinstance(result, CompletionChatResponse)
    assert result.body is typed


def test_wrap_non_streaming_anthropic_message():
    typed = _make_anthropic_message()
    result = _wrap_non_streaming(typed)
    assert isinstance(result, AnthropicChatResponse)
    assert result.body is typed


def test_wrap_non_streaming_unexpected_type_raises_500():
    from nemo_helix_plugin.inference_middleware import InferenceMiddlewareError

    with pytest.raises(InferenceMiddlewareError) as exc:
        _wrap_non_streaming({"bad": "dict"})  # type: ignore[arg-type]
    assert exc.value.status_code == 500


# ---------------------------------------------------------------------------
# write_back_response — non-streaming
# ---------------------------------------------------------------------------


def test_write_back_response_completion_updates_typed_body_and_result():
    typed = _make_chat_completion()
    response = InferenceResponse(result={"id": "raw"}, headers={})
    write_back_response(response, CompletionChatResponse(typed))
    assert response.typed_body is typed
    # result must also be updated so downstream plugins reading result directly
    # (e.g. guardrails) see the translated payload, not the pre-translation dict.
    assert isinstance(response.result, dict)
    assert response.result["id"] == typed.id
    assert "choices" in response.result


def test_write_back_response_anthropic_message_updates_typed_body_and_result():
    typed = _make_anthropic_message()
    response = InferenceResponse(result={"id": "raw"}, headers={})
    write_back_response(response, AnthropicChatResponse(typed))
    assert response.typed_body is typed
    assert isinstance(response.result, dict)
    assert response.result["id"] == typed.id
    assert "content" in response.result


# ---------------------------------------------------------------------------
# write_back_response — streaming
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_back_response_streaming_sets_result():
    from switchyard.lib.chat_response.openai_chat import ResponseStream

    async def _source():
        yield openai_chat_types.ChatCompletionChunk.model_validate(
            {
                "id": "c1",
                "choices": [{"delta": {"content": "hi"}, "index": 0, "finish_reason": None}],
                "created": 0,
                "model": "gpt-4",
                "object": "chat.completion.chunk",
            }
        )

    rs = ResponseStream(_source())
    response = InferenceResponse(result={"id": "raw"}, headers={}, typed_body=_make_chat_completion())
    write_back_response(response, StreamingChatResponse(rs))
    assert response.result is rs
    assert response.typed_body is None


@pytest.mark.asyncio
async def test_write_back_response_anthropic_streaming():
    from switchyard.lib.chat_response.anthropic import AnthropicResponseStream

    async def _src():
        return
        yield  # noqa: unreachable

    ars = AnthropicResponseStream(_src())
    response = InferenceResponse(result={}, headers={})
    write_back_response(response, AnthropicStreamingChatResponse(ars))
    assert response.result is ars
    assert response.typed_body is None


def test_write_back_response_responses_api_streaming():
    from switchyard.lib.chat_response.openai_responses import ResponsesApiStream

    async def _src():
        return
        yield  # noqa: unreachable

    rs = ResponsesApiStream(_src())
    response = InferenceResponse(result={}, headers={})
    write_back_response(response, ResponsesApiStreamingChatResponse(rs))
    assert response.result is rs
    assert response.typed_body is None


def test_write_back_response_streaming_clears_typed_body_sets_result_to_stream():
    """After a streaming write_back_response typed_body is None and result is an iterator.
    A second process_response call should detect this and pass through rather than 500."""
    from switchyard.lib.chat_response.openai_chat import ResponseStream

    async def _src():
        return
        yield  # noqa: unreachable

    rs = ResponseStream(_src())
    response = InferenceResponse(result={"id": "raw"}, headers={}, typed_body=_make_chat_completion())
    from switchyard.lib.chat_response.openai_chat import StreamingChatResponse

    write_back_response(response, StreamingChatResponse(rs))

    # typed_body cleared, result is the stream — hasattr(__aiter__) is True
    assert response.typed_body is None
    assert hasattr(response.result, "__aiter__")


def test_write_back_response_unexpected_type_raises_500():
    from nemo_helix_plugin.inference_middleware import InferenceMiddlewareError

    class _Weird:
        pass

    with pytest.raises(InferenceMiddlewareError) as exc:
        write_back_response(InferenceResponse(result={}, headers={}), _Weird())  # type: ignore[arg-type]
    assert exc.value.status_code == 500


# ---------------------------------------------------------------------------
# write_back_request
# ---------------------------------------------------------------------------


def test_write_back_request_sets_body_and_typed_body():
    from switchyard.lib.chat_request.openai_chat import OpenAIChatRequest
    from switchyard.lib.proxy_context import ProxyContext

    new_body: dict[str, Any] = {"model": "ws/strong", "messages": []}
    request = InferenceRequest(
        body={"model": "ws/orig", "messages": []},
        headers={},
        path="v1/chat/completions",
        typed_body={"model": "ws/orig", "messages": []},
    )  # type: ignore[arg-type]
    write_back_request(request, OpenAIChatRequest(new_body), ProxyContext(metadata={}))
    assert request.body is new_body
    assert request.typed_body is request.body


def test_write_back_request_applies_path_update():
    from nemo_switchyard._processors import CTX_PATH_UPDATE
    from switchyard.lib.chat_request.anthropic import AnthropicChatRequest
    from switchyard.lib.proxy_context import ProxyContext

    new_body: dict[str, Any] = {"model": "ws/c", "messages": [], "max_tokens": 100}
    request = InferenceRequest(
        body={"model": "ws/orig", "messages": []},
        headers={},
        path="v1/chat/completions",
        typed_body={"model": "ws/orig", "messages": []},
    )  # type: ignore[arg-type]
    write_back_request(request, AnthropicChatRequest(new_body), ProxyContext(metadata={CTX_PATH_UPDATE: "v1/messages"}))
    assert request.path == "v1/messages"
    assert request.body is new_body
    assert request.typed_body is request.body


# ---------------------------------------------------------------------------
# Round-trip: _wrap → write_back_response
# ---------------------------------------------------------------------------


def test_roundtrip_completion_identity():
    typed = _make_chat_completion()
    response = InferenceResponse(result={"id": "raw"}, headers={}, typed_body=typed)
    write_back_response(response, _wrap_non_streaming(typed))
    assert response.typed_body is typed  # empty pipeline: identity


def test_roundtrip_anthropic_message_identity():
    typed = _make_anthropic_message()
    response = InferenceResponse(result={"id": "raw"}, headers={}, typed_body=typed)
    write_back_response(response, _wrap_non_streaming(typed))
    assert response.typed_body is typed
