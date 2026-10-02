# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for typed backend response parsing."""

from __future__ import annotations

from typing import Any, AsyncIterator

import anthropic.types as anthropic_types
import openai.types.chat as openai_chat_types
import pytest
from nemo_helix_plugin.inference_middleware import (
    BackendFormat,
)
from nhx.core.inference_gateway.api.typed_response import (
    AnthropicPingEvent,
    parse_typed_response,
    parse_typed_stream,
)


def _openai_response(**extra: Any) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "llama",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
        **extra,
    }


def _anthropic_message(**extra: Any) -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "hello"}],
        "model": "claude",
        "usage": {"input_tokens": 1, "output_tokens": 1},
        **extra,
    }


async def _collect(stream: AsyncIterator[Any]) -> list[Any]:
    return [chunk async for chunk in stream]


def test_parse_openai_response_preserves_extra_fields():
    parsed = parse_typed_response(BackendFormat.OPENAI_CHAT, _openai_response(vendor_passthrough="fp_1"))

    assert isinstance(parsed, openai_chat_types.ChatCompletion)
    assert parsed.choices[0].message is not None
    assert parsed.choices[0].message.content == "hello"
    assert parsed.model_extra is not None
    assert parsed.model_extra["vendor_passthrough"] == "fp_1"


def test_parse_openai_response_dumps_extra_fields():
    parsed = parse_typed_response(BackendFormat.OPENAI_CHAT, _openai_response(vendor_passthrough="fp_1"))

    assert isinstance(parsed, openai_chat_types.ChatCompletion)
    assert parsed.model_dump(mode="json")["vendor_passthrough"] == "fp_1"


def test_parse_anthropic_response_allows_missing_optional_fields_and_preserves_extra_fields():
    parsed = parse_typed_response(BackendFormat.ANTHROPIC_MESSAGES, _anthropic_message(vendor_passthrough="fp_1"))

    assert isinstance(parsed, anthropic_types.Message)
    assert parsed.stop_reason is None
    assert parsed.stop_sequence is None
    assert parsed.model_extra is not None
    assert parsed.model_extra["vendor_passthrough"] == "fp_1"


def test_parse_anthropic_response_dumps_extra_fields():
    parsed = parse_typed_response(BackendFormat.ANTHROPIC_MESSAGES, _anthropic_message(vendor_passthrough="fp_1"))

    assert isinstance(parsed, anthropic_types.Message)
    assert parsed.model_dump(mode="json")["vendor_passthrough"] == "fp_1"


def test_parse_response_returns_none_for_invalid_shape():
    parsed = parse_typed_response(BackendFormat.OPENAI_CHAT, {"id": "chatcmpl-1"})

    assert parsed is None


@pytest.mark.asyncio
async def test_parse_openai_stream_is_lazy_and_typed():
    consumed = False

    async def raw_stream() -> AsyncIterator[dict[str, Any]]:
        nonlocal consumed
        consumed = True
        yield {
            "id": "chunk-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "llama",
            "choices": [{"index": 0, "delta": {"content": "hel"}, "finish_reason": None}],
        }

    typed_stream = parse_typed_stream(BackendFormat.OPENAI_CHAT, raw_stream())

    assert consumed is False
    chunks = await _collect(typed_stream)
    assert consumed is True
    assert isinstance(chunks[0], openai_chat_types.ChatCompletionChunk)
    assert chunks[0].choices[0].delta.content == "hel"


@pytest.mark.asyncio
async def test_parse_anthropic_stream_uses_discriminated_event_models():
    async def raw_stream() -> AsyncIterator[dict[str, Any]]:
        yield {"type": "message_start", "message": _anthropic_message()}
        yield {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}}

    chunks = await _collect(parse_typed_stream(BackendFormat.ANTHROPIC_MESSAGES, raw_stream()))

    assert isinstance(chunks[0], anthropic_types.RawMessageStartEvent)
    assert isinstance(chunks[1], anthropic_types.RawContentBlockDeltaEvent)


@pytest.mark.asyncio
async def test_parse_stream_skips_unrecognised_chunks_without_yielding_raw_dicts():
    """Chunks the SDK Union can't validate (e.g. malformed shapes, Anthropic ping
    heartbeats) are skipped — the typed iterator continues with the next chunk
    instead of tearing down. Raw dicts are never yielded as typed chunks."""

    async def raw_stream() -> AsyncIterator[dict[str, Any]]:
        # Valid OpenAI chunk
        yield {
            "id": "chunk-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "llama",
            "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}],
        }
        # Malformed chunk — should be skipped
        yield {"id": "not-enough-fields"}
        # Another valid chunk after the bad one — proves the iterator survived
        yield {
            "id": "chunk-2",
            "object": "chat.completion.chunk",
            "created": 2,
            "model": "llama",
            "choices": [{"index": 0, "delta": {"content": " there"}, "finish_reason": "stop"}],
        }

    chunks = await _collect(parse_typed_stream(BackendFormat.OPENAI_CHAT, raw_stream()))

    assert len(chunks) == 2
    assert all(isinstance(c, openai_chat_types.ChatCompletionChunk) for c in chunks)
    assert chunks[0].id == "chunk-1"
    assert chunks[1].id == "chunk-2"


@pytest.mark.asyncio
async def test_parse_anthropic_stream_includes_ping_events_as_typed_chunks():
    """Anthropic's streaming protocol sends `{"type": "ping"}` heartbeats that
    aren't part of the SDK's `RawMessageStreamEvent` Union. Our widened Union
    types them explicitly via ``AnthropicPingEvent`` so the
    typed iterator survives the chunk and middleware that wants to inspect
    heartbeats can. Downstream consumers that dispatch on event ``type`` just
    fall through their match chain, so passing ping through is safe."""

    async def raw_stream() -> AsyncIterator[dict[str, Any]]:
        yield {"type": "message_start", "message": _anthropic_message()}
        yield {"type": "ping"}  # heartbeat — typed as AnthropicPingEvent
        yield {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}}
        yield {"type": "ping"}  # another mid-stream ping
        yield {"type": "message_stop"}

    chunks = await _collect(parse_typed_stream(BackendFormat.ANTHROPIC_MESSAGES, raw_stream()))

    # All five events flow through, ping events typed as AnthropicPingEvent.
    assert len(chunks) == 5
    assert isinstance(chunks[0], anthropic_types.RawMessageStartEvent)
    assert isinstance(chunks[1], AnthropicPingEvent)
    assert chunks[1].type == "ping"
    assert isinstance(chunks[2], anthropic_types.RawContentBlockDeltaEvent)
    assert isinstance(chunks[3], AnthropicPingEvent)
    assert isinstance(chunks[4], anthropic_types.RawMessageStopEvent)


@pytest.mark.asyncio
async def test_parse_stream_raw_chunks_unaffected_by_typed_skips():
    """``raw_chunks()`` always delivers every chunk in order — wire-level
    serialization must not be filtered by the typed Union's coverage."""
    from nhx.core.inference_gateway.api.typed_response import TypedResponseStream

    async def raw_stream() -> AsyncIterator[dict[str, Any]]:
        yield {"type": "message_start", "message": _anthropic_message()}
        yield {"type": "ping"}
        yield {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}}

    stream = TypedResponseStream(BackendFormat.ANTHROPIC_MESSAGES, raw_stream())
    raw_chunks = [chunk async for chunk in stream.raw_chunks()]

    assert len(raw_chunks) == 3
    assert raw_chunks[1] == {"type": "ping"}
