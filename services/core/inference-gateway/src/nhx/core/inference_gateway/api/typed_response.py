# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed backend response parsing for inference middleware."""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Literal, cast

import anthropic.types as anthropic_types
import openai.types.chat as openai_chat_types
from nemo_helix_plugin.inference_middleware import (
    BackendFormat,
)
from nemo_helix_plugin.inference_middleware_types import (
    TypedResponse,
    TypedResponseChunk,
)
from pydantic import BaseModel, TypeAdapter, ValidationError

logger = logging.getLogger(__name__)

_RESPONSE_ADAPTERS: dict[BackendFormat, TypeAdapter] = {
    BackendFormat.OPENAI_CHAT: TypeAdapter(openai_chat_types.ChatCompletion),
    BackendFormat.ANTHROPIC_MESSAGES: TypeAdapter(anthropic_types.Message),
}


class AnthropicPingEvent(BaseModel):
    """Anthropic streaming heartbeat event.

    Not part of the Anthropic SDK's ``RawMessageStreamEvent`` Union, but routinely
    emitted by upstream hubs (e.g. NVIDIA Inference API) between content blocks.
    Typed here so middleware that wants to inspect or count heartbeats can do so;
    most middleware just ignores them.
    """

    type: Literal["ping"]


# Wider Anthropic stream Union including the ``ping`` heartbeat the SDK omits.
_AnthropicStreamEvent = anthropic_types.RawMessageStreamEvent | AnthropicPingEvent


_STREAM_ADAPTERS: dict[BackendFormat, TypeAdapter] = {
    BackendFormat.OPENAI_CHAT: TypeAdapter(openai_chat_types.ChatCompletionChunk),
    BackendFormat.ANTHROPIC_MESSAGES: TypeAdapter(_AnthropicStreamEvent),
}


def parse_typed_response(backend_format: BackendFormat, result: dict[str, Any]) -> TypedResponse | None:
    """Parse a non-streaming backend response into the matching typed model.

    Parsing is intentionally non-fatal: middleware can fall back to the raw
    ``result`` when a backend returns a shape outside the modeled subset.
    """
    try:
        return cast(TypedResponse, _RESPONSE_ADAPTERS[backend_format].validate_python(result))
    except ValidationError:
        logger.debug(
            "Failed to parse %s backend response into a typed middleware response",
            backend_format.value,
            exc_info=True,
        )
        return None


class TypedResponseStream:
    """Replayable typed view over a raw streaming backend response.

    Middleware iteration yields only typed chunks. The raw stream is cached as
    it is consumed so response serialization can still replay raw chunks after
    middleware has inspected the typed view.

    Per-chunk validation is intentionally non-fatal:

    1. **Known events outside the SDK Union** (e.g. Anthropic ``ping``
       heartbeats) are encoded explicitly in the local widened Union — they
       flow through as typed instances (``AnthropicPingEvent``) that
       middleware can inspect or ignore.
    2. **Unknown chunk shapes** (future SDK additions, vendor extras,
       malformed data) are logged at DEBUG and skipped from the typed view.

    The raw view always delivers every chunk so wire-level serialization is
    unaffected; only the typed view filters down to validated chunks.
    """

    def __init__(
        self,
        backend_format: BackendFormat,
        result: AsyncIterator[dict[str, Any]],
    ) -> None:
        self._backend_format = backend_format
        self._adapter = _STREAM_ADAPTERS[backend_format]
        self._source = result.__aiter__()
        self._raw_cache: list[dict[str, Any]] = []
        self._source_exhausted = False
        self._typed_iterator: AsyncIterator[TypedResponseChunk] | None = None

    def __aiter__(self) -> TypedResponseStream:
        self._typed_iterator = self.typed_chunks()
        return self

    async def __anext__(self) -> TypedResponseChunk:
        if self._typed_iterator is None:
            self._typed_iterator = self.typed_chunks()
        return await self._typed_iterator.__anext__()

    async def raw_chunks(self) -> AsyncIterator[dict[str, Any]]:
        """Yield raw chunks, replaying chunks already consumed by typed iteration."""
        index = 0
        while True:
            if index < len(self._raw_cache):
                yield self._raw_cache[index]
                index += 1
                continue
            if self._source_exhausted:
                return
            try:
                chunk = await self._source.__anext__()
            except StopAsyncIteration:
                self._source_exhausted = True
                return
            self._raw_cache.append(chunk)
            index += 1
            yield chunk

    async def typed_chunks(self) -> AsyncIterator[TypedResponseChunk]:
        """Yield validated typed chunks only.

        Skips any chunk the SDK's typed Union doesn't recognise (logged at
        DEBUG). This keeps the iterator alive across SDK-coverage gaps such
        as Anthropic ``ping`` events while still preventing raw dicts from
        ever being yielded as a typed chunk.
        """
        async for chunk in self.raw_chunks():
            try:
                yield cast(TypedResponseChunk, self._adapter.validate_python(chunk))
            except ValidationError:
                logger.debug(
                    "Skipping unrecognised %s stream chunk (type=%r) — not in the SDK typed Union",
                    self._backend_format.value,
                    chunk.get("type") if isinstance(chunk, dict) else None,
                    exc_info=True,
                )
                continue


def parse_typed_stream(
    backend_format: BackendFormat,
    result: AsyncIterator[dict[str, Any]],
) -> AsyncIterator[TypedResponseChunk]:
    """Lazily parse streaming backend chunks into typed chunk models.

    Individual chunk parse failures are logged at DEBUG and **skipped** —
    the typed iterator continues with the next chunk. Raw dict payloads are
    never exposed as typed chunks; consumers needing the raw view should
    use ``TypedResponseStream.raw_chunks()`` directly.
    """
    return cast(AsyncIterator[TypedResponseChunk], TypedResponseStream(backend_format, result))
