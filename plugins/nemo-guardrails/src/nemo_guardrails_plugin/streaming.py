# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from nemo_guardrails_plugin.responses import build_chat_completion_response_id
from pydantic import BaseModel, ConfigDict


class _OpenAIStreamingChunkBase(BaseModel):
    """Base for local OpenAI-shaped streaming models.

    These mirror OpenAI-shaped streaming types locally; replace with shared
    source-owned types when available.
    """

    model_config = ConfigDict(extra="allow")

    def to_dict(self, *, exclude_none: bool = False) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=exclude_none, exclude_unset=True)


class DeltaMessage(_OpenAIStreamingChunkBase):
    """Delta for one streaming choice (OpenAI ``chat.completion.chunk``)."""

    content: str | None = None
    role: str | None = None


class ChatCompletionResponseStreamChoice(_OpenAIStreamingChunkBase):
    """One choice in a streaming chat completion chunk."""

    index: int
    delta: DeltaMessage
    finish_reason: str | None = None


class GuardrailChatCompletionStreamResponse(_OpenAIStreamingChunkBase):
    """Top-level streaming chat completion chunk (OpenAI-compatible)."""

    choices: list[ChatCompletionResponseStreamChoice]
    model: str
    id: str | None = None
    created: int | None = None
    object: str | None = None
    system_fingerprint: str | None = None


@dataclass
class ChatCompletionChunkMetadata:
    """Stable OpenAI chunk metadata preserved while filtering tokens.

    ``LLMRails.stream_async`` accepts and yields plain token strings, so the
    OpenAI chat chunk metadata from IGW is lost while output rails run. This class
    captures the first upstream chunk's stable fields, so filtered tokens can be wrapped
    back into valid ``chat.completion.chunk`` responses. If output rails emit a
    token before consuming upstream chunks, synthesize the same stable fields
    once and reuse them for the rest of the stream.
    """

    response: GuardrailChatCompletionStreamResponse | None = None
    finish_reason: str | None = None

    def update_from_chunk(self, chunk: dict[str, Any]) -> None:
        """Capture stable fields from an upstream OpenAI chat completion chunk."""
        self._update_finish_reason(chunk)

        if self.response is None:
            self.response = GuardrailChatCompletionStreamResponse.model_validate(chunk)
            return

        if self.response.system_fingerprint is None and isinstance(chunk.get("system_fingerprint"), str):
            self.response.system_fingerprint = chunk["system_fingerprint"]

    def _update_finish_reason(self, chunk: dict[str, Any]) -> None:
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            return

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            return

        finish_reason = first_choice.get("finish_reason")
        if isinstance(finish_reason, str):
            self.finish_reason = finish_reason

    def _base_response(self, model: str) -> GuardrailChatCompletionStreamResponse:
        """Return the response metadata, synthesizing missing stable fields as needed."""
        if self.response is None:
            self.response = GuardrailChatCompletionStreamResponse(
                id=build_chat_completion_response_id(),
                object="chat.completion.chunk",
                created=int(time.time()),
                model=model,
                choices=[],
            )

        if self.response.id is None:
            self.response.id = build_chat_completion_response_id()
        if self.response.object is None:
            self.response.object = "chat.completion.chunk"
        if self.response.created is None:
            self.response.created = int(time.time())

        return self.response

    def build_chunk(
        self,
        *,
        choices: list[ChatCompletionResponseStreamChoice],
        model: str,
    ) -> dict[str, Any]:
        """Build an OpenAI-compatible ``chat.completion.chunk`` response dict.

        This function is called for each token in the stream returned by ``LLMRails.stream_async``.
        It decorates each chunk with top-level metadata fields from the upstream IGW chunk.

        IGW serializes the returned dict as the next server-sent event in the
        chat-completions stream.

        For example, this input:

        ``choices=[
            ChatCompletionResponseStreamChoice(
                index=0,
                delta=DeltaMessage(content="hello", role="assistant"),
            )
        ]``

        becomes a response-shaped dict like:

        ``{
            "id": "chatcmpl-...",
            "object": "chat.completion.chunk",
            "created": 123,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "hello", "role": "assistant"},
                }
            ],
        }``
        """
        chunk_response = self._base_response(model).model_copy(update={"choices": choices})
        return chunk_response.to_dict(exclude_none=True)


def extract_delta_content(chunk: dict[str, Any]) -> str:
    """Return the first choice's delta content from an OpenAI chat chunk."""
    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return ""

    delta = first_choice.get("delta")
    if not isinstance(delta, dict):
        return ""

    content = delta.get("content")
    return content if isinstance(content, str) else ""


async def chunks_to_strings(
    chunks: AsyncIterator[dict[str, Any]],
    metadata: ChatCompletionChunkMetadata,
) -> AsyncIterator[str]:
    """Transform IGW streaming chunks into the token stream expected by nemoguardrails.

    Our response middleware receives an async stream of OpenAI-compatible chat completion
    chunks. ``LLMRails.stream_async`` expects an async stream of plain token strings, so
    this function transforms each chunk into a token string, while preserving other chunk
    metadata neeed to rebuild the chat completion chunk after output rails run.
    """
    async for chunk in chunks:
        metadata.update_from_chunk(chunk)
        token = extract_delta_content(chunk)
        if token:
            yield token


async def strings_to_chunks(
    tokens: AsyncIterator[str],
    *,
    metadata: ChatCompletionChunkMetadata,
    model: str,
) -> AsyncIterator[dict[str, Any]]:
    """Transform ``LLMRails.stream_async`` tokens into OpenAI-compatible chat completion chunks.

    ``LLMRails.stream_async`` yields plain token strings after output rails run.
    IGW expects middleware to return OpenAI-compatible chat completion chunks as
    dictionaries, so this wraps each filtered token with the preserved token
    metadata, and appends a terminal chunk using the upstream ``finish_reason``.
    """
    is_first_content_chunk = True
    async for token in tokens:
        if error_data := parse_streaming_error_token(token):
            yield error_data
            return

        yield build_chat_completion_chunk(
            token,
            metadata=metadata,
            model=model,
            is_first_content_chunk=is_first_content_chunk,
        )
        is_first_content_chunk = False

    yield build_final_chat_completion_chunk(metadata=metadata, model=model)


def parse_streaming_error_token(token: str) -> dict[str, Any] | None:
    """Return a nemoguardrails streaming error token as a response dict.

    ``LLMRails.stream_async`` emits guardrail violations as a plain JSON string
    with a top-level ``error`` object. IGW expects middleware streaming results
    to be dicts, so pass those error payloads through instead of wrapping them as
    an assistant message delta.
    """
    try:
        data = json.loads(token)
    except json.JSONDecodeError:
        return None

    if isinstance(data, dict) and isinstance(data.get("error"), dict):
        return data

    return None


def build_streaming_error_response(exc: Exception) -> dict[str, Any]:
    """Build the streaming error payload used when iteration fails mid-stream."""
    return {
        "error": {
            "message": str(exc),
            "type": type(exc).__name__,
            "param": "",
            "code": "500",
        }
    }


def build_chat_completion_chunk(
    token: str,
    *,
    metadata: ChatCompletionChunkMetadata,
    model: str,
    is_first_content_chunk: bool,
) -> dict[str, Any]:
    """Build a single chat completion chunk for the given string token."""
    delta = DeltaMessage(content=token, role="assistant" if is_first_content_chunk else None)
    return metadata.build_chunk(
        choices=[ChatCompletionResponseStreamChoice(index=0, delta=delta, finish_reason=None)],
        model=model,
    )


def build_final_chat_completion_chunk(
    *,
    metadata: ChatCompletionChunkMetadata,
    model: str,
) -> dict[str, Any]:
    """Build the terminal chat completion chunk using the upstream finish reason."""
    return metadata.build_chunk(
        choices=[
            ChatCompletionResponseStreamChoice(
                index=0,
                delta=DeltaMessage(),
                finish_reason=metadata.finish_reason or "stop",
            )
        ],
        model=model,
    )


async def close_async_iterator(iterator: AsyncIterator[Any]) -> None:
    """Close an async iterator if it exposes an `aclose` method."""
    close = getattr(iterator, "aclose", None)
    if close is not None:
        await close()
