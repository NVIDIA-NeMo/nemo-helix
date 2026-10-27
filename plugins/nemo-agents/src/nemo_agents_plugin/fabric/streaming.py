# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transport selection and OpenAI-compatible SSE framing for Fabric responses.

Server-Sent Events (SSE) stream newline-delimited HTTP frames. OpenAI Chat
Completions streams encode each JSON chunk as ``data: <json>`` and terminate
with ``data: [DONE]``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterable, AsyncIterator, Mapping
from enum import StrEnum
from typing import Any

from nemo_agents_plugin.fabric.runtime import (
    FabricRuntimeResult,
    FabricRuntimeStream,
    fabric_streaming_options,
)
from nemo_agents_plugin.fabric.serving_models import (
    ChatCompletionStreamChoice,
    ChatCompletionStreamDelta,
    ChatCompletionStreamError,
    ChatCompletionStreamErrorResponse,
    ChatCompletionStreamResponse,
)
from nemo_fabric import FabricConfig, RunPlan

# Fabric streams raw ATOF records. ATOF's envelope tells us which records
# represent completed LLM work; the payload remains producer-defined, so the
# text extraction below only handles known assistant-output shapes.
_ASSISTANT_ROLES = frozenset({"assistant", "ai"})
_ATOF_EVENT_KINDS = frozenset({"scope", "mark"})
_TEXT_EVENT_TYPES = frozenset(
    {
        "agentMessage",
        "agent_message",
        "assistant_message",
        "content_delta",
        "message_delta",
        "output_text",
        "output_text_delta",
        "text_delta",
    }
)


class StreamingMethod(StrEnum):
    """How serving obtains output for an OpenAI streaming response."""

    RELAY = "relay"
    OPENAI = "openai"
    BUFFERED = "buffered"


def resolve_streaming_method(plan: RunPlan, config: FabricConfig) -> StreamingMethod:
    """Prefer Relay, then native streaming, otherwise buffer the completion.

    Inspect the plan before adding serving-owned Relay configuration. Selection
    checks declared support and configuration without modifying either input.
    Telemetry export settings do not control serving-owned Relay streaming.
    Invalid Relay streaming configuration raises rather than falling back.
    """
    resolved = plan.get("adapter_descriptor")
    if not isinstance(resolved, Mapping):
        return StreamingMethod.BUFFERED
    descriptor = resolved.get("descriptor", resolved)
    if not isinstance(descriptor, Mapping):
        return StreamingMethod.BUFFERED

    telemetry = descriptor.get("telemetry")
    providers = telemetry.get("providers") if isinstance(telemetry, Mapping) else None
    if isinstance(providers, Mapping) and isinstance(providers.get("relay"), Mapping):
        fabric_streaming_options(config)
        return StreamingMethod.RELAY

    capabilities = descriptor.get("capabilities")
    if plan.capabilities.streaming and isinstance(capabilities, Mapping) and capabilities.get("streaming") is True:
        return StreamingMethod.OPENAI
    return StreamingMethod.BUFFERED


class FabricStreamResultError(RuntimeError):
    """Raised when the terminal Fabric streaming result is not successful."""

    def __init__(self, result: FabricRuntimeResult) -> None:
        self.result = result
        super().__init__(_failed_result_detail(result))


def extract_assistant_text_delta(record: Mapping[str, Any]) -> str | None:
    """Extract user-visible assistant text from a known ATOF record shape.

    Fabric streams raw Relay ATOF dictionaries rather than a normalized
    text-delta contract. This helper only emits known assistant text shapes and
    skips lifecycle, metadata, tool, and user-message records.
    """
    return _extract_text_delta(record)


async def iter_openai_chat_completion_sse(
    *,
    completion_id: str,
    content_chunks: AsyncIterable[str],
    model: str = "unknown-model",
) -> AsyncIterator[str]:
    """Serialize assistant text deltas as OpenAI Chat Completions SSE frames."""
    yield _to_sse_frame(
        ChatCompletionStreamResponse(
            id=completion_id,
            model=model,
            choices=[
                ChatCompletionStreamChoice(
                    delta=ChatCompletionStreamDelta(role="assistant"),
                )
            ],
        )
    )

    async for content in content_chunks:
        if not content:
            continue
        yield _to_sse_frame(
            ChatCompletionStreamResponse(
                id=completion_id,
                model=model,
                choices=[
                    ChatCompletionStreamChoice(
                        delta=ChatCompletionStreamDelta(content=content),
                    )
                ],
            )
        )

    yield _to_sse_frame(
        ChatCompletionStreamResponse(
            id=completion_id,
            model=model,
            choices=[
                ChatCompletionStreamChoice(
                    delta=ChatCompletionStreamDelta(),
                    finish_reason="stop",
                )
            ],
        )
    )
    yield "data: [DONE]\n\n"


def openai_chat_completion_error_sse(error: BaseException) -> str:
    """Serialize a mid-stream error as an OpenAI-compatible SSE frame."""
    return _to_sse_frame(
        ChatCompletionStreamErrorResponse(
            error=ChatCompletionStreamError(
                message=str(error) or "Streaming response failed.",
                type=type(error).__name__,
            )
        )
    )


async def iter_fabric_assistant_text_deltas(stream: FabricRuntimeStream) -> AsyncIterator[str]:
    """Yield assistant text deltas and validate the terminal Fabric result."""
    emitted_text = False
    async for record in stream.records():
        text = extract_assistant_text_delta(record)
        if text is not None:
            emitted_text = True
            yield text

    result = await stream.result()
    if result.status != "succeeded":
        raise FabricStreamResultError(result)
    if not emitted_text and isinstance(result.response, str) and result.response:
        yield result.response


def _to_sse_frame(chunk: ChatCompletionStreamResponse | ChatCompletionStreamErrorResponse) -> str:
    data = chunk.model_dump(mode="json", exclude_none=True)
    return f"data: {json.dumps(data, separators=(',', ':'))}\n\n"


def _extract_text_delta(value: Any) -> str | None:
    """Walk one ATOF value looking for assistant text in supported shapes."""
    if not isinstance(value, Mapping):
        return None

    if _is_atof_record(value):
        if not _is_llm_output_scope(value):
            return None
        return _extract_payload_text_delta(value.get("data"))

    return _extract_payload_text_delta(value)


def _extract_payload_text_delta(value: Any) -> str | None:
    """Walk producer payloads looking for known assistant text shapes."""
    if not isinstance(value, Mapping):
        return None

    choice_text = _extract_openai_choice_delta(value)
    if choice_text is not None:
        return choice_text

    role = value.get("role")
    if role in _ASSISTANT_ROLES:
        return _text_from_content(value.get("content"))

    event_type = value.get("type") or value.get("name") or value.get("event")
    if event_type in _TEXT_EVENT_TYPES and _role_allows_assistant_text(value):
        for key in ("text", "content", "delta", "payload"):
            text = _text_from_content(value.get(key))
            if text is not None:
                return text

    messages = value.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            text = _extract_text_delta(message)
            if text is not None:
                return text

    for key in ("message", "agent", "data", "payload", "content"):
        text = _extract_text_delta(value.get(key))
        if text is not None:
            return text

    return None


def _is_atof_record(value: Mapping[str, Any]) -> bool:
    """Return true when a dictionary has the ATOF event envelope shape."""
    return value.get("kind") in _ATOF_EVENT_KINDS or "atof_version" in value


def _is_llm_output_scope(value: Mapping[str, Any]) -> bool:
    """Return true for completed LLM scopes that can carry assistant output."""
    if value.get("kind") != "scope":
        return False
    return value.get("scope_category") == "end" and value.get("category") == "llm"


def _extract_openai_choice_delta(value: Mapping[str, Any]) -> str | None:
    """Extract text from OpenAI-style chat completion choices, when present."""
    choices = value.get("choices")
    if not isinstance(choices, list):
        return None
    for choice in choices:
        if not isinstance(choice, Mapping):
            continue
        delta = choice.get("delta")
        if isinstance(delta, Mapping) and _role_allows_assistant_text(delta):
            text = _text_from_content(delta.get("content"))
            if text is not None:
                return text
        message = choice.get("message")
        if isinstance(message, Mapping) and message.get("role") in _ASSISTANT_ROLES:
            text = _text_from_content(message.get("content"))
            if text is not None:
                return text
    return None


def _role_allows_assistant_text(value: Mapping[str, Any]) -> bool:
    """Return true when no role is declared or the role is assistant-like."""
    role = value.get("role")
    return role is None or role in _ASSISTANT_ROLES


def _text_from_content(content: Any) -> str | None:
    """Normalize string content or list-of-text-part content into one delta."""
    if isinstance(content, str):
        return content or None
    if isinstance(content, list):
        parts = [part["text"] for part in content if isinstance(part, Mapping) and isinstance(part.get("text"), str)]
        return "".join(parts) or None
    return None


def _failed_result_detail(result: FabricRuntimeResult) -> str:
    error = result.error
    if isinstance(error, Mapping):
        message = error.get("message")
        if isinstance(message, str):
            return message
    return f"Fabric streaming invocation returned status {result.status!r}."
