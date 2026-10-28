# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI-compatible SSE framing for Fabric-backed streaming responses.

Server-Sent Events (SSE) stream newline-delimited HTTP frames. OpenAI Chat
Completions streams encode each JSON chunk as ``data: <json>`` and terminate
with ``data: [DONE]``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterable, AsyncIterator, Mapping
from typing import TYPE_CHECKING, Any

from nemo_agents_plugin.fabric.serving_models import (
    ChatCompletionStreamChoice,
    ChatCompletionStreamDelta,
    ChatCompletionStreamError,
    ChatCompletionStreamErrorResponse,
    ChatCompletionStreamResponse,
    ChatCompletionStreamToolCall,
    ChatCompletionStreamToolCallFunction,
)

if TYPE_CHECKING:
    from nemo_agents_plugin.fabric.runtime import FabricRuntimeResult, FabricRuntimeStream

# Fabric streams raw ATOF records. ATOF's envelope tells us which records
# represent completed LLM work; the payload remains producer-defined, so the
# extraction below only handles known assistant-output shapes. Relay also
# attaches a provider-neutral ``category_profile.annotated_response`` to LLM
# scopes, which is preferred for tool calls and used as a text fallback.
_ASSISTANT_ROLES = frozenset({"assistant", "ai"})
_ATOF_EVENT_KINDS = frozenset({"scope", "mark"})
# Content blocks carry reasoning under a key matching their type: LangChain uses
# ``{"type": "reasoning", "reasoning": ...}`` and Anthropic ``{"type": "thinking", "thinking": ...}``.
_REASONING_PART_TYPES = frozenset({"reasoning", "thinking"})
_ASSISTANT_MESSAGE_KEYS = ("__nemo_relay_integrations_langchain_model_response", "message", "data", "payload")
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


class FabricStreamResultError(RuntimeError):
    """Raised when the terminal Fabric streaming result is not successful."""

    def __init__(self, result: FabricRuntimeResult) -> None:
        self.result = result
        super().__init__(_failed_result_detail(result))


def extract_assistant_delta(record: Mapping[str, Any]) -> ChatCompletionStreamDelta | None:
    """Extract assistant text, reasoning, and tool calls from a known ATOF record shape.

    Fabric streams raw Relay ATOF dictionaries rather than a normalized delta
    contract. Completed LLM scopes yield text, reasoning, and the tool calls the
    agent issued; other records only yield known assistant text shapes.
    Lifecycle, metadata, tool, and user-message records are skipped.
    """
    if _is_atof_record(record):
        return _extract_llm_output_delta(record) if _is_llm_output_scope(record) else None
    content = _extract_payload_text_delta(record)
    return ChatCompletionStreamDelta(content=content) if content is not None else None


def extract_assistant_text_delta(record: Mapping[str, Any]) -> str | None:
    """Extract user-visible assistant text from a known ATOF record shape."""
    delta = extract_assistant_delta(record)
    return delta.content if delta is not None else None


async def iter_openai_chat_completion_sse(
    *,
    completion_id: str,
    deltas: AsyncIterable[ChatCompletionStreamDelta],
    model: str = "unknown-model",
) -> AsyncIterator[str]:
    """Serialize assistant deltas as OpenAI Chat Completions SSE frames."""
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

    async for delta in deltas:
        yield _to_sse_frame(
            ChatCompletionStreamResponse(
                id=completion_id,
                model=model,
                choices=[ChatCompletionStreamChoice(delta=delta)],
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


async def iter_fabric_assistant_deltas(stream: FabricRuntimeStream) -> AsyncIterator[ChatCompletionStreamDelta]:
    """Yield assistant deltas and validate the terminal Fabric result.

    Tool-call indices are assigned across the whole stream so clients can
    accumulate every call the agent made within one assistant message.
    """
    emitted_text = False
    tool_call_count = 0
    async for record in stream.records():
        delta = extract_assistant_delta(record)
        if delta is None:
            continue
        for tool_call in delta.tool_calls or ():
            tool_call.index = tool_call_count
            tool_call_count += 1
        emitted_text = emitted_text or delta.content is not None
        yield delta

    result = await stream.result()
    if result.status != "succeeded":
        raise FabricStreamResultError(result)
    if not emitted_text and isinstance(result.response, str) and result.response:
        yield ChatCompletionStreamDelta(content=result.response)


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


def _extract_llm_output_delta(record: Mapping[str, Any]) -> ChatCompletionStreamDelta | None:
    """Build one delta from a completed LLM scope's text, reasoning, and tool calls."""
    data = record.get("data")
    profile = record.get("category_profile")
    annotated = profile.get("annotated_response") if isinstance(profile, Mapping) else None
    if not isinstance(annotated, Mapping):
        annotated = {}
    message = _find_assistant_message(data) or {}

    content = _extract_payload_text_delta(data) or _text_from_content(annotated.get("message"))
    reasoning = _reasoning_from_message(message)
    tool_calls = _normalize_tool_calls(annotated.get("tool_calls")) or _tool_calls_from_message(message)
    if content is None and reasoning is None and not tool_calls:
        return None
    return ChatCompletionStreamDelta(content=content, reasoning_content=reasoning, tool_calls=tool_calls or None)


def _find_assistant_message(value: Any) -> Mapping[str, Any] | None:
    """Locate the assistant message in OpenAI, Anthropic, or serialized LangChain payloads."""
    if not isinstance(value, Mapping):
        return None
    if value.get("role") in _ASSISTANT_ROLES or (value.get("type") == "ai" and "content" in value):
        return value

    choices = value.get("choices")
    for choice in choices if isinstance(choices, list) else ():
        if isinstance(choice, Mapping) and (message := _find_assistant_message(choice.get("message"))) is not None:
            return message

    messages = value.get("messages")
    for item in reversed(messages) if isinstance(messages, list) else ():
        if (message := _find_assistant_message(item)) is not None:
            return message

    for key in _ASSISTANT_MESSAGE_KEYS:
        if (message := _find_assistant_message(value.get(key))) is not None:
            return message
    return None


def _reasoning_from_message(message: Mapping[str, Any]) -> str | None:
    """Return reasoning from message fields or reasoning/thinking content blocks."""
    for source in (message, message.get("additional_kwargs")):
        if not isinstance(source, Mapping):
            continue
        for key in ("reasoning_content", "reasoning"):
            value = source.get(key)
            if isinstance(value, str) and value:
                return value

    content = message.get("content")
    if not isinstance(content, list):
        return None
    parts = [
        part[part["type"]]
        for part in content
        if isinstance(part, Mapping)
        and part.get("type") in _REASONING_PART_TYPES
        and isinstance(part.get(part["type"]), str)
    ]
    return "".join(parts) or None


def _tool_calls_from_message(message: Mapping[str, Any]) -> list[ChatCompletionStreamToolCall]:
    """Return OpenAI ``tool_calls`` or, when absent, Anthropic ``tool_use`` content blocks."""
    tool_calls = _normalize_tool_calls(message.get("tool_calls"))
    if tool_calls:
        return tool_calls
    content = message.get("content")
    if not isinstance(content, list):
        return []
    return _normalize_tool_calls(
        [part for part in content if isinstance(part, Mapping) and part.get("type") == "tool_use"]
    )


def _normalize_tool_calls(value: Any) -> list[ChatCompletionStreamToolCall]:
    """Normalize flat, OpenAI-nested, LangChain, or Anthropic tool calls to OpenAI stream shape."""
    if not isinstance(value, list):
        return []
    tool_calls = []
    for call in value:
        if not isinstance(call, Mapping):
            continue
        function = call.get("function")
        if not isinstance(function, Mapping):
            function = call
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        arguments = next(
            (args for args in (function.get("arguments"), call.get("args"), call.get("input")) if args is not None),
            {},
        )
        call_id = call.get("id")
        tool_calls.append(
            ChatCompletionStreamToolCall(
                id=call_id if isinstance(call_id, str) and call_id else None,
                function=ChatCompletionStreamToolCallFunction(
                    name=name,
                    arguments=arguments if isinstance(arguments, str) else json.dumps(arguments),
                ),
            )
        )
    return tool_calls


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
