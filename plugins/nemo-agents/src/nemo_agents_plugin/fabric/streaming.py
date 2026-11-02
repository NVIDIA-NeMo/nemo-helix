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
from typing import Any

from nemo_agents_plugin.fabric.runtime import (
    FabricRuntimeResult,
    FabricRuntimeStream,
    StreamingMethod,
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
# extraction below only handles known assistant-output shapes. Relay also
# attaches a provider-neutral ``category_profile.annotated_response`` to LLM
# scopes, which is used as a text fallback.
_ASSISTANT_ROLES = frozenset({"assistant", "ai"})
_USER_ROLES = frozenset({"user", "human"})
_ATOF_EVENT_KINDS = frozenset({"scope", "mark"})
# Content blocks carry reasoning under a key matching their type: LangChain uses
# ``{"type": "reasoning", "reasoning": ...}`` and Anthropic ``{"type": "thinking", "thinking": ...}``.
_REASONING_PART_TYPES = frozenset({"reasoning", "thinking"})
_REASONING_FIELDS = ("reasoning_content", "reasoning")
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


def extract_assistant_delta(record: Mapping[str, Any]) -> ChatCompletionStreamDelta | None:
    """Extract assistant text and reasoning from a known ATOF record shape.

    Fabric streams raw Relay ATOF dictionaries rather than a normalized delta
    contract. Completed LLM scopes yield text and reasoning; other records only
    yield known assistant text shapes. Lifecycle, metadata, tool, and
    user-message records are skipped. Tool calls are omitted because Chat
    Completions ``tool_calls`` asks the client to run them.
    """
    if _is_atof_record(record):
        return _extract_llm_output_delta(record) if _is_llm_output_scope(record) else None
    content = _extract_payload_text_delta(record)
    return ChatCompletionStreamDelta(content=content) if content is not None else None


def extract_output_reasoning(output: Any) -> str | None:
    """Collect the final turn's reasoning and tool-step narration from a Fabric result's ``messages``."""
    messages = output.get("messages") if isinstance(output, Mapping) else None
    if not isinstance(messages, list):
        return None
    turn_start = 0
    for index, message in enumerate(messages):
        if isinstance(message, Mapping) and message.get("role") in _USER_ROLES:
            turn_start = index + 1

    parts: list[str] = []
    for message in messages[turn_start:]:
        if not isinstance(message, Mapping) or message.get("role") not in _ASSISTANT_ROLES:
            continue
        reasoning = _reasoning_from_message(message)
        narration = _text_from_content(message.get("content")) if _requested_tools({}, message) else None
        parts.extend(part for part in (reasoning, narration) if part)
    return "\n\n".join(parts) or None


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


async def iter_fabric_chat_completion_sse(
    stream: FabricRuntimeStream, *, completion_id: str, model: str
) -> AsyncIterator[str]:
    """Preserve native chunks or frame Relay/buffered text as OpenAI SSE."""
    if stream.streaming_method is StreamingMethod.OPENAI:
        async for chunk in stream.records():
            yield _to_sse_frame(chunk)
        result = await stream.result()
        if result.status != "succeeded":
            raise FabricStreamResultError(result)
        yield "data: [DONE]\n\n"
    else:
        async for event in iter_openai_chat_completion_sse(
            completion_id=completion_id,
            deltas=iter_fabric_assistant_deltas(stream),
            model=model,
        ):
            yield event


async def iter_fabric_assistant_deltas(stream: FabricRuntimeStream) -> AsyncIterator[ChatCompletionStreamDelta]:
    """Yield the primary agent's assistant deltas, one paragraph per LLM step, and validate the terminal result."""
    emitted_text = False
    emitted_reasoning = False
    agents = _PrimaryAgentScopes()
    async for record in stream.records():
        agents.observe(record)
        is_llm_step = _is_atof_record(record) and _is_llm_output_scope(record)
        if is_llm_step and not agents.is_primary(record):
            continue
        delta = extract_assistant_delta(record)
        if delta is None:
            continue
        if is_llm_step:
            delta = _separate_from_previous_step(delta, after_text=emitted_text, after_reasoning=emitted_reasoning)
        emitted_text = emitted_text or delta.content is not None
        emitted_reasoning = emitted_reasoning or delta.reasoning_content is not None
        yield delta

    result = await stream.result()
    if result.status != "succeeded":
        raise FabricStreamResultError(result)
    if stream.streaming_method is StreamingMethod.BUFFERED and not isinstance(result.response, str):
        raise ValueError("Fabric invocation did not return a text response.")
    if not emitted_text and isinstance(result.response, str) and result.response:
        reasoning = None if emitted_reasoning else extract_output_reasoning(result.output)
        yield ChatCompletionStreamDelta(content=result.response, reasoning_content=reasoning)


class _PrimaryAgentScopes:
    """Treat the first agent-scoped LLM step's agent as primary; subagents only run after it delegates."""

    def __init__(self) -> None:
        self._scopes: dict[str, tuple[Any, Any]] = {}
        self._primary_agent: str | None = None

    def observe(self, record: Mapping[str, Any]) -> None:
        uuid = record.get("uuid")
        if record.get("kind") == "scope" and record.get("scope_category") == "start" and isinstance(uuid, str):
            self._scopes[uuid] = (record.get("category"), record.get("parent_uuid"))

    def is_primary(self, record: Mapping[str, Any]) -> bool:
        agent = self._nearest_agent(record.get("parent_uuid"))
        if agent is None:
            return True
        if self._primary_agent is None:
            self._primary_agent = agent
        return agent == self._primary_agent

    def _nearest_agent(self, uuid: Any) -> str | None:
        visited: set[str] = set()
        while isinstance(uuid, str) and uuid in self._scopes and uuid not in visited:
            visited.add(uuid)
            category, parent_uuid = self._scopes[uuid]
            if category == "agent":
                return uuid
            uuid = parent_uuid
        return None


def _separate_from_previous_step(
    delta: ChatCompletionStreamDelta, *, after_text: bool, after_reasoning: bool
) -> ChatCompletionStreamDelta:
    content = f"\n\n{delta.content}" if after_text and delta.content else delta.content
    reasoning = (
        f"\n\n{delta.reasoning_content}" if after_reasoning and delta.reasoning_content else delta.reasoning_content
    )
    return delta.model_copy(update={"content": content, "reasoning_content": reasoning})


def _to_sse_frame(chunk: ChatCompletionStreamResponse | ChatCompletionStreamErrorResponse | Mapping[str, Any]) -> str:
    data = dict(chunk) if isinstance(chunk, Mapping) else chunk.model_dump(mode="json", exclude_none=True)
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
    """Build one delta from a completed LLM scope; a tool-requesting step's text is narration, so it becomes reasoning."""
    data = record.get("data")
    profile = record.get("category_profile")
    annotated = profile.get("annotated_response") if isinstance(profile, Mapping) else None
    if not isinstance(annotated, Mapping):
        annotated = {}
    message = _find_assistant_message(data)
    if message is not None:
        content = _text_from_content(message.get("content"))
    else:
        message = {}
        content = _extract_payload_text_delta(data)
    content = content or _text_from_content(annotated.get("message"))
    reasoning = _reasoning_from_message(message)
    if content is not None and _requested_tools(annotated, message):
        reasoning = "\n\n".join(part for part in (reasoning, content) if part)
        content = None
    if content is None and reasoning is None:
        return None
    return ChatCompletionStreamDelta(content=content, reasoning_content=reasoning)


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


def _requested_tools(annotated: Mapping[str, Any], message: Mapping[str, Any]) -> bool:
    """Return true when an LLM step requested OpenAI/LangChain tool calls or Anthropic ``tool_use`` blocks."""
    if annotated.get("tool_calls") or message.get("tool_calls"):
        return True
    content = message.get("content")
    return isinstance(content, list) and any(
        isinstance(part, Mapping) and part.get("type") == "tool_use" for part in content
    )


def _reasoning_from_message(message: Mapping[str, Any]) -> str | None:
    """Return reasoning from message fields or reasoning/thinking content blocks."""
    for source in (message, message.get("additional_kwargs")):
        if not isinstance(source, Mapping):
            continue
        for key in _REASONING_FIELDS:
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
