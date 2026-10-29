# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from nemo_agents_plugin.fabric.runtime import FabricRuntimeResult, FabricRuntimeStartError, FabricRuntimeStream
from nemo_agents_plugin.fabric.serving_models import ChatCompletionStreamDelta
from nemo_agents_plugin.fabric.streaming import (
    FabricStreamResultError,
    StreamingMethod,
    extract_assistant_delta,
    extract_assistant_text_delta,
    iter_fabric_assistant_deltas,
    iter_fabric_chat_completion_sse,
    iter_openai_chat_completion_sse,
    resolve_streaming_method,
)
from nemo_fabric import FabricConfig, RunPlan


class _FakeFabricRuntimeStream(FabricRuntimeStream):
    """A Fabric runtime stream that replays scripted ATOF records and a terminal result."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        *,
        result: FabricRuntimeResult | None = None,
        streaming_method: StreamingMethod = StreamingMethod.RELAY,
    ) -> None:
        super().__init__(stream=None, streaming_method=streaming_method)
        self._records = records
        self._result = result or FabricRuntimeResult(status="succeeded", response="done")
        self.result_awaited = False

    async def records(self) -> AsyncIterator[dict[str, Any]]:
        for record in self._records:
            yield record

    async def result(self) -> FabricRuntimeResult:
        self.result_awaited = True
        return self._result


async def _deltas() -> AsyncIterator[ChatCompletionStreamDelta]:
    yield ChatCompletionStreamDelta(content="hel")
    yield ChatCompletionStreamDelta(reasoning_content="think")
    yield ChatCompletionStreamDelta(content="lo")


def _event_payload(event: str) -> dict[str, object]:
    assert event.startswith("data: ")
    assert event.endswith("\n\n")
    return json.loads(event.removeprefix("data: ").removesuffix("\n\n"))


@pytest.mark.asyncio
async def test_iter_openai_chat_completion_sse_frames_deltas() -> None:
    events = [
        event
        async for event in iter_openai_chat_completion_sse(
            completion_id="chatcmpl-test",
            deltas=_deltas(),
            model="test-model",
        )
    ]

    assert len(events) == 6
    assert events[-1] == "data: [DONE]\n\n"

    first_chunk = _event_payload(events[0])
    assert first_chunk["id"] == "chatcmpl-test"
    assert first_chunk["object"] == "chat.completion.chunk"
    assert first_chunk["model"] == "test-model"
    assert first_chunk["choices"] == [{"index": 0, "delta": {"role": "assistant"}}]

    second_chunk = _event_payload(events[1])
    assert second_chunk["choices"] == [{"index": 0, "delta": {"content": "hel"}}]

    third_chunk = _event_payload(events[2])
    assert third_chunk["choices"] == [{"index": 0, "delta": {"reasoning_content": "think"}}]

    fourth_chunk = _event_payload(events[3])
    assert fourth_chunk["choices"] == [{"index": 0, "delta": {"content": "lo"}}]

    terminal_chunk = _event_payload(events[4])
    assert terminal_chunk["choices"] == [{"index": 0, "delta": {}, "finish_reason": "stop"}]


@pytest.mark.asyncio
async def test_iter_fabric_chat_completion_sse_preserves_native_chunks() -> None:
    chunks = [
        {
            "id": "native-id",
            "object": "chat.completion.chunk",
            "model": "native-model",
            "created": 123,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "hello"}}],
        },
        {
            "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{}"}}]}}],
        },
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {"completion_tokens": 12, "prompt_tokens": 5, "total_tokens": 17}},
    ]
    stream = _FakeFabricRuntimeStream(chunks, streaming_method=StreamingMethod.OPENAI)

    events = [
        event
        async for event in iter_fabric_chat_completion_sse(stream, completion_id="platform-id", model="platform-model")
    ]

    assert [_event_payload(event) for event in events[:-1]] == chunks
    assert events[-1] == "data: [DONE]\n\n"
    assert stream.result_awaited


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        (
            {"data": {"choices": [{"delta": {"content": "hel"}}]}},
            "hel",
        ),
        (
            {"data": {"choices": [{"delta": {"role": "assistant", "content": "assistant delta"}}]}},
            "assistant delta",
        ),
        (
            {"data": {"type": "agentMessage", "phase": "final_answer", "text": "done"}},
            "done",
        ),
        (
            {"payload": {"agent": {"messages": [{"role": "ai", "content": "deep result"}]}}},
            "deep result",
        ),
        (
            {"data": {"messages": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]}},
            "hi",
        ),
        (
            {"data": {"role": "assistant", "content": [{"type": "text", "text": "hello"}]}},
            "hello",
        ),
        (
            {
                "kind": "scope",
                "scope_category": "end",
                "category": "llm",
                "name": "anthropic.messages",
                "data": {"role": "assistant", "content": [{"type": "text", "text": "llm text"}]},
            },
            "llm text",
        ),
    ],
)
def test_extract_assistant_text_delta_from_known_text_shapes(
    record: dict[str, object],
    expected: str,
) -> None:
    assert extract_assistant_text_delta(record) == expected


@pytest.mark.parametrize(
    "record",
    [
        {"kind": "scope", "scope_category": "start", "name": "request"},
        {"kind": "mark", "uuid": "mark-1", "parent_uuid": "scope-1"},
        {"payload": "lifecycle label"},
        {"data": {"role": "user", "content": "hello"}},
        {"data": {"type": "text_delta", "role": "user", "text": "user delta"}},
        {"data": {"choices": [{"delta": {"role": "user", "content": "user choice"}}]}},
        {"data": {"type": "toolCall", "text": "tool output"}},
        {
            "kind": "scope",
            "scope_category": "end",
            "category": "agent",
            "name": "agent-turn",
            "data": {"role": "assistant", "content": "agent summary"},
        },
        {
            "kind": "scope",
            "scope_category": "end",
            "category": "tool",
            "name": "tool-call",
            "data": {"type": "agentMessage", "text": "tool output"},
        },
        {
            "kind": "scope",
            "scope_category": "end",
            "category": "custom",
            "name": "claude-code-turn",
            "data": {
                "role": "assistant",
                "content": [{"type": "text", "text": "turn summary"}],
            },
        },
    ],
)
def test_extract_assistant_text_delta_skips_non_assistant_records(record: dict[str, object]) -> None:
    assert extract_assistant_text_delta(record) is None


@pytest.mark.asyncio
async def test_iter_fabric_assistant_deltas_preserves_extracted_text_order() -> None:
    stream = _FakeFabricRuntimeStream(
        [
            {"kind": "scope", "scope_category": "start", "name": "request"},
            {"data": {"choices": [{"delta": {"content": "hel"}}]}},
            {"data": {"type": "toolCall", "text": "tool output"}},
            {"data": {"type": "agentMessage", "phase": "final_answer", "text": "lo"}},
        ]
    )

    assert [delta.content async for delta in iter_fabric_assistant_deltas(stream)] == ["hel", "lo"]
    assert stream.result_awaited is True


@pytest.mark.asyncio
async def test_iter_fabric_assistant_deltas_skips_duplicate_turn_summary() -> None:
    stream = _FakeFabricRuntimeStream(
        [
            {
                "kind": "scope",
                "scope_category": "end",
                "name": "anthropic.messages",
                "category": "llm",
                "data": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "hello"}],
                },
            },
            {
                "kind": "scope",
                "scope_category": "end",
                "name": "claude-code-turn",
                "category": "custom",
                "metadata": {"nemo_relay_scope_role": "turn"},
                "data": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "hello"}],
                },
            },
        ]
    )

    assert [delta.content async for delta in iter_fabric_assistant_deltas(stream)] == ["hello"]


@pytest.mark.asyncio
async def test_iter_fabric_assistant_deltas_falls_back_to_terminal_response() -> None:
    stream = _FakeFabricRuntimeStream(
        [{"kind": "scope", "scope_category": "start", "name": "request"}],
        result=FabricRuntimeResult(status="succeeded", response="terminal response"),
    )

    assert [delta.content async for delta in iter_fabric_assistant_deltas(stream)] == ["terminal response"]


@pytest.mark.asyncio
async def test_iter_fabric_assistant_deltas_raises_failed_terminal_result() -> None:
    stream = _FakeFabricRuntimeStream(
        [{"data": {"choices": [{"delta": {"content": "partial"}}]}}],
        result=FabricRuntimeResult(status="failed", response="done", error={"message": "terminal failure"}),
    )

    with pytest.raises(FabricStreamResultError, match="terminal failure"):
        _ = [delta.content async for delta in iter_fabric_assistant_deltas(stream)]


def _streaming_plan(config: FabricConfig, descriptor: dict[str, Any] | None, *, streaming: bool = False) -> RunPlan:
    assert config.harness is not None
    return RunPlan.from_mapping(
        {
            "base_dir": "/tmp",
            "config": config.model_dump(mode="json"),
            "adapter": {"adapter_id": config.harness.adapter_id, "adapter_kind": "python"},
            "capabilities": {"streaming": streaming},
            "adapter_descriptor": {"descriptor": descriptor} if descriptor is not None else None,
        }
    )


@pytest.mark.parametrize(
    ("descriptor", "streaming", "expected"),
    [
        ({"telemetry": {"providers": {"relay": {}}}}, False, StreamingMethod.RELAY),
        (
            {"telemetry": {"providers": {"relay": {}}}, "capabilities": {"streaming": True}},
            True,
            StreamingMethod.RELAY,
        ),
        ({"capabilities": {"streaming": True}}, True, StreamingMethod.OPENAI),
        ({"capabilities": {"streaming": True}}, False, StreamingMethod.BUFFERED),
        ({"capabilities": {"streaming": False}}, True, StreamingMethod.BUFFERED),
        (None, True, StreamingMethod.BUFFERED),
        ({}, False, StreamingMethod.BUFFERED),
    ],
)
def test_resolve_streaming_method(
    descriptor: dict[str, Any] | None, streaming: bool, expected: StreamingMethod
) -> None:
    config = FabricConfig.model_validate(
        {"metadata": {"name": "test-agent"}, "harness": {"adapter_id": "nvidia.fabric.custom"}}
    )
    plan = _streaming_plan(config, descriptor, streaming=streaming)
    before_config, before_plan = config.model_dump(), plan.to_mapping()

    assert resolve_streaming_method(plan, config) is expected
    assert config.model_dump() == before_config
    assert plan.to_mapping() == before_plan


@pytest.mark.parametrize("configured", [False, True])
def test_streaming_method_validates_remote_collector(configured: bool) -> None:
    config = FabricConfig.model_validate(
        {
            "metadata": {"name": "remote"},
            "harness": {"adapter_id": "nvidia.fabric.remote-agent", "settings": {"relay_streaming": True}},
            "relay": {
                "observability": {
                    "atof": {
                        "enabled": True,
                        "sinks": (
                            [{"name": "nemo-fabric-stream", "type": "stream", "url": "http://collector:8080/v1/atof"}]
                            if configured
                            else []
                        ),
                    }
                }
            },
        }
    )
    plan = _streaming_plan(
        config,
        {"telemetry": {"providers": {"relay": {"outputs": ["atof"]}}}, "capabilities": {"streaming": True}},
        streaming=True,
    )
    if configured:
        assert resolve_streaming_method(plan, config) is StreamingMethod.RELAY
    else:
        with pytest.raises(FabricRuntimeStartError, match="shared collector"):
            resolve_streaming_method(plan, config)


def _llm_end(data: object, *, annotated_response: dict[str, object] | None = None) -> dict[str, object]:
    record: dict[str, object] = {
        "kind": "scope",
        "scope_category": "end",
        "category": "llm",
        "name": "model",
        "data": data,
    }
    if annotated_response is not None:
        record["category_profile"] = {"annotated_response": annotated_response, "model_name": "model"}
    return record


# Shape captured from Relay's LangChain middleware for an AIMessage with reasoning and a tool call.
_LANGCHAIN_TOOL_TURN = _llm_end(
    {
        "__nemo_relay_integrations_langchain_model_response": {
            "messages": [
                {
                    "type": "ai",
                    "data": {
                        "type": "ai",
                        "additional_kwargs": {"reasoning_content": "Thinking about adding."},
                        "content": [
                            {"type": "reasoning", "reasoning": "I should add."},
                            {"type": "text", "text": "Let me add."},
                        ],
                        "tool_calls": [{"args": {"a": 1, "b": 2}, "id": "call_1", "name": "add", "type": "tool_call"}],
                    },
                }
            ]
        }
    },
    annotated_response={
        "finish_reason": "tool_use",
        "message": "Let me add.",
        "tool_calls": [{"arguments": {"a": 1, "b": 2}, "id": "call_1", "name": "add"}],
    },
)


def test_extract_assistant_delta_from_relay_langchain_llm_scope() -> None:
    delta = extract_assistant_delta(_LANGCHAIN_TOOL_TURN)

    assert delta is not None
    assert delta.model_dump(exclude_none=True) == {
        "content": "Let me add.",
        "reasoning_content": "Thinking about adding.",
    }


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "reasoning_content": "plan",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "search", "arguments": '{"q":"x"}'},
                                }
                            ],
                        }
                    }
                ]
            },
            {"reasoning_content": "plan"},
        ),
        (
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "hmm"},
                    {"type": "text", "text": "Searching."},
                    {"type": "tool_use", "id": "toolu_1", "name": "search", "input": {"q": "x"}},
                ],
            },
            {"content": "Searching.", "reasoning_content": "hmm"},
        ),
    ],
    ids=["openai", "anthropic"],
)
def test_extract_assistant_delta_from_provider_llm_scopes(data: dict[str, object], expected: dict[str, object]) -> None:
    delta = extract_assistant_delta(_llm_end(data))

    assert delta is not None
    assert delta.model_dump(exclude_none=True) == expected


def test_extract_assistant_delta_skips_tool_scopes() -> None:
    record = {
        "kind": "scope",
        "scope_category": "end",
        "category": "tool",
        "name": "add",
        "category_profile": {"tool_call_id": "call_1"},
        "data": {"content": "3", "tool_call_id": "call_1", "type": "tool"},
    }

    assert extract_assistant_delta(record) is None


@pytest.mark.asyncio
async def test_iter_fabric_assistant_deltas_omits_tool_calls() -> None:
    stream = _FakeFabricRuntimeStream(
        [
            _LANGCHAIN_TOOL_TURN,
            _llm_end(
                {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_2", "name": "add", "input": {}}]}
            ),
            _llm_end({}, annotated_response={"message": "The answer is 3."}),
        ],
        result=FabricRuntimeResult(status="succeeded", response="The answer is 3."),
    )

    deltas = [delta.model_dump(exclude_none=True) async for delta in iter_fabric_assistant_deltas(stream)]

    assert deltas == [
        {"content": "Let me add.", "reasoning_content": "Thinking about adding."},
        {"content": "The answer is 3."},
    ]


@pytest.mark.asyncio
async def test_iter_fabric_assistant_deltas_falls_back_to_response_after_tool_only_turns() -> None:
    stream = _FakeFabricRuntimeStream(
        [
            _llm_end(
                {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_1", "name": "add", "input": {}}]}
            )
        ],
        result=FabricRuntimeResult(status="succeeded", response="final"),
    )

    deltas = [delta async for delta in iter_fabric_assistant_deltas(stream)]

    assert [delta.content for delta in deltas] == ["final"]
