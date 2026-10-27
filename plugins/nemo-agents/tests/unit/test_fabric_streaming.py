# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from nemo_agents_plugin.fabric.runtime import FabricRuntimeResult, FabricRuntimeStartError, FabricRuntimeStream
from nemo_agents_plugin.fabric.streaming import (
    FabricStreamResultError,
    StreamingMethod,
    extract_assistant_text_delta,
    iter_fabric_assistant_text_deltas,
    iter_openai_chat_completion_sse,
    resolve_streaming_method,
)
from nemo_fabric import FabricConfig, RunPlan


class _FakeFabricRuntimeStream(FabricRuntimeStream):
    """A Fabric runtime stream that replays scripted ATOF records and a terminal result."""

    def __init__(self, records: list[dict[str, Any]], *, result: FabricRuntimeResult | None = None) -> None:
        super().__init__(stream=None)
        self._records = records
        self._result = result or FabricRuntimeResult(status="succeeded", response="done")
        self.result_awaited = False

    async def records(self) -> AsyncIterator[dict[str, Any]]:
        for record in self._records:
            yield record

    async def result(self) -> FabricRuntimeResult:
        self.result_awaited = True
        return self._result


async def _content_chunks() -> AsyncIterator[str]:
    yield "hel"
    yield ""
    yield "lo"


def _event_payload(event: str) -> dict[str, object]:
    assert event.startswith("data: ")
    assert event.endswith("\n\n")
    return json.loads(event.removeprefix("data: ").removesuffix("\n\n"))


@pytest.mark.asyncio
async def test_iter_openai_chat_completion_sse_frames_content_chunks() -> None:
    events = [
        event
        async for event in iter_openai_chat_completion_sse(
            completion_id="chatcmpl-test",
            content_chunks=_content_chunks(),
            model="test-model",
        )
    ]

    assert len(events) == 5
    assert events[-1] == "data: [DONE]\n\n"

    first_chunk = _event_payload(events[0])
    assert first_chunk["id"] == "chatcmpl-test"
    assert first_chunk["object"] == "chat.completion.chunk"
    assert first_chunk["model"] == "test-model"
    assert first_chunk["choices"] == [{"index": 0, "delta": {"role": "assistant"}}]

    second_chunk = _event_payload(events[1])
    assert second_chunk["choices"] == [{"index": 0, "delta": {"content": "hel"}}]

    third_chunk = _event_payload(events[2])
    assert third_chunk["choices"] == [{"index": 0, "delta": {"content": "lo"}}]

    terminal_chunk = _event_payload(events[3])
    assert terminal_chunk["choices"] == [{"index": 0, "delta": {}, "finish_reason": "stop"}]


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
async def test_iter_fabric_assistant_text_deltas_preserves_extracted_text_order() -> None:
    stream = _FakeFabricRuntimeStream(
        [
            {"kind": "scope", "scope_category": "start", "name": "request"},
            {"data": {"choices": [{"delta": {"content": "hel"}}]}},
            {"data": {"type": "toolCall", "text": "tool output"}},
            {"data": {"type": "agentMessage", "phase": "final_answer", "text": "lo"}},
        ]
    )

    assert [text async for text in iter_fabric_assistant_text_deltas(stream)] == ["hel", "lo"]
    assert stream.result_awaited is True


@pytest.mark.asyncio
async def test_iter_fabric_assistant_text_deltas_skips_duplicate_turn_summary() -> None:
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

    assert [text async for text in iter_fabric_assistant_text_deltas(stream)] == ["hello"]


@pytest.mark.asyncio
async def test_iter_fabric_assistant_text_deltas_falls_back_to_terminal_response() -> None:
    stream = _FakeFabricRuntimeStream(
        [{"kind": "scope", "scope_category": "start", "name": "request"}],
        result=FabricRuntimeResult(status="succeeded", response="terminal response"),
    )

    assert [text async for text in iter_fabric_assistant_text_deltas(stream)] == ["terminal response"]


@pytest.mark.asyncio
async def test_iter_fabric_assistant_text_deltas_raises_failed_terminal_result() -> None:
    stream = _FakeFabricRuntimeStream(
        [{"data": {"choices": [{"delta": {"content": "partial"}}]}}],
        result=FabricRuntimeResult(status="failed", response="done", error={"message": "terminal failure"}),
    )

    with pytest.raises(FabricStreamResultError, match="terminal failure"):
        _ = [text async for text in iter_fabric_assistant_text_deltas(stream)]


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
