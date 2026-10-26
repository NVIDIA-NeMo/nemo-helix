# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Each VirtualModel call is recorded to Intake as one span, without touching the call."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient
from multidict import CIMultiDict, CIMultiDictProxy
from nemo_helix_plugin.intake.types import DirectSpanInput
from nhx.core.inference_gateway.api.call_recording import (
    CallApi,
    CallObserver,
    CallRecorder,
    CallStatus,
    TokenUsage,
    global_call_recorder,
    set_global_call_recorder,
    usage_from_payload,
)

CHAT_URL = "/v2/workspaces/e2e-test/openai/-/v1/chat/completions"
CHAT_BODY = {"model": "meta_llama-3.2-1b-instruct", "messages": [{"role": "user", "content": "Hello"}]}


class CollectingRecorder(CallRecorder):
    """A recorder that keeps what it is handed instead of queueing it for Intake."""

    def __init__(self) -> None:
        super().__init__(self._unused_sink)
        self.spans: list[tuple[str, DirectSpanInput]] = []

    async def _unused_sink(self, workspace: str, spans: list[DirectSpanInput]) -> None:
        raise AssertionError("CollectingRecorder never sends")

    def record(self, workspace: str, span: DirectSpanInput) -> None:
        self.spans.append((workspace, span))


@pytest.fixture
def recorded(client: TestClient) -> Iterator[CollectingRecorder]:
    previous = global_call_recorder()
    recorder = CollectingRecorder()
    set_global_call_recorder(recorder)
    yield recorder
    set_global_call_recorder(previous)


def _span(recorded: CollectingRecorder) -> DirectSpanInput:
    [(workspace, span)] = recorded.spans
    assert workspace == "e2e-test"
    return span


# --- usage per API ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("api", "payloads", "expected"),
    [
        pytest.param(
            CallApi.CHAT_COMPLETIONS,
            [{"usage": {"prompt_tokens": 100, "completion_tokens": 7, "prompt_tokens_details": {"cached_tokens": 60}}}],
            TokenUsage(uncached_input_tokens=40, cached_input_tokens=60, output_tokens=7),
            id="chat-body",
        ),
        pytest.param(
            CallApi.CHAT_COMPLETIONS,
            [
                {"choices": [{"delta": {"content": "Hi"}}], "usage": None},
                {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 3}},
            ],
            TokenUsage(uncached_input_tokens=12, output_tokens=3),
            id="chat-stream-include-usage",
        ),
        pytest.param(
            CallApi.RESPONSES,
            [{"usage": {"input_tokens": 50, "output_tokens": 9, "input_tokens_details": {"cached_tokens": 20}}}],
            TokenUsage(uncached_input_tokens=30, cached_input_tokens=20, output_tokens=9),
            id="responses-body",
        ),
        pytest.param(
            CallApi.RESPONSES,
            [
                {"type": "response.output_text.delta", "delta": "Hi"},
                {
                    "type": "response.completed",
                    "response": {"usage": {"input_tokens": 40, "output_tokens": 4, "input_tokens_details": {}}},
                },
            ],
            TokenUsage(uncached_input_tokens=40, output_tokens=4),
            id="responses-stream",
        ),
        pytest.param(
            CallApi.MESSAGES,
            [
                {
                    "usage": {
                        "input_tokens": 10,
                        "cache_read_input_tokens": 30,
                        "cache_creation_input_tokens": 5,
                        "output_tokens": 8,
                    }
                }
            ],
            TokenUsage(uncached_input_tokens=10, cached_input_tokens=30, cache_write_input_tokens=5, output_tokens=8),
            id="messages-body",
        ),
        pytest.param(
            CallApi.MESSAGES,
            [
                {
                    "type": "message_start",
                    "message": {"usage": {"input_tokens": 10, "cache_read_input_tokens": 30, "output_tokens": 1}},
                },
                {"type": "content_block_delta", "delta": {"text": "Hi"}},
                {
                    "type": "message_delta",
                    "usage": {"input_tokens": 10, "cache_read_input_tokens": None, "output_tokens": 15},
                },
            ],
            TokenUsage(uncached_input_tokens=10, cached_input_tokens=30, output_tokens=15),
            id="messages-stream",
        ),
    ],
)
def test_usage_is_read_from_every_api_and_stream_shape(
    api: CallApi, payloads: list[dict[str, Any]], expected: TokenUsage
) -> None:
    """Every API's counts land in the same classes; a later chunk's null count keeps the earlier value."""
    usage = TokenUsage()
    for payload in payloads:
        if (reported := usage_from_payload(api, payload)) is not None:
            usage.merge(reported)
    assert usage == expected


def test_input_total_counts_every_class_once() -> None:
    usage = TokenUsage(uncached_input_tokens=10, cached_input_tokens=30, cache_write_input_tokens=5, output_tokens=8)
    assert usage.input_tokens == 45
    assert usage.attributes() == {
        "gen_ai.usage.input_tokens": 45,
        "nemo.usage.uncached_input_tokens": 10,
        "gen_ai.usage.cached_tokens": 30,
        "llm.token_count.prompt_details.cache_write": 5,
        "gen_ai.usage.output_tokens": 8,
    }


@pytest.mark.parametrize(
    ("path", "api"),
    [
        ("v1/chat/completions", CallApi.CHAT_COMPLETIONS),
        ("/v1/responses", CallApi.RESPONSES),
        ("v1/messages", CallApi.MESSAGES),
        ("v1/embeddings", CallApi.OTHER),
    ],
)
def test_api_is_read_from_the_path(path: str, api: CallApi) -> None:
    assert CallApi.from_path(path) is api


# --- the recorder -----------------------------------------------------------------------------


def _bare_span(name: str = "chat.completions") -> DirectSpanInput:
    now = datetime.now(UTC)
    return DirectSpanInput(span_id="s", trace_id=name, name=name, started_at=now, ended_at=now)


async def test_queued_spans_are_sent_in_one_batch_per_workspace() -> None:
    sent: list[tuple[str, list[str]]] = []

    async def sink(workspace: str, spans: list[DirectSpanInput]) -> None:
        sent.append((workspace, [span.trace_id for span in spans]))

    recorder = CallRecorder(sink, flush_interval_sec=0.05)
    recorder.start()
    recorder.record("a", _bare_span("1"))
    recorder.record("b", _bare_span("2"))
    recorder.record("a", _bare_span("3"))
    await asyncio.sleep(0.2)
    await recorder.stop()

    assert sorted(sent) == [("a", ["1", "3"]), ("b", ["2"])]


async def test_an_unreachable_intake_never_blocks_recording_and_drops_its_batches() -> None:
    """Recording only enqueues, so an Intake that hangs or fails costs the call nothing."""
    calls = 0

    async def hanging_sink(workspace: str, spans: list[DirectSpanInput]) -> None:
        nonlocal calls
        calls += 1
        await asyncio.Event().wait()

    recorder = CallRecorder(hanging_sink, max_queued_spans=2, flush_interval_sec=0.01)
    recorder.start()
    recorder.record("ws", _bare_span())
    await asyncio.sleep(0.05)  # the sender is now stuck on that span's batch

    started = time.monotonic()
    for _ in range(5):
        recorder.record("ws", _bare_span())
    assert time.monotonic() - started < 0.01
    assert recorder.dropped == 3  # two fit in the queue

    await recorder.stop(timeout_sec=0.05)
    assert calls >= 1


async def test_a_failing_intake_drops_the_batch_and_keeps_sending() -> None:
    sent: list[int] = []

    async def flaky_sink(workspace: str, spans: list[DirectSpanInput]) -> None:
        if not sent:
            sent.append(0)
            raise RuntimeError("Intake is down")
        sent.append(len(spans))

    recorder = CallRecorder(flaky_sink, flush_interval_sec=0.01)
    recorder.start()
    recorder.record("ws", _bare_span())
    await asyncio.sleep(0.05)
    recorder.record("ws", _bare_span())
    await asyncio.sleep(0.05)
    await recorder.stop()

    assert sent == [0, 1]
    assert recorder.dropped == 1


async def test_stop_sends_what_is_still_queued() -> None:
    sent: list[DirectSpanInput] = []

    async def sink(workspace: str, spans: list[DirectSpanInput]) -> None:
        sent.extend(spans)

    recorder = CallRecorder(sink, flush_interval_sec=60)
    recorder.record("ws", _bare_span())
    await recorder.stop()

    assert len(sent) == 1


# --- through the proxy ------------------------------------------------------------------------


def test_a_call_records_its_caller_target_usage_and_agent_session(
    client: TestClient, recorded: CollectingRecorder, mock_proxy_response: Any
) -> None:
    mock_proxy_response._body = [
        json.dumps(
            {
                "choices": [{"message": {"content": "Hi"}}],
                "usage": {"prompt_tokens": 21, "completion_tokens": 4, "prompt_tokens_details": {"cached_tokens": 16}},
            }
        ).encode()
    ]

    response = client.post(
        CHAT_URL,
        json=CHAT_BODY,
        headers={
            "x-claude-code-session-id": "session-1",
            "x-codex-turn-metadata": '{"turn": 2}',
            "x-forwarded-for": "6.6.6.6",
        },
    )

    assert response.status_code == 200
    span = _span(recorded)
    assert span.status == "success"
    assert span.kind == "LLM"
    assert span.name == "chat.completions"
    assert span.session_id == "session-1"
    assert span.ended_at is not None and span.started_at <= span.ended_at
    assert span.attributes["client.address"] == "testclient"  # the socket peer, never X-Forwarded-For
    assert span.attributes["nemo.virtual_model"] == "e2e-test/meta_llama-3.2-1b-instruct"
    assert span.attributes["gen_ai.request.model"] == "e2e-test/meta_llama-3.2-1b-instruct"
    assert span.attributes["gen_ai.provider.name"] == "ollama"
    assert span.attributes["http.response.status_code"] == 200
    assert span.attributes["gen_ai.usage.input_tokens"] == 21
    assert span.attributes["nemo.usage.uncached_input_tokens"] == 5
    assert span.attributes["gen_ai.usage.cached_tokens"] == 16
    assert span.attributes["gen_ai.usage.output_tokens"] == 4
    assert span.attributes["nemo.agent.header.x-claude-code-session-id"] == "session-1"
    assert span.attributes["nemo.agent.header.x-codex-turn-metadata"] == '{"turn": 2}'


def test_a_streamed_call_records_the_usage_of_its_final_chunk(
    client: TestClient, recorded: CollectingRecorder, mock_proxy_response: Any
) -> None:
    mock_proxy_response.headers = CIMultiDictProxy(CIMultiDict([("content-type", "text/event-stream")]))
    mock_proxy_response._body = [
        b'data: {"choices": [{"delta": {"content": "Hi"}}]}\n\n',
        b'data: {"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 2}}\n\n',
        b"data: [DONE]\n\n",
    ]

    response = client.post(CHAT_URL, json={**CHAT_BODY, "stream": True, "stream_options": {"include_usage": True}})

    assert response.status_code == 200
    assert "data: [DONE]" in response.text
    span = _span(recorded)
    assert span.status == "success"
    assert span.attributes["gen_ai.usage.input_tokens"] == 9
    assert span.attributes["gen_ai.usage.output_tokens"] == 2


def test_a_call_the_upstream_rejects_is_recorded_as_an_error(
    client: TestClient, recorded: CollectingRecorder, mock_proxy_response: Any
) -> None:
    mock_proxy_response.status = 500

    async def _error_body() -> bytes:
        return b'{"error": "boom"}'

    mock_proxy_response.read = _error_body

    response = client.post(CHAT_URL, json=CHAT_BODY)

    assert response.status_code == 500
    span = _span(recorded)
    assert span.status == "error"
    assert span.attributes["http.response.status_code"] == 500
    assert span.attributes["gen_ai.request.model"] == "e2e-test/meta_llama-3.2-1b-instruct"


def test_a_full_recording_queue_does_not_change_the_call(client: TestClient) -> None:
    async def never_called(workspace: str, spans: list[DirectSpanInput]) -> None:
        raise AssertionError("the sender is not running")

    previous = global_call_recorder()
    recorder = CallRecorder(never_called, max_queued_spans=1)
    recorder.record("e2e-test", _bare_span())
    set_global_call_recorder(recorder)
    try:
        response = client.post(CHAT_URL, json=CHAT_BODY)
    finally:
        set_global_call_recorder(previous)

    assert response.status_code == 200
    assert recorder.dropped == 1


def test_plugin_attributes_are_recorded_but_cannot_replace_the_gateways_own() -> None:
    recorder = CollectingRecorder()
    observer = CallObserver(
        recorder,
        workspace="ws",
        virtual_model="vm",
        trailing_uri="v1/responses",
        client_address="10.244.0.7",
        headers={},
    )
    observer.telemetry_attributes = {
        "nemo-evals-closed-book.removed_tools": ["web_search"],
        "client.address": "6.6.6.6",
    }

    observer.finish(CallStatus.SUCCESS, 200)
    observer.finish(CallStatus.ERROR, 500)

    [(_, span)] = recorder.spans
    assert span.attributes["nemo-evals-closed-book.removed_tools"] == ["web_search"]
    assert span.attributes["client.address"] == "10.244.0.7"
    assert span.status == "success"


async def test_stop_is_bounded_even_when_the_sink_ignores_cancellation() -> None:
    release = asyncio.Event()

    async def stubborn_sink(workspace: str, spans: list[DirectSpanInput]) -> None:
        while not release.is_set():
            try:
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                continue

    recorder = CallRecorder(stubborn_sink, flush_interval_sec=0.01)
    recorder.start()
    recorder.record("ws", _bare_span())
    await asyncio.sleep(0.05)

    started = time.monotonic()
    await recorder.stop(timeout_sec=0.1)
    assert time.monotonic() - started < 0.5
    release.set()


async def test_a_complete_error_body_is_recorded_as_an_error() -> None:
    recorder = CollectingRecorder()
    observer = CallObserver(
        recorder,
        workspace="ws",
        virtual_model="vm",
        trailing_uri="v1/chat/completions",
        client_address=None,
        headers={},
    )

    async def body() -> AsyncIterator[bytes]:
        yield b'{"error": "boom"}'

    assert [part async for part in observer.follow_body(body(), 502)] == [b'{"error": "boom"}']

    [(_, span)] = recorder.spans
    assert span.status == "error"


async def test_a_caller_that_disconnects_mid_stream_is_recorded_as_cancelled() -> None:
    recorder = CollectingRecorder()
    observer = CallObserver(
        recorder, workspace="ws", virtual_model="vm", trailing_uri="v1/messages", client_address=None, headers={}
    )

    async def body() -> AsyncIterator[bytes]:
        for part in (b"one", b"two", b"three"):
            yield part

    followed = observer.follow_body(body(), 200)
    assert await anext(followed) == b"one"
    await followed.aclose()

    [(_, span)] = recorder.spans
    assert span.status == "cancelled"
    assert "client.address" not in span.attributes
