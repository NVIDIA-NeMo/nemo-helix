# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Each VirtualModel call from a leased sandbox is recorded in its trial, without touching the call."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient
from multidict import CIMultiDict, CIMultiDictProxy
from nemo_helix_plugin.call_leases.types import CallLease
from nhx.core.inference_gateway.api import lease_cache
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
from nhx.core.inference_gateway.api.lease_cache import CallLeaseCache
from nhx.core.inference_gateway.api.trial_spans import GATEWAY_CALL_ATTRIBUTE, resource_spans
from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans, Span, Status

CHAT_URL = "/v2/workspaces/e2e-test/openai/-/v1/chat/completions"
CHAT_BODY = {"model": "meta_llama-3.2-1b-instruct", "messages": [{"role": "user", "content": "Hello"}]}
NOW = datetime.now(UTC)
LEASE = CallLease(
    name="lease-sbx-1",
    workspace="evals",
    client_address="10.244.0.7",
    session_id="run-1-task-0:hello__abc",
    trace_id="0af7651916cd43dd8448eb211c80319c",
    root_span_id="b7ad6b7169203331",
    role="agent",
    started_at=NOW - timedelta(hours=1),
    evaluation_name="run-1",
)


class CollectingRecorder(CallRecorder):
    """A recorder that keeps what it is handed instead of queueing it for Intake."""

    def __init__(self) -> None:
        super().__init__(self._unused_sink)
        self.spans: list[tuple[str, ResourceSpans]] = []

    async def _unused_sink(self, workspace: str, spans: list[ResourceSpans]) -> None:
        raise AssertionError("CollectingRecorder never sends")

    def record(self, workspace: str, span: ResourceSpans) -> None:
        self.spans.append((workspace, span))


class AnyCallerLeases(CallLeaseCache):
    """A cache whose lease holds every address, and that remembers which addresses were looked up."""

    def __init__(self, lease: CallLease | None = LEASE) -> None:
        super().__init__()
        self.lease = lease
        self.asked: list[str | None] = []

    def lookup(self, client_address: str | None, at: datetime) -> CallLease | None:
        self.asked.append(client_address)
        return self.lease


@pytest.fixture
def leases(monkeypatch: pytest.MonkeyPatch) -> AnyCallerLeases:
    cache = AnyCallerLeases()
    monkeypatch.setattr(lease_cache, "_LEASE_CACHE", cache)
    return cache


@pytest.fixture
def recorded(client: TestClient, leases: AnyCallerLeases) -> Iterator[CollectingRecorder]:
    previous = global_call_recorder()
    recorder = CollectingRecorder()
    set_global_call_recorder(recorder)
    yield recorder
    set_global_call_recorder(previous)


def _value(value: Any) -> Any:
    kind = value.WhichOneof("value")
    if kind == "array_value":
        return [item.string_value for item in value.array_value.values]
    return getattr(value, kind)


def _attributes(span: Span) -> dict[str, Any]:
    return {attribute.key: _value(attribute.value) for attribute in span.attributes}


def _only_span(recorded: CollectingRecorder) -> tuple[Span, dict[str, Any]]:
    [(workspace, batch)] = recorded.spans
    assert workspace == "evals"
    [span] = batch.scope_spans[0].spans
    return span, _attributes(span)


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


def _bare_span(name: str = "chat.completions") -> ResourceSpans:
    return resource_spans([Span(name=name, trace_id=b"\x01" * 16, span_id=b"\x02" * 8)])


def _names(spans: list[ResourceSpans]) -> list[str]:
    return [span.name for batch in spans for scope in batch.scope_spans for span in scope.spans]


async def test_queued_spans_are_sent_in_one_batch_per_workspace() -> None:
    sent: list[tuple[str, list[str]]] = []

    async def sink(workspace: str, spans: list[ResourceSpans]) -> None:
        sent.append((workspace, _names(spans)))

    recorder = CallRecorder(sink, flush_interval_sec=0.05)
    recorder.start()
    recorder.record("a", _bare_span("1"))
    recorder.record("b", _bare_span("2"))
    recorder.record("a", _bare_span("3"))
    await asyncio.sleep(0.2)
    await recorder.stop()

    assert sorted(sent) == [("a", ["1", "3"]), ("b", ["2"])]


async def test_a_batch_is_split_to_fit_intakes_body_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    from nhx.core.inference_gateway.api import call_recording

    sent: list[list[str]] = []

    async def sink(workspace: str, spans: list[ResourceSpans]) -> None:
        sent.append(_names(spans))

    monkeypatch.setattr(call_recording, "MAX_BATCH_BYTES", _bare_span("1").ByteSize() * 2)
    recorder = CallRecorder(sink, flush_interval_sec=60)
    for name in "12345":
        recorder.record("ws", _bare_span(name))
    await recorder.stop()

    assert sent == [["1", "2"], ["3", "4"], ["5"]]


async def test_an_unreachable_intake_never_blocks_recording_and_drops_its_batches() -> None:
    """Recording only enqueues, so an Intake that hangs or fails costs the call nothing."""
    calls = 0

    async def hanging_sink(workspace: str, spans: list[ResourceSpans]) -> None:
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

    async def flaky_sink(workspace: str, spans: list[ResourceSpans]) -> None:
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
    sent: list[ResourceSpans] = []

    async def sink(workspace: str, spans: list[ResourceSpans]) -> None:
        sent.extend(spans)

    recorder = CallRecorder(sink, flush_interval_sec=60)
    recorder.record("ws", _bare_span())
    await recorder.stop()

    assert len(sent) == 1


async def test_stop_is_bounded_even_when_the_sink_ignores_cancellation() -> None:
    release = asyncio.Event()

    async def stubborn_sink(workspace: str, spans: list[ResourceSpans]) -> None:
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


# --- through the proxy ------------------------------------------------------------------------


def test_a_leased_call_is_recorded_in_its_trial(
    client: TestClient, recorded: CollectingRecorder, leases: AnyCallerLeases, mock_proxy_response: Any
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
    assert leases.asked == ["testclient"]  # the socket peer, never X-Forwarded-For
    span, attributes = _only_span(recorded)
    assert span.trace_id.hex() == LEASE.trace_id
    assert span.parent_span_id.hex() == LEASE.root_span_id
    assert span.name == "chat.completions"
    assert span.status.code == Status.STATUS_CODE_OK
    assert span.start_time_unix_nano <= span.end_time_unix_nano
    assert attributes["gen_ai.conversation.id"] == LEASE.session_id
    assert attributes["nemo.evaluation.name"] == "run-1"
    assert attributes[GATEWAY_CALL_ATTRIBUTE] is True
    assert attributes["gen_ai.operation.name"] == "chat"
    assert attributes["client.address"] == "testclient"
    assert attributes["nemo.virtual_model"] == "e2e-test/meta_llama-3.2-1b-instruct"
    assert attributes["gen_ai.request.model"] == "e2e-test/meta_llama-3.2-1b-instruct"
    assert attributes["gen_ai.provider.name"] == "ollama"
    assert attributes["http.response.status_code"] == 200
    assert attributes["gen_ai.usage.input_tokens"] == 21
    assert attributes["nemo.usage.uncached_input_tokens"] == 5
    assert attributes["gen_ai.usage.cached_tokens"] == 16
    assert attributes["gen_ai.usage.output_tokens"] == 4
    assert attributes["nemo.agent.header.x-claude-code-session-id"] == "session-1"
    assert attributes["nemo.agent.header.x-codex-turn-metadata"] == '{"turn": 2}'


def test_a_call_from_an_unleased_address_is_not_recorded(
    client: TestClient, recorded: CollectingRecorder, leases: AnyCallerLeases
) -> None:
    leases.lease = None

    response = client.post(CHAT_URL, json=CHAT_BODY)

    assert response.status_code == 200
    assert recorded.spans == []


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
    span, attributes = _only_span(recorded)
    assert span.status.code == Status.STATUS_CODE_OK
    assert attributes["gen_ai.usage.input_tokens"] == 9
    assert attributes["gen_ai.usage.output_tokens"] == 2


def test_a_call_the_upstream_rejects_is_recorded_as_an_error(
    client: TestClient, recorded: CollectingRecorder, mock_proxy_response: Any
) -> None:
    mock_proxy_response.status = 500

    async def _error_body() -> bytes:
        return b'{"error": "boom"}'

    mock_proxy_response.read = _error_body

    response = client.post(CHAT_URL, json=CHAT_BODY)

    assert response.status_code == 500
    span, attributes = _only_span(recorded)
    assert span.status.code == Status.STATUS_CODE_ERROR
    assert attributes["http.response.status_code"] == 500
    assert attributes["gen_ai.request.model"] == "e2e-test/meta_llama-3.2-1b-instruct"


def test_a_full_recording_queue_does_not_change_the_call(client: TestClient, leases: AnyCallerLeases) -> None:
    async def never_called(workspace: str, spans: list[ResourceSpans]) -> None:
        raise AssertionError("the sender is not running")

    previous = global_call_recorder()
    recorder = CallRecorder(never_called, max_queued_spans=1)
    recorder.record("evals", _bare_span())
    set_global_call_recorder(recorder)
    try:
        response = client.post(CHAT_URL, json=CHAT_BODY)
    finally:
        set_global_call_recorder(previous)

    assert response.status_code == 200
    assert recorder.dropped == 1


# --- the observer -----------------------------------------------------------------------------


def _observer(recorder: CallRecorder, trailing_uri: str = "v1/responses") -> CallObserver:
    return CallObserver(
        recorder,
        AnyCallerLeases(),
        workspace="ws",
        virtual_model="vm",
        trailing_uri=trailing_uri,
        client_address="10.244.0.7",
        headers={},
    )


def test_plugin_attributes_are_recorded_but_cannot_replace_the_gateways_own() -> None:
    recorder = CollectingRecorder()
    observer = _observer(recorder)
    observer.telemetry_attributes = {
        "nemo-evals-closed-book.removed_tools": ["web_search"],
        "client.address": "6.6.6.6",
        "gen_ai.conversation.id": "someone-elses-session",
    }

    observer.finish(CallStatus.SUCCESS, 200)
    observer.finish(CallStatus.ERROR, 500)

    span, attributes = _only_span(recorder)
    assert attributes["nemo-evals-closed-book.removed_tools"] == ["web_search"]
    assert attributes["client.address"] == "10.244.0.7"
    assert attributes["gen_ai.conversation.id"] == LEASE.session_id
    assert span.status.code == Status.STATUS_CODE_OK


async def test_a_complete_error_body_is_recorded_as_an_error() -> None:
    recorder = CollectingRecorder()
    observer = _observer(recorder, "v1/chat/completions")

    async def body() -> AsyncIterator[bytes]:
        yield b'{"error": "boom"}'

    assert [part async for part in observer.follow_body(body(), 502)] == [b'{"error": "boom"}']

    span, _ = _only_span(recorder)
    assert span.status.code == Status.STATUS_CODE_ERROR


async def test_a_caller_that_disconnects_mid_stream_is_recorded_as_cancelled() -> None:
    recorder = CollectingRecorder()
    observer = _observer(recorder, "v1/messages")

    async def body() -> AsyncIterator[bytes]:
        for part in (b"one", b"two", b"three"):
            yield part

    followed = observer.follow_body(body(), 200)
    assert await anext(followed) == b"one"
    await followed.aclose()

    _, attributes = _only_span(recorder)
    assert attributes["status"] == "cancelled"
