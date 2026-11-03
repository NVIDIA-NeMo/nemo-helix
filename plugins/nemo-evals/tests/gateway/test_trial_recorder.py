# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Finished trial calls reach Intake in batches, off the request path, and failures only drop calls."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest
from nemo_evals.gateway import recorder as recorder_module
from nemo_evals.gateway.policy import ModelAccess
from nemo_evals.gateway.recorder import CallRecorder, FinishedCall
from nemo_evals.gateway.tokens import TrialClaims, TrialTokenReader, unsigned_token
from nemo_helix_plugin.client.oidc import decode_jwt_claims
from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans, Span

NOW = datetime.now(UTC)
TRIAL = TrialClaims(
    session="run-1:hello__abc",
    intake_workspace="evals",
    evaluation="run-1",
    task="hello",
    model_access=ModelAccess.OPEN,
)
TOKEN = unsigned_token(TRIAL)


def _spans(batches: list[tuple[str, list[ResourceSpans]]]) -> list[tuple[str, Span]]:
    return [
        (workspace, span)
        for workspace, batch in batches
        for resource in batch
        for scope in resource.scope_spans
        for span in scope.spans
    ]


class Signed:
    """Stands in for the platform's signature check: trusts the claims of exactly the tokens it was given."""

    def __init__(self, *tokens: str) -> None:
        self._tokens = set(tokens)
        self.asked = 0

    async def __call__(self, token: str) -> Mapping[str, Any] | None:
        self.asked += 1
        return decode_jwt_claims(token) if token in self._tokens else None


class Sink:
    def __init__(self) -> None:
        self.sent: list[tuple[str, list[ResourceSpans]]] = []

    async def __call__(self, workspace: str, spans: list[ResourceSpans]) -> int:
        self.sent.append((workspace, spans))
        return 0


def _reader(*tokens: str) -> TrialTokenReader:
    return TrialTokenReader(Signed(*tokens))


def _finished(token: str = TOKEN, name: str = "chat.completions") -> FinishedCall:
    return FinishedCall(token=token, name=name, started_at=NOW, ended_at=NOW, cancelled=False)


async def test_a_token_that_cant_be_checked_drops_the_call_and_keeps_recording() -> None:
    failed = False

    async def flaky(token: str) -> Mapping[str, Any] | None:
        nonlocal failed
        if not failed:
            failed = True
            raise RuntimeError("the platform's signing keys are unreachable")
        return decode_jwt_claims(token)

    sink = Sink()
    recorder = CallRecorder(TrialTokenReader(flaky), sink, flush_interval_sec=60)
    recorder.record(_finished(name="1"))
    recorder.record(_finished(name="2"))
    await recorder.flush()

    assert [span.name for _, span in _spans(sink.sent)] == ["2"]
    assert recorder.dropped == 1


async def test_queued_calls_are_sent_in_one_batch_per_intake_workspace() -> None:
    other = unsigned_token(TRIAL.model_copy(update={"intake_workspace": "other"}))
    sink = Sink()
    recorder = CallRecorder(_reader(TOKEN, other), sink, flush_interval_sec=0.05)
    recorder.start()
    recorder.record(_finished(name="1"))
    recorder.record(_finished(token=other, name="2"))
    recorder.record(_finished(name="3"))
    await asyncio.sleep(0.2)
    await recorder.stop()

    sent = sorted(
        (workspace, [span.name for _, span in _spans([(workspace, batch)])]) for workspace, batch in sink.sent
    )
    assert sent == [("evals", ["1", "3"]), ("other", ["2"])]


async def test_stopping_sends_the_calls_the_sender_was_still_collecting() -> None:
    """The sender holds a batch for up to the flush interval; shutting the gateway down mustn't lose it."""
    sink = Sink()
    recorder = CallRecorder(_reader(TOKEN), sink, flush_interval_sec=60)
    recorder.start()
    recorder.record(_finished(name="1"))
    recorder.record(_finished(name="2"))
    while recorder._queue.qsize():
        await asyncio.sleep(0.01)

    await recorder.stop()

    assert sorted(span.name for _, span in _spans(sink.sent)) == ["1", "2"]


async def test_stopping_lets_a_batch_already_being_sent_finish() -> None:
    sent = asyncio.Event()
    entered = asyncio.Event()
    delivered: list[str] = []

    async def slow_sink(workspace: str, spans: list[ResourceSpans]) -> int:
        entered.set()
        await asyncio.sleep(0.05)
        delivered.extend(span.scope_spans[0].spans[0].name for span in spans)
        sent.set()
        return 0

    recorder = CallRecorder(_reader(TOKEN), slow_sink, flush_interval_sec=0.01)
    recorder.start()
    recorder.record(_finished(name="1"))
    await entered.wait()

    await recorder.stop()

    assert delivered == ["1"]
    assert recorder.dropped == 0


async def test_a_batch_that_cant_be_sent_before_shutdown_counts_as_dropped() -> None:
    entered = asyncio.Event()

    async def hanging_sink(workspace: str, spans: list[ResourceSpans]) -> int:
        entered.set()
        await asyncio.Event().wait()
        return 0

    recorder = CallRecorder(_reader(TOKEN), hanging_sink, flush_interval_sec=0.01)
    recorder.start()
    recorder.record(_finished())
    recorder.record(_finished())
    await entered.wait()

    await recorder.stop(timeout_sec=0.05)

    assert recorder.dropped == 2


async def test_calls_the_shutdown_flush_cant_send_count_as_dropped() -> None:
    async def hanging_sink(workspace: str, spans: list[ResourceSpans]) -> int:
        await asyncio.Event().wait()
        return 0

    recorder = CallRecorder(_reader(TOKEN), hanging_sink, max_batch_size=2, flush_interval_sec=60)
    for _ in range(3):
        recorder.record(_finished())

    await recorder.stop(timeout_sec=0.05)

    assert recorder.dropped == 3


async def test_a_batch_is_split_to_fit_intakes_body_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    sink = Sink()
    recorder = CallRecorder(_reader(TOKEN), sink, flush_interval_sec=60)
    for name in "12345":
        recorder.record(_finished(name=name))
    one_span = await recorder._span(_finished(name="1"))
    assert one_span is not None
    monkeypatch.setattr(recorder_module, "MAX_BATCH_BYTES", one_span[1].ByteSize() * 2)
    await recorder.stop()

    assert [[span.name for _, span in _spans([batch])] for batch in sink.sent] == [["1", "2"], ["3", "4"], ["5"]]


async def test_an_unreachable_intake_never_blocks_recording_and_drops_its_batches() -> None:
    """Recording only enqueues, so an Intake that hangs or fails costs the call nothing."""
    calls = 0

    async def hanging_sink(workspace: str, spans: list[ResourceSpans]) -> int:
        nonlocal calls
        calls += 1
        await asyncio.Event().wait()
        return 0

    recorder = CallRecorder(_reader(TOKEN), hanging_sink, max_queued_calls=2, flush_interval_sec=0.01)
    recorder.start()
    recorder.record(_finished())
    await asyncio.sleep(0.05)  # the sender is now stuck on that call's batch

    started = time.monotonic()
    for _ in range(5):
        recorder.record(_finished())
    assert time.monotonic() - started < 0.01
    assert recorder.dropped == 3  # two fit in the queue

    await recorder.stop(timeout_sec=0.05)
    assert calls >= 1


async def test_a_failing_intake_drops_the_batch_and_keeps_sending() -> None:
    sent: list[int] = []

    async def flaky_sink(workspace: str, spans: list[ResourceSpans]) -> int:
        if not sent:
            sent.append(0)
            raise RuntimeError("Intake is down")
        sent.append(len(spans))
        return 0

    recorder = CallRecorder(_reader(TOKEN), flaky_sink, flush_interval_sec=0.01)
    recorder.start()
    recorder.record(_finished())
    await asyncio.sleep(0.05)
    recorder.record(_finished())
    await asyncio.sleep(0.05)
    await recorder.stop()

    assert sent == [0, 1]
    assert recorder.dropped == 1


async def test_spans_intake_rejects_are_counted_as_dropped() -> None:
    async def picky_sink(workspace: str, spans: list[ResourceSpans]) -> int:
        return 1

    recorder = CallRecorder(_reader(TOKEN), picky_sink, flush_interval_sec=60)
    recorder.record(_finished())
    recorder.record(_finished())
    await recorder.flush()

    assert recorder.dropped == 1


async def test_stop_is_bounded_even_when_the_sink_ignores_cancellation() -> None:
    release = asyncio.Event()

    async def stubborn_sink(workspace: str, spans: list[ResourceSpans]) -> int:
        while not release.is_set():
            try:
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                continue
        return 0

    recorder = CallRecorder(_reader(TOKEN), stubborn_sink, flush_interval_sec=0.01)
    recorder.start()
    recorder.record(_finished())
    await asyncio.sleep(0.05)

    started = time.monotonic()
    await recorder.stop(timeout_sec=0.1)
    assert time.monotonic() - started < 0.5
    release.set()
