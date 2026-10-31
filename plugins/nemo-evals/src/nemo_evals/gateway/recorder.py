# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Delivers finished trial calls to Intake off the request path, in batches per Intake workspace.

Nothing here fails or delays a call: a full queue drops the call, and a failed check or send is logged and dropped.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime

from nemo_evals.gateway.spans import CallerSpan, call_span, export_request
from nemo_evals.gateway.tokens import TrialTokenReader
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.intake.client import AsyncIntakeClient
from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans

logger = logging.getLogger(__name__)

DEFAULT_MAX_QUEUED_CALLS = 10_000
DEFAULT_MAX_BATCH_SIZE = 500
# Intake refuses OTLP bodies over 5 MiB.
MAX_BATCH_BYTES = 4 * 1024 * 1024
DEFAULT_FLUSH_INTERVAL_SEC = 1.0
DEFAULT_SHUTDOWN_FLUSH_TIMEOUT_SEC = 5.0
FAILURE_LOG_INTERVAL_SEC = 60.0


@dataclass(frozen=True)
class FinishedCall:
    """A call the gateway served, waiting to be attributed to the trial its token names."""

    token: str
    name: str
    started_at: datetime
    ended_at: datetime
    cancelled: bool
    attributes: Mapping[str, object] = field(default_factory=dict)
    caller: CallerSpan | None = None
    error: bool = False


SpanSink = Callable[[str, list[ResourceSpans]], Awaitable[int]]
"""Sends a workspace's spans to Intake and returns how many Intake rejected."""


class CallRecorder:
    """Queues finished calls; in the background, attributes them to trials and sends them to Intake in batches."""

    def __init__(
        self,
        trials: TrialTokenReader,
        sink: SpanSink,
        *,
        max_queued_calls: int = DEFAULT_MAX_QUEUED_CALLS,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
        flush_interval_sec: float = DEFAULT_FLUSH_INTERVAL_SEC,
    ) -> None:
        self._trials = trials
        self._sink = sink
        self._queue: asyncio.Queue[FinishedCall] = asyncio.Queue(maxsize=max_queued_calls)
        self._collecting: list[FinishedCall] = []
        self._sending: list[FinishedCall] = []
        self._flushing: list[FinishedCall] = []
        self._max_batch_size = max_batch_size
        self._flush_interval_sec = flush_interval_sec
        self._task: asyncio.Task[None] | None = None
        self._stopping = False
        self._dropped = 0
        self._last_failure_log = float("-inf")

    @property
    def dropped(self) -> int:
        """Calls dropped so far because the queue was full, a token couldn't be checked, or Intake rejected their batch."""
        return self._dropped

    def record(self, call: FinishedCall) -> None:
        """Queue *call*, dropping it if the queue is full."""
        try:
            self._queue.put_nowait(call)
        except asyncio.QueueFull:
            self._dropped += 1
            self._log_failure("Trial call recording queue is full; dropping calls")

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="nemo-evals-trial-recorder")

    async def stop(self, timeout_sec: float = DEFAULT_SHUTDOWN_FLUSH_TIMEOUT_SEC) -> None:
        """Stop the sender, then send what is still queued; the whole stop takes at most *timeout_sec*.

        A batch the sender is already delivering gets the time to finish; one it can't finish counts as dropped.
        """
        deadline = time.monotonic() + timeout_sec
        self._stopping = True
        if self._task is not None:
            task, self._task = self._task, None
            if not self._sending:
                task.cancel()
            await asyncio.wait({task}, timeout=timeout_sec)
            if not task.done():
                self._dropped += len(self._sending)
                logger.warning("Dropped %d trial calls being sent at shutdown", len(self._sending))
                task.cancel()
        try:
            await asyncio.wait_for(self.flush(), timeout=max(0.0, deadline - time.monotonic()))
        except TimeoutError:
            lost = len(self._flushing) + self._queue.qsize()
            self._dropped += lost
            self._flushing = []
            logger.warning("Dropped %d queued trial calls at shutdown", lost)

    async def flush(self) -> None:
        """Send everything queued now, starting with any batch the sender was still collecting."""
        if self._collecting:
            self._flushing, self._collecting = self._collecting, []
            await self._send(self._flushing)
        while not self._queue.empty():
            self._flushing = self._take_batch()
            await self._send(self._flushing)
        self._flushing = []

    async def _run(self) -> None:
        while not self._stopping:
            try:
                await self._send_next_batch()
            except Exception:
                logger.exception("Trial call recording failed; continuing")

    async def _send_next_batch(self) -> None:
        first = await self._queue.get()
        deadline = time.monotonic() + self._flush_interval_sec
        self._collecting = [first]
        while len(self._collecting) < self._max_batch_size and (remaining := deadline - time.monotonic()) > 0:
            try:
                self._collecting.append(await asyncio.wait_for(self._queue.get(), timeout=remaining))
            except TimeoutError:
                break
        self._sending, self._collecting = self._collecting, []
        try:
            await self._send(self._sending)
        finally:
            self._sending = []

    def _take_batch(self) -> list[FinishedCall]:
        batch: list[FinishedCall] = []
        while len(batch) < self._max_batch_size and not self._queue.empty():
            batch.append(self._queue.get_nowait())
        return batch

    async def _span(self, call: FinishedCall) -> tuple[str, ResourceSpans] | None:
        try:
            trial = await self._trials.trial(call.token)
        except Exception:
            self._dropped += 1
            self._log_failure("Could not check a caller's trial token", exc_info=True)
            return None
        if trial is None:
            return None
        attributes = {**call.attributes, "status": "cancelled"} if call.cancelled else call.attributes
        span = call_span(
            trial,
            name=call.name,
            started_at=call.started_at,
            ended_at=call.ended_at,
            error=call.error,
            attributes=attributes,
            caller=call.caller,
        )
        return trial.intake_workspace, span

    async def _send(self, batch: list[FinishedCall]) -> None:
        by_workspace: dict[str, list[ResourceSpans]] = defaultdict(list)
        for call in batch:
            if (attributed := await self._span(call)) is not None:
                by_workspace[attributed[0]].append(attributed[1])
        for workspace, spans in by_workspace.items():
            for chunk in _chunks_under(spans, MAX_BATCH_BYTES):
                try:
                    rejected = await self._sink(workspace, chunk)
                except Exception:
                    self._dropped += len(chunk)
                    self._log_failure(
                        f"Could not send {len(chunk)} trial call spans to Intake for workspace {workspace!r}",
                        exc_info=True,
                    )
                    continue
                if rejected:
                    self._dropped += rejected
                    self._log_failure(f"Intake rejected {rejected} trial call spans for workspace {workspace!r}")

    def _log_failure(self, message: str, *, exc_info: bool = False) -> None:
        now = time.monotonic()
        if now - self._last_failure_log >= FAILURE_LOG_INTERVAL_SEC:
            self._last_failure_log = now
            logger.warning("%s (%d dropped so far)", message, self._dropped, exc_info=exc_info)


def _chunks_under(spans: list[ResourceSpans], max_bytes: int) -> list[list[ResourceSpans]]:
    chunks: list[list[ResourceSpans]] = [[]]
    size = 0
    for span in spans:
        span_size = span.ByteSize()
        if chunks[-1] and size + span_size > max_bytes:
            chunks.append([])
            size = 0
        chunks[-1].append(span)
        size += span_size
    return chunks


def intake_sink(client: AsyncNemoClient) -> SpanSink:
    """Send each workspace's spans to Intake's OTLP route with *client*."""
    intake = AsyncIntakeClient.from_client(client)

    async def sink(workspace: str, spans: list[ResourceSpans]) -> int:
        response = (await intake.create_otlp_traces(workspace=workspace, content=export_request(spans))).data()
        if response.errors:
            logger.debug("Intake rejected trial call spans for workspace %r: %s", workspace, response.errors[:5])
        return len(response.errors)

    return sink
