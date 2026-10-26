# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record each VirtualModel call to Intake as one span, off the request path.

A :class:`CallObserver` follows one call: who sent it (the socket peer), what it resolved to, the
token usage the upstream reported, and when the caller's response finished. It hands the finished
span to the :class:`CallRecorder`, which queues it and sends batches to Intake in the background.
Nothing here can fail, delay or change the call itself: a full queue drops the span, and a failed
send is logged and dropped.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import defaultdict
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.intake.client import AsyncIntakeClient
from nemo_helix_plugin.intake.types import DirectSpanInput, DirectSpansIngestRequest
from pydantic import JsonValue

logger = logging.getLogger(__name__)

SPAN_SOURCE = "inference-gateway"
CLIENT_ADDRESS_ATTRIBUTE = "client.address"

SESSION_HEADERS = ("x-claude-code-session-id", "session_id", "session-id")
AGENT_HEADERS = (*SESSION_HEADERS, "x-codex-turn-metadata", "x-codex-window-id")
AGENT_HEADER_ATTRIBUTE_PREFIX = "nemo.agent.header."
# Intake caps span identifiers at this length; longer header values are cut to fit.
MAX_HEADER_VALUE_LENGTH = 1024

DEFAULT_MAX_QUEUED_SPANS = 10_000
DEFAULT_MAX_BATCH_SIZE = 500
DEFAULT_FLUSH_INTERVAL_SEC = 1.0
DEFAULT_SHUTDOWN_FLUSH_TIMEOUT_SEC = 5.0
FAILURE_LOG_INTERVAL_SEC = 60.0


class CallApi(StrEnum):
    """The inference API a call used, from its path."""

    CHAT_COMPLETIONS = "chat.completions"
    RESPONSES = "responses"
    MESSAGES = "messages"
    OTHER = "other"

    @classmethod
    def from_path(cls, trailing_uri: str) -> CallApi:
        path = trailing_uri.strip("/").removeprefix("v1/")
        if path == "chat/completions":
            return cls.CHAT_COMPLETIONS
        if path == "responses":
            return cls.RESPONSES
        if path == "messages":
            return cls.MESSAGES
        return cls.OTHER


@dataclass
class TokenUsage:
    """Token counts for one call, by class: every input token is exactly one of uncached, cache read, or cache write."""

    uncached_input_tokens: int | None = None
    cached_input_tokens: int | None = None
    cache_write_input_tokens: int | None = None
    output_tokens: int | None = None

    def merge(self, other: TokenUsage) -> None:
        """Take every count *other* reports."""
        for name in ("uncached_input_tokens", "cached_input_tokens", "cache_write_input_tokens", "output_tokens"):
            value = getattr(other, name)
            if value is not None:
                setattr(self, name, value)

    @property
    def input_tokens(self) -> int | None:
        if self.uncached_input_tokens is None:
            return None
        return self.uncached_input_tokens + (self.cached_input_tokens or 0) + (self.cache_write_input_tokens or 0)

    def attributes(self) -> dict[str, JsonValue]:
        counts = {
            "gen_ai.usage.input_tokens": self.input_tokens,
            "nemo.usage.uncached_input_tokens": self.uncached_input_tokens,
            "gen_ai.usage.cached_tokens": self.cached_input_tokens,
            "llm.token_count.prompt_details.cache_write": self.cache_write_input_tokens,
            "gen_ai.usage.output_tokens": self.output_tokens,
        }
        return {key: value for key, value in counts.items() if value is not None}


def _count(mapping: Any, *path: str) -> int | None:
    for key in path:
        if not isinstance(mapping, Mapping):
            return None
        mapping = mapping.get(key)
    if isinstance(mapping, int) and not isinstance(mapping, bool):
        return mapping
    return None


def _openai_usage(total_input: int | None, cached: int | None, output: int | None) -> TokenUsage:
    uncached = None if total_input is None else total_input - (cached or 0)
    return TokenUsage(uncached_input_tokens=uncached, cached_input_tokens=cached, output_tokens=output)


def _openai_chat_usage(usage: Mapping[str, Any]) -> TokenUsage:
    return _openai_usage(
        _count(usage, "prompt_tokens"),
        _count(usage, "prompt_tokens_details", "cached_tokens"),
        _count(usage, "completion_tokens"),
    )


def _openai_responses_usage(usage: Mapping[str, Any]) -> TokenUsage:
    return _openai_usage(
        _count(usage, "input_tokens"),
        _count(usage, "input_tokens_details", "cached_tokens"),
        _count(usage, "output_tokens"),
    )


def _anthropic_usage(usage: Mapping[str, Any]) -> TokenUsage:
    return TokenUsage(
        uncached_input_tokens=_count(usage, "input_tokens"),
        cached_input_tokens=_count(usage, "cache_read_input_tokens"),
        cache_write_input_tokens=_count(usage, "cache_creation_input_tokens"),
        output_tokens=_count(usage, "output_tokens"),
    )


def usage_from_payload(api: CallApi, payload: Mapping[str, Any]) -> TokenUsage | None:
    """The usage a response body or stream chunk reports, or ``None`` if it reports none."""
    if api is CallApi.MESSAGES:
        usage = payload.get("usage")
        if payload.get("type") == "message_start" and isinstance(payload.get("message"), Mapping):
            usage = payload["message"].get("usage")
        return _anthropic_usage(usage) if isinstance(usage, Mapping) else None
    if api is CallApi.RESPONSES:
        usage = payload.get("usage")
        if isinstance(payload.get("response"), Mapping):
            usage = payload["response"].get("usage")
        return _openai_responses_usage(usage) if isinstance(usage, Mapping) else None
    usage = payload.get("usage")
    return _openai_chat_usage(usage) if isinstance(usage, Mapping) else None


SpanSink = Callable[[str, list[DirectSpanInput]], Awaitable[None]]


class CallRecorder:
    """Queues finished call spans and sends them to Intake in per-workspace batches."""

    def __init__(
        self,
        sink: SpanSink,
        *,
        max_queued_spans: int = DEFAULT_MAX_QUEUED_SPANS,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
        flush_interval_sec: float = DEFAULT_FLUSH_INTERVAL_SEC,
    ) -> None:
        self._sink = sink
        self._queue: asyncio.Queue[tuple[str, DirectSpanInput]] = asyncio.Queue(maxsize=max_queued_spans)
        self._max_batch_size = max_batch_size
        self._flush_interval_sec = flush_interval_sec
        self._task: asyncio.Task[None] | None = None
        self._stopping = False
        self._dropped = 0
        self._last_failure_log = float("-inf")

    @property
    def dropped(self) -> int:
        """Spans dropped so far because the queue was full or Intake rejected their batch."""
        return self._dropped

    def record(self, workspace: str, span: DirectSpanInput) -> None:
        """Queue *span* for *workspace*, dropping it if the queue is full."""
        try:
            self._queue.put_nowait((workspace, span))
        except asyncio.QueueFull:
            self._dropped += 1
            self._log_failure("Inference Gateway call recording queue is full; dropping call spans")

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="igw-call-recorder")

    async def stop(self, timeout_sec: float = DEFAULT_SHUTDOWN_FLUSH_TIMEOUT_SEC) -> None:
        """Stop the sender, then send what is still queued; the whole stop takes at most *timeout_sec*."""
        deadline = time.monotonic() + timeout_sec
        self._stopping = True
        if self._task is not None:
            task, self._task = self._task, None
            task.cancel()
            await asyncio.wait({task}, timeout=timeout_sec)
        try:
            await asyncio.wait_for(self.flush(), timeout=max(0.0, deadline - time.monotonic()))
        except TimeoutError:
            logger.warning("Inference Gateway dropped %d queued call spans at shutdown", self._queue.qsize())

    async def flush(self) -> None:
        """Send everything queued now."""
        while not self._queue.empty():
            await self._send(self._take_batch())

    async def _run(self) -> None:
        while not self._stopping:
            try:
                await self._send_next_batch()
            except Exception:
                logger.exception("Inference Gateway call recording failed; continuing")

    async def _send_next_batch(self) -> None:
        first = await self._queue.get()
        deadline = time.monotonic() + self._flush_interval_sec
        batch = [first]
        while len(batch) < self._max_batch_size and (remaining := deadline - time.monotonic()) > 0:
            try:
                batch.append(await asyncio.wait_for(self._queue.get(), timeout=remaining))
            except TimeoutError:
                break
        await self._send(batch)

    def _take_batch(self) -> list[tuple[str, DirectSpanInput]]:
        batch: list[tuple[str, DirectSpanInput]] = []
        while len(batch) < self._max_batch_size and not self._queue.empty():
            batch.append(self._queue.get_nowait())
        return batch

    async def _send(self, batch: list[tuple[str, DirectSpanInput]]) -> None:
        by_workspace: dict[str, list[DirectSpanInput]] = defaultdict(list)
        for workspace, span in batch:
            by_workspace[workspace].append(span)
        for workspace, spans in by_workspace.items():
            try:
                await self._sink(workspace, spans)
            except Exception:
                self._dropped += len(spans)
                self._log_failure(
                    f"Inference Gateway could not send {len(spans)} call spans to Intake for workspace {workspace!r}",
                    exc_info=True,
                )

    def _log_failure(self, message: str, *, exc_info: bool = False) -> None:
        now = time.monotonic()
        if now - self._last_failure_log >= FAILURE_LOG_INTERVAL_SEC:
            self._last_failure_log = now
            logger.warning("%s (%d dropped so far)", message, self._dropped, exc_info=exc_info)


def intake_sink(client: AsyncNemoClient) -> SpanSink:
    """A sink that sends each workspace's spans to Intake with *client*."""
    intake: AsyncIntakeClient | None = None

    async def sink(workspace: str, spans: list[DirectSpanInput]) -> None:
        nonlocal intake
        if intake is None:
            intake = AsyncIntakeClient.from_client(client)
        await intake.create_spans(workspace=workspace, body=DirectSpansIngestRequest(source=SPAN_SOURCE, spans=spans))

    return sink


class CallStatus(StrEnum):
    SUCCESS = "success"
    ERROR = "error"
    CANCELLED = "cancelled"


class CallObserver:
    """Follows one VirtualModel call and records it once, when it finishes."""

    def __init__(
        self,
        recorder: CallRecorder,
        *,
        workspace: str,
        virtual_model: str,
        trailing_uri: str,
        client_address: str | None,
        headers: Mapping[str, str],
    ) -> None:
        self._recorder = recorder
        self.workspace = workspace
        self.virtual_model = virtual_model
        self.api = CallApi.from_path(trailing_uri)
        self.client_address = client_address
        self.started_at = datetime.now(UTC)
        self.model: str | None = None
        self.provider: str | None = None
        self.usage = TokenUsage()
        self.telemetry_attributes: Mapping[str, Any] = {}
        self._agent_headers = {
            name: value[:MAX_HEADER_VALUE_LENGTH] for name in AGENT_HEADERS if (value := headers.get(name))
        }
        self._finished = False

    def observe(self, payload: Any) -> None:
        if isinstance(payload, Mapping) and (usage := usage_from_payload(self.api, payload)) is not None:
            self.usage.merge(usage)

    async def observe_stream(self, chunks: AsyncIterator[Any]) -> AsyncIterator[Any]:
        async for chunk in chunks:
            self.observe(chunk)
            yield chunk

    async def follow_body(self, body: AsyncIterator[Any], status_code: int) -> AsyncIterator[Any]:
        """Pass the response body through, recording the call when the caller has all of it."""
        status = CallStatus.ERROR
        try:
            async for part in body:
                yield part
            if status_code < 400:
                status = CallStatus.SUCCESS
        except (asyncio.CancelledError, GeneratorExit):
            status = CallStatus.CANCELLED
            raise
        finally:
            self.finish(status, status_code)

    def finish(self, status: CallStatus, status_code: int) -> None:
        if self._finished:
            return
        self._finished = True
        try:
            self._recorder.record(self.workspace, self._span(status, status_code))
        except Exception:
            logger.warning("Inference Gateway could not build the call span", exc_info=True)

    def _span(self, status: CallStatus, status_code: int) -> DirectSpanInput:
        attributes: dict[str, JsonValue] = {
            **self.telemetry_attributes,
            "nemo.virtual_model": f"{self.workspace}/{self.virtual_model}",
            "nemo.inference.api": self.api.value,
            "http.response.status_code": status_code,
            **{f"{AGENT_HEADER_ATTRIBUTE_PREFIX}{name}": value for name, value in self._agent_headers.items()},
            **self.usage.attributes(),
        }
        if self.client_address:
            attributes[CLIENT_ADDRESS_ATTRIBUTE] = self.client_address
        if self.model:
            attributes["gen_ai.request.model"] = self.model
        if self.provider:
            attributes["gen_ai.provider.name"] = self.provider
        trace_id = uuid.uuid4().hex
        session_id = next((self._agent_headers[name] for name in SESSION_HEADERS if name in self._agent_headers), None)
        return DirectSpanInput(
            span_id=uuid.uuid4().hex[:16],
            trace_id=trace_id,
            session_id=session_id,
            name=self.api.value,
            kind="LLM",
            status=status.value,
            started_at=self.started_at,
            ended_at=datetime.now(UTC),
            attributes=attributes,
        )


_CALL_RECORDER: CallRecorder | None = None


def set_global_call_recorder(recorder: CallRecorder | None) -> CallRecorder | None:
    global _CALL_RECORDER
    _CALL_RECORDER = recorder
    return _CALL_RECORDER


def global_call_recorder() -> CallRecorder | None:
    """The running recorder, or ``None`` when call recording is off."""
    return _CALL_RECORDER
