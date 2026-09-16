# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Shared telemetry handler for NeMo products.

Environment variables:
- NEMO_TELEMETRY_ENABLED: Whether telemetry is enabled.
- NEMO_DEPLOYMENT_TYPE: The deployment type the event came from.
- NEMO_TELEMETRY_ENDPOINT: The endpoint to send the telemetry events to.
- NEMO_SESSION_PREFIX: Optional prefix to add to session IDs.
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import threading
from collections.abc import Coroutine
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import SplitResult, urlsplit, urlunsplit

import httpx
from nemo_platform_plugin.telemetry.events import PlatformTelemetryEvent

CLIENT_ID = "184482118588404"
NEMO_TELEMETRY_VERSION = "nemo-telemetry/1.0"
DEFAULT_ENDPOINT = "https://events.telemetry.data.nvidia.com/v1.1/events/json"
MAX_RETRIES = 3
# Tight explicit timeout so a hung or black-holed endpoint can never block a command
# at exit for longer than this. httpx's default is ~5s, which is too long for a
# best-effort flush that runs synchronously on the command's exit path.
SEND_TIMEOUT_SECONDS = 2.0
logger = logging.getLogger(__name__)


def _telemetry_enabled() -> bool:
    value = os.getenv("NEMO_TELEMETRY_ENABLED")
    if value is None:
        return True
    return value.strip().lower() == "true"


def _cpu_architecture() -> str:
    return platform.machine() or "undefined"


def _telemetry_endpoint() -> str | None:
    endpoint = os.getenv("NEMO_TELEMETRY_ENDPOINT", DEFAULT_ENDPOINT)
    try:
        parsed = urlsplit(endpoint)
    except ValueError:
        logger.debug("Skipping telemetry because endpoint is invalid: %s", _redact_endpoint(endpoint))
        return None
    if parsed.scheme.lower() != "https" or not parsed.netloc:
        logger.debug("Skipping telemetry because endpoint is not HTTPS: %s", _redact_endpoint(endpoint))
        return None
    return endpoint


def _redacted_netloc(parsed: SplitResult) -> str:
    return parsed.netloc.rsplit("@", 1)[-1]


def _redact_endpoint(endpoint: str) -> str:
    """Redact credentials and query parameters before logging telemetry endpoints."""
    try:
        parsed = urlsplit(endpoint)
    except ValueError:
        return "<invalid-endpoint>"
    query = "<redacted>" if parsed.query else ""
    return urlunsplit((parsed.scheme, _redacted_netloc(parsed), parsed.path, query, parsed.fragment))


def _session_prefix() -> str | None:
    return os.getenv("NEMO_SESSION_PREFIX")


@dataclass
class QueuedEvent:
    event: PlatformTelemetryEvent
    timestamp: datetime
    retry_count: int = 0


def _get_iso_timestamp(dt: datetime | None = None) -> str:
    if dt is None:
        dt = datetime.now(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def build_payload(
    events: list[QueuedEvent], *, source_client_version: str, session_id: str = "undefined"
) -> dict[str, Any]:
    if not events:
        raise ValueError("build_payload requires at least one event")
    return {
        "browserType": "undefined",
        "clientId": CLIENT_ID,
        "clientType": "Native",
        "clientVariant": "Release",
        "clientVer": source_client_version,
        "cpuArchitecture": _cpu_architecture(),
        "deviceGdprBehOptIn": "None",
        "deviceGdprFuncOptIn": "None",
        "deviceGdprTechOptIn": "None",
        "deviceId": "undefined",
        "deviceMake": "undefined",
        "deviceModel": "undefined",
        "deviceOS": "undefined",
        "deviceOSVersion": "undefined",
        "deviceType": "undefined",
        "eventProtocol": "1.6",
        "eventSchemaVer": events[0].event._schema_version,
        "eventSysVer": NEMO_TELEMETRY_VERSION,
        "externalUserId": "undefined",
        "gdprBehOptIn": "None",
        "gdprFuncOptIn": "None",
        "gdprTechOptIn": "None",
        "idpId": "undefined",
        "integrationId": "undefined",
        "productName": "undefined",
        "productVersion": "undefined",
        "sentTs": _get_iso_timestamp(),
        "sessionId": session_id,
        "userId": "undefined",
        "events": [
            {
                "ts": _get_iso_timestamp(queued.timestamp),
                "parameters": queued.event.model_dump(by_alias=True, mode="json"),
                "name": queued.event._event_name,
            }
            for queued in events
        ],
    }


class TelemetryHandler:
    """
    Handles telemetry event batching, flushing, and retry logic for NeMo products.

    Supports two usage patterns:

    - **Background mode**: call ``start()`` (or use ``with handler:``) to spawn
      a daemon thread with its own event loop that drives periodic flushing.
      ``stop()`` schedules a final flush, then stops the loop and joins the thread.
    - **Fire-and-flush mode**: skip ``start()``, ``enqueue()`` events, then call
      ``stop()`` to flush once. No background thread is created unless the caller
      already has a running event loop, in which case the one-shot flush is
      offloaded to a worker thread with its own loop.

    Args:
        flush_interval_seconds (float): The interval in seconds to flush the events.
        max_queue_size (int): The maximum number of events to queue before flushing.
        max_retries (int): The maximum number of times to retry sending an event.
        source_client_version (str): The version of the source client. This should be the version of
            the actual NeMo product that is sending the events, typically the same as the version of
            a PyPi package that a user would install.
        session_id (str): An optional session ID to associate with the events.
            This should be a unique identifier for the session, such as a UUID.
            It is used to group events together.
    """

    def __init__(
        self,
        flush_interval_seconds: float = 120.0,
        max_queue_size: int = 50,
        max_retries: int = MAX_RETRIES,
        source_client_version: str = "undefined",
        session_id: str = "undefined",
    ):
        self._flush_interval = flush_interval_seconds
        self._max_queue_size = max_queue_size
        self._max_retries = max_retries
        self._events: list[QueuedEvent] = []
        self._dlq: list[QueuedEvent] = []
        self._queue_lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._flush_signal: asyncio.Event | None = None
        self._timer_task: asyncio.Task | None = None
        self._running = False
        self._source_client_version = source_client_version
        prefix = _session_prefix()
        self._session_id = f"{prefix}{session_id}" if prefix else session_id

    # -- Async API -----------------------------------------------------------

    async def astart(self) -> None:
        """Start the background timer task on the current event loop."""
        if self._running:
            return
        self._loop = asyncio.get_running_loop()
        self._flush_signal = asyncio.Event()
        self._running = True
        self._timer_task = asyncio.create_task(self._timer_loop())

    async def astop(self) -> None:
        """Cancel the timer task and flush any remaining events."""
        if not self._running:
            await self._flush_events()
            return
        self._running = False
        if self._flush_signal is not None:
            self._flush_signal.set()
        if self._timer_task is not None:
            self._timer_task.cancel()
            try:
                await self._timer_task
            except asyncio.CancelledError:
                pass
            self._timer_task = None
        await self._flush_events()
        self._loop = None
        self._flush_signal = None

    async def aflush(self) -> None:
        """Flush all queued events immediately and await completion."""
        await self._flush_events()

    # -- Sync API ------------------------------------------------------------

    def start(self) -> None:
        """Spawn a daemon thread with a persistent event loop for periodic flushing."""
        if self._running:
            return
        ready = threading.Event()
        startup_error: list[BaseException] = []

        def _run() -> None:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self._loop = loop
                self._flush_signal = asyncio.Event()
                self._timer_task = loop.create_task(self._timer_loop())
                self._running = True
            except BaseException as exc:  # noqa: BLE001
                startup_error.append(exc)
                ready.set()
                return
            ready.set()
            try:
                loop.run_forever()
            finally:
                loop.close()

        self._thread = threading.Thread(target=_run, name="nemo-telemetry", daemon=True)
        self._thread.start()
        ready.wait()
        if startup_error:
            self._thread = None
            raise startup_error[0]

    def stop(self) -> None:
        """Flush pending events. If a background thread is running, shut it down and join."""
        if self._running and self._loop is not None and self._thread is not None:
            loop = self._loop
            future = asyncio.run_coroutine_threadsafe(self._astop_inner(), loop)
            try:
                future.result(timeout=30)
            except Exception:  # noqa: BLE001
                pass
            loop.call_soon_threadsafe(loop.stop)
            self._thread.join(timeout=5)
            self._thread = None
            self._loop = None
            self._flush_signal = None
            self._timer_task = None
            self._running = False
            return
        if self._events or self._dlq:
            try:
                self._run_sync(self._flush_events())
            except Exception:  # noqa: BLE001
                logger.debug("Telemetry stop flush failed", exc_info=True)

    def flush(self) -> None:
        """Flush all queued events immediately and wait for completion."""
        if self._running and self._loop is not None and self._thread is not None:
            future: Future[None] = asyncio.run_coroutine_threadsafe(self._flush_events(), self._loop)
            try:
                future.result(timeout=30)
            except Exception:  # noqa: BLE001
                pass
            return
        if self._events or self._dlq:
            try:
                self._run_sync(self._flush_events())
            except Exception:  # noqa: BLE001
                logger.debug("Telemetry flush failed", exc_info=True)

    @staticmethod
    def _run_sync(coro: Coroutine[Any, Any, Any]) -> Any:
        """Run a coroutine synchronously from sync or async caller contexts.

        ``asyncio.run`` raises when called from a thread that already has a
        running event loop, such as a notebook kernel or an async SDK caller. In
        that case, run the coroutine in a worker thread so telemetry still gets
        a fresh event loop while remaining synchronous to the caller.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result(timeout=30)
        return asyncio.run(coro)

    async def _astop_inner(self) -> None:
        """Async shutdown body run on the background loop."""
        self._running = False
        if self._flush_signal is not None:
            self._flush_signal.set()
        if self._timer_task is not None:
            self._timer_task.cancel()
            try:
                await self._timer_task
            except asyncio.CancelledError:
                pass
            self._timer_task = None
        await self._flush_events()

    # -- Enqueue / signalling ------------------------------------------------

    def enqueue(self, event: object) -> None:
        if not _telemetry_enabled():
            return
        if not isinstance(event, PlatformTelemetryEvent):
            return
        queued = QueuedEvent(event=event, timestamp=datetime.now(timezone.utc))
        with self._queue_lock:
            self._events.append(queued)
            should_signal = len(self._events) >= self._max_queue_size
        if should_signal:
            self._signal_flush()

    def _signal_flush(self) -> None:
        """Set the flush signal, threadsafe across the background-loop boundary."""
        loop = self._loop
        signal = self._flush_signal
        if loop is None or signal is None:
            return
        try:
            loop.call_soon_threadsafe(signal.set)
        except RuntimeError:
            pass

    # -- Context managers ----------------------------------------------------

    def __enter__(self) -> TelemetryHandler:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    async def __aenter__(self) -> TelemetryHandler:
        await self.astart()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.astop()

    # -- Internal loop -------------------------------------------------------

    async def _timer_loop(self) -> None:
        assert self._flush_signal is not None
        while self._running:
            try:
                await asyncio.wait_for(
                    self._flush_signal.wait(),
                    timeout=self._flush_interval,
                )
            except asyncio.TimeoutError:
                pass
            self._flush_signal.clear()
            await self._flush_events()

    async def _flush_events(self) -> None:
        with self._queue_lock:
            dlq_events, self._dlq = self._dlq, []
            new_events, self._events = self._events, []
        events_to_send = dlq_events + new_events
        if events_to_send:
            await self._send_events(events_to_send)

    async def _send_events(self, events: list[QueuedEvent]) -> None:
        endpoint = _telemetry_endpoint()
        if endpoint is None:
            return
        try:
            async with httpx.AsyncClient(timeout=SEND_TIMEOUT_SECONDS) as client:
                await self._send_events_with_client(client, events, endpoint=endpoint)
        except Exception:  # noqa: BLE001
            logger.debug("Telemetry send failed; routing events to DLQ", exc_info=True)
            self._add_to_dlq(events)

    async def _send_events_with_client(
        self, client: httpx.AsyncClient, events: list[QueuedEvent], *, endpoint: str | None = None
    ) -> None:
        if not events:
            return

        endpoint = endpoint or _telemetry_endpoint()
        if endpoint is None:
            return

        payload = build_payload(events, source_client_version=self._source_client_version, session_id=self._session_id)
        logger.debug(
            "Sending telemetry events",
            extra={
                "ctx": {
                    "endpoint": _redact_endpoint(endpoint),
                    "event_count": len(events),
                    "events": [
                        {
                            "name": queued.event._event_name,
                            "task": getattr(queued.event, "task", "unknown"),
                            "task_status": getattr(queued.event, "task_status", "unknown"),
                            "deployment_type": getattr(queued.event, "deployment_type", "unknown"),
                            "retry_count": queued.retry_count,
                        }
                        for queued in events
                    ],
                }
            },
        )
        try:
            response = await client.post(endpoint, json=payload)
            if response.status_code in (400, 422) or response.is_success:
                return
            if response.status_code == 413:
                if len(events) == 1:
                    return
                mid = len(events) // 2
                await self._send_events_with_client(client, events[:mid], endpoint=endpoint)
                await self._send_events_with_client(client, events[mid:], endpoint=endpoint)
                return
            if response.status_code in (408, 429) or response.status_code >= 500:
                self._add_to_dlq(events)
        except Exception:  # noqa: BLE001
            logger.debug("Telemetry POST failed to %s; routing events to DLQ", _redact_endpoint(endpoint))
            self._add_to_dlq(events)

    def _add_to_dlq(self, events: list[QueuedEvent]) -> None:
        with self._queue_lock:
            for queued in events:
                queued.retry_count += 1
                if queued.retry_count > self._max_retries:
                    continue
                self._dlq.append(queued)
