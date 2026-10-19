# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import importlib
import threading
from datetime import datetime, timezone
from typing import ClassVar
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nemo_helix_ext.cli.telemetry.events import DeploymentTypeEnum, HelixTelemetryEvent, _deployment_type
from nemo_helix_ext.cli.telemetry.handler import (
    QueuedEvent,
    TelemetryHandler,
    _telemetry_enabled,
    _telemetry_endpoint,
    build_payload,
)
from pydantic import Field

telemetry_module = importlib.import_module("nemo_helix_ext.cli.telemetry.handler")


# =============================================================================
# Stub Event Model for Testing
# =============================================================================


class _StubEvent(HelixTelemetryEvent):
    """Minimal concrete event for testing, subclassing the real HelixTelemetryEvent.

    ``task_status`` and ``deployment_type`` are redeclared as plain strings to shed
    the base's serialization aliases, so the handler tests keep asserting the
    snake_case keys the Task 1 handler emitted.
    """

    _event_name: ClassVar[str] = "stub_event"
    task: str = Field(default="test_task")
    task_status: str = Field(default="completed")
    deployment_type: str = Field(default="sdk")


# =============================================================================
# Env-var helpers
# =============================================================================


class TestEnvHelpers:
    def test_telemetry_enabled_default(self, monkeypatch):
        monkeypatch.delenv("NEMO_TELEMETRY_ENABLED", raising=False)
        assert _telemetry_enabled() is True

    def test_telemetry_enabled_disabled(self, monkeypatch):
        monkeypatch.setenv("NEMO_TELEMETRY_ENABLED", "false")
        assert _telemetry_enabled() is False

    def test_telemetry_enabled_true_when_explicitly_true(self, monkeypatch):
        monkeypatch.setenv("NEMO_TELEMETRY_ENABLED", "true")
        assert _telemetry_enabled() is True

    @pytest.mark.parametrize("value", ["", "0", "1", "yes", "yep", "for sure"])
    def test_telemetry_enabled_non_true_values_disable(self, monkeypatch, value):
        monkeypatch.setenv("NEMO_TELEMETRY_ENABLED", value)
        assert _telemetry_enabled() is False

    def test_telemetry_endpoint_preserves_case(self, monkeypatch):
        custom = "https://Events.Telemetry.example.COM/v1/Events?Token=AbC"
        monkeypatch.setenv("NEMO_TELEMETRY_ENDPOINT", custom)
        assert _telemetry_endpoint() == custom

    def test_deployment_type_default(self, monkeypatch):
        monkeypatch.delenv("NEMO_DEPLOYMENT_TYPE", raising=False)
        assert _deployment_type() is DeploymentTypeEnum.CLI

    def test_deployment_type_invalid_falls_back_to_undefined(self, monkeypatch):
        monkeypatch.setenv("NEMO_DEPLOYMENT_TYPE", "definitely-not-real")
        assert _deployment_type() is DeploymentTypeEnum.UNDEFINED

    def test_deployment_type_nvidia_internal(self, monkeypatch):
        monkeypatch.setenv("NEMO_DEPLOYMENT_TYPE", "nvidia-internal")
        assert _deployment_type() is DeploymentTypeEnum.NVIDIA_INTERNAL


# =============================================================================
# build_payload
# =============================================================================


class TestBuildPayload:
    def _make_queued(self, task: str = "generate", status: str = "completed") -> QueuedEvent:
        event = _StubEvent(task=task, task_status=status)
        return QueuedEvent(event=event, timestamp=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc))

    def test_structure(self):
        queued = self._make_queued()
        payload = build_payload([queued], source_client_version="1.2.3", session_id="test-session")
        assert payload["clientId"] == "184482118588404"
        assert payload["clientVer"] == "1.2.3"
        assert payload["sessionId"] == "test-session"
        assert len(payload["events"]) == 1

    def test_event_fields_serialize_as_strings(self, monkeypatch):
        monkeypatch.delenv("NEMO_DEPLOYMENT_TYPE", raising=False)

        queued = self._make_queued(task="train", status="error")
        payload = build_payload([queued], source_client_version="0.0.1")
        event_entry = payload["events"][0]
        assert event_entry["name"] == "stub_event"
        assert event_entry["ts"] == "2025-01-01T12:00:00.000Z"
        params = event_entry["parameters"]
        assert params["task"] == "train"
        assert params["task_status"] == "error"

    def test_payload_is_json_serializable(self):
        """Regression: the full payload must be encodable by the stdlib JSON encoder."""
        import json

        queued = self._make_queued(status="error")
        payload = build_payload([queued], source_client_version="1.0.0")
        json.dumps(payload)

    def test_multiple_events(self):
        events = [self._make_queued(task=t) for t in ("train", "generate", "evaluate")]
        payload = build_payload(events, source_client_version="1.0.0")
        assert len(payload["events"]) == 3
        tasks = [e["parameters"]["task"] for e in payload["events"]]
        assert tasks == ["train", "generate", "evaluate"]

    def test_default_session_id(self):
        queued = self._make_queued()
        payload = build_payload([queued], source_client_version="1.0.0")
        assert payload["sessionId"] == "undefined"

    def test_cpu_architecture_uses_platform_machine(self, monkeypatch):
        monkeypatch.setattr(telemetry_module.platform, "machine", lambda: "arm64")
        payload = build_payload([self._make_queued()], source_client_version="1.0.0")
        assert payload["cpuArchitecture"] == "arm64"

    def test_cpu_architecture_empty_value_falls_back_to_undefined(self, monkeypatch):
        monkeypatch.setattr(telemetry_module.platform, "machine", lambda: "")
        payload = build_payload([self._make_queued()], source_client_version="1.0.0")
        assert payload["cpuArchitecture"] == "undefined"

    def test_empty_events_raises(self):
        with pytest.raises(ValueError):
            build_payload([], source_client_version="1.0.0")


# =============================================================================
# TelemetryHandler — telemetry disabled
# =============================================================================


class TestTelemetryDisabled:
    def test_enqueue_noop_when_disabled(self, monkeypatch):
        monkeypatch.setenv("NEMO_TELEMETRY_ENABLED", "false")
        handler = TelemetryHandler()
        event = _StubEvent(task="generate")
        handler.enqueue(event)
        assert handler._events == []

    def test_enqueue_noop_for_non_event(self, monkeypatch):
        """Silently ignores non-TelemetryEvent objects regardless of env."""
        monkeypatch.setenv("NEMO_TELEMETRY_ENABLED", "true")
        handler = TelemetryHandler()
        handler.enqueue("not an event")
        assert handler._events == []


# =============================================================================
# TelemetryHandler — enqueue and flush
# =============================================================================


class TestTelemetryHandlerEnqueue:
    def test_enqueue_adds_event(self, monkeypatch):
        monkeypatch.setenv("NEMO_TELEMETRY_ENABLED", "true")
        handler = TelemetryHandler()
        event = _StubEvent(task="generate")
        handler.enqueue(event)
        assert len(handler._events) == 1
        assert handler._events[0].event is event

    def test_enqueue_does_not_add_event_when_telemetry_is_disabled(self, monkeypatch):
        monkeypatch.setenv("NEMO_TELEMETRY_ENABLED", "false")
        handler = TelemetryHandler()
        event = _StubEvent(task="generate")
        handler.enqueue(event)
        assert len(handler._events) == 0

    def test_enqueue_at_max_queue_size_signals_flush_when_running(self, monkeypatch):
        """When a background loop is up, hitting max_queue_size should signal a flush."""
        monkeypatch.setenv("NEMO_TELEMETRY_ENABLED", "true")
        handler = TelemetryHandler(max_queue_size=3, flush_interval_seconds=60.0)
        flushed = threading.Event()

        async def fake_send(_events):
            flushed.set()

        with patch.object(handler, "_send_events", side_effect=fake_send):
            handler.start()
            try:
                for _ in range(3):
                    handler.enqueue(_StubEvent(task="run"))
                assert flushed.wait(timeout=2.0), "max_queue_size flush signal did not fire"
            finally:
                handler.stop()


# =============================================================================
# TelemetryHandler — _flush_events queue clearing and DLQ
# =============================================================================


class TestFlushEventsQueueClearing:
    async def test_flush_events_clears_queue(self):
        """_flush_events() must drain _events even when the underlying send succeeds."""
        handler = TelemetryHandler(source_client_version="1.0.0")
        event = _StubEvent(task="generate")
        handler._events.append(QueuedEvent(event=event, timestamp=datetime.now(timezone.utc)))

        async def fake_send(events):
            assert len(events) == 1

        with patch.object(handler, "_send_events", side_effect=fake_send):
            await handler._flush_events()

        assert handler._events == []
        assert handler._dlq == []

    async def test_flush_events_includes_dlq(self):
        handler = TelemetryHandler(source_client_version="1.0.0")
        event = _StubEvent(task="generate")
        handler._dlq.append(QueuedEvent(event=event, timestamp=datetime.now(timezone.utc), retry_count=1))
        handler._events.append(QueuedEvent(event=event, timestamp=datetime.now(timezone.utc)))

        sent: list[list[QueuedEvent]] = []

        async def fake_send(events):
            sent.append(list(events))

        with patch.object(handler, "_send_events", side_effect=fake_send):
            await handler._flush_events()

        assert handler._events == []
        assert handler._dlq == []
        assert len(sent) == 1
        assert len(sent[0]) == 2


# =============================================================================
# TelemetryHandler — send and retry
# =============================================================================


class TestTelemetryHandlerSend:
    def _make_handler(self) -> TelemetryHandler:
        return TelemetryHandler(source_client_version="1.0.0", session_id="s1")

    def _make_queued(self) -> QueuedEvent:
        event = _StubEvent(task="generate")
        return QueuedEvent(event=event, timestamp=datetime.now(timezone.utc))

    async def test_debug_logs_event_metadata_before_post(self, monkeypatch):
        handler = self._make_handler()
        queued = self._make_queued()
        events_seen: list[str] = []
        debug_calls = []

        monkeypatch.setenv("NEMO_TELEMETRY_ENDPOINT", "https://Events.Telemetry.example.COM/v1/Events?Token=AbC")

        def fake_debug(message, *, extra):
            events_seen.append("debug")
            debug_calls.append((message, extra))

        async def fake_post(*_args, **_kwargs):
            events_seen.append("post")
            return MagicMock(status_code=200, is_success=True)

        monkeypatch.setattr(telemetry_module.logger, "debug", fake_debug)
        mock_client = AsyncMock()
        mock_client.post.side_effect = fake_post

        await handler._send_events_with_client(mock_client, [queued])

        assert events_seen == ["debug", "post"]
        message, extra = debug_calls[0]
        assert message == "Sending telemetry events"
        ctx = extra["ctx"]
        assert ctx["endpoint"] == "https://Events.Telemetry.example.COM/v1/Events?<redacted>"
        assert ctx["event_count"] == 1
        assert ctx["events"] == [
            {
                "name": "stub_event",
                "task": "generate",
                "task_status": "completed",
                "deployment_type": "sdk",
                "retry_count": 0,
            }
        ]

    async def test_successful_send_does_not_dlq(self):
        handler = self._make_handler()
        queued = self._make_queued()

        mock_response = MagicMock(status_code=200, is_success=True)
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        await handler._send_events_with_client(mock_client, [queued])
        mock_client.post.assert_awaited_once()
        assert handler._dlq == []

    async def test_500_adds_to_dlq(self):
        handler = self._make_handler()
        queued = self._make_queued()

        mock_response = MagicMock(status_code=500, is_success=False)
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        await handler._send_events_with_client(mock_client, [queued])
        assert len(handler._dlq) == 1
        assert handler._dlq[0].retry_count == 1

    async def test_429_adds_to_dlq(self):
        handler = self._make_handler()
        queued = self._make_queued()

        mock_response = MagicMock(status_code=429, is_success=False)
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        await handler._send_events_with_client(mock_client, [queued])
        assert len(handler._dlq) == 1
        assert handler._dlq[0].retry_count == 1

    async def test_exceeds_max_retries_dropped(self):
        handler = self._make_handler()
        queued = self._make_queued()
        queued.retry_count = handler._max_retries

        mock_response = MagicMock(status_code=500, is_success=False)
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        await handler._send_events_with_client(mock_client, [queued])
        assert handler._dlq == []

    async def test_413_splits_and_retries(self):
        handler = self._make_handler()
        event = _StubEvent(task="generate")
        events = [
            QueuedEvent(event=event, timestamp=datetime.now(timezone.utc)),
            QueuedEvent(event=event, timestamp=datetime.now(timezone.utc)),
        ]

        success_response = MagicMock(status_code=200, is_success=True)
        too_large_response = MagicMock(status_code=413, is_success=False)
        mock_client = AsyncMock()
        mock_client.post.side_effect = [too_large_response, success_response, success_response]

        await handler._send_events_with_client(mock_client, events)
        assert mock_client.post.await_count == 3

    async def test_send_events_routes_to_dlq_on_client_setup_failure(self):
        """If httpx client creation raises, events must land in DLQ rather than vanish."""
        handler = self._make_handler()
        queued = self._make_queued()

        with patch("httpx.AsyncClient", side_effect=RuntimeError("boom")):
            await handler._send_events([queued])

        assert len(handler._dlq) == 1

    def test_session_prefix_applied(self, monkeypatch):
        monkeypatch.setenv("NEMO_SESSION_PREFIX", "pfx-")
        handler = TelemetryHandler(session_id="abc")
        assert handler._session_id == "pfx-abc"


# =============================================================================
# TelemetryHandler — aflush awaits a real flush
# =============================================================================


class TestAflushAwaits:
    async def test_aflush_actually_flushes(self, monkeypatch):
        monkeypatch.setenv("NEMO_TELEMETRY_ENABLED", "true")
        handler = TelemetryHandler(source_client_version="1.0.0")
        event = _StubEvent(task="generate")
        handler.enqueue(event)
        assert len(handler._events) == 1

        sent: list[int] = []

        async def fake_send(events):
            sent.append(len(events))

        with patch.object(handler, "_send_events", side_effect=fake_send):
            await handler.aflush()

        assert handler._events == []
        assert sent == [1]


# =============================================================================
# TelemetryHandler — sync flush from async caller contexts
# =============================================================================


class TestFlushFromRunningLoop:
    """Regression coverage for notebooks and async SDK callers."""

    def test_flush_runs_to_completion_when_loop_is_running(self) -> None:
        sent: list[int] = []

        async def fake_flush(self) -> None:  # noqa: ARG001
            sent.append(1)

        async def driver() -> None:
            handler = TelemetryHandler(source_client_version="1.0.0")
            handler._events.append(
                QueuedEvent(
                    event=_StubEvent(task="run"),
                    timestamp=datetime.now(timezone.utc),
                )
            )
            with patch.object(TelemetryHandler, "_flush_events", new=fake_flush):
                handler.flush()

        asyncio.run(driver())
        assert sent == [1]

    def test_stop_flushes_fire_and_flush_path_when_loop_is_running(self, monkeypatch) -> None:
        monkeypatch.setenv("NEMO_TELEMETRY_ENABLED", "true")
        sent: list[int] = []

        async def fake_send(events):
            sent.append(len(events))

        async def driver() -> None:
            handler = TelemetryHandler(source_client_version="1.0.0")
            with patch.object(handler, "_send_events", side_effect=fake_send):
                handler.enqueue(_StubEvent(task="run"))
                handler.stop()
                assert handler._events == []

        asyncio.run(driver())
        assert sent == [1]

    def test_run_sync_propagates_exception_to_flush_boundary(self) -> None:
        async def boom() -> None:
            raise RuntimeError("kaboom")

        async def driver() -> None:
            with pytest.raises(RuntimeError, match="kaboom"):
                TelemetryHandler._run_sync(boom())

        asyncio.run(driver())


# =============================================================================
# TelemetryHandler — sync lifecycle and context manager
# =============================================================================


class TestSyncLifecycle:
    def test_fire_and_flush_without_start(self, monkeypatch):
        """Pattern used by the SDK: construct, enqueue, stop. No start() call. stop() must flush."""
        monkeypatch.setenv("NEMO_TELEMETRY_ENABLED", "true")
        handler = TelemetryHandler(source_client_version="1.0.0")
        event = _StubEvent(task="generate")
        handler.enqueue(event)
        assert len(handler._events) == 1

        sent: list[int] = []

        async def fake_send(events):
            sent.append(len(events))

        with patch.object(handler, "_send_events", side_effect=fake_send):
            handler.stop()

        assert handler._events == []
        assert sent == [1]
        assert handler._thread is None

    def test_start_spawns_thread_and_stop_flushes(self, monkeypatch):
        monkeypatch.setenv("NEMO_TELEMETRY_ENABLED", "true")
        handler = TelemetryHandler(source_client_version="1.0.0", flush_interval_seconds=60.0)

        sent: list[int] = []

        async def fake_send(events):
            sent.append(len(events))

        with patch.object(handler, "_send_events", side_effect=fake_send):
            handler.start()
            assert handler._thread is not None
            assert handler._thread.is_alive()

            handler.enqueue(_StubEvent(task="run"))
            handler.stop()

        assert handler._thread is None
        assert handler._loop is None
        assert sent == [1]

    def test_sync_context_manager(self, monkeypatch):
        monkeypatch.setenv("NEMO_TELEMETRY_ENABLED", "true")
        sent: list[int] = []

        async def fake_send(events):
            sent.append(len(events))

        with patch.object(TelemetryHandler, "_send_events", side_effect=fake_send, autospec=False):
            with TelemetryHandler(source_client_version="1.0.0", flush_interval_seconds=60.0) as handler:
                handler.enqueue(_StubEvent(task="run"))

        assert sent == [1]

    def test_sync_flush_during_background_run(self, monkeypatch):
        monkeypatch.setenv("NEMO_TELEMETRY_ENABLED", "true")
        handler = TelemetryHandler(source_client_version="1.0.0", flush_interval_seconds=60.0)

        sent: list[int] = []

        async def fake_send(events):
            sent.append(len(events))

        with patch.object(handler, "_send_events", side_effect=fake_send):
            handler.start()
            try:
                handler.enqueue(_StubEvent(task="run"))
                handler.flush()
                assert sent == [1]
                assert handler._events == []
            finally:
                handler.stop()

    def test_timer_driven_flush(self, monkeypatch):
        """With a short flush interval, the background timer should drive a flush without explicit calls."""
        monkeypatch.setenv("NEMO_TELEMETRY_ENABLED", "true")
        handler = TelemetryHandler(source_client_version="1.0.0", flush_interval_seconds=0.05)

        flushed = threading.Event()

        async def fake_send(_events):
            flushed.set()

        with patch.object(handler, "_send_events", side_effect=fake_send):
            handler.start()
            try:
                handler.enqueue(_StubEvent(task="run"))
                assert flushed.wait(timeout=2.0), "timer-driven flush did not fire"
            finally:
                handler.stop()


# =============================================================================
# TelemetryHandler — async lifecycle
# =============================================================================


class TestAsyncLifecycle:
    async def test_async_context_manager_flushes_on_exit(self, monkeypatch):
        monkeypatch.setenv("NEMO_TELEMETRY_ENABLED", "true")

        sent: list[int] = []

        async def fake_send(events):
            sent.append(len(events))

        async with TelemetryHandler(source_client_version="1.0.0", flush_interval_seconds=60.0) as handler:
            with patch.object(handler, "_send_events", side_effect=fake_send):
                handler.enqueue(_StubEvent(task="run"))
                await handler.astop()

        assert sent == [1]

    async def test_enqueue_at_max_size_signals_flush_in_async_mode(self, monkeypatch):
        monkeypatch.setenv("NEMO_TELEMETRY_ENABLED", "true")

        flushed = asyncio.Event()

        async def fake_send(_events):
            flushed.set()

        handler = TelemetryHandler(source_client_version="1.0.0", flush_interval_seconds=60.0, max_queue_size=2)
        with patch.object(handler, "_send_events", side_effect=fake_send):
            await handler.astart()
            try:
                handler.enqueue(_StubEvent(task="run"))
                handler.enqueue(_StubEvent(task="run"))
                await asyncio.wait_for(flushed.wait(), timeout=2.0)
            finally:
                await handler.astop()
