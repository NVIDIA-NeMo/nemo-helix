# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from datetime import datetime, timezone
from io import StringIO
from unittest.mock import patch

import httpx
import pytest
from nemo_platform_ext.cli.core.job_watch_renderer import JobWatchRenderResult, render_job_watch_events
from nemo_platform_plugin.client.errors import NotFoundError
from nemo_platform_plugin.jobs.watch_types import JobLogEvent, JobStatusEvent, JobWatchEvent, JobWatchTimeoutError
from rich.console import Console


def _console_pair() -> tuple[Console, StringIO]:
    output = StringIO()
    return Console(file=output, force_terminal=False, color_system=None, width=120), output


def test_render_job_watch_events_returns_true_for_completed_status() -> None:
    console, output = _console_pair()
    error_console, error_output = _console_pair()
    events: list[JobWatchEvent] = [
        JobStatusEvent(
            kind="status",
            job_name="job-a",
            status="active",
            status_details={"phase": "training", "progress_pct": 41},
            terminal=False,
            successful=None,
        ),
        log_event := JobLogEvent(
            kind="log",
            job_name="job-a",
            timestamp=datetime(2026, 7, 31, 12, 0, tzinfo=timezone.utc),
            step_id="step-a",
            task_id="task-a",
            message="started",
        ),
        JobStatusEvent(
            kind="status",
            job_name="job-a",
            status="completed",
            status_details={},
            terminal=True,
            successful=True,
        ),
    ]

    assert (
        render_job_watch_events(events, console=console, error_console=error_console) is JobWatchRenderResult.SUCCEEDED
    )

    rendered = output.getvalue()
    expected_log_time = log_event.timestamp.astimezone().strftime("%H:%M:%S")
    assert "Status: active phase=training progress_pct=41" in rendered
    assert f"[{expected_log_time}] step-a/task-a | started" in rendered
    assert "step-a/task-a | started" in rendered
    assert "Job 'job-a' completed" in rendered
    assert error_output.getvalue() == ""


def test_render_job_watch_events_does_not_emit_client_job_run_event() -> None:
    console, _ = _console_pair()
    event = JobStatusEvent(
        kind="status",
        job_name="job-a",
        status="completed",
        status_details={"model": "nemotron"},
        terminal=True,
        successful=True,
    )

    with patch("nemo_platform_ext.cli.telemetry.emit.emit_event") as emit_event:
        assert (
            render_job_watch_events([event], console=console, resource_label="job", start_time=123.0)
            is JobWatchRenderResult.SUCCEEDED
        )

    emit_event.assert_not_called()


def test_render_job_watch_events_returns_false_for_failed_terminal_status() -> None:
    console, output = _console_pair()
    event = JobStatusEvent(
        kind="status",
        job_name="job-a",
        status="error",
        status_details={},
        error_details={"reason": "container exited", "exit_code": 137, "empty": {}},
        terminal=True,
        successful=False,
    )

    assert render_job_watch_events([event], console=console) is JobWatchRenderResult.FAILED

    assert "Job 'job-a' ended with status 'error': reason=container exited exit_code=137" in output.getvalue()


def test_render_job_watch_events_catches_timeout() -> None:
    console, output = _console_pair()
    error_console, error_output = _console_pair()

    def events() -> Iterator[JobWatchEvent]:
        yield JobStatusEvent(
            kind="status",
            job_name="job-a",
            status="active",
            status_details={},
            terminal=False,
            successful=None,
        )
        raise JobWatchTimeoutError("Timed out watching job 'job-a'")

    assert (
        render_job_watch_events(events(), console=console, error_console=error_console) is JobWatchRenderResult.FAILED
    )

    assert "Status: active" in output.getvalue()
    assert "Timed out watching job 'job-a'" in error_output.getvalue()


@pytest.mark.parametrize("exception_cls", [KeyboardInterrupt, asyncio.CancelledError])
def test_render_job_watch_events_catches_interrupts(exception_cls: type[BaseException]) -> None:
    console, output = _console_pair()
    error_console, error_output = _console_pair()

    def events() -> Iterator[JobWatchEvent]:
        yield JobStatusEvent(
            kind="status",
            job_name="job-a",
            status="active",
            status_details={},
            terminal=False,
            successful=None,
        )
        raise exception_cls()

    assert (
        render_job_watch_events(events(), console=console, error_console=error_console)
        is JobWatchRenderResult.INTERRUPTED
    )

    assert "Status: active" in output.getvalue()
    assert "Interrupted watching job 'job-a'; last known status: active. Exiting." in error_output.getvalue()


def test_render_job_watch_events_catches_interrupt_before_status() -> None:
    console, output = _console_pair()
    error_console, error_output = _console_pair()

    def events() -> Iterator[JobWatchEvent]:
        raise KeyboardInterrupt
        yield JobStatusEvent(
            kind="status",
            job_name="job-a",
            status="active",
            status_details={},
            terminal=False,
            successful=None,
        )

    assert (
        render_job_watch_events(events(), console=console, error_console=error_console)
        is JobWatchRenderResult.INTERRUPTED
    )

    assert output.getvalue() == ""
    assert "Interrupted watching; no status received. Exiting." in error_output.getvalue()


def test_render_job_watch_events_catches_not_found() -> None:
    console, output = _console_pair()
    error_console, error_output = _console_pair()

    def events() -> Iterator[JobWatchEvent]:
        request = httpx.Request("GET", "http://test")
        response = httpx.Response(
            404,
            request=request,
            json={"detail": "Job 'missing' not found in workspace 'default'."},
        )
        raise NotFoundError(response)
        yield JobStatusEvent(
            kind="status",
            job_name="missing",
            status="active",
            status_details={},
            terminal=False,
            successful=None,
        )

    assert (
        render_job_watch_events(events(), console=console, error_console=error_console) is JobWatchRenderResult.FAILED
    )

    assert output.getvalue() == ""
    assert "HTTP 404: Job 'missing' not found in workspace 'default'." in error_output.getvalue()
