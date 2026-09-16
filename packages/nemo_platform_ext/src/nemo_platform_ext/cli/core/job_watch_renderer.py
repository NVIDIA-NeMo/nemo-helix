# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping
from datetime import datetime
from enum import Enum

from nemo_platform import NotFoundError as PlatformNotFoundError
from nemo_platform_plugin.client.errors import NotFoundError as PluginNotFoundError
from nemo_platform_plugin.jobs.watch_types import (
    JobLogEvent,
    JobStatusEvent,
    JobWarningEvent,
    JobWatchEvent,
    JobWatchTimeoutError,
)
from rich.console import Console
from rich.text import Text

from nemo_platform_ext.cli.core.job_log_renderer import render_log_line


class JobWatchRenderResult(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


def render_job_watch_events(
    events: Iterable[JobWatchEvent],
    *,
    console: Console | None = None,
    error_console: Console | None = None,
    resource_label: str | None = None,
    start_time: float | None = None,
) -> JobWatchRenderResult:
    """Render job watch events and return the final watch result."""
    _ = (resource_label, start_time)
    output = console or Console()
    errors = error_console or Console(stderr=True)
    terminal_event: JobStatusEvent | None = None
    last_status_event: JobStatusEvent | None = None

    try:
        for event in events:
            rendered_status_event = _render_event(output, event)
            if rendered_status_event is not None:
                last_status_event = rendered_status_event
                if rendered_status_event.terminal:
                    terminal_event = rendered_status_event
    except (KeyboardInterrupt, asyncio.CancelledError):
        _render_interrupted(errors, last_status_event)
        return JobWatchRenderResult.INTERRUPTED
    except JobWatchTimeoutError as exc:
        errors.print(str(exc), style="red")
        return JobWatchRenderResult.FAILED
    except (PlatformNotFoundError, PluginNotFoundError) as exc:
        errors.print(str(exc), style="red")
        return JobWatchRenderResult.FAILED

    if terminal_event is None:
        return JobWatchRenderResult.FAILED

    if terminal_event.successful:
        output.print(f"Job {terminal_event.job_name!r} completed", style="green")
        return JobWatchRenderResult.SUCCEEDED

    message = f"Job {terminal_event.job_name!r} ended with status {terminal_event.status!r}"
    error_details = _status_details(terminal_event.error_details or {})
    if error_details:
        message = f"{message}: {error_details}"
    output.print(message, style="red")
    return JobWatchRenderResult.FAILED


def _render_event(console: Console, event: JobWatchEvent) -> JobStatusEvent | None:
    if isinstance(event, JobStatusEvent):
        _render_status(console, event)
        return event
    if isinstance(event, JobLogEvent):
        _render_log(console, event)
        return None
    if isinstance(event, JobWarningEvent):
        _render_warning(console, event)
        return None
    return None


def _render_interrupted(console: Console, last_status_event: JobStatusEvent | None) -> None:
    if last_status_event is None:
        console.print("Interrupted watching; no status received. Exiting.", style="yellow")
        return
    status = last_status_event.status
    details = _status_details(last_status_event.status_details)
    if details:
        status = f"{status} {details}"
    console.print(
        f"Interrupted watching job {last_status_event.job_name!r}; last known status: {status}. Exiting.",
        style="yellow",
    )


def _render_status(console: Console, event: JobStatusEvent) -> None:
    line = Text()
    line.append(f"[{_time_label()}] ", style="dim")
    line.append("Status: ")
    line.append(event.status, style=_status_style(event))
    details = _status_details(event.status_details)
    if details:
        line.append(f" {details}", style="dim")
    console.print(line)


def _render_log(console: Console, event: JobLogEvent) -> None:
    render_log_line(
        console,
        timestamp=event.timestamp,
        step_id=event.step_id,
        task_id=event.task_id,
        message=event.message,
    )


def _render_warning(console: Console, event: JobWarningEvent) -> None:
    line = Text()
    line.append(f"[{_time_label()}] ", style="dim")
    line.append(event.message, style="yellow")
    console.print(line)


def _time_label(timestamp: datetime | None = None) -> str:
    value = timestamp.astimezone() if timestamp is not None else datetime.now()
    return value.strftime("%H:%M:%S")


def _status_style(event: JobStatusEvent) -> str:
    if event.successful:
        return "green bold"
    if event.successful is False:
        return "red bold"
    return "cyan bold"


def _status_details(details: Mapping[str, object]) -> str:
    parts = []
    for key, value in details.items():
        if value in (None, "", [], {}):
            continue
        if isinstance(value, str | int | float | bool):
            parts.append(f"{key}={value}")
        if len(parts) >= 6:
            break
    return " ".join(parts)
