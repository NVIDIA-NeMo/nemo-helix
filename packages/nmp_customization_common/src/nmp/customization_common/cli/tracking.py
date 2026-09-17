# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Post-submit feedback for Customizer backends.

After ``submit`` creates a job, two things happen here:

- :func:`print_tracking_instructions` writes a short message naming the job and
  the commands that follow it.
- :func:`follow_job` optionally blocks until the job reaches a terminal state,
  for ``--wait`` (status only) and ``--watch`` (status and logs). Both show a
  spinner, so a phase that produces no output still looks alive.

Everything is written to **stderr**. Submit prints the job JSON to stdout, and
scripts parse that, so stdout stays exactly as it was.

A Customizer job is finished when its **top-level status** is terminal, not when
its steps finish. A job can report every step complete, and ``progress_pct: 100``,
while it is still uploading the trained weights and registering the model entity.
:func:`nemo_platform_plugin.jobs.watch.watch_job` already keys on the top-level
status, so both modes below read ``JobStatusEvent.terminal``.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum

from nemo_platform_plugin.cli_progress import ProgressHandle, request_progress
from nemo_platform_plugin.jobs.client import JobsClient
from nemo_platform_plugin.jobs.watch import watch_job
from nemo_platform_plugin.jobs.watch_types import (
    JobLogEvent,
    JobStatusEvent,
    JobWarningEvent,
    JobWatchEvent,
    JobWatchTimeoutError,
)
from rich.console import Console


class FollowResult(str, Enum):
    """Outcome of following a job to a terminal state."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


def _print(console: Console, text: str, *, style: str | None = None) -> None:
    """Write *text* verbatim.

    Job logs and platform messages are data, not Rich markup. Left to Rich, a log
    line reading ``expected list[int]`` prints as ``expected list``, and numbers
    get recoloured. Both are off here.
    """
    console.print(text, style=style, markup=False, highlight=False)


def print_tracking_instructions(
    *,
    backend: str,
    job_name: str,
    workspace: str,
    console: Console | None = None,
) -> None:
    """Report the created job and the commands that track it.

    Only commands the user can run now. ``--wait`` and ``--watch`` are documented
    on ``submit --help``, because by the time this prints the job is already
    submitted and the flags no longer apply to it.
    """
    out = console if console is not None else Console(stderr=True)
    _print(out, f"\nSubmitted {backend} job {job_name} to workspace {workspace}.")
    _print(out, f"Track it with 'nemo jobs watch {job_name}', or check its status with")
    _print(out, f"'nemo jobs get-status {job_name}'.")


def follow_job(
    *,
    base_url: str,
    job_name: str,
    workspace: str,
    headers: dict[str, str],
    include_logs: bool,
    timeout: int | None,
    poll_interval: int,
    console: Console | None = None,
) -> FollowResult:
    """Block until *job_name* reaches a terminal state, reporting progress.

    With *include_logs* the job's logs are printed as they arrive.
    """
    client = JobsClient(base_url=base_url, workspace=workspace, default_headers=headers or None)
    events = watch_job(
        client,
        job_name,
        workspace=workspace,
        poll_interval=poll_interval,
        timeout=timeout,
        include_logs=include_logs,
    )
    out = console if console is not None else Console(stderr=True)
    return follow_events(events, job_name=job_name, print_logs=include_logs, console=out)


def follow_events(
    events: Iterable[JobWatchEvent],
    *,
    job_name: str,
    print_logs: bool,
    console: Console,
) -> FollowResult:
    """Render *events* until the job finishes, and report the outcome.

    The spinner runs for both ``--wait`` and ``--watch``. A training job can sit
    in one phase for several minutes while it is scheduled and its image is
    pulled, producing no logs, and ``watch_job`` only yields a status event when
    the status actually changes. Without the spinner's elapsed clock there is
    nothing on screen for that whole window and the command looks stuck. Rich
    draws log lines above the spinner, so both can run at once.
    """
    with request_progress(f"{job_name}: waiting for status", console=console) as progress:
        return _consume(
            events,
            job_name=job_name,
            console=console,
            progress=progress,
            print_logs=print_logs,
        )


def _consume(
    events: Iterable[JobWatchEvent],
    *,
    job_name: str,
    console: Console,
    progress: ProgressHandle,
    print_logs: bool,
) -> FollowResult:
    """Drive the event stream and report the terminal outcome."""
    terminal: JobStatusEvent | None = None
    try:
        for event in events:
            if isinstance(event, JobStatusEvent):
                progress.update(f"{job_name}: {event.status}")
                if not progress.is_active:
                    # No spinner is carrying the status, so leave it as a line.
                    _print(console, f"{job_name}: {event.status}")
                if event.terminal:
                    terminal = event
            elif isinstance(event, JobWarningEvent):
                _print(console, f"Warning: {event.message}", style="yellow")
            elif print_logs and isinstance(event, JobLogEvent):
                _print(console, event.message)
    except KeyboardInterrupt:
        _print(console, f"\nStopped following {job_name}. The job is still running.")
        return FollowResult.INTERRUPTED
    except JobWatchTimeoutError as exc:
        _print(console, f"Error: {exc}", style="red")
        return FollowResult.FAILED

    if terminal is None:
        _print(console, f"Error: stopped following {job_name} before it reached a final status.", style="red")
        return FollowResult.FAILED
    if terminal.successful:
        _print(console, f"Job {job_name} completed.", style="green")
        return FollowResult.SUCCEEDED

    _print(console, f"Error: job {job_name} ended with status {terminal.status}.", style="red")
    for line in _detail_lines(terminal):
        _print(console, f"  {line}", style="red")
    return FollowResult.FAILED


def _detail_lines(event: JobStatusEvent) -> list[str]:
    """Flatten the platform's error details into one line per entry."""
    details = event.error_details or {}
    return [f"{key}: {value}" for key, value in details.items() if value not in (None, "", {}, [])]
