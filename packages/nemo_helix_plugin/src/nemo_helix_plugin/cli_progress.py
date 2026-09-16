# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stderr progress indicators for long-running plugin CLI requests.

`request_progress` wraps a synchronous in-flight HTTP call (or any blocking
operation) with a spinner + elapsed-time counter rendered on stderr. The
spinner auto-disables when stderr isn't a TTY or when the user opts out via
``--no-progress`` / ``NO_COLOR`` / ``CI`` / ``NEMO_NO_PROGRESS``, so piping
stdout to ``jq`` or redirecting in CI stays clean.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, Task, TaskID, TextColumn, TimeElapsedColumn
from rich.text import Text

_OPT_OUT_ENV_VARS = ("NO_COLOR", "CI", "NEMO_NO_PROGRESS")


class _MinutesSecondsElapsedColumn(TimeElapsedColumn):
    """Elapsed-time column rendered as MM:SS instead of Rich's default H:MM:SS.

    The leading ``0:`` hour marker visually implies hour-scale waits, which
    these CLI requests never reach in practice. Minutes are allowed to grow
    past 59 rather than rolling into hours.
    """

    def render(self, task: Task) -> Text:
        elapsed = task.finished_time if task.finished else task.elapsed
        if elapsed is None:
            return Text("--:--", style="progress.elapsed")
        seconds = max(0, int(elapsed))
        minutes, secs = divmod(seconds, 60)
        return Text(f"{minutes:02d}:{secs:02d}", style="progress.elapsed")


def _progress_disabled(console: Console, disabled: bool) -> bool:
    if disabled:
        return True
    if not console.is_terminal:
        return True
    return any(os.environ.get(name) for name in _OPT_OUT_ENV_VARS)


class _ProgressHandle:
    """Caller-facing handle for updating the spinner message mid-flight."""

    def __init__(self, progress: Progress | None, task_id: TaskID | None) -> None:
        self._progress = progress
        self._task_id = task_id

    @property
    def is_active(self) -> bool:
        """True when a Rich Progress widget is actually backing this handle."""
        return self._progress is not None and self._task_id is not None

    def update(self, message: str) -> None:
        if self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, description=message)


@contextmanager
def request_progress(
    message: str,
    *,
    disabled: bool = False,
    console: Console | None = None,
) -> Iterator[_ProgressHandle]:
    """Render a stderr spinner with elapsed-time while the block runs.

    Args:
        message: Description shown next to the spinner (e.g. ``"Waiting for agent..."``).
        disabled: Force-disable the spinner regardless of TTY/env detection.
            Wire this to the command's ``--no-progress`` flag.
        console: Optional Rich Console override (mostly for tests). Defaults to
            a stderr console.

    Yields:
        A handle whose ``update(msg)`` swaps the spinner description while the
        block is still active. Useful for surfacing intermediate progress.
    """
    console = console if console is not None else Console(stderr=True)
    if _progress_disabled(console, disabled):
        yield _ProgressHandle(None, None)
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        _MinutesSecondsElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task_id = progress.add_task(message, total=None)
        yield _ProgressHandle(progress, task_id)
