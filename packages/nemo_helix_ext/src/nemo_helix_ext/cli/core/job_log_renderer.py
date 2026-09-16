# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

from nemo_helix_plugin.jobs.schemas import PlatformJobLog
from rich.console import Console
from rich.text import Text


def render_job_logs(logs: Iterable[PlatformJobLog], *, console: Console | None = None) -> None:
    output = console or Console()
    for log in logs:
        render_log_line(
            output,
            timestamp=log.timestamp,
            step_id=log.job_step,
            task_id=log.job_task,
            message=log.message,
        )


def render_log_line(
    console: Console,
    *,
    timestamp: datetime | None,
    step_id: str | None,
    task_id: str | None,
    message: str,
) -> None:
    line = Text()
    line.append(f"[{_time_label(timestamp)}] ", style="dim")
    scope = _scope(step_id, task_id)
    if scope:
        line.append(f"{scope} | ", style="dim")
    line.append(message)
    console.print(line)


def _time_label(timestamp: datetime | None = None) -> str:
    value = timestamp.astimezone() if timestamp is not None else datetime.now()
    return value.strftime("%H:%M:%S")


def _scope(step_id: str | None, task_id: str | None) -> str:
    if step_id and task_id:
        return f"{step_id}/{task_id}"
    if step_id:
        return step_id
    if task_id:
        return task_id
    return ""
