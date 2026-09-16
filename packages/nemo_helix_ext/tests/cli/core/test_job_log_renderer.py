# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime, timezone
from io import StringIO

from nemo_helix_ext.cli.core.job_log_renderer import render_job_logs
from nemo_helix_plugin.jobs.schemas import PlatformJobLog
from rich.console import Console


def _console_pair() -> tuple[Console, StringIO]:
    output = StringIO()
    return Console(file=output, force_terminal=False, color_system=None, width=120), output


def test_render_job_logs_prints_logs_in_order() -> None:
    console, output = _console_pair()
    timestamp = datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc)

    render_job_logs(
        [
            PlatformJobLog(timestamp=timestamp, job="job-a", job_step="step-a", job_task="task-a", message="started"),
            PlatformJobLog(timestamp=timestamp, job="job-a", job_step="step-a", job_task="", message="running"),
        ],
        console=console,
    )

    rendered = output.getvalue()
    expected_time = timestamp.astimezone().strftime("%H:%M:%S")
    assert f"[{expected_time}] step-a/task-a | started" in rendered
    assert f"[{expected_time}] step-a | running" in rendered


def test_render_job_logs_prints_nothing_for_empty_logs() -> None:
    console, output = _console_pair()

    render_job_logs([], console=console)

    assert output.getvalue() == ""
