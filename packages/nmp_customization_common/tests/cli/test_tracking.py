# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the post-submit tracking message and the --wait/--watch follow."""

from __future__ import annotations

import io
from collections.abc import Iterable, Mapping
from contextlib import AbstractContextManager

import pytest
from nemo_platform_plugin.cli_progress import ProgressHandle
from nemo_platform_plugin.jobs.watch_types import (
    JobLogEvent,
    JobStatusEvent,
    JobWarningEvent,
    JobWatchEvent,
    JobWatchTimeoutError,
)
from nmp.customization_common.cli import tracking
from nmp.customization_common.cli.tracking import (
    FollowResult,
    print_tracking_instructions,
)
from rich.console import Console


def _console() -> tuple[Console, io.StringIO]:
    buffer = io.StringIO()
    # width keeps assertions stable; force_terminal stays off so output is plain.
    return Console(file=buffer, width=100, no_color=True), buffer


def _status(
    status: str,
    *,
    terminal: bool,
    successful: bool | None,
    error_details: Mapping[str, object] | None = None,
) -> JobStatusEvent:
    return JobStatusEvent(
        kind="status",
        job_name="automodel-1a2b3c",
        status=status,
        status_details={},
        terminal=terminal,
        successful=successful,
        error_details=error_details,
    )


def _follow(
    events: Iterable[JobWatchEvent],
    *,
    include_logs: bool = False,
    terminal: bool = False,
) -> tuple[FollowResult, str]:
    """Drive the renderer. *terminal* turns the spinner on, as on a real TTY."""
    buffer = io.StringIO()
    console = Console(file=buffer, width=100, no_color=True, force_terminal=terminal or None)
    result = tracking.follow_events(
        events,
        job_name="automodel-1a2b3c",
        print_logs=include_logs,
        console=console,
    )
    return result, buffer.getvalue()


class TestTrackingInstructions:
    def test_names_the_job_backend_and_workspace(self) -> None:
        console, buffer = _console()
        print_tracking_instructions(
            backend="automodel",
            job_name="automodel-1a2b3c",
            workspace="acme",
            console=console,
        )
        output = buffer.getvalue()
        assert "Submitted automodel job automodel-1a2b3c to workspace acme." in output

    def test_gives_the_exact_tracking_commands(self) -> None:
        console, buffer = _console()
        print_tracking_instructions(
            backend="rl",
            job_name="rl-99",
            workspace="default",
            console=console,
        )
        output = buffer.getvalue()
        assert "nemo jobs watch rl-99" in output
        assert "nemo jobs get-status rl-99" in output

    def test_does_not_suggest_flags_for_a_job_already_submitted(self) -> None:
        """--wait and --watch belong on submit --help; they cannot apply to this job now."""
        console, buffer = _console()
        print_tracking_instructions(
            backend="unsloth",
            job_name="unsloth-7",
            workspace="default",
            console=console,
        )
        output = buffer.getvalue()
        assert "--wait" not in output
        assert "--watch" not in output


class TestFollowJob:
    def test_terminal_success_reports_completion(self) -> None:
        events = [
            _status("created", terminal=False, successful=None),
            _status("running", terminal=False, successful=None),
            _status("completed", terminal=True, successful=True),
        ]
        result, output = _follow(events)
        assert result is FollowResult.SUCCEEDED
        assert "completed" in output

    def test_terminal_failure_reports_the_status_and_details(self) -> None:
        events = [
            _status("running", terminal=False, successful=None),
            _status("error", terminal=True, successful=False, error_details={"reason": "out of memory"}),
        ]
        result, output = _follow(events)
        assert result is FollowResult.FAILED
        assert "ended with status error" in output
        assert "reason: out of memory" in output

    def test_cancelled_is_a_failure(self) -> None:
        events = [_status("cancelled", terminal=True, successful=False)]
        result, output = _follow(events)
        assert result is FollowResult.FAILED
        assert "cancelled" in output

    def test_progress_at_100_is_not_treated_as_done(self) -> None:
        """A Customizer job keeps uploading after its steps report complete."""
        events = [
            JobStatusEvent(
                kind="status",
                job_name="automodel-1a2b3c",
                status="active",
                status_details={"phase": "training", "progress_pct": 100},
                terminal=False,
                successful=None,
            ),
        ]
        result, output = _follow(events)
        assert result is FollowResult.FAILED
        assert "before it reached a final status" in output

    def test_watch_prints_logs(self) -> None:
        events = [
            JobLogEvent(
                kind="log",
                job_name="automodel-1a2b3c",
                timestamp=None,
                step_id="train",
                task_id=None,
                message="step 1 loss 0.5",
            ),
            _status("completed", terminal=True, successful=True),
        ]
        result, output = _follow(events, include_logs=True)
        assert result is FollowResult.SUCCEEDED
        assert "step 1 loss 0.5" in output

    def test_wait_does_not_print_logs(self) -> None:
        events = [
            JobLogEvent(
                kind="log",
                job_name="automodel-1a2b3c",
                timestamp=None,
                step_id="train",
                task_id=None,
                message="step 1 loss 0.5",
            ),
            _status("completed", terminal=True, successful=True),
        ]
        result, output = _follow(events)
        assert result is FollowResult.SUCCEEDED
        assert "step 1 loss 0.5" not in output

    def test_warnings_are_surfaced_without_ending_the_watch(self) -> None:
        events = [
            JobWarningEvent(kind="warning", job_name="automodel-1a2b3c", message="log drain retried"),
            _status("completed", terminal=True, successful=True),
        ]
        result, output = _follow(events)
        assert result is FollowResult.SUCCEEDED
        assert "log drain retried" in output

    def test_timeout_is_a_failure(self) -> None:
        def _events() -> Iterable[JobWatchEvent]:
            yield _status("running", terminal=False, successful=None)
            raise JobWatchTimeoutError("timed out waiting for job")

        result, output = _follow(_events())
        assert result is FollowResult.FAILED
        assert "timed out waiting for job" in output

    def test_interrupt_says_the_job_is_still_running(self) -> None:
        def _events() -> Iterable[JobWatchEvent]:
            yield _status("running", terminal=False, successful=None)
            raise KeyboardInterrupt

        result, output = _follow(_events())
        assert result is FollowResult.INTERRUPTED
        assert "still running" in output


class TestProgressDisabling:
    """The spinner must not draw when stderr is not a terminal, or in CI."""

    def test_spinner_is_inactive_off_a_tty(self) -> None:
        console, _ = _console()
        with tracking.request_progress("waiting", console=console) as progress:
            assert not progress.is_active

    def test_spinner_is_inactive_in_ci(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CI", "true")
        console = Console(file=io.StringIO(), width=100, force_terminal=True)
        with tracking.request_progress("waiting", console=console) as progress:
            assert not progress.is_active

    def test_status_updates_still_drive_the_result_without_a_spinner(self) -> None:
        result, _ = _follow([_status("completed", terminal=True, successful=True)])
        assert result is FollowResult.SUCCEEDED


class TestLiveness:
    """A job can sit in one phase for minutes with no logs and no status change."""

    @pytest.mark.parametrize("print_logs", [False, True], ids=["wait", "watch"])
    def test_spinner_runs_in_both_modes(self, monkeypatch: pytest.MonkeyPatch, print_logs: bool) -> None:
        """Regression: --watch printed one status line, then went silent during init."""
        started: list[str] = []
        real = tracking.request_progress

        def spy(
            message: str,
            *,
            disabled: bool = False,
            console: Console | None = None,
        ) -> AbstractContextManager[ProgressHandle]:
            started.append(message)
            return real(message, disabled=disabled, console=console)

        monkeypatch.setattr(tracking, "request_progress", spy)
        result, _ = _follow([_status("completed", terminal=True, successful=True)], include_logs=print_logs)

        assert result is FollowResult.SUCCEEDED
        assert started, "both --wait and --watch need the spinner to show the job is alive"

    def test_logs_still_reach_the_console_alongside_the_spinner(self) -> None:
        events = [
            _status("init", terminal=False, successful=None),
            JobLogEvent(
                kind="log",
                job_name="automodel-1a2b3c",
                timestamp=None,
                step_id="train",
                task_id=None,
                message="pulling image",
            ),
            _status("completed", terminal=True, successful=True),
        ]
        result, output = _follow(events, include_logs=True, terminal=True)
        assert result is FollowResult.SUCCEEDED
        assert "pulling image" in output

    def test_log_lines_with_brackets_survive(self) -> None:
        """Job logs are data. Rich markup would eat 'list[int]' down to 'list'."""
        events = [
            JobLogEvent(
                kind="log",
                job_name="automodel-1a2b3c",
                timestamp=None,
                step_id="train",
                task_id=None,
                message="TypeError: expected list[int], got [1, 2]",
            ),
            _status("completed", terminal=True, successful=True),
        ]
        _, output = _follow(events, include_logs=True)
        assert "expected list[int], got [1, 2]" in output

    def test_error_details_with_brackets_survive(self) -> None:
        events = [
            _status(
                "error",
                terminal=True,
                successful=False,
                error_details={"reason": "bad shape [4, 8]"},
            )
        ]
        _, output = _follow(events)
        assert "bad shape [4, 8]" in output

    def test_status_changes_are_printed_when_the_spinner_is_off(self) -> None:
        """Off a TTY there is no spinner, so each status change is left as a line."""
        events = [
            _status("init", terminal=False, successful=None),
            _status("running", terminal=False, successful=None),
            _status("completed", terminal=True, successful=True),
        ]
        _, output = _follow(events)
        assert "automodel-1a2b3c: init" in output
        assert "automodel-1a2b3c: running" in output
