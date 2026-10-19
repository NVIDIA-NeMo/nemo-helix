# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io

import pytest
from nemo_helix_plugin.cli_progress import _MinutesSecondsElapsedColumn, request_progress
from rich.console import Console


def _tty_console() -> Console:
    return Console(file=io.StringIO(), force_terminal=True, width=80, record=True)


def _pipe_console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False, width=80, record=True)


@pytest.fixture(autouse=True)
def _clear_opt_out_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure CI/NO_COLOR/NEMO_NO_PROGRESS don't leak in from the runner.

    GitHub Actions sets ``CI=true`` automatically, which would otherwise
    force the spinner off regardless of the test's intent.
    """
    for name in ("NO_COLOR", "CI", "NEMO_NO_PROGRESS"):
        monkeypatch.delenv(name, raising=False)


def test_creates_progress_widget_when_stderr_is_a_tty() -> None:
    """A TTY stderr should yield an active Rich Progress backing the handle.

    Tests the contract structurally rather than asserting on rendered text:
    transient Progress + Live's refresh thread make output-capture timing
    flaky in CI, and the value we actually want to guarantee is that the
    spinner *is* wired up when stderr supports it.
    """
    console = _tty_console()
    with request_progress("Waiting for agent...", console=console) as handle:
        assert handle.is_active
        handle.update("Still waiting...")


def test_disabled_flag_yields_inert_handle() -> None:
    console = _tty_console()
    with request_progress("Should not appear", disabled=True, console=console) as handle:
        assert not handle.is_active
    assert console.export_text() == ""


def test_non_tty_stderr_yields_inert_handle() -> None:
    console = _pipe_console()
    with request_progress("Should not appear", console=console) as handle:
        assert not handle.is_active
    assert console.export_text() == ""


@pytest.mark.parametrize("env_var", ["NO_COLOR", "CI", "NEMO_NO_PROGRESS"])
def test_opt_out_env_vars_yield_inert_handle(monkeypatch: pytest.MonkeyPatch, env_var: str) -> None:
    for name in ("NO_COLOR", "CI", "NEMO_NO_PROGRESS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(env_var, "1")
    console = _tty_console()
    with request_progress("Should not appear", console=console) as handle:
        assert not handle.is_active
    assert console.export_text() == ""


def test_handle_update_is_safe_when_disabled() -> None:
    console = _tty_console()
    with request_progress("Should not appear", disabled=True, console=console) as handle:
        handle.update("This must not raise")
    assert console.export_text() == ""


@pytest.mark.parametrize(
    "elapsed,expected",
    [
        (None, "--:--"),
        (0, "00:00"),
        (7, "00:07"),
        (65, "01:05"),
        # Minutes intentionally roll past 59 rather than spilling into an hours field.
        (3725, "62:05"),
    ],
)
def test_elapsed_column_renders_minutes_seconds(elapsed: float | None, expected: str) -> None:
    class _StubTask:
        def __init__(self, elapsed: float | None) -> None:
            self.elapsed = elapsed
            self.finished = False
            self.finished_time = None

    rendered = _MinutesSecondsElapsedColumn().render(_StubTask(elapsed))  # type: ignore[arg-type]
    assert rendered.plain == expected
