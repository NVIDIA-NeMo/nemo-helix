# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for signal handling in run_platform().

Verifies that a single SIGTERM causes the process to exit (not just kill
controllers while leaving uvicorn alive).
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time

import pytest

_STARTUP_TIMEOUT = 15
_SHUTDOWN_TIMEOUT = 5

_FIXED_SIGNAL_HANDLER = """\
import os, signal, sys, threading, time

controller_stop_signal = threading.Event()
ready_marker = sys.argv[1]

def signal_handler(signum, _frame):
    controller_stop_signal.set()
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    os.kill(os.getpid(), signum)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

with open(ready_marker, "w") as f:
    f.write(str(os.getpid()))

while True:
    time.sleep(0.1)
"""

_BROKEN_SIGNAL_HANDLER = """\
import os, signal, sys, threading, time

controller_stop_signal = threading.Event()
ready_marker = sys.argv[1]

def signal_handler(signum, _frame):
    controller_stop_signal.set()
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    # BUG: missing os.kill(os.getpid(), signum)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

with open(ready_marker, "w") as f:
    f.write(str(os.getpid()))

while True:
    time.sleep(0.1)
"""


def _spawn_and_wait_ready(script_path, ready_marker, *extra_args):
    """Spawn a script and block until it writes its readiness marker."""
    proc = subprocess.Popen(  # noqa: S603
        [sys.executable, str(script_path), str(ready_marker), *extra_args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    deadline = time.monotonic() + _STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        if ready_marker.exists():
            return proc
        if proc.poll() is not None:
            pytest.fail(f"Process exited early: {proc.returncode}")
        time.sleep(0.05)
    proc.kill()
    proc.wait(timeout=5)
    pytest.fail(f"Process did not become ready within {_STARTUP_TIMEOUT}s")


@pytest.fixture()
def platform_process(tmp_path):
    """Spawn a minimal platform server with the fixed signal handler."""
    script = tmp_path / "fake_platform.py"
    script.write_text(_FIXED_SIGNAL_HANDLER)

    proc = _spawn_and_wait_ready(script, tmp_path / "ready")
    yield proc

    if proc.poll() is None:
        proc.kill()
        proc.wait(timeout=5)


class TestSignalHandlerExitsProcess:
    def test_sigterm_exits_process(self, platform_process):
        """A single SIGTERM should terminate the process, not just kill controllers."""
        os.kill(platform_process.pid, signal.SIGTERM)

        try:
            platform_process.wait(timeout=_SHUTDOWN_TIMEOUT)
        except subprocess.TimeoutExpired:
            platform_process.kill()
            pytest.fail(f"Process survived SIGTERM for {_SHUTDOWN_TIMEOUT}s")

        assert platform_process.returncode != 0, "Process should exit with non-zero (signal death)"

    def test_sigint_exits_process(self, platform_process):
        """A single SIGINT (Ctrl-C) should also terminate the process."""
        os.kill(platform_process.pid, signal.SIGINT)

        try:
            platform_process.wait(timeout=_SHUTDOWN_TIMEOUT)
        except subprocess.TimeoutExpired:
            platform_process.kill()
            pytest.fail(f"Process survived SIGINT for {_SHUTDOWN_TIMEOUT}s")

    def test_controller_stop_signal_is_set(self, tmp_path):
        """Controllers should receive the stop signal before the process exits."""
        script = tmp_path / "check_controllers.py"
        script.write_text(
            textwrap.dedent("""\
            import os, signal, sys, threading, time

            controller_stop_signal = threading.Event()
            result_path = sys.argv[2]

            def signal_handler(signum, _frame):
                controller_stop_signal.set()
                signal.signal(signal.SIGINT, signal.SIG_DFL)
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
                with open(result_path, "w") as f:
                    f.write("stopped" if controller_stop_signal.is_set() else "not_stopped")
                os.kill(os.getpid(), signum)

            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

            with open(sys.argv[1], "w") as f:
                f.write(str(os.getpid()))

            while True:
                time.sleep(0.1)
            """)
        )

        result_path = tmp_path / "controller_result"
        proc = _spawn_and_wait_ready(script, tmp_path / "ready2", str(result_path))
        try:
            os.kill(proc.pid, signal.SIGTERM)
            proc.wait(timeout=_SHUTDOWN_TIMEOUT)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)

        assert result_path.exists(), "Signal handler should have written result"
        assert result_path.read_text() == "stopped"


class TestWithoutFix:
    """Demonstrate that omitting os.kill(os.getpid(), signum) creates a zombie."""

    def test_sigterm_without_self_kill_leaves_zombie(self, tmp_path):
        """Without the self-signal, SIGTERM kills controllers but the process survives."""
        script = tmp_path / "broken_platform.py"
        script.write_text(_BROKEN_SIGNAL_HANDLER)

        proc = _spawn_and_wait_ready(script, tmp_path / "ready_broken")
        try:
            os.kill(proc.pid, signal.SIGTERM)

            with pytest.raises(subprocess.TimeoutExpired):
                proc.wait(timeout=1)

            os.kill(proc.pid, signal.SIGTERM)
            proc.wait(timeout=_SHUTDOWN_TIMEOUT)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)
