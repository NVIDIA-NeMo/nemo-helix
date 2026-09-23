# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the control loop liveness model.

The liveness check is pure logic over a timestamp, the poll interval, and whether
the loop thread is alive, so these tests drive the clock rather than sleep on it.
"""

import threading
from datetime import datetime, timedelta, timezone

import time_machine
from nhx.common.controller import (
    Controller,
    Heartbeat,
    HeartbeatMixin,
    Loop,
    TimedLoopWaiter,
    TrackLastExecutionTime,
)

# Matches the models controller's default, so the 3x window below is 15s.
INTERVAL_SECONDS = 5.0
BUDGET = timedelta(seconds=INTERVAL_SECONDS * 3)
START = datetime(2026, 1, 1, tzinfo=timezone.utc)


class _PlainController(Controller):
    """Controller that reports no progress beyond entering step()."""

    def __init__(self) -> None:
        self.steps = 0

    def step(self):
        self.steps += 1


class _ReportingController(HeartbeatMixin, Controller):
    """Controller that reports progress as it works through a batch."""

    def __init__(self) -> None:
        self.processed = 0

    def step(self):
        self.process_one()

    def process_one(self) -> None:
        self.processed += 1
        self.emit_heartbeat()


def _build_loop(controller: Controller, *, alive: bool = True) -> tuple[Loop, TrackLastExecutionTime]:
    """Wrap a controller in a Loop without starting a thread.

    Thread liveness is a separate branch of the health check with its own test, so
    it is pinned here to keep the staleness assertions unambiguous.
    """
    stop_signal = threading.Event()
    wrapper = TrackLastExecutionTime(controller)
    loop = Loop(TimedLoopWaiter(INTERVAL_SECONDS, stop_signal=stop_signal), wrapper, stop_signal=stop_signal)
    loop.name = "test-loop"
    if alive:
        loop.is_alive = lambda: True  # type: ignore[method-assign]
    return loop, wrapper


def test_heartbeat_beat_advances_last():
    with time_machine.travel(START, tick=False) as clock:
        heartbeat = Heartbeat()
        assert heartbeat.last() == START

        clock.shift(timedelta(seconds=30))
        heartbeat.beat()

        assert heartbeat.last() == START + timedelta(seconds=30)


def test_wrapper_shares_its_heartbeat_with_a_reporting_controller():
    with time_machine.travel(START, tick=False) as clock:
        controller = _ReportingController()
        wrapper = TrackLastExecutionTime(controller)
        assert wrapper.last_execution_time() == START

        clock.shift(timedelta(seconds=10))
        controller.emit_heartbeat()

        assert wrapper.last_execution_time() == START + timedelta(seconds=10)


def test_emit_heartbeat_before_attach_is_a_noop():
    # Never wrapped, so no heartbeat is attached.
    _ReportingController().emit_heartbeat()


def test_controller_without_the_mixin_still_tracks_step_entry():
    with time_machine.travel(START, tick=False) as clock:
        controller = _PlainController()
        wrapper = TrackLastExecutionTime(controller)

        clock.shift(timedelta(seconds=10))
        wrapper.step()

        assert controller.steps == 1
        assert wrapper.last_execution_time() == START + timedelta(seconds=10)


def test_loop_stays_healthy_while_a_long_step_reports_progress():
    """A step far exceeding its interval stays healthy as long as it progresses.

    This is the case a single heartbeat per step cannot express: the step has not
    returned, yet the loop is demonstrably working.
    """
    with time_machine.travel(START, tick=False) as clock:
        controller = _ReportingController()
        loop, _ = _build_loop(controller)

        # Work through a batch that takes several times the poll interval, reporting
        # progress at intervals shorter than the window.
        for _ in range(12):
            clock.shift(BUDGET / 2)
            controller.process_one()
            assert loop.is_healthy
            assert loop.unhealthy_reason is None

        assert controller.processed == 12


def test_loop_goes_unhealthy_when_a_step_reports_no_progress():
    """A step that stops making progress is still detected."""
    with time_machine.travel(START, tick=False) as clock:
        loop, _ = _build_loop(_ReportingController())
        assert loop.is_healthy

        # Just inside the window.
        clock.shift(BUDGET - timedelta(seconds=1))
        assert loop.is_healthy

        # And past it.
        clock.shift(timedelta(seconds=2))
        assert not loop.is_healthy
        assert loop.unhealthy_reason is not None
        assert "no progress" in loop.unhealthy_reason


def test_loop_recovers_once_progress_resumes_without_the_step_returning():
    with time_machine.travel(START, tick=False) as clock:
        controller = _ReportingController()
        loop, _ = _build_loop(controller)

        clock.shift(BUDGET + timedelta(seconds=1))
        assert not loop.is_healthy

        controller.process_one()

        assert loop.is_healthy
        assert loop.unhealthy_reason is None


def test_loop_reports_reason_when_thread_is_not_alive():
    # Built but never started, so the real is_alive() is False.
    loop, _ = _build_loop(_PlainController(), alive=False)

    assert not loop.is_healthy
    assert loop.unhealthy_reason == "loop thread is not alive"


def test_loop_reports_reason_when_controller_declares_itself_unhealthy():
    class _SelfUnhealthy(HeartbeatMixin, Controller):
        def step(self):
            self.emit_heartbeat()

        @property
        def is_healthy(self) -> bool:
            return False

    with time_machine.travel(START, tick=False):
        loop, _ = _build_loop(_SelfUnhealthy())

        assert not loop.is_healthy
        assert loop.unhealthy_reason == "controller reported itself unhealthy"
