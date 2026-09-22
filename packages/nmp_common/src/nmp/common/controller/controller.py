# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Controller framework for background processes."""

import contextvars
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from logging import getLogger
from typing import Callable

from nmp.common.observability.context import AppContext, initialize_app_ctx

logger = getLogger(__name__)


class Heartbeat:
    """Timestamp of the last observed progress in a control loop.

    Lets :class:`Loop` tell a loop working through a large batch apart from one
    that is stuck, since a single iteration can outlast the poll interval.
    """

    def __init__(self) -> None:
        self._last = datetime.now(timezone.utc)

    def beat(self) -> None:
        self._last = datetime.now(timezone.utc)

    def last(self) -> datetime:
        return self._last


class HeartbeatMixin:
    """Lets a controller report progress while a single ``step()`` is still running.

    Mixed into controllers whose ``step()`` iterates a variable number of items.
    """

    # Class-level default so mixing this in never obliges a subclass to call
    # super().__init__(); emitting before attach_heartbeat() is a no-op.
    _heartbeat: Heartbeat | None = None

    def attach_heartbeat(self, heartbeat: Heartbeat) -> None:
        self._heartbeat = heartbeat

    def emit_heartbeat(self) -> None:
        """Report that a unit of work finished.

        Only call this once work has actually completed. Emitting on a timer, or
        around a call that may block, would report progress that did not happen
        and defeat the liveness check.
        """
        if self._heartbeat is not None:
            self._heartbeat.beat()


class Controller(ABC):
    """Step represents a function intended to be run in a loop."""

    @abstractmethod
    def step(self): ...

    @property
    def is_healthy(self) -> bool:
        """Check if the controller is healthy.

        Returns:
            True if the controller is healthy, False otherwise.
        """
        return True


class LoopWaiter(ABC):
    """
    Loop waiter waits in increments of sleep_secs.
    When wait is called, if it has been < sleep_secs since the last call, it will sleep the remaining time.
    If it has been >= sleep_secs since last call, it will return immediately.
    """

    @abstractmethod
    def wait(self): ...


class ProvidesLastExecutionTime(ABC):
    """ProvidesLastExecutionTime is an interface for objects providing a time of
    last successful execution.
    """

    @abstractmethod
    def last_execution_time(self) -> datetime: ...


class TrackLastExecutionTime(Controller, ProvidesLastExecutionTime):
    """Tracks when the wrapped controller last made progress.

    Entering ``step()`` counts as progress; a controller mixing in
    :class:`HeartbeatMixin` shares this wrapper's :class:`Heartbeat` so its
    mid-iteration progress lands on the same single timestamp.
    """

    def __init__(self, controller: Controller):
        self._internal = controller
        self._heartbeat = Heartbeat()
        if isinstance(controller, HeartbeatMixin):
            controller.attach_heartbeat(self._heartbeat)

    def last_execution_time(self) -> datetime:
        return self._heartbeat.last()

    def step(self):
        self._heartbeat.beat()
        self._internal.step()

    @property
    def is_healthy(self) -> bool:
        """Delegate health check to the wrapped controller."""
        return self._internal.is_healthy


class TimedLoopWaiter(LoopWaiter):
    def __init__(self, sleep_secs: float, stop_signal: threading.Event | None = None):
        self._sleep_secs = sleep_secs
        self._next_step = 0.0
        self._stop_signal = stop_signal

    @property
    def sleep_secs(self) -> float:
        return self._sleep_secs

    def wait(self):
        now = time.time()
        if self._next_step > now:
            sleep_duration = self._next_step - now
            if self._stop_signal is not None:
                # Use Event.wait() which returns immediately if stop signal is set
                self._stop_signal.wait(timeout=sleep_duration)
            else:
                time.sleep(sleep_duration)
            self._next_step = self._next_step + self._sleep_secs
        else:
            self._next_step = now + self._sleep_secs


class Loop(threading.Thread):
    """
    Loop is a loop that runs in a separate Thread. The contents of the 'step' function are called every
    iteration.
    """

    def __init__(
        self,
        waiter: LoopWaiter,
        controller: Controller,
        shutdown_func: Callable | None = None,
        stop_signal: threading.Event | None = None,
    ):
        threading.Thread.__init__(self)
        self._waiter = waiter
        self._internal = controller
        self._stop_signal = stop_signal if stop_signal is not None else threading.Event()
        self._shutdown_func = shutdown_func
        self._unhealthy_reason: str | None = None

        # Capture the creating thread's context so the loop runs with it. Deliberately not
        # `self._context`: Python 3.14 added that exact attribute to `threading.Thread` and enters
        # it in `_bootstrap_inner`, so assigning to it here made the thread enter this context and
        # then `run()` re-enter the *same* object -- `RuntimeError: cannot enter context: ... is
        # already entered`. 3.12 and 3.13 have no such attribute, which is why this only broke on
        # 3.14.
        self._loop_context = contextvars.copy_context()

    def run(self):
        self._loop_context.run(self._run_loop)

    def _run_loop(self):
        initialize_app_ctx(AppContext(service_name=self.name))
        try:
            while not self._stop_signal.is_set():
                self._waiter.wait()
                if self._stop_signal.is_set():
                    break

                try:
                    self._internal.step()
                except Exception as e:
                    logger.exception(f"Error: Control loop caught an exception: {e}")
        finally:
            if self._shutdown_func:
                try:
                    self._shutdown_func()
                except Exception as e:
                    logger.exception(f"Error during loop shutdown: {e}")

    def stop(self):
        self._stop_signal.set()

    @property
    def unhealthy_reason(self) -> str | None:
        """Why the most recent :attr:`is_healthy` evaluation failed, if it did."""
        return self._unhealthy_reason

    @property
    def is_healthy(self) -> bool:
        """Check if the internal controller is healthy.

        Returns:
            True if the thread is active AND last execution is recent (or controller is healthy), False otherwise.
        """
        # Thread must be alive
        if not self.is_alive():
            self._unhealthy_reason = "loop thread is not alive"
            logger.debug(f"Controller thread {self.name} is not alive")
            return False

        # Check if last execution time is within acceptable window
        if isinstance(self._internal, ProvidesLastExecutionTime) and isinstance(self._waiter, TimedLoopWaiter):
            last_execution = self._internal.last_execution_time()
            sleep_secs = self._waiter.sleep_secs
            now = datetime.now(timezone.utc)
            max_delay = sleep_secs * 3  # Allow 3 sleep windows before marking unhealthy
            time_since_last = (now - last_execution).total_seconds()
            if time_since_last > max_delay:
                self._unhealthy_reason = f"no progress for {time_since_last:.2f}s (max allowed: {max_delay:.2f}s)"
                logger.debug(
                    f"Controller thread {self.name} has not executed in {time_since_last:.2f}s "
                    f"(max allowed: {max_delay:.2f}s)"
                )
                return False

        if not self._internal.is_healthy:
            self._unhealthy_reason = "controller reported itself unhealthy"
            logger.debug(f"Controller thread {self.name} internal controller is unhealthy")
            return False

        self._unhealthy_reason = None
        return True
