# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thread lifecycle helpers for runner-owned controllers and sidecars."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterable, Mapping

from nhx.common.controller import ControllerManager
from nhx.platform_runner.loader import ControllerRunFunc

logger = logging.getLogger(__name__)

# Backward-compatible shared timeout and join override. New runner-owned
# shutdown paths use the component-specific defaults below when no override is
# supplied.
RUNNER_JOIN_TIMEOUT_SECONDS = 16.0
DEFAULT_RUNNER_JOIN_TIMEOUT_SECONDS = 10.0
RUNNER_JOIN_TIMEOUT_SECONDS_BY_NAME: dict[str, float] = {
    # Covers auth-proxy's nested Uvicorn and health-loop shutdown budgets.
    "auth-proxy": RUNNER_JOIN_TIMEOUT_SECONDS,
}


class _RunnerThread(threading.Thread):
    """Thread carrying the lifecycle generation it owns."""

    def __init__(
        self,
        *,
        target: Callable[[], None],
        name: str,
        component_name: str,
        generation: int,
    ) -> None:
        super().__init__(target=target, name=name, daemon=True)
        self.component_name = component_name
        self.generation = generation


def start_controller_threads(
    controller_run_funcs: dict[str, ControllerRunFunc],
    stop_signal: threading.Event,
) -> list[threading.Thread]:
    """Start controllers while making pre-registration failures observable."""
    return _start_runner_threads(controller_run_funcs, stop_signal, kind="Controller", thread_prefix="controller")


def start_sidecar_threads(
    sidecar_run_funcs: dict[str, ControllerRunFunc],
    stop_signal: threading.Event,
) -> list[threading.Thread]:
    """Start required sidecars with the same health contract as controllers."""
    return _start_runner_threads(sidecar_run_funcs, stop_signal, kind="Sidecar", thread_prefix="sidecar")


def _start_runner_threads(
    run_funcs: dict[str, ControllerRunFunc],
    stop_signal: threading.Event,
    *,
    kind: str,
    thread_prefix: str,
) -> list[threading.Thread]:
    manager = ControllerManager.get_instance()
    threads: list[threading.Thread] = []

    for name, run_func in run_funcs.items():
        try:
            generation = manager.await_controller_registration(name)
        except Exception as error:
            logger.error("%s '%s' failed to register for startup: %s", kind, name, error)
            _abort_runner_batch(manager, stop_signal, threads)
            raise

        def run_tracked(
            run_func: ControllerRunFunc = run_func,
            name: str = name,
            generation: int = generation,
        ) -> None:
            with manager.controller_registration_context(name, generation):
                try:
                    run_func(stop_signal)
                except BaseException as error:
                    manager.mark_controller_failed(name, generation, reason=f"{kind.lower()} crashed: {error}")
                    logger.exception("%s '%s' crashed", kind, name)
                    return

                if not stop_signal.is_set():
                    manager.mark_controller_failed(name, generation, reason=f"{kind.lower()} exited unexpectedly")
                    logger.error("%s '%s' exited unexpectedly", kind, name)

        thread_name = name if name.startswith(f"{thread_prefix}-") else f"{thread_prefix}-{name}"
        thread = _RunnerThread(
            target=run_tracked,
            name=thread_name,
            component_name=name,
            generation=generation,
        )
        try:
            thread.start()
        except Exception as error:
            manager.mark_controller_failed(name, generation, reason=f"{kind.lower()} thread failed to start: {error}")
            _abort_runner_batch(manager, stop_signal, threads)
            raise
        threads.append(thread)

    return threads


def _abort_runner_batch(
    manager: ControllerManager,
    stop_signal: threading.Event,
    threads: list[threading.Thread],
) -> None:
    """Tear down already-started threads in this batch after a startup failure."""
    stop_signal.set()
    started_by_name = {started.component_name: started for started in threads if isinstance(started, _RunnerThread)}
    join_and_untrack_runner_threads(threads, started_by_name, started_by_name)


def join_and_untrack_runner_threads(
    threads: list[threading.Thread],
    thread_by_name: Mapping[str, threading.Thread],
    names: Iterable[str],
    *,
    join_timeout: float | None = None,
) -> None:
    """Join runner threads using component-specific deadlines.

    A delayed cleanup watcher owns the same generation as its thread, so it
    cannot erase health state belonging to a later restart.

    ``join_timeout`` preserves the previous API as an explicit shared timeout;
    when omitted, each component uses its configured shutdown budget.
    """
    started_waiting = time.monotonic()
    pending = list(threads)
    while pending:
        now = time.monotonic()
        still_pending = []
        for thread in pending:
            # RUNNER_JOIN_TIMEOUT_SECONDS_BY_NAME is keyed by component name
            # (e.g. "auth-proxy"), not by thread name (e.g. "sidecar-auth-proxy") —
            # fall back to the thread name only when no component name is known,
            # so a future non-_RunnerThread caller doesn't silently lose a
            # component's configured shutdown budget.
            component_name = getattr(thread, "component_name", thread.name)
            timeout = (
                join_timeout
                if join_timeout is not None
                else RUNNER_JOIN_TIMEOUT_SECONDS_BY_NAME.get(
                    component_name,
                    DEFAULT_RUNNER_JOIN_TIMEOUT_SECONDS,
                )
            )
            if thread.is_alive() and now - started_waiting < timeout:
                still_pending.append(thread)
        pending = still_pending
        if pending:
            time.sleep(0.05)

    manager = ControllerManager.get_instance()
    for name in names:
        thread = thread_by_name.get(name)
        if thread is None:
            continue
        generation = thread.generation if isinstance(thread, _RunnerThread) else None
        if thread.is_alive():
            logger.warning("Runner thread %s did not finish in time", thread.name)
            if generation is not None and manager.mark_controller_stopping(name, generation):
                manager.watch_delayed_exit(thread, name, generation)
            continue
        if manager.stop_tracking_controller(name, generation):
            continue
        if generation is not None and manager.mark_controller_stopping(name, generation):
            if not manager.watch_controller_loops_exit(name, generation):
                # The last loop can exit after stop_tracking_controller saw it
                # alive but before the delayed-loop watcher snapshots it.
                manager.stop_tracking_controller(name, generation, allow_stopping=True)
