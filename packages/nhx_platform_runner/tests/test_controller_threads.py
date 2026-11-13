# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for runner-owned controller thread health tracking."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from typing import cast

import pytest
from nhx.common.controller import ControllerManager, Loop
from nhx.platform_runner import controller_threads
from nhx.platform_runner.controller_threads import (
    join_and_untrack_runner_threads,
    start_controller_threads,
    start_sidecar_threads,
)

#: How long the test itself waits for something that should happen promptly before failing. Helpers
#: never time out (see :func:`_gate`), so this only bounds how long a broken test takes to fail.
_GUARD_S = 10.0
#: Events a helper thread blocks on, opened at teardown so a failed test leaves no thread blocked. Only
#: the test's own thread creates gates and opens them, so the list needs no lock.
_GATES: list[threading.Event] = []


def _gate() -> threading.Event:
    """An event a helper thread blocks on, untimed, until the test sets it.

    Only the test may end the wait: a helper that gave up on its own would exit while the test still
    expects it running, and the test would see it untracked. Call it from the test, before starting the
    helper that waits on it.
    """
    gate = threading.Event()
    _GATES.append(gate)
    return gate


class _HealthyLoop:
    name = ""
    is_healthy = True
    unhealthy_reason = None


@pytest.fixture(autouse=True)
def reset_controller_manager() -> Iterator[None]:
    ControllerManager._instance = None
    yield
    for gate in _GATES:
        gate.set()
    _GATES.clear()
    ControllerManager._instance = None


def test_controller_is_pending_until_run_func_registers_loop() -> None:
    manager = ControllerManager.get_instance()
    allow_registration = _gate()
    registered = threading.Event()
    stop_signal = _gate()

    def run(stop_signal: threading.Event) -> None:
        allow_registration.wait()
        manager.register("models_controller", cast(Loop, _HealthyLoop()))
        registered.set()
        stop_signal.wait()

    threads = start_controller_threads({"models": run}, stop_signal)
    try:
        assert manager.validate_all_healthy() == (False, {"models": False})

        allow_registration.set()
        assert registered.wait(timeout=_GUARD_S)
        assert manager.validate_all_healthy() == (True, {"models_controller": True})
    finally:
        stop_signal.set()
        for thread in threads:
            thread.join(timeout=_GUARD_S)


def test_controller_that_exits_without_shutdown_is_unhealthy(caplog: pytest.LogCaptureFixture) -> None:
    manager = ControllerManager.get_instance()
    stop_signal = _gate()

    with caplog.at_level(logging.ERROR, logger="nhx.platform_runner.controller_threads"):
        threads = start_controller_threads({"models": lambda _stop_signal: None}, stop_signal)
        for thread in threads:
            thread.join(timeout=_GUARD_S)

    assert manager.validate_all_healthy() == (False, {"models": False})
    assert "Controller 'models' exited unexpectedly" in caplog.text


def test_controller_system_exit_is_recorded_as_a_crash(caplog: pytest.LogCaptureFixture) -> None:
    manager = ControllerManager.get_instance()

    def exit_controller(_stop_signal: threading.Event) -> None:
        raise SystemExit(7)

    with caplog.at_level(logging.ERROR, logger="nhx.platform_runner.controller_threads"):
        threads = start_controller_threads({"models": exit_controller}, threading.Event())
        threads[0].join(timeout=_GUARD_S)

    assert manager.validate_all_healthy() == (False, {"models": False})
    assert "Controller 'models' crashed" in caplog.text


def test_controller_thread_name_does_not_duplicate_prefix() -> None:
    stop_signal = _gate()

    threads = start_controller_threads({"controller-models": lambda _stop_signal: None}, stop_signal)
    for thread in threads:
        thread.join(timeout=_GUARD_S)

    assert threads[0].name == "controller-models"


def test_required_sidecar_failure_is_unhealthy(caplog: pytest.LogCaptureFixture) -> None:
    manager = ControllerManager.get_instance()
    stop_signal = _gate()

    def crash(_stop_signal: threading.Event) -> None:
        raise ValueError("optional configuration is missing")

    with caplog.at_level(logging.ERROR, logger="nhx.platform_runner.controller_threads"):
        threads = start_sidecar_threads({"adapters": crash}, stop_signal)
        for thread in threads:
            thread.join(timeout=_GUARD_S)

    assert manager.validate_all_healthy() == (False, {"adapters": False})
    assert "Sidecar 'adapters' crashed" in caplog.text
    assert threads[0].name == "sidecar-adapters"


def test_self_tracking_sidecar_is_pending_before_its_thread_runs_any_code() -> None:
    """Expose a self-tracking sidecar before its thread is scheduled."""
    manager = ControllerManager.get_instance()
    stop_signal = _gate()
    thread_may_proceed = _gate()

    def run(stop_signal: threading.Event) -> None:
        # Hold the thread before its own registration call.
        thread_may_proceed.wait()
        stop_signal.wait()

    threads = start_sidecar_threads({"auth-proxy": run}, stop_signal)
    try:
        all_healthy, status = manager.validate_all_healthy()
        assert all_healthy is False
        assert status.get("auth-proxy") is False
    finally:
        thread_may_proceed.set()
        stop_signal.set()
        for thread in threads:
            thread.join(timeout=_GUARD_S)
        manager.stop_tracking_controller("auth-proxy")


def test_delayed_runner_exit_is_untracked_after_its_generation_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(controller_threads, "DEFAULT_RUNNER_JOIN_TIMEOUT_SECONDS", 0.05)
    manager = ControllerManager.get_instance()
    stop_signal = _gate()
    release = _gate()

    def run(_stop_signal: threading.Event) -> None:
        release.wait()

    threads = start_controller_threads({"models": run}, stop_signal)
    stop_signal.set()
    join_and_untrack_runner_threads(threads, {"models": threads[0]}, {"models"})

    assert manager.validate_all_healthy() == (False, {"models": False})
    with pytest.raises(RuntimeError, match="still stopping"):
        manager.await_controller_registration("models")

    release.set()
    deadline = time.monotonic() + _GUARD_S
    while manager.validate_all_healthy() != (True, {}) and time.monotonic() < deadline:
        time.sleep(0.01)

    assert manager.validate_all_healthy() == (True, {})


def test_join_timeout_override_remains_supported() -> None:
    manager = ControllerManager.get_instance()
    release = _gate()
    threads = start_controller_threads({"models": lambda _stop_signal: release.wait()}, threading.Event())

    started_at = time.monotonic()
    join_and_untrack_runner_threads(
        threads,
        {"models": threads[0]},
        {"models"},
        join_timeout=0.01,
    )

    assert time.monotonic() - started_at < 1
    assert manager.validate_all_healthy() == (False, {"models": False})
    release.set()
    threads[0].join(timeout=_GUARD_S)
    deadline = time.monotonic() + _GUARD_S
    while manager.validate_all_healthy() != (True, {}) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert manager.validate_all_healthy() == (True, {})


def test_delayed_runner_hands_cleanup_to_delayed_control_loop() -> None:
    manager = ControllerManager.get_instance()
    stop_signal = _gate()
    wrapper_started = threading.Event()
    loop_started = threading.Event()
    release_wrapper = _gate()
    release_loop = _gate()

    class _BlockingLoop(threading.Thread):
        name = ""
        is_healthy = True
        unhealthy_reason = None

        def run(self) -> None:
            loop_started.set()
            release_loop.wait()

    def run(_stop_signal: threading.Event) -> None:
        loop = cast(Loop, _BlockingLoop(daemon=True))
        manager.register("models_controller", loop)
        loop.start()
        wrapper_started.set()
        release_wrapper.wait()

    threads = start_controller_threads({"models": run}, stop_signal)
    assert wrapper_started.wait(timeout=_GUARD_S)
    assert loop_started.wait(timeout=_GUARD_S)

    join_and_untrack_runner_threads(
        threads,
        {"models": threads[0]},
        {"models"},
        join_timeout=0,
    )
    assert manager.validate_all_healthy() == (False, {"models": False, "models_controller": True})

    release_wrapper.set()
    threads[0].join(timeout=_GUARD_S)
    assert manager.validate_all_healthy() == (False, {"models": False, "models_controller": True})

    release_loop.set()
    deadline = time.monotonic() + _GUARD_S
    while manager.validate_all_healthy() != (True, {}) and time.monotonic() < deadline:
        time.sleep(0.01)

    assert manager.validate_all_healthy() == (True, {})


def test_dead_runner_with_live_control_loop_is_untracked_after_loop_exits() -> None:
    manager = ControllerManager.get_instance()
    stop_signal = _gate()
    loop_started = threading.Event()
    release_loop = _gate()

    class _BlockingLoop(threading.Thread):
        name = ""
        is_healthy = True
        unhealthy_reason = None

        def run(self) -> None:
            loop_started.set()
            release_loop.wait()

    def run(stop_signal: threading.Event) -> None:
        loop = cast(Loop, _BlockingLoop(daemon=True))
        manager.register("models_controller", loop)
        loop.start()
        stop_signal.wait()
        # Simulate a controller wrapper whose own shutdown deadline expires
        # before its owned control-loop thread has stopped.

    threads = start_controller_threads({"models": run}, stop_signal)
    assert loop_started.wait(timeout=_GUARD_S)
    stop_signal.set()
    threads[0].join(timeout=_GUARD_S)

    join_and_untrack_runner_threads(threads, {"models": threads[0]}, {"models"})

    assert manager.validate_all_healthy() == (False, {"models": False, "models_controller": True})
    with pytest.raises(RuntimeError, match="still stopping"):
        manager.await_controller_registration("models")

    release_loop.set()
    deadline = time.monotonic() + _GUARD_S
    while manager.validate_all_healthy() != (True, {}) and time.monotonic() < deadline:
        time.sleep(0.01)

    assert manager.validate_all_healthy() == (True, {})
    assert manager.await_controller_registration("models") > 1


def test_loop_exit_between_cleanup_and_watcher_does_not_leave_stopping_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = ControllerManager.get_instance()
    stop_signal = _gate()
    loop_started = threading.Event()
    release_loop = _gate()
    owned_loop: threading.Thread | None = None

    class _BlockingLoop(threading.Thread):
        name = ""
        is_healthy = True
        unhealthy_reason = None

        def run(self) -> None:
            loop_started.set()
            release_loop.wait()

    def run(stop_signal: threading.Event) -> None:
        nonlocal owned_loop
        owned_loop = _BlockingLoop(daemon=True)
        manager.register("models_controller", cast(Loop, owned_loop))
        owned_loop.start()
        stop_signal.wait()

    threads = start_controller_threads({"models": run}, stop_signal)
    assert loop_started.wait(timeout=_GUARD_S)
    stop_signal.set()
    threads[0].join(timeout=_GUARD_S)

    def exit_before_snapshot(*_args: object, **_kwargs: object) -> bool:
        release_loop.set()
        assert owned_loop is not None
        owned_loop.join(timeout=_GUARD_S)
        return False

    monkeypatch.setattr(manager, "watch_controller_loops_exit", exit_before_snapshot)

    join_and_untrack_runner_threads(threads, {"models": threads[0]}, {"models"})

    assert manager.validate_all_healthy() == (True, {})


def test_existing_stopping_owner_keeps_control_of_delayed_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = ControllerManager.get_instance()
    release = _gate()
    threads = start_sidecar_threads({"auth-proxy": lambda _stop: release.wait()}, threading.Event())
    thread = threads[0]
    assert isinstance(thread, controller_threads._RunnerThread)
    manager.mark_controller_stopping("auth-proxy", thread.generation)
    watcher_called = False

    def record_watcher(*_args: object, **_kwargs: object) -> None:
        nonlocal watcher_called
        watcher_called = True

    monkeypatch.setattr(manager, "watch_delayed_exit", record_watcher)

    join_and_untrack_runner_threads(
        threads,
        {"auth-proxy": thread},
        {"auth-proxy"},
        join_timeout=0,
    )

    assert watcher_called is False
    assert manager.validate_all_healthy() == (False, {"auth-proxy": False})
    release.set()
    thread.join(timeout=_GUARD_S)


def test_thread_start_failure_rolls_back_already_started_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = ControllerManager.get_instance()
    stop_signal = _gate()
    first_stopped = threading.Event()

    def first_run(stop_signal: threading.Event) -> None:
        manager.register("first_loop", cast(Loop, _HealthyLoop()))
        stop_signal.wait()
        first_stopped.set()

    original_start = controller_threads._RunnerThread.start

    def fail_second_start(thread: controller_threads._RunnerThread) -> None:
        if thread.component_name == "second":
            raise RuntimeError("thread start failed")
        original_start(thread)

    monkeypatch.setattr(controller_threads._RunnerThread, "start", fail_second_start)

    with pytest.raises(RuntimeError, match="thread start failed"):
        start_controller_threads(
            {
                "first": first_run,
                "second": lambda _stop_signal: None,
            },
            stop_signal,
        )

    assert first_stopped.wait(timeout=_GUARD_S)
    assert manager.get_all_loops() == {}
    assert manager.validate_all_healthy() == (False, {"second": False})
