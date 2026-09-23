# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ControllerManager health checks and Loop shutdown behavior."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import cast

import pytest
from nhx.common.controller import Controller, Loop, TimedLoopWaiter, TrackLastExecutionTime
from nhx.common.controller.controller_manager import ControllerManager


class MockHealthyController(Controller):
    """Mock controller that is always healthy."""

    def step(self):
        """Required abstract method implementation."""
        pass

    @property
    def is_healthy(self) -> bool:
        return True


class MockUnhealthyController(Controller):
    """Mock controller that is always unhealthy."""

    def step(self):
        """Required abstract method implementation."""
        pass

    @property
    def is_healthy(self) -> bool:
        return False


class MockControllerNoHealth(Controller):
    """Mock controller without is_healthy property."""

    def step(self):
        """Required abstract method implementation."""
        pass


class SlowController(Controller):
    """Controller that tracks step execution."""

    def __init__(self):
        self.step_count = 0
        self.stepped = threading.Event()

    def step(self):
        self.step_count += 1
        self.stepped.set()


class FailingController(Controller):
    """Controller whose step always raises."""

    def __init__(self):
        self.stepped = threading.Event()

    def step(self):
        self.stepped.set()
        raise RuntimeError("step failed")


@pytest.fixture(autouse=True)
def clear_singleton():
    """Clear the singleton instance before each test."""
    ControllerManager._instance = None
    yield
    ControllerManager._instance = None


def test_singleton_initialization_is_thread_safe(monkeypatch: pytest.MonkeyPatch):
    """Concurrent first access must return the same manager instance."""
    original_new = ControllerManager.__new__

    def slow_new(cls):
        time.sleep(0.01)
        return original_new(cls)

    monkeypatch.setattr(ControllerManager, "__new__", slow_new)
    with ThreadPoolExecutor(max_workers=16) as executor:
        managers = list(executor.map(lambda _: ControllerManager.get_instance(), range(16)))

    assert all(manager is managers[0] for manager in managers)


def test_all_healthy():
    """Test validation when all loops are healthy."""
    manager = ControllerManager.get_instance()
    controller1 = TrackLastExecutionTime(MockHealthyController())
    controller2 = TrackLastExecutionTime(MockHealthyController())
    # Use short interval for fast tests - we only need threads to be alive, not timing behavior
    loop1 = Loop(TimedLoopWaiter(0.01), controller1)
    loop2 = Loop(TimedLoopWaiter(0.01), controller2)
    manager.register("healthy1", loop1)
    manager.register("healthy2", loop2)

    # Start the threads so they are alive for health checks
    loop1.start()
    loop2.start()

    try:
        all_healthy, status = manager.validate_all_healthy()
        assert all_healthy is True
        assert status == {"healthy1": True, "healthy2": True}
    finally:
        # Clean up threads
        loop1.stop()
        loop2.stop()
        loop1.join(timeout=0.1)
        loop2.join(timeout=0.1)


def test_mixed_health():
    """Test validation with mix of healthy and unhealthy loops."""
    manager = ControllerManager.get_instance()
    healthy_controller = TrackLastExecutionTime(MockHealthyController())
    unhealthy_controller = TrackLastExecutionTime(MockUnhealthyController())
    # Use short interval for fast tests - we only need threads to be alive, not timing behavior
    healthy_loop = Loop(TimedLoopWaiter(0.01), healthy_controller)
    unhealthy_loop = Loop(TimedLoopWaiter(0.01), unhealthy_controller)
    manager.register("healthy", healthy_loop)
    manager.register("unhealthy", unhealthy_loop)

    # Start the threads so they are alive for health checks
    healthy_loop.start()
    unhealthy_loop.start()

    try:
        all_healthy, status = manager.validate_all_healthy()
        assert all_healthy is False
        assert status == {"healthy": True, "unhealthy": False}
    finally:
        # Clean up threads
        healthy_loop.stop()
        unhealthy_loop.stop()
        healthy_loop.join(timeout=0.1)
        unhealthy_loop.join(timeout=0.1)


def test_no_health_property():
    """Test validation treats loops with controllers without is_healthy as healthy."""
    manager = ControllerManager.get_instance()
    controller_no_health = TrackLastExecutionTime(MockControllerNoHealth())
    # Use short interval for fast tests - we only need threads to be alive, not timing behavior
    loop_no_health = Loop(TimedLoopWaiter(0.01), controller_no_health)
    manager.register("no_health", loop_no_health)

    # Start the thread so it is alive for health checks
    loop_no_health.start()

    try:
        all_healthy, status = manager.validate_all_healthy()
        assert all_healthy is True
        assert status == {"no_health": True}
    finally:
        # Clean up thread
        loop_no_health.stop()
        loop_no_health.join(timeout=0.1)


def test_registered_object_without_health_attribute_keeps_legacy_healthy_default():
    manager = ControllerManager.get_instance()

    class LoopWithoutHealth:
        name = ""

    manager.register("legacy", cast(Loop, LoopWithoutHealth()))

    assert manager.validate_all_healthy() == (True, {"legacy": True})


def test_detailed_false():
    """Test validation with detailed=False returns empty status dict."""
    manager = ControllerManager.get_instance()
    unhealthy_controller = TrackLastExecutionTime(MockUnhealthyController())
    # Use short interval for fast tests - we only need threads to be alive, not timing behavior
    unhealthy_loop = Loop(TimedLoopWaiter(0.01), unhealthy_controller)
    manager.register("unhealthy", unhealthy_loop)

    # Start the thread so it is alive for health checks
    unhealthy_loop.start()

    try:
        all_healthy, status = manager.validate_all_healthy(detailed=False)
        assert all_healthy is False
        assert status == {}
    finally:
        # Clean up thread
        unhealthy_loop.stop()
        unhealthy_loop.join(timeout=0.1)


def test_awaited_controller_is_unhealthy_until_it_registers_a_loop():
    manager = ControllerManager.get_instance()
    manager.await_controller_registration("models")

    assert manager.validate_all_healthy() == (False, {"models": False})

    loop = _ToggleLoop(healthy=True)
    with manager.controller_registration_context("models"):
        manager.register("models_controller", loop)

    assert manager.validate_all_healthy() == (True, {"models_controller": True})


@pytest.mark.parametrize(
    ("state", "expected_message"),
    [
        ("starting", "already starting"),
        ("running", "already running"),
    ],
)
def test_duplicate_start_cannot_supersede_active_generation(state: str, expected_message: str):
    manager = ControllerManager.get_instance()
    generation = manager.await_controller_registration("models")
    if state == "running":
        with manager.controller_registration_context("models", generation):
            manager.register("models_controller", _ToggleLoop(healthy=True))

    with pytest.raises(RuntimeError, match=expected_message):
        manager.await_controller_registration("models")

    with manager.controller_registration_context("models", generation):
        if state == "starting":
            manager.register("models_controller", _ToggleLoop(healthy=True))

    assert manager.validate_all_healthy() == (True, {"models_controller": True})


def test_failed_controller_remains_unhealthy_after_registration():
    manager = ControllerManager.get_instance()
    manager.await_controller_registration("models")
    with manager.controller_registration_context("models"):
        manager.register("models_controller", _ToggleLoop(healthy=True))
    manager.mark_controller_failed("models")

    assert manager.validate_all_healthy() == (
        False,
        {"models": False, "models_controller": True},
    )


def test_failed_controller_status_takes_precedence_over_same_named_healthy_loop():
    manager = ControllerManager.get_instance()
    manager.await_controller_registration("models")
    with manager.controller_registration_context("models"):
        manager.register("models", _ToggleLoop(healthy=True))
    manager.mark_controller_failed("models")

    assert manager.validate_all_healthy() == (False, {"models": False})


def test_awaiting_registration_for_restart_clears_previous_failure():
    manager = ControllerManager.get_instance()
    manager.mark_controller_failed("models")

    manager.await_controller_registration("models")
    with manager.controller_registration_context("models"):
        manager.register("models_controller", _ToggleLoop(healthy=True))

    assert manager.validate_all_healthy() == (True, {"models_controller": True})


def test_stale_generation_cleanup_does_not_remove_restarted_controller():
    manager = ControllerManager.get_instance()
    old_generation = manager.await_controller_registration("models")
    manager.mark_controller_failed("models", old_generation)

    new_generation = manager.await_controller_registration("models")
    with manager.controller_registration_context("models", new_generation):
        manager.register("models_controller", _ToggleLoop(healthy=True))

    assert manager.stop_tracking_controller("models", old_generation) is False
    assert manager.validate_all_healthy() == (True, {"models_controller": True})


def test_clear_does_not_reuse_generation_tokens():
    manager = ControllerManager.get_instance()
    old_generation = manager.await_controller_registration("models")
    manager.clear()

    new_generation = manager.await_controller_registration("models")

    assert new_generation > old_generation
    assert manager.stop_tracking_controller("models", old_generation) is False
    assert manager.validate_all_healthy() == (False, {"models": False})


def test_stale_generation_cannot_register_a_loop_into_a_restart():
    manager = ControllerManager.get_instance()
    old_generation = manager.await_controller_registration("models")
    manager.mark_controller_failed("models", old_generation)
    new_generation = manager.await_controller_registration("models")

    with (
        manager.controller_registration_context("models", old_generation),
        pytest.raises(RuntimeError, match="stale generation"),
    ):
        manager.register("stale_loop", _ToggleLoop(healthy=True))

    assert manager.get_all_loops() == {}
    with manager.controller_registration_context("models", new_generation):
        manager.register("models_controller", _ToggleLoop(healthy=True))
    assert manager.validate_all_healthy() == (True, {"models_controller": True})


def test_awaited_controller_is_unhealthy_after_its_last_loop_is_unregistered():
    manager = ControllerManager.get_instance()
    manager.await_controller_registration("models")
    with manager.controller_registration_context("models"):
        manager.register("models_controller", _ToggleLoop(healthy=True))

    manager.unregister("models_controller")

    assert manager.validate_all_healthy() == (False, {"models": False})


def test_stopping_controller_tracking_removes_its_registered_loops():
    manager = ControllerManager.get_instance()
    manager.await_controller_registration("jobs")
    with manager.controller_registration_context("jobs"):
        manager.register("job_scheduler", _ToggleLoop(healthy=True))
        manager.register("job_reconciler", _ToggleLoop(healthy=True))

    manager.stop_tracking_controller("jobs")

    assert manager.get_all_loops() == {}
    assert manager.validate_all_healthy() == (True, {})


def test_stopping_transition_preserves_the_original_failure_reason(caplog: pytest.LogCaptureFixture):
    manager = ControllerManager.get_instance()
    generation = manager.await_controller_registration("auth-proxy")
    manager.mark_controller_failed("auth-proxy", generation, reason="health loop failed to start")
    manager.mark_controller_stopping("auth-proxy", generation)

    with caplog.at_level("WARNING"):
        assert manager.validate_all_healthy() == (False, {"auth-proxy": False})

    assert "health loop failed to start" in caplog.text
    assert "did not stop before the shutdown deadline" not in caplog.text


def test_stop_tracking_with_clear_state_false_does_not_downgrade_stopping():
    """``clear_state=False`` must respect the same STOPPING guard as ``clear_state=True``;
    otherwise a delayed cleanup call downgrades STOPPING back to FAILED and drops the
    "still stopping" guard in :meth:`ControllerManager.await_controller_registration`,
    letting a new generation start while the old one still owns a resource.
    """
    manager = ControllerManager.get_instance()
    generation = manager.await_controller_registration("auth-proxy")
    manager.mark_controller_stopping("auth-proxy", generation)

    assert manager.stop_tracking_controller("auth-proxy", generation, clear_state=False) is False
    assert manager.validate_all_healthy() == (False, {"auth-proxy": False})


def test_stopping_controller_tracking_leaves_still_running_loop_tracked():
    """A loop whose thread hasn't finished must stay tracked (and thus unhealthy),
    so a controller that outlives its wrapper thread doesn't silently vanish from
    health checks (regression test for the auth-proxy shutdown race).
    """
    manager = ControllerManager.get_instance()
    release = threading.Event()

    class BlockingController(Controller):
        def step(self):
            pass

        @property
        def is_healthy(self) -> bool:
            return True

    loop = Loop(
        waiter=TimedLoopWaiter(sleep_secs=0.01, stop_signal=threading.Event()),
        controller=BlockingController(),
        stop_signal=threading.Event(),
    )
    loop.run = lambda: release.wait(timeout=5)

    with manager.controller_registration_context("auth-proxy"):
        manager.register("auth-proxy", loop)
    loop.start()

    try:
        manager.stop_tracking_controller("auth-proxy")

        assert manager.get_loop("auth-proxy") is loop
        assert manager.validate_all_healthy() == (True, {"auth-proxy": True})
    finally:
        release.set()
        loop.join(timeout=5)


def test_registration_context_tracks_loop_ownership_without_requiring_registration():
    manager = ControllerManager.get_instance()
    with manager.controller_registration_context("adapters"):
        manager.register("adapters_controller", _ToggleLoop(healthy=True))

    assert manager.validate_all_healthy() == (True, {"adapters_controller": True})

    manager.stop_tracking_controller("adapters")

    assert manager.validate_all_healthy() == (True, {})


# =============================================================================
# Loop shutdown_func Tests
# =============================================================================


def test_shutdown_func_called_on_stop():
    """Test that shutdown_func is called inside the loop thread when the loop stops."""
    shutdown_called = threading.Event()
    controller = SlowController()
    stop = threading.Event()

    loop = Loop(
        TimedLoopWaiter(0.01, stop_signal=stop),
        controller,
        shutdown_func=shutdown_called.set,
        stop_signal=stop,
    )

    loop.start()
    controller.stepped.wait(timeout=2)
    loop.stop()
    loop.join(timeout=2)

    assert not loop.is_alive()
    assert shutdown_called.is_set(), "shutdown_func should be called when loop exits"
    assert controller.step_count > 0


def test_shutdown_func_called_after_step_exception():
    """Test that shutdown_func runs even if step raises exceptions."""
    shutdown_called = threading.Event()
    controller = FailingController()
    stop = threading.Event()

    loop = Loop(
        TimedLoopWaiter(0.01, stop_signal=stop),
        controller,
        shutdown_func=shutdown_called.set,
        stop_signal=stop,
    )

    loop.start()
    controller.stepped.wait(timeout=2)
    loop.stop()
    loop.join(timeout=2)

    assert not loop.is_alive()
    assert shutdown_called.is_set(), "shutdown_func should run even after step exceptions"


def test_shutdown_func_exception_does_not_crash_thread():
    """Test that exceptions in shutdown_func are caught and don't crash the loop thread."""

    def bad_shutdown():
        raise RuntimeError("shutdown exploded")

    controller = SlowController()
    stop = threading.Event()

    loop = Loop(
        TimedLoopWaiter(0.01, stop_signal=stop),
        controller,
        shutdown_func=bad_shutdown,
        stop_signal=stop,
    )

    loop.start()
    controller.stepped.wait(timeout=2)
    loop.stop()
    loop.join(timeout=2)

    assert not loop.is_alive(), "Thread should exit cleanly even if shutdown_func raises"


def test_shutdown_func_runs_in_loop_thread():
    """Test that shutdown_func runs in the loop thread, not the caller's thread."""
    call_thread_ids: list[int] = []

    def track_shutdown():
        call_thread_ids.append(threading.current_thread().ident)

    controller = SlowController()
    stop = threading.Event()
    main_thread_id = threading.current_thread().ident

    loop = Loop(
        TimedLoopWaiter(0.01, stop_signal=stop),
        controller,
        shutdown_func=track_shutdown,
        stop_signal=stop,
    )

    loop.start()
    controller.stepped.wait(timeout=2)
    loop.stop()
    loop.join(timeout=2)

    assert len(call_thread_ids) == 1, "shutdown_func should be called exactly once"
    assert call_thread_ids[0] != main_thread_id, "shutdown_func should run in the loop thread, not the caller's thread"


def test_stop_without_shutdown_func():
    """Test that stop() works fine when no shutdown_func is provided."""
    controller = SlowController()
    stop = threading.Event()

    loop = Loop(
        TimedLoopWaiter(0.01, stop_signal=stop),
        controller,
        stop_signal=stop,
    )

    loop.start()
    controller.stepped.wait(timeout=2)
    loop.stop()
    loop.join(timeout=2)

    assert not loop.is_alive()
    assert controller.step_count > 0


class _ToggleLoop:
    """Loop stand-in whose health and reason are set by the test."""

    def __init__(self, healthy: bool = True, reason: str | None = None) -> None:
        self.is_healthy = healthy
        self.unhealthy_reason = reason


def test_health_transition_logs_only_on_change(caplog):
    """Health is evaluated on every probe, so only the edges may be logged."""
    manager = ControllerManager.get_instance()
    manager.clear()
    loop = _ToggleLoop(healthy=True)
    manager.register("toggle", loop)

    with caplog.at_level("INFO"):
        manager.validate_all_healthy()
        manager.validate_all_healthy()
        assert [r for r in caplog.records if "toggle" in r.getMessage()] == []

        loop.is_healthy = False
        loop.unhealthy_reason = "no progress for 44.20s (max allowed: 15.00s)"
        manager.validate_all_healthy()
        manager.validate_all_healthy()
        manager.validate_all_healthy()

        warnings = [r for r in caplog.records if r.levelname == "WARNING" and "toggle" in r.getMessage()]
        assert len(warnings) == 1
        assert "no progress for 44.20s" in warnings[0].getMessage()

        loop.is_healthy = True
        loop.unhealthy_reason = None
        manager.validate_all_healthy()
        manager.validate_all_healthy()

        recoveries = [r for r in caplog.records if r.levelname == "INFO" and "healthy again" in r.getMessage()]
        assert len(recoveries) == 1

    manager.clear()


def test_health_transition_reports_placeholder_when_reason_missing(caplog):
    manager = ControllerManager.get_instance()
    manager.clear()
    manager.register("no-reason", _ToggleLoop(healthy=False, reason=None))

    with caplog.at_level("WARNING"):
        manager.validate_all_healthy()

    warnings = [r for r in caplog.records if r.levelname == "WARNING" and "no-reason" in r.getMessage()]
    assert len(warnings) == 1
    assert "reason unavailable" in warnings[0].getMessage()

    manager.clear()


def test_clear_resets_transition_state(caplog):
    """After clear(), a re-registered unhealthy loop is reported again."""
    manager = ControllerManager.get_instance()
    manager.clear()
    manager.register("recycled", _ToggleLoop(healthy=False, reason="stuck"))
    with caplog.at_level("WARNING"):
        manager.validate_all_healthy()
    manager.clear()

    manager.register("recycled", _ToggleLoop(healthy=False, reason="stuck"))
    caplog.clear()
    with caplog.at_level("WARNING"):
        manager.validate_all_healthy()

    assert [r for r in caplog.records if "recycled" in r.getMessage()]
    manager.clear()
