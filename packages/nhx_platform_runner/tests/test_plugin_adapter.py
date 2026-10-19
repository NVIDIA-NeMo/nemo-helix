# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from typing import ClassVar
from unittest.mock import MagicMock

import pytest
from fastapi import APIRouter, Request
from nemo_helix_plugin.controller import NemoController
from nemo_helix_plugin.service import NemoService, RouterSpec
from nhx.common.controller import ControllerManager, Loop
from nhx.platform_runner.plugin_adapter import NemoServiceAdapter, make_controller_run_func
from starlette.responses import JSONResponse

_WAIT_POLL_INTERVAL = 0.01
_WAIT_DEADLINE_SECONDS = 2.0

# Per-test observation lists (cleared by autouse fixture; avoids class-level flake under xdist).
_list_objects_calls: list[float] = []
_on_startup_calls: list[float] = []


class _StubService(NemoService):
    name = "test-plugin"

    def get_routers(self) -> list[RouterSpec]:
        return [RouterSpec(router=APIRouter())]


class _ServiceWithExceptionHandlers(NemoService):
    name = "test-plugin"

    def get_routers(self) -> list[RouterSpec]:
        return [RouterSpec(router=APIRouter())]

    def get_exception_handlers(self):
        async def handle_value_error(request: Request, exc: ValueError) -> JSONResponse:
            return JSONResponse(status_code=422, content={"detail": str(exc)})

        return {ValueError: handle_value_error}


class _StubController(NemoController):
    """Minimal controller for make_controller_run_func tests."""

    name = "test-controller"
    dependencies: ClassVar[list[str]] = []

    def __init__(self) -> None:
        self._interval_seconds = 3600.0

    @property
    def interval_seconds(self) -> float:
        return self._interval_seconds

    async def on_startup(self) -> None:
        _on_startup_calls.append(time.monotonic())

    async def list_objects(self) -> list:
        _list_objects_calls.append(time.monotonic())
        return []

    async def reconcile_one(self, obj: object) -> None:
        pass


class _StubControllerWithDeps(_StubController):
    dependencies: ClassVar[list[str]] = ["entities"]


@pytest.fixture(autouse=True)
def _clear_stub_controller_call_history() -> Iterator[None]:
    _list_objects_calls.clear()
    _on_startup_calls.clear()
    ControllerManager._instance = None
    yield
    ControllerManager._instance = None


@pytest.fixture
def patch_get_platform_config(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock_config = MagicMock()
    monkeypatch.setattr(
        "nhx.platform_runner.plugin_adapter.get_platform_config",
        MagicMock(return_value=mock_config),
    )
    return mock_config


def _run_controller_until_list_objects(
    controller_cls: type[NemoController],
    stop_signal: threading.Event,
) -> threading.Thread:
    run_func = make_controller_run_func(controller_cls)
    thread = threading.Thread(target=run_func, args=(stop_signal,), daemon=True)
    thread.start()
    deadline = time.monotonic() + _WAIT_DEADLINE_SECONDS
    while time.monotonic() < deadline and not _list_objects_calls:
        time.sleep(_WAIT_POLL_INTERVAL)
    return thread


def test_adapter_without_exception_handlers():
    adapter = NemoServiceAdapter(_StubService())
    app = adapter.create_app()
    assert ValueError not in app.exception_handlers


def test_adapter_registers_exception_handlers():
    adapter = NemoServiceAdapter(_ServiceWithExceptionHandlers())
    app = adapter.create_app()
    assert ValueError in app.exception_handlers


@pytest.mark.usefixtures("patch_get_platform_config")
def test_controller_waits_for_dependencies_before_on_startup_and_reconcile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """on_startup and reconcile must not run until wait_for_service_ready returns True."""
    wait_started = threading.Event()
    wait_can_proceed = threading.Event()

    def mock_wait(_config, service_name, _stop_signal, timeout=60.0, poll_interval=0.5):
        assert service_name == "entities"
        wait_started.set()
        return wait_can_proceed.wait(timeout=_WAIT_DEADLINE_SECONDS)

    monkeypatch.setattr(
        "nhx.platform_runner.plugin_adapter.wait_for_service_ready",
        mock_wait,
    )

    stop_signal = threading.Event()
    run_func = make_controller_run_func(_StubControllerWithDeps)
    thread = threading.Thread(target=run_func, args=(stop_signal,), daemon=True)
    thread.start()

    assert wait_started.wait(timeout=_WAIT_DEADLINE_SECONDS)
    assert _on_startup_calls == []
    assert _list_objects_calls == []

    wait_can_proceed.set()
    deadline = time.monotonic() + _WAIT_DEADLINE_SECONDS
    while time.monotonic() < deadline and not _on_startup_calls:
        time.sleep(_WAIT_POLL_INTERVAL)
    assert _on_startup_calls

    deadline = time.monotonic() + _WAIT_DEADLINE_SECONDS
    while time.monotonic() < deadline and not _list_objects_calls:
        time.sleep(_WAIT_POLL_INTERVAL)
    assert _list_objects_calls

    stop_signal.set()
    thread.join(timeout=_WAIT_DEADLINE_SECONDS)


@pytest.mark.usefixtures("patch_get_platform_config")
def test_controller_exits_early_on_stop_during_dependency_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When stop_signal is set during dependency wait, reconcile must not run."""

    def mock_wait(_config, _service_name, stop_signal, timeout=60.0, poll_interval=0.5):
        stop_signal.set()
        return False

    monkeypatch.setattr(
        "nhx.platform_runner.plugin_adapter.wait_for_service_ready",
        mock_wait,
    )

    stop_signal = threading.Event()
    run_func = make_controller_run_func(_StubControllerWithDeps)
    run_func(stop_signal)

    assert _on_startup_calls == []
    assert _list_objects_calls == []


@pytest.mark.usefixtures("patch_get_platform_config")
def test_controller_starts_after_dependency_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When a dependency times out without shutdown, reconcile still starts."""
    monkeypatch.setattr(
        "nhx.platform_runner.plugin_adapter.wait_for_service_ready",
        lambda *_args, **_kwargs: False,
    )

    stop_signal = threading.Event()
    thread = _run_controller_until_list_objects(_StubControllerWithDeps, stop_signal)

    stop_signal.set()
    thread.join(timeout=_WAIT_DEADLINE_SECONDS)

    assert _list_objects_calls


def test_controller_skips_wait_when_no_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Controllers with empty dependencies must not call wait_for_service_ready."""
    mock_wait = MagicMock(return_value=True)
    monkeypatch.setattr(
        "nhx.platform_runner.plugin_adapter.wait_for_service_ready",
        mock_wait,
    )

    stop_signal = threading.Event()
    thread = _run_controller_until_list_objects(_StubController, stop_signal)

    stop_signal.set()
    thread.join(timeout=_WAIT_DEADLINE_SECONDS)

    mock_wait.assert_not_called()
    assert _list_objects_calls


def test_registration_name_collision_still_runs_shutdown_and_reports_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Close the adapter event loop when loop registration collides.

    Registration failures must propagate (like ``loop.start()`` failures do,
    see ``test_loop_start_failure_unregisters_loop_and_runs_shutdown``) so the
    controller_threads.py wrapper unconditionally records the failure instead
    of relying on ``stop_signal`` happening to still be unset.
    """
    manager = ControllerManager.get_instance()
    manager.register("controller-plugin-test-controller", MagicMock(is_healthy=True, unhealthy_reason=None))

    shutdown_calls: list[float] = []

    class _CollidingController(_StubController):
        async def on_shutdown(self) -> None:
            shutdown_calls.append(time.monotonic())

    run_func = make_controller_run_func(_CollidingController)
    with caplog.at_level("ERROR"), pytest.raises(ValueError, match="already registered"):
        run_func(threading.Event())

    assert shutdown_calls
    assert "failed to register its control loop" in caplog.text


def test_plugin_controller_registers_loop_for_health_reporting() -> None:
    stop_signal = threading.Event()
    thread = _run_controller_until_list_objects(_StubController, stop_signal)

    try:
        loops = ControllerManager.get_instance().get_all_loops()
        assert list(loops) == ["controller-plugin-test-controller"]
        assert loops["controller-plugin-test-controller"].is_alive()
    finally:
        stop_signal.set()
        thread.join(timeout=_WAIT_DEADLINE_SECONDS)


def test_loop_start_failure_unregisters_loop_and_runs_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    shutdown_calls: list[float] = []

    class _StartFailingController(_StubController):
        async def on_shutdown(self) -> None:
            shutdown_calls.append(time.monotonic())

    def _fail_start(_loop: Loop) -> None:
        raise RuntimeError("loop thread start failed")

    monkeypatch.setattr(Loop, "start", _fail_start)
    run_func = make_controller_run_func(_StartFailingController)

    with pytest.raises(RuntimeError, match="loop thread start failed"):
        run_func(threading.Event())

    assert ControllerManager.get_instance().get_all_loops() == {}
    assert shutdown_calls


def test_on_startup_failure_raises_and_runs_shutdown_hook(caplog: pytest.LogCaptureFixture) -> None:
    """A failing on_startup() must propagate and still run the plugin's on_shutdown().

    Matches the register()/loop.start() failure paths above: the caller
    (controller_threads.py's wrapper) relies on the exception to unconditionally
    record the failure, and any resources on_startup() partially acquired
    should still be released via on_shutdown().
    """
    shutdown_calls: list[float] = []

    class _StartupFailingController(_StubController):
        async def on_startup(self) -> None:
            await super().on_startup()
            raise RuntimeError("on_startup failed")

        async def on_shutdown(self) -> None:
            shutdown_calls.append(time.monotonic())

    run_func = make_controller_run_func(_StartupFailingController)
    with caplog.at_level("ERROR"), pytest.raises(RuntimeError, match="on_startup failed"):
        run_func(threading.Event())

    assert shutdown_calls
    assert "on_startup() failed" in caplog.text
    assert ControllerManager.get_instance().get_all_loops() == {}
