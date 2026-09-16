# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from nhx.common.config import AuthConfig
from nhx.common.config.base import OIDCConfig
from nhx.common.controller import Controller, ControllerManager, Loop, TimedLoopWaiter
from nhx.common.service import Service
from nhx.platform_runner import config as runner_config
from nhx.platform_runner import controller_threads, server


class DummyService(Service):
    def __init__(self, name: str = "models") -> None:
        super().__init__(name=name, module_name="test.sidecars")

    def get_routers(self):
        return []


def _dummy_sidecar(_stop_signal: threading.Event) -> None:
    return None


def _patch_runner_discovery(
    monkeypatch: pytest.MonkeyPatch,
    *,
    services: dict[str, Service] | None = None,
    controllers: dict[str, Callable[[threading.Event], object]] | None = None,
    sidecars: dict[str, Callable[[threading.Event], object]] | None = None,
) -> None:
    services = services if services is not None else {"models": DummyService("models")}
    controllers = controllers if controllers is not None else {}
    sidecars = sidecars if sidecars is not None else {"adapters": _dummy_sidecar}

    monkeypatch.setattr(runner_config, "get_available_services", lambda: services)
    monkeypatch.setattr(runner_config, "get_available_controllers", lambda: controllers)
    monkeypatch.setattr(
        runner_config,
        "get_service_groups",
        lambda _available: {"all": list(services), "core": list(services), "api": []},
    )
    monkeypatch.setattr(
        runner_config, "get_controller_groups", lambda _available: {"all": list(controllers), "core": list(controllers)}
    )
    monkeypatch.setattr(runner_config, "get_default_controllers", lambda _groups: list(controllers))
    monkeypatch.setattr(runner_config, "AVAILABLE_SIDECARS", sidecars)
    monkeypatch.setattr("nhx.platform_runner.registry.AVAILABLE_SIDECARS", sidecars)
    monkeypatch.setattr(server, "AVAILABLE_SIDECARS", sidecars, raising=False)
    monkeypatch.setattr(server, "order_services_by_dependencies", lambda service_instances: service_instances)


def test_models_service_does_not_pull_in_adapters_sidecar(monkeypatch: pytest.MonkeyPatch) -> None:
    """``adapters`` is a NIM sidecar, so the platform never starts it implicitly."""
    _patch_runner_discovery(monkeypatch)

    resolved = runner_config.resolve_run_configuration(
        runner_config.PlatformAppConfig(services=["models"], controllers=[])
    )

    assert resolved.services == {"models"}
    assert resolved.sidecars == set()


def test_explicit_sidecar_can_run_without_services(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_runner_discovery(monkeypatch)

    resolved = runner_config.resolve_run_configuration(
        runner_config.PlatformAppConfig(services=[], controllers=[], sidecars=["adapters"])
    )

    assert resolved.services == set()
    assert resolved.controllers == set()
    assert resolved.sidecars == {"adapters"}


def _auth_config(enabled: bool = False) -> AuthConfig:
    return AuthConfig(
        enabled=enabled,
        policy_decision_point_base_url="http://localhost:8181",
        oidc=OIDCConfig(enabled=False),
    )


def _sidecar_with_events(started: threading.Event, stopped: threading.Event) -> Callable[[threading.Event], None]:
    def run(stop_signal: threading.Event) -> None:
        started.set()
        stop_signal.wait(timeout=5.0)
        stopped.set()

    return run


def test_create_app_starts_and_stops_dummy_sidecar_with_lifespan() -> None:
    started = threading.Event()
    stopped = threading.Event()

    with (
        patch("nhx.platform_runner.server.get_platform_config") as platform_config,
        patch("nhx.platform_runner.server.get_auth_config", return_value=_auth_config(False)),
        patch("nhx.common.auth.middleware.get_auth_config", return_value=_auth_config(False)),
    ):
        platform_config.return_value.seed_on_startup = False
        platform_config.return_value.redirect_root_to_studio = False
        app = server.create_app(
            services=[],
            sidecar_run_funcs={"adapters": _sidecar_with_events(started, stopped)},
        )
        from fastapi.testclient import TestClient

        with TestClient(app) as client:
            assert started.wait(timeout=1.0)
            assert client.get("/").status_code == 200

    assert stopped.wait(timeout=1.0)


def test_registered_sidecar_loop_is_removed_after_each_lifespan() -> None:
    manager = ControllerManager.get_instance()
    manager.clear()
    registered = threading.Event()

    def run(stop_signal: threading.Event) -> None:
        loop = MagicMock(is_healthy=True, unhealthy_reason=None)
        manager.register("adapters_controller", loop)
        registered.set()
        stop_signal.wait(timeout=5.0)

    try:
        with (
            patch("nhx.platform_runner.server.get_platform_config") as platform_config,
            patch("nhx.platform_runner.server.get_auth_config", return_value=_auth_config(False)),
            patch("nhx.common.auth.middleware.get_auth_config", return_value=_auth_config(False)),
        ):
            platform_config.return_value.seed_on_startup = False
            platform_config.return_value.redirect_root_to_studio = False

            for _ in range(2):
                registered.clear()
                app = server.create_app(services=[], sidecar_run_funcs={"adapters": run})

                from fastapi.testclient import TestClient

                with TestClient(app) as client:
                    assert registered.wait(timeout=1.0)
                    assert client.get("/health/ready").status_code == 200

                assert manager.validate_all_healthy() == (True, {})
    finally:
        manager.clear()


def test_self_tracking_sidecar_survives_generic_shutdown_untracking() -> None:
    """Leave tracking to a sidecar whose resource outlives its wrapper."""
    manager = ControllerManager.get_instance()
    manager.clear()
    registered = threading.Event()
    # Model a resource that outlives its sidecar wrapper.
    hang_forever = threading.Event()
    resource_thread = threading.Thread(target=lambda: hang_forever.wait(timeout=5), daemon=True)

    class _ResourceController(Controller):
        def step(self) -> None:
            pass

        @property
        def is_healthy(self) -> bool:
            return resource_thread.is_alive()

    def run(stop_signal: threading.Event) -> None:
        generation = manager.await_controller_registration("auth-proxy")
        resource_thread.start()
        # The health loop stops independently of the resource thread.
        health_loop = Loop(
            waiter=TimedLoopWaiter(sleep_secs=0.01, stop_signal=stop_signal),
            controller=_ResourceController(),
            stop_signal=stop_signal,
        )
        health_loop.name = "auth-proxy"
        manager.register(health_loop.name, health_loop)
        health_loop.start()
        registered.set()
        stop_signal.wait(timeout=5.0)
        health_loop.join(timeout=2.0)
        # The sidecar intentionally leaves the live resource tracked.
        manager.mark_controller_stopping("auth-proxy", generation)

        def cleanup_after_resource_exit() -> None:
            resource_thread.join()
            manager.stop_tracking_controller(
                "auth-proxy",
                generation,
                allow_stopping=True,
            )

        threading.Thread(target=cleanup_after_resource_exit, daemon=True).start()

    try:
        with (
            patch("nhx.platform_runner.server.get_platform_config") as platform_config,
            patch("nhx.platform_runner.server.get_auth_config", return_value=_auth_config(False)),
            patch("nhx.common.auth.middleware.get_auth_config", return_value=_auth_config(False)),
        ):
            platform_config.return_value.seed_on_startup = False
            platform_config.return_value.redirect_root_to_studio = False
            app = server.create_app(services=[], sidecar_run_funcs={"auth-proxy": run})

            from fastapi.testclient import TestClient

            with TestClient(app) as client:
                assert registered.wait(timeout=1.0)
                assert client.get("/health/ready").status_code == 200

            assert "auth-proxy" in manager.get_all_loops()
    finally:
        hang_forever.set()
        manager.clear()


def test_crashed_required_sidecar_blocks_readiness() -> None:
    crashed = threading.Event()

    def crash(_stop_signal: threading.Event) -> None:
        crashed.set()
        raise ValueError("optional configuration is missing")

    manager = ControllerManager.get_instance()
    manager.clear()
    try:
        with (
            patch("nhx.platform_runner.server.get_platform_config") as platform_config,
            patch("nhx.platform_runner.server.get_auth_config", return_value=_auth_config(False)),
            patch("nhx.common.auth.middleware.get_auth_config", return_value=_auth_config(False)),
        ):
            platform_config.return_value.seed_on_startup = False
            platform_config.return_value.redirect_root_to_studio = False
            app = server.create_app(services=[], sidecar_run_funcs={"adapters": crash})

            from fastapi.testclient import TestClient

            with TestClient(app) as client:
                assert crashed.wait(timeout=1.0)
                assert client.get("/health/ready").status_code == 503
                status = client.get("/status").json()
                assert status["status"] == "unhealthy"
                assert status["controllers"] == {"healthy": False, "status": {"adapters": False}}
    finally:
        manager.clear()


def test_build_platform_app_loads_explicit_sidecar_into_lifespan(monkeypatch: pytest.MonkeyPatch) -> None:
    started = threading.Event()
    stopped = threading.Event()
    _patch_runner_discovery(monkeypatch, sidecars={"adapters": _sidecar_with_events(started, stopped)})

    with (
        patch("nhx.platform_runner.server.get_platform_config") as platform_config,
        patch("nhx.platform_runner.server.get_auth_config", return_value=_auth_config(False)),
        patch("nhx.common.auth.middleware.get_auth_config", return_value=_auth_config(False)),
    ):
        platform_config.return_value.seed_on_startup = False
        platform_config.return_value.redirect_root_to_studio = False
        app = server.build_platform_app(
            runner_config.PlatformAppConfig(services=["models"], controllers=[], sidecars=["adapters"]),
            env={},
        )
        from fastapi.testclient import TestClient

        with TestClient(app) as client:
            assert started.wait(timeout=1.0)
            assert client.get("/").status_code == 200

    assert stopped.wait(timeout=1.0)


def test_sidecar_thread_start_failure_rolls_back_started_controllers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller_stopped = threading.Event()
    manager = ControllerManager.get_instance()
    manager.clear()

    def controller_run(stop_signal: threading.Event) -> None:
        manager.register("models_controller", MagicMock(is_healthy=True, unhealthy_reason=None))
        stop_signal.wait(timeout=2)
        controller_stopped.set()

    original_start = controller_threads._RunnerThread.start

    def fail_sidecar_start(thread: controller_threads._RunnerThread) -> None:
        if thread.component_name == "adapters":
            raise RuntimeError("sidecar thread start failed")
        original_start(thread)

    monkeypatch.setattr(controller_threads._RunnerThread, "start", fail_sidecar_start)

    with (
        patch("nhx.platform_runner.server.get_platform_config") as platform_config,
        patch("nhx.platform_runner.server.get_auth_config", return_value=_auth_config(False)),
        patch("nhx.common.auth.middleware.get_auth_config", return_value=_auth_config(False)),
    ):
        platform_config.return_value.seed_on_startup = False
        platform_config.return_value.redirect_root_to_studio = False
        app = server.create_app(
            services=[],
            controller_run_funcs={"models": controller_run},
            sidecar_run_funcs={"adapters": _dummy_sidecar},
        )

        from fastapi.testclient import TestClient

        with pytest.raises(RuntimeError, match="sidecar thread start failed"), TestClient(app):
            pass

    assert controller_stopped.wait(timeout=2)
    assert manager.get_all_loops() == {}
    assert manager.validate_all_healthy() == (False, {"adapters": False})
    manager.clear()


def test_build_platform_app_rejects_controller_sidecar_name_collision(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_runner_discovery(
        monkeypatch,
        controllers={"adapters": _dummy_sidecar},
        sidecars={"adapters": _dummy_sidecar},
    )

    with pytest.raises(ValueError, match="Controller/sidecar name collision: adapters"):
        server.build_platform_app(
            runner_config.PlatformAppConfig(controllers=["adapters"], sidecars=["adapters"]),
            env={},
        )


def test_real_adapters_sidecar_entrypoint_starts_and_stops_with_required_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from nhx.core.models.sidecars.adapters import main as adapters_main

    started = threading.Event()
    stopped = threading.Event()
    manager = MagicMock()

    class FakeLoop:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def start(self) -> None:
            started.set()

        def stop(self) -> None:
            stopped.set()

        def join(self) -> None:
            return None

    lora_dir = tmp_path / "loras"
    monkeypatch.setenv("NIM_PEFT_SOURCE", str(lora_dir))
    monkeypatch.setenv("NHX_MODEL_ENTITY_WORKSPACE", "default")
    monkeypatch.setenv("NHX_MODEL_ENTITY_NAME", "test-model")
    monkeypatch.setenv("NIM_PEFT_REFRESH_INTERVAL", "30")
    monkeypatch.delenv("VLLM_ENDPOINT", raising=False)

    monkeypatch.setattr(adapters_main, "get_platform_config", lambda: MagicMock(base_url="http://platform.local"))
    monkeypatch.setattr(adapters_main, "get_platform_sdk", lambda **_kwargs: MagicMock())
    monkeypatch.setattr(adapters_main.asyncio, "new_event_loop", lambda: MagicMock())
    monkeypatch.setattr(adapters_main, "Loop", FakeLoop)
    monkeypatch.setattr(adapters_main, "TimedLoopWaiter", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(adapters_main.ControllerManager, "get_instance", classmethod(lambda _cls: manager))

    stop_signal = threading.Event()
    thread = threading.Thread(target=adapters_main.run, args=(stop_signal,), daemon=True)
    try:
        thread.start()

        assert started.wait(timeout=1.0)
        manager.register.assert_called_once()
        assert manager.register.call_args.args[0] == "adapters_controller"

        stop_signal.set()
        thread.join(timeout=1.0)

        assert not thread.is_alive()
        assert stopped.is_set()
    finally:
        stop_signal.set()
        thread.join(timeout=1.0)
        adapters_main.adapters_controller_monitored = None
