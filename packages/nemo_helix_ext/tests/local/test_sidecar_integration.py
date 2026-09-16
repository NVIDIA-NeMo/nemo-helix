# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for sidecar lifecycle in embedded and daemon modes.

These tests let the real ``build_platform_app`` → ``resolve_run_configuration`` →
``create_app`` chain run with a lightweight test sidecar registered in the
platform runner registry.  They verify that sidecar threads actually start and
stop during the FastAPI app lifespan, covering the full resolution path without
mocking away the core wiring.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from nemo_helix_ext.local import services
from nemo_helix_ext.local.services import ServiceRunConfig
from nhx.common.config import AuthConfig
from nhx.common.config.base import OIDCConfig
from nhx.common.service import Service
from nhx.platform_runner import config as runner_config
from nhx.platform_runner import registry, server

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DummyService(Service):
    """Minimal service that registers no routers."""

    def __init__(self, name: str = "models") -> None:
        super().__init__(name=name, module_name="test.sidecar_integration")

    def get_routers(self):
        return []


def _sidecar_with_events(started: threading.Event, stopped: threading.Event) -> Callable[[threading.Event], None]:
    """Return a sidecar ``run(stop_signal)`` that signals start/stop via events."""

    def run(stop_signal: threading.Event) -> None:
        started.set()
        stop_signal.wait(timeout=5.0)
        stopped.set()

    return run


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sidecar_events() -> tuple[threading.Event, threading.Event]:
    return threading.Event(), threading.Event()


@pytest.fixture
def patched_registry(
    monkeypatch: pytest.MonkeyPatch,
    sidecar_events: tuple[threading.Event, threading.Event],
) -> tuple[threading.Event, threading.Event]:
    """Patch the platform runner registry with a dummy models service and a
    test sidecar, plus minimal auth/platform config stubs."""
    started, stopped = sidecar_events
    dummy_services: dict[str, Service] = {"models": _DummyService()}
    dummy_sidecars: dict[str, Callable] = {"adapters": _sidecar_with_events(started, stopped)}

    monkeypatch.setattr(runner_config, "get_available_services", lambda: dummy_services)
    monkeypatch.setattr(runner_config, "get_available_controllers", lambda: {})
    monkeypatch.setattr(
        runner_config,
        "get_service_groups",
        lambda _available: {"all": ["models"], "core": ["models"], "api": []},
    )
    monkeypatch.setattr(runner_config, "get_controller_groups", lambda _available: {"all": [], "core": []})
    monkeypatch.setattr(runner_config, "get_default_controllers", lambda _groups: [])
    monkeypatch.setattr(runner_config, "AVAILABLE_SIDECARS", dummy_sidecars)
    monkeypatch.setattr(registry, "AVAILABLE_SIDECARS", dummy_sidecars)
    monkeypatch.setattr(server, "AVAILABLE_SIDECARS", dummy_sidecars, raising=False)
    monkeypatch.setattr(server, "order_services_by_dependencies", lambda svc: svc)

    auth_cfg = AuthConfig(
        enabled=False,
        policy_decision_point_base_url="http://localhost:8181",
        oidc=OIDCConfig(enabled=False),
    )
    monkeypatch.setattr(server, "get_auth_config", lambda: auth_cfg)
    monkeypatch.setattr("nhx.common.auth.middleware.get_auth_config", lambda: auth_cfg)
    platform_cfg = MagicMock()
    platform_cfg.seed_on_startup = False
    platform_cfg.redirect_root_to_studio = False
    monkeypatch.setattr(server, "get_platform_config", lambda: platform_cfg)

    return started, stopped


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_embedded_sidecar_starts_alongside_selected_services(
    patched_registry: tuple[threading.Event, threading.Event],
    tmp_path: Path,
) -> None:
    """A sidecar requested next to a service starts during app lifespan.

    Sidecars are never derived from the selected services, so ``adapters`` has
    to be named explicitly even when ``models`` is running.
    """
    started, stopped = patched_registry

    cfg = ServiceRunConfig(
        mode=services.ServiceMode.EMBEDDED,
        services=("models",),
        controllers=(),
        sidecars=("adapters",),
        transport="tcp",
        scope="integ-embedded-auto",
        state_dir=tmp_path / "state",
        runtime_dir=tmp_path / "runtime",
    )
    handle = services.start_embedded_services(cfg, env={})

    from fastapi.testclient import TestClient

    with TestClient(handle.app) as client:
        assert started.wait(timeout=2.0), "sidecar thread did not start"
        assert client.get("/").status_code == 200

    assert stopped.wait(timeout=2.0), "sidecar thread did not stop after lifespan exit"


@pytest.mark.integration
def test_embedded_explicit_sidecar_without_services(
    patched_registry: tuple[threading.Event, threading.Event],
    tmp_path: Path,
) -> None:
    """An explicitly requested sidecar runs even when no services are selected."""
    started, stopped = patched_registry

    cfg = ServiceRunConfig(
        mode=services.ServiceMode.EMBEDDED,
        services=(),
        controllers=(),
        sidecars=("adapters",),
        transport="tcp",
        scope="integ-embedded-explicit",
        state_dir=tmp_path / "state",
        runtime_dir=tmp_path / "runtime",
    )
    handle = services.start_embedded_services(cfg, env={})

    from fastapi.testclient import TestClient

    with TestClient(handle.app) as client:
        assert started.wait(timeout=2.0), "sidecar thread did not start"
        assert client.get("/").status_code == 200

    assert stopped.wait(timeout=2.0), "sidecar thread did not stop after lifespan exit"


@pytest.mark.integration
def test_run_services_daemon_mode_starts_sidecar_in_process(
    patched_registry: tuple[threading.Event, threading.Event],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """run_services(_mode='daemon') exercises the daemon code path in-process.
    It calls start_embedded_services then serve_embedded_app.  We intercept
    serve_embedded_app to capture the app and exercise its lifespan, proving
    the daemon path wires sidecars identically to embedded mode."""
    started, stopped = patched_registry
    captured_app = {}

    def fake_serve(app, cfg, socket_path):
        captured_app["app"] = app

    monkeypatch.setattr("nemo_helix_ext.local.services.require_services_extra", lambda: None)
    monkeypatch.setattr("nemo_helix_ext.local.services.process.is_instance_alive", lambda *a, **kw: False)
    monkeypatch.setattr("nemo_helix_ext.local.services._check_tcp_available", lambda *a: None)
    monkeypatch.setattr("nemo_helix_ext.local.services.process.acquire_lock", lambda *a, **kw: 123)
    monkeypatch.setattr("nemo_helix_ext.local.services.process.log_path_for", lambda *a, **kw: tmp_path / "nemo.log")
    monkeypatch.setattr("nemo_helix_ext.local.services.process.write_descriptor", lambda *a, **kw: None)
    monkeypatch.setattr("nemo_helix_ext.local.services.process.remove_descriptor", lambda *a, **kw: None)
    monkeypatch.setattr("nemo_helix_ext.local.services.serve_embedded_app", fake_serve)
    monkeypatch.setattr("nemo_helix_ext.local.services.os.close", lambda fd: None)

    cfg = ServiceRunConfig(
        mode=services.ServiceMode.DAEMON,
        services=("models",),
        controllers=(),
        sidecars=("adapters",),
        transport="tcp",
        scope="integ-daemon",
        state_dir=tmp_path / "state",
        runtime_dir=tmp_path / "runtime",
    )
    services.run_services(cfg, _mode="daemon", env={})

    assert "app" in captured_app, "serve_embedded_app was not called"

    from fastapi.testclient import TestClient

    with TestClient(captured_app["app"]) as client:
        assert started.wait(timeout=2.0), "sidecar thread did not start in daemon mode"
        assert client.get("/").status_code == 200

    assert stopped.wait(timeout=2.0), "sidecar thread did not stop after lifespan exit"


@pytest.mark.integration
def test_embedded_rejects_unknown_sidecar_name(tmp_path: Path, patched_registry) -> None:
    """Requesting a sidecar not in the registry raises ValueError with a clear message."""
    cfg = ServiceRunConfig(
        mode=services.ServiceMode.EMBEDDED,
        services=(),
        controllers=(),
        sidecars=("nonexistent",),
        transport="tcp",
        scope="integ-unknown",
        state_dir=tmp_path / "state",
        runtime_dir=tmp_path / "runtime",
    )
    with pytest.raises(ValueError, match="Unknown sidecars: nonexistent"):
        services.start_embedded_services(cfg, env={})
