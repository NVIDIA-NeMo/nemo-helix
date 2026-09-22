# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from nemo_helix_ext.local import services
from nemo_helix_ext.local.process import StopResult
from nemo_helix_ext.local.services import ServiceRunConfig
from nhx.platform_runner.config import HelixAppConfig


@dataclass(frozen=True)
class ModeContractCase:
    mode: services.ServiceMode
    launcher_patch: str
    existing_handle_patch_value: object | None


@dataclass
class ContractHandle:
    mode: services.ServiceMode
    calls: list[tuple[str, object]] = field(default_factory=list)

    def is_running(self) -> bool:
        self.calls.append(("is_running", None))
        return True

    def wait_until_ready(self, timeout: float | None = None) -> None:
        self.calls.append(("wait_until_ready", timeout))

    async def wait_until_ready_async(self, timeout: float | None = None) -> None:
        self.calls.append(("wait_until_ready_async", timeout))

    def client(self, **kwargs: object) -> tuple[str, services.ServiceMode, dict[str, object]]:
        self.calls.append(("client", kwargs))
        return ("client", self.mode, kwargs)

    def async_client(self, **kwargs: object) -> tuple[str, services.ServiceMode, dict[str, object]]:
        self.calls.append(("async_client", kwargs))
        return ("async_client", self.mode, kwargs)

    def start_services(self, service_names: list[str] | tuple[str, ...]) -> services.StartServicesResult:
        requested = list(service_names)
        self.calls.append(("start_services", requested))
        return services.StartServicesResult(
            requested=requested,
            started=["auth", *requested],
            already_active=[],
            active=["secrets", "auth", *requested],
        )

    async def start_services_async(self, service_names: list[str] | tuple[str, ...]) -> services.StartServicesResult:
        requested = list(service_names)
        self.calls.append(("start_services_async", requested))
        return services.StartServicesResult(
            requested=requested,
            started=["auth", *requested],
            already_active=[],
            active=["secrets", "auth", *requested],
        )

    def stop(self, *, timeout: float = 30.0, force: bool = False) -> StopResult:
        self.calls.append(("stop", {"timeout": timeout, "force": force}))
        return StopResult(stopped_pids=[], swept_children=[])

    async def stop_async(self, *, timeout: float = 30.0, force: bool = False) -> StopResult:
        self.calls.append(("stop_async", {"timeout": timeout, "force": force}))
        return StopResult(stopped_pids=[], swept_children=[])


MODE_CONTRACT_CASES = [
    ModeContractCase(
        mode=services.ServiceMode.EMBEDDED,
        launcher_patch="nemo_helix_ext.local.services.start_embedded_services",
        existing_handle_patch_value=None,
    ),
    ModeContractCase(
        mode=services.ServiceMode.DAEMON,
        launcher_patch="nemo_helix_ext.local.services.daemonize_services",
        existing_handle_patch_value=None,
    ),
]


@pytest.fixture(params=MODE_CONTRACT_CASES, ids=lambda case: case.mode.value)
def mode_case(request: pytest.FixtureRequest) -> ModeContractCase:
    return request.param


def _config_for(case: ModeContractCase, tmp_path: Path) -> ServiceRunConfig:
    return ServiceRunConfig(
        mode=case.mode,
        services=("secrets",),
        scope=f"{case.mode.value}-contract",
        state_dir=tmp_path / case.mode.value / "state",
        runtime_dir=tmp_path / case.mode.value / "runtime",
    )


def test_contract_ensure_services_returns_running_mode_handle(
    mode_case: ModeContractCase,
    tmp_path: Path,
) -> None:
    cfg = _config_for(mode_case, tmp_path)
    handle = ContractHandle(mode_case.mode)

    with (
        patch(
            "nemo_helix_ext.local.services.get_service_handle", return_value=mode_case.existing_handle_patch_value
        ),
        patch(mode_case.launcher_patch, return_value=handle),
    ):
        result = services.ensure_services(cfg)

    assert result is handle
    assert result.is_running() is True
    assert result.calls == [("is_running", None)]


def test_contract_connect_services_returns_client_from_selected_mode(
    mode_case: ModeContractCase,
    tmp_path: Path,
) -> None:
    cfg = _config_for(mode_case, tmp_path)
    handle = ContractHandle(mode_case.mode)

    with patch("nemo_helix_ext.local.services.ensure_services", return_value=handle):
        client = services.connect_services(cfg, api_key="test-key")

    assert client == ("client", mode_case.mode, {"api_key": "test-key"})
    assert handle.calls == [("client", {"api_key": "test-key"})]


@pytest.mark.asyncio
async def test_contract_handle_lifecycle_methods_have_same_semantics(
    mode_case: ModeContractCase,
) -> None:
    handle = ContractHandle(mode_case.mode)

    handle.wait_until_ready(timeout=1.5)
    await handle.wait_until_ready_async(timeout=2.5)
    sync_start = handle.start_services(["jobs"])
    async_start = await handle.start_services_async(["jobs"])
    stop_result = handle.stop(timeout=3.0, force=True)
    async_stop_result = await handle.stop_async(timeout=4.0, force=False)

    assert sync_start == services.StartServicesResult(
        requested=["jobs"],
        started=["auth", "jobs"],
        already_active=[],
        active=["secrets", "auth", "jobs"],
    )
    assert async_start == sync_start
    assert stop_result == StopResult(stopped_pids=[], swept_children=[])
    assert async_stop_result == StopResult(stopped_pids=[], swept_children=[])
    assert handle.calls == [
        ("wait_until_ready", 1.5),
        ("wait_until_ready_async", 2.5),
        ("start_services", ["jobs"]),
        ("start_services_async", ["jobs"]),
        ("stop", {"timeout": 3.0, "force": True}),
        ("stop_async", {"timeout": 4.0, "force": False}),
    ]


def test_contract_real_handles_report_same_staged_start_status_before_staged_start_lands(tmp_path: Path) -> None:
    embedded = services.EmbeddedServiceHandle(app=object(), runtime=object())
    daemon = services.DaemonServiceHandle(
        scope="daemon-contract",
        transport="tcp",
        socket_path=None,
        gateway_base_url=None,
        host="127.0.0.1",
        port=8080,
        pid=None,
        mode="daemon",
        log_path=None,
        state_dir=tmp_path / "state",
        runtime_dir=None,
    )

    for handle in (embedded, daemon):
        with pytest.raises(services.ServicesError, match="Staged service start is not implemented"):
            handle.start_services(["jobs"])


def test_contract_embedded_and_daemon_child_both_delegate_models_to_platform_builder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_build_platform_app(
        config: HelixAppConfig | None = None,
        *,
        env: object = None,
        http_client: object = None,
    ) -> MagicMock:
        calls.append({"config": config, "env": env, "http_client": http_client})
        return MagicMock()

    def service_config(mode: services.ServiceMode) -> ServiceRunConfig:
        return ServiceRunConfig(
            mode=mode,
            services=("models",),
            controllers=(),
            transport="tcp",
            scope="sc-test",
            state_dir=tmp_path / "state",
            runtime_dir=tmp_path / "runtime",
        )

    with patch("nhx.platform_runner.server.build_platform_app", side_effect=fake_build_platform_app):
        services.start_embedded_services(service_config(services.ServiceMode.EMBEDDED))

    with (
        patch("nhx.platform_runner.server.build_platform_app", side_effect=fake_build_platform_app),
        patch("nemo_helix_ext.local.services.require_services_extra"),
        patch("nemo_helix_ext.local.services.process.is_instance_alive", return_value=False),
        patch("nemo_helix_ext.local.services._check_tcp_available"),
        patch("nemo_helix_ext.local.services.process.acquire_lock", return_value=123),
        patch("nemo_helix_ext.local.services.process.log_path_for", return_value=tmp_path / "nemo.log"),
        patch("nemo_helix_ext.local.services.process.write_descriptor"),
        patch("nemo_helix_ext.local.services.process.remove_descriptor"),
        patch("nemo_helix_ext.local.services.serve_embedded_app"),
        patch("nemo_helix_ext.local.services.os.close"),
    ):
        services.run_services(service_config(services.ServiceMode.DAEMON), _mode="daemon")

    configs: list[HelixAppConfig] = []
    for call in calls:
        config = call["config"]
        assert isinstance(config, HelixAppConfig)
        configs.append(config)
    assert [config.services for config in configs] == [("models",), ("models",)]
    assert [config.controllers for config in configs] == [(), ()]
    assert [config.sidecars for config in configs] == [None, None]


def _sidecar_with_events(started: threading.Event, stopped: threading.Event) -> Callable[[threading.Event], None]:
    def run(stop_signal: threading.Event) -> None:
        started.set()
        stop_signal.wait(timeout=5.0)
        stopped.set()

    return run


def _patch_runner_registry(
    monkeypatch: pytest.MonkeyPatch,
    *,
    sidecar_run_func: Callable[[threading.Event], None],
) -> None:
    """Patch the platform runner registry so only a dummy 'models' service
    and a test sidecar are available, avoiding real service imports."""
    from nhx.common.config import AuthConfig
    from nhx.common.config.base import OIDCConfig
    from nhx.common.service import Service
    from nhx.platform_runner import config as runner_config
    from nhx.platform_runner import registry, server

    class _DummyService(Service):
        def __init__(self) -> None:
            super().__init__(name="models", module_name="test.contract")

        def get_routers(self):
            return []

    dummy_services: dict[str, Service] = {"models": _DummyService()}
    dummy_sidecars: dict[str, Callable] = {"adapters": sidecar_run_func}

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


def test_embedded_mode_starts_sidecar_thread_via_full_resolution_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """End-to-end: an explicitly requested sidecar threads through the full
    resolution path and actually runs when the app lifespan starts."""
    started = threading.Event()
    stopped = threading.Event()

    _patch_runner_registry(monkeypatch, sidecar_run_func=_sidecar_with_events(started, stopped))

    cfg = ServiceRunConfig(
        mode=services.ServiceMode.EMBEDDED,
        services=("models",),
        controllers=(),
        sidecars=("adapters",),
        transport="tcp",
        scope="sidecar-e2e",
        state_dir=tmp_path / "state",
        runtime_dir=tmp_path / "runtime",
    )
    handle = services.start_embedded_services(cfg, env={})

    from fastapi.testclient import TestClient

    with TestClient(handle.app) as client:
        assert started.wait(timeout=2.0), "sidecar thread did not start"
        assert client.get("/").status_code == 200

    assert stopped.wait(timeout=2.0), "sidecar thread did not stop"
