# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nemo_helix_ext.local import _service_child, services
from nemo_helix_ext.local.process import (
    DESCRIPTOR_FILENAME,
    InstanceDescriptor,
)
from nemo_helix_ext.local.services import ServiceRunConfig
from nemo_helix_ext.local.transport import UDS_BASE_URL
from nhx.platform_runner.config import (
    HelixAppConfig,
    default_runtime_root,
    default_state_root,
    validate_scope,
)


def _allow_tmp_path_socket_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(services, "_AF_UNIX_PATH_MAX_BYTES", 4096)


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _embedded_handle() -> services.EmbeddedServiceHandle:
    return services.EmbeddedServiceHandle(app=object(), runtime=object())


def test_service_run_config_normalizes_lists_to_tuples() -> None:
    cfg = ServiceRunConfig(services=["entities", "models"], controllers=["jobs"])

    assert cfg.services == ("entities", "models")
    assert cfg.controllers == ("jobs",)


def test_service_run_config_converts_to_platform_app_config(tmp_path: Path) -> None:
    cfg = ServiceRunConfig(
        services=["entities", "models"],
        controllers=[],
        sidecars=["adapters"],
        config_path=tmp_path / "local.yaml",
        socket_path=tmp_path / "nemo.sock",
        keep_alive_timeout_seconds=12,
        mode="embedded",
    )

    app_config = cfg.to_platform_app_config()

    assert app_config.services == ("entities", "models")
    assert app_config.controllers == ()
    assert app_config.sidecars == ("adapters",)
    assert app_config.config_path == str(tmp_path / "local.yaml")
    assert app_config.socket_path == str(tmp_path / "nemo.sock")
    assert app_config.runtime_root is None
    assert app_config.runtime_dir() == tmp_path
    assert app_config.host == "127.0.0.1"
    assert app_config.port == 8080
    assert app_config.keep_alive_timeout_seconds == 12


def test_service_run_config_rejects_non_positive_keep_alive_timeout() -> None:
    with pytest.raises(ValueError, match="keep_alive_timeout_seconds must be greater than 0"):
        ServiceRunConfig(keep_alive_timeout_seconds=0)


def test_instance_descriptor_converts_from_service_run_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = ServiceRunConfig(
        services=["entities", "models"],
        controllers=[],
        sidecars=["adapters"],
        config_path=tmp_path / "local.yaml",
        socket_path=tmp_path / "nemo.sock",
        state_dir=tmp_path / "state",
        runtime_dir=tmp_path / "run",
    )
    monkeypatch.setattr(services.process, "get_create_time", lambda _pid: 123.0)
    app_config = cfg.to_platform_app_config()
    app_config.log_path = str(tmp_path / "nemo.log")

    desc = InstanceDescriptor.from_config(
        app_config,
        pid=4242,
        mode="daemon",
        transport=cfg.transport,
    )

    assert desc.pid == 4242
    assert desc.config.scope == "default"
    assert desc.config.host == "127.0.0.1"
    assert desc.config.port == 8080
    assert desc.transport == "uds"
    assert desc.config.socket_path == str(tmp_path / "nemo.sock")
    assert desc.config.state_root == str(tmp_path / "state")
    assert desc.config.runtime_root == str(tmp_path / "run")
    assert desc.config.state_dir() == tmp_path / "state" / "instances" / "default"
    assert desc.config.runtime_dir() == tmp_path / "run" / "default"
    assert desc.mode == "daemon"
    assert desc.create_time == 123.0
    assert desc.config.services == ("entities", "models")
    assert desc.config.controllers == ()
    assert desc.config.sidecars == ("adapters",)
    assert desc.config.config_path == str(tmp_path / "local.yaml")
    assert desc.config.log_path == str(tmp_path / "nemo.log")
    assert desc.config.log_file_path() == tmp_path / "nemo.log"
    payload = desc.model_dump()
    assert "services" not in payload
    assert "host" not in payload
    assert "state_dir" not in payload
    assert "runtime_dir" not in payload
    assert "log_path" not in payload
    assert payload["config"]["services"] == ("entities", "models")
    assert payload["config"]["socket_path"] == str(tmp_path / "nemo.sock")
    assert payload["config"]["state_root"] == str(tmp_path / "state")
    assert payload["config"]["runtime_root"] == str(tmp_path / "run")
    assert payload["config"]["log_path"] == str(tmp_path / "nemo.log")


def test_service_mode_enum_values() -> None:
    assert services.ServiceMode.EMBEDDED.value == "embedded"
    assert services.ServiceMode.DAEMON.value == "daemon"


def test_service_run_config_defaults_to_daemon_mode() -> None:
    cfg = ServiceRunConfig()

    assert cfg.mode is services.ServiceMode.DAEMON


def test_service_run_config_accepts_mode_strings() -> None:
    cfg = ServiceRunConfig(mode="embedded")

    assert cfg.mode is services.ServiceMode.EMBEDDED


def test_service_run_config_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="mode must be 'embedded' or 'daemon'"):
        ServiceRunConfig(mode="foreground")


def test_embedded_and_daemon_handles_implement_local_service_handle(tmp_path: Path) -> None:
    embedded = services.EmbeddedServiceHandle(app=object(), runtime=object())
    daemon = services.DaemonServiceHandle(
        scope="dev",
        transport="tcp",
        socket_path=None,
        gateway_base_url=None,
        host="127.0.0.1",
        port=8080,
        pid=123,
        mode="daemon",
        log_path=None,
        state_dir=tmp_path / "state" / "instances" / "dev",
        runtime_dir=None,
    )

    assert isinstance(embedded, services.LocalServiceHandle)
    assert isinstance(daemon, services.LocalServiceHandle)


def test_start_services_result_is_shared_result_type() -> None:
    result = services.StartServicesResult(
        requested=["jobs"],
        started=["auth", "jobs"],
        already_active=[],
        active=["secrets", "auth", "jobs"],
    )

    assert result.requested == ["jobs"]
    assert result.started == ["auth", "jobs"]
    assert result.active == ["secrets", "auth", "jobs"]


def test_service_run_config_rejects_services_with_service_group() -> None:
    with pytest.raises(ValueError, match="services cannot be combined with service_group"):
        ServiceRunConfig(services=("entities",), service_group="all")


def test_service_run_config_defaults_to_named_uds_instance() -> None:
    cfg = ServiceRunConfig()

    assert cfg.transport == "uds"
    assert cfg.http_gateway == "disabled"
    assert cfg.scope == "default"
    assert cfg.socket_path is None


@pytest.mark.parametrize("instance", ["has space", "../bad"])
def test_service_run_config_rejects_invalid_scope_names(instance: str) -> None:
    with pytest.raises(ValueError, match="scope"):
        ServiceRunConfig(scope=instance)


def test_service_run_config_rejects_gateway_for_tcp_transport() -> None:
    with pytest.raises(ValueError, match="gateway.*UDS"):
        ServiceRunConfig(transport="tcp", http_gateway="enabled")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("readiness_timeout", 0.0, "readiness_timeout"),
        ("readiness_timeout", -1.0, "readiness_timeout"),
        ("readiness_poll_interval", 0.0, "readiness_poll_interval"),
        ("readiness_poll_interval", -1.0, "readiness_poll_interval"),
    ],
)
def test_service_run_config_rejects_non_positive_readiness_values(field: str, value: float, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        if field == "readiness_timeout":
            ServiceRunConfig(readiness_timeout=value)
        else:
            ServiceRunConfig(readiness_poll_interval=value)


def test_process_paths_follow_existing_nhx_state_convention(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

    assert default_state_root() == tmp_path / "state" / "nhx"
    assert default_runtime_root() == tmp_path / "state" / "nhx" / "run"
    assert (
        HelixAppConfig(scope="dev").socket_file_path()
        == tmp_path / "state" / "nhx" / "run" / "dev" / "nemo-helix.sock"
    )
    assert validate_scope("dev_1-2") == "dev_1-2"


def test_resolved_socket_path_rejects_relative_explicit_path() -> None:
    cfg = ServiceRunConfig(socket_path="relative.sock")

    with pytest.raises(ValueError, match="UDS socket path must be absolute"):
        _ = cfg.resolved_socket_path


def test_resolved_socket_path_rejects_relative_runtime_dir() -> None:
    cfg = ServiceRunConfig(runtime_dir="relative-run")

    with pytest.raises(ValueError, match="runtime root must be absolute"):
        _ = cfg.resolved_socket_path


def test_resolved_socket_path_rejects_relative_socket_path_with_tcp_client() -> None:
    cfg = ServiceRunConfig(transport="tcp", socket_path="relative.sock")

    with pytest.raises(ValueError, match="UDS socket path must be absolute"):
        _ = cfg.resolved_socket_path


def test_tcp_client_can_still_configure_uds_listener(tmp_path: Path) -> None:
    cfg = ServiceRunConfig(transport="tcp", socket_path=tmp_path / "nemo.sock")

    app_config = cfg.to_platform_app_config()

    assert app_config.socket_path == str(tmp_path / "nemo.sock")
    assert app_config.runtime_dir() == tmp_path


def test_instance_descriptor_rejects_uds_client_without_socket_path() -> None:
    with pytest.raises(ValueError, match="UDS client transport requires config.socket_path"):
        InstanceDescriptor(pid=1, config=HelixAppConfig(scope="dev"), transport="uds")


def test_prepare_socket_rejects_long_generated_path_before_creating_runtime_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(services, "_AF_UNIX_PATH_MAX_BYTES", 1, raising=False)
    runtime_root = tmp_path / "runtime"
    cfg = ServiceRunConfig(scope="dev", state_dir=tmp_path / "state", runtime_dir=runtime_root)

    with pytest.raises(ValueError, match="UDS socket path is too long.*AF_UNIX"):
        services._prepare_socket(cfg)

    assert not runtime_root.exists()


def test_validate_socket_path_length_reserves_trailing_nul(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(services, "_AF_UNIX_PATH_MAX_BYTES", 3, raising=False)

    services._validate_socket_path_length(Path("abc"))
    with pytest.raises(ValueError, match=r"4 bytes; maximum is 3 bytes"):
        services._validate_socket_path_length(Path("abcd"))


def test_prepare_socket_rejects_long_explicit_path_before_filesystem_or_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(services, "_AF_UNIX_PATH_MAX_BYTES", 1, raising=False)
    socket_parent = tmp_path / "explicit"
    cfg = ServiceRunConfig(socket_path=socket_parent / "nemo-helix.sock")

    with patch("nemo_helix_ext.local.services.probe_status") as probe_status:
        with pytest.raises(ValueError, match="UDS socket path is too long.*AF_UNIX"):
            services._prepare_socket(cfg)

    probe_status.assert_not_called()
    assert not socket_parent.exists()


def test_run_services_prepares_socket_after_acquiring_lock(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _allow_tmp_path_socket_paths(monkeypatch)
    cfg = ServiceRunConfig(
        scope="dev", port=_free_tcp_port(), state_dir=tmp_path / "state", runtime_dir=tmp_path / "run"
    )
    events: list[str] = []
    real_acquire_lock = services.process.acquire_lock

    def acquire_lock(scope: str, *, base_dir: Path | None = None) -> int:
        events.append("lock")
        return real_acquire_lock(scope, base_dir=base_dir)

    def prepare_socket(config: ServiceRunConfig) -> Path | None:
        events.append("prepare")
        lock_path = (
            services.process.instance_dir(config.scope, base_dir=config.state_root) / services.process.LOCK_FILENAME
        )
        assert lock_path.exists()
        return config.resolved_socket_path

    with (
        patch("nemo_helix_ext.local.services.require_services_extra"),
        patch("nemo_helix_ext.local.services.process.acquire_lock", side_effect=acquire_lock),
        patch("nemo_helix_ext.local.services._prepare_socket", side_effect=prepare_socket),
        patch("nemo_helix_ext.local.services.start_embedded_services", return_value=_embedded_handle()),
        patch("nemo_helix_ext.local.services.serve_embedded_app"),
    ):
        services.run_services(cfg)

    assert events == ["lock", "prepare"]


def test_run_services_cleans_lock_when_socket_prepare_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _allow_tmp_path_socket_paths(monkeypatch)
    cfg = ServiceRunConfig(
        scope="dev", port=_free_tcp_port(), state_dir=tmp_path / "state", runtime_dir=tmp_path / "run"
    )

    with (
        patch("nemo_helix_ext.local.services.require_services_extra"),
        patch(
            "nemo_helix_ext.local.services._prepare_socket",
            side_effect=services.ServicesSocketStaleError("boom"),
        ),
    ):
        with pytest.raises(services.ServicesSocketStaleError, match="boom"):
            services.run_services(cfg)

    assert not services.process.is_instance_alive(cfg.scope, base_dir=cfg.state_root)


def test_run_services_restores_env_and_closes_lock_when_descriptor_cleanup_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _allow_tmp_path_socket_paths(monkeypatch)
    cfg = ServiceRunConfig(
        scope="dev",
        port=_free_tcp_port(),
        state_dir=tmp_path / "state",
        runtime_dir=tmp_path / "run",
        data_dir=tmp_path / "data",
    )
    monkeypatch.delenv("NHX_DATA_DIR", raising=False)
    real_acquire_lock = services.process.acquire_lock
    locked_fd: int | None = None

    def acquire_lock(scope: str, *, base_dir: Path | None = None) -> int:
        nonlocal locked_fd
        locked_fd = real_acquire_lock(scope, base_dir=base_dir)
        return locked_fd

    real_close = os.close
    closed_fds: list[int] = []

    def close(fd: int) -> None:
        closed_fds.append(fd)
        real_close(fd)

    monkeypatch.setattr(os, "close", close)
    with (
        patch("nemo_helix_ext.local.services.require_services_extra"),
        patch("nemo_helix_ext.local.services.process.acquire_lock", side_effect=acquire_lock),
        patch(
            "nemo_helix_ext.local.services.process.remove_descriptor",
            side_effect=RuntimeError("descriptor cleanup failed"),
        ),
        patch("nemo_helix_ext.local.services.start_embedded_services", return_value=_embedded_handle()),
        patch("nemo_helix_ext.local.services.serve_embedded_app"),
    ):
        with pytest.raises(RuntimeError, match="descriptor cleanup failed"):
            services.run_services(cfg)

    assert "NHX_DATA_DIR" not in os.environ
    assert locked_fd is not None
    assert locked_fd in closed_fds
    assert closed_fds[-1] == locked_fd


def test_run_services_foreground_serves_embedded_app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _allow_tmp_path_socket_paths(monkeypatch)
    cfg = ServiceRunConfig(
        mode=services.ServiceMode.EMBEDDED,
        scope="dev",
        port=_free_tcp_port(),
        state_dir=tmp_path / "state",
        runtime_dir=tmp_path / "run",
    )
    app = object()
    handle = services.EmbeddedServiceHandle(app=app, runtime=object())

    with (
        patch("nemo_helix_ext.local.services.require_services_extra"),
        patch("nemo_helix_ext.local.services.start_embedded_services", return_value=handle) as start_embedded,
        patch("nemo_helix_ext.local.services.serve_embedded_app") as serve_embedded,
    ):
        services.run_services(cfg)

    start_embedded.assert_called_once_with(cfg, env=None)
    serve_embedded.assert_called_once()
    assert serve_embedded.call_args.args[0] is app


def test_daemon_service_handle_uds_client_uses_socket_transport(tmp_path: Path) -> None:
    socket_path = tmp_path / "nemo-helix.sock"
    handle = services.DaemonServiceHandle(
        scope="dev",
        transport="uds",
        socket_path=socket_path,
        gateway_base_url="http://127.0.0.1:9999",
        host="127.0.0.1",
        port=8080,
        pid=123,
        mode="daemon",
        log_path=None,
        state_dir=tmp_path / "state" / "instances" / "dev",
        runtime_dir=tmp_path,
    )

    client = handle.client()
    try:
        assert str(client.base_url).rstrip("/") == UDS_BASE_URL
        assert handle.gateway_base_url == "http://127.0.0.1:9999"
    finally:
        client.close()


def test_embedded_handle_async_client_uses_asgi_transport() -> None:
    app = MagicMock()
    runtime = MagicMock()
    http_client = object()
    client_value = object()
    handle = services.EmbeddedServiceHandle(app=app, runtime=runtime)

    with (
        patch(
            "nemo_helix_ext.local.services.build_async_asgi_http_client", return_value=http_client
        ) as build_client,
        patch("nemo_helix_ext.local.services.AsyncNeMoHelix", return_value=client_value) as platform_cls,
    ):
        client = handle.async_client(access_token="test-token")

    build_client.assert_called_once_with(app)
    platform_cls.assert_called_once_with(
        access_token="test-token",
        http_client=http_client,
        base_url=services.EMBEDDED_BASE_URL,
    )
    assert client is client_value


def test_ensure_services_dispatches_to_embedded_mode() -> None:
    cfg = ServiceRunConfig(mode=services.ServiceMode.EMBEDDED)
    embedded_handle = MagicMock(spec=services.EmbeddedServiceHandle)

    with patch("nemo_helix_ext.local.services.start_embedded_services", return_value=embedded_handle):
        handle = services.ensure_services(cfg)

    assert handle is embedded_handle


def test_ensure_services_dispatches_to_daemon_mode() -> None:
    cfg = ServiceRunConfig(mode=services.ServiceMode.DAEMON)
    daemon_handle = MagicMock(spec=services.DaemonServiceHandle)

    with (
        patch("nemo_helix_ext.local.services.get_service_handle", return_value=None),
        patch("nemo_helix_ext.local.services.daemonize_services", return_value=daemon_handle),
    ):
        handle = services.ensure_services(cfg)

    assert handle is daemon_handle


def test_connect_services_uses_selected_mode_handle_client() -> None:
    cfg = ServiceRunConfig(mode=services.ServiceMode.EMBEDDED)
    handle = MagicMock(spec=services.EmbeddedServiceHandle)
    client = object()
    handle.client.return_value = client

    with patch("nemo_helix_ext.local.services.ensure_services", return_value=handle):
        result = services.connect_services(cfg, access_token="test")

    assert result is client
    handle.client.assert_called_once_with(access_token="test")


@pytest.mark.parametrize("mode", [services.ServiceMode.EMBEDDED, services.ServiceMode.DAEMON])
def test_ensure_services_returns_handle_with_parity_methods(mode: services.ServiceMode) -> None:
    cfg = ServiceRunConfig(mode=mode)
    if mode is services.ServiceMode.EMBEDDED:
        handle = MagicMock(spec=services.EmbeddedServiceHandle)
        patch_target = "nemo_helix_ext.local.services.start_embedded_services"
    else:
        handle = MagicMock(spec=services.DaemonServiceHandle)
        patch_target = "nemo_helix_ext.local.services.daemonize_services"

    with (
        patch("nemo_helix_ext.local.services.get_service_handle", return_value=None),
        patch(patch_target, return_value=handle),
    ):
        result = services.ensure_services(cfg)

    assert result is handle
    for method_name in (
        "is_running",
        "wait_until_ready",
        "wait_until_ready_async",
        "client",
        "async_client",
        "start_services",
        "start_services_async",
        "stop",
        "stop_async",
    ):
        assert hasattr(result, method_name), method_name


def test_daemonize_services_starts_child_module(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _allow_tmp_path_socket_paths(monkeypatch)
    cfg = ServiceRunConfig(
        service_group="all",
        scope="dev",
        state_dir=tmp_path / "state",
        runtime_dir=tmp_path / "run",
        readiness_timeout=0.1,
        readiness_poll_interval=0.01,
    )
    proc = MagicMock()
    proc.pid = 4242
    proc.poll.return_value = None

    with (
        patch("nemo_helix_ext.local.services.require_services_extra"),
        patch("nemo_helix_ext.local.services._check_tcp_available"),
        patch("nemo_helix_ext.local.services.probe_status", return_value=True),
        patch("nemo_helix_ext.local.services.subprocess.Popen", return_value=proc) as popen,
    ):
        handle = services.daemonize_services(cfg)

    args = popen.call_args.args[0]
    assert args[:3] == [sys.executable, "-m", f"{services.__package__}._service_child"]
    request_path = Path(args[3])
    assert request_path.parent == tmp_path / "state" / "instances" / "dev"
    assert request_path.suffix == ".json"
    assert request_path.name != "run-request.json"
    assert handle.transport == "uds"
    assert handle.socket_path == tmp_path / "run" / "dev" / "nemo-helix.sock"
    assert handle.pid == 4242
    proc.terminate.assert_not_called()
    proc.kill.assert_not_called()


def test_daemonize_services_leaves_socket_preparation_to_child(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _allow_tmp_path_socket_paths(monkeypatch)
    cfg = ServiceRunConfig(
        scope="dev",
        state_dir=tmp_path / "state",
        runtime_dir=tmp_path / "run",
        readiness_timeout=0.1,
        readiness_poll_interval=0.01,
    )
    socket_path = cfg.resolved_socket_path
    assert socket_path is not None
    socket_path.parent.mkdir(parents=True)
    socket_path.write_text("stale", encoding="utf-8")
    proc = MagicMock()
    proc.pid = 4242
    proc.poll.return_value = None

    with (
        patch("nemo_helix_ext.local.services.require_services_extra"),
        patch("nemo_helix_ext.local.services._prepare_socket", side_effect=AssertionError("parent prepared socket")),
        patch("nemo_helix_ext.local.services.probe_status", side_effect=[False, True]),
        patch("nemo_helix_ext.local.services.subprocess.Popen", return_value=proc),
    ):
        handle = services.daemonize_services(cfg)

    assert handle.socket_path == socket_path
    assert socket_path.read_text(encoding="utf-8") == "stale"


def test_write_run_request_writes_complete_payload_when_os_write_is_short(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = ServiceRunConfig(scope="dev", state_dir=tmp_path / "state", runtime_dir=tmp_path / "run")
    real_write = os.write

    def short_write(fd: int, data: bytes) -> int:
        return real_write(fd, data[: max(1, len(data) // 2)])

    monkeypatch.setattr(services.os, "write", short_write)

    request_path = services._write_run_request(cfg)

    expected_payload = json.dumps(cfg.to_child_payload(), indent=2) + "\n"
    assert request_path.read_text(encoding="utf-8") == expected_payload


def test_service_child_unlinks_request_after_read(tmp_path: Path) -> None:
    request_path = tmp_path / "run-request.json"
    cfg = ServiceRunConfig(scope="dev", state_dir=tmp_path / "state", runtime_dir=tmp_path / "run")
    request_path.write_text(json.dumps(cfg.to_child_payload()), encoding="utf-8")

    with patch("nemo_helix_ext.local._service_child.run_services") as run_services:
        result = _service_child.main([str(request_path)])

    assert result == 0
    assert not request_path.exists()
    child_cfg = run_services.call_args.args[0]
    assert child_cfg.scope == "dev"
    assert child_cfg.state_dir == str(tmp_path / "state")
    assert child_cfg.runtime_dir == str(tmp_path / "run")
    assert run_services.call_args.kwargs == {"_mode": "daemon"}


def test_daemonize_services_terminates_child_on_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _allow_tmp_path_socket_paths(monkeypatch)
    cfg = ServiceRunConfig(
        scope="dev",
        state_dir=tmp_path / "state",
        runtime_dir=tmp_path / "run",
        readiness_timeout=0.01,
        readiness_poll_interval=0.001,
    )
    proc = MagicMock()
    proc.pid = 4242
    proc.poll.return_value = None

    with (
        patch("nemo_helix_ext.local.services.require_services_extra"),
        patch("nemo_helix_ext.local.services.probe_status", return_value=False),
        patch("nemo_helix_ext.local.services.subprocess.Popen", return_value=proc),
    ):
        with pytest.raises(services.ServicesStartupTimeoutError):
            services.daemonize_services(cfg)

    proc.terminate.assert_called_once_with()
    proc.kill.assert_not_called()


def test_daemonize_services_bounds_probe_and_sleep_by_remaining_deadline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _allow_tmp_path_socket_paths(monkeypatch)
    cfg = ServiceRunConfig(
        scope="dev",
        state_dir=tmp_path / "state",
        runtime_dir=tmp_path / "run",
        readiness_timeout=5.0,
        readiness_poll_interval=10.0,
    )
    proc = MagicMock()
    proc.pid = 4242
    proc.poll.return_value = None
    clock = 0.0
    sleep_calls: list[float] = []

    def monotonic() -> float:
        nonlocal clock
        if clock == 0.0:
            clock = 4.0
            return 0.0
        return clock

    def probe_status(*_args: object, **_kwargs: object) -> bool:
        nonlocal clock
        clock = 4.5
        return False

    def sleep(duration: float) -> None:
        nonlocal clock
        sleep_calls.append(duration)
        clock += duration

    with (
        patch("nemo_helix_ext.local.services.require_services_extra"),
        patch("nemo_helix_ext.local.services.probe_status", side_effect=probe_status) as probe_status_mock,
        patch("nemo_helix_ext.local.services.subprocess.Popen", return_value=proc),
        patch("nemo_helix_ext.local.services.time.monotonic", side_effect=monotonic),
        patch("nemo_helix_ext.local.services.time.sleep", side_effect=sleep),
    ):
        with pytest.raises(services.ServicesStartupTimeoutError):
            services.daemonize_services(cfg)

    assert probe_status_mock.call_args.kwargs["timeout"] == pytest.approx(1.0)
    assert sleep_calls == [pytest.approx(0.5)]
    proc.terminate.assert_called_once_with()


def test_daemonize_services_terminates_child_on_handle_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _allow_tmp_path_socket_paths(monkeypatch)
    cfg = ServiceRunConfig(
        scope="dev",
        state_dir=tmp_path / "state",
        runtime_dir=tmp_path / "run",
        transport="tcp",
    )
    proc = MagicMock()
    proc.pid = 4242
    proc.poll.return_value = None

    with (
        patch("nemo_helix_ext.local.services.require_services_extra"),
        patch("nemo_helix_ext.local.services._check_tcp_available"),
        patch("nemo_helix_ext.local.services.subprocess.Popen", return_value=proc),
        patch(
            "nemo_helix_ext.local.services.DaemonServiceHandle.from_config",
            side_effect=RuntimeError("handle failed"),
        ),
    ):
        with pytest.raises(RuntimeError, match="handle failed"):
            services.daemonize_services(cfg)

    proc.terminate.assert_called_once_with()
    proc.kill.assert_not_called()


def test_daemonize_services_terminates_child_on_probe_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _allow_tmp_path_socket_paths(monkeypatch)
    cfg = ServiceRunConfig(
        scope="dev",
        state_dir=tmp_path / "state",
        runtime_dir=tmp_path / "run",
        transport="tcp",
    )
    proc = MagicMock()
    proc.pid = 4242
    proc.poll.return_value = None

    with (
        patch("nemo_helix_ext.local.services.require_services_extra"),
        patch("nemo_helix_ext.local.services._check_tcp_available"),
        patch("nemo_helix_ext.local.services.probe_status", side_effect=RuntimeError("probe failed")),
        patch("nemo_helix_ext.local.services.subprocess.Popen", return_value=proc),
    ):
        with pytest.raises(RuntimeError, match="probe failed"):
            services.daemonize_services(cfg)

    proc.terminate.assert_called_once_with()
    proc.kill.assert_not_called()


def test_daemonize_services_terminates_child_on_sleep_interruption(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _allow_tmp_path_socket_paths(monkeypatch)
    cfg = ServiceRunConfig(
        scope="dev",
        state_dir=tmp_path / "state",
        runtime_dir=tmp_path / "run",
        transport="tcp",
    )
    proc = MagicMock()
    proc.pid = 4242
    proc.poll.return_value = None

    with (
        patch("nemo_helix_ext.local.services.require_services_extra"),
        patch("nemo_helix_ext.local.services._check_tcp_available"),
        patch("nemo_helix_ext.local.services.probe_status", return_value=False),
        patch("nemo_helix_ext.local.services.subprocess.Popen", return_value=proc),
        patch("nemo_helix_ext.local.services.time.sleep", side_effect=KeyboardInterrupt),
    ):
        with pytest.raises(KeyboardInterrupt):
            services.daemonize_services(cfg)

    proc.terminate.assert_called_once_with()
    proc.kill.assert_not_called()


def test_daemonize_services_kills_child_when_terminate_times_out(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _allow_tmp_path_socket_paths(monkeypatch)
    cfg = ServiceRunConfig(
        scope="dev",
        state_dir=tmp_path / "state",
        runtime_dir=tmp_path / "run",
        transport="tcp",
        readiness_timeout=0.01,
        readiness_poll_interval=0.001,
    )
    proc = MagicMock()
    proc.pid = 4242
    proc.poll.return_value = None
    proc.wait.side_effect = [subprocess.TimeoutExpired("nemo services", 5), None]

    with (
        patch("nemo_helix_ext.local.services.require_services_extra"),
        patch("nemo_helix_ext.local.services._check_tcp_available"),
        patch("nemo_helix_ext.local.services.probe_status", return_value=False),
        patch("nemo_helix_ext.local.services.subprocess.Popen", return_value=proc),
    ):
        with pytest.raises(services.ServicesStartupTimeoutError):
            services.daemonize_services(cfg)

    proc.terminate.assert_called_once_with()
    proc.kill.assert_called_once_with()


async def test_daemonize_services_async_uses_thread(tmp_path: Path) -> None:
    cfg = ServiceRunConfig(scope="dev", state_dir=tmp_path / "state", runtime_dir=tmp_path / "run")
    handle = MagicMock()

    with patch("nemo_helix_ext.local.services.asyncio.to_thread", new=AsyncMock(return_value=handle)) as to_thread:
        result = await services.daemonize_services_async(cfg)

    assert result is handle
    to_thread.assert_awaited_once_with(services.daemonize_services, cfg)


def test_run_services_serves_embedded_app_with_socket_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _allow_tmp_path_socket_paths(monkeypatch)
    cfg = ServiceRunConfig(
        services=["entities"],
        controllers=["jobs"],
        scope="dev",
        port=_free_tcp_port(),
        state_dir=tmp_path / "state",
        runtime_dir=tmp_path / "run",
    )
    handle = _embedded_handle()

    with (
        patch("nemo_helix_ext.local.services.require_services_extra"),
        patch("nemo_helix_ext.local.services.start_embedded_services", return_value=handle) as start_embedded,
        patch("nemo_helix_ext.local.services.serve_embedded_app") as serve_embedded,
    ):
        services.run_services(cfg, _mode="daemon")

    start_embedded.assert_called_once_with(cfg, env=None)
    serve_embedded.assert_called_once_with(handle.app, cfg, tmp_path / "run" / "dev" / "nemo-helix.sock")
    assert not (tmp_path / "state" / "instances" / "dev" / DESCRIPTOR_FILENAME).exists()


def test_serve_embedded_app_with_socket_path_listens_on_tcp_and_uds(tmp_path: Path) -> None:
    cfg = ServiceRunConfig(
        transport="tcp",
        host="127.0.0.1",
        port=9090,
        keep_alive_timeout_seconds=12,
    )
    app = object()
    socket_path = tmp_path / "nemo.sock"

    with patch("nhx.platform_runner.server._run_server_on_bound_sockets") as run_bound_sockets:
        services.serve_embedded_app(app, cfg, socket_path)

    run_bound_sockets.assert_called_once_with(
        app,
        host="127.0.0.1",
        port=9090,
        socket_path=str(socket_path),
        keep_alive_timeout_seconds=12,
    )


def test_serve_embedded_app_without_socket_path_sets_keep_alive_timeout() -> None:
    cfg = ServiceRunConfig(
        transport="tcp",
        host="127.0.0.1",
        port=9090,
        keep_alive_timeout_seconds=12,
    )
    app = object()

    with patch("uvicorn.run") as uvicorn_run:
        services.serve_embedded_app(app, cfg, None)

    uvicorn_run.assert_called_once_with(
        app,
        host="127.0.0.1",
        port=9090,
        log_config=None,
        timeout_keep_alive=12,
    )


def test_run_services_cleans_lock_when_log_path_resolution_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _allow_tmp_path_socket_paths(monkeypatch)
    cfg = ServiceRunConfig(
        scope="dev", port=_free_tcp_port(), state_dir=tmp_path / "state", runtime_dir=tmp_path / "run"
    )

    with (
        patch("nemo_helix_ext.local.services.require_services_extra"),
        patch.object(HelixAppConfig, "log_file_path", side_effect=RuntimeError("boom")),
    ):
        with pytest.raises(RuntimeError, match="boom"):
            services.run_services(cfg)

    assert not services.process.is_instance_alive(cfg.scope, base_dir=cfg.state_root)


def test_run_services_restores_data_dir_and_lock_when_descriptor_write_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _allow_tmp_path_socket_paths(monkeypatch)
    cfg = ServiceRunConfig(
        scope="dev",
        port=_free_tcp_port(),
        state_dir=tmp_path / "state",
        runtime_dir=tmp_path / "run",
        data_dir=tmp_path / "data",
    )
    monkeypatch.delenv("NHX_DATA_DIR", raising=False)

    with (
        patch("nemo_helix_ext.local.services.require_services_extra"),
        patch("nemo_helix_ext.local.services.process.write_descriptor", side_effect=RuntimeError("boom")),
    ):
        with pytest.raises(RuntimeError, match="boom"):
            services.run_services(cfg)

    assert "NHX_DATA_DIR" not in os.environ
    assert not services.process.is_instance_alive(cfg.scope, base_dir=cfg.state_root)


def test_run_services_restores_existing_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _allow_tmp_path_socket_paths(monkeypatch)
    cfg = ServiceRunConfig(
        scope="dev", port=_free_tcp_port(), state_dir=tmp_path / "state", runtime_dir=tmp_path / "run"
    )
    monkeypatch.setenv("NHX_DATA_DIR", "/shell/data")

    with (
        patch("nemo_helix_ext.local.services.require_services_extra"),
        patch("nemo_helix_ext.local.services.start_embedded_services", return_value=_embedded_handle()),
        patch("nemo_helix_ext.local.services.serve_embedded_app"),
    ):
        services.run_services(cfg)

    assert os.environ["NHX_DATA_DIR"] == "/shell/data"


def test_daemon_service_handle_tcp_client_uses_tcp_base_url(tmp_path: Path) -> None:
    handle = services.DaemonServiceHandle(
        scope="dev",
        transport="tcp",
        socket_path=None,
        gateway_base_url=None,
        host="0.0.0.0",
        port=9090,
        pid=123,
        mode="daemon",
        log_path=None,
        state_dir=tmp_path / "state" / "instances" / "dev",
        runtime_dir=None,
    )

    with patch("nemo_helix_ext.local.services.NeMoHelix") as sdk:
        handle.client(timeout=12)

    sdk.assert_called_once_with(timeout=12, base_url="http://localhost:9090")


def test_daemon_service_handle_uds_client_requires_socket_path(tmp_path: Path) -> None:
    handle = services.DaemonServiceHandle(
        scope="dev",
        transport="uds",
        socket_path=None,
        gateway_base_url=None,
        host="127.0.0.1",
        port=8080,
        pid=123,
        mode="daemon",
        log_path=None,
        state_dir=tmp_path / "state" / "instances" / "dev",
        runtime_dir=tmp_path / "run",
    )

    with pytest.raises(services.ServicesError, match="missing socket_path"):
        handle.client()


def test_ensure_services_returns_existing_handle(tmp_path: Path) -> None:
    cfg = ServiceRunConfig(scope="dev", state_dir=tmp_path / "state", runtime_dir=tmp_path / "run")
    handle = MagicMock()

    with (
        patch("nemo_helix_ext.local.services.get_service_handle", return_value=handle),
        patch("nemo_helix_ext.local.services.daemonize_services") as daemonize,
    ):
        result = services.ensure_services(cfg)

    assert result is handle
    daemonize.assert_not_called()


def test_connect_services_respects_start_if_needed_false(tmp_path: Path) -> None:
    cfg = ServiceRunConfig(scope="dev", state_dir=tmp_path / "state", runtime_dir=tmp_path / "run")

    with patch("nemo_helix_ext.local.services.get_service_handle", return_value=None):
        with pytest.raises(services.ServicesNotRunningError, match="not running"):
            services.connect_services(cfg, start_if_needed=False)


def test_stop_services_delegates_to_handle(tmp_path: Path) -> None:
    cfg = ServiceRunConfig(scope="dev", state_dir=tmp_path / "state", runtime_dir=tmp_path / "run")
    handle = MagicMock()
    stop_result = MagicMock()
    handle.stop.return_value = stop_result

    with patch("nemo_helix_ext.local.services.get_service_handle", return_value=handle):
        result = services.stop_services(cfg, timeout=3.0, force=True)

    assert result is stop_result
    handle.stop.assert_called_once_with(timeout=3.0, force=True)


def test_get_service_handle_returns_none_without_live_descriptor(tmp_path: Path) -> None:
    cfg = ServiceRunConfig(scope="dev", state_dir=tmp_path / "state", runtime_dir=tmp_path / "run")

    with patch("nemo_helix_ext.local.services.process.read_descriptor", return_value=None):
        assert services.get_service_handle(cfg) is None


def test_list_service_handles_filters_dead_or_descriptorless_instances(tmp_path: Path) -> None:
    live_desc = InstanceDescriptor(
        pid=123,
        transport="tcp",
        config=HelixAppConfig(scope="live", state_root=tmp_path / "state"),
        mode="daemon",
    )
    infos = [
        MagicMock(descriptor=live_desc, alive=True),
        MagicMock(descriptor=None, alive=True),
        MagicMock(descriptor=live_desc, alive=False),
    ]

    with patch("nemo_helix_ext.local.services.process.list_instances", return_value=infos):
        handles = services.list_service_handles(tmp_path / "state")

    assert [handle.scope for handle in handles] == ["live"]


def test_get_service_handle_reads_live_descriptor(tmp_path: Path) -> None:
    cfg = ServiceRunConfig(scope="dev", state_dir=tmp_path / "state", runtime_dir=tmp_path / "run")
    state_dir = tmp_path / "state" / "instances" / "dev"
    state_dir.mkdir(parents=True)
    desc = InstanceDescriptor(
        pid=123,
        config=HelixAppConfig(
            scope="dev",
            socket_path=str(tmp_path / "run" / "dev" / "nemo-helix.sock"),
            state_root=tmp_path / "state",
            runtime_root=tmp_path / "run",
        ),
        transport="uds",
        mode="daemon",
    )
    (state_dir / DESCRIPTOR_FILENAME).write_text(desc.model_dump_json(), encoding="utf-8")

    with patch("nemo_helix_ext.local.process.is_instance_alive", return_value=True):
        handle = services.get_service_handle(cfg)

    assert handle is not None
    assert handle.scope == "dev"
    assert handle.transport == "uds"
