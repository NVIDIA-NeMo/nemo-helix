# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Programmatic local lifecycle API for NeMo Helix services."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Protocol, Self, runtime_checkable

from nemo_helix import AsyncNeMoHelix, NeMoHelix
from nemo_helix_ext.local import process
from nemo_helix_ext.local.install import services_extra_install_command
from nemo_helix_ext.local.transport import (
    EMBEDDED_BASE_URL,
    UDS_BASE_URL,
    build_async_asgi_http_client,
    build_async_http_client,
    build_sync_asgi_http_client,
    build_sync_http_client,
    probe_status,
    tcp_base_url,
    wait_for_status,
    wait_for_status_async,
)
from nhx.platform_runner.config import (
    DEFAULT_SCOPE,
    DEFAULT_UVICORN_KEEP_ALIVE_TIMEOUT_SECONDS,
    HelixAppConfig,
    default_runtime_root,
    default_state_root,
    validate_keep_alive_timeout_seconds,
    validate_scope,
)

_AF_UNIX_PATH_MAX_BYTES = 103 if sys.platform.startswith(("darwin", "freebsd", "openbsd", "netbsd")) else 107


class ServicesError(RuntimeError):
    """Base class for local services lifecycle errors."""


class ServicesExtraRequiredError(ServicesError):
    """Raised when local service dependencies are not installed."""


class ServicesAlreadyRunningError(ServicesError):
    """Raised when a requested local instance is already running."""


class ServicesNotRunningError(ServicesError):
    """Raised when a requested local instance is not running."""


class ServicesPortInUseError(ServicesError):
    """Raised when TCP startup targets an unavailable port."""


class ServicesStartupTimeoutError(ServicesError):
    """Raised when startup does not become healthy before the timeout."""


class ServicesStartupExitedError(ServicesError):
    """Raised when a daemon child exits before becoming healthy."""


class ServicesSocketStaleError(ServicesError):
    """Raised when a stale socket cannot be removed."""


def _as_tuple(value: Sequence[str] | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    return tuple(value)


def _optional_str(value: str | Path | None) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_list(value: Sequence[str] | None) -> list[str] | None:
    if value is None:
        return None
    return list(value)


class ServiceMode(StrEnum):
    EMBEDDED = "embedded"
    DAEMON = "daemon"


@dataclass(frozen=True)
class StartServicesResult:
    requested: list[str]
    started: list[str]
    already_active: list[str]
    active: list[str]


@runtime_checkable
class LocalServiceHandle(Protocol):
    """Shared lifecycle/client contract for local services handles."""

    def is_running(self) -> bool: ...

    def wait_until_ready(self, timeout: float | None = None) -> None: ...

    async def wait_until_ready_async(self, timeout: float | None = None) -> None: ...

    def client(self, **kwargs: Any) -> NeMoHelix: ...

    def async_client(self, **kwargs: Any) -> AsyncNeMoHelix: ...

    def start_services(self, service_names: Sequence[str]) -> StartServicesResult: ...

    async def start_services_async(self, service_names: Sequence[str]) -> StartServicesResult: ...

    def stop(self, *, timeout: float = 30.0, force: bool = False) -> process.StopResult: ...

    async def stop_async(self, *, timeout: float = 30.0, force: bool = False) -> process.StopResult: ...


@dataclass
class ServiceRunConfig:
    services: Sequence[str] | None = None
    service_group: str | None = None
    controllers: Sequence[str] | None = None
    controller_group: str | None = None
    sidecars: Sequence[str] | None = None
    config_path: str | Path | None = None
    transport: Literal["uds", "tcp"] = "uds"
    socket_path: str | Path | None = None
    http_gateway: Literal["enabled", "disabled"] = "disabled"
    http_gateway_host: str = "127.0.0.1"
    http_gateway_port: int | None = None
    host: str = "127.0.0.1"
    port: int = 8080
    scope: str = DEFAULT_SCOPE
    state_dir: str | Path | None = None
    runtime_dir: str | Path | None = None
    data_dir: str | Path | None = None
    readiness_timeout: float = 60.0
    readiness_poll_interval: float = 0.5
    keep_alive_timeout_seconds: int = DEFAULT_UVICORN_KEEP_ALIVE_TIMEOUT_SECONDS
    mode: ServiceMode | str = ServiceMode.DAEMON

    def __post_init__(self) -> None:
        self.services = _as_tuple(self.services)
        self.controllers = _as_tuple(self.controllers)
        self.sidecars = _as_tuple(self.sidecars)
        try:
            self.mode = ServiceMode(self.mode)
        except ValueError as error:
            raise ValueError("mode must be 'embedded' or 'daemon'") from error

        if self.services and self.service_group:
            raise ValueError("services cannot be combined with service_group")
        if self.controllers and self.controller_group:
            raise ValueError("controllers cannot be combined with controller_group")
        if self.transport not in {"uds", "tcp"}:
            raise ValueError("transport must be 'uds' or 'tcp'")
        if self.http_gateway not in {"enabled", "disabled"}:
            raise ValueError("http_gateway must be 'enabled' or 'disabled'")
        if self.http_gateway == "enabled" and self.transport != "uds":
            raise ValueError("gateway can only be enabled for UDS transport")
        if self.readiness_timeout <= 0:
            raise ValueError("readiness_timeout must be greater than 0")
        if self.readiness_poll_interval <= 0:
            raise ValueError("readiness_poll_interval must be greater than 0")
        self.keep_alive_timeout_seconds = validate_keep_alive_timeout_seconds(self.keep_alive_timeout_seconds)
        self.scope = validate_scope(self.scope)

    @property
    def state_root(self) -> Path:
        return Path(self.state_dir).expanduser() if self.state_dir is not None else default_state_root()

    @property
    def runtime_root(self) -> Path:
        return Path(self.runtime_dir).expanduser() if self.runtime_dir is not None else default_runtime_root()

    @property
    def resolved_socket_path(self) -> Path | None:
        if self.socket_path is not None:
            socket_path = Path(self.socket_path).expanduser()
        elif self.transport == "uds":
            socket_path = HelixAppConfig(
                scope=self.scope,
                runtime_root=self.runtime_root,
            ).socket_file_path()
        else:
            return None
        if not socket_path.is_absolute():
            raise ValueError(f"UDS socket path must be absolute: {socket_path}")
        return socket_path

    def to_platform_app_config(self) -> HelixAppConfig:
        return HelixAppConfig(
            services=self.services,
            service_group=self.service_group,
            controllers=self.controllers,
            controller_group=self.controller_group,
            sidecars=self.sidecars,
            config_path=_optional_str(self.config_path),
            scope=self.scope,
            host=self.host,
            port=self.port,
            socket_path=_optional_str(self.resolved_socket_path),
            state_root=_optional_str(self.state_root),
            runtime_root=_optional_str(self.runtime_dir),
            keep_alive_timeout_seconds=self.keep_alive_timeout_seconds,
        )

    def to_child_payload(self) -> dict[str, object]:
        return {
            "mode": ServiceMode(self.mode).value,
            "services": _optional_list(self.services),
            "service_group": self.service_group,
            "controllers": _optional_list(self.controllers),
            "controller_group": self.controller_group,
            "sidecars": _optional_list(self.sidecars),
            "config_path": _optional_str(self.config_path),
            "transport": self.transport,
            "socket_path": _optional_str(self.socket_path),
            "http_gateway": self.http_gateway,
            "http_gateway_host": self.http_gateway_host,
            "http_gateway_port": self.http_gateway_port,
            "host": self.host,
            "port": self.port,
            "scope": self.scope,
            "state_dir": _optional_str(self.state_dir),
            "runtime_dir": _optional_str(self.runtime_dir),
            "data_dir": _optional_str(self.data_dir),
            "readiness_timeout": self.readiness_timeout,
            "readiness_poll_interval": self.readiness_poll_interval,
            "keep_alive_timeout_seconds": self.keep_alive_timeout_seconds,
        }


@dataclass(frozen=True)
class DaemonServiceHandle:
    scope: str
    transport: Literal["uds", "tcp"]
    socket_path: Path | None
    gateway_base_url: str | None
    host: str
    port: int
    pid: int | None
    mode: Literal["foreground", "daemon"]
    log_path: Path | None
    state_dir: Path | None
    runtime_dir: Path | None

    @classmethod
    def from_descriptor(cls, desc: process.InstanceDescriptor) -> Self:
        socket_path = Path(desc.config.socket_path) if desc.config.socket_path else None
        runtime_dir = desc.config.runtime_dir() if socket_path else None
        return cls(
            scope=desc.config.scope,
            transport=desc.transport,
            socket_path=socket_path,
            gateway_base_url=None,
            host=desc.config.host,
            port=desc.config.port,
            pid=desc.pid,
            mode="daemon" if desc.mode == "daemon" else "foreground",
            log_path=desc.config.log_file_path(),
            state_dir=desc.config.state_dir(),
            runtime_dir=runtime_dir,
        )

    @classmethod
    def from_config(
        cls,
        config: ServiceRunConfig,
        *,
        pid: int | None = None,
    ) -> Self:
        app_config = config.to_platform_app_config()
        socket_path = config.resolved_socket_path
        runtime_dir = app_config.runtime_dir() if socket_path else None
        return cls(
            scope=config.scope,
            transport=config.transport,
            socket_path=socket_path,
            gateway_base_url=None,
            host=config.host,
            port=config.port,
            pid=pid,
            mode="daemon",
            log_path=app_config.log_file_path(),
            state_dir=app_config.state_dir(),
            runtime_dir=runtime_dir,
        )

    @property
    def base_url(self) -> str:
        if self.transport == "uds":
            return UDS_BASE_URL
        return tcp_base_url(self.host, self.port)

    def _state_root(self) -> Path | None:
        if self.state_dir is None:
            return None
        if self.state_dir.parent.name == "instances":
            return self.state_dir.parent.parent
        return self.state_dir.parent

    def is_running(self) -> bool:
        state_root = self._state_root()
        return process.is_instance_alive(self.scope, base_dir=state_root)

    def wait_until_ready(self, timeout: float | None = None) -> None:
        if not wait_for_status(
            base_url=self.base_url,
            socket_path=self.socket_path if self.transport == "uds" else None,
            timeout=60.0 if timeout is None else timeout,
        ):
            raise ServicesStartupTimeoutError(f"Timed out waiting for services instance {self.scope!r}")

    async def wait_until_ready_async(self, timeout: float | None = None) -> None:
        if not await wait_for_status_async(
            base_url=self.base_url,
            socket_path=self.socket_path if self.transport == "uds" else None,
            timeout=60.0 if timeout is None else timeout,
        ):
            raise ServicesStartupTimeoutError(f"Timed out waiting for services instance {self.scope!r}")

    def stop(self, *, timeout: float = 30.0, force: bool = False) -> process.StopResult:
        state_root = self._state_root()
        return process.stop_instance(self.scope, base_dir=state_root, timeout=timeout, force=force)

    async def stop_async(self, *, timeout: float = 30.0, force: bool = False) -> process.StopResult:
        return await asyncio.to_thread(self.stop, timeout=timeout, force=force)

    def start_services(self, service_names: Sequence[str]) -> StartServicesResult:
        raise ServicesError("Staged service start is not implemented for daemon mode yet")

    async def start_services_async(self, service_names: Sequence[str]) -> StartServicesResult:
        return await asyncio.to_thread(self.start_services, service_names)

    def client(self, **kwargs: Any) -> NeMoHelix:
        if self.transport == "uds":
            if self.socket_path is None:
                raise ServicesError("UDS service handle is missing socket_path")
            kwargs.setdefault("http_client", build_sync_http_client(self.socket_path))
        kwargs.setdefault("base_url", self.base_url)
        return NeMoHelix(**kwargs)

    def async_client(self, **kwargs: Any) -> AsyncNeMoHelix:
        if self.transport == "uds":
            if self.socket_path is None:
                raise ServicesError("UDS service handle is missing socket_path")
            kwargs.setdefault("http_client", build_async_http_client(self.socket_path))
        kwargs.setdefault("base_url", self.base_url)
        return AsyncNeMoHelix(**kwargs)


@dataclass(frozen=True)
class EmbeddedServiceHandle:
    app: Any
    runtime: object

    def is_running(self) -> bool:
        return True

    def wait_until_ready(self, timeout: float | None = None) -> None:
        return None

    async def wait_until_ready_async(self, timeout: float | None = None) -> None:
        return None

    def client(self, **kwargs: Any) -> NeMoHelix:
        kwargs.setdefault("http_client", build_sync_asgi_http_client(self.app))
        kwargs.setdefault("base_url", EMBEDDED_BASE_URL)
        return NeMoHelix(**kwargs)

    def async_client(self, **kwargs: Any) -> AsyncNeMoHelix:
        kwargs.setdefault("http_client", build_async_asgi_http_client(self.app))
        kwargs.setdefault("base_url", EMBEDDED_BASE_URL)
        return AsyncNeMoHelix(**kwargs)

    def start_services(self, service_names: Sequence[str]) -> StartServicesResult:
        raise ServicesError("Staged service start is not implemented for embedded mode yet")

    async def start_services_async(self, service_names: Sequence[str]) -> StartServicesResult:
        return await asyncio.to_thread(self.start_services, service_names)

    def stop(self, *, timeout: float = 30.0, force: bool = False) -> process.StopResult:
        return process.StopResult(stopped_pids=[], swept_children=[])

    async def stop_async(self, *, timeout: float = 30.0, force: bool = False) -> process.StopResult:
        return self.stop(timeout=timeout, force=force)


def require_services_extra() -> None:
    if importlib.util.find_spec("pyleak") is not None:
        return
    raise ServicesExtraRequiredError(f"Install service dependencies with `{services_extra_install_command()}`.")


def _validate_socket_path_length(socket_path: Path) -> None:
    encoded_length = len(os.fsencode(socket_path))
    if encoded_length > _AF_UNIX_PATH_MAX_BYTES:
        raise ValueError(
            "UDS socket path is too long for AF_UNIX "
            f"({encoded_length} bytes; maximum is {_AF_UNIX_PATH_MAX_BYTES} bytes): {socket_path}"
        )


def _validated_socket_path(config: ServiceRunConfig) -> Path | None:
    socket_path = config.resolved_socket_path
    if socket_path is None:
        return None
    _validate_socket_path_length(socket_path)
    return socket_path


def _prepare_socket(config: ServiceRunConfig) -> Path | None:
    socket_path = _validated_socket_path(config)
    if socket_path is None:
        return None
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    if not socket_path.exists():
        return socket_path
    if probe_status(base_url=UDS_BASE_URL, socket_path=socket_path, timeout=0.5):
        raise ServicesAlreadyRunningError(f"UDS socket is live at {socket_path}")
    try:
        socket_path.unlink()
    except OSError as error:
        raise ServicesSocketStaleError(f"Could not remove stale socket at {socket_path}") from error
    return socket_path


def _check_tcp_available(config: ServiceRunConfig) -> None:
    conflict = process.check_port_available_for_start(
        config.host,
        config.port,
        config.scope,
        base_dir=config.state_root,
    )
    if conflict is not None:
        raise ServicesPortInUseError("\n".join(process.format_port_conflict(conflict)))


def _write_run_request(config: ServiceRunConfig) -> Path:
    state_dir = config.to_platform_app_config().state_dir(create=True)
    fd, tmp = tempfile.mkstemp(dir=state_dir, suffix=".json")
    path = Path(tmp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            fd = -1
            json.dump(config.to_child_payload(), file, indent=2)
            file.write("\n")
    except BaseException:
        if fd >= 0:
            os.close(fd)
            fd = -1
        path.unlink(missing_ok=True)
        raise
    finally:
        if fd >= 0:
            os.close(fd)
    return path


def _terminate_startup_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def serve_embedded_app(app: Any, cfg: ServiceRunConfig, socket_path: Path | None) -> None:
    import uvicorn

    if socket_path is not None:
        from nhx.platform_runner.server import _run_server_on_bound_sockets

        _run_server_on_bound_sockets(
            app,
            host=cfg.host,
            port=cfg.port,
            socket_path=str(socket_path),
            keep_alive_timeout_seconds=cfg.keep_alive_timeout_seconds,
        )
    else:
        uvicorn.run(
            app,
            host=cfg.host,
            port=cfg.port,
            log_config=None,
            timeout_keep_alive=cfg.keep_alive_timeout_seconds,
        )


def run_services(
    config: ServiceRunConfig | None = None,
    *,
    _mode: Literal["foreground", "daemon"] = "foreground",
    env: MutableMapping[str, str] | None = None,
) -> None:
    cfg = config or ServiceRunConfig()
    app_config = cfg.to_platform_app_config()
    require_services_extra()
    if cfg.http_gateway == "enabled":
        raise ServicesError("HTTP gateway support is not implemented yet")
    if process.is_instance_alive(cfg.scope, base_dir=cfg.state_root):
        raise ServicesAlreadyRunningError(f"Instance {cfg.scope!r} is already running")
    _check_tcp_available(cfg)
    lock_fd = process.acquire_lock(cfg.scope, base_dir=cfg.state_root)
    original_data_dir = os.environ.get("NHX_DATA_DIR")
    try:
        socket_path = _prepare_socket(cfg)
        app_config.log_file_path(create_parent=True)
        if cfg.data_dir is not None and "NHX_DATA_DIR" not in os.environ:
            os.environ["NHX_DATA_DIR"] = str(cfg.data_dir)
        desc = process.InstanceDescriptor.from_config(
            app_config,
            mode=_mode,
            transport=cfg.transport,
        )
        process.write_descriptor(desc, base_dir=cfg.state_root)
        embedded_handle = start_embedded_services(cfg, env=env)
        serve_embedded_app(embedded_handle.app, cfg, socket_path)
    finally:
        try:
            process.remove_descriptor(cfg.scope, base_dir=cfg.state_root)
        finally:
            if original_data_dir is None:
                os.environ.pop("NHX_DATA_DIR", None)
            else:
                os.environ["NHX_DATA_DIR"] = original_data_dir
            os.close(lock_fd)


def daemonize_services(config: ServiceRunConfig | None = None) -> DaemonServiceHandle:
    cfg = config or ServiceRunConfig()
    app_config = cfg.to_platform_app_config()
    require_services_extra()
    if cfg.http_gateway == "enabled":
        raise ServicesError("HTTP gateway support is not implemented yet")
    if process.is_instance_alive(cfg.scope, base_dir=cfg.state_root):
        raise ServicesAlreadyRunningError(f"Instance {cfg.scope!r} is already running")
    _check_tcp_available(cfg)
    socket_path = _validated_socket_path(cfg)
    if (
        socket_path is not None
        and socket_path.exists()
        and probe_status(base_url=UDS_BASE_URL, socket_path=socket_path, timeout=0.5)
    ):
        raise ServicesAlreadyRunningError(f"UDS socket is live at {socket_path}")

    request_path = _write_run_request(cfg)
    log_path = process.rotate_log_path(app_config.log_file_path())
    log_file = open(log_path, "a")  # noqa: SIM115
    env = os.environ.copy()
    if cfg.data_dir is not None and "NHX_DATA_DIR" not in env:
        env["NHX_DATA_DIR"] = str(cfg.data_dir)
    proc: subprocess.Popen | None = None
    ownership_transferred = False
    try:
        try:
            child_module = f"{__package__}._service_child"
            proc = subprocess.Popen(
                [sys.executable, "-m", child_module, str(request_path)],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
                stdin=subprocess.DEVNULL,
                close_fds=True,
            )
        finally:
            log_file.close()
        assert proc is not None
        handle = DaemonServiceHandle.from_config(cfg, pid=proc.pid)
        deadline = time.monotonic() + cfg.readiness_timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            if proc.poll() is not None:
                raise ServicesStartupExitedError(f"Services daemon exited with code {proc.returncode}; log: {log_path}")
            if probe_status(
                base_url=handle.base_url,
                socket_path=handle.socket_path if handle.transport == "uds" else None,
                timeout=remaining,
            ):
                ownership_transferred = True
                return handle
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(cfg.readiness_poll_interval, remaining))
        raise ServicesStartupTimeoutError(f"Timed out waiting for services daemon {cfg.scope!r}; log: {log_path}")
    finally:
        if proc is not None and not ownership_transferred:
            _terminate_startup_process(proc)


async def daemonize_services_async(config: ServiceRunConfig | None = None) -> DaemonServiceHandle:
    return await asyncio.to_thread(daemonize_services, config)


def start_embedded_services(
    config: ServiceRunConfig | None = None,
    *,
    env: MutableMapping[str, str] | None = None,
) -> EmbeddedServiceHandle:
    """Start Helix services in the current process.

    Args:
        env: Environment mapping passed to :func:`build_platform_app`.
            Defaults to ``None`` which writes to ``os.environ``.  Tests can
            pass an empty dict to avoid polluting the process environment.
    """
    cfg = config or ServiceRunConfig(mode=ServiceMode.EMBEDDED)
    from nhx.platform_runner.server import build_platform_app

    app = build_platform_app(
        config=cfg.to_platform_app_config(),
        env=env,
    )
    runtime = getattr(app.state, "platform_runtime", None)
    return EmbeddedServiceHandle(app=app, runtime=runtime)


async def start_embedded_services_async(config: ServiceRunConfig | None = None) -> EmbeddedServiceHandle:
    return start_embedded_services(config)


def get_service_handle(config: ServiceRunConfig | None = None) -> DaemonServiceHandle | None:
    cfg = config or ServiceRunConfig()
    desc = process.read_descriptor(cfg.scope, base_dir=cfg.state_root)
    if desc is None or not process.is_instance_alive(cfg.scope, base_dir=cfg.state_root):
        return None
    return DaemonServiceHandle.from_descriptor(desc)


def list_service_handles(state_dir: str | Path | None = None) -> list[DaemonServiceHandle]:
    state_root = Path(state_dir).expanduser() if state_dir is not None else default_state_root()
    handles: list[DaemonServiceHandle] = []
    for info in process.list_instances(base_dir=state_root):
        if info.descriptor is not None and info.alive:
            handles.append(DaemonServiceHandle.from_descriptor(info.descriptor))
    return handles


def ensure_services(
    config: ServiceRunConfig | None = None,
    *,
    daemonize: bool | None = None,
) -> LocalServiceHandle:
    cfg = config or ServiceRunConfig()
    if cfg.mode is ServiceMode.EMBEDDED:
        return start_embedded_services(cfg)

    handle = get_service_handle(cfg)
    if handle is not None:
        return handle
    if daemonize is False:
        raise ServicesNotRunningError(f"Instance {cfg.scope!r} is not running")
    return daemonize_services(cfg)


async def ensure_services_async(
    config: ServiceRunConfig | None = None,
    *,
    daemonize: bool | None = None,
) -> LocalServiceHandle:
    cfg = config or ServiceRunConfig()
    if cfg.mode is ServiceMode.EMBEDDED:
        return await start_embedded_services_async(cfg)

    handle = get_service_handle(cfg)
    if handle is not None:
        return handle
    if daemonize is False:
        raise ServicesNotRunningError(f"Instance {cfg.scope!r} is not running")
    return await daemonize_services_async(cfg)


def connect_services(
    config: ServiceRunConfig | None = None,
    *,
    daemonize: bool | None = None,
    start_if_needed: bool = True,
    **client_kwargs: Any,
) -> NeMoHelix:
    cfg = config or ServiceRunConfig()
    if not start_if_needed and cfg.mode is ServiceMode.DAEMON and get_service_handle(cfg) is None:
        raise ServicesNotRunningError(f"Instance {cfg.scope!r} is not running")
    handle = ensure_services(cfg, daemonize=daemonize)
    return handle.client(**client_kwargs)


async def connect_services_async(
    config: ServiceRunConfig | None = None,
    *,
    daemonize: bool | None = None,
    start_if_needed: bool = True,
    **client_kwargs: Any,
) -> AsyncNeMoHelix:
    cfg = config or ServiceRunConfig()
    if not start_if_needed and cfg.mode is ServiceMode.DAEMON and get_service_handle(cfg) is None:
        raise ServicesNotRunningError(f"Instance {cfg.scope!r} is not running")
    handle = await ensure_services_async(cfg, daemonize=daemonize)
    return handle.async_client(**client_kwargs)


def stop_services(
    config: ServiceRunConfig | None = None,
    *,
    timeout: float = 30.0,
    force: bool = False,
) -> process.StopResult:
    cfg = config or ServiceRunConfig()
    handle = get_service_handle(cfg)
    if handle is None:
        raise ServicesNotRunningError(f"Instance {cfg.scope!r} is not running")
    return handle.stop(timeout=timeout, force=force)


async def stop_services_async(
    config: ServiceRunConfig | None = None,
    *,
    timeout: float = 30.0,
    force: bool = False,
) -> process.StopResult:
    return await asyncio.to_thread(stop_services, config, timeout=timeout, force=force)
