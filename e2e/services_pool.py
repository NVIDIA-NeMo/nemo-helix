# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared E2E services-pool implementation used by pytest fixtures."""

import hashlib
import json
import logging
import os
import socket
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path
from string import Template
from typing import Any, Callable, Literal, NotRequired, TypedDict, cast

import httpx
import pytest
import yaml
from _pytest.nodes import Node
from nhx.testing.e2e import Docker as DockerE2EBackend
from nhx.testing.e2e.config import deep_merge

from e2e.backends.docker_compose import DockerComposeE2EBackend

logger = logging.getLogger(__name__)
_E2E_HARNESS_DEBUG = os.environ.get("E2E_HARNESS_DEBUG") == "1"

_HEALTH_TIMEOUT = 60
_HEALTH_POLL_INTERVAL = 1.0
_AUTH_READY_TIMEOUT = 60
_E2E_ADMIN_EMAIL = "admin@example.com"
_E2E_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_E2E_PLATFORM_CONFIG = _E2E_REPO_ROOT / "packages/nhx_platform/config/local.yaml"
# Layered on the default config so pooled platforms that are not testing
# deployments orphan cleanup cannot delete peer platforms' docker containers.
_DEFAULT_E2E_DISABLE_DEPLOYMENTS_ORPHAN_CLEANUP = _E2E_REPO_ROOT / "e2e/configs/disable-deployments-orphan-cleanup.yaml"
_E2E_COMPOSE_LIFECYCLE_ENV = "NHX_E2E_COMPOSE_LIFECYCLE"


def admin_headers() -> dict[str, str]:
    return {
        "X-NHX-Principal-Id": _E2E_ADMIN_EMAIL,
        "X-NHX-Principal-Email": _E2E_ADMIN_EMAIL,
    }


@dataclass(frozen=True)
class ServicesPoolKey:
    config_hash: str


class DynamicPortConfig(TypedDict, total=False):
    host: str
    port: str | int
    scheme: str


class E2EHarnessConfig(TypedDict, total=False):
    backend: Literal["subprocess", "docker", "docker_compose"]
    compose_file: str
    compose_project_name: str
    service_url: str
    auth_ready_url: str
    wait_url: str
    wait_urls: list[str]
    lifecycle: Literal["fresh", "reuse"]
    compose_project_prefix: str
    env: dict[str, str]
    dynamic_ports: dict[str, DynamicPortConfig]
    # Subprocess backend only; container-reachable host for platform.base_url (see
    # _start_services_subprocess for how it's applied).
    container_base_url_host: str


@dataclass
class RunningServices:
    url: str
    log_path: Path | None
    proc: subprocess.Popen[Any] | None
    config_path: Path | None
    close: Callable[[], None] | None = None
    auth_enabled: bool = False
    key: ServicesPoolKey | None = None
    docker_network_name: str | None = None
    docker_container_alias: str | None = None
    docker_container_port: int | None = None
    compose_project_name: str | None = None


@dataclass(frozen=True)
class ModuleConfigState:
    module_id: str
    key: ServicesPoolKey
    config_path: Path | None
    config_data: dict[str, Any]
    harness_config: E2EHarnessConfig
    config_layers: tuple[str, ...]
    auth_enabled: bool


class E2EServicesPool:
    """Central manager for config-hash-based E2E service pooling."""

    def __init__(self) -> None:
        self._tmp_path_factory: pytest.TempPathFactory | None = None
        self._module_states: dict[str, ModuleConfigState] = {}
        self._remaining_modules_by_key: dict[ServicesPoolKey, set[str]] = {}
        self._running_by_key: dict[ServicesPoolKey, RunningServices] = {}
        self._active_service_key_by_module: dict[str, ServicesPoolKey] = {}
        self._generated_config_dir: Path | None = None
        self._log_dir: Path | None = None

    @staticmethod
    def _log_debug(message: str, **extra: Any) -> None:
        if _E2E_HARNESS_DEBUG:
            logger.info(message, extra=extra)

    def bind_tmp_path_factory(self, tmp_path_factory: pytest.TempPathFactory) -> None:
        if self._tmp_path_factory is None:
            self._tmp_path_factory = tmp_path_factory

    def register_collected_items(self, items: list[pytest.Item]) -> None:
        seen_modules: set[str] = set()
        for item in items:
            module = item.getparent(pytest.Module)
            if module is None or module.nodeid in seen_modules:
                continue
            seen_modules.add(module.nodeid)
            self._ensure_module_registered(module)

    def acquire_for_module(self, module: pytest.Module) -> RunningServices:
        self._ensure_module_registered(module)
        state = self._module_states[module.nodeid]
        return self._acquire_for_registered_owner(module.nodeid, state)

    def acquire_for_config(
        self,
        owner_id: str,
        config_layers: Sequence[str | dict[str, Any]],
        harness_config: Mapping[str, Any],
    ) -> RunningServices:
        self._ensure_config_registered(
            owner_id,
            config_layers=config_layers,
            harness_config=_normalize_e2e_harness_config(harness_config),
        )
        state = self._module_states[owner_id]
        return self._acquire_for_registered_owner(owner_id, state)

    def _acquire_for_registered_owner(self, owner_id: str, state: ModuleConfigState) -> RunningServices:
        external_url = os.environ.get("NHX_BASE_URL")
        if external_url:
            return RunningServices(
                url=external_url,
                log_path=None,
                proc=None,
                config_path=None,
                auth_enabled=state.auth_enabled,
            )
        if state.config_path is None:
            state = self._materialize_config_path(state)
            self._module_states[owner_id] = state
        assert state.config_path is not None
        services = self._running_by_key.get(state.key)
        if services is None:
            log_path = self._get_log_dir() / f"services-{state.key.config_hash}-{uuid.uuid4().hex[:8]}.log"
            services = _start_services(
                state.config_path,
                state.config_data,
                state.harness_config,
                state.key.config_hash,
                log_path,
            )
            self._running_by_key[state.key] = services
        previous_key = self._active_service_key_by_module.get(owner_id)
        if previous_key is not None and previous_key != state.key:
            logger.error(
                "E2E module rebound to a different services pool key",
                extra={
                    "e2e_module": owner_id,
                    "previous_config_hash": previous_key.config_hash,
                    "new_config_hash": state.key.config_hash,
                    "new_url": services.url,
                    "new_pid": services.proc.pid if services.proc is not None else None,
                },
            )
        self._remaining_modules_by_key.setdefault(state.key, set()).add(owner_id)
        self._active_service_key_by_module[owner_id] = state.key
        self._log_debug("E2E services acquire", **self.describe_module_binding(owner_id, services))
        return services

    def release_for_module(self, module: pytest.Module) -> None:
        self._release_for_registered_owner(module.nodeid)

    def release_for_config(self, owner_id: str) -> None:
        self._release_for_registered_owner(owner_id)

    def _release_for_registered_owner(self, owner_id: str) -> None:
        self._active_service_key_by_module.pop(owner_id, None)
        if os.environ.get("NHX_BASE_URL"):
            return
        state = self._module_states.get(owner_id)
        if state is None:
            return
        remaining = self._remaining_modules_by_key.get(state.key)
        if remaining is None or owner_id not in remaining:
            return
        remaining.remove(owner_id)
        self._log_debug(
            "E2E services release",
            **{
                **self.describe_module_binding(owner_id),
                "remaining_modules_for_hash": sorted(remaining),
            },
        )
        if remaining:
            return
        self._remaining_modules_by_key.pop(state.key, None)
        services = self._running_by_key.pop(state.key, None)
        if services is not None:
            self._terminate_services(services)

    def shutdown_all(self) -> None:
        for services in list(self._running_by_key.values()):
            self._terminate_services(services)
        self._running_by_key.clear()
        self._remaining_modules_by_key.clear()

    def _ensure_module_registered(self, module: pytest.Module) -> None:
        if module.nodeid in self._module_states:
            return
        resolved_paths, config_data = _load_effective_e2e_config_from_node(module)
        harness_config = _resolve_e2e_harness_config_from_node(module)
        self._register_config_state(
            module.nodeid,
            resolved_paths=resolved_paths,
            config_data=config_data,
            harness_config=harness_config,
        )

    def _ensure_config_registered(
        self,
        owner_id: str,
        *,
        config_layers: Sequence[str | dict[str, Any]],
        harness_config: E2EHarnessConfig,
    ) -> None:
        if owner_id in self._module_states:
            return
        resolved_paths, config_data = _load_effective_e2e_config_from_layers(config_layers)
        self._register_config_state(
            owner_id,
            resolved_paths=resolved_paths,
            config_data=config_data,
            harness_config=harness_config,
        )

    def _register_config_state(
        self,
        owner_id: str,
        *,
        resolved_paths: Sequence[Path],
        config_data: dict[str, Any],
        harness_config: E2EHarnessConfig,
    ) -> None:
        key = _services_pool_key(_canonical_services_hash(config_data, harness_config))
        auth_enabled = _e2e_auth_enabled(config_data)
        self._module_states[owner_id] = ModuleConfigState(
            module_id=owner_id,
            key=key,
            config_path=None,
            config_data=config_data,
            harness_config=harness_config,
            config_layers=tuple(str(path) for path in resolved_paths),
            auth_enabled=auth_enabled,
        )
        self._remaining_modules_by_key.setdefault(key, set()).add(owner_id)
        self._log_debug(
            "Registered E2E module config",
            e2e_module=owner_id,
            config_hash=key.config_hash,
            harness_backend=harness_config["backend"],
            config_layers=list(self._module_states[owner_id].config_layers),
            auth_enabled=auth_enabled,
        )

    def _materialize_config_path(self, state: ModuleConfigState) -> ModuleConfigState:
        data_dir = e2e_services_data_dir(self._get_log_dir(), state.key.config_hash)
        rendered_config_data = _render_e2e_config_for_backend(
            state.config_data,
            data_dir,
            state.harness_config,
            state.key.config_hash,
        )
        rendered_config = yaml.safe_dump(rendered_config_data, default_flow_style=False, sort_keys=True)
        config_path = self._get_generated_config_dir(state.harness_config) / f"platform-{state.key.config_hash}.yaml"
        if not config_path.exists():
            config_path.write_text(rendered_config)
            self._log_debug(
                "Materialized generated E2E config",
                e2e_module=state.module_id,
                config_hash=state.key.config_hash,
                config_path=str(config_path),
            )
        return ModuleConfigState(
            module_id=state.module_id,
            key=state.key,
            config_path=config_path,
            config_data=state.config_data,
            harness_config=state.harness_config,
            config_layers=state.config_layers,
            auth_enabled=state.auth_enabled,
        )

    def _get_generated_config_dir(self, harness_config: E2EHarnessConfig) -> Path:
        if _e2e_backend(harness_config) == "docker_compose":
            return _docker_compose_generated_config_dir()
        if self._generated_config_dir is None:
            self._generated_config_dir = self._get_log_dir() / "generated-configs"
            self._generated_config_dir.mkdir(parents=True, exist_ok=True)
        return self._generated_config_dir

    def _get_log_dir(self) -> Path:
        if self._tmp_path_factory is None:
            raise RuntimeError("E2E services pool used before tmp_path_factory was bound")
        if self._log_dir is None:
            self._log_dir = _services_log_dir(self._tmp_path_factory)
        return self._log_dir

    @staticmethod
    def _terminate_services(services: RunningServices) -> None:
        if services.proc is None:
            if services.close is not None:
                services.close()
            return
        if services.proc.poll() is not None:
            E2EServicesPool._log_debug(
                "Skipping E2E services terminate for already-exited process",
                config_hash=services.key.config_hash if services.key is not None else None,
                pid=services.proc.pid,
                returncode=services.proc.returncode,
                url=services.url,
            )
            return
        logger.info("Terminating nemo services (pid %d)", services.proc.pid)
        services.proc.terminate()
        try:
            services.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("Process %d did not exit after SIGTERM, sending SIGKILL", services.proc.pid)
            services.proc.kill()
            services.proc.wait(timeout=5)

    def describe_module_binding(
        self,
        module_id: str,
        services: RunningServices | None = None,
    ) -> dict[str, Any]:
        state = self._module_states[module_id]
        details: dict[str, Any] = {
            "e2e_module": module_id,
            "config_hash": state.key.config_hash,
            "auth_enabled": state.auth_enabled,
            "config_layers": list(state.config_layers),
            "config_path": str(state.config_path) if state.config_path is not None else None,
            "harness_backend": state.harness_config["backend"],
        }
        if services is not None:
            details.update(
                {
                    "service_url": services.url,
                    "service_pid": services.proc.pid if services.proc is not None else None,
                    "service_log_path": str(services.log_path) if services.log_path is not None else None,
                    "docker_network_name": services.docker_network_name,
                    "docker_container_alias": services.docker_container_alias,
                    "docker_container_port": services.docker_container_port,
                    "compose_project_name": services.compose_project_name,
                }
            )
        return details

    def describe_active_module_binding(self, module_id: str) -> dict[str, Any] | None:
        key = self._active_service_key_by_module.get(module_id)
        if key is None:
            return None
        active_modules = self._remaining_modules_by_key.get(key)
        if active_modules is None or module_id not in active_modules:
            return None
        services = self._running_by_key.get(key)
        if services is None:
            return None
        return self.describe_module_binding(module_id, services)


def _services_log_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    log_dir = os.environ.get("E2E_SERVICES_LOG_DIR")
    if log_dir:
        directory = Path(log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        return directory
    return tmp_path_factory.mktemp("e2e-services-logs")


def _docker_compose_generated_config_dir() -> Path:
    directory = _E2E_REPO_ROOT / ".pytest_cache" / "e2e-services" / f"generated-configs-{os.getpid()}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _resolve_e2e_config_layers_from_node(node: Node) -> list[str | dict[str, Any]]:
    marker = node.get_closest_marker("e2e_config")
    if marker is None or not marker.args:
        return [
            str(_DEFAULT_E2E_PLATFORM_CONFIG),
            str(_DEFAULT_E2E_DISABLE_DEPLOYMENTS_ORPHAN_CLEANUP),
        ]
    layers: list[str | dict[str, Any]] = []
    for layer in marker.args:
        if isinstance(layer, (str, dict)):
            layers.append(layer)
            continue
        raise pytest.UsageError("pytest.mark.e2e_config arguments must be strings or dicts")
    return layers


def _resolve_e2e_harness_config_from_node(node: Node) -> E2EHarnessConfig:
    marker = node.get_closest_marker("e2e_config")
    if marker is None:
        return {"backend": "subprocess"}
    unknown = set(marker.kwargs) - {"harness"}
    if unknown:
        raise pytest.UsageError(f"pytest.mark.e2e_config only supports the 'harness' keyword, got: {sorted(unknown)}")
    harness = marker.kwargs.get("harness")
    if harness is None:
        return {"backend": "subprocess"}
    return _normalize_e2e_harness_config(harness)


def _normalize_e2e_harness_config(harness: Mapping[str, Any]) -> E2EHarnessConfig:
    if not isinstance(harness, Mapping):
        raise pytest.UsageError("pytest.mark.e2e_config harness must be a mapping")
    normalized = _normalize_config(dict(harness))
    backend = normalized.get("backend", "subprocess")
    if backend not in {"subprocess", "docker", "docker_compose"}:
        raise pytest.UsageError(f"unsupported e2e harness backend: {backend}")
    normalized["backend"] = backend
    if "container_base_url_host" in normalized and backend != "subprocess":
        raise pytest.UsageError("container_base_url_host is only supported with the 'subprocess' harness backend")
    if backend == "docker_compose":
        required = {"compose_file", "service_url"}
        missing = sorted(required - set(normalized))
        if missing:
            raise pytest.UsageError(f"docker_compose harness config missing required keys: {missing}")
        lifecycle = normalized.get("lifecycle", os.environ.get(_E2E_COMPOSE_LIFECYCLE_ENV, "fresh"))
        if lifecycle not in {"fresh", "reuse"}:
            raise pytest.UsageError(
                f"unsupported docker_compose lifecycle from {_E2E_COMPOSE_LIFECYCLE_ENV}: {lifecycle}"
            )
        normalized["lifecycle"] = lifecycle
    return normalized


def _resolve_config_path(config_ref: str) -> Path:
    candidate = Path(config_ref)
    if not candidate.is_absolute():
        candidate = _E2E_REPO_ROOT / config_ref
    return candidate.resolve()


def _normalize_config(value: Any, path: tuple[str, ...] = ()) -> Any:
    if isinstance(value, dict):
        return {key: _normalize_config(value[key], (*path, key)) for key in sorted(value)}
    if isinstance(value, list):
        normalized = [_normalize_config(item, path) for item in value]
        if path == ("jobs", "executors"):
            return sorted(
                normalized,
                key=lambda item: (
                    item.get("provider", "") if isinstance(item, dict) else "",
                    item.get("profile", "") if isinstance(item, dict) else "",
                    item.get("backend", "") if isinstance(item, dict) else "",
                    json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=True),
                ),
            )
        return normalized
    return value


def _render_template_value(value: Any, context: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        return Template(value).safe_substitute(context)
    if isinstance(value, Mapping):
        return {key: _render_template_value(item, context) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [_render_template_value(item, context) for item in value]
    return value


def _dynamic_port_template_context(harness_config: E2EHarnessConfig) -> dict[str, str]:
    context: dict[str, str] = {}
    for name, port_config in harness_config.get("dynamic_ports", {}).items():
        configured_port = port_config.get("port")
        port = _find_free_port() if configured_port is None else int(configured_port)
        host = port_config.get("host", "127.0.0.1")
        scheme = port_config.get("scheme", "http")
        context[f"{name}_host"] = host
        context[f"{name}_port"] = str(port)
        context[f"{name}_scheme"] = scheme
        context[f"{name}_url"] = f"{scheme}://{host}:{port}"
    return context


def _canonical_config_hash(config_data: dict[str, Any]) -> str:
    normalized = _normalize_config(config_data)
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _canonical_services_hash(config_data: dict[str, Any], harness_config: E2EHarnessConfig) -> str:
    payload = json.dumps(
        {
            "platform": _normalize_config(config_data),
            "harness": _normalize_config(harness_config),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _load_effective_e2e_config_from_node(node: Node) -> tuple[list[Path], dict[str, Any]]:
    return _load_effective_e2e_config_from_layers(_resolve_e2e_config_layers_from_node(node))


def _load_effective_e2e_config_from_layers(
    layers: Sequence[str | dict[str, Any]],
) -> tuple[list[Path], dict[str, Any]]:
    effective_config: dict[str, Any] = {}
    resolved_paths: list[Path] = []

    for layer in layers:
        if isinstance(layer, str):
            config_path = _resolve_config_path(layer)
            if not config_path.is_file():
                raise pytest.UsageError(f"E2E platform config not found: {config_path}")
            layer_config = yaml.safe_load(config_path.read_text()) or {}
            resolved_paths.append(config_path)
        else:
            layer_config = layer
        effective_config = deep_merge(effective_config, layer_config)

    return resolved_paths, _normalize_config(effective_config)


def e2e_services_data_dir(log_dir: Path, config_hash: str) -> Path:
    """Return persistent state outside the directory collected as test logs."""
    return log_dir.parent / f"{log_dir.name}-data" / f"data-{config_hash}"


def with_e2e_instance_paths(
    config_data: dict[str, Any],
    data_dir: Path,
    *,
    resource_scope: str | None = None,
) -> dict[str, Any]:
    """Return config data with per-instance filesystem paths rooted under ``data_dir``."""
    rendered = deepcopy(config_data)
    subprocess_working_dir = str(data_dir / "subprocess-jobs")
    files_root = str(data_dir / "files")

    jobs = rendered.get("jobs")
    if isinstance(jobs, dict):
        executors = jobs.get("executors")
        if isinstance(executors, list):
            for executor in executors:
                if not isinstance(executor, dict):
                    continue
                if executor.get("provider") != "subprocess":
                    continue
                config = executor.setdefault("config", {})
                if isinstance(config, dict):
                    config["working_directory"] = subprocess_working_dir

        executor_defaults = jobs.get("executor_defaults")
        if isinstance(executor_defaults, dict):
            subprocess_defaults = executor_defaults.get("subprocess")
            if isinstance(subprocess_defaults, dict):
                subprocess_defaults["working_directory"] = subprocess_working_dir

    files = rendered.get("files")
    if isinstance(files, dict):
        default_storage_config = files.get("default_storage_config")
        if isinstance(default_storage_config, dict) and default_storage_config.get("type") == "local":
            default_storage_config["path"] = files_root

    if resource_scope:
        _stamp_e2e_docker_deployment_scope(rendered, resource_scope)

    return rendered


def _stamp_e2e_docker_deployment_scope(config_data: dict[str, Any], resource_scope: str) -> None:
    deployments = config_data.get("deployments")
    if not isinstance(deployments, dict):
        return
    executors = deployments.get("executors")
    if not isinstance(executors, list):
        return
    for executor in executors:
        if not isinstance(executor, dict) or executor.get("backend") != "docker":
            continue
        executor_config = executor.setdefault("config", {})
        if isinstance(executor_config, dict):
            executor_config.setdefault("resource_scope", resource_scope)


# The authz e2e suite (``e2e/authz_oidc``) editable-installs intentionally-broken
# fixture plugins (named ``harness-*``) into the shared venv to exercise the authz
# fail-modes. Their ``nemo.services`` entry points persist for the rest of the pytest
# session, so a pool platform spawned afterward would otherwise discover them too — and
# with the ``on_invalid_plugin=hard_fail`` default an unruled fixture aborts the whole OPA
# bundle ("Policy data not loaded — refusing to evaluate"), 502-ing every request and
# wedging unrelated auth job integration tests. Pool platforms therefore pin the
# service-plugin allowlist to the real installed plugins, fencing the fixtures out.
# (authz_oidc spawns its own platforms with their own env, opting into the fixtures with
# deny_route/quarantine, so it is unaffected by this.)
_HARNESS_FIXTURE_PREFIX = "harness-"


def _real_service_plugin_allowlist() -> str | None:
    """Comma-joined ``nemo.services`` plugin names minus the ``harness-*`` test fixtures.

    Returns ``None`` (⇒ leave discovery unrestricted) when no real plugins are visible, so
    a stripped-down environment never accidentally disables all plugin discovery.
    """
    names = sorted(
        {ep.name for ep in entry_points(group="nemo.services") if not ep.name.startswith(_HARNESS_FIXTURE_PREFIX)}
    )
    return ",".join(names) if names else None


def _e2e_backend(harness_config: E2EHarnessConfig) -> Literal["subprocess", "docker", "docker_compose"]:
    return harness_config.get("backend", "subprocess")


def _render_e2e_config_for_backend(
    config_data: dict[str, Any], data_dir: Path, harness_config: E2EHarnessConfig, config_hash: str
) -> dict[str, Any]:
    if _e2e_backend(harness_config) in {"docker", "docker_compose"}:
        return deepcopy(config_data)
    return with_e2e_instance_paths(config_data, data_dir, resource_scope=f"e2e-{config_hash}")


class DockerBackendOverrides(TypedDict, total=False):
    registry: str
    tag: str
    gpu_requested: NotRequired[bool]


def _docker_backend_overrides() -> DockerBackendOverrides:
    registry = os.environ.get("NHX_E2E_IMAGE_REGISTRY") or os.environ.get("IMAGE_REGISTRY")
    tag = os.environ.get("NHX_E2E_IMAGE_TAG") or os.environ.get("BAKE_TAG")
    overrides: DockerBackendOverrides = {}
    if registry:
        overrides["registry"] = registry
    if tag:
        overrides["tag"] = tag
    return overrides


def e2e_services_env(config_path: Path, data_dir: Path) -> dict[str, str]:
    """Environment for the ``nemo services run`` child process."""
    env = os.environ.copy()
    env["NHX_SEED_ON_STARTUP"] = "true"
    env["NHX_INFERENCE_GATEWAY_MOCK_PROVIDER_PREFIX"] = "igw-mock-"
    env["NHX_CONFIG_FILE_PATH"] = str(config_path)
    env["NHX_CONFIG_WARNINGS_DISABLED"] = "1"
    env["NHX_DATA_DIR"] = str(data_dir)
    allowlist = _real_service_plugin_allowlist()
    if allowlist is not None:
        env.setdefault("NEMO_PLUGIN_SERVICES_ALLOWLIST", allowlist)
    return env


def _e2e_auth_enabled(config_data: dict[str, Any]) -> bool:
    auth_cfg = config_data.get("auth")
    return isinstance(auth_cfg, dict) and bool(auth_cfg.get("enabled", False))


def _services_pool_key(config_hash: str) -> ServicesPoolKey:
    return ServicesPoolKey(config_hash=config_hash)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _set_platform_base_url(config_path: Path, base_url: str) -> None:
    """Rewrite ``platform.base_url`` in an already-materialized config file."""
    config_data = yaml.safe_load(config_path.read_text()) or {}
    platform = config_data.setdefault("platform", {})
    platform["base_url"] = base_url
    config_path.write_text(yaml.safe_dump(config_data, default_flow_style=False, sort_keys=True))


def _process_exited(proc: subprocess.Popen[Any]) -> bool:
    return proc.poll() is not None


def _wait_for_healthy(url: str, proc: subprocess.Popen[Any], timeout: float = _HEALTH_TIMEOUT) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _process_exited(proc):
            return False
        try:
            resp = httpx.get(f"{url}/status", timeout=2.0)
            if resp.status_code == 200:
                return True
        except httpx.RequestError:
            pass
        if _process_exited(proc):
            return False
        time.sleep(_HEALTH_POLL_INTERVAL)
    return False


def _wait_for_auth_ready(url: str, proc: subprocess.Popen[Any] | None, timeout: float = _AUTH_READY_TIMEOUT) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc is not None and _process_exited(proc):
            return False
        probe_name = f"auth-probe-{uuid.uuid4().hex[:8]}"
        entity_name = f"auth-probe-entity-{uuid.uuid4().hex[:8]}"
        try:
            create_resp = httpx.post(
                f"{url}/apis/entities/v2/workspaces",
                json={"name": probe_name},
                headers=admin_headers(),
                timeout=5.0,
            )
            if create_resp.status_code != 201:
                if proc is not None and _process_exited(proc):
                    return False
                time.sleep(_HEALTH_POLL_INTERVAL)
                continue

            entity_resp = httpx.post(
                f"{url}/apis/entities/v2/workspaces/{probe_name}/entities/e2e-auth-probe",
                json={"name": entity_name, "data": {"ready": True}},
                headers=admin_headers(),
                timeout=5.0,
            )
            if entity_resp.status_code != 201:
                httpx.delete(
                    f"{url}/apis/entities/v2/workspaces/{probe_name}",
                    headers=admin_headers(),
                    timeout=5.0,
                )
                time.sleep(_HEALTH_POLL_INTERVAL)
                continue

            httpx.delete(
                f"{url}/apis/entities/v2/workspaces/{probe_name}/entities/e2e-auth-probe/{entity_name}",
                headers=admin_headers(),
                timeout=5.0,
            )
            httpx.delete(
                f"{url}/apis/entities/v2/workspaces/{probe_name}",
                headers=admin_headers(),
                timeout=5.0,
            )
            return True
        except httpx.RequestError as exc:
            logger.debug("Auth readiness probe failed; will retry: %s", exc)
        if proc is not None and _process_exited(proc):
            return False
        time.sleep(_HEALTH_POLL_INTERVAL)
    return False


def _request_verify_from_env(env: Mapping[str, str] | None = None) -> str | bool:
    source = dict(os.environ)
    if env:
        source.update(env)
    return (
        source.get("NHX_CLIENT_SSL_CERT_FILE")
        or source.get("REQUESTS_CA_BUNDLE")
        or source.get("SSL_CERT_FILE")
        or True
    )


def _wait_for_auth_ready_url(
    url: str,
    proc: subprocess.Popen[Any] | None,
    *,
    env: Mapping[str, str] | None = None,
    timeout: float = _AUTH_READY_TIMEOUT,
) -> bool:
    deadline = time.monotonic() + timeout
    verify = _request_verify_from_env(env)
    while time.monotonic() < deadline:
        if proc is not None and _process_exited(proc):
            return False
        try:
            response = httpx.get(url, timeout=5.0, verify=verify)
            if response.status_code == 200:
                return True
        except httpx.RequestError as exc:
            logger.debug("Auth readiness URL probe failed; will retry: %s", exc)
        if proc is not None and _process_exited(proc):
            return False
        time.sleep(_HEALTH_POLL_INTERVAL)
    return False


def _auth_ready_url(harness_config: E2EHarnessConfig, service_url: str) -> str | None:
    url = harness_config.get("auth_ready_url")
    if not url:
        return None
    return Template(url).safe_substitute({"service_url": service_url})


def _wait_for_configured_auth_ready(
    service_url: str,
    proc: subprocess.Popen[Any] | None,
    harness_config: E2EHarnessConfig,
) -> bool:
    ready_url = _auth_ready_url(harness_config, service_url)
    if ready_url is not None:
        return _wait_for_auth_ready_url(ready_url, proc, env=harness_config.get("env"))
    return _wait_for_auth_ready(service_url, proc)


def _start_services(
    config_path: Path,
    config_data: dict[str, Any],
    harness_config: E2EHarnessConfig,
    config_hash: str,
    log_path: Path,
) -> RunningServices:
    backend = _e2e_backend(harness_config)
    if backend == "docker":
        return _start_services_docker(config_path, config_data, harness_config, config_hash)
    if backend == "docker_compose":
        return _start_services_docker_compose(config_path, config_data, harness_config, config_hash, log_path)
    return _start_services_subprocess(config_path, config_data, harness_config, config_hash, log_path)


def _start_services_subprocess(
    config_path: Path,
    config_data: dict[str, Any],
    harness_config: E2EHarnessConfig,
    config_hash: str,
    log_path: Path,
) -> RunningServices:
    port = _find_free_port()
    url = f"http://127.0.0.1:{port}"

    nemo_bin = str(Path(sys.executable).parent / "nemo")
    args = [
        nemo_bin,
        "services",
        "run",
        "--service-group",
        "all",
        "--controller-group",
        "all",
        "--port",
        str(port),
    ]

    # For container-mode agent deployments the platform must advertise a base URL
    # reachable from inside the deployed agent container, not the loopback the
    # platform binds by default. Two coordinated changes make that work:
    #   1. Bind all interfaces (--host 0.0.0.0) instead of the CLI default
    #      127.0.0.1, so the platform is reachable on the container-facing host
    #      (e.g. the docker bridge 172.17.0.1) as well as loopback.
    #   2. Rewrite platform.base_url on disk to http://<host>:<this port>. The
    #      free port is only known here, after the config file was materialized.
    #      The platform seeds NHX_BASE_URL from this, so both the platform's own
    #      in-process clients and the injected agent LLM base_url point at a
    #      host the agent container can reach.
    container_host = harness_config.get("container_base_url_host")
    if container_host:
        args += ["--host", "0.0.0.0"]
        _set_platform_base_url(config_path, f"http://{container_host}:{port}")
    data_dir = e2e_services_data_dir(log_path.parent, config_hash)
    data_dir.mkdir(parents=True, exist_ok=True)
    env = e2e_services_env(config_path, data_dir)

    logger.info("Starting nemo services on port %d with config %s", port, config_path)

    log_file = open(log_path, "w")
    try:
        proc = subprocess.Popen(args, stdout=log_file, stderr=subprocess.STDOUT, env=env)
    finally:
        log_file.close()

    if not _wait_for_healthy(url, proc):
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
            proc.wait(timeout=5)
        pytest.fail(
            f"nemo services run did not become healthy within {_HEALTH_TIMEOUT}s.\nlog:\n{log_path.read_text()}"
        )
    auth_enabled = _e2e_auth_enabled(config_data)
    if auth_enabled and not _wait_for_configured_auth_ready(url, proc, harness_config):
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
            proc.wait(timeout=5)
        pytest.fail(
            f"Platform auth seed did not become ready within {_AUTH_READY_TIMEOUT}s.\nlog:\n{log_path.read_text()}"
        )

    logger.info("Platform services ready on port %d (pid %d)", port, proc.pid)
    return RunningServices(
        url=url,
        log_path=log_path,
        proc=proc,
        config_path=config_path,
        auth_enabled=auth_enabled,
        key=_services_pool_key(config_hash),
    )


def _start_services_docker(
    config_path: Path,
    config_data: dict[str, Any],
    harness_config: E2EHarnessConfig,
    config_hash: str,
) -> RunningServices:
    backend = DockerE2EBackend(config_path=config_path, **_docker_backend_overrides())
    try:
        backend.start()
    except Exception:
        backend.stop()
        raise

    auth_enabled = _e2e_auth_enabled(config_data)
    if auth_enabled and not _wait_for_configured_auth_ready(backend.base_url, None, harness_config):
        backend.stop()
        pytest.fail(f"Platform auth seed did not become ready within {_AUTH_READY_TIMEOUT}s.")

    services = RunningServices(
        url=backend.base_url,
        log_path=None,
        proc=None,
        config_path=config_path,
        close=backend.stop,
        auth_enabled=auth_enabled,
        key=_services_pool_key(config_hash),
        docker_network_name=backend.network_name,
        docker_container_alias=backend.network_alias,
        docker_container_port=backend.container_port,
    )

    return services


def _start_services_docker_compose(
    config_path: Path,
    config_data: dict[str, Any],
    harness_config: E2EHarnessConfig,
    config_hash: str,
    log_path: Path,
) -> RunningServices:
    compose_file = _resolve_config_path(harness_config["compose_file"])
    project_name = harness_config.get("compose_project_name")
    if project_name is None:
        project_prefix = harness_config.get("compose_project_prefix", "e2e-compose")
        project_name = f"{project_prefix}-{config_hash}"
        if harness_config["lifecycle"] == "fresh":
            project_name = f"{project_name}-{uuid.uuid4().hex[:8]}"
    if harness_config["lifecycle"] != "fresh":
        missing_fixed_ports = sorted(
            name for name, port_config in harness_config.get("dynamic_ports", {}).items() if "port" not in port_config
        )
        if missing_fixed_ports:
            raise pytest.UsageError(
                f"docker_compose dynamic_ports require explicit ports with lifecycle='reuse': {missing_fixed_ports}"
            )
    template_context = _dynamic_port_template_context(harness_config)
    if template_context:
        config_data = cast(dict[str, Any], _render_template_value(config_data, template_context))
        rendered_harness_config = cast(E2EHarnessConfig, _render_template_value(dict(harness_config), template_context))
        rendered_harness_config.pop("dynamic_ports", None)
        harness_config = rendered_harness_config
        config_path.write_text(yaml.safe_dump(config_data, default_flow_style=False, sort_keys=True))
    auth_enabled = _e2e_auth_enabled(config_data)
    service_url = harness_config["service_url"]
    auth_ready_url = _auth_ready_url(harness_config, service_url) if auth_enabled else None
    wait_url = auth_ready_url or harness_config.get("wait_url")
    wait_urls = [auth_ready_url] if auth_ready_url is not None else harness_config.get("wait_urls")
    backend = DockerComposeE2EBackend(
        compose_file=compose_file,
        config_path=config_path,
        project_name=project_name,
        service_url=service_url,
        wait_url=wait_url,
        wait_urls=wait_urls,
        env=harness_config.get("env"),
        lifecycle=harness_config["lifecycle"],
    )
    try:
        backend.start()
    except Exception:
        _write_docker_compose_logs(backend, log_path)
        backend.stop()
        raise

    def close() -> None:
        try:
            _write_docker_compose_logs(backend, log_path)
        finally:
            backend.stop()

    return RunningServices(
        url=backend.service_url,
        log_path=log_path,
        proc=None,
        config_path=config_path,
        close=close,
        auth_enabled=auth_enabled,
        key=_services_pool_key(config_hash),
        compose_project_name=project_name,
    )


def _write_docker_compose_logs(backend: DockerComposeE2EBackend, log_path: Path) -> None:
    try:
        backend.write_logs(log_path)
    except Exception:
        logger.exception("Could not write docker compose services log", extra={"log_path": str(log_path)})


def _read_log_text(log_path: Path) -> str:
    try:
        return log_path.read_text()
    except OSError as exc:
        return f"<could not read {log_path}: {exc}>"
