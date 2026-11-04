# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Service and controller registries for the platform runner."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator, Mapping
from functools import cache
from importlib.metadata import EntryPoint

from nemo_helix_plugin.discovery import discover_controllers, discover_entry_points
from nhx.common.service import Service
from nhx.platform_runner.loader import ControllerRunFunc
from nhx.platform_runner.plugin_adapter import NemoServiceAdapter, make_controller_run_func

logger = logging.getLogger(__name__)

AVAILABLE_SERVICES: dict[str, str] = {
    "hello-world": "nhx.hello_world.main:service",
    "studio": "nhx.studio.main:service",
    "auth": "nhx.core.auth.main:service",
    "jobs": "nhx.core.jobs.main:service",
    "files": "nhx.core.files.main:service",
    "guardrails": "nhx.guardrails.main:service",
    "intake": "nhx.intake.main:service",
    "entities": "nhx.core.entities.main:service",
    "secrets": "nhx.core.secrets.main:service",
    "models": "nhx.core.models.main:service",
    "inference-gateway": "nhx.core.inference_gateway.main:service",
}

AVAILABLE_CONTROLLERS: dict[str, str] = {
    "jobs": "nhx.core.jobs.controllers.main:run",
    "models": "nhx.core.models.controllers.main:run",
    "entities": "nhx.core.entities.controllers.main:run",
}

AVAILABLE_SIDECARS: dict[str, str] = {
    "adapters": "nhx.core.models.sidecars.adapters.main:run",
    "auth-proxy": "nhx.common.auth.workload_proxy.main:run",
}


def check_no_controller_sidecar_collision(controller_names: Iterable[str], sidecar_names: Iterable[str]) -> None:
    """Reject names that make controller and sidecar health indistinguishable."""
    collisions = set(controller_names) & set(sidecar_names)
    if collisions:
        raise ValueError(f"Controller/sidecar name collision: {', '.join(sorted(collisions))}")


CORE_SERVICES = [
    "auth",
    "models",
    "files",
    "inference-gateway",
    "jobs",
    "secrets",
    "entities",
]

API_SERVICES = [
    "studio",
    "guardrails",
    "intake",
    # Safe Synthesizer is intentionally excluded from default runtime groups
    # while remaining available for OpenAPI generation.
    # "safe-synthesizer",
    "hello-world",
]

OPENAPI_SERVICES = [
    "auth",
    "customization",
    "entities",
    "files",
    "guardrails",
    "intake",
    "inference-gateway",
    "jobs",
    "models",
    "safe-synthesizer",
    "secrets",
]


@cache
def get_available_controllers() -> dict[str, str | ControllerRunFunc]:
    """Return all available controller run functions for the current run.

    Merges built-in core controllers (stored as ``"module:object"`` import
    strings) with plugin controllers discovered via the ``nemo.controllers``
    entry-point group (converted to ``run(stop_signal)`` callables via
    :func:`~nhx.platform_runner.plugin_adapter.make_controller_run_func`).

    Returns:
        Mapping of controller name → string import path or run callable.
    """
    controllers: dict[str, str | ControllerRunFunc] = dict(AVAILABLE_CONTROLLERS)

    for name, controller_cls in discover_controllers().items():
        try:
            controllers[name] = make_controller_run_func(controller_cls)
            logger.debug("Registered plugin controller %r", name)
        except Exception:
            logger.warning(
                "Failed to create run function for plugin controller %r from %s.%s",
                name,
                controller_cls.__module__,
                controller_cls.__qualname__,
                exc_info=True,
            )

    return controllers


def get_controller_groups(
    available_controllers: dict[str, str | ControllerRunFunc] | None = None,
) -> dict[str, list[str]]:
    """Return dynamic controller groups for the current run."""
    available_controllers = available_controllers or get_available_controllers()
    core = list(AVAILABLE_CONTROLLERS)
    plugin_controllers = sorted(name for name in available_controllers if name not in AVAILABLE_CONTROLLERS)
    return {
        "core": core,
        "all": [*core, *plugin_controllers],
    }


def get_controllers_for_group(
    controller_groups: dict[str, list[str]],
    group: str,
) -> list[str]:
    """Resolve the controllers for a named group."""
    if group not in controller_groups:
        valid_groups = ", ".join(sorted(controller_groups))
        raise ValueError(f"Unknown controller group: {group}. Available groups: {valid_groups}")
    return controller_groups[group]


def get_default_controllers(
    controller_groups: dict[str, list[str]] | None = None,
) -> list[str]:
    """Return the default controller set for a run."""
    controller_groups = controller_groups or get_controller_groups()
    return get_controllers_for_group(controller_groups, "all")


class AvailableServices(Mapping[str, "str | Service"]):
    """Every service this run can start, by name; plugin services load only when looked up.

    Names come from ``nemo.services`` entry-point metadata, so checking, listing or grouping them
    imports no plugin code. Importing every installed plugin costs seconds per process, and most
    runs start a few services. Looking a plugin up imports, instantiates and wraps it once. A
    plugin that fails to load is logged and then treated as absent (``KeyError``, and it drops
    out of iteration), matching the eager discovery this replaces. A plugin overrides a built-in
    service of the same name.
    """

    def __init__(self, core: Mapping[str, str], plugins: Mapping[str, EntryPoint]) -> None:
        self._core = dict(core)
        self._plugins = dict(plugins)
        self._loaded: dict[str, Service] = {}
        self._failed: set[str] = set()

    def __getitem__(self, name: str) -> str | Service:
        if name in self._loaded:
            return self._loaded[name]
        if name in self._plugins and name not in self._failed:
            service = _load_plugin_service(name, self._plugins[name])
            if service is not None:
                self._loaded[name] = service
                return service
            self._failed.add(name)
        return self._core[name]

    def __contains__(self, name: object) -> bool:
        return name in self._core or (name in self._plugins and name not in self._failed)

    def __iter__(self) -> Iterator[str]:
        names = dict.fromkeys([*self._core, *self._plugins])
        return iter([name for name in names if name in self])

    def __len__(self) -> int:
        return sum(1 for _ in self)


def _load_plugin_service(name: str, entry_point: EntryPoint) -> Service | None:
    try:
        service_cls = entry_point.load()
    except Exception:
        logger.warning("Failed to load %r from 'nemo.services' (%s) — skipping", name, entry_point.value)
        logger.debug("Traceback for failed load of %r from 'nemo.services':", name, exc_info=True)
        return None
    cls_name = getattr(service_cls, "name", None)
    if cls_name != name:
        logger.warning(
            "nemo.services entry %r: class %s declares name=%r — name must match the pyproject.toml entry-point key",
            name,
            getattr(service_cls, "__qualname__", service_cls),
            cls_name,
        )
    try:
        return NemoServiceAdapter(service_cls())
    except Exception:
        logger.warning(
            "Failed to instantiate plugin service %r from %s.%s",
            name,
            service_cls.__module__,
            service_cls.__qualname__,
            exc_info=True,
        )
        return None


@cache
def get_available_services() -> AvailableServices:
    """Return all available services for the current run, loading plugin services on lookup."""
    return AvailableServices(AVAILABLE_SERVICES, discover_entry_points("nemo.services"))


def get_service_groups(
    available_services: Mapping[str, str | Service] | None = None,
) -> dict[str, list[str]]:
    """Return dynamic service groups for the current run."""
    available_services = available_services or get_available_services()
    core = list(CORE_SERVICES)
    api = [name for name in API_SERVICES if name in available_services]
    plugin_services = sorted(name for name in available_services if name not in AVAILABLE_SERVICES)
    api.extend(plugin_services)
    return {
        "core": core,
        "api": api,
        "all": [*core, *api],
    }


def get_services_for_group(service_groups: dict[str, list[str]], group: str) -> list[str]:
    """Resolve the services for a named group."""
    if group not in service_groups:
        valid_groups = ", ".join(sorted(service_groups))
        raise ValueError(f"Unknown service group: {group}. Available groups: {valid_groups}")
    return service_groups[group]


def get_openapi_service_names(available_services: Mapping[str, str | Service] | None = None) -> list[str]:
    """Return the intentionally included services for platform OpenAPI generation."""
    available_services = available_services or get_available_services()
    return [service_name for service_name in OPENAPI_SERVICES if service_name in available_services]
