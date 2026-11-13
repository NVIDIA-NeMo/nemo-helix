# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Service and controller registries for the platform runner."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator, Mapping
from functools import cache
from importlib.metadata import EntryPoint

from nemo_helix_plugin.controller import NemoController
from nemo_helix_plugin.discovery import discover_entry_points
from nemo_helix_plugin.service import NemoService
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
    "model-metadata": "nhx.core.models.controllers.metadata_main:run",
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
def get_available_controllers() -> PluginRegistry[ControllerRunFunc]:
    """Return all available controllers for the current run, loading plugin controllers on lookup.

    Built-in core controllers are ``"module:object"`` import strings. Plugin controllers come from
    the ``nemo.controllers`` entry-point group and become ``run(stop_signal)`` callables via
    :func:`~nhx.platform_runner.plugin_adapter.make_controller_run_func` when looked up.
    """
    return PluginRegistry(AVAILABLE_CONTROLLERS, discover_entry_points("nemo.controllers"), _load_plugin_controller)


def get_controller_groups(
    available_controllers: Mapping[str, str | ControllerRunFunc] | None = None,
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


class PluginRegistry[V](Mapping[str, str | V]):
    """Built-in ``"module:object"`` import strings plus installed plugins, keyed by name.

    A plugin is imported only when looked up: importing every installed plugin costs seconds per
    process, and most runs start only a few of them.
    """

    def __init__(
        self,
        core: Mapping[str, str],
        plugins: Mapping[str, EntryPoint],
        load: Callable[[str, EntryPoint], V | None],
    ) -> None:
        self._core = dict(core)
        self._plugins = dict(plugins)
        self._load = load
        self._loaded: dict[str, V] = {}
        self._failed: set[str] = set()

    def __getitem__(self, name: str) -> str | V:
        if name in self._loaded:
            return self._loaded[name]
        if name in self._plugins and name not in self._failed:
            value = self._load(name, self._plugins[name])
            if value is not None:
                self._loaded[name] = value
                return value
            self._failed.add(name)
        return self._core[name]

    def __contains__(self, name: object) -> bool:
        return name in self._core or (name in self._plugins and name not in self._failed)

    def __iter__(self) -> Iterator[str]:
        names = dict.fromkeys([*self._core, *self._plugins])
        return iter([name for name in names if name in self])

    def __len__(self) -> int:
        return sum(1 for _ in self)


def _load_plugin_class[T](group: str, name: str, entry_point: EntryPoint, base: type[T]) -> type[T] | None:
    try:
        cls = entry_point.load()
    except Exception:
        logger.warning("Failed to load %r from %r (%s) — skipping", name, group, entry_point.value)
        logger.debug("Traceback for failed load of %r from %r:", name, group, exc_info=True)
        return None
    if not (isinstance(cls, type) and issubclass(cls, base)):
        logger.warning(
            "%s entry %r (%s) is not a %s subclass — skipping", group, name, entry_point.value, base.__name__
        )
        return None
    cls_name = getattr(cls, "name", None)
    if cls_name != name:
        logger.warning(
            "%s entry %r: class %s declares name=%r — name must match the pyproject.toml entry-point key",
            group,
            name,
            getattr(cls, "__qualname__", cls),
            cls_name,
        )
    return cls


def _load_plugin_controller(name: str, entry_point: EntryPoint) -> ControllerRunFunc | None:
    controller_cls = _load_plugin_class("nemo.controllers", name, entry_point, NemoController)
    if controller_cls is None:
        return None
    try:
        run_func = make_controller_run_func(controller_cls)
    except Exception:
        logger.warning(
            "Failed to create run function for plugin controller %r from %s.%s",
            name,
            controller_cls.__module__,
            controller_cls.__qualname__,
            exc_info=True,
        )
        return None
    logger.debug("Registered plugin controller %r", name)
    return run_func


def _load_plugin_service(name: str, entry_point: EntryPoint) -> Service | None:
    service_cls = _load_plugin_class("nemo.services", name, entry_point, NemoService)
    if service_cls is None:
        return None
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
def get_available_services() -> PluginRegistry[Service]:
    """Return all available services for the current run, loading plugin services on lookup."""
    return PluginRegistry(AVAILABLE_SERVICES, discover_entry_points("nemo.services"), _load_plugin_service)


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
