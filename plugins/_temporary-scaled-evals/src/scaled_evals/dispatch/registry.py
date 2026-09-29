# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime backend registry.

Runtime backends are loaded from explicit plugin modules (``module`` or
``module:function`` specs), so adding a backend requires a backend module, a
plugin registration hook, and tests; it should not require editing the
dispatcher lifecycle.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from functools import cache

from scaled_evals.api.settings import settings
from scaled_evals.dispatch.runtime_backend import (
    RuntimeBackend,
    RuntimeBackendCapabilities,
    RuntimeBackendRegistration,
)

_BUILTIN_RUNTIME_BACKEND_PLUGINS = (
    "scaled_evals.dispatch.sandbox_k8s",
    "scaled_evals.dispatch.harbor_opensandbox",
)


class RuntimeBackendRegistry:
    """Lookup table for runtime backends and their control-plane capabilities."""

    def __init__(self, registrations: Iterable[RuntimeBackendRegistration] = ()) -> None:
        self._registrations: dict[str, RuntimeBackendRegistration] = {}
        for registration in registrations:
            self.register(registration)

    def register(self, registration: RuntimeBackendRegistration) -> None:
        names = (registration.name, *registration.aliases)
        conflicts = [name for name in names if name in self._registrations]
        if conflicts:
            joined = ", ".join(sorted(conflicts))
            raise ValueError(f"runtime backend already registered: {joined}")
        for name in names:
            self._registrations[name] = registration

    def get(self, runtime: str) -> RuntimeBackendRegistration:
        try:
            return self._registrations[runtime]
        except KeyError as exc:
            known = ", ".join(self.names())
            raise ValueError(f"unknown runtime backend: {runtime!r}; known runtimes: {known}") from exc

    def build(self, runtime: str) -> RuntimeBackend:
        return self.get(runtime).build()

    def capabilities(self, runtime: str) -> RuntimeBackendCapabilities:
        return self.get(runtime).capabilities

    def validate_config(self, runtime: str) -> None:
        self.get(runtime).validate_config()

    def names(self) -> tuple[str, ...]:
        return tuple(sorted({registration.name for registration in self._registrations.values()}))


def _plugin_specs_from_settings() -> tuple[str, ...]:
    raw = settings.runtime_backend_plugins or ""
    return tuple(spec.strip() for spec in raw.split(",") if spec.strip())


def load_runtime_backend_plugin(registry: RuntimeBackendRegistry, plugin_spec: str) -> None:
    """Load one runtime plugin and let it register backends.

    ``plugin_spec`` is ``module`` or ``module:function``. A bare module must
    expose ``register_runtime_backends(registry)``.
    """
    module_name, sep, function_name = plugin_spec.partition(":")
    if not module_name:
        raise RuntimeError("runtime backend plugin spec is empty")
    register_name = function_name if sep else "register_runtime_backends"
    module = importlib.import_module(module_name)
    register = getattr(module, register_name, None)
    if not callable(register):
        raise RuntimeError(f"runtime backend plugin {plugin_spec!r} must expose callable {register_name!r}")
    register(registry)


def load_runtime_backend_plugins(
    registry: RuntimeBackendRegistry,
    plugin_specs: Iterable[str],
) -> None:
    for plugin_spec in plugin_specs:
        load_runtime_backend_plugin(registry, plugin_spec)


def _additional_plugin_specs(plugin_specs: Iterable[str]) -> tuple[str, ...]:
    builtin_modules = {spec.partition(":")[0] for spec in _BUILTIN_RUNTIME_BACKEND_PLUGINS}
    filtered: list[str] = []
    for plugin_spec in plugin_specs:
        module_name, sep, function_name = plugin_spec.partition(":")
        register_name = function_name if sep else "register_runtime_backends"
        if module_name in builtin_modules and register_name == "register_runtime_backends":
            continue
        filtered.append(plugin_spec)
    return tuple(filtered)


def build_runtime_backend_registry(
    plugin_specs: Iterable[str] | None = None,
) -> RuntimeBackendRegistry:
    registry = RuntimeBackendRegistry()
    load_runtime_backend_plugins(registry, _BUILTIN_RUNTIME_BACKEND_PLUGINS)
    extra_plugin_specs = _plugin_specs_from_settings() if plugin_specs is None else plugin_specs
    load_runtime_backend_plugins(
        registry,
        _additional_plugin_specs(extra_plugin_specs),
    )
    return registry


@cache
def default_runtime_backends() -> RuntimeBackendRegistry:
    """Build the process-wide registry once, on first use.

    Deferred because it reads settings: building at import time makes a missing
    env var an import error, and the platform's entry-point loader turns that
    into a silently skipped plugin.
    """
    return build_runtime_backend_registry()


def get_backend(runtime: str) -> RuntimeBackend:
    return default_runtime_backends().build(runtime)


def get_backend_capabilities(runtime: str) -> RuntimeBackendCapabilities:
    return default_runtime_backends().capabilities(runtime)


def get_backend_registration(runtime: str) -> RuntimeBackendRegistration:
    return default_runtime_backends().get(runtime)


def registered_runtime_names() -> tuple[str, ...]:
    return default_runtime_backends().names()


def validate_backend_config(runtime: str) -> None:
    default_runtime_backends().validate_config(runtime)
