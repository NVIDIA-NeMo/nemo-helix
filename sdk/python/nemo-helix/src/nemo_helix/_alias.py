# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: I001 - the generated SDK and workspace use different import-order settings.

from __future__ import annotations

import sys
from importlib import import_module, util
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from types import CodeType, ModuleType
from typing import Any


class _AliasLoader(Loader):
    def __init__(self, alias_name: str, target_name: str) -> None:
        self._alias_name = alias_name
        self._target_name = target_name

    def create_module(self, spec: ModuleSpec) -> ModuleType:
        del spec
        target = import_module(self._target_name)
        module = ModuleType(self._alias_name, target.__doc__)
        _populate_alias_namespace(module.__dict__, self._alias_name, self._target_name, target)
        return module

    def exec_module(self, module: ModuleType) -> None:
        del module
        return None

    def get_code(self, fullname: str) -> CodeType | None:
        """Return the target module's code so ``python -m`` and debugpy can run aliases."""
        del fullname
        target_spec = util.find_spec(self._target_name)
        if target_spec is None or target_spec.loader is None:
            return None
        get_code = getattr(target_spec.loader, "get_code", None)
        if get_code is None:
            return None
        return get_code(self._target_name)

    def get_resource_reader(self, fullname: str) -> Any:
        if fullname != self._alias_name:
            return None

        target = import_module(self._target_name)
        target_loader = getattr(target, "__loader__", None)
        get_resource_reader = getattr(target_loader, "get_resource_reader", None)
        if get_resource_reader is None:
            return None
        return get_resource_reader(self._target_name)


class _AliasFinder(MetaPathFinder):
    def __init__(self) -> None:
        self._aliases: dict[str, str] = {}

    def add_alias(self, alias_name: str, target_name: str) -> None:
        self._aliases[alias_name] = target_name

    def find_spec(
        self,
        fullname: str,
        _path: object | None = None,
        _target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        target_name = self._target_for(fullname)
        if target_name is None:
            return None

        target_spec = util.find_spec(target_name)
        if target_spec is None:
            return None

        is_package = target_spec.submodule_search_locations is not None
        spec = ModuleSpec(
            fullname,
            _AliasLoader(fullname, target_name),
            origin=target_spec.origin,
            is_package=is_package,
        )
        spec.cached = target_spec.cached
        spec.has_location = target_spec.has_location
        if is_package:
            spec.submodule_search_locations = target_spec.submodule_search_locations
        return spec

    def _target_for(self, fullname: str) -> str | None:
        for alias_name, target_name in sorted(self._aliases.items(), key=lambda item: len(item[0]), reverse=True):
            if fullname == alias_name:
                return target_name
            prefix = f"{alias_name}."
            if fullname.startswith(prefix):
                suffix = fullname[len(alias_name) :]
                return f"{target_name}{suffix}"
        return None


_FINDER: _AliasFinder | None = None

_MODULE_METADATA_NAMES = frozenset(
    {
        "__builtins__",
        "__cached__",
        "__dir__",
        "__doc__",
        "__file__",
        "__getattr__",
        "__loader__",
        "__name__",
        "__package__",
        "__path__",
        "__spec__",
    }
)


def _module_alias_name(value: ModuleType, alias_name: str, target_name: str) -> str | None:
    module_name = value.__name__
    if module_name == target_name:
        return alias_name

    target_prefix = f"{target_name}."
    if module_name.startswith(target_prefix):
        return f"{alias_name}{module_name[len(target_name) :]}"

    return None


def _alias_value(value: Any, alias_name: str, target_name: str) -> Any:
    if not isinstance(value, ModuleType):
        return value

    alias_module_name = _module_alias_name(value, alias_name, target_name)
    if alias_module_name is None:
        return value

    if alias_module_name == alias_name:
        return sys.modules.get(alias_name, value)

    return import_module(alias_module_name)


def _alias_spec(alias_name: str, target: ModuleType) -> ModuleSpec | None:
    target_spec = target.__spec__
    if target_spec is None:
        return None

    is_package = target_spec.submodule_search_locations is not None
    spec = ModuleSpec(
        alias_name,
        _AliasLoader(alias_name, target.__name__),
        origin=target_spec.origin,
        is_package=is_package,
    )
    spec.cached = target_spec.cached
    spec.has_location = target_spec.has_location
    if is_package:
        spec.submodule_search_locations = target_spec.submodule_search_locations
    return spec


def _populate_alias_namespace(
    namespace: dict[str, Any],
    alias_name: str,
    target_name: str,
    target: ModuleType,
) -> None:
    namespace["__doc__"] = target.__doc__
    namespace["__package__"] = alias_name if hasattr(target, "__path__") else alias_name.rpartition(".")[0]

    alias_spec = _alias_spec(alias_name, target)
    if alias_spec is not None:
        namespace["__spec__"] = alias_spec
        namespace["__loader__"] = alias_spec.loader

    target_file = getattr(target, "__file__", None)
    if target_file is not None:
        namespace["__file__"] = target_file

    target_cached = getattr(target, "__cached__", None)
    if target_cached is not None:
        namespace["__cached__"] = target_cached

    target_path = getattr(target, "__path__", None)
    if target_path is not None:
        namespace["__path__"] = list(target_path)

    for name, value in target.__dict__.items():
        if name in _MODULE_METADATA_NAMES:
            continue
        if isinstance(value, ModuleType) and _module_alias_name(value, alias_name, target_name) is not None:
            continue
        namespace.setdefault(name, value)

    def __getattr__(name: str) -> Any:
        value = _alias_value(getattr(target, name), alias_name, target_name)
        namespace[name] = value
        return value

    def __dir__() -> list[str]:
        return sorted({*namespace, *dir(target)})

    namespace["__getattr__"] = __getattr__
    namespace["__dir__"] = __dir__


def alias_package(target_name: str, namespace: dict[str, Any]) -> ModuleType:
    """Expose a native source package through a ``nemo_helix`` package path.

    Tiny generated ``__init__.py`` files call this from legacy SDK locations
    such as ``nemo_helix.filesets``. The finder below maps submodule imports
    like ``nemo_helix.filesets.resources`` to the real staged package path
    (``filesets.resources``), so runtime resolution works without copying the
    package tree into ``nemo_helix``.
    """
    alias_name = str(namespace["__name__"])
    target = import_module(target_name)
    _alias_finder().add_alias(alias_name, target_name)
    _populate_alias_namespace(namespace, alias_name, target_name, target)
    return target


def _alias_finder() -> _AliasFinder:
    global _FINDER

    if _FINDER is None:
        _FINDER = _AliasFinder()
        sys.meta_path.insert(0, _FINDER)
    return _FINDER
