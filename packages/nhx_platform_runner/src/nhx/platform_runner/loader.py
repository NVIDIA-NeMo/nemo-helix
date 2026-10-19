# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load services and controllers by import path."""

from __future__ import annotations

import importlib
import logging
import threading
from collections.abc import Callable
from typing import cast

from nhx.common.service import Service
from nhx.common.service.deptree import resolve_service_loading_order

logger = logging.getLogger(__name__)

ControllerRunFunc = Callable[[threading.Event], object]


def load_service(
    service_name: str,
    import_path: str,
    dependencies: list[str] | None = None,
) -> Service:
    """Load a service instance from an import path."""
    if ":" not in import_path:
        raise ValueError(f"Import path must be in format 'module:variable', got: {import_path}")

    module_path, variable_name = import_path.split(":", 1)
    module = importlib.import_module(module_path)
    service_instance = getattr(module, variable_name)

    if not isinstance(service_instance, Service):
        raise TypeError(f"Service {service_name} must be an instance of Service, got {type(service_instance)}")

    if dependencies is not None:
        service_instance._dependencies = list(dependencies)

    logger.debug("Loaded service %s", service_name)
    return service_instance


def order_services_by_dependencies(services: list[Service]) -> list[Service]:
    """Return services in startup order."""
    if not services:
        return []

    dependency_tree = {service.name: list(service._dependencies) for service in services}
    ordered_names = resolve_service_loading_order([service.name for service in services], dependency_tree)
    services_by_name = {service.name: service for service in services}
    return [services_by_name[name] for name in ordered_names if name in services_by_name]


def load_controller_run_func(controller_name: str, import_path: str) -> ControllerRunFunc:
    """Load a controller run function from an import path."""
    if ":" not in import_path:
        raise ValueError(f"Import path must be in format 'module:function', got: {import_path}")

    module_path, function_name = import_path.split(":", 1)
    module = importlib.import_module(module_path)
    run_func = getattr(module, function_name)

    if not callable(run_func):
        raise TypeError(f"Controller {controller_name} must be a callable, got {type(run_func)}")

    logger.debug("Loaded controller %s", controller_name)
    return cast(ControllerRunFunc, run_func)
