# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common FastAPI dependencies for NeMo Helix services.

Simple placeholders are re-exported from nemo_helix_plugin.dependencies.
get_service_config_factory (which needs ServiceConfig type) and the service
lookups (which need the Service type) are defined here.
"""

from __future__ import annotations

from typing import Callable, TypeVar

from fastapi import Request
from nemo_helix_plugin.dependencies import get_effective_principal_id as get_effective_principal_id
from nemo_helix_plugin.dependencies import get_entity_client as get_entity_client
from nemo_helix_plugin.dependencies import get_nemo_client as get_nemo_client
from nemo_helix_plugin.dependencies import get_platform_config as get_platform_config
from nemo_helix_plugin.dependencies import get_service_config as get_service_config
from nemo_helix_plugin.dependencies import get_sync_nemo_client as get_sync_nemo_client
from nhx.common.config import ServiceConfig
from nhx.common.service.base import Service

T = TypeVar("T", bound=ServiceConfig)


def get_service_config_factory(config_class: type[T]) -> Callable[[Request], T]:
    """Factory that creates a FastAPI dependency for getting a specific service config.

    This is the preferred way to access service configs when multiple services
    are loaded together in the platform.

    Args:
        config_class: The ServiceConfig subclass to retrieve

    Returns:
        A FastAPI dependency function that retrieves the config from app.state
    """

    def _get_config(request: Request) -> T:
        registry: dict[type[T], T] = getattr(request.app.state, "service_configs", {})
        if config_class not in registry:
            raise RuntimeError(
                f"Service config {config_class.__name__} not registered. "
                "Ensure the service is loaded and its config is added to app.state.service_configs."
            )
        return registry[config_class]

    return _get_config


def find_service(request: Request, name: str) -> Service | None:
    """Return the service registered on the request's app under ``name``, or None.

    Both app shapes keep services in ``app.state.services``: a standalone
    ``Service.create_app`` app holds its own service, and the platform runner's
    merged app holds every mounted service. Use this for optional services.
    """
    services: dict[str, Service] = getattr(request.app.state, "services", {})
    return services.get(name)


def get_service(request: Request, name: str) -> Service:
    """Return the service registered on the request's app under ``name``.

    Raises:
        RuntimeError: If no service with that name is registered on the app.
    """
    service = find_service(request, name)
    if service is None:
        raise RuntimeError(
            f"Service {name!r} is not registered on this app. "
            "Ensure the service is loaded and added to app.state.services."
        )
    return service
