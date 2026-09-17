# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common FastAPI dependencies for NeMo Platform services.

Simple placeholders are re-exported from nemo_platform_plugin.dependencies.
get_service_config_factory (which needs ServiceConfig type) is defined here.
"""

from __future__ import annotations

from typing import Callable, TypeVar

from fastapi import Request
from nemo_platform_plugin.dependencies import get_effective_principal_id as get_effective_principal_id
from nemo_platform_plugin.dependencies import get_entity_client as get_entity_client
from nemo_platform_plugin.dependencies import get_nemo_client as get_nemo_client
from nemo_platform_plugin.dependencies import get_platform_config as get_platform_config
from nemo_platform_plugin.dependencies import get_sdk_client as get_sdk_client
from nemo_platform_plugin.dependencies import get_service_config as get_service_config
from nemo_platform_plugin.dependencies import get_sync_sdk_client as get_sync_sdk_client
from nmp.common.config import ServiceConfig

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
