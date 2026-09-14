# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo Platform common service module."""

from nmp.common.service.base import DependencyProvider, RouterConfig, Service
from nmp.common.service.dependencies import (
    get_entity_client,
    get_nemo_client,
    get_platform_config,
    get_sdk_client,
    get_service_config,
    get_sync_sdk_client,
)
from nmp.common.service.deptree import CircularDependencyError, resolve_service_loading_order
from nmp.common.service.headers import build_downstream_service_headers

__all__ = [
    "CircularDependencyError",
    "DependencyProvider",
    "Service",
    "RouterConfig",
    "build_downstream_service_headers",
    "get_entity_client",
    "get_nemo_client",
    "get_platform_config",
    "get_sdk_client",
    "get_service_config",
    "get_sync_sdk_client",
    "resolve_service_loading_order",
]
