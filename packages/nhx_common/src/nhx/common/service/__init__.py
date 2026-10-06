# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo Helix common service module."""

from nhx.common.service.base import DependencyProvider, RouterConfig, Service
from nhx.common.service.dependencies import (
    get_entity_client,
    get_nemo_client,
    get_platform_config,
    get_service_config,
    get_sync_nemo_client,
)
from nhx.common.service.deptree import CircularDependencyError, resolve_service_loading_order
from nhx.common.service.headers import build_downstream_service_headers, build_downstream_service_headers_async

__all__ = [
    "CircularDependencyError",
    "DependencyProvider",
    "Service",
    "RouterConfig",
    "build_downstream_service_headers",
    "build_downstream_service_headers_async",
    "get_entity_client",
    "get_nemo_client",
    "get_platform_config",
    "get_service_config",
    "get_sync_nemo_client",
    "resolve_service_loading_order",
]
