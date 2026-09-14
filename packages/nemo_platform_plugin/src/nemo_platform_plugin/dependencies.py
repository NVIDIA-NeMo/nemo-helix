# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common FastAPI dependency placeholders for NeMo Platform services.

These are stub functions that raise RuntimeError if called directly.
The platform injects real implementations via app.dependency_overrides.
"""

from __future__ import annotations

from typing import Any

from nemo_platform import AsyncNeMoPlatform, NeMoPlatform
from nemo_platform_plugin.client.client import AsyncNemoClient
from nemo_platform_plugin.config import PlatformConfig
from nemo_platform_plugin.entities import EntityClient


def get_platform_config() -> PlatformConfig:
    """FastAPI dependency for getting the platform config.

    This is a placeholder — the actual config is injected via
    app.dependency_overrides in Service.create_app().
    """
    raise RuntimeError(
        "get_platform_config() was called without being overridden. "
        "Ensure your Service subclass calls super().create_app()."
    )


def get_service_config() -> Any:
    """FastAPI dependency for getting the service-specific config.

    DEPRECATED: Use get_service_config_factory(ConfigClass) instead.
    """
    raise RuntimeError(
        "get_service_config() was called without being overridden. "
        "Ensure your Service subclass specifies a config type via Service[YourConfig]."
    )


def get_sdk_client() -> AsyncNeMoPlatform:
    """FastAPI dependency for getting the async platform SDK client.

    This is a placeholder — the actual client is injected via
    app.dependency_overrides in Service.create_app().
    """
    raise RuntimeError(
        "get_sdk_client() was called without being overridden. Ensure your Service subclass calls super().create_app()."
    )


def get_sync_sdk_client() -> NeMoPlatform:
    """FastAPI dependency for getting the sync platform SDK client.

    This is a placeholder — the actual client is injected via
    app.dependency_overrides in Service.create_app().
    """
    raise RuntimeError(
        "get_sync_sdk_client() was called without being overridden. "
        "Ensure your Service subclass calls super().create_app()."
    )


def get_nemo_client() -> AsyncNemoClient:
    """FastAPI dependency for getting the async NemoClient.

    This is a placeholder. The actual client is injected via
    app.dependency_overrides in Service.create_app().
    """
    raise RuntimeError(
        "get_nemo_client() was called without being overridden. "
        "Ensure your Service subclass calls super().create_app()."
    )


def get_effective_principal_id() -> str:
    """FastAPI dependency for getting the request's effective principal ID.

    This is a placeholder — the actual principal ID is injected via
    app.dependency_overrides in Service.create_app().
    """
    raise RuntimeError(
        "get_effective_principal_id() was called without being overridden. "
        "Ensure your Service subclass calls super().create_app()."
    )


def get_entity_client() -> EntityClient:
    """FastAPI dependency for getting the EntityClient.

    This is a placeholder — the actual client is injected via
    app.dependency_overrides in Service.create_app().
    """
    raise RuntimeError(
        "get_entity_client() was called without being overridden. "
        "Ensure your Service subclass calls super().create_app() or "
        "configure entity_client in the service."
    )
