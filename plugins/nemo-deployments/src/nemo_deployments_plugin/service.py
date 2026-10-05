# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deployments plugin service registration."""

from __future__ import annotations

import logging
from typing import ClassVar

from nemo_deployments_plugin.backends.registry import ExecutorRegistry, ExecutorSpec
from nemo_deployments_plugin.config import DeploymentsConfig
from nemo_helix import AsyncNeMoHelix
from nemo_helix_plugin.sdk_provider import get_async_platform_sdk
from nemo_helix_plugin.service import NemoService, RouterSpec

logger = logging.getLogger(__name__)


class DeploymentsService(NemoService):
    """HTTP service for deployment configs, deployments, volumes, and controller status."""

    name: ClassVar[str] = "deployments"
    dependencies: ClassVar[list[str]] = ["entities", "auth"]

    def __init__(self) -> None:
        self._executor_registry: ExecutorRegistry | None = None

    @property
    def executor_registry(self) -> ExecutorRegistry:
        if self._executor_registry is None:
            self._executor_registry = ExecutorRegistry.empty()
        return self._executor_registry

    def get_routers(self) -> list[RouterSpec]:
        from nemo_deployments_plugin.api.v2 import (
            deployment_configs,
            deployments,
            status,
            volumes,
        )

        prefix = "/v2/workspaces/{workspace}"
        return [
            RouterSpec(
                deployment_configs.router,
                tag="Deployment Configs",
                description="Immutable deployment templates",
                prefix=prefix,
            ),
            RouterSpec(
                deployments.router,
                tag="Deployments",
                description="Deployment lifecycle",
                prefix=prefix,
            ),
            RouterSpec(
                volumes.router,
                tag="Volumes",
                description="Volume lifecycle",
                prefix=prefix,
            ),
            RouterSpec(
                status.router,
                tag="Deployment Status",
                description="Controller-only status projection",
                prefix=prefix,
            ),
        ]

    async def on_startup(self) -> None:
        config = DeploymentsConfig.get()
        sdk: AsyncNeMoHelix = get_async_platform_sdk(as_service="deployments", internal=True)
        specs = [ExecutorSpec(name=e.name, backend=e.backend, config=e.config) for e in config.executors]
        if specs:
            self._executor_registry = ExecutorRegistry.from_config(
                sdk,
                specs,
                default_executor=config.default_executor,
            )
        else:
            self._executor_registry = ExecutorRegistry.empty()
            if config.default_executor:
                logger.warning(
                    "default_executor '%s' is configured but no executors are registered.",
                    config.default_executor,
                )
            logger.info("Deployments plugin started with zero registered executors.")

    async def on_shutdown(self) -> None:
        if self._executor_registry is not None:
            self._executor_registry.shutdown_all()
