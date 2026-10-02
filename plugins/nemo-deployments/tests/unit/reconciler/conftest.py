# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for reconciler unit tests."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from nemo_deployments_plugin.backends.base import BackendStatusUpdate, DeploymentBackend, LogResult, VolumeStatusUpdate
from nemo_deployments_plugin.backends.registry import ExecutorRegistry
from nemo_deployments_plugin.config import ControllerConfig
from nemo_deployments_plugin.reconciler.deployment_reconciler import DeploymentReconciler
from nemo_deployments_plugin.reconciler.volume_reconciler import VolumeReconciler
from nemo_helix import AsyncNeMoHelix


class MockDeploymentBackend(DeploymentBackend):
    """Configurable stub backend for reconciler tests."""

    def __init__(
        self,
        sdk: AsyncNeMoHelix | None = None,
        config: dict[str, Any] | None = None,
        *,
        create_status: BackendStatusUpdate | None = None,
        read_status: BackendStatusUpdate | None = None,
        delete_status: BackendStatusUpdate | None = None,
        managed_names: list[str] | None = None,
        volume_create_status: VolumeStatusUpdate | None = None,
        volume_delete_status: VolumeStatusUpdate | None = None,
    ) -> None:
        self.create_status = create_status or BackendStatusUpdate(status="STARTING", status_message="created")
        self.read_status_result = read_status or BackendStatusUpdate(status="READY", status_message="running")
        self.delete_status = delete_status or BackendStatusUpdate(status="SUCCEEDED", status_message="deleted")
        self.managed_names = list(managed_names or [])
        self.volume_create_status = volume_create_status or VolumeStatusUpdate(status="BOUND")
        self.volume_delete_status = volume_delete_status or VolumeStatusUpdate(status="RELEASED")
        self.create_calls: list[dict[str, Any]] = []
        self.read_calls: list[tuple[str, str]] = []
        self.deployment_delete_calls: list[tuple[str, str]] = []
        self.volume_delete_calls: list[tuple[str, str]] = []
        super().__init__(sdk or AsyncMock(), config or {})

    def shutdown(self) -> None:
        pass

    async def create_deployment(self, **kwargs: Any) -> BackendStatusUpdate:
        self.create_calls.append(kwargs)
        return self.create_status

    async def read_status(self, *, workspace: str, name: str) -> BackendStatusUpdate:
        self.read_calls.append((workspace, name))
        return self.read_status_result

    async def delete_deployment(self, workspace: str, name: str) -> BackendStatusUpdate:
        self.deployment_delete_calls.append((workspace, name))
        return self.delete_status

    async def list_managed_deployment_names(self) -> list[str]:
        return list(self.managed_names)

    async def get_logs(self, **kwargs: Any) -> LogResult:
        return LogResult(lines=[])

    async def create_volume(self, **kwargs: Any) -> VolumeStatusUpdate:
        return self.volume_create_status

    async def read_volume_status(
        self,
        *,
        workspace: str,
        name: str,
        backend_config: dict[str, Any] | None = None,
    ) -> VolumeStatusUpdate:
        return VolumeStatusUpdate(status="BOUND")

    async def delete_volume(
        self,
        workspace: str,
        name: str,
        *,
        backend_config: dict[str, Any] | None = None,
    ) -> VolumeStatusUpdate:
        self.volume_delete_calls.append((workspace, name))
        return self.volume_delete_status


@pytest.fixture
def controller_config() -> ControllerConfig:
    return ControllerConfig(
        interval_seconds=5,
        drift_recovery_max_attempts=3,
        drift_recovery_initial_delay_seconds=1,
        drift_recovery_max_delay_seconds=10,
    )


@pytest.fixture
def mock_entities() -> AsyncMock:
    client = AsyncMock()
    client.update = AsyncMock(side_effect=lambda entity: entity)
    return client


@pytest.fixture
def mock_backend() -> MockDeploymentBackend:
    return MockDeploymentBackend()


@pytest.fixture
def executor_registry(mock_backend: MockDeploymentBackend) -> ExecutorRegistry:
    return ExecutorRegistry({"default": mock_backend}, default_executor="default")


@pytest.fixture
def deployment_reconciler(
    mock_entities: AsyncMock,
    executor_registry: ExecutorRegistry,
    controller_config: ControllerConfig,
) -> DeploymentReconciler:
    return DeploymentReconciler(mock_entities, executor_registry, controller_config)


@pytest.fixture
def volume_reconciler(mock_entities: AsyncMock, executor_registry: ExecutorRegistry) -> VolumeReconciler:
    return VolumeReconciler(mock_entities, executor_registry)
