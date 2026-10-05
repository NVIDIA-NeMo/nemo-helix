# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deployment backend ABC and status projection types."""

from __future__ import annotations

import abc
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

from nemo_deployments_plugin.types import DeploymentStatus, Endpoint, VolumeStatus
from nemo_helix import AsyncNeMoHelix
from nemo_helix_plugin.auth import AuthContext
from nemo_helix_plugin.capabilities import CapabilityUnavailableError
from pydantic import BaseModel, Field


class MissingBackendDependencyError(CapabilityUnavailableError):
    """A backend cannot initialize because a required capability is unavailable.

    Raised when an optional packaging extra is missing (e.g. ``openshell``) or when
    a runtime substrate is unreachable (e.g. Docker daemon/socket). The executor
    registry catches this, skips that executor with a warning, and continues
    starting the deployments service. Subclasses
    :class:`~nemo_helix_plugin.capabilities.CapabilityUnavailableError` (and
    therefore ``RuntimeError``) so existing ``except RuntimeError`` paths keep
    working and jobs/deployments share one unavailable-capability type.
    """


class BackendStatusUpdate(BaseModel):
    """Status projection returned by backends — consumed by the reconciler (758)."""

    status: DeploymentStatus
    status_message: str = ""
    error_details: dict[str, Any] | None = None
    endpoints: list[Endpoint] = Field(default_factory=list)
    exit_code: int | None = None


class VolumeStatusUpdate(BaseModel):
    status: VolumeStatus
    status_message: str = ""
    error_details: dict[str, Any] | None = None


@dataclass(frozen=True)
class LogResult:
    lines: list[str]
    truncated: bool = False


class DeploymentBackend(abc.ABC):
    """Abstract substrate backend for deployment and volume lifecycle."""

    def __init__(self, sdk: AsyncNeMoHelix, config: dict[str, Any]) -> None:
        self._sdk = sdk
        self._config = config
        self.init()

    def init(self) -> None:
        """Optional hook for backend-specific startup."""

    @abstractmethod
    def shutdown(self) -> None:
        """Release backend resources."""
        raise NotImplementedError

    @abstractmethod
    async def create_deployment(
        self,
        *,
        workspace: str,
        name: str,
        config_name: str,
        labels: dict[str, str],
        backend_config: dict[str, Any],
        auth_context: AuthContext | None = None,
    ) -> BackendStatusUpdate:
        raise NotImplementedError

    @abstractmethod
    async def read_status(self, *, workspace: str, name: str) -> BackendStatusUpdate:
        raise NotImplementedError

    @abstractmethod
    async def delete_deployment(self, workspace: str, name: str) -> BackendStatusUpdate:
        raise NotImplementedError

    @abstractmethod
    async def list_managed_deployment_names(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    async def get_logs(
        self,
        *,
        workspace: str,
        name: str,
        tail: int = 100,
    ) -> LogResult:
        raise NotImplementedError

    @abstractmethod
    async def create_volume(
        self,
        *,
        workspace: str,
        name: str,
        size: str,
        access_modes: list[str],
        backend_config: dict[str, Any],
    ) -> VolumeStatusUpdate:
        raise NotImplementedError

    @abstractmethod
    async def read_volume_status(
        self,
        *,
        workspace: str,
        name: str,
        backend_config: dict[str, Any] | None = None,
    ) -> VolumeStatusUpdate:
        raise NotImplementedError

    @abstractmethod
    async def delete_volume(
        self,
        workspace: str,
        name: str,
        *,
        backend_config: dict[str, Any] | None = None,
    ) -> VolumeStatusUpdate:
        raise NotImplementedError
