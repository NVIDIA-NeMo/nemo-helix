# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base backend interface for Models Controller service."""

from abc import ABC, abstractmethod
from typing import Any, Dict

from nemo_platform import AsyncNeMoPlatform
from nemo_platform_plugin.models.types import ModelDeploymentStatus
from nmp.core.models.controllers.context import ModelContext
from pydantic import BaseModel


class DeploymentStatusUpdate(BaseModel):
    """
    Status update for a model deployment.
    This is the message that the service backend returns to the controller for every operation.
    """

    status: ModelDeploymentStatus
    status_message: str = ""
    error_details: Dict[str, Any] | None = None
    host_url: str | None = None


class ServiceBackend(ABC):
    """
    Abstract base class for service backends that manage ModelDeployment lifecycle.

    Each backend implementation handles the actual creation, updating, and monitoring
    of model deployments in a specific environment (e.g., Docker, Kubernetes).
    """

    def __init__(
        self,
        nmp_sdk: AsyncNeMoPlatform,
        config: Dict[str, Any],
    ) -> None:
        """Initialize the service backend.

        Args:
            nmp_sdk: NeMo Platform SDK client for API interactions (includes secrets access)
            config: Backend-specific configuration dictionary
        """
        self._nmp_sdk = nmp_sdk
        self._config = config
        self.init()

    def init(self) -> None:
        """Optional initialization hook for backend-specific setup.

        Override this method to perform any additional initialization
        after the backend is instantiated.
        """
        ...

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the backend.

        Override this method to perform any cleanup when the backend
        is shutting down (e.g., closing connections, releasing resources).
        """
        ...

    @abstractmethod
    async def create_model_deployment(self, ctx: ModelContext) -> DeploymentStatusUpdate:
        """Create a new model deployment.

        Args:
            ctx: The reconciliation context bundling the ModelDeployment, its
                ModelDeploymentConfig, and the optional Model entity.

        Returns:
            DeploymentStatusUpdate with the current status after creation attempt

        Raises:
            Exception: If deployment creation fails
        """
        ...

    @abstractmethod
    async def update_model_deployment(self, ctx: ModelContext) -> DeploymentStatusUpdate:
        """Update an existing model deployment.

        Args:
            ctx: The reconciliation context bundling the ModelDeployment, its
                (possibly new-version) ModelDeploymentConfig, and the optional
                Model entity.

        Returns:
            DeploymentStatusUpdate with the current status after update attempt

        Raises:
            Exception: If deployment update fails
        """
        ...

    @abstractmethod
    async def get_model_deployment_status(self, ctx: ModelContext) -> DeploymentStatusUpdate:
        """Get the current status of a model deployment.

        Args:
            ctx: The reconciliation context bundling the ModelDeployment, its
                ModelDeploymentConfig, and the optional Model entity. Some backends
                need the config to advance creation (e.g. the k8s vLLM path emits
                the serving Deployment once the weight-puller Job completes).

        Returns:
            DeploymentStatusUpdate with the current deployment status

        Raises:
            Exception: If status check fails
        """
        ...

    @abstractmethod
    async def delete_model_deployment(
        self,
        workspace: str,
        name: str,
        *,
        deleting_elapsed_seconds: float | None = None,
    ) -> DeploymentStatusUpdate:
        """Delete a model deployment by workspace and name (model deployment ID).

        Used for both regular reconciliation (controller passes deployment.workspace/name)
        and orphan cleanup (controller passes workspace/name from list_managed_deployment_names).

        Args:
            workspace: Deployment workspace
            name: Deployment name
            deleting_elapsed_seconds: Seconds the deployment has been in DELETING; used by
                backends that escalate stuck teardown to ERROR.

        Returns:
            DeploymentStatusUpdate with the current status after deletion attempt

        Raises:
            Exception: If deployment deletion fails
        """
        ...

    @abstractmethod
    async def list_managed_deployment_names(self) -> list[str]:
        """List deployment names the backend currently manages.

        Uses backend-specific labels (e.g. docker/k8s managed-by) to enumerate
        resources. Used for orphan reconciliation.

        Returns:
            List of "workspace/name" strings for each managed deployment.
        """
        ...
