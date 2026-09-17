# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RunnerBackend abstraction — manages the lifecycle of agent runtime processes.

Implementations:
- :class:`~nemo_agents_plugin.runner.in_memory.InMemoryRunnerBackend` — spawns
  ``nat serve`` subprocesses (subprocess mode).
- :class:`~nemo_agents_plugin.runner.deployments_backend.DeploymentsRunnerBackend`
  — durable docker/k8s containers via the nemo-deployments plugin.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from nemo_agents_plugin.entities import ComputeResources, DeploymentMode, DeploymentStatus, Endpoint
from nemo_agents_plugin.spec_revision import SpecRevision
from nemo_platform_plugin.auth import AuthContext


@dataclass(frozen=True)
class LocalLog:
    """Log file lives on this host."""

    path: Path
    kind: Literal["local"] = "local"


@dataclass(frozen=True)
class NotYetAvailable:
    """Deployment exists; runtime hasn't produced log output yet."""

    kind: Literal["not_yet_available"] = "not_yet_available"


@dataclass(frozen=True)
class ExternalLog:
    """Logs fetched via a backend-specific channel (``docker logs``, ``kubectl logs``, ...)."""

    kind: Literal["external"] = "external"
    hint: str = ""


LogLocation = LocalLog | NotYetAvailable | ExternalLog


@dataclass
class DeploymentInfo:
    """Runtime snapshot of a deployment managed by the backend.

    This is the backend's in-memory view, distinct from the ``AgentDeployment``
    entity in the store.  The controller reads this to update the entity.
    """

    name: str
    status: DeploymentStatus = "pending"
    endpoint: str = ""
    """HTTP endpoint of the process, e.g. ``http://localhost:9001`` (subprocess)."""
    endpoints: list[Endpoint] = field(default_factory=list)
    """Projected routable endpoints for container modes."""
    port: int = 0
    pid: int = 0
    error: str = ""
    log_path: str = ""
    """Absolute path to the subprocess log file (empty if not applicable)."""
    extra: dict[str, Any] = field(default_factory=dict)
    """Backend-specific metadata (e.g. container ID for Docker)."""
    staged_spec: SpecRevision | None = None
    """What the spec fileset resolved to when staged; None when the backend staged nothing."""


class RunnerBackend(ABC):
    """Abstract base class for managing agent runtime processes.

    All async methods are called from the asyncio reconcile loop.
    Blocking operations must use ``asyncio.to_thread`` internally.
    """

    def allocate_port(self) -> int:
        """Return the next port to use for a new deployment.

        Backends that manage port allocation override this.  The default
        returns 0, meaning the backend handles port allocation internally
        (e.g. Docker, Kubernetes).
        """
        return 0

    @abstractmethod
    async def create_deployment(
        self,
        workspace: str,
        name: str,
        config: dict[str, Any],
        port: int,
        *,
        agent: str = "",
        image: str | None = None,
        deployment_mode: DeploymentMode = "subprocess",
        created_by: str | None = None,
        auth_context: AuthContext | None = None,
        resources: ComputeResources | None = None,
        secrets: dict[str, str] | None = None,
        use_image_entrypoint: bool = False,
    ) -> DeploymentInfo:
        """Start the agent process; returns status="starting".

        ``created_by`` is the principal id that created the deployment. When
        platform auth is enabled, container-mode backends delegate inference calls
        through the auth-proxy sidecar with this principal as on-behalf-of.

        ``auth_context`` is the full creator identity snapshot used by container
        backends that request workload token exchange for direct platform SDK
        calls from inside the deployed agent.

        ``resources`` is the snapshotted compute spec's k8s-style
        requests/limits. Container backends compile it into the execute
        container's resources (k8s passes both; docker consolidates to limits).
        Subprocess mode ignores it.

        ``secrets`` maps ENV_VAR_NAME -> "workspace/secret-name" from the
        resolved environment. Container backends compile these into secret-backed
        container env vars (never plaintext); the deployments-plugin substrate
        materializes/mounts them. Subprocess mode ignores it.

        ``use_image_entrypoint`` is only meaningful for container backends. When
        true, those backends preserve the image ENTRYPOINT/CMD instead of
        injecting the platform-owned server command.
        """
        ...

    @abstractmethod
    async def get_deployment_status(self, workspace: str, name: str) -> DeploymentInfo | None:
        """Return the current runtime state of a deployment, or ``None`` if unknown."""
        ...

    @abstractmethod
    async def delete_deployment(self, workspace: str, name: str) -> bool:
        """Stop + clean up.

        Returns ``True`` when teardown is complete (or already gone) and the
        ``AgentDeployment`` entity may be deleted. Returns ``False`` when
        teardown is still in progress and the controller should keep the entity
        in ``deleting`` for a later reconcile pass.
        """
        ...

    @abstractmethod
    async def list_deployments(self, workspace: str | None = None) -> list[DeploymentInfo]:
        """Snapshot of deployments; ``workspace=None`` returns every workspace."""
        ...

    @abstractmethod
    async def health_check(self, endpoint: str) -> bool:
        """Return ``True`` if the agent at *endpoint* passes ``GET /health``."""
        ...

    def get_log_location(self, workspace: str, name: str) -> "LogLocation":
        """Where to read logs for (workspace, name). Default: NotYetAvailable."""
        del workspace, name
        return NotYetAvailable()

    @abstractmethod
    async def shutdown(self) -> None:
        """Terminate all managed processes and release resources.

        Called during service shutdown.  Must be idempotent.
        """
        ...
