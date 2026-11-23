# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenSandbox deployment backend: one sandbox per deployment.

The sandbox starts with an idle entrypoint. Once it is running, ``read_status`` delivers the
config files through execd, starts the serve command as an execd background command with the
deployment's environment, and reports READY when the readiness probe passes. The serve command's
id and the ready flag are kept in the sandbox's metadata.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
from typing import Any

from nemo_deployments_plugin.backends.base import (
    BackendStatusUpdate,
    DeploymentBackend,
    LogResult,
    MissingBackendDependencyError,
    VolumeStatusUpdate,
)
from nemo_deployments_plugin.backends.labels import (
    CONFIG_NAME_LABEL,
    DEPLOYMENT_NAME_LABEL,
    DEPLOYMENT_WORKSPACE_LABEL,
    MANAGED_BY_KEY,
    deployment_identity_labels,
)
from nemo_deployments_plugin.backends.opensandbox.client import (
    EXECD_ACCESS_TOKEN_ENV,
    OpenSandboxClient,
    OpenSandboxError,
)
from nemo_deployments_plugin.backends.opensandbox.config import OpenSandboxExecutorConfig
from nemo_deployments_plugin.backends.opensandbox.endpoint import sandbox_endpoint_url
from nemo_deployments_plugin.backends.workload_identity import (
    workload_identity_activation_error,
    workload_identity_requested,
)
from nemo_deployments_plugin.constants import MANAGED_BY_LABEL
from nemo_deployments_plugin.entities import Container, DeploymentConfig
from nemo_deployments_plugin.secrets import SecretResolutionError, resolve_deployment_config_secrets
from nemo_deployments_plugin.types import Endpoint
from nemo_helix_plugin.auth import AuthContext, platform_auth_enabled
from nemo_helix_plugin.entities.client import AsyncEntitiesClient
from nemo_helix_plugin.entity_client import NemoEntitiesClient, NemoEntityNotFoundError

logger = logging.getLogger(__name__)

SERVE_COMMAND_KEY = "nemo.nvidia.com/serve-command"
SERVE_READY_KEY = "nemo.nvidia.com/serve-ready"

_IDLE_ENTRYPOINT = ["tail", "-f", "/dev/null"]
_DEAD_STATES = frozenset({"Failed", "Terminated", "Stopping"})
_STARTING_STATES = frozenset({"Pending", "Resuming"})
_LOG_TAIL_CHARS = 2000

Sandbox = dict[str, Any]


class OpenSandboxDeploymentBackend(DeploymentBackend):
    """Manage deployments as OpenSandbox sandboxes through the lifecycle and execd APIs."""

    def init(self) -> None:
        self._executor_config = OpenSandboxExecutorConfig.model_validate(self._config)
        api_key = os.environ.get(self._executor_config.api_key_env)
        if not api_key:
            raise MissingBackendDependencyError(
                f"The OpenSandbox API key is not set: {self._executor_config.api_key_env} is empty."
            )
        self._api_key = api_key
        self._entities = NemoEntitiesClient(AsyncEntitiesClient.from_client(self._nemo_client))

    def shutdown(self) -> None:
        pass

    def _client(self) -> OpenSandboxClient:
        return OpenSandboxClient(self._executor_config, self._api_key)

    def _execd_token(self, workspace: str, name: str) -> str:
        """execd access token of a deployment's sandbox, derived from the API key."""
        return hmac.new(self._api_key.encode(), f"{workspace}/{name}".encode(), hashlib.sha256).hexdigest()

    @staticmethod
    async def _sandboxes(client: OpenSandboxClient, workspace: str, name: str) -> tuple[Sandbox | None, list[Sandbox]]:
        """The deployment's live sandbox, if any, and its failed or terminating ones."""
        sandboxes = await client.list_sandboxes(
            {MANAGED_BY_KEY: MANAGED_BY_LABEL, DEPLOYMENT_WORKSPACE_LABEL: workspace, DEPLOYMENT_NAME_LABEL: name}
        )
        live = [sandbox for sandbox in sandboxes if _state(sandbox) not in _DEAD_STATES]
        dead = [sandbox for sandbox in sandboxes if _state(sandbox) in _DEAD_STATES]
        return (live[0] if live else None), dead

    async def _load_config(self, workspace: str, config_name: str) -> DeploymentConfig:
        return await self._entities.get(DeploymentConfig, config_name, workspace=workspace)

    # -- deployments -----------------------------------------------------------------------------

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
        try:
            config = await self._load_config(workspace, config_name)
        except NemoEntityNotFoundError:
            return _failed(f"DeploymentConfig '{config_name}' not found in workspace '{workspace}'")
        problem = _unsupported(config, auth_context)
        if problem is not None:
            return _failed(problem)
        container = config.containers[0]

        metadata = {
            **labels,
            **config.labels,
            **deployment_identity_labels(
                workspace, name, config.restart_policy, config_name=config_name, backoff_limit=config.backoff_limit
            ),
        }
        egress = self._executor_config.platform_egress
        body: dict[str, Any] = {
            "image": {"uri": container.image},
            "entrypoint": _IDLE_ENTRYPOINT,
            "timeout": None,
            "resourceLimits": dict(container.resources.limits) or dict(self._executor_config.default_resources),
            "metadata": metadata,
            "env": {EXECD_ACCESS_TOKEN_ENV: self._execd_token(workspace, name)},
            "networkPolicy": {"defaultAction": "deny", "egress": [egress.egress_rule()] if egress else []},
            "secureAccess": True,
        }
        if container.resources.requests:
            body["resourceRequests"] = dict(container.resources.requests)

        try:
            async with self._client() as client:
                live, dead = await self._sandboxes(client, workspace, name)
                if live is not None:
                    return await self._status(client, workspace, name, live)
                # Drift recovery: a failed sandbox is replaced, not restarted.
                for sandbox in dead:
                    await client.delete_sandbox(sandbox["id"])
                created = await client.create_sandbox(body)
        except OpenSandboxError as exc:
            return _failed(str(exc))
        return BackendStatusUpdate(status="STARTING", status_message=f"Sandbox {created.get('id')} created")

    async def read_status(self, *, workspace: str, name: str) -> BackendStatusUpdate:
        try:
            async with self._client() as client:
                live, dead = await self._sandboxes(client, workspace, name)
                if live is None:
                    detail = (dead[0].get("status") or {}).get("message") if dead else None
                    return BackendStatusUpdate(
                        status="LOST", status_message=f"Sandbox failed: {detail}" if detail else "Sandbox not found"
                    )
                return await self._status(client, workspace, name, live)
        except OpenSandboxError as exc:
            return BackendStatusUpdate(status="UNKNOWN", status_message=str(exc), error_details={"error": str(exc)})

    async def _status(
        self, client: OpenSandboxClient, workspace: str, name: str, sandbox: Sandbox
    ) -> BackendStatusUpdate:
        state = _state(sandbox)
        if state in _STARTING_STATES:
            return BackendStatusUpdate(status="STARTING", status_message=f"Sandbox is {state}")
        if state != "Running":
            return BackendStatusUpdate(status="UNKNOWN", status_message=f"Sandbox is {state}")

        metadata: dict[str, str] = sandbox.get("metadata") or {}
        config_name = metadata.get(CONFIG_NAME_LABEL)
        if not config_name:
            return _failed("Sandbox is missing its deployment-config label")
        try:
            config = await self._load_config(workspace, config_name)
        except NemoEntityNotFoundError:
            return _failed(f"DeploymentConfig '{config_name}' not found in workspace '{workspace}'")
        container = config.containers[0]
        sandbox_id: str = sandbox["id"]
        token = self._execd_token(workspace, name)

        command_id = metadata.get(SERVE_COMMAND_KEY)
        if command_id is None:
            return await self._launch(client, sandbox_id, token, config)

        try:
            status = await client.command_status(sandbox_id, token, command_id)
        except OpenSandboxError as exc:
            if exc.status_code != 404:
                raise
            return _failed("execd no longer tracks the serve process; deployment is not serving")
        if not status.get("running"):
            exit_code = status.get("exit_code")
            tail = (await client.command_logs(sandbox_id, token, command_id))[-_LOG_TAIL_CHARS:].strip()
            message = f"Serve process exited with code {exit_code}; deployment is not serving"
            return BackendStatusUpdate(
                status="FAILED",
                status_message=f"{message}. Last output: {tail}" if tail else message,
                exit_code=exit_code if isinstance(exit_code, int) else None,
            )

        if metadata.get(SERVE_READY_KEY) != "true":
            if not await self._ready(client, sandbox_id, container):
                return BackendStatusUpdate(status="STARTING", status_message="Serve running; awaiting readiness")
            await client.patch_metadata(sandbox_id, {SERVE_READY_KEY: "true"})
        return BackendStatusUpdate(
            status="READY", status_message="Sandbox serving", endpoints=_endpoints(sandbox_id, container)
        )

    async def _launch(
        self, client: OpenSandboxClient, sandbox_id: str, token: str, config: DeploymentConfig
    ) -> BackendStatusUpdate:
        """Deliver the config files and start the serve command, recording its id on the sandbox."""
        try:
            resolved = await resolve_deployment_config_secrets(self._nemo_client, config)
        except SecretResolutionError as exc:
            return _failed(str(exc))
        container = resolved.containers[0]
        unresolved = sorted(var.name for var in container.env if var.value is None)
        if unresolved:
            return _failed(f"Cannot resolve a value for environment variable(s): {', '.join(unresolved)}")
        envs = {var.name: var.value for var in container.env if var.value is not None}
        try:
            for item in resolved.config_files:
                await client.upload_file(
                    sandbox_id, token, path=item.path, content=item.content.encode(), mode=item.mode
                )
            command_id = await client.start_command(sandbox_id, token, [*container.command, *container.args], envs=envs)
        except OpenSandboxError as exc:
            if exc.status_code is None or exc.status_code >= 500:
                # execd may not be listening yet.
                return BackendStatusUpdate(status="STARTING", status_message=f"Waiting for execd: {exc}")
            return _failed(str(exc))
        try:
            await client.patch_metadata(sandbox_id, {SERVE_COMMAND_KEY: command_id})
        except OpenSandboxError:
            # The next poll would start a second serve process.
            await client.interrupt_command(sandbox_id, token, command_id)
            raise
        return BackendStatusUpdate(status="STARTING", status_message="Serve command launched")

    @staticmethod
    async def _ready(client: OpenSandboxClient, sandbox_id: str, container: Container) -> bool:
        """Whether the container's HTTP readiness probe passes. No HTTP probe counts as ready."""
        probe = container.readiness_probe
        if probe is None or probe.http_get is None:
            return True
        port = _resolve_port(probe.http_get.port, container)
        if port is None:
            return True
        try:
            status = await client.http_get(sandbox_id, port, probe.http_get.path, timeout=probe.timeout_seconds)
        except OpenSandboxError:
            return False
        return 200 <= status < 400

    async def delete_deployment(self, workspace: str, name: str) -> BackendStatusUpdate:
        try:
            async with self._client() as client:
                live, dead = await self._sandboxes(client, workspace, name)
                for sandbox in [*([live] if live else []), *dead]:
                    if _state(sandbox) != "Stopping":
                        await client.delete_sandbox(sandbox["id"])
        except OpenSandboxError as exc:
            return _failed(str(exc))
        return BackendStatusUpdate(status="SUCCEEDED", status_message="Sandbox deleted")

    async def list_managed_deployment_names(self) -> list[str]:
        try:
            async with self._client() as client:
                sandboxes = await client.list_sandboxes({MANAGED_BY_KEY: MANAGED_BY_LABEL})
        except OpenSandboxError:
            logger.warning("Failed to list managed sandboxes", exc_info=True)
            return []
        names: set[str] = set()
        for sandbox in sandboxes:
            metadata = sandbox.get("metadata") or {}
            workspace, name = metadata.get(DEPLOYMENT_WORKSPACE_LABEL), metadata.get(DEPLOYMENT_NAME_LABEL)
            if workspace and name:
                names.add(f"{workspace}/{name}")
        return sorted(names)

    async def get_logs(self, *, workspace: str, name: str, tail: int = 100) -> LogResult:
        """Output of the serve command."""
        try:
            async with self._client() as client:
                live, _ = await self._sandboxes(client, workspace, name)
                if live is None:
                    return LogResult(lines=["Sandbox not found"])
                command_id = (live.get("metadata") or {}).get(SERVE_COMMAND_KEY)
                if command_id is None:
                    return LogResult(lines=["The serve command has not started"])
                output = await client.command_logs(live["id"], self._execd_token(workspace, name), command_id)
        except OpenSandboxError as exc:
            return LogResult(lines=[f"Failed to fetch logs: {exc}"])
        lines = output.splitlines()
        return LogResult(lines=lines[-tail:], truncated=len(lines) > tail)

    # -- volumes ---------------------------------------------------------------------------------

    async def create_volume(
        self, *, workspace: str, name: str, size: str, access_modes: list[str], backend_config: dict[str, Any]
    ) -> VolumeStatusUpdate:
        return _volumes_unsupported()

    async def read_volume_status(
        self, *, workspace: str, name: str, backend_config: dict[str, Any] | None = None
    ) -> VolumeStatusUpdate:
        return _volumes_unsupported()

    async def delete_volume(
        self, workspace: str, name: str, *, backend_config: dict[str, Any] | None = None
    ) -> VolumeStatusUpdate:
        return _volumes_unsupported()


def _state(sandbox: Sandbox) -> str:
    return str((sandbox.get("status") or {}).get("state") or "")


def _failed(message: str) -> BackendStatusUpdate:
    return BackendStatusUpdate(status="FAILED", status_message=message)


def _volumes_unsupported() -> VolumeStatusUpdate:
    return VolumeStatusUpdate(status="FAILED", status_message="Volumes are not supported by the opensandbox backend")


def _unsupported(config: DeploymentConfig, auth_context: AuthContext | None) -> str | None:
    """Why the opensandbox backend can't run ``config``, or None when it can."""
    if workload_identity_requested(config):
        return (
            workload_identity_activation_error(config=config, auth_context=auth_context)
            or "workload_identity is not supported by the opensandbox backend"
        )
    if config.auth_proxy_sidecar and platform_auth_enabled():
        return "Platform auth is enabled, and the opensandbox backend has no platform identity for deployments yet"
    if config.gateway_providers:
        return "Gateway providers require an OpenShell executor"
    if len(config.containers) != 1:
        return "The opensandbox backend runs exactly one container per deployment"
    if config.init_containers:
        return "initContainers are not supported by the opensandbox backend"
    container = config.containers[0]
    if config.volume_mounts or container.volume_mounts:
        return "Volume mounts are not supported by the opensandbox backend"
    if not container.image:
        return "containers[0].image is required"
    if not (container.command or container.args):
        return (
            "The opensandbox backend requires containers[0].command (the serve command); "
            "the image entrypoint is not run"
        )
    if any(var.value_from is not None for var in container.env):
        return "valueFrom environment variables are not supported by the opensandbox backend"
    return None


def _resolve_port(port: int | str, container: Container) -> int | None:
    if isinstance(port, int):
        return port
    if port.isdigit():
        return int(port)
    return next((item.container_port for item in container.ports if item.name == port), None)


def _endpoints(sandbox_id: str, container: Container) -> list[Endpoint]:
    return [
        Endpoint(
            name=port.name or f"port-{port.container_port}",
            url=sandbox_endpoint_url(sandbox_id, port.container_port),
            protocol="http",
        )
        for port in container.ports
    ]
