# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenSandbox deployment backend: one sandbox per deployment.

The sandbox starts with an idle entrypoint. Once it is running, ``read_status`` delivers the
config files through execd, starts the serve command as an execd background command with the
deployment's environment, and reports READY when the readiness probe passes. The serve command's
id and the ready flag are kept in the sandbox's metadata.

The backend needs the ``opensandbox`` extra.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
from ipaddress import ip_address
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
from nemo_deployments_plugin.backends.opensandbox.config import OpenSandboxExecutorConfig, PlatformEgressConfig
from nemo_deployments_plugin.backends.opensandbox.endpoint import RESERVED_PORTS, sandbox_endpoint_url
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

_OPENSANDBOX_IMPORT_ERROR: ImportError | None = None
try:
    from opensandbox.exceptions import SandboxApiException, SandboxException
    from opensandbox.models.execd import RunCommandOpts
    from opensandbox.models.filesystem import WriteEntry
    from opensandbox.models.sandboxes import NetworkPolicy, NetworkRule, SandboxImageSpec, SandboxInfo

    from .client import EXECD_ACCESS_TOKEN_ENV, Execd, OpenSandboxClient, describe
except ImportError as exc:
    _OPENSANDBOX_IMPORT_ERROR = exc

logger = logging.getLogger(__name__)

_OPENSANDBOX_INSTALL_HINT = (
    "The 'opensandbox' package is required for OpenSandboxDeploymentBackend. "
    'Install it with: uv pip install "opensandbox>=1.2.0"'
)

SERVE_COMMAND_KEY = "nemo.nvidia.com/serve-command"
SERVE_READY_KEY = "nemo.nvidia.com/serve-ready"

_IDLE_ENTRYPOINT = ["tail", "-f", "/dev/null"]
_DEAD_STATES = frozenset({"Failed", "Terminated", "Stopping"})
_STARTING_STATES = frozenset({"Pending", "Allocated", "Resuming"})
_LOG_TAIL_CHARS = 2000


class OpenSandboxDeploymentBackend(DeploymentBackend):
    """Manage deployments as OpenSandbox sandboxes through the lifecycle and execd APIs."""

    def init(self) -> None:
        if _OPENSANDBOX_IMPORT_ERROR is not None:
            raise MissingBackendDependencyError(_OPENSANDBOX_INSTALL_HINT) from _OPENSANDBOX_IMPORT_ERROR
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
    async def _sandboxes(
        client: OpenSandboxClient, workspace: str, name: str
    ) -> tuple[SandboxInfo | None, list[SandboxInfo]]:
        """The deployment's live sandbox, if any, and its failed or terminating ones."""
        sandboxes = await client.list_sandboxes(
            {MANAGED_BY_KEY: MANAGED_BY_LABEL, DEPLOYMENT_WORKSPACE_LABEL: workspace, DEPLOYMENT_NAME_LABEL: name}
        )
        live = [sandbox for sandbox in sandboxes if sandbox.status.state not in _DEAD_STATES]
        dead = [sandbox for sandbox in sandboxes if sandbox.status.state in _DEAD_STATES]
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

        try:
            async with self._client() as client:
                live, dead = await self._sandboxes(client, workspace, name)
                if live is not None:
                    return await self._status(client, workspace, name, live)
                # Drift recovery: a failed sandbox is replaced, not restarted.
                for sandbox in dead:
                    await client.delete_sandbox(sandbox.id)
                created = await client.sandbox_creator.create_sandbox(
                    spec=SandboxImageSpec(image=container.image),
                    entrypoint=_IDLE_ENTRYPOINT,
                    env={EXECD_ACCESS_TOKEN_ENV: self._execd_token(workspace, name)},
                    metadata=metadata,
                    timeout=None,
                    resource=dict(container.resources.limits) or dict(self._executor_config.default_resources),
                    resource_requests=dict(container.resources.requests) or None,
                    network_policy=NetworkPolicy(
                        default_action="deny", egress=[_egress_rule(egress)] if egress else []
                    ),
                    extensions={},
                    volumes=None,
                    secure_access=True,
                )
        except SandboxException as exc:
            return _failed(describe("create sandbox", exc))
        return BackendStatusUpdate(status="STARTING", status_message=f"Sandbox {created.id} created")

    async def read_status(self, *, workspace: str, name: str) -> BackendStatusUpdate:
        try:
            async with self._client() as client:
                live, dead = await self._sandboxes(client, workspace, name)
                if live is None:
                    detail = dead[0].status.message if dead else None
                    return BackendStatusUpdate(
                        status="LOST", status_message=f"Sandbox failed: {detail}" if detail else "Sandbox not found"
                    )
                return await self._status(client, workspace, name, live)
        except SandboxException as exc:
            message = describe("read sandbox", exc)
            return BackendStatusUpdate(status="UNKNOWN", status_message=message, error_details={"error": message})

    async def _status(
        self, client: OpenSandboxClient, workspace: str, name: str, sandbox: SandboxInfo
    ) -> BackendStatusUpdate:
        state = sandbox.status.state
        if state in _STARTING_STATES:
            return BackendStatusUpdate(status="STARTING", status_message=f"Sandbox is {state}")
        if state != "Running":
            return BackendStatusUpdate(status="UNKNOWN", status_message=f"Sandbox is {state}")

        metadata = sandbox.metadata or {}
        config_name = metadata.get(CONFIG_NAME_LABEL)
        if not config_name:
            return _failed("Sandbox is missing its deployment-config label")
        try:
            config = await self._load_config(workspace, config_name)
        except NemoEntityNotFoundError:
            return _failed(f"DeploymentConfig '{config_name}' not found in workspace '{workspace}'")
        container = config.containers[0]
        execd = await client.execd(sandbox.id, self._execd_token(workspace, name))

        command_id = metadata.get(SERVE_COMMAND_KEY)
        if command_id is None:
            return await self._launch(client, execd, sandbox.id, config)

        try:
            status = await execd.commands.get_command_status(command_id)
        except SandboxApiException as exc:
            if exc.status_code != 404:
                raise
            return _failed("execd no longer tracks the serve process; deployment is not serving")
        if not status.running:
            try:
                logs = await execd.commands.get_background_command_logs(command_id, cursor=0)
            except SandboxException:
                logger.warning("Failed to read the output of an exited serve process", exc_info=True)
                tail = ""
            else:
                tail = logs.content[-_LOG_TAIL_CHARS:].strip()
            message = f"Serve process exited with code {status.exit_code}; deployment is not serving"
            return BackendStatusUpdate(
                status="FAILED",
                status_message=f"{message}. Last output: {tail}" if tail else message,
                exit_code=status.exit_code,
            )

        if metadata.get(SERVE_READY_KEY) != "true":
            if not await self._ready(client, sandbox.id, container):
                return BackendStatusUpdate(status="STARTING", status_message="Serve running; awaiting readiness")
            await client.sandboxes.patch_sandbox_metadata(sandbox.id, {SERVE_READY_KEY: "true"})
        return BackendStatusUpdate(
            status="READY", status_message="Sandbox serving", endpoints=_endpoints(sandbox.id, container)
        )

    async def _launch(
        self, client: OpenSandboxClient, execd: Execd, sandbox_id: str, config: DeploymentConfig
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
            await execd.files.write_files(
                [
                    # WriteEntry takes the permission bits as their octal digits: 0o644 is 644.
                    WriteEntry(path=item.path, data=item.content.encode(), mode=int(f"{item.mode:o}"))
                    for item in resolved.config_files
                ]
            )
            execution = await execd.commands.run(
                [*container.command, *container.args], opts=RunCommandOpts(background=True, envs=envs)
            )
        except SandboxException as exc:
            message = describe("start serve command", exc)
            status_code = exc.status_code if isinstance(exc, SandboxApiException) else None
            if status_code is None or status_code >= 500:
                # execd may not be listening yet.
                return BackendStatusUpdate(status="STARTING", status_message=f"Waiting for execd: {message}")
            return _failed(message)
        if execution.id is None:
            reason = execution.error.value if execution.error else "execd returned no command id"
            return _failed(f"start serve command failed: {reason}")
        try:
            await client.sandboxes.patch_sandbox_metadata(sandbox_id, {SERVE_COMMAND_KEY: execution.id})
        except SandboxException:
            # The next poll would start a second serve process.
            await execd.commands.interrupt(execution.id)
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
        status = await client.http_get(sandbox_id, port, probe.http_get.path, timeout=probe.timeout_seconds)
        return status is not None and 200 <= status < 400

    async def delete_deployment(self, workspace: str, name: str) -> BackendStatusUpdate:
        try:
            async with self._client() as client:
                live, dead = await self._sandboxes(client, workspace, name)
                for sandbox in [*([live] if live else []), *dead]:
                    if sandbox.status.state != "Stopping":
                        await client.delete_sandbox(sandbox.id)
        except SandboxException as exc:
            return _failed(describe("delete sandbox", exc))
        return BackendStatusUpdate(status="SUCCEEDED", status_message="Sandbox deleted")

    async def list_managed_deployment_names(self) -> list[str]:
        try:
            async with self._client() as client:
                sandboxes = await client.list_sandboxes({MANAGED_BY_KEY: MANAGED_BY_LABEL})
        except SandboxException:
            logger.warning("Failed to list managed sandboxes", exc_info=True)
            return []
        names: set[str] = set()
        for sandbox in sandboxes:
            metadata = sandbox.metadata or {}
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
                command_id = (live.metadata or {}).get(SERVE_COMMAND_KEY)
                if command_id is None:
                    return LogResult(lines=["The serve command has not started"])
                execd = await client.execd(live.id, self._execd_token(workspace, name))
                logs = await execd.commands.get_background_command_logs(command_id, cursor=0)
        except SandboxException as exc:
            return LogResult(lines=[f"Failed to fetch logs: {describe('read serve logs', exc)}"])
        lines = logs.content.splitlines()
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


def _egress_rule(egress: PlatformEgressConfig) -> NetworkRule:
    """Allow rule for the platform endpoint. Only an IP target can be limited to the port."""
    try:
        ip_address(egress.host)
    except ValueError:
        return NetworkRule(action="allow", target=egress.host)
    return NetworkRule(action="allow", target=egress.host, ports=[egress.port])


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
    ports = {port.container_port for port in container.ports}
    probe = container.readiness_probe
    probe_port = _resolve_port(probe.http_get.port, container) if probe and probe.http_get else None
    if probe_port is not None:
        ports.add(probe_port)
    reserved = sorted(RESERVED_PORTS & ports)
    if reserved:
        return f"Port(s) {', '.join(map(str, reserved))} are reserved by OpenSandbox"
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
