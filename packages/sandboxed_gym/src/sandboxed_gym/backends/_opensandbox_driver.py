# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The seven sandbox operations the episode backend needs, driven straight at the OpenSandbox SDK.

Replaces ``nemo_gym.sandbox.providers.opensandbox.OpenSandboxProvider``. That class is 1541 lines
covering PTY sessions, streaming, handle serialization and connect -- of which the episode backend
used seven methods -- and depending on it meant depending on NeMo-Gym, which cannot be installed in
this workspace at all (it floors at CPython 3.13.14 and pulls ``mlflow-skinny>=3.15.1``, which
``services/unsloth`` contradicts). The SDK underneath it is a normal PyPI package with four light
dependencies, and this package already called it directly for connection config and sandbox
listing. Going straight there removes the last import of Gym from the broker path.

Handles keep the live ``Sandbox`` in ``SandboxHandle.raw``, matching what the Gym provider stored,
so egress verification can still ask the sandbox what policy it actually applied.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from collections.abc import Mapping
from datetime import timedelta
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from sandboxed_gym.backends.base import EpisodeBackendError, UnsupportedEpisodeOperationError
from sandboxed_gym.execd import EXECD_ACCESS_TOKEN_ENV, EXECD_ACCESS_TOKEN_HEADER
from sandboxed_gym.sandbox_types import (
    SandboxExecResult,
    SandboxHandle,
    SandboxSpec,
    SandboxStatus,
)

if TYPE_CHECKING:
    from opensandbox import Sandbox
    from opensandbox.models.execd import OutputMessage

LOGGER = logging.getLogger(__name__)

#: Provider name stamped onto handles.
PROVIDER_NAME = "opensandbox"

#: Correlates one create request with a resource whose response may have been lost.
_SANDBOX_CREATE_ATTEMPT_ID_METADATA_KEY = "sandbox_create_attempt_id"

#: Sandbox lifecycle names the SDK reports that do not match ours one-for-one.
_STATUS_ALIASES = {
    "creating": SandboxStatus.STARTING,
    "pending": SandboxStatus.STARTING,
    "ready": SandboxStatus.RUNNING,
    "active": SandboxStatus.RUNNING,
    "terminated": SandboxStatus.STOPPED,
    "destroyed": SandboxStatus.STOPPED,
    "failed": SandboxStatus.ERROR,
}


def _resource_requests(spec: SandboxSpec) -> dict[str, str]:
    """Map typed resources onto the SDK's Kubernetes-style request strings.

    The SDK documents the format: ``cpu`` in millicores, ``memory`` in binary suffixes, ``gpu`` as
    a device count, with the key set open-ended. ``disk_gib`` uses Kubernetes' own
    ``ephemeral-storage`` name for the same reason.
    """
    resources = spec.resources
    requests: dict[str, str] = {}
    if resources.cpu is not None:
        requests["cpu"] = f"{int(resources.cpu * 1000)}m"
    if resources.memory_mib is not None:
        requests["memory"] = f"{resources.memory_mib}Mi"
    if resources.disk_gib is not None:
        requests["ephemeral-storage"] = f"{resources.disk_gib}Gi"
    if resources.gpu:
        requests["gpu"] = str(resources.gpu)
    if resources.gpu_type is not None:
        # Refused rather than dropped: `resource_requests` has no documented key for a device
        # model, and an episode that silently lands on the wrong GPU grades under conditions it
        # did not ask for.
        raise UnsupportedEpisodeOperationError(
            "this backend cannot request a specific GPU type; the OpenSandbox resource request has no field for one"
        )
    return requests


def _exec_identity(user: str | int | None) -> dict[str, int]:
    """Turn the contract's ``user`` into the SDK's numeric ``uid``.

    A named user is refused rather than guessed: resolving a name to an id requires reading the
    image's passwd database, and running as the wrong id is exactly the silent downgrade the
    contract forbids.
    """
    if user is None:
        return {}
    if isinstance(user, int):
        return {"uid": user}
    if user.isdigit():
        return {"uid": int(user)}
    raise UnsupportedEpisodeOperationError(f"this backend can only exec as a numeric uid, not the name {user!r}")


def _resource_limits(create_options: Mapping[str, Any]) -> dict[str, str] | None:
    """Operator-configured resource *limits*, distinct from the per-episode requests.

    ``resource_requests`` is what the episode asks to be scheduled with; ``resource`` is the hard
    cap the sandbox runs under. They are separate arguments on the SDK, and omitting the second is
    not neutral: ``Sandbox.create`` substitutes ``{"cpu": "1", "memory": "2Gi"}``. A host given
    larger requests but the default cap is killed or throttled partway through an episode, which
    surfaces as an unexplained episode failure rather than as a resource error.
    """
    limits = create_options.get("resource")
    return {str(k): str(v) for k, v in limits.items()} if limits else None


def _network_policy(create_options: Mapping[str, Any]) -> Any:
    """Lift the egress policy dict `create_options_with_policy` produced into the SDK's model."""
    policy = create_options.get("network_policy")
    if policy is None:
        return None

    from opensandbox.models.sandboxes import NetworkPolicy

    if isinstance(policy, NetworkPolicy):
        return policy
    return NetworkPolicy.model_validate(policy)


def _extensions(create_options: Mapping[str, Any]) -> dict[str, str] | None:
    """Operator-configured OpenSandbox server extensions, such as ``bootstrap.execd.isolation``.

    Only the deployment's create options carry these; nothing on a ``SandboxSpec`` can, so a job or an
    episode request can't reach them. The server receives them as strings.
    """
    extensions = create_options.get("extensions")
    return {str(k): str(v) for k, v in extensions.items()} if extensions else None


def _volumes(spec: SandboxSpec) -> list[Any]:
    """Translate ``provider_options["volumes"]`` into the SDK's ``Volume`` models.

    The job host mounts the environment, workspace and dataset PVCs this way. ``Volume`` validates
    both camelCase and snake_case keys, so the mapping the host provider already builds passes
    through unchanged.
    """
    volumes = spec.provider_options.get("volumes") or []
    if not volumes:
        return []

    from opensandbox.models.sandboxes import Volume

    return [volume if isinstance(volume, Volume) else Volume.model_validate(volume) for volume in volumes]


def _joined_output(messages: list[OutputMessage] | None) -> str | None:
    """Rejoin the SDK's per-line output messages into one stream, or ``None`` if there was none.

    execd strips line terminators, so whether the output ended in a newline is not recoverable.
    """
    if not messages or not any(message.text for message in messages):
        return None
    return "\n".join(message.text.rstrip("\n") for message in messages)


#: Added to ``ready_timeout_s`` for the create request. The server answers only once the pod is
#: Running, so a pod that makes it just inside the ready wait still needs the response to arrive.
CREATE_REQUEST_HEADROOM_S = 30.0


def connection_config(
    connection: Mapping[str, Any] | None,
    *,
    execd_access_token: str | None = None,
    config_class: Any = None,
) -> Any:
    """Build an SDK ``ConnectionConfig`` from this package's own connection mapping.

    Shared so every call site -- the episode backend, its sandbox listing, and the job host
    provider -- reads the same keys the same way. Only keys that are set are forwarded, so the
    SDK's own defaults still apply to the rest.

    Args:
        connection: This package's connection mapping.
        execd_access_token: One sandbox's execd token. The SDK sends a config's headers on every call made
            with it, and a sandbox keeps the config it was created or connected with, so every execd call
            on that sandbox carries the token. Leave it out for server-only calls such as listing and killing.
        config_class: The SDK's ``ConnectionConfig``, for a caller that holds its own copy of the SDK.
    """
    if config_class is None:
        from opensandbox.config import ConnectionConfig as config_class

    connection = connection or {}
    kwargs: dict[str, Any] = {}
    for key in ("domain", "api_key", "protocol"):
        if connection.get(key) is not None:
            kwargs[key] = connection[key]
    if connection.get("request_timeout_s") is not None:
        kwargs["request_timeout"] = timedelta(seconds=connection["request_timeout_s"])
    if connection.get("use_server_proxy"):
        kwargs["use_server_proxy"] = True
    if execd_access_token is not None:
        kwargs["headers"] = {EXECD_ACCESS_TOKEN_HEADER: execd_access_token}
    return config_class(**kwargs)


def load_sdk() -> dict[str, Any]:
    """The OpenSandbox SDK classes the driver calls, imported on first use."""
    from opensandbox import Sandbox
    from opensandbox.config import ConnectionConfig
    from opensandbox.exceptions import SandboxTimeoutException

    return {
        "Sandbox": Sandbox,
        "ConnectionConfig": ConnectionConfig,
        "SandboxTimeoutException": SandboxTimeoutException,
    }


class OpenSandboxDriver:
    """Minimal OpenSandbox provider: create, exec, file I/O, status, close.

    The constructor mirrors the NeMo-Gym provider it replaces -- four option mappings -- so the
    episode backend and the job host provider construct it exactly as they constructed that.
    """

    name = PROVIDER_NAME

    def __init__(
        self,
        *,
        connection: Mapping[str, Any] | None = None,
        create: Mapping[str, Any] | None = None,
        probe: Mapping[str, Any] | None = None,
        operations: Mapping[str, Any] | None = None,
        sdk: Mapping[str, Any] | None = None,
    ) -> None:
        self._sdk = dict(sdk) if sdk is not None else None
        self._connection = dict(connection) if connection else {}
        # Listing and orphan cleanup use this one, so it carries no sandbox's execd token. Every call made through a
        # sandbox's own SDK object -- execd, its egress API, its lifecycle calls including the kill -- sends that
        # sandbox's token, as Harbor's driver does; all of those go to the sandbox or to the OpenSandbox server.
        self._connection_config = connection_config(self._connection, config_class=self.sdk["ConnectionConfig"])
        self._create_options = dict(create) if create else {}
        # `probe` configured NeMo-Gym's create-time health probe. This driver exposes the SDK's own
        # switch instead (`skip_health_check`), so the mapping is accepted and ignored rather than
        # silently changing what a caller asked for.
        self._probe = dict(probe) if probe else {}
        operations = operations or {}
        self._default_exec_timeout_s = operations.get("exec_timeout_s")
        # `workdir` is not a create-time field on this SDK -- it is a per-command option -- so it
        # is held here per sandbox and applied as the default working directory for every exec.
        # Kept off `SandboxHandle`, which mirrors NeMo-Gym's type field for field.
        self._workdirs: dict[str, str] = {}

    @property
    def sdk(self) -> dict[str, Any]:
        """The SDK classes, imported on first use."""
        if self._sdk is None:
            self._sdk = load_sdk()
        return self._sdk

    def _create_request_timeout_s(self, spec: SandboxSpec) -> float | None:
        """The create POST stays open until the pod is Running, so its timeout must cover the ready wait.

        None when the configured timeout applies as is: the operator set one, or there is no ready wait.
        """
        if self._connection.get("request_timeout_s") is not None or spec.ready_timeout_s is None:
            return None
        return spec.ready_timeout_s + CREATE_REQUEST_HEADROOM_S

    def _sandbox_connection_config(self, execd_access_token: str, request_timeout_s: float | None = None) -> Any:
        """One sandbox's connection: the configured one, plus its execd token and any longer timeout."""
        connection = dict(self._connection)
        if request_timeout_s is not None:
            connection["request_timeout_s"] = request_timeout_s
        return connection_config(
            connection, execd_access_token=execd_access_token, config_class=self.sdk["ConnectionConfig"]
        )

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        """Create a sandbox whose execd answers only this driver.

        Each call generates a fresh execd token, set as ``EXECD_ACCESS_TOKEN`` in the sandbox's environment
        (overriding any value in ``spec.env``) and held only by the returned SDK object, which sends it on
        every call. It never reaches the spec, the handle's metadata or the logs.
        """
        Sandbox = self.sdk["Sandbox"]
        SandboxTimeoutException = self.sdk["SandboxTimeoutException"]

        if spec.image is None:
            raise EpisodeBackendError("an episode image is required")

        create_attempt_id = uuid4().hex
        metadata = {**spec.metadata, _SANDBOX_CREATE_ATTEMPT_ID_METADATA_KEY: create_attempt_id}
        execd_access_token = secrets.token_urlsafe(32)
        create_request_timeout_s = self._create_request_timeout_s(spec)
        create_connection_config = self._sandbox_connection_config(execd_access_token, create_request_timeout_s)
        try:
            sandbox = await Sandbox.create(
                spec.image,
                timeout=timedelta(seconds=spec.ttl_s) if spec.ttl_s is not None else None,
                ready_timeout=timedelta(seconds=spec.ready_timeout_s or 30.0),
                env={**spec.env, EXECD_ACCESS_TOKEN_ENV: execd_access_token},
                metadata=metadata,
                resource=_resource_limits(self._create_options),
                resource_requests=_resource_requests(spec) or None,
                network_policy=_network_policy(self._create_options),
                extensions=_extensions(self._create_options),
                entrypoint=list(spec.entrypoint) if spec.entrypoint else None,
                volumes=_volumes(spec) or None,
                # Set only by the job-host provider; the broker refuses provider_options on episode requests.
                extensions=dict(spec.provider_options.get("extensions") or {}) or None,
                # Large runtime images flake on the SDK's create-time probe, and the job host polls
                # `/health` itself once routes resolve. The caller decides; the SDK default stands.
                skip_health_check=bool(self._create_options.get("skip_health_check", False)),
                connection_config=create_connection_config,
            )
            sandbox_id = getattr(sandbox, "id", None) or (await sandbox.get_info()).id
            if create_request_timeout_s is not None:
                sandbox = await self._reattach_under_base_connection(
                    sandbox, sandbox_id, self._sandbox_connection_config(execd_access_token)
                )
            if spec.workdir:
                self._workdirs[sandbox_id] = spec.workdir
            return SandboxHandle(sandbox_id=sandbox_id, provider_name=self.name, raw=sandbox)
        except BaseException as exc:
            try:
                removed = await asyncio.wait_for(
                    self.destroy_sandboxes_matching({_SANDBOX_CREATE_ATTEMPT_ID_METADATA_KEY: create_attempt_id}),
                    timeout=30,
                )
                if removed:
                    LOGGER.warning(
                        "destroyed sandboxes after create attempt %s failed: %s",
                        create_attempt_id,
                        ", ".join(removed),
                    )
            except BaseException:
                # Preserve the create failure: it is the actionable cause.
                LOGGER.exception("failed to reconcile sandbox create attempt %s", create_attempt_id)
            if isinstance(exc, SandboxTimeoutException):
                request_timeout_s = create_connection_config.request_timeout.total_seconds()
                raise EpisodeBackendError(
                    f"OpenSandbox did not answer the create request within the SDK request timeout of "
                    f"{request_timeout_s:g}s; the server may still have been creating the sandbox (image pull, VM "
                    f"boot). Set connection.request_timeout_s to at least the server's sandbox_create_timeout_seconds."
                ) from exc
            raise

    async def _reattach_under_base_connection(
        self, created: Sandbox, sandbox_id: str, sandbox_connection: Any
    ) -> Sandbox:
        """Swap the create-time connection for the configured one, keeping the sandbox's execd token."""
        try:
            reattached = await self.sdk["Sandbox"].connect(
                sandbox_id, connection_config=sandbox_connection, skip_health_check=True
            )
        except Exception:
            LOGGER.warning(
                "sandbox %s keeps its create-time request timeout: reattaching under the configured connection failed",
                sandbox_id,
                exc_info=True,
            )
            return created
        await created.close()
        return reattached

    def _sandbox(self, handle: SandboxHandle) -> Sandbox:
        return handle.raw  # ty: ignore[invalid-return-type] - provider-owned opaque state

    async def exec(
        self,
        handle: SandboxHandle,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: float | None = None,
        user: str | int | None = None,
    ) -> SandboxExecResult:
        from opensandbox.models.execd import RunCommandOpts

        effective_timeout = timeout_s if timeout_s is not None else self._default_exec_timeout_s
        working_directory = cwd or self._workdirs.get(handle.sandbox_id)
        execution = await self._sandbox(handle).commands.run(
            command,
            opts=RunCommandOpts(
                working_directory=working_directory,
                timeout=timedelta(seconds=effective_timeout) if effective_timeout is not None else None,
                envs=dict(env) if env else None,
                **_exec_identity(user),
            ),
        )
        error = getattr(execution, "error", None)
        return SandboxExecResult(
            stdout=_joined_output(getattr(execution.logs, "stdout", None)),
            stderr=_joined_output(getattr(execution.logs, "stderr", None)),
            # A command the sandbox never ran reports no exit code; the contract's callers treat a
            # non-zero code as "the command failed", so an absent one must not read as success.
            return_code=execution.exit_code if execution.exit_code is not None else 1,
            error_type=type(error).__name__ if error is not None else None,
        )

    async def upload_file(self, handle: SandboxHandle, source_path: Any, target_path: str) -> None:
        from pathlib import Path

        await self._sandbox(handle).files.write_file(target_path, Path(source_path).read_bytes())

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Any) -> None:
        from pathlib import Path

        content = await self._sandbox(handle).files.read_bytes(source_path)
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        try:
            info = await self._sandbox(handle).get_info()
        except Exception:
            # A sandbox the control plane no longer knows about is gone, which the contract models
            # as a status rather than an error.
            return SandboxStatus.UNKNOWN
        reported = info.status.state.lower()
        if reported in _STATUS_ALIASES:
            return _STATUS_ALIASES[reported]
        try:
            return SandboxStatus(reported)
        except ValueError:
            return SandboxStatus.UNKNOWN

    async def close(self, handle: SandboxHandle) -> None:
        self._workdirs.pop(handle.sandbox_id, None)
        await self._sandbox(handle).destroy()

    async def destroy_sandboxes_matching(self, metadata: Mapping[str, str]) -> tuple[str, ...]:
        """Destroy every sandbox matching ``metadata``.

        OpenSandbox creates the Kubernetes resource before waiting for it to become ready. If the
        client loses that request, no ``SandboxHandle`` comes back, so normal handle-based cleanup
        cannot reach the resource. A unique create-attempt ID makes metadata lookup an exact rollback.
        """
        if not metadata:
            raise ValueError("sandbox cleanup requires at least one metadata selector")

        from opensandbox.manager import SandboxManager
        from opensandbox.models.sandboxes import SandboxFilter

        manager = await SandboxManager.create(connection_config=self._connection_config)
        try:
            sandbox_ids: list[str] = []
            page = 1

            while True:
                result = await manager.list_sandbox_infos(SandboxFilter(metadata=dict(metadata), page=page))
                sandbox_ids.extend(info.id for info in result.sandbox_infos)
                if not result.pagination.has_next_page:
                    break
                page += 1

            killed: list[str] = []
            for sandbox_id in sandbox_ids:
                try:
                    await manager.kill_sandbox(sandbox_id)
                except Exception:
                    # One sandbox the control plane already dropped must not spare the rest.
                    LOGGER.exception("failed to destroy sandbox %s", sandbox_id)
                    continue
                killed.append(sandbox_id)

            return tuple(killed)
        finally:
            await manager.close()

    async def aclose(self) -> None:
        """No provider-scoped client to close: each sandbox owns its own SDK connection."""
        return
