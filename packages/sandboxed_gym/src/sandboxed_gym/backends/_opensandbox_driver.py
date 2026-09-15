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

import logging
from collections.abc import Mapping
from datetime import timedelta
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from sandboxed_gym.backends.base import EpisodeBackendError, UnsupportedEpisodeOperationError
from sandboxed_gym.sandbox_types import (
    SandboxExecResult,
    SandboxHandle,
    SandboxSpec,
    SandboxStatus,
)

if TYPE_CHECKING:
    from opensandbox import Sandbox

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


def _network_policy(create_options: Mapping[str, Any]) -> Any:
    """Lift the egress policy dict `create_options_with_policy` produced into the SDK's model."""
    from opensandbox.models.sandboxes import NetworkPolicy

    policy = create_options.get("network_policy")
    if policy is None or isinstance(policy, NetworkPolicy):
        return policy
    return NetworkPolicy.model_validate(policy)


def _volumes(spec: SandboxSpec) -> list[Any]:
    """Translate ``provider_options["volumes"]`` into the SDK's ``Volume`` models.

    The job host mounts the environment, workspace and dataset PVCs this way. ``Volume`` validates
    both camelCase and snake_case keys, so the mapping the host provider already builds passes
    through unchanged.
    """
    from opensandbox.models.sandboxes import Volume

    volumes = spec.provider_options.get("volumes") or []
    return [volume if isinstance(volume, Volume) else Volume.model_validate(volume) for volume in volumes]


def _joined_output(messages: Any) -> str | None:
    """Concatenate the SDK's per-line output messages into one stream, or ``None`` if empty."""
    if not messages:
        return None
    parts = [str(getattr(message, "content", message) or "") for message in messages]
    joined = "".join(parts)
    return joined or None


def connection_config(connection: Mapping[str, Any] | None) -> Any:
    """Build an SDK ``ConnectionConfig`` from this package's own connection mapping.

    Shared so every call site -- the episode backend, its sandbox listing, and the job host
    provider -- reads the same keys the same way. Only keys that are set are forwarded, so the
    SDK's own defaults still apply to the rest.
    """
    from opensandbox.config import ConnectionConfig

    connection = connection or {}
    kwargs: dict[str, Any] = {}
    for key in ("domain", "api_key", "protocol"):
        if connection.get(key) is not None:
            kwargs[key] = connection[key]
    if connection.get("request_timeout_s") is not None:
        kwargs["request_timeout"] = timedelta(seconds=connection["request_timeout_s"])
    if connection.get("use_server_proxy"):
        kwargs["use_server_proxy"] = True
    return ConnectionConfig(**kwargs)


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
    ) -> None:
        self._connection = dict(connection) if connection else {}
        self._connection_config = connection_config(self._connection)
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

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        from opensandbox import Sandbox

        if spec.image is None:
            raise EpisodeBackendError("an episode image is required")

        create_attempt_id = uuid4().hex
        metadata = {**spec.metadata, _SANDBOX_CREATE_ATTEMPT_ID_METADATA_KEY: create_attempt_id}
        try:
            sandbox = await Sandbox.create(
                spec.image,
                timeout=timedelta(seconds=spec.ttl_s) if spec.ttl_s is not None else None,
                ready_timeout=timedelta(seconds=spec.ready_timeout_s or 30.0),
                env=dict(spec.env) or None,
                metadata=metadata,
                resource_requests=_resource_requests(spec) or None,
                network_policy=_network_policy(self._create_options),
                entrypoint=list(spec.entrypoint) if spec.entrypoint else None,
                volumes=_volumes(spec) or None,
                # Large runtime images flake on the SDK's create-time probe, and the job host polls
                # `/health` itself once routes resolve. The caller decides; the SDK default stands.
                skip_health_check=bool(self._create_options.get("skip_health_check", False)),
                connection_config=self._connection_config,
            )
            sandbox_id = getattr(sandbox, "sandbox_id", None) or (await sandbox.get_info()).id
            if spec.workdir:
                self._workdirs[sandbox_id] = spec.workdir
            return SandboxHandle(sandbox_id=sandbox_id, provider_name=self.name, raw=sandbox)
        except BaseException:
            try:
                removed = await self.destroy_sandboxes_matching(
                    {_SANDBOX_CREATE_ATTEMPT_ID_METADATA_KEY: create_attempt_id}
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
            raise

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
        reported = str(getattr(info.status, "value", info.status)).lower()
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

            for sandbox_id in sandbox_ids:
                await manager.kill_sandbox(sandbox_id)

            return tuple(sandbox_ids)
        finally:
            await manager.close()

    async def aclose(self) -> None:
        """No provider-scoped client to close: each sandbox owns its own SDK connection."""
        return
