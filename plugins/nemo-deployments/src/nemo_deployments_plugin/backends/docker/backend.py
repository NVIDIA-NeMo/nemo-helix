# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Docker substrate backend for the deployments plugin."""

from __future__ import annotations

import asyncio
import io
import logging
import os
import tarfile
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from nemo_deployments_plugin.auth_proxy import is_auth_proxy_container
from nemo_deployments_plugin.backends.base import (
    BackendStatusUpdate,
    DeploymentBackend,
    LogResult,
    MissingBackendDependencyError,
    VolumeStatusUpdate,
)
from nemo_deployments_plugin.backends.docker import volumes as volume_ops
from nemo_deployments_plugin.backends.docker.config import DockerExecutorConfig
from nemo_deployments_plugin.backends.docker.containers import (
    DeploymentConfigError,
    build_docker_plan,
    build_port_bindings,
    build_volume_bindings,
    device_requests_for_gpus,
    env_dict,
    gpu_count_from_container,
    merged_volume_mounts,
    parse_docker_backend_config,
    restart_policy_kwargs,
)
from nemo_deployments_plugin.backends.docker.gpu import (
    GPUAllocationError,
    get_shared_gpu_pool,
)
from nemo_deployments_plugin.backends.docker.ports import (
    PortEnumerationError,
    find_available_port,
)
from nemo_deployments_plugin.backends.docker.probes import (
    check_readiness_probe,
    host_url_for_port,
)
from nemo_deployments_plugin.backends.docker.status import (
    LOG_MAX_CHARS,
    map_docker_state_to_starting,
    map_exited_status,
    missing_container_status,
)
from nemo_deployments_plugin.backends.labels import (
    BACKOFF_LIMIT_LABEL,
    CONFIG_NAME_LABEL,
    CONTAINER_ROLE_LABEL,
    CONTAINER_ROLE_SERVER,
    DEPLOYMENT_NAME_LABEL,
    DEPLOYMENT_WORKSPACE_LABEL,
    MANAGED_BY_KEY,
    RESOURCE_SCOPE_LABEL,
    RESTART_POLICY_LABEL,
    companion_container_name,
    container_name,
    deployment_identity_labels,
    deployment_key,
    managed_by_filter,
)
from nemo_deployments_plugin.backends.workload_identity import (
    workload_delegation_scope,
    workload_identity_activation_error,
    workload_identity_requested,
)
from nemo_deployments_plugin.constants import MANAGED_BY_LABEL
from nemo_deployments_plugin.entities import (
    ConfigFile,
    Container,
    Deployment,
    DeploymentConfig,
)
from nemo_deployments_plugin.secrets import (
    SecretResolutionError,
    resolve_deployment_config_secrets,
)
from nemo_deployments_plugin.types import NON_TERMINAL_DEPLOYMENT_STATUSES, Endpoint, RestartPolicy
from nemo_helix_plugin.auth import AuthContext
from nemo_helix_plugin.auth.workload_delegations import (
    WorkloadDelegationConflictError,
    WorkloadDelegationEntity,
    WorkloadDelegationStore,
    as_aware_utc,
)
from nemo_helix_plugin.auth.workload_identity import (
    WORKLOAD_IDENTITY_TOKEN_FILE_PATH,
    WORKLOAD_IDENTITY_VOLUME_PATH,
    build_docker_opaque_workload_delegation,
    build_token_archive,
    get_workload_delegation_audience,
    workload_identity_env,
)
from nemo_helix_plugin.capabilities import docker_from_env_kwargs, probe_docker
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.config import LOOPBACK_ADDRESSES
from nemo_helix_plugin.entities.client import AsyncEntitiesClient
from nemo_helix_plugin.entity_client import (
    NemoEntitiesClient,
    NemoEntityNotFoundError,
)
from nemo_helix_plugin.k8s_naming import k8s_safe_name
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import ReadTimeout
from requests.exceptions import Timeout as RequestsTimeout
from urllib3.exceptions import ReadTimeoutError as Urllib3ReadTimeoutError

if TYPE_CHECKING:
    from docker.models.containers import Container as DockerContainer
    from docker.models.volumes import Volume as DockerVolume

    import docker

logger = logging.getLogger(__name__)


_ONE_SHOT_RESTART_POLICIES = frozenset({"Never", "OnFailure"})
_EXITED_CONTAINER_STATES = frozenset({"exited", "dead"})
# Docker rejects a publish with this message when its own allocator already holds
# the port. Its reservations are invisible to a host-socket probe, so a container
# created concurrently can claim a port between allocation and run.
_PORT_CONFLICT_MARKER = "port is already allocated"
_PORT_CONFLICT_ATTEMPTS = 3
NGC_IMAGE_REGISTRY = os.getenv("NGC_IMAGE_REGISTRY", "nvcr.io")
NGC_IMAGE_REGISTRY_USER_NAME = os.getenv("NGC_IMAGE_REGISTRY_USER_NAME", "$oauthtoken")
DOCKER_WORKLOAD_IDENTITY_TOKEN_FILE_LABEL = "nemo.nvidia.com/workload-identity-token-file"
DOCKER_WORKLOAD_IDENTITY_VOLUME_LABEL = "nemo.nvidia.com/workload-identity-volume"


def _is_ngc_image(image: str) -> bool:
    """Return whether an image belongs to the configured NGC registry."""
    return image == NGC_IMAGE_REGISTRY or image.startswith(f"{NGC_IMAGE_REGISTRY}/")


def _config_files_tar(config_files: list[ConfigFile]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        seen_dirs: set[str] = set()
        for cf in config_files:
            rel = cf.path.lstrip("/")
            parts = rel.split("/")
            for i in range(1, len(parts)):
                d = "/".join(parts[:i])
                if d in seen_dirs:
                    continue
                seen_dirs.add(d)
                info = tarfile.TarInfo(name=d)
                info.type = tarfile.DIRTYPE
                if d == "tmp":
                    info.mode = 0o1777
                elif parts[0] == "tmp" and i == 2:
                    # Configs under /tmp need one writable deployment-owned root
                    # so hardened, non-root runtimes can create sibling workspace
                    # and artifact directories beside the delivered config.
                    info.mode = 0o777
                else:
                    info.mode = 0o755
                tar.addfile(info)
            data = cf.content.encode("utf-8")
            info = tarfile.TarInfo(name=rel)
            info.size = len(data)
            info.mode = cf.mode
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _docker_inspect_attrs(container: DockerContainer) -> dict[str, Any]:
    return container.attrs or {}


def _docker_inspect_exit_code(container: DockerContainer) -> int:
    state = _docker_inspect_attrs(container).get("State") or {}
    if not isinstance(state, dict):
        return 1
    return int(state.get("ExitCode", 1))


def _docker_inspect_restart_count(container: DockerContainer) -> int:
    return int(_docker_inspect_attrs(container).get("RestartCount", 0))


class DockerDeploymentBackend(DeploymentBackend):
    """Manage deployments and volumes as Docker containers and volumes."""

    _client: docker.DockerClient

    def init(self) -> None:
        try:
            from docker.errors import DockerException

            import docker
            from docker import errors as docker_errors
        except ImportError as exc:
            raise MissingBackendDependencyError(
                "docker package is required for DockerDeploymentBackend. "
                "Install with: uv sync --package nemo-deployments-plugin --extra docker"
            ) from exc

        self._docker = docker
        self._docker_errors = docker_errors
        self._executor_config = DockerExecutorConfig.model_validate(self._config)
        self._entities = NemoEntitiesClient(client_from_platform(self._sdk, AsyncEntitiesClient))
        self._workload_delegations = WorkloadDelegationStore(self._entities)
        self._gpu_pool = get_shared_gpu_pool()
        docker_host = self._executor_config.docker_host
        probe = probe_docker(docker_host=docker_host)
        if not probe.available:
            detail = probe.detail or "Docker daemon unreachable"
            raise MissingBackendDependencyError(
                f"Docker daemon is unavailable ({detail}). Docker-backed deployments will be disabled."
            )
        try:
            self._client = self._create_client()
        except (
            DockerException,
            RequestsConnectionError,
            RequestsTimeout,
            OSError,
        ) as exc:
            raise MissingBackendDependencyError(
                f"Docker daemon is unavailable ({exc}). Docker-backed deployments will be disabled."
            ) from exc

    def _create_client(self) -> docker.DockerClient:
        # docker-py 7.x rejects base_url= on from_env; override DOCKER_HOST instead
        # so TLS env vars (DOCKER_TLS_VERIFY, cert paths) still apply.
        client = self._docker.from_env(
            **docker_from_env_kwargs(
                timeout=self._executor_config.docker_timeout,
                docker_host=self._executor_config.docker_host,
            )
        )
        client.api.timeout = self._executor_config.docker_timeout
        client.ping()
        return client

    def shutdown(self) -> None:
        if hasattr(self, "_client") and self._client is not None:
            self._client.close()

    async def _load_deployment_config(self, workspace: str, config_name: str) -> DeploymentConfig:
        return await self._entities.get(DeploymentConfig, config_name, workspace=workspace)

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
        c_name = container_name(workspace, name)
        recreate_exited_one_shot = False
        try:
            existing = await asyncio.to_thread(self._client.containers.get, c_name)
            if self._container_matches_deployment(existing, workspace, name, config_name):
                restart_policy = (existing.labels or {}).get(RESTART_POLICY_LABEL, "Always")
                if restart_policy in _ONE_SHOT_RESTART_POLICIES and existing.status in _EXITED_CONTAINER_STATES:
                    logger.info(
                        "Removing exited one-shot container before recreate",
                        extra={
                            "container": c_name,
                            "restart_policy": restart_policy,
                            "status": existing.status,
                        },
                    )
                    try:
                        await asyncio.to_thread(existing.remove, force=True)
                    except self._docker_errors.APIError as exc:
                        return BackendStatusUpdate(
                            status="FAILED",
                            status_message=f"Failed to remove exited container before recreate: {exc}",
                        )
                    recreate_exited_one_shot = True
                else:
                    return await self.read_status(workspace=workspace, name=name)
            else:
                return BackendStatusUpdate(
                    status="FAILED",
                    status_message=f"Container name collision: {c_name} exists with different labels",
                )
        except self._docker_errors.NotFound:
            pass

        try:
            config = await self._load_deployment_config(workspace, config_name)
            config = await resolve_deployment_config_secrets(self._sdk, config)
            plan = build_docker_plan(config)
        except DeploymentConfigError as exc:
            return BackendStatusUpdate(status="FAILED", status_message=str(exc))
        except SecretResolutionError as exc:
            return BackendStatusUpdate(status="FAILED", status_message=str(exc))
        except NemoEntityNotFoundError:
            return BackendStatusUpdate(
                status="FAILED",
                status_message=f"DeploymentConfig '{config_name}' not found in workspace '{workspace}'",
            )
        except Exception as exc:
            logger.exception("Failed to load deployment config %s/%s", workspace, config_name)
            return BackendStatusUpdate(
                status="FAILED",
                status_message=f"Failed to load deployment config: {exc}",
            )

        if recreate_exited_one_shot:
            try:
                await self._revoke_workload_delegations_for_deployment(workspace=workspace, name=name)
                await self._cleanup_workload_identity_volumes(workspace=workspace, name=name)
            except Exception as exc:
                logger.exception("Failed to clean up workload identity before recreating %s/%s", workspace, name)
                return BackendStatusUpdate(
                    status="FAILED",
                    status_message=f"Failed to clean up workload identity before recreate: {exc}",
                )

        container_spec = plan.primary
        docker_cfg = parse_docker_backend_config(backend_config)
        if config.backend_config.docker is not None:
            docker_cfg = config.backend_config.docker
        if docker_cfg.network is None and self._executor_config.network is not None:
            docker_cfg = docker_cfg.model_copy(update={"network": self._executor_config.network})

        dep_key = deployment_key(workspace, name)
        gpu_pool = self._gpu_pool
        gpu_ids: list[int] = []
        gpu_count = gpu_count_from_container(container_spec)
        if gpu_count > 0:
            if gpu_pool is None:
                return BackendStatusUpdate(
                    status="FAILED",
                    status_message="GPU requested but no GPUs detected on this host",
                )
            try:
                gpu_ids = gpu_pool.allocate_gpu(dep_key, num_requested=gpu_count)
            except GPUAllocationError as exc:
                return BackendStatusUpdate(status="FAILED", status_message=str(exc))

        try:
            host_ports = await self._allocate_host_ports(container_spec)
        except PortEnumerationError as exc:
            # Could not determine what is in use — report that, rather than blaming the port range.
            if gpu_ids and gpu_pool is not None:
                gpu_pool.release_gpu(dep_key)
            return BackendStatusUpdate(
                status="FAILED",
                status_message=f"Could not determine host ports in use: {exc}",
            )
        if host_ports is None:
            if gpu_ids and gpu_pool is not None:
                gpu_pool.release_gpu(dep_key)
            return BackendStatusUpdate(
                status="FAILED",
                status_message="No host ports available in configured range",
            )

        # Pull all images in the group (init + primary + sidecars) up front.
        if self._executor_config.pull_images:
            for container in [*plan.init_containers, container_spec, *plan.sidecars]:
                pull_error = await self._pull_image(
                    container.image,
                    ngc_api_key=env_dict(container).get("NGC_API_KEY"),
                )
                if pull_error is not None:
                    # A pull failure is only fatal if the image is not already
                    # present locally. Locally-built/loaded images (e.g. the LoRA
                    # adapters sidecar `nhx-api:local`) are not pullable from a
                    # registry, so fall back to the local copy when it exists.
                    # Only a genuine "not found locally" is fatal here; other
                    # docker client errors from images.get propagate rather than
                    # being masked as "image present".
                    try:
                        await asyncio.to_thread(self._client.images.get, container.image)
                        logger.info(
                            "Image %s not pullable but present locally; using local copy",
                            container.image,
                        )
                    except (
                        self._docker_errors.ImageNotFound,
                        self._docker_errors.NotFound,
                    ):
                        if gpu_ids and gpu_pool is not None:
                            gpu_pool.release_gpu(dep_key)
                        return BackendStatusUpdate(status="FAILED", status_message=pull_error)

        base_labels = {
            **labels,
            **config.labels,
            **deployment_identity_labels(
                workspace,
                name,
                config.restart_policy,
                config_name=config_name,
                backoff_limit=config.backoff_limit,
                resource_scope=self._executor_config.resource_scope,
            ),
        }
        identity_error = workload_identity_activation_error(config=config, auth_context=auth_context)
        if identity_error is not None:
            if gpu_ids and gpu_pool is not None:
                gpu_pool.release_gpu(dep_key)
            return BackendStatusUpdate(status="FAILED", status_message=identity_error)

        # 1) Init containers run to completion, in order, before the group starts.
        for init in plan.init_containers:
            init_status = await self._run_init_container(
                workspace=workspace,
                name=name,
                config=config,
                init=init,
                base_labels=base_labels,
                dep_key=dep_key,
                gpu_ids=gpu_ids,
                auth_context=auth_context,
            )
            if init_status is not None:
                return init_status

        # 2) Primary (server) container: publishes ports, owns GPUs.
        primary_identity: tuple[str, str] | None = None
        primary_labels = {**base_labels, CONTAINER_ROLE_LABEL: CONTAINER_ROLE_SERVER}
        try:
            primary_identity = await self._prepare_workload_identity_for_container(
                workspace=workspace,
                deployment_name=name,
                config=config,
                role=CONTAINER_ROLE_SERVER,
                base_labels=primary_labels,
                auth_context=auth_context,
            )
        except Exception as exc:
            if gpu_ids and gpu_pool is not None:
                gpu_pool.release_gpu(dep_key)
            return BackendStatusUpdate(status="FAILED", status_message=f"Failed to prepare workload identity: {exc}")
        if primary_identity is not None:
            primary_labels = self._workload_identity_volume_labels(
                base_labels=base_labels,
                role=CONTAINER_ROLE_SERVER,
                volume_name=primary_identity[0],
            )
        server_container, host_ports, run_error = await self._run_server_container(
            workspace=workspace,
            config=config,
            container_spec=container_spec,
            name=c_name,
            labels=primary_labels,
            host_ports=host_ports,
            gpu_ids=gpu_ids,
            network=docker_cfg.network,
            workload_identity_volume_name=primary_identity[0] if primary_identity is not None else None,
        )
        if server_container is None:
            await self._cleanup_prepared_workload_identity(primary_identity, reason="server start failure")
            if gpu_ids and gpu_pool is not None:
                gpu_pool.release_gpu(dep_key)
            return BackendStatusUpdate(status="FAILED", status_message=run_error)

        # 3) Sidecars share the primary's network namespace + volumes; no ports/GPU.
        #
        # Known limitation: a sidecar joins ``network=container:<server>``, so it
        # is tied to *this* server container instance. Docker's Always restart
        # policy restarts the same server container in place (netns preserved), so
        # ordinary server crashes are fine. But if the server container is ever
        # replaced by a *new* container (a fresh create rather than an in-place
        # restart), the sidecar's netns reference dangles and nothing re-runs the
        # sidecars. Today a full recreate goes through delete_model_deployment
        # (which tears down the whole group) before create, so the sidecars are
        # re-run together; a future partial/primary-only recreate would need to
        # recreate the whole group to stay consistent.
        for sidecar in plan.sidecars:
            sidecar_name = companion_container_name(workspace, name, sidecar.name)
            sidecar_identity: tuple[str, str] | None = None
            sidecar_labels = {**base_labels, CONTAINER_ROLE_LABEL: sidecar.name}
            try:
                if not is_auth_proxy_container(sidecar) or workload_identity_requested(config):
                    sidecar_identity = await self._prepare_workload_identity_for_container(
                        workspace=workspace,
                        deployment_name=name,
                        config=config,
                        role=sidecar.name,
                        base_labels=sidecar_labels,
                        auth_context=auth_context,
                    )
                if sidecar_identity is not None:
                    sidecar_labels = self._workload_identity_volume_labels(
                        base_labels=base_labels,
                        role=sidecar.name,
                        volume_name=sidecar_identity[0],
                    )
                sidecar_create_kwargs = self._build_run_kwargs(
                    workspace=workspace,
                    config=config,
                    container=sidecar,
                    name=sidecar_name,
                    labels=sidecar_labels,
                    host_ports={},
                    gpu_ids=[],
                    network=f"container:{c_name}",
                    workload_identity_volume_name=sidecar_identity[0] if sidecar_identity is not None else None,
                )
                sidecar_container = await asyncio.to_thread(self._client.containers.create, **sidecar_create_kwargs)
                await asyncio.to_thread(sidecar_container.start)
            except Exception as exc:
                logger.exception("Failed to start sidecar container %s", sidecar_name)
                await self._cleanup_prepared_workload_identity(sidecar_identity, reason="sidecar start failure")
                # Tear the whole group down so we don't leave a half-started deployment.
                await self.delete_deployment(workspace, name)
                return BackendStatusUpdate(
                    status="FAILED",
                    status_message=f"Failed to start sidecar {sidecar.name}: {exc}",
                )

        endpoints = self._build_endpoints(container_spec, host_ports, target_name=c_name)
        if config.restart_policy in _ONE_SHOT_RESTART_POLICIES:
            return await self._observe_one_shot_primary_after_create(
                workspace=workspace,
                name=name,
                config=config,
                container=server_container,
                dep_key=dep_key,
                endpoints=endpoints,
            )
        return BackendStatusUpdate(
            status="STARTING",
            status_message=f"Container {c_name} created",
            endpoints=endpoints,
        )

    async def _allocate_host_ports(
        self,
        container_spec: Container,
        *,
        exclude_ports: set[int] | None = None,
    ) -> dict[int, int] | None:
        """Map each published container port to a free host port.

        Returns None when the configured range holds no more free ports. Propagates
        :class:`PortEnumerationError` when the ports in use could not be determined at all — that is
        a different (usually transient) condition and must not be reported as an exhausted range.
        """
        excluded = set(exclude_ports or ())
        host_ports: dict[int, int] = {}
        for port_spec in container_spec.ports:
            host_port = await find_available_port(
                self._client,
                self._executor_config.port_range_start,
                self._executor_config.port_range_end,
                exclude_ports=excluded | set(host_ports.values()),
            )
            if host_port is None:
                return None
            host_ports[port_spec.container_port] = host_port
        return host_ports

    def _workload_identity_volume_name(self, *, workspace: str, name: str, role: str) -> str:
        return k8s_safe_name(
            f"dep-wi-{workspace}-{name}-{role}",
            hash_input=f"{deployment_key(workspace, name)}/{role}/workload-identity",
        )

    def _workload_identity_volume_labels(
        self,
        *,
        base_labels: dict[str, str],
        role: str,
        volume_name: str,
    ) -> dict[str, str]:
        return {
            **base_labels,
            CONTAINER_ROLE_LABEL: role,
            DOCKER_WORKLOAD_IDENTITY_TOKEN_FILE_LABEL: WORKLOAD_IDENTITY_TOKEN_FILE_PATH,
            DOCKER_WORKLOAD_IDENTITY_VOLUME_LABEL: volume_name,
        }

    def _volume_labels(self, volume: DockerVolume) -> dict[str, str]:
        attrs = volume.attrs
        if not isinstance(attrs, Mapping):
            return {}
        labels = attrs.get("Labels")
        if not isinstance(labels, Mapping):
            return {}
        return {key: value for key, value in labels.items() if isinstance(key, str) and isinstance(value, str)}

    def _volume_matches_deployment_group(self, volume: DockerVolume, workspace: str, name: str) -> bool:
        labels = self._volume_labels(volume)
        return (
            labels.get(DEPLOYMENT_WORKSPACE_LABEL) == workspace
            and labels.get(DEPLOYMENT_NAME_LABEL) == name
            and labels.get(MANAGED_BY_KEY) == MANAGED_BY_LABEL
            and self._labels_match_resource_scope(labels)
        )

    async def _ensure_workload_identity_volume(self, *, volume_name: str, labels: dict[str, str]) -> None:
        def _ensure() -> None:
            try:
                volume = self._client.volumes.get(volume_name)
                volume.reload()
                existing_labels = self._volume_labels(volume)
                if not all(existing_labels.get(key) == value for key, value in labels.items()):
                    logger.info("Recreating workload identity volume %s with updated labels", volume_name)
                    volume.remove(force=True)
                    self._client.volumes.create(name=volume_name, labels=labels)
            except self._docker_errors.NotFound:
                self._client.volumes.create(name=volume_name, labels=labels)

        await asyncio.to_thread(_ensure)

    async def _remove_workload_identity_volume(self, volume_name: str) -> None:
        def _remove() -> None:
            try:
                volume = self._client.volumes.get(volume_name)
                volume.remove(force=True)
            except self._docker_errors.NotFound:
                return

        try:
            await asyncio.to_thread(_remove)
        except Exception:
            logger.warning("Failed to remove workload identity volume %s", volume_name, exc_info=True)

    async def _cleanup_workload_identity_volumes(self, *, workspace: str, name: str) -> None:
        def _delete() -> None:
            try:
                volumes = self._client.volumes.list(
                    filters={
                        "label": [
                            f"{MANAGED_BY_KEY}={MANAGED_BY_LABEL}",
                            f"{DEPLOYMENT_WORKSPACE_LABEL}={workspace}",
                            f"{DEPLOYMENT_NAME_LABEL}={name}",
                            *self._resource_scope_filter_label(),
                        ]
                    }
                )
            except Exception:
                logger.warning("Failed to list workload identity volumes for %s/%s", workspace, name, exc_info=True)
                return
            for volume in volumes:
                if not self._volume_matches_deployment_group(volume, workspace, name):
                    continue
                try:
                    volume.remove(force=True)
                except self._docker_errors.NotFound:
                    continue
                except Exception:
                    logger.warning(
                        "Failed to remove workload identity volume %s",
                        volume.name,
                        exc_info=True,
                    )

        await asyncio.to_thread(_delete)

    async def _try_revoke_workload_delegation(self, delegation_name: str | None, *, reason: str) -> None:
        if not delegation_name:
            return
        try:
            await self._workload_delegations.revoke(delegation_name)
        except Exception:
            logger.warning(
                "Failed to revoke Docker workload delegation",
                extra={"delegation_name": delegation_name, "reason": reason},
                exc_info=True,
            )

    async def _revoke_workload_delegations_for_deployment(
        self,
        *,
        workspace: str,
        name: str,
    ) -> None:
        try:
            await self._workload_delegations.revoke_by_workload(
                workload_delegation_scope(workspace=workspace, deployment_name=name, config=None)
            )
        except Exception:
            logger.warning("Failed to revoke Docker workload delegations for %s/%s", workspace, name, exc_info=True)

    async def _refresh_workload_delegations_for_config(
        self,
        *,
        workspace: str,
        name: str,
        config: DeploymentConfig | None,
    ) -> None:
        if config is None or config.workload_identity is None or not workload_identity_requested(config):
            return
        now = datetime.now(UTC)
        refresh_threshold_seconds = config.workload_identity.token_expiration_seconds / 2
        try:
            delegations = await self._workload_delegations.list_by_workload(
                workload_delegation_scope(workspace=workspace, deployment_name=name, config=config)
            )
        except Exception:
            logger.warning("Failed to refresh Docker workload delegations for %s/%s", workspace, name, exc_info=True)
            return

        for delegation in delegations:
            try:
                if delegation.revoked_at is not None or not delegation.opaque_subject_token_hash:
                    continue
                remaining_seconds = (as_aware_utc(delegation.expires_at) - now).total_seconds()
                if remaining_seconds > refresh_threshold_seconds:
                    continue
                await self._refresh_docker_workload_delegation(
                    workspace=workspace,
                    name=name,
                    config=config,
                    delegation=delegation,
                    now=now,
                )
            except WorkloadDelegationConflictError:
                logger.debug(
                    "Skipping stale Docker workload delegation refresh",
                    extra={"delegation_name": delegation.name},
                )
            except Exception:
                logger.warning(
                    "Failed to refresh Docker workload delegation",
                    extra={"delegation_name": delegation.name},
                    exc_info=True,
                )

    async def _refresh_docker_workload_delegation(
        self,
        *,
        workspace: str,
        name: str,
        config: DeploymentConfig,
        delegation: WorkloadDelegationEntity,
        now: datetime,
    ) -> None:
        if config.workload_identity is None or not delegation.workload_generation:
            return
        scope = workload_delegation_scope(workspace=workspace, deployment_name=name, config=config)
        refreshed, proof_token = build_docker_opaque_workload_delegation(
            scope=scope,
            workload_audience=get_workload_delegation_audience(),
            workload_generation=delegation.workload_generation,
            auth_context=delegation.auth_context,
            ttl_seconds_active=config.workload_identity.token_expiration_seconds,
            now=now,
        )
        replacement = delegation.model_copy(
            update={
                "workload_subject": refreshed.workload_subject,
                "workload_audience": refreshed.workload_audience,
                "workload_workspace": refreshed.workload_workspace,
                "workload_kind": refreshed.workload_kind,
                "workload_id": refreshed.workload_id,
                "workload_claim_id": refreshed.workload_claim_id,
                "workload_generation": refreshed.workload_generation,
                "auth_context": refreshed.auth_context,
                "opaque_subject_token_hash": refreshed.opaque_subject_token_hash,
                "expires_at": refreshed.expires_at,
            }
        )
        volume_name = self._workload_identity_volume_name(
            workspace=workspace,
            name=name,
            role=delegation.workload_generation,
        )
        await self._write_workload_identity_subject_token(
            volume_name,
            proof_token,
            workspace=workspace,
            deployment_name=name,
        )
        await self._workload_delegations.update(replacement, expected_db_version=delegation.db_version)

    async def _sync_workload_identity_for_status(
        self,
        *,
        workspace: str,
        name: str,
        config: DeploymentConfig | None,
        status_update: BackendStatusUpdate,
    ) -> None:
        if status_update.status in NON_TERMINAL_DEPLOYMENT_STATUSES:
            await self._refresh_workload_delegations_for_config(workspace=workspace, name=name, config=config)
            return
        await self._revoke_workload_delegations_for_deployment(workspace=workspace, name=name)
        await self._cleanup_workload_identity_volumes(workspace=workspace, name=name)

    async def _cleanup_missing_workload_identity(
        self,
        *,
        workspace: str,
        name: str,
    ) -> None:
        try:
            await self._revoke_workload_delegations_for_deployment(workspace=workspace, name=name)
            await self._cleanup_workload_identity_volumes(workspace=workspace, name=name)
        except Exception:
            logger.warning(
                "Failed to clean up missing Docker workload identity for %s/%s", workspace, name, exc_info=True
            )

    async def _prepare_workload_identity_for_container(
        self,
        *,
        workspace: str,
        deployment_name: str,
        config: DeploymentConfig,
        role: str,
        base_labels: dict[str, str],
        auth_context: AuthContext | None,
    ) -> tuple[str, str] | None:
        if not workload_identity_requested(config):
            return None
        error = workload_identity_activation_error(config=config, auth_context=auth_context)
        if error is not None:
            raise RuntimeError(error)
        if auth_context is None or config.workload_identity is None:
            return None

        volume_name = self._workload_identity_volume_name(workspace=workspace, name=deployment_name, role=role)
        scope = workload_delegation_scope(workspace=workspace, deployment_name=deployment_name, config=config)
        delegation, proof_token = build_docker_opaque_workload_delegation(
            scope=scope,
            workload_audience=get_workload_delegation_audience(),
            workload_generation=role,
            auth_context=auth_context,
            ttl_seconds_active=config.workload_identity.token_expiration_seconds,
        )
        await self._ensure_workload_identity_volume(
            volume_name=volume_name,
            labels=self._workload_identity_volume_labels(
                base_labels=base_labels,
                role=role,
                volume_name=volume_name,
            ),
        )
        registered = False
        try:
            await self._workload_delegations.register(delegation, require_opaque_subject_token_hash=True)
            registered = True
            await self._write_workload_identity_subject_token(
                volume_name,
                proof_token,
                workspace=workspace,
                deployment_name=deployment_name,
            )
        except Exception:
            if registered:
                await self._try_revoke_workload_delegation(delegation.name, reason="token provisioning failure")
            await self._remove_workload_identity_volume(volume_name)
            raise
        return volume_name, delegation.name

    async def _cleanup_prepared_workload_identity(
        self,
        material: tuple[str, str] | None,
        *,
        reason: str,
    ) -> None:
        if material is None:
            return
        volume_name, delegation_name = material
        await self._try_revoke_workload_delegation(delegation_name, reason=reason)
        await self._remove_workload_identity_volume(volume_name)

    async def _write_workload_identity_subject_token(
        self,
        volume_name: str,
        token: str,
        *,
        workspace: str,
        deployment_name: str,
    ) -> None:
        token_volume_path = "/workload-identity-vol"
        container_name_for_write = f"deployment-workload-token-write-{uuid.uuid4().hex[:8]}"
        finalize_token_command = (
            f"mv {token_volume_path}/token.tmp {token_volume_path}/token && chmod 0444 {token_volume_path}/token"
        )
        image = self._executor_config.workload_token_writer_image
        container_args = {
            "name": container_name_for_write,
            "image": image,
            "command": ["sh", "-c", finalize_token_command],
            "volumes": {volume_name: {"bind": token_volume_path, "mode": "rw"}},
            "labels": {
                MANAGED_BY_KEY: MANAGED_BY_LABEL,
                DEPLOYMENT_WORKSPACE_LABEL: workspace,
                DEPLOYMENT_NAME_LABEL: deployment_name,
                RESOURCE_SCOPE_LABEL: self._executor_config.resource_scope,
            },
        }
        container = await self._create_container_with_image_pull(
            container_args=container_args,
            image=image,
        )
        try:
            await asyncio.to_thread(container.put_archive, token_volume_path, build_token_archive(token))
            await asyncio.to_thread(container.start)
            result = await asyncio.to_thread(container.wait, timeout=self._executor_config.docker_timeout)
            exit_code = self._exit_code_from_wait_result(result)
            if exit_code != 0:
                raise RuntimeError(f"workload identity token writer exited with code {exit_code}")
        finally:
            try:
                await asyncio.to_thread(container.remove, force=True)
            except Exception:
                logger.warning("Failed to remove workload identity token writer container", exc_info=True)

    async def _create_container_with_image_pull(self, *, container_args: dict[str, Any], image: str) -> DockerContainer:
        try:
            return await asyncio.to_thread(self._client.containers.create, **container_args)
        except (self._docker_errors.ImageNotFound, self._docker_errors.NotFound):
            pull_error = await self._pull_image(image, ngc_api_key=None)
            if pull_error is not None:
                raise RuntimeError(pull_error)
            return await asyncio.to_thread(self._client.containers.create, **container_args)

    async def _remove_container_by_name(self, name: str) -> None:
        """Best-effort removal of a container this call just created."""
        try:
            container = await asyncio.to_thread(self._client.containers.get, name)
            await asyncio.to_thread(container.remove, force=True)
        except self._docker_errors.NotFound:
            pass
        except Exception:
            logger.warning("Failed to remove container %s", name, exc_info=True)

    async def _run_server_container(
        self,
        *,
        workspace: str,
        config: DeploymentConfig,
        container_spec: Container,
        name: str,
        labels: dict[str, str],
        host_ports: dict[int, int],
        gpu_ids: list[int],
        network: str | None,
        workload_identity_volume_name: str | None = None,
    ) -> tuple[DockerContainer | None, dict[int, int], str]:
        """Start the primary container, reallocating host ports on Docker port conflicts.

        Returns the started container (None on failure), the host ports it actually
        published, and an error message when it could not be started.
        """
        rejected_ports: set[int] = set()
        attempt = 0
        while True:
            attempt += 1
            create_kwargs = self._build_run_kwargs(
                workspace=workspace,
                config=config,
                container=container_spec,
                name=name,
                labels=labels,
                host_ports=host_ports,
                gpu_ids=gpu_ids,
                network=network,
                workload_identity_volume_name=workload_identity_volume_name,
            )
            created: DockerContainer | None = None
            try:
                created = await asyncio.to_thread(self._client.containers.create, **create_kwargs)
                if config.config_files:
                    await self._deliver_config_files(created, config.config_files)
                await asyncio.to_thread(created.start)
                return created, host_ports, ""
            except Exception as exc:
                if created is not None:
                    await asyncio.to_thread(created.remove, force=True)
                last_attempt = attempt == _PORT_CONFLICT_ATTEMPTS
                if not host_ports or last_attempt or _PORT_CONFLICT_MARKER not in str(exc):
                    logger.exception("Failed to start container %s", name)
                    return None, host_ports, f"Failed to start container: {exc}"

                rejected_ports |= set(host_ports.values())
                logger.warning(
                    "Host port conflict starting %s (attempt %d/%d); reallocating outside %s",
                    name,
                    attempt,
                    _PORT_CONFLICT_ATTEMPTS,
                    sorted(rejected_ports),
                )
                try:
                    reallocated = await self._allocate_host_ports(container_spec, exclude_ports=rejected_ports)
                except PortEnumerationError as port_exc:
                    return (
                        None,
                        host_ports,
                        f"Could not determine host ports in use: {port_exc}",
                    )
                if reallocated is None:
                    return (
                        None,
                        host_ports,
                        "No host ports available in configured range",
                    )
                host_ports = reallocated

    async def _deliver_config_files(
        self,
        container: DockerContainer,
        config_files: list[ConfigFile],
    ) -> None:
        archive = _config_files_tar(config_files)
        await asyncio.to_thread(container.put_archive, "/", archive)

    def _build_run_kwargs(
        self,
        *,
        workspace: str,
        config: DeploymentConfig,
        container: Container,
        name: str,
        labels: dict[str, str],
        host_ports: dict[int, int],
        gpu_ids: list[int],
        network: str | None,
        workload_identity_volume_name: str | None = None,
    ) -> dict[str, Any]:
        """Build docker ``containers.run`` kwargs for one container in a group."""
        environment = env_dict(container)
        if workload_identity_volume_name is not None:
            environment.update(workload_identity_env(token_file_path=WORKLOAD_IDENTITY_TOKEN_FILE_PATH))
        run_kwargs: dict[str, Any] = {
            "image": container.image,
            "name": name,
            "labels": labels,
            "environment": environment,
            **restart_policy_kwargs(config.restart_policy, config.backoff_limit),
        }
        # Mirror Kubernetes semantics (see the k8s compiler): a container spec's
        # ``command`` is the entrypoint override and ``args`` is the command
        # (CMD). Map them onto docker's distinct ``entrypoint`` / ``command``
        # kwargs so a driven container overrides the image's baked-in ENTRYPOINT
        # instead of appending to it. Conflating both into docker ``command``
        # leaves the image ENTRYPOINT in force and silently drops the spec.
        if container.command:
            run_kwargs["entrypoint"] = list(container.command)
        if container.args:
            run_kwargs["command"] = list(container.args)

        volume_bindings = {
            **self._executor_volume_bindings(),
            **build_volume_bindings(workspace, merged_volume_mounts(config, container)),
        }
        if workload_identity_volume_name is not None:
            volume_bindings = {
                **volume_bindings,
                workload_identity_volume_name: {"bind": WORKLOAD_IDENTITY_VOLUME_PATH, "mode": "ro"},
            }
        if volume_bindings:
            run_kwargs["volumes"] = volume_bindings

        # When joining another container's network namespace, docker forbids
        # publishing ports (they belong to the primary) and also forbids
        # ExtraHosts. Only the primary maps host ports / host.docker.internal.
        # Drop the image HEALTHCHECK on netns-joined sidecars: nhx-api ships a
        # probe for localhost:8080/health/ready, which fails forever for LoRA
        # adapters (``python -m ...adapters.main`` has no HTTP listener).
        if network is not None and network.startswith("container:"):
            run_kwargs["network"] = network
            run_kwargs["healthcheck"] = {"test": ["NONE"]}
        else:
            # Linux Docker Engine does not define host.docker.internal by default;
            # jobs/agents rewrite loopback platform URLs to that hostname.
            run_kwargs["extra_hosts"] = {"host.docker.internal": "host-gateway"}
            if container.ports:
                run_kwargs["ports"] = build_port_bindings(container, host_ports)
            if network:
                run_kwargs["network"] = network

        device_requests = device_requests_for_gpus(gpu_ids)
        if device_requests:
            run_kwargs["device_requests"] = device_requests

        return run_kwargs

    async def _run_init_container(
        self,
        *,
        workspace: str,
        name: str,
        config: DeploymentConfig,
        init: Container,
        base_labels: dict[str, str],
        dep_key: str,
        gpu_ids: list[int],
        auth_context: AuthContext | None = None,
    ) -> BackendStatusUpdate | None:
        """Run one init container to completion. Return an error status or None on success."""
        init_name = companion_container_name(workspace, name, f"init-{init.name}")
        # Best-effort clean any stale init container from a prior attempt.
        try:
            stale = await asyncio.to_thread(self._client.containers.get, init_name)
            if self._container_matches_deployment_group(stale, workspace, name):
                await asyncio.to_thread(stale.remove, force=True)
        except self._docker_errors.NotFound:
            pass
        except Exception:
            logger.warning("Failed to remove stale init container %s", init_name, exc_info=True)

        # Init containers are intentionally CPU-only: no device_requests is set,
        # so they never receive a GPU. This suits the current init workload
        # (lora-cache-init prepares the scratch dir). If the models compiler ever
        # emits a GPU-needing init container, this would need to plumb GPUs
        # through here.
        role = f"init-{init.name}"
        init_identity: tuple[str, str] | None = None
        labels = {**base_labels, CONTAINER_ROLE_LABEL: role}
        environment = env_dict(init)
        try:
            init_identity = await self._prepare_workload_identity_for_container(
                workspace=workspace,
                deployment_name=name,
                config=config,
                role=role,
                base_labels=labels,
                auth_context=auth_context,
            )
        except Exception as exc:
            if gpu_ids and self._gpu_pool is not None:
                self._gpu_pool.release_gpu(dep_key)
            return BackendStatusUpdate(status="FAILED", status_message=f"Failed to prepare workload identity: {exc}")
        if init_identity is not None:
            labels = self._workload_identity_volume_labels(
                base_labels=base_labels,
                role=role,
                volume_name=init_identity[0],
            )
            environment.update(workload_identity_env(token_file_path=WORKLOAD_IDENTITY_TOKEN_FILE_PATH))
        create_kwargs: dict[str, Any] = {
            "image": init.image,
            "name": init_name,
            "labels": labels,
            "environment": environment,
        }
        if init.command:
            create_kwargs["entrypoint"] = list(init.command)
        if init.args:
            create_kwargs["command"] = list(init.args)
        volume_bindings = {
            **self._executor_volume_bindings(),
            **build_volume_bindings(workspace, merged_volume_mounts(config, init)),
        }
        if init_identity is not None:
            volume_bindings = {
                **volume_bindings,
                init_identity[0]: {"bind": WORKLOAD_IDENTITY_VOLUME_PATH, "mode": "ro"},
            }
        if volume_bindings:
            create_kwargs["volumes"] = volume_bindings

        def _run_and_wait() -> int:
            container = self._client.containers.create(**create_kwargs)
            try:
                container.start()
                result = container.wait(timeout=self._executor_config.docker_timeout)
                return self._exit_code_from_wait_result(result)
            finally:
                try:
                    container.remove(force=True)
                except Exception:
                    logger.warning("Failed to remove init container %s", init_name, exc_info=True)

        try:
            exit_code = await asyncio.to_thread(_run_and_wait)
        except Exception as exc:
            if gpu_ids and self._gpu_pool is not None:
                self._gpu_pool.release_gpu(dep_key)
            logger.exception("Init container %s failed to run", init_name)
            return BackendStatusUpdate(
                status="FAILED",
                status_message=f"Init container {init.name} failed: {exc}",
            )
        finally:
            await self._cleanup_prepared_workload_identity(init_identity, reason="init container completed")

        if exit_code != 0:
            if gpu_ids and self._gpu_pool is not None:
                self._gpu_pool.release_gpu(dep_key)
            return BackendStatusUpdate(
                status="FAILED",
                status_message=f"Init container {init.name} exited with code {exit_code}",
            )
        return None

    async def read_status(self, *, workspace: str, name: str) -> BackendStatusUpdate:
        c_name = container_name(workspace, name)
        dep_key = deployment_key(workspace, name)
        try:
            container = await asyncio.to_thread(self._client.containers.get, c_name)
            if not self._container_matches_deployment_group(container, workspace, name):
                restart_policy = await self._resolve_restart_policy(workspace, name)
                status_update = missing_container_status(restart_policy, container_name=c_name)
                await self._cleanup_missing_workload_identity(workspace=workspace, name=name)
                if status_update.status not in NON_TERMINAL_DEPLOYMENT_STATUSES and self._gpu_pool is not None:
                    self._gpu_pool.release_gpu(dep_key)
                return status_update
            await asyncio.to_thread(container.reload)
        except self._docker_errors.NotFound:
            restart_policy = await self._resolve_restart_policy(workspace, name)
            status_update = missing_container_status(restart_policy, container_name=c_name)
            await self._cleanup_missing_workload_identity(workspace=workspace, name=name)
            if status_update.status not in NON_TERMINAL_DEPLOYMENT_STATUSES and self._gpu_pool is not None:
                self._gpu_pool.release_gpu(dep_key)
            return status_update
        except (
            self._docker_errors.APIError,
            ReadTimeout,
            Urllib3ReadTimeoutError,
            RequestsConnectionError,
        ) as exc:
            logger.error("Transient Docker API error checking container %s: %s", c_name, exc)
            return BackendStatusUpdate(
                status="UNKNOWN",
                status_message=f"Docker API error while checking container status: {exc}",
                error_details={"error": str(exc), "container_name": c_name},
            )
        except Exception as exc:
            return BackendStatusUpdate(status="FAILED", status_message=f"Docker API error: {exc}")

        labels = container.labels or {}
        restart_policy: RestartPolicy = labels.get(RESTART_POLICY_LABEL, "Always")
        state = container.status
        container_id = (container.id or "")[:12]
        host_ports = self._extract_host_ports(container)
        endpoints = self._endpoints_from_container_ports(host_ports, target_name=c_name)

        if state in ("created", "restarting"):
            return map_docker_state_to_starting(container_id, state)

        if state == "running":
            # The default (no-declared-probe) reachability check TCP-connects the port, so
            # it must see only TCP mappings: a UDP-only workload has no TCP listener and
            # would otherwise be gated STARTING forever. Endpoints still carry every port.
            tcp_host_ports = self._extract_host_ports(container, protocol="tcp")
            host_url = self._primary_host_url(tcp_host_ports, target_name=c_name)
            probe_ports = self._probe_ports(tcp_host_ports)
            config = await self._load_config_from_labels(workspace, labels)
            await self._refresh_workload_delegations_for_config(workspace=workspace, name=name, config=config)
            probe = None
            if config is not None and config.containers:
                probe = config.containers[0].readiness_probe
            ready, reason = await check_readiness_probe(
                container=container,
                probe=probe,
                host_url=host_url,
                host_ports=probe_ports,
            )
            if ready and restart_policy == "Always":
                sidecar_ok, sidecar_reason = await self._sidecars_healthy(workspace, name, config)
                if not sidecar_ok:
                    return BackendStatusUpdate(
                        status="STARTING",
                        status_message=f"Server ready but sidecar not ready ({sidecar_reason})",
                        endpoints=endpoints,
                    )
                return BackendStatusUpdate(
                    status="READY",
                    status_message=f"Container running and ready ({reason})",
                    endpoints=endpoints,
                )
            if ready and restart_policy in ("OnFailure", "Never"):
                return BackendStatusUpdate(
                    status="STARTING",
                    status_message=f"Container running ({reason})",
                    endpoints=endpoints,
                )
            return BackendStatusUpdate(
                status="STARTING",
                status_message=f"Container running but not ready ({reason})",
                endpoints=endpoints,
            )

        if state in ("exited", "dead"):
            exit_code = _docker_inspect_exit_code(container)
            restart_count = _docker_inspect_restart_count(container)
            config = await self._load_config_from_labels(workspace, labels)
            status_update = self._status_from_exited_container(
                exit_code=exit_code,
                restart_policy=restart_policy,
                labels=labels,
                dep_key=dep_key,
                restart_count=restart_count,
                endpoints=endpoints,
            )
            await self._sync_workload_identity_for_status(
                workspace=workspace,
                name=name,
                config=config,
                status_update=status_update,
            )
            return status_update

        if state == "removing":
            return BackendStatusUpdate(
                status="DELETING",
                status_message=f"Container removing (ID: {container_id})",
            )

        return BackendStatusUpdate(status="STARTING", status_message=f"Container state: {state}")

    @staticmethod
    def _exit_code_from_wait_result(result: dict[str, Any] | int) -> int:
        """Normalize a docker `container.wait()` result to an exit code."""
        return int(result.get("StatusCode", 1)) if isinstance(result, dict) else int(result)

    def _status_from_exited_container(
        self,
        *,
        exit_code: int,
        restart_policy: RestartPolicy,
        labels: dict[str, str],
        dep_key: str,
        restart_count: int = 0,
        endpoints: list[Endpoint] | None = None,
    ) -> BackendStatusUpdate:
        """Map a stopped container's exit code to a deployment status update."""
        resolved_endpoints = endpoints or []
        if exit_code == 0 and restart_policy in ("Never", "OnFailure"):
            if self._gpu_pool is not None:
                self._gpu_pool.release_gpu(dep_key)
            return BackendStatusUpdate(
                status="SUCCEEDED",
                status_message="Container exited successfully (code 0)",
                exit_code=exit_code,
                endpoints=resolved_endpoints,
            )
        if restart_policy == "Always":
            return BackendStatusUpdate(
                status="STARTING",
                status_message=f"Container exited (code {exit_code}); restart policy will recreate it",
                exit_code=exit_code,
                endpoints=resolved_endpoints,
            )
        if restart_policy == "OnFailure":
            backoff_limit = int(labels.get(BACKOFF_LIMIT_LABEL, "6"))
            if backoff_limit == 0:
                return BackendStatusUpdate(
                    status="STARTING",
                    status_message=(f"Container exited (code {exit_code}); retry {restart_count}/unlimited"),
                    exit_code=exit_code,
                    endpoints=resolved_endpoints,
                )
            if restart_count < backoff_limit:
                return BackendStatusUpdate(
                    status="STARTING",
                    status_message=(f"Container exited (code {exit_code}); retry {restart_count}/{backoff_limit}"),
                    exit_code=exit_code,
                    endpoints=resolved_endpoints,
                )
        if self._gpu_pool is not None:
            self._gpu_pool.release_gpu(dep_key)
        status = map_exited_status(exit_code, restart_policy)
        return BackendStatusUpdate(
            status=status,
            status_message=f"Container exited with code {exit_code}",
            exit_code=exit_code,
            endpoints=resolved_endpoints,
        )

    async def _observe_one_shot_primary_after_create(
        self,
        *,
        workspace: str,
        name: str,
        config: DeploymentConfig,
        container: DockerContainer,
        dep_key: str,
        endpoints: list[Endpoint],
    ) -> BackendStatusUpdate:
        """Observe a one-shot primary container immediately after create."""
        restart_policy = config.restart_policy
        labels = container.labels or {}

        if restart_policy == "Never":
            observe_timeout = self._executor_config.oneshot_observe_timeout_seconds

            def _wait_for_exit() -> int:
                return self._exit_code_from_wait_result(container.wait(timeout=observe_timeout))

            try:
                exit_code = await asyncio.to_thread(_wait_for_exit)
            except (
                ReadTimeout,
                Urllib3ReadTimeoutError,
                RequestsConnectionError,
                self._docker_errors.APIError,
            ):
                return BackendStatusUpdate(
                    status="STARTING",
                    status_message=(
                        f"Container {container.name} still running or status unavailable "
                        f"after observe wait ({observe_timeout}s)"
                    ),
                    endpoints=endpoints,
                )
            except Exception as exc:
                logger.exception("Failed waiting for one-shot container %s to exit", container.name)
                await self.delete_deployment(workspace, name)
                return BackendStatusUpdate(
                    status="FAILED",
                    status_message=f"Failed waiting for container to exit: {exc}",
                )
            status_update = self._status_from_exited_container(
                exit_code=exit_code,
                restart_policy=restart_policy,
                labels=labels,
                dep_key=dep_key,
                endpoints=endpoints,
            )
            await self._sync_workload_identity_for_status(
                workspace=workspace,
                name=name,
                config=config,
                status_update=status_update,
            )
            return status_update

        try:
            await asyncio.to_thread(container.reload)
        except (
            self._docker_errors.APIError,
            ReadTimeout,
            Urllib3ReadTimeoutError,
            RequestsConnectionError,
        ) as exc:
            return BackendStatusUpdate(
                status="STARTING",
                status_message=f"Container created but status check failed: {exc}",
                endpoints=endpoints,
            )
        if container.status in _EXITED_CONTAINER_STATES:
            exit_code = _docker_inspect_exit_code(container)
            restart_count = _docker_inspect_restart_count(container)
            status_update = self._status_from_exited_container(
                exit_code=exit_code,
                restart_policy=restart_policy,
                labels=labels,
                dep_key=dep_key,
                restart_count=restart_count,
                endpoints=endpoints,
            )
            await self._sync_workload_identity_for_status(
                workspace=workspace,
                name=name,
                config=config,
                status_update=status_update,
            )
            return status_update

        container_name_label = container.name or "primary"
        return BackendStatusUpdate(
            status="STARTING",
            status_message=f"Container {container_name_label} created",
            endpoints=endpoints,
        )

    async def _sidecars_healthy(self, workspace: str, name: str, config: DeploymentConfig | None) -> tuple[bool, str]:
        """Return (all_healthy, reason) for a deployment's expected sidecar containers.

        The set of expected sidecars is derived from the deployment ``config``
        (every container after the primary), so a sidecar that has been *removed*
        entirely — not just exited — is still detected as not-ready. A sidecar is
        healthy only when a container for its role is present AND running; an
        exited or missing sidecar keeps the deployment out of READY (its Always
        restart policy recreates an exited one; a missing one signals a broken
        group).

        Deployments with no sidecars (the common single-container case) skip the
        Docker enumeration entirely.
        """
        expected_roles = {sidecar.name for sidecar in build_docker_plan(config).sidecars} if config else set()
        if not expected_roles:
            return True, "no sidecars"

        def _list() -> list[Any]:
            return self._client.containers.list(
                all=True,
                filters={
                    "label": [
                        f"{MANAGED_BY_KEY}={MANAGED_BY_LABEL}",
                        f"{DEPLOYMENT_WORKSPACE_LABEL}={workspace}",
                        f"{DEPLOYMENT_NAME_LABEL}={name}",
                        *self._resource_scope_filter_label(),
                    ]
                },
            )

        try:
            containers = await asyncio.to_thread(_list)
        except Exception:
            # If we cannot enumerate the group, do not block readiness on it.
            logger.warning("Failed to list sidecars for %s/%s", workspace, name, exc_info=True)
            return True, "sidecar check skipped"

        running_roles: set[str] = set()
        for container in containers:
            if not self._container_matches_deployment_group(container, workspace, name):
                continue
            role = (container.labels or {}).get(CONTAINER_ROLE_LABEL, "")
            if role not in expected_roles:
                continue
            if container.status != "running":
                return False, f"sidecar '{role}' is {container.status}"
            running_roles.add(role)

        missing = expected_roles - running_roles
        if missing:
            return False, f"sidecar '{sorted(missing)[0]}' is missing"
        return True, "sidecars running"

    async def delete_deployment(self, workspace: str, name: str) -> BackendStatusUpdate:
        c_name = container_name(workspace, name)
        dep_key = deployment_key(workspace, name)

        def _delete() -> None:
            # Remove every container in the deployment group: the primary
            # (dep-<hash>) plus any companion sidecar/init containers, which are
            # discoverable by the shared deployment identity labels.
            group: dict[str, Any] = {}
            try:
                for container in self._client.containers.list(
                    all=True,
                    filters={
                        "label": [
                            f"{MANAGED_BY_KEY}={MANAGED_BY_LABEL}",
                            f"{DEPLOYMENT_WORKSPACE_LABEL}={workspace}",
                            f"{DEPLOYMENT_NAME_LABEL}={name}",
                            *self._resource_scope_filter_label(),
                        ]
                    },
                ):
                    if self._container_matches_deployment_group(container, workspace, name):
                        group[container.name] = container
            except Exception:
                logger.warning(
                    "Failed to list group containers for %s; falling back to primary",
                    c_name,
                    exc_info=True,
                )
            # Ensure the primary is included even if the label list query missed it.
            if c_name not in group:
                try:
                    primary = self._client.containers.get(c_name)
                    if self._container_matches_deployment_group(primary, workspace, name):
                        group[c_name] = primary
                except self._docker_errors.NotFound:
                    pass
            for container in group.values():
                try:
                    container.stop(timeout=30)
                    container.remove(force=True)
                except self._docker_errors.NotFound:
                    continue

        try:
            await asyncio.to_thread(_delete)
        except Exception as exc:
            return BackendStatusUpdate(status="FAILED", status_message=f"Failed to delete container: {exc}")
        finally:
            if self._gpu_pool is not None:
                self._gpu_pool.release_gpu(dep_key)

        try:
            await self._revoke_workload_delegations_for_deployment(workspace=workspace, name=name)
            await self._cleanup_workload_identity_volumes(workspace=workspace, name=name)
        except Exception as exc:
            return BackendStatusUpdate(
                status="FAILED",
                status_message=f"Container {c_name} deleted, but workload identity cleanup failed: {exc}",
            )

        return BackendStatusUpdate(status="SUCCEEDED", status_message=f"Container {c_name} deleted")

    async def list_managed_deployment_names(self) -> list[str]:
        try:
            containers = await asyncio.to_thread(
                self._client.containers.list,
                all=True,
                filters=managed_by_filter(resource_scope=self._executor_config.resource_scope),
            )
        except Exception:
            logger.warning("Failed to list managed containers", exc_info=True)
            return []

        seen: set[str] = set()
        for container in containers:
            container_labels = container.labels or {}
            if container_labels.get(MANAGED_BY_KEY) != MANAGED_BY_LABEL:
                continue
            if not self._labels_match_resource_scope(container_labels):
                continue
            ws = container_labels.get(DEPLOYMENT_WORKSPACE_LABEL)
            dep_name = container_labels.get(DEPLOYMENT_NAME_LABEL)
            if ws and dep_name:
                seen.add(f"{ws}/{dep_name}")
        return sorted(seen)

    def _executor_volume_bindings(self) -> dict[str, dict[str, str]]:
        return {
            mount.volume_name: {
                "bind": mount.mount_path,
                "mode": "ro" if mount.read_only else "rw",
            }
            for mount in self._executor_config.additional_volume_mounts
        }

    def _resource_scope_filter_label(self) -> list[str]:
        return [f"{RESOURCE_SCOPE_LABEL}={self._executor_config.resource_scope}"]

    def _labels_match_resource_scope(self, labels: dict[str, Any]) -> bool:
        return labels.get(RESOURCE_SCOPE_LABEL) == self._executor_config.resource_scope

    def _container_matches_deployment_group(self, container: DockerContainer, workspace: str, name: str) -> bool:
        labels = container.labels or {}
        return (
            labels.get(DEPLOYMENT_WORKSPACE_LABEL) == workspace
            and labels.get(DEPLOYMENT_NAME_LABEL) == name
            and labels.get(MANAGED_BY_KEY) == MANAGED_BY_LABEL
            and self._labels_match_resource_scope(labels)
        )

    async def get_logs(self, *, workspace: str, name: str, tail: int = 100) -> LogResult:
        c_name = container_name(workspace, name)

        def _logs() -> bytes:
            container = self._client.containers.get(c_name)
            if not self._container_matches_deployment_group(container, workspace, name):
                raise self._docker_errors.NotFound(f"Container {c_name} not found")
            return container.logs(tail=tail, timestamps=True)

        try:
            raw = await asyncio.to_thread(_logs)
            text = raw.decode("utf-8", errors="ignore")
            lines = text.splitlines()
            truncated = len(text) > LOG_MAX_CHARS
            if truncated:
                lines = lines[-tail:]
            return LogResult(lines=lines, truncated=truncated)
        except self._docker_errors.NotFound:
            return LogResult(lines=[f"Container {c_name} not found"])
        except Exception as exc:
            return LogResult(lines=[f"Failed to fetch logs: {exc}"])

    async def create_volume(
        self,
        *,
        workspace: str,
        name: str,
        size: str,
        access_modes: list[str],
        backend_config: dict[str, Any],
    ) -> VolumeStatusUpdate:
        del size, access_modes
        driver = "local"
        init_chmod: str | None = None
        init_image: str | None = None
        docker_section = backend_config.get("docker") or {}
        if isinstance(docker_section, dict):
            if docker_section.get("driver"):
                driver = str(docker_section["driver"])
            # initChmod/initImage let the compiler request the volume be made
            # writable by non-root workloads (docker analogue of k8s fsGroup).
            if docker_section.get("initChmod"):
                init_chmod = str(docker_section["initChmod"])
            if docker_section.get("initImage"):
                init_image = str(docker_section["initImage"])
        return await volume_ops.create_volume(
            self._client,
            workspace=workspace,
            name=name,
            driver=driver,
            init_chmod=init_chmod,
            init_image=init_image,
            resource_scope=self._executor_config.resource_scope,
        )

    async def read_volume_status(
        self,
        *,
        workspace: str,
        name: str,
        backend_config: dict[str, Any] | None = None,
    ) -> VolumeStatusUpdate:
        # backend_config is part of the DeploymentBackend ABC so the reconciler can pass
        # K8s namespace overrides; Docker volume names are global to the daemon.
        return await volume_ops.read_volume_status(self._client, workspace=workspace, name=name)

    async def delete_volume(
        self,
        workspace: str,
        name: str,
        *,
        backend_config: dict[str, Any] | None = None,
    ) -> VolumeStatusUpdate:
        return await volume_ops.delete_volume(self._client, workspace=workspace, name=name)

    def _container_matches_deployment(
        self,
        container: DockerContainer,
        workspace: str,
        name: str,
        config_name: str,
    ) -> bool:
        labels = container.labels or {}
        return (
            self._container_matches_deployment_group(container, workspace, name)
            and labels.get(CONFIG_NAME_LABEL) == config_name
        )

    async def _pull_image(self, image: str, *, ngc_api_key: str | None) -> str | None:
        """Pull ``image``, authenticating to NGC when needed. Returns an error message or None."""
        pull_kwargs: dict[str, Any] = {}
        if _is_ngc_image(image):
            if ngc_api_key:
                pull_kwargs["auth_config"] = {
                    "username": NGC_IMAGE_REGISTRY_USER_NAME,
                    "password": ngc_api_key,
                }
            else:
                logger.warning(
                    "Pulling NGC image without NGC credentials",
                    extra={"image": image},
                )
        try:
            await asyncio.to_thread(self._client.images.pull, image, **pull_kwargs)
        except (self._docker_errors.APIError, self._docker_errors.ImageNotFound) as exc:
            message = f"Failed to pull image {image}: {exc}"
            if _is_ngc_image(image):
                message += (
                    ". Ensure the image exists and NGC_API_KEY (or platform.ngc_api_key_secret) is set correctly."
                )
            return message
        return None

    async def _resolve_restart_policy(self, workspace: str, name: str) -> RestartPolicy:
        config = await self._load_config_for_deployment_entity(workspace, name)
        if config is not None:
            return config.restart_policy
        return "Always"

    async def _load_config_from_labels(self, workspace: str, labels: dict[str, str]) -> DeploymentConfig | None:
        config_name = labels.get(CONFIG_NAME_LABEL)
        if not config_name:
            return await self._load_config_for_deployment_entity(
                workspace,
                labels.get(DEPLOYMENT_NAME_LABEL, ""),
            )
        try:
            config = await self._entities.get(DeploymentConfig, config_name, workspace=workspace)
        except Exception:
            return None
        return config if isinstance(config, DeploymentConfig) else None

    async def _load_config_for_deployment_entity(
        self,
        workspace: str,
        deployment_name: str,
    ) -> DeploymentConfig | None:
        if not deployment_name:
            return None
        try:
            deployment = await self._entities.get(Deployment, deployment_name, workspace=workspace)
            if not deployment.deployment_config:
                return None
            config = await self._entities.get(
                DeploymentConfig,
                deployment.deployment_config,
                workspace=workspace,
            )
        except Exception:
            return None
        return config if isinstance(config, DeploymentConfig) else None

    def _extract_host_ports(self, container: DockerContainer, *, protocol: str | None = None) -> dict[int, int]:
        """Map container port -> published host port.

        With *protocol* (e.g. ``"tcp"``) only mappings of that protocol are returned;
        docker keys the port map as ``"<port>/<proto>"``. Defaults to every protocol.
        """
        result: dict[int, int] = {}
        ports = container.ports or {}
        for key, bindings in ports.items():
            if not bindings:
                continue
            key_str = str(key)
            if protocol is not None and not key_str.endswith(f"/{protocol}"):
                continue
            container_port = int(key_str.split("/")[0])
            host_port = bindings[0].get("HostPort")
            if host_port:
                result[container_port] = int(host_port)
        return result

    def _url_for_port(
        self,
        *,
        target_name: str,
        container_port: int,
        host_port: int | None,
        scheme: str = "http",
    ) -> str | None:
        if self._executor_config.endpoint_mode == "network":
            return host_url_for_port(target_name, container_port, scheme=scheme)
        if host_port is None:
            return None
        host = os.environ.get("NHX_LOOPBACK_ADDRESS", LOOPBACK_ADDRESSES[0])
        return host_url_for_port(host, host_port, scheme=scheme)

    def _primary_host_url(self, host_ports: dict[int, int], *, target_name: str) -> str | None:
        if not host_ports:
            return None
        container_port, host_port = next(iter(host_ports.items()))
        return self._url_for_port(target_name=target_name, container_port=container_port, host_port=host_port)

    def _probe_ports(self, host_ports: dict[int, int]) -> dict[int, int]:
        if self._executor_config.endpoint_mode == "network":
            return {container_port: container_port for container_port in host_ports}
        return host_ports

    def _build_endpoints(
        self,
        container_spec: Container,
        host_ports: dict[int, int],
        *,
        target_name: str,
    ) -> list[Endpoint]:
        endpoints: list[Endpoint] = []
        for port_spec in container_spec.ports:
            host_port = host_ports.get(port_spec.container_port)
            endpoint_url = self._url_for_port(
                target_name=target_name,
                container_port=port_spec.container_port,
                host_port=host_port,
            )
            if endpoint_url is None:
                continue
            endpoint_name = port_spec.name or f"port-{port_spec.container_port}"
            protocol = "tcp" if port_spec.protocol == "UDP" else "http"
            endpoints.append(
                Endpoint(
                    name=endpoint_name,
                    url=endpoint_url,
                    protocol=protocol,
                )
            )
        return endpoints

    def _endpoints_from_container_ports(self, host_ports: dict[int, int], *, target_name: str) -> list[Endpoint]:
        if not host_ports:
            return []
        endpoints: list[Endpoint] = []
        for container_port, host_port in host_ports.items():
            endpoint_url = self._url_for_port(
                target_name=target_name,
                container_port=container_port,
                host_port=host_port,
            )
            if endpoint_url is None:
                continue
            endpoints.append(
                Endpoint(
                    name=f"port-{container_port}",
                    url=endpoint_url,
                    protocol="http",
                )
            )
        return endpoints
