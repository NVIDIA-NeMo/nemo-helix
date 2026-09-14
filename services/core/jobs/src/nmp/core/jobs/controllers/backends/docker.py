# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import datetime
import hashlib
import io
import json
import logging
import os
import sys
import tarfile
import threading
import time
import uuid
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import docker.types
from docker.errors import APIError, DockerException, ImageNotFound, NotFound
from docker.models.containers import Container
from docker.types import LogConfig, Mount
from nemo_platform_plugin.auth import AuthContext as PluginAuthContext
from nemo_platform_plugin.capabilities import CapabilityUnavailableError, probe_docker
from nemo_platform_plugin.client.errors import NemoClientError
from nemo_platform_plugin.client.errors import NotFoundError as ClientNotFoundError
from nemo_platform_plugin.jobs.execution_profiles import (
    DockerJobExecutionProfile as PluginDockerJobExecutionProfile,
)
from nemo_platform_plugin.jobs.execution_profiles import (
    DockerJobExecutionProfileConfig as PluginDockerJobExecutionProfileConfig,
)
from nemo_platform_plugin.jobs.execution_profiles import (
    DockerJobNetworkConfig as PluginDockerJobNetworkConfig,
)
from nemo_platform_plugin.jobs.execution_profiles import (
    DockerJobStorageConfig as DockerJobStorageConfig,
)
from nemo_platform_plugin.jobs.execution_profiles import (
    DockerVolumeMount as DockerVolumeMount,
)
from nemo_platform_plugin.jobs.types import (
    PlatformJobStatusUpdateRequest,
    PlatformJobStepWithContext,
    PlatformJobTaskUpdate,
)
from nmp.common.auth import (
    AuthContext,
    WorkloadDelegationError,
    WorkloadDelegationScope,
    build_docker_opaque_workload_delegation,
    docker_delegation_name,
    get_workload_delegation_audience,
)
from nmp.common.config import get_platform_config, nmp_user_data_dir
from nmp.common.docker.gpu_pool import GPUAllocationError
from nmp.common.entities import EntityStoreError
from nmp.common.jobs.constants import (
    CONFIG_TASK_STORAGE_PATH_ENVVAR,
    DEFAULT_CONFIG_STORAGE_PATH,
    DEFAULT_NEMO_JOB_STEP_CONFIG_FILE_PATH,
    DEFAULT_TASK_STORAGE_PATH,
    EPHEMERAL_TASK_STORAGE_PATH_ENVVAR,
    NEMO_JOB_ATTEMPT_ID_ENVVAR,
    NEMO_JOB_FILESET_ENVVAR,
    NEMO_JOB_ID_ENVVAR,
    NEMO_JOB_SECRETS_ENVVAR,
    NEMO_JOB_STEP_CONFIG_FILE_NAME,
    NEMO_JOB_STEP_CONFIG_FILE_PATH_ENVVAR,
    NEMO_JOB_STEP_ENVVAR,
    NEMO_JOB_TASK_ENVVAR,
    NEMO_JOB_WORKSPACE_ENVVAR,
    PERSISTENT_JOB_STORAGE_PATH_ENVVAR,
    TERMINAL_EXIT_CODES,
)
from nmp.common.jobs.schemas import PlatformJobStatus
from nmp.common.observability import start_span_with_ctx
from nmp.common.resources import SharedResourceManager
from nmp.core.jobs.app.constants import (
    DEFAULT_VOLUME_PERMISSIONS_IMAGE,
    JOB_ATTEMPT_ID_LABEL,
    JOB_CONTROLLER_INSTANCE_ID_LABEL,
    JOB_EXECUTION_BACKEND_LABEL,
    JOB_EXECUTION_PROFILE_LABEL,
    JOB_ID_LABEL,
    JOB_MANAGED_BY_JOBS_CONTROLLER,
    JOB_MANAGED_BY_LABEL,
    JOB_STEP_ID_LABEL,
    JOB_STEP_NAME_LABEL,
    JOB_TASK_ID_LABEL,
    JOB_TYPE_JOB,
    JOB_TYPE_LABEL,
    JOB_TYPE_STORAGE_CLEANUP,
    JOB_USES_PERSISTENT_STORAGE_LABEL,
    JOB_WORKSPACE_ID_LABEL,
)
from nmp.core.jobs.app.ctx import JobContext
from nmp.core.jobs.app.providers import (
    ComputeResources,
    CPUExecutionProvider,
    GPUExecutionProvider,
)
from nmp.core.jobs.app.schemas import PlatformJobStepSpec
from nmp.core.jobs.controllers.backends.base import (
    NMP_JOB_LAUNCHER_OTLP_LOGS_ENDPOINT_ENVVAR,
    WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR,
    WORKLOAD_IDENTITY_TOKEN_FILE_PATH,
    WORKLOAD_IDENTITY_VOLUME_PATH,
    JobBackend,
    JobUpdate,
    get_job_runtime_shared_envvars,
    get_logs_endpoint_from_fileset,
    is_workload_identity_token_exchange_enabled,
    resolve_gpu_job_shm_size,
    resolve_task_image,
    staleness_error_message,
    validate_no_reserved_managed_job_environment_variable_names,
)
from nmp.core.jobs.controllers.backends.exceptions import (
    FailedToScheduleError,
    JobStorageError,
    ResourceAllocationError,
    SchedulingDeferred,
)
from nmp.core.jobs.controllers.backends.workload_tokens import (
    build_token_archive,
    create_authenticated_workload_delegation_store,
)
from opentelemetry import trace
from pydantic import Field

import docker

tracer = trace.get_tracer(__name__)
logger = logging.getLogger(__name__)
DOCKER_CONTAINER_START_WORKERS = 10
_WORKLOAD_DELEGATION_CLEANUP_ERRORS: tuple[type[Exception], ...] = (
    JobStorageError,
    WorkloadDelegationError,
    EntityStoreError,
    NemoClientError,
)


def k8s_shm_quantity_to_docker(quantity: str) -> str:
    """Convert a Kubernetes-style memory quantity to docker-py's shm_size string (e.g. '1g', '512m')."""
    q = quantity.strip()
    if q.endswith("Gi"):
        return f"{q[:-2]}g"
    if q.endswith("Mi"):
        return f"{q[:-2]}m"
    if q.endswith("G") and not q.endswith("Gi"):
        return q
    return q


NEMO_JOBS_IMAGE_REGISTRY_PASSWORD = os.getenv("NEMO_JOBS_IMAGE_REGISTRY_PASSWORD")
NEMO_JOBS_IMAGE_REGISTRY = os.getenv("NEMO_JOBS_IMAGE_REGISTRY")
NEMO_JOBS_IMAGE_REGISTRY_USER_NAME = os.getenv("NEMO_JOBS_IMAGE_REGISTRY_USER_NAME")
NEMO_JOBS_DEFAULT_DOCKER_NETWORK = os.getenv("NEMO_JOBS_DEFAULT_DOCKER_NETWORK", "host")

# Timeout for stopping Docker containers gracefully with SIGTERM before SIGKILL is sent.
# Default is 30 seconds which matches the Kubernetes default grace period for pod termination.
DOCKER_STOP_TIMEOUT = int(os.getenv("NEMO_JOBS_DEFAULT_DOCKER_STOP_TIMEOUT", "30"))
NMP_JOBS_DOCKER_OWNER_ID_ENVVAR = "NMP_JOBS_DOCKER_OWNER_ID"
DOCKER_WORKLOAD_IDENTITY_TOKEN_FILE_LABEL = "nmp.nvidia.com/workload_identity_token_file"
DOCKER_WORKLOAD_IDENTITY_VOLUME_LABEL = "nmp.nvidia.com/workload_identity_volume"


ProviderT = TypeVar("ProviderT", CPUExecutionProvider, GPUExecutionProvider)


# DockerVolumeMount and DockerJobStorageConfig are pure data shapes shared with
# the typed HTTP client — imported from the plugin leaf node (see imports).


def _resolve_jobs_controller_instance_id() -> str:
    configured = os.getenv(NMP_JOBS_DOCKER_OWNER_ID_ENVVAR)
    if configured:
        return configured

    owner_source = f"nmp-data-dir:{nmp_user_data_dir().expanduser().resolve()}"
    return hashlib.sha256(owner_source.encode("utf-8")).hexdigest()[:32]


@dataclass(frozen=True, slots=True)
class DockerTimestampParseResult:
    parsed: datetime.datetime | None
    parse_error: str | None
    is_zero: bool


@dataclass(frozen=True, slots=True)
class DockerContainerCreateResult:
    container: Container
    image_source: str
    duration_seconds: float


# Server-side override: the default network name comes from the
# ``NEMO_JOBS_DEFAULT_DOCKER_NETWORK`` env var (used by quickstart and e2e).
# No docstring on purpose — a docstring would surface as the schema
# ``description``, and this type carries none on the wire.
class DockerJobNetworkConfig(PluginDockerJobNetworkConfig):
    job_container_network: str = Field(
        default=NEMO_JOBS_DEFAULT_DOCKER_NETWORK, description="Docker network for the job container"
    )


class DockerJobExecutionProfileConfig(PluginDockerJobExecutionProfileConfig):
    """Configuration for Docker Job execution profile."""

    # ``networking`` re-typed to the server ``DockerJobNetworkConfig`` (env-var default).
    networking: DockerJobNetworkConfig = Field(
        default_factory=DockerJobNetworkConfig, description="Docker networking configuration"
    )


class DockerJobExecutionProfile(PluginDockerJobExecutionProfile):
    """
    Execution configuration for a Docker Job.
    This is used to define the executor type, provider, profile, and any additional configuration
    required for the executor to run the job on Docker
    """

    config: DockerJobExecutionProfileConfig = Field(description="Additional configuration for the docker executor")


class DockerJobBackend(JobBackend[ProviderT, DockerJobExecutionProfileConfig], Generic[ProviderT]):
    BACKEND_NAME: str = "docker"

    def init(self) -> None:
        self._jobs_controller_instance_id = _resolve_jobs_controller_instance_id()
        self._container_start_admission = threading.BoundedSemaphore(DOCKER_CONTAINER_START_WORKERS)
        self._container_run_threadpool = ThreadPoolExecutor(max_workers=DOCKER_CONTAINER_START_WORKERS)
        # Short probe first — avoid docker.from_env(timeout=180) hanging when the
        # daemon is down. CapabilityUnavailableError is soft-skipped by the registry.
        probe = probe_docker()
        if not probe.available:
            raise CapabilityUnavailableError(probe.detail or "Docker daemon is unavailable")
        self._client = docker.from_env(timeout=180)
        if NEMO_JOBS_IMAGE_REGISTRY:
            logger.info(
                f"Got image registry config, logging into {NEMO_JOBS_IMAGE_REGISTRY} with {NEMO_JOBS_IMAGE_REGISTRY_USER_NAME}"
            )
            self._client.login(
                username=NEMO_JOBS_IMAGE_REGISTRY_USER_NAME,
                password=NEMO_JOBS_IMAGE_REGISTRY_PASSWORD,
                registry=NEMO_JOBS_IMAGE_REGISTRY,
            )
        self._workload_delegation_store = create_authenticated_workload_delegation_store(self._nmp_sdk)

    def shutdown(self) -> None:
        self._container_run_threadpool.shutdown(wait=True)
        self._client.close()

    def _create_container_once(self, container_args: dict, *, image_source: str) -> DockerContainerCreateResult:
        create_started_at = time.monotonic()
        container = self._client.containers.create(**container_args)
        return DockerContainerCreateResult(
            container=container,
            image_source=image_source,
            duration_seconds=time.monotonic() - create_started_at,
        )

    def _create_container_with_image_pull(self, *, container_args: dict, image: str) -> DockerContainerCreateResult:
        try:
            return self._create_container_once(container_args, image_source="local")
        except ImageNotFound:
            self._client.images.pull(image)
            return self._create_container_once(container_args, image_source="pulled")

    def _is_workload_identity_enabled(self) -> bool:
        return is_workload_identity_token_exchange_enabled()

    def _should_enable_workload_identity_for_step(self, step: PlatformJobStepWithContext) -> bool:
        if not self._is_workload_identity_enabled():
            return False
        if step.auth_context is None:
            logger.debug(
                "Docker workload identity is enabled, but the job step has no auth_context",
                extra={"workspace": step.workspace, "job": step.job, "step": step.name},
            )
            return False
        return True

    @staticmethod
    def _require_step_spec(step: PlatformJobStepWithContext) -> PlatformJobStepSpec:
        if step.step_spec is None:
            raise ValueError("Docker job step requires a step_spec")
        return step.step_spec

    def _prepare_workload_identity_for_step(
        self,
        *,
        step: PlatformJobStepWithContext,
        workload_identity_volume_name: str,
    ) -> str:
        if step.auth_context is None:
            raise JobStorageError("Docker workload identity requires a job auth_context for on-behalf-of delegation")

        auth_context = PluginAuthContext.model_validate(step.auth_context.model_dump(mode="python", exclude_none=True))
        delegation, proof_token = build_docker_opaque_workload_delegation(
            scope=WorkloadDelegationScope(
                workload_workspace=step.workspace,
                workload_kind="job",
                workload_instance_id=step.job,
            ),
            workload_audience=get_workload_delegation_audience(),
            workload_generation=f"{step.attempt_id}/{step.id}",
            job_id=step.job,
            attempt_id=step.attempt_id,
            step_id=step.id,
            auth_context=auth_context,
            ttl_seconds_active=self._execution_profile_config.ttl_seconds_active,
        )
        self._workload_delegation_store.register(
            delegation,
            require_opaque_subject_token_hash=True,
        )

        token_written = False
        try:
            self._write_workload_identity_subject_token(workload_identity_volume_name, proof_token)
            token_written = True
        finally:
            if not token_written:
                self._try_revoke_workload_delegation(
                    delegation.name,
                    reason="token provisioning failure",
                    cause=sys.exception(),
                )

        return delegation.name

    def _try_revoke_workload_delegation(
        self,
        delegation_name: str | None,
        *,
        reason: str,
        cause: BaseException | None = None,
    ) -> Exception | None:
        if not delegation_name:
            return None

        try:
            self._workload_delegation_store.revoke(delegation_name)
        except _WORKLOAD_DELEGATION_CLEANUP_ERRORS as exc:
            logger.exception(
                "Failed to revoke Docker workload delegation",
                extra={"delegation_name": delegation_name, "reason": reason},
            )
            if cause is not None:
                cause.add_note(f"Failed to revoke Docker workload delegation {delegation_name} after {reason}: {exc!r}")
            return exc
        return None

    def _workload_delegation_name_for_step(self, step: PlatformJobStepWithContext) -> str:
        return docker_delegation_name(
            workload_workspace=step.workspace,
            job_id=step.job,
            attempt_id=step.attempt_id,
            step_id=step.id,
        )

    @staticmethod
    def _workload_delegation_name_from_container(container: Container) -> str | None:
        labels = container.labels or {}
        if DOCKER_WORKLOAD_IDENTITY_TOKEN_FILE_LABEL not in labels:
            return None

        workload_workspace = labels.get(JOB_WORKSPACE_ID_LABEL)
        job_id = labels.get(JOB_ID_LABEL)
        attempt_id = labels.get(JOB_ATTEMPT_ID_LABEL)
        step_id = labels.get(JOB_STEP_ID_LABEL)
        if not isinstance(workload_workspace, str) or not workload_workspace:
            return None
        if not isinstance(job_id, str) or not job_id:
            return None
        if not isinstance(attempt_id, str) or not attempt_id:
            return None
        if not isinstance(step_id, str) or not step_id:
            return None

        return docker_delegation_name(
            workload_workspace=workload_workspace,
            job_id=job_id,
            attempt_id=attempt_id,
            step_id=step_id,
        )

    def _write_workload_identity_subject_token(self, volume_name: str, token: str) -> None:
        storage_config = self._execution_profile_config.storage
        permissions_image = (
            storage_config.volume_permissions_image if storage_config is not None else DEFAULT_VOLUME_PERMISSIONS_IMAGE
        )
        token_volume_path = "/workload-identity-vol"
        container_name = f"workload-token-write-{uuid.uuid4().hex[:8]}"
        finalize_token_command = (
            f"mv {token_volume_path}/token.tmp {token_volume_path}/token && chmod 0444 {token_volume_path}/token"
        )
        container_args = {
            "name": container_name,
            "image": permissions_image,
            "command": [
                "sh",
                "-c",
                finalize_token_command,
            ],
            "volumes": {volume_name: {"bind": token_volume_path, "mode": "rw"}},
            "labels": {JOB_MANAGED_BY_LABEL: JOB_MANAGED_BY_JOBS_CONTROLLER},
        }
        try:
            container = self._create_container_with_image_pull(
                container_args=container_args, image=permissions_image
            ).container
        except DockerException as exc:
            raise JobStorageError("Error creating workload identity token writer container") from exc

        try:
            container.put_archive(path=token_volume_path, data=build_token_archive(token))
            container.start()
            exit_status = container.wait()
            if exit_status["StatusCode"] != 0:
                raise JobStorageError(
                    f"Workload identity token writer exited with non-zero status {exit_status['StatusCode']}"
                )
        except DockerException as exc:
            raise JobStorageError("Error writing workload identity subject token") from exc
        finally:
            try:
                container.remove()
            except DockerException:
                logger.debug("Failed to remove workload identity token writer container", exc_info=True)

    @staticmethod
    def get_label_from_container(container: Container, label: str) -> str:
        return container.labels[label]

    @staticmethod
    def _is_container_managed_by_jobs_controller(container: Container) -> bool:
        """Return True if the container has the jobs-controller managed-by label."""
        labels = container.labels or {}
        return labels.get(JOB_MANAGED_BY_LABEL) == JOB_MANAGED_BY_JOBS_CONTROLLER

    def _base_controller_labels(self) -> dict[str, str]:
        return {
            JOB_MANAGED_BY_LABEL: JOB_MANAGED_BY_JOBS_CONTROLLER,
            JOB_CONTROLLER_INSTANCE_ID_LABEL: self._jobs_controller_instance_id,
            JOB_EXECUTION_BACKEND_LABEL: self.BACKEND_NAME,
            JOB_EXECUTION_PROFILE_LABEL: self._profile_name,
        }

    def _is_container_owned_by_this_controller(self, container: Container) -> bool:
        labels = container.labels or {}
        return (
            labels.get(JOB_MANAGED_BY_LABEL) == JOB_MANAGED_BY_JOBS_CONTROLLER
            and labels.get(JOB_CONTROLLER_INSTANCE_ID_LABEL) == self._jobs_controller_instance_id
        )

    def _cleanup_container_filters(self) -> dict[str, list[str]]:
        return {
            "label": [
                f"{JOB_MANAGED_BY_LABEL}={JOB_MANAGED_BY_JOBS_CONTROLLER}",
                f"{JOB_CONTROLLER_INSTANCE_ID_LABEL}={self._jobs_controller_instance_id}",
                f"{JOB_EXECUTION_BACKEND_LABEL}={self.BACKEND_NAME}",
                f"{JOB_EXECUTION_PROFILE_LABEL}={self._profile_name}",
            ]
        }

    def _release_step_resources(self, step: PlatformJobStepWithContext, *, reason: str) -> None:
        """No-op resource hook for Docker backends without per-step resources."""
        del step, reason
        return

    def _release_container_resources(self, container: Container, *, reason: str) -> None:
        """No-op resource hook for Docker backends without per-container resources."""
        del container, reason
        return

    def job_storage_subpath(self, workspace: str, job: str) -> str:
        return f"jobs/{workspace}/{job}"

    def task_storage_volume_name(self, workspace: str, job: str, task: str) -> str:
        """Generate a unique volume name for task storage space."""
        return f"task-storage-{workspace}-{job}-{task}"

    def task_config_volume_name(self, workspace: str, job: str, task: str) -> str:
        """Generate a unique volume name for task config space."""
        return f"task-config-{workspace}-{job}-{task}"

    def task_workload_identity_volume_name(self, workspace: str, job: str, task: str) -> str:
        """Generate a unique volume name for workload identity token material."""
        return f"task-workload-identity-{workspace}-{job}-{task}"

    def cleanup_task_storage_volumes(self, workspace: str, job: str, task: str) -> None:
        """Remove the task storage volume after the container is done."""

        volumes_to_delete = [
            self.task_storage_volume_name(workspace, job, task),
            self.task_config_volume_name(workspace, job, task),
            self.task_workload_identity_volume_name(workspace, job, task),
        ]
        for volume_name in volumes_to_delete:
            try:
                volume = self._client.volumes.get(volume_name)
                volume.remove(force=True)
                logger.debug("Cleaned up task storage volume", extra={"volume_name": volume_name})
            except NotFound:
                logger.warning(
                    "Task storage volume not found, may have been already cleaned up",
                    extra={"volume_name": volume_name},
                )
            except DockerException:
                logger.exception("Failed to clean up task storage volume", extra={"volume_name": volume_name})

    def cleanup_job_persistent_storage(self, workspace: str, job: str) -> None:
        """Remove persistent job storage from the shared volume after successful job completion."""
        storage_config = self._execution_profile_config.storage
        if storage_config is None or storage_config.volume_name == "":
            logger.debug(
                "No persistent storage configured, skipping cleanup for job", extra={"workspace": workspace, "job": job}
            )
            return

        job_storage_subpath = self.job_storage_subpath(workspace, job)
        cleanup_script = f"""#!/bin/sh
set -ex
# Remove the job's storage directory
if [ -d "/vol/{job_storage_subpath}" ]; then
    rm -rf "/vol/{job_storage_subpath}"
    echo "Removed persistent storage for job {workspace}/{job}"
else
    echo "Storage directory not found, may have been already cleaned up"
fi
"""

        fileobj = io.BytesIO()
        with tarfile.open(fileobj=fileobj, mode="w") as tar:
            info = tarfile.TarInfo(name="cleanup.sh")
            info.size = len(cleanup_script)
            info.mode = 0o755
            tar.addfile(info, io.BytesIO(cleanup_script.encode("utf-8")))
        fileobj.seek(0)

        volumes = {storage_config.volume_name: {"bind": "/vol", "mode": "rw"}}

        labels = {
            **self._base_controller_labels(),
            JOB_WORKSPACE_ID_LABEL: workspace,
            JOB_ID_LABEL: job,
            JOB_TYPE_LABEL: JOB_TYPE_STORAGE_CLEANUP,
        }
        container_args = {
            "name": f"job-cleanup-{workspace}-{job}-{uuid.uuid4().hex[:8]}",
            "image": storage_config.volume_permissions_image,
            "command": ["sh", "/cleanup.sh"],
            "volumes": volumes,
            "labels": labels,
        }

        try:
            container = self._create_container_with_image_pull(
                container_args=container_args, image=storage_config.volume_permissions_image
            ).container
        except APIError:
            logger.error(
                "Error creating cleanup container for job", extra={"workspace": workspace, "job": job}, exc_info=True
            )
            return

        container.put_archive(path="/", data=fileobj)
        try:
            container.start()
        except APIError:
            logger.error(
                "Error starting cleanup container for job", extra={"workspace": workspace, "job": job}, exc_info=True
            )
            self.cleanup_container(container)
            return

        try:
            exit_status = container.wait()
            if exit_status["StatusCode"] != 0:
                logger.warning(
                    "Cleanup container for job exited with non-zero status",
                    extra={"workspace": workspace, "job": job, "exit_code": exit_status["StatusCode"]},
                )
        except APIError:
            logger.error(
                "Error waiting for cleanup container for job", extra={"workspace": workspace, "job": job}, exc_info=True
            )
        finally:
            self.cleanup_container(container)

    def cleanup_container(self, container: Container) -> None:
        """Remove a Docker container. Only removes containers owned by this jobs controller."""
        if not self._is_container_owned_by_this_controller(container):
            logger.warning(
                "Skipping container remove (not owned by this jobs-controller)",
                extra={
                    "container_id": container.id[:16] if container.id else "unknown",
                    "owner_label": (container.labels or {}).get(JOB_CONTROLLER_INSTANCE_ID_LABEL),
                },
            )
            return
        try:
            container.remove(force=True)
            logger.debug("Removed container", extra={"container_id": container.id[:16]})
        except NotFound:
            logger.warning(
                "Container not found, may have been already removed", extra={"container_id": container.id[:16]}
            )
        except DockerException:
            logger.exception("Failed to remove container", extra={"container_id": container.id[:16]})

    def cleanup_container_network(self, container: Container) -> None:
        """Remove a Docker network."""

        # First check if the container is attached to the network
        network_name = self._execution_profile_config.networking.job_container_network
        attrs = container.attrs or {}
        if network_name not in attrs.get("NetworkSettings", {}).get("Networks", {}):
            logger.debug(
                "Container already detached from network",
                extra={"container_id": container.id[:16], "network_name": network_name},
            )
            return

        # If it is, disconnect it
        network = self._client.networks.get(network_name)
        network.disconnect(container)
        logger.debug(
            "Removed network from container", extra={"container_id": container.id[:16], "network_name": network_name}
        )

    def get_mounts(
        self,
        workspace: str,
        job: str,
        job_volume_name: str,
        job_volume_path: str,
        config_volume_name: str,
        config_volume_path: str,
        task_volume_name: str,
        task_volume_path: str,
        workload_identity_volume_name: str | None = None,
        additional_volume_mounts: list[DockerVolumeMount] | None = None,
    ) -> list[Mount]:
        """
        Create `Mount` objects that attach persistent storage to the container.
        We need the more advanced `mounts` over `volumes` so we can utilize the `Subpath` option.
        This allows us to mount in a subpath of an existing volume, ensuring that the mount
        can't see more than we explicitly allow. This is essential so one job can't see
        the activity of another one.
        """

        task_storage_mount = docker.types.Mount(
            type="volume",
            source=task_volume_name,
            target=task_volume_path,
        )
        config_storage_mount = docker.types.Mount(
            type="volume",
            source=config_volume_name,
            target=config_volume_path,
        )

        mounts = [
            task_storage_mount,
            config_storage_mount,
        ]
        if workload_identity_volume_name is not None:
            workload_identity_mount = docker.types.Mount(
                type="volume",
                source=workload_identity_volume_name,
                target=WORKLOAD_IDENTITY_VOLUME_PATH,
                read_only=True,
            )
            mounts.append(workload_identity_mount)

        if job_volume_path != "":
            job_storage_mount = docker.types.Mount(
                type="volume",
                source=job_volume_name,
                target=job_volume_path,
            )
            job_storage_mount["VolumeOptions"] = {"Subpath": self.job_storage_subpath(workspace, job)}
            mounts.append(job_storage_mount)

        if additional_volume_mounts:
            for vol_mount in additional_volume_mounts:
                mount = docker.types.Mount(
                    type=vol_mount.kind,
                    source=vol_mount.volume_name,
                    target=vol_mount.mount_path,
                )
                if vol_mount.options:
                    mount["VolumeOptions"] = vol_mount.options
                mounts.append(mount)

        return mounts

    def ensure_job_storage(
        self,
        job_storage_volume_name: str,
        permissions_image: str,
        workspace: str,
        job: str,
        task: str,
        step_config_json: str,
        additional_volumes_mounts: list[DockerVolumeMount] | None = None,
        workload_identity_volume_name: str | None = None,
    ) -> None:
        """
        Ensure Docker volumes exist for the job and task, with proper permissions.
        This creates:
        1. A task-specific storage volume for temporary task storage
        2. Job-specific subpaths in the shared job volume for persistent storage, if requested

        Both volumes are configured with proper permissions so non-root-user containers can access them.
        """

        task_vol = "/task-vol"
        task_volume_name = self.task_storage_volume_name(workspace, job, task)
        try:
            self._client.volumes.create(task_volume_name)
            logger.debug("Created task storage volume", extra={"volume_name": task_volume_name})
        except DockerException as exc:
            raise JobStorageError(f"Error creating task storage volume {task_volume_name}") from exc

        # create a config volume for placing config files if needed
        config_vol = "/config-vol"
        config_volume_name = self.task_config_volume_name(workspace, job, task)
        try:
            self._client.volumes.create(config_volume_name)
            logger.debug("Created task config volume", extra={"volume_name": config_volume_name})
        except DockerException as exc:
            raise JobStorageError(f"Error creating task config volume {config_volume_name}") from exc

        if workload_identity_volume_name is not None:
            try:
                self._client.volumes.create(workload_identity_volume_name)
                logger.debug("Created workload identity volume", extra={"volume_name": workload_identity_volume_name})
            except DockerException as exc:
                raise JobStorageError(
                    f"Error creating workload identity volume {workload_identity_volume_name}"
                ) from exc

        script = f"""#!/bin/sh
set -ex
chmod -R 777 {task_vol}
cat > {config_vol}/{NEMO_JOB_STEP_CONFIG_FILE_NAME} << 'EOF'
{step_config_json}
EOF
cat {config_vol}/{NEMO_JOB_STEP_CONFIG_FILE_NAME}
chmod -R 777 {config_vol}
"""
        volumes = {
            task_volume_name: {"bind": task_vol, "mode": "rw"},
            config_volume_name: {"bind": config_vol, "mode": "rw"},
        }
        if workload_identity_volume_name is not None:
            volumes[workload_identity_volume_name] = {"bind": "/workload-identity-vol", "mode": "rw"}
            script += """
mkdir -p /workload-identity-vol
chmod -R 777 /workload-identity-vol
"""

        if job_storage_volume_name != "":
            job_vol = "/job-vol"
            storage_subpath = self.job_storage_subpath(workspace, job)
            script += f"""
mkdir -p {job_vol}/{storage_subpath}
chmod -R 777 {job_vol}/{storage_subpath}
"""
            try:
                self._client.volumes.get(job_storage_volume_name)
                logger.debug(
                    "Volume exists for job",
                    extra={"volume_name": job_storage_volume_name, "workspace": workspace, "job": job},
                )
            except NotFound:
                logger.info(
                    "Could not find storage volume, creating one now", extra={"volume_name": job_storage_volume_name}
                )
                self._client.volumes.create(job_storage_volume_name)
            volumes[job_storage_volume_name] = {"bind": job_vol, "mode": "rw"}

        if additional_volumes_mounts:
            for vol_mount in additional_volumes_mounts:
                # Check for existence of the additional volume
                try:
                    self._client.volumes.get(vol_mount.volume_name)
                    logger.debug("Additional volume already exists", extra={"volume_name": vol_mount.volume_name})
                except NotFound as e:
                    if vol_mount.allow_create_volume:
                        logger.info(
                            "Could not find additional volume, creating one now",
                            extra={"volume_name": vol_mount.volume_name},
                        )
                        self._client.volumes.create(vol_mount.volume_name)
                    else:
                        raise JobStorageError(f"Additional volume {vol_mount.volume_name} not found") from e

                volumes[vol_mount.volume_name] = {"bind": vol_mount.mount_path, "mode": "rw"}

        script += "\necho 'Job init completed.'\n"

        fileobj = io.BytesIO()
        with tarfile.open(fileobj=fileobj, mode="w") as tar:
            info = tarfile.TarInfo(name="job-init.sh")
            info.size = len(script)
            info.mode = 0o755
            tar.addfile(info, io.BytesIO(script.encode("utf-8")))
        fileobj.seek(0)

        container_args = {
            "name": f"job-init-{workspace}-{job}-{task}",
            "image": permissions_image,
            "command": ["sh", "/job-init.sh"],
            "volumes": volumes,
        }
        try:
            container = self._create_container_with_image_pull(
                container_args=container_args, image=permissions_image
            ).container
        except DockerException as exc:
            raise JobStorageError(f"Error creating job init container with image {permissions_image}") from exc

        try:
            container.put_archive(path="/", data=fileobj)
            container.start()
        except DockerException as exc:
            self.cleanup_task_storage_volumes(workspace, job, task)
            raise JobStorageError("Error starting job init container") from exc

        try:
            exit_status = container.wait()
            if exit_status["StatusCode"] != 0:
                self.cleanup_task_storage_volumes(workspace, job, task)
                raise JobStorageError(f"Job init container exited with non-zero status {exit_status['StatusCode']}")
        except DockerException as exc:
            self.cleanup_task_storage_volumes(workspace, job, task)
            raise JobStorageError("Error waiting for job init container") from exc

        try:
            container.remove()
        except DockerException as e:
            raise JobStorageError("Failed to remove job init container") from e

    def schedule_single_container(
        self,
        executor_config: ProviderT,
        step: PlatformJobStepWithContext,
    ) -> JobUpdate:
        step_spec = self._require_step_spec(step)
        platform_config = get_platform_config()

        # Profile-level env vars first (e.g. HOME=/tmp); system, step, and shared env override these
        env = self._execution_profile_config.env.copy()
        validate_no_reserved_managed_job_environment_variable_names(
            (envvar.name for envvar in step_spec.environment or []),
            source="Job step environment keys",
        )

        # identify the task using a uuid.  In docker, there's only one task per step.
        # because parallelism and completions are not supported.
        task_id = f"task-{uuid.uuid4().hex}"
        env.update(
            {
                NEMO_JOB_ID_ENVVAR: step.job,
                NEMO_JOB_ATTEMPT_ID_ENVVAR: step.attempt_id,
                NEMO_JOB_STEP_ENVVAR: step.name,
                NEMO_JOB_TASK_ENVVAR: task_id,
                NEMO_JOB_WORKSPACE_ENVVAR: step.workspace,
                NEMO_JOB_FILESET_ENVVAR: step.fileset,
                EPHEMERAL_TASK_STORAGE_PATH_ENVVAR: DEFAULT_TASK_STORAGE_PATH,
                CONFIG_TASK_STORAGE_PATH_ENVVAR: DEFAULT_CONFIG_STORAGE_PATH,
                NEMO_JOB_STEP_CONFIG_FILE_PATH_ENVVAR: DEFAULT_NEMO_JOB_STEP_CONFIG_FILE_PATH,
                # Private env vars for jobs-launcher to export captured logs.
                NMP_JOB_LAUNCHER_OTLP_LOGS_ENDPOINT_ENVVAR: get_logs_endpoint_from_fileset(
                    platform_config,
                    step.workspace,
                    step.fileset,
                ),
                # Inject secret environment variable mappings for the jobs-launcher to fetch
                NEMO_JOB_SECRETS_ENVVAR: self.get_secrets_environment_variable_for_injection(step),
            }
        )
        workload_identity_enabled = self._should_enable_workload_identity_for_step(step)
        if workload_identity_enabled:
            env[WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR] = WORKLOAD_IDENTITY_TOKEN_FILE_PATH

        # Set auth context env var for job containers to make authenticated API calls
        if step.auth_context and not workload_identity_enabled:
            sdk_auth_context = step.auth_context
            auth_context = AuthContext.model_validate(sdk_auth_context.model_dump(mode="python", exclude_none=True))
            principal = auth_context.to_principal()
            env_var_dict = principal.get_env_var()
            for name, value in env_var_dict.items():
                env[name] = value

        step_config_json = json.dumps(step_spec.config)

        job_storage_mount = ""
        task_storage_mount = DEFAULT_TASK_STORAGE_PATH

        # Update the step container's environment variables for non-secret values
        if step_spec.environment:
            for envvar in step_spec.environment:
                if envvar.value is not None:
                    # If the job has requested persistent job storage path, capture it for use when constructing the volume mount.
                    if envvar.name == PERSISTENT_JOB_STORAGE_PATH_ENVVAR:
                        job_storage_mount = envvar.value

                    # The job has explicitly overridden the mount path for task storage.
                    # Since this fields has already been set via environment variables, we should update the appropriate variable instead.
                    elif envvar.name == EPHEMERAL_TASK_STORAGE_PATH_ENVVAR:
                        task_storage_mount = envvar.value

                    env[envvar.name] = envvar.value

        # Thread through shared platform envvars to the job
        # Note: address_override defaults to None, which triggers automatic loopback detection
        env.update(get_job_runtime_shared_envvars(platform_config))

        log_config = LogConfig(
            type=LogConfig.types.JSON,
            config={
                "labels": ",".join(
                    [JOB_WORKSPACE_ID_LABEL, JOB_ID_LABEL, JOB_ATTEMPT_ID_LABEL, JOB_STEP_NAME_LABEL, JOB_TASK_ID_LABEL]
                )
            },
        )

        if not self._container_start_admission.acquire(blocking=False):
            logger.debug(
                "Docker start admission full, deferring scheduling",
                extra={"job": step.job, "step": step.name},
            )
            raise SchedulingDeferred("Docker start worker capacity is full")

        # The admission slot is owned by this method until submit succeeds.
        # After that, run_container releases it when the start worker exits.
        container_args = None
        submitted = False
        try:
            container_args = self._prepare_container_args_for_start(
                executor_config=executor_config,
                step=step,
                task_id=task_id,
                env=env,
                log_config=log_config,
                job_storage_mount=job_storage_mount,
                task_storage_mount=task_storage_mount,
                step_config_json=step_config_json,
            )
            submitted_to_threadpool_at = time.monotonic()
            self._container_run_threadpool.submit(self.run_container, step, container_args, submitted_to_threadpool_at)
            submitted = True
        finally:
            if not submitted:
                if container_args is not None:
                    self._try_revoke_workload_delegation(
                        container_args.get("_nmp_workload_delegation_name"),
                        reason="container scheduling failure",
                        cause=sys.exception(),
                    )
                    self.cleanup_task_storage_volumes(step.workspace, step.job, task_id)
                self._release_step_resources(step, reason="schedule_failed_before_start")
                self._container_start_admission.release()
        logger.debug(
            "Docker run_container submitted",
            extra={
                "job": step.job,
                "step": step.name,
                "task": task_id,
            },
        )
        return JobUpdate(
            status=PlatformJobStatus.PENDING,
            status_details={"message": "Container schedule pending, checking for existing image and container"},
        )

    def _prepare_container_args_for_start(
        self,
        *,
        executor_config: ProviderT,
        step: PlatformJobStepWithContext,
        task_id: str,
        env: dict,
        log_config: LogConfig,
        job_storage_mount: str,
        task_storage_mount: str,
        step_config_json: str,
    ) -> dict:
        storage_config = self._execution_profile_config.storage
        job_volume_name = storage_config.volume_name if storage_config is not None else ""
        task_volume_name = self.task_storage_volume_name(workspace=step.workspace, job=step.job, task=task_id)
        config_volume_name = self.task_config_volume_name(workspace=step.workspace, job=step.job, task=task_id)
        workload_identity_enabled = self._should_enable_workload_identity_for_step(step)
        workload_identity_volume_name = (
            self.task_workload_identity_volume_name(workspace=step.workspace, job=step.job, task=task_id)
            if workload_identity_enabled
            else None
        )
        additional_volume_mounts = storage_config.additional_volume_mounts if storage_config else None
        ensure_storage_started_at = time.monotonic()
        self.ensure_job_storage(
            # if the job storage mount is not used, pass empty string to avoid creating unnecessary job storage volume
            job_storage_volume_name=job_volume_name if job_storage_mount != "" else "",
            permissions_image=storage_config.volume_permissions_image
            if storage_config is not None
            else DEFAULT_VOLUME_PERMISSIONS_IMAGE,
            workspace=step.workspace,
            job=step.job,
            task=task_id,
            additional_volumes_mounts=additional_volume_mounts,
            step_config_json=step_config_json,
            workload_identity_volume_name=workload_identity_volume_name,
        )
        workload_delegation_name = None
        logger.debug(
            "Docker job storage ensured",
            extra={
                "job": step.job,
                "step": step.name,
                "task": task_id,
                "duration_seconds": time.monotonic() - ensure_storage_started_at,
            },
        )

        labels = {
            **self._base_controller_labels(),
            JOB_WORKSPACE_ID_LABEL: step.workspace,
            JOB_ID_LABEL: step.job,
            JOB_ATTEMPT_ID_LABEL: step.attempt_id,
            JOB_STEP_NAME_LABEL: step.name,
            # identify the task using a uuid.  In docker, there's only one task per step.
            # because parallelism is not supported.
            JOB_TASK_ID_LABEL: task_id,
            JOB_STEP_ID_LABEL: step.id,
            JOB_TYPE_LABEL: JOB_TYPE_JOB,
        }

        # Mark container if it uses persistent storage so we can clean it up later
        if job_storage_mount != "":
            labels[JOB_USES_PERSISTENT_STORAGE_LABEL] = "true"
        else:
            labels[JOB_USES_PERSISTENT_STORAGE_LABEL] = "false"
        if workload_identity_volume_name is not None:
            labels[DOCKER_WORKLOAD_IDENTITY_TOKEN_FILE_LABEL] = WORKLOAD_IDENTITY_TOKEN_FILE_PATH
            labels[DOCKER_WORKLOAD_IDENTITY_VOLUME_LABEL] = workload_identity_volume_name

        task_image = resolve_task_image(
            executor_config.container.image, self._execution_profile_config.default_task_image
        )
        container_args = {
            "name": self.name_for_step(step),
            "entrypoint": executor_config.container.entrypoint or [],
            "command": executor_config.container.command or [],
            "image": task_image,
            "labels": labels,
            "log_config": log_config,
            "environment": env,
            "detach": True,
            "init": True,
            "mounts": self.get_mounts(
                workspace=step.workspace,
                job=step.job,
                job_volume_name=job_volume_name,
                job_volume_path=job_storage_mount,
                config_volume_name=config_volume_name,
                config_volume_path=DEFAULT_CONFIG_STORAGE_PATH,
                task_volume_name=task_volume_name,
                task_volume_path=task_storage_mount,
                workload_identity_volume_name=workload_identity_volume_name,
                additional_volume_mounts=additional_volume_mounts,
            ),
        }
        if workload_identity_volume_name is not None:
            workload_identity_prepared = False
            try:
                workload_delegation_name = self._prepare_workload_identity_for_step(
                    step=step,
                    workload_identity_volume_name=workload_identity_volume_name,
                )
                container_args["_nmp_workload_delegation_name"] = workload_delegation_name
                workload_identity_prepared = True
            finally:
                if not workload_identity_prepared:
                    self.cleanup_task_storage_volumes(step.workspace, step.job, task_id)

        container_args["network"] = self._execution_profile_config.networking.job_container_network
        configured = False
        try:
            configured_container_args = self.configure_container(container_args, executor_config)
            configured = True
            return configured_container_args
        finally:
            if not configured:
                self._try_revoke_workload_delegation(
                    workload_delegation_name,
                    reason="container scheduling failure",
                    cause=sys.exception(),
                )
                self.cleanup_task_storage_volumes(step.workspace, step.job, task_id)

    def _release_step_resources_after_scheduling_stop(
        self,
        step: PlatformJobStepWithContext,
        *,
        reason: str,
        created_container: Container | None = None,
    ) -> None:
        """Release step resources once any already-created container is gone."""
        if created_container is not None and not self._remove_container_before_resource_release(
            step, created_container, reason=reason
        ):
            return
        self._release_step_resources(step, reason=reason)

    def cancel_scheduling(
        self, step: PlatformJobStepWithContext, *, created_container: Container | None = None
    ) -> bool:
        """Check if the job step is cancelling or pausing, and update status accordingly."""
        updated_step = self.get_step_safe(step_name=step.name, job=step.job, workspace=step.workspace)
        if updated_step is None:
            logger.info(
                "Job step disappeared before Docker container start; cancelling scheduling",
                extra={"workspace": step.workspace, "job": step.job, "step": step.name},
            )
            self._release_step_resources_after_scheduling_stop(
                step, reason="step_missing_before_start", created_container=created_container
            )
            return True

        is_cancelling_or_pausing = updated_step.status in (
            PlatformJobStatus.CANCELLING,
            PlatformJobStatus.CANCELLED,
            PlatformJobStatus.PAUSING,
            PlatformJobStatus.PAUSED,
        )

        if is_cancelling_or_pausing:
            if updated_step.status in (
                PlatformJobStatus.CANCELLED,
                PlatformJobStatus.PAUSED,
            ):
                logger.info(
                    "Job step is already in terminal state, no update required", extra={"status": updated_step.status}
                )
                self._release_step_resources_after_scheduling_stop(
                    step, reason="step_terminal_before_start", created_container=created_container
                )
                return True  # Already in terminal state, no step updates needed

            status_details = {}
            if updated_step.status == PlatformJobStatus.PAUSING:
                status = PlatformJobStatus.PAUSED
                status_details["message"] = "Job is paused, not creating container"
            else:
                status = PlatformJobStatus.CANCELLED
                status_details["message"] = "Job is cancelled, not creating container"
            logger.info("Job step is not scheduling container", extra={"status": updated_step.status})
            try:
                self._jobs.update_job_step_status(
                    name=step.name,
                    workspace=step.workspace,
                    job=step.job,
                    body=PlatformJobStatusUpdateRequest(status=status, status_details=status_details),
                )
            except ClientNotFoundError:
                logger.info(
                    "Job step disappeared while Docker scheduling was stopping",
                    extra={"workspace": step.workspace, "job": step.job, "step": step.name},
                )
            self._release_step_resources_after_scheduling_stop(
                step, reason="step_stopped_before_start", created_container=created_container
            )
        return is_cancelling_or_pausing

    def get_jobs_launcher_binary(self) -> io.BytesIO | None:
        """Get a copy of the jobs-launcher binary as a tar stream to include in the job container."""
        jobs_launcher_stream = None
        if os.path.exists(self._execution_profile_config.launcher_tool_path):
            jobs_launcher_stream = io.BytesIO()
            with (
                tarfile.open(fileobj=jobs_launcher_stream, mode="w") as tar,
                open(self._execution_profile_config.launcher_tool_path, "rb") as f,
            ):
                file_data = f.read()
                tarinfo = tarfile.TarInfo(name="jobs-launcher")
                tarinfo.size = len(file_data)
                tarinfo.mode = 0o755  # Make it executable
                tar.addfile(tarinfo, io.BytesIO(file_data))
            jobs_launcher_stream.seek(0)
            return jobs_launcher_stream
        return None

    def run_container(
        self,
        step: PlatformJobStepWithContext,
        container_args: dict,
        submitted_to_threadpool_at: float | None = None,
    ):
        with start_span_with_ctx(
            tracer, "jobs_controller/docker_backend/run_container", JobContext(id=step.job, step_name=step.name)
        ):
            log_extra = {"job": step.job, "step": step.name}
            if submitted_to_threadpool_at is not None:
                log_extra["queue_delay_seconds"] = time.monotonic() - submitted_to_threadpool_at
            logger.debug("Docker run_container worker started", extra=log_extra)
            workload_delegation_name = container_args.get("_nmp_workload_delegation_name")
            try:
                self._run_container_in_thread(step, container_args)
            except FailedToScheduleError as e:
                logger.exception("Failed to schedule container for job step")
                self._try_revoke_workload_delegation(
                    workload_delegation_name,
                    reason="container scheduling failure",
                    cause=e,
                )
                status = PlatformJobStatus.ERROR
                try:
                    self._jobs.update_job_step_status(
                        name=step.name,
                        workspace=step.workspace,
                        job=step.job,
                        body=PlatformJobStatusUpdateRequest(status=status, error_details=e.error_details),
                    )
                except Exception:
                    logger.exception("Failed to persist scheduling error for job step")
                if self._remove_container_after_failed_schedule(step, reason="scheduling_error"):
                    self._release_step_resources(step, reason="scheduling_error")
            except Exception as e:
                logger.exception("Unexpected error while scheduling container for job step")
                self._try_revoke_workload_delegation(
                    workload_delegation_name,
                    reason="container scheduling failure",
                    cause=e,
                )
                if self._remove_container_after_failed_schedule(step, reason="unexpected_scheduling_error"):
                    self._release_step_resources(step, reason="unexpected_scheduling_error")
            finally:
                self._container_start_admission.release()

    def _update_step_schedule_status(
        self,
        step: PlatformJobStepWithContext,
        status: PlatformJobStatus,
        status_details: dict,
        message: str,
    ) -> None:
        status_details["message"] = message
        self._jobs.update_job_step_status(
            name=step.name,
            workspace=step.workspace,
            job=step.job,
            body=PlatformJobStatusUpdateRequest(status=status, status_details=status_details),
        )

    def _log_job_step_container_created(
        self,
        step: PlatformJobStepWithContext,
        container_args: dict,
        result: DockerContainerCreateResult,
    ) -> None:
        attrs = result.container.attrs or {}
        host_config = attrs.get("HostConfig", {})
        if not isinstance(host_config, dict):
            host_config = {}
        logger.debug(
            "Docker container create succeeded",
            extra={
                "job": step.job,
                "step": step.name,
                "container_name": result.container.name,
                "image": container_args["image"],
                "duration_seconds": result.duration_seconds,
                "image_source": result.image_source,
                "requested_auto_remove": container_args.get("auto_remove"),
                "host_config_auto_remove": host_config.get("AutoRemove"),
            },
        )

    def _create_job_step_container(
        self,
        *,
        step: PlatformJobStepWithContext,
        container_args: dict,
        status: PlatformJobStatus,
        status_details: dict,
        workload_delegation_name: str | None,
    ) -> Container | None:
        image = container_args["image"]
        try:
            result = self._create_container_once(container_args, image_source="local")
            # Container will create successfully only if image is found locally
            logger.info("Image found locally", extra={"image": image})
            self._log_job_step_container_created(step, container_args, result)
            return result.container
        except (ImageNotFound, NotFound):
            # Image not found locally, pull it
            logger.info("Image not found locally, pulling from registry", extra={"image": image})
            self._update_step_schedule_status(
                step,
                status,
                status_details,
                f"Pulling image {image} from registry",
            )
            try:
                pull_start = time.time()
                self._client.images.pull(image)
                pull_elapsed = time.time() - pull_start
                logger.info(
                    "Successfully pulled image for job step",
                    extra={"image": image, "pull_elapsed_s": f"{pull_elapsed:.1f}"},
                )
            except APIError as e:
                raise FailedToScheduleError(
                    "Failed to pull image",
                    error_details={"message": f"Failed to pull image {image}: {e}"},
                ) from e

            # If a request to pause or cancel came in while we were pulling down an image,
            # cancel scheduling the container
            logger.debug("Checking for cancellation or pausing before creating container after image pull")
            if self.cancel_scheduling(step):
                self._try_revoke_workload_delegation(
                    workload_delegation_name,
                    reason="container scheduling cancellation",
                )
                return None

            self._update_step_schedule_status(
                step,
                status,
                status_details,
                f"Creating container with image {image}",
            )

            # Now create it with the pulled container image
            logger.debug("Creating container for job step after pulling image")
            try:
                result = self._create_container_once(container_args, image_source="pulled")
                self._log_job_step_container_created(step, container_args, result)
                return result.container
            except APIError as e:
                raise FailedToScheduleError(
                    "Failed to create container for job step",
                    error_details={"message": f"Failed to create container after pulling image: {e}"},
                ) from e
        except APIError as e:
            raise FailedToScheduleError(
                "Failed to create container for job step",
                error_details={"message": f"Failed to create container: {e}"},
            ) from e

    def _remove_container_before_resource_release(
        self, step: PlatformJobStepWithContext, container: Container, *, reason: str
    ) -> bool:
        """Remove an owned created container before releasing resources tied to it."""
        if not self._is_container_owned_by_this_controller(container):
            logger.warning(
                "Skipping Docker resource-release cleanup for unowned container",
                extra={
                    "job": step.job,
                    "step": step.name,
                    "container_name": getattr(container, "name", None),
                    "owner_label": (getattr(container, "labels", None) or {}).get(JOB_CONTROLLER_INSTANCE_ID_LABEL),
                    "reason": reason,
                },
            )
            return False

        logger.info(
            "Removing Docker container before releasing scheduling resources",
            extra={
                "job": step.job,
                "step": step.name,
                "container_name": container.name,
                "reason": reason,
            },
        )
        try:
            container.remove(force=True)
        except NotFound:
            logger.info(
                "Container disappeared during Docker resource-release cleanup",
                extra={"job": step.job, "step": step.name, "container_name": container.name, "reason": reason},
            )
        except Exception:
            logger.exception(
                "Failed to remove container before Docker resource release",
                extra={"job": step.job, "step": step.name, "container_name": container.name, "reason": reason},
            )
            return False

        labels = getattr(container, "labels", None) or {}
        workspace = labels.get(JOB_WORKSPACE_ID_LABEL)
        job = labels.get(JOB_ID_LABEL)
        task = labels.get(JOB_TASK_ID_LABEL)
        if workspace and job and task:
            self.cleanup_task_storage_volumes(workspace, job, task)
        else:
            logger.debug(
                "Skipping task storage cleanup before Docker resource release because labels are incomplete",
                extra={"job": step.job, "step": step.name, "container_name": container.name, "reason": reason},
            )
        return True

    def _remove_container_after_failed_schedule(self, step: PlatformJobStepWithContext, *, reason: str) -> bool:
        container = self.get_container(step)
        if container is None:
            return True
        return self._remove_container_before_resource_release(step, container, reason=reason)

    def _run_container_in_thread(self, step: PlatformJobStepWithContext, container_args: dict):
        status_details = {}
        status = PlatformJobStatus.PENDING
        workload_delegation_name = container_args.pop("_nmp_workload_delegation_name", None)

        # If a request to pause or cancel came in while we were waiting for scheduling loop,
        # cancel scheduling the container
        logger.debug("Checking for cancellation or pausing before creating container")
        cancel_check_started_at = time.monotonic()
        if self.cancel_scheduling(step):
            logger.debug(
                "Docker pre-create cancellation check stopped scheduling",
                extra={
                    "job": step.job,
                    "step": step.name,
                    "duration_seconds": time.monotonic() - cancel_check_started_at,
                },
            )
            self._try_revoke_workload_delegation(
                workload_delegation_name,
                reason="container scheduling cancellation",
            )
            return
        logger.debug(
            "Docker pre-create cancellation check completed",
            extra={
                "job": step.job,
                "step": step.name,
                "duration_seconds": time.monotonic() - cancel_check_started_at,
            },
        )

        # For resuming containers, check if it already exists
        logger.debug("Checking for existing container for job step")
        get_container_started_at = time.monotonic()
        container = self.get_container(step)
        logger.debug(
            "Docker existing container lookup completed",
            extra={
                "job": step.job,
                "step": step.name,
                "found": container is not None,
                "duration_seconds": time.monotonic() - get_container_started_at,
            },
        )
        if container is not None:
            logger.info("Container already exists, not creating a new one", extra={"container_name": container.name})
        else:
            # Container not found, create it
            logger.debug("Creating container for job step")

            # Find the jobs launcher binary inside this running python container and also include it in the job container
            launcher_lookup_started_at = time.monotonic()
            jobs_launcher_stream = self.get_jobs_launcher_binary()
            logger.debug(
                "Docker jobs launcher lookup completed",
                extra={
                    "job": step.job,
                    "step": step.name,
                    "found": jobs_launcher_stream is not None,
                    "duration_seconds": time.monotonic() - launcher_lookup_started_at,
                },
            )
            if jobs_launcher_stream is not None:
                # Modify the container entrypoint and command to use the jobs-launcher
                original_entrypoint = container_args.get("entrypoint", [])
                container_args["entrypoint"] = ["/jobs-launcher", "run", "--"] + original_entrypoint
                logger.debug("Jobs launcher found, will be included in container")
            else:
                logger.warning(
                    "Jobs launcher not found, container will use original entrypoint",
                    extra={"launcher_tool_path": self._execution_profile_config.launcher_tool_path},
                )

            container = self._create_job_step_container(
                step=step,
                container_args=container_args,
                status=status,
                status_details=status_details,
                workload_delegation_name=workload_delegation_name,
            )
            if container is None:
                return

            # Insert the jobs-launcher into the container if the launcher exists
            if jobs_launcher_stream is not None:
                try:
                    put_archive_started_at = time.monotonic()
                    container.put_archive(path="/", data=jobs_launcher_stream)
                    logger.debug(
                        "Jobs launcher inserted into container successfully",
                        extra={
                            "job": step.job,
                            "step": step.name,
                            "container_name": container.name,
                            "duration_seconds": time.monotonic() - put_archive_started_at,
                        },
                    )
                except APIError as e:
                    raise FailedToScheduleError(
                        "Failed to insert jobs-launcher into container",
                        error_details={"message": f"Failed to add jobs-launcher to job container: {e}"},
                    ) from e

        # At this point we now have container created successfully, including the jobs launcher container if applicable
        logger.info("Created container for job step", extra={"container_name": container.name})

        # If a request to pause or cancel came in while we were waiting for scheduling loop,
        # cancel scheduling the container
        logger.debug("Checking for cancellation or pausing before starting container")
        pre_start_cancel_check_started_at = time.monotonic()
        if self.cancel_scheduling(step, created_container=container):
            logger.debug(
                "Docker pre-start cancellation check stopped scheduling",
                extra={
                    "job": step.job,
                    "step": step.name,
                    "container_name": container.name,
                    "duration_seconds": time.monotonic() - pre_start_cancel_check_started_at,
                },
            )
            self._try_revoke_workload_delegation(
                workload_delegation_name,
                reason="container scheduling cancellation",
            )
            return
        logger.debug(
            "Docker pre-start cancellation check completed",
            extra={
                "job": step.job,
                "step": step.name,
                "container_name": container.name,
                "duration_seconds": time.monotonic() - pre_start_cancel_check_started_at,
            },
        )

        try:
            # If no errors to this point, start the container
            status_details["message"] = "Starting container"
            pre_start_status_write_started_at = time.monotonic()
            self._jobs.update_job_step_status(
                name=step.name,
                workspace=step.workspace,
                job=step.job,
                body=PlatformJobStatusUpdateRequest(status=status, status_details=status_details),
            )
            logger.debug(
                "Docker pre-start status update succeeded",
                extra={
                    "job": step.job,
                    "step": step.name,
                    "container_name": container.name,
                    "duration_seconds": time.monotonic() - pre_start_status_write_started_at,
                },
            )
            started = False
            max_attempts = 3
            attempts = 0
            start_started_at = time.monotonic()
            while not started and attempts < max_attempts:
                attempts += 1
                try:
                    container.start()
                    started = True
                except APIError as e:
                    logger.warning(
                        "Attempt to start container failed",
                        extra={"attempt": attempts, "container_name": container.name},
                        exc_info=True,
                    )
                    # Raise the exception if max attempts reached
                    if attempts >= max_attempts:
                        raise e

                    time.sleep(5)  # brief pause before retrying
            logger.debug(
                "Started container for job step",
                extra={
                    "job": step.job,
                    "step": step.name,
                    "container_name": container.name,
                    "attempts": attempts,
                    "duration_seconds": time.monotonic() - start_started_at,
                },
            )
        except DockerException as e:
            raise FailedToScheduleError(
                f"Failed to start container {container.name} for job step",
                error_details={"message": f"Failed to start container: {e}"},
            ) from e

    def _sync(self, step: PlatformJobStepWithContext) -> JobUpdate:
        step_spec = self._require_step_spec(step)
        container: Container | None = self.get_container(step)
        if step.status == PlatformJobStatus.ACTIVE:
            if container is None:
                task_fallback = self._get_terminal_task_fallback_update(step)
                if task_fallback is not None:
                    return task_fallback
            if result := self.enforce_sync_ttl(
                step, self._execution_profile_config.ttl_seconds_active, container, before_active=False
            ):
                return result
            if container is not None and self.check_step_is_stale(step):
                message = staleness_error_message(step_spec.lifecycle.staleness_timeout_seconds)
                return self._kill_container_with_error(step, container, message)
            return self.sync_active(step, container)
        elif step.status == PlatformJobStatus.PENDING:
            if container is not None and container.status in ("running", "exited", "dead"):
                return self.sync_pending(step, container)
            if result := self.enforce_sync_ttl(
                step,
                self._execution_profile_config.ttl_seconds_before_active,
                container,
                before_active=True,
            ):
                return result
            return self.sync_pending(step, container)
        elif step.status == PlatformJobStatus.CANCELLING:
            # Handle cases where container is already gone, or was never created in the first place
            if container is None:
                return JobUpdate(
                    status=PlatformJobStatus.CANCELLED,
                    status_details={"message": "Container not found, job cancelled"},
                )
            return self.sync_stop_container(step, container)
        elif step.status == PlatformJobStatus.PAUSING:
            # Handle cases where container is already gone, or was never created in the first place
            if container is None:
                return JobUpdate(
                    status=PlatformJobStatus.PAUSED, status_details={"message": "Container not found, job paused"}
                )
            return self.sync_stop_container(step, container)
        else:
            raise ValueError(f"Unhandled job status during sync: {step.status}")

    def get_container(self, step: PlatformJobStepWithContext) -> Container | None:
        container_name = self.name_for_step(step)
        try:
            return self._client.containers.get(container_name)
        except NotFound:
            return None

    def enforce_sync_ttl(
        self,
        step: PlatformJobStepWithContext,
        ttl_seconds: int,
        container: Container | None,
        *,
        before_active: bool = False,
    ) -> JobUpdate | None:
        ttl_exceeded = (
            self.check_step_ttl_before_active(step, ttl_seconds)
            if before_active
            else self.check_step_ttl(step, ttl_seconds)
        )
        if not ttl_exceeded:
            return None

        message = f"Job timed out after reaching max TTL of {ttl_seconds} seconds"

        if container is None:
            return JobUpdate(
                status=PlatformJobStatus.ERROR,
                status_details={"message": message},
                error_details={"message": message},
            )

        return self._kill_container_with_error(step, container, message)

    def _kill_container_with_error(
        self, step: PlatformJobStepWithContext, container: Container, message: str
    ) -> JobUpdate:
        """Kill a managed container, update its task to ERROR, and return a JobUpdate."""
        status_details = {"message": message}
        error_details = {"message": message}

        if not self._is_container_owned_by_this_controller(container):
            logger.warning(
                "Skipping container kill (not owned by this jobs-controller)",
                extra={
                    "container_name": container.name,
                    "owner_label": (container.labels or {}).get(JOB_CONTROLLER_INSTANCE_ID_LABEL),
                },
            )
            return JobUpdate(
                status=PlatformJobStatus.ERROR,
                status_details=status_details,
                error_details=error_details,
            )

        try:
            container.kill()
        except APIError as e:
            if e.status_code == 409:
                logger.warning("Container already stopping or stopped", extra={"container_name": container.name})
            else:
                raise

        task_id = self.get_label_from_container(container, JOB_TASK_ID_LABEL)
        self._jobs.update_job_step_task(
            name=task_id,
            workspace=step.workspace,
            job=step.job,
            step=step.name,
            body=PlatformJobTaskUpdate(
                status=PlatformJobStatus.ERROR,
                status_details=status_details,
                error_details=error_details,
            ),
        )
        logger.info(
            "Updated task",
            extra={
                "job": step.job,
                "step_name": step.name,
                "task_id": task_id,
                "status": PlatformJobStatus.ERROR,
                "error_details": error_details,
            },
        )
        return JobUpdate(status=PlatformJobStatus.ERROR, status_details=status_details, error_details=error_details)

    def sync_pending(self, step: PlatformJobStepWithContext, container: Container | None) -> JobUpdate:
        if container is None:
            # Job doesn't exist yet
            return JobUpdate(
                status=PlatformJobStatus.PENDING,
                # status details are not set in this case to avoid overwriting any existing details
                # propagated from the scheduling loop
            )
        else:
            return self.create_step_update(step, container)

    def _get_terminal_task_fallback_update(self, step: PlatformJobStepWithContext) -> JobUpdate | None:
        """Return the latest persisted terminal task status for a missing Docker container."""
        try:
            tasks = self._jobs.list_job_step_tasks(
                name=step.name,
                job=step.job,
                workspace=step.workspace,
            ).data()
        except Exception:
            logger.warning(
                "Failed to fetch tasks for Docker missing-container fallback",
                extra={"job": step.job, "step": step.name, "workspace": step.workspace},
            )
            return None

        if not tasks.data:
            return None

        latest_task = max(
            tasks.data,
            key=lambda task: (
                task.updated_at or task.created_at or datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)
            ),
        )
        status = PlatformJobStatus(latest_task.status)
        if not status.is_terminal():
            return None

        return JobUpdate(
            status=status,
            status_details=latest_task.status_details,
            error_details=latest_task.error_details or {},
        )

    def sync_active(self, step: PlatformJobStepWithContext, container: Container | None) -> JobUpdate:
        if container is None:
            task_fallback = self._get_terminal_task_fallback_update(step)
            if task_fallback is not None:
                return task_fallback

            container_name = self.name_for_step(step)
            logger.error("Container not found while syncing active step: %s", container_name)
            return JobUpdate(
                status=PlatformJobStatus.ERROR,
                error_details={"message": "Container not found while syncing active step"},
            )
        else:
            return self.create_step_update(step, container)

    def sync_stop_container(self, step: PlatformJobStepWithContext, container: Container | None) -> JobUpdate:
        if container is None:
            container_name = self.name_for_step(step)
            logger.error("Container not found while stopping container: %s", container_name)
            # Job was deleted
            return JobUpdate(
                status=PlatformJobStatus.ERROR,
                error_details={"message": "Container not found while stopping container"},
            )
        else:
            if not self._is_container_owned_by_this_controller(container):
                logger.warning(
                    "Skipping container stop (not owned by this jobs-controller)",
                    extra={
                        "container_name": container.name,
                        "owner_label": (container.labels or {}).get(JOB_CONTROLLER_INSTANCE_ID_LABEL),
                    },
                )
                return JobUpdate(
                    status=PlatformJobStatus.ERROR,
                    error_details={"message": "Container not owned by this jobs controller"},
                )
            try:
                logger.debug(
                    "Stopping container for job step", extra={"container_name": container.name, "step_name": step.name}
                )
                container.stop(timeout=DOCKER_STOP_TIMEOUT)
                return self.create_step_update(step, container)
            except APIError as e:
                if e.status_code == 409:
                    # Container already stopping or stopped
                    logger.warning(
                        "Container already stopping or stopped when calling stop",
                        extra={"container_name": container.name},
                    )
                    return self.create_step_update(step, container)
                else:
                    raise e

    @staticmethod
    def parse_docker_timestamp(timestamp: str | None) -> DockerTimestampParseResult:
        """Parse a timestamp from Docker container state.

        Docker stores lifecycle timestamps as Go time.Time values and exposes them
        through inspect as formatted strings. An unset Go time.Time formats as
        0001-01-01T00:00:00Z, so treat that value as absent while keeping a
        structured flag for debugging.
        """
        if not timestamp:
            return DockerTimestampParseResult(parsed=None, parse_error=None, is_zero=False)
        is_zero_time = timestamp.startswith("0001-01-01")
        if is_zero_time:
            return DockerTimestampParseResult(parsed=None, parse_error=None, is_zero=True)
        try:
            parsed = datetime.datetime.fromisoformat(timestamp)
        except ValueError as exc:
            return DockerTimestampParseResult(parsed=None, parse_error=str(exc), is_zero=False)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=datetime.UTC)
        return DockerTimestampParseResult(parsed=parsed, parse_error=None, is_zero=False)

    def docker_state_debug_fields(self, container: Container) -> dict[str, Any]:
        attrs = container.attrs or {}
        state = attrs.get("State", {})
        now = datetime.datetime.now(datetime.UTC)
        finished_at_raw = state.get("FinishedAt")
        finished_at_result = self.parse_docker_timestamp(finished_at_raw)
        finished_at = finished_at_result.parsed
        cleanup_after_finished_at = (
            finished_at + datetime.timedelta(seconds=self._execution_profile_config.ttl_seconds_after_finished)
            if finished_at
            else None
        )

        return {
            "container_status": container.status,
            "docker_state_status": state.get("Status"),
            "docker_state_started_at": state.get("StartedAt"),
            "docker_state_finished_at": finished_at_raw,
            "docker_state_finished_at_parsed": finished_at.isoformat() if finished_at else None,
            "docker_state_finished_at_parse_error": finished_at_result.parse_error,
            "docker_state_finished_at_is_zero": finished_at_result.is_zero,
            "docker_state_finished_age_seconds": (now - finished_at).total_seconds() if finished_at else None,
            "docker_state_exit_code": state.get("ExitCode"),
            "docker_state_error": state.get("Error"),
            "docker_state_oom_killed": state.get("OOMKilled"),
            "docker_state_dead": state.get("Dead"),
            "docker_state_running": state.get("Running"),
            "docker_state_paused": state.get("Paused"),
            "host_config_auto_remove": attrs.get("HostConfig", {}).get("AutoRemove"),
            "cleanup_completed_jobs_immediately": self._execution_profile_config.cleanup_completed_jobs_immediately,
            "ttl_seconds_after_finished": self._execution_profile_config.ttl_seconds_after_finished,
            "cleanup_after_finished_at": cleanup_after_finished_at.isoformat() if cleanup_after_finished_at else None,
            "cleanup_ttl_remaining_seconds": (cleanup_after_finished_at - now).total_seconds()
            if cleanup_after_finished_at
            else None,
            "cleanup_ttl_due": cleanup_after_finished_at <= now if cleanup_after_finished_at else None,
            "now_utc": now.isoformat(),
        }

    def create_step_update(self, step: PlatformJobStepWithContext, container: Container) -> JobUpdate:
        status, status_details, error_stack = self.map_docker_container_status_to_platform_status(step, container)
        task_id = self.get_label_from_container(container, JOB_TASK_ID_LABEL)
        error_details = {}
        if status == PlatformJobStatus.ERROR:
            error_details["message"] = status_details.get("message", "Job encountered an error")
        if status in PlatformJobStatus.terminals() and self._should_enable_workload_identity_for_step(step):
            self._try_revoke_workload_delegation(
                self._workload_delegation_name_for_step(step),
                reason="terminal status",
            )

        logger.debug(
            "Docker container status mapped to platform status",
            extra={
                "workspace": step.workspace,
                "job": step.job,
                "step": step.name,
                "task": task_id,
                "container_name": container.name,
                "platform_status": status.value,
                **self.docker_state_debug_fields(container),
            },
        )

        # Upsert the task against the Jobs API.
        self._jobs.update_job_step_task(
            name=task_id,
            workspace=step.workspace,
            job=step.job,
            step=step.name,
            body=PlatformJobTaskUpdate(
                status=status,
                status_details=status_details,
                error_details=error_details,
                error_stack=error_stack,
            ),
        )
        logger.info("Updated task", extra={"task_id": task_id, "status": status})
        return JobUpdate(status=status, status_details=status_details, error_details=error_details)

    def map_docker_container_status_to_platform_status(
        self, step: PlatformJobStepWithContext, container: Container
    ) -> tuple[PlatformJobStatus, dict, str]:
        status_details = {}
        error_stack = ""
        is_cancelling = step.status == PlatformJobStatus.CANCELLING
        is_pausing = step.status == PlatformJobStatus.PAUSING
        if container.status == "running":
            if is_cancelling:
                return PlatformJobStatus.CANCELLING, {"message": "Job is cancelling"}, error_stack
            elif is_pausing:
                return PlatformJobStatus.PAUSING, {"message": "Job is pausing"}, error_stack
            else:
                return PlatformJobStatus.ACTIVE, {"message": "Job is running"}, error_stack
        elif container.status in ("exited", "dead"):
            attrs = container.attrs or {}
            exit_code = attrs.get("State", {}).get("ExitCode", 0)
            status_details["exit_code"] = exit_code
            if is_cancelling:
                # Docker stop/kill exit codes vary with the container entrypoint and
                # signal handling (for example 0, 1, 137, 143, or 255). Once the
                # persisted step is cancelling, the user's cancellation intent is
                # authoritative; the observed post-stop exit code remains diagnostic.
                return (
                    PlatformJobStatus.CANCELLED,
                    {"message": f"Job was cancelled successfully with exit code {exit_code}"},
                    error_stack,
                )
            if exit_code == 0:
                if is_pausing:
                    return (
                        PlatformJobStatus.PAUSED,
                        {"message": f"Job paused successfully with exit code {exit_code}"},
                        error_stack,
                    )
                else:
                    return (
                        PlatformJobStatus.COMPLETED,
                        {"message": f"Job completed successfully with exit code {exit_code}"},
                        error_stack,
                    )
            else:
                # Get logs for error stack
                try:
                    # Get last 80 lines of logs and up to 2048 characters
                    logs = container.logs(tail=80).decode("utf-8", errors="ignore")
                    error_stack = logs
                    if len(error_stack) > 2048:
                        error_stack = error_stack[-2048:]
                except DockerException as exc:
                    logger.error("Failed to get logs for container %s: %s", container.name, exc)

                return (
                    PlatformJobStatus.ERROR,
                    {
                        "message": f"Job exited with non-zero code {exit_code}, check logs for details.",
                        "exit_code": exit_code,
                    },
                    error_stack,
                )
        elif container.status in ("created"):
            return PlatformJobStatus.PENDING, {"message": "Job is pending"}, error_stack

        raise ValueError("Unable to determine status of Docker container")

    def apply_resource_limits(self, container_args: dict, resources: ComputeResources | None) -> dict:
        """Apply resource limits from executor config to container arguments.

        Args:
            container_args: Container arguments dictionary to modify
            executor_config: The execution provider configuration

        Returns:
            Updated container arguments dictionary
        """
        if resources and resources.limits:
            limits = resources.limits
            if limits.memory is not None:
                # Convert Kubernetes memory format to Docker format
                memory_limit = limits.memory
                # Simple conversion - Docker expects format like "1g", "512m"
                if memory_limit.endswith("Gi"):
                    container_args["mem_limit"] = memory_limit.replace("Gi", "g")
                elif memory_limit.endswith("Mi"):
                    container_args["mem_limit"] = memory_limit.replace("Mi", "m")
                else:
                    container_args["mem_limit"] = memory_limit

            if limits.cpu is not None:
                cpu_limit = limits.cpu
                # Convert CPU format - Docker expects float/int
                if cpu_limit.endswith("m"):
                    # Convert millicores to cores
                    cpu_cores = int(float(cpu_limit[:-1]) / 1000)
                    container_args["cpu_count"] = cpu_cores
                else:
                    container_args["cpu_count"] = int(cpu_limit)

        return container_args

    def cleanup_steps(self):
        containers = self._client.containers.list(
            all=True,
            filters=self._cleanup_container_filters(),
            ignore_removed=True,
        )
        for container in containers:
            try:
                if not self._is_container_owned_by_this_controller(container):
                    logger.debug(
                        "Skipping Docker cleanup for unowned job container",
                        extra={
                            "container_name": container.name,
                            "owner_label": (container.labels or {}).get(JOB_CONTROLLER_INSTANCE_ID_LABEL),
                        },
                    )
                    continue
                if container.labels.get(JOB_TYPE_LABEL) != JOB_TYPE_JOB:
                    continue
                if container.status in ("exited", "dead"):
                    state_debug = self.docker_state_debug_fields(container)
                    exit_code = state_debug.get("docker_state_exit_code") or 0
                    auto_remove = state_debug.get("host_config_auto_remove")
                    job = self.get_label_from_container(container, JOB_ID_LABEL)
                    step_name = self.get_label_from_container(container, JOB_STEP_NAME_LABEL)
                    workspace = self.get_label_from_container(container, JOB_WORKSPACE_ID_LABEL)

                    # Verify the step is terminal before cleaning up.
                    # This prevents cleaning up resources that we last marked in active state,
                    # were prematurely cleaned up, and then sync active to error because the resource is gone.
                    step_is_terminal = self.check_step_is_terminal(job=job, step_name=step_name, workspace=workspace)
                    cleanup_log_extra = {
                        "workspace": workspace,
                        "job": job,
                        "step": step_name,
                        "container_name": container.name,
                        "container_id": container.id[:16],
                        "exit_code": exit_code,
                        "host_config_auto_remove": auto_remove,
                    }
                    logger.debug(
                        "Docker cleanup inspected exited job container",
                        extra={
                            **cleanup_log_extra,
                            "step_is_terminal": step_is_terminal,
                            **state_debug,
                        },
                    )
                    if not step_is_terminal:
                        logger.debug(
                            "Skipping cleanup for job container because step is not in terminal state",
                            extra=cleanup_log_extra,
                        )
                        continue

                    self._release_container_resources(container, reason="terminal_container_cleanup")

                    # Always disconnect the container from its network first if not already done.
                    # We do this to avoid dangling containers being connected to user-defined networks.
                    self.cleanup_container_network(container)

                    # Containers in terminal state can be cleaned up immediately if configured to do so
                    if (
                        self._execution_profile_config.cleanup_completed_jobs_immediately
                        and exit_code in TERMINAL_EXIT_CODES
                    ):
                        logger.debug(
                            "Docker cleanup removing terminal job container immediately",
                            extra=cleanup_log_extra,
                        )
                        self.cleanup_single_container(container)
                        continue

                    # Otherwise, check if the TTL has expired for errored jobs or completed jobs if not cleaned up immediately
                    last_transition_time_str = container.attrs.get("State", {}).get("FinishedAt")
                    finished_at_result = self.parse_docker_timestamp(last_transition_time_str)
                    cleanup_after_finished_at = (
                        finished_at_result.parsed
                        + datetime.timedelta(seconds=self._execution_profile_config.ttl_seconds_after_finished)
                        if finished_at_result.parsed
                        else None
                    )
                    if cleanup_after_finished_at and cleanup_after_finished_at < datetime.datetime.now(datetime.UTC):
                        logger.debug(
                            "Docker cleanup removing expired job container",
                            extra={
                                **cleanup_log_extra,
                                "finished_at": last_transition_time_str,
                                "finished_at_parse_error": finished_at_result.parse_error,
                                "finished_at_is_zero": finished_at_result.is_zero,
                                **state_debug,
                            },
                        )
                        self.cleanup_single_container(container)
                    else:
                        logger.debug(
                            "Docker cleanup retaining terminal job container until TTL",
                            extra={
                                **cleanup_log_extra,
                                "finished_at": last_transition_time_str,
                                "finished_at_parse_error": finished_at_result.parse_error,
                                "finished_at_is_zero": finished_at_result.is_zero,
                                **state_debug,
                            },
                        )
            except NotFound:
                # Container may disappear between list and inspect/attribute access.
                # Ignore and continue cleanup for remaining containers.
                logger.debug("Container disappeared during cleanup loop; skipping")

    def cleanup_single_container(self, container: Container) -> None:
        """Cleanup a single container and its associated task storage volume."""
        workspace = self.get_label_from_container(container, JOB_WORKSPACE_ID_LABEL)
        job = self.get_label_from_container(container, JOB_ID_LABEL)
        task = self.get_label_from_container(container, JOB_TASK_ID_LABEL)
        attrs: dict[str, Any] = container.attrs or {}
        state = attrs.get("State", {})
        exit_code = state.get("ExitCode", 0) if isinstance(state, dict) else 0
        delegation_name = self._workload_delegation_name_from_container(container)

        self._try_revoke_workload_delegation(
            delegation_name,
            reason="terminal status",
        )
        self.cleanup_container(container)
        logger.debug(
            "Cleaned up container",
            extra={"container_name": container.name, "workspace": workspace, "job": job, "task": task},
        )

        self.cleanup_task_storage_volumes(workspace, job, task)
        logger.debug("Cleaned up task storage volume", extra={"workspace": workspace, "job": job, "task": task})

        if self._is_container_owned_by_this_controller(container):
            self._release_container_resources(container, reason="container_cleanup")

        # Clean up persistent storage for successful jobs that used it
        # Only clean up if the job itself is in a terminal state to prevent premature cleanup
        if JOB_USES_PERSISTENT_STORAGE_LABEL in container.labels:
            uses_persistent_storage = (
                self.get_label_from_container(container, JOB_USES_PERSISTENT_STORAGE_LABEL) == "true"
            )
            if uses_persistent_storage and exit_code == 0:
                step_name = self.get_label_from_container(container, JOB_STEP_NAME_LABEL)
                if self.check_job_persistent_storage_cleanup_allowed(job=job, step_name=step_name, workspace=workspace):
                    logger.debug(
                        "Cleaning up persistent storage for successful job", extra={"workspace": workspace, "job": job}
                    )
                    self.cleanup_job_persistent_storage(workspace, job)
                else:
                    logger.debug(
                        "Skipping persistent storage cleanup for job",
                        extra={"workspace": workspace, "job": job, "step": step_name},
                    )

    @abstractmethod
    def configure_container(self, container_args: dict, executor_config: ProviderT) -> dict:
        """Customize container arguments based on the execution provider.

        Args:
            container_args: Base container arguments dictionary
            executor_config: The execution provider configuration

        Returns:
            Updated container arguments dictionary
        """
        return container_args

    def name_for_step(self, step: PlatformJobStepWithContext) -> str:
        return f"{step.job}-{step.name}"


class CPUDockerJobBackend(DockerJobBackend[CPUExecutionProvider]):
    """Docker job backend for CPU execution."""

    def schedule(
        self,
        executor_config: CPUExecutionProvider,
        step: PlatformJobStepWithContext,
    ) -> JobUpdate:
        return self.schedule_single_container(executor_config, step)

    def sync(
        self,
        step: PlatformJobStepWithContext,
    ) -> JobUpdate:
        return self._sync(step)

    def configure_container(self, container_args: dict, executor_config: CPUExecutionProvider) -> dict:
        """Customize container arguments for CPU execution."""
        return self.apply_resource_limits(container_args, executor_config.resources)


class GPUDockerJobBackend(DockerJobBackend[GPUExecutionProvider]):
    """Docker job backend for GPU execution."""

    def init(self) -> None:
        super().init()
        # Get shared GPU pool from resource manager (shared with models service)
        # Pool is auto-detected from available GPUs on the system
        resource_manager = SharedResourceManager.get_instance()
        self.gpu_pool = resource_manager.get_gpu_pool()

        if self.gpu_pool is None:
            logger.warning(
                "No GPU pool available - no GPUs were detected on this system. "
                "GPU jobs will fail until GPUs are available."
            )

    def schedule(
        self,
        executor_config: GPUExecutionProvider,
        step: PlatformJobStepWithContext,
    ) -> JobUpdate:
        return self.schedule_single_container(executor_config, step)

    def sync(
        self,
        step: PlatformJobStepWithContext,
    ) -> JobUpdate:
        job_update = self._sync(step)
        # Release GPU on any terminal state. Do not wait for cleanup_steps - that method
        # is called on the base class and has no visibility into the GPU pool!
        # Note: job_update.status may be a string or enum depending on code path
        terminal_states = {s.value for s in PlatformJobStatus.terminals()}
        status_value = (
            job_update.status.value if isinstance(job_update.status, PlatformJobStatus) else job_update.status
        )
        if self.gpu_pool is not None and status_value in terminal_states:
            self._release_gpu_for_step_id(step.id, reason="terminal_sync")
        return job_update

    def _release_gpu_for_step_id(self, step_id: str, *, reason: str) -> list[int]:
        if self.gpu_pool is None:
            return []
        released = self.gpu_pool.release_gpu(step_id)
        if released:
            logger.info(
                "Released Docker GPU allocation for job step",
                extra={"step_id": step_id, "gpu_ids": released, "reason": reason},
            )
        return released

    def _release_step_resources(self, step: PlatformJobStepWithContext, *, reason: str) -> None:
        self._release_gpu_for_step_id(step.id, reason=reason)

    def _release_container_resources(self, container: Container, *, reason: str) -> None:
        labels = getattr(container, "labels", None) or {}
        step_id = labels.get(JOB_STEP_ID_LABEL)
        if step_id:
            self._release_gpu_for_step_id(step_id, reason=reason)

    def configure_container(self, container_args: dict, executor_config: GPUExecutionProvider) -> dict:
        """Customize container arguments for GPU execution."""
        # Apply resource limits
        container_args = self.apply_resource_limits(container_args, executor_config.resources)

        if executor_config.resources is not None and executor_config.resources.num_gpus is not None:
            num_gpus = executor_config.resources.num_gpus
        else:
            num_gpus = 1

        shm = resolve_gpu_job_shm_size(executor_config.resources, None, num_gpus)
        container_args["shm_size"] = k8s_shm_quantity_to_docker(shm)

        # If no GPU pool is available (no GPUs detected), raise an error
        if self.gpu_pool is None:
            raise ResourceAllocationError(
                "No GPUs available on this system. GPU jobs require a system with NVIDIA GPUs."
            )

        # Allocate explicit device IDs from the pool to prevent conflicts
        # This will raise a GPUAllocationError if not enough GPUs are available.
        try:
            gpu_ids = self.gpu_pool.allocate_gpu(container_args["labels"][JOB_STEP_ID_LABEL], num_requested=num_gpus)
        except GPUAllocationError as e:
            if e.is_transient_capacity_exhaustion:
                raise SchedulingDeferred(str(e)) from e
            raise ResourceAllocationError(str(e)) from e

        container_args["device_requests"] = [
            docker.types.DeviceRequest(
                driver="nvidia",
                device_ids=[str(gpu_id) for gpu_id in gpu_ids],
                capabilities=[["gpu"]],
            )
        ]

        return container_args
