# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import datetime
import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Generic, Literal, Optional, TypeVar
from urllib.parse import SplitResult, quote, urlsplit

import nmp.common.auth.workload_identity as _workload_identity
from nemo_platform import NeMoPlatform
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.client.errors import NotFoundError as ClientNotFoundError
from nemo_platform_plugin.jobs import execution_profiles as _execution_profiles
from nemo_platform_plugin.jobs.client import JobsClient
from nemo_platform_plugin.jobs.schemas import PlatformJobStatus
from nemo_platform_plugin.jobs.types import PlatformJobStepResponse, PlatformJobStepWithContext
from nmp.common.auth import WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR
from nmp.common.auth.models import NMP_PRINCIPAL_ENVVAR
from nmp.common.config.base import (
    LOOPBACK_ADDRESSES,
    NMP_CONFIG_WARNINGS_DISABLED_ENV_VAR,
    PlatformConfig,
    determine_loopback_override,
    get_common_service_config,
)
from nmp.common.jobs.constants import (
    CONFIG_TASK_STORAGE_PATH_ENVVAR,
    EPHEMERAL_TASK_STORAGE_PATH_ENVVAR,
    NEMO_JOB_ATTEMPT_ID_ENVVAR,
    NEMO_JOB_FILESET_ENVVAR,
    NEMO_JOB_ID_ENVVAR,
    NEMO_JOB_SECRETS_ENVVAR,
    NEMO_JOB_STEP_CONFIG_FILE_PATH_ENVVAR,
    NEMO_JOB_STEP_ENVVAR,
    NEMO_JOB_TASK_ENVVAR,
    NEMO_JOB_WORKSPACE_ENVVAR,
    PERSISTENT_JOB_STORAGE_PATH_ENVVAR,
    TASK_CONFIG_ENVVAR,
)
from nmp.common.platform_endpoint import parse_platform_endpoint
from nmp.common.sdk_factory import get_entity_parts
from nmp.core.jobs.app.providers import ComputeResources
from nmp.core.jobs.entities import get_step_spec_name, is_final_platform_step
from pydantic import BaseModel, model_validator

logger = logging.getLogger(__name__)

WORKLOAD_IDENTITY_TOKEN_FILE_PATH = _workload_identity.WORKLOAD_IDENTITY_TOKEN_FILE_PATH
WORKLOAD_IDENTITY_VOLUME_NAME = _workload_identity.WORKLOAD_IDENTITY_VOLUME_NAME
WORKLOAD_IDENTITY_VOLUME_PATH = _workload_identity.WORKLOAD_IDENTITY_VOLUME_PATH
get_workload_identity_token_audience = _workload_identity.get_workload_identity_token_audience
is_workload_identity_token_exchange_enabled = _workload_identity.is_workload_identity_token_exchange_enabled

ExecutionProviderConfigT = TypeVar("ExecutionProviderConfigT")
ExecutionProfileConfigT = TypeVar("ExecutionProfileConfigT")

DEFAULT_PROFILE = "default"
DEFAULT_PROVIDER = "cpu"

# The base ``JobExecutionProfileConfig`` lives in the shared plugin leaf node
# (imported above) so that both the server and the typed HTTP client agree on
# the wire shape.

JOB_LOGS_ENDPOINT_ENVVAR = _execution_profiles.JOB_LOGS_ENDPOINT_ENVVAR

RESERVED_MANAGED_JOB_AUTH_ENVIRONMENT_VARIABLE_NAMES: frozenset[str] = frozenset(
    {
        JOB_LOGS_ENDPOINT_ENVVAR,
        "NMP_ACCESS_TOKEN",
        "NEMO_WORKLOAD_TOKEN",
        "NEMO_WORKLOAD_TOKEN_FILE",
        WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR,
    }
)

RESERVED_MANAGED_JOB_ENVIRONMENT_VARIABLE_PREFIXES: tuple[str, ...] = (
    "NEMO_WORKLOAD_",
    "NMP_WORKLOAD_",
    "NEMO_WORKFLOW_",
    "NMP_WORKFLOW_",
)

# Env var names set by the platform during job creation; user-provided profile environment must not conflict.
RESERVED_JOB_ENVIRONMENT_VARIABLE_NAMES: frozenset[str] = (
    frozenset(
        {
            # From nmp.common.jobs.constants
            CONFIG_TASK_STORAGE_PATH_ENVVAR,
            EPHEMERAL_TASK_STORAGE_PATH_ENVVAR,
            NEMO_JOB_ATTEMPT_ID_ENVVAR,
            NEMO_JOB_FILESET_ENVVAR,
            NEMO_JOB_ID_ENVVAR,
            NEMO_JOB_SECRETS_ENVVAR,
            NEMO_JOB_STEP_CONFIG_FILE_PATH_ENVVAR,
            NEMO_JOB_STEP_ENVVAR,
            NEMO_JOB_TASK_ENVVAR,
            NEMO_JOB_WORKSPACE_ENVVAR,
            PERSISTENT_JOB_STORAGE_PATH_ENVVAR,
            TASK_CONFIG_ENVVAR,
            # Auth
            NMP_PRINCIPAL_ENVVAR,
            # Platform launcher logs
            JOB_LOGS_ENDPOINT_ENVVAR,
            # Platform shared envvars (to_shared_envvars with NMP_ prefix)
            NMP_CONFIG_WARNINGS_DISABLED_ENV_VAR,
            "NMP_AUTH_URL",
            "NMP_BASE_URL",
            "NMP_JOBS_URL",
            "NMP_FILES_URL",
            "NMP_MODELS_URL",
            "NMP_SECRETS_URL",
        }
    )
    | RESERVED_MANAGED_JOB_AUTH_ENVIRONMENT_VARIABLE_NAMES
)


def find_reserved_managed_job_environment_variable_names(env_names: Iterable[str]) -> list[str]:
    """Return user-provided managed-job env names reserved for platform-managed injection."""
    return sorted(
        {
            name
            for name in env_names
            if name in RESERVED_MANAGED_JOB_AUTH_ENVIRONMENT_VARIABLE_NAMES
            or name.startswith(RESERVED_MANAGED_JOB_ENVIRONMENT_VARIABLE_PREFIXES)
        }
    )


def validate_no_reserved_managed_job_environment_variable_names(env_names: Iterable[str], *, source: str) -> None:
    """Reject user-provided platform-managed env names for managed jobs."""
    conflicting = find_reserved_managed_job_environment_variable_names(env_names)
    if conflicting:
        raise ValueError(f"{source} must not use platform-reserved managed environment keys: {conflicting}")


NMP_JOB_LAUNCHER_OTLP_LOGS_ENDPOINT_ENVVAR = "NMP_JOB_LAUNCHER_OTLP_LOGS_ENDPOINT"
NMP_JOB_LAUNCHER_OTLP_LOGS_SOCKET_PATH_ENVVAR = "NMP_JOB_LAUNCHER_OTLP_LOGS_SOCKET_PATH"
NMP_JOB_LAUNCHER_OTLP_LOGS_TRANSPORT_ENVVAR = "NMP_JOB_LAUNCHER_OTLP_LOGS_TRANSPORT"


@dataclass(frozen=True)
class OtlpLogsEndpointConfig:
    endpoint: str
    transport: Literal["tcp", "uds"]
    socket_path: str | None = None

    def to_env(self) -> dict[str, str]:
        env = {
            NMP_JOB_LAUNCHER_OTLP_LOGS_ENDPOINT_ENVVAR: self.endpoint,
            NMP_JOB_LAUNCHER_OTLP_LOGS_TRANSPORT_ENVVAR: self.transport,
        }
        if self.socket_path is not None:
            env[NMP_JOB_LAUNCHER_OTLP_LOGS_SOCKET_PATH_ENVVAR] = self.socket_path
        return env


class JobUpdate(BaseModel):
    status: PlatformJobStatus
    status_details: dict | None = None
    error_details: dict | None = None


class JobExecutionProfileConfig(_execution_profiles.JobExecutionProfileConfig):
    """Server-side extension adding workload-identity managed-auth env validation."""

    @model_validator(mode="after")
    def validate_env_no_reserved_names(self) -> JobExecutionProfileConfig:
        conflicting = [k for k in self.env if k in RESERVED_JOB_ENVIRONMENT_VARIABLE_NAMES]
        if conflicting:
            raise ValueError(
                f"Profile environment keys must not conflict with platform-reserved names: {sorted(conflicting)}"
            )
        return self

    @model_validator(mode="after")
    def validate_env_no_reserved_managed_names(self) -> JobExecutionProfileConfig:
        conflicting = find_reserved_managed_job_environment_variable_names(self.env)
        if conflicting:
            raise ValueError(
                f"Profile environment keys must not conflict with platform-reserved auth environment keys: {sorted(conflicting)}"
            )
        return self


_DEFAULT_TASK_IMAGE_NAME = "nmp-cpu-tasks"
JOB_RUNTIME_SERVICE_NAMES = ("auth", "jobs", "files", "models", "secrets")


def resolve_task_image(container_image: str | None, default_task_image: str | None) -> str:
    """Resolve the container image for a job task.

    Priority:
    1. Explicit container.image from the job step
    2. default_task_image from the execution profile config
    3. Platform CPU tasks image derived from platform.image_registry / image_tag
    """
    from nemo_platform_plugin.jobs.image import get_qualified_image

    return container_image or default_task_image or get_qualified_image(_DEFAULT_TASK_IMAGE_NAME)


def resolve_gpu_job_shm_size(
    executor_resources: ComputeResources | None,
    profile_resources: ComputeResources | None,
    num_gpus: int,
) -> str:
    """Pick SHM size for GPU workloads: explicit shm_size on executor, then profile, then 1Gi x GPU count."""
    for res in (executor_resources, profile_resources):
        if res is not None and res.shm_size:
            return res.shm_size
    return f"{max(1, num_gpus)}Gi"


def _split_url(url: str) -> SplitResult | None:
    try:
        parsed = urlsplit(url)
        parsed.hostname
        parsed.port
    except ValueError:
        return None
    return parsed


def _is_loopback_url(url: str) -> bool:
    parsed = _split_url(url)
    return parsed is not None and parsed.hostname in LOOPBACK_ADDRESSES


def _format_netloc_with_hostname(parsed: SplitResult, hostname: str) -> str:
    host = hostname[1:-1] if hostname.startswith("[") and hostname.endswith("]") else hostname
    if ":" in host:
        host = f"[{host}]"

    userinfo = ""
    if parsed.username is not None:
        userinfo = parsed.username
        if parsed.password is not None:
            userinfo = f"{userinfo}:{parsed.password}"
        userinfo = f"{userinfo}@"

    netloc = userinfo + host
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    return netloc


def _replace_loopback_address(url: str, loopback_address: str | None) -> str:
    if not loopback_address:
        return url

    parsed = _split_url(url)
    if parsed is None or parsed.hostname not in LOOPBACK_ADDRESSES:
        return url

    netloc = _format_netloc_with_hostname(parsed, loopback_address)
    return parsed._replace(netloc=netloc).geturl()


def _contains_loopback_address(url: str) -> bool:
    return _is_loopback_url(url)


def _job_runtime_base_url(platform_config: PlatformConfig) -> str:
    service_discovery_url = platform_config.service_discovery.get("platform") or platform_config.service_discovery.get(
        "base"
    )
    if service_discovery_url:
        return service_discovery_url

    return platform_config.base_url


def get_job_runtime_shared_envvars(
    platform_config: PlatformConfig,
    *,
    disable_warnings: bool = True,
    loopback_address: str | None = None,
) -> dict[str, str]:
    """Return platform URLs for job containers/pods.

    Jobs run outside the API server process. When services run in-process,
    PlatformConfig.get_service_url() intentionally returns a local server URL,
    but job runtimes need URLs reachable from their own network, usually the
    configured service_discovery gateway.
    """
    effective_loopback_address = loopback_address or platform_config.loopback_address or determine_loopback_override()
    base_url = _job_runtime_base_url(platform_config)
    base_url = _replace_loopback_address(base_url, effective_loopback_address)

    envvars = {"NMP_BASE_URL": base_url}
    for service_name in JOB_RUNTIME_SERVICE_NAMES:
        service_url = platform_config.service_discovery.get(service_name) or base_url
        envvars[f"NMP_{service_name.upper()}_URL"] = _replace_loopback_address(
            service_url,
            effective_loopback_address,
        )

    if disable_warnings:
        envvars[NMP_CONFIG_WARNINGS_DISABLED_ENV_VAR] = "1"

    # A task container does not get the platform config file, so the deployment's
    # logging settings would otherwise not reach it: a chart configured with
    # ``service.log_format: json`` would still start its tasks on ``plain``.
    # Resolve them here, where the file *is* readable, and pass the answer down.
    # Best-effort - logging configuration is not worth failing a schedule over,
    # and the task falls back to its own defaults.
    try:
        service_config = get_common_service_config()
    except Exception:
        logger.warning("Could not resolve logging settings for job runtime env.", exc_info=True)
    else:
        envvars["LOG_LEVEL"] = service_config.log_level
        envvars["LOG_FORMAT"] = service_config.log_format

    return envvars


class JobBackend(Generic[ExecutionProviderConfigT, ExecutionProfileConfigT], ABC):
    BACKEND_NAME: str = "generic_backend"

    def __init__(
        self,
        nmp_sdk: NeMoPlatform,
        execution_profile_config: ExecutionProfileConfigT,
        profile_name: str,
    ):
        self._nmp_sdk = nmp_sdk
        # Typed Jobs client sharing the SDK's transport/headers. Built once; every
        # call passes ``workspace=`` explicitly (including the cross-workspace "-"),
        # so the client's default workspace is never relied upon.
        self._jobs = client_from_platform(nmp_sdk, JobsClient)
        self._execution_profile_config = execution_profile_config
        self._profile_name = profile_name
        self.init()

    def init(self): ...

    @abstractmethod
    def schedule(
        self,
        executor_config: ExecutionProviderConfigT,
        step: PlatformJobStepWithContext,
    ) -> JobUpdate: ...

    @abstractmethod
    def sync(self, step: PlatformJobStepWithContext) -> JobUpdate: ...

    @abstractmethod
    def cleanup_steps(self): ...

    @abstractmethod
    def shutdown(self): ...

    def __str__(self) -> str:
        return self.BACKEND_NAME

    def get_secrets_environment_variable_for_injection(self, step: PlatformJobStepWithContext) -> str:
        """Fetch the secret value for an inline secret environment variable.

        Args:
            step: The job step containing the environment variable reference.
        Returns:
            The environment variable to create to inform the job which secrets to fetch.
            It has the format "ENV_VAR1=workspace/secret_name1,ENV_VAR2=workspace/secret_name2".
        """
        env_var_str = ""
        for envvar in (step.step_spec.environment or []) if step.step_spec else []:
            if envvar.from_secret is not None:
                if env_var_str != "":
                    env_var_str += ","
                workspace, secret_name = get_entity_parts(envvar.from_secret.name, default_workspace=step.workspace)
                env_var_str += f"{envvar.name}={workspace}/{secret_name}"
        return env_var_str

    def get_step(self, job: str, step_name: str, workspace: str) -> PlatformJobStepResponse:
        """Fetch the latest state of a job step via the typed Jobs client."""
        return self._jobs.get_job_step(name=step_name, workspace=workspace, job=job).data()

    def get_step_safe(self, job: str, step_name: str, workspace: str) -> Optional[PlatformJobStepResponse]:
        """Fetch the job step, or None if not found (e.g. 404, workspace deleted)."""
        try:
            return self.get_step(job=job, step_name=step_name, workspace=workspace)
        except ClientNotFoundError:
            return None
        except Exception as e:
            raise e

    def check_step_is_terminal(self, job: str, step_name: str, workspace: str) -> bool:
        """Return True if the job step is terminal (cancelled/error/completed) or the step entity is not found (e.g. 404).

        Used by cleanup: when the entity is missing (e.g. workspace deleted) but the backend job is terminal and
        has our labels, we treat as terminal so we can clean up backend resources.
        """
        try:
            step = self.get_step(job=job, step_name=step_name, workspace=workspace)
            return step.status in ("cancelled", "error", "completed")
        except ClientNotFoundError:
            # If the job step entity is not found, we treat it as terminal so cleanup can proceed.
            return True
        except Exception as e:
            raise RuntimeError(f"Could not fetch job step '{job}/{step_name}' to check if terminal") from e

    def check_job_is_terminal(self, job: str, workspace: str) -> bool:
        """Check if a job is in a terminal state."""
        try:
            job_response = self._jobs.get_job(name=job, workspace=workspace).data()
            return job_response.status in ("cancelled", "error", "completed")
        except ClientNotFoundError:
            # If the job entity is not found, we treat it as terminal so cleanup can proceed.
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to fetch job '{workspace}/{job}' to check if terminal") from e

    def check_job_persistent_storage_cleanup_allowed(self, job: str, step_name: str, workspace: str) -> bool:
        """Return whether a completed step may delete persistent job storage.

        The jobs controller owns aggregate terminal-state transitions. Backend cleanup only checks that the
        successful resource belongs to the final configured step before deleting storage shared by the attempt.
        """
        try:
            job_response = self._jobs.get_job(name=job, workspace=workspace).data()
        except ClientNotFoundError:
            # If the job entity is gone (e.g. workspace deletion), allow backend cleanup to reclaim storage.
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to fetch job '{workspace}/{job}' to check storage cleanup eligibility") from e

        job_status = getattr(job_response.status, "value", job_response.status)
        if job_status not in ("cancelled", "error", "completed"):
            return False

        try:
            step = self.get_step(job=job, step_name=step_name, workspace=workspace)
        except ClientNotFoundError:
            return True
        except Exception as e:
            raise RuntimeError(
                f"Could not fetch job step '{job}/{step_name}' to check storage cleanup eligibility"
            ) from e

        step_spec_name = get_step_spec_name(step.config, getattr(step, "name", None) or step_name)

        platform_spec = getattr(job_response, "platform_spec", None)
        steps = getattr(platform_spec, "steps", None) or []
        if not steps:
            return True

        if is_final_platform_step(platform_spec, step_spec_name):
            return True

        logger.debug(
            "Skipping persistent storage cleanup because completed step is not the final job step",
            extra={
                "workspace": workspace,
                "job": job,
                "step": step_name,
                "step_spec_name": step_spec_name,
                "final_step_name": getattr(steps[-1], "name", None),
            },
        )
        return False

    def check_step_ttl(self, step: PlatformJobStepWithContext, ttl_seconds: int) -> bool:
        # Ensure created_at is timezone-aware (assume UTC if naive)
        if step.created_at is None:
            return False

        created_at = step.created_at
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=datetime.timezone.utc)

        return (created_at + datetime.timedelta(seconds=ttl_seconds)) < datetime.datetime.now(datetime.timezone.utc)

    @staticmethod
    def should_enforce_before_active_ttl(step: PlatformJobStepWithContext) -> bool:
        """Whether ttl_seconds_before_active applies. Skipped while resuming from pause."""
        if step.status is None:
            return True
        if isinstance(step.status, str):
            status_val = step.status
        elif isinstance(step.status, Enum):
            status_val = step.status.value
        else:
            status_val = str(step.status)
        return status_val != "resuming"

    def check_step_ttl_before_active(self, step: PlatformJobStepWithContext, ttl_seconds: int) -> bool:
        """Return True if the step exceeded ttl_seconds in a pre-active (pending) state.

        Uses max(created_at, updated_at) so pause/resume does not consume the scheduling
        budget; updated_at advances on lifecycle transitions (e.g. resume, PENDING).
        """
        if step.created_at is None:
            return False

        created_at = step.created_at
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=datetime.timezone.utc)

        anchor = created_at
        if step.updated_at is not None:
            updated_at = step.updated_at
            if updated_at.tzinfo is None:
                updated_at = updated_at.replace(tzinfo=datetime.timezone.utc)
            anchor = max(created_at, updated_at)

        return (anchor + datetime.timedelta(seconds=ttl_seconds)) < datetime.datetime.now(datetime.timezone.utc)

    def check_step_is_stale(self, step: PlatformJobStepWithContext) -> bool:
        """Check if all active tasks for a step have gone stale.

        Returns True if staleness_timeout_seconds is configured
        in the step lifecycle AND all active tasks have updated_at older than
        the staleness threshold.
        """
        staleness_timeout = configured_staleness_timeout_seconds(step)
        if not staleness_timeout or staleness_timeout <= 0:
            return False

        # Skip if step hasn't been alive long enough for staleness to be possible.
        if not self.check_step_ttl(step, staleness_timeout):
            return False

        try:
            tasks = self._jobs.list_job_step_tasks(
                name=step.name,
                job=step.job,
                workspace=step.workspace,
            ).data()
        except Exception:
            logger.warning("Failed to fetch tasks for staleness check", extra={"step": step.name, "job": step.job})
            return False

        active_tasks = [t for t in tasks.data if t.status == "active"]
        if not active_tasks:
            return False

        now = datetime.datetime.now(datetime.timezone.utc)
        threshold = datetime.timedelta(seconds=staleness_timeout)

        for task in active_tasks:
            if task.updated_at is None:
                logger.error("Task missing updated_at timestamp", extra={"task": task.name, "step": step.name})
                return False
            updated_at = task.updated_at
            if updated_at.tzinfo is None:
                updated_at = updated_at.replace(tzinfo=datetime.timezone.utc)
            if (now - updated_at) < threshold:
                return False

        logger.warning(
            "Stale job step detected, job step will be killed",
            extra={
                "step": step.name,
                "job": step.job,
                "staleness_timeout_seconds": staleness_timeout,
                "num_stale_tasks": len(active_tasks),
            },
        )
        return True


def staleness_error_message(staleness_timeout: int) -> str:
    return f"Job terminated: no task updates received within {staleness_timeout}s staleness threshold"


def configured_staleness_timeout_seconds(step: PlatformJobStepWithContext) -> int | None:
    if step.step_spec is None or step.step_spec.lifecycle is None:
        return None
    return step.step_spec.lifecycle.staleness_timeout_seconds


def require_staleness_timeout_seconds(step: PlatformJobStepWithContext) -> int:
    staleness_timeout = configured_staleness_timeout_seconds(step)
    if staleness_timeout is None or staleness_timeout <= 0:
        raise ValueError("stale job step requires a positive staleness_timeout_seconds")
    return staleness_timeout


def extract_provider_profile(step: PlatformJobStepWithContext) -> tuple[str, str]:
    step_spec = step.step_spec
    if step_spec is None:
        raise ValueError("Job step requires a step_spec")

    profile = DEFAULT_PROFILE
    if isinstance(step_spec.executor, str):
        provider = step_spec.executor
    else:
        provider = step_spec.executor.provider or DEFAULT_PROVIDER
        profile = step_spec.executor.profile or profile

    return provider, profile


def get_logs_endpoint_from_fileset(
    platform_config: PlatformConfig,
    workspace: str,
    fileset_id: str,
    loopback_address: str | None = None,
) -> str:
    """Get the OTLP logs endpoint URL for a fileset.

    Args:
        platform_config: Platform configuration containing files_url and loopback_address
        workspace: Workspace ID
        fileset_id: Fileset ID
        loopback_address: Optional override for loopback address replacement. If omitted, uses
            platform_config.loopback_address, then automatic loopback detection.

    Returns:
        Full OTLP logs endpoint URL with appropriate loopback address applied.
    """
    return get_logs_endpoint_config_from_fileset(
        platform_config,
        workspace,
        fileset_id,
        loopback_address=loopback_address,
    ).endpoint


def get_logs_endpoint_config_from_fileset(
    platform_config: PlatformConfig,
    workspace: str,
    fileset_id: str,
    loopback_address: str | None = None,
    artifact_base_path: str | None = None,
) -> OtlpLogsEndpointConfig:
    """Get OTLP logs endpoint config, preserving transport metadata for local UDS runtimes.

    Job telemetry is emitted from a separate process/container/pod. When Files
    runs in-process with the API server, local service URLs are not necessarily
    routable from job runtime networks, so fall back through the same
    workload-facing base URL used for job SDK env vars.
    """
    # Check service_discovery for a files-specific URL first, then fall back to
    # the job runtime base URL (service_discovery["platform"] / base_url).
    files_discovery_url = platform_config.service_discovery.get("files")
    if files_discovery_url:
        platform_endpoint = parse_platform_endpoint(files_discovery_url)
    else:
        runtime_base = _job_runtime_base_url(platform_config)
        platform_endpoint = parse_platform_endpoint(runtime_base)

    base_url = platform_endpoint.connect_base_url

    # Use configured loopback_address, or fall back to automatic detection
    effective_override = loopback_address or platform_config.loopback_address or determine_loopback_override()
    base_url = _replace_loopback_address(base_url, effective_override)

    endpoint = f"{base_url}/apis/files/v2/workspaces/{workspace}/filesets/{fileset_id}/otlp/v1/logs"
    if artifact_base_path:
        endpoint = f"{endpoint}?artifact_base_path={quote(artifact_base_path, safe='')}"

    return OtlpLogsEndpointConfig(
        endpoint=endpoint,
        transport=platform_endpoint.transport,
        socket_path=str(platform_endpoint.socket_path) if platform_endpoint.socket_path is not None else None,
    )
