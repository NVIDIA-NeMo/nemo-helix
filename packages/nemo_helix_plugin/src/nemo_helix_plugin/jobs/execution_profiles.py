# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execution-profile types for the Jobs service.

An *execution profile* describes a configured backend (docker, kubernetes,
volcano, subprocess, e2e) that the jobs controller can schedule steps onto.
These are returned by the ``get_execution_profiles`` endpoint.

This module holds the **data shapes** as pure pydantic — no docker or
kubernetes runtime dependencies.  Server-side behaviour that needs those
libraries (``KubernetesVolume.to_k8s()`` etc.) lives in the Jobs service,
which subclasses these models.  Both the server and the typed HTTP client
share these definitions.
"""

from __future__ import annotations

from typing import Any, Literal
from urllib.parse import urlparse

from nemo_helix_plugin.client.constants import WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR
from nemo_helix_plugin.config import NHX_CONFIG_WARNINGS_DISABLED_ENV_VAR
from nemo_helix_plugin.jobs.constants import (
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
from nemo_helix_plugin.jobs.providers import ComputeResources
from nemo_helix_plugin.jobs.spec import BaseExecutionProfile, ProviderRef
from pydantic import BaseModel, ConfigDict, Field, model_validator

# Default image used to set filesystem permissions on job storage volumes.
DEFAULT_VOLUME_PERMISSIONS_IMAGE = "docker.io/library/busybox:stable"
JOB_LOGS_ENDPOINT_ENVVAR = "NHX_JOB_LOGS_ENDPOINT"

# Env var names set by the platform during job creation; user-provided profile
# environment must not conflict.  The job-scoped names come from the shared
# ``jobs.constants`` leaf; the auth / config / logging / telemetry names are stable env
# var strings kept here to avoid importing server-side auth/config modules.
RESERVED_JOB_ENVIRONMENT_VARIABLE_NAMES: frozenset[str] = frozenset(
    {
        # Job runtime (from nemo_helix_plugin.jobs.constants)
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
        "NHX_PRINCIPAL",
        WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR,
        # Platform launcher logs
        JOB_LOGS_ENDPOINT_ENVVAR,
        # OTEL (telemetry)
        "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT",
        "OTEL_LOGS_EXPORTER",
        "OTEL_SERVICE_NAME",
        "OTEL_EXPORTER_OTLP_LOGS_HEADERS",
        # Platform shared envvars (to_shared_envvars with NHX_ prefix)
        NHX_CONFIG_WARNINGS_DISABLED_ENV_VAR,
        "NHX_AUTH_URL",
        "NHX_BASE_URL",
        "NHX_JOBS_URL",
        "NHX_FILES_URL",
        "NHX_MODELS_URL",
        "NHX_SECRETS_URL",
    }
)


class ImagePullSecret(BaseModel):
    """Kubernetes image pull secret reference."""

    # extra=forbid keeps additionalProperties: false on the generated schema.
    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Kubernetes Secret name for pulling images")


class JobExecutionProfileConfig(BaseModel):
    ttl_seconds_before_active: int = 30 * 60  # 30 minutes
    ttl_seconds_active: int = 24 * 60 * 60  # 24 hours
    ttl_seconds_after_finished: int = 60 * 60  # 1 hour
    cleanup_completed_jobs_immediately: bool = True
    launcher_tool_path: str = Field(default="/tools/jobs-launcher", description="Path to the jobs launcher tool")
    default_task_image: str | None = Field(
        default=None,
        min_length=1,
        description="Default container image for job task pods. Used when a job step omits container.image. "
        "When unset, falls back to the platform CPU tasks image (platform.image_registry/nhx-tasks:platform.image_tag).",
    )
    env: dict[str, str] = Field(
        default_factory=dict,
        description="Optional env vars applied to all jobs (e.g. HOME=/tmp). Keys must not conflict with platform-reserved names. Job steps may override these variables.",
    )

    @model_validator(mode="after")
    def validate_env_no_reserved_names(self) -> JobExecutionProfileConfig:
        conflicting = [k for k in self.env if k in RESERVED_JOB_ENVIRONMENT_VARIABLE_NAMES]
        if conflicting:
            raise ValueError(
                f"Profile environment keys must not conflict with platform-reserved names: {sorted(conflicting)}"
            )
        return self


# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------


class DockerVolumeMount(BaseModel):
    volume_name: str = Field(description="Name of the Docker volume to mount")
    mount_path: str = Field(description="Path inside the container where the volume will be mounted")
    kind: Literal["volume", "tmpfs"] = Field(
        default="volume",
        description="Type of the Docker volume to mount. Options are 'volume' or 'tmpfs' (default: 'volume'). tmpfs volumes are only supported on Linux hosts.",
    )
    options: dict | None = Field(default=None, description="Additional options for the volume")
    allow_create_volume: bool = Field(
        default=False, description="Whether to allow the creation of the volume if it does not exist (default: false)."
    )


class DockerJobStorageConfig(BaseModel):
    """Configuration for persistent storage in Docker jobs."""

    volume_name: str = Field(
        default="nemo-jobs-storage", description="Name of the Docker volume for persistent storage"
    )
    volume_permissions_image: str = Field(
        default=DEFAULT_VOLUME_PERMISSIONS_IMAGE, description="Docker image used to set permissions on the volume"
    )
    additional_volume_mounts: list[DockerVolumeMount] = Field(
        default_factory=list,
        description="List of additional Docker volume mounts for the job",
    )


class DockerJobNetworkConfig(BaseModel):
    job_container_network: str = Field(default="host", description="Docker network for the job container")


class DockerJobExecutionProfileConfig(JobExecutionProfileConfig):
    """Configuration for Docker Job execution profile."""

    storage: DockerJobStorageConfig = Field(
        default_factory=DockerJobStorageConfig, description="Docker storage configuration"
    )
    networking: DockerJobNetworkConfig = Field(
        default_factory=DockerJobNetworkConfig, description="Docker networking configuration"
    )


class DockerJobExecutionProfile(BaseExecutionProfile):
    """
    Execution configuration for a Docker Job.
    This is used to define the executor type, provider, profile, and any additional configuration
    required for the executor to run the job on Docker
    """

    backend: Literal["docker"] = "docker"
    config: DockerJobExecutionProfileConfig = Field(description="Additional configuration for the docker executor")

    @property
    def supports_persistent_storage(self) -> bool:
        """Indicates if the execution profile supports persistent storage."""
        return self.config.storage is not None and self.config.storage.volume_name != ""


# ---------------------------------------------------------------------------
# Kubernetes (shared)
# ---------------------------------------------------------------------------


class KubernetesObjectMetadata(BaseModel):
    labels: dict[str, str] = Field(default_factory=dict)
    annotations: dict[str, str] = Field(default_factory=dict)


class KubernetesPersistentVolumeClaim(BaseModel):
    """Kubernetes Persistent Volume Claim definition."""

    claim_name: str = Field(description="Persistent Volume Claim Name")
    read_only: bool = Field(default=False, description="Whether the volume is mounted read-only")


class KubernetesEmptyDirVolume(BaseModel):
    """Kubernetes EmptyDir Volume definition."""

    medium: str | None = Field(default=None, description="The medium of the emptyDir volume (e.g., 'Memory')")
    size_limit: str | None = Field(default=None, description="The size limit of the emptyDir volume (e.g., '1Gi')")


class KubernetesVolume(BaseModel):
    """Kubernetes Volume definition.

    Data shape only.  The server subclass adds ``to_k8s()`` which requires the
    ``kubernetes`` client library.
    """

    name: str = Field(description="Volume Name")
    persistent_volume_claim: KubernetesPersistentVolumeClaim | None = Field(
        default=None, description="Persistent Volume Claim configuration"
    )
    empty_dir: KubernetesEmptyDirVolume | None = Field(default=None, description="EmptyDir Volume configuration")

    @model_validator(mode="after")
    def validate_self(self) -> KubernetesVolume:
        """Ensure that exactly one volume source is specified."""
        if sum(source is not None for source in [self.persistent_volume_claim, self.empty_dir]) != 1:
            raise ValueError("Exactly one of 'persistent_volume_claim' or 'empty_dir' must be specified.")
        return self


class KubernetesVolumeMount(BaseModel):
    """Kubernetes Volume Mount definition.

    Data shape only.  The server subclass adds ``to_k8s()``.
    """

    name: str = Field(description="Volume Name")
    mount_path: str = Field(description="Mount Path in the container")
    sub_path: str | None = Field(default=None, description="Sub-path within the volume to mount")
    read_only: bool = Field(default=False, description="Whether the volume mount is read-only")


class KubernetesJobStorageConfig(BaseModel):
    """Configuration for persistent storage in Kubernetes jobs."""

    pvc_name: str = Field(default="", description="Persistent Volume Claim Name to use for job storage.")
    volume_permissions_image: str = Field(
        default=DEFAULT_VOLUME_PERMISSIONS_IMAGE, description="Image used to set volume permissions"
    )
    additional_volumes: list[KubernetesVolume] = Field(default_factory=list, description="Additional volumes to mount")
    additional_volume_mounts: list[KubernetesVolumeMount] = Field(
        default_factory=list, description="Additional volume mounts"
    )


class KubernetesWorkloadIdentityConfig(BaseModel):
    """Kubernetes workload identity token projection configuration."""

    model_config = ConfigDict(extra="forbid")

    token_expiration_seconds: int = Field(
        default=600,
        ge=600,
        description="Requested expirationSeconds for the projected service account token used as the workload identity subject token.",
    )
    token_audience: str | None = Field(
        default=None,
        description="Audience for the projected service account token. Defaults to auth.oidc.workload.client_id, then 'nemo-helix'.",
    )


class BaseKubernetesExecutionProfileConfig(JobExecutionProfileConfig):
    """Common configuration for Kubernetes execution environment."""

    # Only a pull that has already *failed* is counted -- a slow pull of a large
    # image reports ContainerCreating and never reaches this budget -- so the
    # question is how long a recoverable fault deserves. Registry rate limiting,
    # a 5xx blip and a mid-rotation pull secret can all take minutes to clear,
    # and killing a job that would have succeeded is worse than being slow to
    # report one that never will. Hence generous, but still 3x faster than
    # ttl_seconds_before_active and naming the image when it fires.
    ttl_seconds_image_pull: int = Field(
        default=10 * 60,
        ge=0,
        description="How long a pod may spend in image-pull backoff before the pull is treated as "
        "unrecoverable and the step fails naming the image. 0 disables it.",
    )

    namespace: str | None = Field(
        default=None,
        description="Kubernetes namespace to submit the job to. If not set, it will be determined from the environment.",
    )

    service_account_name: str = Field(
        default="default",
        description="Kubernetes service account name for job pods. Uses the Kubernetes default service account when set to 'default'.",
    )

    # Scheduling and resource configuration
    tolerations: list[dict[str, Any]] = Field(
        default_factory=list, description="Tolerations for the Kubernetes job pods."
    )
    node_selector: dict[str, str] = Field(
        default_factory=dict, description="Node selector for the Kubernetes job pods."
    )
    affinity: dict[str, Any] = Field(default_factory=dict, description="Affinity for the Kubernetes job pods.")
    resources: ComputeResources = Field(
        default_factory=ComputeResources, description="Resource requests and limits for the Kubernetes job pods."
    )
    pod_security_context: dict[str, Any] = Field(
        default_factory=dict, description="Pod security context for the Kubernetes job pods."
    )

    # Image pull secrets
    image_pull_secrets: list[ImagePullSecret] = Field(
        default_factory=list, description="Image pull secrets for the Kubernetes job pods."
    )

    # Optional metadata to add to each job object
    job_metadata: KubernetesObjectMetadata = Field(
        default_factory=KubernetesObjectMetadata,
        description="Metadata to add to each job object in the Kubernetes job.",
    )

    # Optional metadata to add to each pod in the job
    pod_metadata: KubernetesObjectMetadata = Field(
        default_factory=KubernetesObjectMetadata, description="Metadata to add to each pod in the Kubernetes job."
    )

    # Storage configurations for the job
    storage: KubernetesJobStorageConfig = Field(
        default_factory=KubernetesJobStorageConfig, description="Storage configuration for the Kubernetes job pods."
    )

    num_gpus: int = Field(default=1, description="Number of GPUs to request for the job")

    scheduler_name: str = Field(
        default="",
        description="The scheduler name to use for the pod spec. When non-empty, this value is applied to the pod's schedulerName field, enabling custom schedulers such as KAI Scheduler. Empty string omits schedulerName so the cluster default scheduler is used.",
    )

    launcher_image: str = Field(
        default="nvcr.io/nvidia/nemo-microservices/jobs-launcher:latest",
        description="Container image that contains the jobs-launcher binary.",
    )
    workload_identity: KubernetesWorkloadIdentityConfig = Field(
        default_factory=KubernetesWorkloadIdentityConfig,
        description="Kubernetes workload identity configuration.",
    )


class KubernetesJobExecutionProfileConfig(BaseKubernetesExecutionProfileConfig):
    """Configuration for Kubernetes execution environment."""


class KubernetesJobExecutionProfile(BaseExecutionProfile):
    """
    Execution configuration for a Kubernetes Job.
    This is used to define the executor type, provider, profile, and any additional configuration
    required for the executor to run the job on Kubernetes
    """

    backend: Literal["kubernetes_job"] = "kubernetes_job"
    config: KubernetesJobExecutionProfileConfig = Field(
        description="Additional configuration for the kubernetes executor",
    )

    @property
    def supports_persistent_storage(self) -> bool:
        """Indicates if the execution profile supports persistent storage."""
        return self.config.storage is not None and self.config.storage.pvc_name != ""


# ---------------------------------------------------------------------------
# Volcano
# ---------------------------------------------------------------------------


class VolcanoJobExecutionProfileConfig(BaseKubernetesExecutionProfileConfig):
    """Configuration for Volcano Job Execution Profile"""

    queue: str = Field(
        default="default",
        description="The Volcano queue to submit the job to.",
    )
    scheduler_name: str = Field(
        default="volcano",
        description="The scheduler name to use for the Volcano job.",
    )

    max_retry: int = Field(default=0, description="maxRetry indicates the maximum number of retries allowed by the job")

    plugins: dict[str, Any] = Field(
        default_factory=dict,
        description="plugins indicates the plugins used by Volcano when the job is scheduled. We always add the pytorch plugin if more than one node.",
    )

    enable_multi_node_networking: bool = Field(
        default=True,
        description="Enable multi-node networking injection. Sets annotations to trigger Kyverno policy mutations.",
    )


class VolcanoJobExecutionProfile(BaseExecutionProfile):
    """Volcano Job Execution Profile"""

    backend: Literal["volcano_job"] = "volcano_job"
    config: VolcanoJobExecutionProfileConfig = Field(
        description="Additional configuration for the kubernetes executor",
    )

    @property
    def supports_persistent_storage(self) -> bool:
        """Indicates if the execution profile supports persistent storage."""
        return self.config.storage is not None and self.config.storage.pvc_name != ""


# ---------------------------------------------------------------------------
# Subprocess
# ---------------------------------------------------------------------------


class SubprocessJobExecutionProfileConfig(JobExecutionProfileConfig):
    working_directory: str = Field(
        default="/tmp/nhx-subprocess-jobs",
        description="Root directory for subprocess job state, config, storage, and logs.",
    )
    graceful_shutdown_timeout_seconds: int = Field(
        default=10,
        description="How long to wait after SIGTERM before force killing the process group.",
    )
    cleanup_completed_jobs_immediately: bool = Field(
        default=False,
        description="Keep subprocess working directories by default so runs remain inspectable.",
    )


class SubprocessJobExecutionProfile(BaseExecutionProfile):
    provider: ProviderRef = Field(default="subprocess")
    backend: Literal["subprocess"] = "subprocess"
    config: SubprocessJobExecutionProfileConfig = Field(
        default_factory=SubprocessJobExecutionProfileConfig,
        description="Additional configuration for the subprocess executor",
    )

    @property
    def supports_persistent_storage(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# OpenShell
# ---------------------------------------------------------------------------


class OpenShellJobTLSConfig(BaseModel):
    """mTLS material for an https OpenShell gateway. Unused for plaintext gateways."""

    model_config = ConfigDict(extra="forbid")

    ca_cert_path: str | None = Field(default=None, description="Path to the CA bundle that signed the gateway cert.")
    client_cert_path: str | None = Field(default=None, description="Path to the client certificate (mTLS).")
    client_key_path: str | None = Field(default=None, description="Path to the client private key (mTLS).")

    @model_validator(mode="after")
    def validate_client_identity_pair(self) -> OpenShellJobTLSConfig:
        if (self.client_cert_path is None) != (self.client_key_path is None):
            raise ValueError("client_cert_path and client_key_path must be set together")
        return self


class OpenShellJobEgressConfig(BaseModel):
    """The NeMo Helix endpoint a job sandbox must be able to reach directly.

    Becomes the platform egress rule in the generated sandbox policy. This shape
    is separate from the deployments plugin's ``HelixEgressConfig`` so the jobs
    service does not depend on that plugin; the values track it.
    """

    model_config = ConfigDict(extra="forbid")

    host: str = Field(default="host.docker.internal", description="Platform host reachable from inside a sandbox.")
    port: int = Field(default=8080, ge=1, description="Platform port (the inference gateway / API listener).")
    # Value sets track OpenShell's authored policy schema: protocol also allows "graphql" and
    # "" (L4-only). TLS "" inspects automatically, "skip" turns inspection off.
    protocol: Literal["rest", "websocket", "graphql", "sql", ""] = Field(
        default="rest",
        description='OpenShell L7 protocol: "rest", "websocket", "graphql", "sql", or "" for L4-only.',
    )
    tls: Literal["", "skip"] = Field(
        default="",
        description='TLS handling: "" (default) for automatic inspection, "skip" to disable inspection.',
    )
    access: Literal["read-only", "read-write", "full"] = Field(
        default="full",
        description='OpenShell access preset: "read-only", "read-write", or "full".',
    )
    binaries: list[str] = Field(
        default_factory=list,
        description=(
            "Binaries permitted to open the egress connection. Empty uses the backend default set, "
            "which must include the jobs-launcher and the task venv python."
        ),
    )


class OpenShellJobExecutionProfileConfig(JobExecutionProfileConfig):
    """Configuration for the OpenShell job execution profile.

    Schedules job steps as OpenShell sandboxes via the gateway gRPC API. The
    sandbox runs the jobs-launcher as its canonical main process
    (``SandboxSpec.command``), which fetches the step config from the platform
    (``jobs-launcher run --fetch-step-config``) and then runs the task command.
    """

    model_config = ConfigDict(extra="forbid")

    gateway_endpoint: str = Field(
        default="http://127.0.0.1:17670",
        description=(
            "OpenShell gateway endpoint as a URL (http://host:port or https://host:port). "
            "The gRPC target is the same host:port; http implies plaintext, https implies TLS."
        ),
    )
    workspace: str = Field(
        default="default",
        description="OpenShell gateway workspace that every sandbox RPC is scoped to. The gateway creates 'default'.",
    )
    insecure: bool | None = Field(
        default=None,
        description="Force plaintext (True) or TLS (False). When None, derived from the endpoint scheme.",
    )
    tls: OpenShellJobTLSConfig | None = Field(default=None, description="mTLS material for an https gateway.")
    request_timeout_seconds: int = Field(
        default=120,
        ge=1,
        description="Per-RPC deadline for control-plane calls (create/get/delete/list).",
    )
    platform_egress: OpenShellJobEgressConfig | None = Field(
        default_factory=OpenShellJobEgressConfig,
        description=(
            "Platform endpoint a job sandbox reaches directly. Drives the generated default sandbox "
            "policy and is injected into a policy_path policy as a mandatory egress rule. Set to null "
            "to grant the sandbox NO direct egress (breaks secrets fetch and OTLP log export)."
        ),
    )
    policy_path: str | None = Field(
        default=None,
        description=(
            "Path to a hand-written OpenShell SandboxPolicy YAML applied to created sandboxes. When unset, "
            "a default-deny policy is generated from platform_egress plus the sandbox filesystem defaults."
        ),
    )
    image: str | None = Field(
        default=None,
        description=(
            "Sandbox image for job steps. When unset, falls back to the platform CPU tasks image "
            "(nhx-tasks); OpenShell needs the openshell variant (nhx-tasks-openshell)."
        ),
    )
    default_entrypoint: list[str] | None = Field(
        default=None,
        description=(
            "Entrypoint for job steps that do not declare one. The OpenShell supervisor ignores "
            "the image ENTRYPOINT, so the sandbox command must name the binary explicitly. "
            "When unset, defaults to the nhx-tasks entrypoint (/app/.venv/bin/nemo-helix)."
        ),
    )
    egress_proxy: str | None = Field(
        default=None,
        description=(
            "HTTP proxy URL injected as HTTP_PROXY/HTTPS_PROXY for the sandbox workload. "
            "The docker driver gives sandboxes no network of their own: all egress goes "
            "through the supervisor's proxy at http://127.0.0.1:3128, which enforces the "
            "sandbox policy. The k8s driver fences with NetworkPolicies instead, so leave "
            "this unset there."
        ),
    )

    @model_validator(mode="after")
    def _validate_endpoint(self) -> OpenShellJobExecutionProfileConfig:
        parsed = urlparse(self.gateway_endpoint)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("gateway_endpoint must be a URL like http://host:port or https://host:port")
        return self

    def grpc_target(self) -> str:
        """host:port for the gRPC channel, derived from the endpoint URL."""
        parsed = urlparse(self.gateway_endpoint)
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        return f"{host}:{port}"

    def use_insecure(self) -> bool:
        """Whether to open a plaintext channel (explicit override, else scheme-derived)."""
        if self.insecure is not None:
            return self.insecure
        return urlparse(self.gateway_endpoint).scheme == "http"


class OpenShellJobExecutionProfile(BaseExecutionProfile):
    """Execution configuration for an OpenShell job."""

    backend: Literal["openshell"] = "openshell"
    config: OpenShellJobExecutionProfileConfig = Field(
        description="Additional configuration for the OpenShell executor",
    )

    @property
    def supports_persistent_storage(self) -> bool:
        """OpenShell sandboxes do not support persistent storage."""
        return False


# ---------------------------------------------------------------------------
# E2E test backend
# ---------------------------------------------------------------------------


class E2EJobExecutionProfile(BaseExecutionProfile):
    """
    Execution configuration for E2E testing.
    This backend auto-completes jobs without actually running containers,
    making tests fast and deterministic.
    """

    backend: Literal["e2e"] = "e2e"
    config: JobExecutionProfileConfig = Field(
        default_factory=JobExecutionProfileConfig,
        description="Configuration for the e2e test executor",
    )

    @property
    def supports_persistent_storage(self) -> bool:
        """E2E backend claims to support persistent storage since jobs auto-complete without execution."""
        return True
