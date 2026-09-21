# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deployment plugin entity definitions and supporting Pydantic types."""

from __future__ import annotations

from typing import Any, Literal, Self

from nemo_deployments_plugin.constants import (
    ENTITY_TYPE_DEPLOYMENT,
    ENTITY_TYPE_DEPLOYMENT_CONFIG,
    ENTITY_TYPE_VOLUME,
)
from nemo_deployments_plugin.types import (
    AccessMode,
    DeploymentStatus,
    DesiredState,
    DriftRecoveryAction,
    Endpoint,
    PrerequisiteCondition,
    RestartPolicy,
    VolumeStatus,
)
from nemo_platform_plugin.auth import AuthContext
from nemo_platform_plugin.client.constants import WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR
from nemo_platform_plugin.entity import NemoEntity
from pydantic import BaseModel, Field, PrivateAttr, computed_field, model_validator

WORKLOAD_IDENTITY_TOKEN_MAX_EXPIRATION_SECONDS = 86400


class SecretRef(BaseModel):
    """Reference to a Platform secret resolved by the substrate backend."""

    workspace: str
    name: str


class EnvVar(BaseModel):
    name: str
    value: str | None = None
    value_from: dict[str, Any] | None = Field(default=None, alias="valueFrom")
    secret_ref: SecretRef | None = Field(default=None, alias="secretRef")

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def validate_single_source(self) -> EnvVar:
        """Reject ambiguous environment variables with multiple value sources."""
        sources = (self.value, self.value_from, self.secret_ref)
        if sum(source is not None for source in sources) > 1:
            raise ValueError("EnvVar may define only one of value, valueFrom, or secretRef")
        return self


class ContainerPort(BaseModel):
    name: str | None = None
    container_port: int = Field(alias="containerPort")
    protocol: Literal["TCP", "UDP"] = "TCP"

    model_config = {"populate_by_name": True}


class ResourceRequirements(BaseModel):
    limits: dict[str, str] = Field(default_factory=dict)
    requests: dict[str, str] = Field(default_factory=dict)


class VolumeMount(BaseModel):
    name: str
    mount_path: str = Field(alias="mountPath")
    read_only: bool = Field(default=False, alias="readOnly")
    sub_path: str | None = Field(default=None, alias="subPath")

    model_config = {"populate_by_name": True}


class ExecAction(BaseModel):
    command: list[str] = Field(default_factory=list)


class HTTPGetAction(BaseModel):
    path: str = "/"
    port: int | str = 8080
    scheme: Literal["HTTP", "HTTPS"] = "HTTP"


class TCPSocketAction(BaseModel):
    port: int | str


class Probe(BaseModel):
    exec_action: ExecAction | None = Field(default=None, alias="exec")
    http_get: HTTPGetAction | None = Field(default=None, alias="httpGet")
    tcp_socket: TCPSocketAction | None = Field(default=None, alias="tcpSocket")
    initial_delay_seconds: int = Field(default=0, alias="initialDelaySeconds")
    period_seconds: int = Field(default=10, alias="periodSeconds")
    timeout_seconds: int = Field(default=1, alias="timeoutSeconds")
    failure_threshold: int = Field(default=3, alias="failureThreshold")

    model_config = {"populate_by_name": True}


class Container(BaseModel):
    name: str
    image: str
    command: list[str] = Field(default_factory=list)
    args: list[str] = Field(default_factory=list)
    env: list[EnvVar] = Field(default_factory=list)
    ports: list[ContainerPort] = Field(default_factory=list)
    resources: ResourceRequirements = Field(default_factory=ResourceRequirements)
    volume_mounts: list[VolumeMount] = Field(default_factory=list, alias="volumeMounts")
    liveness_probe: Probe | None = Field(default=None, alias="livenessProbe")
    readiness_probe: Probe | None = Field(default=None, alias="readinessProbe")
    restart_policy: RestartPolicy | None = Field(
        default=None,
        alias="restartPolicy",
        description="Per-container restart policy for init containers; Always enables k8s native sidecar.",
    )

    model_config = {"populate_by_name": True}


class ConfigFile(BaseModel):
    path: str
    content: str
    mode: int = 0o644


class Toleration(BaseModel):
    key: str | None = None
    operator: Literal["Equal", "Exists"] = "Equal"
    value: str | None = None
    effect: Literal["NoSchedule", "PreferNoSchedule", "NoExecute"] | None = None
    toleration_seconds: int | None = Field(default=None, alias="tolerationSeconds")

    model_config = {"populate_by_name": True}


class LabelSelector(BaseModel):
    match_labels: dict[str, str] = Field(default_factory=dict, alias="matchLabels")

    model_config = {"populate_by_name": True}


class LocalObjectReference(BaseModel):
    name: str


class PodSecurityContext(BaseModel):
    run_as_user: int | None = Field(default=None, alias="runAsUser")
    run_as_group: int | None = Field(default=None, alias="runAsGroup")
    fs_group: int | None = Field(default=None, alias="fsGroup")

    model_config = {"populate_by_name": True}


class Affinity(BaseModel):
    node_affinity: dict[str, Any] | None = Field(default=None, alias="nodeAffinity")
    pod_affinity: dict[str, Any] | None = Field(default=None, alias="podAffinity")
    pod_anti_affinity: dict[str, Any] | None = Field(default=None, alias="podAntiAffinity")

    model_config = {"populate_by_name": True}


class WorkloadIdentitySpec(BaseModel):
    """Workload token-exchange settings for deployment backends."""

    enabled: bool = False
    workload_kind: str | None = Field(default=None, alias="workloadKind")
    workload_id: str | None = Field(default=None, alias="workloadId")
    token_audience: str | None = Field(default=None, alias="tokenAudience")
    service_account_name: str | None = Field(default=None, alias="serviceAccountName")
    token_expiration_seconds: int = Field(
        default=3600,
        ge=600,
        le=WORKLOAD_IDENTITY_TOKEN_MAX_EXPIRATION_SECONDS,
        alias="tokenExpirationSeconds",
        description="Requested subject-token lifetime in seconds, capped at 24 hours.",
    )

    model_config = {"populate_by_name": True}


class DockerDeploymentConfig(BaseModel):
    network: str | None = None


class K8sDeploymentConfig(BaseModel):
    namespace: str | None = None
    service_account: str | None = Field(default=None, alias="serviceAccount")
    node_selector: dict[str, str] = Field(default_factory=dict, alias="nodeSelector")
    tolerations: list[Toleration] = Field(default_factory=list)
    affinity: Affinity | None = None
    topology_spread_constraints: list[dict[str, Any]] = Field(
        default_factory=list,
        alias="topologySpreadConstraints",
        description=(
            "Pod topology spread constraints applied to every Job/Deployment pod rendered "
            "from this config. Each entry is a raw Kubernetes topologySpreadConstraint object "
            "(maxSkew, topologyKey, whenUnsatisfiable, labelSelector, ...). k8s-only."
        ),
    )
    security_context: PodSecurityContext | None = Field(default=None, alias="securityContext")
    pod_annotations: dict[str, str] = Field(
        default_factory=dict,
        alias="podAnnotations",
        description=(
            "Annotations stamped onto the pod-template metadata for every Job/Deployment "
            "pod rendered from this config (e.g. an Istio native-sidecar annotation so a "
            "mesh-injected proxy terminates when a puller Job's main container exits). "
            "k8s-only; ignored on docker/openshell backends."
        ),
    )

    job_ttl_seconds_after_finished: int | None = Field(
        default=None,
        alias="jobTtlSecondsAfterFinished",
        description=(
            "Per-deployment override for ttlSecondsAfterFinished on a finite (Never/OnFailure) Job, "
            "such as the weight-puller. When set, wins over the executor default; when unset, the "
            "executor default applies. Controls how quickly a completed Job's pod (and its "
            "ReadWriteOnce volume attachment) is reaped. A set value must be at least "
            "MIN_JOB_TTL_SECONDS_AFTER_FINISHED so the reconciler can observe Job completion before "
            "the pod is reaped. k8s-only."
        ),
    )

    model_config = {"populate_by_name": True}


class OpenShellDeploymentConfig(BaseModel):
    policy_path: str | None = Field(
        default=None,
        alias="policyPath",
        description=(
            "Path to a sandbox policy YAML file on the backend host. "
            "Overrides the executor-level default_policy_path for this deployment only."
        ),
    )

    model_config = {"populate_by_name": True}


class DeploymentBackendConfig(BaseModel):
    docker: DockerDeploymentConfig | None = None
    k8s: K8sDeploymentConfig | None = None
    openshell: OpenShellDeploymentConfig | None = None


class DockerVolumeConfig(BaseModel):
    driver: str = "local"
    mount_point: str | None = None
    init_chmod: str | None = Field(
        default=None,
        alias="initChmod",
        description=(
            "When set, an init container chmods the freshly created (root-owned) "
            "named volume to this mode (e.g. '0777') so a non-root workload (e.g. "
            "the HF weight-puller) can write to it — the docker analogue of a k8s fsGroup."
        ),
    )
    init_image: str | None = Field(
        default=None,
        alias="initImage",
        description="Image used for the init_chmod container (e.g. busybox).",
    )

    model_config = {"populate_by_name": True}


class K8sVolumeConfig(BaseModel):
    storage_class: str | None = Field(default=None, alias="storageClass")
    namespace: str | None = None

    model_config = {"populate_by_name": True}


class VolumeBackendConfig(BaseModel):
    docker: DockerVolumeConfig | None = None
    k8s: K8sVolumeConfig | None = None


class DriftRecoveryPolicy(BaseModel):
    action: DriftRecoveryAction = "recreate"
    max_attempts: int | None = Field(
        default=None,
        ge=0,
        description="Override controller drift_recovery_max_attempts when set.",
    )
    initial_delay_seconds: int | None = Field(
        default=None,
        ge=0,
        description="Override controller drift_recovery_initial_delay_seconds when set.",
    )
    max_delay_seconds: int | None = Field(
        default=None,
        ge=0,
        description="Override controller drift_recovery_max_delay_seconds when set.",
    )

    @model_validator(mode="after")
    def _validate_delays(self) -> DriftRecoveryPolicy:
        if (
            self.initial_delay_seconds is not None
            and self.max_delay_seconds is not None
            and self.initial_delay_seconds > self.max_delay_seconds
        ):
            raise ValueError("initial_delay_seconds must not exceed max_delay_seconds")
        return self


class Prerequisite(BaseModel):
    deployment_name: str = Field(
        description=(
            "Name of another Deployment in the same workspace (or workspace/name) that must "
            "satisfy condition before this deployment may start."
        ),
    )
    condition: PrerequisiteCondition = Field(
        default="succeeded",
        description=(
            "ready: prerequisite Deployment.status == READY. "
            "succeeded: prerequisite Deployment.status == SUCCEEDED with exit_code == 0."
        ),
    )


class StatusEvent(BaseModel):
    status: DeploymentStatus
    message: str = ""
    timestamp: str = ""


class DeploymentConfig(NemoEntity, entity_type=ENTITY_TYPE_DEPLOYMENT_CONFIG):
    """Immutable PodSpec-shaped deployment template."""

    containers: list[Container] = Field(default_factory=list)
    init_containers: list[Container] = Field(default_factory=list, alias="initContainers")
    volume_mounts: list[VolumeMount] = Field(default_factory=list, alias="volumeMounts")
    config_files: list[ConfigFile] = Field(default_factory=list, alias="configFiles")
    restart_policy: RestartPolicy = Field(default="Always", alias="restartPolicy")
    backoff_limit: int = Field(
        default=6,
        ge=1,
        alias="backoffLimit",
        description="Retry limit for OnFailure deployments; must be positive. Never deployments disable retries internally.",
    )
    drift_recovery: DriftRecoveryPolicy = Field(default_factory=DriftRecoveryPolicy, alias="driftRecovery")
    labels: dict[str, str] = Field(default_factory=dict)
    backend_config: DeploymentBackendConfig = Field(default_factory=DeploymentBackendConfig, alias="backendConfig")
    auth_proxy_sidecar: bool = Field(
        default=False,
        alias="authProxySidecar",
        description=(
            "Inject a loopback auth-proxy sidecar that stamps a service-principal identity header on the "
            "workload's platform calls. No-op when platform auth is disabled. The workload must target the "
            "proxy on localhost (see auth_proxy_sidecar_port)."
        ),
    )
    auth_proxy_sidecar_identity: str | None = Field(
        default=None,
        alias="authProxySidecarIdentity",
        description=(
            "Service-principal name the auth-proxy sidecar stamps (interpolated into "
            "'X-NMP-Principal-Id: service:<identity>'). Required when auth_proxy_sidecar is True."
        ),
    )
    auth_proxy_sidecar_on_behalf_of: str | None = Field(
        default=None,
        alias="authProxySidecarOnBehalfOf",
        description=(
            "Optional principal id the auth-proxy sidecar delegates to via "
            "'X-NMP-Principal-On-Behalf-Of'. When set, the service principal acts on behalf of this "
            "identity so the platform scopes the workload's access to what that principal can reach "
            "(e.g. the deployment's creator) rather than the service principal's full reach. "
            "Only meaningful when auth_proxy_sidecar is True."
        ),
    )
    workload_identity: WorkloadIdentitySpec | None = Field(default=None, alias="workloadIdentity")

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def _validate_auth_proxy_identity(self) -> DeploymentConfig:
        if self.auth_proxy_sidecar and not self.auth_proxy_sidecar_identity:
            raise ValueError("auth_proxy_sidecar_identity is required when auth_proxy_sidecar is True")
        return self

    @model_validator(mode="after")
    def _validate_workload_identity_env(self) -> DeploymentConfig:
        if self.workload_identity is None or not self.workload_identity.enabled:
            return self
        for container in [*self.init_containers, *self.containers]:
            for env_var in container.env:
                if env_var.name == WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR:
                    raise ValueError(f"{WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR} is managed by workload_identity")
        return self


class Deployment(NemoEntity, entity_type=ENTITY_TYPE_DEPLOYMENT):
    """Desired and observed deployment state."""

    _auth_context: AuthContext | None = PrivateAttr(default=None)

    deployment_config: str = Field(description="Name of the DeploymentConfig entity.")
    desired_state: DesiredState = Field(default="READY")
    executor: str | None = Field(
        default=None,
        description="Named executor registry entry; falls back to plugin default_executor.",
    )
    prerequisites: list[Prerequisite] = Field(default_factory=list)
    status: DeploymentStatus = Field(default="PENDING")
    status_message: str = Field(default="")
    endpoints: list[Endpoint] = Field(default_factory=list)
    exit_code: int | None = None
    error_details: dict[str, Any] | None = None
    status_history: list[StatusEvent] = Field(default_factory=list)

    # Reconciler (758) enforces restart_policy vs terminal status on DeploymentConfig:
    # Never -> SUCCEEDED terminal; Always/OnFailure -> READY while running.

    @computed_field(json_schema_extra={"nullable": True})
    @property
    def auth_context(self) -> AuthContext | None:
        return self._auth_context

    def with_auth_context(self, auth_context: AuthContext | None) -> Self:
        self._auth_context = auth_context
        return self


def _default_volume_access_modes() -> list[AccessMode]:
    return ["ReadWriteOnce"]


class Volume(NemoEntity, entity_type=ENTITY_TYPE_VOLUME):
    """Persistent volume request and observed state."""

    size: str = Field(default="1Gi", description="Requested storage size (Kubernetes quantity).")
    access_modes: list[AccessMode] = Field(default_factory=_default_volume_access_modes)
    backend_config: VolumeBackendConfig = Field(default_factory=VolumeBackendConfig, alias="backendConfig")
    status: VolumeStatus = Field(default="PENDING")
    status_message: str = Field(default="")
    error_details: dict[str, Any] | None = None

    model_config = {"populate_by_name": True}
