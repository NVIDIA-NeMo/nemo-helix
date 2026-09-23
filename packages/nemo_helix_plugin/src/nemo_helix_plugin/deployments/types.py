# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed request/response models for the deployments-plugin client.

Pure-Pydantic mirrors of the ``nemo-deployments`` HTTP wire contract
(``/apis/deployments/v2/...``) for Volumes, Deployments, and DeploymentConfigs.
Kept independent of ``nemo_deployments_plugin`` on purpose: that plugin already
depends on ``nemo-helix-plugin``, so importing it back would be a dependency
cycle. Same client-DTO boundary as ``models``/``files``/``secrets``.
"""

from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict

from pydantic import BaseModel, ConfigDict, Field


def _default_access_modes() -> list[AccessMode]:
    return ["ReadWriteOnce"]


# ---------------------------------------------------------------------------
# Shared literals (mirror nemo_deployments_plugin.types)
# ---------------------------------------------------------------------------

DeploymentStatus = Literal[
    "PENDING",
    "STARTING",
    "READY",
    "SUCCEEDED",
    "FAILED",
    "LOST",
    "UNKNOWN",
    "DELETING",
]
VolumeStatus = Literal["PENDING", "BOUND", "DELETING", "RELEASED", "FAILED"]
DesiredState = Literal["READY", "STOPPED"]
RestartPolicy = Literal["Always", "OnFailure", "Never"]
AccessMode = Literal["ReadWriteOnce", "ReadOnlyMany", "ReadWriteMany"]
DriftRecoveryAction = Literal["recreate", "ignore"]
PrerequisiteCondition = Literal["ready", "succeeded"]
EndpointProtocol = Literal["http", "https", "grpc", "tcp"]


# ---------------------------------------------------------------------------
# Nested value objects (PodSpec-shaped; carried in DeploymentConfig)
# ---------------------------------------------------------------------------


class _AliasModel(BaseModel):
    """Base for wire models that use camelCase aliases but accept snake_case."""

    model_config = ConfigDict(populate_by_name=True)


class SecretRef(_AliasModel):
    workspace: str
    name: str


class EnvVar(_AliasModel):
    name: str
    value: str | None = None
    value_from: dict[str, Any] | None = Field(default=None, alias="valueFrom")
    secret_ref: SecretRef | None = Field(default=None, alias="secretRef")


class ContainerPort(_AliasModel):
    name: str | None = None
    container_port: int = Field(alias="containerPort")
    protocol: Literal["TCP", "UDP"] = "TCP"


class ResourceRequirements(BaseModel):
    limits: dict[str, str] = Field(default_factory=dict)
    requests: dict[str, str] = Field(default_factory=dict)


class VolumeMount(_AliasModel):
    name: str
    mount_path: str = Field(alias="mountPath")
    read_only: bool = Field(default=False, alias="readOnly")
    sub_path: str | None = Field(default=None, alias="subPath")


class ExecAction(BaseModel):
    command: list[str] = Field(default_factory=list)


class HTTPGetAction(BaseModel):
    path: str = "/"
    port: int | str = 8080
    scheme: Literal["HTTP", "HTTPS"] = "HTTP"


class TCPSocketAction(BaseModel):
    port: int | str


class Probe(_AliasModel):
    exec_action: ExecAction | None = Field(default=None, alias="exec")
    http_get: HTTPGetAction | None = Field(default=None, alias="httpGet")
    tcp_socket: TCPSocketAction | None = Field(default=None, alias="tcpSocket")
    initial_delay_seconds: int = Field(default=0, alias="initialDelaySeconds")
    period_seconds: int = Field(default=10, alias="periodSeconds")
    timeout_seconds: int = Field(default=1, alias="timeoutSeconds")
    failure_threshold: int = Field(default=3, alias="failureThreshold")


class Container(_AliasModel):
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
    restart_policy: RestartPolicy | None = Field(default=None, alias="restartPolicy")


class ConfigFile(BaseModel):
    path: str
    content: str
    mode: int = 0o644


class Toleration(_AliasModel):
    key: str | None = None
    operator: Literal["Equal", "Exists"] = "Equal"
    value: str | None = None
    effect: Literal["NoSchedule", "PreferNoSchedule", "NoExecute"] | None = None
    toleration_seconds: int | None = Field(default=None, alias="tolerationSeconds")


class PodSecurityContext(_AliasModel):
    run_as_user: int | None = Field(default=None, alias="runAsUser")
    run_as_group: int | None = Field(default=None, alias="runAsGroup")
    fs_group: int | None = Field(default=None, alias="fsGroup")


class Affinity(_AliasModel):
    node_affinity: dict[str, Any] | None = Field(default=None, alias="nodeAffinity")
    pod_affinity: dict[str, Any] | None = Field(default=None, alias="podAffinity")
    pod_anti_affinity: dict[str, Any] | None = Field(default=None, alias="podAntiAffinity")


class WorkloadIdentitySpec(_AliasModel):
    enabled: bool = False
    workload_kind: str | None = Field(default=None, alias="workloadKind")
    workload_id: str | None = Field(default=None, alias="workloadId")
    token_audience: str | None = Field(default=None, alias="tokenAudience")
    service_account_name: str | None = Field(default=None, alias="serviceAccountName")
    token_expiration_seconds: int = Field(default=3600, alias="tokenExpirationSeconds")


class DockerDeploymentConfig(BaseModel):
    network: str | None = None


class K8sDeploymentConfig(_AliasModel):
    namespace: str | None = None
    service_account: str | None = Field(default=None, alias="serviceAccount")
    node_selector: dict[str, str] = Field(default_factory=dict, alias="nodeSelector")
    tolerations: list[Toleration] = Field(default_factory=list)
    affinity: Affinity | None = None
    topology_spread_constraints: list[dict[str, Any]] = Field(default_factory=list, alias="topologySpreadConstraints")
    security_context: PodSecurityContext | None = Field(default=None, alias="securityContext")
    pod_annotations: dict[str, str] = Field(default_factory=dict, alias="podAnnotations")


class OpenShellDeploymentConfig(_AliasModel):
    policy_path: str | None = Field(default=None, alias="policyPath")


class DeploymentBackendConfig(BaseModel):
    docker: DockerDeploymentConfig | None = None
    k8s: K8sDeploymentConfig | None = None
    openshell: OpenShellDeploymentConfig | None = None


class DockerVolumeConfig(_AliasModel):
    driver: str = "local"
    mount_point: str | None = None
    init_chmod: str | None = Field(default=None, alias="initChmod")
    init_image: str | None = Field(default=None, alias="initImage")


class K8sVolumeConfig(_AliasModel):
    storage_class: str | None = Field(default=None, alias="storageClass")
    namespace: str | None = None


class VolumeBackendConfig(BaseModel):
    docker: DockerVolumeConfig | None = None
    k8s: K8sVolumeConfig | None = None


class DriftRecoveryPolicy(BaseModel):
    action: DriftRecoveryAction = "recreate"
    max_attempts: int | None = None
    initial_delay_seconds: int | None = None
    max_delay_seconds: int | None = None


class Prerequisite(BaseModel):
    deployment_name: str
    condition: PrerequisiteCondition = "succeeded"


class Endpoint(BaseModel):
    name: str
    url: str
    protocol: EndpointProtocol = "http"


class StatusEvent(BaseModel):
    status: DeploymentStatus
    message: str = ""
    timestamp: str = ""


# ---------------------------------------------------------------------------
# Entity response DTOs (mirror the entity-store wire envelope + payload)
# ---------------------------------------------------------------------------


class _EntityEnvelope(_AliasModel):
    """Common entity-store fields present on every response entity.

    The server serializes these alongside the entity payload; they are
    read-only from a client's perspective. ``model_config`` allows the extra
    fields to arrive without error and preserves forward-compat.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    name: str = ""
    workspace: str
    project: str | None = None
    id: str = ""
    entity_id: str = ""
    parent: str | None = None
    db_version: int = 1
    created_at: str | None = None
    created_by: str | None = None
    updated_at: str | None = None
    updated_by: str | None = None


class Volume(_EntityEnvelope):
    """Persistent volume request and observed state (response DTO)."""

    size: str = "1Gi"
    access_modes: list[AccessMode] = Field(default_factory=_default_access_modes)
    backend_config: VolumeBackendConfig = Field(default_factory=VolumeBackendConfig, alias="backendConfig")
    status: VolumeStatus = "PENDING"
    status_message: str = ""
    error_details: dict[str, Any] | None = None


class DeploymentConfig(_EntityEnvelope):
    """Immutable PodSpec-shaped deployment template (response DTO)."""

    containers: list[Container] = Field(default_factory=list)
    init_containers: list[Container] = Field(default_factory=list, alias="initContainers")
    volume_mounts: list[VolumeMount] = Field(default_factory=list, alias="volumeMounts")
    config_files: list[ConfigFile] = Field(default_factory=list, alias="configFiles")
    restart_policy: RestartPolicy = Field(default="Always", alias="restartPolicy")
    backoff_limit: int = Field(default=6, alias="backoffLimit")
    drift_recovery: DriftRecoveryPolicy = Field(default_factory=DriftRecoveryPolicy, alias="driftRecovery")
    labels: dict[str, str] = Field(default_factory=dict)
    backend_config: DeploymentBackendConfig = Field(default_factory=DeploymentBackendConfig, alias="backendConfig")
    auth_proxy_sidecar: bool = Field(default=False, alias="authProxySidecar")
    auth_proxy_sidecar_identity: str | None = Field(default=None, alias="authProxySidecarIdentity")
    auth_proxy_sidecar_on_behalf_of: str | None = Field(default=None, alias="authProxySidecarOnBehalfOf")
    workload_identity: WorkloadIdentitySpec | None = Field(default=None, alias="workloadIdentity")


class Deployment(_EntityEnvelope):
    """Desired and observed deployment state (response DTO)."""

    deployment_config: str
    desired_state: DesiredState = "READY"
    executor: str | None = None
    prerequisites: list[Prerequisite] = Field(default_factory=list)
    status: DeploymentStatus = "PENDING"
    status_message: str = ""
    endpoints: list[Endpoint] = Field(default_factory=list)
    exit_code: int | None = None
    error_details: dict[str, Any] | None = None
    status_history: list[StatusEvent] = Field(default_factory=list)
    auth_context: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Request bodies (mirror nemo_deployments_plugin.schema)
# ---------------------------------------------------------------------------


class RequestEnvVar(_AliasModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    name: str
    value: str | None = None
    value_from: dict[str, Any] | None = Field(default=None, alias="valueFrom")
    secret_ref: SecretRef | None = Field(default=None, alias="secretRef")


class RequestContainer(_AliasModel):
    name: str
    image: str
    command: list[str] = Field(default_factory=list)
    args: list[str] = Field(default_factory=list)
    env: list[RequestEnvVar] = Field(default_factory=list)
    ports: list[ContainerPort] = Field(default_factory=list)
    resources: ResourceRequirements = Field(default_factory=ResourceRequirements)
    volume_mounts: list[VolumeMount] = Field(default_factory=list, alias="volumeMounts")
    liveness_probe: Probe | None = Field(default=None, alias="livenessProbe")
    readiness_probe: Probe | None = Field(default=None, alias="readinessProbe")
    restart_policy: RestartPolicy | None = Field(default=None, alias="restartPolicy")


class CreateVolumeRequest(BaseModel):
    name: str
    size: str = "1Gi"
    access_modes: list[AccessMode] = Field(default_factory=_default_access_modes)
    backend_config: VolumeBackendConfig = Field(default_factory=VolumeBackendConfig)


class CreateDeploymentConfigRequest(_AliasModel):
    name: str
    containers: list[RequestContainer] = Field(default_factory=list)
    init_containers: list[RequestContainer] = Field(default_factory=list, alias="initContainers")
    volume_mounts: list[VolumeMount] = Field(default_factory=list, alias="volumeMounts")
    config_files: list[ConfigFile] = Field(default_factory=list, alias="configFiles")
    restart_policy: RestartPolicy = Field(default="Always", alias="restartPolicy")
    backoff_limit: int = Field(default=6, ge=1, alias="backoffLimit")
    drift_recovery: DriftRecoveryPolicy | None = Field(default=None, alias="driftRecovery")
    labels: dict[str, str] = Field(default_factory=dict)
    backend_config: DeploymentBackendConfig = Field(default_factory=DeploymentBackendConfig, alias="backendConfig")
    workload_identity: WorkloadIdentitySpec | None = Field(default=None, alias="workloadIdentity")


class CreateDeploymentRequest(BaseModel):
    name: str
    deployment_config: str
    desired_state: DesiredState = "READY"
    executor: str | None = None
    prerequisites: list[Prerequisite] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Query parameter types
#
# Per-field filtering uses the deepObject form ``filter[field]=value`` — the
# plugin routes parse only bracketed keys (``parse_deep_object`` skips a bare
# ``filter=...`` key), so the per-field ``filter[...]`` entries below are what
# actually filter. This mirrors the agents/iam/agent-hardener clients.
# ``status_in`` on deployments is a real explicit (non-filter) query param.
# The bare ``filter`` key is kept for the whole-object escape hatch some routes
# accept, and ``filter[<field>]`` allows any entity field (e.g.
# ``filter[workspace]``) beyond the typed ones enumerated here.
# ---------------------------------------------------------------------------


ListVolumesQueryParams = TypedDict(
    "ListVolumesQueryParams",
    {
        "page": NotRequired[int],
        "page_size": NotRequired[int],
        "sort": NotRequired[str],
        "filter": NotRequired[str],
        "filter[status]": NotRequired[VolumeStatus],
    },
    total=False,
)


ListDeploymentsQueryParams = TypedDict(
    "ListDeploymentsQueryParams",
    {
        "page": NotRequired[int],
        "page_size": NotRequired[int],
        "sort": NotRequired[str],
        "filter": NotRequired[str],
        "status_in": NotRequired[str],
        "filter[status]": NotRequired[DeploymentStatus],
        "filter[deployment_config]": NotRequired[str],
        "filter[desired_state]": NotRequired[DesiredState],
        "filter[executor]": NotRequired[str],
    },
    total=False,
)


ListDeploymentConfigsQueryParams = TypedDict(
    "ListDeploymentConfigsQueryParams",
    {
        "page": NotRequired[int],
        "page_size": NotRequired[int],
        "sort": NotRequired[str],
        "filter": NotRequired[str],
        "filter[restart_policy]": NotRequired[RestartPolicy],
    },
    total=False,
)
