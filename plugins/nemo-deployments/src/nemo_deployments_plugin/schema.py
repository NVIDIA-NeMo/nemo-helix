# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deployments plugin API schema definitions — request bodies and filters."""

from __future__ import annotations

from typing import Any

from nemo_deployments_plugin.entities import (
    AccessMode,
    ConfigFile,
    ContainerPort,
    Deployment,
    DeploymentBackendConfig,
    DeploymentConfig,
    DeploymentStatus,
    DesiredState,
    DriftRecoveryPolicy,
    EnvVar,
    Prerequisite,
    Probe,
    ResourceRequirements,
    RestartPolicy,
    SecretRef,
    Volume,
    VolumeBackendConfig,
    VolumeMount,
    VolumeStatus,
    WorkloadIdentitySpec,
)
from nemo_helix_plugin.schema import NemoFilter, NemoListResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator


def _default_request_access_modes() -> list[AccessMode]:
    return ["ReadWriteOnce"]


class RequestEnvVar(BaseModel):
    """Public request env vars.

    An env var carries exactly one value source: a plaintext ``value``, a
    ``valueFrom`` projection, or a ``secretRef`` pointing at a Platform secret.
    ``secretRef`` values are resolved by the substrate backend at deploy time —
    docker injects the resolved value as a plaintext container env var, while
    k8s materializes a single per-deployment ``Secret`` and mounts it via
    ``envFrom`` so the plaintext never lands in the pod manifest.
    """

    name: str
    value: str | None = None
    value_from: dict[str, Any] | None = Field(default=None, alias="valueFrom")
    secret_ref: SecretRef | None = Field(default=None, alias="secretRef")

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
        json_schema_extra={
            # Keep the OpenAPI contract aligned with validate_single_source.
            "not": {"required": ["value", "valueFrom", "secretRef"]},
        },
    )

    @model_validator(mode="after")
    def validate_single_source(self) -> RequestEnvVar:
        sources = (self.value, self.value_from, self.secret_ref)
        if sum(source is not None for source in sources) > 1:
            raise ValueError("EnvVar may define only one of value, valueFrom, or secretRef")
        return self

    def to_entity(self) -> EnvVar:
        return EnvVar(
            name=self.name,
            value=self.value,
            valueFrom=self.value_from,
            secretRef=self.secret_ref,
        )


class RequestContainer(BaseModel):
    """Public request container."""

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

    model_config = ConfigDict(populate_by_name=True)


class CreateDeploymentConfigRequest(BaseModel):
    name: str
    containers: list[RequestContainer] = Field(default_factory=list)
    init_containers: list[RequestContainer] = Field(default_factory=list, alias="initContainers")
    volume_mounts: list[VolumeMount] = Field(default_factory=list, alias="volumeMounts")
    config_files: list[ConfigFile] = Field(default_factory=list, alias="configFiles")
    restart_policy: RestartPolicy = Field(default="Always", alias="restartPolicy")
    backoff_limit: int = Field(
        default=6,
        ge=1,
        alias="backoffLimit",
        description="Retry limit for OnFailure deployments; must be positive. Never deployments disable retries internally.",
    )
    drift_recovery: DriftRecoveryPolicy | None = Field(default=None, alias="driftRecovery")
    labels: dict[str, str] = Field(default_factory=dict)
    backend_config: DeploymentBackendConfig = Field(default_factory=DeploymentBackendConfig, alias="backendConfig")
    workload_identity: WorkloadIdentitySpec | None = Field(default=None, alias="workloadIdentity")

    model_config = ConfigDict(populate_by_name=True)


class CreateDeploymentRequest(BaseModel):
    name: str
    deployment_config: str = Field(
        description="DeploymentConfig name in this workspace, or workspace/name for cross-workspace refs.",
    )
    desired_state: DesiredState = "READY"
    executor: str | None = None
    prerequisites: list[Prerequisite] = Field(default_factory=list)


class CreateVolumeRequest(BaseModel):
    name: str
    size: str = "1Gi"
    access_modes: list[AccessMode] = Field(default_factory=_default_request_access_modes)
    backend_config: VolumeBackendConfig = Field(default_factory=VolumeBackendConfig)


class UpdateDeploymentStatusRequest(BaseModel):
    status: DeploymentStatus
    status_message: str = ""
    exit_code: int | None = None
    error_details: dict[str, Any] | None = None


class UpdateVolumeStatusRequest(BaseModel):
    status: VolumeStatus
    status_message: str = ""
    error_details: dict[str, Any] | None = None


class DeploymentConfigFilter(NemoFilter):
    restart_policy: RestartPolicy | None = None


class DeploymentFilter(NemoFilter):
    deployment_config: str | None = None
    desired_state: DesiredState | None = None
    executor: str | None = None
    status: DeploymentStatus | None = None


class VolumeFilter(NemoFilter):
    status: VolumeStatus | None = None


DeploymentConfigPage = NemoListResponse[DeploymentConfig]
DeploymentPage = NemoListResponse[Deployment]
VolumePage = NemoListResponse[Volume]
