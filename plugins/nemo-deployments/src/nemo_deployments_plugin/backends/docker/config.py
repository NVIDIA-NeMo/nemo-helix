# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Executor-level Docker backend configuration."""

from __future__ import annotations

import os
from typing import Literal

from nemo_deployments_plugin.backends.labels import DEFAULT_RESOURCE_SCOPE
from pydantic import BaseModel, Field, field_validator, model_validator

DOCKER_ENDPOINT_MODE_ENV_VAR = "NEMO_DEPLOYMENTS_DOCKER_ENDPOINT_MODE"
DOCKER_NETWORK_ENV_VAR = "NEMO_DEPLOYMENTS_DOCKER_NETWORK"
LEGACY_MODELS_DOCKER_NETWORKING_MODE_ENV_VAR = "MODELS_DOCKER_NETWORKING_MODE"
LEGACY_MODELS_DOCKER_NETWORK_ENV_VAR = "MODELS_DOCKER_NETWORK"
DEFAULT_WORKLOAD_TOKEN_WRITER_IMAGE = (
    "docker.io/library/busybox:1.37.0@sha256:9532d8c39891ca2ecde4d30d7710e01fb739c87a8b9299685c63704296b16028"
)
DockerEndpointMode = Literal["host", "network"]


class DockerAdditionalVolumeMount(BaseModel):
    """Executor-level Docker volume or bind mount applied to every deployment container."""

    volume_name: str = Field(min_length=1, description="Docker volume name or host path to mount.")
    mount_path: str = Field(min_length=1, description="Container path where the volume is mounted.")
    read_only: bool = Field(default=False, description="Mount the source read-only when true.")


def _default_network() -> str | None:
    for env_var in (DOCKER_NETWORK_ENV_VAR, LEGACY_MODELS_DOCKER_NETWORK_ENV_VAR):
        value = os.getenv(env_var)
        if value:
            return value
    return None


def _default_endpoint_mode() -> str:
    value = os.getenv(DOCKER_ENDPOINT_MODE_ENV_VAR)
    if value:
        return _normalize_endpoint_mode(value)
    return _normalize_endpoint_mode(os.getenv(LEGACY_MODELS_DOCKER_NETWORKING_MODE_ENV_VAR, "host"))


def _normalize_endpoint_mode(value: str) -> str:
    mode = value.lower()
    if mode == "dond":
        return "network"
    if mode in {"local", "dind"}:
        return "host"
    return mode


class DockerExecutorConfig(BaseModel):
    """Knobs for a named docker executor instance (not entity backend_config)."""

    docker_host: str | None = Field(default=None, description="Override DOCKER_HOST for this executor.")
    docker_timeout: int = Field(
        default=600,
        ge=1,
        description="Docker client timeout in seconds for pull/create/status operations (default: 10 minutes).",
    )
    oneshot_observe_timeout_seconds: int = Field(
        default=5,
        ge=1,
        description=(
            "Max seconds to wait for a Never one-shot container to exit during create. "
            "Should stay near the deployments controller reconcile interval (default 5s). "
            "Longer jobs return STARTING and finish via read_status polling."
        ),
    )
    pull_images: bool = Field(default=True, description="Pull container images before run when missing locally.")
    network: str | None = Field(
        default_factory=_default_network,
        description=(
            "Default Docker network for containers created by this executor. "
            f"Can also be set with {DOCKER_NETWORK_ENV_VAR}; "
            f"{LEGACY_MODELS_DOCKER_NETWORK_ENV_VAR} is accepted for compatibility."
        ),
    )
    endpoint_mode: DockerEndpointMode = Field(
        default_factory=_default_endpoint_mode,
        validate_default=True,
        description=(
            "'host' reports and probes host-published ports. 'network' reports and probes "
            "Docker-network container names and container ports, for controller/API "
            "processes that run inside a container on the same Docker network. Can also "
            f"be set with {DOCKER_ENDPOINT_MODE_ENV_VAR}; "
            f"{LEGACY_MODELS_DOCKER_NETWORKING_MODE_ENV_VAR}=dond selects 'network' for compatibility."
        ),
    )
    additional_volume_mounts: list[DockerAdditionalVolumeMount] = Field(
        default_factory=list,
        description=(
            "Executor-level Docker volume mounts applied to every deployment container. "
            "Useful for shared platform assets such as gateway CA bundles."
        ),
    )
    workload_token_writer_image: str = Field(
        default=DEFAULT_WORKLOAD_TOKEN_WRITER_IMAGE,
        min_length=1,
        description="Image used for the workload identity token-writer container.",
    )
    resource_scope: str = Field(
        default=DEFAULT_RESOURCE_SCOPE,
        min_length=1,
        description=(
            "Owner scope label for Docker-managed resources. Orphan cleanup only lists resources with the "
            "same scope, allowing multiple platform instances to share one Docker daemon."
        ),
    )
    port_range_start: int = Field(
        default=49152,
        ge=1,
        le=65535,
        description="First host port to consider when publishing container ports for this executor.",
    )
    port_range_end: int = Field(
        default=49251,
        ge=1,
        le=65535,
        description="Last host port (inclusive) to consider when publishing container ports for this executor.",
    )

    @field_validator("endpoint_mode", mode="before")
    @classmethod
    def _normalize_endpoint_mode(cls, value: object) -> object:
        if isinstance(value, str):
            return _normalize_endpoint_mode(value)
        return value

    @model_validator(mode="after")
    def _validate_config(self) -> DockerExecutorConfig:
        if self.port_range_start > self.port_range_end:
            raise ValueError("port_range_start must not exceed port_range_end")
        if self.endpoint_mode == "network" and not self.network:
            raise ValueError("endpoint_mode='network' requires network")
        return self
