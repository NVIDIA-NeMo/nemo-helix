# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the Agents plugin."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from nemo_platform_plugin.config import NemoConfig, nmp_user_data_dir
from pydantic import BaseModel, Field, model_validator


class ControllerConfig(BaseModel):
    """Configuration for the AgentDeploymentController reconcile loop."""

    interval_seconds: int = Field(default=2, description="Reconciliation loop interval in seconds.")
    health_check_timeout_seconds: int = Field(
        default=120, description="Maximum time to wait for agent health check to succeed."
    )
    health_check_interval_seconds: int = Field(default=2, description="Health check poll interval in seconds.")
    port_range_start: int = Field(
        default=49152,
        description="First port in the range available for spawned agent processes (default: start of IANA dynamic/private range).",
    )
    port_range_end: int = Field(
        default=65535,
        description="Last port (inclusive) in the range available for spawned agent processes (default: end of IANA dynamic/private range).",
    )

    @model_validator(mode="after")
    def _validate_port_range(self) -> "ControllerConfig":
        if self.port_range_end < self.port_range_start:
            raise ValueError(
                f"port_range_end ({self.port_range_end}) must be >= port_range_start ({self.port_range_start})"
            )
        return self

    workspace_dir: Path = Field(
        default_factory=lambda: nmp_user_data_dir() / "agents",
        description=(
            "Root directory used by the in-memory runner backend for storing runtime "
            "artifacts (rendered NAT configs and per-deployment logs) under a "
            "'system/' subdirectory. Defaults to ``nmp_user_data_dir() / 'agents'`` "
            "(typically ``~/.local/share/nemo/agents``), so artifacts survive ``/tmp`` "
            "cleanup on macOS reboots and live in a documented, user-accessible "
            "location. Override the user-data root via ``NMP_DATA_DIR`` or "
            "``XDG_DATA_HOME``."
        ),
    )


class DeploymentsRunnerConfig(BaseModel):
    """Settings for container-mode agent deployments via the nemo-deployments plugin."""

    default_executor: str | None = Field(
        default=None,
        description="Named deployments-plugin executor used when mode-specific executors are unset.",
    )
    docker_executor: str | None = Field(
        default=None,
        description="Named executor for deployment_mode=docker. Falls back to default_executor.",
    )
    k8s_executor: str | None = Field(
        default=None,
        description="Named executor for deployment_mode=k8s. Falls back to default_executor.",
    )
    default_image: str = Field(
        default="",
        description="Default container image when CreateDeploymentRequest.image is omitted.",
    )
    container_port: int = Field(
        default=8000,
        description="Container port the agent server listens on (and readiness probe target).",
    )
    gateway_url_override: str | None = Field(
        default=None,
        description=(
            "Platform base URL baked into deployed agents as their inference endpoint, used verbatim "
            "for both docker and k8s. When unset, the deploy path derives a container-reachable URL: "
            "docker rewrites loopback hosts to host.docker.internal; k8s uses k8s_internal_base_url."
        ),
    )
    k8s_internal_base_url: str | None = Field(
        default=None,
        description=(
            "In-cluster platform base URL (the API Service DNS, e.g. http://<release>-nmp-api:8080) "
            "used as the inference endpoint for k8s-mode agents. Set automatically by the Helm chart. "
            "Read from NEMO_INTERNAL_BASE_URL, then NMP_INTERNAL_BASE_URL, when unset."
        ),
    )
    plugin_wheels_init_image: str | None = Field(
        default=None,
        description=(
            "Optional init-container image that stages workspace plugin wheels (k8s only). "
            "When unset, init_containers are omitted; AIRCORE-863 hardens the full contract."
        ),
    )
    config_mount_path: str = Field(
        default="/tmp/nemo/config.yaml",
        description=(
            "Path inside the container where the NAT workflow config is placed for nat-workflow-v1 "
            "deployments. Fabric deployments use agent.yaml in the same directory. Must be writable "
            "by every runtime user this image can run as: the plain-docker container user and the "
            "openshell sandbox user (uid 999, which cannot write the image's /workspace). /tmp is "
            "the writable intersection; k8s mounts it read-only there via a ConfigMap subPath."
        ),
    )


class AgentJobsConfig(BaseModel):
    """Settings for containerized agent jobs (``agents.execute``)."""

    default_image: str = Field(
        default="",
        description=(
            "Default container image for agents.execute jobs when "
            "ExecuteAgentJobConfig.image is omitted. Distinct from "
            "deployments.default_image, which is the AgentDeployment fallback. "
            "When both are empty the job inherits the jobs substrate's chain: "
            "the execution profile's default_task_image, then the platform "
            "CPU tasks image."
        ),
    )


class AgentsConfig(NemoConfig):
    """Configuration for the Agents plugin."""

    plugin_name: ClassVar[str] = "agents"
    plugin_description: ClassVar[str] = "Configuration for the NeMo Platform agents plugin."

    controller: ControllerConfig = Field(
        default_factory=ControllerConfig,
        description="Deployment controller settings.",
    )
    runner_backend: str = Field(
        default="in_memory",
        description=(
            "Default runner for subprocess-mode deployments. Container modes always use the "
            "deployments-plugin backend regardless of this setting."
        ),
    )
    deployments: DeploymentsRunnerConfig = Field(
        default_factory=DeploymentsRunnerConfig,
        description="Container-mode (docker/k8s) settings for the deployments-plugin runner.",
    )
    jobs: AgentJobsConfig = Field(
        default_factory=AgentJobsConfig,
        description="Containerized agent job settings.",
    )
