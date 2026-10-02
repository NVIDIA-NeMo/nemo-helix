# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deployments plugin configuration."""

from __future__ import annotations

from typing import Any, ClassVar

from nemo_helix_plugin.config import NemoConfig
from pydantic import BaseModel, Field, model_validator


class ExecutorConfigEntry(BaseModel):
    name: str = Field(description="Unique executor name used by Deployment.executor.")
    backend: str = Field(description="Backend type key registered in BACKEND_CLASSES.")
    config: dict[str, Any] = Field(default_factory=dict)


class ControllerConfig(BaseModel):
    """Configuration for the deployments reconcile controller."""

    interval_seconds: int = Field(default=5, gt=0, description="Reconciliation loop interval in seconds.")
    drift_recovery_max_attempts: int = Field(default=5, ge=0, description="Max drift recovery attempts before FAILED.")
    drift_recovery_initial_delay_seconds: int = Field(
        default=5, ge=0, description="Initial delay for drift recovery backoff."
    )
    drift_recovery_max_delay_seconds: int = Field(
        default=300, ge=0, description="Max delay cap for drift recovery backoff."
    )
    orphan_cleanup_interval_seconds: int = Field(
        default=30,
        ge=0,
        description="Run orphaned backend resource cleanup after this many seconds (0 disables).",
    )
    terminal_orphan_grace_seconds: int = Field(
        default=3600,
        ge=0,
        description=(
            "Keep terminal deployment backend resources (SUCCEEDED/FAILED) protected from "
            "orphan cleanup for this many seconds after the terminal status transition (0 disables)."
        ),
    )
    starting_timeout_seconds: int = Field(
        default=3600,
        ge=0,
        description="Max seconds in STARTING before FAILED (0 disables).",
    )
    deleting_timeout_seconds: int = Field(
        default=300,
        ge=0,
        description="Max seconds in DELETING before FAILED (0 disables).",
    )

    @model_validator(mode="after")
    def _validate_backoff(self) -> ControllerConfig:
        if self.drift_recovery_initial_delay_seconds > self.drift_recovery_max_delay_seconds:
            raise ValueError("drift_recovery_initial_delay_seconds must not exceed drift_recovery_max_delay_seconds")
        return self


class DeploymentsConfig(NemoConfig):
    plugin_name: ClassVar[str] = "deployments"
    plugin_description: ClassVar[str] = "Configuration for the NeMo Helix deployments plugin."

    executors: list[ExecutorConfigEntry] = Field(
        default_factory=list,
        description="Named executor instances. May be empty at scaffold time.",
    )
    default_executor: str | None = Field(
        default=None,
        description="Default executor when Deployment.executor is unset.",
    )
    controller: ControllerConfig = Field(default_factory=ControllerConfig)
    auth_proxy_image_name: str = Field(
        default="nhx-api",
        description=(
            "Image name for the auth-proxy sidecar (qualified with the platform image registry/tag). "
            "Must be an nhx-api image (runs `nemo services run --sidecars auth-proxy`)."
        ),
    )
    auth_proxy_image: str = Field(
        default="",
        description="Optional fully-qualified image override for the auth-proxy sidecar.",
    )
    auth_proxy_port: int = Field(
        default=8090,
        description="Loopback port the auth-proxy sidecar listens on; workloads target it as their platform base URL.",
    )
