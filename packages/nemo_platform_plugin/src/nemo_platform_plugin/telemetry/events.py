# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared anonymous telemetry event models for NeMo Platform clients.

Field names and aliases follow the shared NeMo telemetry schema
(aire/microservices/nemo-telemetry, schemas/anonymous_events.json, v1.10).
"""

from __future__ import annotations

import os
from enum import Enum
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

_CI_ENV_VARS = (
    "CI",
    "GITLAB_CI",
    "GITHUB_ACTIONS",
    "BUILDKITE",
    "CIRCLECI",
    "JENKINS_URL",
    "TEAMCITY_VERSION",
    "TF_BUILD",
    "TRAVIS",
)
_FALSEY = ("", "0", "false", "no", "off")


def is_ci_environment() -> bool:
    return any(os.getenv(v, "").lower() not in _FALSEY for v in _CI_ENV_VARS)


class TaskStatusEnum(str, Enum):
    COMPLETED = "completed"
    ERROR = "error"
    CANCELED = "canceled"
    UNDEFINED = "undefined"


class DeploymentTypeEnum(str, Enum):
    CLI = "cli"
    SDK = "sdk"
    NVIDIA_INTERNAL = "nvidia-internal"
    UNDEFINED = "undefined"


def _deployment_type() -> DeploymentTypeEnum:
    raw = os.getenv("NEMO_DEPLOYMENT_TYPE", "cli").lower()
    try:
        return DeploymentTypeEnum(raw)
    except ValueError:
        return DeploymentTypeEnum.UNDEFINED


class PlatformTelemetryEvent(BaseModel):
    """Base for all platform events. extra="forbid" is a privacy guard."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    _event_name: ClassVar[str] = "undefined"
    _schema_version: ClassVar[str] = "1.10"

    nemo_source: str = Field(default="platform", serialization_alias="nemoSource")
    task_status: TaskStatusEnum = Field(serialization_alias="taskStatus")
    deployment_type: DeploymentTypeEnum = Field(default_factory=_deployment_type, serialization_alias="deploymentType")
    is_ci: bool = Field(default_factory=is_ci_environment, serialization_alias="isCi")


class OnboardingStepEvent(PlatformTelemetryEvent):
    _event_name: ClassVar[str] = "onboarding_step"

    step: str
    provider_type: str = Field(default="undefined", serialization_alias="providerType")
    models_discovered_bucket: str = Field(default="undefined", serialization_alias="modelsDiscoveredBucket")
    skills_target: str = Field(default="undefined", serialization_alias="skillsTarget")
    agent_deployed: bool = Field(default=False, serialization_alias="agentDeployed")


class CommandInvokedEvent(PlatformTelemetryEvent):
    _event_name: ClassVar[str] = "command_invoked"

    command: str
    duration_sec: float = Field(serialization_alias="durationSec")
    agent_mode: bool = Field(default=False, serialization_alias="agentMode")


class JobRunEvent(PlatformTelemetryEvent):
    _event_name: ClassVar[str] = "job_run"

    job_type: str = Field(serialization_alias="jobType")
    duration_sec: float = Field(default=-1.0, serialization_alias="durationSec")
    plugins: list[str] = Field(default_factory=list)
    model: str = "undefined"
    input_tokens: int = Field(default=-1, serialization_alias="inputTokens")
    output_tokens: int = Field(default=-1, serialization_alias="outputTokens")
