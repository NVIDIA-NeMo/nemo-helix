# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility imports for CLI telemetry event models."""

from __future__ import annotations

from nemo_helix_plugin.telemetry.events import (
    _CI_ENV_VARS,
    CommandInvokedEvent,
    DeploymentTypeEnum,
    JobRunEvent,
    OnboardingStepEvent,
    HelixTelemetryEvent,
    TaskStatusEnum,
    _deployment_type,
    is_ci_environment,
)

__all__ = [
    "_CI_ENV_VARS",
    "CommandInvokedEvent",
    "DeploymentTypeEnum",
    "JobRunEvent",
    "OnboardingStepEvent",
    "HelixTelemetryEvent",
    "TaskStatusEnum",
    "_deployment_type",
    "is_ci_environment",
]
