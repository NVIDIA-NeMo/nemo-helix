# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared auto-deployment schemas for customization job specs."""

from nemo_platform_plugin.deployment.schemas import (
    DEPLOYMENT_CONFIG_DESCRIPTION,
    LORA_ENABLED_REQUIRED_MESSAGE,
    DeploymentParams,
    ToolCallParams,
    reject_lora_without_lora_enabled,
)

__all__ = [
    "DEPLOYMENT_CONFIG_DESCRIPTION",
    "LORA_ENABLED_REQUIRED_MESSAGE",
    "DeploymentParams",
    "ToolCallParams",
    "reject_lora_without_lora_enabled",
]
