# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared auto-deployment schemas and rules for customization job specs."""

from nemo_platform_plugin.deployment.rules import (
    LORA_ENABLED_REQUIRED_MESSAGE,
    is_unbound_deployment_config,
    reject_lora_without_lora_enabled,
)
from nemo_platform_plugin.deployment.schemas import (
    DEPLOYMENT_CONFIG_DESCRIPTION,
    DeploymentParams,
    ToolCallParams,
)

__all__ = [
    "DEPLOYMENT_CONFIG_DESCRIPTION",
    "LORA_ENABLED_REQUIRED_MESSAGE",
    "DeploymentParams",
    "ToolCallParams",
    "is_unbound_deployment_config",
    "reject_lora_without_lora_enabled",
]
