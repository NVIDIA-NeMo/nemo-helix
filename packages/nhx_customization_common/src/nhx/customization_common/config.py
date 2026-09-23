# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared customization configuration (CPU tasks image override)."""

from nhx.common.config import create_service_config_class, get_service_config
from pydantic import Field


class CustomizationCommonConfig(create_service_config_class("customizer")):  # ty: ignore[unsupported-base]
    """Environment variables use the ``NHX_CUSTOMIZER_`` prefix."""

    tasks_image: str | None = Field(
        default=None,
        description=(
            "Override entire CPU tasks image (registry/name:tag) for all customization backends. "
            "Set via NHX_CUSTOMIZER_TASKS_IMAGE."
        ),
    )


config = get_service_config(CustomizationCommonConfig)
