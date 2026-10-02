# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the nhx-automodel compiler and tasks."""

from nhx.common.config import create_service_config_class, get_platform_config, get_service_config
from pydantic import Field


class AutomodelConfig(create_service_config_class("automodel")):  # type: ignore
    """Environment variables use the NHX_AUTOMODEL_ prefix."""

    image_registry: str | None = Field(
        default=None,
        description=(
            "Registry host/path prefix for nhx-customizer-tasks and nhx-automodel-training. "
            "Override via NHX_AUTOMODEL_IMAGE_REGISTRY for other environments, defaults to the platform's image registry."
        ),
    )
    training_image: str | None = Field(
        default=None,
        description="Override entire GPU training image (registry/name:tag).",
    )
    tasks_image: str | None = Field(
        default=None,
        description="Override entire CPU tasks image (registry/name:tag).",
    )

    default_job_resource_cpu_request: str = Field(default="1")
    default_job_resource_memory_request: str = Field(default="8Gi")
    default_job_resource_cpu_limit: str = Field(default="4")
    default_job_resource_memory_limit: str = Field(default="16Gi")

    training_staleness_timeout_seconds: int = Field(
        default=3600,
        description="Terminate training if no task progress within this many seconds (0 disables).",
    )

    default_training_execution_profile: str = Field(
        default="gpu",
        description="Default GPU execution profile when the job spec omits training.execution_profile.",
    )


config = get_service_config(AutomodelConfig)
platform_config = get_platform_config()

# Legacy compiler attribute names
config.training_automodel_image = config.training_image
