# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the nhx-unsloth compiler and container tasks.

Modeled after :mod:`nhx.automodel.config`. Environment variables use
the ``NHX_UNSLOTH_`` prefix and drive image resolution for the 4-step
container ``PlatformJobSpec`` the plugin's :meth:`~nemo_unsloth_plugin.jobs.jobs.UnslothJob.compile` builds.
"""

from nhx.common.config import create_service_config_class, get_platform_config, get_service_config
from pydantic import Field


class UnslothConfig(create_service_config_class("unsloth")):  # ty: ignore[unsupported-base]
    """Environment variables use the ``NHX_UNSLOTH_`` prefix."""

    image_registry: str | None = Field(
        default=None,
        description=(
            "Registry host/path prefix for nhx-unsloth-tasks and nhx-unsloth-training. "
            "Override via NHX_UNSLOTH_IMAGE_REGISTRY for other environments, defaults to the platform's image registry."
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

    default_training_execution_profile: str = Field(
        default="gpu",
        description="Default GPU execution profile when the job spec omits training.execution_profile.",
    )


config = get_service_config(UnslothConfig)
platform_config = get_platform_config()
