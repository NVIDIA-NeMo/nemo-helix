# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the Safe Synthesizer plugin."""

from typing import ClassVar

from nemo_helix_plugin.config import NemoConfig
from pydantic import Field


class SafeSynthesizerConfig(NemoConfig):
    """Configuration for Safe Synthesizer plugin API and task compilation."""

    plugin_name: ClassVar[str] = "safe_synthesizer"
    plugin_description: ClassVar[str] = "Configuration for the NeMo Safe Synthesizer plugin."

    host: str = "0.0.0.0"
    port: int = 8000
    entrypoint: list[str] = Field(
        default_factory=lambda: ["python", "-m", "nemo_safe_synthesizer_plugin.tasks.safe_synthesizer"]
    )
    job_executor_profile: str = "default"
    container_image: str = "nhx-safe-synthesizer-tasks"
    container_image_ref: str | None = Field(
        default=None,
        description=(
            "Optional fully qualified task image reference. When set, this bypasses platform "
            "NHX_IMAGE_REGISTRY / NHX_IMAGE_TAG qualification for Safe Synthesizer jobs."
        ),
    )
    default_job_resource_memory_request: str = "16G"
    default_job_resource_cpu_request: str = "4"
    default_job_resource_memory_limit: str = "16G"
    default_job_resource_cpu_limit: str = "4"


def get_config() -> SafeSynthesizerConfig:
    """Return the Safe Synthesizer plugin configuration singleton."""
    return SafeSynthesizerConfig.get()


config = get_config()
