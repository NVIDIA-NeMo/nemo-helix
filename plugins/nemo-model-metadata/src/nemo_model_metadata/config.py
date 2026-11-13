# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import StrEnum
from typing import ClassVar

from nemo_helix_plugin.config import NemoConfig
from pydantic import Field


class SourceName(StrEnum):
    PROVIDER = "provider"
    NGC = "ngc"


class ModelMetadataConfig(NemoConfig):
    plugin_name: ClassVar[str] = "model_metadata"
    plugin_description: ClassVar[str] = "Configuration for the model metadata enrichment controller."

    sources: list[SourceName] = Field(
        default_factory=lambda: [SourceName.PROVIDER, SourceName.NGC],
        description=(
            "Metadata sources in priority order. A field is taken from the first source that has it. "
            "Remove a source to stop calling it, for example on air-gapped installs."
        ),
    )
    interval_seconds: float = Field(default=300.0, gt=0, description="Seconds between reconcile cycles.")
    refresh_interval_seconds: float = Field(
        default=86400.0,
        gt=0,
        description="Seconds before a model's metadata, or a lookup that found nothing, is fetched again.",
    )
    request_timeout_seconds: float = Field(default=15.0, gt=0, description="Timeout for each metadata request.")
    ngc_url: str = Field(default="https://api.ngc.nvidia.com", description="NGC catalog API base URL.")
    ngc_org: str = Field(default="qc69jvmznzxy", description="NGC org that publishes the build.nvidia.com endpoints.")
