# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import ClassVar

from nemo_helix_plugin.config import NemoConfig
from pydantic import Field


class NgcConfig(NemoConfig):
    plugin_name: ClassVar[str] = "ngc"
    plugin_description: ClassVar[str] = "Configuration for reading model metadata from the NGC catalog."

    url: str = Field(default="https://api.ngc.nvidia.com", description="NGC catalog API base URL.")
    org: str = Field(default="qc69jvmznzxy", description="NGC org that publishes the build.nvidia.com endpoints.")
