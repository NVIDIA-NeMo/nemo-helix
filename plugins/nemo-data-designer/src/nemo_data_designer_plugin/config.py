# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the Data Designer plugin."""

from __future__ import annotations

import os
from pathlib import Path
from typing import ClassVar, Self

from nemo_helix_plugin.config import NemoConfig
from pydantic import BaseModel, Field, model_validator

# Point tiktoken at the bundled cl100k_base encoding so the plugin works without
# outbound network access when the cache is bundled in the repo or image.
_bundled_tiktoken_cache = Path(__file__).parents[2] / "tiktoken-cache"
if _bundled_tiktoken_cache.is_dir():
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(_bundled_tiktoken_cache))


class PreviewNumRecords(BaseModel):
    max: int = Field(default=10, ge=1)
    default: int = Field(default=10, ge=1)

    @model_validator(mode="after")
    def check_compatibility(self) -> Self:
        if self.default > self.max:
            raise ValueError("Default num records cannot be greater than max num records")
        return self


class DataDesignerPluginConfig(NemoConfig):
    plugin_name: ClassVar[str] = "data_designer"
    plugin_description: ClassVar[str] = "Data Designer plugin configuration"

    preview_num_records: PreviewNumRecords = PreviewNumRecords()
    job_executor_profile: str = "default"


def get_config() -> DataDesignerPluginConfig:
    return DataDesignerPluginConfig.get()
