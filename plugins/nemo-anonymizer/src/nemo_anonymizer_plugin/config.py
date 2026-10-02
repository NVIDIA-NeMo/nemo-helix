# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the Anonymizer plugin."""

from __future__ import annotations

import os
from pathlib import Path
from typing import ClassVar, Self

from nemo_helix_plugin.config import NemoConfig
from pydantic import BaseModel, Field, model_validator

# Reuse a bundled tiktoken cache if present (Anonymizer relies on Data Designer
# under the hood, which uses tiktoken). In the cpu-tasks image we expect the
# DD plugin's bundled cache to already be on disk; if it isn't, fall back to
# the network-based tiktoken behavior.
_dd_bundled_tiktoken_cache = Path(__file__).parents[3] / "nemo-data-designer" / "tiktoken-cache"
if _dd_bundled_tiktoken_cache.is_dir():
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(_dd_bundled_tiktoken_cache))


#: Advertised as ``maximum`` on ``PreviewRequest.num_records``; deployments may raise the
#: runtime bound past it, so the schema states the default rather than the effective limit.
DEFAULT_MAX_PREVIEW_NUM_RECORDS = 10


class PreviewNumRecords(BaseModel):
    max: int = Field(default=DEFAULT_MAX_PREVIEW_NUM_RECORDS, ge=1)
    default: int = Field(default=DEFAULT_MAX_PREVIEW_NUM_RECORDS, ge=1)

    @model_validator(mode="after")
    def check_compatibility(self) -> Self:
        if self.default > self.max:
            raise ValueError("Default num records cannot be greater than max num records")
        return self


class AnonymizerPluginConfig(NemoConfig):
    plugin_name: ClassVar[str] = "anonymizer"
    plugin_description: ClassVar[str] = "Anonymizer plugin configuration"

    preview_num_records: PreviewNumRecords = PreviewNumRecords()
    job_executor_profile: str = "default"


def get_config() -> AnonymizerPluginConfig:
    return AnonymizerPluginConfig.get()
