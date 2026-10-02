# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import data_designer.config as dd
from data_designer_nemo.columns import validate_no_custom_columns
from data_designer_nemo.seed import validate_seed_source_for_execution_context
from pydantic import BaseModel, model_validator


# This is the user-facing type expected in an API request to create a job
class DataDesignerJobConfig(BaseModel):
    # This is needed to ensure we don't generate separate
    # -Input/-Output schemas for the child objects.
    model_config = {"json_schema_mode_override": "validation"}

    num_records: int
    config: dd.DataDesignerConfig

    @model_validator(mode="before")
    @classmethod
    def validate_config_for_platform(cls, data: Any) -> Any:
        validate_no_custom_columns(data)
        validate_seed_source_for_execution_context(data)
        return data


# This is the internal object we store on the HelixJobStep to pass to the task.
class DataDesignerStepConfig(BaseModel):
    job_config: DataDesignerJobConfig
    model_providers: list[dd.ModelProvider]
    model_configs: list[dd.ModelConfig]
