# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Annotated, Any, Literal, TypeAlias

import data_designer.config as dd
from data_designer.config.analysis.dataset_profiler import DatasetProfilerResults
from data_designer.config.dataset_metadata import DatasetMetadata
from data_designer_nemo.columns import validate_no_custom_columns
from data_designer_nemo.seed import validate_seed_source_for_execution_context
from nemo_helix_plugin.functions.frames import Done, Error, Heartbeat
from pydantic import BaseModel, Field, model_validator

LogLevel = Literal["debug", "info", "warn", "warning", "error"]


class PreviewSpec(BaseModel):
    config: dd.DataDesignerConfig
    num_records: int | None = None

    @model_validator(mode="before")
    @classmethod
    def validate_config_for_platform(cls, data: Any) -> Any:
        validate_no_custom_columns(data)
        validate_seed_source_for_execution_context(data)
        return data


class LogFrame(BaseModel):
    kind: Literal["log"] = "log"
    level: LogLevel
    message: str


class DatasetMetadataFrame(BaseModel):
    kind: Literal["dataset_metadata"] = "dataset_metadata"
    metadata: DatasetMetadata


class AnalysisFrame(BaseModel):
    kind: Literal["analysis"] = "analysis"
    analysis: DatasetProfilerResults


class DatasetFrame(BaseModel):
    kind: Literal["dataset"] = "dataset"
    records: list[dict[str, Any]]


class ProcessorOutputFrame(BaseModel):
    kind: Literal["processor_output"] = "processor_output"
    processor_name: str
    records: list[dict[str, Any]]


PreviewFrame: TypeAlias = Annotated[
    LogFrame | DatasetMetadataFrame | AnalysisFrame | DatasetFrame | ProcessorOutputFrame | Heartbeat | Done | Error,
    Field(discriminator="kind"),
]
