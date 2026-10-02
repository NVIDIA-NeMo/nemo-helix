# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validated inputs for scaled-evals Platform Jobs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class TaskImageBuildSpec(BaseModel):
    """Describe one immutable task-image build attempt."""

    task_id: str = Field(min_length=1)
    revision: int = Field(ge=1)
    build_attempt: int = Field(ge=1)
    backend: Literal["buildkit", "cloudbuild", "image_builder_service", "prebuilt"]
    object_key: str = Field(min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid", frozen=True)


class EvaluationExecutionSpec(BaseModel):
    """Describe one immutable evaluation execution."""

    evaluation_id: str = Field(min_length=1)
    execution_number: int = Field(ge=1)
    runtime: str = Field(min_length=1)
    deadline_seconds: int = Field(ge=1)

    model_config = ConfigDict(extra="forbid", frozen=True)
