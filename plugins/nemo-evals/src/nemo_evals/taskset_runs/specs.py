# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolved execution plan stored as digest-verified bytes in Files."""

from typing import Literal

from nemo_evals.jobs.agent_spec import AgentEvalSpec
from pydantic import BaseModel, ConfigDict, Field


class TasksetRunPlan(BaseModel):
    """Frozen M1 inputs, pinned at acceptance and never rebound by the store.

    Top-level fields are frozen; consumers receive deep copies of the nested spec.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    workspace: str
    taskset_ref: str
    spec: AgentEvalSpec
    missing_prebuilt_image: Literal["fail", "build"]
    max_concurrent_task_jobs: int | None = Field(ge=1)
    max_failed_tasks: int | None = Field(ge=0)
