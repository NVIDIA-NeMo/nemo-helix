# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Digest-pinned images for one Harbor task, and the input of its task Job.

Contract module (M1 story 5) plus M2 story 6a's ``HarborTaskJobSpec`` and pure input builder.
"""

from datetime import datetime

from nemo_evals.harbor.remote.images import PreparedHarborEnvironment, PreparedImage
from nemo_evals.harbor.remote.profile import PINNED_IMAGE_REF_PATTERN, ResolvedHarborSandboxPolicy
from nemo_evals.jobs.agent_spec import AgentEvalSpec
from nemo_evals.taskset_runs.entities import TasksetRunEntity
from nemo_evals.taskset_runs.identifiers import member_name
from nemo_evals.taskset_runs.specs import TasksetRunPlan
from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "PINNED_IMAGE_REF_PATTERN",
    "HarborTaskJobSpec",
    "PreparedHarborEnvironment",
    "PreparedImage",
    "build_task_job_input",
]


class HarborTaskJobSpec(BaseModel):
    """The input of one task Job: one task of a run, its images, and the run's frozen sandbox policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    member_id: str = Field(description="task-<index>, the member this Job runs.")
    task_index: int = Field(ge=0)
    spec: AgentEvalSpec = Field(description="The run's spec with only this task.")
    environment: PreparedHarborEnvironment
    sandbox_policy: ResolvedHarborSandboxPolicy
    dispatcher_image: str = Field(pattern=PINNED_IMAGE_REF_PATTERN)
    execution_deadline_at: datetime


def build_task_job_input(
    plan: TasksetRunPlan, run: TasksetRunEntity, task_index: int, environment: PreparedHarborEnvironment
) -> HarborTaskJobSpec:
    """Build task ``task_index``'s Job input from the frozen plan only; pure.

    Raises:
        IndexError: ``task_index`` is outside the plan's tasks.
    """
    return HarborTaskJobSpec(
        run_id=plan.run_id,
        member_id=member_name(task_index),
        task_index=task_index,
        spec=plan.spec.only_task(task_index),
        environment=environment,
        sandbox_policy=plan.sandbox_policy,
        dispatcher_image=plan.dispatcher_image,
        execution_deadline_at=run.execution_deadline_at,
    )
