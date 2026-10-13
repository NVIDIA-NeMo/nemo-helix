# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Caller-facing settings for running a Harbor taskset as one Kubernetes Job per task."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class HarborTaskJobsExecutionInput(BaseModel):
    """Settings that run a Harbor taskset as many independent task Jobs instead of one Job.

    Without this input, an agent evaluation runs the whole taskset inside one Job on one
    Docker host. Passing it as ``execution`` switches to the Kubernetes path: submit pins the
    taskset revision, registers one Jobs Service Job per task, and returns a single taskset
    run that tracks progress and combines the per-task results.

    It exists because these settings mean nothing on the Docker path: where task Jobs are
    placed, which sandbox they may use, and when to give up on a run. Keeping them in their
    own model, selected by ``kind``, lets future execution strategies add new models rather
    than new ``submit`` methods. It never changes benchmark content or which agent runs.

    Submit validates this input, resolves the percent form of ``max_failed_tasks`` into a
    count, and freezes the result into the run's stored plan; later changes to operator
    profiles or defaults do not affect an accepted run.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["harbor-task-jobs"] = Field(
        default="harbor-task-jobs",
        description="Execution strategy: one Kubernetes Job per Harbor task. Future strategies add new kinds.",
    )
    sandbox_profile: str = Field(
        description=(
            "Name of an operator-approved sandbox profile: the sandbox provider and policy each task's "
            "agent and environment run under. Operators define the available profiles."
        ),
        examples=["harbor-standard"],
    )
    dispatcher_profile: str = Field(
        default="local-k8s",
        description=(
            "Name of a Jobs Service profile that decides where task Jobs run and what resources they get. "
            "Operators define the available profiles; defaults to local-k8s, the profile the Helm chart ships."
        ),
        examples=["local-k8s"],
    )
    missing_prebuilt_image: Literal["fail", "build"] = Field(
        default="fail",
        description=(
            "What to do when a task pins a prebuilt image (its docker_image) and that image is no longer "
            "in the registry. 'fail' (default) fails that task with image_unavailable. 'build' builds the "
            "image from the task's Dockerfile instead and records the substitution; the result may differ "
            "from the image the task author benchmarked, and a task without a Dockerfile still fails. "
            "Does not apply to tasks without a pinned image, which are always built from their Dockerfile, "
            "or to an image this run built that disappears mid-run, which is always rebuilt."
        ),
        examples=["fail", "build"],
    )
    max_concurrent_task_jobs: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Maximum number of this run's task Jobs running at once. "
            "Omit to rely only on Jobs Service admission limits."
        ),
        examples=[5],
    )
    max_failed_tasks: Annotated[int, Field(ge=0)] | Annotated[str, Field(pattern=r"^\d{1,3}%$")] | None = Field(
        default=None,
        description=(
            "Failed tasks the run tolerates, as a count or a percent of the taskset's tasks. "
            "Once exceeded, the run fails and its remaining task Jobs are cancelled. "
            "Submit resolves a percent into a count. Omit for no limit."
        ),
        examples=[3, "10%"],
    )
