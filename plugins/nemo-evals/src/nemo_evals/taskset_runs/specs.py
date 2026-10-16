# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolved execution plan stored as digest-verified bytes in Files."""

from typing import Literal

from nemo_evals.jobs.agent_spec import AgentEvalSpec
from pydantic import BaseModel, ConfigDict, Field

MissingPrebuiltImagePolicy = Literal["fail", "build"]
"""What to do when a task's pinned prebuilt image is gone: fail the task, or build it from its Dockerfile.

Shared by the submit request and the frozen plan so the two cannot drift.
"""


class TasksetRunPlan(BaseModel):
    """Freeze what submit resolved, so every later step runs exactly what was accepted.

    Submit resolves references once (taskset to a pinned revision, agent and metrics
    to their current definitions, a failure percent to a count). The controller and
    task Jobs read this plan instead of resolving again, so a new taskset revision,
    re-registered agent, or edited metric after submit cannot change a running run.

    The plan is stored in Files as ``plan.json`` in its own fileset
    (``<workspace>/<run_id>-plan-<6 hex>#plan.json``); the run record holds that
    ref as ``plan_ref`` and the SHA-256 of its bytes as ``plan_digest``.

    Inputs are a run ID and workspace, pinned taskset revision, resolved agent
    evaluation spec, missing-image policy, and explicit concurrency/failure caps.
    Each cap must be supplied even when None denotes no limit; percentages must
    already have been resolved into a nonnegative failure count.

    Top-level fields are frozen, but nested spec models remain mutable;
    each verified storage read reconstructs independent models for its consumer.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(
        description=(
            "Name of the run record this plan belongs to (run-<16 hex>). "
            "Also prefixes the plan fileset and each task Job name (<run_id>-task-<index>)."
        )
    )
    workspace: str = Field(description="Workspace that owns the run, its plan fileset, and its task Jobs.")
    taskset_ref: str = Field(
        description=(
            "Taskset revision pinned at submit (workspace/name#revision, for example default/sort-suite#3), "
            "so later taskset edits do not change what this run evaluates."
        )
    )
    spec: AgentEvalSpec = Field(
        description=(
            "Resolved agent evaluation spec: target agent, metrics, and self-contained task snapshots. "
            "A member's task_index is its position in spec.tasks."
        )
    )
    missing_prebuilt_image: MissingPrebuiltImagePolicy = Field(
        description=(
            "Copied from the request: when a task's pinned prebuilt image is gone from the registry, "
            "'fail' fails that task and 'build' builds it from the task's Dockerfile instead."
        )
    )
    max_concurrent_task_jobs: int | None = Field(
        ge=1,
        description="Maximum task Jobs of this run running at once; None relies only on Jobs Service admission limits.",
    )
    max_failed_tasks: int | None = Field(
        ge=0,
        description=(
            "Failed tasks the run tolerates before it fails and cancels remaining task Jobs. "
            "Always a count: submit resolves a request percent (10% of 100 tasks -> 10). None means no limit."
        ),
    )
