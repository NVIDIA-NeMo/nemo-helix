# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolved execution plan stored as digest-verified bytes in Files."""

from datetime import datetime, timezone
from typing import Self

from nemo_evals.harbor.remote.images import PreparedHarborEnvironment
from nemo_evals.harbor.remote.profile import ResolvedHarborSandboxPolicy
from nemo_evals.jobs.agent_spec import AgentEvalSpec
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class RunIntakePublication(BaseModel):
    """Where the run publishes its trials: one Intake Evaluation, named after the run, in this Experiment.

    Frozen at submit with the Experiment resolved to its id and the agent identity resolved from the
    target, so task Jobs and the controller never resolve them again. The Evaluation's name is the
    run record's name, so it isn't stored again here.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    experiment: str = Field(min_length=1, description="The Experiment's name, as the request gave it.")
    experiment_id: str = Field(min_length=1, description="The Experiment's id, resolved from its name at submit.")
    agent_name: str = Field(min_length=1, description="Agent name recorded on each published trajectory.")
    dataset_name: str = Field(
        min_length=1, description="The Evaluation's dataset: the request's, or the taskset without its digest."
    )
    dataset_version: str = Field(
        min_length=1, description="The dataset's version: the request's, or the pinned taskset's content digest."
    )


class PlannedAgentLayers(BaseModel):
    """The agent layers a run's tasks wait for: which install, and until when."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str = Field(min_length=1, description="The run's agent install key (`AgentInstall.layer_key`).")
    wait_until: datetime = Field(
        description="When a task stops waiting for a layer that isn't ready and runs on its task_environments image."
    )


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
    evaluation spec, an explicit concurrency cap (None denotes no limit), the sandbox
    policy, dispatcher image and dispatcher profile resolved from operator
    configuration, and each task's prepared images.
    Task Jobs read only the plan, so a config change, a moved image tag or a newer
    build between a partial submit and its retry can't give sibling Jobs different
    images, lifetimes or egress.
    Development runs from before these fields existed are deleted, not migrated.

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
    max_concurrent_task_jobs: int | None = Field(
        ge=1,
        description="Maximum task Jobs of this run running at once; None relies only on Jobs Service admission limits.",
    )
    sandbox_policy: ResolvedHarborSandboxPolicy = Field(
        description="The operator sandbox profile execution.sandbox.profile named (or the default), frozen at submit."
    )
    dispatcher_image: str = Field(
        min_length=1, description="Task Job image from evals.runs.harbor, or the platform's nhx-tasks image."
    )
    dispatcher_profile: str = Field(
        min_length=1, description="Jobs execution profile every task Job of this run uses (CPU class)."
    )
    task_environments: list[PreparedHarborEnvironment] = Field(
        description="Each task's images, pinned by digest at submit; task_environments[i] belongs to spec.tasks[i]."
    )
    intake: RunIntakePublication | None = Field(
        default=None, description="Publish the run's trials to this Intake Experiment; None publishes nowhere."
    )
    agent_layers: PlannedAgentLayers | None = Field(
        default=None,
        description=(
            "The agent layers submit started for the run's agent: each task's image with the agent installed. Each "
            "task waits in image_preparing for its layer, then runs on it, or on its task_environments image when the "
            "layer fails or isn't ready by wait_until. None: the run's agent isn't layered."
        ),
    )

    @model_validator(mode="after")
    def _one_environment_per_task(self) -> Self:
        if len(self.task_environments) != len(self.spec.tasks):
            raise ValueError(
                f"task_environments has {len(self.task_environments)} entries for {len(self.spec.tasks)} tasks"
            )
        return self


class AcceptedTaskOutput(BaseModel):
    """One task Job whose result the run's bundle includes (MVP slice of M5 story 7).

    TODO(M5 story 7): add ``manifest_ref`` and ``attempt_id`` once task Jobs write output manifests.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_index: int = Field(ge=0, description="Position of the task in the plan's spec.tasks.")
    job: str = Field(min_length=1, description="The task Job, <run_id>-task-<index>.")
    bundle_ref: str = Field(
        min_length=1, description="The task Job's agent-eval-results artifact: workspace/fileset#path."
    )


class TasksetBundleSpec(BaseModel):
    """The bundle Job's whole input: which task bundles to merge into the run's one bundle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    workspace: str
    plan_ref: str
    plan_digest: str
    accepted: list[AcceptedTaskOutput] = Field(min_length=1, description="Sorted by task_index.")
    started_at: datetime = Field(description="The run's accepted_at.")
    finished_at: datetime = Field(description="When the newest accepted task Job last changed.")

    @field_validator("started_at", "finished_at")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        """Read a naive time as UTC: the Jobs API returns naive timestamps."""
        return value if value.utcoffset() is not None else value.replace(tzinfo=timezone.utc)
