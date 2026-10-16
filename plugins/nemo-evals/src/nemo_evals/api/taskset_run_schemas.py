# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Caller-facing settings for running a Harbor taskset as one Kubernetes Job per task."""

import hashlib
import json
from datetime import datetime
from typing import Annotated, Literal, Self

from nemo_evals.api.fields import TasksetRef
from nemo_evals.jobs.agent_spec import AgentEvalInputSpec, HarborRunnerTarget
from nemo_evals.taskset_runs.entities import ResultState, TasksetRunEntity, TasksetRunPhase
from nemo_evals.taskset_runs.specs import MissingPrebuiltImagePolicy
from pydantic import BaseModel, ConfigDict, Field, model_validator


class HarborTaskJobsExecutionInput(BaseModel):
    """Settings that run a Harbor taskset as many independent task Jobs instead of one Job.

    Passing it as ``execution`` selects the Kubernetes taskset-run path: submit pins
    the taskset revision and returns a durable run handle. Later execution stages
    register one Jobs Service Job per task and combine the per-task results.

    It exists because these settings mean nothing on the Docker path: where task Jobs are
    placed, which sandbox they may use, and when to give up on a run. Keeping them in their
    own model, selected by ``kind``, lets future execution strategies add new models rather
    than new ``submit`` methods. It never changes benchmark content or which agent runs.

    Submit validates this input, resolves the percent form of ``max_failed_tasks`` into a
    count, and stores the M1 execution caps and missing-image policy in the run's plan.
    In M1, ``sandbox_policy`` and ``dispatcher_profile`` participate in request identity
    only; their names are intentionally neither resolved nor stored while execution is
    unavailable. M2 Story 3 validates these selections and adds the resolved sandbox
    policy, dispatcher profile and pinned dispatcher image as required plan fields before
    execution is enabled. Earlier development runs are deleted rather than migrated.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["harbor-task-jobs"] = Field(
        default="harbor-task-jobs",
        description="Execution strategy: one Kubernetes Job per Harbor task. Future strategies add new kinds.",
    )
    sandbox_policy: str = Field(
        description=(
            "Name of an operator-approved sandbox policy: which sandbox provider and isolation policy "
            "each task's agent and environment run under, meaning the security and isolation boundary. "
            "It does not control compute resources; see dispatcher_profile. Operators define the available policies."
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
    missing_prebuilt_image: MissingPrebuiltImagePolicy = Field(
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
            "Omit to rely only on the job controller's admission limits."
        ),
        examples=[5],
    )
    max_failed_tasks: Annotated[int, Field(ge=0)] | Annotated[str, Field(pattern=r"^(100|[1-9]?\d)%$")] | None = Field(
        default=None,
        description=(
            "Failed tasks the run tolerates, as a count or a percent of the taskset's tasks. "
            "Once exceeded, the run fails and its remaining task Jobs are cancelled. "
            "Submit resolves a percent into a count. Omit for no limit."
        ),
        examples=[3, "10%"],
    )

    def resolve_max_failed_tasks(self, task_count: int) -> int | None:
        """Convert a percentage tolerance to a floored task count.

        Args:
            task_count: Number of resolved tasks, including zero for pure arithmetic.

        Returns:
            The count tolerance, or None when the run has no failure limit.
        """
        if isinstance(self.max_failed_tasks, str):
            return int(self.max_failed_tasks[:-1]) * task_count // 100
        return self.max_failed_tasks


class TasksetRunInput(BaseModel):
    """Validate a stored Harbor taskset request for independent Kubernetes task Jobs.

    Construction produces a validated request without resolving references or doing
    service I/O. Submit uses it to derive request identity and acceptance budgets.

    Raises:
        ValidationError: Fields are invalid or the spec violates Kubernetes run rules.
    """

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    spec: AgentEvalInputSpec = Field(
        description="Evaluation inputs: a stored taskset reference and Harbor runner target. "
        "Inline tasks and precomputed trials are not supported for Kubernetes runs."
    )
    execution: HarborTaskJobsExecutionInput = Field(
        description="Kubernetes task Job settings: sandbox policy, dispatcher profile, "
        "run-wide task Job concurrency and failure tolerance."
    )
    idempotency_key: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
        description="Optional caller-supplied key, scoped to workspace and caller, "
        "so retries of the same request return the accepted run. Omit for a new keyless run.",
    )
    timeout_seconds: int | None = Field(
        default=None,
        gt=0,
        description="Execution budget in seconds from acceptance. Omit to use evals.runs.default_timeout_seconds. "
        "Must not exceed evals.runs.max_timeout_seconds.",
    )

    @model_validator(mode="after")
    def _require_stored_taskset(self) -> Self:
        """Check this request's validated task input for a stored taskset reference.

        Returns:
            This request unchanged when its tasks identify a stored taskset.

        Raises:
            ValueError: The spec supplies inline tasks or individual task references.
        """
        if not isinstance(self.spec.tasks, TasksetRef):
            raise ValueError(
                "Kubernetes runs take a stored taskset reference in spec.tasks; publish the tasks as a taskset first"
            )
        return self

    @model_validator(mode="after")
    def _require_harbor_target(self) -> Self:
        """Check this request's validated trial source for a Harbor runner.

        Returns:
            This request unchanged when its target is Harbor.

        Raises:
            ValueError: The spec supplies another runner or precomputed trials.
        """
        if not isinstance(self.spec.target, HarborRunnerTarget):
            raise ValueError('execution requires a Harbor runner target (spec.target.kind = "harbor")')
        return self

    @model_validator(mode="after")
    def _require_serial_trials(self) -> Self:
        """Check that this request leaves Harbor trial concurrency to the task Job.

        Returns:
            This request unchanged when n_concurrent_trials is None.

        Raises:
            ValueError: The Harbor target sets an explicit trial concurrency value.
        """
        if isinstance(self.spec.target, HarborRunnerTarget) and self.spec.target.n_concurrent_trials is not None:
            raise ValueError(
                "Kubernetes runs trials one at a time in v1; cap the run with execution.max_concurrent_task_jobs"
            )
        return self


def request_digest(req: TasksetRunInput) -> str:
    """Hash the validated request before resolving any references.

    Dictionary key order and omitted defaults do not affect identity. List order
    and numeric representation are retained; reference strings are hashed as supplied.

    Args:
        req: Validated caller request, with model defaults filled.

    Returns:
        SHA-256 of compact JSON with sorted keys and only idempotency_key excluded.
    """
    body = req.model_dump(mode="json", exclude={"idempotency_key"})
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


class TasksetRunStatus(BaseModel):
    """Represent public run state for submit, read and cancellation responses.

    Instances expose stored domain state and caller identity while excluding plan
    bindings, request digests and Entity Store persistence metadata.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(description="Run name used to address the accepted run in API routes.")
    workspace: str = Field(description="Workspace containing the run.")
    taskset_ref: str = Field(
        description="Pinned taskset revision this run evaluates, for example default/sort-suite#<64-hex digest>."
    )
    phase: TasksetRunPhase = Field(description="Execution progress. Starts accepted; terminal phases never change.")
    result_state: ResultState = Field(
        description="Availability of the combined results, tracked independently of phase."
    )
    accepted_at: datetime = Field(description="When submit accepted the run.")
    execution_deadline_at: datetime = Field(
        description="accepted_at plus the effective timeout; unfinished tasks expire then. Retries do not reset it."
    )
    finalization_deadline_at: datetime = Field(
        description="execution_deadline_at plus operator grace; the final result must be saved by then."
    )
    cancel_requested_at: datetime | None = Field(
        default=None, description="When a caller requested cancellation; None if never requested."
    )
    error: str | None = Field(default=None, description="Why the run failed or expired; None otherwise.")
    created_by: str | None = Field(
        default=None,
        description="Principal ID of the caller who created the run; None when creator metadata is unavailable.",
    )
    idempotency_key: str | None = Field(default=None, description="Caller-supplied submit key; None for a keyless run.")


_INTERNAL_FIELDS = {"plan_ref", "plan_digest", "request_digest"}


class TasksetRunSubmitResponse(TasksetRunStatus):
    """Return the accepted run with advisory warnings for this submission only."""

    warnings: list[str] = Field(
        default_factory=list,
        description="Advisory notices about this caller's overlapping active runs on the same taskset revision.",
    )


def to_status(run: TasksetRunEntity) -> TasksetRunStatus:
    """Project stored domain fields into the public run status.

    Args:
        run: Stored run, including Entity Store creator metadata when available.

    Returns:
        Public status with name mapped to run_id and persistence details excluded.

    Raises:
        ValidationError: A domain field has no corresponding public status field or its value is invalid.
    """
    data = run.model_dump(exclude=run.__base_fields__ | _INTERNAL_FIELDS, exclude_computed_fields=True)
    return TasksetRunStatus.model_validate(
        {**data, "run_id": run.name, "workspace": run.workspace, "created_by": run.created_by}
    )
