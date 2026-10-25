# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mutable taskset run records, independent of their frozen execution plans."""

from datetime import datetime
from enum import StrEnum
from typing import ClassVar

from nemo_helix_plugin.entities import EntityBase
from nemo_helix_plugin.jobs.schemas import HelixJobStatus
from pydantic import Field


class TasksetRunPhase(StrEnum):
    """Encode execution progress as stable string values for persisted run records.

    Controllers move from accepted to running or a terminal phase; result
    finalization is tracked separately by ``ResultState`` so execution completion
    does not imply that combined results are already available.
    """

    ACCEPTED = "accepted"  # Created by submit with its plan; no task Job started yet. There is no earlier phase.
    RUNNING = "running"  # The run's first task got a slot; its execution clock started (started_at).
    COMPLETED = "completed"  # Every task settled and at least one succeeded, within the failure tolerance.
    FAILED = (
        "failed"  # The run could not proceed, e.g. too many failed tasks, no task succeeded, or a plan digest mismatch.
    )
    CANCELLED = "cancelled"  # A caller cancelled the run; remaining task Jobs are cancelled.
    EXPIRED = "expired"  # execution_deadline_at passed before every task settled.


TERMINAL_RUN_PHASES = frozenset(
    {
        TasksetRunPhase.COMPLETED,
        TasksetRunPhase.FAILED,
        TasksetRunPhase.CANCELLED,
        TasksetRunPhase.EXPIRED,
    }
)


class ResultState(StrEnum):
    """Encode result availability independently of the run's execution phase.

    Persisted values distinguish pending finalization, ready results, failed
    finalization, and no result, allowing readers to interpret terminal runs.
    """

    PENDING = "pending"  # Combined results are not finalized yet.
    READY = "ready"  # Combined results are saved and readable.
    FAILED = "failed"  # The finalization deadline passed before the result was saved.
    NONE = "none"  # Nothing to combine, e.g. no task succeeded.


class TasksetRunEntity(EntityBase):
    """Persist mutable run progress separately from its accepted execution plan.

    Construction takes the inherited name/workspace, a plan ref and byte digest,
    a pinned taskset ref, the logical request digest, and precomputed acceptance
    and deadline timestamps. An optional idempotency key identifies keyed submits.
    Named creation makes these bindings authoritative for later retries.

    The record starts accepted with a pending result. Controllers can update
    execution phase, result state, cancellation time, and error independently,
    while readers retrieve the original plan through its integrity-checked ref.
    Entity Store supplies inherited creator, parent, and version metadata.
    """

    __entity_type__: ClassVar[str] = "taskset_run"

    plan_ref: str = Field(
        description=(
            "Files ref (workspace/fileset#plan.json) of the frozen execution plan. "
            "Set by the run's create and never updated."
        )
    )
    plan_digest: str = Field(
        description=(
            "sha256 of the stored plan bytes, set with plan_ref. "
            "Every plan load re-hashes the bytes and rejects a mismatch."
        )
    )
    idempotency_key: str | None = Field(
        default=None,
        description=(
            "Key that, with workspace and caller, derives the run name, so retries of the same submit "
            "return this run: the caller's, or one submit generated when the request omitted it."
        ),
    )
    request_digest: str = Field(
        description=(
            "Digest of the logical submit request. A retry with the same idempotency key "
            "but a different digest is rejected as a conflict."
        )
    )
    taskset_ref: str = Field(
        description="Pinned taskset revision this run evaluates (for example default/sort-suite#3); filterable."
    )
    # A run record only comes into existence at the moment it's accepted, so it always starts as accepted / pending
    phase: TasksetRunPhase = Field(
        default=TasksetRunPhase.ACCEPTED,
        description="Execution progress. Starts accepted; terminal phases never change.",
    )
    result_state: ResultState = Field(
        default=ResultState.PENDING,
        description="Availability of the combined results, tracked independently of phase.",
    )
    accepted_at: datetime = Field(description="When submit accepted the run.")
    timeout_seconds: int = Field(gt=0, description="Execution budget in seconds, counted from started_at.")
    started_at: datetime | None = Field(
        default=None,
        description="When the run's first task got a slot and its execution clock started; None until then.",
    )
    execution_deadline_at: datetime = Field(
        description=(
            "Unfinished tasks expire then. Until the run starts, accepted_at plus the queue timeout; from its "
            "first slot, started_at plus timeout_seconds. Fixed before the run's first task Job; retries do not "
            "reset it."
        )
    )
    finalization_deadline_at: datetime = Field(
        description=(
            "execution_deadline_at plus the operator grace at submit; the final result must be saved by then. "
            "Starting the run keeps that grace."
        )
    )
    cancel_requested_at: datetime | None = Field(
        default=None,
        description="When a caller requested cancellation; None if never requested.",
    )
    error: str | None = Field(default=None, description="Why the run failed or expired; None otherwise.")


class MemberPhase(StrEnum):
    """Encode preparation and execution progress for an individual run member.

    String values persist image preparation, Job registration, and settled task
    outcomes so controllers can track each task without changing the run's plan.
    """

    IMAGE_PREPARING = "image_preparing"  # The task's environment image is being located or built.
    IMAGE_READY = "image_ready"  # The image exists; the task Job is not registered yet.
    REGISTERED = "registered"  # The task Job is registered with the Jobs Service (see task_job).
    SUCCEEDED = "succeeded"  # The task Job finished and its outputs were collected.
    FAILED = "failed"  # The task could not run or its Job failed; see error.
    CANCELLED = "cancelled"  # The task was cancelled with its run.
    EXPIRED = "expired"  # The run's execution deadline passed before this task settled.


SETTLED_MEMBER_PHASES = frozenset(
    {
        MemberPhase.SUCCEEDED,
        MemberPhase.FAILED,
        MemberPhase.CANCELLED,
        MemberPhase.EXPIRED,
    }
)


class TasksetRunMemberEntity(EntityBase):
    """Persist one task's progress beneath its run's inherited parent entity ID.

    Inputs are the inherited name/workspace, a nonnegative resolved task index,
    and its current phase; task Job ref and error are optional until known.
    The controller uses the resulting ``task-N`` record to track preparation and
    execution independently of other members (tasks) and the frozen run plan.
    """

    __entity_type__: ClassVar[str] = "taskset_run_member"

    task_index: int = Field(ge=0, description="Zero-based index of the task in the run's resolved task list.")
    phase: MemberPhase = Field(description="Image preparation and execution progress of this task.")
    task_job: str | None = Field(
        default=None,
        description=(
            "Name of this task's Jobs Service Job (<run_id>-task-<index>). Written before the create (the create "
            "intent: the task holds a slot from then on) and never changed after; None while queued."
        ),
    )
    job_id: str | None = Field(
        default=None,
        description=(
            "Id of the Job registered under task_job (created, or adopted as ours); written with registered and "
            "never changed. A Job under task_job with another id isn't ours: ours ended and was deleted."
        ),
    )
    error: str | None = Field(default=None, description="Why this task failed or expired; None otherwise.")
    bundle_ref: str | None = Field(
        default=None, description="The task's agent-eval-results bundle; set when the member succeeds."
    )
    job_status: HelixJobStatus | None = Field(
        default=None,
        description="The task Job's status as the controller last read it; display only, at most one cycle old.",
    )
    create_sent_at: datetime | None = Field(
        default=None,
        description=(
            "When the last create of task_job with an unknown outcome was sent. Until create_grace after it, "
            "a 404 or stepless read of the Job doesn't prove that no create of ours is still saving."
        ),
    )


def is_queued(member: TasksetRunMemberEntity) -> bool:
    """Waiting for a slot: images in the plan, no create intent yet."""
    return member.phase is MemberPhase.IMAGE_READY and member.task_job is None


def holds_slot(member: TasksetRunMemberEntity) -> bool:
    """Counted against the caps: registered, or a create intent (``task_job`` written before the create)."""
    return member.phase is MemberPhase.REGISTERED or (
        member.phase is MemberPhase.IMAGE_READY and member.task_job is not None
    )


_ALLOWED_RUN_TRANSITIONS = {
    TasksetRunPhase.ACCEPTED: frozenset(
        {
            TasksetRunPhase.RUNNING,
            TasksetRunPhase.FAILED,
            TasksetRunPhase.CANCELLED,
            TasksetRunPhase.EXPIRED,
        }
    ),
    TasksetRunPhase.RUNNING: TERMINAL_RUN_PHASES,
}


def assert_run_transition(old: TasksetRunPhase, new: TasksetRunPhase) -> None:
    """Guard controller writes against reversing or reopening the run lifecycle.

    Args:
        old: Current persisted phase.
        new: Phase to persist; callers skip this guard when unchanged.

    Returns:
        None for a legal transition; the guard does not mutate a run record.

    Raises:
        ValueError: The transition is illegal, unchanged, or starts at a terminal phase.
    """
    if new not in _ALLOWED_RUN_TRANSITIONS.get(old, frozenset()):
        raise ValueError(f"illegal taskset run transition: {old} -> {new}")


# Members are created in image_ready (their images are in the plan), so image_preparing is unused. Settled phases
# have no way out. Same-phase writes are allowed: image_ready → image_ready (the create intent, create_sent_at) and
# registered → registered (job_status).
_ALLOWED_MEMBER_TRANSITIONS = {
    MemberPhase.IMAGE_READY: frozenset(
        {
            MemberPhase.IMAGE_READY,
            MemberPhase.REGISTERED,
            MemberPhase.FAILED,
            MemberPhase.CANCELLED,
            MemberPhase.EXPIRED,
        }
    ),
    MemberPhase.REGISTERED: frozenset(
        {MemberPhase.REGISTERED, MemberPhase.SUCCEEDED, MemberPhase.FAILED, MemberPhase.CANCELLED, MemberPhase.EXPIRED}
    ),
}


def assert_member_transition(old: MemberPhase, new: MemberPhase) -> None:
    """Guard member writes: a settled member never changes, and a member never goes back to the queue.

    Raises:
        ValueError: The transition is illegal or starts at a settled phase.
    """
    if new not in _ALLOWED_MEMBER_TRANSITIONS.get(old, frozenset()):
        raise ValueError(f"illegal taskset run member transition: {old} -> {new}")
