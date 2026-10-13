# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mutable taskset run records, independent of their frozen execution plans."""

from datetime import datetime
from enum import StrEnum
from typing import ClassVar

from nemo_helix_plugin.entities import EntityBase
from pydantic import Field


class TasksetRunPhase(StrEnum):
    """Run execution lifecycle; finalization is tracked separately by ResultState."""

    ACCEPTED = "accepted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


TERMINAL_RUN_PHASES = frozenset(
    {
        TasksetRunPhase.COMPLETED,
        TasksetRunPhase.FAILED,
        TasksetRunPhase.CANCELLED,
        TasksetRunPhase.EXPIRED,
    }
)


class ResultState(StrEnum):
    """Availability of the finalized run result."""

    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"
    NONE = "none"


class TasksetRunEntity(EntityBase):
    """Run accepted by a named create that permanently binds its plan and pinned taskset."""

    __entity_type__: ClassVar[str] = "taskset_run"

    plan_ref: str
    plan_digest: str
    idempotency_key: str | None = None
    request_digest: str
    taskset_ref: str
    phase: TasksetRunPhase = TasksetRunPhase.ACCEPTED
    result_state: ResultState = ResultState.PENDING
    accepted_at: datetime
    execution_deadline_at: datetime
    finalization_deadline_at: datetime
    cancel_requested_at: datetime | None = None
    error: str | None = None


class MemberPhase(StrEnum):
    """Task preparation and execution lifecycle within one run."""

    IMAGE_PREPARING = "image_preparing"
    IMAGE_READY = "image_ready"
    REGISTERED = "registered"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


SETTLED_MEMBER_PHASES = frozenset(
    {
        MemberPhase.SUCCEEDED,
        MemberPhase.FAILED,
        MemberPhase.CANCELLED,
        MemberPhase.EXPIRED,
    }
)


class TasksetRunMemberEntity(EntityBase):
    """Task record named task-N beneath its run's inherited parent entity ID."""

    __entity_type__: ClassVar[str] = "taskset_run_member"

    task_index: int = Field(ge=0)
    phase: MemberPhase
    task_job: str | None = None
    error: str | None = None


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
    """Raise ValueError for an illegal phase change, including unchanged or terminal phases.

    Args:
        old: Current persisted phase.
        new: Phase to persist; callers skip this guard when unchanged.
    """
    if new not in _ALLOWED_RUN_TRANSITIONS.get(old, frozenset()):
        raise ValueError(f"illegal taskset run transition: {old} -> {new}")
