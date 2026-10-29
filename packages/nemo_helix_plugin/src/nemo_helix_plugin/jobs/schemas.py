# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from enum import Enum
from typing import Any

from nemo_helix_plugin.schema import Value
from pydantic import BaseModel, Field, field_validator


class FileStorageType(str, Enum):
    FILESET = "fileset"


class HelixJobResultCreateRequest(BaseModel):
    attempt_id: str = Field(description="Concrete ID of the attempt that produced the result")
    artifact_url: str
    artifact_storage_type: FileStorageType

    @field_validator("attempt_id")
    @classmethod
    def require_concrete_attempt_id(cls, attempt_id: str) -> str:
        if not attempt_id or attempt_id == "current":
            raise ValueError("attempt_id must be a concrete attempt ID")
        return attempt_id


class HelixJobResultResponse(BaseModel):
    name: str
    job: str
    workspace: str
    project: str | None = None
    attempt_id: str | None = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    artifact_url: str
    artifact_storage_type: FileStorageType
    download_url: str | None = None


class HelixJobListResultResponse(Value):
    data: list[HelixJobResultResponse]


class HelixJobStatus(str, Enum):
    """Enumeration of possible job statuses.

    This enum represents the various states a job can be in during its lifecycle,
    from creation to a terminal state.
    """

    CREATED = "created"
    PENDING = "pending"
    ACTIVE = "active"
    CANCELLED = "cancelled"
    CANCELLING = "cancelling"
    ERROR = "error"
    COMPLETED = "completed"
    PAUSED = "paused"
    PAUSING = "pausing"
    RESUMING = "resuming"

    @staticmethod
    def terminals() -> list["HelixJobStatus"]:
        return [HelixJobStatus.COMPLETED, HelixJobStatus.ERROR, HelixJobStatus.CANCELLED]

    @staticmethod
    def non_terminals() -> list["HelixJobStatus"]:
        return [
            HelixJobStatus.CREATED,
            HelixJobStatus.PENDING,
            HelixJobStatus.ACTIVE,
            HelixJobStatus.CANCELLING,
            HelixJobStatus.PAUSING,
            HelixJobStatus.PAUSED,
            HelixJobStatus.RESUMING,
        ]

    def is_terminal(self) -> bool:
        return self in HelixJobStatus.terminals()

    def can_transition_to(self, new_status: "HelixJobStatus") -> bool:
        """Validate if a status transition is valid."""

        if self == new_status:
            return True

        # Define valid transitions
        valid_transitions = {
            HelixJobStatus.CREATED: {
                HelixJobStatus.PENDING,
                HelixJobStatus.ACTIVE,
                HelixJobStatus.CANCELLING,
                HelixJobStatus.CANCELLED,
                HelixJobStatus.PAUSING,
                HelixJobStatus.PAUSED,
                HelixJobStatus.ERROR,
            },
            HelixJobStatus.PENDING: {
                HelixJobStatus.ACTIVE,
                HelixJobStatus.CANCELLING,
                HelixJobStatus.CANCELLED,
                HelixJobStatus.ERROR,
                HelixJobStatus.PAUSING,
                HelixJobStatus.PAUSED,
                HelixJobStatus.COMPLETED,
            },
            HelixJobStatus.ACTIVE: {
                HelixJobStatus.COMPLETED,
                HelixJobStatus.ERROR,
                HelixJobStatus.CANCELLING,
                HelixJobStatus.CANCELLED,
                HelixJobStatus.PAUSING,
                HelixJobStatus.PAUSED,
            },
            HelixJobStatus.PAUSED: {
                HelixJobStatus.RESUMING,
                HelixJobStatus.ACTIVE,
                HelixJobStatus.CANCELLING,
            },
            HelixJobStatus.CANCELLING: {HelixJobStatus.CANCELLED, HelixJobStatus.ERROR},
            HelixJobStatus.PAUSING: {
                HelixJobStatus.PAUSED,
                HelixJobStatus.ERROR,
                HelixJobStatus.CANCELLING,
                HelixJobStatus.COMPLETED,
            },
            HelixJobStatus.RESUMING: {
                HelixJobStatus.PENDING,
                HelixJobStatus.ACTIVE,
                HelixJobStatus.ERROR,
                HelixJobStatus.CANCELLING,
            },
            HelixJobStatus.COMPLETED: set(),
            HelixJobStatus.CANCELLED: set(),
            HelixJobStatus.ERROR: set(),
        }

        return new_status in valid_transitions.get(self, set())


class JobGroupStatus(str, Enum):
    """Derived status of a :class:`JobGroup`, rolled up from its member jobs.

    A group is terminal only once every member is terminal. The group ``status``
    is never stored; it is computed on read from the member jobs' statuses
    (see :func:`aggregate_job_group_status`).
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"

    @staticmethod
    def terminals() -> list["JobGroupStatus"]:
        return [JobGroupStatus.COMPLETED, JobGroupStatus.FAILED, JobGroupStatus.CANCELLED]

    def is_terminal(self) -> bool:
        return self in JobGroupStatus.terminals()


def aggregate_job_group_status(member_statuses: list[HelixJobStatus]) -> JobGroupStatus:
    """Roll up member job statuses into a single :class:`JobGroupStatus`.

    Rules:
    - An empty group (no members) is ``PENDING`` -- nothing has run.
    - The group is terminal only when *every* member is terminal.
      - ``FAILED`` if any member errored.
      - ``CANCELLED`` if any member was cancelled (and none errored).
      - ``COMPLETED`` only when every member completed successfully.
    - Otherwise the group is non-terminal:
      - ``CANCELLING`` if any member is cancelling or already cancelled while
        others are still running (a cancel is sweeping the group).
      - ``RUNNING`` if any member is active/pending/past-created but not terminal.
      - ``PENDING`` when every member is still in ``CREATED``.

    Cancel is driven by a separate ticket; the ``CANCELLING``/``CANCELLED``
    arms exist so the derivation is coherent once cancel lands.
    """
    if not member_statuses:
        return JobGroupStatus.PENDING

    all_terminal = all(status.is_terminal() for status in member_statuses)
    any_error = any(status == HelixJobStatus.ERROR for status in member_statuses)
    any_cancelled = any(status == HelixJobStatus.CANCELLED for status in member_statuses)

    if all_terminal:
        if any_error:
            return JobGroupStatus.FAILED
        if any_cancelled:
            return JobGroupStatus.CANCELLED
        return JobGroupStatus.COMPLETED

    any_cancel_in_flight = any(
        status in (HelixJobStatus.CANCELLING, HelixJobStatus.CANCELLED) for status in member_statuses
    )
    if any_cancel_in_flight:
        return JobGroupStatus.CANCELLING

    if all(status == HelixJobStatus.CREATED for status in member_statuses):
        return JobGroupStatus.PENDING

    return JobGroupStatus.RUNNING


def aggregate_job_group_status_from_counts(total: int, status_counts: dict[HelixJobStatus, int]) -> JobGroupStatus:
    """Roll up a group's status from per-status member COUNTS, not a full status list.

    Equivalent to :func:`aggregate_job_group_status` but driven by cheap count
    queries (``count(parent=group AND status=X)``) so a roll-up never fans a read
    out over every member -- which matters for groups with thousands of members.
    The precedence rules live in one place: this reconstructs a minimal
    representative status multiset from the counts and delegates to
    :func:`aggregate_job_group_status`, so the two can never diverge.

    ``total`` is the member count (from the list pagination envelope).
    ``status_counts`` need only carry the buckets the caller probed; any member
    not covered by the provided buckets is treated as ``CREATED`` (the lowest-
    precedence state), so a caller may omit buckets it has proven empty.
    """
    if total <= 0:
        return JobGroupStatus.PENDING
    representative: list[HelixJobStatus] = []
    for status, count in status_counts.items():
        representative.extend([status] * max(count, 0))
    accounted = len(representative)
    if accounted < total:
        # Members not covered by a probed bucket are CREATED (lowest precedence);
        # padding with CREATED keeps the aggregate correct without reading them.
        representative.extend([HelixJobStatus.CREATED] * (total - accounted))
    return aggregate_job_group_status(representative)


class HelixJobTaskStatusResponse(BaseModel):
    id: str
    name: str
    status: HelixJobStatus
    status_details: dict[str, Any]
    error_details: dict[str, Any] | None
    error_stack: str | None
    created_at: datetime
    updated_at: datetime


class HelixJobStepStatusResponse(BaseModel):
    id: str
    name: str
    status: HelixJobStatus
    status_details: dict[str, Any]
    error_details: dict[str, Any] | None
    tasks: list[HelixJobTaskStatusResponse]
    created_at: datetime
    updated_at: datetime


class HelixJobStatusResponse(BaseModel):
    id: str
    name: str
    status: HelixJobStatus
    status_details: dict[str, Any]
    error_details: dict[str, Any] | None
    steps: list[HelixJobStepStatusResponse]
    created_at: datetime
    updated_at: datetime


class HelixJobLog(BaseModel):
    timestamp: datetime
    job: str
    job_step: str
    job_task: str
    message: str


class HelixJobLogPage(BaseModel):
    data: list[HelixJobLog]
    total: int
    next_page: str | None
    prev_page: str | None
