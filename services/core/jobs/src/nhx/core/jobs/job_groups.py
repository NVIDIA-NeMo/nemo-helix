# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Template submissions and durable JobGroup reconciliation state."""

import hashlib
import logging
from collections.abc import Iterator
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, ClassVar, Literal, Self

from nemo_helix_plugin.entities.types import Entity
from nemo_helix_plugin.jobs.types import JobName
from nhx.common.auth import AuthContext
from nhx.common.entities.client import EntityBase
from nhx.common.jobs.schemas import HelixJobStatus
from nhx.core.jobs.app.schemas import HelixJobSpec
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError, model_validator


def _preserve_nullable_properties(schema: dict[str, Any]) -> None:
    """Keep null in the exported OpenAPI types for JobGroup API models."""
    null_schema = {"type": "null"}
    for prop in schema["properties"].values():
        variants = prop.get("anyOf", [])
        if len(variants) != 2 or null_schema not in variants:
            continue
        value = next(variant for variant in variants if variant != null_schema)
        if "type" not in value:
            continue
        del prop["anyOf"]
        prop.update(value)
        prop["type"] = [value["type"], "null"]


class JobGroupMember(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra=_preserve_nullable_properties)

    config_patch: dict[str, Any] = Field(
        default_factory=dict,
        description="Overrides merged recursively into each template step's config.",
    )
    timeout_seconds: int | None = Field(default=None, ge=0, description="Overrides the group step timeout.")


class JobGroupSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    template: HelixJobSpec
    spec: dict[str, Any] = Field(default_factory=dict, description="Plugin payload copied into every member Job.")
    members: list[JobGroupMember] = Field(min_length=1)
    max_concurrent_jobs: int | None = Field(default=None, ge=1)
    fail_fast: bool = False
    default_timeout_seconds: int | None = Field(default=None, ge=0)


class CreateJobGroupRequest(JobGroupSubmission):
    model_config = ConfigDict(regex_engine="python-re", json_schema_extra=_preserve_nullable_properties)

    name: JobName | None = None
    description: str | None = None
    project: str | None = None
    custom_fields: dict[str, Any] | None = None
    ownership: dict[str, Any] | None = None


class UpdateJobGroupRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra=_preserve_nullable_properties)

    description: str | None = None
    custom_fields: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_patch(self) -> Self:
        if not self.model_fields_set:
            raise ValueError("at least one mutable JobGroup field is required")
        return self


class JobGroupStatus(StrEnum):
    PENDING = "pending"
    ACTIVE = "active"
    CANCELLING = "cancelling"
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    DELETION_PENDING = "deletion_pending"

    @property
    def terminal(self) -> bool:
        return self in {self.COMPLETE, self.INCOMPLETE}


class JobGroupMemberOutcome(StrEnum):
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


class _MemberPhase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class _Pending(_MemberPhase):
    phase: Literal["pending"] = "pending"


class _Preparing(_MemberPhase):
    phase: Literal["preparing"] = "preparing"
    attempt_name: str
    reason: str | None = None


class _Tracking(_MemberPhase):
    phase: Literal["tracking"] = "tracking"
    attempt_id: str


class _Admitted(_MemberPhase):
    phase: Literal["admitted"] = "admitted"
    attempt_id: str


class _Completed(_MemberPhase):
    phase: Literal["completed"] = "completed"
    attempt_id: str


class _Unsuccessful(_MemberPhase):
    phase: Literal["unsuccessful"] = "unsuccessful"
    outcome: Literal[JobGroupMemberOutcome.ERROR, JobGroupMemberOutcome.CANCELLED]
    attempt_id: str | None = None
    reason: str | None = None


class JobGroupMemberState(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    job_id: str | None = None
    progress: Annotated[
        _Pending | _Preparing | _Tracking | _Admitted | _Completed | _Unsuccessful, Field(discriminator="phase")
    ] = Field(default_factory=_Pending)

    @model_validator(mode="after")
    def validate_job_identity(self) -> Self:
        if self.attempt_id is not None and self.job_id is None:
            raise ValueError("A recorded Attempt needs a Job ID")
        return self

    @property
    def phase(self) -> Literal["pending", "preparing", "tracking", "admitted", "completed", "unsuccessful"]:
        return self.progress.phase

    @property
    def attempt_name(self) -> str | None:
        if isinstance(self.progress, _Preparing):
            return self.progress.attempt_name
        return None

    @property
    def attempt_id(self) -> str | None:
        if isinstance(self.progress, (_Tracking, _Admitted, _Completed, _Unsuccessful)):
            return self.progress.attempt_id
        return None

    @property
    def outcome(self) -> JobGroupMemberOutcome | None:
        if isinstance(self.progress, _Completed):
            return JobGroupMemberOutcome.COMPLETED
        if isinstance(self.progress, _Unsuccessful):
            return self.progress.outcome
        return None

    @property
    def reason(self) -> str | None:
        if isinstance(self.progress, (_Preparing, _Unsuccessful)):
            return self.progress.reason
        return None


class JobGroupTransitionError(ValueError):
    pass


def log_unreadable_job_group(entity: Entity, error: ValidationError) -> None:
    state = entity.data.get("controller_state")
    version = state.get("schema_version") if isinstance(state, dict) else None
    logging.getLogger(__name__).error(
        "Skipping unreadable JobGroup %s (schema version %s): %s validation error(s)",
        entity.id,
        version,
        error.error_count(),
    )


def group_member_job_name(group_id: str, index: int) -> str:
    identity = hashlib.sha256(f"{group_id}/{index}".encode()).hexdigest()[:24]
    return f"group-{identity}"


class JobGroupControllerState(BaseModel):
    schema_version: Literal[7] = 7
    status: JobGroupStatus = JobGroupStatus.PENDING
    retry_revision: int = Field(default=0, ge=0)
    members: dict[int, JobGroupMemberState] = Field(default_factory=dict)

    def member(self, index: int) -> JobGroupMemberState:
        return self.members.get(index) or JobGroupMemberState()

    def _find_member_by_job_id(self, job_id: str) -> tuple[int, JobGroupMemberState] | None:
        return next(
            ((key, member) for key, member in self.members.items() if member.job_id == job_id),
            None,
        )

    def members_in_flight(self, member_count: int) -> list[int]:
        return [
            index
            for index in range(member_count)
            if (saved := self.members.get(index)) is not None
            and isinstance(saved.progress, (_Preparing, _Tracking, _Admitted))
        ]

    def pending_members(self, member_count: int) -> Iterator[int]:
        for index in range(member_count):
            saved = self.members.get(index)
            if saved is None or isinstance(saved.progress, _Pending):
                yield index

    def available_claims(self, limit: int | None, batch_size: int) -> int:
        occupied = self.occupied_slots()
        preparing = sum(isinstance(member.progress, (_Preparing, _Tracking)) for member in self.members.values())
        demand = batch_size if limit is None else min(batch_size, limit - occupied)
        return max(0, demand - preparing)

    def occupied_slots(self) -> int:
        return sum(isinstance(member.progress, _Admitted) for member in self.members.values())

    def claim_member(self, index: int, attempt_name: str) -> None:
        member = self.member(index)
        if self.status not in {JobGroupStatus.PENDING, JobGroupStatus.ACTIVE} or not isinstance(
            member.progress, _Pending
        ):
            raise JobGroupTransitionError("Only pending members of an active submission can be claimed")
        progress = _Preparing(attempt_name=attempt_name)
        self.members[index] = member.model_copy(update={"progress": progress})
        self.status = JobGroupStatus.ACTIVE

    def record_job(self, index: int, job_id: str) -> None:
        member = self.member(index)
        if not isinstance(member.progress, _Preparing) or member.job_id not in (None, job_id):
            raise JobGroupTransitionError("Only a preparing member can record its Job ID")
        self.members[index] = member.model_copy(update={"job_id": job_id})

    def record_creation_failure(self, index: int, job_id: str, reason: str) -> None:
        self.record_job(index, job_id)
        member = self.member(index)
        self.members[index] = member.model_copy(
            update={"progress": member.progress.model_copy(update={"reason": member.reason or reason})}
        )

    def record_abandoned_creation(self, job_id: str, *, job_name: str, group_id: str) -> bool:
        entry = self._find_member_by_job_id(job_id)
        if entry is None:
            entry = next(
                (
                    (index, member)
                    for index, member in self.members.items()
                    if isinstance(member.progress, _Preparing)
                    and member.job_id is None
                    and group_member_job_name(group_id, index) == job_name
                ),
                None,
            )
        if entry is None:
            return False
        index, member = entry
        if not isinstance(member.progress, _Preparing) or member.reason is not None:
            return False
        self.record_creation_failure(index, job_id, "Job creation did not finish before the creation deadline.")
        return True

    def finish_missing_job(self, index: int, outcome: JobGroupMemberOutcome, reason: str | None = None) -> None:
        self.finish_member(index, outcome, reason)
        member = self.member(index)
        if member.attempt_id is None:
            self.members[index] = member.model_copy(update={"job_id": None})

    def record_attempt(self, index: int, job_id: str, attempt_id: str) -> None:
        member = self.member(index)
        if isinstance(member.progress, (_Tracking, _Admitted)) and (member.job_id, member.attempt_id) == (
            job_id,
            attempt_id,
        ):
            return
        if not isinstance(member.progress, _Preparing) or member.job_id != job_id:
            raise JobGroupTransitionError("Only the recorded Job can record a new Attempt")
        self.members[index] = member.model_copy(update={"job_id": job_id, "progress": _Tracking(attempt_id=attempt_id)})

    def admit_attempt(self, job_id: str, attempt_id: str, limit: int | None) -> bool:
        entry = self._find_member_by_job_id(job_id)
        if entry is None:
            return False
        member_key, member = entry
        if member.attempt_id != attempt_id:
            return False
        if self.status not in {JobGroupStatus.PENDING, JobGroupStatus.ACTIVE}:
            raise JobGroupTransitionError("The JobGroup is no longer accepting dispatch")
        if not isinstance(member.progress, (_Tracking, _Admitted)):
            raise JobGroupTransitionError("The group member has already settled")
        if isinstance(member.progress, _Admitted):
            return False
        if limit is not None and self.occupied_slots() >= limit:
            raise JobGroupTransitionError("Waiting for JobGroup capacity")
        self.members[member_key] = member.model_copy(update={"progress": _Admitted(attempt_id=attempt_id)})
        return True

    def finish_member(self, index: int, outcome: JobGroupMemberOutcome, reason: str | None = None) -> None:
        member = self.member(index)
        if member.outcome is not None:
            raise JobGroupTransitionError("The member has already finished")
        if outcome == JobGroupMemberOutcome.COMPLETED:
            if member.attempt_id is None:
                raise JobGroupTransitionError("A completed member must have an Attempt")
            progress = _Completed(attempt_id=member.attempt_id)
        else:
            progress = _Unsuccessful(outcome=outcome, attempt_id=member.attempt_id, reason=reason)
        self.members[index] = member.model_copy(update={"progress": progress})

    def settle(self, member_count: int) -> bool:
        if self.status == JobGroupStatus.DELETION_PENDING or self.status.terminal:
            return False
        if len(self.members) != member_count or any(member.outcome is None for member in self.members.values()):
            return False
        self.status = (
            JobGroupStatus.COMPLETE
            if all(member.outcome == JobGroupMemberOutcome.COMPLETED for member in self.members.values())
            else JobGroupStatus.INCOMPLETE
        )
        return True

    def cancel(self) -> bool:
        if self.status == JobGroupStatus.CANCELLING:
            return False
        if self.status not in {JobGroupStatus.PENDING, JobGroupStatus.ACTIVE}:
            raise JobGroupTransitionError("JobGroup is already terminal")
        self.status = JobGroupStatus.CANCELLING
        return True

    def retry_unsuccessful_members(self) -> bool:
        if self.status != JobGroupStatus.INCOMPLETE:
            raise JobGroupTransitionError("Only an incomplete JobGroup can be retried")
        selected = [index for index, member in self.members.items() if isinstance(member.progress, _Unsuccessful)]
        if not selected:
            raise JobGroupTransitionError("JobGroup has no unsuccessful members to retry")
        self.retry_revision += 1
        for index in selected:
            self.members[index] = self.members[index].model_copy(update={"progress": _Pending()})
        self.status = JobGroupStatus.PENDING
        return True

    def request_deletion(self) -> bool:
        if self.status == JobGroupStatus.DELETION_PENDING:
            return False
        if not self.status.terminal:
            raise JobGroupTransitionError("Cancel outstanding work and wait for the JobGroup to settle before deleting")
        self.status = JobGroupStatus.DELETION_PENDING
        return True


class HelixJobGroup(EntityBase):
    __entity_type__: ClassVar[str] = "platform_job_group"

    submission: JobGroupSubmission
    controller_state: JobGroupControllerState
    description: str | None = None
    custom_fields: dict[str, Any] | None = None
    ownership: dict[str, Any] | None = None
    _auth_context: AuthContext | None = PrivateAttr(default=None)

    def member_job_name(self, index: int) -> str:
        return group_member_job_name(self.id, index)

    def member_state(self, index: int) -> JobGroupMemberState:
        return self.controller_state.member(index)

    @property
    def auth_context(self) -> AuthContext | None:
        return self._auth_context

    def with_auth_context(self, auth_context: AuthContext) -> Self:
        self._auth_context = auth_context
        return self


class JobGroupResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra=_preserve_nullable_properties)

    id: str
    name: str
    workspace: str
    project: str | None
    description: str | None
    source: str
    custom_fields: dict[str, Any] | None
    ownership: dict[str, Any] | None
    max_concurrent_jobs: int | None
    fail_fast: bool
    default_timeout_seconds: int | None
    status: JobGroupStatus
    member_count: int
    created_at: datetime | None
    updated_at: datetime | None

    @classmethod
    def from_entity(cls, group: HelixJobGroup) -> Self:
        return cls(
            id=group.id,
            name=group.name,
            workspace=group.workspace,
            project=group.project,
            description=group.description,
            source=group.submission.source,
            custom_fields=group.custom_fields,
            ownership=group.ownership,
            max_concurrent_jobs=group.submission.max_concurrent_jobs,
            fail_fast=group.submission.fail_fast,
            default_timeout_seconds=group.submission.default_timeout_seconds,
            status=group.controller_state.status,
            member_count=len(group.submission.members),
            created_at=group.created_at,
            updated_at=group.updated_at,
        )


class JobGroupMemberResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra=_preserve_nullable_properties)

    index: int = Field(ge=0, description="Zero-based position in the submitted members array.")
    job_id: str | None = None
    attempt_id: str | None = None
    status: HelixJobStatus = HelixJobStatus.PENDING
    reason: str | None = None
