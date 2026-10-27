# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entity definitions for the Jobs service.

These entities use EntityBase and are stored via EntityClient.
Jobs are identified by ID rather than by a unique name within a namespace.
"""

import hashlib
from collections.abc import Mapping
from typing import Any, ClassVar, Dict, Optional, Self

from nemo_helix_plugin.jobs.types import JobControl
from nhx.common.auth import AuthContext
from nhx.common.entities.client import EntityBase
from nhx.common.jobs.schemas import (
    FileStorageType,
    HelixJobResultResponse,
    HelixJobStatus,
    JobGroupStatus,
)
from nhx.core.jobs.app.schemas import HelixJobSpec, HelixJobStepSpec
from pydantic import BaseModel, Field, PrivateAttr, computed_field, model_validator

STEP_SPEC_NAME_CONFIG_KEY = "_step_spec_name"


def get_step_spec_name(config: Mapping[str, Any] | None, fallback_name: str | None = None) -> str | None:
    """Return the original platform step spec name stored on a step entity."""
    if isinstance(config, Mapping):
        step_spec_name = config.get(STEP_SPEC_NAME_CONFIG_KEY)
        if isinstance(step_spec_name, str):
            return step_spec_name
    return fallback_name


def is_final_platform_step(platform_spec: HelixJobSpec, step_spec_name: str | None) -> bool:
    """Return whether ``step_spec_name`` names the final configured step."""
    if step_spec_name is None or not platform_spec.steps:
        return False
    return platform_spec.steps[-1].name == step_spec_name


class JobGroup(EntityBase):
    """A set of related jobs submitted and managed as one unit.

    A JobGroup is a management/accounting unit, not an execution-ordering
    construct: its member jobs are independent (no ordering, no DAG). Membership
    is the Entity Store ``parent`` relationship -- each member :class:`HelixJob`
    is parented to the group, so members are unique within
    ``(workspace, platform_job, parent=group, name)`` and are found by querying
    jobs whose ``parent`` is this group's ID.

    The group ``status`` is a *denormalized* field persisted on the entity (like
    ``HelixJob.status``), not recomputed on every read -- deriving it per GET
    would fan out a read over every member, which does not scale to groups with
    thousands of members. ``aggregate_job_group_status`` computes the value; it
    is written at create time and kept current by the controller's group roll-up
    as members change.
    """

    __entity_type__: ClassVar[str] = "platform_job_group"

    source: str = Field(..., description="Source service that created this job group")
    description: Optional[str] = Field(default=None, description="Job group description")
    max_concurrent_jobs: Optional[int] = Field(
        default=None,
        ge=1,
        description="Caller-set ceiling on how many member jobs may run at once. Recorded on the group; "
        "admission enforcement is handled separately and does not gate creation.",
    )
    fail_fast: bool = Field(
        default=False,
        description="When true, a single member failure should stop the rest of the group. Recorded on the group; "
        "enforcement is handled separately.",
    )
    default_timeout_seconds: Optional[int] = Field(
        default=None,
        ge=0,
        description="Group-default per-step staleness timeout propagated into each member's step lifecycle "
        "(staleness_timeout_seconds) unless the member overrides it. None leaves the template's value in place.",
    )
    status: JobGroupStatus = Field(
        default=JobGroupStatus.PENDING,
        description="Aggregate group status, denormalized onto the entity (like a HelixJob's status) rather than "
        "recomputed on every read. Set from the members at create time; the controller's group roll-up keeps it "
        "current as members change. aggregate_job_group_status computes the value that is written here.",
    )
    custom_fields: Optional[Dict[str, Any]] = Field(default=None, description="Custom fields")
    ownership: Optional[Dict[str, Any]] = Field(default=None, description="Ownership info")


class HelixJob(EntityBase):
    """Platform job storage entity.

    Jobs are identified by ID, not by a unique name within a namespace.
    The name field is a convenience (often derived from the ID).
    """

    __entity_type__: ClassVar[str] = "platform_job"

    # Core job fields
    source: str = Field(..., description="Source service that created this job")
    spec: Dict[str, Any] = Field(default_factory=dict, description="Job Spec")
    platform_spec: HelixJobSpec = Field(..., description="Platform job specification")
    control: JobControl = Field(
        default_factory=JobControl,
        description="Mutable job behavior. This is not part of the execution spec.",
    )
    current_attempt_id: Optional[str] = Field(default=None, description="Current Attempt ID")
    status: Optional[HelixJobStatus] = Field(default=None, description="Current attempt status")
    fileset: str = Field(..., description="Fileset ID for storing job artifacts (logs, results)")
    output_location: Optional[str] = Field(
        default=None,
        description="Caller-supplied artifact fileset; None when the dispatcher auto-created the fileset",
    )
    project: Optional[str] = Field(default=None, description="Project URN")

    # Optional metadata (explicitly added since EntityBase doesn't include these)
    description: Optional[str] = Field(default=None, description="Job description")
    custom_fields: Optional[Dict[str, Any]] = Field(default=None, description="Custom fields")
    ownership: Optional[Dict[str, Any]] = Field(default=None, description="Ownership info")

    # Auth context from the request that created this job. Used to propagate auth for task execution.
    _auth_context: Optional[AuthContext] = PrivateAttr(default=None)

    @computed_field
    @property
    def auth_context(self) -> Optional[AuthContext]:
        return self._auth_context

    def with_auth_context(self, auth_context: AuthContext | None) -> Self:
        """
        Updates the job (in-place) with auth context for task execution.
        """
        self._auth_context = auth_context
        return self


class HelixJobAttempt(EntityBase):
    """A single execution attempt of a job.

    Captures the job specification at the time of the attempt (immutable copy).
    Parent-scoped: unique within (workspace, entity_type, parent=job).
    """

    __entity_type__: ClassVar[str] = "platform_job_attempt"

    job: str = Field(..., description="Parent job ID")
    seq: int = Field(default=0, description="Attempt sequence number")
    spec: Dict[str, Any] = Field(default_factory=dict, description="Job Spec (immutable copy)")
    platform_spec: HelixJobSpec = Field(..., description="Platform job specification (immutable copy)")
    status: HelixJobStatus = Field(default=HelixJobStatus.CREATED, description="Attempt status")
    status_details: Dict[str, Any] = Field(default_factory=dict, description="Details about the job status")
    error_details: Optional[Dict[str, Any]] = Field(default=None, description="Error details if applicable")

    @model_validator(mode="after")
    def set_parent_from_job(self):
        """Set _parent to the job ID for parent-scoped uniqueness."""
        self._parent = self.job
        return self

    def get_first_step(self) -> HelixJobStepSpec:
        """Get the first step in the job specification."""
        return self.platform_spec.steps[0]

    def get_step_spec(self, step_name: str) -> HelixJobStepSpec:
        """Get a specific step's specification by name."""
        for step in self.platform_spec.steps:
            if step.name == step_name:
                return step
        raise ValueError(f"Step '{step_name}' not found in job attempt '{self.id}'")

    def get_next_step_spec(self, current_step_name: str) -> Optional[HelixJobStepSpec]:
        """Get the next step's specification in the job after the current step."""
        found_current = False
        for step in self.platform_spec.steps:
            if found_current:
                return step
            if step.name == current_step_name:
                found_current = True
        return None

    def is_final_step_spec(self, step_name: str | None) -> bool:
        """Return whether ``step_name`` is the final configured step in this attempt."""
        return is_final_platform_step(self.platform_spec, step_name)


class HelixJobStep(EntityBase):
    """A single step within an attempt.

    Parent-scoped: unique within (workspace, entity_type, parent=attempt_id).
    """

    __entity_type__: ClassVar[str] = "platform_job_step"

    attempt_id: str = Field(..., description="Parent attempt ID")
    config: Dict[str, Any] = Field(default_factory=dict, description="Configuration for the step")
    status: HelixJobStatus = Field(default=HelixJobStatus.CREATED, description="Step status")
    status_details: Dict[str, Any] = Field(default_factory=dict, description="Status details")
    error_details: Optional[Dict[str, Any]] = Field(default=None, description="Error details if applicable")

    @model_validator(mode="after")
    def set_parent_from_attempt(self):
        """Set _parent to the attempt_id for parent-scoped uniqueness."""
        self._parent = self.attempt_id
        return self


class HelixJobTask(EntityBase):
    """A task within a step (for parallel execution).

    Parent-scoped: unique within (workspace, entity_type, parent=step_id).
    """

    __entity_type__: ClassVar[str] = "platform_job_task"

    step_id: str = Field(..., description="Parent step ID")
    status: HelixJobStatus = Field(default=HelixJobStatus.PENDING, description="Task status")
    status_details: Dict[str, Any] = Field(default_factory=dict, description="Details about the task status")
    error_details: Optional[Dict[str, Any]] = Field(default=None, description="Details about task errors")
    error_stack: Optional[str] = Field(default=None, description="Error stack trace if applicable")

    @model_validator(mode="after")
    def set_parent_from_step(self):
        """Set _parent to the step_id for parent-scoped uniqueness."""
        self._parent = self.step_id
        return self


class HelixJobResult(EntityBase):
    """Result/artifact from a completed job.

    Results remain parented to the job so names are unique across all attempts.
    ``attempt_id`` records the attempt that produced newer results; existing
    records without it remain valid.
    """

    __entity_type__: ClassVar[str] = "platform_job_result"

    job: str = Field(..., description="Parent job ID")
    attempt_id: Optional[str] = Field(default=None, description="ID of the attempt that produced the result")
    artifact_url: str = Field(..., description="URL to the artifact")
    artifact_storage_type: FileStorageType = Field(..., description="Type of artifact storage")

    @model_validator(mode="after")
    def set_result_parent(self):
        """Set the job parent for job-wide result-name uniqueness."""
        self._parent = self.job
        return self

    def to_response(self) -> HelixJobResultResponse:
        """Convert to API response format."""
        return HelixJobResultResponse(
            name=self.name,
            job=self.job,
            workspace=self.workspace,
            attempt_id=self.attempt_id,
            created_at=self.created_at,  # type: ignore
            updated_at=self.updated_at,  # type: ignore
            artifact_url=self.artifact_url,
            artifact_storage_type=self.artifact_storage_type,
        )


class JobStorageTarget(BaseModel):
    """A backend that holds a job's persistent storage."""

    provider: str = Field(..., description="Execution provider of a step that mounts job storage")
    profile: str = Field(..., description="Execution profile of a step that mounts job storage")


class HelixJobStorageReclaim(EntityBase):
    """Job storage waiting for the jobs controller to reclaim it, from a deleted job or one stopped too long.

    Job storage is keyed by the job name, and deleting the job removes the entities the
    storage sweep reads. This record outlives the job, so the controller can still reach
    the backends that hold the storage, and it holds the name until the storage is gone.
    It lives in the system workspace, so deleting the job's workspace does not drop it.
    """

    __entity_type__: ClassVar[str] = "platform_job_storage_reclaim"

    job_id: str = Field(..., description="ID of the job")
    job_workspace: str = Field(..., description="Workspace the job belongs to")
    job_name: str = Field(..., description="Name of the job. Job storage is keyed by it.")
    targets: list[JobStorageTarget] = Field(default_factory=list, description="Backends that hold the storage")

    @staticmethod
    def name_for_job(job_id: str) -> str:
        """Return a stable record name for a job. Job IDs have upper case, which entity names do not allow."""
        return f"reclaim-{hashlib.sha256(job_id.encode()).hexdigest()[:40]}"
