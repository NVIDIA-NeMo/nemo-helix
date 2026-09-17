# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entity definitions for the Jobs service.

These entities use EntityBase and are stored via EntityClient.
Jobs are identified by ID rather than by a unique name within a namespace.
"""

from collections.abc import Mapping
from typing import Any, ClassVar, Dict, Optional, Self

from nmp.common.auth import AuthContext
from nmp.common.entities.client import EntityBase
from nmp.common.jobs.schemas import FileStorageType, PlatformJobResultResponse, PlatformJobStatus
from nmp.core.jobs.app.schemas import PlatformJobSpec, PlatformJobStepSpec
from pydantic import Field, PrivateAttr, computed_field, model_validator

STEP_SPEC_NAME_CONFIG_KEY = "_step_spec_name"


def get_step_spec_name(config: Mapping[str, Any] | None, fallback_name: str | None = None) -> str | None:
    """Return the original platform step spec name stored on a step entity."""
    if isinstance(config, Mapping):
        step_spec_name = config.get(STEP_SPEC_NAME_CONFIG_KEY)
        if isinstance(step_spec_name, str):
            return step_spec_name
    return fallback_name


def is_final_platform_step(platform_spec: PlatformJobSpec, step_spec_name: str | None) -> bool:
    """Return whether ``step_spec_name`` names the final configured step."""
    if step_spec_name is None or not platform_spec.steps:
        return False
    return platform_spec.steps[-1].name == step_spec_name


class PlatformJob(EntityBase):
    """Platform job storage entity.

    Jobs are identified by ID, not by a unique name within a namespace.
    The name field is a convenience (often derived from the ID).
    """

    __entity_type__: ClassVar[str] = "platform_job"

    # Core job fields
    source: str = Field(..., description="Source service that created this job")
    spec: Dict[str, Any] = Field(default_factory=dict, description="Job Spec")
    platform_spec: PlatformJobSpec = Field(..., description="Platform job specification")
    current_attempt_id: Optional[str] = Field(default=None, description="Current Attempt ID")
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


class PlatformJobAttempt(EntityBase):
    """A single execution attempt of a job.

    Captures the job specification at the time of the attempt (immutable copy).
    Parent-scoped: unique within (workspace, entity_type, parent=job).
    """

    __entity_type__: ClassVar[str] = "platform_job_attempt"

    job: str = Field(..., description="Parent job ID")
    seq: int = Field(default=0, description="Attempt sequence number")
    spec: Dict[str, Any] = Field(default_factory=dict, description="Job Spec (immutable copy)")
    platform_spec: PlatformJobSpec = Field(..., description="Platform job specification (immutable copy)")
    status: PlatformJobStatus = Field(default=PlatformJobStatus.CREATED, description="Attempt status")
    status_details: Dict[str, Any] = Field(default_factory=dict, description="Details about the job status")
    error_details: Optional[Dict[str, Any]] = Field(default=None, description="Error details if applicable")

    @model_validator(mode="after")
    def set_parent_from_job(self):
        """Set _parent to the job ID for parent-scoped uniqueness."""
        self._parent = self.job
        return self

    def get_first_step(self) -> PlatformJobStepSpec:
        """Get the first step in the job specification."""
        return self.platform_spec.steps[0]

    def get_step_spec(self, step_name: str) -> PlatformJobStepSpec:
        """Get a specific step's specification by name."""
        for step in self.platform_spec.steps:
            if step.name == step_name:
                return step
        raise ValueError(f"Step '{step_name}' not found in job attempt '{self.id}'")

    def get_next_step_spec(self, current_step_name: str) -> Optional[PlatformJobStepSpec]:
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


class PlatformJobStep(EntityBase):
    """A single step within an attempt.

    Parent-scoped: unique within (workspace, entity_type, parent=attempt_id).
    """

    __entity_type__: ClassVar[str] = "platform_job_step"

    attempt_id: str = Field(..., description="Parent attempt ID")
    config: Dict[str, Any] = Field(default_factory=dict, description="Configuration for the step")
    status: PlatformJobStatus = Field(default=PlatformJobStatus.CREATED, description="Step status")
    status_details: Dict[str, Any] = Field(default_factory=dict, description="Status details")
    error_details: Optional[Dict[str, Any]] = Field(default=None, description="Error details if applicable")

    @model_validator(mode="after")
    def set_parent_from_attempt(self):
        """Set _parent to the attempt_id for parent-scoped uniqueness."""
        self._parent = self.attempt_id
        return self


class PlatformJobTask(EntityBase):
    """A task within a step (for parallel execution).

    Parent-scoped: unique within (workspace, entity_type, parent=step_id).
    """

    __entity_type__: ClassVar[str] = "platform_job_task"

    step_id: str = Field(..., description="Parent step ID")
    status: PlatformJobStatus = Field(default=PlatformJobStatus.PENDING, description="Task status")
    status_details: Dict[str, Any] = Field(default_factory=dict, description="Details about the task status")
    error_details: Optional[Dict[str, Any]] = Field(default=None, description="Details about task errors")
    error_stack: Optional[str] = Field(default=None, description="Error stack trace if applicable")

    @model_validator(mode="after")
    def set_parent_from_step(self):
        """Set _parent to the step_id for parent-scoped uniqueness."""
        self._parent = self.step_id
        return self


class PlatformJobResult(EntityBase):
    """Result/artifact from a completed job.

    Parent-scoped: unique within (workspace, entity_type, parent=job).
    """

    __entity_type__: ClassVar[str] = "platform_job_result"

    job: str = Field(..., description="Parent job ID")
    artifact_url: str = Field(..., description="URL to the artifact")
    artifact_storage_type: FileStorageType = Field(..., description="Type of artifact storage")

    @model_validator(mode="after")
    def set_parent_from_job(self):
        """Set _parent to the job ID for parent-scoped uniqueness."""
        self._parent = self.job
        return self

    def to_response(self) -> PlatformJobResultResponse:
        """Convert to API response format."""
        return PlatformJobResultResponse(
            name=self.name,
            job=self.job,
            workspace=self.workspace,
            created_at=self.created_at,  # type: ignore
            updated_at=self.updated_at,  # type: ignore
            artifact_url=self.artifact_url,
            artifact_storage_type=self.artifact_storage_type,
        )
