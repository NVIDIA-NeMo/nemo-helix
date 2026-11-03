# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request/response DTOs for the Jobs service HTTP contract.

These types define what job endpoints accept and return.  Both the server
(FastAPI routes in ``nhx.core.jobs.api``) and the typed HTTP client import
from here, so there is one source of truth.

The deep spec types live in sibling modules:
- :mod:`nemo_helix_plugin.jobs.spec` — ``HelixJobSpec`` and children
- :mod:`nemo_helix_plugin.jobs.providers` — the executor tree
- :mod:`nemo_helix_plugin.jobs.execution_profiles` — backend profiles
- :mod:`nemo_helix_plugin.jobs.schemas` — status/result/log DTOs
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, NotRequired, Optional, Self, TypedDict

from nemo_helix_plugin.entities.types import HTTPValidationError
from nemo_helix_plugin.entity_naming import NAME_MAX_LENGTH, NAME_PATTERN, NAME_PATTERN_DESCRIPTION
from nemo_helix_plugin.files.types import MAX_LENGTH as FILESET_NAME_MAX_LENGTH
from nemo_helix_plugin.files.types import NAME_PATTERN as FILESET_NAME_PATTERN
from nemo_helix_plugin.files.types import NAME_PATTERN_DESCRIPTION as FILESET_NAME_DESCRIPTION
from nemo_helix_plugin.jobs.schemas import (
    HelixJobResultResponse,
    HelixJobStatus,
)
from nemo_helix_plugin.jobs.spec import HelixJobSpec, HelixJobStepSpec
from nemo_helix_plugin.k8s_naming import k8s_safe_name, workspace_name_identity
from nemo_helix_plugin.schema import Value
from pydantic import BaseModel, ConfigDict, Field, RootModel, field_validator, model_validator

_FILESET_NAME_RE = re.compile(FILESET_NAME_PATTERN)


def validate_output_location(value: str | None) -> str | None:
    """Syntactic check for the ``output_location`` job-request field.

    Rejects anything that could not name an existing fileset, so a bad value fails
    before the dispatcher spends a lookup on it.
    """
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        raise ValueError("output_location must not be empty")
    if "#" in stripped:
        raise ValueError("subpath in output_location is not yet supported")
    if "/" in stripped:
        raise ValueError("output_location must be a bare fileset name; the workspace is implied by the request")
    if len(stripped) > FILESET_NAME_MAX_LENGTH:
        raise ValueError(f"output_location must be at most {FILESET_NAME_MAX_LENGTH} characters")
    if not _FILESET_NAME_RE.match(stripped):
        raise ValueError(f"output_location is not a valid fileset name. {FILESET_NAME_DESCRIPTION}")
    return stripped


JOB_FILESET_NAME_PREFIX = "job-fileset-"
# Longest job name that still fits verbatim in ``job-fileset-<name>``; a longer name is normalized,
# truncated, and given a hash suffix by ``derive_job_fileset_name``.
JOB_NAME_MAX_LENGTH_WITH_DERIVED_FILESET = NAME_MAX_LENGTH - len(JOB_FILESET_NAME_PREFIX)


JobName = Annotated[
    str,
    Field(
        pattern=NAME_PATTERN,
        max_length=NAME_MAX_LENGTH,
        description=f"Job name. {NAME_PATTERN_DESCRIPTION} Auto-generated when omitted.",
    ),
]


def derive_job_fileset_name(job_name: str, workspace: str) -> str:
    """Return a valid, length-safe ``job-fileset-`` name for a job's auto-created artifact fileset.

    A short job name nests verbatim as ``job-fileset-<name>``; a longer one is normalized,
    truncated, and given a hash suffix via :func:`k8s_safe_name`, keyed on the ``workspace/name``
    identity so the result is deterministic and collision-safe.
    """
    if len(job_name) <= JOB_NAME_MAX_LENGTH_WITH_DERIVED_FILESET:
        return f"{JOB_FILESET_NAME_PREFIX}{job_name}"
    safe_name_part = k8s_safe_name(
        base_name=job_name,
        max_length=JOB_NAME_MAX_LENGTH_WITH_DERIVED_FILESET,
        hash_input=workspace_name_identity(workspace, job_name),
        name_type="label",
    )
    return f"{JOB_FILESET_NAME_PREFIX}{safe_name_part}"


# Root folder for job artifacts inside a caller-supplied ``output_location`` fileset, so a
# fileset the caller also uses for their own files keeps job output in one predictable place.
JOB_ARTIFACT_ROOT = "jobs"


def job_artifact_base_path(job_name: str, output_location: str | None) -> str | None:
    """Folder inside the fileset that this job's artifacts (logs + results) nest under.

    ``None`` for an auto-created ``job-fileset-<name>``: the job owns the whole fileset, so
    its artifacts sit at the root. Writers and readers must both derive the value from here
    or logs get written where the log query cannot find them.
    """
    return f"{JOB_ARTIFACT_ROOT}/{job_name}" if output_location else None


# ---------------------------------------------------------------------------
# Auth context (data-only mirror of nhx.common.auth.AuthContext)
# ---------------------------------------------------------------------------


class AuthContext(BaseModel):
    """Auth context captured at resource creation for delegated access.

    Stores a snapshot of the creating principal's identity so that controllers
    can later act on their behalf (e.g., accessing secrets).

    This is the wire/data shape.  The server's ``nhx.common.auth.AuthContext``
    adds ``from_principal`` / ``to_principal`` behaviour on top of the same
    fields.
    """

    principal_id: str = Field(..., description="The principal's unique identifier")
    principal_account_id: Optional[str] = Field(
        default=None, description="Stable NeMo account identifier for the principal"
    )
    principal_email: Optional[str] = Field(default=None, description="The principal's email address")
    principal_groups: list[str] = Field(default_factory=list, description="Groups the principal belongs to")
    principal_authz_aliases: list[str] = Field(
        default_factory=list,
        description="Alternate trusted identifiers for the principal",
    )
    principal_on_behalf_of: Optional[str] = Field(
        default=None, description="If acting on behalf of another principal, their principal ID"
    )
    principal_on_behalf_of_groups: Optional[list[str]] = Field(
        default=None, description="Groups the on-behalf-of principal belongs to"
    )
    principal_on_behalf_of_email: Optional[str] = Field(
        default=None, description="The on-behalf-of principal's email address"
    )
    principal_on_behalf_of_account_id: Optional[str] = Field(
        default=None,
        description="Stable NeMo account identifier for the on-behalf-of principal",
    )
    principal_on_behalf_of_authz_aliases: list[str] = Field(
        default_factory=list,
        description="Alternate trusted identifiers for the on-behalf-of principal",
    )


# ---------------------------------------------------------------------------
# Sort fields
# ---------------------------------------------------------------------------


class HelixJobLogSortField(str, Enum):
    TIMESTAMP_ASC = "timestamp"
    TIMESTAMP_DESC = "-timestamp"

    def get_field_name(self) -> str:
        return self.value.lstrip("-")

    def get_sort_direction(self) -> str:
        return "desc" if self.value.startswith("-") else "asc"


class HelixJobSortField(str, Enum):
    CREATED_AT_ASC = "created_at"
    CREATED_AT_DESC = "-created_at"
    UPDATED_AT_ASC = "updated_at"
    UPDATED_AT_DESC = "-updated_at"

    def get_field_name(self) -> str:
        return self.value.lstrip("-")

    def get_sort_direction(self) -> str:
        return "desc" if self.value.startswith("-") else "asc"


class HelixJobListSortField(str, Enum):
    """Sort fields for the job *list* endpoint."""

    # Superset of HelixJobSortField with `source`; only the job list can sort
    # by source (steps/results/logs have no source field).
    CREATED_AT_ASC = "created_at"
    CREATED_AT_DESC = "-created_at"
    UPDATED_AT_ASC = "updated_at"
    UPDATED_AT_DESC = "-updated_at"
    SOURCE_ASC = "source"
    SOURCE_DESC = "-source"

    def get_field_name(self) -> str:
        return self.value.lstrip("-")

    def get_sort_direction(self) -> str:
        return "desc" if self.value.startswith("-") else "asc"


class HelixJobAttemptSortField(str, Enum):
    SEQ_ASC = "seq"
    SEQ_DESC = "-seq"

    def get_field_name(self) -> str:
        return self.value.lstrip("-")

    def get_sort_direction(self) -> str:
        return "desc" if self.value.startswith("-") else "asc"


# ---------------------------------------------------------------------------
# Response DTOs
# ---------------------------------------------------------------------------


class JobControl(BaseModel):
    """Mutable job behavior, stored on the job and kept off the execution spec.

    The attempt spec stays the immutable snapshot of what a run executes. Rerun
    copies the job spec into the next attempt and keeps using this object. In a
    control update, an omitted or null value keeps the job's current value.
    """

    pause_ttl_seconds: Optional[int] = Field(
        default=None,
        ge=0,
        description="How long this job keeps storage after it pauses, measured from stopped_at. "
        "When the job has no value, the platform paused-storage TTL applies. Cannot exceed the platform "
        "maximum. 0 reclaims on the next retention pass. In a control update, an omitted or null value "
        "keeps the current value. Changing this value does not move the start time.",
        json_schema_extra={"nullable": True},
    )


class JobDryRunCheck(BaseModel):
    """One check a dry run (create with ``?dry_run=true``) made."""

    name: str = Field(description="What was checked.")
    passed: bool
    message: str = Field(description="What the check found; on failure, what to fix.")


class JobDryRunFailureDetail(BaseModel):
    """What a dry run whose checks fail returns in ``detail``."""

    message: str = Field(description="The checks that failed.")
    checks: list[JobDryRunCheck] = Field(description="Every check the dry run made, passed and failed.")


class JobDryRunFailure(BaseModel):
    """The 422 body of a dry run whose checks fail."""

    detail: JobDryRunFailureDetail


class JobDryRunValidationErrorResponse(RootModel[HTTPValidationError | JobDryRunFailure]):
    """Request validation error, or a dry run whose checks fail.

    Create routes that take ``?dry_run=true`` declare this for 422. Declaring any 422 model
    replaces FastAPI's request-validation schema, so the union keeps both shapes.
    """


class HelixJobResponse(BaseModel):
    """Response model for a platform job."""

    id: str
    attempt_id: str
    name: str
    workspace: str = Field(..., description="Workspace identifier")
    project: Optional[str] = Field(default=None, description="Project URN")
    description: str | None = None
    source: str
    spec: dict[str, Any] = Field(default_factory=dict, description="Job Spec")
    platform_spec: HelixJobSpec
    control: JobControl = Field(
        default_factory=JobControl,
        description="Mutable job behavior. This is not part of the execution spec.",
    )
    fileset: str = Field(..., description="Fileset ID for storing job artifacts")
    output_location: Optional[str] = Field(
        default=None, description="Caller-supplied artifact fileset; None when the fileset was auto-created"
    )
    status: HelixJobStatus
    status_details: dict[str, Any] = Field(default_factory=dict, description="Details about the job status")
    error_details: dict[str, Any] | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ownership: Optional[dict[str, Any]] = None
    custom_fields: Optional[dict[str, Any]] = Field(default=None, description="Custom Fields")


class HelixJobAttemptResponse(BaseModel):
    """Response model for one execution attempt of a platform job."""

    id: str
    name: str
    job: str = Field(..., description="Parent job ID")
    workspace: str
    seq: int
    spec: dict[str, Any] = Field(default_factory=dict)
    platform_spec: HelixJobSpec
    status: HelixJobStatus
    status_details: dict[str, Any] = Field(default_factory=dict)
    error_details: dict[str, Any] | None = None
    created_at: datetime
    updated_at: datetime
    is_current: bool = False


class HelixJobStepResponse(BaseModel):
    """Response model for a job step (wire shape of the ``HelixJobStep`` entity)."""

    id: str
    entity_id: str
    parent: str = Field(..., description="Parent entity ID (the attempt ID)")
    attempt_id: str = Field(..., description="Parent attempt ID")
    name: str | None = None
    workspace: str
    project: str | None = None
    config: dict[str, Any] = Field(default_factory=dict, description="Configuration for the step")
    status: HelixJobStatus = HelixJobStatus.CREATED
    status_details: dict[str, Any] = Field(default_factory=dict, description="Status details")
    error_details: dict[str, Any] | None = None
    created_at: datetime | None = None
    created_by: str | None = None
    updated_at: datetime | None = None
    updated_by: str | None = None


class HelixJobStepWithContext(BaseModel):
    """Step with additional context from parent job/attempt."""

    id: str
    job: str
    job_id: str | None = Field(
        default=None,
        description="ID of the platform job. Unlike the job name, it is never reused by a later job.",
    )
    attempt_id: str
    fileset: str
    artifact_base_path: str | None = Field(
        default=None,
        description=(
            "Folder inside the fileset that this job's artifacts nest under; None when the job "
            "owns the whole fileset and its artifacts sit at the root"
        ),
    )
    workspace: str
    name: str
    step_spec: HelixJobStepSpec | None = None
    status: HelixJobStatus = HelixJobStatus.CREATED
    status_details: dict[str, Any] | None = None
    error_details: dict[str, Any] | None = None
    auth_context: Optional[AuthContext] = Field(default=None, description="Auth context for task execution")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HelixJobTaskResponse(BaseModel):
    """Response model for a job task (wire shape of the ``HelixJobTask`` entity)."""

    id: str
    entity_id: str
    parent: str = Field(..., description="Parent entity ID (the step ID)")
    step_id: str = Field(..., description="Parent step ID")
    name: str | None = None
    workspace: str
    project: str | None = None
    status: HelixJobStatus = HelixJobStatus.PENDING
    status_details: dict[str, Any] = Field(default_factory=dict, description="Details about the task status")
    error_details: dict[str, Any] | None = None
    error_stack: str | None = None
    created_at: datetime | None = None
    created_by: str | None = None
    updated_at: datetime | None = None
    updated_by: str | None = None


class HelixJobListResultResponse(Value):
    """Response model for listing job results."""

    data: list[HelixJobResultResponse]


class HelixJobListTaskResponse(Value):
    """Response model for listing job tasks."""

    data: list[HelixJobTaskResponse]


# ---------------------------------------------------------------------------
# Request DTOs
# ---------------------------------------------------------------------------


class CreateHelixJobRequest(BaseModel):
    """Request model for creating a new platform job."""

    model_config = ConfigDict(
        regex_engine="python-re",
        json_schema_extra={
            "examples": [
                {
                    "name": "hello-world",
                    "spec": {},
                    "platform_spec": {
                        "steps": [
                            {
                                "name": "run",
                                "executor": {
                                    "provider": "cpu",
                                    "container": {"image": "python:3.11"},
                                },
                                "config": {},
                            }
                        ]
                    },
                    "source": "api-reference",
                }
            ]
        },
    )

    name: Optional[JobName] = None
    description: Optional[str] = None
    project: Optional[str] = None
    spec: dict
    platform_spec: HelixJobSpec
    control: Optional[JobControl] = Field(
        default=None,
        description="Mutable job behavior, including how long storage is kept after a pause. "
        "When omitted, the platform paused-storage TTL applies.",
    )
    source: str
    ownership: Optional[dict] = None
    custom_fields: Optional[dict] = None
    output_location: Optional[str] = None

    _validate_output_location = field_validator("output_location")(validate_output_location)


class UpdateHelixJobRequest(BaseModel):
    """Validated partial update for a job description or definition."""

    model_config = ConfigDict(extra="forbid")

    description: str | None = Field(default=None, json_schema_extra={"nullable": True})
    spec: dict[str, Any] | None = None
    platform_spec: HelixJobSpec | None = None

    @model_validator(mode="after")
    def validate_paired_specs(self) -> Self:
        if not self.model_fields_set:
            raise ValueError("at least one mutable job field is required")
        spec_supplied = "spec" in self.model_fields_set
        platform_spec_supplied = "platform_spec" in self.model_fields_set
        if spec_supplied != platform_spec_supplied:
            raise ValueError("spec and platform_spec must be supplied together")
        if spec_supplied and (self.spec is None or self.platform_spec is None):
            raise ValueError("spec and platform_spec cannot be null")
        return self


class HelixJobTaskUpdate(BaseModel):
    """Request model for updating a platform job task."""

    status: HelixJobStatus = HelixJobStatus.PENDING
    status_details: dict[str, Any] | None = None
    error_details: dict[str, Any] | None = None
    error_stack: str | None = None


class HelixJobStatusUpdateRequest(BaseModel):
    """Request model for updating job status."""

    status: HelixJobStatus = Field(..., description="The new status to set for the job.")
    expected_step_id: str | None = Field(None, description="Reject the update if the step's entity ID has changed.")
    status_details: dict[str, Any] | None = Field(
        default_factory=dict, description="Optional status details related to the status update."
    )
    error_details: dict[str, Any] | None = Field(
        default_factory=dict, description="Optional error details related to the status update."
    )


# Status-details PATCH body: a free-form dict of status details.  The server
# accepts a bare JSON object (typed as ``dict[str, Any]``); the client uses the
# ``JobStatusDetailsUpdate`` RootModel wrapper so it can be passed as a typed
# request ``body`` (it serialises to the same bare object).
HelixJobStatusDetailsUpdateRequest = dict[str, Any]


class JobStatusDetailsUpdate(RootModel[dict[str, Any]]):
    """Client request body for attempt status-details updates (a bare JSON object)."""


# NB: list *filter* models (``HelixJobsListFilter`` etc.) are intentionally
# NOT defined here.  They subclass the entity-store ``Filter`` (with field
# mapping / translation) and are server-side only.  Clients pass a ``filter``
# query-param string via the query-param TypedDicts below.


# ---------------------------------------------------------------------------
# Query parameter types (client-side)
# ---------------------------------------------------------------------------


class ListJobsQueryParams(TypedDict, total=False):
    page: NotRequired[int]
    page_size: NotRequired[int]
    sort: NotRequired[str]
    filter: NotRequired[str | dict[str, Any]]


class ListStepsQueryParams(TypedDict, total=False):
    page: NotRequired[int]
    page_size: NotRequired[int]
    sort: NotRequired[str]
    filter: NotRequired[str | dict[str, Any]]


class ListAttemptsQueryParams(TypedDict, total=False):
    page: NotRequired[int]
    page_size: NotRequired[int]
    sort: NotRequired[str]
    filter: NotRequired[str | dict[str, Any]]


class ListJobResultsQueryParams(TypedDict, total=False):
    sort: NotRequired[str]
    attempt_id: NotRequired[str]


class JobStatusDetailsQueryParams(TypedDict, total=False):
    expected_step_id: NotRequired[str]
    expected_step_status: NotRequired[str]


class JobLogsQueryParams(TypedDict, total=False):
    limit: NotRequired[int]
    page_cursor: NotRequired[str]
    attempt_id: NotRequired[str | int]
    step_id: NotRequired[str]
    task_id: NotRequired[str]
    tail: NotRequired[int]
