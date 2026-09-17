# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request/response DTOs for the Jobs service HTTP contract.

These types define what job endpoints accept and return.  Both the server
(FastAPI routes in ``nmp.core.jobs.api``) and the typed HTTP client import
from here — one source of truth, no Stainless-generated duplicates.

The deep spec types live in sibling modules:
- :mod:`nemo_platform_plugin.jobs.spec` — ``PlatformJobSpec`` and children
- :mod:`nemo_platform_plugin.jobs.providers` — the executor tree
- :mod:`nemo_platform_plugin.jobs.execution_profiles` — backend profiles
- :mod:`nemo_platform_plugin.jobs.schemas` — status/result/log DTOs
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, NotRequired, Optional, TypedDict

from nemo_platform_plugin.files.types import MAX_LENGTH as FILESET_NAME_MAX_LENGTH
from nemo_platform_plugin.files.types import NAME_PATTERN as FILESET_NAME_PATTERN
from nemo_platform_plugin.files.types import NAME_PATTERN_DESCRIPTION as FILESET_NAME_DESCRIPTION
from nemo_platform_plugin.jobs.schemas import (
    PlatformJobResultResponse,
    PlatformJobStatus,
)
from nemo_platform_plugin.jobs.spec import PlatformJobSpec, PlatformJobStepSpec
from nemo_platform_plugin.schema import Value
from pydantic import BaseModel, Field, RootModel, field_validator

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
# Auth context (data-only mirror of nmp.common.auth.AuthContext)
# ---------------------------------------------------------------------------


class AuthContext(BaseModel):
    """Auth context captured at resource creation for delegated access.

    Stores a snapshot of the creating principal's identity so that controllers
    can later act on their behalf (e.g., accessing secrets).

    This is the wire/data shape.  The server's ``nmp.common.auth.AuthContext``
    adds ``from_principal`` / ``to_principal`` behaviour on top of the same
    fields.
    """

    principal_id: str = Field(..., description="The principal's unique identifier")
    principal_email: Optional[str] = Field(default=None, description="The principal's email address")
    principal_groups: list[str] = Field(default_factory=list, description="Groups the principal belongs to")
    principal_on_behalf_of: Optional[str] = Field(
        default=None, description="If acting on behalf of another principal, their principal ID"
    )
    principal_on_behalf_of_groups: Optional[list[str]] = Field(
        default=None, description="Groups the on-behalf-of principal belongs to"
    )
    principal_on_behalf_of_email: Optional[str] = Field(
        default=None, description="The on-behalf-of principal's email address"
    )


# ---------------------------------------------------------------------------
# Sort fields
# ---------------------------------------------------------------------------


class PlatformJobLogSortField(str, Enum):
    TIMESTAMP_ASC = "timestamp"
    TIMESTAMP_DESC = "-timestamp"

    def get_field_name(self) -> str:
        return self.value.lstrip("-")

    def get_sort_direction(self) -> str:
        return "desc" if self.value.startswith("-") else "asc"


class PlatformJobSortField(str, Enum):
    CREATED_AT_ASC = "created_at"
    CREATED_AT_DESC = "-created_at"
    UPDATED_AT_ASC = "updated_at"
    UPDATED_AT_DESC = "-updated_at"

    def get_field_name(self) -> str:
        return self.value.lstrip("-")

    def get_sort_direction(self) -> str:
        return "desc" if self.value.startswith("-") else "asc"


class PlatformJobListSortField(str, Enum):
    """Sort fields for the job *list* endpoint."""

    # Superset of PlatformJobSortField with `source`; only the job list can sort
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


class PlatformJobAttemptSortField(str, Enum):
    SEQ_ASC = "seq"
    SEQ_DESC = "-seq"

    def get_field_name(self) -> str:
        return self.value.lstrip("-")

    def get_sort_direction(self) -> str:
        return "desc" if self.value.startswith("-") else "asc"


# ---------------------------------------------------------------------------
# Response DTOs
# ---------------------------------------------------------------------------


class PlatformJobResponse(BaseModel):
    """Response model for a platform job."""

    id: str
    attempt_id: str
    name: str
    workspace: str = Field(..., description="Workspace identifier")
    project: Optional[str] = Field(default=None, description="Project URN")
    description: str | None = None
    source: str
    spec: dict[str, Any] = Field(default_factory=dict, description="Job Spec")
    platform_spec: PlatformJobSpec
    fileset: str = Field(..., description="Fileset ID for storing job artifacts")
    output_location: Optional[str] = Field(
        default=None, description="Caller-supplied artifact fileset; None when the fileset was auto-created"
    )
    status: PlatformJobStatus
    status_details: dict[str, Any] = Field(default_factory=dict, description="Details about the job status")
    error_details: dict[str, Any] | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ownership: Optional[dict[str, Any]] = None
    custom_fields: Optional[dict[str, Any]] = Field(default=None, description="Custom Fields")


class PlatformJobStepResponse(BaseModel):
    """Response model for a job step (wire shape of the ``PlatformJobStep`` entity)."""

    id: str
    entity_id: str
    parent: str = Field(..., description="Parent entity ID (the attempt ID)")
    attempt_id: str = Field(..., description="Parent attempt ID")
    name: str | None = None
    workspace: str
    project: str | None = None
    config: dict[str, Any] = Field(default_factory=dict, description="Configuration for the step")
    status: PlatformJobStatus = PlatformJobStatus.CREATED
    status_details: dict[str, Any] = Field(default_factory=dict, description="Status details")
    error_details: dict[str, Any] | None = None
    created_at: datetime | None = None
    created_by: str | None = None
    updated_at: datetime | None = None
    updated_by: str | None = None


class PlatformJobStepWithContext(BaseModel):
    """Step with additional context from parent job/attempt."""

    id: str
    job: str
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
    step_spec: PlatformJobStepSpec | None = None
    status: PlatformJobStatus = PlatformJobStatus.CREATED
    status_details: dict[str, Any] | None = None
    error_details: dict[str, Any] | None = None
    auth_context: Optional[AuthContext] = Field(default=None, description="Auth context for task execution")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PlatformJobTaskResponse(BaseModel):
    """Response model for a job task (wire shape of the ``PlatformJobTask`` entity)."""

    id: str
    entity_id: str
    parent: str = Field(..., description="Parent entity ID (the step ID)")
    step_id: str = Field(..., description="Parent step ID")
    name: str | None = None
    workspace: str
    project: str | None = None
    status: PlatformJobStatus = PlatformJobStatus.PENDING
    status_details: dict[str, Any] = Field(default_factory=dict, description="Details about the task status")
    error_details: dict[str, Any] | None = None
    error_stack: str | None = None
    created_at: datetime | None = None
    created_by: str | None = None
    updated_at: datetime | None = None
    updated_by: str | None = None


class PlatformJobListResultResponse(Value):
    """Response model for listing job results."""

    data: list[PlatformJobResultResponse]


class PlatformJobListTaskResponse(Value):
    """Response model for listing job tasks."""

    data: list[PlatformJobTaskResponse]


# ---------------------------------------------------------------------------
# Request DTOs
# ---------------------------------------------------------------------------


class CreatePlatformJobRequest(BaseModel):
    """Request model for creating a new platform job."""

    name: Optional[str] = None
    description: Optional[str] = None
    project: Optional[str] = None
    spec: dict
    platform_spec: PlatformJobSpec
    source: str
    ownership: Optional[dict] = None
    custom_fields: Optional[dict] = None
    output_location: Optional[str] = None

    _validate_output_location = field_validator("output_location")(validate_output_location)


class PlatformJobTaskUpdate(BaseModel):
    """Request model for updating a platform job task."""

    status: PlatformJobStatus = PlatformJobStatus.PENDING
    status_details: dict[str, Any] | None = None
    error_details: dict[str, Any] | None = None
    error_stack: str | None = None


class PlatformJobStatusUpdateRequest(BaseModel):
    """Request model for updating job status."""

    status: PlatformJobStatus = Field(..., description="The new status to set for the job.")
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
PlatformJobStatusDetailsUpdateRequest = dict[str, Any]


class JobStatusDetailsUpdate(RootModel[dict[str, Any]]):
    """Client request body for ``update_job_status_details`` (a bare JSON object)."""


# NB: list *filter* models (``PlatformJobsListFilter`` etc.) are intentionally
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


class ListJobResultsQueryParams(TypedDict, total=False):
    sort: NotRequired[str]


class JobLogsQueryParams(TypedDict, total=False):
    limit: NotRequired[int]
    page_cursor: NotRequired[str]
    attempt_id: NotRequired[int]
    step_id: NotRequired[str]
    task_id: NotRequired[str]
    tail: NotRequired[int]
