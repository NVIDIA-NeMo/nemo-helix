# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schemas for the v2 Jobs Service.

Most request/response/filter/sort types now live in
:mod:`nemo_platform_plugin.jobs.types` so that both the server and the typed
HTTP client (``JobsClient``) share one source of truth; this module
re-exports them.

Two list-response wrappers (``PlatformJobListResultResponse``,
``PlatformJobListTaskResponse``) remain defined here because they wrap raw
entity instances server-side; the plugin exposes DTO equivalents for clients.
"""

from typing import Annotated, Any, Dict, List, Optional

# Re-exported shared types (single source of truth in the plugin).
# NB: ``AuthContext`` is intentionally NOT re-exported from the plugin here —
# this module exposes the behaviour-carrying ``nmp.common.auth.AuthContext``
# (imported above), which the ``PlatformJobStepWithContext`` subclass uses.
from nemo_platform_plugin.jobs import types as _types
from nmp.common.auth import AuthContext
from nmp.common.entities import (
    DatetimeFilter,
    Filter,
    StringFilter,
    Value,
    get_random_id,
    map_entity_field,
)
from nmp.common.jobs.schemas import PlatformJobResultResponse
from nmp.common.jobs.schemas import PlatformJobStatus as PlatformJobStatus
from nmp.core.jobs.entities import PlatformJobTask
from pydantic import Field

CreatePlatformJobRequest = _types.CreatePlatformJobRequest
PlatformJobAttemptSortField = _types.PlatformJobAttemptSortField
PlatformJobLogSortField = _types.PlatformJobLogSortField
PlatformJobListSortField = _types.PlatformJobListSortField
PlatformJobResponse = _types.PlatformJobResponse
PlatformJobSortField = _types.PlatformJobSortField
PlatformJobStatusDetailsUpdateRequest = _types.PlatformJobStatusDetailsUpdateRequest
PlatformJobStatusUpdateRequest = _types.PlatformJobStatusUpdateRequest
PlatformJobTaskUpdate = _types.PlatformJobTaskUpdate
job_artifact_base_path = _types.job_artifact_base_path

# =============================================================================
# Utilities
# =============================================================================


def get_model_id(prefix: str) -> str:
    """Generate a random ID with the given prefix.

    Uses lowercase characters for compatibility with all platform identifiers.
    """
    # `get_random_id` includes uppercase characters, which won't
    # work with all platform identifiers.
    return get_random_id(prefix).lower()


# =============================================================================
# Response Schemas (server-side — wrap raw entity instances)
# =============================================================================


class PlatformJobListResultResponse(Value):
    """Response model for listing job results."""

    data: List[PlatformJobResultResponse]


class PlatformJobListTaskResponse(Value):
    """Response model for listing job tasks."""

    data: List[PlatformJobTask]


class PlatformJobStepWithContext(_types.PlatformJobStepWithContext):
    """Step with additional context from parent job/attempt."""

    # Overrides ``auth_context`` with the behaviour-carrying
    # ``nmp.common.auth.AuthContext`` (``to_principal`` / ``from_principal``);
    # the plugin base uses the data-only mirror for the wire shape.
    auth_context: Optional[AuthContext] = Field(default=None, description="Auth context for task execution")


# =============================================================================
# Filter Schemas (server-side — subclass the entity-store ``Filter`` for
# field-mapping / translation support; not part of the client wire contract)
# =============================================================================


class PlatformJobsListFilter(Filter):
    """Filter options for listing platform jobs."""

    workspace: Optional[str] = Field(None, description="Workspace of the job.")
    project: Optional[str] = Field(None, description="Project of the job.")
    name: StringFilter | str | None = Field(None, description="Name of the job.")
    created_at: Optional[DatetimeFilter] = Field(None, description="Jobs created at 'gte' datetime or 'lte' datetime.")
    updated_at: Optional[DatetimeFilter] = Field(None, description="Jobs updated at 'gte' datetime or 'lte' datetime.")
    status: Optional[PlatformJobStatus | list[PlatformJobStatus]] = Field(None, description="The current status.")
    source: StringFilter | str | None = Field(None, description="The source of the job.")
    spec: Annotated[Optional[Dict[str, Any]], map_entity_field("data.spec", namespace=True)] = Field(
        None, description="Filter on a path within the job's plugin-defined spec, e.g. `spec.target.format`."
    )


class PlatformJobAttemptsListFilter(Filter):
    """Filter options for listing platform job attempts."""

    status: Optional[PlatformJobStatus] = Field(None, description="The current status.")


class PlatformJobStepsListFilter(Filter):
    """Filter options for listing platform job steps."""

    # job/source kept as str pending AIRCORE-388 (read as scalars by the in-memory dispatcher)
    job: Optional[str] = Field(None, description="The ID of the job to filter steps by.")
    status: Optional[List[PlatformJobStatus]] = Field(None, description="The list of statuses to filter steps by.")
    source: Optional[str] = Field(None, description="The source of the job steps.")
