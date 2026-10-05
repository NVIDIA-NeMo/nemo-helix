# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schemas for the v2 Jobs Service.

Most request/response/filter/sort types now live in
:mod:`nemo_helix_plugin.jobs.types` so that both the server and the typed
HTTP client (``JobsClient``) share one source of truth; this module
re-exports them.

Two list-response wrappers (``HelixJobListResultResponse``,
``HelixJobListTaskResponse``) remain defined here because they wrap raw
entity instances server-side; the plugin exposes DTO equivalents for clients.
"""

from typing import Annotated, Any, Dict, List, Optional

# Re-exported shared types (single source of truth in the plugin).
# NB: ``AuthContext`` is intentionally NOT re-exported from the plugin here —
# this module exposes the behaviour-carrying ``nhx.common.auth.AuthContext``
# (imported above), which the ``HelixJobStepWithContext`` subclass uses.
from nemo_helix_plugin.jobs import types as _types
from nhx.common.auth import AuthContext
from nhx.common.entities import (
    DatetimeFilter,
    Filter,
    StringFilter,
    Value,
    get_random_id,
    map_entity_field,
)
from nhx.common.jobs.schemas import HelixJobResultResponse
from nhx.common.jobs.schemas import HelixJobStatus as HelixJobStatus
from nhx.core.jobs.entities import HelixJobTask
from pydantic import Field

CreateHelixJobRequest = _types.CreateHelixJobRequest
HelixJobAttemptSortField = _types.HelixJobAttemptSortField
HelixJobLogSortField = _types.HelixJobLogSortField
HelixJobListSortField = _types.HelixJobListSortField
HelixJobResponse = _types.HelixJobResponse
HelixJobSortField = _types.HelixJobSortField
HelixJobStatusDetailsUpdateRequest = _types.HelixJobStatusDetailsUpdateRequest
HelixJobStatusUpdateRequest = _types.HelixJobStatusUpdateRequest
HelixJobTaskUpdate = _types.HelixJobTaskUpdate
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


class HelixJobListResultResponse(Value):
    """Response model for listing job results."""

    data: List[HelixJobResultResponse]


class HelixJobListTaskResponse(Value):
    """Response model for listing job tasks."""

    data: List[HelixJobTask]


class HelixJobStepWithContext(_types.HelixJobStepWithContext):
    """Step with additional context from parent job/attempt."""

    # Overrides ``auth_context`` with the behaviour-carrying
    # ``nhx.common.auth.AuthContext`` (``to_principal`` / ``from_principal``);
    # the plugin base uses the data-only mirror for the wire shape.
    auth_context: Optional[AuthContext] = Field(default=None, description="Auth context for task execution")


# =============================================================================
# Filter Schemas (server-side — subclass the entity-store ``Filter`` for
# field-mapping / translation support; not part of the client wire contract)
# =============================================================================


class HelixJobsListFilter(Filter):
    """Filter options for listing platform jobs."""

    workspace: Optional[str] = Field(None, description="Workspace of the job.")
    project: Optional[str] = Field(None, description="Project of the job.")
    name: StringFilter | str | None = Field(None, description="Name of the job.")
    created_at: Optional[DatetimeFilter] = Field(None, description="Jobs created at 'gte' datetime or 'lte' datetime.")
    updated_at: Optional[DatetimeFilter] = Field(None, description="Jobs updated at 'gte' datetime or 'lte' datetime.")
    status: Optional[HelixJobStatus | list[HelixJobStatus]] = Field(None, description="The current status.")
    source: StringFilter | str | None = Field(None, description="The source of the job.")
    spec: Annotated[Optional[Dict[str, Any]], map_entity_field("data.spec", namespace=True)] = Field(
        None, description="Filter on a path within the job's plugin-defined spec, e.g. `spec.target.format`."
    )


class HelixJobAttemptsListFilter(Filter):
    """Filter options for listing platform job attempts."""

    status: Optional[HelixJobStatus] = Field(None, description="The current status.")


class HelixJobStepsListFilter(Filter):
    """Filter options for listing platform job steps."""

    # job/source kept as str pending AIRCORE-388 (read as scalars by the in-memory dispatcher)
    job: Optional[str] = Field(None, description="The ID of the job to filter steps by.")
    status: Optional[List[HelixJobStatus]] = Field(None, description="The list of statuses to filter steps by.")
    source: Optional[str] = Field(None, description="The source of the job steps.")
