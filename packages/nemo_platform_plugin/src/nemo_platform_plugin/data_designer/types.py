# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared request/response types for the Data Designer service.

Single source of truth for the HTTP contract. The Data Designer plugin
exposes a streaming preview function and a job-submission collection
(``/jobs/create``) rather than standard entity CRUD.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, NotRequired, TypedDict

from nemo_platform_plugin.jobs.schemas import PlatformJobStatus
from nemo_platform_plugin.schema import Page
from pydantic import BaseModel, Field, JsonValue

JsonMap = dict[str, JsonValue]
DataDesignerJobCollection = Literal["create", "retrieval-generate", "retrieval-prepare", "retrieval-run"]

# ---------------------------------------------------------------------------
# Preview types
# ---------------------------------------------------------------------------


class PreviewRequest(BaseModel):
    """Request body for the Data Designer preview endpoint.

    Mirrors ``PreviewSpec`` from the plugin's function layer.  The ``config``
    field is an opaque dict because ``DataDesignerConfig`` is defined in the
    ``data_designer`` package, not in ``nemo_platform_plugin``.  Callers that
    have a ``DataDesignerConfigBuilder`` should call ``.build()`` and
    ``.to_dict()`` (or ``model_dump(mode="json")``) to produce this dict.
    """

    config: JsonMap
    num_records: int | None = None


class RetrievalPreviewRequest(BaseModel):
    """Request body for the retrieval-preview endpoint."""

    generate: JsonMap
    num_records: int = Field(default=1, ge=1)


class PreviewFrameData(BaseModel):
    """A single NDJSON frame from the streaming preview response.

    The ``kind`` field discriminates the frame type.  Callers that need
    typed frame parsing should use the plugin's ``PreviewFrame`` union
    (``nemo_data_designer_plugin.functions._types``) to validate each line.
    """

    model_config = {"extra": "allow"}

    kind: str


# ---------------------------------------------------------------------------
# Job types
# ---------------------------------------------------------------------------


class DataDesignerJobRequest(BaseModel):
    """Request body for creating a Data Designer generation job.

    Mirrors ``BaseJobRequest`` with ``spec`` set to the Data Designer job
    config dict.  The ``spec`` is an opaque dict because
    ``DataDesignerJobConfig`` is defined in the plugin package.
    """

    name: str | None = None
    description: str | None = None
    project: str | None = None
    spec: JsonMap = Field(default_factory=dict)
    profile: str | None = None
    options: JsonMap | None = None
    ownership: JsonMap | None = None
    custom_fields: JsonMap | None = None
    output_location: str | None = None


class DataDesignerJobResponse(BaseModel):
    """Response model for a Data Designer job.

    Mirrors ``BaseJob`` with ``spec`` as an opaque dict.  The platform job
    status and timestamps come from the jobs service.
    """

    id: str | None = None
    name: str
    description: str | None = None
    project: str | None = None
    workspace: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    spec: JsonMap = Field(default_factory=dict)
    status: PlatformJobStatus | None = None
    status_details: JsonMap | None = None
    error_details: JsonMap | None = None
    ownership: JsonMap | None = None
    custom_fields: JsonMap | None = None


DataDesignerJobPage = Page[DataDesignerJobResponse]


# ---------------------------------------------------------------------------
# Query parameter types
# ---------------------------------------------------------------------------


class ListDataDesignerJobsQueryParams(TypedDict, total=False):
    page: NotRequired[int]
    page_size: NotRequired[int]
    sort: NotRequired[str]
    filter: NotRequired[str]


class DataDesignerJobLogsQueryParams(TypedDict, total=False):
    limit: NotRequired[int]
    page_cursor: NotRequired[str]
    tail: NotRequired[int]
