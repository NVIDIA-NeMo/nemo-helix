# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request and response types for the Safe Synthesizer plugin API."""

from __future__ import annotations

from datetime import datetime
from typing import NotRequired, TypedDict

from nemo_helix_plugin.jobs.schemas import HelixJobStatus
from pydantic import BaseModel, ConfigDict, Field, JsonValue

JsonMap = dict[str, JsonValue]


class CreateSafeSynthesizerJobRequest(BaseModel):
    """Request body for creating a Safe Synthesizer job.

    ``spec`` is an opaque mapping validated server-side against the plugin's
    job config.
    """

    name: str | None = None
    description: str | None = None
    project: str | None = None
    spec: JsonMap
    ownership: JsonMap | None = None
    custom_fields: JsonMap | None = None


class SafeSynthesizerJobResponse(BaseModel):
    """A Safe Synthesizer job as returned by the plugin API."""

    model_config = ConfigDict(extra="allow")

    id: str | None = None
    name: str
    description: str | None = None
    project: str | None = None
    workspace: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    spec: JsonMap = Field(default_factory=dict)
    status: HelixJobStatus | None = None
    status_details: JsonMap | None = None
    error_details: JsonMap | None = None
    ownership: JsonMap | None = None
    custom_fields: JsonMap | None = None


class ListSafeSynthesizerJobsQueryParams(TypedDict, total=False):
    page: NotRequired[int]
    page_size: NotRequired[int]
    sort: NotRequired[str]
    filter: NotRequired[str]
