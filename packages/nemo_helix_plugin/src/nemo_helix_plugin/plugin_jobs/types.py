# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request/response types shared by every plugin job collection.

Each :class:`~nemo_helix_plugin.job.NemoJob` mounted with
:func:`~nemo_helix_plugin.jobs.routes.add_job_routes` serves the same list contract at
``/apis/<plugin>/v2/workspaces/{workspace}/<collection>``; only the stored ``spec`` differs.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, NotRequired, TypedDict

from nemo_helix_plugin.jobs.schemas import HelixJobStatus
from pydantic import BaseModel, ConfigDict, Field


class PluginJob(BaseModel):
    """One job as a plugin's job collection returns it. The spec is kept as plugin-specific JSON."""

    model_config = ConfigDict(extra="allow")

    id: str | None = None
    attempt_id: str | None = None
    name: str
    description: str | None = None
    project: str | None = None
    workspace: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    spec: dict[str, Any] = Field(default_factory=dict)
    status: HelixJobStatus | None = None
    status_details: dict[str, Any] | None = None
    error_details: dict[str, Any] | None = None
    ownership: dict[str, Any] | None = None
    custom_fields: dict[str, Any] | None = None


class ListPluginJobsQueryParams(TypedDict, total=False):
    """Query parameters every plugin job collection's list route accepts."""

    page: NotRequired[int]
    page_size: NotRequired[int]
    sort: NotRequired[str]
    filter: NotRequired[str]
