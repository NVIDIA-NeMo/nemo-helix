# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed request/response DTOs for the Insights plugin HTTP client."""

from __future__ import annotations

from datetime import datetime
from typing import NotRequired, TypedDict

from nemo_platform_plugin.jobs.schemas import PlatformJobStatus
from pydantic import BaseModel, ConfigDict, Field, JsonValue

ANALYSIS_JOB_NAME = "analyze-job"


class AnalyzeSpec(BaseModel):
    """Canonical input for one insights analyst run."""

    model_config = ConfigDict(extra="forbid")

    agent: str = Field(description="Agent under test.")
    ethos: str | None = Field(
        default=None,
        description="Optional Ethos Markdown for the agent under test.",
    )
    base_url: str | None = Field(
        default=None,
        description="Optional platform base URL. Unset uses the active platform context.",
    )
    insights_output: str | None = Field(
        default=None,
        description=(
            "Optional local YAML path mirroring the Insights the platform stored. "
            "Container-local unless it points at mounted storage."
        ),
    )
    since: datetime | None = Field(
        default=None,
        description="Optional lower bound for incremental trace/span analysis.",
    )
    update_analysis_config: bool = Field(
        default=True,
        description="Update the matching AnalysisRunStatus with run metadata.",
    )
    default_model: str = Field(description="Workspace-qualified default Model Entity ID selected during setup.")
    fast_model: str = Field(description="Workspace-qualified fast Model Entity ID selected during setup.")


class ListInsightsQueryParams(TypedDict, total=False):
    page: NotRequired[int]
    page_size: NotRequired[int]
    sort: NotRequired[str]
    agent: NotRequired[str]
    status: NotRequired[str]


class ListAnalysisConfigsQueryParams(TypedDict, total=False):
    page: NotRequired[int]
    page_size: NotRequired[int]
    sort: NotRequired[str]
    enabled: NotRequired[bool]


class ListAnalysisRunsQueryParams(TypedDict, total=False):
    page: NotRequired[int]
    page_size: NotRequired[int]
    sort: NotRequired[str]
    agent: NotRequired[str]


class ListAnalysisRunStatusesQueryParams(TypedDict, total=False):
    page: NotRequired[int]
    page_size: NotRequired[int]
    sort: NotRequired[str]


class CreateAnalysisJobRequest(BaseModel):
    """Body accepted by the Insights analysis job route."""

    name: str | None = None
    description: str | None = None
    project: str | None = None
    spec: AnalyzeSpec
    ownership: dict[str, JsonValue] | None = None
    custom_fields: dict[str, JsonValue] | None = None
    output_location: str | None = None


class AnalysisJob(BaseModel):
    """Insights analysis job response from the plugin-owned job route."""

    id: str | None = None
    name: str
    description: str | None = None
    project: str | None = None
    workspace: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    spec: AnalyzeSpec
    status: PlatformJobStatus | None = None
    status_details: dict[str, JsonValue] | None = None
    error_details: dict[str, JsonValue] | None = None
    ownership: dict[str, JsonValue] | None = None
    custom_fields: dict[str, JsonValue] | None = None


class ListAnalysisJobsQueryParams(TypedDict, total=False):
    page: NotRequired[int]
    page_size: NotRequired[int]
    sort: NotRequired[str]
    filter: NotRequired[str | dict[str, JsonValue]]
