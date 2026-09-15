# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed endpoint definitions for the Insights plugin API."""

from __future__ import annotations

from abc import abstractmethod

from nemo_insights_plugin.entities import AnalysisConfig, AnalysisRun, AnalysisRunStatus, Insight
from nemo_insights_plugin.schema import (
    AnalysisRunResponse,
    CreateAnalysisRunRequest,
    CreateInsightRequest,
    EnableAnalysisConfigRequest,
    InsightListItem,
    UpdateAnalysisConfigRequest,
    UpdateAnalysisRunStatusRequest,
    UpdateInsightRequest,
)
from nemo_insights_plugin.types import (
    ANALYSIS_JOB_NAME,
    AnalysisJob,
    CreateAnalysisJobRequest,
    ListAnalysisConfigsQueryParams,
    ListAnalysisJobsQueryParams,
    ListAnalysisRunsQueryParams,
    ListAnalysisRunStatusesQueryParams,
    ListInsightsQueryParams,
)
from nemo_platform_plugin.client.endpoint import delete, get, patch, post
from nemo_platform_plugin.client.types import Paginated

_INSIGHTS_BASE = "/apis/insights/v2/workspaces/{workspace}"
_ANALYSIS_JOBS = f"{_INSIGHTS_BASE}/jobs/{ANALYSIS_JOB_NAME}"


@post(f"{_INSIGHTS_BASE}/insights")
@abstractmethod
def create_insight(*, workspace: str | None = None, body: CreateInsightRequest) -> Insight: ...


@get(f"{_INSIGHTS_BASE}/insights")
@abstractmethod
def list_insights(
    *, workspace: str | None = None, query_params: ListInsightsQueryParams | None = None
) -> Paginated[InsightListItem]: ...


@get(f"{_INSIGHTS_BASE}/insights/{{insight_id}}")
@abstractmethod
def get_insight(*, workspace: str | None = None, insight_id: str) -> Insight: ...


@patch(f"{_INSIGHTS_BASE}/insights/{{insight_id}}")
@abstractmethod
def update_insight(*, workspace: str | None = None, insight_id: str, body: UpdateInsightRequest) -> Insight: ...


@delete(f"{_INSIGHTS_BASE}/insights/{{insight_id}}")
@abstractmethod
def delete_insight(*, workspace: str | None = None, insight_id: str) -> None: ...


@post(f"{_INSIGHTS_BASE}/analysis-configs/{{agent}}/enable")
@abstractmethod
def enable_analysis_config(
    *, workspace: str | None = None, agent: str, body: EnableAnalysisConfigRequest
) -> AnalysisConfig: ...


@post(f"{_INSIGHTS_BASE}/analysis-configs/{{agent}}/disable")
@abstractmethod
def disable_analysis_config(*, workspace: str | None = None, agent: str) -> AnalysisConfig: ...


@get(f"{_INSIGHTS_BASE}/analysis-configs")
@abstractmethod
def list_analysis_configs(
    *, workspace: str | None = None, query_params: ListAnalysisConfigsQueryParams | None = None
) -> Paginated[AnalysisConfig]: ...


@get(f"{_INSIGHTS_BASE}/analysis-configs/{{agent}}")
@abstractmethod
def get_analysis_config(*, workspace: str | None = None, agent: str) -> AnalysisConfig: ...


@patch(f"{_INSIGHTS_BASE}/analysis-configs/{{agent}}")
@abstractmethod
def update_analysis_config(
    *, workspace: str | None = None, agent: str, body: UpdateAnalysisConfigRequest
) -> AnalysisConfig: ...


@get(f"{_INSIGHTS_BASE}/analysis-run-statuses")
@abstractmethod
def list_analysis_run_statuses(
    *, workspace: str | None = None, query_params: ListAnalysisRunStatusesQueryParams | None = None
) -> Paginated[AnalysisRunStatus]: ...


@get(f"{_INSIGHTS_BASE}/analysis-run-statuses/{{agent}}")
@abstractmethod
def get_analysis_run_status(*, workspace: str | None = None, agent: str) -> AnalysisRunStatus: ...


@patch(f"{_INSIGHTS_BASE}/analysis-run-statuses/{{agent}}")
@abstractmethod
def update_analysis_run_status(
    *, workspace: str | None = None, agent: str, body: UpdateAnalysisRunStatusRequest
) -> AnalysisRunStatus: ...


@post(f"{_INSIGHTS_BASE}/analysis-runs")
@abstractmethod
def create_analysis_run(*, workspace: str | None = None, body: CreateAnalysisRunRequest) -> AnalysisRunResponse: ...


@get(f"{_INSIGHTS_BASE}/analysis-runs")
@abstractmethod
def list_analysis_runs(
    *, workspace: str | None = None, query_params: ListAnalysisRunsQueryParams | None = None
) -> Paginated[AnalysisRun]: ...


@get(f"{_INSIGHTS_BASE}/analysis-runs/{{name}}")
@abstractmethod
def get_analysis_run(*, workspace: str | None = None, name: str) -> AnalysisRunResponse: ...


@post(_ANALYSIS_JOBS)
@abstractmethod
def create_analysis_job(*, workspace: str | None = None, body: CreateAnalysisJobRequest) -> AnalysisJob: ...


@get(_ANALYSIS_JOBS)
@abstractmethod
def list_analysis_jobs(
    *, workspace: str | None = None, query_params: ListAnalysisJobsQueryParams | None = None
) -> Paginated[AnalysisJob]: ...


@get(f"{_ANALYSIS_JOBS}/{{name}}")
@abstractmethod
def get_analysis_job(*, workspace: str | None = None, name: str) -> AnalysisJob: ...
