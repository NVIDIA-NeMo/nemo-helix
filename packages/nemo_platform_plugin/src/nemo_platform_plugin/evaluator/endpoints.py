# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed endpoint definitions for the Evaluator service.

Single source of truth for the HTTP contract. Replaces the hand-written
``nemo_evaluator.sdk`` resource layer's direct ``NeMoPlatform._client`` usage.

The evaluator's high-level ``submit()`` convenience method (with its overloaded
signatures for row vs. taskset evaluation) stays in the SDK layer — it packages
parameters into a job spec and calls ``submit_evaluate_job`` or
``submit_agent_eval_job`` here for the actual HTTP POST.
"""

from __future__ import annotations

from abc import abstractmethod

from nemo_platform_plugin.client.endpoint import delete, get, post, put
from nemo_platform_plugin.client.types import BinaryContent, CursorPagination, Paginated
from nemo_platform_plugin.evaluator.types import (
    AgentEvalJob,
    AgentEvalResult,
    CreateMetricRequest,
    CreateTaskRequest,
    CreateTasksetRequest,
    EvalResult,
    EvaluateJob,
    EvaluatorHealth,
    EvaluatorJobLogsQueryParams,
    FlatQueryParams,
    HelloResponse,
    ListAgentEvalResultsQueryParams,
    ListEvalResultsQueryParams,
    ListMetricsQueryParams,
    ListRevisionsQueryParams,
    ListTasksetsQueryParams,
    ListTasksQueryParams,
    Metric,
    ProjectQueryParams,
    ReplaceTaskRequest,
    ReplaceTasksetRequest,
    RetrieveEvalJob,
    Revision,
    RevisionQueryParams,
    SubmitAgentEvalJobRequest,
    SubmitEvaluateJobRequest,
    SubmitRetrieveEvalJobRequest,
    Task,
    Taskset,
)
from nemo_platform_plugin.jobs.schemas import PlatformJobLog, PlatformJobResultResponse, PlatformJobStatusResponse
from nemo_platform_plugin.jobs.types import (
    ListJobsQueryParams,
    PlatformJobListResultResponse,
)

_EVAL_BASE = "/apis/evaluator/v2/workspaces/{workspace}"
_EVAL_JOBS = f"{_EVAL_BASE}/evaluate/jobs"
_AGENT_EVAL_JOBS = f"{_EVAL_BASE}/agent-evaluate/jobs"
_RETRIEVE_EVAL_JOBS = f"{_EVAL_BASE}/retrieve-eval/jobs"
_EVAL_RESULTS = f"{_EVAL_BASE}/eval-results"
_AGENT_EVAL_RESULTS = f"{_EVAL_BASE}/agent-eval-results"
_METRICS = f"{_EVAL_BASE}/metrics"
_TASKS = f"{_EVAL_BASE}/tasks"
_TASKSETS = f"{_EVAL_BASE}/tasksets"


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@get("/apis/evaluator/v1/healthz")
@abstractmethod
def get_health() -> EvaluatorHealth: ...


@get("/apis/evaluator/v1/hello/{name}")
@abstractmethod
def hello(*, name: str) -> HelloResponse: ...


# ---------------------------------------------------------------------------
# Evaluate jobs
# ---------------------------------------------------------------------------


@post(_EVAL_JOBS)
@abstractmethod
def submit_evaluate_job(*, workspace: str | None = None, body: SubmitEvaluateJobRequest) -> EvaluateJob: ...


@get(_EVAL_JOBS)
@abstractmethod
def list_evaluate_jobs(
    *, workspace: str | None = None, query_params: ListJobsQueryParams | FlatQueryParams | None = None
) -> Paginated[EvaluateJob]: ...


@get(f"{_EVAL_JOBS}/{{name}}")
@abstractmethod
def get_evaluate_job(*, workspace: str | None = None, name: str) -> EvaluateJob: ...


@get(f"{_EVAL_JOBS}/{{name}}/status")
@abstractmethod
def get_evaluate_job_status(*, workspace: str | None = None, name: str) -> PlatformJobStatusResponse: ...


@delete(f"{_EVAL_JOBS}/{{name}}")
@abstractmethod
def delete_evaluate_job(*, workspace: str | None = None, name: str) -> None: ...


@post(f"{_EVAL_JOBS}/{{name}}/cancel")
@abstractmethod
def cancel_evaluate_job(*, workspace: str | None = None, name: str) -> EvaluateJob: ...


@get(f"{_EVAL_JOBS}/{{name}}/logs")
@abstractmethod
def list_evaluate_job_logs(
    *, workspace: str | None = None, name: str, query_params: EvaluatorJobLogsQueryParams | None = None
) -> Paginated[PlatformJobLog, CursorPagination]: ...


@get(f"{_EVAL_JOBS}/{{name}}/results")
@abstractmethod
def list_evaluate_job_results(*, workspace: str | None = None, name: str) -> PlatformJobListResultResponse: ...


@get(f"{_EVAL_JOBS}/{{job}}/results/{{name}}")
@abstractmethod
def get_evaluate_job_result(*, workspace: str | None = None, job: str, name: str) -> PlatformJobResultResponse: ...


@get(f"{_EVAL_JOBS}/{{job}}/results/{{name}}/download")
@abstractmethod
def download_evaluate_job_result(*, workspace: str | None = None, job: str, name: str) -> BinaryContent: ...


# ---------------------------------------------------------------------------
# Agent-evaluate jobs
# ---------------------------------------------------------------------------


@post(_AGENT_EVAL_JOBS)
@abstractmethod
def submit_agent_eval_job(*, workspace: str | None = None, body: SubmitAgentEvalJobRequest) -> AgentEvalJob: ...


@get(_AGENT_EVAL_JOBS)
@abstractmethod
def list_agent_eval_jobs(
    *, workspace: str | None = None, query_params: ListJobsQueryParams | FlatQueryParams | None = None
) -> Paginated[AgentEvalJob]: ...


@get(f"{_AGENT_EVAL_JOBS}/{{name}}")
@abstractmethod
def get_agent_eval_job(*, workspace: str | None = None, name: str) -> AgentEvalJob: ...


@get(f"{_AGENT_EVAL_JOBS}/{{name}}/status")
@abstractmethod
def get_agent_eval_job_status(*, workspace: str | None = None, name: str) -> PlatformJobStatusResponse: ...


@delete(f"{_AGENT_EVAL_JOBS}/{{name}}")
@abstractmethod
def delete_agent_eval_job(*, workspace: str | None = None, name: str) -> None: ...


@post(f"{_AGENT_EVAL_JOBS}/{{name}}/cancel")
@abstractmethod
def cancel_agent_eval_job(*, workspace: str | None = None, name: str) -> AgentEvalJob: ...


@get(f"{_AGENT_EVAL_JOBS}/{{name}}/logs")
@abstractmethod
def list_agent_eval_job_logs(
    *, workspace: str | None = None, name: str, query_params: EvaluatorJobLogsQueryParams | None = None
) -> Paginated[PlatformJobLog, CursorPagination]: ...


@get(f"{_AGENT_EVAL_JOBS}/{{name}}/results")
@abstractmethod
def list_agent_eval_job_results(*, workspace: str | None = None, name: str) -> PlatformJobListResultResponse: ...


@get(f"{_AGENT_EVAL_JOBS}/{{job}}/results/{{name}}")
@abstractmethod
def get_agent_eval_job_result(*, workspace: str | None = None, job: str, name: str) -> PlatformJobResultResponse: ...


@get(f"{_AGENT_EVAL_JOBS}/{{job}}/results/{{name}}/download")
@abstractmethod
def download_agent_eval_job_result(*, workspace: str | None = None, job: str, name: str) -> BinaryContent: ...


# ---------------------------------------------------------------------------
# Retrieve-eval jobs
# ---------------------------------------------------------------------------


@post(_RETRIEVE_EVAL_JOBS)
@abstractmethod
def submit_retrieve_eval_job(
    *, workspace: str | None = None, body: SubmitRetrieveEvalJobRequest
) -> RetrieveEvalJob: ...


@get(_RETRIEVE_EVAL_JOBS)
@abstractmethod
def list_retrieve_eval_jobs(
    *, workspace: str | None = None, query_params: ListJobsQueryParams | FlatQueryParams | None = None
) -> Paginated[RetrieveEvalJob]: ...


@get(f"{_RETRIEVE_EVAL_JOBS}/{{name}}")
@abstractmethod
def get_retrieve_eval_job(*, workspace: str | None = None, name: str) -> RetrieveEvalJob: ...


@get(f"{_RETRIEVE_EVAL_JOBS}/{{name}}/status")
@abstractmethod
def get_retrieve_eval_job_status(*, workspace: str | None = None, name: str) -> PlatformJobStatusResponse: ...


@delete(f"{_RETRIEVE_EVAL_JOBS}/{{name}}")
@abstractmethod
def delete_retrieve_eval_job(*, workspace: str | None = None, name: str) -> None: ...


@post(f"{_RETRIEVE_EVAL_JOBS}/{{name}}/cancel")
@abstractmethod
def cancel_retrieve_eval_job(*, workspace: str | None = None, name: str) -> RetrieveEvalJob: ...


@get(f"{_RETRIEVE_EVAL_JOBS}/{{name}}/logs")
@abstractmethod
def list_retrieve_eval_job_logs(
    *, workspace: str | None = None, name: str, query_params: EvaluatorJobLogsQueryParams | None = None
) -> Paginated[PlatformJobLog, CursorPagination]: ...


@get(f"{_RETRIEVE_EVAL_JOBS}/{{name}}/results")
@abstractmethod
def list_retrieve_eval_job_results(*, workspace: str | None = None, name: str) -> PlatformJobListResultResponse: ...


@get(f"{_RETRIEVE_EVAL_JOBS}/{{job}}/results/{{name}}")
@abstractmethod
def get_retrieve_eval_job_result(*, workspace: str | None = None, job: str, name: str) -> PlatformJobResultResponse: ...


@get(f"{_RETRIEVE_EVAL_JOBS}/{{job}}/results/{{name}}/download")
@abstractmethod
def download_retrieve_eval_job_result(*, workspace: str | None = None, job: str, name: str) -> BinaryContent: ...


# ---------------------------------------------------------------------------
# Metrics CRUD
# ---------------------------------------------------------------------------


@get(f"{_METRICS}/{{name}}")
@abstractmethod
def get_metric(*, workspace: str | None = None, name: str) -> Metric: ...


@get(_METRICS)
@abstractmethod
def list_metrics(
    *, workspace: str | None = None, query_params: ListMetricsQueryParams | FlatQueryParams | None = None
) -> Paginated[Metric]: ...


@post(f"{_METRICS}/{{name}}")
@abstractmethod
def create_metric(
    *,
    workspace: str | None = None,
    name: str,
    body: CreateMetricRequest,
    query_params: ProjectQueryParams | FlatQueryParams | None = None,
) -> Metric: ...


@delete(f"{_METRICS}/{{name}}")
@abstractmethod
def delete_metric(*, workspace: str | None = None, name: str) -> None: ...


# ---------------------------------------------------------------------------
# Tasks CRUD + revisions
# ---------------------------------------------------------------------------


@get(f"{_TASKS}/{{name}}")
@abstractmethod
def get_task(*, workspace: str | None = None, name: str) -> Task: ...


@get(_TASKS)
@abstractmethod
def list_tasks(
    *, workspace: str | None = None, query_params: ListTasksQueryParams | FlatQueryParams | None = None
) -> Paginated[Task]: ...


@post(f"{_TASKS}/{{name}}")
@abstractmethod
def create_task(
    *,
    workspace: str | None = None,
    name: str,
    body: CreateTaskRequest,
    query_params: ProjectQueryParams | FlatQueryParams | None = None,
) -> Task: ...


@put(f"{_TASKS}/{{name}}")
@abstractmethod
def replace_task(
    *,
    workspace: str | None = None,
    name: str,
    body: ReplaceTaskRequest,
    query_params: ProjectQueryParams | FlatQueryParams | None = None,
) -> Task: ...


@get(f"{_TASKS}/{{name}}/revisions")
@abstractmethod
def list_task_revisions(
    *, workspace: str | None = None, name: str, query_params: ListRevisionsQueryParams | FlatQueryParams | None = None
) -> Paginated[Revision]: ...


@get(f"{_TASKS}/{{name}}/revisions/{{revision}}")
@abstractmethod
def get_task_revision(*, workspace: str | None = None, name: str, revision: str) -> Task: ...


@put(f"{_TASKS}/{{name}}/tags/{{tag}}")
@abstractmethod
def tag_task_revision(
    *,
    workspace: str | None = None,
    name: str,
    tag: str,
    query_params: RevisionQueryParams | FlatQueryParams | None = None,
) -> Task: ...


@delete(f"{_TASKS}/{{name}}")
@abstractmethod
def delete_task(*, workspace: str | None = None, name: str) -> None: ...


# ---------------------------------------------------------------------------
# Tasksets CRUD + revisions
# ---------------------------------------------------------------------------


@get(f"{_TASKSETS}/{{name}}")
@abstractmethod
def get_taskset(*, workspace: str | None = None, name: str) -> Taskset: ...


@get(_TASKSETS)
@abstractmethod
def list_tasksets(
    *, workspace: str | None = None, query_params: ListTasksetsQueryParams | FlatQueryParams | None = None
) -> Paginated[Taskset]: ...


@post(f"{_TASKSETS}/{{name}}")
@abstractmethod
def create_taskset(
    *,
    workspace: str | None = None,
    name: str,
    body: CreateTasksetRequest,
    query_params: ProjectQueryParams | FlatQueryParams | None = None,
) -> Taskset: ...


@put(f"{_TASKSETS}/{{name}}")
@abstractmethod
def replace_taskset(
    *,
    workspace: str | None = None,
    name: str,
    body: ReplaceTasksetRequest,
    query_params: ProjectQueryParams | FlatQueryParams | None = None,
) -> Taskset: ...


@get(f"{_TASKSETS}/{{name}}/revisions")
@abstractmethod
def list_taskset_revisions(
    *, workspace: str | None = None, name: str, query_params: ListRevisionsQueryParams | FlatQueryParams | None = None
) -> Paginated[Revision]: ...


@get(f"{_TASKSETS}/{{name}}/revisions/{{revision}}")
@abstractmethod
def get_taskset_revision(*, workspace: str | None = None, name: str, revision: str) -> Taskset: ...


@put(f"{_TASKSETS}/{{name}}/tags/{{tag}}")
@abstractmethod
def tag_taskset_revision(
    *,
    workspace: str | None = None,
    name: str,
    tag: str,
    query_params: RevisionQueryParams | FlatQueryParams | None = None,
) -> Taskset: ...


@delete(f"{_TASKSETS}/{{name}}")
@abstractmethod
def delete_taskset(*, workspace: str | None = None, name: str) -> None: ...


# ---------------------------------------------------------------------------
# Eval results
# ---------------------------------------------------------------------------


@get(f"{_EVAL_RESULTS}/{{name}}")
@abstractmethod
def get_eval_result(*, workspace: str | None = None, name: str) -> EvalResult: ...


@get(_EVAL_RESULTS)
@abstractmethod
def list_eval_results(
    *, workspace: str | None = None, query_params: ListEvalResultsQueryParams | FlatQueryParams | None = None
) -> Paginated[EvalResult]: ...


@delete(f"{_EVAL_RESULTS}/{{name}}")
@abstractmethod
def delete_eval_result(*, workspace: str | None = None, name: str) -> None: ...


# ---------------------------------------------------------------------------
# Agent eval results
# ---------------------------------------------------------------------------


@get(f"{_AGENT_EVAL_RESULTS}/{{name}}")
@abstractmethod
def get_agent_eval_result(*, workspace: str | None = None, name: str) -> AgentEvalResult: ...


@get(_AGENT_EVAL_RESULTS)
@abstractmethod
def list_agent_eval_results(
    *, workspace: str | None = None, query_params: ListAgentEvalResultsQueryParams | FlatQueryParams | None = None
) -> Paginated[AgentEvalResult]: ...


@delete(f"{_AGENT_EVAL_RESULTS}/{{name}}")
@abstractmethod
def delete_agent_eval_result(*, workspace: str | None = None, name: str) -> None: ...
