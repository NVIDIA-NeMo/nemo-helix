# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed endpoint definitions for the Evals service.

Single source of truth for the HTTP contract.

The evaluator's high-level ``submit()`` convenience method (with its overloaded
signatures for row vs. taskset evaluation) stays in the SDK layer — it packages
parameters into a job spec and calls ``submit_evaluate_job`` or
``submit_agent_eval_job`` here for the actual HTTP POST.
"""

from __future__ import annotations

from abc import abstractmethod

from nemo_helix_plugin.client.endpoint import delete, get, post, put
from nemo_helix_plugin.client.types import BinaryContent, CursorPagination, Paginated, PreparedRequest
from nemo_helix_plugin.evals.types import (
    AgentEvalJob,
    AgentEvalResult,
    CreateMetricRequest,
    CreateTaskImageBuildRequest,
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
    ListTaskImageBuildsQueryParams,
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
    TaskImageBuild,
    Taskset,
)
from nemo_helix_plugin.jobs.schemas import HelixJobLog, HelixJobResultResponse, HelixJobStatusResponse
from nemo_helix_plugin.jobs.selectors import JobReferenceKind
from nemo_helix_plugin.jobs.types import (
    HelixJobListResultResponse,
    JobSelectorQueryParams,
    ListJobsQueryParams,
)

_EVAL_BASE = "/apis/evals/v2/workspaces/{workspace}"
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


@get("/apis/evals/v1/healthz")
@abstractmethod
def get_health() -> EvaluatorHealth: ...


@get("/apis/evals/v1/hello/{name}")
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
def _raw_get_evaluate_job(
    *, workspace: str | None = None, name: str, query_params: JobSelectorQueryParams | None = None
) -> EvaluateJob: ...


def get_evaluate_job(
    *,
    workspace: str | None = None,
    name: str,
    query_params: JobSelectorQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[EvaluateJob]:
    return _raw_get_evaluate_job(
        workspace=workspace,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, job_reference_kind),
    )


def get_evaluate_job_by_id(
    *, workspace: str | None = None, job_id: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[EvaluateJob]:
    return _raw_get_evaluate_job(
        workspace=workspace,
        name=job_id,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.ID),
    )


def get_evaluate_job_by_name(
    *, workspace: str | None = None, job_name: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[EvaluateJob]:
    return _raw_get_evaluate_job(
        workspace=workspace,
        name=job_name,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.NAME),
    )


@get(f"{_EVAL_JOBS}/{{name}}/status")
@abstractmethod
def _raw_get_evaluate_job_status(
    *, workspace: str | None = None, name: str, query_params: JobSelectorQueryParams | None = None
) -> HelixJobStatusResponse: ...


def get_evaluate_job_status(
    *,
    workspace: str | None = None,
    name: str,
    query_params: JobSelectorQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[HelixJobStatusResponse]:
    return _raw_get_evaluate_job_status(
        workspace=workspace,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, job_reference_kind),
    )


def get_evaluate_job_status_by_id(
    *, workspace: str | None = None, job_id: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[HelixJobStatusResponse]:
    return _raw_get_evaluate_job_status(
        workspace=workspace,
        name=job_id,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.ID),
    )


def get_evaluate_job_status_by_name(
    *, workspace: str | None = None, job_name: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[HelixJobStatusResponse]:
    return _raw_get_evaluate_job_status(
        workspace=workspace,
        name=job_name,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.NAME),
    )


@delete(f"{_EVAL_JOBS}/{{name}}")
@abstractmethod
def _raw_delete_evaluate_job(
    *, workspace: str | None = None, name: str, query_params: JobSelectorQueryParams | None = None
) -> None: ...


def delete_evaluate_job(
    *,
    workspace: str | None = None,
    name: str,
    query_params: JobSelectorQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[None]:
    return _raw_delete_evaluate_job(
        workspace=workspace,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, job_reference_kind),
    )


def delete_evaluate_job_by_id(
    *, workspace: str | None = None, job_id: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[None]:
    return _raw_delete_evaluate_job(
        workspace=workspace,
        name=job_id,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.ID),
    )


def delete_evaluate_job_by_name(
    *, workspace: str | None = None, job_name: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[None]:
    return _raw_delete_evaluate_job(
        workspace=workspace,
        name=job_name,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.NAME),
    )


@post(f"{_EVAL_JOBS}/{{name}}/cancel")
@abstractmethod
def _raw_cancel_evaluate_job(
    *, workspace: str | None = None, name: str, query_params: JobSelectorQueryParams | None = None
) -> EvaluateJob: ...


def cancel_evaluate_job(
    *,
    workspace: str | None = None,
    name: str,
    query_params: JobSelectorQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[EvaluateJob]:
    return _raw_cancel_evaluate_job(
        workspace=workspace,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, job_reference_kind),
    )


def cancel_evaluate_job_by_id(
    *, workspace: str | None = None, job_id: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[EvaluateJob]:
    return _raw_cancel_evaluate_job(
        workspace=workspace,
        name=job_id,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.ID),
    )


def cancel_evaluate_job_by_name(
    *, workspace: str | None = None, job_name: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[EvaluateJob]:
    return _raw_cancel_evaluate_job(
        workspace=workspace,
        name=job_name,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.NAME),
    )


@get(f"{_EVAL_JOBS}/{{name}}/logs")
@abstractmethod
def _raw_list_evaluate_job_logs(
    *, workspace: str | None = None, name: str, query_params: EvaluatorJobLogsQueryParams | None = None
) -> Paginated[HelixJobLog, CursorPagination]: ...


def list_evaluate_job_logs(
    *,
    workspace: str | None = None,
    name: str,
    query_params: EvaluatorJobLogsQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[Paginated[HelixJobLog, CursorPagination]]:
    return _raw_list_evaluate_job_logs(
        workspace=workspace,
        name=name,
        query_params=_compose_evaluator_job_logs_query_params(query_params, job_reference_kind),
    )


def list_evaluate_job_logs_by_id(
    *, workspace: str | None = None, job_id: str, query_params: EvaluatorJobLogsQueryParams | None = None
) -> PreparedRequest[Paginated[HelixJobLog, CursorPagination]]:
    return _raw_list_evaluate_job_logs(
        workspace=workspace,
        name=job_id,
        query_params=_compose_evaluator_job_logs_query_params(query_params, JobReferenceKind.ID),
    )


def list_evaluate_job_logs_by_name(
    *, workspace: str | None = None, job_name: str, query_params: EvaluatorJobLogsQueryParams | None = None
) -> PreparedRequest[Paginated[HelixJobLog, CursorPagination]]:
    return _raw_list_evaluate_job_logs(
        workspace=workspace,
        name=job_name,
        query_params=_compose_evaluator_job_logs_query_params(query_params, JobReferenceKind.NAME),
    )


@get(f"{_EVAL_JOBS}/{{name}}/results")
@abstractmethod
def _raw_list_evaluate_job_results(
    *, workspace: str | None = None, name: str, query_params: JobSelectorQueryParams | None = None
) -> HelixJobListResultResponse: ...


def list_evaluate_job_results(
    *,
    workspace: str | None = None,
    name: str,
    query_params: JobSelectorQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[HelixJobListResultResponse]:
    return _raw_list_evaluate_job_results(
        workspace=workspace,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, job_reference_kind),
    )


def list_evaluate_job_results_by_id(
    *, workspace: str | None = None, job_id: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[HelixJobListResultResponse]:
    return _raw_list_evaluate_job_results(
        workspace=workspace,
        name=job_id,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.ID),
    )


def list_evaluate_job_results_by_name(
    *, workspace: str | None = None, job_name: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[HelixJobListResultResponse]:
    return _raw_list_evaluate_job_results(
        workspace=workspace,
        name=job_name,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.NAME),
    )


@get(f"{_EVAL_JOBS}/{{job}}/results/{{name}}")
@abstractmethod
def _raw_get_evaluate_job_result(
    *, workspace: str | None = None, job: str, name: str, query_params: JobSelectorQueryParams | None = None
) -> HelixJobResultResponse: ...


def get_evaluate_job_result(
    *,
    workspace: str | None = None,
    job: str,
    name: str,
    query_params: JobSelectorQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[HelixJobResultResponse]:
    return _raw_get_evaluate_job_result(
        workspace=workspace,
        job=job,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, job_reference_kind),
    )


def get_evaluate_job_result_by_id(
    *, workspace: str | None = None, job_id: str, name: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[HelixJobResultResponse]:
    return _raw_get_evaluate_job_result(
        workspace=workspace,
        job=job_id,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.ID),
    )


def get_evaluate_job_result_by_name(
    *, workspace: str | None = None, job_name: str, name: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[HelixJobResultResponse]:
    return _raw_get_evaluate_job_result(
        workspace=workspace,
        job=job_name,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.NAME),
    )


@get(f"{_EVAL_JOBS}/{{job}}/results/{{name}}/download")
@abstractmethod
def _raw_download_evaluate_job_result(
    *, workspace: str | None = None, job: str, name: str, query_params: JobSelectorQueryParams | None = None
) -> BinaryContent: ...


def download_evaluate_job_result(
    *,
    workspace: str | None = None,
    job: str,
    name: str,
    query_params: JobSelectorQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[BinaryContent]:
    return _raw_download_evaluate_job_result(
        workspace=workspace,
        job=job,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, job_reference_kind),
    )


def download_evaluate_job_result_by_id(
    *, workspace: str | None = None, job_id: str, name: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[BinaryContent]:
    return _raw_download_evaluate_job_result(
        workspace=workspace,
        job=job_id,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.ID),
    )


def download_evaluate_job_result_by_name(
    *, workspace: str | None = None, job_name: str, name: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[BinaryContent]:
    return _raw_download_evaluate_job_result(
        workspace=workspace,
        job=job_name,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.NAME),
    )


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
def _raw_get_agent_eval_job(
    *, workspace: str | None = None, name: str, query_params: JobSelectorQueryParams | None = None
) -> AgentEvalJob: ...


def get_agent_eval_job(
    *,
    workspace: str | None = None,
    name: str,
    query_params: JobSelectorQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[AgentEvalJob]:
    return _raw_get_agent_eval_job(
        workspace=workspace,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, job_reference_kind),
    )


def get_agent_eval_job_by_id(
    *, workspace: str | None = None, job_id: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[AgentEvalJob]:
    return _raw_get_agent_eval_job(
        workspace=workspace,
        name=job_id,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.ID),
    )


def get_agent_eval_job_by_name(
    *, workspace: str | None = None, job_name: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[AgentEvalJob]:
    return _raw_get_agent_eval_job(
        workspace=workspace,
        name=job_name,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.NAME),
    )


@get(f"{_AGENT_EVAL_JOBS}/{{name}}/status")
@abstractmethod
def _raw_get_agent_eval_job_status(
    *, workspace: str | None = None, name: str, query_params: JobSelectorQueryParams | None = None
) -> HelixJobStatusResponse: ...


def get_agent_eval_job_status(
    *,
    workspace: str | None = None,
    name: str,
    query_params: JobSelectorQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[HelixJobStatusResponse]:
    return _raw_get_agent_eval_job_status(
        workspace=workspace,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, job_reference_kind),
    )


def get_agent_eval_job_status_by_id(
    *, workspace: str | None = None, job_id: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[HelixJobStatusResponse]:
    return _raw_get_agent_eval_job_status(
        workspace=workspace,
        name=job_id,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.ID),
    )


def get_agent_eval_job_status_by_name(
    *, workspace: str | None = None, job_name: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[HelixJobStatusResponse]:
    return _raw_get_agent_eval_job_status(
        workspace=workspace,
        name=job_name,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.NAME),
    )


@delete(f"{_AGENT_EVAL_JOBS}/{{name}}")
@abstractmethod
def _raw_delete_agent_eval_job(
    *, workspace: str | None = None, name: str, query_params: JobSelectorQueryParams | None = None
) -> None: ...


def delete_agent_eval_job(
    *,
    workspace: str | None = None,
    name: str,
    query_params: JobSelectorQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[None]:
    return _raw_delete_agent_eval_job(
        workspace=workspace,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, job_reference_kind),
    )


def delete_agent_eval_job_by_id(
    *, workspace: str | None = None, job_id: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[None]:
    return _raw_delete_agent_eval_job(
        workspace=workspace,
        name=job_id,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.ID),
    )


def delete_agent_eval_job_by_name(
    *, workspace: str | None = None, job_name: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[None]:
    return _raw_delete_agent_eval_job(
        workspace=workspace,
        name=job_name,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.NAME),
    )


@post(f"{_AGENT_EVAL_JOBS}/{{name}}/cancel")
@abstractmethod
def _raw_cancel_agent_eval_job(
    *, workspace: str | None = None, name: str, query_params: JobSelectorQueryParams | None = None
) -> AgentEvalJob: ...


def cancel_agent_eval_job(
    *,
    workspace: str | None = None,
    name: str,
    query_params: JobSelectorQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[AgentEvalJob]:
    return _raw_cancel_agent_eval_job(
        workspace=workspace,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, job_reference_kind),
    )


def cancel_agent_eval_job_by_id(
    *, workspace: str | None = None, job_id: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[AgentEvalJob]:
    return _raw_cancel_agent_eval_job(
        workspace=workspace,
        name=job_id,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.ID),
    )


def cancel_agent_eval_job_by_name(
    *, workspace: str | None = None, job_name: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[AgentEvalJob]:
    return _raw_cancel_agent_eval_job(
        workspace=workspace,
        name=job_name,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.NAME),
    )


@get(f"{_AGENT_EVAL_JOBS}/{{name}}/logs")
@abstractmethod
def _raw_list_agent_eval_job_logs(
    *, workspace: str | None = None, name: str, query_params: EvaluatorJobLogsQueryParams | None = None
) -> Paginated[HelixJobLog, CursorPagination]: ...


def list_agent_eval_job_logs(
    *,
    workspace: str | None = None,
    name: str,
    query_params: EvaluatorJobLogsQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[Paginated[HelixJobLog, CursorPagination]]:
    return _raw_list_agent_eval_job_logs(
        workspace=workspace,
        name=name,
        query_params=_compose_evaluator_job_logs_query_params(query_params, job_reference_kind),
    )


def list_agent_eval_job_logs_by_id(
    *, workspace: str | None = None, job_id: str, query_params: EvaluatorJobLogsQueryParams | None = None
) -> PreparedRequest[Paginated[HelixJobLog, CursorPagination]]:
    return _raw_list_agent_eval_job_logs(
        workspace=workspace,
        name=job_id,
        query_params=_compose_evaluator_job_logs_query_params(query_params, JobReferenceKind.ID),
    )


def list_agent_eval_job_logs_by_name(
    *, workspace: str | None = None, job_name: str, query_params: EvaluatorJobLogsQueryParams | None = None
) -> PreparedRequest[Paginated[HelixJobLog, CursorPagination]]:
    return _raw_list_agent_eval_job_logs(
        workspace=workspace,
        name=job_name,
        query_params=_compose_evaluator_job_logs_query_params(query_params, JobReferenceKind.NAME),
    )


@get(f"{_AGENT_EVAL_JOBS}/{{name}}/results")
@abstractmethod
def _raw_list_agent_eval_job_results(
    *, workspace: str | None = None, name: str, query_params: JobSelectorQueryParams | None = None
) -> HelixJobListResultResponse: ...


def list_agent_eval_job_results(
    *,
    workspace: str | None = None,
    name: str,
    query_params: JobSelectorQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[HelixJobListResultResponse]:
    return _raw_list_agent_eval_job_results(
        workspace=workspace,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, job_reference_kind),
    )


def list_agent_eval_job_results_by_id(
    *, workspace: str | None = None, job_id: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[HelixJobListResultResponse]:
    return _raw_list_agent_eval_job_results(
        workspace=workspace,
        name=job_id,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.ID),
    )


def list_agent_eval_job_results_by_name(
    *, workspace: str | None = None, job_name: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[HelixJobListResultResponse]:
    return _raw_list_agent_eval_job_results(
        workspace=workspace,
        name=job_name,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.NAME),
    )


@get(f"{_AGENT_EVAL_JOBS}/{{job}}/results/{{name}}")
@abstractmethod
def _raw_get_agent_eval_job_result(
    *, workspace: str | None = None, job: str, name: str, query_params: JobSelectorQueryParams | None = None
) -> HelixJobResultResponse: ...


def get_agent_eval_job_result(
    *,
    workspace: str | None = None,
    job: str,
    name: str,
    query_params: JobSelectorQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[HelixJobResultResponse]:
    return _raw_get_agent_eval_job_result(
        workspace=workspace,
        job=job,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, job_reference_kind),
    )


def get_agent_eval_job_result_by_id(
    *, workspace: str | None = None, job_id: str, name: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[HelixJobResultResponse]:
    return _raw_get_agent_eval_job_result(
        workspace=workspace,
        job=job_id,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.ID),
    )


def get_agent_eval_job_result_by_name(
    *, workspace: str | None = None, job_name: str, name: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[HelixJobResultResponse]:
    return _raw_get_agent_eval_job_result(
        workspace=workspace,
        job=job_name,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.NAME),
    )


@get(f"{_AGENT_EVAL_JOBS}/{{job}}/results/{{name}}/download")
@abstractmethod
def _raw_download_agent_eval_job_result(
    *, workspace: str | None = None, job: str, name: str, query_params: JobSelectorQueryParams | None = None
) -> BinaryContent: ...


def download_agent_eval_job_result(
    *,
    workspace: str | None = None,
    job: str,
    name: str,
    query_params: JobSelectorQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[BinaryContent]:
    return _raw_download_agent_eval_job_result(
        workspace=workspace,
        job=job,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, job_reference_kind),
    )


def download_agent_eval_job_result_by_id(
    *, workspace: str | None = None, job_id: str, name: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[BinaryContent]:
    return _raw_download_agent_eval_job_result(
        workspace=workspace,
        job=job_id,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.ID),
    )


def download_agent_eval_job_result_by_name(
    *, workspace: str | None = None, job_name: str, name: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[BinaryContent]:
    return _raw_download_agent_eval_job_result(
        workspace=workspace,
        job=job_name,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.NAME),
    )


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
def _raw_get_retrieve_eval_job(
    *, workspace: str | None = None, name: str, query_params: JobSelectorQueryParams | None = None
) -> RetrieveEvalJob: ...


def get_retrieve_eval_job(
    *,
    workspace: str | None = None,
    name: str,
    query_params: JobSelectorQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[RetrieveEvalJob]:
    return _raw_get_retrieve_eval_job(
        workspace=workspace,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, job_reference_kind),
    )


def get_retrieve_eval_job_by_id(
    *, workspace: str | None = None, job_id: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[RetrieveEvalJob]:
    return _raw_get_retrieve_eval_job(
        workspace=workspace,
        name=job_id,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.ID),
    )


def get_retrieve_eval_job_by_name(
    *, workspace: str | None = None, job_name: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[RetrieveEvalJob]:
    return _raw_get_retrieve_eval_job(
        workspace=workspace,
        name=job_name,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.NAME),
    )


@get(f"{_RETRIEVE_EVAL_JOBS}/{{name}}/status")
@abstractmethod
def _raw_get_retrieve_eval_job_status(
    *, workspace: str | None = None, name: str, query_params: JobSelectorQueryParams | None = None
) -> HelixJobStatusResponse: ...


def get_retrieve_eval_job_status(
    *,
    workspace: str | None = None,
    name: str,
    query_params: JobSelectorQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[HelixJobStatusResponse]:
    return _raw_get_retrieve_eval_job_status(
        workspace=workspace,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, job_reference_kind),
    )


def get_retrieve_eval_job_status_by_id(
    *, workspace: str | None = None, job_id: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[HelixJobStatusResponse]:
    return _raw_get_retrieve_eval_job_status(
        workspace=workspace,
        name=job_id,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.ID),
    )


def get_retrieve_eval_job_status_by_name(
    *, workspace: str | None = None, job_name: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[HelixJobStatusResponse]:
    return _raw_get_retrieve_eval_job_status(
        workspace=workspace,
        name=job_name,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.NAME),
    )


@delete(f"{_RETRIEVE_EVAL_JOBS}/{{name}}")
@abstractmethod
def _raw_delete_retrieve_eval_job(
    *, workspace: str | None = None, name: str, query_params: JobSelectorQueryParams | None = None
) -> None: ...


def delete_retrieve_eval_job(
    *,
    workspace: str | None = None,
    name: str,
    query_params: JobSelectorQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[None]:
    return _raw_delete_retrieve_eval_job(
        workspace=workspace,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, job_reference_kind),
    )


def delete_retrieve_eval_job_by_id(
    *, workspace: str | None = None, job_id: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[None]:
    return _raw_delete_retrieve_eval_job(
        workspace=workspace,
        name=job_id,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.ID),
    )


def delete_retrieve_eval_job_by_name(
    *, workspace: str | None = None, job_name: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[None]:
    return _raw_delete_retrieve_eval_job(
        workspace=workspace,
        name=job_name,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.NAME),
    )


@post(f"{_RETRIEVE_EVAL_JOBS}/{{name}}/cancel")
@abstractmethod
def _raw_cancel_retrieve_eval_job(
    *, workspace: str | None = None, name: str, query_params: JobSelectorQueryParams | None = None
) -> RetrieveEvalJob: ...


def cancel_retrieve_eval_job(
    *,
    workspace: str | None = None,
    name: str,
    query_params: JobSelectorQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[RetrieveEvalJob]:
    return _raw_cancel_retrieve_eval_job(
        workspace=workspace,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, job_reference_kind),
    )


def cancel_retrieve_eval_job_by_id(
    *, workspace: str | None = None, job_id: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[RetrieveEvalJob]:
    return _raw_cancel_retrieve_eval_job(
        workspace=workspace,
        name=job_id,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.ID),
    )


def cancel_retrieve_eval_job_by_name(
    *, workspace: str | None = None, job_name: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[RetrieveEvalJob]:
    return _raw_cancel_retrieve_eval_job(
        workspace=workspace,
        name=job_name,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.NAME),
    )


@get(f"{_RETRIEVE_EVAL_JOBS}/{{name}}/logs")
@abstractmethod
def _raw_list_retrieve_eval_job_logs(
    *, workspace: str | None = None, name: str, query_params: EvaluatorJobLogsQueryParams | None = None
) -> Paginated[HelixJobLog, CursorPagination]: ...


def list_retrieve_eval_job_logs(
    *,
    workspace: str | None = None,
    name: str,
    query_params: EvaluatorJobLogsQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[Paginated[HelixJobLog, CursorPagination]]:
    return _raw_list_retrieve_eval_job_logs(
        workspace=workspace,
        name=name,
        query_params=_compose_evaluator_job_logs_query_params(query_params, job_reference_kind),
    )


def list_retrieve_eval_job_logs_by_id(
    *, workspace: str | None = None, job_id: str, query_params: EvaluatorJobLogsQueryParams | None = None
) -> PreparedRequest[Paginated[HelixJobLog, CursorPagination]]:
    return _raw_list_retrieve_eval_job_logs(
        workspace=workspace,
        name=job_id,
        query_params=_compose_evaluator_job_logs_query_params(query_params, JobReferenceKind.ID),
    )


def list_retrieve_eval_job_logs_by_name(
    *, workspace: str | None = None, job_name: str, query_params: EvaluatorJobLogsQueryParams | None = None
) -> PreparedRequest[Paginated[HelixJobLog, CursorPagination]]:
    return _raw_list_retrieve_eval_job_logs(
        workspace=workspace,
        name=job_name,
        query_params=_compose_evaluator_job_logs_query_params(query_params, JobReferenceKind.NAME),
    )


@get(f"{_RETRIEVE_EVAL_JOBS}/{{name}}/results")
@abstractmethod
def _raw_list_retrieve_eval_job_results(
    *, workspace: str | None = None, name: str, query_params: JobSelectorQueryParams | None = None
) -> HelixJobListResultResponse: ...


def list_retrieve_eval_job_results(
    *,
    workspace: str | None = None,
    name: str,
    query_params: JobSelectorQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[HelixJobListResultResponse]:
    return _raw_list_retrieve_eval_job_results(
        workspace=workspace,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, job_reference_kind),
    )


def list_retrieve_eval_job_results_by_id(
    *, workspace: str | None = None, job_id: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[HelixJobListResultResponse]:
    return _raw_list_retrieve_eval_job_results(
        workspace=workspace,
        name=job_id,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.ID),
    )


def list_retrieve_eval_job_results_by_name(
    *, workspace: str | None = None, job_name: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[HelixJobListResultResponse]:
    return _raw_list_retrieve_eval_job_results(
        workspace=workspace,
        name=job_name,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.NAME),
    )


@get(f"{_RETRIEVE_EVAL_JOBS}/{{job}}/results/{{name}}")
@abstractmethod
def _raw_get_retrieve_eval_job_result(
    *, workspace: str | None = None, job: str, name: str, query_params: JobSelectorQueryParams | None = None
) -> HelixJobResultResponse: ...


def get_retrieve_eval_job_result(
    *,
    workspace: str | None = None,
    job: str,
    name: str,
    query_params: JobSelectorQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[HelixJobResultResponse]:
    return _raw_get_retrieve_eval_job_result(
        workspace=workspace,
        job=job,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, job_reference_kind),
    )


def get_retrieve_eval_job_result_by_id(
    *, workspace: str | None = None, job_id: str, name: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[HelixJobResultResponse]:
    return _raw_get_retrieve_eval_job_result(
        workspace=workspace,
        job=job_id,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.ID),
    )


def get_retrieve_eval_job_result_by_name(
    *, workspace: str | None = None, job_name: str, name: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[HelixJobResultResponse]:
    return _raw_get_retrieve_eval_job_result(
        workspace=workspace,
        job=job_name,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.NAME),
    )


@get(f"{_RETRIEVE_EVAL_JOBS}/{{job}}/results/{{name}}/download")
@abstractmethod
def _raw_download_retrieve_eval_job_result(
    *, workspace: str | None = None, job: str, name: str, query_params: JobSelectorQueryParams | None = None
) -> BinaryContent: ...


def download_retrieve_eval_job_result(
    *,
    workspace: str | None = None,
    job: str,
    name: str,
    query_params: JobSelectorQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[BinaryContent]:
    return _raw_download_retrieve_eval_job_result(
        workspace=workspace,
        job=job,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, job_reference_kind),
    )


def download_retrieve_eval_job_result_by_id(
    *, workspace: str | None = None, job_id: str, name: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[BinaryContent]:
    return _raw_download_retrieve_eval_job_result(
        workspace=workspace,
        job=job_id,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.ID),
    )


def download_retrieve_eval_job_result_by_name(
    *, workspace: str | None = None, job_name: str, name: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[BinaryContent]:
    return _raw_download_retrieve_eval_job_result(
        workspace=workspace,
        job=job_name,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.NAME),
    )


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


@post(f"{_TASKS}/{{name}}/revisions/{{revision}}/image-builds")
@abstractmethod
def create_task_image_build(
    *, workspace: str | None = None, name: str, revision: str, body: CreateTaskImageBuildRequest
) -> TaskImageBuild: ...


@get(f"{_TASKS}/{{name}}/revisions/{{revision}}/image-builds")
@abstractmethod
def list_task_image_builds(
    *,
    workspace: str | None = None,
    name: str,
    revision: str,
    query_params: ListTaskImageBuildsQueryParams | FlatQueryParams | None = None,
) -> Paginated[TaskImageBuild]: ...


@get(f"{_TASKS}/{{name}}/revisions/{{revision}}/image-builds/{{build_id}}")
@abstractmethod
def get_task_image_build(
    *, workspace: str | None = None, name: str, revision: str, build_id: str
) -> TaskImageBuild: ...


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


def _compose_job_selector_query_params(
    query: JobSelectorQueryParams | None, mode: JobReferenceKind | None
) -> JobSelectorQueryParams:
    result: JobSelectorQueryParams = {} if query is None else query.copy()
    existing = result.pop("job_reference_kind", None)
    existing_mode = JobReferenceKind(existing) if existing is not None else None
    selected_mode = JobReferenceKind(mode) if mode is not None else None
    if existing_mode is not None and selected_mode is not None and existing_mode != selected_mode:
        raise ValueError("Conflicting Job reference modes")
    selected_mode = selected_mode if selected_mode is not None else existing_mode
    if selected_mode is not None:
        result["job_reference_kind"] = selected_mode
    return result


def _compose_evaluator_job_logs_query_params(
    query: EvaluatorJobLogsQueryParams | None, mode: JobReferenceKind | None
) -> EvaluatorJobLogsQueryParams:
    result: EvaluatorJobLogsQueryParams = {} if query is None else query.copy()
    existing = result.pop("job_reference_kind", None)
    existing_mode = JobReferenceKind(existing) if existing is not None else None
    selected_mode = JobReferenceKind(mode) if mode is not None else None
    if existing_mode is not None and selected_mode is not None and existing_mode != selected_mode:
        raise ValueError("Conflicting Job reference modes")
    selected_mode = selected_mode if selected_mode is not None else existing_mode
    if selected_mode is not None:
        result["job_reference_kind"] = selected_mode
    return result
