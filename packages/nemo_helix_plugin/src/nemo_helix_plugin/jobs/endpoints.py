# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed endpoint definitions for the Jobs service.

These are the single source of truth for the HTTP contract.  Each function
is decorated with an HTTP-method decorator and ``@abstractmethod``; the
decorator turns the signature into a :class:`PreparedRequest` builder.
"""

from __future__ import annotations

from abc import abstractmethod

from nemo_helix_plugin.client.endpoint import delete, get, patch, post, put
from nemo_helix_plugin.client.types import BinaryContent, CursorPagination, Paginated, PreparedRequest
from nemo_helix_plugin.jobs.execution_profiles import (
    DockerJobExecutionProfile,
    E2EJobExecutionProfile,
    KubernetesJobExecutionProfile,
    OpenShellJobExecutionProfile,
    SubprocessJobExecutionProfile,
    VolcanoJobExecutionProfile,
)
from nemo_helix_plugin.jobs.schemas import (
    HelixJobLog,
    HelixJobResultCreateRequest,
    HelixJobResultResponse,
    HelixJobStatusResponse,
)
from nemo_helix_plugin.jobs.selectors import JobMetadataReferenceKind, JobReferenceKind
from nemo_helix_plugin.jobs.types import (
    CreateHelixJobRequest,
    HelixJobAttemptResponse,
    HelixJobListResultResponse,
    HelixJobListTaskResponse,
    HelixJobResponse,
    HelixJobStatusUpdateRequest,
    HelixJobStepResponse,
    HelixJobStepWithContext,
    HelixJobTaskResponse,
    HelixJobTaskUpdate,
    JobControl,
    JobLogsQueryParams,
    JobMetadataQueryParams,
    JobMetadataResponse,
    JobSelectorQueryParams,
    JobStatusDetailsQueryParams,
    JobStatusDetailsUpdate,
    ListAttemptsQueryParams,
    ListJobResultsQueryParams,
    ListJobsQueryParams,
    ListStepsQueryParams,
    UpdateHelixJobRequest,
)

# The execution-profiles endpoint returns a union over all configured backend
# profile types.
ExecutionProfile = (
    DockerJobExecutionProfile
    | KubernetesJobExecutionProfile
    | VolcanoJobExecutionProfile
    | SubprocessJobExecutionProfile
    | OpenShellJobExecutionProfile
    | E2EJobExecutionProfile
)

# ---------------------------------------------------------------------------
# Execution profiles
# ---------------------------------------------------------------------------


@get("/apis/jobs/v2/execution-profiles")
@abstractmethod
def get_execution_profiles() -> list[ExecutionProfile]: ...


# ---------------------------------------------------------------------------
# Job CRUD + lifecycle
# ---------------------------------------------------------------------------


@post("/apis/jobs/v2/workspaces/{workspace}/jobs")
@abstractmethod
def create_job(*, workspace: str | None = None, body: CreateHelixJobRequest) -> HelixJobResponse: ...


@get("/apis/jobs/v2/workspaces/{workspace}/jobs")
@abstractmethod
def list_jobs(
    *, workspace: str | None = None, query_params: ListJobsQueryParams | None = None
) -> Paginated[HelixJobResponse]: ...


@get("/apis/jobs/v2/workspaces/{workspace}/jobs/{name}")
@abstractmethod
def _raw_get_job(
    *, workspace: str | None = None, name: str, query_params: JobSelectorQueryParams | None = None
) -> HelixJobResponse: ...


def get_job(
    *,
    workspace: str | None = None,
    name: str,
    query_params: JobSelectorQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[HelixJobResponse]:
    return _raw_get_job(
        workspace=workspace,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, job_reference_kind),
    )


def get_job_by_id(
    *, workspace: str | None = None, job_id: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[HelixJobResponse]:
    return _raw_get_job(
        workspace=workspace,
        name=job_id,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.ID),
    )


def get_job_by_name(
    *, workspace: str | None = None, job_name: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[HelixJobResponse]:
    return _raw_get_job(
        workspace=workspace,
        name=job_name,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.NAME),
    )


@patch("/apis/jobs/v2/workspaces/{workspace}/jobs/{name}")
@abstractmethod
def _raw_update_job(
    *,
    workspace: str | None = None,
    name: str,
    body: UpdateHelixJobRequest,
    query_params: JobSelectorQueryParams | None = None,
) -> HelixJobResponse: ...


def update_job(
    *,
    workspace: str | None = None,
    name: str,
    body: UpdateHelixJobRequest,
    query_params: JobSelectorQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[HelixJobResponse]:
    return _raw_update_job(
        workspace=workspace,
        name=name,
        body=body,
        query_params=_compose_job_selector_query_params(query_params, job_reference_kind),
    )


def update_job_by_id(
    *,
    workspace: str | None = None,
    job_id: str,
    body: UpdateHelixJobRequest,
    query_params: JobSelectorQueryParams | None = None,
) -> PreparedRequest[HelixJobResponse]:
    return _raw_update_job(
        workspace=workspace,
        name=job_id,
        body=body,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.ID),
    )


def update_job_by_name(
    *,
    workspace: str | None = None,
    job_name: str,
    body: UpdateHelixJobRequest,
    query_params: JobSelectorQueryParams | None = None,
) -> PreparedRequest[HelixJobResponse]:
    return _raw_update_job(
        workspace=workspace,
        name=job_name,
        body=body,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.NAME),
    )


@delete("/apis/jobs/v2/workspaces/{workspace}/jobs/{name}")
@abstractmethod
def _raw_delete_job(
    *, workspace: str | None = None, name: str, query_params: JobSelectorQueryParams | None = None
) -> None: ...


def delete_job(
    *,
    workspace: str | None = None,
    name: str,
    query_params: JobSelectorQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[None]:
    return _raw_delete_job(
        workspace=workspace,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, job_reference_kind),
    )


def delete_job_by_id(
    *, workspace: str | None = None, job_id: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[None]:
    return _raw_delete_job(
        workspace=workspace,
        name=job_id,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.ID),
    )


def delete_job_by_name(
    *, workspace: str | None = None, job_name: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[None]:
    return _raw_delete_job(
        workspace=workspace,
        name=job_name,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.NAME),
    )


@post("/apis/jobs/v2/workspaces/{workspace}/jobs/{name}/cancel")
@abstractmethod
def _raw_cancel_job(
    *, workspace: str | None = None, name: str, query_params: JobSelectorQueryParams | None = None
) -> HelixJobResponse: ...


def cancel_job(
    *,
    workspace: str | None = None,
    name: str,
    query_params: JobSelectorQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[HelixJobResponse]:
    return _raw_cancel_job(
        workspace=workspace,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, job_reference_kind),
    )


def cancel_job_by_id(
    *, workspace: str | None = None, job_id: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[HelixJobResponse]:
    return _raw_cancel_job(
        workspace=workspace,
        name=job_id,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.ID),
    )


def cancel_job_by_name(
    *, workspace: str | None = None, job_name: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[HelixJobResponse]:
    return _raw_cancel_job(
        workspace=workspace,
        name=job_name,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.NAME),
    )


@post("/apis/jobs/v2/workspaces/{workspace}/jobs/{name}/pause")
@abstractmethod
def _raw_pause_job(
    *, workspace: str | None = None, name: str, query_params: JobSelectorQueryParams | None = None
) -> HelixJobResponse: ...


def pause_job(
    *,
    workspace: str | None = None,
    name: str,
    query_params: JobSelectorQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[HelixJobResponse]:
    return _raw_pause_job(
        workspace=workspace,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, job_reference_kind),
    )


def pause_job_by_id(
    *, workspace: str | None = None, job_id: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[HelixJobResponse]:
    return _raw_pause_job(
        workspace=workspace,
        name=job_id,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.ID),
    )


def pause_job_by_name(
    *, workspace: str | None = None, job_name: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[HelixJobResponse]:
    return _raw_pause_job(
        workspace=workspace,
        name=job_name,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.NAME),
    )


@post("/apis/jobs/v2/workspaces/{workspace}/jobs/{name}/resume")
@abstractmethod
def _raw_resume_job(
    *, workspace: str | None = None, name: str, query_params: JobSelectorQueryParams | None = None
) -> HelixJobResponse: ...


def resume_job(
    *,
    workspace: str | None = None,
    name: str,
    query_params: JobSelectorQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[HelixJobResponse]:
    return _raw_resume_job(
        workspace=workspace,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, job_reference_kind),
    )


def resume_job_by_id(
    *, workspace: str | None = None, job_id: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[HelixJobResponse]:
    return _raw_resume_job(
        workspace=workspace,
        name=job_id,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.ID),
    )


def resume_job_by_name(
    *, workspace: str | None = None, job_name: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[HelixJobResponse]:
    return _raw_resume_job(
        workspace=workspace,
        name=job_name,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.NAME),
    )


@post("/apis/jobs/v2/workspaces/{workspace}/jobs/{job}/rerun")
@abstractmethod
def _raw_rerun_job(
    *, workspace: str | None = None, job: str, query_params: JobSelectorQueryParams | None = None
) -> HelixJobResponse: ...


def rerun_job(
    *,
    workspace: str | None = None,
    job: str,
    query_params: JobSelectorQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[HelixJobResponse]:
    return _raw_rerun_job(
        workspace=workspace, job=job, query_params=_compose_job_selector_query_params(query_params, job_reference_kind)
    )


def rerun_job_by_id(
    *, workspace: str | None = None, job_id: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[HelixJobResponse]:
    return _raw_rerun_job(
        workspace=workspace,
        job=job_id,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.ID),
    )


def rerun_job_by_name(
    *, workspace: str | None = None, job_name: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[HelixJobResponse]:
    return _raw_rerun_job(
        workspace=workspace,
        job=job_name,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.NAME),
    )


@patch("/apis/jobs/v2/workspaces/{workspace}/jobs/{name}/control")
@abstractmethod
def _raw_update_job_control(
    *, workspace: str | None = None, name: str, body: JobControl, query_params: JobSelectorQueryParams | None = None
) -> HelixJobResponse: ...


def update_job_control(
    *,
    workspace: str | None = None,
    name: str,
    body: JobControl,
    query_params: JobSelectorQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[HelixJobResponse]:
    return _raw_update_job_control(
        workspace=workspace,
        name=name,
        body=body,
        query_params=_compose_job_selector_query_params(query_params, job_reference_kind),
    )


def update_job_control_by_id(
    *, workspace: str | None = None, job_id: str, body: JobControl, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[HelixJobResponse]:
    return _raw_update_job_control(
        workspace=workspace,
        name=job_id,
        body=body,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.ID),
    )


def update_job_control_by_name(
    *, workspace: str | None = None, job_name: str, body: JobControl, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[HelixJobResponse]:
    return _raw_update_job_control(
        workspace=workspace,
        name=job_name,
        body=body,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.NAME),
    )


# ---------------------------------------------------------------------------
# Job status
# ---------------------------------------------------------------------------


@get("/apis/jobs/v2/workspaces/{workspace}/jobs/{name}/status")
@abstractmethod
def _raw_get_job_status(
    *, workspace: str | None = None, name: str, query_params: JobSelectorQueryParams | None = None
) -> HelixJobStatusResponse: ...


def get_job_status(
    *,
    workspace: str | None = None,
    name: str,
    query_params: JobSelectorQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[HelixJobStatusResponse]:
    return _raw_get_job_status(
        workspace=workspace,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, job_reference_kind),
    )


def get_job_status_by_id(
    *, workspace: str | None = None, job_id: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[HelixJobStatusResponse]:
    return _raw_get_job_status(
        workspace=workspace,
        name=job_id,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.ID),
    )


def get_job_status_by_name(
    *, workspace: str | None = None, job_name: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[HelixJobStatusResponse]:
    return _raw_get_job_status(
        workspace=workspace,
        name=job_name,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.NAME),
    )


# ---------------------------------------------------------------------------
# Job attempts
# ---------------------------------------------------------------------------


@get("/apis/jobs/v2/workspaces/{workspace}/jobs/{job}/attempts")
@abstractmethod
def _raw_list_job_attempts(
    *, workspace: str | None = None, job: str, query_params: ListAttemptsQueryParams | None = None
) -> Paginated[HelixJobAttemptResponse]: ...


def list_job_attempts(
    *,
    workspace: str | None = None,
    job: str,
    query_params: ListAttemptsQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[Paginated[HelixJobAttemptResponse]]:
    return _raw_list_job_attempts(
        workspace=workspace, job=job, query_params=_compose_list_attempts_query_params(query_params, job_reference_kind)
    )


def list_job_attempts_by_id(
    *, workspace: str | None = None, job_id: str, query_params: ListAttemptsQueryParams | None = None
) -> PreparedRequest[Paginated[HelixJobAttemptResponse]]:
    return _raw_list_job_attempts(
        workspace=workspace,
        job=job_id,
        query_params=_compose_list_attempts_query_params(query_params, JobReferenceKind.ID),
    )


def list_job_attempts_by_name(
    *, workspace: str | None = None, job_name: str, query_params: ListAttemptsQueryParams | None = None
) -> PreparedRequest[Paginated[HelixJobAttemptResponse]]:
    return _raw_list_job_attempts(
        workspace=workspace,
        job=job_name,
        query_params=_compose_list_attempts_query_params(query_params, JobReferenceKind.NAME),
    )


@get("/apis/jobs/v2/workspaces/{workspace}/jobs/{job}/attempts/{attempt}")
@abstractmethod
def _raw_get_job_attempt(
    *, workspace: str | None = None, job: str, attempt: str, query_params: JobSelectorQueryParams | None = None
) -> HelixJobAttemptResponse: ...


def get_job_attempt(
    *,
    workspace: str | None = None,
    job: str,
    attempt: str,
    query_params: JobSelectorQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[HelixJobAttemptResponse]:
    return _raw_get_job_attempt(
        workspace=workspace,
        job=job,
        attempt=attempt,
        query_params=_compose_job_selector_query_params(query_params, job_reference_kind),
    )


def get_job_attempt_by_id(
    *, workspace: str | None = None, job_id: str, attempt: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[HelixJobAttemptResponse]:
    return _raw_get_job_attempt(
        workspace=workspace,
        job=job_id,
        attempt=attempt,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.ID),
    )


def get_job_attempt_by_name(
    *, workspace: str | None = None, job_name: str, attempt: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[HelixJobAttemptResponse]:
    return _raw_get_job_attempt(
        workspace=workspace,
        job=job_name,
        attempt=attempt,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.NAME),
    )


@patch("/apis/jobs/v2/workspaces/{workspace}/jobs/{job}/attempts/{attempt}/status-details")
@abstractmethod
def _raw_update_job_attempt_status_details(
    *,
    workspace: str | None = None,
    job: str,
    attempt: str,
    body: JobStatusDetailsUpdate,
    query_params: JobStatusDetailsQueryParams | None = None,
) -> None: ...


def update_job_attempt_status_details(
    *,
    workspace: str | None = None,
    job: str,
    attempt: str,
    body: JobStatusDetailsUpdate,
    query_params: JobStatusDetailsQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[None]:
    return _raw_update_job_attempt_status_details(
        workspace=workspace,
        job=job,
        attempt=attempt,
        body=body,
        query_params=_compose_job_status_details_query_params(query_params, job_reference_kind),
    )


def update_job_attempt_status_details_by_id(
    *,
    workspace: str | None = None,
    job_id: str,
    attempt: str,
    body: JobStatusDetailsUpdate,
    query_params: JobStatusDetailsQueryParams | None = None,
) -> PreparedRequest[None]:
    return _raw_update_job_attempt_status_details(
        workspace=workspace,
        job=job_id,
        attempt=attempt,
        body=body,
        query_params=_compose_job_status_details_query_params(query_params, JobReferenceKind.ID),
    )


def update_job_attempt_status_details_by_name(
    *,
    workspace: str | None = None,
    job_name: str,
    attempt: str,
    body: JobStatusDetailsUpdate,
    query_params: JobStatusDetailsQueryParams | None = None,
) -> PreparedRequest[None]:
    return _raw_update_job_attempt_status_details(
        workspace=workspace,
        job=job_name,
        attempt=attempt,
        body=body,
        query_params=_compose_job_status_details_query_params(query_params, JobReferenceKind.NAME),
    )


# ---------------------------------------------------------------------------
# Job logs
# ---------------------------------------------------------------------------


@get("/apis/jobs/v2/workspaces/{workspace}/jobs/{name}/logs")
@abstractmethod
def _raw_list_job_logs(
    *, workspace: str | None = None, name: str, query_params: JobLogsQueryParams | None = None
) -> Paginated[HelixJobLog, CursorPagination]: ...


def list_job_logs(
    *,
    workspace: str | None = None,
    name: str,
    query_params: JobLogsQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[Paginated[HelixJobLog, CursorPagination]]:
    return _raw_list_job_logs(
        workspace=workspace, name=name, query_params=_compose_job_logs_query_params(query_params, job_reference_kind)
    )


def list_job_logs_by_id(
    *, workspace: str | None = None, job_id: str, query_params: JobLogsQueryParams | None = None
) -> PreparedRequest[Paginated[HelixJobLog, CursorPagination]]:
    return _raw_list_job_logs(
        workspace=workspace, name=job_id, query_params=_compose_job_logs_query_params(query_params, JobReferenceKind.ID)
    )


def list_job_logs_by_name(
    *, workspace: str | None = None, job_name: str, query_params: JobLogsQueryParams | None = None
) -> PreparedRequest[Paginated[HelixJobLog, CursorPagination]]:
    return _raw_list_job_logs(
        workspace=workspace,
        name=job_name,
        query_params=_compose_job_logs_query_params(query_params, JobReferenceKind.NAME),
    )


# ---------------------------------------------------------------------------
# Job results
# ---------------------------------------------------------------------------


@post("/apis/jobs/v2/workspaces/{workspace}/jobs/{job}/results/{name}")
@abstractmethod
def _raw_create_job_result(
    *,
    workspace: str | None = None,
    job: str,
    name: str,
    body: HelixJobResultCreateRequest,
    query_params: JobSelectorQueryParams | None = None,
) -> HelixJobResultResponse: ...


def create_job_result(
    *,
    workspace: str | None = None,
    job: str,
    name: str,
    body: HelixJobResultCreateRequest,
    query_params: JobSelectorQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[HelixJobResultResponse]:
    return _raw_create_job_result(
        workspace=workspace,
        job=job,
        name=name,
        body=body,
        query_params=_compose_job_selector_query_params(query_params, job_reference_kind),
    )


def create_job_result_by_id(
    *,
    workspace: str | None = None,
    job_id: str,
    name: str,
    body: HelixJobResultCreateRequest,
    query_params: JobSelectorQueryParams | None = None,
) -> PreparedRequest[HelixJobResultResponse]:
    return _raw_create_job_result(
        workspace=workspace,
        job=job_id,
        name=name,
        body=body,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.ID),
    )


def create_job_result_by_name(
    *,
    workspace: str | None = None,
    job_name: str,
    name: str,
    body: HelixJobResultCreateRequest,
    query_params: JobSelectorQueryParams | None = None,
) -> PreparedRequest[HelixJobResultResponse]:
    return _raw_create_job_result(
        workspace=workspace,
        job=job_name,
        name=name,
        body=body,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.NAME),
    )


@get("/apis/jobs/v2/workspaces/{workspace}/jobs/{name}/results")
@abstractmethod
def _raw_list_job_results(
    *, workspace: str | None = None, name: str, query_params: ListJobResultsQueryParams | None = None
) -> HelixJobListResultResponse: ...


def list_job_results(
    *,
    workspace: str | None = None,
    name: str,
    query_params: ListJobResultsQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[HelixJobListResultResponse]:
    return _raw_list_job_results(
        workspace=workspace,
        name=name,
        query_params=_compose_list_job_results_query_params(query_params, job_reference_kind),
    )


def list_job_results_by_id(
    *, workspace: str | None = None, job_id: str, query_params: ListJobResultsQueryParams | None = None
) -> PreparedRequest[HelixJobListResultResponse]:
    return _raw_list_job_results(
        workspace=workspace,
        name=job_id,
        query_params=_compose_list_job_results_query_params(query_params, JobReferenceKind.ID),
    )


def list_job_results_by_name(
    *, workspace: str | None = None, job_name: str, query_params: ListJobResultsQueryParams | None = None
) -> PreparedRequest[HelixJobListResultResponse]:
    return _raw_list_job_results(
        workspace=workspace,
        name=job_name,
        query_params=_compose_list_job_results_query_params(query_params, JobReferenceKind.NAME),
    )


@get("/apis/jobs/v2/workspaces/{workspace}/jobs/{job}/results/{name}")
@abstractmethod
def _raw_get_job_result(
    *, workspace: str | None = None, job: str, name: str, query_params: JobSelectorQueryParams | None = None
) -> HelixJobResultResponse: ...


def get_job_result(
    *,
    workspace: str | None = None,
    job: str,
    name: str,
    query_params: JobSelectorQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[HelixJobResultResponse]:
    return _raw_get_job_result(
        workspace=workspace,
        job=job,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, job_reference_kind),
    )


def get_job_result_by_id(
    *, workspace: str | None = None, job_id: str, name: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[HelixJobResultResponse]:
    return _raw_get_job_result(
        workspace=workspace,
        job=job_id,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.ID),
    )


def get_job_result_by_name(
    *, workspace: str | None = None, job_name: str, name: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[HelixJobResultResponse]:
    return _raw_get_job_result(
        workspace=workspace,
        job=job_name,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.NAME),
    )


@get("/apis/jobs/v2/workspaces/{workspace}/jobs/{job}/results/{name}/download")
@abstractmethod
def _raw_download_job_result(
    *, workspace: str | None = None, job: str, name: str, query_params: JobSelectorQueryParams | None = None
) -> BinaryContent: ...


def download_job_result(
    *,
    workspace: str | None = None,
    job: str,
    name: str,
    query_params: JobSelectorQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[BinaryContent]:
    return _raw_download_job_result(
        workspace=workspace,
        job=job,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, job_reference_kind),
    )


def download_job_result_by_id(
    *, workspace: str | None = None, job_id: str, name: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[BinaryContent]:
    return _raw_download_job_result(
        workspace=workspace,
        job=job_id,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.ID),
    )


def download_job_result_by_name(
    *, workspace: str | None = None, job_name: str, name: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[BinaryContent]:
    return _raw_download_job_result(
        workspace=workspace,
        job=job_name,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.NAME),
    )


# ---------------------------------------------------------------------------
# Job steps
# ---------------------------------------------------------------------------


@get("/apis/jobs/v2/workspaces/{workspace}/jobs/{name}/steps")
@abstractmethod
def _raw_list_steps(
    *, workspace: str | None = None, name: str, query_params: ListStepsQueryParams | None = None
) -> Paginated[HelixJobStepWithContext]: ...


def list_steps(
    *,
    workspace: str | None = None,
    name: str,
    query_params: ListStepsQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[Paginated[HelixJobStepWithContext]]:
    return _raw_list_steps(
        workspace=workspace, name=name, query_params=_compose_list_steps_query_params(query_params, job_reference_kind)
    )


def list_steps_by_id(
    *, workspace: str | None = None, job_id: str, query_params: ListStepsQueryParams | None = None
) -> PreparedRequest[Paginated[HelixJobStepWithContext]]:
    return _raw_list_steps(
        workspace=workspace,
        name=job_id,
        query_params=_compose_list_steps_query_params(query_params, JobReferenceKind.ID),
    )


def list_steps_by_name(
    *, workspace: str | None = None, job_name: str, query_params: ListStepsQueryParams | None = None
) -> PreparedRequest[Paginated[HelixJobStepWithContext]]:
    return _raw_list_steps(
        workspace=workspace,
        name=job_name,
        query_params=_compose_list_steps_query_params(query_params, JobReferenceKind.NAME),
    )


@get("/apis/jobs/v2/workspaces/{workspace}/jobs/{job}/steps/{name}")
@abstractmethod
def _raw_get_job_step(
    *, workspace: str | None = None, job: str, name: str, query_params: JobSelectorQueryParams | None = None
) -> HelixJobStepResponse: ...


def get_job_step(
    *,
    workspace: str | None = None,
    job: str,
    name: str,
    query_params: JobSelectorQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[HelixJobStepResponse]:
    return _raw_get_job_step(
        workspace=workspace,
        job=job,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, job_reference_kind),
    )


def get_job_step_by_id(
    *, workspace: str | None = None, job_id: str, name: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[HelixJobStepResponse]:
    return _raw_get_job_step(
        workspace=workspace,
        job=job_id,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.ID),
    )


def get_job_step_by_name(
    *, workspace: str | None = None, job_name: str, name: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[HelixJobStepResponse]:
    return _raw_get_job_step(
        workspace=workspace,
        job=job_name,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.NAME),
    )


@patch("/apis/jobs/v2/workspaces/{workspace}/jobs/{job}/steps/{name}/status")
@abstractmethod
def _raw_update_job_step_status(
    *,
    workspace: str | None = None,
    job: str,
    name: str,
    body: HelixJobStatusUpdateRequest,
    query_params: JobSelectorQueryParams | None = None,
) -> HelixJobStepResponse: ...


def update_job_step_status(
    *,
    workspace: str | None = None,
    job: str,
    name: str,
    body: HelixJobStatusUpdateRequest,
    query_params: JobSelectorQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[HelixJobStepResponse]:
    return _raw_update_job_step_status(
        workspace=workspace,
        job=job,
        name=name,
        body=body,
        query_params=_compose_job_selector_query_params(query_params, job_reference_kind),
    )


def update_job_step_status_by_id(
    *,
    workspace: str | None = None,
    job_id: str,
    name: str,
    body: HelixJobStatusUpdateRequest,
    query_params: JobSelectorQueryParams | None = None,
) -> PreparedRequest[HelixJobStepResponse]:
    return _raw_update_job_step_status(
        workspace=workspace,
        job=job_id,
        name=name,
        body=body,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.ID),
    )


def update_job_step_status_by_name(
    *,
    workspace: str | None = None,
    job_name: str,
    name: str,
    body: HelixJobStatusUpdateRequest,
    query_params: JobSelectorQueryParams | None = None,
) -> PreparedRequest[HelixJobStepResponse]:
    return _raw_update_job_step_status(
        workspace=workspace,
        job=job_name,
        name=name,
        body=body,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.NAME),
    )


@get("/apis/jobs/v2/workspaces/{workspace}/jobs/{job}/attempts/{attempt}/steps")
@abstractmethod
def _raw_list_attempt_steps(
    *, workspace: str | None = None, job: str, attempt: str, query_params: ListStepsQueryParams | None = None
) -> Paginated[HelixJobStepWithContext]: ...


def list_attempt_steps(
    *,
    workspace: str | None = None,
    job: str,
    attempt: str,
    query_params: ListStepsQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[Paginated[HelixJobStepWithContext]]:
    return _raw_list_attempt_steps(
        workspace=workspace,
        job=job,
        attempt=attempt,
        query_params=_compose_list_steps_query_params(query_params, job_reference_kind),
    )


def list_attempt_steps_by_id(
    *, workspace: str | None = None, job_id: str, attempt: str, query_params: ListStepsQueryParams | None = None
) -> PreparedRequest[Paginated[HelixJobStepWithContext]]:
    return _raw_list_attempt_steps(
        workspace=workspace,
        job=job_id,
        attempt=attempt,
        query_params=_compose_list_steps_query_params(query_params, JobReferenceKind.ID),
    )


def list_attempt_steps_by_name(
    *, workspace: str | None = None, job_name: str, attempt: str, query_params: ListStepsQueryParams | None = None
) -> PreparedRequest[Paginated[HelixJobStepWithContext]]:
    return _raw_list_attempt_steps(
        workspace=workspace,
        job=job_name,
        attempt=attempt,
        query_params=_compose_list_steps_query_params(query_params, JobReferenceKind.NAME),
    )


@get("/apis/jobs/v2/workspaces/{workspace}/jobs/{job}/attempts/{attempt}/steps/{name}")
@abstractmethod
def _raw_get_attempt_step(
    *,
    workspace: str | None = None,
    job: str,
    attempt: str,
    name: str,
    query_params: JobSelectorQueryParams | None = None,
) -> HelixJobStepResponse: ...


def get_attempt_step(
    *,
    workspace: str | None = None,
    job: str,
    attempt: str,
    name: str,
    query_params: JobSelectorQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[HelixJobStepResponse]:
    return _raw_get_attempt_step(
        workspace=workspace,
        job=job,
        attempt=attempt,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, job_reference_kind),
    )


def get_attempt_step_by_id(
    *,
    workspace: str | None = None,
    job_id: str,
    attempt: str,
    name: str,
    query_params: JobSelectorQueryParams | None = None,
) -> PreparedRequest[HelixJobStepResponse]:
    return _raw_get_attempt_step(
        workspace=workspace,
        job=job_id,
        attempt=attempt,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.ID),
    )


def get_attempt_step_by_name(
    *,
    workspace: str | None = None,
    job_name: str,
    attempt: str,
    name: str,
    query_params: JobSelectorQueryParams | None = None,
) -> PreparedRequest[HelixJobStepResponse]:
    return _raw_get_attempt_step(
        workspace=workspace,
        job=job_name,
        attempt=attempt,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.NAME),
    )


@patch("/apis/jobs/v2/workspaces/{workspace}/jobs/{job}/attempts/{attempt}/steps/{name}/status")
@abstractmethod
def _raw_update_attempt_step_status(
    *,
    workspace: str | None = None,
    job: str,
    attempt: str,
    name: str,
    body: HelixJobStatusUpdateRequest,
    query_params: JobSelectorQueryParams | None = None,
) -> HelixJobStepResponse: ...


def update_attempt_step_status(
    *,
    workspace: str | None = None,
    job: str,
    attempt: str,
    name: str,
    body: HelixJobStatusUpdateRequest,
    query_params: JobSelectorQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[HelixJobStepResponse]:
    return _raw_update_attempt_step_status(
        workspace=workspace,
        job=job,
        attempt=attempt,
        name=name,
        body=body,
        query_params=_compose_job_selector_query_params(query_params, job_reference_kind),
    )


def update_attempt_step_status_by_id(
    *,
    workspace: str | None = None,
    job_id: str,
    attempt: str,
    name: str,
    body: HelixJobStatusUpdateRequest,
    query_params: JobSelectorQueryParams | None = None,
) -> PreparedRequest[HelixJobStepResponse]:
    return _raw_update_attempt_step_status(
        workspace=workspace,
        job=job_id,
        attempt=attempt,
        name=name,
        body=body,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.ID),
    )


def update_attempt_step_status_by_name(
    *,
    workspace: str | None = None,
    job_name: str,
    attempt: str,
    name: str,
    body: HelixJobStatusUpdateRequest,
    query_params: JobSelectorQueryParams | None = None,
) -> PreparedRequest[HelixJobStepResponse]:
    return _raw_update_attempt_step_status(
        workspace=workspace,
        job=job_name,
        attempt=attempt,
        name=name,
        body=body,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.NAME),
    )


# ---------------------------------------------------------------------------
# Job tasks
# ---------------------------------------------------------------------------


@get("/apis/jobs/v2/workspaces/{workspace}/jobs/{job}/steps/{name}/tasks")
@abstractmethod
def _raw_list_job_step_tasks(
    *, workspace: str | None = None, job: str, name: str, query_params: JobSelectorQueryParams | None = None
) -> HelixJobListTaskResponse: ...


def list_job_step_tasks(
    *,
    workspace: str | None = None,
    job: str,
    name: str,
    query_params: JobSelectorQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[HelixJobListTaskResponse]:
    return _raw_list_job_step_tasks(
        workspace=workspace,
        job=job,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, job_reference_kind),
    )


def list_job_step_tasks_by_id(
    *, workspace: str | None = None, job_id: str, name: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[HelixJobListTaskResponse]:
    return _raw_list_job_step_tasks(
        workspace=workspace,
        job=job_id,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.ID),
    )


def list_job_step_tasks_by_name(
    *, workspace: str | None = None, job_name: str, name: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[HelixJobListTaskResponse]:
    return _raw_list_job_step_tasks(
        workspace=workspace,
        job=job_name,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.NAME),
    )


@get("/apis/jobs/v2/workspaces/{workspace}/jobs/{job}/steps/{step}/tasks/{name}")
@abstractmethod
def _raw_get_job_step_task(
    *, workspace: str | None = None, job: str, step: str, name: str, query_params: JobSelectorQueryParams | None = None
) -> HelixJobTaskResponse: ...


def get_job_step_task(
    *,
    workspace: str | None = None,
    job: str,
    step: str,
    name: str,
    query_params: JobSelectorQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[HelixJobTaskResponse]:
    return _raw_get_job_step_task(
        workspace=workspace,
        job=job,
        step=step,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, job_reference_kind),
    )


def get_job_step_task_by_id(
    *,
    workspace: str | None = None,
    job_id: str,
    step: str,
    name: str,
    query_params: JobSelectorQueryParams | None = None,
) -> PreparedRequest[HelixJobTaskResponse]:
    return _raw_get_job_step_task(
        workspace=workspace,
        job=job_id,
        step=step,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.ID),
    )


def get_job_step_task_by_name(
    *,
    workspace: str | None = None,
    job_name: str,
    step: str,
    name: str,
    query_params: JobSelectorQueryParams | None = None,
) -> PreparedRequest[HelixJobTaskResponse]:
    return _raw_get_job_step_task(
        workspace=workspace,
        job=job_name,
        step=step,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.NAME),
    )


@get("/apis/jobs/v2/workspaces/{workspace}/jobs/{job}/attempts/{attempt}/steps/{step}/tasks")
@abstractmethod
def _raw_list_attempt_step_tasks(
    *,
    workspace: str | None = None,
    job: str,
    attempt: str,
    step: str,
    query_params: JobSelectorQueryParams | None = None,
) -> HelixJobListTaskResponse: ...


def list_attempt_step_tasks(
    *,
    workspace: str | None = None,
    job: str,
    attempt: str,
    step: str,
    query_params: JobSelectorQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[HelixJobListTaskResponse]:
    return _raw_list_attempt_step_tasks(
        workspace=workspace,
        job=job,
        attempt=attempt,
        step=step,
        query_params=_compose_job_selector_query_params(query_params, job_reference_kind),
    )


def list_attempt_step_tasks_by_id(
    *,
    workspace: str | None = None,
    job_id: str,
    attempt: str,
    step: str,
    query_params: JobSelectorQueryParams | None = None,
) -> PreparedRequest[HelixJobListTaskResponse]:
    return _raw_list_attempt_step_tasks(
        workspace=workspace,
        job=job_id,
        attempt=attempt,
        step=step,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.ID),
    )


def list_attempt_step_tasks_by_name(
    *,
    workspace: str | None = None,
    job_name: str,
    attempt: str,
    step: str,
    query_params: JobSelectorQueryParams | None = None,
) -> PreparedRequest[HelixJobListTaskResponse]:
    return _raw_list_attempt_step_tasks(
        workspace=workspace,
        job=job_name,
        attempt=attempt,
        step=step,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.NAME),
    )


@put("/apis/jobs/v2/workspaces/{workspace}/jobs/{job}/attempts/{attempt}/steps/{step}/tasks/{name}")
@abstractmethod
def _raw_update_attempt_step_task(
    *,
    workspace: str | None = None,
    job: str,
    attempt: str,
    step: str,
    name: str,
    body: HelixJobTaskUpdate,
    query_params: JobSelectorQueryParams | None = None,
) -> HelixJobTaskResponse: ...


def update_attempt_step_task(
    *,
    workspace: str | None = None,
    job: str,
    attempt: str,
    step: str,
    name: str,
    body: HelixJobTaskUpdate,
    query_params: JobSelectorQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[HelixJobTaskResponse]:
    return _raw_update_attempt_step_task(
        workspace=workspace,
        job=job,
        attempt=attempt,
        step=step,
        name=name,
        body=body,
        query_params=_compose_job_selector_query_params(query_params, job_reference_kind),
    )


def update_attempt_step_task_by_id(
    *,
    workspace: str | None = None,
    job_id: str,
    attempt: str,
    step: str,
    name: str,
    body: HelixJobTaskUpdate,
    query_params: JobSelectorQueryParams | None = None,
) -> PreparedRequest[HelixJobTaskResponse]:
    return _raw_update_attempt_step_task(
        workspace=workspace,
        job=job_id,
        attempt=attempt,
        step=step,
        name=name,
        body=body,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.ID),
    )


def update_attempt_step_task_by_name(
    *,
    workspace: str | None = None,
    job_name: str,
    attempt: str,
    step: str,
    name: str,
    body: HelixJobTaskUpdate,
    query_params: JobSelectorQueryParams | None = None,
) -> PreparedRequest[HelixJobTaskResponse]:
    return _raw_update_attempt_step_task(
        workspace=workspace,
        job=job_name,
        attempt=attempt,
        step=step,
        name=name,
        body=body,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.NAME),
    )


@get("/apis/jobs/v2/workspaces/{workspace}/jobs/{job}/attempts/{attempt}/steps/{step}/tasks/{name}")
@abstractmethod
def _raw_get_attempt_step_task(
    *,
    workspace: str | None = None,
    job: str,
    attempt: str,
    step: str,
    name: str,
    query_params: JobSelectorQueryParams | None = None,
) -> HelixJobTaskResponse: ...


def get_attempt_step_task(
    *,
    workspace: str | None = None,
    job: str,
    attempt: str,
    step: str,
    name: str,
    query_params: JobSelectorQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[HelixJobTaskResponse]:
    return _raw_get_attempt_step_task(
        workspace=workspace,
        job=job,
        attempt=attempt,
        step=step,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, job_reference_kind),
    )


def get_attempt_step_task_by_id(
    *,
    workspace: str | None = None,
    job_id: str,
    attempt: str,
    step: str,
    name: str,
    query_params: JobSelectorQueryParams | None = None,
) -> PreparedRequest[HelixJobTaskResponse]:
    return _raw_get_attempt_step_task(
        workspace=workspace,
        job=job_id,
        attempt=attempt,
        step=step,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.ID),
    )


def get_attempt_step_task_by_name(
    *,
    workspace: str | None = None,
    job_name: str,
    attempt: str,
    step: str,
    name: str,
    query_params: JobSelectorQueryParams | None = None,
) -> PreparedRequest[HelixJobTaskResponse]:
    return _raw_get_attempt_step_task(
        workspace=workspace,
        job=job_name,
        attempt=attempt,
        step=step,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.NAME),
    )


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


def _compose_list_attempts_query_params(
    query: ListAttemptsQueryParams | None, mode: JobReferenceKind | None
) -> ListAttemptsQueryParams:
    result: ListAttemptsQueryParams = {} if query is None else query.copy()
    existing = result.pop("job_reference_kind", None)
    existing_mode = JobReferenceKind(existing) if existing is not None else None
    selected_mode = JobReferenceKind(mode) if mode is not None else None
    if existing_mode is not None and selected_mode is not None and existing_mode != selected_mode:
        raise ValueError("Conflicting Job reference modes")
    selected_mode = selected_mode if selected_mode is not None else existing_mode
    if selected_mode is not None:
        result["job_reference_kind"] = selected_mode
    return result


def _compose_job_status_details_query_params(
    query: JobStatusDetailsQueryParams | None, mode: JobReferenceKind | None
) -> JobStatusDetailsQueryParams:
    result: JobStatusDetailsQueryParams = {} if query is None else query.copy()
    existing = result.pop("job_reference_kind", None)
    existing_mode = JobReferenceKind(existing) if existing is not None else None
    selected_mode = JobReferenceKind(mode) if mode is not None else None
    if existing_mode is not None and selected_mode is not None and existing_mode != selected_mode:
        raise ValueError("Conflicting Job reference modes")
    selected_mode = selected_mode if selected_mode is not None else existing_mode
    if selected_mode is not None:
        result["job_reference_kind"] = selected_mode
    return result


def _compose_job_logs_query_params(
    query: JobLogsQueryParams | None, mode: JobReferenceKind | None
) -> JobLogsQueryParams:
    result: JobLogsQueryParams = {} if query is None else query.copy()
    existing = result.pop("job_reference_kind", None)
    existing_mode = JobReferenceKind(existing) if existing is not None else None
    selected_mode = JobReferenceKind(mode) if mode is not None else None
    if existing_mode is not None and selected_mode is not None and existing_mode != selected_mode:
        raise ValueError("Conflicting Job reference modes")
    selected_mode = selected_mode if selected_mode is not None else existing_mode
    if selected_mode is not None:
        result["job_reference_kind"] = selected_mode
    return result


def _compose_list_job_results_query_params(
    query: ListJobResultsQueryParams | None, mode: JobReferenceKind | None
) -> ListJobResultsQueryParams:
    result: ListJobResultsQueryParams = {} if query is None else query.copy()
    existing = result.pop("job_reference_kind", None)
    existing_mode = JobReferenceKind(existing) if existing is not None else None
    selected_mode = JobReferenceKind(mode) if mode is not None else None
    if existing_mode is not None and selected_mode is not None and existing_mode != selected_mode:
        raise ValueError("Conflicting Job reference modes")
    selected_mode = selected_mode if selected_mode is not None else existing_mode
    if selected_mode is not None:
        result["job_reference_kind"] = selected_mode
    return result


@get("/apis/jobs/v2/workspaces/{workspace}/jobs/{job}/metadata")
@abstractmethod
def _raw_get_job_metadata(
    *, job: str, query_params: JobMetadataQueryParams, workspace: str | None = None
) -> JobMetadataResponse: ...


def get_job_metadata(
    *, job: str, job_reference_kind: JobMetadataReferenceKind, workspace: str | None = None
) -> PreparedRequest[JobMetadataResponse]:
    return _raw_get_job_metadata(
        job=job, workspace=workspace, query_params={"job_reference_kind": JobMetadataReferenceKind(job_reference_kind)}
    )


def _compose_list_steps_query_params(
    query: ListStepsQueryParams | None, mode: JobReferenceKind | None
) -> ListStepsQueryParams:
    result: ListStepsQueryParams = {} if query is None else query.copy()
    existing = result.pop("job_reference_kind", None)
    existing_mode = JobReferenceKind(existing) if existing is not None else None
    selected_mode = JobReferenceKind(mode) if mode is not None else None
    if existing_mode is not None and selected_mode is not None and existing_mode != selected_mode:
        raise ValueError("Conflicting Job reference modes")
    selected_mode = selected_mode if selected_mode is not None else existing_mode
    if selected_mode is not None:
        result["job_reference_kind"] = selected_mode
    return result
