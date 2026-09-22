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
from nemo_helix_plugin.client.types import BinaryContent, CursorPagination, Paginated
from nemo_helix_plugin.jobs.execution_profiles import (
    DockerJobExecutionProfile,
    E2EJobExecutionProfile,
    KubernetesJobExecutionProfile,
    SubprocessJobExecutionProfile,
    VolcanoJobExecutionProfile,
)
from nemo_helix_plugin.jobs.schemas import (
    HelixJobLog,
    HelixJobResultCreateRequest,
    HelixJobResultResponse,
    HelixJobStatusResponse,
)
from nemo_helix_plugin.jobs.types import (
    CreateHelixJobRequest,
    JobLogsQueryParams,
    JobStatusDetailsUpdate,
    ListJobResultsQueryParams,
    ListJobsQueryParams,
    ListStepsQueryParams,
    HelixJobListResultResponse,
    HelixJobListTaskResponse,
    HelixJobResponse,
    HelixJobStatusUpdateRequest,
    HelixJobStepResponse,
    HelixJobStepWithContext,
    HelixJobTaskResponse,
    HelixJobTaskUpdate,
)

# The execution-profiles endpoint returns a union over all configured backend
# profile types (matches the Stainless ``JobListExecutionProfilesResponseItem``).
ExecutionProfile = (
    DockerJobExecutionProfile
    | KubernetesJobExecutionProfile
    | VolcanoJobExecutionProfile
    | SubprocessJobExecutionProfile
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
def get_job(*, workspace: str | None = None, name: str) -> HelixJobResponse: ...


@delete("/apis/jobs/v2/workspaces/{workspace}/jobs/{name}")
@abstractmethod
def delete_job(*, workspace: str | None = None, name: str) -> None: ...


@post("/apis/jobs/v2/workspaces/{workspace}/jobs/{name}/cancel")
@abstractmethod
def cancel_job(*, workspace: str | None = None, name: str) -> HelixJobResponse: ...


@post("/apis/jobs/v2/workspaces/{workspace}/jobs/{name}/pause")
@abstractmethod
def pause_job(*, workspace: str | None = None, name: str) -> HelixJobResponse: ...


@post("/apis/jobs/v2/workspaces/{workspace}/jobs/{name}/resume")
@abstractmethod
def resume_job(*, workspace: str | None = None, name: str) -> HelixJobResponse: ...


# NOTE: no ``rerun_job`` — the server's ``/rerun`` route is test-only and not
# mounted in the release service (see services/core/jobs/.../api/v2/jobs/rerun.py),
# so exposing it on the client would 404 in production.


# ---------------------------------------------------------------------------
# Job status
# ---------------------------------------------------------------------------


@get("/apis/jobs/v2/workspaces/{workspace}/jobs/{name}/status")
@abstractmethod
def get_job_status(*, workspace: str | None = None, name: str) -> HelixJobStatusResponse: ...


@patch("/apis/jobs/v2/workspaces/{workspace}/jobs/{name}/status-details")
@abstractmethod
def update_job_status_details(*, workspace: str | None = None, name: str, body: JobStatusDetailsUpdate) -> None: ...


# ---------------------------------------------------------------------------
# Job logs
# ---------------------------------------------------------------------------


@get("/apis/jobs/v2/workspaces/{workspace}/jobs/{name}/logs")
@abstractmethod
def list_job_logs(
    *, workspace: str | None = None, name: str, query_params: JobLogsQueryParams | None = None
) -> Paginated[HelixJobLog, CursorPagination]: ...


# ---------------------------------------------------------------------------
# Job results
# ---------------------------------------------------------------------------


@post("/apis/jobs/v2/workspaces/{workspace}/jobs/{job}/results/{name}")
@abstractmethod
def create_job_result(
    *, workspace: str | None = None, job: str, name: str, body: HelixJobResultCreateRequest
) -> HelixJobResultResponse: ...


@get("/apis/jobs/v2/workspaces/{workspace}/jobs/{name}/results")
@abstractmethod
def list_job_results(
    *, workspace: str | None = None, name: str, query_params: ListJobResultsQueryParams | None = None
) -> HelixJobListResultResponse: ...


@get("/apis/jobs/v2/workspaces/{workspace}/jobs/{job}/results/{name}")
@abstractmethod
def get_job_result(*, workspace: str | None = None, job: str, name: str) -> HelixJobResultResponse: ...


@get("/apis/jobs/v2/workspaces/{workspace}/jobs/{job}/results/{name}/download")
@abstractmethod
def download_job_result(*, workspace: str | None = None, job: str, name: str) -> BinaryContent: ...


# ---------------------------------------------------------------------------
# Job steps
# ---------------------------------------------------------------------------


@get("/apis/jobs/v2/workspaces/{workspace}/jobs/{name}/steps")
@abstractmethod
def list_steps(
    *, workspace: str | None = None, name: str, query_params: ListStepsQueryParams | None = None
) -> Paginated[HelixJobStepWithContext]: ...


@get("/apis/jobs/v2/workspaces/{workspace}/jobs/{job}/steps/{name}")
@abstractmethod
def get_job_step(*, workspace: str | None = None, job: str, name: str) -> HelixJobStepResponse: ...


@patch("/apis/jobs/v2/workspaces/{workspace}/jobs/{job}/steps/{name}/status")
@abstractmethod
def update_job_step_status(
    *, workspace: str | None = None, job: str, name: str, body: HelixJobStatusUpdateRequest
) -> HelixJobStepResponse: ...


# ---------------------------------------------------------------------------
# Job tasks
# ---------------------------------------------------------------------------


@get("/apis/jobs/v2/workspaces/{workspace}/jobs/{job}/steps/{name}/tasks")
@abstractmethod
def list_job_step_tasks(*, workspace: str | None = None, job: str, name: str) -> HelixJobListTaskResponse: ...


@put("/apis/jobs/v2/workspaces/{workspace}/jobs/{job}/steps/{step}/tasks/{name}")
@abstractmethod
def update_job_step_task(
    *, workspace: str | None = None, job: str, step: str, name: str, body: HelixJobTaskUpdate
) -> HelixJobTaskResponse: ...


@get("/apis/jobs/v2/workspaces/{workspace}/jobs/{job}/steps/{step}/tasks/{name}")
@abstractmethod
def get_job_step_task(*, workspace: str | None = None, job: str, step: str, name: str) -> HelixJobTaskResponse: ...
